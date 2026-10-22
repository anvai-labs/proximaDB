// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! The streaming provider must serve exactly the rows the materialized path
//! serves. Every test drives a REAL `SpillRecordStorage` over a local filesystem
//! (segments written by the production flush) and compares against that store's
//! own merged scan, projected the way `full_table_records` projects it.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use proximadb_catalog_schema::{CatalogColumn, CatalogTableSchema};
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_records::{ProximaRecord, ProximaTreeNode, RecordKey, RecordScan, RecordStore};
use proximadb_storage_common::proxima_arrow::{
    proxima_records_to_record_batch, proxima_records_to_record_batch_with_residual,
};
use proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes;

use super::LayerATableProvider;
use super::conform::{ConformContext, Conformed};
use super::merge::LayerAMerge;
use crate::services::SpillRecordStorage;
use crate::services::dml::DmlService;
use crate::services::layer_a_snapshot::{LayerARowAccess, LayerASnapshot, NativeLayerASnapshot};
use crate::services::record_spill::SpillSchemaSource;
use crate::storage::persistence::filesystem::FileSystem;
use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

const TABLE: &str = "orders";

/// `id` (the key) and `name` are NOT NULL, as SQL declares them: persisted
/// tombstones are NULL there, and must never reach an emitted batch.
fn schema() -> CatalogTableSchema {
    let not_null = |column: CatalogColumn| CatalogColumn {
        nullable: false,
        ..column
    };
    CatalogTableSchema::new(TABLE)
        .with_column(not_null(CatalogColumn::new(1, "id", ProximaType::Int64)))
        .with_column(not_null(CatalogColumn::new(2, "name", ProximaType::String)))
        .with_column(CatalogColumn::new(3, "score", ProximaType::Float64))
        .with_primary_key(vec!["id".to_string()])
}

fn row(id: i64, name: &str, score: f64) -> ProximaRecord {
    let mut r = ProximaRecord {
        oid: id.to_string(),
        tenant_id: "tenant-a".to_string(),
        variation_id: Some(TABLE.to_string()),
        ..Default::default()
    };
    r.props
        .insert("id".into(), ProximaTreeNode::Value(ProximaValue::Int64(id)));
    r.props.insert(
        "name".into(),
        ProximaTreeNode::Value(ProximaValue::String(name.to_string())),
    );
    r.props.insert(
        "score".into(),
        ProximaTreeNode::Value(ProximaValue::Float64(score)),
    );
    r
}

async fn fixture() -> (Arc<dyn FileSystem>, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().to_string_lossy().to_string();
    std::mem::forget(dir);
    let fs: Arc<dyn FileSystem> = Arc::new(
        LocalFileSystem::new(LocalConfig::default())
            .await
            .expect("local filesystem"),
    );
    (fs, base)
}

async fn spill_store() -> (SpillRecordStorage, Arc<dyn FileSystem>) {
    let (fs, base) = fixture().await;
    let store = SpillRecordStorage::with_flush_threshold(fs.clone(), base, None)
        .with_tenant("tenant-a")
        .with_schema_source(SpillSchemaSource::new(|| Some(schema())));
    (store, fs)
}

async fn snapshot_of(store: &SpillRecordStorage, fs: &Arc<dyn FileSystem>) -> LayerASnapshot {
    LayerASnapshot {
        view: RecordScan::scan_view(store).await.expect("view"),
        filesystem: Some(fs.clone()),
        table_name: TABLE.to_string(),
    }
}

async fn query(table: NativeLayerASnapshot, sql: &str) -> RecordBatch {
    let ctx = SessionContext::new();
    let provider = LayerATableProvider::new(table);
    let schema = provider_schema(&provider);
    ctx.register_table(TABLE, Arc::new(provider))
        .expect("register");
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan")
        .collect()
        .await
        .expect("execute");
    let schema = batches.first().map(|b| b.schema()).unwrap_or(schema);
    concat_batches(&schema, &batches).expect("concat")
}

fn provider_schema(provider: &LayerATableProvider) -> arrow::datatypes::SchemaRef {
    datafusion::datasource::TableProvider::schema(provider)
}

/// What the materialized path serves: the store's merged scan, member rows
/// only, each projected by the shared helper, ordered by id.
async fn materialized(store: &SpillRecordStorage) -> RecordBatch {
    let schema = schema();
    let names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let mut rows: Vec<ProximaRecord> = RecordScan::scan_records(store, usize::MAX)
        .await
        .expect("scan")
        .iter()
        .filter(|r| r.variation_id.as_deref().is_none_or(|v| v == TABLE))
        .filter_map(|r| DmlService::relational_record_for(r, &schema, &names))
        .collect();
    rows.sort_by_key(|r| match r.props.get("id") {
        Some(ProximaTreeNode::Value(ProximaValue::Int64(id))) => *id,
        _ => i64::MIN,
    });
    proxima_records_to_record_batch(&rows, &schema).expect("encode")
}

fn unrestricted(snapshot: LayerASnapshot) -> NativeLayerASnapshot {
    NativeLayerASnapshot {
        schema: schema(),
        snapshot: Some(snapshot),
        access: LayerARowAccess::Unrestricted,
    }
}

fn columns(batch: &RecordBatch) -> Vec<ArrayRef> {
    batch.columns().to_vec()
}

fn ids(batch: &RecordBatch) -> Vec<i64> {
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id column");
    ids.iter().map(|v| v.expect("id")).collect()
}

/// Updates across segments, a delete of a flushed row (memory tombstone, then
/// persisted), a resident update shadowing a segment copy, a re-insert after a
/// delete, an expired row, a foreign-table row, a residual value and a row with
/// no `id` prop (its key is its oid): the stream equals the materialized merge.
#[tokio::test]
async fn streams_exactly_the_rows_of_the_materialized_merge() {
    let (store, fs) = spill_store().await;
    for i in 0..40 {
        store
            .upsert_record(row(i, &format!("v0-{i}"), i as f64))
            .await
            .unwrap();
    }
    store.flush().await.unwrap().expect("segment 1");
    for i in (0..40).step_by(3) {
        store
            .upsert_record(row(i, &format!("v1-{i}"), -(i as f64)))
            .await
            .unwrap();
    }
    store
        .delete_record(&RecordKey::new("5".to_string()))
        .await
        .unwrap();
    store
        .delete_record(&RecordKey::new("7".to_string()))
        .await
        .unwrap();
    // A value the typed column cannot hold goes through the residual.
    let mut odd = row(8, "residual", 0.0);
    odd.props.insert(
        "score".into(),
        ProximaTreeNode::Value(ProximaValue::String("not-a-float".into())),
    );
    store.upsert_record(odd).await.unwrap();
    // No `id` prop: the key is served from the oid.
    let mut keyless = row(41, "keyless", 4.1);
    keyless.props.remove("id");
    store.upsert_record(keyless).await.unwrap();
    // Expired, and foreign: both invisible, both still shadowing.
    let mut expired = row(10, "expired", 1.0);
    expired.valid_to_ns = Some(1);
    store.upsert_record(expired).await.unwrap();
    let mut foreign = row(11, "foreign", 1.0);
    foreign.variation_id = Some("other".into());
    store.upsert_record(foreign).await.unwrap();
    store.flush().await.unwrap().expect("segment 2");
    // Memory layer on top: a tombstone over a flushed row, a resident update,
    // and a re-insert of a deleted oid.
    store
        .delete_record(&RecordKey::new("1".to_string()))
        .await
        .unwrap();
    store.upsert_record(row(2, "resident", 2.5)).await.unwrap();
    store.upsert_record(row(5, "back", 5.5)).await.unwrap();

    let view = snapshot_of(&store, &fs).await;
    assert_eq!(view.view.segments.len(), 2);
    assert!(!view.view.tombstones.is_empty() && !view.view.resident.is_empty());

    let got = query(
        unrestricted(view),
        "select id, name, score from orders order by id",
    )
    .await;
    let want = materialized(&store).await;
    assert_eq!(ids(&got), ids(&want));
    assert_eq!(columns(&got), columns(&want));
    assert!(!ids(&got).contains(&1) && !ids(&got).contains(&7) && !ids(&got).contains(&10));
}

/// Aggregates, a projection that leaves the key out, and a limit all run on
/// the streamed rows.
#[tokio::test]
async fn projection_count_and_limit() {
    let (store, fs) = spill_store().await;
    for i in 0..25 {
        store.upsert_record(row(i, "x", 1.0)).await.unwrap();
    }
    store.flush().await.unwrap();
    store
        .delete_record(&RecordKey::new("3".to_string()))
        .await
        .unwrap();
    let count = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select count(*) as n from orders",
    )
    .await;
    let n = count
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(n.value(0), 24);

    let limited = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select name from orders limit 5",
    )
    .await;
    assert_eq!(limited.num_rows(), 5);
}

/// A row predicate is applied per row, before the limit; a denied caller sees
/// an empty table without the partition being read.
#[tokio::test]
async fn abac_restricts_before_the_limit_and_denial_is_empty() {
    let (store, fs) = spill_store().await;
    for i in 0..10 {
        store
            .upsert_record(row(i, if i % 2 == 0 { "even" } else { "odd" }, 0.0))
            .await
            .unwrap();
    }
    store.flush().await.unwrap();
    store.upsert_record(row(20, "odd", 0.0)).await.unwrap();
    let only_even = || {
        LayerARowAccess::Restricted(Box::new(
            |r: &ProximaRecord| matches!(r.props.get("name"), Some(ProximaTreeNode::Value(ProximaValue::String(s))) if s == "even"),
        ))
    };
    let table = NativeLayerASnapshot {
        schema: schema(),
        snapshot: Some(snapshot_of(&store, &fs).await),
        access: only_even(),
    };
    let got = query(table, "select id from orders order by id").await;
    assert_eq!(ids(&got), [0, 2, 4, 6, 8]);

    let table = NativeLayerASnapshot {
        schema: schema(),
        snapshot: Some(snapshot_of(&store, &fs).await),
        access: only_even(),
    };
    assert_eq!(
        query(table, "select id from orders limit 3")
            .await
            .num_rows(),
        3
    );

    let denied = NativeLayerASnapshot {
        schema: schema(),
        snapshot: None,
        access: LayerARowAccess::Unrestricted,
    };
    assert_eq!(query(denied, "select id from orders").await.num_rows(), 0);
}

/// A segment that does not declare oid order (an older or foreign writer) is
/// read whole and sorted, never merged as if it were ordered.
#[tokio::test]
async fn an_undeclared_segment_is_sorted_before_it_is_merged() {
    let (fs, base) = fixture().await;
    let records: Vec<ProximaRecord> = [9, 3, 7, 1].iter().map(|&i| row(i, "seg", 0.0)).collect();
    let batch = proxima_records_to_record_batch_with_residual(&records, &schema()).unwrap();
    let bytes = record_batches_to_parquet_bytes(&[batch.clone()], batch.schema(), None).unwrap();
    let path = format!("{base}/unsorted.parquet");
    fs.write(&path, &bytes, None).await.unwrap();

    let snapshot = LayerASnapshot {
        view: proximadb_records::RecordScanView {
            resident: vec![row(3, "resident", 3.0)],
            tombstones: vec!["7".into()],
            segments: vec![path],
        },
        filesystem: Some(fs),
        table_name: TABLE.to_string(),
    };
    let got = query(
        unrestricted(snapshot),
        "select id, name from orders order by id",
    )
    .await;
    assert_eq!(ids(&got), [1, 3, 9]);
}

fn conformed(ctx: &ConformContext, oids: &[i64]) -> Conformed {
    let records: Vec<ProximaRecord> = oids.iter().map(|&i| row(i, "m", 0.0)).collect();
    Conformed::from_records(ctx, &records).unwrap()
}

async fn drain(mut merge: LayerAMerge) -> anyhow::Result<Vec<RecordBatch>> {
    let mut out = Vec::new();
    while let Some(batch) = merge.next_batch().await? {
        out.push(batch);
    }
    Ok(out)
}

/// The merge itself: newest wins across segments and batch boundaries, output
/// batches respect the batch size, and a segment out of oid order fails.
#[tokio::test]
async fn merge_is_newest_wins_and_verifies_order() {
    let ctx = Arc::new(
        ConformContext::new(schema(), None, Arc::new(LayerARowAccess::Unrestricted), 0).unwrap(),
    );
    // Oids compare as strings: "1" < "10" < "2".
    let older = futures::stream::iter(vec![
        Ok(conformed(&ctx, &[1, 10])),
        Ok(conformed(&ctx, &[2])),
    ])
    .boxed();
    let mut newer_rows = conformed(&ctx, &[10]);
    newer_rows.visible = arrow::array::BooleanArray::from(vec![false]);
    let newer = futures::stream::iter(vec![Ok(newer_rows)]).boxed();
    let merge = LayerAMerge::new(
        vec![older, newer],
        HashSet::new(),
        None,
        ctx.output.clone(),
        1,
        None,
    );
    let batches = drain(merge).await.unwrap();
    assert!(batches.iter().all(|b| b.num_rows() == 1));
    let got: Vec<i64> = batches.iter().flat_map(ids).collect();
    assert_eq!(
        got,
        [1, 2],
        "the invisible newest copy of 10 hides the older one"
    );

    let unsorted = futures::stream::iter(vec![Ok(conformed(&ctx, &[2, 1]))]).boxed();
    let merge = LayerAMerge::new(
        vec![unsorted],
        HashSet::new(),
        None,
        ctx.output.clone(),
        8,
        None,
    );
    let err = drain(merge).await.expect_err("out of order");
    assert!(err.to_string().contains("not strictly ordered"), "{err}");
}
