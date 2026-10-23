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
use datafusion::logical_expr::Expr;
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
        ConformContext::new(
            schema(),
            TABLE.to_string(),
            None,
            Arc::new(LayerARowAccess::Unrestricted),
            0,
        )
        .unwrap(),
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

/// A composite primary key cannot be served from the oid: a row taking the
/// record route must be decoded with its key columns even when the query does
/// not project them, or its projection fails and the row is lost (review of
/// #2153). `count(*)` and a key-less projection must match the materialized
/// path row for row.
#[tokio::test]
async fn a_record_route_row_keeps_a_composite_key_it_does_not_project() {
    let schema = CatalogTableSchema::new(TABLE)
        .with_column(CatalogColumn::new(1, "a", ProximaType::Int64))
        .with_column(CatalogColumn::new(2, "b", ProximaType::Int64))
        .with_column(CatalogColumn::new(3, "c", ProximaType::Float64))
        .with_primary_key(vec!["a".to_string(), "b".to_string()]);
    let (fs, base) = fixture().await;
    let encode = schema.clone();
    let store = SpillRecordStorage::with_flush_threshold(fs.clone(), base, None)
        .with_tenant("tenant-a")
        .with_schema_source(SpillSchemaSource::new(move || Some(encode.clone())));
    for (a, b) in [(1_i64, 1_i64), (1, 2), (2, 1)] {
        let mut r = ProximaRecord {
            oid: format!("{a}\u{1f}{b}"),
            tenant_id: "tenant-a".to_string(),
            variation_id: Some(TABLE.to_string()),
            ..Default::default()
        };
        r.props
            .insert("a".into(), ProximaTreeNode::Value(ProximaValue::Int64(a)));
        r.props
            .insert("b".into(), ProximaTreeNode::Value(ProximaValue::Int64(b)));
        // (1, 2) carries an integer in the float column: a residual row.
        let c = if (a, b) == (1, 2) {
            ProximaValue::Int64(12)
        } else {
            ProximaValue::Float64((a * 10 + b) as f64)
        };
        r.props.insert("c".into(), ProximaTreeNode::Value(c));
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");

    let table = || async {
        NativeLayerASnapshot {
            schema: schema.clone(),
            snapshot: Some(snapshot_of(&store, &fs).await),
            access: LayerARowAccess::Unrestricted,
        }
    };
    let count = query(table().await, "select count(*) as n from orders").await;
    let n = count
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(n.value(0), 3, "no row lost");
    let got = query(table().await, "select c from orders order by c").await;
    let c = got
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert_eq!(c.values().to_vec(), vec![11.0, 12.0, 21.0]);
}

// ---------------------------------------------------------------------------
// Row-group pruning (ADR-097 P3.1 PR 5)
// ---------------------------------------------------------------------------

/// Zero-padded oids, so oid order (the segment order) is id order and a row
/// group covers a contiguous id range.
fn padded(id: i64, score: f64) -> ProximaRecord {
    ProximaRecord {
        oid: format!("{id:04}"),
        ..row(id, "p", score)
    }
}

/// A spill store whose segments have `rows`-row row groups.
async fn small_row_group_store(rows: usize) -> (SpillRecordStorage, Arc<dyn FileSystem>) {
    let (store, fs) = spill_store().await;
    let policy = proximadb_storage_common::parquet_write_policy::ParquetWritePolicy {
        row_group_rows: rows,
        ..Default::default()
    };
    (store.with_write_policy(policy), fs)
}

/// Run the provider's scan with `filter` pushed in; the rows it returned
/// (before DataFusion re-applies the filter) and its row-group counters.
async fn pruned_scan(table: NativeLayerASnapshot, filter: Expr) -> (Vec<i64>, usize, usize) {
    let (scanned, count) = pruned_scan_counting(table, filter).await;
    (
        scanned,
        count("row_groups_pruned"),
        count("row_groups_total"),
    )
}

/// [`pruned_scan`], with every metric by name.
async fn pruned_scan_counting(
    table: NativeLayerASnapshot,
    filter: Expr,
) -> (Vec<i64>, impl Fn(&str) -> usize) {
    let ctx = SessionContext::new();
    let provider = LayerATableProvider::new(table);
    let plan = datafusion::datasource::TableProvider::scan(
        &provider,
        &ctx.state(),
        Some(&vec![0]),
        &[filter],
        None,
    )
    .await
    .expect("scan");
    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx())
        .await
        .expect("execute");
    let metrics = plan.metrics().expect("metrics");
    let count = move |name: &str| metrics.sum_by_name(name).map_or(0, |v| v.as_usize());
    (batches.iter().flat_map(ids).collect(), count)
}

/// Pruning skips row groups whose statistics rule the filter out, and the SQL
/// answer is the unpruned one.
#[tokio::test]
async fn pruning_skips_row_groups_and_keeps_the_answer() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..100 {
        store.upsert_record(padded(i, i as f64)).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");

    let (scanned, pruned, total) = pruned_scan(
        unrestricted(snapshot_of(&store, &fs).await),
        col("id").gt_eq(lit(90_i64)),
    )
    .await;
    assert_eq!(total, 10, "ten row groups of ten");
    assert_eq!(pruned, 9, "only the 90..99 row group can match");
    assert_eq!(scanned, (90..100).collect::<Vec<_>>());

    let got = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select id from orders where id >= 90 order by id",
    )
    .await;
    assert_eq!(ids(&got), (90..100).collect::<Vec<_>>());
}

/// A pruned row group still shadows older copies of its oids (the key spine):
/// row 5 was updated to a name the filter excludes, and its older matching
/// copy must not come back.
#[tokio::test]
async fn a_pruned_newer_copy_still_hides_an_older_matching_one() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..50 {
        store.upsert_record(padded(i, i as f64)).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment 1");
    let mut renamed = padded(5, 5.0);
    renamed.props.insert(
        "name".into(),
        ProximaTreeNode::Value(ProximaValue::String("z".to_string())),
    );
    store.upsert_record(renamed).await.unwrap();
    store.flush().await.unwrap().expect("segment 2");

    let (scanned, pruned, _) = pruned_scan(
        unrestricted(snapshot_of(&store, &fs).await),
        col("name").eq(lit("p")),
    )
    .await;
    assert!(pruned >= 1, "the newer copy's row group is pruned");
    assert!(
        !scanned.contains(&5),
        "the stale copy of 5 must stay hidden"
    );

    let got = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select id from orders where name = 'p' order by id",
    )
    .await;
    let want: Vec<i64> = (0..50).filter(|&i| i != 5).collect();
    assert_eq!(ids(&got), want);
    assert_eq!(
        columns(&got),
        columns(
            &query(
                unrestricted(snapshot_of(&store, &fs).await),
                "select id from orders where name = 'p' and id <> 5 order by id",
            )
            .await
        )
    );
}

/// A residual row overrides only the props its residual names. Row 3's
/// residual names `score` (an integer in a float column), so its row group's
/// `id` statistics still describe every served `id`: `id >= 10` prunes it
/// after the residual check, and the residual value is still served.
#[tokio::test]
async fn a_residual_row_not_naming_a_filter_column_does_not_block_pruning() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..20 {
        let mut r = padded(i, 1000.0 + i as f64);
        if i == 3 {
            r.props.insert(
                "score".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(1003)),
            );
        }
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");
    let (scanned, count) = pruned_scan_counting(
        unrestricted(snapshot_of(&store, &fs).await),
        col("id").gt_eq(lit(10_i64)),
    )
    .await;
    assert_eq!(count("row_groups_total"), 2);
    assert_eq!(count("row_groups_residual_checked"), 1);
    assert_eq!(
        count("row_groups_pruned"),
        1,
        "the residual names no filter column"
    );
    assert_eq!(scanned, (10..20).collect::<Vec<_>>());

    let got = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select id from orders where id >= 10 order by id",
    )
    .await;
    assert_eq!(ids(&got), (10..20).collect::<Vec<_>>());
    let three = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select score from orders where id = 3",
    )
    .await;
    let score = three
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert_eq!(score.value(0), 1003.0, "the residual value is still served");
}

/// A row group the residual check clears in a NEWER segment is still read as
/// a key spine: its rows (renamed away from the filter) keep shadowing the
/// older matching copies, which must not come back.
#[tokio::test]
async fn a_residual_cleared_row_group_still_shadows_older_copies() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..20 {
        store.upsert_record(padded(i, 0.0)).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment 1");
    for i in 0..10 {
        let mut r = padded(i, 0.0);
        r.props.insert(
            "name".into(),
            ProximaTreeNode::Value(ProximaValue::String("z".to_string())),
        );
        if i == 3 {
            // A residual naming `score`, not the filter column.
            r.props.insert(
                "score".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(7)),
            );
        }
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment 2");

    let (scanned, count) = pruned_scan_counting(
        unrestricted(snapshot_of(&store, &fs).await),
        col("name").eq(lit("p")),
    )
    .await;
    assert_eq!(count("row_groups_residual_checked"), 1);
    assert_eq!(count("row_groups_pruned"), 1, "segment 2's renamed rows");
    assert_eq!(scanned, (10..20).collect::<Vec<_>>(), "no stale copy");
    let got = query(
        unrestricted(snapshot_of(&store, &fs).await),
        "select id from orders where name = 'p' order by id",
    )
    .await;
    assert_eq!(ids(&got), (10..20).collect::<Vec<_>>());
}

/// A residual row that names a filter column keeps its row group, whatever
/// the statistics say: that column's served value is not in them.
#[tokio::test]
async fn a_residual_row_naming_a_filter_column_keeps_its_row_group() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..20 {
        let mut r = padded(i, 0.0);
        if i == 3 {
            // Not a string in a string column: the typed column cannot hold
            // it, so it is residual.
            r.props.insert(
                "name".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(1003)),
            );
        }
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");
    let (scanned, count) = pruned_scan_counting(
        unrestricted(snapshot_of(&store, &fs).await),
        col("name").eq(lit("zzz")),
    )
    .await;
    assert_eq!(count("row_groups_total"), 2);
    assert_eq!(count("row_groups_residual_checked"), 1);
    assert_eq!(
        count("row_groups_pruned"),
        1,
        "only the row group without residual rows is pruned"
    );
    assert!(scanned.contains(&3), "{scanned:?}");
}

/// The residual check errs only towards keeping: a residual key matches a
/// filter column case-insensitively and through a dotted column's top-level
/// prop; a residual that does not decode keeps the row group, and so does one
/// replacing embeddings when a filter references a vector column (the only
/// columns served from embeddings).
#[test]
fn the_residual_check_keeps_any_row_group_it_cannot_clear() {
    use datafusion::common::Column;
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::prelude::lit;
    use std::collections::HashMap;

    #[derive(serde::Serialize)]
    struct Cell {
        #[serde(skip_serializing_if = "HashMap::is_empty")]
        props: HashMap<String, ProximaTreeNode>,
        #[serde(skip_serializing_if = "Option::is_none")]
        embeddings: Option<Vec<u8>>,
    }
    let cell = |key: Option<&str>, embeddings: bool| {
        let props = key
            .map(|k| {
                HashMap::from([(
                    k.to_string(),
                    ProximaTreeNode::Value(ProximaValue::Int64(1)),
                )])
            })
            .unwrap_or_default();
        rmp_serde::to_vec_named(&Cell {
            props,
            embeddings: embeddings.then(Vec::new),
        })
        .unwrap()
    };

    use arrow::datatypes::{DataType, Field, Schema};
    let catalog = Arc::new(Schema::new(vec![
        Field::new("Name", DataType::Utf8, true),
        Field::new("a.b", DataType::Int64, true),
        Field::new("other", DataType::Int64, true),
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        ),
    ]));
    let ctx = SessionContext::new();
    let pruner = super::prune::SegmentPruner::try_new(
        &[
            Expr::Column(Column::new_unqualified("Name")).eq(lit("x")),
            Expr::Column(Column::new_unqualified("a.b")).gt(lit(1_i64)),
        ],
        catalog.clone(),
        None,
        &ctx.state(),
        &ExecutionPlanMetricsSet::new(),
    )
    .expect("a pruning predicate");
    let vector_pruner = super::prune::SegmentPruner::try_new(
        &[Expr::Column(Column::new_unqualified("v")).is_not_null()],
        catalog,
        None,
        &ctx.state(),
        &ExecutionPlanMetricsSet::new(),
    )
    .expect("a pruning predicate");

    let allows = |cells: &[Vec<u8>]| pruner.residual_allows_prune(cells.iter().map(Vec::as_slice));
    assert!(
        allows(&[cell(Some("other"), false)]),
        "names no filter column"
    );
    assert!(allows(&[]), "no residual cell");
    assert!(!allows(&[cell(Some("NAME"), false)]), "case-insensitive");
    assert!(!allows(&[cell(Some("a"), false)]), "a.b reads through a");
    assert!(!allows(&[cell(Some("A.B"), false)]), "same head, folded");
    assert!(
        allows(&[cell(None, true)]),
        "embeddings serve no scalar filter column"
    );
    assert!(
        !vector_pruner.residual_allows_prune([cell(None, true).as_slice()]),
        "replaces the embeddings a vector filter reads"
    );
    assert!(!allows(&[b"\xc1".to_vec()]), "undecodable");
    assert!(
        !allows(&[cell(Some("other"), false), cell(Some("name"), false)]),
        "any cell naming a filter column"
    );
}

/// A row whose key column is NULL serves its key from the oid, which the
/// statistics do not cover: its row group must not be pruned on the key.
/// Every other row's typed key is 1000+, so the statistics alone would prune
/// `id = 7`; the keyless row's key is 7, from its oid.
#[tokio::test]
async fn a_row_group_whose_key_is_null_in_some_row_is_not_pruned_on_the_key() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..10 {
        let mut r = padded(i, 0.0);
        if i == 7 {
            r.props.remove("id");
        } else {
            r.props.insert(
                "id".into(),
                ProximaTreeNode::Value(ProximaValue::Int64(1000 + i)),
            );
        }
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");
    let (scanned, pruned, _) = pruned_scan(
        unrestricted(snapshot_of(&store, &fs).await),
        col("id").eq(lit(7_i64)),
    )
    .await;
    assert_eq!(
        pruned, 0,
        "the key statistics are withheld for this row group"
    );
    assert!(scanned.contains(&7), "{scanned:?}");
}

/// A row group whose key column is NULL in every row is not pruned on the key
/// either: its null count would otherwise prove "no non-NULL key" to the
/// predicate, while every row's served key is its oid (review of PR 5).
#[tokio::test]
async fn a_row_group_whose_key_is_null_in_every_row_is_not_pruned_on_the_key() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..10 {
        let mut r = padded(i, 0.0);
        r.props.remove("id");
        store.upsert_record(r).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");
    for filter in [col("id").eq(lit(7_i64)), col("id").is_not_null()] {
        let (scanned, pruned, _) =
            pruned_scan(unrestricted(snapshot_of(&store, &fs).await), filter.clone()).await;
        assert_eq!(pruned, 0, "{filter}: the key statistics are withheld");
        assert!(scanned.contains(&7), "{filter}: {scanned:?}");
    }
}

/// Parquet leaves NaN out of float min/max while DataFusion orders NaN above
/// every number, so a float column is never pruned on (review of PR 5).
#[tokio::test]
async fn a_nan_float_row_is_never_pruned_away() {
    use datafusion::prelude::{col, lit};
    let (store, fs) = small_row_group_store(10).await;
    for i in 0..10 {
        let score = if i == 7 { f64::NAN } else { 0.5 };
        store.upsert_record(padded(i, score)).await.unwrap();
    }
    store.flush().await.unwrap().expect("segment");
    let (scanned, pruned, _) = pruned_scan(
        unrestricted(snapshot_of(&store, &fs).await),
        col("score").gt(lit(1.0_f64)),
    )
    .await;
    assert_eq!(pruned, 0, "float statistics are not used");
    assert!(scanned.contains(&7), "{scanned:?}");
}
