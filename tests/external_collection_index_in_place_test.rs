//! Phase 8 F5 (TD-090) integration: index an external Parquet table **in place**
//! and search it through the public crate API, asserting the source is never
//! copied.
//!
//! Exercises the full F5 Slice 1 path end-to-end via the public facade
//! (`proximadb::services::external_collection`): write an external Parquet file →
//! register (catalog `FederatedRead` + `ExternalSnapshotRegistered`) → build the
//! ProximaDB-owned IVF index in place → search returns the exact nearest row.
//!
//! No-copy proof: (a) the catalog models the source as `FederatedRead`
//! (ProximaDB owns the index, not the records); (b) the source fingerprint is
//! unchanged after build (ProximaDB never rewrote the source); (c) the
//! republished projection's `source_range` is the external snapshot id.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, StringBuilder};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use proximadb_catalog::{CatalogAuthorityMode, ProjectionFreshnessState};

use proximadb::catalog::CatalogManager;
use proximadb::index::{AxisConfig, AxisManager};
use proximadb::services::external_collection::{
    ExternalCollectionRegistry, ExternalCollectionService, ExternalCollectionSpec,
    ExternalCollectionStatus,
};

/// Write `n` rows with unique vector directions (one-hot at position `i`, so
/// under cosine each row's nearest neighbor is itself) plus a `title: Utf8`
/// metadata column for federated-fetch assertions. Requires `dim >= n`.
fn write_external_parquet(path: &std::path::Path, n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    assert!(dim >= n, "one-hot fixture requires dim >= n");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));
    let mut id_b = StringBuilder::new();
    let mut title_b = StringBuilder::new();
    let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    let mut expect = Vec::new();
    for i in 0..n {
        let id = format!("doc-{i}");
        let mut v = vec![0.0f32; dim];
        v[i] = 1.0 + (i as f32) * 0.01;
        id_b.append_value(&id);
        title_b.append_value(format!("title-{i}"));
        for x in &v {
            vec_b.values().append_value(*x);
        }
        vec_b.append(true);
        expect.push((id, v));
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_b.finish()),
            Arc::new(title_b.finish()),
            Arc::new(vec_b.finish()),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    expect
}

async fn catalog_manager() -> Arc<CatalogManager> {
    let tmp = std::env::temp_dir().join(format!(
        "proximadb_f5_it_cat_{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let manager = Arc::new(CatalogManager::new());
    manager
        .create_native_catalog("default", &format!("file://{}", tmp.display()))
        .await
        .unwrap();
    manager.set_default_catalog("default").await.unwrap();
    manager
}

#[tokio::test]
async fn external_parquet_indexed_in_place_and_searchable_without_copy() {
    let dim = 56;
    let parquet = std::env::temp_dir().join(format!(
        "proximadb_f5_it_{}.parquet",
        uuid::Uuid::new_v4().simple()
    ));
    let expect = write_external_parquet(&parquet, 40, dim);
    allowlist_parent_of(&parquet);

    let cat = catalog_manager().await;
    let axis = Arc::new(AxisManager::new(AxisConfig::default()).await.unwrap());
    let svc = ExternalCollectionService::new(
        Arc::new(ExternalCollectionRegistry::new()),
        cat.clone(),
        axis.clone(),
    )
    .with_axis_runtime_enabled(true);

    // Register un-copied.
    let spec = ExternalCollectionSpec::parquet(
        "ext_papers",
        parquet.to_str().unwrap(),
        "id",
        "vector",
        dim,
    );
    let ec = svc.register(spec).await.unwrap();
    assert_eq!(ec.status, ExternalCollectionStatus::Registered);

    // (a) Catalog authority is FederatedRead — ProximaDB does not own the records.
    let (catalog, identifier) = cat.resolve_table("ext_papers").await.unwrap();
    let schema = catalog.get_table(&identifier).await.unwrap();
    assert_eq!(
        schema.storage_layouts[0].authority,
        CatalogAuthorityMode::FederatedRead
    );
    assert_eq!(
        schema.storage_layouts[0].location.as_deref(),
        Some(parquet.to_str().unwrap())
    );

    // Build the ProximaDB-owned index in place.
    let count = svc.build(&ec.id).await.unwrap();
    assert_eq!(count, 40);
    assert!(axis.has_ivf_index("ext_papers").await);
    let built = svc.get(&ec.id).unwrap();
    assert_eq!(built.status, ExternalCollectionStatus::Ready);
    assert_eq!(built.indexed_record_count, 40);

    // (c) Projection republished Fresh with the external snapshot id as lineage.
    let proj = svc.projection("ext_papers").await.unwrap().unwrap();
    assert_eq!(proj.freshness_state, ProjectionFreshnessState::Fresh);
    assert_eq!(proj.source_range.as_deref(), Some(ec.snapshot_id.as_str()));

    // Search routes through the same AxisManager path and returns the exact row
    // WITH its full record federated from the source (Slice 2: props, not just id).
    let (qid, qvec) = &expect[7];
    let hits = svc.search(&ec.id, qvec.clone(), 5).await.unwrap();
    assert!(!hits.is_empty(), "external index must return hits");
    assert_eq!(&hits[0].id, qid, "top-1 must be the exact external row");
    match &hits[0].record.props["title"] {
        proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
            s,
        )) => assert_eq!(s, "title-7", "hit must carry the source's title metadata"),
        other => panic!("title prop wrong: {other:?}"),
    }

    // (b) No copy: the snapshot id is stable across register+build (build did
    // not rewrite the source) and the source file is still present. The index
    // lives only in the in-memory AxisManager; no storage engine was wired, so
    // ProximaDB structurally cannot have copied the records.
    assert_eq!(
        ec.snapshot_id, built.snapshot_id,
        "snapshot id must be stable across register+build (source untouched)"
    );
    assert!(parquet.exists(), "external source must still be present");

    // Slice 2 staleness-refresh: unchanged source is not stale; mutating the
    // source (more rows) makes refresh rebuild in place and advance the snapshot.
    assert!(!svc.is_stale(&ec.id).unwrap());
    write_external_parquet(&parquet, 50, dim);
    assert!(
        svc.is_stale(&ec.id).unwrap(),
        "changed source must be stale"
    );
    let outcome = svc.refresh(&ec.id).await.unwrap();
    assert!(
        outcome.stale_detected && outcome.rebuilt,
        "stale source must rebuild"
    );
    assert_eq!(outcome.indexed_record_count, 50);
    assert_ne!(
        outcome.snapshot_id, built.snapshot_id,
        "snapshot must advance"
    );
    let refreshed = svc.get(&ec.id).unwrap();
    assert_eq!(refreshed.indexed_record_count, 50);
    let proj2 = svc.projection("ext_papers").await.unwrap().unwrap();
    assert_eq!(proj2.freshness_state, ProjectionFreshnessState::Fresh);
    assert_eq!(
        proj2.source_range.as_deref(),
        Some(outcome.snapshot_id.as_str())
    );

    let _ = std::fs::remove_file(&parquet);
}

/// F5 Slice 3: register with a BM25 text column → build → hybrid (vector + text)
/// search fuses a lexical-only match in alongside the vector match.
#[tokio::test]
async fn external_hybrid_search_fuses_text_and_vector() {
    let dim = 48;
    let parquet = std::env::temp_dir().join(format!(
        "proximadb_f5_hyb_{}.parquet",
        uuid::Uuid::new_v4().simple()
    ));
    let expect = write_external_parquet(&parquet, 40, dim);
    allowlist_parent_of(&parquet);

    let cat = catalog_manager().await;
    let axis = Arc::new(AxisManager::new(AxisConfig::default()).await.unwrap());
    let svc =
        ExternalCollectionService::new(Arc::new(ExternalCollectionRegistry::new()), cat, axis)
            .with_axis_runtime_enabled(true);

    let spec = ExternalCollectionSpec::parquet(
        "ext_hybrid",
        parquet.to_str().unwrap(),
        "id",
        "vector",
        dim,
    )
    .with_text_column("title");
    let ec = svc.register(spec).await.unwrap();
    svc.build(&ec.id).await.unwrap();
    assert!(
        svc.has_fulltext_index("ext_hybrid"),
        "BM25 index built over the title column"
    );

    // Vector query points at doc-3; lexical query is doc-7's title.
    let (_qid, qvec) = &expect[3];
    let hybrid = svc
        .hybrid_search(&ec.id, qvec.clone(), Some("title-7".to_string()), 5)
        .await
        .unwrap();
    let ids: Vec<&str> = hybrid.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"doc-7"), "lexical match fused in: {ids:?}");
    assert!(ids.contains(&"doc-3"), "vector match retained: {ids:?}");
    // Fused hits still carry federated props from the source.
    let d7 = hybrid.iter().find(|h| h.id == "doc-7").unwrap();
    assert!(d7.record.props.contains_key("title"));

    let _ = std::fs::remove_file(&parquet);
}

/// Allowlist the parquet's own directory for this test process.
///
/// `POST /api/v2/external-collections` is now gated by the same fail-closed
/// `PROXIMADB_EXTERNAL_TABLE_ROOTS` allowlist as `CREATE EXTERNAL TABLE`
/// (TD-SECSCAN-1: the location previously went straight to `fs::metadata` /
/// `fs::read_dir`). Unset means every location is rejected, which is the
/// shipped default, so a test that reads a real file must opt in.
///
/// The root is written as a BARE path, not `file://...`, because these tests
/// pass a bare path as the location and the allowlist match is textual.
fn allowlist_parent_of(path: &std::path::Path) {
    let dir = path
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "/".to_string());
    // Safe here: integration tests get their own process, and this runs before
    // the service under test reads the gate.
    unsafe { std::env::set_var("PROXIMADB_EXTERNAL_TABLE_ROOTS", &dir) };
}
