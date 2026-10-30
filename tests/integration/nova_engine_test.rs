//! Integration tests for the Nova storage engine.

use std::sync::Arc;
use tempfile::TempDir;

use proximadb::proto::proximadb_v1::{
    Collection, DistanceMetric, FilterableDataType, StorageEngine,
};
use proximadb::services::collection::manager::CollectionService;
use proximadb::services::operations::vectors::VectorOperationsService;
use proximadb::storage::engines::nova::NovaEngine;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::persistence::write_ahead_log::BatchId;
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine, UnifiedStorageFormat};
use proximadb_data_model::ProximaValue;
use proximadb_records::{
    EmbeddingCell, EmbeddingValues, LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode,
};
use std::collections::HashMap;

// Import test utilities
#[path = "../common/collection_builder.rs"]
mod collection_builder;
#[path = "../common/vector_generator.rs"]
mod vector_generator;

use collection_builder::TestCollectionBuilder;
use vector_generator::sequential;

/// Test setup helper - DEPRECATED, use direct Collection creation instead
/// This setup requires complex CollectionService initialization which is not needed for engine tests
#[allow(dead_code)]
async fn create_test_setup() -> (Arc<NovaEngine>, Arc<CollectionService>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let filesystem_config = FilesystemConfig::default();
    let _filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());

    let storage_config = proximadb::core::config::StorageConfig::default();
    let collection_service = Arc::new(CollectionService::new(storage_config).await.unwrap());

    let nova_engine = Arc::new(NovaEngine::new().await.unwrap());

    (nova_engine, collection_service, temp_dir)
}

/// Create test vectors
/// REFACTORED: Now uses vector_generator::sequential()
fn create_test_vectors(count: usize) -> Vec<ProximaRecord> {
    sequential("nova_test_collection", count, 128)
}

/// Build a single `ProximaRecord` with explicit `record_version`/`valid_to_ns`
/// (TD-DSEFF-2 MVCC/tombstone tests need full control over these — the
/// shared `vector_generator` helpers always default them to `1`/`None`).
fn versioned_record(
    oid: &str,
    dim: usize,
    version: u64,
    valid_to_ns: Option<i64>,
) -> ProximaRecord {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64;
    let vector: Vec<f32> = (0..dim).map(|d| (version as f32) + d as f32).collect();
    ProximaRecord {
        schema_version: proximadb_records::schema_version::default_schema_version(),
        oid: oid.to_string(),
        local_id: None,
        tid: None,
        variation_id: None,
        record_version: version,
        spec_version: 1,
        tenant_id: String::new(),
        permitted_principals: Vec::new(),
        rls_policy_id: None,
        created_at_ns: now,
        updated_at_ns: now,
        valid_from_ns: None,
        valid_to_ns,
        origin: None,
        actor: None,
        method: Some("test".to_string()),
        memory_type: None,
        props: ProximaTree::new(),
        refs: Vec::new(),
        edge: None,
        embeddings: vec![EmbeddingCell::new_fp32(
            "default",
            "dense_vector",
            dim as u32,
            vector,
        )],
        sequence: None,
        labels: LabelSet::new(),
        branch_id: None,
    }
}

/// Flush a single record as its own batch (its own Parquet file) — used to
/// simulate multiple flushes of the same oid landing in separate files,
/// exactly as NOVA's real flush path does per-batch (TD-DSEFF-2).
async fn flush_one(
    nova_engine: &NovaEngine,
    collection_id: &str,
    collection: &Collection,
    record: ProximaRecord,
) {
    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        vector_records: vec![record],
        batch_ids: vec![],
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection.clone()),
        estimated_size: 1024,
    };
    let result = nova_engine.flush(flush_params).await.unwrap();
    assert!(result.success, "flush must succeed");
}

#[tokio::test]
async fn test_nova_engine_creation_and_insertion() {
    let nova_engine = NovaEngine::new().await.unwrap();

    let vectors = create_test_vectors(100);
    let collection_id = "nova_test_collection";

    // REFACTORED: Use TestCollectionBuilder
    let (mut collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(128)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let base_location = _temp.path().to_str().unwrap().to_string();

    let batch_ids: Vec<BatchId> = (0..100).map(|_| BatchId::new()).collect();

    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        vector_records: vectors,
        batch_ids,
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection),
        estimated_size: 1024 * 1024,
    };

    println!(
        "🧪 Test: Flushing {} vectors",
        flush_params.vector_records.len()
    );
    let flush_result = nova_engine.flush(flush_params).await.unwrap();

    println!(
        "🧪 Test: Flush result - success={}, entries={:?}",
        flush_result.success, flush_result.entries_flushed
    );
    assert!(flush_result.success, "Flush should succeed");
    assert_eq!(
        flush_result.entries_flushed,
        Some(100),
        "Should flush 100 vectors"
    );

    // Verify files were created using filesystem API for cloud compatibility
    let data_path = format!("{}/{}/data", base_location, collection_id);
    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
    let fs = filesystem.get_filesystem(&data_path).unwrap();

    let entries = fs.list(&data_path).await.unwrap();
    let parquet_files: Vec<_> = entries
        .into_iter()
        .filter(|e| !e.metadata.is_directory && e.name.ends_with(".parquet"))
        .collect();

    println!("🧪 Test: Found {} parquet files", parquet_files.len());
    assert!(
        !parquet_files.is_empty(),
        "Should have created at least one parquet file"
    );

    // Note: vector_by_id requires ID index which may not be built yet, so we skip that check
    // The important verification is that flush succeeded and files were created
}

#[tokio::test]
async fn test_nova_flush_basic() {
    let nova_engine = NovaEngine::new().await.unwrap();

    let vectors = create_test_vectors(10);
    let collection_id = "test_flush";

    // REFACTORED: Use TestCollectionBuilder
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(128)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let base_location = _temp.path().to_str().unwrap().to_string();

    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        vector_records: vectors,
        batch_ids: vec![],
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection),
        estimated_size: 1024 * 1024,
    };

    println!(
        "🧪 Test: Flushing {} vectors to {}",
        flush_params.vector_records.len(),
        base_location
    );
    let flush_result = nova_engine.flush(flush_params).await.unwrap();

    println!(
        "🧪 Test: Flush result - success={}, entries={:?}, bytes={:?}",
        flush_result.success, flush_result.entries_flushed, flush_result.bytes_written
    );

    assert!(flush_result.success, "Flush should succeed");
    assert_eq!(
        flush_result.entries_flushed,
        Some(10),
        "Should flush 10 vectors"
    );
    assert!(
        flush_result.bytes_written.unwrap() > 0,
        "Should write some bytes"
    );

    // Verify files exist using filesystem API (cloud-compatible)
    let data_path = format!("{}/{}/data", base_location, collection_id);
    println!("🧪 Test: Checking for files in {}", data_path);

    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
    let fs = filesystem.get_filesystem(&data_path).unwrap();

    let entries = fs.list(&data_path).await.unwrap();
    let parquet_files: Vec<_> = entries
        .into_iter()
        .filter(|e| !e.metadata.is_directory && e.name.ends_with(".parquet"))
        .collect();

    println!("🧪 Test: Found {} parquet files", parquet_files.len());
    for file in &parquet_files {
        println!("  - {}", file.name);
    }

    assert!(
        !parquet_files.is_empty(),
        "Should have created at least one parquet file"
    );
}

#[tokio::test]
async fn test_nova_search_basic() {
    use proximadb::core::search::SearchParams;
    use proximadb::storage::traits::{StorageQueryContext, StorageQueryMetadata};

    let nova_engine = NovaEngine::new().await.unwrap();

    let vectors = create_test_vectors(20);
    let collection_id = "test_search";

    // REFACTORED: Use TestCollectionBuilder
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(128)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let base_location = _temp.path().to_str().unwrap().to_string();

    // First flush some vectors
    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        vector_records: vectors.clone(),
        batch_ids: vec![],
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection.clone()),
        estimated_size: 1024 * 1024,
    };

    println!("🧪 Test: Flushing {} vectors", vectors.len());
    let flush_result = nova_engine.flush(flush_params).await.unwrap();
    assert!(flush_result.success);

    // Now search
    let query_vector = vectors[0]
        .embeddings
        .first()
        .map(|e| e.values.to_fp32_owned())
        .unwrap_or_default();
    let search_params = Arc::new(SearchParams {
        vector: Some(query_vector),
        top_k: Some(5),
        ..Default::default()
    });

    let ctx = StorageQueryContext {
        search_params,
        collection: Arc::new(collection),
        metadata: StorageQueryMetadata {
            collection_id: collection_id.to_string(),
            dimension: 128,
            storage_path: base_location.clone(),
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };

    // Check what files exist before searching (using filesystem API for cloud compatibility)
    let data_path = format!("{}/{}/data", base_location, collection_id);
    println!("🧪 Test: Checking data_path: {}", data_path);

    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
    let fs = filesystem.get_filesystem(&data_path).unwrap();

    if let Ok(entries) = fs.list(&data_path).await {
        println!("🧪 Test: Found {} files in data directory", entries.len());
        for file in entries {
            println!("  - {} (is_dir={})", file.name, file.metadata.is_directory);
        }
    } else {
        println!("🧪 Test: data_path does not exist or cannot be read");
    }

    println!("🧪 Test: Searching...");
    let results = nova_engine.search_vectors_unified(&ctx).await.unwrap();

    println!("🧪 Test: Found {} results", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  Result {}: id={}, score={}", i, r.id, r.score);
    }

    assert!(!results.is_empty(), "Should find at least one result");
    assert!(results.len() <= 5, "Should return at most 5 results");
    assert_eq!(
        results[0].id, "vec_0",
        "First result should be vec_0 (exact match)"
    );
}

#[tokio::test]
async fn test_nova_compact_basic() {
    use proximadb::storage::traits::CompactionParameters;

    let nova_engine = NovaEngine::new().await.unwrap();

    let collection_id = "test_compact";

    // REFACTORED: Use TestCollectionBuilder
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(128)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let base_location = _temp.path().to_str().unwrap().to_string();

    // Flush multiple batches to create multiple files
    for i in 0..3 {
        let vectors = create_test_vectors(10);
        let flush_params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vectors,
            batch_ids: vec![],
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: Some(30000),
            trigger_compaction: false,
            collection_config: Some(collection.clone()),
            estimated_size: 1024 * 1024,
        };

        println!("🧪 Test: Flushing batch {}", i);
        let flush_result = nova_engine.flush(flush_params).await.unwrap();
        assert!(flush_result.success);
    }

    // Compact
    let compact_params = CompactionParameters {
        collection_id: Some(collection_id.to_string()),
        collection_config: Some(collection),
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(60000),
        priority: proximadb::storage::traits::OperationPriority::Medium,
        estimated_input_size: 1024 * 1024 * 10, // 10MB estimate
    };

    println!("🧪 Test: Starting compaction");
    let compact_result = nova_engine.compact(compact_params).await.unwrap();

    println!(
        "🧪 Test: Compaction result - success={}, input_files={:?}, output_files={:?}, entries_processed={:?}",
        compact_result.success,
        compact_result.input_files,
        compact_result.output_files,
        compact_result.entries_processed
    );

    assert!(compact_result.success, "Compaction should succeed");
}

// ---------------------------------------------------------------------------
// TD-DSEFF-2: `vector_by_id` MVCC resolution + tombstone/TTL filtering.
//
// NOVA's flush path writes a brand-new uniquely-named Parquet file per flush
// (`NovaFlushOperations::write_nova_file_to_disk`) — nothing merges by id
// until compaction runs, so an id updated across two flushes genuinely
// exists in two files simultaneously. Before this fix, `vector_by_id`
// returned the first file/row match with no version comparison and no
// dead-record filtering at all.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_nova_vector_by_id_returns_latest_version_across_flushes() {
    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_mvcc_version";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    // v1 flushed first, v2 (same oid, higher version) flushed second — into
    // a SEPARATE file, since each flush call writes its own file.
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("mvcc_vec", 4, 1, None),
    )
    .await;
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("mvcc_vec", 4, 2, None),
    )
    .await;

    let found = nova_engine
        .vector_by_id(collection_id, &base_location, "mvcc_vec")
        .await
        .unwrap()
        .expect("the record must be found");
    assert_eq!(
        found.record_version, 2,
        "vector_by_id must return the HIGHER version across files, not the first match"
    );
    let vector = found.embeddings[0].values.to_fp32_owned();
    assert_eq!(
        vector[0], 2.0,
        "returned vector data must belong to the v2 record, not v1"
    );
}

#[tokio::test]
async fn test_nova_vector_by_id_hides_tombstoned_record() {
    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_mvcc_tombstone";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    // A live v1, then a tombstone at v2 (higher version, valid_to_ns =
    // Some(0)) in a LATER flush — the tombstone must shadow the live
    // version (CLAUDE.md invariant 16d), not be skipped in favor of it.
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("tombstoned_vec", 4, 1, None),
    )
    .await;
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("tombstoned_vec", 4, 2, Some(0)),
    )
    .await;

    let found = nova_engine
        .vector_by_id(collection_id, &base_location, "tombstoned_vec")
        .await
        .unwrap();
    assert!(
        found.is_none(),
        "a tombstoned record (higher version, valid_to_ns=Some(0)) must not be returned"
    );
}

#[tokio::test]
async fn test_nova_vector_by_id_filters_ttl_expired_and_returns_live() {
    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_mvcc_ttl";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64;
    let one_hour_ns = 3_600_000_000_000i64;

    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("expired_vec", 4, 1, Some(now_ns - one_hour_ns)),
    )
    .await;
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("future_vec", 4, 1, Some(now_ns + one_hour_ns)),
    )
    .await;
    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("forever_vec", 4, 1, None),
    )
    .await;

    assert!(
        nova_engine
            .vector_by_id(collection_id, &base_location, "expired_vec")
            .await
            .unwrap()
            .is_none(),
        "a record with a past valid_to_ns must not be returned"
    );
    assert!(
        nova_engine
            .vector_by_id(collection_id, &base_location, "future_vec")
            .await
            .unwrap()
            .is_some(),
        "a record with a future valid_to_ns must be returned"
    );
    assert!(
        nova_engine
            .vector_by_id(collection_id, &base_location, "forever_vec")
            .await
            .unwrap()
            .is_some(),
        "a record with no valid_to_ns must be returned"
    );
}

#[tokio::test]
async fn test_nova_vector_by_id_preserves_metadata_types_and_origin() {
    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_point_read_metadata";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .with_filterable_column("category", FilterableDataType::FilterableString)
        .with_filterable_column("count", FilterableDataType::FilterableInteger)
        .with_filterable_column("score", FilterableDataType::FilterableFloat)
        .with_filterable_column("active", FilterableDataType::FilterableBoolean)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    let mut record = versioned_record("metadata_vec", 4, 1, None);
    record.origin = Some("fixture-origin".to_string());
    record.props.insert(
        "category".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("books".to_string())),
    );
    record.props.insert(
        "count".to_string(),
        ProximaTreeNode::Value(ProximaValue::Int64(7)),
    );
    record.props.insert(
        "score".to_string(),
        ProximaTreeNode::Value(ProximaValue::Float64(9.5)),
    );
    record.props.insert(
        "active".to_string(),
        ProximaTreeNode::Value(ProximaValue::Boolean(true)),
    );
    record.props.insert(
        "note".to_string(),
        ProximaTreeNode::Value(ProximaValue::String("extra".to_string())),
    );
    flush_one(&nova_engine, collection_id, &collection, record).await;

    let found = nova_engine
        .vector_by_id(collection_id, &base_location, "metadata_vec")
        .await
        .unwrap()
        .expect("record must exist");

    assert_eq!(found.origin.as_deref(), Some("fixture-origin"));
    assert_eq!(
        found.props.get("category"),
        Some(&ProximaTreeNode::Value(ProximaValue::String(
            "books".to_string()
        )))
    );
    assert_eq!(
        found.props.get("count"),
        Some(&ProximaTreeNode::Value(ProximaValue::Int64(7)))
    );
    assert_eq!(
        found.props.get("score"),
        Some(&ProximaTreeNode::Value(ProximaValue::Float64(9.5)))
    );
    assert_eq!(
        found.props.get("active"),
        Some(&ProximaTreeNode::Value(ProximaValue::Boolean(true)))
    );
    assert_eq!(
        found.props.get("note"),
        Some(&ProximaTreeNode::Value(ProximaValue::String(
            "extra".to_string()
        )))
    );
    assert!(
        !found.props.contains_key("source"),
        "source is record provenance, not user metadata"
    );
}

#[tokio::test]
async fn test_nova_vector_by_id_fails_closed_on_unreadable_candidate_file() {
    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_point_read_corrupt_file";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("uncertain_vec", 4, 1, None),
    )
    .await;
    let data_dir = _temp.path().join(collection_id).join("data");
    std::fs::write(data_dir.join("unknown-newer.parquet"), b"not parquet").unwrap();

    let result = nova_engine
        .vector_by_id(collection_id, &base_location, "uncertain_vec")
        .await;
    assert!(
        result.is_err(),
        "point reads must not return a stale candidate when another data file is unreadable"
    );
}

#[tokio::test]
async fn test_nova_vector_by_id_does_not_serve_cached_pre_update_version() {
    use proximadb::storage::cache::orchestrator::CrossCacheOrchestrator;
    use proximadb::storage::cache::specialized::vector_cache::VectorCache;

    let orchestrator = Arc::new(
        CrossCacheOrchestrator::new(1024 * 1024).with_vector_cache(Arc::new(VectorCache::new(1))),
    );
    CrossCacheOrchestrator::register_global(orchestrator);

    let nova_engine = NovaEngine::new().await.unwrap();
    let collection_id = "test_point_read_cache_freshness";
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_id(collection_id)
        .with_name(collection_id)
        .with_dimension(4)
        .with_engine(StorageEngine::Nova)
        .build();
    let base_location = _temp.path().to_str().unwrap().to_string();

    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("cached_vec", 4, 1, None),
    )
    .await;
    let first = nova_engine
        .vector_by_id(collection_id, &base_location, "cached_vec")
        .await
        .unwrap()
        .expect("v1 must exist");
    assert_eq!(first.record_version, 1);

    flush_one(
        &nova_engine,
        collection_id,
        &collection,
        versioned_record("cached_vec", 4, 2, None),
    )
    .await;
    let second = nova_engine
        .vector_by_id(collection_id, &base_location, "cached_vec")
        .await
        .unwrap()
        .expect("v2 must exist");
    assert_eq!(
        second.record_version, 2,
        "an uninvalidated point-read cache must not hide a newer flushed version"
    );
}
