#![cfg(feature = "experimental-engines")] // RAPTOR needs the `experimental-engines` feature.
//! Integration tests for the Raptor storage engine.

use std::sync::Arc;
use tempfile::TempDir;

use proximadb::proto::proximadb_v1::{CollectionConfig, DistanceMetric, StorageEngine};
use proximadb::storage::engines::raptor::RaptorEngine;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::persistence::write_ahead_log::BatchId;
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};
use proximadb_records::ProximaRecord;

// Import test utilities
#[path = "../common/collection_builder.rs"]
mod collection_builder;
#[path = "../common/vector_generator.rs"]
mod vector_generator;

use collection_builder::TestCollectionBuilder;
use vector_generator::sequential;

/// Test setup helper
async fn create_test_setup() -> (Arc<RaptorEngine>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());

    // Create collection service - not needed for this test since we're testing the engine directly
    // The collection service would normally manage collections at a higher level

    // Create cache orchestrator for RAPTOR engine with memory budget
    let cache_orchestrator = Arc::new(
        proximadb::storage::cache::orchestrator::CrossCacheOrchestrator::new(
            1024 * 1024 * 1024, // 1GB memory budget
        ),
    );

    let raptor_engine = Arc::new(RaptorEngine::new().await.unwrap());

    (raptor_engine, temp_dir)
}

/// Create test vectors
/// REFACTORED: Now uses vector_generator::sequential()
fn create_test_vectors(count: usize) -> Vec<ProximaRecord> {
    sequential("raptor_test_collection", count, 128)
}

#[tokio::test]
async fn test_raptor_engine_creation_and_insertion() {
    let (raptor_engine, _temp_dir) = create_test_setup().await;

    // REFACTORED: Use TestCollectionBuilder
    let (mut collection, _temp) = TestCollectionBuilder::new()
        .with_id("raptor_test_collection")
        .with_name("raptor_test_collection")
        .with_dimension(128)
        .with_engine(StorageEngine::Raptor)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    // Override storage path and timestamps
    if let Some(ref mut assignment) = collection.storage_assignment {
        assignment.primary_path = _temp_dir.path().to_str().unwrap().to_string();
        assignment.base_location = _temp_dir.path().to_str().unwrap().to_string();
        assignment.assigned_at = chrono::Utc::now().timestamp();
    }
    collection.created_at = chrono::Utc::now().timestamp();
    collection.updated_at = chrono::Utc::now().timestamp();

    let vectors = create_test_vectors(100);
    let batch_ids: Vec<BatchId> = (0..100).map(|_i| BatchId::new()).collect();

    let flush_params = proximadb::storage::traits::FlushParameters {
        collection_id: Some("raptor_test_collection".to_string()),
        vector_records: vectors,
        batch_ids,
        force: true,
        synchronous: true,
        hints: std::collections::HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection),
        estimated_size: 1024 * 1024,
        collection_properties: std::collections::HashMap::new(),
    };

    let flush_result = raptor_engine.flush(flush_params).await.unwrap();
    assert!(flush_result.success);
    assert_eq!(flush_result.entries_flushed, Some(100));

    // Test would verify vectors are persisted - actual retrieval depends on engine implementation
    // The test verifies the flush was successful which means vectors were written
    // Individual vector retrieval is typically done through search operations
}
