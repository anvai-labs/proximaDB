//! Integration tests for the Swift storage engine.

// SWIFT is compiled only under `experimental-engines` (TD-152 deprecation gate).
#![cfg(feature = "experimental-engines")]

use std::sync::Arc;
use tempfile::TempDir;

use proximadb::proto::proximadb_v1::{CollectionConfig, DistanceMetric, StorageEngine};
use proximadb::services::collection::manager::CollectionService;
use proximadb::storage::engines::swift::SwiftEngine;
use proximadb::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
use proximadb::storage::persistence::write_ahead_log::BatchId;
use proximadb::storage::traits::{FlushParameters, UnifiedStorageEngine};
use proximadb_records::ProximaRecord;
use std::collections::HashMap;

// Import test utilities
#[path = "../common/collection_builder.rs"]
mod collection_builder;
#[path = "../common/vector_generator.rs"]
mod vector_generator;

use collection_builder::TestCollectionBuilder;
use vector_generator::sequential;

/// Test setup helper
async fn create_test_setup() -> (Arc<SwiftEngine>, Arc<CollectionService>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let filesystem_config = FilesystemConfig::default();
    let _filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());

    let metadata_path = temp_dir.path().join("metadata");

    // Create storage config with tempdir-based path
    let mut storage_config = proximadb::core::config::StorageConfig::default();
    storage_config.storage_locations = vec![proximadb::core::config::StorageLocation {
        url: format!("file://{}", temp_dir.path().display()),
        weight: 1,
        tags: vec!["test".to_string()],
        io_budget: None,
    }];
    storage_config.metadata_url = format!("file://{}", metadata_path.display());

    let collection_service = Arc::new(CollectionService::new(storage_config).await.unwrap());

    let swift_engine = Arc::new(SwiftEngine::new().await.unwrap());

    (swift_engine, collection_service, temp_dir)
}

/// Create test vectors
/// REFACTORED: Now uses vector_generator::sequential() - much cleaner!
fn create_test_vectors(count: usize) -> Vec<ProximaRecord> {
    // OLD: 18 lines of manual vector construction
    // NEW: 1 line using test utility
    sequential("swift_test_collection", count, 128)
}

#[tokio::test]
async fn test_swift_engine_creation_and_insertion() {
    let (swift_engine, collection_service, _temp_dir) = create_test_setup().await;

    // REFACTORED: Use TestCollectionBuilder instead of manual construction
    // OLD: 30+ lines of boilerplate
    // NEW: 6 lines using test utility
    let (mut collection, _temp) = TestCollectionBuilder::new()
        .with_id("swift_test_collection")
        .with_name("swift_test_collection")
        .with_dimension(128)
        .with_engine(StorageEngine::Swift)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    // Override storage path to use the test setup's temp_dir
    if let Some(ref mut assignment) = collection.storage_assignment {
        assignment.primary_path = _temp_dir.path().to_str().unwrap().to_string();
        assignment.base_location = _temp_dir.path().to_str().unwrap().to_string();
    }

    // Update timestamps
    collection.created_at = chrono::Utc::now().timestamp();
    collection.updated_at = chrono::Utc::now().timestamp();
    if let Some(ref mut assignment) = collection.storage_assignment {
        assignment.assigned_at = chrono::Utc::now().timestamp();
    }

    // Extract config for collection service
    let config = collection.config.as_ref().unwrap().clone();
    collection_service.create_collection(&config).await.unwrap();

    let vectors = create_test_vectors(100);
    let batch_ids: Vec<BatchId> = (0..100).map(|_i| BatchId::new()).collect();

    let flush_params = FlushParameters {
        collection_id: Some("swift_test_collection".to_string()),
        vector_records: vectors,
        batch_ids,
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: Some(30000),
        trigger_compaction: false,
        collection_config: Some(collection),
        estimated_size: 1024 * 1024, // 1MB estimate
    };

    let flush_result = swift_engine.flush(flush_params).await.unwrap();
    assert!(flush_result.success);
    assert_eq!(flush_result.entries_flushed, Some(100));

    // Note: vector_by_id is a placeholder that returns None in SWIFT engine
    // The actual vector retrieval would require loading SWIFT files from disk
    // This test verifies that the flush operation completed successfully
}
