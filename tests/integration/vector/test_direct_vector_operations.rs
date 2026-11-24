//! Integration tests for VectorOperationsService operations
//!
//! Tests the complete vector lifecycle using the current VectorOperationsService API:
//! - Vector insertion (single and batch)
//! - Vector search (unified and streaming)
//! - Collection management
//! - Performance metrics

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;
#[path = "../common/mod.rs"]
mod common;



use std::sync::Arc;
use std::collections::HashMap;
use tempfile::TempDir;
use proximadb::utils::uuid::Uuid;

use proximadb::proto::proximadb_v1::VectorRecord;
use proximadb::proto::proximadb_v1::{
    CollectionConfig, DistanceMetric, StorageEngine, IndexingAlgorithm, MetadataItem
};
use proximadb::services::VectorOperationsService;
use proximadb::services::collection_service::CollectionService;
use proximadb::storage::persistence::filesystem::{FilesystemFactory, FilesystemConfig};
use proximadb::storage::persistence::write_ahead_log::WriteBufferManager;
use proximadb::storage::memtable::implementations::global_partitioned::GlobalPartitionedMemtable;

/// Helper to create test configuration
async fn create_test_services() -> (VectorOperationsService, CollectionService, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    
    // Create filesystem
    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
    
    // Create global memtable
    let memtable = Arc::new(GlobalPartitionedMemtable::new(
        16 * 1024 * 1024, // 16MB
        1000,             // 1000 partitions
        2 * 1024 * 1024,  // 2MB flush threshold
    ));
    
    // Create VectorOperationsService using test utilities
    let direct_vector_service = tests::common::integration_test_helpers::create_test_vector_operations_service()
        .await
        .expect("Failed to create VectorOperationsService");
    
    // Create CollectionService
    let collection_service = CollectionService::new(
        filesystem.clone(),
        temp_dir.path().to_path_buf(),
    );
    
    (direct_vector_service, collection_service, temp_dir)
}

/// Create test vectors with metadata
fn create_test_vectors(collection_id: &str, count: usize) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| {
            let vector = (0..128)
                .map(|j| (i * 128 + j) as f32 / (count * 128) as f32)
                .collect();
            
            let metadata = vec![
                MetadataItem {
                    key: "category".to_string(),
                    value: format!("category_{}", i % 3),
                },
                MetadataItem {
                    key: "score".to_string(),
                    value: Some(proximadb::proto::proximadb_v1::metadata_item::Value::StringValue((i as f64 / count as f64).to_string())),
                },
                MetadataItem {
                    key: "is_active".to_string(),
                    value: Some(proximadb::proto::proximadb_v1::metadata_item::Value::StringValue((i % 2 == 0).to_string())),
                },
            ];
            
            VectorRecord {
                id: Some(format!("vec_{,
            timestamp: 0,
            updated_at: None,
            expires_at: None,
            distance: None,
            rank: None,
            score: None,
        }", i)),
                vector,
                metadata,
                timestamp: chrono::Utc::now().timestamp() as u32,
                created_at: chrono::Utc::now().timestamp_millis(),
                updated_at: Some(chrono::Utc::now().timestamp() as u32),
                expires_at: None,
                distance: 0.0,
                rank: 0,
                score: 0.0,
            }
        })
        .collect()
}

/// Test basic vector insertion and search
#[tokio::test]
async fn test_basic_vector_operations() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Hnsw as i32,
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Insert test vectors
    let vectors = create_test_vectors("test_collection", 100);
    let vectors_arc = Arc::new(vectors);
    
    let sequences = direct_service
        .insert_vectors_direct("test_collection", vectors_arc.clone())
        .await
        .unwrap();
    
    assert_eq!(sequences.len(), 100);
    
    // Test search
    let query_vector = vec![0.5; 128];
    let search_results = direct_service
        .search_vectors_unified(
            "test_collection",
            &query_vector,
            10,
            DistanceMetric::Cosine,
            None,
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    assert!(search_results.vectors.len() <= 10);
    assert!(search_results.vectors.len() > 0);
    
    // Verify results are sorted by distance (lower = more similar)
    for i in 1..search_results.vectors.len() {
        assert!(search_results.vectors[i-1].distance <= search_results.vectors[i].distance);
    }
    
    // Verify vectors and metadata are included
    for result in &search_results.vectors {
        assert!(result.vector.len() == 128);
        assert!(result.metadata.len() > 0);
    }
}

/// Test batch vector insertion
#[tokio::test]
async fn test_batch_vector_insertion() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "batch_test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Euclidean as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Ivf as i32,
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Insert multiple batches
    let batch_size = 50;
    let num_batches = 5;
    let mut all_sequences = Vec::new();
    
    for batch_idx in 0..num_batches {
        let batch_vectors = create_test_vectors("batch_test_collection", batch_size);
        let batch_vectors_arc = Arc::new(batch_vectors);
        
        let sequences = direct_service
            .insert_vectors_direct("batch_test_collection", batch_vectors_arc)
            .await
            .unwrap();
        
        assert_eq!(sequences.len(), batch_size);
        all_sequences.extend(sequences);
    }
    
    // Verify all sequences are unique
    let mut unique_sequences = all_sequences.clone();
    unique_sequences.sort();
    unique_sequences.dedup();
    assert_eq!(unique_sequences.len(), all_sequences.len());
    
    // Test search across all batches
    let query_vector = vec![0.7; 128];
    let search_results = direct_service
        .search_vectors_unified(
            "batch_test_collection",
            &query_vector,
            20,
            DistanceMetric::Euclidean,
            None,
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    assert!(search_results.vectors.len() <= 20);
    assert!(search_results.vectors.len() > 0);
}

/// Test streaming search
#[tokio::test]
async fn test_streaming_search() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "streaming_test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Manhattan as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Hnsw as i32,
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Insert large dataset for streaming test
    let vectors = create_test_vectors("streaming_test_collection", 500);
    let vectors_arc = Arc::new(vectors);
    
    direct_service
        .insert_vectors_direct("streaming_test_collection", vectors_arc)
        .await
        .unwrap();
    
    // Test streaming search
    let query_vector = vec![0.3; 128];
    let streaming_config = Default::default(); // Use default streaming config
    
    let search_results = direct_service
        .search_vectors_streaming(
            "streaming_test_collection",
            &query_vector,
            30,
            DistanceMetric::Manhattan,
            streaming_config,
        )
        .await
        .unwrap();
    
    assert!(search_results.vectors.len() <= 30);
    assert!(search_results.vectors.len() > 0);
    
    // Verify streaming results are properly deduplicated
    let mut ids = HashSet::new();
    for result in &search_results.vectors {
        if let Some(id) = &result.id {
            assert!(!ids.contains(id), "Duplicate ID found in streaming results");
            ids.insert(id.clone());
        }
    }
}

/// Test metadata filtering
#[tokio::test]
async fn test_metadata_filtering() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "filter_test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Hnsw as i32,
        filterable_columns: vec![
            "category".to_string(),
            "score".to_string(),
            "is_active".to_string(),
        ],
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Insert test vectors with diverse metadata
    let vectors = create_test_vectors("filter_test_collection", 200);
    let vectors_arc = Arc::new(vectors);
    
    direct_service
        .insert_vectors_direct("filter_test_collection", vectors_arc)
        .await
        .unwrap();
    
    // Test category filter
    let category_filter = HashMap::from([
        ("category".to_string(), serde_json::Value::String("category_1".to_string())),
    ]);
    
    let query_vector = vec![0.4; 128];
    let filtered_results = direct_service
        .search_vectors_unified(
            "filter_test_collection",
            &query_vector,
            15,
            DistanceMetric::Cosine,
            Some(category_filter),
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    // Verify all results match the filter
    for result in &filtered_results.vectors {
        let category = result.metadata.iter()
            .find(|item| item.key == "category")
            .map(|item| &item.value)
            .unwrap();
        assert_eq!(category, "category_1");
    }
    
    // Test score range filter
    let score_filter = HashMap::from([
        ("score".to_string(), serde_json::json!({
            "$gte": 0.5,
            "$lte": 0.8
        })),
    ]);
    
    let score_filtered_results = direct_service
        .search_vectors_unified(
            "filter_test_collection",
            &query_vector,
            15,
            DistanceMetric::Cosine,
            Some(score_filter),
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    // Verify all results are in score range
    for result in &score_filtered_results.vectors {
        let score_str = result.metadata.iter()
            .find(|item| item.key == "score")
            .map(|item| &item.value)
            .unwrap();
        let score: f64 = score_str.parse().unwrap();
        assert!(score >= 0.5 && score <= 0.8);
    }
}

/// Test flush operations
#[tokio::test]
async fn test_flush_operations() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "flush_test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::DotProduct as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Pq as i32,
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Insert vectors
    let vectors = create_test_vectors("flush_test_collection", 100);
    let vectors_arc = Arc::new(vectors);
    
    direct_service
        .insert_vectors_direct("flush_test_collection", vectors_arc)
        .await
        .unwrap();
    
    // Force flush specific collection
    direct_service
        .force_flush_collection("flush_test_collection")
        .await
        .unwrap();
    
    // Verify data is still searchable after flush
    let query_vector = vec![0.6; 128];
    let search_results = direct_service
        .search_vectors_unified(
            "flush_test_collection",
            &query_vector,
            10,
            DistanceMetric::DotProduct,
            None,
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    assert!(search_results.vectors.len() > 0);
    
    // Test flush all collections
    direct_service.force_flush_all().await.unwrap();
    
    // Verify data is still searchable after flush all
    let search_results_after_flush = direct_service
        .search_vectors_unified(
            "flush_test_collection",
            &query_vector,
            10,
            DistanceMetric::DotProduct,
            None,
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    assert!(search_results_after_flush.vectors.len() > 0);
}

/// Test metrics and health check
#[tokio::test]
async fn test_metrics_and_health() {
    setup_hardware_capabilities();
    let (direct_service, collection_service, _temp_dir) = create_test_services().await;
    
    // Create test collection
    let config = CollectionConfig {
        name: "metrics_test_collection".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine as i32,
        storage_engine: StorageEngine::Sst as i32,
        primary_indexing_algorithm: IndexingAlgorithm::Hnsw as i32,
        ..Default::default()
                compression: None,
                optimization_hints: None,
            };
    
    collection_service.create_collection(&config).await.unwrap();
    
    // Perform some operations to generate metrics
    let vectors = create_test_vectors("metrics_test_collection", 50);
    let vectors_arc = Arc::new(vectors);
    
    direct_service
        .insert_vectors_direct("metrics_test_collection", vectors_arc)
        .await
        .unwrap();
    
    let query_vector = vec![0.5; 128];
    direct_service
        .search_vectors_unified(
            "metrics_test_collection",
            &query_vector,
            10,
            DistanceMetric::Cosine,
            None,
            None,
            true,
            true,
        )
        .await
        .unwrap();
    
    // Test health check
    let health_status = direct_service.health_check().await.unwrap();
    assert!(health_status.is_healthy);
    
    // Test metrics
    let metrics = direct_service.get_metrics().await.unwrap();
    assert!(metrics.contains_key("insert_count"));
    assert!(metrics.contains_key("search_count"));
    assert!(metrics.contains_key("memtable_size"));
    
    // Verify metrics have reasonable values
    let insert_count = metrics.get("enable_two_stage_search").unwrap();
    assert!(*insert_count >= 50.0);
    
    let search_count = metrics.get("enable_two_stage_search").unwrap();
    assert!(*search_count >= 1.0);
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() {
    setup_hardware_capabilities();
    let (direct_service, _collection_service, _temp_dir) = create_test_services().await;
    
    // Test search on non-existent collection
    let query_vector = vec![0.5; 128];
    let result = direct_service
        .search_vectors_unified(
            "non_existent_collection",
            &query_vector,
            10,
            DistanceMetric::Cosine,
            None,
            None,
            true,
            true,
        )
        .await;
    
    assert!(result.is_err());
    
    // Test insert on non-existent collection
    let vectors = create_test_vectors("non_existent_collection", 10);
    let vectors_arc = Arc::new(vectors);
    
    let result = direct_service
        .insert_vectors_direct("non_existent_collection", vectors_arc)
        .await;
    
    assert!(result.is_err());
    
    // Test flush on non-existent collection
    let result = direct_service
        .force_flush_collection("non_existent_collection")
        .await;
    
    assert!(result.is_err());
}

use std::collections::HashSet;
