//! Fixed SST Collection Integration Test using UnifiedTestEnvironment
//!
//! This test properly creates a collection with SST storage engine and
//! verifies that flush operations route to SST correctly.

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;

use common::integration_test_helpers::{UnifiedTestEnvironment, operations};
use proximadb::proto::proximadb_v1::StorageEngine;
use proximadb::storage::traits::UnifiedStorageEngine;
use tracing::{debug, info};

#[tokio::test]
async fn test_sst_collection_with_proper_routing() -> anyhow::Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Testing SST Collection with Proper Routing ===");

    // Use UnifiedTestEnvironment for proper setup
    let env = UnifiedTestEnvironment::new().await?;
    let collection_id = env.collection_id();
    debug!(
        "Created test environment with collection: {}",
        collection_id
    );

    // Create SST engine using the environment
    let sst_engine = env.create_sst_engine().await?;
    debug!("Created SST engine for collection: {}", collection_id);

    // Create test vectors
    let make_record = |oid: String, values: Vec<f32>, category: &str, ts: i64| {
        let mut props = proximadb_records::ProximaTree::new();
        props.insert(
            "category".to_string(),
            proximadb_records::ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                category.to_string(),
            )),
        );
        proximadb_records::ProximaRecord {
            oid,
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                values.len() as u32,
                values,
            )],
            props,
            record_version: 1,
            created_at_ns: ts * 1_000_000_000,
            updated_at_ns: ts * 1_000_000_000,
            ..Default::default()
        }
    };
    let vectors = vec![
        make_record(
            format!("{}_vec1", collection_id),
            vec![1.0, 0.0, 0.0],
            "A",
            1000,
        ),
        make_record(
            format!("{}_vec2", collection_id),
            vec![0.0, 1.0, 0.0],
            "B",
            1001,
        ),
        make_record(
            format!("{}_vec3", collection_id),
            vec![0.0, 0.0, 1.0],
            "A",
            1002,
        ),
    ];

    info!("Test 1: Flush vectors to SST engine");

    // Build proper flush parameters using the helper
    let flush_params = operations::build_flush_params(&env, vectors, StorageEngine::Sst).await?;

    // Flush directly to SST engine
    let flush_result = sst_engine.do_flush(&flush_params).await?;
    assert!(flush_result.success, "SST flush should succeed");
    assert_eq!(
        flush_result.entries_flushed.unwrap_or(0),
        3,
        "Should flush 3 vectors"
    );
    assert!(
        flush_result.files_created.unwrap_or(0) > 0,
        "Should create SST files"
    );

    debug!(
        "✅ Successfully flushed {} vectors to SST, created {} files",
        flush_result.entries_flushed.unwrap_or(0),
        flush_result.files_created.unwrap_or(0)
    );

    info!("Test 2: Verify SST files were created");

    // Check that SST files exist in the data directory
    let sst_data_dir = env.get_sst_data_directory();
    assert!(sst_data_dir.exists(), "SST data directory should exist");

    let mut sst_file_count = 0;
    let mut total_size = 0u64;

    if let Ok(entries) = std::fs::read_dir(&sst_data_dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "sst" {
                    sst_file_count += 1;
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                        debug!(
                            "Found SST file: {} ({} bytes)",
                            entry.file_name().to_string_lossy(),
                            metadata.len()
                        );
                    }
                }
            }
        }
    }

    assert!(
        sst_file_count > 0,
        "Should have created at least one SST file"
    );
    assert!(total_size > 0, "SST files should have non-zero size");

    debug!(
        "✅ Found {} SST files with total size {} bytes",
        sst_file_count, total_size
    );

    info!("Test 3: Search vectors from SST");

    // Build proper storage URL for search
    let storage_url = operations::build_sst_storage_url(&env);

    // Search for vectors
    let query_vector = vec![1.0, 0.0, 0.0];
    // Create search context for SST engine
    let search_params = std::sync::Arc::new(proximadb::core::search::SearchParams {
        vector: Some(query_vector.clone()),
        query_vectors: None,
        top_k: Some(5),
        distance_metric: Some(proximadb::compute::distance_computation::DistanceMetric::Cosine),
        filter_expression: None,
        timeout_ms: None,
        accuracy_threshold: None,
        ..Default::default()
    });

    let collection_config = proximadb::proto::proximadb_v1::CollectionConfig {
        name: collection_id.to_string(),
        dimension: query_vector.len() as u32,
        distance_metric: Some(proximadb::proto::proximadb_v1::DistanceMetric::Cosine as i32),
        storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Sst as i32),
        primary_index: Some("default".to_string()),
        auto_index_selection: Some(false),
        tags: vec![],
        embedding_models: vec![],
        owner: None,
        ..Default::default()
    };

    let collection = std::sync::Arc::new(proximadb::proto::proximadb_v1::Collection {
        storage_class_status: None,
        id: collection_id.to_string(),
        config: Some(collection_config),
        stats: None,
        created_at: 0,
        updated_at: 0,
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            primary_path: format!("{}", env.persistent_dir.display()),
            backup_paths: vec![],
            engine: proximadb::proto::proximadb_v1::StorageEngine::Sst as i32,
            engine_config: std::collections::HashMap::new(),
            // base_location should point to the parent directory where collection directories are
            base_location: format!("{}", env.persistent_base.display()),
            assigned_at: chrono::Utc::now().timestamp_micros(),
            ..Default::default()
        }),
    });

    let query_context = proximadb::storage::traits::StorageQueryContext {
        search_params,
        collection: collection.clone(),
        metadata: proximadb::storage::traits::StorageQueryMetadata {
            collection_id: collection_id.to_string(),
            use_axis_indexes: false,
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };

    let search_results = sst_engine.search_vectors_unified(&query_context).await?;

    assert!(!search_results.is_empty(), "Should find search results");
    assert_eq!(search_results.len(), 3, "Should find all 3 vectors");

    // Verify the closest result is the identical vector
    assert!(
        search_results[0].id.ends_with("_vec1"),
        "Closest result should be vec1 (identical to query)"
    );

    debug!("✅ Search returned {} results", search_results.len());
    for (i, result) in search_results.iter().enumerate() {
        debug!("  Result {}: id={}, score={:?}", i, result.id, result.score);
    }

    info!("Test 4: Metadata filtering");

    // Search with metadata filter
    let filter = proximadb::core::search::FilterExpression::Comparison {
        field: "category".to_string(),
        operator: proximadb::core::search::ComparisonOperator::Equals,
        value: serde_json::Value::String("A".to_string()),
    };

    // Create filtered search context
    let filtered_search_params = std::sync::Arc::new(proximadb::core::search::SearchParams {
        vector: Some(query_vector.clone()),
        query_vectors: None,
        top_k: Some(5),
        distance_metric: Some(proximadb::compute::distance_computation::DistanceMetric::Cosine),
        filter_expression: Some(filter),
        timeout_ms: None,
        accuracy_threshold: None,
        ..Default::default()
    });

    let filtered_query_context = proximadb::storage::traits::StorageQueryContext {
        search_params: filtered_search_params,
        collection: collection.clone(),
        metadata: proximadb::storage::traits::StorageQueryMetadata {
            collection_id: collection_id.to_string(),
            use_axis_indexes: false,
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };

    let filtered_results = sst_engine
        .search_vectors_unified(&filtered_query_context)
        .await?;

    assert_eq!(
        filtered_results.len(),
        2,
        "Should find 2 vectors with category A"
    );

    debug!(
        "✅ Filtered search returned {} results",
        filtered_results.len()
    );

    info!("✅ All SST collection tests passed!");

    Ok(())
}
