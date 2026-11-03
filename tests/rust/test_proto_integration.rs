//! Protocol Integration Tests
//!
//! Tests to verify that the new proto definitions work correctly
//! with the server implementation and quantization features.

use anyhow::Result;
use proximadb::proto::proximadb_v1::StorageEngine as StorageEngineType;
use serde_json::json;
use tracing::debug;

// Import test utilities - accessible from tests/common/
#[path = "../common/collection_builder.rs"]
mod collection_builder;
use collection_builder::TestCollectionBuilder;

#[tokio::test]
async fn test_quantization_config_fields() -> Result<()> {
    // Test that quantization field exists and can be used
    use proximadb::proto::proximadb_v1::*;

    // BEFORE: 15+ lines of boilerplate CollectionConfig creation
    // AFTER: Use the test utility builder
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_name("basic_collection")
        .with_dimension(128)
        .with_engine(StorageEngineType::Viper)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let basic_config = collection.config.as_ref().unwrap();

    assert_eq!(basic_config.name, "basic_collection");
    assert!(basic_config.quantization.is_none());

    debug!("✅ Quantization config field exists in CollectionConfig");
    Ok(())
}

#[tokio::test]
async fn test_index_config_field() -> Result<()> {
    // Test that index_config field exists and can be used
    use proximadb::proto::proximadb_v1::*;

    // Using test utility - much cleaner!
    let (collection, _temp) = TestCollectionBuilder::new()
        .with_name("indexed_collection")
        .with_dimension(256)
        .with_engine(StorageEngineType::Viper)
        .with_distance_metric(DistanceMetric::Cosine)
        .build();

    let config_with_index = collection.config.as_ref().unwrap();

    assert_eq!(config_with_index.name, "indexed_collection");
    assert!(config_with_index.index_configs.is_empty());

    debug!("✅ Index config field exists in CollectionConfig");
    Ok(())
}

#[tokio::test]
async fn test_search_optimization_hints_field() -> Result<()> {
    // Test that optimization_hints field exists in VectorSearchRequest
    use proximadb::proto::proximadb_v1::*;

    // Create a search query
    let search_query = SearchQuery {
        vector: vec![1.0, 2.0, 3.0, 4.0],
        filters: std::collections::HashMap::new(),
        advanced_filter: None,
    };

    let search_request = VectorSearchRequest {
        collection_id: "test_collection".to_string(),
        queries: vec![search_query],
        top_k: 10,
        distance_metric_override: None,
        search_params: None,
        include_fields: Some(IncludeFields::default()),
        search_optimization: None, // This field should exist from our proto updates
    };

    assert_eq!(search_request.collection_id, "test_collection");
    assert_eq!(search_request.queries.len(), 1);
    assert_eq!(search_request.queries[0].vector.len(), 4);
    assert_eq!(search_request.top_k, 10);
    assert!(search_request.search_optimization.is_none());

    debug!("✅ Search optimization hints field exists in VectorSearchRequest");
    Ok(())
}

#[tokio::test]
async fn test_quantization_message_types_exist() -> Result<()> {
    // Test that the quantization message types are generated and accessible
    use proximadb::proto::proximadb_v1::*;

    // This test verifies that our proto quantization messages were generated correctly
    // Even if we can't test complex nested structures, we should be able to reference the types

    // Let's create a simple test that proves the types exist
    let _ = std::any::type_name::<QuantizationConfig>();
    let _ = std::any::type_name::<SearchParams>();

    // If these types don't exist, the compilation will fail
    debug!(
        "✅ Quantization message types exist: {:?}",
        std::any::type_name::<QuantizationConfig>()
    );
    debug!(
        "✅ Search params type exists: {:?}",
        std::any::type_name::<SearchParams>()
    );

    Ok(())
}

#[tokio::test]
async fn test_handler_hint_processing() -> Result<()> {
    // Test that optimization hints can be processed by handlers
    // This simulates what the gRPC and REST handlers do

    let hints_json = json!({
        "enable_two_stage_search": true,
        "candidate_multiplier": 2.0,
        "min_candidates": 100,
        "max_candidates": 1000,
        "accuracy_threshold": 0.95,
        "quantization_hint": "PQ8"
    });

    // Simulate gRPC handler processing
    let enable_two_stage = hints_json
        .get("enable_two_stage_search")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let candidate_multiplier = hints_json
        .get("candidate_multiplier")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let quantization_hint = hints_json
        .get("quantization_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("FP32")
        .to_string();

    // Create processed hints JSON as handlers do
    let processed_hints = json!({
        "enable_two_stage_search": enable_two_stage,
        "candidate_multiplier": candidate_multiplier,
        "quantization_hint": quantization_hint,
        "min_candidates": hints_json.get("min_candidates"),
        "max_candidates": hints_json.get("max_candidates"),
        "accuracy_threshold": hints_json.get("accuracy_threshold")
    });

    assert_eq!(processed_hints["enable_two_stage_search"], true);
    assert_eq!(processed_hints["candidate_multiplier"], 2.0);
    assert_eq!(processed_hints["quantization_hint"], "PQ8");

    debug!("✅ Handler hint processing test passed");
    Ok(())
}
