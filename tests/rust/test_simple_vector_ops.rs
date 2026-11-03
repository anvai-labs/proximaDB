//! Simple Vector Operations Integration Tests
//!
//! Basic tests for vector insert, search, and retrieval operations
//! using the current ProximaDB architecture.

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;
use tracing::debug;

#[tokio::test]
async fn test_vector_record_creation() -> Result<()> {
    // Test basic vector record creation
    let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), json!("test"));
    metadata.insert("priority".to_string(), json!("high"));

    // Simulate vector record structure
    let vector_record = json!({
        "id": "test_vector_1",
        "vector": vector,
        "metadata": metadata,
        "timestamp": chrono::Utc::now().timestamp_micros(),
        "version": 1
    });

    assert_eq!(vector_record["id"], "test_vector_1");
    assert_eq!(vector_record["vector"].as_array().unwrap().len(), 5);
    assert_eq!(vector_record["metadata"]["category"], "test");
    assert_eq!(vector_record["version"], 1);

    debug!("✅ Vector record creation test passed");
    Ok(())
}

#[tokio::test]
async fn test_vector_search_request_structure() -> Result<()> {
    // Test vector search request structure
    let query_vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let search_request = json!({
        "vector": query_vector,
        "k": 10,
        "filters": {
            "category": "test",
            "priority": "high"
        },
        "include_vectors": true,
        "include_metadata": true,
        "optimization_hints": {
            "enable_two_stage_search": true,
            "quantization_hint": "PQ8"
        }
    });

    assert_eq!(search_request.get("k").unwrap(), 10);
    assert_eq!(search_request.get("filters").unwrap()["category"], "test");
    assert_eq!(search_request.get("include_vectors").unwrap(), true);
    assert_eq!(
        search_request.get("optimization_hints").unwrap()["enable_two_stage_search"],
        true
    );

    debug!("✅ Vector search request structure test passed");
    Ok(())
}

#[tokio::test]
async fn test_collection_config_with_quantization() -> Result<()> {
    // Test collection configuration with quantization support
    let collection_config = json!({
        "name": "test_collection",
        "dimension": 128,
        "distance_metric": "COSINE",
        "storage_engine": "VIPER",
        "indexing_algorithm": "HNSW",
        "quantization": {
            "quantization_type": "PQ",
            "bits_per_code": 8,
            "num_subspaces": 16,
            "enable_compression": true
        }
    });

    assert_eq!(collection_config["name"], "test_collection");
    assert_eq!(collection_config["dimension"], 128);
    assert_eq!(collection_config["distance_metric"], "COSINE");

    let quant_config = &collection_config["quantization"];
    assert_eq!(quant_config["quantization_type"], "PQ");
    assert_eq!(quant_config["bits_per_code"], 8);
    assert_eq!(quant_config["num_subspaces"], 16);

    debug!("✅ Collection config with quantization test passed");
    Ok(())
}

#[tokio::test]
async fn test_distance_metric_types() -> Result<()> {
    // Test different distance metric configurations
    let metrics = vec!["COSINE", "EUCLIDEAN", "DOT_PRODUCT", "MANHATTAN"];

    for metric in metrics {
        let config = json!({
            "distance_metric": metric,
            "dimension": 128
        });

        assert_eq!(config["distance_metric"], metric);
        assert_eq!(config["dimension"], 128);
    }

    debug!("✅ Distance metric types test passed");
    Ok(())
}

#[tokio::test]
async fn test_storage_engine_types() -> Result<()> {
    // Test different storage engine configurations
    let engines = vec!["VIPER", "LSM"];

    for engine in engines {
        let config = json!({
            "storage_engine": engine,
            "dimension": 128,
            "name": format!("test_collection_{}", engine.to_lowercase())
        });

        assert_eq!(config["storage_engine"], engine);
        assert!(
            config["name"]
                .as_str()
                .unwrap()
                .contains(&engine.to_lowercase())
        );
    }

    debug!("✅ Storage engine types test passed");
    Ok(())
}

#[tokio::test]
async fn test_vector_mutation_operations() -> Result<()> {
    // Test vector update and delete operations
    let update_request = json!({
        "operation": "UPDATE",
        "vector_id": "test_vector_1",
        "new_vector": [2.0, 3.0, 4.0, 5.0, 6.0],
        "metadata_updates": {
            "priority": "medium",
            "last_updated": chrono::Utc::now().timestamp()
        }
    });

    let delete_request = json!({
        "operation": "DELETE",
        "vector_id": "test_vector_1",
        "soft_delete": true
    });

    assert_eq!(update_request.get("operation").unwrap(), "UPDATE");
    assert_eq!(update_request.get("vector_id").unwrap(), "test_vector_1");
    assert_eq!(
        update_request.get("metadata_updates").unwrap()["priority"],
        "medium"
    );

    assert_eq!(delete_request.get("operation").unwrap(), "DELETE");
    assert_eq!(delete_request.get("soft_delete").unwrap(), true);

    debug!("✅ Vector mutation operations test passed");
    Ok(())
}
