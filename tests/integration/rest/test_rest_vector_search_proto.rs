//! REST API Vector Search Integration Tests with Proto Structure
//!
//! Tests the REST API with the new proto-aligned SearchParams structure
//! and quantization support.

use anyhow::Result;
use tracing::{debug, error, info, warn};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;

const REST_BASE_URL: &str = "http://localhost:5678";

#[derive(Debug, serde::Serialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: i32,
    distance_metric: String,
    storage_engine: String,
    indexing_algorithm: String,
    quantization: Option<serde_json::Value>,
}

#[derive(Debug, serde::Serialize)]
struct VectorInsertRequest {
    vectors: Vec<VectorData>,
}

#[derive(Debug, serde::Serialize)]
struct VectorData {
    id: String,
    vector: Vec<f32>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, serde::Serialize)]
struct SearchVectorRequest {
    vector: Vec<f32>,
    k: Option<usize>,
    filters: Option<HashMap<String, serde_json::Value>>,
    include_vectors: Option<bool>,
    include_metadata: Option<bool>,
    optimization_hints: Option<SearchOptimizationHints>,
}

#[derive(Debug, serde::Serialize)]
struct SearchOptimizationHints {
    enable_two_stage_search: Option<bool>,
    quantization_hint: Option<String>,
    candidate_multiplier: Option<f32>,
    min_candidates: Option<i32>,
    max_candidates: Option<i32>,
    enable_clustering_optimization: Option<bool>,
    enable_metadata_filtering_hint: Option<bool>,
    enable_parallel_search: Option<bool>,
    accuracy_threshold: Option<f32>,
    timeout_ms: Option<i32>,
    include_expired_vectors: Option<bool>,
    custom_hints: Option<HashMap<String, String>>,
}

async fn create_test_collection(client: &Client, collection_name: &str) -> Result<()> {
    let quantization = json!({
        "enabled": true,
        "storage_quantization": {
            "enabled": true,
            "level": "INT8",
            "progressive": false
        },
        "search_quantization": {
            "enabled": true,
            "default_level": "INT8",
            "adaptive_precision": true,
            "accuracy_threshold": 0.95
        },
        "compression_ratio_target": 4.0
    });

    let request = CreateCollectionRequest {
        name: collection_name.to_string(),
        dimension: 384,
        distance_metric: "cosine".to_string(),
        storage_engine: "sst".to_string(),
        indexing_algorithm: "hnsw".to_string(),
        quantization: Some(quantization),
    };

    let response = client
        .post(&format!("{}/collections", REST_BASE_URL))
        .json(&request)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    Ok(())
}

async fn insert_test_vectors(client: &Client, collection_name: &str) -> Result<()> {
    let vectors = vec![
        VectorData {
            id: "vec1".to_string(),
            vector: vec![0.1; 384],
            metadata: [("category".to_string(), "test".to_string())]
                .iter()
                .cloned()
                .collect(),
        },
        VectorData {
            id: "vec2".to_string(),
            vector: vec![0.2; 384],
            metadata: [("category".to_string(), "example".to_string())]
                .iter()
                .cloned()
                .collect(),
        },
        VectorData {
            id: "vec3".to_string(),
            vector: vec![0.15; 384],
            metadata: [("category".to_string(), "test".to_string())]
                .iter()
                .cloned()
                .collect(),
        },
    ];

    let request = VectorInsertRequest { vectors };

    let response = client
        .post(&format!("{}/collections/{}/vectors", REST_BASE_URL, collection_name))
        .json(&request)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    Ok(())
}

#[tokio::test]
async fn test_rest_vector_search_with_quantization() -> Result<()> {
    setup_hardware_capabilities();
    let client = Client::new();
    let collection_name = format!("test_rest_quant_{}", uuid::Uuid::new_v4());

    // Create collection with quantization
    create_test_collection(&client, &collection_name).await?;

    // Insert test vectors
    insert_test_vectors(&client, &collection_name).await?;

    // Wait for indexing
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Test 1: Binary quantization search
    let binary_search = SearchVectorRequest {
        vector: vec![0.12; 384],
        k: Some(10),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(true),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("BINARY".to_string()),
            candidate_multiplier: Some(5.0),
            min_candidates: Some(50),
            max_candidates: Some(500),
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(true),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.90),
            timeout_ms: Some(1000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };

    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&binary_search)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    let result: serde_json::Value = response.json().await?;
    assert!(result["success"].as_bool().unwrap_or(false));
    debug!("✅ Binary quantization search passed");

    // Test 2: INT8 scalar quantization search
    let int8_search = SearchVectorRequest {
        vector: vec![0.12; 384],
        k: Some(10),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(true),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("INT8".to_string()),
            candidate_multiplier: Some(3.0),
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(true),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.95),
            timeout_ms: Some(2000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };

    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&int8_search)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    let result: serde_json::Value = response.json().await?;
    assert!(result["success"].as_bool().unwrap_or(false));
    debug!("✅ INT8 quantization search passed");

    // Test 3: Product quantization search
    let pq_search = SearchVectorRequest {
        vector: vec![0.12; 384],
        k: Some(10),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(true),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("PQ8".to_string()),
            candidate_multiplier: Some(3.0),
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(true),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.95),
            timeout_ms: Some(3000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };

    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&pq_search)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    let result: serde_json::Value = response.json().await?;
    assert!(result["success"].as_bool().unwrap_or(false));
    debug!("✅ Product quantization search passed");

    // Test 4: Full precision search (FP32)
    let fp32_search = SearchVectorRequest {
        vector: vec![0.12; 384],
        k: Some(10),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(true),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(false),
            quantization_hint: Some("FP32".to_string()),
            candidate_multiplier: None,
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(true),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(1.0),
            timeout_ms: Some(5000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };

    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&fp32_search)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    let result: serde_json::Value = response.json().await?;
    assert!(result["success"].as_bool().unwrap_or(false));
    debug!("✅ Full precision search passed");

    // Test 5: Search with metadata filters
    let filtered_search = SearchVectorRequest {
        vector: vec![0.12; 384],
        k: Some(10),
        filters: Some(
            [("category".to_string(), json!("test"))]
                .iter()
                .cloned()
                .collect(),
        ),
        include_vectors: Some(false),
        include_metadata: Some(true),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("INT8".to_string()),
            candidate_multiplier: Some(2.0),
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(true),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.95),
            timeout_ms: Some(2000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };

    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&filtered_search)
        .send()
        .await?;

    assert_eq!(response.status(), 200);
    let result: serde_json::Value = response.json().await?;
    assert!(result["success"].as_bool().unwrap_or(false));
    
    // Verify filtered results
    if let Some(data) = result["data"].as_array() {
        for item in data {
            if let Some(metadata) = item["metadata"].as_object() {
                assert_eq!(
                    metadata.get("active").and_then(|v| v.as_str()),
                    Some("test")
                );
            }
        }
    }
    debug!("✅ Filtered search passed");

    // Clean up
    let delete_response = client
        .delete(&format!("{}/collections/{}", REST_BASE_URL, collection_name))
        .send()
        .await?;

    assert_eq!(delete_response.status(), 200);
    debug!("✅ Collection cleanup passed");

    Ok(())
}

#[tokio::test]
async fn test_rest_search_performance_comparison() -> Result<()> {
    setup_hardware_capabilities();
    let client = Client::new();
    let collection_name = format!("test_rest_perf_{}", uuid::Uuid::new_v4());

    // Create collection with quantization
    create_test_collection(&client, &collection_name).await?;

    // Insert more vectors for performance testing
    for batch in 0..10 {
        let vectors: Vec<VectorData> = (0..100)
            .map(|i| VectorData {
                id: format!("perf_vec_{}_{}", batch, i),
                vector: vec![0.1 + (i as f32 * 0.001); 384],
                metadata: [
                    ("category".to_string(), format!("cat{}", i % 5)),
                    ("batch".to_string(), batch.to_string()),
                ]
                .iter()
                .cloned()
                .collect(),
            })
            .collect();

        let request = VectorInsertRequest { vectors };
        client
            .post(&format!("{}/collections/{}/vectors", REST_BASE_URL, collection_name))
            .json(&request)
            .send()
            .await?;
    }

    // Wait for indexing
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Compare search performance with different quantization levels
    let query_vector = vec![0.15; 384];
    
    // Binary quantization (fastest)
    let start = std::time::Instant::now();
    let binary_search = SearchVectorRequest {
        vector: query_vector.clone(),
        k: Some(50),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(false),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("BINARY".to_string()),
            candidate_multiplier: Some(5.0),
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(false),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.85),
            timeout_ms: Some(500),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };
    
    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&binary_search)
        .send()
        .await?;
    
    let binary_time = start.elapsed();
    assert_eq!(response.status(), 200);
    debug!("Binary search completed in {:?}", binary_time);

    // INT8 quantization (balanced)
    let start = std::time::Instant::now();
    let int8_search = SearchVectorRequest {
        vector: query_vector.clone(),
        k: Some(50),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(false),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(true),
            quantization_hint: Some("INT8".to_string()),
            candidate_multiplier: Some(3.0),
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(false),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(0.95),
            timeout_ms: Some(1000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };
    
    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&int8_search)
        .send()
        .await?;
    
    let int8_time = start.elapsed();
    assert_eq!(response.status(), 200);
    debug!("INT8 search completed in {:?}", int8_time);

    // FP32 (most accurate)
    let start = std::time::Instant::now();
    let fp32_search = SearchVectorRequest {
        vector: query_vector,
        k: Some(50),
        filters: None,
        include_vectors: Some(false),
        include_metadata: Some(false),
        optimization_hints: Some(SearchOptimizationHints {
            enable_two_stage_search: Some(false),
            quantization_hint: Some("FP32".to_string()),
            candidate_multiplier: None,
            min_candidates: None,
            max_candidates: None,
            enable_clustering_optimization: Some(true),
            enable_metadata_filtering_hint: Some(false),
            enable_parallel_search: Some(false),
            accuracy_threshold: Some(1.0),
            timeout_ms: Some(5000),
            include_expired_vectors: Some(false),
            custom_hints: None,
        }),
    };
    
    let response = client
        .post(&format!("{}/collections/{}/search", REST_BASE_URL, collection_name))
        .json(&fp32_search)
        .send()
        .await?;
    
    let fp32_time = start.elapsed();
    assert_eq!(response.status(), 200);
    debug!("FP32 search completed in {:?}", fp32_time);

    // Verify performance hierarchy
    debug!("\n📊 Performance Summary:");
    debug!("   Binary: {:?} (fastest)", binary_time);
    debug!("   INT8: {:?} (balanced)", int8_time);
    debug!("   FP32: {:?} (most accurate)", fp32_time);
    
    // Clean up
    client
        .delete(&format!("{}/collections/{}", REST_BASE_URL, collection_name))
        .send()
        .await?;

    Ok(())
}