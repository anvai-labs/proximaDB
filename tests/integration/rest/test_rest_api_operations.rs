//! Integration tests for REST API operations
//!
//! Tests the complete REST API functionality using the current unified handlers:
//! - Collection operations (create, read, update, delete)
//! - Vector batch operations
//! - Vector search operations
//! - Health and metrics endpoints

// Import the common test helpers
#[path = "../common/mod.rs"]
mod common;
#[path = "../common/mod.rs"]
mod common;



use std::collections::HashMap;
use serde_json::json;
use tempfile::TempDir;
use tokio::net::TcpListener;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::Response,
    Router,
};
use tower::ServiceExt;

use proximadb::network::rest::create_rest_router;
use proximadb::services::VectorOperationsService;
use proximadb::services::collection_service::CollectionService;
use proximadb::storage::persistence::filesystem::{FilesystemFactory, FilesystemConfig};
use proximadb::storage::memtable::implementations::global_partitioned::GlobalPartitionedMemtable;
use proximadb::proto::proximadb_v1::{
    CollectionConfig, DistanceMetric, StorageEngine, IndexingAlgorithm
};
use std::sync::Arc;

/// Test setup helper
async fn create_test_app() -> (Router, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    
    // Create filesystem
    let filesystem_config = FilesystemConfig::default();
    let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
    
    // Create memtable
    let memtable = Arc::new(GlobalPartitionedMemtable::new(
        16 * 1024 * 1024, // 16MB
        1000,             // 1000 partitions
        2 * 1024 * 1024,  // 2MB flush threshold
    ));
    
    // Create services using test utilities
    let direct_vector_service = Arc::new(
        tests::common::integration_test_helpers::create_test_vector_operations_service()
            .await
            .expect("Failed to create VectorOperationsService")
    );
    
    let collection_service = Arc::new(CollectionService::new(
        filesystem.clone(),
        temp_dir.path().to_path_buf(),
    ));
    
    // Create router
    let app = create_rest_router(
        direct_vector_service,
        collection_service,
    ).await;
    
    (app, temp_dir)
}

/// Helper to make HTTP requests
async fn make_request(
    app: &Router,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let mut request_builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    
    let body_bytes = if let Some(body) = body {
        serde_json::to_vec(&body).unwrap()
    } else {
        vec![]
    };
    
    let request = request_builder.body(Body::from(body_bytes)).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    
    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: serde_json::Value = if body_bytes.is_empty() {
        json!({})
    } else {
        serde_json::from_slice(&body_bytes).unwrap_or_else(|_| {
            json!({ "raw": String::from_utf8_lossy(&body_bytes) })
        })
    };
    
    (status, response_json)
}

/// Test collection operations
#[tokio::test]
async fn test_collection_operations() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Test create collection
    let create_request = json!({
        "name": "test_collection",
        "dimension": 128,
        "distance_metric": "COSINE",
        "storage_engine": "SST",
        "primary_indexing_algorithm": "HNSW",
        "filterable_columns": ["category", "score"]
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/collection", Some(create_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["collection_id"].is_string());
    
    let collection_id = response["collection_id"].as_str().unwrap();
    
    // Test get collection
    let get_path = format!("/api/v1/collection/{}", collection_id);
    let (status, response) = make_request(&app, "GET", &get_path, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["name"], "test_collection");
    assert_eq!(response["dimension"], 128);
    
    // Test list collections
    let (status, response) = make_request(&app, "GET", "/api/v1/collection", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["collections"].is_array());
    assert!(response["collections"].as_array().unwrap().len() >= 1);
    
    // Test update collection
    let update_request = json!({
        "name": "updated_test_collection",
        "filterable_columns": ["category", "score", "is_active"]
    });
    
    let (status, response) = make_request(&app, "PUT", &get_path, Some(update_request)).await;
    assert_eq!(status, StatusCode::OK);
    
    // Verify update
    let (status, response) = make_request(&app, "GET", &get_path, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["name"], "updated_test_collection");
    assert_eq!(response["filterable_columns"].as_array().unwrap().len(), 3);
    
    // Test delete collection
    let (status, _response) = make_request(&app, "DELETE", &get_path, None).await;
    assert_eq!(status, StatusCode::OK);
    
    // Verify deletion
    let (status, _response) = make_request(&app, "GET", &get_path, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Test vector batch operations
#[tokio::test]
async fn test_vector_batch_operations() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Create test collection first
    let create_request = json!({
        "name": "batch_test_collection",
        "dimension": 128,
        "distance_metric": "EUCLIDEAN",
        "storage_engine": "LSM",
        "primary_indexing_algorithm": "IVF"
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/collection", Some(create_request)).await;
    assert_eq!(status, StatusCode::OK);
    let collection_id = response["collection_id"].as_str().unwrap();
    
    // Test vector batch insert
    let batch_request = json!({
        "collection_id": collection_id,
        "vectors": [
            {
                "id": "vec_1",
                "vector": (0..128).map(|i| i as f32 / 128.0).collect::<Vec<f32>>(),
                "metadata": [
                    {"key": "category", "value": "electronics"},
                    {"key": "score", "value": "0.85"}
                ]
            },
            {
                "id": "vec_2",
                "vector": (0..128).map(|i| (i + 64) as f32 / 128.0).collect::<Vec<f32>>(),
                "metadata": [
                    {"key": "category", "value": "books"},
                    {"key": "score", "value": "0.92"}
                ]
            },
            {
                "id": "vec_3",
                "vector": (0..128).map(|i| (i + 32) as f32 / 128.0).collect::<Vec<f32>>(),
                "metadata": [
                    {"key": "category", "value": "electronics"},
                    {"key": "score", "value": "0.78"}
                ]
            }
        ]
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/batch", Some(batch_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["sequences"].is_array());
    assert_eq!(response["sequences"].as_array().unwrap().len(), 3);
    
    // Test batch with invalid data
    let invalid_batch_request = json!({
        "collection_id": collection_id,
        "vectors": [
            {
                "id": "vec_invalid",
                "vector": (0..64).map(|i| i as f32).collect::<Vec<f32>>(), // Wrong dimension
                "metadata": []
            }
        ]
    });
    
    let (status, _response) = make_request(&app, "POST", "/api/v1/vector/batch", Some(invalid_batch_request)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    
    // Test batch on non-existent collection
    let invalid_collection_request = json!({
        "collection_id": "non_existent_collection",
        "vectors": [
            {
                "id": "vec_test",
                "vector": (0..128).map(|i| i as f32).collect::<Vec<f32>>(),
                "metadata": []
            }
        ]
    });
    
    let (status, _response) = make_request(&app, "POST", "/api/v1/vector/batch", Some(invalid_collection_request)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Test vector search operations
#[tokio::test]
async fn test_vector_search_operations() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Create test collection
    let create_request = json!({
        "name": "search_test_collection",
        "dimension": 128,
        "distance_metric": "COSINE",
        "storage_engine": "SST",
        "primary_indexing_algorithm": "HNSW",
        "filterable_columns": ["category", "score"]
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/collection", Some(create_request)).await;
    assert_eq!(status, StatusCode::OK);
    let collection_id = response["collection_id"].as_str().unwrap();
    
    // Insert test vectors
    let batch_request = json!({
        "collection_id": collection_id,
        "vectors": (0..50).map(|i| json!({
            "id": format!("vec_{}", i),
            "vector": (0..128).map(|j| ((i * 128 + j) as f32) / (50.0 * 128.0)).collect::<Vec<f32>>(),
            "metadata": [
                {"key": "category", "value": format!("category_{}", i % 3)},
                {"key": "score", "value": format!("{}", i as f64 / 50.0)}
            ]
        })).collect::<Vec<_>>()
    });
    
    let (status, _response) = make_request(&app, "POST", "/api/v1/vector/batch", Some(batch_request)).await;
    assert_eq!(status, StatusCode::OK);
    
    // Test basic search
    let search_request = json!({
        "collection_id": collection_id,
        "queries": [
            {
                "vector": (0..128).map(|i| 0.5).collect::<Vec<f32>>()
            }
        ],
        "top_k": 10
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/search", Some(search_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["results"].is_array());
    
    let results = response["results"].as_array().unwrap();
    assert!(results.len() > 0);
    assert!(results[0]["vectors"].as_array().unwrap().len() <= 10);
    
    // Verify search results structure
    let first_result = &results[0]["vectors"].as_array().unwrap()[0];
    assert!(first_result["id"].is_string());
    assert!(first_result["vector"].is_array());
    assert!(first_result["metadata"].is_array());
    assert!(first_result["distance"].is_number());
    
    // Test search with filters
    let filtered_search_request = json!({
        "collection_id": collection_id,
        "queries": [
            {
                "vector": (0..128).map(|i| 0.5).collect::<Vec<f32>>(),
                "metadata_filter": {
                    "category": "category_1"
                }
            }
        ],
        "top_k": 5
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/search", Some(filtered_search_request)).await;
    assert_eq!(status, StatusCode::OK);
    
    let results = response["results"].as_array().unwrap();
    let vectors = results[0]["vectors"].as_array().unwrap();
    
    // Verify all results match filter
    for vector in vectors {
        let metadata = vector["metadata"].as_array().unwrap();
        let category = metadata.iter()
            .find(|item| item["key"] == "category")
            .unwrap()["value"]
            .as_str()
            .unwrap();
        assert_eq!(category, "category_1");
    }
    
    // Test batch search
    let batch_search_request = json!({
        "collection_id": collection_id,
        "queries": [
            {
                "vector": (0..128).map(|i| 0.3).collect::<Vec<f32>>()
            },
            {
                "vector": (0..128).map(|i| 0.7).collect::<Vec<f32>>()
            }
        ],
        "top_k": 5
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/search", Some(batch_search_request)).await;
    assert_eq!(status, StatusCode::OK);
    
    let results = response["results"].as_array().unwrap();
    assert_eq!(results.len(), 2); // Should have results for both queries
    
    // Test search with optimization hints
    let optimized_search_request = json!({
        "collection_id": collection_id,
        "queries": [
            {
                "vector": (0..128).map(|i| 0.5).collect::<Vec<f32>>()
            }
        ],
        "top_k": 10,
        "search_optimization": {
            "enable_two_stage": true,
            "quantization_hint": "PQ8"
        }
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/search", Some(optimized_search_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["results"].is_array());
}

/// Test health and metrics endpoints
#[tokio::test]
async fn test_health_and_metrics() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Test health endpoint
    let (status, response) = make_request(&app, "GET", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["status"], "healthy");
    assert!(response["timestamp"].is_string());
    
    // Test metrics endpoint
    let (status, response) = make_request(&app, "GET", "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["metrics"].is_object());
    
    let metrics = response["metrics"].as_object().unwrap();
    assert!(metrics.contains_key("system"));
    assert!(metrics.contains_key("service"));
    
    // Perform some operations to generate metrics
    let create_request = json!({
        "name": "metrics_test_collection",
        "dimension": 128,
        "distance_metric": "COSINE",
        "storage_engine": "SST",
        "primary_indexing_algorithm": "HNSW"
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/collection", Some(create_request)).await;
    assert_eq!(status, StatusCode::OK);
    let collection_id = response["collection_id"].as_str().unwrap();
    
    // Insert some vectors
    let batch_request = json!({
        "collection_id": collection_id,
        "vectors": [
            {
                "id": "metrics_vec_1",
                "vector": (0..128).map(|i| i as f32 / 128.0).collect::<Vec<f32>>(),
                "metadata": []
            }
        ]
    });
    
    make_request(&app, "POST", "/api/v1/vector/batch", Some(batch_request)).await;
    
    // Check metrics again
    let (status, response) = make_request(&app, "GET", "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    
    let metrics = response["metrics"].as_object().unwrap();
    let service_metrics = metrics["service"].as_object().unwrap();
    
    // Should have operation counters
    assert!(service_metrics.contains_key("collections_created"));
    assert!(service_metrics.contains_key("vectors_inserted"));
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Test invalid collection creation
    let invalid_create_request = json!({
        "name": "",
        "dimension": 0,
        "distance_metric": "INVALID_METRIC"
    });
    
    let (status, _response) = make_request(&app, "POST", "/api/v1/collection", Some(invalid_create_request)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    
    // Test get non-existent collection
    let (status, _response) = make_request(&app, "GET", "/api/v1/collection/non_existent", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    
    // Test search on non-existent collection
    let search_request = json!({
        "collection_id": "non_existent_collection",
        "queries": [
            {
                "vector": (0..128).map(|i| 0.5).collect::<Vec<f32>>()
            }
        ],
        "top_k": 10
    });
    
    let (status, _response) = make_request(&app, "POST", "/api/v1/vector/search", Some(search_request)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    
    // Test malformed JSON
    let malformed_request = r#"{"invalid": json"#;
    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/collection")
        .header("content-type", "application/json")
        .body(Body::from(malformed_request))
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// Test internal endpoints
#[tokio::test]
async fn test_internal_endpoints() {
    setup_hardware_capabilities();
    let (app, _temp_dir) = create_test_app().await;
    
    // Create test collection and add some data
    let create_request = json!({
        "name": "flush_test_collection",
        "dimension": 128,
        "distance_metric": "COSINE",
        "storage_engine": "SST",
        "primary_indexing_algorithm": "HNSW"
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/collection", Some(create_request)).await;
    assert_eq!(status, StatusCode::OK);
    let collection_id = response["collection_id"].as_str().unwrap();
    
    // Add some vectors
    let batch_request = json!({
        "collection_id": collection_id,
        "vectors": [
            {
                "id": "flush_vec_1",
                "vector": (0..128).map(|i| i as f32 / 128.0).collect::<Vec<f32>>(),
                "metadata": []
            }
        ]
    });
    
    make_request(&app, "POST", "/api/v1/vector/batch", Some(batch_request)).await;
    
    // Test flush endpoint
    let flush_request = json!({
        "collection_id": collection_id
    });
    
    let (status, response) = make_request(&app, "POST", "/internal/flush", Some(flush_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["success"].as_bool().unwrap());
    
    // Test flush all endpoint
    let (status, response) = make_request(&app, "POST", "/internal/flush", Some(json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["success"].as_bool().unwrap());
    
    // Verify data is still searchable after flush
    let search_request = json!({
        "collection_id": collection_id,
        "queries": [
            {
                "vector": (0..128).map(|i| 0.5).collect::<Vec<f32>>()
            }
        ],
        "top_k": 10
    });
    
    let (status, response) = make_request(&app, "POST", "/api/v1/vector/search", Some(search_request)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["results"].as_array().unwrap().len() > 0);
}
