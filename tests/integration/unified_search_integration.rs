//! Simplified Integration Test for Unified Search Interface
//!
//! This test focuses on the core unified search functionality using the actual
//! API structure and testing the search engines directly.

use serde_json::json;
use std::collections::HashMap;

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::compute::distance_computation::UnifiedDistanceCompute;
use proximadb::core::search::SearchParams;
use proximadb_data_model::ProximaValue;
use proximadb_records::{
    EmbeddingCell, EmbeddingValues, ProximaRecord, ProximaTree, ProximaTreeNode,
};

/// Generate test vectors with basic metadata
fn generate_test_vectors(count: usize, dimension: usize) -> Vec<ProximaRecord> {
    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

    (0..count)
        .map(|i| {
            let values: Vec<f32> = (0..dimension)
                .map(|j| (i * dimension + j) as f32 / (count * dimension) as f32)
                .collect();
            let dim = values.len() as u32;

            let mut props = ProximaTree::new();
            props.insert(
                "category".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("cat_{}", i % 3))),
            );
            props.insert(
                "score".to_string(),
                ProximaTreeNode::Value(ProximaValue::Float64(i as f64 / count as f64)),
            );
            props.insert(
                "active".to_string(),
                ProximaTreeNode::Value(ProximaValue::Boolean(i % 2 == 0)),
            );

            ProximaRecord {
                oid: format!("vec_{}", i),
                created_at_ns: now_ns,
                updated_at_ns: now_ns,
                record_version: 1,
                props,
                embeddings: vec![EmbeddingCell::new_fp32("default", "vector", dim, values)],
                ..Default::default()
            }
        })
        .collect()
}

/// Test basic search functionality with SearchParams
#[tokio::test]
async fn test_search_params_functionality() {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Test various SearchParams configurations
    let query_vectors = vec![vec![0.1; 16], vec![0.5; 16], vec![0.9; 16]];

    // Test single vector search
    let single_search = SearchParams::single_vector(query_vectors[0].clone());
    assert_eq!(single_search.query_vectors.as_ref().unwrap().len(), 1);
    assert_eq!(single_search.top_k, Some(10)); // Default value

    // Test batch vector search
    let batch_search = SearchParams::batch_vectors(query_vectors.clone());
    assert_eq!(batch_search.query_vectors.as_ref().unwrap().len(), 3);

    // Test search with filters
    let mut filters = HashMap::new();
    filters.insert("category".to_string(), json!("cat_1"));
    filters.insert("active".to_string(), json!(true));

    let filtered_search = SearchParams {
        query_vectors: Some(vec![query_vectors[1].clone()]),
        top_k: Some(20),
        distance_metric: Some(DistanceMetric::Manhattan),
        ..Default::default()
    }
    .with_simple_filters(filters.clone());

    // Can't assert on filters directly anymore, but we can verify the filter_expression
    assert!(filtered_search.filter_expression.is_some());
    assert_eq!(
        filtered_search.distance_metric,
        Some(DistanceMetric::Manhattan)
    );

    // Test completed
}

/// Test distance metric functionality
#[tokio::test]
async fn test_distance_metrics() {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let distance_compute = UnifiedDistanceCompute::default();

    // Test vectors
    let vector1 = vec![1.0, 0.0, 0.0];
    let vector2 = vec![0.0, 1.0, 0.0];
    let identical = vec![1.0, 0.0, 0.0];

    // Test all distance metrics
    let metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ];

    for metric in metrics {
        // Test distance calculation
        let result1 = distance_compute.calculate_distance(&vector1, &vector2, &metric);
        let result2 = distance_compute.calculate_distance(&vector1, &identical, &metric);

        // Verify the result structure
        // For DotProduct, raw_value can be negative and rank_value is -raw_value
        if metric == DistanceMetric::DotProduct {
            // DotProduct can have any value
            assert!(result1.normalized_score >= 0.0 && result1.normalized_score <= 1.0);
            assert!(result2.normalized_score >= 0.0 && result2.normalized_score <= 1.0);
        } else {
            // Distance metrics should have non-negative raw values
            assert!(result1.raw_value >= 0.0);
            assert!(result1.normalized_score >= 0.0 && result1.normalized_score <= 1.0);
            assert!(result1.rank_value >= 0.0);

            assert!(result2.raw_value >= 0.0);
            assert!(result2.normalized_score >= 0.0 && result2.normalized_score <= 1.0);
            assert!(result2.rank_value >= 0.0);
        }

        // Identical vectors should have higher similarity (lower rank_value)
        // This tests the semantic consistency of the unified distance system
        assert!(
            result2.rank_value <= result1.rank_value,
            "Identical vectors should have lower rank_value for {:?}",
            metric
        );

        // Distance metric test completed
    }
}

/// Test ProximaRecord structure and metadata handling
#[tokio::test]
async fn test_vector_record_structure() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let test_vectors = generate_test_vectors(5, 64);

    for (i, record) in test_vectors.iter().enumerate() {
        assert_eq!(record.oid, format!("vec_{}", i));
        assert_eq!(record.embeddings[0].values.len(), 64);
        assert_eq!(record.props.len(), 3);
        assert!(record.created_at_ns > 0);

        let category = record.props.get("category").unwrap();
        if let ProximaTreeNode::Value(ProximaValue::String(s)) = category {
            assert!(s.starts_with("cat_"));
        } else {
            panic!("Expected string value for category");
        }

        let score = record.props.get("score").unwrap();
        if let ProximaTreeNode::Value(ProximaValue::Float64(n)) = score {
            assert!(*n >= 0.0 && *n <= 1.0);
        } else {
            panic!("Expected float value for score");
        }

        let active = record.props.get("active").unwrap();
        if let ProximaTreeNode::Value(ProximaValue::Boolean(b)) = active {
            assert_eq!(*b, i % 2 == 0);
        } else {
            panic!("Expected bool value for active");
        }
    }
}

/// Test SearchParams default values and edge cases
#[tokio::test]
async fn test_search_params_edge_cases() {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Test default SearchParams
    let default_params = SearchParams::default();
    assert_eq!(default_params.query_vectors, None);
    assert_eq!(default_params.top_k, Some(10));
    assert_eq!(default_params.distance_metric, Some(DistanceMetric::Cosine));
    assert!(default_params.filter_expression.is_none());
    assert_eq!(default_params.accuracy_threshold, Some(0.95));
    assert_eq!(default_params.timeout_ms, Some(5000));

    // Test empty query vectors
    let empty_query_params = SearchParams {
        query_vectors: Some(vec![]),
        top_k: Some(5),
        distance_metric: Some(DistanceMetric::Cosine),
        ..Default::default()
    };
    assert_eq!(empty_query_params.query_vectors.as_ref().unwrap().len(), 0);

    // Test large top_k
    let large_k_params = SearchParams {
        query_vectors: Some(vec![vec![0.1; 64]]),
        top_k: Some(10000),
        distance_metric: Some(DistanceMetric::Euclidean),
        ..Default::default()
    };
    assert_eq!(large_k_params.top_k, Some(10000));

    // Test zero-dimension vectors (edge case)
    let zero_dim_params = SearchParams {
        query_vectors: Some(vec![vec![]]),
        top_k: Some(1),
        distance_metric: Some(DistanceMetric::Cosine),
        ..Default::default()
    };
    assert_eq!(zero_dim_params.query_vectors.as_ref().unwrap()[0].len(), 0);

    // Test completed
}

/// Test basic API usage patterns
#[tokio::test]
async fn test_api_usage_patterns() {
    // Initialize hardware capabilities
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Test single vector search helper
    let query_vector = vec![0.1, 0.2, 0.3];
    let single_search = SearchParams::single_vector(query_vector.clone());
    assert_eq!(single_search.first_query_vector(), Some(&query_vector));

    // Test batch vector search helper
    let batch_vectors = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ];
    let batch_search = SearchParams::batch_vectors(batch_vectors.clone());
    assert_eq!(batch_search.query_vectors.as_ref().unwrap().len(), 3);
    // Test that it's a batch search by checking the number of vectors
    assert!(batch_search.query_vectors.as_ref().unwrap().len() > 1);

    // Test with different distance metrics
    let euclidean_search = SearchParams {
        query_vectors: Some(vec![vec![1.0, 0.0, 0.0]]),
        distance_metric: Some(DistanceMetric::Euclidean),
        ..Default::default()
    };
    assert_eq!(
        euclidean_search.distance_metric,
        Some(DistanceMetric::Euclidean)
    );

    // Test with filters
    let mut filters = HashMap::new();
    filters.insert("category".to_string(), json!("electronics"));
    filters.insert("price".to_string(), json!({"less_than": 100}));

    let filtered_search = SearchParams {
        query_vectors: Some(vec![vec![0.5; 128]]),
        top_k: Some(50),
        ..Default::default()
    }
    .with_simple_filters(filters);

    // Can't assert on filters directly anymore, but we can verify the filter_expression
    assert!(filtered_search.filter_expression.is_some());
    assert_eq!(filtered_search.top_k, Some(50));

    // Test completed
}
