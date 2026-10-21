//! Comprehensive Filter Tests for SST Engine
//!
//! Tests all filtering scenarios with typed metadata to ensure SST engine
//! properly preserves and filters on all data types (String, Integer, Float, Boolean).

use proximadb::compute::distance_computation::UnifiedDistanceCompute;
use proximadb::core::search::{ComparisonOperator, FilterExpression, SearchParams};
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, FilterableColumnSpec, FilterableDataType,
};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageEngine};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

/// Helper to create test collection with filterable columns
fn create_test_collection(temp_dir: &TempDir) -> Collection {
    Collection {
        id: "test_sst_filter".to_string(),
        config: Some(CollectionConfig {
            dimension: 128,
            distance_metric: Some(1), // Cosine
            filterable_columns: vec![
                FilterableColumnSpec {
                    name: "category".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: Some(10),
                },
                FilterableColumnSpec {
                    name: "price".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: Some(100),
                },
                FilterableColumnSpec {
                    name: "enabled".to_string(),
                    data_type: FilterableDataType::FilterableBoolean as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: Some(2),
                },
                FilterableColumnSpec {
                    name: "score".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: Some(50),
                },
                FilterableColumnSpec {
                    name: "status".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: Some(5),
                },
                FilterableColumnSpec {
                    name: "count".to_string(),
                    data_type: FilterableDataType::FilterableInteger as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: Some(100),
                },
            ],
            ..Default::default()
        }),
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create test vectors with diverse metadata values
fn create_test_vectors(collection_id: &str, count: usize) -> Vec<proximadb_records::ProximaRecord> {
    (0..count)
        .map(|i| {
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "category".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(format!("cat_{}", i % 10)),
                ),
            );
            props.insert(
                "price".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Float64(10.0 + (i as f64) * 5.0),
                ),
            );
            props.insert(
                "enabled".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Boolean(i % 2 == 0),
                ),
            );
            props.insert(
                "score".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Float64(0.1 * (i as f64)),
                ),
            );
            props.insert(
                "status".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(
                        if i % 3 == 0 {
                            "active"
                        } else if i % 3 == 1 {
                            "pending"
                        } else {
                            "inactive"
                        }
                        .to_string(),
                    ),
                ),
            );
            props.insert(
                "count".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Int64(i as i64 * 10),
                ),
            );

            proximadb_records::ProximaRecord {
                oid: format!("{}_{}", collection_id, i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![0.1; 128],
                )],
                props,
                record_version: 1,
                created_at_ns: i as i64 * 1_000_000_000,
                updated_at_ns: i as i64 * 1_000_000_000,
                ..Default::default()
            }
        })
        .collect()
}

#[tokio::test]
async fn test_sst_string_equals_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    // Insert test data
    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors.clone(),
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: category = "cat_5"
    let filter = FilterExpression::Comparison {
        field: "category".to_string(),
        operator: ComparisonOperator::Equals,
        value: json!("cat_5"),
    };

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // Should return ~10 results (indices 5, 15, 25, ..., 95)
    assert!(
        results.len() >= 8 && results.len() <= 12,
        "Expected ~10 results, got {}",
        results.len()
    );
}

#[tokio::test]
async fn test_sst_number_less_than_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: price < 50.0
    let filter = FilterExpression::Comparison {
        field: "price".to_string(),
        operator: ComparisonOperator::LessThan,
        value: json!(50.0),
    };

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // price = 10 + i*5, so i < 8 => 8 results
    assert_eq!(results.len(), 8, "Expected 8 results for price < 50");
}

#[tokio::test]
async fn test_sst_boolean_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: enabled = true
    let filter = FilterExpression::Comparison {
        field: "enabled".to_string(),
        operator: ComparisonOperator::Equals,
        value: json!(true),
    };

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // enabled=true for even indices: 0, 2, 4, ..., 98 => 50 results
    assert_eq!(results.len(), 50, "Expected 50 results for enabled=true");
}

#[tokio::test]
async fn test_sst_integer_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: count >= 500 (i >= 50)
    let filter = FilterExpression::Comparison {
        field: "count".to_string(),
        operator: ComparisonOperator::GreaterThanOrEqual,
        value: json!(500),
    };

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // count = i*10, so count >= 500 => i >= 50 => 50 results (50..99)
    assert_eq!(results.len(), 50, "Expected 50 results for count >= 500");
}

#[tokio::test]
async fn test_sst_and_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: category = "cat_5" AND enabled = true
    let filter = FilterExpression::And(vec![
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("cat_5"),
        },
        FilterExpression::Comparison {
            field: "enabled".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(true),
        },
    ]);

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // cat_5: indices 5,15,25,35,45,55,65,75,85,95 (all odd)
    // enabled=true: even indices
    // Intersection: NONE (mutually exclusive)
    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for mutually exclusive AND"
    );
}

#[tokio::test]
async fn test_sst_or_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: category = "cat_0" OR category = "cat_1"
    let filter = FilterExpression::Or(vec![
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("cat_0"),
        },
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!("cat_1"),
        },
    ]);

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // cat_0: 0,10,20,30,40,50,60,70,80,90 (10 results)
    // cat_1: 1,11,21,31,41,51,61,71,81,91 (10 results)
    // Total: 20 results
    assert_eq!(results.len(), 20, "Expected 20 results for OR filter");
}

#[tokio::test]
async fn test_sst_not_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 50); // Smaller dataset for NOT
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: NOT (enabled = true)
    let filter = FilterExpression::Not(Box::new(FilterExpression::Comparison {
        field: "enabled".to_string(),
        operator: ComparisonOperator::Equals,
        value: json!(true),
    }));

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // NOT enabled=true => enabled=false => odd indices => 25 results
    assert_eq!(results.len(), 25, "Expected 25 results for NOT filter");
}

#[tokio::test]
async fn test_sst_complex_nested_filter() {
    let temp_dir = TempDir::new().unwrap();
    let collection = create_test_collection(&temp_dir);
    let filesystem = Arc::new(FilesystemFactory::create_default().await.unwrap());
    let distance_compute = Arc::new(UnifiedDistanceCompute::default());
    let engine = SstEngine::new_with_config(Default::default(), filesystem, distance_compute)
        .await
        .unwrap();

    let vectors = create_test_vectors("test_sst_filter", 100);
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        collection_config: Some(collection.clone()),
        vector_records: vectors,
        force: true,
        synchronous: true,
        ..Default::default()
    };
    engine.flush(params).await.unwrap();

    // Test: (category = "cat_0" OR category = "cat_2") AND enabled = true
    let filter = FilterExpression::And(vec![
        FilterExpression::Or(vec![
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("cat_0"),
            },
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: json!("cat_2"),
            },
        ]),
        FilterExpression::Comparison {
            field: "enabled".to_string(),
            operator: ComparisonOperator::Equals,
            value: json!(true),
        },
    ]);

    let search_params = Arc::new(SearchParams {
        vector: Some(vec![0.1; 128]),
        top_k: Some(100),
        filter_expression: Some(filter),
        ..Default::default()
    });

    let ctx = StorageQueryContext::new(search_params, Arc::new(collection));
    let results = engine.search_vectors_unified(&ctx).await.unwrap();

    // cat_0: 0,10,20,30,40,50,60,70,80,90 (all even, so enabled=true) ✓
    // cat_2: 2,12,22,32,42,52,62,72,82,92 (all even, so enabled=true) ✓
    // Total: 20 results
    assert!(
        results.len() >= 18 && results.len() <= 22,
        "Expected ~20 results, got {}",
        results.len()
    );
}
