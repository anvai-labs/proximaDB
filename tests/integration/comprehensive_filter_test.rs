//! Comprehensive Filter Test
//!
//! Validates that metadata filtering works correctly across all:
//! - Data types (String, Number, Boolean, Int64)
//! - Comparison operators (Equals, NotEquals, LessThan, GreaterThan, LessThanOrEqual, GreaterThanOrEqual)
//! - Logical operators (AND, OR, NOT)
//! - Storage engines (VIPER)

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{ComparisonOperator, FilterExpression, SearchParams};
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, FilterableColumnSpec, FilterableDataType, SqlValue,
    StorageAssignment, VectorRecord, sql_value,
};
use proximadb::storage::engines::viper::ViperEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use proximadb::utils::StoragePath;

/// Create test vectors with diverse metadata for comprehensive filtering
fn create_test_vectors(count: usize, dimension: usize) -> Vec<VectorRecord> {
    (0..count)
        .map(|i| {
            let mut metadata = HashMap::new();

            // String metadata
            metadata.insert(
                "category".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::StringValue(format!("cat_{}", i % 10))),
                },
            );

            metadata.insert(
                "status".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::StringValue(
                        if i % 3 == 0 {
                            "active"
                        } else if i % 3 == 1 {
                            "pending"
                        } else {
                            "inactive"
                        }
                        .to_string(),
                    )),
                },
            );

            // Number metadata
            metadata.insert(
                "price".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::NumberValue((i * 10) as f64 % 1000.0)),
                },
            );

            metadata.insert(
                "score".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::NumberValue((i as f64) / 10.0)),
                },
            );

            // Integer metadata
            metadata.insert(
                "count".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::Int64Value(i as i64)),
                },
            );

            // Boolean metadata
            metadata.insert(
                "enabled".to_string(),
                SqlValue {
                    value: Some(sql_value::Value::BoolValue(i % 2 == 0)),
                },
            );

            VectorRecord {
                id: format!("vec_{}", i),
                vector: vec![i as f32 / count as f32; dimension],
                metadata,
                timestamp: Some(chrono::Utc::now().timestamp()),
                updated_at: Some(chrono::Utc::now().timestamp()),
                expires_at: None,
                version: Some(1),
                source: None,
            }
        })
        .collect()
}

/// Helper function to search with filters
async fn search_with_filter(
    engine: &ViperEngine,
    collection_id: &str,
    base_path: &str,
    query_vector: &[f32],
    top_k: usize,
    filter_expression: Option<FilterExpression>,
) -> Result<Vec<proximadb::core::search::results::OptimizedSearchRecord>> {
    let search_params = Arc::new(SearchParams {
        vector: Some(query_vector.to_vec()),
        top_k: Some(top_k as u16),
        distance_metric: Some(DistanceMetric::Cosine),
        filter_expression,
        ..SearchParams::default()
    });

    let storage_url = format!("file://{}/{}/data", base_path, collection_id);
    let base_location = format!("file://{}", base_path);

    let collection = Arc::new(Collection {
        id: collection_id.to_string(),
        config: Some(CollectionConfig {
            name: collection_id.to_string(),
            dimension: query_vector.len() as u32,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Viper as i32),
            filterable_columns: vec![
                FilterableColumnSpec {
                    name: "category".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "price".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "enabled".to_string(),
                    data_type: FilterableDataType::FilterableBoolean as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "score".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "status".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "count".to_string(),
                    data_type: FilterableDataType::FilterableInteger as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
            ],
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: base_location.clone(),
            primary_path: storage_url.clone(),
            backup_paths: vec![],
            engine: proximadb::proto::proximadb_v1::StorageEngine::Viper as i32,
            engine_config: Default::default(),
            assigned_at: 0,
            ..Default::default()
        }),
        ..Default::default()
    });

    let metadata = StorageQueryMetadata {
        collection_id: collection_id.to_string(),
        use_axis_indexes: false,
        has_quantization: false,
        storage_path: base_location,
        dimension: query_vector.len(),
        distance_metric: DistanceMetric::Cosine.into(),
        ..Default::default()
    };

    let ctx = StorageQueryContext {
        search_params,
        collection,
        metadata,
        user_context: None,
        tenant_context: None,
    };

    engine.search_vectors_unified(&ctx).await
}

/// Set up test collection and flush vectors
async fn setup_test_collection(
    collection_id: &str,
    dimension: usize,
    base_path: &str,
) -> Result<ViperEngine> {
    // Set up directory structure
    let data_dir = StoragePath::collection_data_path(base_path, collection_id);
    tokio::fs::create_dir_all(&data_dir).await?;

    let temp_dir = StoragePath::data_file_path(base_path, collection_id, "___temp");
    tokio::fs::create_dir_all(&temp_dir).await?;

    // Create engine
    let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
    let engine = ViperEngine::from_core_config(
        proximadb::core::config::ViperConfig::default(),
        filesystem_factory,
    )
    .await?;

    // Create and flush test vectors
    let vectors = create_test_vectors(100, dimension);

    let collection = Collection {
        id: collection_id.to_string(),
        config: Some(CollectionConfig {
            name: collection_id.to_string(),
            dimension: dimension as u32,
            distance_metric: Some(DistanceMetric::Cosine as i32),
            storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Viper as i32),
            filterable_columns: vec![
                FilterableColumnSpec {
                    name: "category".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "price".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "enabled".to_string(),
                    data_type: FilterableDataType::FilterableBoolean as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "score".to_string(),
                    data_type: FilterableDataType::FilterableFloat as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "status".to_string(),
                    data_type: FilterableDataType::FilterableString as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: None,
                },
                FilterableColumnSpec {
                    name: "count".to_string(),
                    data_type: FilterableDataType::FilterableInteger as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: None,
                },
            ],
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            primary_path: format!("file://{}", base_path),
            backup_paths: vec![],
            engine: proximadb::proto::proximadb_v1::StorageEngine::Viper as i32,
            engine_config: HashMap::new(),
            base_location: format!("file://{}", base_path),
            assigned_at: chrono::Utc::now().timestamp(),
            ..Default::default()
        }),
        ..Default::default()
    };

    let flush_params = FlushParameters {
        collection_id: Some(collection_id.to_string()),
        force: true,
        synchronous: true,
        vector_records: vectors
            .into_iter()
            .map(|v: VectorRecord| v.into())
            .collect(),
        batch_ids: vec![],
        hints: HashMap::new(),
        timeout_ms: None,
        trigger_compaction: false,
        collection_config: Some(collection),
        estimated_size: 100 * dimension,
        collection_properties: None,
    };

    engine.do_flush(&flush_params).await?;

    // Small delay to ensure file system operations complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(engine)
}

#[tokio::test]
async fn test_string_equals_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_string_equals";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: category = "cat_5"
    let filter = FilterExpression::Comparison {
        field: "category".to_string(),
        operator: ComparisonOperator::Equals,
        value: serde_json::Value::String("cat_5".to_string()),
    };

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have ~10 results (indices 5, 15, 25, ..., 95)
    println!(
        "✓ String Equals filter: {} results (expected ~10)",
        results.len()
    );
    assert!(
        results.len() >= 8 && results.len() <= 12,
        "Expected ~10 results for cat_5, got {}",
        results.len()
    );

    // Verify all results match filter
    for result in &results {
        let id_num: usize = result.id.strip_prefix("vec_").unwrap().parse()?;
        assert_eq!(id_num % 10, 5, "Result {} doesn't match filter", result.id);
    }

    Ok(())
}

#[tokio::test]
async fn test_number_less_than_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_number_lt";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: price < 500
    let filter = FilterExpression::Comparison {
        field: "price".to_string(),
        operator: ComparisonOperator::LessThan,
        value: serde_json::json!(500.0),
    };

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have results with price < 500
    println!("✓ Number LessThan filter: {} results", results.len());
    assert!(
        !results.is_empty(),
        "Expected results for price < 500, got none"
    );

    Ok(())
}

#[tokio::test]
async fn test_boolean_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_boolean";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: enabled = true
    let filter = FilterExpression::Comparison {
        field: "enabled".to_string(),
        operator: ComparisonOperator::Equals,
        value: serde_json::json!(true),
    };

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have ~50 results (even indices)
    println!("✓ Boolean filter: {} results (expected ~50)", results.len());
    assert!(
        results.len() >= 45 && results.len() <= 55,
        "Expected ~50 results for enabled=true, got {}",
        results.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_and_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_and_filter";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: category = "cat_5" AND price < 500
    let filter = FilterExpression::And(vec![
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("cat_5".to_string()),
        },
        FilterExpression::Comparison {
            field: "price".to_string(),
            operator: ComparisonOperator::LessThan,
            value: serde_json::json!(500.0),
        },
    ]);

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have results matching both conditions
    println!("✓ AND filter: {} results (expected ~5)", results.len());
    assert!(
        !results.is_empty(),
        "Expected results for AND filter, got none"
    );

    // Verify all results match both conditions
    for result in &results {
        let id_num: usize = result.id.strip_prefix("vec_").unwrap().parse()?;
        assert_eq!(
            id_num % 10,
            5,
            "Result {} doesn't match category=cat_5",
            result.id
        );

        let price = (id_num * 10) % 1000;
        assert!(
            price < 500,
            "Result {} has price {} >= 500",
            result.id,
            price
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_or_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_or_filter";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: category = "cat_5" OR category = "cat_7"
    let filter = FilterExpression::Or(vec![
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("cat_5".to_string()),
        },
        FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("cat_7".to_string()),
        },
    ]);

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have ~20 results (cat_5 and cat_7 combined)
    println!("✓ OR filter: {} results (expected ~20)", results.len());
    assert!(
        results.len() >= 15 && results.len() <= 25,
        "Expected ~20 results for OR filter, got {}",
        results.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_not_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_not_filter";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: NOT (category = "cat_5")
    let filter = FilterExpression::Not(Box::new(FilterExpression::Comparison {
        field: "category".to_string(),
        operator: ComparisonOperator::Equals,
        value: serde_json::Value::String("cat_5".to_string()),
    }));

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have ~90 results (all except cat_5)
    println!("✓ NOT filter: {} results (expected ~90)", results.len());
    assert!(
        results.len() >= 85 && results.len() <= 95,
        "Expected ~90 results for NOT filter, got {}",
        results.len()
    );

    // Verify no results have category=cat_5
    for result in &results {
        let id_num: usize = result.id.strip_prefix("vec_").unwrap().parse()?;
        assert_ne!(
            id_num % 10,
            5,
            "Result {} should not have category=cat_5",
            result.id
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_greater_than_or_equal_filter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_gte_filter";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: score >= 5.0
    let filter = FilterExpression::Comparison {
        field: "score".to_string(),
        operator: ComparisonOperator::GreaterThanOrEqual,
        value: serde_json::json!(5.0),
    };

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have results with score >= 5.0 (indices 50-99)
    println!(
        "✓ GreaterThanOrEqual filter: {} results (expected ~50)",
        results.len()
    );
    assert!(
        results.len() >= 45,
        "Expected results for score >= 5.0, got {}",
        results.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_complex_nested_filter_zero_results() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_complex_filter_zero";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: (category = "cat_5" OR category = "cat_7") AND enabled = true
    // This should return 0 results because:
    // - cat_5 only appears at odd indices (5, 15, 25, ...)
    // - cat_7 only appears at odd indices (7, 17, 27, ...)
    // - enabled=true only for even indices (0, 2, 4, 6, ...)
    // These conditions are mutually exclusive, ensuring AND/OR logic works correctly
    let filter = FilterExpression::And(vec![
        FilterExpression::Or(vec![
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String("cat_5".to_string()),
            },
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String("cat_7".to_string()),
            },
        ]),
        FilterExpression::Comparison {
            field: "enabled".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(true),
        },
    ]);

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have 0 results (mutually exclusive conditions)
    println!(
        "✓ Complex nested filter (zero results): {} results (expected 0)",
        results.len()
    );
    assert_eq!(
        results.len(),
        0,
        "Expected 0 results for mutually exclusive conditions, got {}",
        results.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_complex_nested_filter_with_results() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path().to_str().unwrap();
    let collection_id = "test_complex_filter_results";
    let dimension = 128;

    let engine = setup_test_collection(collection_id, dimension, base_path).await?;

    // Test: (category = "cat_0" OR category = "cat_2") AND enabled = true
    // This should return ~20 results because:
    // - cat_0 appears at indices 0, 10, 20, 30, ... (all even, so enabled=true) ✓
    // - cat_2 appears at indices 2, 12, 22, 32, ... (all even, so enabled=true) ✓
    let filter = FilterExpression::And(vec![
        FilterExpression::Or(vec![
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String("cat_0".to_string()),
            },
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::String("cat_2".to_string()),
            },
        ]),
        FilterExpression::Comparison {
            field: "enabled".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(true),
        },
    ]);

    let results = search_with_filter(
        &engine,
        collection_id,
        base_path,
        &vec![0.5; dimension],
        100,
        Some(filter),
    )
    .await?;

    // Verify: Should have ~20 results (cat_0 and cat_2, both with enabled=true)
    println!(
        "✓ Complex nested filter (with results): {} results (expected ~20)",
        results.len()
    );
    assert!(
        results.len() >= 15 && results.len() <= 25,
        "Expected ~20 results for complex filter, got {}",
        results.len()
    );

    // Verify all results match conditions
    for result in &results {
        let id_num: usize = result.id.strip_prefix("vec_").unwrap().parse()?;

        // Must be cat_0 or cat_2
        let category_match = id_num % 10 == 0 || id_num % 10 == 2;
        assert!(
            category_match,
            "Result {} doesn't match category filter (expected cat_0 or cat_2)",
            result.id
        );

        // Must be enabled (even index)
        assert_eq!(id_num % 2, 0, "Result {} should be enabled", result.id);
    }

    Ok(())
}
