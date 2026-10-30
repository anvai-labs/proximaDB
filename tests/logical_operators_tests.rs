//! Comprehensive tests for logical metadata operators (AND/OR/NOT)
//!
//! Tests the metadata query engine and its integration with search systems.

use serde_json::json;
use std::collections::HashMap;
use tracing::info;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use proximadb::core::search::{
        DataFreshnessTier, DeduplicationStorageEngine, MultiTierDeduplicator, TieredSearchCandidate,
    };
    use proximadb::core::{
        ComparisonOperator, FieldQuery, MetadataQuery, MetadataQueryBuilder, MetadataQueryEngine,
    };
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{
        EmbeddingCell, EmbeddingValues, ProximaRecord, ProximaTree, ProximaTreeNode,
    };

    /// Create a test vector record with metadata
    fn create_test_vector(id: &str, metadata: HashMap<String, serde_json::Value>) -> ProximaRecord {
        let mut props = ProximaTree::new();
        for (key, value) in &metadata {
            let pv = match value {
                serde_json::Value::String(s) => ProximaValue::String(s.clone()),
                serde_json::Value::Number(n) => ProximaValue::Float64(n.as_f64().unwrap_or(0.0)),
                serde_json::Value::Bool(b) => ProximaValue::Boolean(*b),
                _ => ProximaValue::String(value.to_string()),
            };
            props.insert(key.clone(), ProximaTreeNode::Value(pv));
        }
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                4,
                vec![1.0, 2.0, 3.0, 4.0],
            )],
            props,
            record_version: 1,
            created_at_ns: Utc::now().timestamp_nanos_opt().unwrap_or(0),
            updated_at_ns: Utc::now().timestamp_nanos_opt().unwrap_or(0),
            ..Default::default()
        }
    }

    /// Create test tiered search result
    fn create_tiered_result(record: ProximaRecord, similarity: f32) -> TieredSearchCandidate {
        TieredSearchCandidate {
            record,
            similarity,
            tier: DataFreshnessTier::Unflushed,
            engine: DeduplicationStorageEngine::WAL,
            timestamp: chrono::Utc::now(),
            sequence: 0,
            file_path: None,
        }
    }

    #[test]
    fn test_logical_and_operator() {
        let mut engine = MetadataQueryEngine::new();

        // Test data: electronics with price and brand
        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(299.99));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta
        };

        // AND query: category = "electronics" AND price < 500
        let query = MetadataQuery::And(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::Field(FieldQuery {
                field: "price".to_string(),
                operator: ComparisonOperator::LessThan,
                value: json!(500.0),
            }),
        ]);

        assert!(engine.evaluate(&query, &metadata).unwrap());

        // Should fail if one condition is false
        let query_fail = MetadataQuery::And(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::Field(FieldQuery {
                field: "price".to_string(),
                operator: ComparisonOperator::LessThan,
                value: json!(200.0), // Too low
            }),
        ]);

        assert!(!engine.evaluate(&query_fail, &metadata).unwrap());
    }

    #[test]
    fn test_logical_or_operator() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("books"));
            meta.insert("price".to_string(), json!(15.99));
            meta
        };

        // OR query: category = "electronics" OR category = "books"
        let query = MetadataQuery::Or(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::field_eq("category", json!("books")),
        ]);

        assert!(engine.evaluate(&query, &metadata).unwrap());

        // Should fail if no conditions match
        let query_fail = MetadataQuery::Or(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::field_eq("category", json!("clothing")),
        ]);

        assert!(!engine.evaluate(&query_fail, &metadata).unwrap());
    }

    #[test]
    fn test_logical_not_operator() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta
        };

        // NOT query: NOT (category = "books")
        let query = MetadataQuery::Not(Box::new(MetadataQuery::field_eq(
            "category",
            json!("books"),
        )));

        assert!(engine.evaluate(&query, &metadata).unwrap());

        // Should fail when negating a true condition
        let query_fail = MetadataQuery::Not(Box::new(MetadataQuery::field_eq(
            "category",
            json!("electronics"),
        )));

        assert!(!engine.evaluate(&query_fail, &metadata).unwrap());
    }

    #[test]
    fn test_complex_logical_combinations() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(199.99));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta.insert("rating".to_string(), json!(4.5));
            meta.insert("in_stock".to_string(), json!(true));
            meta
        };

        // Complex query: (category = "electronics" AND price < 300) OR (brand = "TechCorp" AND rating >= 4.0)
        let query = MetadataQuery::Or(vec![
            MetadataQuery::And(vec![
                MetadataQuery::field_eq("category", json!("electronics")),
                MetadataQuery::Field(FieldQuery {
                    field: "price".to_string(),
                    operator: ComparisonOperator::LessThan,
                    value: json!(300.0),
                }),
            ]),
            MetadataQuery::And(vec![
                MetadataQuery::field_eq("brand", json!("TechCorp")),
                MetadataQuery::Field(FieldQuery {
                    field: "rating".to_string(),
                    operator: ComparisonOperator::GreaterThanOrEqual,
                    value: json!(4.0),
                }),
            ]),
        ]);

        assert!(engine.evaluate(&query, &metadata).unwrap());

        // Test with NOT: NOT (price > 500) AND in_stock = true
        let query_with_not = MetadataQuery::And(vec![
            MetadataQuery::Not(Box::new(MetadataQuery::Field(FieldQuery {
                field: "price".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: json!(500.0),
            }))),
            MetadataQuery::field_eq("in_stock", json!(true)),
        ]);

        assert!(engine.evaluate(&query_with_not, &metadata).unwrap());
    }

    #[test]
    fn test_comparison_operators() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("price".to_string(), json!(99.99));
            meta.insert("year".to_string(), json!(2023));
            meta.insert("name".to_string(), json!("Gaming Laptop"));
            meta.insert(
                "description".to_string(),
                json!("High-performance gaming laptop with RGB lighting"),
            );
            meta
        };

        // Numeric comparisons
        assert!(
            engine
                .evaluate(
                    &MetadataQuery::Field(FieldQuery {
                        field: "price".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        value: json!(50.0),
                    }),
                    &metadata
                )
                .unwrap()
        );

        assert!(
            engine
                .evaluate(
                    &MetadataQuery::Field(FieldQuery {
                        field: "year".to_string(),
                        operator: ComparisonOperator::LessThanOrEqual,
                        value: json!(2023),
                    }),
                    &metadata
                )
                .unwrap()
        );

        // String operations
        assert!(
            engine
                .evaluate(
                    &MetadataQuery::Field(FieldQuery {
                        field: "description".to_string(),
                        operator: ComparisonOperator::Contains,
                        value: json!("gaming"),
                    }),
                    &metadata
                )
                .unwrap()
        );

        assert!(
            engine
                .evaluate(
                    &MetadataQuery::Field(FieldQuery {
                        field: "name".to_string(),
                        operator: ComparisonOperator::StartsWith,
                        value: json!("Gaming"),
                    }),
                    &metadata
                )
                .unwrap()
        );
    }

    #[test]
    fn test_field_existence_operators() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(199.99));
            // Note: "discount" field is missing
            meta
        };

        // Field exists
        let exists_query = MetadataQuery::field_exists("category");
        assert!(engine.evaluate(&exists_query, &metadata).unwrap());

        // Field does not exist
        let not_exists_query = MetadataQuery::Field(FieldQuery {
            field: "discount".to_string(),
            operator: ComparisonOperator::NotExists,
            value: serde_json::Value::Null,
        });
        assert!(engine.evaluate(&not_exists_query, &metadata).unwrap());

        // Should fail for missing field existence check
        let missing_exists = MetadataQuery::field_exists("discount");
        assert!(!engine.evaluate(&missing_exists, &metadata).unwrap());
    }

    #[test]
    fn test_array_operations() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("tags".to_string(), json!(["laptop", "gaming", "portable"]));
            meta
        };

        // IN operation - check if category is in allowed list
        let in_query =
            MetadataQuery::field_in("category", vec![json!("electronics"), json!("books")]);
        assert!(engine.evaluate(&in_query, &metadata).unwrap());

        // NOT IN operation
        let not_in_query = MetadataQuery::Field(FieldQuery {
            field: "category".to_string(),
            operator: ComparisonOperator::NotIn,
            value: json!(["clothing", "furniture"]),
        });
        assert!(engine.evaluate(&not_in_query, &metadata).unwrap());
    }

    #[test]
    fn test_range_queries() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("price".to_string(), json!(150.00));
            meta.insert("rating".to_string(), json!(4.2));
            meta
        };

        // Price range: 100 <= price <= 200
        let price_range = MetadataQuery::field_range("price", 100.0, 200.0);
        assert!(engine.evaluate(&price_range, &metadata).unwrap());

        // Rating range: 4.0 <= rating <= 5.0
        let rating_range = MetadataQuery::field_range("rating", 4.0, 5.0);
        assert!(engine.evaluate(&rating_range, &metadata).unwrap());

        // Out of range test
        let out_of_range = MetadataQuery::field_range("price", 200.0, 300.0);
        assert!(!engine.evaluate(&out_of_range, &metadata).unwrap());
    }

    #[test]
    fn test_query_builder() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(299.99));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta
        };

        // Build query using builder pattern
        let query = MetadataQueryBuilder::new()
            .field_equals("category", json!("electronics"))
            .field_compare("price", ComparisonOperator::LessThan, json!(500.0))
            .build();

        assert!(engine.evaluate(&query, &metadata).unwrap());
    }

    #[test]
    fn test_integration_with_multi_tier_deduplicator() {
        // Test logical operators with the multi-tier deduplication system

        // Create test vectors with different metadata
        let vector1 = create_test_vector("vec1", {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(199.99));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta
        });

        let vector2 = create_test_vector("vec2", {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("books"));
            meta.insert("price".to_string(), json!(29.99));
            meta.insert("author".to_string(), json!("John Doe"));
            meta
        });

        let vector3 = create_test_vector("vec3", {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("price".to_string(), json!(599.99));
            meta.insert("brand".to_string(), json!("OtherCorp"));
            meta
        });

        // Create query: category = "electronics" AND price < 300
        let query = MetadataQuery::And(vec![
            MetadataQuery::field_eq("category", json!("electronics")),
            MetadataQuery::Field(FieldQuery {
                field: "price".to_string(),
                operator: ComparisonOperator::LessThan,
                value: json!(300.0),
            }),
        ]);

        // Create deduplicator with logical query
        let mut deduplicator = MultiTierDeduplicator::with_query(query);

        // Add test results
        let results = vec![
            create_tiered_result(vector1, 0.9),
            create_tiered_result(vector2, 0.8),
            create_tiered_result(vector3, 0.7),
        ];

        deduplicator.add_tier_results(results);

        // Get final results - should only include vector1 (electronics + price < 300)
        let final_results = deduplicator.get_final_results(10);

        assert_eq!(final_results.len(), 1);
        assert_eq!(final_results[0].record.oid, "vec1".to_string());
        assert_eq!(final_results[0].similarity, 0.9);
    }

    #[test]
    fn test_complex_business_logic_queries() {
        let mut engine = MetadataQueryEngine::new();

        // E-commerce product filtering scenario
        let product = {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), json!("electronics"));
            meta.insert("subcategory".to_string(), json!("laptops"));
            meta.insert("price".to_string(), json!(899.99));
            meta.insert("brand".to_string(), json!("TechCorp"));
            meta.insert("rating".to_string(), json!(4.3));
            meta.insert("in_stock".to_string(), json!(true));
            meta.insert("shipping_time_days".to_string(), json!(2));
            meta.insert("warranty_years".to_string(), json!(3));
            meta
        };

        // Business logic: Premium electronics with fast shipping OR budget friendly options
        let business_query = MetadataQuery::Or(vec![
            // Premium: electronics + price > 500 + rating >= 4.0 + fast shipping
            MetadataQuery::And(vec![
                MetadataQuery::field_eq("category", json!("electronics")),
                MetadataQuery::Field(FieldQuery {
                    field: "price".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: json!(500.0),
                }),
                MetadataQuery::Field(FieldQuery {
                    field: "rating".to_string(),
                    operator: ComparisonOperator::GreaterThanOrEqual,
                    value: json!(4.0),
                }),
                MetadataQuery::Field(FieldQuery {
                    field: "shipping_time_days".to_string(),
                    operator: ComparisonOperator::LessThanOrEqual,
                    value: json!(3),
                }),
            ]),
            // Budget: price < 100 + in stock
            MetadataQuery::And(vec![
                MetadataQuery::Field(FieldQuery {
                    field: "price".to_string(),
                    operator: ComparisonOperator::LessThan,
                    value: json!(100.0),
                }),
                MetadataQuery::field_eq("in_stock", json!(true)),
            ]),
        ]);

        // This product should match the premium criteria
        assert!(engine.evaluate(&business_query, &product).unwrap());

        // Test exclusion logic: NOT (brand = "BadBrand" OR warranty < 1 year)
        let exclusion_query = MetadataQuery::Not(Box::new(MetadataQuery::Or(vec![
            MetadataQuery::field_eq("brand", json!("BadBrand")),
            MetadataQuery::Field(FieldQuery {
                field: "warranty_years".to_string(),
                operator: ComparisonOperator::LessThan,
                value: json!(1),
            }),
        ])));

        assert!(engine.evaluate(&exclusion_query, &product).unwrap());
    }

    #[test]
    fn test_error_handling() {
        let mut engine = MetadataQueryEngine::new();

        let metadata = {
            let mut meta = HashMap::new();
            meta.insert("price".to_string(), json!("not_a_number")); // Invalid for numeric comparison
            meta
        };

        // This should handle the error gracefully
        let query = MetadataQuery::Field(FieldQuery {
            field: "price".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: json!(100.0),
        });

        // The query engine should handle type mismatches gracefully
        let result = engine.evaluate(&query, &metadata);
        assert!(result.is_ok()); // Should not panic, even with type mismatch
    }

    #[test]
    fn test_performance_with_large_metadata() {
        let mut engine = MetadataQueryEngine::new();

        // Create metadata with many fields
        let mut metadata = HashMap::new();
        for i in 0..1000 {
            metadata.insert(format!("field_{}", i), json!(format!("value_{}", i)));
        }
        metadata.insert("target_field".to_string(), json!("target_value"));

        // Create a query that needs to find a specific field
        let query = MetadataQuery::field_eq("target_field", json!("target_value"));

        let start = std::time::Instant::now();
        let result = engine.evaluate(&query, &metadata).unwrap();
        let duration = start.elapsed();

        assert!(result);
        assert!(duration.as_millis() < 50); // Should be fast even with large metadata

        info!("Query on 1000 fields completed in {:?}", duration);
    }
}
