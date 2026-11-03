//! Integration tests for HELIX storage engine
//!
//! These tests verify the complete functionality of the HELIX engine
//! including clustering, compaction, and query performance.

// Import test utilities
#[path = "common/collection_builder.rs"]
mod collection_builder;
#[path = "common/vector_generator.rs"]
mod vector_generator;

#[cfg(test)]
mod helix_integration_tests {
    use proximadb::compute::distance_computation::DistanceMetric;
    use proximadb::core::search::SearchParams;
    use proximadb::proto::proximadb_v1::Collection;
    use proximadb::storage::engines::helix::{HelixConfig, HelixEngine};
    use proximadb::storage::traits::{
        CompactionParameters, FlushParameters, OperationPriority, StorageQueryContext,
        StorageQueryMetadata, UnifiedStorageEngine,
    };

    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::TempDir;

    use super::vector_generator;

    /// Helper to clean up old HELIX test files from previous runs
    /// This ensures tests always start with fresh data
    fn cleanup_old_helix_test_files() {
        use std::fs;

        // Clean /tmp directory
        if let Ok(entries) = fs::read_dir("/tmp") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    // Remove old .helix files and helix_test_ directories
                    if name_str.ends_with(".helix") || name_str.starts_with("helix_test_") {
                        let _ = if path.is_dir() {
                            fs::remove_dir_all(&path)
                        } else {
                            fs::remove_file(&path)
                        };
                    }
                }
            }
        }

        // Clean system temp directory (macOS: /var/folders, Linux: /tmp)
        if cfg!(target_os = "macos")
            && let Ok(entries) = fs::read_dir("/var/folders")
        {
            for entry in entries.flatten() {
                let path = entry.path();
                // Search in /var/folders/*/T/ directories
                let temp_path = path.join("T");
                if temp_path.exists()
                    && let Ok(temp_entries) = fs::read_dir(&temp_path)
                {
                    for temp_entry in temp_entries.flatten() {
                        let temp_file = temp_entry.path();
                        if let Some(name) = temp_file.file_name() {
                            let name_str = name.to_string_lossy();
                            if name_str.starts_with("helix_test_") || name_str.ends_with(".helix") {
                                let _ = if temp_file.is_dir() {
                                    fs::remove_dir_all(&temp_file)
                                } else {
                                    fs::remove_file(&temp_file)
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    /// Helper to create test vectors
    /// REFACTORED: Now uses vector_generator::random_seeded_with_prefix() with seed for determinism
    fn create_test_vectors(
        count: usize,
        dims: usize,
        seed: u64,
    ) -> Vec<proximadb_records::ProximaRecord> {
        vector_generator::random_seeded_with_prefix("test_vec", count, dims, seed)
    }

    #[tokio::test]
    async fn test_helix_basic_operations() {
        // Clean up old test files before starting
        cleanup_old_helix_test_files();

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let _config = HelixConfig::default();

        let engine = HelixEngine::new().await.unwrap();

        // Test engine properties
        assert_eq!(engine.engine_name(), "helix");
        assert_eq!(engine.engine_version(), "1.0.0");

        // Test flush
        let vectors = create_test_vectors(100, 128, 42);
        let query = vectors[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .unwrap_or_default(); // Store query before moving vectors

        // Create collection config with storage assignment
        let collection_config = Collection {
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let flush_params = FlushParameters {
            collection_id: Some("test_collection".to_string()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection_config),
            estimated_size: 0,
        };

        let flush_result = engine.do_flush(&flush_params).await.unwrap();
        assert_eq!(flush_result.entries_flushed.unwrap_or(0), 100);
        assert!(flush_result.bytes_written.unwrap_or(0) > 0);

        // Test search
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(5),
            distance_metric: Some(DistanceMetric::Euclidean),
            ..Default::default()
        });

        let collection = Arc::new(Collection {
            storage_class_status: None,
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Sst as i32),
                ..Default::default()
            }),
            stats: Some(proximadb::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
        });

        let ctx = StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // The first result should be the query vector itself
        assert_eq!(results[0].id, "test_vec_0");
        assert!(results[0].similarity.unwrap_or(0.0) > 0.999); // Should be very close to 1 for cosine similarity
    }

    #[tokio::test]
    async fn test_helix_compaction() {
        // Clean up old test files before starting
        cleanup_old_helix_test_files();

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();

        // Ensure /tmp directory exists (needed by HelixEngine::new())
        // Using std::fs directly since this is a simple directory check in tests
        let _ = std::fs::create_dir_all("/tmp");

        let mut _config = HelixConfig::default();
        _config.level0_file_num_compaction_trigger = 2;

        let engine = HelixEngine::new().await.unwrap();

        // Create collection config with storage assignment
        let collection_config = Collection {
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Create multiple flushes to trigger compaction
        for batch in 0..3 {
            let vectors = create_test_vectors(50, 128, batch);
            let flush_params = FlushParameters {
                collection_id: Some("test_collection".to_string()),
                vector_records: vectors,
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection_config.clone()),
                estimated_size: 0,
            };
            engine.do_flush(&flush_params).await.unwrap();
        }

        // Trigger manual compaction
        let compact_params = CompactionParameters {
            full: false,
            collection_id: Some("test_collection".to_string()),
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            priority: OperationPriority::Medium,
            collection_config: Some(collection_config.clone()),
            estimated_input_size: 0,
        };

        let compact_result = engine.do_compact(&compact_params).await.unwrap();
        assert!(compact_result.success, "Compaction should succeed");
        // Note: bytes_written may be 0 if compaction determined no work was needed
        // The important thing is that compaction succeeded and data remains searchable

        // Verify data is still searchable after compaction
        let query = vec![0.0; 128];
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(10),
            distance_metric: Some(DistanceMetric::Euclidean),
            ..Default::default()
        });

        let collection = Arc::new(Collection {
            storage_class_status: None,
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Sst as i32),
                ..Default::default()
            }),
            stats: Some(proximadb::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
        });

        let ctx = StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_helix_clustering_effectiveness() {
        // Clean up old test files before starting
        cleanup_old_helix_test_files();

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let _config = HelixConfig::default();

        let engine = HelixEngine::new().await.unwrap();

        // Create clustered data
        let mut all_vectors = Vec::new();

        // Cluster 1: vectors around [1.0, 0.0, ...]
        for i in 0..50 {
            let mut values = vec![1.0f32; 128];
            values[0] += (i as f32) * 0.01;
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "cluster".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String("1".to_string()),
                ),
            );
            all_vectors.push(proximadb_records::ProximaRecord {
                oid: format!("cluster1_vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default", "vector", 128, values,
                )],
                props,
                record_version: 1,
                created_at_ns: i as i64 * 1_000_000_000,
                updated_at_ns: i as i64 * 1_000_000_000,
                ..Default::default()
            });
        }

        // Cluster 2: vectors around [-1.0, 0.0, ...]
        for i in 0..50 {
            let mut values = vec![-1.0f32; 128];
            values[0] += (i as f32) * 0.01;
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "cluster".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String("2".to_string()),
                ),
            );
            all_vectors.push(proximadb_records::ProximaRecord {
                oid: format!("cluster2_vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default", "vector", 128, values,
                )],
                props,
                record_version: 1,
                created_at_ns: (50 + i) as i64 * 1_000_000_000,
                updated_at_ns: (50 + i) as i64 * 1_000_000_000,
                ..Default::default()
            });
        }

        // Create collection config with storage assignment
        let collection_config = Collection {
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Flush the data
        let flush_params = FlushParameters {
            collection_id: Some("test_collection".to_string()),
            vector_records: all_vectors,
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection_config),
            estimated_size: 0,
        };
        engine.do_flush(&flush_params).await.unwrap();

        // Search for a vector from cluster 1
        let query = vec![1.0; 128];
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(10),
            distance_metric: Some(DistanceMetric::Euclidean),
            ..Default::default()
        });

        let collection = Arc::new(Collection {
            storage_class_status: None,
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Sst as i32),
                ..Default::default()
            }),
            stats: Some(proximadb::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
        });

        let ctx = StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();

        // Most results should be from cluster 1
        let cluster1_count = results
            .iter()
            .filter(|r| r.id.starts_with("cluster1_"))
            .count();

        assert!(
            cluster1_count >= 8,
            "Expected at least 8/10 results from cluster 1, got {}",
            cluster1_count
        );
    }

    #[tokio::test]
    async fn test_helix_with_metadata_filtering() {
        // Clean up old test files before starting
        cleanup_old_helix_test_files();

        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

        let temp_dir = TempDir::new().unwrap();
        let _config = HelixConfig::default();

        let engine = HelixEngine::new().await.unwrap();

        // Create vectors with different metadata (String, Integer, Boolean, Float)
        let mut vectors: Vec<proximadb_records::ProximaRecord> = Vec::new();
        for i in 0..100 {
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "category".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(if i % 2 == 0 {
                        "even".to_string()
                    } else {
                        "odd".to_string()
                    }),
                ),
            );
            props.insert(
                "batch".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Float64((i / 10) as f64),
                ),
            );
            props.insert(
                "count".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Int64(i as i64 * 10),
                ),
            );
            props.insert(
                "enabled".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Boolean(i % 2 == 0),
                ),
            );
            vectors.push(proximadb_records::ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![i as f32 / 100.0; 128],
                )],
                props,
                record_version: 1,
                created_at_ns: i as i64 * 1_000_000_000,
                updated_at_ns: i as i64 * 1_000_000_000,
                ..Default::default()
            });
        }

        // Create collection config with filterable columns for type-safe filtering
        let collection_config = Collection {
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                filterable_columns: vec![
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "category".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableString
                                as i32,
                        indexed: false,
                        supports_range: false,
                        estimated_cardinality: Some(2),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "batch".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableFloat
                                as i32,
                        indexed: false,
                        supports_range: true,
                        estimated_cardinality: Some(10),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "count".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableInteger
                                as i32,
                        indexed: false,
                        supports_range: true,
                        estimated_cardinality: Some(100),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "enabled".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableBoolean
                                as i32,
                        indexed: false,
                        supports_range: false,
                        estimated_cardinality: Some(2),
                    },
                ],
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let flush_params = FlushParameters {
            collection_id: Some("test_collection".to_string()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection_config),
            estimated_size: 0,
        };
        engine.do_flush(&flush_params).await.unwrap();

        // Search with metadata filter
        use proximadb::core::search::{ComparisonOperator, FilterExpression};
        let filter = FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("even".to_string()),
        };

        let query = vec![0.5; 128];
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(10),
            distance_metric: Some(DistanceMetric::Euclidean),
            filter_expression: Some(filter),
            ..Default::default()
        });

        let collection = Arc::new(Collection {
            storage_class_status: None,
            id: "test_collection".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                filterable_columns: vec![
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "category".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableString
                                as i32,
                        indexed: false,
                        supports_range: false,
                        estimated_cardinality: Some(2),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "batch".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableFloat
                                as i32,
                        indexed: false,
                        supports_range: true,
                        estimated_cardinality: Some(10),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "count".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableInteger
                                as i32,
                        indexed: false,
                        supports_range: true,
                        estimated_cardinality: Some(100),
                    },
                    proximadb::proto::proximadb_v1::FilterableColumnSpec {
                        name: "enabled".to_string(),
                        data_type:
                            proximadb::proto::proximadb_v1::FilterableDataType::FilterableBoolean
                                as i32,
                        indexed: false,
                        supports_range: false,
                        estimated_cardinality: Some(2),
                    },
                ],
                ..Default::default()
            }),
            stats: Some(proximadb::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                index_size_bytes: 0,
                data_size_bytes: 0,
            }),
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
        });

        let ctx = StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();

        // All results should have even category (even indices: 0, 2, 4, ..., 98)
        assert!(results.len() <= 10, "Should return at most top_k results");
        for result in &results {
            let vec_id: usize = result
                .id
                .strip_prefix("vec_")
                .and_then(|s| s.parse().ok())
                .expect("Invalid ID format");
            assert_eq!(
                vec_id % 2,
                0,
                "Expected only even-indexed vectors (filter: category=even)"
            );
        }
    }

    #[tokio::test]
    async fn test_helix_integer_filter() {
        cleanup_old_helix_test_files();
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let engine = HelixEngine::new().await.unwrap();

        // Create vectors with integer metadata
        let mut vectors: Vec<proximadb_records::ProximaRecord> = Vec::new();
        for i in 0..100 {
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "count".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Int64(i as i64 * 10),
                ),
            );
            vectors.push(proximadb_records::ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![i as f32 / 100.0; 128],
                )],
                props,
                record_version: 1,
                created_at_ns: i as i64 * 1_000_000_000,
                updated_at_ns: i as i64 * 1_000_000_000,
                ..Default::default()
            });
        }

        let collection_config = Collection {
            id: "test_helix_int".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_helix_int".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                filterable_columns: vec![proximadb::proto::proximadb_v1::FilterableColumnSpec {
                    name: "count".to_string(),
                    data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableInteger
                        as i32,
                    indexed: false,
                    supports_range: true,
                    estimated_cardinality: Some(100),
                }],
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        engine
            .do_flush(&FlushParameters {
                collection_id: Some("test_helix_int".to_string()),
                vector_records: vectors,
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection_config.clone()),
                estimated_size: 0,
            })
            .await
            .unwrap();

        // Test: count >= 500 (should get indices 50..99 = 50 results)
        use proximadb::core::search::{ComparisonOperator, FilterExpression};
        let filter = FilterExpression::Comparison {
            field: "count".to_string(),
            operator: ComparisonOperator::GreaterThanOrEqual,
            value: serde_json::json!(500),
        };

        let query = vec![0.5; 128];
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(100),
            distance_metric: Some(DistanceMetric::Euclidean),
            filter_expression: Some(filter),
            ..Default::default()
        });

        let ctx = StorageQueryContext {
            search_params,
            collection: Arc::new(collection_config),
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();

        // Verify all results have count >= 500
        assert_eq!(
            results.len(),
            50,
            "Expected 50 results for count >= 500 (indices 50-99)"
        );
        for result in &results {
            let vec_id: usize = result
                .id
                .strip_prefix("vec_")
                .and_then(|s| s.parse().ok())
                .expect("Invalid ID format");
            assert!(vec_id >= 50, "Expected vec_id >= 50, got {}", vec_id);
        }
    }

    #[tokio::test]
    async fn test_helix_boolean_filter() {
        cleanup_old_helix_test_files();
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
        let temp_dir = TempDir::new().unwrap();
        let engine = HelixEngine::new().await.unwrap();

        // Create vectors with boolean metadata
        let mut vectors: Vec<proximadb_records::ProximaRecord> = Vec::new();
        for i in 0..100 {
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "enabled".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::Boolean(i % 2 == 0),
                ),
            );
            vectors.push(proximadb_records::ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![i as f32 / 100.0; 128],
                )],
                props,
                record_version: 1,
                created_at_ns: i as i64 * 1_000_000_000,
                updated_at_ns: i as i64 * 1_000_000_000,
                ..Default::default()
            });
        }

        let collection_config = Collection {
            id: "test_helix_bool".to_string(),
            config: Some(proximadb::proto::proximadb_v1::CollectionConfig {
                name: "test_helix_bool".to_string(),
                dimension: 128,
                distance_metric: Some(DistanceMetric::Euclidean as i32),
                storage_engine: Some(proximadb::proto::proximadb_v1::StorageEngine::Helix as i32),
                filterable_columns: vec![proximadb::proto::proximadb_v1::FilterableColumnSpec {
                    name: "enabled".to_string(),
                    data_type: proximadb::proto::proximadb_v1::FilterableDataType::FilterableBoolean
                        as i32,
                    indexed: false,
                    supports_range: false,
                    estimated_cardinality: Some(2),
                }],
                ..Default::default()
            }),
            storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        engine
            .do_flush(&FlushParameters {
                collection_id: Some("test_helix_bool".to_string()),
                vector_records: vectors,
                force: true,
                synchronous: true,
                hints: HashMap::new(),
                timeout_ms: None,
                trigger_compaction: false,
                batch_ids: vec![],
                collection_config: Some(collection_config.clone()),
                estimated_size: 0,
            })
            .await
            .unwrap();

        // Test: enabled = true (should get even indices = 50 results)
        use proximadb::core::search::{ComparisonOperator, FilterExpression};
        let filter = FilterExpression::Comparison {
            field: "enabled".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!(true),
        };

        let query = vec![0.5; 128];
        let search_params = Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(100),
            distance_metric: Some(DistanceMetric::Euclidean),
            filter_expression: Some(filter),
            ..Default::default()
        });

        let ctx = StorageQueryContext {
            search_params,
            collection: Arc::new(collection_config),
            metadata: StorageQueryMetadata::default(),
            user_context: None,
            tenant_context: None,
        };

        let results = engine.search_vectors_unified(&ctx).await.unwrap();

        // Verify all results have enabled = true
        assert_eq!(
            results.len(),
            50,
            "Expected 50 results for enabled=true (even indices)"
        );
        for result in &results {
            let vec_id: usize = result
                .id
                .strip_prefix("vec_")
                .and_then(|s| s.parse().ok())
                .expect("Invalid ID format");
            assert_eq!(
                vec_id % 2,
                0,
                "Expected only even-indexed vectors (enabled=true)"
            );
        }
    }
}
