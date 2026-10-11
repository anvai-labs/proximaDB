//! # Arrow-Native FileFormat API Integration Tests
//!
//! Tests the success criteria for the Arrow-Native FileFormat API:
//! 1. Schema creation and conversion to Arrow
//! 2. FileSplit abstraction for parallel reading
//! 3. Compute connectors (Spark/Trino/DuckDB)
//! 4. Hadoop compatibility shim

use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Schema Tests
// ============================================================================

mod schema_tests {
    use super::*;
    use proximadb::storage::schema::{ProximaColumn, ProximaSchema};
    use proximadb_data_model::{ProximaType, VectorElement};

    fn make_column(id: i32, name: &str, data_type: ProximaType, nullable: bool) -> ProximaColumn {
        ProximaColumn {
            id,
            object_id: None,
            name: name.to_string(),
            data_type,
            nullable,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
    }

    #[test]
    fn test_proxima_schema_creation() {
        let schema = ProximaSchema::from_columns(
            "test_schema".to_string(),
            vec![
                make_column(0, "id", ProximaType::String, false),
                make_column(
                    1,
                    "embedding",
                    ProximaType::DenseVector {
                        element: VectorElement::Float32,
                        dim: 128,
                    },
                    false,
                ),
                make_column(2, "metadata", ProximaType::Json, true),
            ],
            vec![0],
        );

        assert_eq!(schema.schema_id, "test_schema");
        assert_eq!(schema.version, 1);
        assert_eq!(schema.columns.len(), 3);
        assert!(schema.fingerprint > 0);
    }

    #[test]
    fn test_proxima_schema_to_arrow() {
        let schema = ProximaSchema::from_columns(
            "arrow_test".to_string(),
            vec![
                make_column(0, "id", ProximaType::String, false),
                make_column(1, "count", ProximaType::Int64, true),
            ],
            vec![0],
        );

        let arrow_schema = schema.to_arrow_schema();
        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "id");
        assert_eq!(arrow_schema.field(1).name(), "count");
    }

    #[test]
    fn test_schema_fingerprint_consistency() {
        let schema1 = ProximaSchema::from_columns(
            "fingerprint_test".to_string(),
            vec![make_column(0, "field1", ProximaType::String, false)],
            vec![0],
        );

        let schema2 = ProximaSchema::from_columns(
            "fingerprint_test".to_string(),
            vec![make_column(0, "field1", ProximaType::String, false)],
            vec![0],
        );

        // Same columns should produce same fingerprint
        assert_eq!(schema1.fingerprint, schema2.fingerprint);
    }

    #[test]
    fn test_legacy_vector_schema() {
        let schema = ProximaSchema::vector_record_schema(128);

        assert!(schema.is_legacy_vector_record);
        assert!(!schema.columns.is_empty());
    }
}

// ============================================================================
// FileSplit Tests
// ============================================================================

mod file_split_tests {

    use proximadb::storage::formats::{
        CacheStatus, FileSplit, SpatialBounds, SplitLocality, SplitPlanner, SplitType, StorageTier,
    };

    #[test]
    fn test_block_split_creation() {
        let split =
            FileSplit::new_block("/data/collection/file.sst".to_string(), 0, 0, 65536, 1000);

        assert_eq!(split.split_id, "/data/collection/file.sst:block:0");
        assert_eq!(split.statistics.row_count, Some(1000));
        assert!(matches!(
            split.split_type,
            SplitType::Block { block_id: 0, .. }
        ));
    }

    #[test]
    fn test_row_group_split_creation() {
        let split = FileSplit::new_row_group(
            "/data/collection/file.parquet".to_string(),
            0,
            0,
            1048576,
            10000,
        );

        assert!(split.split_id.contains(":rg:"));
        assert_eq!(split.statistics.row_count, Some(10000));
        assert!(matches!(split.split_type, SplitType::RowGroup { .. }));
    }

    #[test]
    fn test_hilbert_range_split_creation() {
        let split = FileSplit::new_hilbert_range(
            "/data/collection/file.helix".to_string(),
            0,
            1000,
            8,
            0,
            65536,
        );

        assert!(split.split_id.contains(":hilbert:"));
        assert!(matches!(split.split_type, SplitType::HilbertRange { .. }));

        // Should have spatial bounds
        assert!(split.statistics.spatial_bounds.is_some());
        if let Some(SpatialBounds::Hilbert {
            min_code,
            max_code,
            order,
        }) = split.statistics.spatial_bounds
        {
            assert_eq!(min_code, 0);
            assert_eq!(max_code, 1000);
            assert_eq!(order, 8);
        }
    }

    #[test]
    fn test_superblock_split_creation() {
        let split = FileSplit::new_superblock(
            "/data/collection/file.swift".to_string(),
            0,
            vec![0, 1, 2, 3],
            0,
            262144,
        );

        assert!(split.split_id.contains(":superblock:"));
        if let SplitType::SuperBlock {
            block_count,
            block_ids,
            ..
        } = &split.split_type
        {
            assert_eq!(*block_count, 4);
            assert_eq!(block_ids.len(), 4);
        } else {
            panic!("Expected SuperBlock split type");
        }
    }

    #[test]
    fn test_split_planner() {
        let planner = SplitPlanner::default();

        let splits = vec![
            FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f1.sst".to_string(), 1, 1000, 2000, 200),
            FileSplit::new_block("/f2.sst".to_string(), 0, 0, 1500, 150),
            FileSplit::new_block("/f2.sst".to_string(), 1, 1500, 2500, 250),
        ];

        let partitions = planner.plan_splits(splits, 2);

        // Should distribute into 2 partitions
        assert_eq!(partitions.len(), 2);

        // Total splits should be preserved
        let total_splits: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total_splits, 4);
    }

    #[test]
    fn test_split_estimated_cost() {
        let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

        // Default (unknown) cache status
        let cost1 = split.estimated_cost();

        // Set to cached - should have lower cost
        split.locality.cache_status = CacheStatus::Cached;
        let cost2 = split.estimated_cost();

        // Set to remote - should have higher cost
        split.locality.cache_status = CacheStatus::Remote;
        let cost3 = split.estimated_cost();

        assert!(cost2 < cost1, "Cached should be cheaper than unknown");
        assert!(
            cost3 > cost1,
            "Remote should be more expensive than unknown"
        );
    }

    #[test]
    fn test_split_locality() {
        let locality = SplitLocality {
            preferred_hosts: vec!["host1.cluster".to_string(), "host2.cluster".to_string()],
            storage_tier: StorageTier::Hot,
            cache_status: CacheStatus::Cached,
        };

        let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100)
            .with_locality(locality);

        assert_eq!(split.locality.preferred_hosts.len(), 2);
        assert_eq!(split.locality.storage_tier, StorageTier::Hot);
        assert_eq!(split.locality.cache_status, CacheStatus::Cached);
    }
}

// ============================================================================
// Spark Connector Tests
// ============================================================================

mod spark_tests {
    use super::*;
    use proximadb::storage::formats::FileSplit;
    use proximadb::storage::schema::{ProximaColumn, ProximaSchema};
    use proximadb_data_model::ProximaType;
    use proximadb_embedded::spark::{
        SparkConnectorConfig, SparkFilter, SparkFilterType, SparkInputPartition, SparkScanBuilder,
        SparkTable, SparkWriteBuilder, SparkWriteMode,
    };

    fn make_column(id: i32, name: &str, data_type: ProximaType) -> ProximaColumn {
        ProximaColumn {
            id,
            object_id: None,
            name: name.to_string(),
            data_type,
            nullable: false,
            default_value: None,
            comment: None,
            properties: HashMap::new(),
            is_deleted: false,
            original_id: None,
        }
    }

    #[test]
    fn test_spark_config_default() {
        let config = SparkConnectorConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5678);
        assert!(config.enable_filter_pushdown);
        assert!(config.enable_projection_pushdown);
        assert!(config.enable_aggregate_pushdown);
    }

    #[test]
    fn test_spark_input_partition_from_splits() {
        let splits = vec![
            FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f1.sst".to_string(), 1, 1000, 2000, 200),
        ];

        let partition = SparkInputPartition::from_splits(0, splits);

        assert_eq!(partition.partition_id, 0);
        assert_eq!(partition.splits.len(), 2);
        assert_eq!(partition.estimated_rows, Some(300)); // 100 + 200
    }

    #[test]
    fn test_spark_scan_builder() {
        let proxima_schema = Arc::new(ProximaSchema::from_columns(
            "test".to_string(),
            vec![
                make_column(0, "id", ProximaType::String),
                make_column(1, "value", ProximaType::Int64),
            ],
            vec![0],
        ));

        let table = SparkTable {
            name: "test_table".to_string(),
            schema: proxima_schema.to_arrow_schema(),
            proxima_schema,
            properties: HashMap::new(),
            partition_columns: Vec::new(),
        };

        let builder = SparkScanBuilder::new(table)
            .with_projection(vec!["id".to_string()])
            .with_filter(SparkFilter {
                filter_type: SparkFilterType::EqualTo,
                column: Some("id".to_string()),
                value: Some(serde_json::json!("test_id")),
                children: Vec::new(),
            })
            .with_limit(100);

        assert!(builder.projection.is_some());
        assert_eq!(builder.filters.len(), 1);
        assert_eq!(builder.limit, Some(100));
    }

    #[test]
    fn test_spark_filter_types() {
        // Comparison filters
        let eq_filter = SparkFilter {
            filter_type: SparkFilterType::EqualTo,
            column: Some("col".to_string()),
            value: Some(serde_json::json!(42)),
            children: Vec::new(),
        };
        assert_eq!(eq_filter.filter_type, SparkFilterType::EqualTo);

        // Logical filters
        let and_filter = SparkFilter {
            filter_type: SparkFilterType::And,
            column: None,
            value: None,
            children: vec![
                SparkFilter {
                    filter_type: SparkFilterType::GreaterThan,
                    column: Some("a".to_string()),
                    value: Some(serde_json::json!(10)),
                    children: Vec::new(),
                },
                SparkFilter {
                    filter_type: SparkFilterType::LessThan,
                    column: Some("a".to_string()),
                    value: Some(serde_json::json!(100)),
                    children: Vec::new(),
                },
            ],
        };
        assert_eq!(and_filter.children.len(), 2);
    }

    #[test]
    fn test_spark_write_builder() {
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, true),
        ]));

        let builder = SparkWriteBuilder::new("output_table".to_string(), schema)
            .with_mode(SparkWriteMode::Overwrite)
            .with_partition_by(vec!["date".to_string()])
            .with_option("compression".to_string(), "lz4".to_string());

        assert_eq!(builder.table_name, "output_table");
        assert_eq!(builder.mode, SparkWriteMode::Overwrite);
        assert_eq!(builder.partition_by, vec!["date"]);
        assert!(builder.options.contains_key("compression"));
    }
}

// ============================================================================
// Trino Connector Tests
// ============================================================================

mod trino_tests {
    use super::*;
    use proximadb::connectors::{
        TrinoConnectorConfig, TrinoDomain, TrinoRange, TrinoSplit, TrinoTupleDomain,
    };
    use proximadb::storage::formats::{FileSplit, SplitLocality, SplitStatistics};

    #[test]
    fn test_trino_config_default() {
        let config = TrinoConnectorConfig::default();
        assert_eq!(config.flight_endpoint, "grpc://localhost:5680");
        assert!(config.enable_predicate_pushdown);
        assert!(config.enable_dynamic_filtering);
        assert!(config.enable_topn_pushdown);
    }

    #[test]
    fn test_trino_tuple_domain_all() {
        let domain = TrinoTupleDomain::all();
        assert!(domain.is_all);
        assert!(!domain.is_none);
        assert!(domain.domains.is_empty());
    }

    #[test]
    fn test_trino_tuple_domain_none() {
        let domain = TrinoTupleDomain::none();
        assert!(!domain.is_all);
        assert!(domain.is_none);
    }

    #[test]
    fn test_trino_tuple_domain_with_constraints() {
        let mut domains = HashMap::new();
        domains.insert(
            "category".to_string(),
            TrinoDomain {
                column: "category".to_string(),
                ranges: vec![TrinoRange::equal(serde_json::json!("science"))],
                null_allowed: false,
            },
        );

        let constrained = TrinoTupleDomain::with_domains(domains);
        assert!(!constrained.is_all);
        assert!(!constrained.is_none);
        assert!(constrained.domains.contains_key("category"));
    }

    #[test]
    fn test_trino_range_operations() {
        // Equal range
        let eq = TrinoRange::equal(serde_json::json!(42));
        assert!(eq.low_inclusive);
        assert!(eq.high_inclusive);
        assert_eq!(eq.low, eq.high);

        // Greater than
        let gt = TrinoRange::greater_than(serde_json::json!(10), false);
        assert!(!gt.low_inclusive);
        assert!(gt.high.is_none());

        // Less than or equal
        let lte = TrinoRange::less_than(serde_json::json!(100), true);
        assert!(lte.high_inclusive);
        assert!(lte.low.is_none());

        // Between
        let between =
            TrinoRange::between(serde_json::json!(10), true, serde_json::json!(100), false);
        assert!(between.low_inclusive);
        assert!(!between.high_inclusive);
    }

    #[test]
    fn test_trino_split_from_file_split() {
        let file_split = FileSplit {
            split_id: "test:0".to_string(),
            file_path: "/data/file.sst".to_string(),
            offset: 0,
            length: 65536,
            split_type: proximadb::storage::formats::SplitType::Block {
                block_id: 0,
                record_count: 1000,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality {
                preferred_hosts: vec!["node1".to_string()],
                ..Default::default()
            },
            object_size: None,
        };

        let trino_split = TrinoSplit::from_file_split(
            "proximadb".to_string(),
            "default".to_string(),
            "vectors".to_string(),
            file_split,
        );

        assert_eq!(trino_split.split_id, "test:0");
        assert_eq!(trino_split.catalog, "proximadb");
        assert_eq!(trino_split.schema, "default");
        assert_eq!(trino_split.table, "vectors");
        assert!(trino_split.remotely_accessible);
    }
}

// ============================================================================
// DuckDB Connector Tests
// ============================================================================

mod duckdb_tests {
    use super::*;
    use arrow::datatypes::Schema as ArrowSchema;
    use httpmock::{Method, MockServer};
    use proximadb::connectors::{
        DuckDBColumnRef, DuckDBConnectorConfig, DuckDBFilter, DuckDBFilterType, DuckDBGlobalState,
        DuckDBInitData, DuckDBLocalState, DuckDBTableScan, DuckDBVectorSearchParams,
    };
    use proximadb::storage::formats::{FileSplit, SplitLocality, SplitStatistics, SplitType};
    use proximadb_distance_types::DistanceMetric;

    #[test]
    fn test_duckdb_config_default() {
        let config = DuckDBConnectorConfig::default();
        assert_eq!(config.server_url, "http://localhost:5678");
        assert!(config.enable_filter_pushdown);
        assert!(config.enable_projection_pushdown);
        assert!(config.enable_parallel_scan);
        assert_eq!(config.max_threads, 8);
    }

    #[tokio::test]
    async fn test_duckdb_table_scan_bind() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(Method::GET)
                    .path("/api/v2/collections/test_collection/schema");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(r#"{"name":"test_collection","columns":[]}"#);
            })
            .await;

        let config = DuckDBConnectorConfig {
            server_url: server.base_url(),
            ..DuckDBConnectorConfig::default()
        };
        let mut scan = DuckDBTableScan::new(config);

        let result = scan.bind("test_collection").await;
        assert!(result.is_ok(), "bind() must succeed: {result:?}");

        let bind_data = result.unwrap();
        assert_eq!(bind_data.collection, "test_collection");
        mock.assert_async().await;
    }

    #[test]
    fn test_duckdb_table_scan_pushdown() {
        let config = DuckDBConnectorConfig::default();
        let scan = DuckDBTableScan::new(config);

        assert!(scan.supports_filter_pushdown());
        assert!(scan.supports_projection_pushdown());
        assert_eq!(scan.max_threads(), 8);
    }

    #[test]
    fn test_duckdb_filter_types() {
        let filter = DuckDBFilter {
            filter_type: DuckDBFilterType::Equal,
            column_ref: Some(DuckDBColumnRef {
                column_idx: 0,
                column_name: "category".to_string(),
            }),
            constant: Some(serde_json::json!("science")),
            children: Vec::new(),
        };

        assert_eq!(filter.filter_type, DuckDBFilterType::Equal);
        assert!(filter.column_ref.is_some());
    }

    #[test]
    fn test_duckdb_vector_search_params() {
        let params = DuckDBVectorSearchParams {
            collection: "embeddings".to_string(),
            query_vector: vec![0.1, 0.2, 0.3, 0.4],
            top_k: 10,
            metric: DistanceMetric::Cosine,
            filter: None,
            include_distances: true,
        };

        assert_eq!(params.collection, "embeddings");
        assert_eq!(params.query_vector.len(), 4);
        assert_eq!(params.top_k, 10);
        assert!(params.include_distances);
    }

    #[test]
    fn test_duckdb_distance_metrics() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::L2);

        let _l2 = DistanceMetric::L2;
        let _cosine = DistanceMetric::Cosine;
        let _inner = DistanceMetric::InnerProduct;
        let _l1 = DistanceMetric::L1;
    }

    #[test]
    fn test_duckdb_init_data() {
        let splits = vec![FileSplit {
            split_id: "s0".to_string(),
            file_path: "/data/0.sst".to_string(),
            offset: 0,
            length: 1024,
            split_type: SplitType::Block {
                block_id: 0,
                record_count: 100,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality::default(),
            object_size: None,
        }];

        let init_data = DuckDBInitData::new(splits);
        assert_eq!(init_data.current_split, 0);
        assert_eq!(init_data.splits.len(), 1);
        assert!(!init_data.finished);
    }

    #[test]
    fn test_duckdb_global_state() {
        let schema = Arc::new(ArrowSchema::empty());
        let splits = vec![
            FileSplit {
                split_id: "s0".to_string(),
                file_path: String::new(),
                offset: 0,
                length: 0,
                split_type: SplitType::ByteRange {
                    estimated_records: 0,
                },
                statistics: SplitStatistics::default(),
                locality: SplitLocality::default(),
                object_size: None,
            },
            FileSplit {
                split_id: "s1".to_string(),
                file_path: String::new(),
                offset: 0,
                length: 0,
                split_type: SplitType::ByteRange {
                    estimated_records: 0,
                },
                statistics: SplitStatistics::default(),
                locality: SplitLocality::default(),
                object_size: None,
            },
        ];

        let state = DuckDBGlobalState::new("test".to_string(), schema, splits, 4);
        assert_eq!(state.max_threads, 4);
        assert_eq!(state.collection, "test");

        // Get splits atomically
        assert!(state.get_next_split().is_some());
        assert!(state.get_next_split().is_some());
        assert!(state.get_next_split().is_none()); // No more splits
    }

    #[test]
    fn test_duckdb_local_state() {
        let state = DuckDBLocalState::new(0);
        assert_eq!(state.thread_id, 0);
        assert!(state.current_split.is_none());
        assert!(state.batch_buffer.is_empty());
        assert_eq!(state.rows_read, 0);
    }
}

// ============================================================================
// Hadoop Compatibility Tests
// ============================================================================

mod hadoop_tests {
    use super::*;
    use proximadb::connectors::{
        HadoopInputSplit, HadoopShimConfig, HadoopWritable, ProximaInputFormat,
        ProximaOutputFormat, ProximaSerDe,
    };
    use proximadb::storage::formats::{FileSplit, SplitLocality, SplitStatistics, SplitType};

    #[test]
    fn test_hadoop_config_default() {
        let config = HadoopShimConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5678);
        assert!(config.speculative_execution);
        assert_eq!(config.split_size_hint, 128 * 1024 * 1024);
    }

    #[test]
    fn test_hadoop_input_split() {
        let file_split = FileSplit {
            split_id: "test:0".to_string(),
            file_path: "/data/file.sst".to_string(),
            offset: 0,
            length: 65536,
            split_type: SplitType::Block {
                block_id: 0,
                record_count: 1000,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality {
                preferred_hosts: vec!["node1.cluster".to_string()],
                ..Default::default()
            },
            object_size: None,
        };

        let hadoop_split = HadoopInputSplit::from_file_split(file_split);

        assert_eq!(hadoop_split.split_id, "test:0");
        assert_eq!(hadoop_split.get_length(), 65536);
        assert_eq!(hadoop_split.get_locations(), &["node1.cluster"]);

        // Test serialization round-trip
        let bytes = hadoop_split.write_fields();
        assert!(!bytes.is_empty());

        let deserialized = HadoopInputSplit::read_fields(&bytes);
        assert!(deserialized.is_some());
        assert_eq!(deserialized.unwrap().split_id, "test:0");
    }

    #[test]
    fn test_hadoop_writable_primitives() {
        // Integer
        let int_w = HadoopWritable::IntWritable(42);
        assert_eq!(int_w.to_json(), serde_json::json!(42));

        // Long
        let long_w = HadoopWritable::LongWritable(9999999999i64);
        assert_eq!(long_w.to_json(), serde_json::json!(9999999999i64));

        // Float
        let float_w = HadoopWritable::FloatWritable(3.5);
        let json = float_w.to_json();
        assert!((json.as_f64().unwrap() - 3.5).abs() < 0.001);

        // Text
        let text_w = HadoopWritable::Text("hello".to_string());
        assert_eq!(text_w.to_json(), serde_json::json!("hello"));

        // Boolean
        let bool_w = HadoopWritable::BooleanWritable(true);
        assert_eq!(bool_w.to_json(), serde_json::json!(true));

        // Null
        let null_w = HadoopWritable::NullWritable;
        assert!(null_w.to_json().is_null());
    }

    #[test]
    fn test_hadoop_writable_collections() {
        // Array
        let arr = HadoopWritable::ArrayWritable(vec![
            HadoopWritable::IntWritable(1),
            HadoopWritable::IntWritable(2),
            HadoopWritable::IntWritable(3),
        ]);
        let json = arr.to_json();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 3);

        // Map
        let mut map = HashMap::new();
        map.insert("key1".to_string(), HadoopWritable::IntWritable(1));
        map.insert(
            "key2".to_string(),
            HadoopWritable::Text("value".to_string()),
        );

        let map_w = HadoopWritable::MapWritable(map);
        let json = map_w.to_json();
        assert!(json.is_object());
        assert!(json.get("key1").is_some());
        assert!(json.get("key2").is_some());
    }

    #[test]
    fn test_hadoop_writable_from_json() {
        // Round-trip conversion
        let original = HadoopWritable::MapWritable({
            let mut m = HashMap::new();
            m.insert("count".to_string(), HadoopWritable::IntWritable(42));
            m.insert("name".to_string(), HadoopWritable::Text("test".to_string()));
            m
        });

        let json = original.to_json();
        let back = HadoopWritable::from_json(&json);

        if let HadoopWritable::MapWritable(m) = back {
            assert_eq!(m.len(), 2);
        } else {
            panic!("Expected MapWritable");
        }
    }

    #[test]
    fn test_input_format() {
        let config = HadoopShimConfig {
            collection: "test_collection".to_string(),
            ..Default::default()
        };

        let input_format = ProximaInputFormat::new(config);
        let splits = input_format.get_splits(4);

        assert!(!splits.is_empty());
    }

    #[test]
    fn test_output_format_validation() {
        // Valid config
        let valid_config = HadoopShimConfig {
            collection: "test_collection".to_string(),
            ..Default::default()
        };
        let output_format = ProximaOutputFormat::new(valid_config);
        assert!(output_format.check_output_specs().is_ok());

        // Invalid config (no collection)
        let invalid_config = HadoopShimConfig::default();
        let output_format = ProximaOutputFormat::new(invalid_config);
        assert!(output_format.check_output_specs().is_err());
    }

    #[test]
    fn test_proxima_serde() {
        let mut serde = ProximaSerDe::new();

        let mut props = HashMap::new();
        props.insert("columns".to_string(), "id,name,score".to_string());
        props.insert(
            "columns.types".to_string(),
            "string:string:double".to_string(),
        );

        let result = serde.initialize(&props);
        assert!(result.is_ok());
    }
}
