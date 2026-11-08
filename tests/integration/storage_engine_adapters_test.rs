//! # Engine Adapters Unit Tests
//!
//! Tests for the engine-specific TableProvider adapters (SST, HELIX, VIPER).
//!
//! ## Test Coverage
//!
//! - Split generation from file metadata
//! - Pruning statistics accuracy
//! - Predicate pushdown capability flags
//! - Schema generation and projection
//! - Split reader interface compliance

#![cfg(feature = "datafusion-integration")]

use std::collections::HashMap;

use proximadb::datafusion::{CollectionInfo, EngineType, PruningStatistics};
use proximadb::storage::formats::{
    CacheStatus, ColumnBounds, FileSplit, ScalarPredicate, ScalarValue, SpatialBounds,
    SplitStatistics, SplitType, StorageTier,
};

// ============================================================================
// SST Adapter Tests
// ============================================================================

mod sst_adapter_tests {
    use super::*;

    fn create_sst_collection_info() -> CollectionInfo {
        CollectionInfo::new("test_sst".to_string(), 128, EngineType::Sst)
            .with_vector_count(10000)
            .with_storage_size(1024 * 1024 * 100) // 100MB
            .with_file_count(10)
            .with_base_path("/data/sst_test".to_string())
    }

    #[test]
    fn test_sst_collection_info() {
        let info = create_sst_collection_info();
        assert_eq!(info.engine_type, EngineType::Sst);
        assert_eq!(info.dimension, 128);
        assert_eq!(info.vector_count, 10000);
        assert_eq!(info.file_count, 10);
    }

    #[test]
    fn test_sst_block_split_creation() {
        let split = FileSplit::new_block("/data/test/file.sst".to_string(), 0, 0, 65536, 1000);

        assert_eq!(split.file_path, "/data/test/file.sst");
        assert_eq!(split.offset, 0);
        assert_eq!(split.length, 65536);
        assert_eq!(split.statistics.row_count, Some(1000));

        // Verify split type
        match split.split_type {
            SplitType::Block {
                block_id,
                record_count,
            } => {
                assert_eq!(block_id, 0);
                assert_eq!(record_count, 1000);
            }
            _ => panic!("Expected Block split type"),
        }
    }

    #[test]
    fn test_sst_split_with_bloom_filter() {
        let mut split = FileSplit::new_block("/data/test/file.sst".to_string(), 0, 0, 65536, 1000);

        // Add bloom filter
        split.statistics.bloom_filter = Some(vec![0xFF; 32]);

        assert!(split.statistics.bloom_filter.is_some());
        assert_eq!(split.statistics.bloom_filter.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn test_sst_split_with_column_statistics() {
        let mut split = FileSplit::new_block("/data/test/file.sst".to_string(), 0, 0, 65536, 1000);

        // Add column statistics
        split.statistics.column_stats.insert(
            "timestamp".to_string(),
            ColumnBounds {
                min: Some(serde_json::json!(1704067200000i64)),
                max: Some(serde_json::json!(1704153600000i64)),
                null_count: 0,
                distinct_count: Some(1000),
            },
        );

        assert!(split.statistics.column_stats.contains_key("timestamp"));
    }

    #[test]
    fn test_sst_pruning_statistics_from_splits() {
        let splits = vec![
            create_sst_split_with_stats(0, true, false),
            create_sst_split_with_stats(1, true, true),
            create_sst_split_with_stats(2, false, false),
        ];

        let stats = PruningStatistics::from_splits(&splits);

        assert_eq!(stats.total_splits, 3);
        assert_eq!(stats.splits_with_stats, 2); // 2 splits have row_count
        assert_eq!(stats.splits_with_bloom, 1); // 1 split has bloom filter
        assert!(stats.is_pruning_effective());
    }

    fn create_sst_split_with_stats(block_id: u32, has_stats: bool, has_bloom: bool) -> FileSplit {
        let mut split = FileSplit::new_block(
            format!("/data/file_{}.sst", block_id),
            block_id,
            block_id as u64 * 65536,
            65536,
            1000,
        );

        if has_stats {
            split.statistics.row_count = Some(1000);
        } else {
            split.statistics.row_count = None;
        }

        if has_bloom {
            split.statistics.bloom_filter = Some(vec![0u8; 32]);
        }

        split
    }

    #[test]
    fn test_sst_split_estimated_cost() {
        let mut split = FileSplit::new_block("/data/test/file.sst".to_string(), 0, 0, 65536, 1000);

        // Test with different cache statuses
        split.locality.cache_status = CacheStatus::Cached;
        let cached_cost = split.estimated_cost();

        split.locality.cache_status = CacheStatus::Remote;
        let remote_cost = split.estimated_cost();

        // Remote should cost more than cached
        assert!(remote_cost > cached_cost);
    }
}

// ============================================================================
// HELIX Adapter Tests
// ============================================================================

mod helix_adapter_tests {
    use super::*;

    fn create_helix_collection_info() -> CollectionInfo {
        CollectionInfo::new("test_helix".to_string(), 768, EngineType::Helix)
            .with_vector_count(100000)
            .with_storage_size(1024 * 1024 * 500) // 500MB
            .with_file_count(20)
            .with_base_path("/data/helix_test".to_string())
    }

    #[test]
    fn test_helix_collection_info() {
        let info = create_helix_collection_info();
        assert_eq!(info.engine_type, EngineType::Helix);
        assert_eq!(info.dimension, 768);
        assert_eq!(info.vector_count, 100000);
    }

    #[test]
    fn test_helix_hilbert_range_split_creation() {
        let split = FileSplit::new_hilbert_range(
            "/data/test/file.helix".to_string(),
            1000,
            2000,
            16,
            0,
            1024 * 1024,
        );

        assert_eq!(split.file_path, "/data/test/file.helix");

        // Verify split type
        match split.split_type {
            SplitType::HilbertRange {
                start_code,
                end_code,
                hilbert_order,
            } => {
                assert_eq!(start_code, 1000);
                assert_eq!(end_code, 2000);
                assert_eq!(hilbert_order, 16);
            }
            _ => panic!("Expected HilbertRange split type"),
        }

        // Verify spatial bounds
        match split.statistics.spatial_bounds {
            Some(SpatialBounds::Hilbert {
                min_code,
                max_code,
                order,
            }) => {
                assert_eq!(min_code, 1000);
                assert_eq!(max_code, 2000);
                assert_eq!(order, 16);
            }
            _ => panic!("Expected Hilbert spatial bounds"),
        }
    }

    #[test]
    fn test_helix_split_with_centroid() {
        let mut split = FileSplit::new_hilbert_range(
            "/data/test/file.helix".to_string(),
            1000,
            2000,
            16,
            0,
            1024 * 1024,
        );

        // Add centroid for vector pruning
        let centroid = vec![0.1f32; 768];
        split.statistics.centroid = Some(centroid.clone());

        assert!(split.statistics.centroid.is_some());
        assert_eq!(split.statistics.centroid.as_ref().unwrap().len(), 768);
    }

    #[test]
    fn test_helix_vector_pruning() {
        let mut split = FileSplit::new_hilbert_range(
            "/data/test/file.helix".to_string(),
            1000,
            2000,
            16,
            0,
            1024 * 1024,
        );

        // Add centroid at origin
        split.statistics.centroid = Some(vec![0.0f32; 3]);
        split.statistics.spatial_bounds = Some(SpatialBounds::BoundingBox {
            min_corner: vec![-1.0, -1.0, -1.0],
            max_corner: vec![1.0, 1.0, 1.0],
        });

        // Query close to centroid - should NOT be pruned
        let query_close = vec![0.5, 0.5, 0.5];
        assert!(!split.can_prune_vector(&query_close, 10.0));

        // Query very far from centroid - should be pruned
        let query_far = vec![100.0, 100.0, 100.0];
        assert!(split.can_prune_vector(&query_far, 1.0));
    }

    #[test]
    fn test_helix_split_cost_calculation() {
        let split = FileSplit::new_hilbert_range(
            "/data/test/file.helix".to_string(),
            1000,
            2000,
            16,
            0,
            1024 * 1024, // 1MB
        );

        let cost = split.split_cost();

        // Hilbert splits have decode complexity > 1.0
        assert!(cost.decode_complexity > 1.0);
        assert!(cost.decode_complexity < 2.0); // But not too high
    }

    #[test]
    fn test_helix_pruning_statistics() {
        let splits = vec![
            create_helix_split_with_centroid(0, true),
            create_helix_split_with_centroid(1, true),
            create_helix_split_with_centroid(2, false),
        ];

        let stats = PruningStatistics::from_splits(&splits);

        assert_eq!(stats.total_splits, 3);
        assert_eq!(stats.splits_with_centroid, 2);
    }

    fn create_helix_split_with_centroid(index: usize, has_centroid: bool) -> FileSplit {
        let mut split = FileSplit::new_hilbert_range(
            format!("/data/file_{}.helix", index),
            index as u64 * 1000,
            (index + 1) as u64 * 1000 - 1,
            16,
            0,
            1024 * 1024,
        );

        split.statistics.row_count = Some(10000);

        if has_centroid {
            split.statistics.centroid = Some(vec![0.0f32; 768]);
        }

        split
    }
}

// ============================================================================
// VIPER Adapter Tests
// ============================================================================

mod viper_adapter_tests {
    use super::*;

    fn create_viper_collection_info() -> CollectionInfo {
        CollectionInfo::new("test_viper".to_string(), 1536, EngineType::Viper)
            .with_vector_count(1000000)
            .with_storage_size(1024 * 1024 * 1024 * 2) // 2GB
            .with_file_count(50)
            .with_base_path("/data/viper_test".to_string())
    }

    #[test]
    fn test_viper_collection_info() {
        let info = create_viper_collection_info();
        assert_eq!(info.engine_type, EngineType::Viper);
        assert_eq!(info.dimension, 1536);
        assert_eq!(info.vector_count, 1000000);
    }

    #[test]
    fn test_viper_row_group_split_creation() {
        let split = FileSplit::new_row_group(
            "/data/test/file.parquet".to_string(),
            0,
            0,
            50 * 1024 * 1024, // 50MB
            128000,
        );

        assert_eq!(split.file_path, "/data/test/file.parquet");
        assert_eq!(split.statistics.row_count, Some(128000));

        // Verify split type
        match split.split_type {
            SplitType::RowGroup {
                row_group_index,
                row_count,
            } => {
                assert_eq!(row_group_index, 0);
                assert_eq!(row_count, 128000);
            }
            _ => panic!("Expected RowGroup split type"),
        }
    }

    #[test]
    fn test_viper_split_with_column_statistics() {
        let mut split = FileSplit::new_row_group(
            "/data/test/file.parquet".to_string(),
            0,
            0,
            50 * 1024 * 1024,
            128000,
        );

        // Add column statistics for predicate pushdown
        let mut column_stats = HashMap::new();
        column_stats.insert(
            "price".to_string(),
            ColumnBounds {
                min: Some(serde_json::json!(10.0)),
                max: Some(serde_json::json!(1000.0)),
                null_count: 100,
                distinct_count: Some(500),
            },
        );
        column_stats.insert(
            "category".to_string(),
            ColumnBounds {
                min: Some(serde_json::json!("A")),
                max: Some(serde_json::json!("Z")),
                null_count: 0,
                distinct_count: Some(26),
            },
        );

        split.statistics.column_stats = column_stats;

        assert!(split.statistics.column_stats.contains_key("price"));
        assert!(split.statistics.column_stats.contains_key("category"));
    }

    #[test]
    fn test_viper_scalar_predicate_pruning() {
        let bounds = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // Test equality pruning
        assert!(bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(5))));
        assert!(bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(150))));
        assert!(!bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(50))));

        // Test range pruning
        assert!(bounds.can_prune(&ScalarPredicate::GreaterThan(ScalarValue::Int64(100))));
        assert!(bounds.can_prune(&ScalarPredicate::LessThan(ScalarValue::Int64(10))));
        assert!(!bounds.can_prune(&ScalarPredicate::GreaterThan(ScalarValue::Int64(50))));

        // Test between pruning
        assert!(bounds.can_prune(&ScalarPredicate::Between(
            ScalarValue::Int64(200),
            ScalarValue::Int64(300)
        )));
        assert!(!bounds.can_prune(&ScalarPredicate::Between(
            ScalarValue::Int64(50),
            ScalarValue::Int64(80)
        )));
    }

    #[test]
    fn test_viper_split_cost_row_group() {
        let split = FileSplit::new_row_group(
            "/data/test/file.parquet".to_string(),
            0,
            0,
            50 * 1024 * 1024,
            128000,
        );

        let cost = split.split_cost();

        // Row groups are efficient (columnar format)
        assert!(cost.decode_complexity < 1.0);
        assert_eq!(cost.estimated_rows, 128000);
    }

    #[test]
    fn test_viper_pruning_statistics() {
        let splits = vec![
            create_viper_split_with_stats(0),
            create_viper_split_with_stats(1),
            create_viper_split_with_stats(2),
        ];

        let stats = PruningStatistics::from_splits(&splits);

        assert_eq!(stats.total_splits, 3);
        assert_eq!(stats.splits_with_stats, 3); // All have row counts
        assert!(stats.columns_with_stats.contains(&"price".to_string()));
        assert!(stats.is_pruning_effective());
    }

    fn create_viper_split_with_stats(index: usize) -> FileSplit {
        let mut split = FileSplit::new_row_group(
            format!("/data/file_{}.parquet", index),
            index,
            index as u64 * 50 * 1024 * 1024,
            50 * 1024 * 1024,
            128000,
        );

        split.statistics.column_stats.insert(
            "price".to_string(),
            ColumnBounds {
                min: Some(serde_json::json!(10.0 + index as f64 * 100.0)),
                max: Some(serde_json::json!(100.0 + index as f64 * 100.0)),
                null_count: 0,
                distinct_count: Some(500),
            },
        );

        split
    }

    #[test]
    fn test_viper_storage_tier_assignment() {
        let mut split = FileSplit::new_row_group(
            "/data/test/file.parquet".to_string(),
            0,
            0,
            50 * 1024 * 1024,
            128000,
        );

        // VIPER/Parquet files are typically on warm/cold tier
        split.locality.storage_tier = StorageTier::Warm;
        assert_eq!(split.locality.storage_tier, StorageTier::Warm);

        // Cold tier for older files
        split.locality.storage_tier = StorageTier::Cold;
        assert_eq!(split.locality.storage_tier, StorageTier::Cold);
    }
}

// ============================================================================
// Cross-Engine Tests
// ============================================================================

mod cross_engine_tests {
    use super::*;

    #[test]
    fn test_engine_type_conversions() {
        use proximadb::storage::traits::StorageEngineStrategy;

        // Test EngineType -> StorageEngineStrategy
        assert_eq!(
            StorageEngineStrategy::from(EngineType::Sst),
            StorageEngineStrategy::Sst
        );
        assert_eq!(
            StorageEngineStrategy::from(EngineType::Helix),
            StorageEngineStrategy::Helix
        );
        assert_eq!(
            StorageEngineStrategy::from(EngineType::Viper),
            StorageEngineStrategy::Viper
        );

        // Test StorageEngineStrategy -> EngineType
        assert_eq!(
            EngineType::from(StorageEngineStrategy::Sst),
            EngineType::Sst
        );
        assert_eq!(
            EngineType::from(StorageEngineStrategy::Helix),
            EngineType::Helix
        );
        assert_eq!(
            EngineType::from(StorageEngineStrategy::Viper),
            EngineType::Viper
        );
    }

    #[test]
    fn test_split_type_decoding_complexity() {
        // Block splits have baseline complexity
        let block_split = FileSplit::new_block("/f.sst".to_string(), 0, 0, 1024, 100);
        assert!((block_split.split_cost().decode_complexity - 1.0).abs() < 0.01);

        // Row group splits are efficient (< 1.0)
        let rg_split = FileSplit::new_row_group("/f.parquet".to_string(), 0, 0, 65536, 10000);
        assert!(rg_split.split_cost().decode_complexity < 1.0);

        // Hilbert splits have overhead (> 1.0)
        let hilbert_split =
            FileSplit::new_hilbert_range("/f.helix".to_string(), 0, 1000, 16, 0, 65536);
        assert!(hilbert_split.split_cost().decode_complexity > 1.0);
    }

    #[test]
    fn test_collection_info_vector_size_calculation() {
        // SST with 128D vectors
        let sst_info = CollectionInfo::new("sst".to_string(), 128, EngineType::Sst);
        assert_eq!(sst_info.avg_vector_size_bytes(), 128 * 4); // 512 bytes

        // HELIX with 768D vectors
        let helix_info = CollectionInfo::new("helix".to_string(), 768, EngineType::Helix);
        assert_eq!(helix_info.avg_vector_size_bytes(), 768 * 4); // 3072 bytes

        // VIPER with 1536D vectors (OpenAI ada-002)
        let viper_info = CollectionInfo::new("viper".to_string(), 1536, EngineType::Viper);
        assert_eq!(viper_info.avg_vector_size_bytes(), 1536 * 4); // 6144 bytes
    }

    #[test]
    fn test_pruning_statistics_empty() {
        let stats = PruningStatistics::empty();

        assert_eq!(stats.total_splits, 0);
        assert_eq!(stats.splits_with_stats, 0);
        assert_eq!(stats.splits_with_bloom, 0);
        assert_eq!(stats.splits_with_centroid, 0);
        assert!(!stats.is_pruning_effective());
    }

    #[test]
    fn test_scalar_value_from_json() {
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(42)),
            Some(ScalarValue::Int64(42))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(3.5)),
            Some(ScalarValue::Float64(3.5))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!("hello")),
            Some(ScalarValue::String("hello".to_string()))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(true)),
            Some(ScalarValue::Bool(true))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::Value::Null),
            Some(ScalarValue::Null)
        );
    }
}

// ============================================================================
// Schema Tests
// ============================================================================

mod schema_tests {
    use proximadb::datafusion::engine_adapters::common::{
        estimate_record_size, flat_vector_schema, vector_collection_schema,
    };

    #[test]
    fn test_vector_collection_schema_structure() {
        let schema = vector_collection_schema(128);

        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
        assert_eq!(schema.field(2).name(), "metadata");
        assert_eq!(schema.field(3).name(), "timestamp");
        assert_eq!(schema.field(4).name(), "updated_at");
        assert_eq!(schema.field(5).name(), "expires_at");
        assert_eq!(schema.field(6).name(), "version");
    }

    #[test]
    fn test_flat_vector_schema_structure() {
        let schema = flat_vector_schema(768);

        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
        assert_eq!(schema.field(2).name(), "metadata");

        // Vector field should be FixedSizeBinary with correct size
        let vector_field = schema.field(1);
        if let arrow_schema::DataType::FixedSizeBinary(size) = vector_field.data_type() {
            assert_eq!(*size, 768 * 4); // 768 floats * 4 bytes = 3072
        } else {
            panic!("Expected FixedSizeBinary for vector field");
        }
    }

    #[test]
    fn test_estimate_record_size() {
        // 128D vector: 32 + 512 + 256 + 32 = 832 bytes
        assert_eq!(estimate_record_size(128), 832);

        // 768D vector: 32 + 3072 + 256 + 32 = 3392 bytes
        assert_eq!(estimate_record_size(768), 3392);

        // 1536D vector: 32 + 6144 + 256 + 32 = 6464 bytes
        assert_eq!(estimate_record_size(1536), 6464);
    }
}
