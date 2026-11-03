//! # DataFusion TableProvider Integration Tests
//!
//! TDD tests for the DataFusion TableProvider infrastructure.
//!
//! ## Test Coverage
//!
//! - FileSplit creation and statistics
//! - Scalar predicate pruning
//! - Vector distance pruning
//! - Split cost estimation
//! - ProximaScanExec partitioning
//! - ProximaDataFusionTable schema and scanning

#![cfg(feature = "datafusion-integration")]

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::TableProvider;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;

use proximadb::datafusion::{
    CollectionInfo, EngineType, FileSplit, NullProximaTableProvider, NullSplitReader,
    ProximaDataFusionTable, ProximaDataFusionTableConfig, ProximaScanExec, ProximaTableProvider,
    PruningStatistics,
};
use proximadb::storage::formats::splits::{
    ColumnBounds, ScalarPredicate, ScalarValue, SpatialBounds, SplitStatistics,
};

// ============================================================================
// Test Utilities
// ============================================================================

fn test_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeBinary(512), false),
        Field::new("price", DataType::Float64, true),
        Field::new("category", DataType::Utf8, true),
    ]))
}

fn create_test_split(path: &str, block_id: u32, row_count: u64) -> FileSplit {
    FileSplit::new_block(
        path.to_string(),
        block_id,
        block_id as u64 * 1024,
        1024,
        row_count,
    )
}

fn create_split_with_stats(path: &str, block_id: u32, row_count: u64) -> FileSplit {
    let mut split = create_test_split(path, block_id, row_count);
    split.statistics.column_stats.insert(
        "price".to_string(),
        ColumnBounds {
            min: Some(serde_json::json!(10.0)),
            max: Some(serde_json::json!(100.0)),
            null_count: 0,
            distinct_count: Some(50),
        },
    );
    split.statistics.column_stats.insert(
        "category".to_string(),
        ColumnBounds {
            min: Some(serde_json::json!("electronics")),
            max: Some(serde_json::json!("sports")),
            null_count: 5,
            distinct_count: Some(10),
        },
    );
    split
}

fn create_split_with_centroid(path: &str, block_id: u32, centroid: Vec<f32>) -> FileSplit {
    let mut split = create_test_split(path, block_id, 100);
    split.statistics.centroid = Some(centroid);
    split.statistics.spatial_bounds = Some(SpatialBounds::BoundingBox {
        min_corner: vec![-1.0, -1.0, -1.0],
        max_corner: vec![1.0, 1.0, 1.0],
    });
    split
}

// ============================================================================
// FileSplit Tests
// ============================================================================

#[test]
fn test_file_split_creation() {
    let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

    assert_eq!(split.split_id, "/data/file.sst:block:0");
    assert_eq!(split.file_path, "/data/file.sst");
    assert_eq!(split.offset, 0);
    assert_eq!(split.length, 1024);
    assert_eq!(split.statistics.row_count, Some(100));
}

#[test]
fn test_file_split_row_group() {
    let split = FileSplit::new_row_group("/data/file.parquet".to_string(), 0, 0, 65536, 10000);

    assert!(matches!(
        split.split_type,
        proximadb::storage::formats::splits::SplitType::RowGroup { .. }
    ));
    assert_eq!(split.statistics.row_count, Some(10000));
}

#[test]
fn test_file_split_hilbert_range() {
    let split =
        FileSplit::new_hilbert_range("/data/file.helix".to_string(), 0, 1000, 10, 100, 1000);

    assert!(matches!(
        split.split_type,
        proximadb::storage::formats::splits::SplitType::HilbertRange { .. }
    ));
}

// ============================================================================
// Scalar Pruning Tests
// ============================================================================

#[test]
fn test_split_scalar_pruning_equal() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // Value within range - cannot prune
    assert!(!split.can_prune_scalar("price", &ScalarPredicate::Equal(ScalarValue::Float64(50.0))));

    // Value below range - can prune
    assert!(split.can_prune_scalar("price", &ScalarPredicate::Equal(ScalarValue::Float64(5.0))));

    // Value above range - can prune
    assert!(split.can_prune_scalar(
        "price",
        &ScalarPredicate::Equal(ScalarValue::Float64(150.0))
    ));
}

#[test]
fn test_split_scalar_pruning_less_than() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // price < 10 when min is 10 - can prune
    assert!(split.can_prune_scalar(
        "price",
        &ScalarPredicate::LessThan(ScalarValue::Float64(10.0))
    ));

    // price < 50 when min is 10 - cannot prune (some values match)
    assert!(!split.can_prune_scalar(
        "price",
        &ScalarPredicate::LessThan(ScalarValue::Float64(50.0))
    ));
}

#[test]
fn test_split_scalar_pruning_greater_than() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // price > 100 when max is 100 - can prune
    assert!(split.can_prune_scalar(
        "price",
        &ScalarPredicate::GreaterThan(ScalarValue::Float64(100.0))
    ));

    // price > 50 when max is 100 - cannot prune
    assert!(!split.can_prune_scalar(
        "price",
        &ScalarPredicate::GreaterThan(ScalarValue::Float64(50.0))
    ));
}

#[test]
fn test_split_scalar_pruning_between() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // BETWEEN 50 AND 80 - overlaps with [10, 100]
    assert!(!split.can_prune_scalar(
        "price",
        &ScalarPredicate::Between(ScalarValue::Float64(50.0), ScalarValue::Float64(80.0))
    ));

    // BETWEEN 200 AND 300 - no overlap
    assert!(split.can_prune_scalar(
        "price",
        &ScalarPredicate::Between(ScalarValue::Float64(200.0), ScalarValue::Float64(300.0))
    ));
}

#[test]
fn test_split_scalar_pruning_is_null() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // price has null_count = 0, so IS NULL can be pruned
    assert!(split.can_prune_scalar("price", &ScalarPredicate::IsNull));

    // category has null_count = 5, so IS NULL cannot be pruned
    assert!(!split.can_prune_scalar("category", &ScalarPredicate::IsNull));
}

#[test]
fn test_split_scalar_pruning_unknown_column() {
    let split = create_split_with_stats("/data/file.sst", 0, 100);

    // Unknown column - cannot prune (no statistics)
    assert!(!split.can_prune_scalar("unknown", &ScalarPredicate::Equal(ScalarValue::Int64(50))));
}

// ============================================================================
// Vector Pruning Tests
// ============================================================================

#[test]
fn test_split_vector_pruning_close() {
    let split = create_split_with_centroid("/data/file.sst", 0, vec![0.0, 0.0, 0.0]);

    // Query close to centroid with large distance threshold - cannot prune
    let query = vec![0.5, 0.5, 0.5];
    assert!(!split.can_prune_vector(&query, 10.0));
}

#[test]
fn test_split_vector_pruning_far() {
    let split = create_split_with_centroid("/data/file.sst", 0, vec![0.0, 0.0, 0.0]);

    // Query very far from centroid with small threshold - can prune
    let query = vec![100.0, 100.0, 100.0];
    assert!(split.can_prune_vector(&query, 1.0));
}

#[test]
fn test_split_vector_pruning_no_centroid() {
    let split = create_test_split("/data/file.sst", 0, 100);

    // No centroid - cannot prune
    let query = vec![0.5, 0.5, 0.5];
    assert!(!split.can_prune_vector(&query, 1.0));
}

// ============================================================================
// Split Statistics Tests
// ============================================================================

#[test]
fn test_split_statistics() {
    let mut split = create_test_split("/data/file.sst", 0, 100);
    split.statistics.byte_size = Some(4096);
    split.statistics.bloom_filter = Some(vec![0u8; 32]);
    split.statistics.centroid = Some(vec![0.5, 0.5, 0.5]);

    assert_eq!(split.statistics.row_count, Some(100));
    assert_eq!(split.statistics.byte_size, Some(4096));
    assert!(split.statistics.bloom_filter.is_some());
    assert!(split.statistics.centroid.is_some());
}

#[test]
fn test_split_cost() {
    let split = create_test_split("/data/file.sst", 0, 100);
    let cost = split.split_cost();

    assert_eq!(cost.io_bytes, 1024);
    assert_eq!(cost.estimated_rows, 100);
    assert!((cost.decode_complexity - 1.0).abs() < 0.01);
}

#[test]
fn test_split_cost_row_group() {
    let split = FileSplit::new_row_group("/data/file.parquet".to_string(), 0, 0, 65536, 10000);
    let cost = split.split_cost();

    // Row groups have lower decode complexity
    assert!((cost.decode_complexity - 0.8).abs() < 0.01);
}

// ============================================================================
// Pruning Statistics Tests
// ============================================================================

#[test]
fn test_pruning_statistics_empty() {
    let stats = PruningStatistics::empty();
    assert_eq!(stats.total_splits, 0);
    assert!(!stats.is_pruning_effective());
}

#[test]
fn test_pruning_statistics_from_splits() {
    let mut split1 = create_test_split("/data/file1.sst", 0, 100);
    split1.statistics.bloom_filter = Some(vec![0u8; 32]);

    let mut split2 = create_test_split("/data/file2.sst", 0, 200);
    split2.statistics.column_stats.insert(
        "price".to_string(),
        ColumnBounds {
            min: Some(serde_json::json!(10.0)),
            max: Some(serde_json::json!(100.0)),
            null_count: 0,
            distinct_count: None,
        },
    );

    let stats = PruningStatistics::from_splits(&[split1, split2]);

    assert_eq!(stats.total_splits, 2);
    assert_eq!(stats.splits_with_stats, 2); // Both have row_count
    assert_eq!(stats.splits_with_bloom, 1);
    assert!(stats.columns_with_stats.contains(&"price".to_string()));
}

// ============================================================================
// ProximaScanExec Tests
// ============================================================================

#[test]
fn test_proxima_scan_exec_partitioning() {
    let schema = test_schema();
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Sst));
    let splits = vec![
        create_test_split("/data/file1.sst", 0, 100),
        create_test_split("/data/file1.sst", 1, 100),
        create_test_split("/data/file2.sst", 0, 100),
        create_test_split("/data/file2.sst", 1, 100),
    ];

    let exec = ProximaScanExec::builder()
        .schema(schema)
        .splits(splits)
        .reader(reader)
        .collection_name("test".to_string())
        .target_partitions(2)
        .build()
        .expect("Failed to build ProximaScanExec");

    assert!(exec.partition_count() <= 2);
    assert_eq!(exec.total_split_count(), 4);
    assert_eq!(exec.collection_name(), "test");
}

#[test]
fn test_proxima_scan_exec_projection() {
    let schema = test_schema();
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Viper));

    let exec = ProximaScanExec::builder()
        .schema(schema)
        .splits(vec![])
        .reader(reader)
        .projection(Some(vec![0, 2])) // id and price
        .build()
        .expect("Failed to build ProximaScanExec");

    // Projected schema should have 2 fields
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.projection(), Some(&[0, 2][..]));
}

#[test]
fn test_proxima_scan_exec_limit() {
    let schema = test_schema();
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Nova));

    let exec = ProximaScanExec::builder()
        .schema(schema)
        .splits(vec![])
        .reader(reader)
        .limit(Some(100))
        .build()
        .expect("Failed to build ProximaScanExec");

    assert_eq!(exec.limit(), Some(100));
}

// ============================================================================
// ProximaDataFusionTable Tests
// ============================================================================

#[test]
fn test_datafusion_table_schema() {
    let schema = test_schema();
    let info = CollectionInfo::new("test_vectors".to_string(), 128, EngineType::Sst);
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Sst));

    let table =
        ProximaDataFusionTable::new("test_vectors".to_string(), info, schema.clone(), reader);

    assert_eq!(table.schema().fields().len(), 4);
    assert_eq!(table.engine_type(), EngineType::Sst);
    assert_eq!(table.collection_info().dimension, 128);
}

#[test]
fn test_datafusion_table_with_splits() {
    let schema = test_schema();
    let info = CollectionInfo::new("test".to_string(), 768, EngineType::Viper);
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Viper));

    let splits = vec![
        FileSplit::new_row_group("/data/part1.parquet".to_string(), 0, 0, 65536, 10000),
        FileSplit::new_row_group("/data/part1.parquet".to_string(), 1, 65536, 65536, 10000),
        FileSplit::new_row_group("/data/part2.parquet".to_string(), 0, 0, 65536, 8000),
    ];

    let table =
        ProximaDataFusionTable::new("test".to_string(), info, schema, reader).with_splits(splits);

    let stats = table.pruning_stats().expect("Should have pruning stats");
    assert_eq!(stats.total_splits, 3);
}

#[tokio::test]
async fn test_datafusion_table_scan() {
    let schema = test_schema();
    let info =
        CollectionInfo::new("vectors".to_string(), 512, EngineType::Helix).with_vector_count(10000);
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Helix));

    let splits = vec![
        create_test_split("/data/file1.helix", 0, 1000),
        create_test_split("/data/file1.helix", 1, 1000),
    ];

    let table = ProximaDataFusionTable::new("vectors".to_string(), info, schema.clone(), reader)
        .with_splits(splits);

    // Create session context
    let ctx = SessionContext::new();

    // Get execution plan
    let state = ctx.state();
    let plan = table
        .scan(&state, None, &[], None)
        .await
        .expect("Failed to create scan plan");

    assert_eq!(plan.schema().fields().len(), 4);
}

#[tokio::test]
async fn test_datafusion_table_get_splits() {
    let schema = test_schema();
    let info = CollectionInfo::new("test".to_string(), 256, EngineType::Swift);
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Swift));

    let splits = vec![create_test_split("/data/file.swift", 0, 500)];

    let table =
        ProximaDataFusionTable::new("test".to_string(), info, schema, reader).with_splits(splits);

    let retrieved = table.get_splits(&[]).await.expect("Should get splits");
    assert_eq!(retrieved.len(), 1);
}

// ============================================================================
// Collection Info Tests
// ============================================================================

#[test]
fn test_collection_info_creation() {
    let info = CollectionInfo::new("my_collection".to_string(), 1536, EngineType::Raptor)
        .with_vector_count(100000)
        .with_storage_size(1024 * 1024 * 100) // 100MB
        .with_file_count(10)
        .with_base_path("/data/my_collection".to_string());

    assert_eq!(info.name, "my_collection");
    assert_eq!(info.dimension, 1536);
    assert_eq!(info.vector_count, 100000);
    assert_eq!(info.engine_type, EngineType::Raptor);
    assert_eq!(info.storage_size_bytes, 104857600);
    assert_eq!(info.file_count, 10);
    assert_eq!(info.base_path, "/data/my_collection");
}

#[test]
fn test_collection_info_vector_size() {
    let info = CollectionInfo::new("test".to_string(), 768, EngineType::Viper);
    assert_eq!(info.avg_vector_size_bytes(), 768 * 4); // Float32 = 4 bytes
}

#[test]
fn test_collection_info_estimated_size() {
    let info =
        CollectionInfo::new("test".to_string(), 512, EngineType::Nova).with_vector_count(10000);
    assert_eq!(info.estimated_vector_data_size(), 10000 * 512 * 4);
}

// ============================================================================
// Engine Type Tests
// ============================================================================

#[test]
fn test_engine_type_display() {
    assert_eq!(format!("{}", EngineType::Sst), "SST");
    assert_eq!(format!("{}", EngineType::Helix), "HELIX");
    assert_eq!(format!("{}", EngineType::Swift), "SWIFT");
    assert_eq!(format!("{}", EngineType::Nova), "NOVA");
    assert_eq!(format!("{}", EngineType::Viper), "VIPER");
    assert_eq!(format!("{}", EngineType::Raptor), "RAPTOR");
}

#[test]
fn test_engine_type_conversion() {
    use proximadb::storage::traits::StorageEngineStrategy;

    assert_eq!(
        EngineType::from(StorageEngineStrategy::Sst),
        EngineType::Sst
    );
    assert_eq!(
        EngineType::from(StorageEngineStrategy::Viper),
        EngineType::Viper
    );
    assert_eq!(
        StorageEngineStrategy::from(EngineType::Nova),
        StorageEngineStrategy::Nova
    );
}

// ============================================================================
// Null Provider Tests
// ============================================================================

#[test]
fn test_null_provider() {
    let schema = test_schema();
    let info = CollectionInfo::new("null_test".to_string(), 256, EngineType::Sst);
    let provider = NullProximaTableProvider::new(schema.clone(), info);

    assert_eq!(provider.engine_type(), EngineType::Sst);
    assert_eq!(provider.collection_info().name, "null_test");
    assert!(provider.supports_vector_search());
    assert_eq!(provider.vector_column_name(), Some("vector"));
}

#[tokio::test]
async fn test_null_provider_get_splits() {
    let schema = test_schema();
    let info = CollectionInfo::new("test".to_string(), 128, EngineType::Helix);
    let provider = NullProximaTableProvider::new(schema, info);

    let splits = provider
        .get_splits(&[])
        .await
        .expect("Should return empty splits");
    assert!(splits.is_empty());
}

// ============================================================================
// Integration-Style Tests
// ============================================================================

#[tokio::test]
async fn test_full_pipeline() {
    // Create collection info
    let info = CollectionInfo::new("products".to_string(), 768, EngineType::Viper)
        .with_vector_count(50000)
        .with_storage_size(1024 * 1024 * 500) // 500MB
        .with_file_count(5);

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("embedding", DataType::FixedSizeBinary(768 * 4), false),
        Field::new("category", DataType::Utf8, true),
        Field::new("price", DataType::Float64, true),
    ]));

    // Create reader
    let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Viper));

    // Create splits with statistics
    let splits: Vec<FileSplit> = (0..5)
        .map(|i| {
            let mut split = FileSplit::new_row_group(
                format!("/data/products_{}.parquet", i),
                0,
                0,
                65536 * 10,
                10000,
            );
            split.statistics.column_stats.insert(
                "price".to_string(),
                ColumnBounds {
                    min: Some(serde_json::json!(i as f64 * 100.0)),
                    max: Some(serde_json::json!((i + 1) as f64 * 100.0)),
                    null_count: 0,
                    distinct_count: Some(100),
                },
            );
            split.statistics.bloom_filter = Some(vec![0u8; 64]);
            split.statistics.centroid = Some(vec![0.1 * i as f32; 768]);
            split
        })
        .collect();

    // Create table
    let table = ProximaDataFusionTable::new("products".to_string(), info, schema.clone(), reader)
        .with_splits(splits);

    // Verify pruning statistics
    let stats = table.pruning_stats().unwrap();
    assert_eq!(stats.total_splits, 5);
    assert_eq!(stats.splits_with_stats, 5);
    assert_eq!(stats.splits_with_bloom, 5);
    assert_eq!(stats.splits_with_centroid, 5);
    assert!(stats.is_pruning_effective());

    // Create scan plan
    let ctx = SessionContext::new();
    let plan = table
        .scan(&ctx.state(), None, &[], Some(1000))
        .await
        .expect("Failed to create scan");

    // Verify plan
    assert_eq!(plan.schema().fields().len(), 4);
}
