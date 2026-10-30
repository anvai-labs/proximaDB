// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # Arrow-Native FileFormat API Integration Tests
//!
//! This integration test verifies the complete Arrow-Native FileFormat API stack:
//!
//! - **WS1**: CentroidTree + BloomConsolidator for pruning
//! - **WS2**: SIMD Decoders for hardware-accelerated decoding
//! - **WS3**: Smart I/O Layer with ParallelReader
//! - **WS4**: DataFusion TableProvider integration
//! - **WS5**: ProximaRecordBridge for Arrow conversions
//! - **WS6**: Engine TableProvider Adapters (SST, HELIX, VIPER)
//!
//! ## Test Categories
//!
//! 1. **Component Unit Tests**: Individual component functionality
//! 2. **Integration Tests**: Cross-component interaction
//! 3. **End-to-End Tests**: Full query execution pipeline

use std::collections::HashMap;

// Test imports - only compile when datafusion-integration feature is enabled
#[cfg(feature = "datafusion-integration")]
use arrow_schema::{DataType, Field, Schema};

use proximadb::proto::proximadb_v1::SqlValue;
use proximadb::proto::proximadb_v1::VectorRecord;
use proximadb::storage::formats::{
    CacheStatus, FileSplit, ScalarPredicate, ScalarValue, SpatialBounds, SplitLocality,
    SplitPlanner, SplitType, StorageTier,
};
use proximadb::storage::schema::{
    BloomConsolidator, CentroidTree, CentroidTreeConfig, IncrementalBloomBuilder,
};

// ============================================================================
// WS1: CentroidTree Tests
// ============================================================================

#[test]
fn test_centroid_tree_construction() {
    // Create centroids for rowgroups
    let centroids: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],    // Rowgroup 0: near origin
        vec![1.0, 0.0, 0.0],    // Rowgroup 1
        vec![0.0, 1.0, 0.0],    // Rowgroup 2
        vec![0.0, 0.0, 1.0],    // Rowgroup 3
        vec![10.0, 10.0, 10.0], // Rowgroup 4: far from origin
    ];

    // Build the tree with max_depth=8
    let tree = CentroidTree::build(&centroids, 8).expect("Should build CentroidTree");

    assert_eq!(tree.dimension(), 3, "Tree dimension should be 3");
    assert_eq!(tree.num_rowgroups(), 5, "Tree should have 5 rowgroups");
}

#[test]
fn test_centroid_tree_empty() {
    let tree = CentroidTree::build(&[], 8).expect("Should build empty tree");
    assert_eq!(tree.dimension(), 0);
    assert_eq!(tree.num_rowgroups(), 0);
}

#[test]
fn test_centroid_tree_pruning() {
    let centroids: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],    // Rowgroup 0: near origin
        vec![1.0, 0.0, 0.0],    // Rowgroup 1
        vec![0.0, 1.0, 0.0],    // Rowgroup 2
        vec![10.0, 10.0, 10.0], // Rowgroup 3: far from origin
    ];

    let tree = CentroidTree::build_with_config(
        &centroids,
        CentroidTreeConfig {
            max_depth: 8,
            min_leaf_size: 1,
            use_quantized: false,
            quantization_bits: 8,
        },
    )
    .expect("Should build tree");

    // Query near origin - should match rowgroups 0-2, prune rowgroup 3
    let query_near_origin = vec![0.5, 0.5, 0.0];
    let result = tree.prune(&query_near_origin, 2.0);

    // Verify we found some matches
    assert!(result.has_matches(), "Should have matches near origin");

    // Verify we pruned some rowgroups
    assert!(
        result.included_indices.len() < 4,
        "Should prune at least one distant rowgroup"
    );
    assert!(
        !result.included_indices.contains(&3),
        "Distant rowgroup should be pruned"
    );
}

#[test]
fn test_centroid_tree_quantized_pruning() {
    let centroids: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![10.0, 10.0, 10.0],
    ];

    let config = CentroidTreeConfig {
        max_depth: 8,
        min_leaf_size: 1,
        use_quantized: true,
        quantization_bits: 8,
    };
    let tree = CentroidTree::build_with_config(&centroids, config).expect("Should build tree");

    let query = vec![0.5, 0.5, 0.5];
    let exact_result = tree.prune(&query, 2.0);
    let quantized_result = tree.prune_quantized(&query, 2.0);

    // Quantized should be conservative (include at least as many as exact)
    assert!(
        quantized_result.included_indices.len() >= exact_result.included_indices.len(),
        "Quantized pruning should be conservative"
    );
}

#[test]
fn test_centroid_tree_serialization() {
    let centroids: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
    ];

    let tree = CentroidTree::build(&centroids, 8).expect("Should build tree");
    let bytes = tree.serialize().expect("Should serialize");
    let restored = CentroidTree::deserialize(&bytes).expect("Should deserialize");

    assert_eq!(restored.dimension(), tree.dimension());
    assert_eq!(restored.num_rowgroups(), tree.num_rowgroups());
}

// ============================================================================
// WS1: BloomConsolidator Tests
// ============================================================================

#[test]
fn test_bloom_consolidator_empty() {
    let consolidator = BloomConsolidator::new(1000, 0.01);
    let bloom = consolidator.build().expect("Should build empty bloom");
    assert!(
        bloom.is_empty(),
        "Empty consolidator should produce empty bloom"
    );
}

#[test]
fn test_incremental_bloom_builder() {
    let mut builder = IncrementalBloomBuilder::new(1000, 0.01);

    // Add some IDs
    builder.add("id_001");
    builder.add("id_002");
    builder.add("id_003");

    assert_eq!(builder.count(), 3);

    let bloom = builder.build().expect("Should build bloom");
    assert_eq!(bloom.num_items(), 3);

    // Test membership
    assert!(bloom.might_contain("id_001"), "Should find inserted ID");
    assert!(bloom.might_contain("id_002"), "Should find inserted ID");
}

#[test]
fn test_incremental_bloom_batch() {
    let mut builder = IncrementalBloomBuilder::new(1000, 0.01);

    let ids = vec!["id1", "id2", "id3", "id4", "id5"];
    builder.add_batch(ids.into_iter());

    assert_eq!(builder.count(), 5);

    let bloom = builder.build().expect("Should build bloom");
    assert!(bloom.might_contain("id3"), "Should find batch-inserted ID");
}

#[test]
fn test_consolidated_bloom_serialization() {
    let mut builder = IncrementalBloomBuilder::new(1000, 0.01);

    for i in 0..50 {
        builder.add(&format!("item:{}", i));
    }

    let bloom = builder.build().expect("Should build bloom");
    let bytes = bloom.serialize().expect("Should serialize");

    let restored = proximadb::storage::schema::ConsolidatedBloom::deserialize(&bytes)
        .expect("Should deserialize");

    assert_eq!(restored.num_items(), bloom.num_items());
    assert!(
        restored.might_contain("item:25"),
        "Should find item after roundtrip"
    );
}

// ============================================================================
// WS3: FileSplit Tests
// ============================================================================

#[test]
fn test_file_split_block_creation() {
    let split = FileSplit::new_block(
        "/data/collection/segment_001.sst".to_string(),
        0,
        0,
        65536,
        1000,
    );

    assert_eq!(split.split_id, "/data/collection/segment_001.sst:block:0");
    assert_eq!(split.offset, 0);
    assert_eq!(split.length, 65536);
    assert_eq!(split.statistics.row_count, Some(1000));

    if let SplitType::Block {
        block_id,
        record_count,
    } = split.split_type
    {
        assert_eq!(block_id, 0);
        assert_eq!(record_count, 1000);
    } else {
        panic!("Expected Block split type");
    }
}

#[test]
fn test_file_split_row_group_creation() {
    let split = FileSplit::new_row_group(
        "/data/collection/segment_001.parquet".to_string(),
        0,
        0,
        1048576,
        10000,
    );

    assert!(split.split_id.contains("rg:0"));

    if let SplitType::RowGroup {
        row_group_index,
        row_count,
    } = split.split_type
    {
        assert_eq!(row_group_index, 0);
        assert_eq!(row_count, 10000);
    } else {
        panic!("Expected RowGroup split type");
    }
}

#[test]
fn test_file_split_hilbert_creation() {
    let split = FileSplit::new_hilbert_range(
        "/data/collection/segment_001.helix".to_string(),
        0,
        1000,
        16, // Hilbert order
        0,
        32768,
    );

    if let SplitType::HilbertRange {
        start_code,
        end_code,
        hilbert_order,
    } = split.split_type
    {
        assert_eq!(start_code, 0);
        assert_eq!(end_code, 1000);
        assert_eq!(hilbert_order, 16);
    } else {
        panic!("Expected HilbertRange split type");
    }
}

#[test]
fn test_file_split_superblock_creation() {
    let block_ids = vec![0, 1, 2, 3];
    let split = FileSplit::new_superblock(
        "/data/collection/segment_001.swift".to_string(),
        0,
        block_ids.clone(),
        0,
        262144,
    );

    if let SplitType::SuperBlock {
        superblock_id,
        block_count,
        block_ids: ids,
    } = split.split_type
    {
        assert_eq!(superblock_id, 0);
        assert_eq!(block_count, 4);
        assert_eq!(ids, block_ids);
    } else {
        panic!("Expected SuperBlock split type");
    }
}

// ============================================================================
// Split Statistics and Pruning Tests
// ============================================================================

#[test]
fn test_split_scalar_predicate_pruning() {
    let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 65536, 1000);

    // Add column statistics
    split.statistics.column_stats.insert(
        "price".to_string(),
        proximadb::storage::formats::ColumnBounds {
            min: Some(serde_json::json!(10.0)),
            max: Some(serde_json::json!(100.0)),
            null_count: 0,
            distinct_count: Some(90),
        },
    );

    // Test pruning with GreaterThan predicate
    // If we're looking for price > 100, and max is 100, we can prune
    assert!(
        split.can_prune_scalar(
            "price",
            &ScalarPredicate::GreaterThan(ScalarValue::Float64(100.0))
        ),
        "Should prune when looking for values greater than max"
    );

    // If we're looking for price > 50, we cannot prune (some values may match)
    assert!(
        !split.can_prune_scalar(
            "price",
            &ScalarPredicate::GreaterThan(ScalarValue::Float64(50.0))
        ),
        "Should NOT prune when values may exist in range"
    );

    // If we're looking for price < 10, and min is 10, we can prune
    assert!(
        split.can_prune_scalar(
            "price",
            &ScalarPredicate::LessThan(ScalarValue::Float64(10.0))
        ),
        "Should prune when looking for values less than min"
    );
}

#[test]
fn test_split_vector_pruning() {
    let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 65536, 1000);

    // Add centroid for vector pruning
    split.statistics.centroid = Some(vec![0.0, 0.0, 0.0]);
    split.statistics.spatial_bounds = Some(SpatialBounds::BoundingBox {
        min_corner: vec![-1.0, -1.0, -1.0],
        max_corner: vec![1.0, 1.0, 1.0],
    });

    // Query close to centroid - cannot prune
    let query_close = vec![0.5, 0.5, 0.5];
    assert!(
        !split.can_prune_vector(&query_close, 10.0),
        "Should NOT prune for queries close to centroid"
    );

    // Query very far from centroid - can prune
    let query_far = vec![100.0, 100.0, 100.0];
    assert!(
        split.can_prune_vector(&query_far, 1.0),
        "Should prune for queries far from centroid with small threshold"
    );
}

// ============================================================================
// Split Planner Tests
// ============================================================================

#[test]
fn test_split_planner_load_balancing() {
    let planner = SplitPlanner::default();

    // Create splits with varying costs
    let splits = vec![
        FileSplit::new_block("/f1.sst".to_string(), 0, 0, 10000, 100),
        FileSplit::new_block("/f1.sst".to_string(), 1, 10000, 20000, 200),
        FileSplit::new_block("/f2.sst".to_string(), 0, 0, 15000, 150),
        FileSplit::new_block("/f2.sst".to_string(), 1, 15000, 5000, 50),
    ];

    // Plan for 2 partitions
    let partitions = planner.plan_splits(splits, 2);

    assert_eq!(partitions.len(), 2, "Should create 2 partitions");

    // Verify all splits are assigned
    let total_splits: usize = partitions.iter().map(|p| p.len()).sum();
    assert_eq!(total_splits, 4, "All splits should be assigned");
}

#[test]
fn test_split_planner_empty_input() {
    let planner = SplitPlanner::default();
    let partitions = planner.plan_splits(vec![], 4);
    assert!(
        partitions.is_empty(),
        "Empty input should produce empty output"
    );
}

// ============================================================================
// Split Cost Estimation Tests
// ============================================================================

#[test]
fn test_split_cost_estimation() {
    // Block split cost
    let block_split = FileSplit::new_block("/f.sst".to_string(), 0, 0, 65536, 1000);
    let block_cost = block_split.split_cost();
    assert_eq!(block_cost.io_bytes, 65536);
    assert_eq!(block_cost.estimated_rows, 1000);
    assert!((block_cost.decode_complexity - 1.0).abs() < 0.01);

    // Row group split cost (columnar is more efficient)
    let rg_split = FileSplit::new_row_group("/f.parquet".to_string(), 0, 0, 65536, 1000);
    let rg_cost = rg_split.split_cost();
    assert!(
        (rg_cost.decode_complexity - 0.8).abs() < 0.01,
        "Columnar should be more efficient"
    );
}

#[test]
fn test_split_locality_cost_multiplier() {
    // Cached split
    let mut cached_split = FileSplit::new_block("/f.sst".to_string(), 0, 0, 1000, 100);
    cached_split.locality = SplitLocality {
        preferred_hosts: vec![],
        storage_tier: StorageTier::Hot,
        cache_status: CacheStatus::Cached,
    };

    // Remote split
    let mut remote_split = FileSplit::new_block("/f.sst".to_string(), 1, 1000, 1000, 100);
    remote_split.locality = SplitLocality {
        preferred_hosts: vec![],
        storage_tier: StorageTier::Cold,
        cache_status: CacheStatus::Remote,
    };

    assert!(
        cached_split.estimated_cost() < remote_split.estimated_cost(),
        "Cached split should have lower cost than remote"
    );
}

// ============================================================================
// WS2: SIMD Decode Tests
// ============================================================================

#[test]
fn test_simd_decode_availability() {
    use proximadb::storage::engines::core::ops::simd_decode::{
        best_decoder, detected_features, has_simd_support,
    };

    // Get the best decoder (will be Scalar, AVX2, or NEON depending on platform)
    let decoder = best_decoder();
    assert!(!decoder.name().is_empty());

    // Check SIMD support
    let has_simd = has_simd_support();
    let features = detected_features();
    assert_eq!(has_simd, features.has_simd());

    println!("Best decoder: {}", decoder.name());
    println!("SIMD support: {}, features: {:?}", has_simd, features);
}

#[test]
fn test_delta_decode_functions() {
    use proximadb::storage::engines::core::ops::simd_decode::delta_decode_f32;

    // Test delta decode
    let base = 1.0f32;
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let base_bits = base.to_bits() as i64;
    let deltas: Vec<i64> = values
        .iter()
        .map(|&v| (v.to_bits() as i64) - base_bits)
        .collect();

    let mut output = vec![0.0f32; 4];
    let count = delta_decode_f32(&deltas, base, &mut output).expect("Should decode");

    assert_eq!(count, 4);
    for (i, (&expected, &actual)) in values.iter().zip(output.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-6,
            "Delta decode mismatch at {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_fused_quantization_decode() {
    use proximadb::storage::engines::core::ops::simd_decode::{
        QuantizationParams, fused_decode_int8_to_f32,
    };

    // Test INT8 -> FP32
    let input = [0u8, 127, 128, 255]; // 0, 127, -128, -1 as signed
    let mut output = vec![0.0f32; 4];
    let params = QuantizationParams::symmetric(0.01);

    let count = fused_decode_int8_to_f32(&input, &mut output, &params).expect("Should decode");
    assert_eq!(count, 4);

    let expected = [0.0f32, 1.27, -1.28, -0.01];
    for (i, (&e, &a)) in expected.iter().zip(output.iter()).enumerate() {
        assert!(
            (e - a).abs() < 1e-4,
            "INT8->FP32 mismatch at {}: expected {}, got {}",
            i,
            e,
            a
        );
    }
}

#[test]
fn test_binary_decode() {
    use proximadb::storage::engines::core::ops::simd_decode::fused_decode_binary_to_f32;

    let input = [0b11110000u8];
    let mut output = vec![0.0f32; 8];

    let count = fused_decode_binary_to_f32(&input, &mut output, true).expect("Should decode");
    assert_eq!(count, 8);

    // First 4 bits are 0 (bipolar: -1), next 4 are 1 (bipolar: +1)
    for i in 0..4 {
        assert_eq!(output[i], -1.0, "Binary decode mismatch at {}", i);
    }
    for i in 4..8 {
        assert_eq!(output[i], 1.0, "Binary decode mismatch at {}", i);
    }
}

// ============================================================================
// WS5: VectorRecord Bridge Tests
// ============================================================================

#[test]
fn test_vector_record_creation() {
    let record = VectorRecord {
        id: "vec_001".to_string(),
        vector: vec![1.0, 2.0, 3.0, 4.0],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert(
                "category".to_string(),
                SqlValue {
                    value: Some(
                        proximadb::proto::proximadb_v1::sql_value::Value::StringValue(
                            "science".to_string(),
                        ),
                    ),
                },
            );
            meta
        },
        timestamp: Some(1234567890),
        ..Default::default()
    };

    assert_eq!(record.id, "vec_001");
    assert_eq!(record.vector.len(), 4);

    // Check that metadata contains the category key
    let category_val = record.metadata.get("category");
    assert!(category_val.is_some(), "Should have category metadata");
    if let Some(val) = category_val {
        match &val.value {
            Some(proximadb::proto::proximadb_v1::sql_value::Value::StringValue(s)) => {
                assert_eq!(s, "science");
            }
            _ => panic!("Expected StringValue"),
        }
    }
}

#[cfg(feature = "datafusion-integration")]
#[test]
fn test_vector_record_to_arrow_roundtrip() {
    use proximadb::storage::schema::{
        DefaultProximaRecordBridge, MetadataMode, ProximaRecordBridge, ProximaSchema,
    };
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

    // Create test records
    let records: Vec<ProximaRecord> = (0..10)
        .map(|i| {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            ProximaRecord {
                oid: format!("vec_{}", i),
                props: {
                    let mut props = HashMap::new();
                    props.insert(
                        "index".to_string(),
                        ProximaTreeNode::Value(ProximaValue::Int64(i as i64)),
                    );
                    props
                },
                created_at_ns: (i as i64).saturating_mul(1_000_000),
                updated_at_ns: (i as i64).saturating_mul(1_000_000),
                record_version: 1,
                embeddings: vec![EmbeddingCell::new_fp32(
                    "test",
                    "dense_vector",
                    vector.len() as u32,
                    vector,
                )],
                ..Default::default()
            }
        })
        .collect();

    // Create bridge with JSON metadata mode
    let bridge = DefaultProximaRecordBridge::new(ProximaSchema::vector_record_schema(4))
        .with_metadata_mode(MetadataMode::JsonString);

    // Convert to RecordBatch
    let batch = bridge
        .records_to_batch(&records)
        .expect("Should convert to RecordBatch");

    assert_eq!(batch.num_rows(), 10);
    assert!(batch.num_columns() >= 3); // id, vector, metadata (at minimum)

    // Convert back to ProximaRecords
    let recovered = bridge
        .batch_to_records(&batch)
        .expect("Should convert back to ProximaRecords");

    assert_eq!(recovered.len(), 10);

    // Verify roundtrip integrity
    for (original, recovered) in records.iter().zip(recovered.iter()) {
        assert_eq!(original.oid, recovered.oid, "ID should match");
        assert_eq!(
            original.embeddings[0].values, recovered.embeddings[0].values,
            "Vector should match"
        );
    }
}

// ============================================================================
// WS3: Smart I/O Layer Tests
// ============================================================================

#[tokio::test]
async fn test_smart_io_parallel_reader_concept() {
    // This test verifies the ParallelReader conceptual interface
    use proximadb::storage::persistence::filesystem::smart_io::{
        ByteRange, IoCostEstimate, ParallelReaderConfig,
    };

    // Test configuration for local storage
    let local_config = ParallelReaderConfig::for_local();
    assert_eq!(local_config.max_concurrent_reads, 4);
    assert!(!local_config.adaptive_concurrency);

    // Test configuration for cloud storage
    let cloud_config = ParallelReaderConfig::for_cloud();
    assert_eq!(cloud_config.max_concurrent_reads, 16);
    assert!(cloud_config.adaptive_concurrency);

    // Test byte range creation
    let range = ByteRange::new(0, 1000);
    assert_eq!(range.start, 0);
    assert_eq!(range.end, 1000);
    assert_eq!(range.len(), 1000);

    // Test cost estimation
    let estimate = IoCostEstimate::new(4, 4096);
    assert_eq!(estimate.io_operations, 4);
    assert_eq!(estimate.bytes_to_read, 4096);
}

// ============================================================================
// DataFusion Integration Tests (Feature-gated)
// ============================================================================

#[cfg(feature = "datafusion-integration")]
mod datafusion_tests {
    use super::*;
    use datafusion::logical_expr::Expr;
    use proximadb::datafusion::{
        CollectionInfo, EngineType, NullProximaTableProvider, ProximaTableProvider,
    };
    use std::sync::Arc;

    fn null_provider_collection_info() -> CollectionInfo {
        CollectionInfo::new("test_collection".to_string(), 768, EngineType::Sst)
            .with_base_path("/data/collections/test".to_string())
    }

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 768),
                true,
            ),
        ]))
    }

    #[tokio::test]
    async fn test_null_table_provider() {
        // NullProximaTableProvider is a no-op implementation for testing.
        let provider =
            NullProximaTableProvider::new(test_schema(), null_provider_collection_info());

        let info = provider.collection_info();
        assert_eq!(info.dimension, 768);
        assert_eq!(info.name, "test_collection");

        let filters: Vec<Expr> = Vec::new();
        let splits = provider
            .get_splits(&filters)
            .await
            .expect("null provider should return an empty split set");
        assert!(splits.is_empty());

        assert_eq!(
            provider
                .pruning_stats()
                .expect("null provider exposes empty pruning stats")
                .total_splits,
            0
        );
        assert!(provider.supports_vector_search());
    }

    #[test]
    fn test_collection_info() {
        let info = null_provider_collection_info();

        assert_eq!(info.name, "test_collection");
        assert_eq!(info.dimension, 768);
        assert!(matches!(info.engine_type, EngineType::Sst));
        assert_eq!(info.base_path, "/data/collections/test");
    }
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

#[test]
fn test_split_based_query_planning() {
    // Simulate a query planning scenario with multiple splits

    // Create splits for a collection
    let mut splits = Vec::new();
    for i in 0..10 {
        let mut split = FileSplit::new_block(
            format!("/data/segment_{}.sst", i / 2),
            i % 2,
            i as u64 * 65536,
            65536,
            1000,
        );

        // Add statistics for pruning
        split.statistics.column_stats.insert(
            "category_id".to_string(),
            proximadb::storage::formats::ColumnBounds {
                min: Some(serde_json::json!(i * 10)),
                max: Some(serde_json::json!((i + 1) * 10)),
                null_count: 0,
                distinct_count: Some(10),
            },
        );

        splits.push(split);
    }

    // Apply predicate pushdown: category_id = 25
    // Only splits where min <= 25 <= max should be kept
    let predicate = ScalarPredicate::Equal(ScalarValue::Int64(25));
    let pruned_splits: Vec<_> = splits
        .into_iter()
        .filter(|s| !s.can_prune_scalar("category_id", &predicate))
        .collect();

    // Split at i=2 has range [20, 30] which contains 25
    assert!(
        !pruned_splits.is_empty(),
        "Should have at least one matching split"
    );
    assert!(pruned_splits.len() < 10, "Should have pruned some splits");
}

#[test]
fn test_centroid_tree_with_splits() {
    // Integration test: CentroidTree with FileSplit

    // Create centroids for rowgroups
    let centroids: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            // Create a centroid (random-ish vector with 128 dimensions)
            (0..128).map(|j| ((i * 10 + j) as f32) / 100.0).collect()
        })
        .collect();

    let tree = CentroidTree::build(&centroids, 8).expect("Should build tree");

    // Query with a vector similar to centroid 2
    let query: Vec<f32> = (0..128).map(|j| ((2 * 10 + j) as f32) / 100.0).collect();
    let result = tree.prune(&query, 0.5);

    // Should find rowgroup 2 as a candidate
    assert!(result.has_matches(), "Should find matches");
    assert!(
        result.included_indices.contains(&2),
        "Should include rowgroup 2 for similar query"
    );
}

#[test]
fn test_bloom_with_centroid_combined_pruning() {
    // Combined pruning: first CentroidTree for vector similarity, then Bloom for ID check

    // Build CentroidTree
    let centroids: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![10.0, 10.0, 10.0],
    ];
    let tree = CentroidTree::build(&centroids, 8).expect("Should build tree");

    // Build Bloom filters for each rowgroup's IDs
    let mut builder = IncrementalBloomBuilder::new(1000, 0.01);
    builder.add("id_in_rg0");
    builder.add("id_in_rg1");
    // Note: id_in_rg2 is NOT added

    let bloom = builder.build().expect("Should build bloom");

    // Phase 1: Vector pruning
    let query = vec![0.5, 0.5, 0.5];
    let vector_candidates = tree.prune(&query, 2.0);

    // Phase 2: ID check on candidates
    let target_id = "id_in_rg0";
    let id_might_exist = bloom.might_contain(target_id);

    // Combined result
    assert!(
        vector_candidates.has_matches(),
        "Should have vector matches"
    );
    assert!(id_might_exist, "ID should exist in bloom filter");

    // Check for non-existent ID
    let missing_id = "id_not_in_any_rg";
    // Due to FPR, might_contain could return true, but for many missing IDs it will return false
    // This is probabilistic, so we just verify the method works
    let _ = bloom.might_contain(missing_id);
}
