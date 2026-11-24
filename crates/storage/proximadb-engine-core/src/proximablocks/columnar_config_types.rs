// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Columnar config + file-metadata types — hoisted from root `formats/columnar/mod.rs`
//! (TD-DECOMP-76) so footer_cache + utilities can live in engine-core.

use proximadb_distance_kernel::DistanceMetric;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ColumnarConfig {
    /// Enable predicate pushdown optimization
    /// When true, filters are pushed to the storage layer to minimize data transfer
    pub enable_predicate_pushdown: bool,

    /// Enable column projection optimization  
    /// When true, only requested columns are read from Parquet files
    pub enable_projection: bool,

    /// Enable row group pruning
    /// When true, use min/max statistics to skip irrelevant row groups
    pub enable_row_group_pruning: bool,

    /// Maximum cache size for row groups (bytes)
    /// Controls memory usage for caching frequently accessed data
    pub max_cache_size_bytes: usize,

    /// Quantization configuration for progressive search
    pub quantization: QuantizationConfig,

    /// Optimization thresholds
    pub optimization_thresholds: OptimizationThresholds,
    /// Filterable metadata columns (have dedicated columns in Parquet)
    pub filterable_metadata_columns: Option<Vec<String>>,
}

// DEPRECATED: Replaced with proto-generated config
// All quantization configs now use the canonical proto version
pub use proximadb_proto::proximadb_v1::QuantizationConfig;

/// Optimization thresholds
#[derive(Debug, Clone)]
pub struct OptimizationThresholds {
    /// Row group pruning threshold (records)
    pub row_group_pruning_threshold: usize,

    /// Column projection threshold (columns)
    pub projection_threshold: usize,

    /// SIMD batch size threshold
    pub simd_threshold: usize,

    /// GPU computation threshold
    pub gpu_threshold: usize,
}

/// File metadata common to both NOVA and NOVA
#[derive(Debug, Clone)]
pub struct ColumnarFileMetadata {
    /// Collection ID
    pub collection_id: String,

    /// Number of vectors
    pub num_vectors: u64,

    /// Vector dimension
    pub dimension: usize,

    /// Distance metric
    pub distance_metric: DistanceMetric,

    /// Quantization configuration
    pub quantization: QuantizationConfig,

    /// Column statistics
    pub column_stats: HashMap<String, ColumnarColumnStatistics>,

    /// File version
    pub version: u32,

    /// Creation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
}

/// Backwards-compat alias for [`ColumnarColumnStatistics`].
pub type ColumnStatistics = ColumnarColumnStatistics;

/// Column statistics for query optimization
#[derive(Debug, Clone)]
pub struct ColumnarColumnStatistics {
    pub null_count: u64,
    pub distinct_count: u64,
    pub min_value: Option<serde_json::Value>,
    pub max_value: Option<serde_json::Value>,
    pub avg_size_bytes: u64,
    pub compression_ratio: f32,
}

impl Default for ColumnarConfig {
    fn default() -> Self {
        Self {
            enable_predicate_pushdown: true,
            enable_projection: true,
            enable_row_group_pruning: true,
            max_cache_size_bytes: 512 * 1024 * 1024, // 512MB
            quantization: QuantizationConfig::default(),
            optimization_thresholds: OptimizationThresholds::default(),
            filterable_metadata_columns: None,
        }
    }
}

// Default implementation removed - using proto-generated Default

impl Default for OptimizationThresholds {
    fn default() -> Self {
        Self {
            row_group_pruning_threshold: 1000,
            projection_threshold: 5,
            simd_threshold: 10000,
            gpu_threshold: 100000,
        }
    }
}
