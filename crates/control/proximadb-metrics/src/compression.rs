// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Compression metrics tracking for the storage engines
//!
//! Tracks compression ratios, performance, and storage savings across different
//! compression strategies and engines.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Compression metrics for a single collection
#[derive(Debug, Clone, Default)]
pub struct CompressionMetrics {
    pub collection_id: String,
    pub engine_type: String, // engine label ("sst", "nova", …)

    // === Overall Compression Statistics ===
    pub total_uncompressed_bytes: u64,
    pub total_compressed_bytes: u64,
    pub overall_compression_ratio: f32, // uncompressed/compressed
    pub space_saved_bytes: i64,
    pub space_saved_percent: f32,

    // === SST Block Compression ===
    pub sst_blocks_written: u64,
    pub sst_avg_block_size_bytes: u64,
    pub sst_avg_compression_ratio: f32,
    pub sst_best_compression_ratio: f32,
    pub sst_worst_compression_ratio: f32,
    pub sst_compression_algorithm: String, // "zstd", "lz4", etc.
    pub sst_compression_level: i32,

    // === Columnar compression (column engines) ===
    pub viper_fp32_column_bytes: u64,
    pub viper_fp32_compressed_bytes: u64,
    pub viper_fp32_compression_ratio: f32,
    pub viper_quantized_column_bytes: u64,
    pub viper_quantized_type: Option<String>, // "int8", "pq8", "pq4"
    pub viper_quantization_reduction: f32,    // % reduction from FP32
    pub viper_normalization_method: Option<String>, // "mean", "trimmed_mean", "median"

    // === Performance Metrics ===
    pub avg_compression_time_ms: f64,
    pub avg_decompression_time_ms: f64,
    pub total_compression_operations: u64,
    pub total_decompression_operations: u64,
    pub compression_throughput_mb_per_sec: f64,
    pub decompression_throughput_mb_per_sec: f64,

    // === Block Size Distribution (SST) ===
    pub block_size_distribution: BlockSizeDistribution,

    // === Compression Effectiveness by Data Type ===
    pub vector_compression_ratio: f32,
    pub metadata_compression_ratio: f32,
    pub index_compression_ratio: f32,

    // === Adaptive Compression Stats ===
    pub adaptive_adjustments: u64, // Number of times compression was adjusted
    pub current_adaptive_level: Option<i32>,
    pub last_adjustment_reason: Option<String>,

    // === Timestamps ===
    pub first_compression_at: Option<i64>,
    pub last_compression_at: Option<i64>,
    pub last_updated: i64,
}

/// Distribution of block sizes for analysis
#[derive(Debug, Clone, Default)]
pub struct BlockSizeDistribution {
    pub min_bytes: u64,
    pub max_bytes: u64,
    pub median_bytes: u64,
    pub p95_bytes: u64,
    pub p99_bytes: u64,
    pub buckets: HashMap<String, u64>, // e.g., "0-4MB": count, "4-8MB": count
}

/// Compression operation result for metrics tracking
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub compression_time: Duration,
    pub algorithm: String,
    pub level: i32,
    pub data_type: CompressionData,
}

/// Decompression operation result
#[derive(Debug, Clone)]
pub struct DecompressionResult {
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub decompression_time: Duration,
}

// `CompressionData` hoisted to `proximadb-compression` (TD-DECOMP-81);
// re-exported so existing `metrics::compression::CompressionData` paths resolve.
pub use proximadb_compression::CompressionData;

/// Global compression stats tracker
pub struct CompressionMetricsTracker {
    metrics: Arc<DashMap<String, CompressionMetrics>>,
    operation_buffer: Arc<DashMap<String, Vec<CompressionResult>>>,
    buffer_size_limit: usize,
}

impl CompressionMetricsTracker {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(DashMap::new()),
            operation_buffer: Arc::new(DashMap::new()),
            buffer_size_limit: 1000, // Buffer up to 1000 operations before aggregating
        }
    }

    /// Record a compression operation
    pub fn record_compression(&self, collection_id: &str, result: CompressionResult) {
        let mut buffer = self
            .operation_buffer
            .entry(collection_id.to_string())
            .or_default();

        buffer.push(result.clone());

        // Aggregate if buffer is full
        if buffer.len() >= self.buffer_size_limit {
            self.aggregate_compression_results(collection_id, &buffer);
            buffer.clear();
        }
    }

    /// Record a decompression operation
    pub fn record_decompression(&self, collection_id: &str, result: DecompressionResult) {
        let mut metrics = self
            .metrics
            .entry(collection_id.to_string())
            .or_insert_with(|| CompressionMetrics {
                collection_id: collection_id.to_string(),
                ..Default::default()
            });

        metrics.total_decompression_operations += 1;

        // Update running average of decompression time
        let new_time_ms = result.decompression_time.as_millis() as f64;
        let n = metrics.total_decompression_operations as f64;
        metrics.avg_decompression_time_ms =
            (metrics.avg_decompression_time_ms * (n - 1.0) + new_time_ms) / n;

        // Calculate decompression throughput
        if result.decompression_time.as_secs_f64() > 0.0 {
            let mb = result.uncompressed_size as f64 / (1024.0 * 1024.0);
            let throughput = mb / result.decompression_time.as_secs_f64();

            // Update running average of throughput
            metrics.decompression_throughput_mb_per_sec =
                (metrics.decompression_throughput_mb_per_sec * (n - 1.0) + throughput) / n;
        }

        metrics.last_updated = chrono::Utc::now().timestamp_millis();
    }

    /// Record SST block compression
    pub fn record_sst_block(
        &self,
        collection_id: &str,
        block_size: u64,
        compressed_size: u64,
        compression_time: Duration,
        algorithm: &str,
        level: i32,
    ) {
        let result = CompressionResult {
            uncompressed_size: block_size,
            compressed_size,
            compression_time,
            algorithm: algorithm.to_string(),
            level,
            data_type: CompressionData::Mixed,
        };

        self.record_compression(collection_id, result.clone());

        // Update SST-specific metrics
        let mut metrics = self
            .metrics
            .entry(collection_id.to_string())
            .or_insert_with(|| CompressionMetrics {
                collection_id: collection_id.to_string(),
                engine_type: "sst".to_string(),
                ..Default::default()
            });

        metrics.sst_blocks_written += 1;
        metrics.sst_compression_algorithm = algorithm.to_string();
        metrics.sst_compression_level = level;

        let ratio = block_size as f32 / compressed_size.max(1) as f32;

        // Update average block size
        let n = metrics.sst_blocks_written as f64;
        metrics.sst_avg_block_size_bytes =
            ((metrics.sst_avg_block_size_bytes as f64 * (n - 1.0) + block_size as f64) / n) as u64;

        // Update compression ratios
        metrics.sst_avg_compression_ratio =
            (metrics.sst_avg_compression_ratio * (n as f32 - 1.0) + ratio) / n as f32;

        if ratio > metrics.sst_best_compression_ratio {
            metrics.sst_best_compression_ratio = ratio;
        }

        if metrics.sst_worst_compression_ratio == 0.0 || ratio < metrics.sst_worst_compression_ratio
        {
            metrics.sst_worst_compression_ratio = ratio;
        }

        // Update block size distribution
        self.update_block_size_distribution(&mut metrics.block_size_distribution, compressed_size);

        metrics.last_compression_at = Some(chrono::Utc::now().timestamp_millis());
        if metrics.first_compression_at.is_none() {
            metrics.first_compression_at = metrics.last_compression_at;
        }
    }

    /// Record VIPER column compression
    pub fn record_viper_compression(
        &self,
        collection_id: &str,
        fp32_size: u64,
        fp32_compressed: u64,
        quantized_size: Option<u64>,
        quantization_type: Option<&str>,
        normalization: Option<&str>,
    ) {
        let mut metrics = self
            .metrics
            .entry(collection_id.to_string())
            .or_insert_with(|| CompressionMetrics {
                collection_id: collection_id.to_string(),
                engine_type: "viper".to_string(),
                ..Default::default()
            });

        metrics.viper_fp32_column_bytes = fp32_size;
        metrics.viper_fp32_compressed_bytes = fp32_compressed;
        metrics.viper_fp32_compression_ratio = fp32_size as f32 / fp32_compressed.max(1) as f32;

        if let Some(q_size) = quantized_size {
            metrics.viper_quantized_column_bytes = q_size;
            metrics.viper_quantization_reduction =
                (1.0 - (q_size as f32 / fp32_size as f32)) * 100.0;
        }

        metrics.viper_quantized_type = quantization_type.map(|s| s.to_string());
        metrics.viper_normalization_method = normalization.map(|s| s.to_string());

        // Update overall metrics
        metrics.total_uncompressed_bytes += fp32_size;
        metrics.total_compressed_bytes += fp32_compressed;

        if let Some(q_size) = quantized_size {
            metrics.total_compressed_bytes += q_size;
        }

        self.update_overall_metrics(&mut metrics);
        metrics.last_updated = chrono::Utc::now().timestamp_millis();
    }

    /// Get metrics for a collection
    pub fn get_metrics(&self, collection_id: &str) -> Option<CompressionMetrics> {
        self.metrics.get(collection_id).map(|m| m.clone())
    }

    /// Get all compression metrics
    pub fn get_all_metrics(&self) -> Vec<CompressionMetrics> {
        self.metrics
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Generate compression recommendations based on metrics
    pub fn get_recommendations(
        &self,
        collection_id: &str,
    ) -> Vec<MetricsCompressionRecommendation> {
        let mut recommendations = Vec::new();

        if let Some(metrics) = self.get_metrics(collection_id) {
            // SST recommendations
            if metrics.engine_type == "sst" {
                if metrics.sst_avg_compression_ratio < 1.2 {
                    recommendations.push(MetricsCompressionRecommendation {
                        recommendation_type: RecommendationType::IncreaseCompressionLevel,
                        description: format!(
                            "Current compression ratio {:.2} is low. Consider increasing ZSTD level from {} to {}",
                            metrics.sst_avg_compression_ratio,
                            metrics.sst_compression_level,
                            (metrics.sst_compression_level + 3).min(9)
                        ),
                        expected_benefit: "10-20% better compression".to_string(),
                        priority: RecommendationPriority::Medium,
                    });
                }

                if metrics.avg_compression_time_ms > 100.0 {
                    recommendations.push(MetricsCompressionRecommendation {
                        recommendation_type: RecommendationType::DecreaseCompressionLevel,
                        description: format!(
                            "Compression taking {:.1}ms on average. Consider reducing ZSTD level to {}",
                            metrics.avg_compression_time_ms,
                            (metrics.sst_compression_level - 2).max(1)
                        ),
                        expected_benefit: "2-3x faster compression".to_string(),
                        priority: RecommendationPriority::High,
                    });
                }
            }

            // VIPER recommendations
            if metrics.engine_type == "viper" {
                if metrics.viper_quantized_type.is_none()
                    && metrics.total_uncompressed_bytes > 100_000_000
                {
                    recommendations.push(MetricsCompressionRecommendation {
                        recommendation_type: RecommendationType::EnableQuantization,
                        description: format!(
                            "Collection has {:.1}GB of FP32 data. Enable PQ8 quantization for faster search",
                            metrics.total_uncompressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                        ),
                        expected_benefit: "24x storage reduction, 3-5x search speedup".to_string(),
                        priority: RecommendationPriority::High,
                    });
                }

                if let Some(ref q_type) = metrics.viper_quantized_type
                    && q_type == "pq8"
                    && metrics.viper_quantization_reduction < 90.0
                {
                    recommendations.push(MetricsCompressionRecommendation {
                            recommendation_type: RecommendationType::OptimizeQuantization,
                            description: "PQ8 achieving only {:.1}% reduction. Consider PQ4 or adjusting normalization".to_string(),
                            expected_benefit: "Additional 2x compression possible".to_string(),
                            priority: RecommendationPriority::Medium,
                        });
                }
            }
        }

        recommendations
    }

    // === Private helper methods ===

    fn aggregate_compression_results(&self, collection_id: &str, results: &[CompressionResult]) {
        if results.is_empty() {
            return;
        }

        let mut metrics = self
            .metrics
            .entry(collection_id.to_string())
            .or_insert_with(|| CompressionMetrics {
                collection_id: collection_id.to_string(),
                ..Default::default()
            });

        for result in results {
            metrics.total_uncompressed_bytes += result.uncompressed_size;
            metrics.total_compressed_bytes += result.compressed_size;
            metrics.total_compression_operations += 1;

            // Update compression time average
            let new_time_ms = result.compression_time.as_millis() as f64;
            let n = metrics.total_compression_operations as f64;
            metrics.avg_compression_time_ms =
                (metrics.avg_compression_time_ms * (n - 1.0) + new_time_ms) / n;

            // Calculate throughput
            if result.compression_time.as_secs_f64() > 0.0 {
                let mb = result.uncompressed_size as f64 / (1024.0 * 1024.0);
                let throughput = mb / result.compression_time.as_secs_f64();
                metrics.compression_throughput_mb_per_sec =
                    (metrics.compression_throughput_mb_per_sec * (n - 1.0) + throughput) / n;
            }

            // Update data type specific ratios
            let ratio = result.uncompressed_size as f32 / result.compressed_size.max(1) as f32;
            match result.data_type {
                CompressionData::Vector => {
                    metrics.vector_compression_ratio = ratio;
                }
                CompressionData::Metadata => {
                    metrics.metadata_compression_ratio = ratio;
                }
                CompressionData::Index => {
                    metrics.index_compression_ratio = ratio;
                }
                _ => {}
            }
        }

        self.update_overall_metrics(&mut metrics);
    }

    fn update_overall_metrics(&self, metrics: &mut CompressionMetrics) {
        if metrics.total_compressed_bytes > 0 {
            metrics.overall_compression_ratio =
                metrics.total_uncompressed_bytes as f32 / metrics.total_compressed_bytes as f32;
        }

        metrics.space_saved_bytes =
            metrics.total_uncompressed_bytes as i64 - metrics.total_compressed_bytes as i64;

        if metrics.total_uncompressed_bytes > 0 {
            metrics.space_saved_percent = (metrics.space_saved_bytes as f32
                / metrics.total_uncompressed_bytes as f32)
                * 100.0;
        }
    }

    fn update_block_size_distribution(&self, dist: &mut BlockSizeDistribution, size: u64) {
        if dist.min_bytes == 0 || size < dist.min_bytes {
            dist.min_bytes = size;
        }
        if size > dist.max_bytes {
            dist.max_bytes = size;
        }

        // Update bucket counts
        let bucket = match size {
            0..=4_194_304 => "0-4MB",
            4_194_305..=8_388_608 => "4-8MB",
            8_388_609..=16_777_216 => "8-16MB",
            _ => "16MB+",
        };

        *dist.buckets.entry(bucket.to_string()).or_insert(0) += 1;
    }
}

/// Backwards-compat alias for [`MetricsCompressionRecommendation`].
pub type CompressionRecommendation = MetricsCompressionRecommendation;

/// Compression recommendation
#[derive(Debug, Clone)]
pub struct MetricsCompressionRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub expected_benefit: String,
    pub priority: RecommendationPriority,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    EnableQuantization,
    OptimizeQuantization,
    IncreaseCompressionLevel,
    DecreaseCompressionLevel,
    EnableAdaptiveCompression,
    ChangeAlgorithm,
    AdjustBlockSize,
}

#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

impl Default for CompressionMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_metrics_tracking() {
        let tracker = CompressionMetricsTracker::new();

        // Record SST block compression
        tracker.record_sst_block(
            "test_collection",
            8_000_000, // 8MB uncompressed
            2_000_000, // 2MB compressed
            Duration::from_millis(50),
            "zstd",
            3,
        );

        let metrics = tracker.get_metrics("test_collection").unwrap();
        assert_eq!(metrics.sst_blocks_written, 1);
        assert_eq!(metrics.sst_avg_compression_ratio, 4.0);
        assert_eq!(metrics.sst_compression_algorithm, "zstd");
        assert_eq!(metrics.sst_compression_level, 3);
    }

    #[test]
    fn test_viper_compression_metrics() {
        let tracker = CompressionMetricsTracker::new();

        tracker.record_viper_compression(
            "test_viper",
            100_000_000,     // 100MB FP32
            25_000_000,      // 25MB compressed FP32
            Some(4_000_000), // 4MB quantized
            Some("pq8"),
            Some("trimmed_mean"),
        );

        let metrics = tracker.get_metrics("test_viper").unwrap();
        assert_eq!(metrics.engine_type, "viper");
        assert_eq!(metrics.viper_fp32_compression_ratio, 4.0);
        assert_eq!(metrics.viper_quantization_reduction, 96.0);
        assert_eq!(metrics.viper_quantized_type, Some("pq8".to_string()));
    }

    #[test]
    fn test_compression_recommendations() {
        let tracker = CompressionMetricsTracker::new();

        // Create metrics with poor compression
        tracker.record_sst_block(
            "poor_compression",
            1_000_000,
            900_000,                    // Only 1.11x compression
            Duration::from_millis(150), // Slow
            "zstd",
            3,
        );

        let recommendations = tracker.get_recommendations("poor_compression");
        assert!(!recommendations.is_empty());

        // Should recommend both increasing compression level and decreasing it due to slow speed
        let has_increase = recommendations.iter().any(|r| {
            matches!(
                r.recommendation_type,
                RecommendationType::IncreaseCompressionLevel
            )
        });
        let has_decrease = recommendations.iter().any(|r| {
            matches!(
                r.recommendation_type,
                RecommendationType::DecreaseCompressionLevel
            )
        });

        assert!(has_increase || has_decrease);
    }
}
