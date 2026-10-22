// Columnar Storage Utilities
// Common utilities for NOVA and VIPER engines

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::proximablocks::columnar_config_types::{ColumnarConfig, ColumnarFileMetadata};
// FilesystemPort seam (TD-DECOMP-75 pattern): root FilesystemFactory implements it.
use proximadb_storage_ports::FilesystemPort;

/// Utility functions for columnar storage operations
pub struct ColumnarUtilities {
    /// Filesystem factory for storage operations
    filesystem: Arc<dyn FilesystemPort>,

    /// Configuration
    #[allow(dead_code)]
    config: ColumnarConfig,

    /// Performance metrics cache
    metrics_cache: Arc<RwLock<HashMap<String, ColumnarUtilsPerformanceMetrics>>>,
}

/// Recommend a Parquet page size given row group sizing and vector characteristics.
///
/// Targets 2–8 pages per row group to balance skip granularity and I/O overhead.
/// Returns a size clamped to 64KB–512KB to avoid tiny or oversized pages.
pub fn recommend_page_size_for_dimension(
    row_group_rows: usize,
    dimension: usize,
    metadata_overhead_bytes: usize,
) -> usize {
    // Estimate bytes per row (FP32 vectors + metadata/ID)
    let row_bytes = dimension
        .saturating_mul(4)
        .saturating_add(metadata_overhead_bytes);
    let total_bytes = row_bytes.saturating_mul(row_group_rows.max(1));

    // Target around 4 pages per row group, but clamp to 2–8 pages.
    let target_pages = 4;
    let candidate = (total_bytes / target_pages.max(1)).max(1);
    let min_for_eight_pages = total_bytes.div_ceil(8); // ensure ≤8 pages
    let max_for_two_pages = total_bytes.div_ceil(2); // ensure ≥2 pages
    let bounded = candidate.clamp(min_for_eight_pages, max_for_two_pages);

    // Global clamps to keep pages practical
    bounded.clamp(64 * 1024, 512 * 1024)
}

impl ColumnarUtilities {
    /// Create new columnar utilities
    pub fn new(
        filesystem: Arc<dyn FilesystemPort>,
        _hardware: Arc<proximadb_hardware_caps::HardwareCapabilities>,
        config: ColumnarConfig,
    ) -> Self {
        Self {
            filesystem,
            config,
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Optimize file layout for columnar access patterns
    pub async fn optimize_file_layout(
        &self,
        file_paths: &[String],
        target_row_group_size: Option<usize>,
    ) -> Result<FileLayoutOptimization> {
        info!("Optimizing file layout for {} files", file_paths.len());

        let mut file_stats = Vec::new();
        let mut total_vectors = 0;
        let mut total_size_bytes = 0;

        // Analyze current file layout
        for file_path in file_paths {
            let stats = self.analyze_file_structure(file_path).await?;
            total_vectors += stats.total_vectors;
            total_size_bytes += stats.total_size_bytes;
            file_stats.push(stats);
        }

        // Calculate optimal layout
        let optimal_row_group_size = target_row_group_size
            .unwrap_or(self.calculate_optimal_row_group_size(total_vectors, total_size_bytes));

        let recommended_file_count =
            self.calculate_optimal_file_count(total_vectors, optimal_row_group_size);

        // Generate optimization recommendations
        let mut recommendations = Vec::new();

        // Check for undersized row groups
        for stats in &file_stats {
            if stats.avg_row_group_size < optimal_row_group_size / 2 {
                recommendations.push(ColumnarUtilsOptimizationRecommendation {
                    file_path: stats.file_path.clone(),
                    issue: "Undersized row groups".to_string(),
                    action: format!(
                        "Compact {} small row groups into {} optimal groups",
                        stats.row_group_count,
                        stats.total_vectors.div_ceil(optimal_row_group_size)
                    ),
                    priority: RecommendationPriority::High,
                });
            }
        }

        // Check for oversized files
        for stats in &file_stats {
            if stats.total_vectors > optimal_row_group_size * 20 {
                recommendations.push(ColumnarUtilsOptimizationRecommendation {
                    file_path: stats.file_path.clone(),
                    issue: "Oversized file".to_string(),
                    action: format!(
                        "Split into {} smaller files",
                        stats.total_vectors.div_ceil(optimal_row_group_size * 10)
                    ),
                    priority: RecommendationPriority::Medium,
                });
            }
        }

        Ok(FileLayoutOptimization {
            current_file_count: file_stats.len(),
            recommended_file_count,
            current_total_vectors: total_vectors,
            current_total_size_bytes: total_size_bytes,
            optimal_row_group_size,
            current_avg_row_group_size: file_stats
                .iter()
                .map(|s| s.avg_row_group_size)
                .sum::<usize>()
                / file_stats.len().max(1),
            recommendations,
            file_statistics: file_stats,
        })
    }

    /// Analyze structure of a single file
    async fn analyze_file_structure(&self, file_path: &str) -> Result<FileStatistics> {
        debug!("Analyzing file structure: {}", file_path);

        // In production, would read actual Parquet metadata
        // For now, provide reasonable estimates
        let estimated_vectors = 10000; // Placeholder
        let estimated_size = 10 * 1024 * 1024; // 10MB placeholder
        let estimated_row_groups = 5; // Placeholder

        Ok(FileStatistics {
            file_path: file_path.to_string(),
            total_vectors: estimated_vectors,
            total_size_bytes: estimated_size,
            row_group_count: estimated_row_groups,
            avg_row_group_size: estimated_vectors / estimated_row_groups,
            compression_ratio: 3.5, // Typical Parquet compression
            has_quantization: true,
        })
    }

    /// Calculate optimal row group size based on hardware and data characteristics
    fn calculate_optimal_row_group_size(
        &self,
        total_vectors: usize,
        total_size_bytes: usize,
    ) -> usize {
        // Fallback assumes a 768-dim fp32 vector when no live samples exist.
        let avg_vector_size = total_size_bytes.checked_div(total_vectors).unwrap_or(3072);

        // Target 128MB row groups for optimal Parquet performance
        const TARGET_ROW_GROUP_SIZE_BYTES: usize = 128 * 1024 * 1024;

        let optimal_size = TARGET_ROW_GROUP_SIZE_BYTES / avg_vector_size.max(1);

        // Clamp to reasonable bounds
        optimal_size.clamp(1000, 100000)
    }

    /// Calculate optimal number of files
    fn calculate_optimal_file_count(&self, total_vectors: usize, row_group_size: usize) -> usize {
        // Target 10 row groups per file for good parallelism
        const TARGET_ROW_GROUPS_PER_FILE: usize = 10;

        let vectors_per_file = row_group_size * TARGET_ROW_GROUPS_PER_FILE;
        total_vectors.div_ceil(vectors_per_file)
    }

    /// Analyze query patterns for optimization
    pub async fn analyze_query_patterns(
        &self,
        query_log: &[QueryLogEntry],
    ) -> Result<QueryPatternAnalysis> {
        info!("Analyzing {} query log entries", query_log.len());

        let mut column_access_count = HashMap::new();
        let mut filter_frequency = HashMap::new();
        let mut row_group_access_patterns = HashMap::new();

        for entry in query_log {
            // Count column accesses
            for column in &entry.accessed_columns {
                *column_access_count.entry(column.clone()).or_insert(0) += 1;
            }

            // Count filter usage
            for filter in &entry.filters_used {
                *filter_frequency.entry(filter.clone()).or_insert(0) += 1;
            }

            // Track row group access patterns
            for rg_id in &entry.row_groups_accessed {
                *row_group_access_patterns.entry(*rg_id).or_insert(0) += 1;
            }
        }

        // Identify hot columns (frequently accessed)
        let mut hot_columns: Vec<_> = column_access_count
            .iter()
            .map(|(col, count)| (col.clone(), *count))
            .collect();
        hot_columns.sort_by_key(|c| std::cmp::Reverse(c.1));
        hot_columns.truncate(10); // Top 10 hot columns

        // Identify frequently used filters
        let mut popular_filters: Vec<_> = filter_frequency
            .iter()
            .map(|(filter, count)| (filter.clone(), *count))
            .collect();
        popular_filters.sort_by_key(|p| std::cmp::Reverse(p.1));
        popular_filters.truncate(10); // Top 10 filters

        // Identify hot row groups
        let mut hot_row_groups: Vec<_> = row_group_access_patterns
            .iter()
            .map(|(rg_id, count)| (*rg_id, *count))
            .collect();
        hot_row_groups.sort_by_key(|h| std::cmp::Reverse(h.1));
        hot_row_groups.truncate(20); // Top 20 hot row groups

        Ok(QueryPatternAnalysis {
            total_queries: query_log.len(),
            hot_columns,
            popular_filters,
            hot_row_groups,
            avg_columns_per_query: column_access_count.len() as f64 / query_log.len() as f64,
            avg_row_groups_per_query: row_group_access_patterns.len() as f64
                / query_log.len() as f64,
        })
    }

    /// Generate compression recommendations
    pub async fn recommend_compression(
        &self,
        file_metadata: &[ColumnarFileMetadata],
    ) -> Result<ColumnarUtilsCompressionRecommendation> {
        info!("Analyzing compression for {} files", file_metadata.len());

        let mut total_uncompressed = 0;
        let mut total_compressed = 0;
        let mut dimension_frequencies = HashMap::new();
        let mut quantization_usage = HashMap::new();

        for metadata in file_metadata {
            // Estimate sizes (in production, would read from file stats)
            let estimated_uncompressed = metadata.num_vectors * (metadata.dimension as u64) * 4; // float32
            let estimated_compressed = (estimated_uncompressed as f32 * 0.3) as u64; // Estimate 30% compression

            total_uncompressed += estimated_uncompressed;
            total_compressed += estimated_compressed;

            *dimension_frequencies.entry(metadata.dimension).or_insert(0) += 1;

            // Track quantization usage
            if metadata.quantization.enable_binary.unwrap_or(false) {
                *quantization_usage.entry("binary".to_string()).or_insert(0) += 1;
            }
            if metadata.quantization.enable_int8.unwrap_or(false) {
                *quantization_usage.entry("int8".to_string()).or_insert(0) += 1;
            }
            if metadata.quantization.enable_pq.unwrap_or(false) {
                *quantization_usage.entry("pq".to_string()).or_insert(0) += 1;
            }
        }

        let overall_ratio = if total_compressed > 0 {
            total_uncompressed as f64 / total_compressed as f64
        } else {
            1.0
        };

        // Generate recommendations based on analysis
        let mut recommendations = Vec::new();

        if overall_ratio < 2.0 {
            recommendations.push("Consider enabling more aggressive quantization".to_string());
        }

        if quantization_usage.get("pq").copied().unwrap_or(0) < file_metadata.len() / 2 {
            recommendations.push("Enable PQ quantization for better compression".to_string());
        }

        // Find most common dimension for optimization
        let most_common_dimension = dimension_frequencies
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(dim, _)| *dim);

        if let Some(dim) = most_common_dimension
            && dim >= 512
        {
            recommendations.push(format!(
                "Consider PQ with more segments for {}-dimensional vectors",
                dim
            ));
        }

        Ok(ColumnarUtilsCompressionRecommendation {
            overall_compression_ratio: overall_ratio,
            total_uncompressed_bytes: total_uncompressed,
            total_compressed_bytes: total_compressed,
            space_saved_bytes: total_uncompressed.saturating_sub(total_compressed),
            quantization_usage,
            recommendations,
            most_common_dimension,
        })
    }

    /// Validate file integrity
    pub async fn validate_file_integrity(
        &self,
        file_paths: &[String],
    ) -> Result<FileIntegrityReport> {
        info!("Validating integrity of {} files", file_paths.len());

        let mut valid_files = Vec::new();
        let mut corrupted_files = Vec::new();
        let mut missing_files = Vec::new();
        let mut total_size_bytes = 0;

        for file_path in file_paths {
            match self.check_single_file_integrity(file_path).await {
                Ok(stats) => {
                    valid_files.push(stats.clone());
                    total_size_bytes += stats.size_bytes;
                }
                Err(e) => {
                    if e.to_string().contains("not found") || e.to_string().contains("No such file")
                    {
                        missing_files.push(file_path.clone());
                    } else {
                        corrupted_files.push(FileCorruption {
                            file_path: file_path.clone(),
                            error: e.to_string(),
                        });
                    }
                }
            }
        }

        let integrity_score = valid_files.len() as f64 / file_paths.len() as f64;

        Ok(FileIntegrityReport {
            total_files_checked: file_paths.len(),
            valid_files,
            corrupted_files,
            missing_files,
            total_size_bytes,
            integrity_score,
        })
    }

    /// Check integrity of a single file
    async fn check_single_file_integrity(&self, file_path: &str) -> Result<ValidFileInfo> {
        debug!("Checking file integrity: {}", file_path);

        // Get filesystem for the file
        let fs = self.filesystem.get_filesystem(file_path)?;

        // Check if file exists and get metadata
        let file_info = fs.metadata(file_path).await?;

        // For Parquet files, we would validate:
        // 1. File can be opened
        // 2. Metadata is readable
        // 3. All row groups are accessible
        // 4. Schema is valid

        Ok(ValidFileInfo {
            file_path: file_path.to_string(),
            size_bytes: file_info.size,
            last_modified: file_info.modified.unwrap_or_else(chrono::Utc::now),
            row_group_count: 5,  // Placeholder
            vector_count: 10000, // Placeholder
        })
    }

    /// Record performance metrics
    pub async fn record_operation_metrics(
        &self,
        operation: &str,
        metrics: ColumnarUtilsOperationMetrics,
    ) -> Result<()> {
        debug!("Recording metrics for operation: {}", operation);

        let perf_metrics = ColumnarUtilsPerformanceMetrics {
            operation: operation.to_string(),
            duration_ms: metrics.duration_ms,
            bytes_processed: metrics.bytes_processed,
            vectors_processed: metrics.vectors_processed,
            cpu_usage_percent: metrics.cpu_usage_percent,
            memory_usage_bytes: metrics.memory_usage_bytes,
            timestamp: chrono::Utc::now(),
        };

        let mut cache = self.metrics_cache.write().await;
        cache.insert(operation.to_string(), perf_metrics);

        // Keep only recent metrics (last 100 operations)
        if cache.len() > 100 {
            let oldest_key = cache.keys().next().cloned();
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        Ok(())
    }

    /// Get performance statistics
    pub async fn get_performance_stats(
        &self,
        operation: Option<&str>,
    ) -> Result<Vec<ColumnarUtilsPerformanceMetrics>> {
        let cache = self.metrics_cache.read().await;

        if let Some(op) = operation {
            if let Some(metrics) = cache.get(op) {
                Ok(vec![metrics.clone()])
            } else {
                Ok(vec![])
            }
        } else {
            Ok(cache.values().cloned().collect())
        }
    }

    /// Clear performance metrics cache
    pub async fn clear_metrics_cache(&self) {
        let mut cache = self.metrics_cache.write().await;
        cache.clear();
        info!("Cleared performance metrics cache_info");
    }
}

/// File layout optimization result
#[derive(Debug)]
pub struct FileLayoutOptimization {
    pub current_file_count: usize,
    pub recommended_file_count: usize,
    pub current_total_vectors: usize,
    pub current_total_size_bytes: usize,
    pub optimal_row_group_size: usize,
    pub current_avg_row_group_size: usize,
    pub recommendations: Vec<ColumnarUtilsOptimizationRecommendation>,
    pub file_statistics: Vec<FileStatistics>,
}

/// File statistics
#[derive(Debug, Clone)]
pub struct FileStatistics {
    pub file_path: String,
    pub total_vectors: usize,
    pub total_size_bytes: usize,
    pub row_group_count: usize,
    pub avg_row_group_size: usize,
    pub compression_ratio: f64,
    pub has_quantization: bool,
}

/// Backwards-compat alias for [`ColumnarUtilsOptimizationRecommendation`].
pub type OptimizationRecommendation = ColumnarUtilsOptimizationRecommendation;

/// Optimization recommendation
#[derive(Debug)]
pub struct ColumnarUtilsOptimizationRecommendation {
    pub file_path: String,
    pub issue: String,
    pub action: String,
    pub priority: RecommendationPriority,
}

#[derive(Debug)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

/// Query log entry for pattern analysis
#[derive(Debug)]
pub struct QueryLogEntry {
    pub query_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub accessed_columns: Vec<String>,
    pub filters_used: Vec<String>,
    pub row_groups_accessed: Vec<usize>,
    pub duration_ms: f64,
}

/// Query pattern analysis result
#[derive(Debug)]
pub struct QueryPatternAnalysis {
    pub total_queries: usize,
    pub hot_columns: Vec<(String, usize)>,
    pub popular_filters: Vec<(String, usize)>,
    pub hot_row_groups: Vec<(usize, usize)>,
    pub avg_columns_per_query: f64,
    pub avg_row_groups_per_query: f64,
}

/// Backwards-compat alias for [`ColumnarUtilsCompressionRecommendation`].
pub type CompressionRecommendation = ColumnarUtilsCompressionRecommendation;

/// Compression recommendation
#[derive(Debug)]
pub struct ColumnarUtilsCompressionRecommendation {
    pub overall_compression_ratio: f64,
    pub total_uncompressed_bytes: u64,
    pub total_compressed_bytes: u64,
    pub space_saved_bytes: u64,
    pub quantization_usage: HashMap<String, usize>,
    pub recommendations: Vec<String>,
    pub most_common_dimension: Option<usize>,
}

/// File integrity report
#[derive(Debug)]
pub struct FileIntegrityReport {
    pub total_files_checked: usize,
    pub valid_files: Vec<ValidFileInfo>,
    pub corrupted_files: Vec<FileCorruption>,
    pub missing_files: Vec<String>,
    pub total_size_bytes: u64,
    pub integrity_score: f64,
}

/// Valid file information
#[derive(Debug, Clone)]
pub struct ValidFileInfo {
    pub file_path: String,
    pub size_bytes: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub row_group_count: usize,
    pub vector_count: usize,
}

/// File corruption information
#[derive(Debug)]
pub struct FileCorruption {
    pub file_path: String,
    pub error: String,
}

/// Backwards-compat alias for [`ColumnarUtilsOperationMetrics`].
pub type OperationMetrics = ColumnarUtilsOperationMetrics;

/// Operation metrics for recording
#[derive(Debug)]
pub struct ColumnarUtilsOperationMetrics {
    pub duration_ms: f64,
    pub bytes_processed: u64,
    pub vectors_processed: usize,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
}

/// Backwards-compat alias for [`ColumnarUtilsPerformanceMetrics`].
pub type PerformanceMetrics = ColumnarUtilsPerformanceMetrics;

/// Performance metrics
#[derive(Debug, Clone)]
pub struct ColumnarUtilsPerformanceMetrics {
    pub operation: String,
    pub duration_ms: f64,
    pub bytes_processed: u64,
    pub vectors_processed: usize,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    // No-op FilesystemPort lives in the crate's shared `testutil` (TD-DECOMP-75
    // pattern): these tests exercise construction, metrics recording, and pure
    // calculations — no real filesystem reads.
    use crate::testutil::noop_filesystem_port as noop_port;

    #[tokio::test]
    async fn test_columnar_utilities_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let filesystem = noop_port();
        let hardware = proximadb_hardware_caps::get_hardware_capabilities();
        let config = ColumnarConfig::default();

        let utilities = ColumnarUtilities::new(filesystem, hardware, config);

        // Test metrics recording
        let metrics = ColumnarUtilsOperationMetrics {
            duration_ms: 100.0,
            bytes_processed: 1024,
            vectors_processed: 10,
            cpu_usage_percent: 50.0,
            memory_usage_bytes: 1024 * 1024,
        };

        utilities
            .record_operation_metrics("test_operation", metrics)
            .await
            .unwrap();

        let stats = utilities
            .get_performance_stats(Some("test_operation"))
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].operation, "test_operation");
    }

    #[test]
    fn test_row_group_size_calculation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let filesystem = noop_port();
        let hardware = proximadb_hardware_caps::get_hardware_capabilities();
        let config = ColumnarConfig::default();

        let utilities = ColumnarUtilities::new(filesystem, hardware, config);

        // Test with typical vector data
        let total_vectors = 100000;
        let total_size_bytes = 100000 * 768 * 4; // 768-dim float32 vectors

        let optimal_size =
            utilities.calculate_optimal_row_group_size(total_vectors, total_size_bytes);

        // Should be reasonable size for 128MB target
        assert!(optimal_size >= 1000);
        assert!(optimal_size <= 100000);
    }
}
