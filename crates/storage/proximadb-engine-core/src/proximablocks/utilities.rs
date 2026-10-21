// Shared Utilities for SST and SWIFT engines
// Common utility functions and helpers

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::proximablocks::block_structures::ProximaDataBlock;
use crate::proximablocks::row_config::RowBasedConfig;
use proximadb_hardware_caps::HardwareCapabilities;
use proximadb_records::ProximaRecord;

/// Row-based utilities collection
pub struct RowBasedUtilities;

impl RowBasedUtilities {
    /// Calculate memory usage for a collection of blocks
    pub fn calculate_memory_usage(blocks: &[ProximaDataBlock]) -> MemoryUsageReport {
        let mut total_records = 0;
        let mut total_vector_bytes = 0;
        let mut total_metadata_bytes = 0;
        let mut total_quantized_bytes = 0;
        let mut total_index_bytes = 0;

        for block in blocks {
            total_records += block.records.len();

            // Calculate vector data size
            for record in &block.records {
                total_vector_bytes += record
                    .embeddings
                    .first()
                    .map_or(0, |embedding| embedding.values.len() * 4);
                if !record.props.is_empty() {
                    total_metadata_bytes += record.props.len() * 32; // Rough estimate per metadata item
                }
            }

            // Calculate quantized data size
            if let Some(ref quantized_vecs) = block.quantized_vectors {
                total_quantized_bytes += quantized_vecs.iter().map(|v| v.len()).sum::<usize>();
            }

            // Calculate index size (rough estimate)
            total_index_bytes += 8 * 2; // u32 block_id takes 8 bytes in index structures
        }

        let total_size =
            total_vector_bytes + total_metadata_bytes + total_quantized_bytes + total_index_bytes;

        MemoryUsageReport {
            total_records,
            total_size_bytes: total_size,
            vector_data_bytes: total_vector_bytes,
            metadata_bytes: total_metadata_bytes,
            quantized_data_bytes: total_quantized_bytes,
            index_data_bytes: total_index_bytes,
            memory_efficiency: if total_vector_bytes > 0 {
                1.0 - (total_quantized_bytes as f32 / total_vector_bytes as f32)
            } else {
                0.0
            },
            compression_ratio: if total_size > 0 {
                (total_vector_bytes + total_metadata_bytes) as f32 / total_size as f32
            } else {
                1.0
            },
        }
    }

    /// Estimate optimal configuration for given workload
    pub fn recommend_configuration(
        workload: &WorkloadCharacteristics,
        hardware: &HardwareCapabilities,
    ) -> RowBasedConfig {
        let mut config = RowBasedConfig::default();

        // Adjust based on workload
        match workload.workload_type {
            WorkloadType::HighThroughput => {
                config.records_per_block = 4000; // Larger blocks for throughput
                config.compression.compression_level = 1; // Fast compression
                config.performance.max_concurrent_operations = 16;
            }
            WorkloadType::LowLatency => {
                config.records_per_block = 1000; // Smaller blocks for faster access
                config.performance.prefetch_enabled = true;
                config.indexing.bloom_filter_per_block = true;
            }
            WorkloadType::MemoryConstrained => {
                config.quantization.enable_progressive = true;
                config.compression.compression_level = 6; // Better compression
                config.performance.cache_size_bytes = 256 * 1024 * 1024; // 256MB
            }
            WorkloadType::AnalyticsHeavy => {
                config.superblock_size_target = 2 * 1024 * 1024 * 1024; // 2GB superblocks
                config.indexing.enable_hierarchical_index = true;
                config.indexing.enable_metadata_index = true;
            }
        }

        // Adjust based on hardware
        if hardware.memory.total_memory / (1024 * 1024 * 1024) > 32 {
            config.performance.cache_size_bytes = 2 * 1024 * 1024 * 1024; // 2GB cache
        }

        if hardware.cpu.logical_cores > 8 {
            config.performance.max_concurrent_operations = hardware.cpu.logical_cores;
        }

        // NVMe detection would require checking actual storage devices
        // For now, use a conservative buffer size
        config.performance.io_buffer_size = 256 * 1024; // 256KB default

        config
    }

    /// Validate record integrity
    pub fn validate_records(records: &[ProximaRecord]) -> ValidationReport {
        let mut report = ValidationReport::default();

        for (idx, record) in records.iter().enumerate() {
            let mut record_issues = Vec::new();

            // Check ID
            if record.oid.is_empty() {
                record_issues.push("Missing or empty ID".to_string());
            }

            // Check vector
            let vector = record
                .embeddings
                .first()
                .map(|embedding| embedding.as_fp32_slice());
            if vector.is_none_or(|values| values.is_empty()) {
                record_issues.push("Empty vector".to_string());
            }

            // Check for NaN or infinite values
            if let Some(vector) = vector {
                for (i, &value) in vector.iter().enumerate() {
                    if value.is_nan() {
                        record_issues.push(format!("NaN value at position {i}"));
                    }
                    if value.is_infinite() {
                        record_issues.push(format!("Infinite value at position {i}"));
                    }
                }
            }

            // Check timestamp
            if record.created_at_ns < 0 {
                record_issues.push("Invalid timestamp".to_string());
            }

            if record_issues.is_empty() {
                report.valid_records += 1;
            } else {
                report.invalid_records += 1;
                report.validation_errors.push(RecordValidationError {
                    record_index: idx,
                    record_id: Some(record.oid.clone()),
                    issues: record_issues,
                });
            }
        }

        report.total_records = records.len();
        report.success_rate = if report.total_records > 0 {
            report.valid_records as f64 / report.total_records as f64
        } else {
            1.0
        };

        report
    }

    /// Optimize vector layout for hardware
    pub fn optimize_vector_layout(
        vectors: &[Vec<f32>],
        hardware: &HardwareCapabilities,
        dimension: usize, // From CollectionConfig
    ) -> OptimizedVectorLayout {
        // Dimension comes from collection config, not from vectors

        // Determine optimal SIMD alignment
        let simd_width = if hardware.has_avx512() {
            16 // 512 bits / 32 bits per float
        } else if hardware.cpu.features.avx2_support {
            8 // 256 bits / 32 bits per float
        } else if hardware.cpu.features.sse42_support {
            4 // 128 bits / 32 bits per float
        } else {
            1 // No SIMD
        };

        // Calculate padding needed for alignment
        let padding_needed = if !dimension.is_multiple_of(simd_width) {
            simd_width - (dimension % simd_width)
        } else {
            0
        };

        let aligned_dimension = dimension + padding_needed;

        // Create optimized layout
        let mut optimized_vectors = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let mut aligned_vector = vector.clone();
            // Add padding with zeros
            aligned_vector.extend(vec![0.0; padding_needed]);
            optimized_vectors.push(aligned_vector);
        }

        OptimizedVectorLayout {
            original_dimension: dimension,
            aligned_dimension,
            simd_width,
            padding_per_vector: padding_needed,
            optimized_vectors,
            memory_overhead_bytes: vectors.len() * padding_needed * 4,
            expected_speedup: Self::estimate_simd_speedup(simd_width),
        }
    }

    /// Estimate SIMD speedup factor
    fn estimate_simd_speedup(simd_width: usize) -> f32 {
        match simd_width {
            16 => 8.0, // AVX-512: ~8x speedup
            8 => 4.0,  // AVX2: ~4x speedup
            4 => 2.0,  // SSE: ~2x speedup
            _ => 1.0,  // No SIMD
        }
    }

    /// Analyze access patterns for optimization
    pub fn analyze_access_patterns(access_log: &[AccessLogEntry]) -> AccessPatternAnalysis {
        let mut temporal_gaps = Vec::new();
        let mut spatial_distances = Vec::new();
        let mut operation_counts = HashMap::new();

        // Sort by timestamp
        let mut sorted_log = access_log.to_vec();
        sorted_log.sort_by_key(|entry| entry.timestamp);

        // Analyze temporal patterns
        for window in sorted_log.windows(2) {
            let gap = window[1].timestamp - window[0].timestamp;
            temporal_gaps.push(gap);
        }

        // Analyze spatial patterns (simplified)
        for window in sorted_log.windows(2) {
            let distance =
                Self::calculate_access_distance(&window[0].record_id, &window[1].record_id);
            spatial_distances.push(distance);
        }

        // Count operations
        for entry in access_log {
            *operation_counts
                .entry(entry.operation_type.clone())
                .or_insert(0) += 1;
        }

        AccessPatternAnalysis {
            total_accesses: access_log.len(),
            temporal_locality: Self::calculate_temporal_locality(&temporal_gaps),
            spatial_locality: Self::calculate_spatial_locality(&spatial_distances),
            operation_distribution: operation_counts,
            recommended_optimizations: Self::recommend_access_optimizations(
                &temporal_gaps,
                &spatial_distances,
            ),
        }
    }

    /// Calculate temporal locality score
    fn calculate_temporal_locality(gaps: &[i64]) -> f64 {
        if gaps.is_empty() {
            return 0.0;
        }

        let avg_gap = gaps.iter().sum::<i64>() as f64 / gaps.len() as f64;
        let variance = gaps
            .iter()
            .map(|&gap| (gap as f64 - avg_gap).powi(2))
            .sum::<f64>()
            / gaps.len() as f64;

        // Lower variance means higher temporal locality
        1.0 / (1.0 + variance.sqrt() / avg_gap)
    }

    /// Calculate spatial locality score
    fn calculate_spatial_locality(distances: &[u64]) -> f64 {
        if distances.is_empty() {
            return 0.0;
        }

        let avg_distance = distances.iter().sum::<u64>() as f64 / distances.len() as f64;

        // Lower average distance means higher spatial locality
        1.0 / (1.0 + avg_distance / 1000.0) // Normalize by typical block size
    }

    /// Calculate access distance between record IDs (simplified)
    fn calculate_access_distance(id1: &str, id2: &str) -> u64 {
        // This is a simplified version - in practice would consider actual layout
        id1.len().abs_diff(id2.len()) as u64
    }

    /// Recommend optimizations based on access patterns
    fn recommend_access_optimizations(
        temporal_gaps: &[i64],
        spatial_distances: &[u64],
    ) -> Vec<BlockUtilsOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        let temporal_locality = Self::calculate_temporal_locality(temporal_gaps);
        let spatial_locality = Self::calculate_spatial_locality(spatial_distances);

        if temporal_locality > 0.7 {
            recommendations.push(BlockUtilsOptimizationRecommendation {
                optimization_type: "enable_prefetching".to_string(),
                description: "High temporal locality detected - enable prefetching".to_string(),
                expected_improvement: 15.0,
                implementation_cost: "Low".to_string(),
            });
        }

        if spatial_locality > 0.8 {
            recommendations.push(BlockUtilsOptimizationRecommendation {
                optimization_type: "sequential_layout".to_string(),
                description: "High spatial locality - optimize for sequential access".to_string(),
                expected_improvement: 25.0,
                implementation_cost: "Medium".to_string(),
            });
        }

        if temporal_locality < 0.3 && spatial_locality < 0.3 {
            recommendations.push(BlockUtilsOptimizationRecommendation {
                optimization_type: "random_access_optimization".to_string(),
                description: "Random access pattern - optimize index structures".to_string(),
                expected_improvement: 20.0,
                implementation_cost: "High".to_string(),
            });
        }

        recommendations
    }
}

// Filename generation now uses unified FilenameCodec from compaction_orchestrator
// For index and bloom files, use format!("{}.{}.idx", base_filename, index_type) directly

/// Path resolver for different storage backends
pub struct PathResolver;

impl PathResolver {
    /// Resolve collection path
    pub fn resolve_collection_path(
        base_path: &Path,
        collection_id: &str,
        engine_name: &str,
    ) -> PathBuf {
        base_path.join(collection_id).join(engine_name)
    }

    /// Resolve file path within collection
    pub fn resolve_file_path(collection_path: &Path, filename: &str) -> PathBuf {
        collection_path.join(filename)
    }

    /// Resolve index path
    pub fn resolve_index_path(collection_path: &Path, index_type: &str) -> PathBuf {
        collection_path.join("indexes").join(index_type)
    }

    /// Resolve backup path
    pub fn resolve_backup_path(collection_path: &Path, timestamp: i64) -> PathBuf {
        collection_path
            .join("backups")
            .join(format!("backup_{timestamp}"))
    }
}

/// Memory estimator for planning operations
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory for vector storage
    pub fn estimate_vector_memory(record_count: usize, dimension: usize) -> usize {
        record_count * dimension * 4 // 4 bytes per f32
    }

    /// Estimate memory for quantized storage
    pub fn estimate_quantized_memory(
        record_count: usize,
        dimension: usize,
        quantization_ratio: f32,
    ) -> usize {
        (Self::estimate_vector_memory(record_count, dimension) as f32 * quantization_ratio) as usize
    }

    /// Estimate memory for index structures
    pub fn estimate_index_memory(record_count: usize, avg_id_length: usize) -> usize {
        // Rough estimate: ID + pointer + overhead
        record_count * (avg_id_length + 8 + 16)
    }

    /// Estimate total memory usage
    pub fn estimate_total_memory(
        record_count: usize,
        dimension: usize,
        avg_id_length: usize,
        quantization_ratio: f32,
    ) -> MemoryEstimate {
        let vector_memory = Self::estimate_vector_memory(record_count, dimension);
        let quantized_memory =
            Self::estimate_quantized_memory(record_count, dimension, quantization_ratio);
        let index_memory = Self::estimate_index_memory(record_count, avg_id_length);
        let metadata_memory = record_count * 256; // Rough estimate for metadata

        MemoryEstimate {
            vector_memory,
            quantized_memory,
            index_memory,
            metadata_memory,
            total_memory: vector_memory + quantized_memory + index_memory + metadata_memory,
            memory_savings: vector_memory - quantized_memory,
        }
    }
}

/// Performance profiler for operations
pub struct BlockUtilsPerformanceProfiler {
    start_time: std::time::Instant,
    checkpoints: Vec<PerformanceCheckpoint>,
}

impl BlockUtilsPerformanceProfiler {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self, name: String) {
        let elapsed = self.start_time.elapsed();
        self.checkpoints.push(PerformanceCheckpoint {
            name,
            timestamp: elapsed,
            memory_usage: Self::get_current_memory_usage(),
        });
    }

    pub fn finish(self) -> BlockUtilsPerformanceProfile {
        let total_time = self.start_time.elapsed();
        let peak_memory = self.checkpoints.iter().map(|cp| cp.memory_usage).max();

        BlockUtilsPerformanceProfile {
            total_time_ms: total_time.as_millis() as u64,
            checkpoints: self.checkpoints,
            peak_memory_bytes: peak_memory.unwrap_or(0),
        }
    }

    fn get_current_memory_usage() -> usize {
        // Simplified - would use actual memory profiling
        0
    }
}

impl Default for BlockUtilsPerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// Data structures for utility functions

#[derive(Debug, Clone)]
pub struct MemoryUsageReport {
    pub total_records: usize,
    pub total_size_bytes: usize,
    pub vector_data_bytes: usize,
    pub metadata_bytes: usize,
    pub quantized_data_bytes: usize,
    pub index_data_bytes: usize,
    pub memory_efficiency: f32,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub workload_type: WorkloadType,
    pub expected_record_count: u64,
    pub average_dimension: usize,
    pub read_write_ratio: f64,
    pub concurrent_operations: usize,
    pub memory_constraints: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum WorkloadType {
    HighThroughput,
    LowLatency,
    MemoryConstrained,
    AnalyticsHeavy,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub total_records: usize,
    pub valid_records: usize,
    pub invalid_records: usize,
    pub success_rate: f64,
    pub validation_errors: Vec<RecordValidationError>,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            total_records: 0,
            valid_records: 0,
            invalid_records: 0,
            success_rate: 1.0,
            validation_errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordValidationError {
    pub record_index: usize,
    pub record_id: Option<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OptimizedVectorLayout {
    pub original_dimension: usize,
    pub aligned_dimension: usize,
    pub simd_width: usize,
    pub padding_per_vector: usize,
    pub optimized_vectors: Vec<Vec<f32>>,
    pub memory_overhead_bytes: usize,
    pub expected_speedup: f32,
}

#[derive(Debug, Clone)]
pub struct AccessLogEntry {
    pub timestamp: i64,
    pub record_id: String,
    pub operation_type: String,
    pub response_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    pub total_accesses: usize,
    pub temporal_locality: f64,
    pub spatial_locality: f64,
    pub operation_distribution: HashMap<String, u64>,
    pub recommended_optimizations: Vec<BlockUtilsOptimizationRecommendation>,
}

/// Backwards-compat alias for [`BlockUtilsOptimizationRecommendation`].
pub type OptimizationRecommendation = BlockUtilsOptimizationRecommendation;

#[derive(Debug, Clone)]
pub struct BlockUtilsOptimizationRecommendation {
    pub optimization_type: String,
    pub description: String,
    pub expected_improvement: f64, // Percentage
    pub implementation_cost: String,
}

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub vector_memory: usize,
    pub quantized_memory: usize,
    pub index_memory: usize,
    pub metadata_memory: usize,
    pub total_memory: usize,
    pub memory_savings: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceCheckpoint {
    pub name: String,
    pub timestamp: std::time::Duration,
    pub memory_usage: usize,
}

/// Backwards-compat alias for [`BlockUtilsPerformanceProfile`].
pub type PerformanceProfile = BlockUtilsPerformanceProfile;

#[derive(Debug, Clone)]
pub struct BlockUtilsPerformanceProfile {
    pub total_time_ms: u64,
    pub checkpoints: Vec<PerformanceCheckpoint>,
    pub peak_memory_bytes: usize,
}

/// Compute a deterministic score from a centroid for clustering.
///
/// Uses the sum of the first 8 dimensions to create a simple but effective
/// ordering that groups similar vectors together in physical storage.
/// This is used by SST, SWIFT, and HELIX for block clustering.
///
/// # Arguments
/// * `centroid` - Vector centroid (typically mean of block vectors)
///
/// # Returns
/// Floating-point score for sorting blocks
///
/// # Example
/// ```rust,ignore
/// let centroid = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let score = centroid_score(&centroid);
/// // score = 36.0 (sum of first 8 elements)
/// ```
pub fn centroid_score(centroid: &[f32]) -> f32 {
    centroid.iter().take(8).copied().sum()
}

/// Cluster blocks by centroid distance to improve pruning and cache locality.
///
/// This function reorders blocks based on their centroid scores while preserving
/// the logical ID-to-block mapping through index updates. Used by SST, SWIFT, and
/// HELIX engines for spatial locality optimization.
///
/// # Type Parameters
/// * `B` - Block type
/// * `I` - Index entry type (must provide centroid via `get_centroid`)
///
/// # Arguments
/// * `blocks` - Vector of data blocks to cluster
/// * `index_entries` - Corresponding index entries with centroid information
/// * `get_centroid` - Function to extract centroid from index entry
///
/// # Returns
/// Tuple of (clustered_blocks, reordered_index_entries)
///
/// # Example
/// ```rust,ignore
/// let (clustered_blocks, clustered_index) = cluster_blocks_by_centroid(
///     data_blocks,
///     index_entries,
///     |entry| &entry.block_centroid
/// );
/// // Physical layout is now optimized for locality
/// // Logical index still provides correct key-based lookups
/// ```
pub fn cluster_blocks_by_centroid<B, I, F>(
    blocks: Vec<B>,
    index_entries: Vec<I>,
    get_centroid: F,
) -> (Vec<B>, Vec<I>)
where
    F: Fn(&I) -> &[f32],
{
    if blocks.len() != index_entries.len() {
        // Mismatched lengths - return unchanged
        return (blocks, index_entries);
    }

    // Compute score for each block and pair with data
    let mut clustered: Vec<(f32, B, I)> = blocks
        .into_iter()
        .zip(index_entries)
        .map(|(block, entry)| {
            let score = centroid_score(get_centroid(&entry));
            (score, block, entry)
        })
        .collect();

    // Sort by centroid score (deterministic ordering)
    clustered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Unzip into separate vectors maintaining clustered order
    let (blocks, index_entries): (Vec<B>, Vec<I>) = clustered
        .into_iter()
        .map(|(_, block, entry)| (block, entry))
        .unzip();

    (blocks, index_entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proximablocks::block_structures::BlockStatistics;
    use proximadb_records::{EmbeddingCell, ProximaRecord};

    fn test_record(oid: &str, values: Vec<f32>, timestamp_ms: i64) -> ProximaRecord {
        let mut record = ProximaRecord {
            oid: oid.to_string(),
            created_at_ns: timestamp_ms * 1_000_000,
            updated_at_ns: timestamp_ms * 1_000_000,
            ..Default::default()
        };
        record.embeddings.push(EmbeddingCell::new_fp32(
            "default",
            "vector",
            values.len() as u32,
            values,
        ));
        record
    }

    #[test]
    fn test_memory_usage_calculation() {
        let records = vec![
            test_record("test1", vec![1.0, 2.0, 3.0], 1000),
            test_record("test2", vec![4.0, 5.0, 6.0], 1001),
        ];

        // Deferred: Update to use proper data structure
        // Temporarily using ProximaDataBlock
        let blocks = vec![ProximaDataBlock {
            encoding_marker: 0x00,
            encoding_metadata: None,
            block_id: 0,
            records: records.clone(), // Include the test records
            quantized_vectors: None,
            quantization_level: None,
            encoded_vectors: None,
            vector_layout: crate::proximablocks::block_structures::VectorEncodingLayout::Auto,
            quantized_section: None,
            metadata: crate::proximablocks::block_structures::ProximaBlockMetadata::default(),
            compression_config:
                crate::proximablocks::block_structures::BlockCompressionConfig::default(),
            compression_algorithm: proximadb_compression::CompressionAlgorithm::None,
            uncompressed_size: 0,
            bloom_filter: None,
            block_bloom_filter: None,
            id_range: (String::new(), String::new()),
            timestamp_range: (0, 0),
            statistics: BlockStatistics {
                read_count: 0,
                write_count: 0,
                search_count: 0,
                cache_hits: 0,
                cache_misses: 0,
                avg_read_time_ms: 0.0,
                avg_search_time_ms: 0.0,
                last_accessed_at: 0,
            },
            metadata_stats: None,
            has_deletes: false,
        }];

        let report = RowBasedUtilities::calculate_memory_usage(&blocks);

        assert_eq!(report.total_records, 2);
        assert!(report.total_size_bytes > 0);
        assert!(report.vector_data_bytes > 0);
    }

    #[test]
    fn test_record_validation() {
        let records = vec![
            test_record("valid", vec![1.0, 2.0, 3.0], 1000),
            test_record("", vec![4.0, 5.0, 6.0], 2000),
            test_record("invalid_vector", vec![f32::NAN, 2.0, f32::INFINITY], 3000),
        ];

        let report = RowBasedUtilities::validate_records(&records);

        assert_eq!(report.total_records, 3);
        assert_eq!(report.valid_records, 1);
        assert_eq!(report.invalid_records, 2);
        assert!(report.success_rate < 0.5);
    }

    // NOTE: `test_filename_generation` (root `FilenameCodec`) stayed behind in the
    // root `formats/proximablocks/mod.rs` test module — FilenameCodec is coupled
    // to the unextracted root filesystem layer (TD-DECOMP-72).

    #[test]
    fn test_memory_estimation() {
        let estimate = MemoryEstimator::estimate_total_memory(
            1000, // records
            768,  // dimension
            32,   // avg ID length
            0.25, // quantization ratio (75% savings)
        );

        assert!(estimate.vector_memory > 0);
        assert!(estimate.quantized_memory < estimate.vector_memory);
        assert!(estimate.memory_savings > 0);
        assert_eq!(
            estimate.total_memory,
            estimate.vector_memory
                + estimate.quantized_memory
                + estimate.index_memory
                + estimate.metadata_memory
        );
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = BlockUtilsPerformanceProfiler::new();

        profiler.checkpoint("start".to_string());
        std::thread::sleep(std::time::Duration::from_millis(1));
        profiler.checkpoint("middle".to_string());
        std::thread::sleep(std::time::Duration::from_millis(1));

        let profile = profiler.finish();

        assert!(profile.total_time_ms >= 2);
        assert_eq!(profile.checkpoints.len(), 2);
        assert_eq!(profile.checkpoints[0].name, "start");
        assert_eq!(profile.checkpoints[1].name, "middle");
    }
}
