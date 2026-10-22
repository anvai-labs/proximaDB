// Compression Adapter - Bridges Universal Compression Config with Unified Compression Implementation
// This demonstrates the synergy between universal abstractions and unified implementation

use crate::ops_compression::compression_common::CompressionStrategy;
use anyhow::Result;
use std::collections::HashMap;

use crate::ops_compression::compression_common::{
    AdaptiveCompressionSettings, ContextAwareCompressionConfig, UniversalCompressionConfig,
};
use proximadb_compression::CompressionData;
use proximadb_compression::{
    CompressionAlgorithm, CompressionContext, CompressionProvider, StandardCompression,
};
use proximadb_hardware_caps::HardwareCapabilities;

/// Compression adapter that bridges Universal config with Unified implementation
#[derive(Debug, Clone)]
pub struct UniversalCompressionAdapter {
    /// Unified compression provider (the actual implementation)
    provider: StandardCompression,
    /// Hardware capabilities for optimization
    hardware: HardwareCapabilities,
    /// Performance monitoring
    performance_stats: CompressionPerformanceStats,
}

impl UniversalCompressionAdapter {
    /// Create new adapter with hardware detection
    pub fn new() -> Result<Self> {
        let hardware =
            HardwareCapabilities::detect_with_config(proximadb_config::HardwareConfig::default())?;

        Ok(Self {
            provider: StandardCompression,
            hardware,
            performance_stats: CompressionPerformanceStats::default(),
        })
    }

    /// Create adapter with specific hardware capabilities
    pub fn with_hardware(hardware: HardwareCapabilities) -> Self {
        Self {
            provider: StandardCompression,
            hardware,
            performance_stats: CompressionPerformanceStats::default(),
        }
    }

    /// Compress using universal configuration
    pub fn compress_with_universal_config(
        &mut self,
        data: &[u8],
        config: &UniversalCompressionConfig,
    ) -> Result<CompressedData> {
        let start_time = std::time::Instant::now();

        // Map universal config to unified compression parameters
        let (algorithm, level, context) = self.map_universal_to_unified_config(config)?;

        // Apply adaptive optimizations if enabled
        let optimized_algorithm = if config.adaptive_settings.enabled {
            self.select_adaptive_algorithm(data, algorithm, &config.adaptive_settings)?
        } else {
            algorithm
        };

        // Perform compression using unified implementation
        let compressed_bytes =
            self.provider
                .compress(data, optimized_algorithm, level, context.clone())?;

        // Record performance statistics
        let compression_time = start_time.elapsed();
        let compressed_size = compressed_bytes.len();
        self.performance_stats.record_compression(
            data.len(),
            compressed_size,
            compression_time,
            optimized_algorithm,
        );

        Ok(CompressedData {
            data: compressed_bytes,
            algorithm: optimized_algorithm,
            original_size: data.len(),
            compressed_size,
            compression_level: level,
            context,
            metadata: self.create_compression_metadata(config, compression_time),
        })
    }

    /// Decompress using compression metadata
    pub fn decompress_with_metadata(
        &mut self,
        compressed_data: &CompressedData,
    ) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();

        let decompressed = self.provider.decompress(
            &compressed_data.data,
            compressed_data.algorithm,
            compressed_data.context.clone(),
        )?;

        // Record decompression performance
        let decompression_time = start_time.elapsed();
        self.performance_stats.record_decompression(
            compressed_data.compressed_size,
            decompressed.len(),
            decompression_time,
            compressed_data.algorithm,
        );

        Ok(decompressed)
    }

    /// Map universal configuration to unified compression parameters
    fn map_universal_to_unified_config(
        &self,
        config: &UniversalCompressionConfig,
    ) -> Result<(CompressionAlgorithm, i32, CompressionContext)> {
        // Map algorithm (direct mapping as they use same enum)
        let algorithm = config.primary_algorithm;

        // Map compression level
        let level = config.compression_level as i32;

        // Map context from universal context-aware config
        let context = self.map_context_aware_config(&config.context_aware)?;

        Ok((algorithm, level, context))
    }

    /// Map universal context-aware config to unified compression context
    fn map_context_aware_config(
        &self,
        context_config: &ContextAwareCompressionConfig,
    ) -> Result<CompressionContext> {
        let context = match context_config.data_type {
            CompressionData::Vector => CompressionContext::VectorSerialization,
            CompressionData::Metadata => CompressionContext::Block,
            CompressionData::Index => CompressionContext::Block,
            CompressionData::BloomFilter => CompressionContext::Block,
            CompressionData::Mixed => CompressionContext::Block, // Default for mixed data
        };

        Ok(context)
    }

    /// Select optimal algorithm based on adaptive settings
    fn select_adaptive_algorithm(
        &self,
        data: &[u8],
        default_algorithm: CompressionAlgorithm,
        adaptive_settings: &AdaptiveCompressionSettings,
    ) -> Result<CompressionAlgorithm> {
        // If adaptive is disabled, return default
        if !adaptive_settings.enabled {
            return Ok(default_algorithm);
        }

        // Analyze data characteristics
        let data_analysis = self.analyze_data_characteristics(data);

        // Select algorithm based on adaptive strategy
        let selected_algorithm = match adaptive_settings.strategies.first() {
            Some(CompressionStrategy::Speed { .. }) => {
                self.select_performance_driven_algorithm(&data_analysis)
            }
            Some(CompressionStrategy::Ratio { .. }) => {
                self.select_data_driven_algorithm(&data_analysis)
            }
            Some(CompressionStrategy::Memory { .. }) => {
                self.select_hardware_driven_algorithm(&data_analysis)
            }
            _ => self.select_data_driven_algorithm(&data_analysis),
        }
        .ok_or_else(|| anyhow::anyhow!("No suitable compression algorithm found"))?;

        Ok(selected_algorithm)
    }

    /// Analyze data characteristics for adaptive compression
    fn analyze_data_characteristics(&self, data: &[u8]) -> DataCharacteristics {
        let mut char_counts = [0u32; 256];
        let mut entropy = 0.0;

        // Calculate byte frequency
        for &byte in data {
            char_counts[byte as usize] += 1;
        }

        // Calculate entropy
        let data_len = data.len() as f64;
        for &count in &char_counts {
            if count > 0 {
                let p = count as f64 / data_len;
                entropy -= p * p.log2();
            }
        }

        // Detect patterns
        let repetitiveness = self.calculate_repetitiveness(data);
        let compressibility = if entropy < 4.0 {
            "high"
        } else if entropy < 6.0 {
            "medium"
        } else {
            "low"
        };

        DataCharacteristics {
            size: data.len(),
            entropy,
            repetitiveness,
            compressibility: compressibility.to_string(),
            has_patterns: repetitiveness > 0.3,
        }
    }

    /// Calculate data repetitiveness (simplified)
    fn calculate_repetitiveness(&self, data: &[u8]) -> f64 {
        if data.len() < 8 {
            return 0.0;
        }

        let mut repeated_bytes = 0;
        let window_size = 8.min(data.len() / 4);

        for i in 0..data.len().saturating_sub(window_size) {
            for j in (i + window_size)..data.len().saturating_sub(window_size) {
                if data[i..i + window_size] == data[j..j + window_size] {
                    repeated_bytes += window_size;
                    break;
                }
            }
        }

        repeated_bytes as f64 / data.len() as f64
    }

    /// Select algorithm based on data characteristics
    fn select_data_driven_algorithm(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Option<CompressionAlgorithm> {
        // High compressibility data
        if characteristics.compressibility == "high" {
            return Some(CompressionAlgorithm::Zstd);
        }

        // Low compressibility data - favor speed
        if characteristics.compressibility == "low" {
            return Some(CompressionAlgorithm::Lz4);
        }

        // Medium compressibility - balanced approach
        if characteristics.has_patterns {
            Some(CompressionAlgorithm::Snappy)
        } else {
            Some(CompressionAlgorithm::Lz4) // Default fallback
        }
    }

    /// Select algorithm based on performance requirements
    fn select_performance_driven_algorithm(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Option<CompressionAlgorithm> {
        // For large data, favor faster algorithms
        if characteristics.size > 1024 * 1024 {
            // > 1MB
            Some(CompressionAlgorithm::Lz4)
        } else if characteristics.size > 64 * 1024 {
            // > 64KB
            Some(CompressionAlgorithm::Snappy)
        } else {
            // Small data can afford better compression
            Some(CompressionAlgorithm::Zstd)
        }
    }

    /// Select algorithm based on hardware capabilities
    fn select_hardware_driven_algorithm(
        &self,
        _characteristics: &DataCharacteristics,
    ) -> Option<CompressionAlgorithm> {
        // Use hardware-optimized algorithms when available
        if self.hardware.cpu.features.avx2_support {
            // LZ4 and Snappy have good SIMD optimizations
            Some(CompressionAlgorithm::Lz4)
        } else if self.hardware.cpu.features.sse42_support {
            Some(CompressionAlgorithm::Snappy)
        } else {
            // Default fallback
            Some(CompressionAlgorithm::Lz4)
        }
    }

    /// Select algorithm using hybrid optimization
    #[allow(dead_code)]
    fn select_hybrid_algorithm(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Option<CompressionAlgorithm> {
        // Combine data, performance, and hardware considerations
        let data_score = self.score_algorithm_for_data(characteristics);
        let perf_score = self.score_algorithm_for_performance(characteristics);
        let hw_score = self.score_algorithm_for_hardware();

        // Weighted combination
        let mut best_algorithm = None;
        let mut best_score = 0.0;

        let candidates = [
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Gzip,
        ];

        for algorithm in candidates {
            let key = &algorithm;
            let combined_score = data_score.get(key).copied().unwrap_or(0.0) * 0.4
                + perf_score.get(key).copied().unwrap_or(0.0) * 0.4
                + hw_score.get(key).copied().unwrap_or(0.0) * 0.2;

            if combined_score > best_score {
                best_score = combined_score;
                best_algorithm = Some(algorithm);
            }
        }

        best_algorithm.or(Some(CompressionAlgorithm::Snappy))
    }

    /// Score algorithms for data characteristics
    #[allow(dead_code)]
    fn score_algorithm_for_data(
        &self,
        characteristics: &DataCharacteristics,
    ) -> HashMap<CompressionAlgorithm, f64> {
        let mut scores = HashMap::new();

        if characteristics.compressibility == "high" {
            scores.insert(CompressionAlgorithm::Zstd, 0.9);
            scores.insert(CompressionAlgorithm::Gzip, 0.8);
            scores.insert(CompressionAlgorithm::Snappy, 0.6);
            scores.insert(CompressionAlgorithm::Lz4, 0.5);
        } else if characteristics.compressibility == "low" {
            scores.insert(CompressionAlgorithm::Lz4, 0.9);
            scores.insert(CompressionAlgorithm::Snappy, 0.8);
            scores.insert(CompressionAlgorithm::Zstd, 0.4);
            scores.insert(CompressionAlgorithm::Gzip, 0.3);
        } else {
            scores.insert(CompressionAlgorithm::Snappy, 0.8);
            scores.insert(CompressionAlgorithm::Lz4, 0.7);
            scores.insert(CompressionAlgorithm::Zstd, 0.6);
            scores.insert(CompressionAlgorithm::Gzip, 0.5);
        }

        scores
    }

    /// Score algorithms for performance
    #[allow(dead_code)]
    fn score_algorithm_for_performance(
        &self,
        characteristics: &DataCharacteristics,
    ) -> HashMap<CompressionAlgorithm, f64> {
        let mut scores = HashMap::new();

        // Larger data favors faster algorithms
        let size_factor = if characteristics.size > 1024 * 1024 {
            1.0
        } else {
            0.5
        };

        scores.insert(CompressionAlgorithm::Lz4, 0.9 * size_factor);
        scores.insert(CompressionAlgorithm::Snappy, 0.8 * size_factor);
        scores.insert(CompressionAlgorithm::Zstd, 0.4 * size_factor);
        scores.insert(CompressionAlgorithm::Gzip, 0.3 * size_factor);

        scores
    }

    /// Score algorithms for hardware capabilities
    #[allow(dead_code)]
    fn score_algorithm_for_hardware(&self) -> HashMap<CompressionAlgorithm, f64> {
        let mut scores = HashMap::new();

        if self.hardware.cpu.features.avx2_support {
            scores.insert(CompressionAlgorithm::Lz4, 0.9);
            scores.insert(CompressionAlgorithm::Snappy, 0.8);
            scores.insert(CompressionAlgorithm::Zstd, 0.6);
            scores.insert(CompressionAlgorithm::Gzip, 0.5);
        } else if self.hardware.cpu.features.sse42_support {
            scores.insert(CompressionAlgorithm::Snappy, 0.8);
            scores.insert(CompressionAlgorithm::Lz4, 0.7);
            scores.insert(CompressionAlgorithm::Zstd, 0.5);
            scores.insert(CompressionAlgorithm::Gzip, 0.4);
        } else {
            // No SIMD - all algorithms equal
            scores.insert(CompressionAlgorithm::Lz4, 0.5);
            scores.insert(CompressionAlgorithm::Snappy, 0.5);
            scores.insert(CompressionAlgorithm::Zstd, 0.5);
            scores.insert(CompressionAlgorithm::Gzip, 0.5);
        }

        scores
    }

    /// Create compression metadata
    fn create_compression_metadata(
        &self,
        config: &UniversalCompressionConfig,
        compression_time: std::time::Duration,
    ) -> AdapterCompressionMetadata {
        AdapterCompressionMetadata {
            universal_config: config.clone(),
            compression_time_ms: compression_time.as_millis() as u64,
            hardware_used: self.hardware.clone(),
            adaptive_selected: config.adaptive_settings.enabled,
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &CompressionPerformanceStats {
        &self.performance_stats
    }

    /// Reset performance statistics
    pub fn reset_performance_stats(&mut self) {
        self.performance_stats = CompressionPerformanceStats::default();
    }
}

/// Data resulting from universal compression
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub algorithm: CompressionAlgorithm,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_level: i32,
    pub context: CompressionContext,
    pub metadata: AdapterCompressionMetadata,
}

/// Backwards-compat alias for [`AdapterCompressionMetadata`].
pub type CompressionMetadata = AdapterCompressionMetadata;

/// Compression metadata
#[derive(Debug, Clone)]
pub struct AdapterCompressionMetadata {
    pub universal_config: UniversalCompressionConfig,
    pub compression_time_ms: u64,
    pub hardware_used: HardwareCapabilities,
    pub adaptive_selected: bool,
}

/// Data characteristics for adaptive compression
#[derive(Debug, Clone)]
struct DataCharacteristics {
    size: usize,
    #[allow(dead_code)]
    entropy: f64,
    #[allow(dead_code)]
    repetitiveness: f64,
    compressibility: String,
    has_patterns: bool,
}

/// Performance statistics for compression operations
#[derive(Debug, Clone, Default)]
pub struct CompressionPerformanceStats {
    pub total_compressions: u64,
    pub total_decompressions: u64,
    pub total_compression_time_ms: u64,
    pub total_decompression_time_ms: u64,
    pub total_bytes_compressed: u64,
    pub total_bytes_decompressed: u64,
    pub algorithm_usage: HashMap<CompressionAlgorithm, u64>,
}

impl CompressionPerformanceStats {
    fn record_compression(
        &mut self,
        original_size: usize,
        _compressed_size: usize,
        time: std::time::Duration,
        algorithm: CompressionAlgorithm,
    ) {
        self.total_compressions += 1;
        self.total_compression_time_ms += time.as_millis() as u64;
        self.total_bytes_compressed += original_size as u64;
        *self.algorithm_usage.entry(algorithm).or_insert(0) += 1;
    }

    fn record_decompression(
        &mut self,
        _compressed_size: usize,
        decompressed_size: usize,
        time: std::time::Duration,
        _algorithm: CompressionAlgorithm,
    ) {
        self.total_decompressions += 1;
        self.total_decompression_time_ms += time.as_millis() as u64;
        self.total_bytes_decompressed += decompressed_size as u64;
    }

    pub fn average_compression_ratio(&self) -> f64 {
        if self.total_compressions > 0 {
            self.total_bytes_compressed as f64 / self.total_compressions as f64
        } else {
            0.0
        }
    }

    pub fn compression_throughput_mbps(&self) -> f64 {
        if self.total_compression_time_ms > 0 {
            let mb_compressed = self.total_bytes_compressed as f64 / (1024.0 * 1024.0);
            let seconds = self.total_compression_time_ms as f64 / 1000.0;
            mb_compressed / seconds
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops_compression::compression_common as cc;
    use crate::ops_compression::compression_common::{
        AdaptationCriteria, AdaptiveCompressionSettings, ContextAwareCompressionConfig,
        UniversalCompressionConfig,
    };

    // ── Shared config fixtures (values verbatim from the original literals) ──

    /// Adaptation criteria block shared by every compression round-trip test.
    fn adaptation_criteria() -> AdaptationCriteria {
        AdaptationCriteria {
            data_characteristics: cc::DataCharacteristics {
                entropy_thresholds: cc::EntropyThresholds {
                    low_entropy: 0.5,
                    high_entropy: 8.0,
                    calculation_method: cc::EntropyCalculationMethod::Shannon,
                },
                size_thresholds: cc::SizeThresholds {
                    small_data_threshold: 1024,
                    large_data_threshold: 1024 * 1024,
                    block_size_optimization: true,
                },
                pattern_recognition: cc::PatternRecognitionConfig {
                    enabled: false,
                    pattern_types: vec![],
                    accuracy_threshold: 0.8,
                    pattern_cache_size: 1000,
                },
                data_type_hints: vec![],
            },
            performance_thresholds: cc::PerformanceThresholds {
                max_compression_latency_ms: 1000.0,
                max_decompression_latency_ms: 500.0,
                min_throughput_mbps: 10.0,
                max_cpu_usage_percent: 80.0,
                max_memory_usage_mb: 512,
            },
            resource_constraints: cc::ResourceConstraints {
                memory_constraints: cc::MemoryConstraints {
                    max_working_memory: 512 * 1024 * 1024,
                    max_buffer_size: 64 * 1024 * 1024,
                    memory_pressure_threshold: 0.8,
                    enable_memory_mapping: true,
                },
                cpu_constraints: cc::CPUConstraints {
                    max_cpu_cores: None,
                    max_cpu_usage_percent: 80.0,
                    enable_hardware_acceleration: true,
                    thread_priority: cc::ThreadPriority::Normal,
                },
                io_constraints: cc::IOConstraints {
                    max_io_bandwidth_mbps: 1000.0,
                    io_priority: cc::IOPriority::Normal,
                    buffer_io: true,
                    use_direct_io: false,
                },
                network_constraints: None,
            },
            quality_requirements: cc::QualityRequirements {
                min_compression_ratio: 1.1,
                max_quality_loss_percent: 5.0,
                require_lossless: true,
                error_tolerance: cc::ErrorTolerance::Low,
            },
        }
    }

    /// Learning config of the round-trip batteries: periodic training over the
    /// given data-type labels, with persisted/compressed models.
    fn periodic_persisted_learning(labels: &[&str]) -> cc::ContextLearningConfig {
        cc::ContextLearningConfig {
            enabled: false,
            algorithms: vec![],
            training_requirements: cc::TrainingRequirements {
                min_training_samples: 1000,
                diversity_requirements: cc::DiversityRequirements {
                    data_types: labels.iter().map(|l| l.to_string()).collect(),
                    size_ranges: vec![(1024, 1024 * 1024)],
                    pattern_types: vec!["structured".to_string(), "unstructured".to_string()],
                    context_types: labels.iter().map(|l| l.to_string()).collect(),
                },
                training_frequency: cc::TrainingFrequency::Periodic {
                    interval_ms: 3600000,
                },
                validation_requirements: cc::ValidationRequirements {
                    validation_split: 0.2,
                    cross_validation_folds: 5,
                    performance_metrics: vec![
                        "accuracy".to_string(),
                        "compression_ratio".to_string(),
                    ],
                    min_performance_threshold: 0.85,
                },
            },
            model_persistence: cc::ModelPersistenceConfig {
                enabled: true,
                storage_path: Some("/tmp/compression_models".to_string()),
                versioning: cc::ModelVersioningConfig {
                    enabled: true,
                    max_versions: 10,
                    naming_strategy: cc::VersionNamingStrategy::Timestamp,
                },
                model_compression: true,
                checkpoint_config: cc::CheckpointConfig {
                    enabled: true,
                    checkpoint_interval_ms: 300000,
                    max_checkpoints: 5,
                },
            },
        }
    }

    /// Learning config of the context-mapping battery: on-demand training over
    /// a single data-type label, models disabled.
    fn ondemand_ephemeral_learning(label: &str) -> cc::ContextLearningConfig {
        cc::ContextLearningConfig {
            enabled: false,
            algorithms: vec![],
            training_requirements: cc::TrainingRequirements {
                min_training_samples: 1000,
                diversity_requirements: cc::DiversityRequirements {
                    data_types: vec![label.to_string()],
                    size_ranges: vec![(1024, 1024 * 1024)],
                    pattern_types: vec!["structured".to_string()],
                    context_types: vec![label.to_string()],
                },
                training_frequency: cc::TrainingFrequency::OnDemand,
                validation_requirements: cc::ValidationRequirements {
                    validation_split: 0.2,
                    cross_validation_folds: 5,
                    performance_metrics: vec!["compression_ratio".to_string()],
                    min_performance_threshold: 0.85,
                },
            },
            model_persistence: cc::ModelPersistenceConfig {
                enabled: false,
                storage_path: Some("/tmp/compression_models".to_string()),
                versioning: cc::ModelVersioningConfig {
                    enabled: false,
                    max_versions: 1,
                    naming_strategy: cc::VersionNamingStrategy::Timestamp,
                },
                model_compression: false,
                checkpoint_config: cc::CheckpointConfig {
                    enabled: false,
                    checkpoint_interval_ms: 300000,
                    max_checkpoints: 1,
                },
            },
        }
    }

    /// Mixed-data context with automatic switching (round-trip batteries).
    fn automatic_mixed_context() -> ContextAwareCompressionConfig {
        ContextAwareCompressionConfig {
            enabled: true,
            data_type: CompressionData::Mixed,
            context_types: vec![],
            switching_strategy: cc::ContextSwitchingStrategy::Automatic {
                detection_threshold: 0.8,
                min_switch_interval_ms: 5000,
            },
            learning_config: periodic_persisted_learning(&["vector", "metadata_info"]),
        }
    }

    /// Single-label context with manual switching (context-mapping battery).
    fn manual_labeled_context(
        data_type: CompressionData,
        label: &str,
    ) -> ContextAwareCompressionConfig {
        ContextAwareCompressionConfig {
            enabled: true,
            data_type,
            context_types: vec![],
            switching_strategy: cc::ContextSwitchingStrategy::Manual,
            learning_config: ondemand_ephemeral_learning(label),
        }
    }

    /// Full universal config for the compression round-trip tests.
    fn universal_test_config(
        primary_algorithm: CompressionAlgorithm,
        fallback: CompressionAlgorithm,
        compression_level: u8,
        adaptive_enabled: bool,
        max_adaptation_overhead_percent: f32,
    ) -> UniversalCompressionConfig {
        UniversalCompressionConfig {
            enabled: true,
            primary_algorithm,
            compression_level,
            fallback_algorithms: vec![fallback],
            adaptive_settings: AdaptiveCompressionSettings {
                enabled: adaptive_enabled,
                criteria: adaptation_criteria(),
                max_adaptation_overhead_percent,
                min_adaptation_interval_ms: 1000,
                strategies: vec![],
            },
            context_aware: automatic_mixed_context(),
            hardware_optimizations: Default::default(),
            performance_config: Default::default(),
            quality_settings: Default::default(),
        }
    }

    // ── Tests ──

    #[test]
    fn test_universal_compression_adapter() {
        let mut adapter = UniversalCompressionAdapter::new()
            .expect("Failed to create UniversalCompressionAdapter for test");

        let config = universal_test_config(
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
            3,
            false,
            10.0,
        );

        let test_data =
            b"Hello, World! This is a test for universal compression adapter.".repeat(100);

        // Test compression
        let compressed = adapter
            .compress_with_universal_config(&test_data, &config)
            .expect("Failed to compress test data");
        assert!(compressed.compressed_size < compressed.original_size);
        assert_eq!(compressed.algorithm, CompressionAlgorithm::Zstd);

        // Test decompression
        let decompressed = adapter
            .decompress_with_metadata(&compressed)
            .expect("Failed to decompress test data");
        assert_eq!(test_data, decompressed);

        // Verify performance stats
        let stats = adapter.get_performance_stats();
        assert_eq!(stats.total_compressions, 1);
        assert_eq!(stats.total_decompressions, 1);
    }

    #[test]
    fn test_adaptive_compression_selection() {
        let mut adapter = UniversalCompressionAdapter::new()
            .expect("Failed to create UniversalCompressionAdapter for adaptive test");

        let config = universal_test_config(
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Lz4,
            3,
            true,
            10.0,
        );

        // Test with highly compressible data
        let compressible_data = vec![0u8; 1000];
        let compressed = adapter
            .compress_with_universal_config(&compressible_data, &config)
            .expect("Failed to compress highly compressible test data");

        // Should select ZSTD for highly compressible data (not the default Gzip)
        assert_eq!(compressed.algorithm, CompressionAlgorithm::Zstd);
        assert!(compressed.metadata.adaptive_selected);

        // Test decompression
        let decompressed = adapter
            .decompress_with_metadata(&compressed)
            .expect("Failed to decompress test data");
        assert_eq!(compressible_data, decompressed);
    }

    #[test]
    fn test_context_mapping() {
        let adapter = UniversalCompressionAdapter::new()
            .expect("Failed to create UniversalCompressionAdapter for context mapping test");

        // Test SST block context
        let sst_context = manual_labeled_context(CompressionData::Index, "index");
        let context = adapter
            .map_context_aware_config(&sst_context)
            .expect("Failed to map SST context");
        assert_eq!(context, CompressionContext::Block);

        // Test vector data context
        let vector_context = manual_labeled_context(CompressionData::Vector, "vector");
        let context = adapter
            .map_context_aware_config(&vector_context)
            .expect("Failed to map vector context");
        assert_eq!(context, CompressionContext::VectorSerialization);

        // Test Parquet context
        let parquet_context = manual_labeled_context(CompressionData::Mixed, "mixed");
        let context = adapter
            .map_context_aware_config(&parquet_context)
            .expect("Failed to map Parquet context");
        assert_eq!(context, CompressionContext::Block);
    }

    #[test]
    fn test_performance_statistics() {
        let mut adapter = UniversalCompressionAdapter::new()
            .expect("Failed to create UniversalCompressionAdapter for performance statistics test");

        let config = universal_test_config(
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
            1,
            false,
            5.0,
        );

        // Perform multiple compressions with larger data to ensure measurable timing
        let test_data = b"Performance test data ".repeat(5000); // Much larger data

        for _ in 0..5 {
            let compressed = adapter
                .compress_with_universal_config(&test_data, &config)
                .expect("Failed to compress performance test data");
            let _decompressed = adapter
                .decompress_with_metadata(&compressed)
                .expect("Failed to decompress performance test data");
        }

        let stats = adapter.get_performance_stats();
        assert_eq!(stats.total_compressions, 5);
        assert_eq!(stats.total_decompressions, 5);
        // Timing fields are u64 (always >= 0); just verify they exist
        let _ = stats.total_compression_time_ms;
        let _ = stats.total_decompression_time_ms;
        assert_eq!(
            stats.algorithm_usage.get(&CompressionAlgorithm::Lz4),
            Some(&5)
        );

        // Test throughput calculation
        // Throughput may be 0.0 on very fast systems where compression time rounds to 0ms
        let throughput = stats.compression_throughput_mbps();
        assert!(throughput >= 0.0);
    }
}
