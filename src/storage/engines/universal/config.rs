//! Configuration for Universal Distance Adapter
//!
//! This module provides configuration structures for the universal adapter system.

use std::collections::HashMap;

use super::progressive_refinement::RefinementStage;

/// Cache eviction policy
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used  
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
    /// Time-based TTL eviction
    TTL,
}

/// Main configuration for the universal adapter
#[derive(Debug, Clone)]
pub struct UniversalAdapterConfig {
    /// Enable progressive refinement pipeline
    pub enable_progressive_refinement: bool,

    /// Enable hardware acceleration
    pub enable_hardware_acceleration: bool,

    /// Enable distance table caching
    pub enable_distance_caching: bool,

    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,

    /// SIMD threshold for activation
    pub simd_threshold: usize,

    /// Refinement stages to use
    pub refinement_stages: Vec<RefinementStage>,

    /// Hardware acceleration configuration
    pub hardware_acceleration: HardwareAccelerationConfig,

    /// Progressive refinement configuration
    pub progressive_refinement: UniversalConfigProgressiveRefinementConfig,

    /// Cache configuration
    pub cache_config: UniversalCacheConfig,

    /// Storage engine configurations
    pub storage_engines: Vec<StorageEngineConfig>,
}

/// Backwards-compat alias for [`UniversalConfigProgressiveRefinementConfig`].
pub type ProgressiveRefinementConfig = UniversalConfigProgressiveRefinementConfig;

/// Progressive refinement configuration
#[derive(Debug, Clone)]
pub struct UniversalConfigProgressiveRefinementConfig {
    /// Refinement strategy

    /// Number of candidates to keep at each stage
    pub candidates_per_stage: HashMap<RefinementStage, usize>,

    /// Quality thresholds for each stage
    pub quality_thresholds: HashMap<RefinementStage, f32>,

    /// Enable parallel processing within stages
    pub enable_parallel_processing: bool,

    /// Maximum memory usage in MB
    pub max_memory_usage_mb: usize,

    /// Enable stage skipping optimization
    pub enable_stage_skipping: bool,

    /// Minimum improvement required to continue refinement
    pub min_improvement_threshold: f32,
}

impl Default for UniversalConfigProgressiveRefinementConfig {
    fn default() -> Self {
        let mut candidates_per_stage = HashMap::new();
        candidates_per_stage.insert(RefinementStage::Binary, 1000);
        candidates_per_stage.insert(RefinementStage::INT8, 500);
        candidates_per_stage.insert(RefinementStage::PQ, 200);
        candidates_per_stage.insert(RefinementStage::FP32, 100);

        let mut quality_thresholds = HashMap::new();
        quality_thresholds.insert(RefinementStage::Binary, 0.6);
        quality_thresholds.insert(RefinementStage::INT8, 0.7);
        quality_thresholds.insert(RefinementStage::PQ, 0.8);
        quality_thresholds.insert(RefinementStage::FP32, 0.9);

        Self {
            candidates_per_stage,
            quality_thresholds,
            enable_parallel_processing: true,
            max_memory_usage_mb: 256,
            enable_stage_skipping: true,
            min_improvement_threshold: 0.05,
        }
    }
}

/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct HardwareAccelerationConfig {
    /// Enable SIMD instructions
    pub enable_simd: bool,

    /// Enable AVX instructions
    pub enable_avx: bool,

    /// Enable AVX2 instructions
    pub enable_avx2: bool,

    /// Enable AVX-512 instructions
    pub enable_avx512: bool,

    /// Enable ARM NEON instructions
    pub enable_neon: bool,

    /// Minimum vector size for hardware acceleration
    pub min_vector_size_for_acceleration: usize,

    /// Enable automatic fallback to scalar operations
    pub enable_scalar_fallback: bool,

    /// Batch size for SIMD operations
    pub simd_batch_size: usize,

    /// Enable prefetching optimizations
    pub enable_prefetching: bool,
}

/// Backwards-compat alias for [`UniversalCacheConfig`].
pub type CacheConfig = UniversalCacheConfig;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct UniversalCacheConfig {
    /// Maximum number of entries in cache
    pub max_entries: usize,

    /// TTL for cache entries in seconds
    pub ttl_seconds: u64,

    /// Eviction policy
    pub eviction_policy: CacheEvictionPolicy,

    /// Enable cache compression
    pub compression: bool,

    /// Cache warming on startup
    pub enable_cache_warming: bool,

    /// Maximum memory usage for cache in MB
    pub max_memory_mb: usize,
}

/// Storage engine specific configuration
#[derive(Debug, Clone)]
pub struct StorageEngineConfig {
    /// Engine type
    pub engine_type: super::storage_integration::EngineType,

    /// Engine-specific settings
    pub settings: HashMap<String, String>,

    /// Supported quantization formats
    pub supported_formats: Vec<super::conversion::StorageFormat>,

    /// Default quantization format
    pub default_format: super::conversion::StorageFormat,

    /// Performance preferences
    pub performance_preferences: PerformancePreferences,

    /// Memory limits
    pub memory_limits: MemoryLimits,
}

/// Performance preferences for storage engines
#[derive(Debug, Clone)]
pub struct PerformancePreferences {
    /// Prefer speed over accuracy
    pub prefer_speed: bool,

    /// Target latency in microseconds
    pub target_latency_us: u64,

    /// Target throughput in operations per second
    pub target_throughput_ops: u64,

    /// Memory vs storage tradeoff (0.0 = prefer storage, 1.0 = prefer memory)
    pub memory_storage_tradeoff: f32,

    /// Quality vs speed tradeoff (0.0 = prefer speed, 1.0 = prefer quality)
    pub quality_speed_tradeoff: f32,
}

/// Memory limits for storage engines
#[derive(Debug, Clone)]
pub struct MemoryLimits {
    /// Maximum memory usage per operation in MB
    pub max_memory_per_operation_mb: usize,

    /// Maximum total memory usage in MB
    pub max_total_memory_mb: usize,

    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f32,

    /// Enable memory pressure handling
    pub enable_memory_pressure_handling: bool,
}

impl Default for HardwareAccelerationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_avx: true,
            enable_avx2: true,
            enable_avx512: false, // Conservative default
            enable_neon: true,
            min_vector_size_for_acceleration: 64,
            enable_scalar_fallback: true,
            simd_batch_size: 32,
            enable_prefetching: true,
        }
    }
}

impl Default for UniversalCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 3600, // 1 hour
            eviction_policy: CacheEvictionPolicy::LRU,
            compression: false,
            enable_cache_warming: true,
            max_memory_mb: 256,
        }
    }
}

impl Default for PerformancePreferences {
    fn default() -> Self {
        Self {
            prefer_speed: false,
            target_latency_us: 1000, // 1ms
            target_throughput_ops: 1000,
            memory_storage_tradeoff: 0.5,
            quality_speed_tradeoff: 0.7, // Prefer quality
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_memory_per_operation_mb: 100,
            max_total_memory_mb: 1024,
            memory_pressure_threshold: 0.8,
            enable_memory_pressure_handling: true,
        }
    }
}

impl StorageEngineConfig {
    /// Default INT8 quantization descriptor advertised by every engine.
    fn default_int8_format() -> super::conversion::StorageFormat {
        super::conversion::StorageFormat::QuantizedINT8 {
            scale: 1.0,
            zero_point: 0,
        }
    }

    /// Default PQ descriptor advertised by every engine.
    fn default_pq_format() -> super::conversion::StorageFormat {
        super::conversion::StorageFormat::QuantizedPQ {
            segments: 8,
            bits: 8,
        }
    }

    /// The formats every engine supports (FP32 + INT8 + PQ), plus Binary for
    /// the engines that keep a raw fast path.
    fn supported_formats(with_binary: bool) -> Vec<super::conversion::StorageFormat> {
        let mut formats = vec![
            super::conversion::StorageFormat::FP32,
            Self::default_int8_format(),
            Self::default_pq_format(),
        ];
        if with_binary {
            formats.push(super::conversion::StorageFormat::Binary);
        }
        formats
    }

    /// Create default configuration for PRISM engine
    pub fn prism_default() -> Self {
        let mut settings = HashMap::new();
        settings.insert("cache_size_mb".to_string(), "512".to_string());
        settings.insert("tree_fanout".to_string(), "64".to_string());

        Self {
            engine_type: super::storage_integration::EngineType::PRISM,
            settings,
            supported_formats: Self::supported_formats(false),
            default_format: Self::default_pq_format(),
            performance_preferences: PerformancePreferences {
                prefer_speed: true,
                target_latency_us: 500,
                memory_storage_tradeoff: 0.8, // Prefer memory
                ..Default::default()
            },
            memory_limits: MemoryLimits {
                max_memory_per_operation_mb: 256,
                max_total_memory_mb: 2048,
                ..Default::default()
            },
        }
    }

    /// Create default configuration for NOVA engine
    pub fn nova_default() -> Self {
        let mut settings = HashMap::new();
        settings.insert("column_batch_size".to_string(), "10000".to_string());
        settings.insert("compression_level".to_string(), "6".to_string());

        Self {
            engine_type: super::storage_integration::EngineType::NOVA,
            settings,
            supported_formats: Self::supported_formats(true),
            default_format: Self::default_int8_format(),
            performance_preferences: PerformancePreferences {
                prefer_speed: false,
                target_latency_us: 2000,
                quality_speed_tradeoff: 0.8, // Prefer quality
                ..Default::default()
            },
            memory_limits: MemoryLimits {
                max_memory_per_operation_mb: 512,
                max_total_memory_mb: 4096,
                ..Default::default()
            },
        }
    }

    /// Create default configuration for SWIFT engine
    pub fn swift_default() -> Self {
        let mut settings = HashMap::new();
        settings.insert("block_size".to_string(), "65536".to_string());
        settings.insert("enable_compression".to_string(), "true".to_string());

        Self {
            engine_type: super::storage_integration::EngineType::SWIFT,
            settings,
            supported_formats: Self::supported_formats(true),
            default_format: super::conversion::StorageFormat::FP32,
            performance_preferences: PerformancePreferences {
                prefer_speed: true,
                target_latency_us: 200,
                memory_storage_tradeoff: 0.6,
                ..Default::default()
            },
            memory_limits: MemoryLimits {
                max_memory_per_operation_mb: 128,
                max_total_memory_mb: 1024,
                ..Default::default()
            },
        }
    }

    /// Create default configuration for VIPER engine
    pub fn viper_default() -> Self {
        let mut settings = HashMap::new();
        settings.insert("parquet_row_group_size".to_string(), "100000".to_string());
        settings.insert("compression_codec".to_string(), "ZSTD".to_string());

        Self {
            engine_type: super::storage_integration::EngineType::VIPER,
            settings,
            supported_formats: Self::supported_formats(false),
            default_format: Self::default_int8_format(),
            performance_preferences: PerformancePreferences {
                prefer_speed: false,
                target_latency_us: 5000,
                quality_speed_tradeoff: 0.9, // Strongly prefer quality
                ..Default::default()
            },
            memory_limits: MemoryLimits {
                max_memory_per_operation_mb: 1024,
                max_total_memory_mb: 8192,
                ..Default::default()
            },
        }
    }

    /// Create default configuration for SST engine
    pub fn sst_default() -> Self {
        let mut settings = HashMap::new();
        settings.insert("block_size".to_string(), "32768".to_string());
        settings.insert("bloom_filter_fp_rate".to_string(), "0.01".to_string());

        Self {
            engine_type: super::storage_integration::EngineType::SST,
            settings,
            supported_formats: Self::supported_formats(true),
            default_format: super::conversion::StorageFormat::FP32,
            performance_preferences: PerformancePreferences {
                prefer_speed: true,
                target_latency_us: 1000,
                memory_storage_tradeoff: 0.3, // Prefer storage
                ..Default::default()
            },
            memory_limits: MemoryLimits {
                max_memory_per_operation_mb: 64,
                max_total_memory_mb: 512,
                ..Default::default()
            },
        }
    }
}
