/*
 * Copyright 2024 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Compute Module - Hardware-Accelerated Vector Operations
//!
//! This module provides ProximaDB's high-performance computation engine with automatic
//! hardware detection and optimization. It leverages CPU SIMD instructions and GPU
//! acceleration to achieve maximum throughput for vector similarity operations.
//!
//! ## Role in ProximaDB Architecture
//!
//! The compute layer provides hardware-accelerated operations:
//! ```text
//! Query Request → Compute Layer → Hardware Detection
//!                      ↓                  ↓
//!              Distance Metrics    Optimal Backend Selection
//!                      ↓                  ↓
//!              ┌──────────────────────────────────┐
//!              │   Hardware Acceleration Layer     │
//!              ├──────────────────────────────────┤
//!              │ AVX-512 │ AVX2 │ NEON │ CUDA │   │
//!              └──────────────────────────────────┘
//!                      ↓
//!              Quantization Pipeline
//!              (Binary → INT8 → PQ → FP32)
//! ```
//!
//! ## Key Features
//!
//! ### 1. **Multi-Backend Support**
//! Automatic selection of optimal compute backend:
//! - **CPU SIMD**: AVX-512, AVX2, SSE4.2, NEON (ARM)
//! - **GPU**: CUDA, ROCm, OpenCL, Metal (macOS)
//! - **Math Libraries**: Intel MKL, OpenBLAS, BLIS
//!
//! ### 2. **13 Distance Metrics**
//! Comprehensive similarity measurement support:
//! - Euclidean (L2), Manhattan (L1), Cosine
//! - Dot Product, Hamming, Jaccard
//! - Chebyshev, Canberra, Minkowski
//! - Wasserstein, Jensen-Shannon, Kullback-Leibler
//! - Haversine (geographic)
//!
//! ### 3. **Advanced Quantization**
//! Multi-level quantization for memory efficiency:
//! - **Binary**: 1-bit for initial filtering (32x compression)
//! - **INT8**: 8-bit integers (4x compression, 10x speedup)
//! - **PQ4/PQ8**: Product quantization (16x compression)
//! - **Adaptive**: Automatic selection based on data
//!
//! ### 4. **Hardware Detection**
//! Runtime capability detection and optimization:
//! - CPU feature detection (CPUID)
//! - GPU enumeration and selection
//! - NUMA topology awareness
//! - Cache size optimization
//!
//! ## Performance Characteristics
//!
//! - **SIMD Speedup**: 4-16x over scalar operations
//! - **GPU Throughput**: 100M+ comparisons/sec
//! - **Quantization Speed**: 10x faster with INT8
//! - **Memory Bandwidth**: Optimized for L1/L2 cache
//! - **Parallel Efficiency**: Near-linear scaling to 32 cores
//!
//! ## Module Organization
//!
//! - **`distance_computation/`**: Core distance algorithms
//!   - `engine.rs`: Unified distance compute engine
//!   - `simd/`: SIMD implementations for each metric
//!   - `traits.rs`: Distance computation traits
//!
//! - **`gpu/`**: GPU acceleration layer
//!   - `cuda.rs`: NVIDIA CUDA backend
//!   - `rocm.rs`: AMD ROCm backend
//!   - `distance.rs`: GPU distance kernels
//!
//! - **`quantization/`**: Vector quantization
//!   - `unified.rs`: Unified quantization engine
//!   - `storage_engine.rs`: Storage-optimized quantization
//!   - `types.rs`: Quantization types and configs
//!
//! ## Configuration
//!
//! ```toml
//! [compute]
//! # Hardware acceleration
//! backend_priority = ["cuda", "rocm", "avx2", "neon"]
//! auto_detect = true
//!
//! # CPU vectorization
//! [compute.cpu]
//! avx512 = true
//! avx2 = true
//! sse42 = true
//! neon = true  # ARM
//!
//! # GPU configuration
//! [compute.gpu]
//! memory_pool = "pooled"
//! batch_size = 1024
//! unified_memory = true
//! memory_limit_gb = 8.0
//!
//! # Quantization
//! [compute.quantization]
//! default = "adaptive"
//! int8_threshold = 0.95  # Accuracy threshold
//! pq_subspaces = 8
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use proximadb::compute::{ComputeConfig, UnifiedDistanceCompute};
//!
//! // Auto-detect hardware and create engine
//! let config = ComputeConfig::default();
//! let engine = UnifiedDistanceCompute::new(config)?;
//!
//! // Compute distances with automatic acceleration
//! let distances = engine.compute_distances(
//!     query_vector,
//!     database_vectors,
//!     DistanceMetric::Cosine
//! )?;
//!
//! // Use quantization for speed
//! let quantized = engine.quantize_int8(vectors)?;
//! let approx_distances = engine.compute_int8_distances(
//!     query_quantized,
//!     database_quantized
//! )?;
//! ```
//!
//! ## Hardware Optimization Strategy
//!
//! 1. **Detection**: Runtime CPU/GPU capability detection
//! 2. **Selection**: Choose optimal backend for workload
//! 3. **Dispatch**: Route computation to best implementation
//! 4. **Fallback**: Graceful degradation if hardware unavailable
//!
//! ## Memory Optimization
//!
//! - **Prefetching**: Adaptive prefetch for sequential access
//! - **Memory Mapping**: Zero-copy access for large datasets
//! - **NUMA Awareness**: Pin threads to local memory nodes
//! - **Huge Pages**: 2MB pages for reduced TLB misses

// Semantic module organization
// distance_computation extracted to the `proximadb-distance-kernel` foundation crate
// (decomposition contracts step 2; GPU edge inverted via register_gpu_accelerator_factory,
// issue #162). Re-exported as the module so existing `crate::compute::distance_computation::*`
// paths (storage + 50+ consumers) resolve unchanged.
pub use proximadb_distance_kernel as distance_computation;
pub mod gpu;
pub use proximadb_montecarlo::montecarlo;
pub mod pipeline_executor;
pub mod proximacodec;
pub mod quantization;

// NOTE (2026-06-04): the pluggable `provider` / `plan` / `scheduler` "Hadoop-style
// storage-compute separation" scaffold was DELETED — a dead stub
// (`LocalComputeProvider::execute` returned empty streams) with zero live callers,
// superseded by the DataFusion-direct OLAP path. A real distributed worker model,
// when needed, is a fresh coordinator/worker design, not this.

// Legacy distance module removed - all functionality moved to distance_computation::core

// Unit tests - will be added as modules are completed
// #[cfg(test)]
// pub mod tests;

// Re-export main APIs from semantic modules
pub use distance_computation::*;
pub use pipeline_executor::*;
pub use quantization::*;

/// Vector computation configuration
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Hardware acceleration preferences
    pub acceleration: AccelerationConfig,
    /// Algorithm selection and tuning
    pub algorithms: AlgorithmConfig,
    /// Memory optimization settings
    pub memory: MemoryConfig,
    /// Performance tuning parameters
    pub performance: ComputePerformanceConfig,
}

#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Preferred compute backend order
    pub backend_priority: Vec<ComputeBackend>,
    /// Enable CPU vectorization
    pub cpu_vectorization: CpuVectorization,
    /// GPU configuration
    pub gpu: GpuConfig,
    /// Math library preferences
    pub math_library: MathLibrary,
}

// ComputeBackend: use proximadb_hardware::SimdLevel directly for SIMD dispatch
pub use proximadb_hardware::SimdLevel as ComputeBackend;

#[derive(Debug, Clone)]
pub struct CpuVectorization {
    /// Enable AVX-512 instructions
    pub avx512: bool,
    /// Enable AVX2 instructions
    pub avx2: bool,
    /// Enable SSE4.2 instructions
    pub sse42: bool,
    /// Enable NEON instructions (ARM)
    pub neon: bool,
    /// Auto-detect best instruction set
    pub auto_detect: bool,
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Memory allocation strategy
    pub memory_pool: GpuMemoryPool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Enable unified memory (CUDA/ROCm)
    pub unified_memory: bool,
    /// GPU memory limit (GB)
    pub memory_limit_gb: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum GpuMemoryPool {
    /// Simple allocation/deallocation
    Simple,
    /// Memory pool for reuse
    Pooled { pool_size_gb: f32 },
    /// Unified memory management
    Unified,
}

#[derive(Debug, Clone)]
pub enum MathLibrary {
    /// Intel Math Kernel Library
    IntelMKL,
    /// OpenBLAS library
    OpenBLAS,
    /// BLIS (BLAS-like Library Instantiation Software)
    BLIS,
    /// Native Rust implementation
    Native,
    /// Auto-select best available
    Auto,
}

#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Default similarity metric
    pub default_metric: DistanceMetric,
    /// Index algorithm preferences
    pub index_algorithm: IndexAlgorithm,
    /// Search algorithm tuning
    pub search_params: ComputeSearchParams,
    /// Quantization settings
    pub quantization: UnifiedQuantizationLevel,
}

#[derive(Debug, Clone)]
pub enum IndexAlgorithm {
    /// Hierarchical Navigable Small World
    HNSW {
        m: usize,               // Number of bi-directional links
        ef_construction: usize, // Size of candidate set
        max_elements: usize,    // Maximum number of elements
    },
    /// Inverted File Index
    IVF {
        nlist: usize,  // Number of clusters
        nprobe: usize, // Number of clusters to search
    },
    /// Locality Sensitive Hashing
    LSH {
        num_tables: usize, // Number of hash tables
        hash_size: usize,  // Size of each hash
    },
    /// Product Quantization
    PQ {
        subspace_count: usize, // Number of subspaces
        bits_per_code: usize,  // Bits per subspace code
    },
    /// Brute force (exact search)
    BruteForce,
    /// Auto-select based on data characteristics
    Auto,
}

#[derive(Debug, Clone)]
pub struct ComputeSearchParams {
    /// Search accuracy vs speed trade-off
    pub accuracy_target: f32, // 0.0 = fastest, 1.0 = most accurate
    /// Maximum search time (milliseconds)
    pub max_search_time_ms: u32,
    /// Early termination threshold
    pub early_termination_threshold: Option<f32>,
    /// Parallel search threads
    pub parallel_threads: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Prefetch strategy for vector data
    pub prefetch_strategy: PrefetchStrategy,
    /// Memory mapping configuration
    pub mmap_config: MmapConfig,
    /// Cache configuration
    pub cache_config: ComputeCacheConfig,
}

#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential { similarity: usize },
    /// Pattern-based prefetching
    Pattern { pattern_buffer_size: usize },
    /// ML-driven prefetching
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// Memory mapping advice
    pub madvise: MadviseHint,
    /// Populate pages immediately
    pub populate: bool,
    /// Use huge pages
    pub huge_pages: bool,
    /// NUMA node binding
    pub numa_node: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum MadviseHint {
    Normal,
    Random,
    Sequential,
    WillNeed,
    DontNeed,
}

/// Backwards-compat alias for [`ComputeCacheConfig`].
pub type CacheConfig = ComputeCacheConfig;

#[derive(Debug, Clone)]
pub struct ComputeCacheConfig {
    /// L1 cache size (vectors in memory)
    pub l1_cache_size: usize,
    /// L2 cache size (compressed vectors)
    pub l2_cache_size: usize,
    /// Cache replacement policy
    pub replacement_policy: CachePolicy,
}

#[derive(Debug, Clone)]
pub enum CachePolicy {
    LRU,  // Least Recently Used
    LFU,  // Least Frequently Used
    ARC,  // Adaptive Replacement Cache
    TwoQ, // Two Queue
}

/// Backwards-compat alias for [`ComputePerformanceConfig`].
pub type PerformanceConfig = ComputePerformanceConfig;

#[derive(Debug, Clone)]
pub struct ComputePerformanceConfig {
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
    /// Unroll loops for better performance
    pub loop_unrolling: bool,
    /// Enable branch prediction optimization
    pub branch_prediction: bool,
    /// Use memory prefaulting
    pub memory_prefault: bool,
    /// Thread affinity configuration
    pub thread_affinity: ThreadAffinity,
}

#[derive(Debug, Clone)]
pub enum ThreadAffinity {
    /// No specific affinity
    None,
    /// Bind to specific CPU cores
    Cores(Vec<usize>),
    /// Bind to NUMA node
    NumaNode(u32),
    /// Auto-detect optimal affinity
    Auto,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            acceleration: AccelerationConfig {
                backend_priority: vec![
                    ComputeBackend::AVX512,
                    ComputeBackend::AVX2,
                    ComputeBackend::NEON,
                    ComputeBackend::SSE41,
                ],
                cpu_vectorization: CpuVectorization {
                    avx512: true,
                    avx2: true,
                    sse42: true,
                    neon: true,
                    auto_detect: true,
                },
                gpu: GpuConfig {
                    memory_pool: GpuMemoryPool::Pooled { pool_size_gb: 4.0 },
                    batch_size: 1024,
                    unified_memory: true,
                    memory_limit_gb: Some(8.0),
                },
                math_library: MathLibrary::Auto,
            },
            algorithms: AlgorithmConfig {
                default_metric: DistanceMetric::Cosine,
                index_algorithm: IndexAlgorithm::Auto,
                search_params: ComputeSearchParams {
                    accuracy_target: 0.95,
                    max_search_time_ms: 100,
                    early_termination_threshold: None,
                    parallel_threads: None,
                },
                quantization: UnifiedQuantizationLevel {
                    level_type: None, // No quantization
                },
            },
            memory: MemoryConfig {
                prefetch_strategy: PrefetchStrategy::Adaptive,
                mmap_config: MmapConfig {
                    madvise: MadviseHint::WillNeed,
                    populate: true,
                    huge_pages: true,
                    numa_node: None,
                },
                cache_config: ComputeCacheConfig {
                    l1_cache_size: 100_000,   // 100K vectors
                    l2_cache_size: 1_000_000, // 1M vectors compressed
                    replacement_policy: CachePolicy::ARC,
                },
            },
            performance: ComputePerformanceConfig {
                simd_enabled: true,
                loop_unrolling: true,
                branch_prediction: true,
                memory_prefault: true,
                thread_affinity: ThreadAffinity::Auto,
            },
        }
    }
}

/// Hardware capability detection
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_features: CpuFeatures,
    pub gpu_devices: Vec<GpuDevice>,
    pub memory_info: MemoryInfo,
    pub numa_topology: NumaTopology,
}

// CpuFeatures and CacheSizes are root-crate-specific (complex NUMA/cache metadata)
// Use crate::core::hardware_capabilities directly for GPU and detailed CPU info
pub use crate::core::hardware_capabilities::{CacheSizes, CpuFeatures};

// Using central GpuDevice and GpuBackend from hardware_capabilities module
pub use crate::core::hardware_capabilities::{GpuBackend, GpuDevice};

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory: u64,
    pub available_memory: u64,
    pub page_size: usize,
    pub huge_page_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub node_count: usize,
    pub nodes: Vec<NumaNode>,
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpu_cores: Vec<usize>,
    pub memory_total: u64,
    pub memory_free: u64,
}

/// Get hardware info from centralized hardware capabilities (no duplicate detection)
pub fn get_hardware_info() -> HardwareInfo {
    let caps = crate::core::hardware_capabilities::get_hardware_capabilities();

    HardwareInfo {
        cpu_features: caps.cpu.features.clone(),
        gpu_devices: caps.gpu.devices.clone(),
        memory_info: MemoryInfo {
            total_memory: caps.memory.total_memory,
            available_memory: caps.memory.total_memory / 2, // Rough estimate
            page_size: 4096,
            huge_page_size: Some(2 * 1024 * 1024),
        },
        numa_topology: NumaTopology {
            node_count: 1,
            nodes: vec![NumaNode {
                node_id: 0,
                cpu_cores: (0..caps.cpu.logical_cores).collect(),
                memory_total: caps.memory.total_memory,
                memory_free: caps.memory.total_memory / 2,
            }],
        },
    }
}

#[cfg(test)]
mod unified_quantization_tests {
    use crate::compute::quantization::types::{
        BinaryQuantization, NoQuantization, ProductQuantization, QuantizationScheme,
        ScalarQuantization, UnifiedQuantizationLevel, UniformQuantization,
    };
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn setup_hardware_capabilities() {
        INIT.call_once(|| {
            let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        });
    }

    #[test]
    fn test_quantization_level_creation() {
        setup_hardware_capabilities();

        // Test PQ8 creation
        let pq8 = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::Pq(ProductQuantization {
                bits_per_code: 8,
                num_subvectors: 16,
                codebook_id: None,
                adaptive_subvectors: false,
            })),
        };

        assert!(pq8.level_type.is_some());
        if let Some(QuantizationScheme::Pq(pq)) = &pq8.level_type {
            assert_eq!(pq.bits_per_code, 8);
            assert_eq!(pq.num_subvectors, 16);
        }

        // Test Uniform quantization
        let uniform4 = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::Uniform(UniformQuantization {
                bits: 4,
                scale: None,
                offset: None,
            })),
        };

        assert!(uniform4.level_type.is_some());
        if let Some(QuantizationScheme::Uniform(uniform)) = &uniform4.level_type {
            assert_eq!(uniform.bits, 4);
        }

        // Test Binary quantization
        let binary = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::Binary(BinaryQuantization {
                threshold: None,
                sign_based: false,
            })),
        };

        assert!(binary.level_type.is_some());
        if let Some(QuantizationScheme::Binary(bin)) = &binary.level_type {
            assert!(!bin.sign_based);
        }
    }

    #[test]
    fn test_quantization_none() {
        setup_hardware_capabilities();

        let none_quant = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::None(NoQuantization {})),
        };

        assert!(none_quant.level_type.is_some());
        assert!(matches!(
            none_quant.level_type,
            Some(QuantizationScheme::None(_))
        ));
    }

    #[test]
    fn test_scalar_quantization() {
        setup_hardware_capabilities();

        let scalar = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::Scalar(ScalarQuantization {
                bits: 8,
                scale: 1.0,
                offset: 0.0,
                clamp_values: false,
            })),
        };

        assert!(scalar.level_type.is_some());
        if let Some(QuantizationScheme::Scalar(s)) = &scalar.level_type {
            assert_eq!(s.bits, 8);
        }
    }

    #[test]
    fn test_quantization_level_display() {
        setup_hardware_capabilities();

        let pq_level = UnifiedQuantizationLevel {
            level_type: Some(QuantizationScheme::Pq(ProductQuantization {
                bits_per_code: 8,
                num_subvectors: 16,
                codebook_id: None,
                adaptive_subvectors: false,
            })),
        };

        let display_str = format!("{:?}", pq_level);
        assert!(display_str.contains("UnifiedQuantizationLevel"));
    }

    #[test]
    fn test_product_quantization_subvectors() {
        setup_hardware_capabilities();

        let pq = ProductQuantization {
            bits_per_code: 8,
            num_subvectors: 32,
            codebook_id: None,
            adaptive_subvectors: true,
        };

        assert_eq!(pq.bits_per_code, 8);
        assert_eq!(pq.num_subvectors, 32);
        assert!(pq.adaptive_subvectors);
        assert!(pq.codebook_id.is_none());
    }

    #[test]
    fn test_uniform_quantization_params() {
        setup_hardware_capabilities();

        let uniform = UniformQuantization {
            bits: 4,
            scale: Some(0.01),
            offset: Some(-128.0),
        };

        assert_eq!(uniform.bits, 4);
        assert_eq!(uniform.scale, Some(0.01));
        assert_eq!(uniform.offset, Some(-128.0));
    }

    #[test]
    fn test_binary_quantization_threshold() {
        setup_hardware_capabilities();

        let binary = BinaryQuantization {
            threshold: Some(0.5),
            sign_based: true,
        };

        assert_eq!(binary.threshold, Some(0.5));
        assert!(binary.sign_based);
    }

    #[test]
    fn test_no_quantization() {
        setup_hardware_capabilities();

        let none = NoQuantization {};
        let display_str = format!("{:?}", none);
        assert!(display_str.contains("NoQuantization"));
    }

    #[test]
    fn test_quantization_levels() {
        setup_hardware_capabilities();

        // Test that all quantization levels can be created
        let levels = vec![
            UnifiedQuantizationLevel {
                level_type: Some(QuantizationScheme::None(NoQuantization {})),
            },
            UnifiedQuantizationLevel {
                level_type: Some(QuantizationScheme::Binary(BinaryQuantization {
                    threshold: None,
                    sign_based: false,
                })),
            },
            UnifiedQuantizationLevel {
                level_type: Some(QuantizationScheme::Scalar(ScalarQuantization {
                    bits: 8,
                    scale: 1.0,
                    offset: 0.0,
                    clamp_values: false,
                })),
            },
        ];

        assert_eq!(levels.len(), 3);
        for level in levels {
            assert!(level.level_type.is_some());
        }
    }
}
