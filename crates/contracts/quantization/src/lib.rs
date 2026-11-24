//! # Quantization Contracts
//!
//! This crate provides abstraction traits for quantization operations,
//! enabling the quantization engine to work across different storage backends
//! and hardware acceleration platforms without tight coupling.
//!
//! ## Design Principles
//!
//! - **Storage Agnostic**: Traits work across NOVA, SST, HELIX, and other engines
//! - **Hardware Abstraction**: SIMD/GPU acceleration hidden behind traits
//! - **Cache Flexibility**: Pluggable caching implementations
//! - **Async First**: All storage operations are async for scalability
//!
//! ## Contract Layers
//!
//! 1. **HardwareAcceleration**: Abstract SIMD/GPU capabilities
//! 2. **QuantizationCache**: Abstract codebook and vector caching
//! 3. **CodebookStore**: Abstract persistent codebook storage
//!
//! ## Migration Path
//!
//! Current implementations in `src/compute/quantization/` will implement these traits,
//! enabling gradual migration without breaking existing functionality.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// HARDWARE ACCELERATION CONTRACT
// ============================================================================

/// Hardware backend type for quantization operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareBackend {
    /// No acceleration (scalar fallback)
    Scalar,
    /// SSE4.2 SIMD (128-bit)
    SSE,
    /// AVX2 SIMD (256-bit)
    AVX2,
    /// AVX-512 SIMD (512-bit)
    AVX512,
    /// ARM NEON SIMD (128-bit)
    NEON,
    /// NVIDIA CUDA GPU
    CUDA,
    /// AMD ROCm GPU
    ROCm,
    /// Apple Metal Performance Shaders
    MPS,
    /// OpenCL GPU compute
    OpenCL,
}

impl fmt::Display for HardwareBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::SSE => write!(f, "sse"),
            Self::AVX2 => write!(f, "avx2"),
            Self::AVX512 => write!(f, "avx512"),
            Self::NEON => write!(f, "neon"),
            Self::CUDA => write!(f, "cuda"),
            Self::ROCm => write!(f, "rocm"),
            Self::MPS => write!(f, "mps"),
            Self::OpenCL => write!(f, "opencl"),
        }
    }
}

/// Hardware capability descriptor used by the [`HardwareAcceleration`] trait.
///
/// This is a *contract type*, not a detector. Implementations of the trait
/// build this descriptor from whatever runtime detection they prefer — the
/// canonical SIMD/CPU/memory probe lives in `proximadb_hardware`, and the
/// richer CPU-topology/GPU detector lives in `src/core/hardware_capabilities`.
///
/// Naming note: this type used to be called `HardwareCapabilities`, which
/// collided with both of the above. It was renamed because its role is to
/// describe quantization-relevant fields (backend choice + GPU batch
/// threshold) to a trait method, not to perform detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationHardwareDescriptor {
    /// Selected hardware backend
    pub backend: HardwareBackend,

    /// CPU SIMD feature support
    pub cpu_features: CpuFeatures,

    /// GPU availability and characteristics
    pub gpu: GpuFeatures,

    /// Recommended batch size for GPU operations
    pub gpu_batch_threshold: usize,
}

/// CPU SIMD feature support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuFeatures {
    pub avx512_support: bool,
    pub avx2_support: bool,
    pub sse42_support: bool,
    pub neon_support: bool,
}

/// GPU feature support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuFeatures {
    pub backend: GpuBackend,
    pub available: bool,
    pub device_memory_mb: Option<usize>,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    None,
    CUDA,
    ROCm,
    MPS,
    OpenCL,
}

/// Abstract hardware acceleration interface for quantization operations
///
/// This trait abstracts the underlying hardware (SIMD/GPU) to enable
/// the quantization engine to work across different platforms without
/// direct dependencies on hardware detection libraries.
#[async_trait::async_trait]
pub trait HardwareAcceleration: Send + Sync {
    /// Get current hardware capabilities
    async fn get_capabilities(&self) -> Result<QuantizationHardwareDescriptor>;

    /// Select optimal backend for the given operation and data size
    async fn select_backend(&self, data_size: usize) -> Result<HardwareBackend>;

    /// Check if GPU should be used for the given batch size
    fn should_use_gpu(
        &self,
        data_size: usize,
        capabilities: &QuantizationHardwareDescriptor,
    ) -> bool {
        data_size >= capabilities.gpu_batch_threshold
    }

    /// Quantize to 8-bit using optimal hardware path
    async fn quantize_u8(&self, values: &[f32]) -> Result<Vec<u8>>;

    /// Quantize to 4-bit using optimal hardware path
    async fn quantize_u4(&self, values: &[f32]) -> Result<Vec<u8>>;

    /// Dequantize from 8-bit using optimal hardware path
    async fn dequantize_u8(&self, quantized: &[u8], scale: f32, offset: f32) -> Result<Vec<f32>>;
}

// ============================================================================
// QUANTIZATION CACHE CONTRACT
// ============================================================================

/// Cache key for quantization codebooks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuantizationCacheKey {
    pub collection_id: String,
    pub quantization_type: String,
    pub level_params: String,
}

impl QuantizationCacheKey {
    /// Create cache key for PQ quantization
    pub fn pq(collection_id: &str, bits_per_code: u8, num_subvectors: u32) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "pq".to_string(),
            level_params: format!("{}_{}", bits_per_code, num_subvectors),
        }
    }

    /// Create cache key for binary quantization
    pub fn binary(collection_id: &str) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "binary".to_string(),
            level_params: "1".to_string(),
        }
    }

    /// Create cache key for INT8 quantization
    pub fn int8(collection_id: &str) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "int8".to_string(),
            level_params: "8".to_string(),
        }
    }
}

impl fmt::Display for QuantizationCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}#{}#{}",
            self.collection_id, self.quantization_type, self.level_params
        )
    }
}

/// Cache statistics for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}

/// Abstract quantization cache interface
///
/// This trait abstracts the caching mechanism for codebooks and quantized
/// vectors, allowing different implementations (in-memory, distributed, etc.)
/// without changing the quantization engine core.
#[async_trait::async_trait]
pub trait QuantizationCache: Send + Sync {
    /// Retrieve a codebook from cache
    async fn get_codebook(&self, key: &QuantizationCacheKey) -> Result<Option<Vec<u8>>>;

    /// Store a codebook in cache
    async fn put_codebook(&self, key: &QuantizationCacheKey, codebook: &[u8]) -> Result<()>;

    /// Check if a codebook exists in cache
    async fn contains_codebook(&self, key: &QuantizationCacheKey) -> Result<bool>;

    /// Remove a codebook from cache
    async fn remove_codebook(&self, key: &QuantizationCacheKey) -> Result<bool>;

    /// Clear all codebooks for a collection
    async fn clear_collection(&self, collection_id: &str) -> Result<usize>;

    /// Get cache statistics
    async fn get_statistics(&self) -> Result<CacheStatistics>;

    /// Warm cache with pre-loaded codebooks
    async fn warm_cache(&self, codebooks: HashMap<QuantizationCacheKey, Vec<u8>>) -> Result<()> {
        for (key, codebook) in codebooks {
            self.put_codebook(&key, &codebook).await?;
        }
        Ok(())
    }
}

// ============================================================================
// CODEBOOK STORAGE CONTRACT
// ============================================================================

/// Codebook data structure
///
/// Codebooks contain the learned parameters for quantization:
/// - **PQ**: Centroid vectors for each subspace
/// - **Scalar**: Scale and offset per dimension
/// - **Binary**: Threshold values per dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Unique identifier
    pub id: String,

    /// Quantization type (pq, int8, binary)
    pub quantization_type: String,

    /// Quantization level parameters
    pub level_params: String,

    /// Codebook data (serialized for storage flexibility)
    pub data: Vec<u8>,

    /// Metadata for codebook management
    pub metadata: CodebookMetadata,
}

/// Codebook metadata for management operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookMetadata {
    pub created_at: i64,
    pub updated_at: i64,
    pub version: u64,
    pub size_bytes: usize,
    pub num_vectors: usize,
    pub dimension: usize,
}

/// Abstract codebook storage interface
///
/// This trait abstracts persistent storage for codebooks, enabling
/// different storage backends (in-memory, file-based, distributed) without
/// changing the quantization engine core.
#[async_trait::async_trait]
pub trait CodebookStore: Send + Sync {
    /// Store a codebook
    async fn store_codebook(&self, id: &str, codebook: &Codebook) -> Result<()>;

    /// Retrieve a codebook
    async fn get_codebook(&self, id: &str) -> Result<Option<Codebook>>;

    /// List available codebooks
    async fn list_codebooks(&self) -> Result<Vec<String>>;

    /// Delete a codebook
    async fn delete_codebook(&self, id: &str) -> Result<bool>;

    /// Check if a codebook exists
    async fn contains_codebook(&self, id: &str) -> Result<bool>;

    /// Get codebook metadata without loading full data
    async fn get_codebook_metadata(&self, id: &str) -> Result<Option<CodebookMetadata>>;
}

// ============================================================================
// QUANTIZATION ENGINE CONTRACT
// ============================================================================

/// Parameter shape for the [`QuantizationEngine`] trait methods.
///
/// This is a *contract type*, not the in-process or wire `QuantizationConfig`.
/// Use the proto-generated `proximadb_proto::v1::QuantizationConfig` for
/// wire/REST surfaces (dominant in-process type, ~80 consumers) or the
/// strongly-typed `proximadb_quantization_types::QuantizationConfig` for the
/// storage-layer typed selector. This struct intentionally keeps the
/// `level_params: String` escape hatch so trait implementors can accept any
/// level encoding without being locked to a specific enum.
///
/// Naming note: this type used to be called `QuantizationConfig`, which
/// collided with both of the above. It was renamed because its role is to
/// describe trait-method parameters, not to be the canonical config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationServiceParams {
    /// Quantization type (pq, int8, binary)
    pub quantization_type: String,

    /// Quantization level parameters
    pub level_params: String,

    /// Hardware acceleration enabled
    pub enable_hardware_acceleration: bool,

    /// Cache enabled
    pub enable_cache: bool,
}

/// Abstract quantization engine interface
///
/// This is the main abstraction that enables the quantization system
/// to work across different storage engines and hardware platforms.
#[async_trait::async_trait]
pub trait QuantizationEngine: Send + Sync {
    /// Quantize vectors to the specified level
    async fn quantize(
        &self,
        vectors: &[f32],
        config: &QuantizationServiceParams,
    ) -> Result<Vec<u8>>;

    /// Dequantize vectors back to FP32
    async fn dequantize(
        &self,
        quantized: &[u8],
        config: &QuantizationServiceParams,
        dimension: usize,
    ) -> Result<Vec<f32>>;

    /// Train a new codebook for the given data
    async fn train_codebook(
        &self,
        collection_id: &str,
        vectors: &[f32],
        config: &QuantizationServiceParams,
    ) -> Result<Codebook>;

    /// Get or load a codebook for quantization operations
    async fn get_or_train_codebook(
        &self,
        collection_id: &str,
        config: &QuantizationServiceParams,
    ) -> Result<Codebook>;
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_backend_display() {
        assert_eq!(HardwareBackend::Scalar.to_string(), "scalar");
        assert_eq!(HardwareBackend::AVX2.to_string(), "avx2");
        assert_eq!(HardwareBackend::CUDA.to_string(), "cuda");
    }

    #[test]
    fn test_cache_key_creation() {
        let pq_key = QuantizationCacheKey::pq("test_collection", 8, 16);
        assert_eq!(pq_key.collection_id, "test_collection");
        assert_eq!(pq_key.quantization_type, "pq");
        assert_eq!(pq_key.level_params, "8_16");

        let binary_key = QuantizationCacheKey::binary("test_collection");
        assert_eq!(binary_key.level_params, "1");

        let int8_key = QuantizationCacheKey::int8("test_collection");
        assert_eq!(int8_key.level_params, "8");
    }

    #[test]
    fn test_cache_key_display() {
        let key = QuantizationCacheKey::pq("my_collection", 4, 8);
        assert_eq!(key.to_string(), "my_collection#pq#4_8");
    }

    #[test]
    fn test_cache_statistics_hit_rate() {
        let stats = CacheStatistics {
            total_entries: 100,
            total_size_bytes: 1024000,
            hit_count: 80,
            miss_count: 20,
            eviction_count: 5,
        };
        assert_eq!(stats.hit_rate(), 0.8);

        let empty_stats = CacheStatistics {
            total_entries: 0,
            total_size_bytes: 0,
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
        };
        assert_eq!(empty_stats.hit_rate(), 0.0);
    }
}
