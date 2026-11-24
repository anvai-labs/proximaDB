//! Common Quantized Data Structure for All Storage Engines
//!
//! This module provides a unified quantized data structure that works across
//! all storage engines (ProximaBlock-based: SST/SWIFT/HELIX and Parquet-based: NOVA).
//!
//! The design ensures:
//! - File-level quantization for optimal performance
//! - Consistent quantization formats across engines
//! - Memory-efficient storage of quantized representations
//! - Integration with existing quantization infrastructure
//!
//! ## Layer Distinction
//!
//! This module defines **STORAGE FORMAT** types, not API types:
//!
//! - **API Level** (`proximadb-quantization-types`): `QuantizationLevel` describes precision (Int4, Int8, FP32)
//! - **Storage Level** (this module): `StorageQuantizationFormat` describes on-disk format (Binary, PQ4, PQ8)
//! - **Config Level** (engine-specific): `QuantizationAggressiveness` describes compression tradeoff (Low, Medium, High)
//!
//! ### Example Mapping
//!
//! ```text
//! API Request: QuantizationLevel::Int8 (8-bit precision)
//!     ↓ converts to
//! Storage Format: StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
//!     ↓ stored as
//! On-Disk Format: Int8 scalar quantized vectors (1 byte per dimension)
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::storage::engines::core::formats::columnar::constants::DEFAULT_ROW_GROUP_SIZE;

// ============================================================================
// API-to-Storage Conversion Layer
// ============================================================================

/// Conversion module between API-level foundation types and storage-level format types
///
/// This module provides type-safe conversions between:
/// - **API Level**: `proximadb_quantization_types::QuantizationLevel` (user-facing precision)
/// - **Storage Level**: `StorageQuantizationFormat` (on-disk format)
///
/// ## Conversion Logic
///
/// The conversion from API precision to storage format follows these rules:
///
/// 1. **No Quantization** → FP32 precision stored as uncompressed (BinaryFormat as fallback)
/// 2. **Integer Precision** → Corresponding scalar format (Int4, Int8, UInt8)
/// 3. **Floating Point Precision** → Approximated as nearest scalar format
/// 4. **Binary Method** → Binary format directly
///
/// ## Examples
///
/// ```ignore
/// use proximadb_quantization_types::QuantizationLevel as ApiQuantizationLevel;
/// use crate::storage::engines::core::formats::common_quantization::StorageQuantizationFormat;
///
/// // API: 8-bit precision
/// let api_level = ApiQuantizationLevel::Int8;
///
/// // Storage: 8-bit scalar format
/// let storage_format = StorageQuantizationFormat::from(api_level);
/// assert_eq!(storage_format, StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8));
/// ```
pub mod conversion {
    use super::*;
    use proximadb_quantization_types::{
        QuantizationLevel as ApiQuantizationLevel, QuantizationType,
    };

    /// Convert API quantization level to storage format
    ///
    /// This implements the conversion from user-facing precision levels to
    /// on-disk storage formats. The conversion is lossless for integer types
    /// and uses best-effort approximation for floating point types.
    ///
    /// ## Conversion Table
    ///
    /// | API Level | Storage Format | Notes |
    /// |-----------|----------------|-------|
    /// | `None` | `BinaryFormat` | Stored as binary for fast filtering |
    /// | `Int4` | `ScalarFormat(Int4)` | Direct mapping |
    /// | `Int8` | `ScalarFormat(Int8)` | Direct mapping |
    /// | `UInt8` | `ScalarFormat(UInt8)` | Direct mapping |
    /// | `FP16` | `ScalarFormat(Int8)` | Approximated as 8-bit scalar |
    /// | `FP32` | `BinaryFormat` | No quantization, use binary for speed |
    ///
    /// ## Rationale
    ///
    /// - **BinaryFormat for None/FP32**: Binary quantization provides fast
    ///   candidate filtering even when full precision is requested. The full
    ///   precision vectors are stored separately and binary is used only for
    ///   acceleration.
    ///
    /// - **FP16 → Int8**: We don't currently support FP16 storage, so we
    ///   approximate as 8-bit scalar which provides good accuracy/performance
    ///   tradeoff.
    impl From<ApiQuantizationLevel> for StorageQuantizationFormat {
        fn from(api_level: ApiQuantizationLevel) -> Self {
            match api_level {
                // No quantization → Use binary for fast filtering
                // (Full precision vectors stored separately)
                ApiQuantizationLevel::None => StorageQuantizationFormat::BinaryFormat,

                // Integer precision → Direct mapping to scalar format
                ApiQuantizationLevel::Int4 => {
                    StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int4)
                }
                ApiQuantizationLevel::Int8 => {
                    StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
                }
                ApiQuantizationLevel::UInt8 => {
                    StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::UInt8)
                }

                // Floating point → Approximate as scalar (we don't store FP16/FP32 directly)
                ApiQuantizationLevel::FP16 => {
                    // FP16 approximated as 8-bit scalar for storage efficiency
                    StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
                }
                ApiQuantizationLevel::FP32 => {
                    // FP32 stored as binary for fast candidate filtering
                    // (Full precision vectors stored in main column)
                    StorageQuantizationFormat::BinaryFormat
                }
            }
        }
    }

    /// Convert storage format back to API quantization level
    ///
    /// This reverse conversion is approximate for some formats since storage
    /// formats are more detailed than API precision levels.
    ///
    /// ## Conversion Table
    ///
    /// | Storage Format | API Level | Notes |
    /// |---------------|-----------|-------|
    /// | `BinaryFormat` | `None` | Binary is used for no-quantization case |
    /// | `ScalarFormat(Int4)` | `Int4` | Direct mapping |
    /// | `ScalarFormat(Int8)` | `Int8` | Direct mapping |
    /// | `ScalarFormat(UInt8)` | `UInt8` | Direct mapping |
    /// | `ProductFormat(*)` | `Int8` | PQ approximated as 8-bit precision |
    impl From<StorageQuantizationFormat> for ApiQuantizationLevel {
        fn from(storage_format: StorageQuantizationFormat) -> Self {
            match storage_format {
                StorageQuantizationFormat::BinaryFormat => ApiQuantizationLevel::None,
                StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                    ScalarQuantizationBits::Int4 => ApiQuantizationLevel::Int4,
                    ScalarQuantizationBits::Int8 => ApiQuantizationLevel::Int8,
                    ScalarQuantizationBits::UInt8 => ApiQuantizationLevel::UInt8,
                },
                StorageQuantizationFormat::ProductFormat(_) => {
                    // Product quantization approximated as 8-bit precision
                    // (This is a simplification; in reality PQ provides variable precision)
                    ApiQuantizationLevel::Int8
                }
            }
        }
    }

    /// Convert API quantization type to storage format
    ///
    /// This converts the quantization method (Scalar, Product, Binary) to
    /// a recommended storage format.
    ///
    /// ## Conversion Logic
    ///
    /// - `None` → `BinaryFormat` (No quantization, use binary for speed)
    /// - `Scalar` → `ScalarFormat(Int8)` (Default 8-bit scalar)
    /// - `Product` → `ProductFormat(PQ8)` (Default 8-bit product quantization)
    /// - `Binary` → `BinaryFormat` (Direct mapping)
    impl From<QuantizationType> for StorageQuantizationFormat {
        fn from(qtype: QuantizationType) -> Self {
            match qtype {
                QuantizationType::None => StorageQuantizationFormat::BinaryFormat,
                QuantizationType::Scalar => {
                    // Default to 8-bit scalar for scalar quantization
                    StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
                }
                QuantizationType::Product => {
                    // Default to 8-bit product quantization
                    StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
                }
                QuantizationType::Binary => StorageQuantizationFormat::BinaryFormat,
                // TurboQuant has no analogue in the legacy
                // StorageQuantizationFormat enum. Map to the binary format
                // sentinel here; real routing lives in
                // `proximadb_vector::quantization::turboquant` (P8).
                #[cfg(feature = "experimental-turboquant")]
                QuantizationType::TurboQuant => StorageQuantizationFormat::BinaryFormat,
            }
        }
    }

    /// Try to determine the best storage format for a given API quantization config
    ///
    /// This function considers both the quantization type and level to recommend
    /// the most appropriate storage format.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// use proximadb_quantization_types::{QuantizationConfig, QuantizationLevel, QuantizationType};
    ///
    /// let config = QuantizationConfig {
    ///     quantization_type: QuantizationType::Scalar,
    ///     quantization_level: QuantizationLevel::Int4,
    ///     asymmetric: false,
    ///     cache: true,
    /// };
    ///
    /// let storage_format = recommend_storage_format(&config);
    /// // Returns: StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int4)
    /// ```
    pub fn recommend_storage_format(
        config: &proximadb_quantization_types::QuantizationConfig,
    ) -> StorageQuantizationFormat {
        // Prefer quantization level if specified, otherwise fall back to type
        if config.quantization_level != ApiQuantizationLevel::None {
            StorageQuantizationFormat::from(config.quantization_level)
        } else {
            StorageQuantizationFormat::from(config.quantization_type)
        }
    }

    /// Check if a storage format is compatible with an API quantization level
    ///
    /// Returns true if the storage format can adequately represent the requested precision.
    ///
    /// ## Compatibility Rules
    ///
    /// - `BinaryFormat` is compatible with all API levels (used for fast filtering)
    /// - `ScalarFormat(Int4)` requires `Int4` or lower precision
    /// - `ScalarFormat(Int8)` requires `Int8` or lower precision
    /// - `ProductFormat(*)` is compatible with all API levels (variable precision)
    pub fn is_compatible(
        api_level: ApiQuantizationLevel,
        storage_format: &StorageQuantizationFormat,
    ) -> bool {
        match storage_format {
            StorageQuantizationFormat::BinaryFormat => true, // Binary compatible with everything
            StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                ScalarQuantizationBits::Int4 => api_level.bit_width() <= 4,
                ScalarQuantizationBits::Int8 => {
                    api_level.bit_width() <= 8 || api_level.is_floating_point()
                }
                ScalarQuantizationBits::UInt8 => {
                    api_level.bit_width() <= 8 || api_level.is_floating_point()
                }
            },
            StorageQuantizationFormat::ProductFormat(_) => true, // PQ compatible with everything
        }
    }
}

// Re-export conversion module for convenience
pub use conversion::*;

/// On-disk storage format for quantized vectors
///
/// This enum describes the **physical storage format** of quantized data on disk,
/// NOT the precision level or compression aggressiveness.
///
/// ## Distinction from API Types
///
/// - **API Level** (`proximadb_quantization_types::QuantizationLevel`): Describes precision (Int4, Int8, UInt8, FP16, FP32)
/// - **Storage Level** (this enum): Describes on-disk binary format (Binary, PQ4, PQ8)
///
/// ## Storage Formats
///
/// - **BinaryFormat**: 1-bit per dimension (sign bit only)
/// - **ScalarFormat**: Uniform scalar quantization (Int4, Int8, UInt8)
/// - **ProductFormat**: Product quantization (PQ4, PQ8, PQ16, PQ32)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageQuantizationFormat {
    /// Binary quantization format (1 bit per dimension)
    BinaryFormat,

    /// Scalar quantization format with specified bit width
    ScalarFormat(ScalarQuantizationBits),

    /// Product quantization format with specified code width
    ProductFormat(ProductQuantizationBits),
}

/// Bit width options for scalar quantization
///
/// Defines the number of bits per dimension for scalar quantized storage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScalarQuantizationBits {
    /// 4-bit signed scalar quantization
    Int4,
    /// 8-bit signed scalar quantization
    Int8,
    /// 8-bit unsigned scalar quantization
    UInt8,
}

/// Code width options for product quantization
///
/// Defines the number of bits per PQ code in storage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProductQuantizationBits {
    /// 4-bit PQ codes
    PQ4,
    /// 8-bit PQ codes
    PQ8,
    /// 16-bit PQ codes
    PQ16,
    /// 32-bit PQ codes
    PQ32,
}

/// Legacy type alias for backward compatibility
/// TODO: Migrate all uses to StorageQuantizationFormat (Phase 3.2)
pub type QuantizationLevel = StorageQuantizationFormat;

/// Unified quantized file structure for all storage engines
///
/// This structure holds quantized representations at the file level,
/// allowing efficient batch operations and avoiding per-vector quantization overhead.
///
/// Used by:
/// - ProximaBlock engines: Embedded in QuantizedSection
/// - Parquet engines: Stored as separate columns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQuantizedFile {
    /// File metadata
    pub file_id: String,
    pub collection_id: String,
    pub quantization_timestamp: i64,

    /// Quantization configuration used
    pub quantization_config: QuantizationFileConfig,

    /// Quantized vector data (parallel to original vectors)
    pub quantized_data: QuantizedVectorData,

    /// Quantization statistics and metrics
    pub quantization_stats: CommonQuantizationStats,
}

/// Configuration for file-level quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationFileConfig {
    /// Enabled quantization formats for this file
    pub enabled_levels: Vec<StorageQuantizationFormat>,

    /// Quantization was performed during flush or compaction
    pub quantization_trigger: QuantizationTrigger,

    /// Memory budget used for quantization (MB)
    pub memory_budget_mb: usize,

    /// Batch size used for quantization
    pub batch_size: usize,

    /// Engine-specific configuration
    pub engine_config: EngineQuantizationConfig,
}

/// When quantization was triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationTrigger {
    /// During flush operation
    Flush,
    /// During compaction operation (LSM-style)
    Compaction { level: u32 },
    /// Background recompaction
    Background,
}

/// Engine-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineQuantizationConfig {
    /// ProximaBlock configuration (SST, SWIFT, HELIX)
    ProximaBlock {
        /// Store quantization in same blocks or separate blocks
        storage_strategy: ProximaBlockQuantizationStorage,
        /// Block size for quantized data
        quantized_block_size_kb: usize,
    },
    /// Parquet configuration (NOVA)
    Parquet {
        /// Separate columns for each quantization level
        separate_columns: bool,
        /// Compression algorithm for quantized columns
        quantized_compression: String,
        /// Row group size for quantized data
        row_group_size: usize,
    },
}

/// Storage strategy for ProximaBlock quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProximaBlockQuantizationStorage {
    /// Store quantized data inline with original data (same blocks)
    Inline,
    /// Store quantized data in separate blocks within same file
    SeparateBlocks,
    /// Store quantized data in separate file
    SeparateFile,
}

/// Columnar Quantized Vector Data - Direct Optional Columns
///
/// This structure represents quantized vectors as individual optional columns,
/// enabling direct columnar pruning and eliminating map key overhead.
///
/// Column Design Philosophy:
/// - Each quantization level is a separate optional column
/// - Direct columnar access (no nested sections or maps)
/// - Enables column-level pruning and projection
/// - Works efficiently in both ProximaBlocks and Parquet
///
/// ProximaBlock Storage:
/// ```text
/// ProximaDataBlock {
///   vector_fp32: Vec<f32>,           // main vector column
///   q_binary: Option<Vec<u8>>,       // optional binary column
///   q_int8: Option<Vec<i8>>,         // optional int8 column
///   q_pq4: Option<Vec<u8>>,          // optional pq4 column
///   q_pq8: Option<Vec<u8>>,          // optional pq8 column
///   // ... codebook columns
/// }
/// ```
///
/// Parquet Storage:
/// ```text
/// Schema: [
///   vector_fp32: LIST<FLOAT> NOT NULL,
///   q_binary: BINARY OPTIONAL,
///   q_int8: LIST<INT8> OPTIONAL,
///   q_pq4: BINARY OPTIONAL,
///   q_pq8: BINARY OPTIONAL,
///   // ... codebook columns
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVectorData {
    /// Original vector count and dimension (for validation)
    pub vector_count: usize,
    pub dimension: usize,

    // ==================== QUANTIZED VECTOR COLUMNS ====================
    // Each column represents one quantization level - direct columnar access
    /// q_binary: Binary quantized vectors (1 bit per dimension)
    /// Column name: "q_binary"
    /// Storage: Bit-packed bytes, (dimension + 7) / 8 bytes per vector
    /// Pruning: Can skip column entirely if not needed
    pub q_binary: Option<Vec<Vec<u8>>>,

    /// q_int8: INT8 quantized vectors (8 bits per dimension)
    /// Column name: "q_int8"
    /// Storage: i8 values, dimension bytes per vector
    /// Pruning: Column-level statistics available
    pub q_int8: Option<Vec<Vec<i8>>>,

    /// q_pq4: PQ4 quantized vectors (4 bits per code)
    /// Column name: "q_pq4"
    /// Storage: Packed 4-bit codes, (num_subquantizers + 1) / 2 bytes per vector
    /// Pruning: Can skip if higher precision available
    pub q_pq4: Option<Vec<Vec<u8>>>,

    /// q_pq8: PQ8 quantized vectors (8 bits per code)
    /// Column name: "q_pq8"
    /// Storage: u8 codes, num_subquantizers bytes per vector
    /// Pruning: Most commonly used quantization level
    pub q_pq8: Option<Vec<Vec<u8>>>,

    /// q_pq16: PQ16 quantized vectors (16 bits per code)
    /// Column name: "q_pq16"
    /// Storage: u16 codes as bytes, num_subquantizers * 2 bytes per vector
    /// Pruning: Higher precision alternative to PQ8
    pub q_pq16: Option<Vec<Vec<u8>>>,

    /// q_pq32: PQ32 quantized vectors (32 bits per code)
    /// Column name: "q_pq32"
    /// Storage: u32 codes as bytes, num_subquantizers * 4 bytes per vector
    /// Pruning: Highest precision quantized option
    pub q_pq32: Option<Vec<Vec<u8>>>,

    // ==================== CODEBOOK STORAGE ====================
    // NOTE: Codebooks are stored as FILE-LEVEL metadata, not per-row columns!
    // This is much more efficient since codebooks are shared across all vectors
    // Storage locations:
    // - Parquet: In file footer metadata
    // - ProximaBlocks: In dedicated codebook section
    // - Sidecar: Optional .codebook files

    // ==================== QUANTIZATION PARAMETER COLUMNS ====================
    // Direct optional columns for quantization parameters - enable parameter pruning
    /// qp_binary_threshold: Binary quantization threshold column
    /// Column name: "qp_binary_threshold"
    /// Storage: Single f32 value per file (constant column)
    /// Pruning: Can skip loading if binary not used
    pub qp_binary_threshold: Option<f32>,

    /// qp_int8_min: INT8 quantization min value column
    /// Column name: "qp_int8_min"
    /// Storage: Single f32 value per file
    pub qp_int8_min: Option<f32>,

    /// qp_int8_max: INT8 quantization max value column
    /// Column name: "qp_int8_max"
    /// Storage: Single f32 value per file
    pub qp_int8_max: Option<f32>,

    /// qp_int8_scale: INT8 quantization scale factor column
    /// Column name: "qp_int8_scale"
    /// Storage: Single f32 value per file
    pub qp_int8_scale: Option<f32>,

    /// qp_pq_subquantizers: Number of PQ subquantizers column
    /// Column name: "qp_pq_subquantizers"
    /// Storage: Single usize value per file (applies to all PQ levels)
    pub qp_pq_subquantizers: Option<usize>,

    /// qp_pq_centroids: Number of PQ centroids column
    /// Column name: "qp_pq_centroids"
    /// Storage: Single usize value per file
    pub qp_pq_centroids: Option<usize>,
}

/// Metadata for a specific quantization format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationLevelMetadata {
    /// Storage quantization format
    pub level: StorageQuantizationFormat,

    /// Compression ratio achieved (original_size / quantized_size)
    pub compression_ratio: f32,

    /// Quantization quality metrics
    pub quality_metrics: QuantizationQuality,

    /// Memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Encoding-specific parameters
    pub encoding_params: EncodingParameters,
}

/// Quality metrics for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationQuality {
    /// Mean Squared Error compared to original
    pub mse: f32,

    /// Cosine similarity preservation (0.0 - 1.0)
    pub cosine_similarity_preservation: f32,

    /// Estimated recall at different k values
    pub estimated_recall_at_k: HashMap<usize, f32>, // k -> recall
}

/// Encoding-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingParameters {
    Binary {
        threshold: f32,
    },
    Int8 {
        min_value: f32,
        max_value: f32,
        scale_factor: f32,
    },
    ProductQuantization {
        num_subquantizers: usize,
        subquantizer_dimension: usize,
        num_centroids: usize,
        training_iterations: usize,
    },
}

/// Backwards-compat alias for [`CommonQuantizationStats`].
pub type QuantizationStats = CommonQuantizationStats;

/// File-level quantization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonQuantizationStats {
    /// Total time spent on quantization (ms)
    pub quantization_time_ms: u64,

    /// Memory usage during quantization (bytes)
    pub peak_memory_usage_bytes: usize,

    /// Storage savings achieved
    pub storage_savings: StorageSavings,

    /// Performance impact estimates
    pub performance_impact: PerformanceImpact,
}

/// Storage savings from quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSavings {
    /// Original file size (bytes)
    pub original_size_bytes: usize,

    /// Total size with quantization (bytes)
    pub quantized_size_bytes: usize,

    /// Overall compression ratio
    pub overall_compression_ratio: f32,

    /// Per-level storage breakdown
    pub per_level_savings: HashMap<String, usize>, // level -> bytes
}

/// Estimated performance impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Estimated search speedup with quantization
    pub estimated_search_speedup: HashMap<String, f32>, // level -> speedup_factor

    /// Memory footprint change
    pub memory_footprint_change_percent: f32,

    /// I/O reduction for search operations
    pub io_reduction_percent: f32,
}

impl UnifiedQuantizedFile {
    /// Create new unified quantized file
    pub fn new(file_id: String, collection_id: String, config: QuantizationFileConfig) -> Self {
        Self {
            file_id,
            collection_id,
            quantization_timestamp: chrono::Utc::now().timestamp(),
            quantization_config: config,
            quantized_data: QuantizedVectorData::empty(),
            quantization_stats: CommonQuantizationStats::default(),
        }
    }

    /// Check if file has quantization for specific storage format
    pub fn has_quantization_level(&self, level: &StorageQuantizationFormat) -> bool {
        self.quantization_config.enabled_levels.contains(level)
    }

    /// Get quantized vectors for specific storage format
    pub fn get_quantized_vectors(&self, level: &StorageQuantizationFormat) -> Option<&[Vec<u8>]> {
        match level {
            StorageQuantizationFormat::BinaryFormat => self.quantized_data.q_binary.as_deref(),
            StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                ScalarQuantizationBits::Int4 => None, // Deferred: implement conversion
                ScalarQuantizationBits::Int8 => None, // Convert i8 to u8 for unified interface
                ScalarQuantizationBits::UInt8 => None, // Deferred: implement conversion
            },
            StorageQuantizationFormat::ProductFormat(bits) => match bits {
                ProductQuantizationBits::PQ4 => self.quantized_data.q_pq4.as_deref(),
                ProductQuantizationBits::PQ8 => self.quantized_data.q_pq8.as_deref(),
                ProductQuantizationBits::PQ16 => self.quantized_data.q_pq16.as_deref(),
                ProductQuantizationBits::PQ32 => self.quantized_data.q_pq32.as_deref(),
            },
        }
    }

    /// Get compression ratio for file
    pub fn overall_compression_ratio(&self) -> f32 {
        self.quantization_stats
            .storage_savings
            .overall_compression_ratio
    }

    /// Estimate search performance improvement
    pub fn estimated_search_improvement(&self, level: &StorageQuantizationFormat) -> f32 {
        self.quantization_stats
            .performance_impact
            .estimated_search_speedup
            .get(&format!("{:?}", level))
            .copied()
            .unwrap_or(1.0)
    }
}

impl QuantizedVectorData {
    /// Create empty quantized data structure
    pub fn empty() -> Self {
        Self {
            vector_count: 0,
            dimension: 0,
            q_binary: None,
            q_int8: None,
            q_pq4: None,
            q_pq8: None,
            q_pq16: None,
            q_pq32: None,
            qp_binary_threshold: None,
            qp_int8_min: None,
            qp_int8_max: None,
            qp_int8_scale: None,
            qp_pq_subquantizers: None,
            qp_pq_centroids: None,
        }
    }

    /// Check if any quantization data exists
    pub fn has_any_quantization(&self) -> bool {
        self.q_binary.is_some()
            || self.q_int8.is_some()
            || self.q_pq4.is_some()
            || self.q_pq8.is_some()
            || self.q_pq16.is_some()
            || self.q_pq32.is_some()
    }

    /// Get memory usage for quantized data
    pub fn memory_usage(&self) -> usize {
        let binary_size = self
            .q_binary
            .as_ref()
            .map_or(0, |v| v.iter().map(|vec| vec.len()).sum::<usize>());

        let int8_size = self
            .q_int8
            .as_ref()
            .map_or(0, |v| v.iter().map(|vec| vec.len()).sum::<usize>());

        let pq_size = [&self.q_pq4, &self.q_pq8, &self.q_pq16, &self.q_pq32]
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|v| v.iter().map(|vec| vec.len()).sum::<usize>())
            .sum::<usize>();

        binary_size + int8_size + pq_size
    }
}

impl Default for CommonQuantizationStats {
    fn default() -> Self {
        Self {
            quantization_time_ms: 0,
            peak_memory_usage_bytes: 0,
            storage_savings: StorageSavings {
                original_size_bytes: 0,
                quantized_size_bytes: 0,
                overall_compression_ratio: 1.0,
                per_level_savings: HashMap::new(),
            },
            performance_impact: PerformanceImpact {
                estimated_search_speedup: HashMap::new(),
                memory_footprint_change_percent: 0.0,
                io_reduction_percent: 0.0,
            },
        }
    }
}

/// Factory functions for engine-specific configurations
impl EngineQuantizationConfig {
    /// Create ProximaBlock configuration for SST/SWIFT/HELIX
    pub fn proxima_block_inline() -> Self {
        Self::ProximaBlock {
            storage_strategy: ProximaBlockQuantizationStorage::Inline,
            quantized_block_size_kb: 64, // 64KB blocks
        }
    }

    /// Create Parquet configuration for NOVA
    pub fn parquet_separate_columns() -> Self {
        Self::Parquet {
            separate_columns: true,
            quantized_compression: "lz4".to_string(),
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_quantization_types::{
        QuantizationConfig, QuantizationLevel as ApiQuantizationLevel, QuantizationType,
    };

    #[test]
    fn test_unified_quantized_file_creation() {
        let config = QuantizationFileConfig {
            enabled_levels: vec![
                StorageQuantizationFormat::BinaryFormat,
                StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8),
            ],
            quantization_trigger: QuantizationTrigger::Flush,
            memory_budget_mb: 100,
            batch_size: 1000,
            engine_config: EngineQuantizationConfig::proxima_block_inline(),
        };

        let quantized_file = UnifiedQuantizedFile::new(
            "test_file_001".to_string(),
            "test_collection".to_string(),
            config,
        );

        assert_eq!(quantized_file.file_id, "test_file_001");
        assert_eq!(quantized_file.collection_id, "test_collection");
        assert!(quantized_file.has_quantization_level(&StorageQuantizationFormat::BinaryFormat));
        assert!(
            quantized_file.has_quantization_level(&StorageQuantizationFormat::ScalarFormat(
                ScalarQuantizationBits::Int8
            ))
        );
        assert!(
            !quantized_file.has_quantization_level(&StorageQuantizationFormat::ProductFormat(
                ProductQuantizationBits::PQ8
            ))
        );
    }

    // ========================================================================
    // Conversion Layer Tests
    // ========================================================================

    #[test]
    fn test_api_to_storage_conversion_int8() {
        let api_level = ApiQuantizationLevel::Int8;
        let storage_format = StorageQuantizationFormat::from(api_level);

        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
        );
    }

    #[test]
    fn test_api_to_storage_conversion_int4() {
        let api_level = ApiQuantizationLevel::Int4;
        let storage_format = StorageQuantizationFormat::from(api_level);

        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int4)
        );
    }

    #[test]
    fn test_api_to_storage_conversion_uint8() {
        let api_level = ApiQuantizationLevel::UInt8;
        let storage_format = StorageQuantizationFormat::from(api_level);

        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::UInt8)
        );
    }

    #[test]
    fn test_api_to_storage_conversion_none() {
        let api_level = ApiQuantizationLevel::None;
        let storage_format = StorageQuantizationFormat::from(api_level);

        // No quantization → Binary format for fast filtering
        assert_eq!(storage_format, StorageQuantizationFormat::BinaryFormat);
    }

    #[test]
    fn test_api_to_storage_conversion_fp32() {
        let api_level = ApiQuantizationLevel::FP32;
        let storage_format = StorageQuantizationFormat::from(api_level);

        // FP32 → Binary format for fast candidate filtering
        assert_eq!(storage_format, StorageQuantizationFormat::BinaryFormat);
    }

    #[test]
    fn test_api_to_storage_conversion_fp16() {
        let api_level = ApiQuantizationLevel::FP16;
        let storage_format = StorageQuantizationFormat::from(api_level);

        // FP16 → Approximated as Int8 scalar
        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
        );
    }

    #[test]
    fn test_storage_to_api_conversion_binary() {
        let storage_format = StorageQuantizationFormat::BinaryFormat;
        let api_level = ApiQuantizationLevel::from(storage_format);

        assert_eq!(api_level, ApiQuantizationLevel::None);
    }

    #[test]
    fn test_storage_to_api_conversion_scalar_int8() {
        let storage_format = StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8);
        let api_level = ApiQuantizationLevel::from(storage_format);

        assert_eq!(api_level, ApiQuantizationLevel::Int8);
    }

    #[test]
    fn test_storage_to_api_conversion_product_pq8() {
        let storage_format = StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8);
        let api_level = ApiQuantizationLevel::from(storage_format);

        // Product quantization approximated as Int8 precision
        assert_eq!(api_level, ApiQuantizationLevel::Int8);
    }

    #[test]
    fn test_quantization_type_to_storage_conversion_scalar() {
        let qtype = QuantizationType::Scalar;
        let storage_format = StorageQuantizationFormat::from(qtype);

        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8)
        );
    }

    #[test]
    fn test_quantization_type_to_storage_conversion_binary() {
        let qtype = QuantizationType::Binary;
        let storage_format = StorageQuantizationFormat::from(qtype);

        assert_eq!(storage_format, StorageQuantizationFormat::BinaryFormat);
    }

    #[test]
    fn test_quantization_type_to_storage_conversion_product() {
        let qtype = QuantizationType::Product;
        let storage_format = StorageQuantizationFormat::from(qtype);

        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
        );
    }

    #[test]
    fn test_recommend_storage_format_from_config() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Scalar,
            quantization_level: ApiQuantizationLevel::Int4,
            asymmetric: false,
            cache: true,
        };

        let storage_format = recommend_storage_format(&config);

        // Should prefer quantization level over type
        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int4)
        );
    }

    #[test]
    fn test_recommend_storage_format_fallback_to_type() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::Product,
            quantization_level: ApiQuantizationLevel::None, // No level specified
            asymmetric: false,
            cache: true,
        };

        let storage_format = recommend_storage_format(&config);

        // Should fall back to quantization type
        assert_eq!(
            storage_format,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
        );
    }

    #[test]
    fn test_is_compatible_binary_format() {
        let binary_format = StorageQuantizationFormat::BinaryFormat;

        // Binary format compatible with all API levels
        assert!(is_compatible(ApiQuantizationLevel::None, &binary_format));
        assert!(is_compatible(ApiQuantizationLevel::Int4, &binary_format));
        assert!(is_compatible(ApiQuantizationLevel::Int8, &binary_format));
        assert!(is_compatible(ApiQuantizationLevel::FP32, &binary_format));
    }

    #[test]
    fn test_is_compatible_scalar_int4() {
        let int4_format = StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int4);

        // Int4 only compatible with Int4 precision.
        assert!(!is_compatible(ApiQuantizationLevel::None, &int4_format));
        assert!(is_compatible(ApiQuantizationLevel::Int4, &int4_format));
        assert!(!is_compatible(ApiQuantizationLevel::Int8, &int4_format));
        assert!(!is_compatible(ApiQuantizationLevel::FP32, &int4_format));
    }

    #[test]
    fn test_is_compatible_scalar_int8() {
        let int8_format = StorageQuantizationFormat::ScalarFormat(ScalarQuantizationBits::Int8);

        // Int8 compatible with any <=8-bit scalar precision or floating point.
        assert!(!is_compatible(ApiQuantizationLevel::None, &int8_format));
        assert!(is_compatible(ApiQuantizationLevel::Int4, &int8_format));
        assert!(is_compatible(ApiQuantizationLevel::Int8, &int8_format));
        assert!(is_compatible(ApiQuantizationLevel::FP16, &int8_format));
        assert!(is_compatible(ApiQuantizationLevel::FP32, &int8_format));
        assert!(is_compatible(ApiQuantizationLevel::UInt8, &int8_format));
    }

    #[test]
    fn test_is_compatible_product_format() {
        let pq8_format = StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8);

        // Product format compatible with all API levels
        assert!(is_compatible(ApiQuantizationLevel::None, &pq8_format));
        assert!(is_compatible(ApiQuantizationLevel::Int4, &pq8_format));
        assert!(is_compatible(ApiQuantizationLevel::Int8, &pq8_format));
        assert!(is_compatible(ApiQuantizationLevel::FP32, &pq8_format));
    }

    #[test]
    fn test_roundtrip_conversion_int8() {
        let api_level = ApiQuantizationLevel::Int8;
        let storage_format = StorageQuantizationFormat::from(api_level);
        let api_level_reconstructed = ApiQuantizationLevel::from(storage_format);

        // Roundtrip should preserve the type
        assert_eq!(api_level, api_level_reconstructed);
    }

    #[test]
    fn test_roundtrip_conversion_int4() {
        let api_level = ApiQuantizationLevel::Int4;
        let storage_format = StorageQuantizationFormat::from(api_level);
        let api_level_reconstructed = ApiQuantizationLevel::from(storage_format);

        // Roundtrip should preserve the type
        assert_eq!(api_level, api_level_reconstructed);
    }

    #[test]
    fn test_roundtrip_conversion_binary() {
        let api_level = ApiQuantizationLevel::None;
        let storage_format = StorageQuantizationFormat::from(api_level);
        let api_level_reconstructed = ApiQuantizationLevel::from(storage_format);

        // Roundtrip should preserve the type
        assert_eq!(api_level, api_level_reconstructed);
    }

    #[test]
    fn test_quantized_vector_data_memory_usage() {
        let mut data = QuantizedVectorData::empty();
        data.q_binary = Some(vec![vec![0u8; 32], vec![1u8; 32]]); // 2 vectors, 32 bytes each

        assert_eq!(data.memory_usage(), 64);
        assert!(data.has_any_quantization());
    }

    #[test]
    fn test_engine_specific_configs() {
        let proxima_config = EngineQuantizationConfig::proxima_block_inline();
        match proxima_config {
            EngineQuantizationConfig::ProximaBlock {
                storage_strategy, ..
            } => {
                assert!(matches!(
                    storage_strategy,
                    ProximaBlockQuantizationStorage::Inline
                ));
            }
            _ => panic!("Expected ProximaBlock configuration"),
        }

        let parquet_config = EngineQuantizationConfig::parquet_separate_columns();
        match parquet_config {
            EngineQuantizationConfig::Parquet {
                separate_columns, ..
            } => {
                assert!(separate_columns);
            }
            _ => panic!("Expected Parquet configuration"),
        }
    }
}
