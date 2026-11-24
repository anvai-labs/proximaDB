//! Quantized Schema Builder for Both ProximaBlocks and Parquet
//!
//! This module provides unified schema building for quantized vector storage
//! across both ProximaBlock-based engines (SST, SWIFT, HELIX) and Parquet-based
//! engines (NOVA).
//!
//! Key Design:
//! - Schema is built once and reused across all files in a collection
//! - Optional fields for each quantization level avoid runtime overhead
//! - Same logical schema maps to different physical storage formats

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::storage::engines::core::formats::columnar::constants::*;
use crate::storage::engines::core::formats::common_quantization::ProductQuantizationBits;
use crate::storage::engines::core::formats::common_quantization::{
    QuantizationFileConfig, ScalarQuantizationBits, StorageQuantizationFormat,
};

/// Schema definition for quantized vector storage
///
/// This schema is built once per collection and defines which quantization
/// levels are available. It maps to:
/// - ProximaBlock: Optional fields in block structure
/// - Parquet: Optional columns in Arrow schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVectorSchema {
    /// Collection this schema belongs to
    pub collection_id: String,

    /// Schema version (for evolution)
    pub schema_version: u32,

    /// Enabled quantization storage formats
    pub enabled_levels: Vec<StorageQuantizationFormat>,

    /// Vector dimension (constant across collection)
    pub dimension: usize,

    /// Physical storage mapping
    pub storage_mapping: StorageMapping,

    /// Field definitions for each quantization level
    pub field_definitions: Vec<QuantizedFieldDefinition>,

    /// Schema creation timestamp
    pub created_at: i64,
}

/// Mapping between logical schema and physical storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageMapping {
    /// ProximaBlock storage mapping
    ProximaBlock {
        /// Fields stored in main data section
        main_section_fields: Vec<String>,
        /// Fields stored in quantized section
        quantized_section_fields: Vec<String>,
        /// Codebook storage strategy
        codebook_storage: CodebookStorageStrategy,
    },
    /// Parquet storage mapping
    Parquet {
        /// Column names for each quantization level
        column_mapping: HashMap<String, String>,
        /// Compression settings per column type
        compression_mapping: HashMap<String, String>,
        /// Row group organization
        row_group_strategy: RowGroupStrategy,
    },
}

/// Strategy for storing codebooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodebookStorageStrategy {
    /// Store codebooks in metadata
    InMetadata,
    /// Store codebooks as separate fields
    SeparateFields,
    /// Store codebooks in external file
    ExternalFile { path_template: String },
}

/// Row group organization for Parquet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RowGroupStrategy {
    /// All quantization levels in same row groups
    Unified { target_row_group_size: usize },
    /// Separate row groups for each quantization level
    Separated {
        size_per_level: HashMap<String, usize>,
    },
}

/// Definition of a quantized field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedFieldDefinition {
    /// Quantization level
    pub level: StorageQuantizationFormat,

    /// Logical field name
    pub field_name: String,

    /// Physical field specifications
    pub physical_spec: PhysicalFieldSpec,

    /// Whether this field is required or optional
    pub required: bool,

    /// Expected data characteristics
    pub data_characteristics: DataCharacteristics,
}

/// Physical field specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicalFieldSpec {
    /// Binary quantization field
    Binary {
        /// Bits per dimension (always 1 for binary)
        bits_per_dimension: u8,
        /// Bit packing strategy
        packing_strategy: BitPackingStrategy,
    },
    /// INT8 quantization field
    Int8 {
        /// Signed or unsigned
        signed: bool,
        /// Scale factor encoding
        scale_encoding: ScaleEncoding,
    },
    /// Product Quantization field
    ProductQuantization {
        /// Bits per code (4, 8, 16, or 32)
        bits_per_code: u8,
        /// Number of subquantizers
        num_subquantizers: usize,
        /// Code packing for sub-byte codes
        code_packing: Option<CodePackingStrategy>,
    },
    /// Codebook field (for PQ methods)
    Codebook {
        /// Associated PQ field
        pq_field_name: String,
        /// Codebook dimensions
        dimensions: CodebookDimensions,
    },
    /// Quantization parameters field
    Parameters {
        /// Parameter type
        param_type: ParameterType,
        /// Storage format
        storage_format: ParameterStorageFormat,
    },
}

/// Bit packing strategy for binary quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BitPackingStrategy {
    /// Pack bits sequentially (standard approach)
    Sequential,
    /// Pack bits for SIMD alignment
    SimdAligned { alignment: usize },
}

/// Scale factor encoding for INT8
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleEncoding {
    /// Global min/max for entire collection
    Global { min: f32, max: f32 },
    /// Per-file min/max
    PerFile,
    /// Per-block min/max
    PerBlock,
}

/// Code packing for sub-byte PQ codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodePackingStrategy {
    /// 4-bit codes: 2 codes per byte
    FourBit { codes_per_byte: u8 },
}

/// Codebook dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookDimensions {
    /// Number of subquantizers
    pub num_subquantizers: usize,
    /// Number of centroids per subquantizer
    pub num_centroids: usize,
    /// Dimension of each centroid
    pub centroid_dimension: usize,
}

/// Parameter types for quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    BinaryThreshold,
    Int8MinMax,
    Int8ScaleFactor,
    PqSubquantizerCount,
    PqCentroidCount,
}

/// Storage format for parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterStorageFormat {
    /// Store as metadata
    Metadata,
    /// Store as dedicated field
    DedicatedField,
    /// Store inline with data
    Inline,
}

/// Expected data characteristics for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Expected compression ratio
    pub expected_compression_ratio: f32,
    /// Expected memory usage per vector (bytes)
    pub expected_memory_per_vector: usize,
    /// Expected search performance improvement
    pub expected_search_speedup: f32,
    /// Sparsity characteristics
    pub sparsity_info: Option<SparsityInfo>,
}

/// Sparsity information for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityInfo {
    /// Expected percentage of zero values
    pub zero_percentage: f32,
    /// Whether to use sparse encoding
    pub use_sparse_encoding: bool,
}

/// Schema builder for quantized vector storage
pub struct QuantizedVectorSchemaBuilder {
    collection_id: String,
    dimension: usize,
    enabled_levels: Vec<StorageQuantizationFormat>,
    storage_type: SchemaStorageType,
}

/// Target storage type for schema
#[derive(Debug, Clone)]
pub enum SchemaStorageType {
    ProximaBlock,
    Parquet,
    Both,
}

impl QuantizedVectorSchemaBuilder {
    /// Create new schema builder
    pub fn new(collection_id: String, dimension: usize, storage_type: SchemaStorageType) -> Self {
        Self {
            collection_id,
            dimension,
            enabled_levels: Vec::new(),
            storage_type,
        }
    }

    /// Add quantization level to schema
    pub fn add_quantization_level(mut self, level: StorageQuantizationFormat) -> Self {
        if !self.enabled_levels.contains(&level) {
            self.enabled_levels.push(level);
        }
        self
    }

    /// Add multiple quantization levels
    pub fn add_quantization_levels(mut self, levels: Vec<StorageQuantizationFormat>) -> Self {
        for level in levels {
            self = self.add_quantization_level(level);
        }
        self
    }

    /// Build schema from configuration
    pub fn from_config(
        collection_id: String,
        dimension: usize,
        config: &QuantizationFileConfig,
        storage_type: SchemaStorageType,
    ) -> Self {
        Self::new(collection_id, dimension, storage_type)
            .add_quantization_levels(config.enabled_levels.clone())
    }

    /// Build the schema
    pub fn build(self) -> Result<QuantizedVectorSchema> {
        let mut field_definitions = Vec::new();

        // Build field definitions for each enabled level
        for level in &self.enabled_levels {
            let field_def = self.build_field_definition(level)?;
            field_definitions.push(field_def);

            // Codebooks are now stored as file-level metadata, not per-row columns
            // Skip adding codebook fields - they will be handled by CodebookSerializer
            // if self.requires_codebook(level) {
            //     // Codebooks moved to file-level metadata
            // }

            // Add parameter fields if needed
            let param_defs = self.build_parameter_definitions(level)?;
            field_definitions.extend(param_defs);
        }

        let storage_mapping = self.build_storage_mapping()?;

        Ok(QuantizedVectorSchema {
            collection_id: self.collection_id,
            schema_version: 1,
            enabled_levels: self.enabled_levels,
            dimension: self.dimension,
            storage_mapping,
            field_definitions,
            created_at: chrono::Utc::now().timestamp(),
        })
    }

    /// Build field definition for quantization level using columnar constants
    fn build_field_definition(
        &self,
        level: &StorageQuantizationFormat,
    ) -> Result<QuantizedFieldDefinition> {
        // Use constants from columnar module for consistent naming
        let field_name = match level {
            StorageQuantizationFormat::BinaryFormat => FIELD_Q_BINARY.to_string(),
            StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                ScalarQuantizationBits::Int4 => FIELD_Q_INT8.to_string(), // TODO: Add FIELD_Q_INT4 constant
                ScalarQuantizationBits::Int8 => FIELD_Q_INT8.to_string(),
                ScalarQuantizationBits::UInt8 => FIELD_Q_INT8.to_string(), // TODO: Add FIELD_Q_UINT8 constant
            },
            StorageQuantizationFormat::ProductFormat(bits) => match bits {
                ProductQuantizationBits::PQ4 => FIELD_Q_PQ4.to_string(),
                ProductQuantizationBits::PQ8 => FIELD_Q_PQ8.to_string(),
                ProductQuantizationBits::PQ16 => FIELD_Q_PQ16.to_string(),
                ProductQuantizationBits::PQ32 => FIELD_Q_PQ32.to_string(),
            },
        };

        let physical_spec = match level {
            StorageQuantizationFormat::BinaryFormat => PhysicalFieldSpec::Binary {
                bits_per_dimension: 1,
                packing_strategy: BitPackingStrategy::Sequential,
            },
            StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                ScalarQuantizationBits::Int4 => PhysicalFieldSpec::Int8 {
                    signed: true,
                    scale_encoding: ScaleEncoding::PerFile,
                }, // TODO: Add Int4 support
                ScalarQuantizationBits::Int8 => PhysicalFieldSpec::Int8 {
                    signed: true,
                    scale_encoding: ScaleEncoding::PerFile,
                },
                ScalarQuantizationBits::UInt8 => PhysicalFieldSpec::Int8 {
                    signed: false,
                    scale_encoding: ScaleEncoding::PerFile,
                },
            },
            StorageQuantizationFormat::ProductFormat(bits) => match bits {
                ProductQuantizationBits::PQ4 => PhysicalFieldSpec::ProductQuantization {
                    bits_per_code: 4,
                    num_subquantizers: self.calculate_subquantizers(4)?,
                    code_packing: Some(CodePackingStrategy::FourBit { codes_per_byte: 2 }),
                },
                ProductQuantizationBits::PQ8 => PhysicalFieldSpec::ProductQuantization {
                    bits_per_code: 8,
                    num_subquantizers: self.calculate_subquantizers(8)?,
                    code_packing: None,
                },
                ProductQuantizationBits::PQ16 => PhysicalFieldSpec::ProductQuantization {
                    bits_per_code: 16,
                    num_subquantizers: self.calculate_subquantizers(16)?,
                    code_packing: None,
                },
                ProductQuantizationBits::PQ32 => PhysicalFieldSpec::ProductQuantization {
                    bits_per_code: 32,
                    num_subquantizers: self.calculate_subquantizers(32)?,
                    code_packing: None,
                },
            },
        };

        let data_characteristics = self.estimate_data_characteristics(level);

        Ok(QuantizedFieldDefinition {
            level: level.clone(),
            field_name,
            physical_spec,
            required: false, // All quantized fields are optional
            data_characteristics,
        })
    }

    /// Check if quantization storage format requires codebook
    #[allow(dead_code)]
    fn requires_codebook(&self, level: &StorageQuantizationFormat) -> bool {
        matches!(
            level,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4)
                | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
                | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16)
                | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32)
        )
    }

    /// Build codebook field definition (NOT USED - codebooks are file-level metadata)
    /// This method is kept for backward compatibility but returns an error
    #[allow(dead_code)]
    fn build_codebook_definition(
        &self,
        level: &StorageQuantizationFormat,
    ) -> Result<QuantizedFieldDefinition> {
        // Codebooks are now stored as file-level metadata, not per-row columns
        // For ProximaBlock engines: stored in footer
        // For Parquet engines: stored as sidecar files
        Err(anyhow::anyhow!(
            "Codebooks are now stored as file-level metadata, not per-row columns. \
             Use CodebookSerializer from codebook_metadata module instead. \
             Level: {:?}",
            level
        ))
    }

    /// Build parameter field definitions using columnar constants
    fn build_parameter_definitions(
        &self,
        level: &StorageQuantizationFormat,
    ) -> Result<Vec<QuantizedFieldDefinition>> {
        let mut param_defs = Vec::new();

        // Use constants for consistent parameter column naming
        match level {
            StorageQuantizationFormat::BinaryFormat => {
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_BINARY_THRESHOLD,
                    ParameterType::BinaryThreshold,
                )?);
            }
            StorageQuantizationFormat::ScalarFormat(
                ScalarQuantizationBits::Int4
                | ScalarQuantizationBits::Int8
                | ScalarQuantizationBits::UInt8,
            ) => {
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_INT8_MIN,
                    ParameterType::Int8MinMax,
                )?);
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_INT8_MAX,
                    ParameterType::Int8MinMax,
                )?);
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_INT8_SCALE,
                    ParameterType::Int8ScaleFactor,
                )?);
            }
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4)
            | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
            | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16)
            | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) => {
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_PQ_SUBQUANTIZERS,
                    ParameterType::PqSubquantizerCount,
                )?);
                param_defs.push(self.build_parameter_field(
                    level,
                    FIELD_QP_PQ_CENTROIDS,
                    ParameterType::PqCentroidCount,
                )?);
            }
        }

        Ok(param_defs)
    }

    /// Build single parameter field
    fn build_parameter_field(
        &self,
        level: &StorageQuantizationFormat,
        name: &str,
        param_type: ParameterType,
    ) -> Result<QuantizedFieldDefinition> {
        Ok(QuantizedFieldDefinition {
            level: level.clone(),
            field_name: name.to_string(),
            physical_spec: PhysicalFieldSpec::Parameters {
                param_type,
                storage_format: ParameterStorageFormat::Metadata, // Store in metadata by default
            },
            required: false,
            data_characteristics: DataCharacteristics {
                expected_compression_ratio: 1.0,
                expected_memory_per_vector: 0,
                expected_search_speedup: 1.0,
                sparsity_info: None,
            },
        })
    }

    /// Calculate number of subquantizers for PQ
    #[allow(dead_code)]
    fn calculate_subquantizers(&self, bits_per_code: u8) -> Result<usize> {
        // Standard PQ: aim for 8-32 dimensions per subquantizer
        let target_dims_per_subq = match bits_per_code {
            4 => 16,  // 4-bit: smaller subquantizers for better quality
            8 => 32,  // 8-bit: standard subquantizer size
            16 => 32, // 16-bit: can handle larger subquantizers
            32 => 64, // 32-bit: very large subquantizers
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported bits per code: {}",
                    bits_per_code
                ));
            }
        };

        let num_subquantizers = (self.dimension.div_ceil(target_dims_per_subq)).max(1);
        Ok(num_subquantizers)
    }

    /// Get bits per code for quantization level
    #[allow(dead_code)]
    fn get_bits_per_code(&self, level: &StorageQuantizationFormat) -> u8 {
        match level {
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4) => 4,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8) => 8,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16) => 16,
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) => 32,
            _ => 8, // Default
        }
    }

    /// Estimate data characteristics for quantization level
    fn estimate_data_characteristics(
        &self,
        level: &StorageQuantizationFormat,
    ) -> DataCharacteristics {
        match level {
            StorageQuantizationFormat::BinaryFormat => DataCharacteristics {
                expected_compression_ratio: 32.0, // 32x compression vs FP32
                expected_memory_per_vector: self.dimension.div_ceil(8), // Bit-packed
                expected_search_speedup: 15.0,
                sparsity_info: Some(SparsityInfo {
                    zero_percentage: 50.0,      // Binary typically has high sparsity
                    use_sparse_encoding: false, // Bit-packing already efficient
                }),
            },
            StorageQuantizationFormat::ScalarFormat(bits) => match bits {
                ScalarQuantizationBits::Int4 => DataCharacteristics {
                    expected_compression_ratio: 8.0, // 8x compression vs FP32
                    expected_memory_per_vector: self.dimension.div_ceil(2),
                    expected_search_speedup: 10.0,
                    sparsity_info: None,
                },
                ScalarQuantizationBits::Int8 | ScalarQuantizationBits::UInt8 => {
                    DataCharacteristics {
                        expected_compression_ratio: 4.0, // 4x compression vs FP32
                        expected_memory_per_vector: self.dimension,
                        expected_search_speedup: 8.0,
                        sparsity_info: None,
                    }
                }
            },
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8) => {
                DataCharacteristics {
                    expected_compression_ratio: 4.0, // Depends on subquantizers
                    expected_memory_per_vector: self.calculate_subquantizers(8).unwrap_or(8),
                    expected_search_speedup: 5.0,
                    sparsity_info: None,
                }
            }
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4) => {
                DataCharacteristics {
                    expected_compression_ratio: 8.0, // Better compression with 4-bit codes
                    expected_memory_per_vector: self.calculate_subquantizers(4).unwrap_or(4) / 2,
                    expected_search_speedup: 6.0,
                    sparsity_info: None,
                }
            }
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16) => {
                DataCharacteristics {
                    expected_compression_ratio: 2.0, // Less compression with 16-bit codes
                    expected_memory_per_vector: self.calculate_subquantizers(16).unwrap_or(16) * 2,
                    expected_search_speedup: 3.0,
                    sparsity_info: None,
                }
            }
            StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) => {
                DataCharacteristics {
                    expected_compression_ratio: 1.0, // No compression gain with 32-bit codes
                    expected_memory_per_vector: self.calculate_subquantizers(32).unwrap_or(32) * 4,
                    expected_search_speedup: 1.5,
                    sparsity_info: None,
                }
            }
        }
    }

    /// Build storage mapping with consistent column names across ProximaBlock and Parquet
    fn build_storage_mapping(&self) -> Result<StorageMapping> {
        match self.storage_type {
            SchemaStorageType::ProximaBlock => {
                // ProximaBlock fields - use constants for consistency
                let main_fields = vec![FIELD_VECTOR_FP32.to_string()];

                let mut quantized_fields = Vec::new();
                for level in &self.enabled_levels {
                    let field_name = match level {
                        StorageQuantizationFormat::BinaryFormat => FIELD_Q_BINARY.to_string(),
                        StorageQuantizationFormat::ScalarFormat(
                            ScalarQuantizationBits::Int4
                            | ScalarQuantizationBits::Int8
                            | ScalarQuantizationBits::UInt8,
                        ) => FIELD_Q_INT8.to_string(),
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4) => {
                            FIELD_Q_PQ4.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8) => {
                            FIELD_Q_PQ8.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16) => {
                            FIELD_Q_PQ16.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) => {
                            FIELD_Q_PQ32.to_string()
                        }
                    };
                    quantized_fields.push(field_name);
                }

                Ok(StorageMapping::ProximaBlock {
                    main_section_fields: main_fields,
                    quantized_section_fields: quantized_fields,
                    codebook_storage: CodebookStorageStrategy::SeparateFields, // Use separate columns
                })
            }
            SchemaStorageType::Parquet => {
                // Parquet columns - use constants for identical names to ProximaBlock
                let mut column_mapping = HashMap::new();

                // Main vector column using constant
                column_mapping.insert(FIELD_VECTOR_FP32.to_string(), FIELD_VECTOR_FP32.to_string());

                // Quantized columns - use constants for same names as ProximaBlock
                for level in &self.enabled_levels {
                    let column_name = match level {
                        StorageQuantizationFormat::BinaryFormat => FIELD_Q_BINARY.to_string(),
                        StorageQuantizationFormat::ScalarFormat(
                            ScalarQuantizationBits::Int4
                            | ScalarQuantizationBits::Int8
                            | ScalarQuantizationBits::UInt8,
                        ) => FIELD_Q_INT8.to_string(),
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4) => {
                            FIELD_Q_PQ4.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8) => {
                            FIELD_Q_PQ8.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16) => {
                            FIELD_Q_PQ16.to_string()
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) => {
                            FIELD_Q_PQ32.to_string()
                        }
                    };
                    // Logical name = Physical name for consistency
                    column_mapping.insert(column_name.clone(), column_name);
                }

                // Codebooks are now stored as file-level metadata, not columns
                // Skip codebook column mapping - handled by CodebookSerializer

                // Parameter columns - use constants
                for level in &self.enabled_levels {
                    let param_columns = match level {
                        StorageQuantizationFormat::BinaryFormat => vec![FIELD_QP_BINARY_THRESHOLD],
                        StorageQuantizationFormat::ScalarFormat(
                            ScalarQuantizationBits::Int4
                            | ScalarQuantizationBits::Int8
                            | ScalarQuantizationBits::UInt8,
                        ) => {
                            vec![FIELD_QP_INT8_MIN, FIELD_QP_INT8_MAX, FIELD_QP_INT8_SCALE]
                        }
                        StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ4)
                        | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ8)
                        | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ16)
                        | StorageQuantizationFormat::ProductFormat(ProductQuantizationBits::PQ32) =>
                        {
                            vec![FIELD_QP_PQ_SUBQUANTIZERS, FIELD_QP_PQ_CENTROIDS]
                        }
                    };

                    for param_col in param_columns {
                        column_mapping.insert(param_col.to_string(), param_col.to_string());
                    }
                }

                // Compression mapping - optimized per column type using constants
                let mut compression_mapping = HashMap::new();
                compression_mapping.insert(FIELD_VECTOR_FP32.to_string(), "lz4".to_string());
                compression_mapping.insert(FIELD_Q_BINARY.to_string(), "snappy".to_string()); // Good for binary data
                compression_mapping.insert(FIELD_Q_INT8.to_string(), "lz4".to_string());
                compression_mapping.insert(FIELD_Q_PQ4.to_string(), "snappy".to_string());
                compression_mapping.insert(FIELD_Q_PQ8.to_string(), "snappy".to_string());
                compression_mapping.insert(FIELD_Q_PQ16.to_string(), "lz4".to_string());
                compression_mapping.insert(FIELD_Q_PQ32.to_string(), "lz4".to_string());

                Ok(StorageMapping::Parquet {
                    column_mapping,
                    compression_mapping,
                    row_group_strategy: RowGroupStrategy::Unified {
                        target_row_group_size: DEFAULT_ROW_GROUP_SIZE,
                    },
                })
            }
            SchemaStorageType::Both => {
                // Return ProximaBlock mapping as default, but this ensures consistency
                self.build_storage_mapping()
            }
        }
    }
}

impl QuantizedVectorSchema {
    /// Check if schema supports quantization level
    pub fn supports_level(&self, level: &StorageQuantizationFormat) -> bool {
        self.enabled_levels.contains(level)
    }

    /// Get field definition for quantization level
    pub fn get_field_definition(
        &self,
        level: &StorageQuantizationFormat,
    ) -> Option<&QuantizedFieldDefinition> {
        self.field_definitions
            .iter()
            .find(|def| &def.level == level)
    }

    /// Get all vector field definitions (excluding codebooks and parameters)
    pub fn get_vector_field_definitions(&self) -> Vec<&QuantizedFieldDefinition> {
        self.field_definitions
            .iter()
            .filter(|def| {
                matches!(
                    def.physical_spec,
                    PhysicalFieldSpec::Binary { .. }
                        | PhysicalFieldSpec::Int8 { .. }
                        | PhysicalFieldSpec::ProductQuantization { .. }
                )
            })
            .collect()
    }

    /// Get estimated storage savings
    pub fn estimated_storage_savings(&self) -> f32 {
        let total_compression: f32 = self
            .get_vector_field_definitions()
            .iter()
            .map(|def| def.data_characteristics.expected_compression_ratio)
            .sum();

        if self.enabled_levels.is_empty() {
            1.0
        } else {
            total_compression / self.enabled_levels.len() as f32
        }
    }

    /// Validate schema consistency
    pub fn validate(&self) -> Result<()> {
        // Check that all enabled levels have field definitions
        for level in &self.enabled_levels {
            if self.get_field_definition(level).is_none() {
                return Err(anyhow::anyhow!(
                    "Missing field definition for level: {:?}",
                    level
                ));
            }
        }

        // Check dimension consistency
        if self.dimension == 0 {
            return Err(anyhow::anyhow!("Invalid dimension: 0"));
        }

        // Validate PQ subquantizer calculations
        for def in self.get_vector_field_definitions() {
            if let PhysicalFieldSpec::ProductQuantization {
                num_subquantizers, ..
            } = &def.physical_spec
                && !self.dimension.is_multiple_of(*num_subquantizers)
            {
                tracing::warn!(
                    "Dimension {} not evenly divisible by subquantizers {} for {:?}",
                    self.dimension,
                    num_subquantizers,
                    def.level
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder_basic() -> Result<()> {
        let schema = QuantizedVectorSchemaBuilder::new(
            "test_collection".to_string(),
            128,
            SchemaStorageType::ProximaBlock,
        )
        .add_quantization_level(StorageQuantizationFormat::BinaryFormat)
        .add_quantization_level(StorageQuantizationFormat::ScalarFormat(
            ScalarQuantizationBits::Int8,
        ))
        .build()?;

        assert_eq!(schema.collection_id, "test_collection");
        assert_eq!(schema.dimension, 128);
        assert_eq!(schema.enabled_levels.len(), 2);
        assert!(schema.supports_level(&StorageQuantizationFormat::BinaryFormat));
        assert!(
            schema.supports_level(&StorageQuantizationFormat::ScalarFormat(
                ScalarQuantizationBits::Int8
            ))
        );
        assert!(
            !schema.supports_level(&StorageQuantizationFormat::ProductFormat(
                ProductQuantizationBits::PQ8
            ))
        );

        schema.validate()?;
        Ok(())
    }

    #[test]
    fn test_pq_subquantizer_calculation() -> Result<()> {
        let builder =
            QuantizedVectorSchemaBuilder::new("test".to_string(), 256, SchemaStorageType::Parquet);

        // 256 dimensions with 8-bit PQ should create 8 subquantizers (256/32 = 8)
        let subquantizers = builder.calculate_subquantizers(8)?;
        assert_eq!(subquantizers, 8);

        Ok(())
    }

    #[test]
    fn test_storage_mapping_parquet() -> Result<()> {
        let schema =
            QuantizedVectorSchemaBuilder::new("test".to_string(), 128, SchemaStorageType::Parquet)
                .add_quantization_level(StorageQuantizationFormat::BinaryFormat)
                .build()?;

        match &schema.storage_mapping {
            StorageMapping::Parquet { column_mapping, .. } => {
                assert!(column_mapping.contains_key(FIELD_VECTOR_FP32));
                assert!(column_mapping.contains_key(FIELD_Q_BINARY));
            }
            _ => panic!("Expected Parquet storage mapping"),
        }

        Ok(())
    }

    #[test]
    fn test_estimated_compression() -> Result<()> {
        let schema = QuantizedVectorSchemaBuilder::new(
            "test".to_string(),
            128,
            SchemaStorageType::ProximaBlock,
        )
        .add_quantization_level(StorageQuantizationFormat::BinaryFormat)
        .add_quantization_level(StorageQuantizationFormat::ScalarFormat(
            ScalarQuantizationBits::Int8,
        ))
        .build()?;

        let savings = schema.estimated_storage_savings();
        assert!(savings > 1.0); // Should have some compression
        assert!(savings < 50.0); // Should be reasonable

        Ok(())
    }
}
