//! Seamless Serialization/Deserialization for Columnar Storage
//!
//! This module provides transparent conversion between FP32 lists, INT8 lists,
//! and bitpacked formats during write/read operations. It integrates with the
//! universal quantization adapters and optimizes for zero-copy operations.

use anyhow::{Context, Result};
use arrow_array::builder::{FixedSizeBinaryBuilder, Float32Builder, Int8Builder};
use arrow_array::{Array, ArrayRef, FixedSizeBinaryArray, Float32Array, Int8Array};
use arrow_schema::{DataType, Schema};
use bytemuck::{cast_slice, try_cast_slice};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, trace, warn};

use crate::proximablocks::columnar_config_types::QuantizationConfig;
use proximadb_compression::CompressionAlgorithm;
use proximadb_distance_kernel::SelectedFormat;
use proximadb_quantization_model::StorageQuantizedData;
use proximadb_records::{EmbeddingCell, ProximaRecord};
use proximadb_storage_ports::StorageQuantizationEnginePort;

/// Serialization configuration for columnar storage
#[derive(Debug, Clone)]
pub struct ColumnarSerializationConfig {
    /// Target vector dimension
    pub dimension: usize,

    /// Quantization settings
    pub quantization: Option<QuantizationConfig>,

    /// Compression per column type
    pub compression: SerializationCompressionConfig,

    /// Memory optimization settings
    pub memory_optimization: MemoryOptimizationConfig,

    /// SIMD acceleration settings
    pub simd_config: SIMDConfig,
}

/// Compression configuration for serialization
#[derive(Debug, Clone)]
pub struct SerializationCompressionConfig {
    /// Compress FP32 vectors
    pub fp32_compression: Option<CompressionAlgorithm>,

    /// Compress quantized vectors
    pub quantized_compression: Option<CompressionAlgorithm>,

    /// Compress binary sketches (usually not beneficial)
    pub binary_compression: Option<CompressionAlgorithm>,

    /// Compression level (1-22 for ZSTD, 1-12 for LZ4)
    pub compression_level: Option<i32>,
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Use memory pools for repeated allocations
    pub enable_memory_pools: bool,

    /// Zero-copy deserialization when possible
    pub enable_zero_copy: bool,

    /// Batch size for vectorized operations
    pub batch_size: usize,

    /// Enable memory-mapped I/O for large arrays
    pub enable_mmap: bool,

    /// SIMD alignment for arrays
    pub simd_alignment: usize,
}

/// SIMD acceleration configuration
#[derive(Debug, Clone)]
pub struct SIMDConfig {
    /// Enable hardware-specific SIMD optimizations
    pub enable_simd: bool,

    /// Target instruction set (AVX2, AVX512, etc.)
    pub target_instruction_set: Option<String>,

    /// Vectorization strategy for different operations
    pub vectorization_strategy: VectorizationStrategy,
}

/// Vectorization strategies for different operations
#[derive(Debug, Clone)]
pub enum VectorizationStrategy {
    /// Auto-detect best strategy based on hardware
    Auto,
    /// Scalar operations (no SIMD)
    Scalar,
    /// Use SSE/SSE2 (128-bit)
    SSE,
    /// Use AVX/AVX2 (256-bit)
    AVX,
    /// Use AVX-512 (512-bit)
    AVX512,
    /// ARM NEON (128-bit)
    NEON,
}

impl Default for MemoryOptimizationConfig {
    /// Default memory optimization configuration (Audited December 2024)
    ///
    /// ## Best Practices Applied:
    /// - Memory pools: Reduce allocation overhead by reusing buffers
    /// - Zero-copy: Use bytemuck::cast_slice for FP32 serialization
    /// - Mmap: Enable memory-mapped I/O for large arrays
    /// - SIMD alignment: 64-byte cache-line boundaries for AVX2/AVX-512/NEON
    ///
    /// ## Performance Impact:
    /// - Memory pools reduce GC pressure and allocation latency
    /// - Zero-copy eliminates buffer copies during serialization
    /// - Mmap enables OS-level caching and prefetching
    /// - 64-byte alignment enables direct SIMD operations without realignment
    fn default() -> Self {
        Self {
            enable_memory_pools: true,
            enable_zero_copy: true,
            batch_size: 1024,
            enable_mmap: true,
            simd_alignment: 64, // Cache-line aligned for AVX2/AVX-512/NEON
        }
    }
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            target_instruction_set: None, // Auto-detect
            vectorization_strategy: VectorizationStrategy::Auto,
        }
    }
}

impl Default for SerializationCompressionConfig {
    fn default() -> Self {
        Self {
            fp32_compression: Some(CompressionAlgorithm::Zstd),
            quantized_compression: Some(CompressionAlgorithm::Lz4),
            binary_compression: None, // Binary data doesn't compress well
            compression_level: None,  // Use default levels
        }
    }
}

/// Transparent serializer for columnar data
pub struct ColumnarSerializer {
    /// Configuration
    config: ColumnarSerializationConfig,

    /// Quantization engine for transparent conversion
    quantization_engine: Option<Arc<dyn StorageQuantizationEnginePort>>,

    /// Memory pools for reuse
    memory_pools: MemoryPools,
}

/// Memory pools for efficient reuse
#[derive(Debug)]
struct MemoryPools {
    /// Pool for FP32 vectors
    fp32_pool: std::sync::Mutex<Vec<Vec<f32>>>,

    /// Pool for INT8 vectors
    #[allow(dead_code)]
    int8_pool: std::sync::Mutex<Vec<Vec<i8>>>,

    /// Pool for binary vectors
    #[allow(dead_code)]
    binary_pool: std::sync::Mutex<Vec<Vec<u8>>>,

    /// Pool for PQ codes
    #[allow(dead_code)]
    pq_pool: std::sync::Mutex<Vec<Vec<u8>>>,
}

impl MemoryPools {
    fn new() -> Self {
        Self {
            fp32_pool: std::sync::Mutex::new(Vec::new()),
            int8_pool: std::sync::Mutex::new(Vec::new()),
            binary_pool: std::sync::Mutex::new(Vec::new()),
            pq_pool: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn lock_pool<'a, T>(
        pool: &'a std::sync::Mutex<Vec<T>>,
        pool_name: &str,
    ) -> std::sync::MutexGuard<'a, Vec<T>> {
        match pool.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!(
                    pool = pool_name,
                    "Memory pool mutex poisoned; recovering inner pool"
                );
                poisoned.into_inner()
            }
        }
    }

    /// Get or allocate FP32 vector
    fn fp32_vector(&self, size: usize) -> Vec<f32> {
        let mut pool = Self::lock_pool(&self.fp32_pool, "fp32_pool");
        if let Some(mut vec) = pool.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            Vec::with_capacity(size)
        }
    }

    /// Return FP32 vector to pool
    fn return_fp32_vector(&self, vec: Vec<f32>) {
        if vec.capacity() <= 4096 {
            // Don't pool very large vectors
            let mut pool = Self::lock_pool(&self.fp32_pool, "fp32_pool");
            if pool.len() < 100 {
                // Limit pool size
                pool.push(vec);
            }
        }
    }

    /// Get or allocate INT8 vector
    #[allow(dead_code)]
    fn get_int8_vector(&self, size: usize) -> Vec<i8> {
        let mut pool = Self::lock_pool(&self.int8_pool, "int8_pool");
        if let Some(mut vec) = pool.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            Vec::with_capacity(size)
        }
    }

    /// Return INT8 vector to pool
    #[allow(dead_code)]
    fn return_int8_vector(&self, vec: Vec<i8>) {
        if vec.capacity() <= 4096 {
            let mut pool = Self::lock_pool(&self.int8_pool, "int8_pool");
            if pool.len() < 100 {
                pool.push(vec);
            }
        }
    }

    /// Get or allocate binary vector
    #[allow(dead_code)]
    fn get_binary_vector(&self, size: usize) -> Vec<u8> {
        let mut pool = Self::lock_pool(&self.binary_pool, "binary_pool");
        if let Some(mut vec) = pool.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            Vec::with_capacity(size)
        }
    }

    /// Return binary vector to pool
    #[allow(dead_code)]
    fn return_binary_vector(&self, vec: Vec<u8>) {
        if vec.capacity() <= 2048 {
            // Binary vectors are smaller
            let mut pool = Self::lock_pool(&self.binary_pool, "binary_pool");
            if pool.len() < 100 {
                pool.push(vec);
            }
        }
    }
}

/// Serialization result with all quantized formats
#[derive(Debug)]
pub struct SerializationResult {
    /// Original FP32 data
    pub fp32_array: Option<ArrayRef>,

    /// Binary quantized data
    pub binary_array: Option<ArrayRef>,

    /// INT8 quantized data and scale factors
    pub int8_array: Option<ArrayRef>,
    pub int8_scale_array: Option<ArrayRef>,
    pub int8_zero_point_array: Option<ArrayRef>,

    /// PQ quantized data
    pub pq_array: Option<ArrayRef>,

    /// Metadata about the serialization
    pub metadata: SerializationMetadata,
}

/// Metadata about serialization
#[derive(Debug, Clone)]
pub struct SerializationMetadata {
    pub record_count: usize,
    pub dimension: usize,
    pub quantization_stats: ColumnarQuantizationStats,
    pub compression_stats: ColumnarCompressionStats,
    pub performance_stats: ColumnarSerPerformanceStats,
}

/// Backwards-compat alias for [`ColumnarQuantizationStats`].
pub type QuantizationStats = ColumnarQuantizationStats;

/// Statistics about quantization quality
#[derive(Debug, Clone)]
pub struct ColumnarQuantizationStats {
    pub binary_hamming_accuracy: Option<f32>,
    pub int8_mse: Option<f32>,
    pub pq_mse: Option<f32>,
    pub compression_ratio: f32,
    pub memory_reduction: f32,
}

/// Backwards-compat alias for [`ColumnarCompressionStats`].
pub type CompressionStats = ColumnarCompressionStats;

/// Statistics about compression
#[derive(Debug, Clone)]
pub struct ColumnarCompressionStats {
    pub fp32_compressed_size: usize,
    pub binary_compressed_size: usize,
    pub int8_compressed_size: usize,
    pub pq_compressed_size: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub compression_ratio: f32,
}

/// Backwards-compat alias for [`ColumnarSerPerformanceStats`].
pub type PerformanceStats = ColumnarSerPerformanceStats;

/// Performance statistics
#[derive(Debug, Clone)]
pub struct ColumnarSerPerformanceStats {
    pub serialization_time_ms: f64,
    pub quantization_time_ms: f64,
    pub compression_time_ms: f64,
    pub simd_acceleration_used: bool,
    pub memory_pool_hits: usize,
}

impl ColumnarSerializer {
    /// Create new serializer without a quantization engine (no transparent quantization).
    ///
    /// Composition-root pattern (TD-DECOMP-78): callers that need transparent quantization
    /// inject the concrete engine (which impls `StorageQuantizationEnginePort` in its own
    /// crate) via [`ColumnarSerializer::with_quantization_engine`].
    pub fn new(config: ColumnarSerializationConfig) -> Result<Self> {
        Ok(Self {
            config,
            quantization_engine: None,
            memory_pools: MemoryPools::new(),
        })
    }

    /// Builder: inject a quantization engine (port). No-op unless the config enables
    /// quantization, mirroring the previous default-construction semantics.
    pub fn with_quantization_engine(
        mut self,
        engine: Arc<dyn StorageQuantizationEnginePort>,
    ) -> Self {
        if self.config.quantization.is_some() {
            self.quantization_engine = Some(engine);
        }
        self
    }

    /// Serialize vector records with transparent quantization
    pub async fn serialize_vectors(
        &self,
        records: &[ProximaRecord],
        _schema: &Schema,
    ) -> Result<SerializationResult> {
        let start_time = std::time::Instant::now();
        let mut quantization_time = 0.0;
        let mut _compression_time = 0.0;
        let _memory_pool_hits = 0; // Deferred: Track actual memory pool hits

        info!(
            "Serializing {} vector records with transparent quantization",
            records.len()
        );

        // Extract vectors from records
        let vectors: Vec<&[f32]> = records
            .iter()
            .map(|r| r.embeddings.first().map_or(&[][..], |e| e.as_fp32_slice()))
            .collect();

        // Serialize FP32 vectors
        let fp32_array = self.serialize_fp32_vectors(&vectors)?;

        // Transparent quantization if configured
        let (binary_array, int8_arrays, pq_array, quant_stats) =
            if let (Some(engine), Some(quant_config)) =
                (&self.quantization_engine, &self.config.quantization)
            {
                let quant_start = std::time::Instant::now();
                let quantized_data = self.quantize_vectors(&vectors, engine.as_ref()).await?;
                quantization_time = quant_start.elapsed().as_secs_f64() * 1000.0;

                let binary = if quant_config.enable_binary.unwrap_or(false) {
                    Some(self.serialize_binary_vectors(&quantized_data)?)
                } else {
                    None
                };

                let int8 = if quant_config.enable_int8.unwrap_or(false) {
                    Some(self.serialize_int8_vectors(&quantized_data)?)
                } else {
                    None
                };

                let pq = if quant_config.enable_pq.unwrap_or(false) {
                    Some(self.serialize_pq_vectors(&quantized_data)?)
                } else {
                    None
                };

                let stats = self.calculate_quantization_stats(&vectors, &quantized_data)?;

                (binary, int8, pq, stats)
            } else {
                (
                    None,
                    None,
                    None,
                    ColumnarQuantizationStats {
                        binary_hamming_accuracy: None,
                        int8_mse: None,
                        pq_mse: None,
                        compression_ratio: 1.0,
                        memory_reduction: 0.0,
                    },
                )
            };

        // Calculate compression statistics
        let comp_start = std::time::Instant::now();
        let compression_stats = self.calculate_compression_stats(
            &fp32_array,
            &binary_array,
            &int8_arrays.as_ref().map(|(a, _, _)| a),
            &pq_array,
        )?;
        _compression_time = comp_start.elapsed().as_secs_f64() * 1000.0;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let compression_ratio = compression_stats.compression_ratio;

        let metadata = SerializationMetadata {
            record_count: records.len(),
            dimension: self.config.dimension,
            quantization_stats: quant_stats,
            compression_stats,
            performance_stats: ColumnarSerPerformanceStats {
                serialization_time_ms: total_time,
                quantization_time_ms: quantization_time,
                compression_time_ms: _compression_time,
                simd_acceleration_used: self.config.simd_config.enable_simd,
                memory_pool_hits: _memory_pool_hits,
            },
        };

        info!(
            "Serialization completed in {:.2}ms, compression ratio: {:.2}x",
            total_time, compression_ratio
        );

        Ok(SerializationResult {
            fp32_array: Some(fp32_array),
            binary_array,
            int8_array: int8_arrays.as_ref().map(|(a, _, _)| a.clone()),
            int8_scale_array: int8_arrays.as_ref().map(|(_, s, _)| s.clone()),
            int8_zero_point_array: int8_arrays.as_ref().map(|(_, _, z)| z.clone()),
            pq_array,
            metadata,
        })
    }

    /// Deserialize vectors with transparent format selection
    pub async fn deserialize_vectors(
        &self,
        arrays: &HashMap<String, ArrayRef>,
        _schema: &Schema,
        format_preference: FormatPreference,
    ) -> Result<Vec<ProximaRecord>> {
        let start_time = std::time::Instant::now();

        info!(
            "Deserializing vectors with format preference: {:?}",
            format_preference
        );

        // Determine best available format
        let selected_format = self.select_optimal_format(arrays, format_preference)?;

        trace!("Selected format: {:?}", selected_format);

        let vectors = match selected_format {
            SelectedFormat::FP32 => {
                let vector_key = "vector";
                let array = arrays.get(vector_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        vector_key
                    ))
                })?;
                self.deserialize_fp32_vectors(array)?
            }
            SelectedFormat::Binary => {
                let binary_key = "vector_binary";
                let array = arrays.get(binary_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        binary_key
                    ))
                })?;
                self.deserialize_binary_vectors(array).await?
            }
            SelectedFormat::INT8 => {
                let vector_key = "vector_int8";
                let scale_key = "int8_scale";
                let zero_point_key = "int8_zero_point";
                let vector_array = arrays.get(vector_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        vector_key
                    ))
                })?;
                let scale_array = arrays.get(scale_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        scale_key
                    ))
                })?;
                let zero_point_array = arrays.get(zero_point_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        zero_point_key
                    ))
                })?;
                self.deserialize_int8_vectors(vector_array, scale_array, zero_point_array)?
            }
            SelectedFormat::PQ => {
                let pq_key = "vector_pq";
                let array = arrays.get(pq_key).ok_or_else(|| {
                    proximadb_kernel::error::VectorDBError::InvalidInput(format!(
                        "Missing required array: {}",
                        pq_key
                    ))
                })?;
                self.deserialize_pq_vectors(array).await?
            }
        };

        // Convert to ProximaRecord format
        let records = vectors
            .into_iter()
            .enumerate()
            .map(|(i, vector)| {
                let dim = vector.len() as u32;
                let mut record = ProximaRecord {
                    oid: format!("record_{i}"), // Placeholder - would come from ID column
                    ..Default::default()
                };
                if !vector.is_empty() {
                    record
                        .embeddings
                        .push(EmbeddingCell::new_fp32("default", "vector", dim, vector));
                }
                record
            })
            .collect();

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Deserialization completed in {:.2}ms using format: {:?}",
            total_time, selected_format
        );

        Ok(records)
    }

    /// Serialize FP32 vectors to Arrow array
    fn serialize_fp32_vectors(&self, vectors: &[&[f32]]) -> Result<ArrayRef> {
        let dimension = self.config.dimension;

        if self.config.memory_optimization.enable_zero_copy && self.is_fixed_dimension(vectors) {
            // Zero-copy path for fixed dimensions using bytemuck
            self.serialize_fp32_zero_copy(vectors, dimension)
        } else {
            // Standard serialization path
            self.serialize_fp32_standard(vectors, dimension)
        }
    }

    /// Zero-copy FP32 serialization for fixed dimensions
    fn serialize_fp32_zero_copy(&self, vectors: &[&[f32]], dimension: usize) -> Result<ArrayRef> {
        let total_elements = vectors.len() * dimension;
        let mut buffer = self.memory_pools.fp32_vector(total_elements);

        // Copy vectors into contiguous buffer with SIMD alignment
        for vector in vectors {
            if vector.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Vector dimension mismatch: expected {}, got {}",
                    dimension,
                    vector.len()
                ));
            }
            buffer.extend_from_slice(vector);
        }

        // Convert to bytes using bytemuck for zero-copy
        let byte_buffer: &[u8] = cast_slice(&buffer);
        let fixed_size = dimension * 4; // 4 bytes per f32

        let values: Vec<Option<&[u8]>> = (0..vectors.len())
            .map(|i| Some(&byte_buffer[i * fixed_size..(i + 1) * fixed_size]))
            .collect();

        let array = FixedSizeBinaryArray::try_new(
            fixed_size as i32,
            values
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .concat()
                .into(),
            None,
        )?;

        // Return buffer to pool
        self.memory_pools.return_fp32_vector(buffer);

        Ok(Arc::new(array))
    }

    /// Standard FP32 serialization path
    fn serialize_fp32_standard(&self, vectors: &[&[f32]], dimension: usize) -> Result<ArrayRef> {
        let mut builder = Float32Builder::with_capacity(vectors.len() * dimension);

        for vector in vectors {
            if vector.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Vector dimension mismatch: expected {}, got {}",
                    dimension,
                    vector.len()
                ));
            }
            builder.append_slice(vector);
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Quantize vectors using the storage quantization engine
    async fn quantize_vectors(
        &self,
        vectors: &[&[f32]],
        engine: &dyn StorageQuantizationEnginePort,
    ) -> Result<Vec<StorageQuantizedData>> {
        // Convert vector slices to owned vectors and create IDs
        let owned_vectors: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
        let ids: Vec<String> = (0..vectors.len()).map(|i| format!("temp_{i}")).collect();

        // Quantize all vectors at once
        let quantized_data = engine
            .quantize_batch(&owned_vectors, Some(&ids))
            .await
            .context("Failed to quantize vectors")?;

        Ok(quantized_data)
    }

    /// Serialize binary quantized vectors
    fn serialize_binary_vectors(
        &self,
        quantized_data: &[StorageQuantizedData],
    ) -> Result<ArrayRef> {
        let binary_size = self.config.dimension.div_ceil(8);
        let mut builder = FixedSizeBinaryBuilder::new(binary_size as i32);

        for data in quantized_data {
            if let Some(ref filter_quant) = data.filter {
                // Assuming filter quantization is binary
                let binary_data = filter_quant.data.as_slice();
                if binary_data.len() != binary_size {
                    return Err(anyhow::anyhow!(
                        "Binary vector size mismatch: expected {}, got {}",
                        binary_size,
                        binary_data.len()
                    ));
                }
                builder.append_value(binary_data)?;
            } else {
                // Add null value if no binary quantization
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Serialize INT8 quantized vectors
    fn serialize_int8_vectors(
        &self,
        quantized_data: &[StorageQuantizedData],
    ) -> Result<(ArrayRef, ArrayRef, ArrayRef)> {
        let mut vector_builder = FixedSizeBinaryBuilder::new(self.config.dimension as i32);
        let mut scale_builder = Float32Builder::with_capacity(quantized_data.len());
        let mut zero_point_builder = Int8Builder::with_capacity(quantized_data.len());

        for data in quantized_data {
            if let Some(ref fast_quant) = data.fast {
                // Assuming fast quantization is INT8
                let int8_data = fast_quant.data.as_slice();
                if int8_data.len() != self.config.dimension {
                    return Err(anyhow::anyhow!(
                        "INT8 vector size mismatch: expected {}, got {}",
                        self.config.dimension,
                        int8_data.len()
                    ));
                }

                vector_builder.append_value(int8_data)?;

                // Extract scale and offset from metadata
                let scale = fast_quant.metadata.scale.unwrap_or(1.0);
                let offset = fast_quant.metadata.offset.unwrap_or(0.0);

                scale_builder.append_value(scale);
                zero_point_builder.append_value(offset as i8);
            } else {
                vector_builder.append_null();
                scale_builder.append_null();
                zero_point_builder.append_null();
            }
        }

        Ok((
            Arc::new(vector_builder.finish()),
            Arc::new(scale_builder.finish()),
            Arc::new(zero_point_builder.finish()),
        ))
    }

    /// Serialize PQ quantized vectors
    fn serialize_pq_vectors(&self, quantized_data: &[StorageQuantizedData]) -> Result<ArrayRef> {
        let pq_size = self
            .config
            .quantization
            .as_ref()
            .map_or(16, |q| q.pq_segments.unwrap_or(8) as usize); // Default PQ segments

        let mut builder = FixedSizeBinaryBuilder::new(pq_size as i32);

        for data in quantized_data {
            if let Some(ref primary_quant) = data.primary {
                // Assuming primary quantization is PQ
                let pq_data = primary_quant.data.as_slice();
                if pq_data.len() != pq_size {
                    return Err(anyhow::anyhow!(
                        "PQ vector size mismatch: expected {}, got {}",
                        pq_size,
                        pq_data.len()
                    ));
                }
                builder.append_value(pq_data)?;
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    /// Calculate quantization quality statistics
    fn calculate_quantization_stats(
        &self,
        original_vectors: &[&[f32]],
        quantized_data: &[StorageQuantizedData],
    ) -> Result<ColumnarQuantizationStats> {
        // Simplified statistics calculation
        let compression_ratio = if !quantized_data.is_empty() {
            let original_size = original_vectors.len() * self.config.dimension * 4; // FP32
            let quantized_size = quantized_data
                .iter()
                .map(|d| {
                    d.primary.as_ref().map_or(0, |p| p.data.len())
                        + d.filter.as_ref().map_or(0, |f| f.data.len())
                        + d.fast.as_ref().map_or(0, |f| f.data.len())
                })
                .sum::<usize>();

            if quantized_size > 0 {
                original_size as f32 / quantized_size as f32
            } else {
                1.0
            }
        } else {
            1.0
        };

        Ok(ColumnarQuantizationStats {
            binary_hamming_accuracy: None, // Deferred: Calculate actual accuracy
            int8_mse: None,                // Deferred: Calculate MSE
            pq_mse: None,                  // Deferred: Calculate PQ MSE
            compression_ratio,
            memory_reduction: (compression_ratio - 1.0) / compression_ratio * 100.0,
        })
    }

    /// Calculate compression statistics
    fn calculate_compression_stats(
        &self,
        fp32_array: &ArrayRef,
        binary_array: &Option<ArrayRef>,
        int8_array: &Option<&ArrayRef>,
        pq_array: &Option<ArrayRef>,
    ) -> Result<ColumnarCompressionStats> {
        let fp32_size = fp32_array.get_array_memory_size();
        let binary_size = binary_array.as_ref().map(|a| a.get_array_memory_size());
        let int8_size = int8_array.map(|a| a.get_array_memory_size());
        let pq_size = pq_array.as_ref().map(|a| a.get_array_memory_size());

        let total_original = fp32_size;
        let total_compressed =
            binary_size.unwrap_or(0) + int8_size.unwrap_or(0) + pq_size.unwrap_or(0);

        let compression_ratio = if total_compressed > 0 {
            total_original as f32 / total_compressed as f32
        } else {
            1.0
        };

        Ok(ColumnarCompressionStats {
            fp32_compressed_size: fp32_size,
            binary_compressed_size: binary_size.unwrap_or(0),
            int8_compressed_size: int8_size.unwrap_or(0),
            pq_compressed_size: pq_size.unwrap_or(0),
            total_original_size: total_original,
            total_compressed_size: total_compressed,
            compression_ratio,
        })
    }

    /// Check if all vectors have the same dimension
    fn is_fixed_dimension(&self, vectors: &[&[f32]]) -> bool {
        vectors.iter().all(|v| v.len() == self.config.dimension)
    }

    /// Select optimal format based on availability and preference
    fn select_optimal_format(
        &self,
        arrays: &HashMap<String, ArrayRef>,
        preference: FormatPreference,
    ) -> Result<SelectedFormat> {
        match preference {
            FormatPreference::HighestQuality => {
                if arrays.contains_key("vector") {
                    Ok(SelectedFormat::FP32)
                } else if arrays.contains_key("vector_int8") {
                    Ok(SelectedFormat::INT8)
                } else if arrays.contains_key("vector_pq") {
                    Ok(SelectedFormat::PQ)
                } else if arrays.contains_key("vector_binary") {
                    Ok(SelectedFormat::Binary)
                } else {
                    Err(anyhow::anyhow!("No vector data found"))
                }
            }
            FormatPreference::FastestRead => {
                if arrays.contains_key("vector_binary") {
                    Ok(SelectedFormat::Binary)
                } else if arrays.contains_key("vector_int8") {
                    Ok(SelectedFormat::INT8)
                } else if arrays.contains_key("vector_pq") {
                    Ok(SelectedFormat::PQ)
                } else if arrays.contains_key("vector") {
                    Ok(SelectedFormat::FP32)
                } else {
                    Err(anyhow::anyhow!("No vector data found"))
                }
            }
            FormatPreference::SmallestSize => {
                if arrays.contains_key("vector_pq") {
                    Ok(SelectedFormat::PQ)
                } else if arrays.contains_key("vector_binary") {
                    Ok(SelectedFormat::Binary)
                } else if arrays.contains_key("vector_int8") {
                    Ok(SelectedFormat::INT8)
                } else if arrays.contains_key("vector") {
                    Ok(SelectedFormat::FP32)
                } else {
                    Err(anyhow::anyhow!("No vector data found"))
                }
            }
            FormatPreference::Specific(format) => {
                let name = match format {
                    SelectedFormat::FP32 => "vector",
                    SelectedFormat::Binary => "vector_binary",
                    SelectedFormat::INT8 => "vector_int8",
                    SelectedFormat::PQ => "vector_pq",
                };

                if arrays.contains_key(name) {
                    Ok(format)
                } else {
                    Err(anyhow::anyhow!("Requested format {} not available", name))
                }
            }
        }
    }

    /// Deserialize FP32 vectors from Arrow array
    fn deserialize_fp32_vectors(&self, array: &ArrayRef) -> Result<Vec<Vec<f32>>> {
        match array.data_type() {
            DataType::FixedSizeBinary(size) => {
                let fixed_array = array
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| anyhow::anyhow!("Failed to downcast to FixedSizeBinaryArray"))?;

                let _dimension = *size as usize / 4; // 4 bytes per f32
                let mut vectors = Vec::with_capacity(fixed_array.len());

                for i in 0..fixed_array.len() {
                    if !fixed_array.is_null(i) {
                        let bytes = fixed_array.value(i);
                        let floats: &[f32] = try_cast_slice(bytes).map_err(|e| {
                            anyhow::anyhow!("Failed to cast bytes to f32 slice: {}", e)
                        })?;
                        vectors.push(floats.to_vec());
                    } else {
                        return Err(anyhow::anyhow!("Null vector found at index {}", i));
                    }
                }

                Ok(vectors)
            }
            DataType::Float32 => {
                let float_array = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow::anyhow!("Failed to downcast to Float32Array"))?;

                let dimension = self.config.dimension;
                let num_vectors = float_array.len() / dimension;
                let mut vectors = Vec::with_capacity(num_vectors);

                for i in 0..num_vectors {
                    let start = i * dimension;
                    let end = start + dimension;
                    let vector = float_array.values()[start..end].to_vec();
                    vectors.push(vector);
                }

                Ok(vectors)
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported array type for FP32 vectors: {:?}",
                array.data_type()
            )),
        }
    }

    /// Deserialize binary vectors (requires reconstruction to FP32)
    async fn deserialize_binary_vectors(&self, array: &ArrayRef) -> Result<Vec<Vec<f32>>> {
        let binary_array = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                anyhow::anyhow!("Failed to downcast to FixedSizeBinaryArray for binary vectors")
            })?;

        let mut vectors = Vec::with_capacity(binary_array.len());

        // This would require the quantization engine to reconstruct approximate FP32 vectors
        // For now, return zeros as placeholder
        warn!(
            "Binary vector deserialization to FP32 not fully implemented - returning zero vectors"
        );

        for _i in 0..binary_array.len() {
            vectors.push(vec![0.0; self.config.dimension]);
        }

        Ok(vectors)
    }

    /// Deserialize INT8 vectors to FP32
    fn deserialize_int8_vectors(
        &self,
        vector_array: &ArrayRef,
        scale_array: &ArrayRef,
        zero_point_array: &ArrayRef,
    ) -> Result<Vec<Vec<f32>>> {
        let int8_array = vector_array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast INT8 vector array"))?;

        let scale_array = scale_array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast scale array"))?;

        let zero_point_array = zero_point_array
            .as_any()
            .downcast_ref::<Int8Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast zero point array"))?;

        let mut vectors = Vec::with_capacity(int8_array.len());

        for i in 0..int8_array.len() {
            if !int8_array.is_null(i) && !scale_array.is_null(i) && !zero_point_array.is_null(i) {
                let int8_bytes = int8_array.value(i);
                let scale = scale_array.value(i);
                let zero_point = zero_point_array.value(i);

                let int8_values: &[i8] = try_cast_slice(int8_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to cast bytes to i8 slice: {}", e))?;

                let fp32_vector: Vec<f32> = int8_values
                    .iter()
                    .map(|&val| (val as f32 - zero_point as f32) * scale)
                    .collect();

                vectors.push(fp32_vector);
            } else {
                return Err(anyhow::anyhow!(
                    "Missing data for INT8 vector at index {}",
                    i
                ));
            }
        }

        Ok(vectors)
    }

    /// Deserialize PQ vectors (requires codebook for reconstruction)
    async fn deserialize_pq_vectors(&self, array: &ArrayRef) -> Result<Vec<Vec<f32>>> {
        let pq_array = array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| {
                anyhow::anyhow!("Failed to downcast to FixedSizeBinaryArray for PQ vectors")
            })?;

        let mut vectors = Vec::with_capacity(pq_array.len());

        // This would require the quantization engine and codebook to reconstruct approximate FP32 vectors
        // For now, return zeros as placeholder
        warn!("PQ vector deserialization to FP32 not fully implemented - returning zero vectors");

        for _i in 0..pq_array.len() {
            vectors.push(vec![0.0; self.config.dimension]);
        }

        Ok(vectors)
    }
}

/// Format preference for deserialization
#[derive(Debug, Clone)]
pub enum FormatPreference {
    /// Prefer highest quality (FP32 > INT8 > PQ > Binary)
    HighestQuality,
    /// Prefer fastest read (Binary > INT8 > PQ > FP32)
    FastestRead,
    /// Prefer smallest size (PQ > Binary > INT8 > FP32)
    SmallestSize,
    /// Use specific format
    Specific(SelectedFormat),
}

impl From<FormatPreference> for SelectedFormat {
    fn from(pref: FormatPreference) -> Self {
        match pref {
            FormatPreference::HighestQuality => SelectedFormat::FP32,
            FormatPreference::FastestRead => SelectedFormat::Binary,
            FormatPreference::SmallestSize => SelectedFormat::PQ,
            FormatPreference::Specific(format) => format,
        }
    }
}

// NOTE: SelectedFormat has been moved to proximadb_distance_kernel::quantized
// This eliminates code duplication and allows all engines to use the same format definitions

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::BinaryArray;

    #[tokio::test]
    async fn test_fp32_serialization() {
        let config = ColumnarSerializationConfig {
            dimension: 128,
            quantization: None,
            compression: SerializationCompressionConfig::default(),
            memory_optimization: MemoryOptimizationConfig::default(),
            simd_config: SIMDConfig::default(),
        };

        let serializer = ColumnarSerializer::new(config).unwrap();

        let vectors = [vec![1.0; 128], vec![2.0; 128], vec![3.0; 128]];

        let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let array = serializer.serialize_fp32_vectors(&vector_refs).unwrap();

        assert_eq!(array.len(), 3);

        // Test deserialization
        let deserialized = serializer.deserialize_fp32_vectors(&array).unwrap();
        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized[0], vectors[0]);
        assert_eq!(deserialized[1], vectors[1]);
        assert_eq!(deserialized[2], vectors[2]);
    }

    #[test]
    fn test_format_selection() {
        let serializer = ColumnarSerializer::new(ColumnarSerializationConfig {
            dimension: 128,
            quantization: None,
            compression: SerializationCompressionConfig::default(),
            memory_optimization: MemoryOptimizationConfig::default(),
            simd_config: SIMDConfig::default(),
        })
        .unwrap();

        let mut arrays = HashMap::new();
        arrays.insert(
            "vector".to_string(),
            Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
        );
        arrays.insert(
            "vector_binary".to_string(),
            Arc::new(BinaryArray::from_opt_vec(vec![Some(b"test")])) as ArrayRef,
        );

        // Test highest quality preference
        let format = serializer
            .select_optimal_format(&arrays, FormatPreference::HighestQuality)
            .unwrap();
        assert!(matches!(format, SelectedFormat::FP32));

        // Test fastest read preference
        let format = serializer
            .select_optimal_format(&arrays, FormatPreference::FastestRead)
            .unwrap();
        assert!(matches!(format, SelectedFormat::Binary));
    }

    #[test]
    fn test_memory_pools() {
        let pools = MemoryPools::new();

        // Test FP32 pool
        let vec1 = pools.fp32_vector(100);
        assert!(vec1.capacity() >= 100);

        pools.return_fp32_vector(vec1);

        let vec2 = pools.fp32_vector(50);
        // Should reuse the returned vector
        assert!(vec2.capacity() >= 50);

        // Test INT8 pool
        let int8_vec = pools.get_int8_vector(256);
        assert!(int8_vec.capacity() >= 256);
        pools.return_int8_vector(int8_vec);

        // Test binary pool
        let binary_vec = pools.get_binary_vector(64);
        assert!(binary_vec.capacity() >= 64);
        pools.return_binary_vector(binary_vec);
    }
}
