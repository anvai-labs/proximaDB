//! Unified Quantized Distance Computation
//!
//! This module provides SIMD-optimized distance calculations for quantized vectors
//! across all storage engines (SST, VIPER, NOVA). It integrates with the unified
//! distance computation infrastructure and provides specialized implementations
//! for binary, INT8, and PQ quantization with proper data types.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use tracing::{info, trace};

use crate::{DistanceMetric, engine::UnifiedDistanceCompute};
use proximadb_hardware::{SimdLevel, best_simd_level};

/// Configuration for distance calculations on quantized data
#[derive(Debug, Clone)]
pub struct QuantizedDistanceConfig {
    /// Target distance metric
    pub distance_metric: DistanceMetric,

    /// SIMD optimization settings
    pub simd_optimization: SIMDOptimization,

    /// Cache configuration for distance tables
    pub cache_config: DistanceCacheConfig,

    /// Approximation settings
    pub approximation: ApproximationConfig,

    /// Hardware acceleration preferences
    pub hardware_preferences: HardwarePreferences,
}

/// SIMD optimization configuration
#[derive(Debug, Clone)]
pub struct SIMDOptimization {
    /// Enable SIMD instructions
    pub enable_simd: bool,

    /// Minimum batch size for SIMD operations
    pub simd_threshold: usize,

    /// Instruction set preference
    pub instruction_set: InstructionSet,

    /// Enable hardware-specific optimizations
    pub enable_hardware_specific: bool,

    /// Vectorization strategy
    pub vectorization_strategy: VectorizationStrategy,
}

/// Supported instruction sets
#[derive(Debug, Clone)]
pub enum InstructionSet {
    /// Auto-detect best available
    Auto,
    /// Scalar (no SIMD)
    Scalar,
    /// SSE/SSE2 (128-bit)
    SSE,
    /// AVX/AVX2 (256-bit)
    AVX,
    /// AVX-512 (512-bit)
    AVX512,
    /// ARM NEON (128-bit)
    NEON,
}

/// Vectorization strategies
#[derive(Debug, Clone)]
pub enum VectorizationStrategy {
    /// Process vectors individually
    Individual,
    /// Batch multiple vectors for better cache utilization
    Batched,
    /// Streaming processing for large datasets
    Streaming,
    /// Adaptive based on dataset size
    Adaptive,
}

/// Distance table cache configuration
#[derive(Debug, Clone)]
pub struct DistanceCacheConfig {
    /// Enable distance table caching for PQ
    pub enable_pq_cache: bool,

    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,

    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,

    /// Precompute distance tables on collection load
    pub precompute_on_load: bool,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
}

/// Approximation configuration
#[derive(Debug, Clone)]
pub struct ApproximationConfig {
    /// Quality vs speed trade-off (0.0 = fastest, 1.0 = highest quality)
    pub quality_factor: f32,

    /// Early termination threshold for progressive search
    pub early_termination_threshold: f32,

    /// Maximum candidates to consider in each stage
    pub max_candidates_per_stage: usize,

    /// Enable progressive refinement
    pub enable_progressive_refinement: bool,
}

/// Hardware acceleration preferences
#[derive(Debug, Clone)]
pub struct HardwarePreferences {
    /// Prefer GPU acceleration when available
    pub prefer_gpu: bool,

    /// Minimum problem size for GPU acceleration
    pub gpu_threshold: usize,

    /// Memory bandwidth optimization
    pub optimize_memory_bandwidth: bool,

    /// Cache-aware optimizations
    pub enable_cache_optimization: bool,
}

/// Apply the RaBitQ-style length-renormalization correction (P7).
///
/// Multiplies the raw inner-product estimate by `renorm` when present.
/// For distance-shaped metrics (Euclidean, Hamming), the correction would
/// require knowing the candidate's stored norm separately, so this
/// helper only applies to inner-product-shaped metrics (Cosine,
/// DotProduct). Other metrics fall through unchanged — pre-P7 callers
/// are unaffected.
///
/// `None` always returns the input unchanged. This is the unconditional
/// pre-P7 path.
#[inline]
fn apply_length_renorm(raw: f32, renorm: Option<f32>, metric: &DistanceMetric) -> f32 {
    match (renorm, metric) {
        (Some(s), DistanceMetric::Cosine) | (Some(s), DistanceMetric::DotProduct) => raw * s,
        _ => raw,
    }
}

/// Format selection for quantized distance computation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedFormat {
    /// Full precision floating-point
    FP32,
    /// Binary quantization (1-bit)
    Binary,
    /// 8-bit integer quantization
    INT8,
    /// Product quantization
    PQ,
}

/// Quantized vector data for distance computation
#[derive(Debug, Clone)]
pub struct QuantizedVectorData {
    /// Original FP32 vector (optional)
    pub fp32: Option<Vec<f32>>,

    /// Binary quantized vector
    pub binary: Option<Vec<u8>>,

    /// INT8 quantized vector with scale/zero point
    pub int8: Option<Int8VectorData>,

    /// Product quantized vector
    pub pq: Option<PQVectorData>,
}

/// INT8 quantized vector data with proper scaling
#[derive(Debug, Clone)]
pub struct Int8VectorData {
    /// Quantized values as signed 8-bit integers
    pub values: Vec<i8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for affine quantization
    pub zero_point: i8,
    /// Per-vector RaBitQ-style length renormalization scalar (P7).
    ///
    /// When `Some(s)`, the scoring path multiplies its raw inner-product
    /// estimate by `s` to remove the systematic downward bias of scalar
    /// quantization. The mathematical form is
    /// `s = ||v|| / <u, x_hat>` where `u = v / ||v||` is the unit
    /// direction and `x_hat` is the dequantized reconstruction.
    ///
    /// `None` preserves pre-P7 behaviour (no correction applied).
    /// Construct via `with_length_renorm` for the corrected variant.
    /// See `TURBOQUANT_LLD_2026_05_30.adoc` §"Implementation Status" P7.
    pub length_renorm: Option<f32>,
}

impl Int8VectorData {
    /// Compute the RaBitQ-style length-renormalization scalar from the
    /// original FP32 vector `v` and its quantized + scaled reconstruction
    /// `x_hat`. Returns `None` for degenerate inputs (zero norm or
    /// orthogonal reconstruction) so the scoring path falls through to
    /// the uncorrected branch.
    pub fn compute_length_renorm(v: &[f32], x_hat: &[f32]) -> Option<f32> {
        if v.len() != x_hat.len() {
            return None;
        }
        let mut sumsq = 0.0f64;
        let mut inner = 0.0f64;
        for (vi, xi) in v.iter().zip(x_hat.iter()) {
            sumsq += (*vi as f64) * (*vi as f64);
            inner += (*vi as f64) * (*xi as f64);
        }
        let norm = sumsq.sqrt();
        if norm < 1e-10 || inner.abs() < 1e-10 {
            return None;
        }
        let scale = (norm / inner) as f32;
        if scale.is_finite() { Some(scale) } else { None }
    }

    /// Builder-style: attach a precomputed length-renorm scalar.
    pub fn with_length_renorm(mut self, renorm: f32) -> Self {
        self.length_renorm = Some(renorm);
        self
    }
}

/// Product quantized vector data
#[derive(Debug, Clone)]
pub struct PQVectorData {
    /// Quantized codes (one per subvector)
    pub codes: Vec<u8>,
    /// Codebook: [subvector][centroid][dimension]
    pub codebook: Vec<Vec<f32>>,
    /// Hash of codebook for caching
    pub codebook_hash: u64,
    /// Per-vector RaBitQ-style length renormalization scalar (P7). See
    /// [`Int8VectorData::length_renorm`] for the semantics. `None`
    /// preserves pre-P7 behaviour.
    pub length_renorm: Option<f32>,
}

impl PQVectorData {
    /// Compute the length-renorm scalar from the original FP32 vector
    /// and the dequantized PQ reconstruction. Returns `None` for
    /// degenerate inputs.
    pub fn compute_length_renorm(v: &[f32], x_hat: &[f32]) -> Option<f32> {
        // Same numerical contract as Int8VectorData::compute_length_renorm
        // — wrapping the helper keeps the formula in one place.
        Int8VectorData::compute_length_renorm(v, x_hat)
    }

    /// Builder-style: attach a precomputed length-renorm scalar.
    pub fn with_length_renorm(mut self, renorm: f32) -> Self {
        self.length_renorm = Some(renorm);
        self
    }
}

/// TurboQuant-quantized vector data (ADR-021, TURBOQUANT_LLD_2026_05_30
/// §"Locked Type Signatures"). Mirrors [`Int8VectorData`]'s
/// `values + scale + zero_point` shape: bit-packed codes plus a per-vector
/// `scale` that carries the RaBitQ-style length-renormalization correction
/// `||v|| / <u_rot, x_hat>`. Applying `scale` at the final
/// score-multiplication site recovers an unbiased inner-product estimator
/// — the recall gain is largest at low bit-widths.
///
/// `encoded_epoch` tracks the collection's precision/calibration epoch
/// (per `EMBEDDING_PRECISION_LLD_2026_05_22` Q12); a mismatch against the
/// collection's current epoch routes through repair-from-`ProximaRecord`.
///
/// All field semantics are paper-driven (no derivation of turbovec's MIT
/// implementation); the struct shape is locked in
/// `TURBOQUANT_LLD_2026_05_30.adoc` §3 wire format and §"Locked Type
/// Signatures".
#[cfg(feature = "experimental-turboquant")]
#[derive(Debug, Clone)]
pub struct TurboQuantVectorData {
    /// Bit-packed codes. Length = `ceil(dim * bit_width / 8)` bytes.
    pub codes: Vec<u8>,
    /// Bit-width per coordinate (2, 3, or 4). 3-bit deferred to P10.
    pub bit_width: u8,
    /// Per-vector length-renormalization scalar:
    ///   `scale = ||v|| / <u_rot, x_hat>`
    /// Applied at the SIMD kernel's final multiplication so the inner
    /// product `<v, q>` is recovered unbiased.
    pub scale: f32,
    /// Collection epoch this code was encoded under. Mismatch with the
    /// collection's current epoch routes through repair source. See
    /// `EMBEDDING_PRECISION_LLD_2026_05_22` Q12.
    pub encoded_epoch: u64,
}

/// Distance computation result with metadata
#[derive(Debug, Clone)]
pub struct QuantizedDistanceResult {
    /// Computed distance
    pub similarity: f32,

    /// Quality estimate (0.0 = low quality, 1.0 = exact)
    pub quality_estimate: f32,

    /// Computation method used
    pub method: ComputationMethod,

    /// Performance metrics
    pub metrics: DistanceMetrics,
}

/// Computation methods for distance calculation
#[derive(Debug, Clone)]
pub enum ComputationMethod {
    /// Exact FP32 computation
    ExactFP32,
    /// Binary approximation
    BinaryApproximation,
    /// INT8 approximation
    INT8Approximation,
    /// Product Quantization approximation
    PQApproximation,
    /// Progressive refinement (multiple stages)
    ProgressiveRefinement { stages: Vec<String> },
}

/// Performance metrics for distance computation
#[derive(Debug, Clone)]
pub struct DistanceMetrics {
    /// Computation time in microseconds
    pub computation_time_us: f64,

    /// SIMD acceleration used
    pub simd_used: bool,

    /// Cache hits/misses
    pub cache_hits: usize,
    pub cache_misses: usize,

    /// Memory bandwidth utilization
    pub memory_bandwidth_mb_s: f32,

    /// Number of operations performed
    pub operation_count: usize,
}

impl Default for DistanceMetrics {
    fn default() -> Self {
        Self {
            computation_time_us: 0.0,
            simd_used: false,
            cache_hits: 0,
            cache_misses: 0,
            memory_bandwidth_mb_s: 0.0,
            operation_count: 0,
        }
    }
}

/// Optimized distance calculator for quantized data
pub struct QuantizedDistanceCalculator {
    /// Configuration
    config: QuantizedDistanceConfig,

    /// Unified distance compute engine
    distance_engine: Arc<UnifiedDistanceCompute>,

    /// Whether SIMD is available on this platform
    has_simd: bool,

    /// Distance table cache for PQ
    #[allow(dead_code)]
    pq_distance_cache: Arc<std::sync::RwLock<PQDistanceCache>>,

    /// Binary Hamming LUT for fast binary distance
    hamming_lut: Arc<HammingLookupTable>,

    /// INT8 distance tables
    #[allow(dead_code)]
    int8_distance_tables: Arc<std::sync::RwLock<Int8DistanceTables>>,
}

/// Product Quantization distance table cache
#[derive(Debug)]
struct PQDistanceCache {
    /// Cached distance tables by codebook hash
    #[allow(dead_code)]
    tables: std::collections::HashMap<u64, Arc<PQDistanceTable>>,

    /// Cache statistics
    #[allow(dead_code)]
    hits: usize,
    /// Cache miss count
    #[allow(dead_code)]
    misses: usize,

    /// Total memory usage
    #[allow(dead_code)]
    memory_usage_bytes: usize,
}

/// PQ distance table for O(1) distance lookups
#[derive(Debug)]
struct PQDistanceTable {
    /// Distance table [subvector][centroid] = distance
    #[allow(dead_code)]
    tables: Vec<Vec<f32>>,

    /// Number of subvectors
    #[allow(dead_code)]
    num_subvectors: usize,

    /// Number of centroids per subvector
    #[allow(dead_code)]
    num_centroids: usize,

    /// Distance metric used
    #[allow(dead_code)]
    distance_metric: DistanceMetric,

    /// Creation timestamp for cache eviction
    #[allow(dead_code)]
    timestamp: std::time::Instant,

    /// Access count for LFU eviction
    #[allow(dead_code)]
    access_count: std::sync::atomic::AtomicUsize,
}

/// Hamming distance lookup table for binary quantization
#[derive(Debug)]
struct HammingLookupTable {
    /// Precomputed hamming weights for all 8-bit values
    hamming_weights: [u8; 256],

    /// Popcnt instruction availability
    has_popcnt: bool,
}

/// INT8 distance tables for accelerated computation
#[derive(Debug)]
struct Int8DistanceTables {
    /// Distance computation lookup tables
    #[allow(dead_code)]
    tables: std::collections::HashMap<(DistanceMetric, usize), Arc<Int8DistanceTable>>,

    /// Memory usage tracking
    #[allow(dead_code)]
    memory_usage_bytes: usize,
}

/// INT8 distance computation table
#[derive(Debug)]
struct Int8DistanceTable {
    /// Precomputed squared differences for INT8 values
    #[allow(dead_code)]
    squared_diff_table: Vec<Vec<f32>>,

    /// Distance metric
    #[allow(dead_code)]
    distance_metric: DistanceMetric,

    /// Dimension
    #[allow(dead_code)]
    dimension: usize,
}

impl Default for QuantizedDistanceConfig {
    fn default() -> Self {
        Self {
            distance_metric: DistanceMetric::Cosine,
            simd_optimization: SIMDOptimization::default(),
            cache_config: DistanceCacheConfig::default(),
            approximation: ApproximationConfig::default(),
            hardware_preferences: HardwarePreferences::default(),
        }
    }
}

impl Default for SIMDOptimization {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_threshold: 64,
            instruction_set: InstructionSet::Auto,
            enable_hardware_specific: true,
            vectorization_strategy: VectorizationStrategy::Adaptive,
        }
    }
}

impl Default for DistanceCacheConfig {
    fn default() -> Self {
        Self {
            enable_pq_cache: true,
            max_cache_size_mb: 256,
            eviction_policy: CacheEvictionPolicy::LRU,
            precompute_on_load: true,
        }
    }
}

impl Default for ApproximationConfig {
    fn default() -> Self {
        Self {
            quality_factor: 0.8,
            early_termination_threshold: 0.95,
            max_candidates_per_stage: 1000,
            enable_progressive_refinement: true,
        }
    }
}

impl Default for HardwarePreferences {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            gpu_threshold: 10000,
            optimize_memory_bandwidth: true,
            enable_cache_optimization: true,
        }
    }
}

impl QuantizedDistanceCalculator {
    /// Create new distance calculator
    pub fn new(config: QuantizedDistanceConfig) -> Result<Self> {
        let has_simd = best_simd_level() > SimdLevel::Scalar;
        let distance_engine = Arc::new(UnifiedDistanceCompute::new(DistanceMetric::Cosine));

        let pq_distance_cache = Arc::new(std::sync::RwLock::new(PQDistanceCache {
            tables: std::collections::HashMap::new(),
            hits: 0,
            misses: 0,
            memory_usage_bytes: 0,
        }));

        let hamming_lut = Arc::new(HammingLookupTable::new(has_simd)?);

        let int8_distance_tables = Arc::new(std::sync::RwLock::new(Int8DistanceTables {
            tables: std::collections::HashMap::new(),
            memory_usage_bytes: 0,
        }));

        info!(
            "Initialized quantized distance calculator with SIMD: {}",
            config.simd_optimization.enable_simd
        );

        Ok(Self {
            config,
            distance_engine,
            has_simd,
            pq_distance_cache,
            hamming_lut,
            int8_distance_tables,
        })
    }

    /// Compute distance between query vector and quantized data
    pub async fn compute_distance(
        &self,
        query: &[f32],
        quantized_vector: &QuantizedVectorData,
        format: SelectedFormat,
    ) -> Result<QuantizedDistanceResult> {
        let start_time = std::time::Instant::now();
        let mut cache_hits = 0;
        let cache_misses = 0;

        trace!("Computing distance using format: {:?}", format);

        let (similarity, quality_estimate, method) = match format {
            SelectedFormat::FP32 => {
                let fp32_data = quantized_vector
                    .fp32
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("FP32 data not available"))?;

                let result = self.distance_engine.calculate_distance(
                    query,
                    fp32_data,
                    &self.config.distance_metric,
                );

                (result.raw_value, 1.0, ComputationMethod::ExactFP32)
            }

            SelectedFormat::Binary => {
                let binary_data = quantized_vector
                    .binary
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Binary data not available"))?;

                let similarity = self.compute_binary_distance(query, binary_data)?;

                // Calculate quality based on dimension
                // Binary quantization preserves more information in higher dimensions
                // due to the concentration of measure phenomenon
                let dimension = query.len();
                let quality_estimate = if dimension < 64 {
                    0.60 // Low dimension: 60% quality (more information loss)
                } else if dimension < 128 {
                    0.65 // Small dimension: 65% quality
                } else if dimension < 256 {
                    0.70 // Medium dimension: 70% quality
                } else if dimension < 512 {
                    0.75 // Large dimension: 75% quality
                } else if dimension < 1024 {
                    0.80 // Very large dimension: 80% quality
                } else {
                    0.85 // Huge dimension: 85% quality (binary works well at scale)
                };

                (
                    similarity,
                    quality_estimate,
                    ComputationMethod::BinaryApproximation,
                )
            }

            SelectedFormat::INT8 => {
                let int8_data = quantized_vector
                    .int8
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("INT8 data not available"))?;

                // Use native INT8 distance computation from unified engine
                let query_int8 =
                    self.quantize_query_to_int8(query, int8_data.scale, int8_data.zero_point)?;
                let result = self.distance_engine.calculate_int8_distance(
                    &query_int8,
                    &int8_data.values,
                    int8_data.scale,
                    int8_data.scale, // Assume same scale for simplicity
                    int8_data.zero_point,
                    int8_data.zero_point,
                    &self.config.distance_metric,
                );

                // P7: RaBitQ-style length renormalization. When the
                // candidate's `length_renorm` was computed at encode time,
                // multiply the raw inner-product estimate by it to remove
                // the systematic downward bias of scalar quantization.
                // Pre-P7 callers never set the field; that path is
                // unchanged.
                let raw_value = apply_length_renorm(
                    result.raw_value,
                    int8_data.length_renorm,
                    &self.config.distance_metric,
                );
                let quality = if int8_data.length_renorm.is_some() {
                    0.93
                } else {
                    0.9
                };
                (raw_value, quality, ComputationMethod::INT8Approximation)
            }

            SelectedFormat::PQ => {
                let pq_data = quantized_vector
                    .pq
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("PQ data not available"))?;

                // Use native PQ distance computation from unified engine
                let result = self.distance_engine.calculate_pq_distance(
                    query,
                    &pq_data.codes,
                    &pq_data.codebook,
                    &self.config.distance_metric,
                );

                cache_hits += 1; // PQ computation typically uses cached distance tables
                // P7: same renorm as INT8 above. Pre-P7 PQ vectors keep
                // their downward bias; P7-encoded vectors get the
                // correction at zero scoring overhead.
                let raw_value = apply_length_renorm(
                    result.raw_value,
                    pq_data.length_renorm,
                    &self.config.distance_metric,
                );
                let quality = if pq_data.length_renorm.is_some() {
                    0.9
                } else {
                    0.85
                };
                (raw_value, quality, ComputationMethod::PQApproximation)
            }
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1_000_000.0; // Convert to microseconds

        let metrics = DistanceMetrics {
            computation_time_us: computation_time,
            simd_used: self.config.simd_optimization.enable_simd
                && self.should_use_simd(query.len()),
            cache_hits,
            cache_misses,
            memory_bandwidth_mb_s: self.estimate_memory_bandwidth(query.len(), format.clone()),
            operation_count: self.estimate_operation_count(query.len(), format.clone()),
        };

        trace!(
            "Distance computation completed in {:.2}μs",
            computation_time
        );

        Ok(QuantizedDistanceResult {
            similarity,
            quality_estimate,
            method,
            metrics,
        })
    }

    /// Compute distances for multiple vectors (batch processing)
    pub async fn compute_batch_distances(
        &self,
        query: &[f32],
        quantized_vectors: &[QuantizedVectorData],
        format: SelectedFormat,
    ) -> Result<Vec<QuantizedDistanceResult>> {
        info!(
            "Computing batch distances for {} vectors using format: {:?}",
            quantized_vectors.len(),
            format
        );

        let start_time = std::time::Instant::now();

        // Per-vector computation for every batch size. The former
        // `enable_simd`-gated "batch path" (compute_batch_distances_simd) ran
        // an identical per-vector loop under a SIMD name — both arms of the
        // removed branch were the same code (TD-FACADE-1). A real batched
        // kernel for mixed-format QuantizedVectorData does not exist in this
        // crate; the f32-pair SIMD kernels live in the `engine` module
        // (`batch_distance_pooled_simd`). `simd_optimization.enable_simd`
        // still gates the hardware-detected popcnt Hamming path inside
        // compute_distance; it does not vectorize this loop.
        let mut results = Vec::with_capacity(quantized_vectors.len());
        for vector in quantized_vectors {
            let result = self.compute_distance(query, vector, format.clone()).await?;
            results.push(result);
        }

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Batch distance computation completed in {:.2}ms ({:.2} vectors/ms)",
            total_time,
            quantized_vectors.len() as f64 / total_time
        );

        Ok(results)
    }

    /// Progressive distance computation with quality refinement
    pub async fn compute_progressive_distance(
        &self,
        query: &[f32],
        quantized_vector: &QuantizedVectorData,
        target_quality: f32,
    ) -> Result<QuantizedDistanceResult> {
        if !self.config.approximation.enable_progressive_refinement {
            // Fall back to highest quality available
            let format = self.select_best_format(quantized_vector);
            return self.compute_distance(query, quantized_vector, format).await;
        }

        let start_time = std::time::Instant::now();
        let mut stages = Vec::new();
        let mut current_quality = 0.0;
        let mut final_distance = 0.0;

        trace!(
            "Starting progressive distance computation (target quality: {:.2})",
            target_quality
        );

        // Stage 1: Binary filtering (if available and quality target allows)
        if let Some(_binary_data) = &quantized_vector.binary
            && current_quality < target_quality
        {
            let result = self
                .compute_distance(query, quantized_vector, SelectedFormat::Binary)
                .await?;
            final_distance = result.similarity;
            current_quality = result.quality_estimate;
            stages.push("Binary".to_string());

            trace!(
                "Binary stage: distance={:.4}, quality={:.2}",
                final_distance, current_quality
            );
        }

        // Stage 2: INT8 approximation (if available and needed)
        if let Some(_int8_data) = &quantized_vector.int8
            && current_quality < target_quality
        {
            let result = self
                .compute_distance(query, quantized_vector, SelectedFormat::INT8)
                .await?;
            final_distance = result.similarity;
            current_quality = result.quality_estimate;
            stages.push("INT8".to_string());

            trace!(
                "INT8 stage: distance={:.4}, quality={:.2}",
                final_distance, current_quality
            );
        }

        // Stage 3: PQ approximation (if available and needed)
        if let Some(_pq_data) = &quantized_vector.pq
            && current_quality < target_quality
        {
            let result = self
                .compute_distance(query, quantized_vector, SelectedFormat::PQ)
                .await?;
            final_distance = result.similarity;
            current_quality = result.quality_estimate;
            stages.push("PQ".to_string());

            trace!(
                "PQ stage: distance={:.4}, quality={:.2}",
                final_distance, current_quality
            );
        }

        // Stage 4: Full precision (if available and needed)
        if let Some(_fp32_data) = &quantized_vector.fp32
            && current_quality < target_quality
        {
            // For FP32, we need to use the distance engine directly
            let fp32_vector = quantized_vector
                .fp32
                .as_ref()
                .ok_or_else(|| anyhow!("FP32 data not available"))?;
            let similarity = self.distance_engine.calculate_distance(
                query,
                fp32_vector,
                &DistanceMetric::Cosine, // Use default metric
            );
            let result = QuantizedDistanceResult {
                similarity: similarity.normalized_score,
                quality_estimate: 1.0,
                method: ComputationMethod::ExactFP32,
                metrics: DistanceMetrics::default(),
            };
            final_distance = result.similarity;
            current_quality = result.quality_estimate;
            stages.push("FP32".to_string());

            trace!(
                "FP32 stage: distance={:.4}, quality={:.2}",
                final_distance, current_quality
            );
        }

        let computation_time = start_time.elapsed().as_secs_f64() * 1_000_000.0;

        info!(
            "Progressive computation completed in {:.2}μs with {} stages, final quality: {:.2}",
            computation_time,
            stages.len(),
            current_quality
        );

        Ok(QuantizedDistanceResult {
            similarity: final_distance,
            quality_estimate: current_quality,
            method: ComputationMethod::ProgressiveRefinement { stages },
            metrics: DistanceMetrics {
                computation_time_us: computation_time,
                simd_used: self.config.simd_optimization.enable_simd,
                cache_hits: 0, // Deferred: Aggregate from stages
                cache_misses: 0,
                memory_bandwidth_mb_s: self
                    .estimate_memory_bandwidth(query.len(), SelectedFormat::FP32),
                operation_count: self.estimate_operation_count(query.len(), SelectedFormat::FP32),
            },
        })
    }

    /// Compute binary distance using Hamming distance
    fn compute_binary_distance(&self, query: &[f32], binary_data: &[u8]) -> Result<f32> {
        // Convert query to binary representation
        let query_binary = self.quantize_query_to_binary(query)?;

        // Compute Hamming distance
        let hamming_distance =
            if self.hamming_lut.has_popcnt && self.config.simd_optimization.enable_simd {
                self.compute_hamming_distance_simd(&query_binary, binary_data)?
            } else {
                self.compute_hamming_distance_lut(&query_binary, binary_data)?
            };

        // Convert Hamming distance to similarity score
        let max_distance = query.len() as f32;
        let similarity = 1.0 - (hamming_distance as f32 / max_distance);

        // Convert to distance based on metric
        let distance = match self.config.distance_metric {
            DistanceMetric::Cosine => 1.0 - similarity,
            DistanceMetric::Euclidean => hamming_distance as f32,
            DistanceMetric::DotProduct => -similarity, // Higher similarity = lower distance
            _ => hamming_distance as f32,
        };

        Ok(distance)
    }

    /// Compute INT8 distance with caching
    #[allow(dead_code)]
    async fn compute_int8_distance(
        &self,
        query: &[f32],
        int8_data: &Int8VectorData,
    ) -> Result<f32> {
        let cache_key = (self.config.distance_metric, query.len());

        // Check cache for precomputed tables
        let distance_table = {
            let tables = self
                .int8_distance_tables
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tables.tables.get(&cache_key).cloned()
        };

        let table = if let Some(table) = distance_table {
            table
        } else {
            // Create and cache new distance table
            let new_table = Arc::new(Int8DistanceTable::new(
                self.config.distance_metric,
                query.len(),
            )?);

            let mut tables = self
                .int8_distance_tables
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tables.tables.insert(cache_key, new_table.clone());
            tables.memory_usage_bytes += new_table.estimated_size();

            new_table
        };

        // Compute distance using cached table
        self.compute_int8_distance_with_table(query, int8_data, &table)
    }

    /// Compute PQ distance with distance table caching
    #[allow(dead_code)]
    async fn compute_pq_distance(
        &self,
        query: &[f32],
        pq_data: &PQVectorData,
    ) -> Result<(f32, bool)> {
        let codebook_hash = pq_data.codebook_hash;

        // Check cache for precomputed distance table
        let distance_table = {
            let cache = self
                .pq_distance_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.tables.get(&codebook_hash).cloned()
        };

        let (table, cache_hit) = if let Some(table) = distance_table {
            // Update access count for LFU eviction
            table
                .access_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (table, true)
        } else {
            // Create new distance table
            let new_table = Arc::new(PQDistanceTable::new(
                query,
                &pq_data.codebook,
                self.config.distance_metric,
            )?);

            // Cache the table
            {
                let mut cache = self
                    .pq_distance_cache
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                cache.tables.insert(codebook_hash, new_table.clone());
                cache.memory_usage_bytes += new_table.estimated_size();
                cache.misses += 1;

                // Evict if cache is too large
                if cache.memory_usage_bytes
                    > self.config.cache_config.max_cache_size_mb * 1024 * 1024
                {
                    self.evict_pq_cache_entries(&mut cache);
                }
            }

            (new_table, false)
        };

        if cache_hit {
            let mut cache = self
                .pq_distance_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.hits += 1;
        }

        // Compute distance using cached table
        let distance = self.compute_pq_distance_with_table(&pq_data.codes, &table)?;

        Ok((distance, cache_hit))
    }

    /// Helper functions for various distance computations
    fn should_use_simd(&self, dimension: usize) -> bool {
        self.config.simd_optimization.enable_simd
            && dimension >= self.config.simd_optimization.simd_threshold
            && self.has_simd
    }

    /// Return true when the batch is large enough to benefit from SIMD batch processing.
    /// Compute batch distances for columnar storage operations
    /// This is the centralized method for all columnar engines (VIPER, NOVA)
    pub async fn compute_columnar_batch_distances(
        &self,
        query_vector: &[f32],
        quantized_vectors: &[QuantizedVectorData],
        format_preference: SelectedFormat,
    ) -> Result<Vec<(f32, SelectedFormat)>> {
        let mut results = Vec::with_capacity(quantized_vectors.len());

        // Per-vector computation. The former `chunks(64)` "batch" branch made
        // exactly the same per-vector calls through one extra iterator level —
        // no kernel, no parallelism, no cache effect — so the distinction is
        // removed (TD-FACADE-1). Same SIMD note as compute_batch_distances.
        for vector_data in quantized_vectors {
            let result = self
                .compute_distance(query_vector, vector_data, format_preference.clone())
                .await?;
            results.push((result.similarity, format_preference.clone()));
        }

        Ok(results)
    }

    /// Select the highest-fidelity format available in the quantized vector data.
    fn select_best_format(&self, vector: &QuantizedVectorData) -> SelectedFormat {
        if vector.fp32.is_some() {
            SelectedFormat::FP32
        } else if vector.int8.is_some() {
            SelectedFormat::INT8
        } else if vector.pq.is_some() {
            SelectedFormat::PQ
        } else if vector.binary.is_some() {
            SelectedFormat::Binary
        } else {
            // Default to FP32 if no quantized data available
            // This allows graceful fallback instead of panicking
            tracing::warn!("No quantized data available, falling back to FP32 format");
            SelectedFormat::FP32
        }
    }

    /// Quantize a query vector to binary using the median as the threshold.
    fn quantize_query_to_binary(&self, query: &[f32]) -> Result<Vec<u8>> {
        // Simplified binary quantization - use median threshold
        let median = {
            let mut sorted = query.to_vec();
            // Use total_cmp for safe NaN handling instead of partial_cmp fallback logic
            sorted.sort_by(|a, b| a.total_cmp(b));
            sorted[sorted.len() / 2]
        };

        let mut binary = vec![0u8; query.len().div_ceil(8)];
        for (i, &value) in query.iter().enumerate() {
            if value > median {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                binary[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(binary)
    }

    /// Compute Hamming distance between binary vectors using popcount.
    fn compute_hamming_distance_simd(&self, a: &[u8], b: &[u8]) -> Result<usize> {
        // SIMD Hamming distance computation
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Binary vector length mismatch"));
        }

        let mut distance = 0;
        for (byte_a, byte_b) in a.iter().zip(b.iter()) {
            let xor = byte_a ^ byte_b;
            distance += xor.count_ones() as usize;
        }

        Ok(distance)
    }

    /// Compute Hamming distance between binary vectors using a 256-entry lookup table.
    fn compute_hamming_distance_lut(&self, a: &[u8], b: &[u8]) -> Result<usize> {
        // LUT-based Hamming distance computation
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Binary vector length mismatch"));
        }

        let mut distance = 0;
        for (byte_a, byte_b) in a.iter().zip(b.iter()) {
            let xor = byte_a ^ byte_b;
            distance += self.hamming_lut.hamming_weights[xor as usize] as usize;
        }

        Ok(distance)
    }

    /// Compute distance between a query and an INT8 vector using a precomputed lookup table.
    #[allow(dead_code)]
    fn compute_int8_distance_with_table(
        &self,
        query: &[f32],
        int8_data: &Int8VectorData,
        table: &Int8DistanceTable,
    ) -> Result<f32> {
        // Convert query to INT8 for distance computation
        let query_int8 =
            self.quantize_query_to_int8(query, int8_data.scale, int8_data.zero_point)?;

        // Compute distance using precomputed table
        let mut distance = 0.0;
        for (i, (&q_val, &_d_val)) in query_int8.iter().zip(int8_data.values.iter()).enumerate() {
            if i < table.squared_diff_table.len()
                && (q_val as usize) < table.squared_diff_table[i].len()
            {
                distance += table.squared_diff_table[i][q_val as usize];
            }
        }

        match self.config.distance_metric {
            DistanceMetric::Euclidean => Ok(distance.sqrt()),
            DistanceMetric::Cosine => {
                // Compute cosine distance from squared differences
                // This is a simplified approximation
                Ok(1.0 - (1.0 / (1.0 + distance)))
            }
            _ => Ok(distance),
        }
    }

    /// Compute distance from PQ codes using a precomputed distance table for O(1) lookups.
    #[allow(dead_code)]
    fn compute_pq_distance_with_table(
        &self,
        pq_codes: &[u8],
        table: &PQDistanceTable,
    ) -> Result<f32> {
        if pq_codes.len() != table.num_subvectors {
            return Err(anyhow::anyhow!("PQ code length mismatch"));
        }

        let mut total_distance = 0.0;
        for (subvector_idx, &code) in pq_codes.iter().enumerate() {
            if subvector_idx < table.tables.len()
                && (code as usize) < table.tables[subvector_idx].len()
            {
                total_distance += table.tables[subvector_idx][code as usize];
            }
        }

        Ok(total_distance)
    }

    /// Quantize a float query vector to INT8 using the given scale and zero point.
    fn quantize_query_to_int8(&self, query: &[f32], scale: f32, zero_point: i8) -> Result<Vec<i8>> {
        let mut quantized = Vec::with_capacity(query.len());
        for &value in query {
            // Apply quantization formula: quantized = round(value / scale) + zero_point
            let scaled = (value / scale).round() + zero_point as f32;
            let clamped = scaled.clamp(-128.0, 127.0) as i8;
            quantized.push(clamped);
        }
        Ok(quantized)
    }

    /// Estimate memory bandwidth in KB for a distance operation given dimension and format.
    fn estimate_memory_bandwidth(&self, dimension: usize, format: SelectedFormat) -> f32 {
        let bytes_per_element = match format {
            SelectedFormat::FP32 => 4.0,
            SelectedFormat::INT8 => 1.0,
            SelectedFormat::Binary => 1.0 / 8.0, // Bits to bytes
            SelectedFormat::PQ => 1.0,           // Assuming 8-bit codes
        };

        // Rough estimate based on dimension and data type
        (dimension as f32 * bytes_per_element * 2.0) / 1024.0 // MB/s estimate
    }

    /// Estimate the number of arithmetic operations for a distance computation.
    fn estimate_operation_count(&self, dimension: usize, format: SelectedFormat) -> usize {
        match format {
            SelectedFormat::FP32 => dimension * 2, // Multiply + accumulate
            SelectedFormat::INT8 => dimension,     // Simpler operations
            SelectedFormat::Binary => dimension / 8, // Bit operations
            SelectedFormat::PQ => 16,              // Table lookups
        }
    }

    /// Evict stale entries from the PQ distance table cache based on the configured policy.
    #[allow(dead_code)]
    fn evict_pq_cache_entries(&self, cache: &mut PQDistanceCache) {
        // Bounded ARBITRARY-ORDER eviction — NOT LRU (TD-FACADE-1).
        //
        // This used to `match CacheEvictionPolicy::LRU` and then remove the
        // first 10 HashMap keys, which is iteration order, not recency, while
        // every other policy silently no-opped: a config surface promising
        // policy-directed eviction this cache never implemented. The match is
        // gone; `eviction_policy` is advisory in this calculator.
        //
        // Correctness is unaffected either way: PQ distance tables are pure
        // performance caches, so evicting the "wrong" entries only costs
        // recomputation. Real LRU/FIFO would need recency tracking on insert
        // (roadmap item).
        if cache.tables.len() > 100 {
            let keys_to_remove: Vec<u64> = cache.tables.keys().take(10).cloned().collect();
            for key in keys_to_remove {
                if let Some(table) = cache.tables.remove(&key) {
                    cache.memory_usage_bytes -= table.estimated_size();
                }
            }
        }
    }
}

// Implementation of helper structs
impl HammingLookupTable {
    /// Build a 256-entry popcount lookup table, detecting hardware POPCNT support.
    fn new(has_simd: bool) -> Result<Self> {
        let mut hamming_weights = [0u8; 256];
        for (i, weight) in hamming_weights.iter_mut().enumerate() {
            *weight = (i as u8).count_ones() as u8;
        }

        Ok(Self {
            hamming_weights,
            has_popcnt: has_simd,
        })
    }
}

impl Int8DistanceTable {
    /// Create a new precomputed distance table for INT8 quantized vectors.
    #[allow(dead_code)]
    fn new(distance_metric: DistanceMetric, dimension: usize) -> Result<Self> {
        let mut squared_diff_table = Vec::with_capacity(dimension);

        for _dim in 0..dimension {
            let mut dim_table = Vec::with_capacity(256);
            for i in 0..256 {
                let diff = i as f32 - 128.0; // Center around 0
                dim_table.push(diff * diff);
            }
            squared_diff_table.push(dim_table);
        }

        Ok(Self {
            squared_diff_table,
            distance_metric,
            dimension,
        })
    }

    /// Estimate the heap memory usage of this distance table in bytes.
    #[allow(dead_code)]
    fn estimated_size(&self) -> usize {
        self.dimension * 256 * std::mem::size_of::<f32>()
    }
}

impl PQDistanceTable {
    /// Build a PQ distance table by computing distances from the query to all centroids.
    #[allow(dead_code)]
    fn new(query: &[f32], codebook: &[Vec<f32>], distance_metric: DistanceMetric) -> Result<Self> {
        let num_subvectors = codebook.len();
        let num_centroids = codebook.first().map_or(256, |c| c.len());

        let mut tables = Vec::with_capacity(num_subvectors);

        for (subvector_idx, centroids) in codebook.iter().enumerate() {
            let mut centroid_distances = Vec::with_capacity(num_centroids);
            let subvector_size = query.len() / num_subvectors;
            let query_subvector =
                &query[subvector_idx * subvector_size..(subvector_idx + 1) * subvector_size];

            for centroid in centroids.chunks(subvector_size) {
                let distance = match distance_metric {
                    DistanceMetric::Euclidean => query_subvector
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| (q - c).powi(2))
                        .sum::<f32>(),
                    DistanceMetric::Cosine => {
                        let dot: f32 = query_subvector
                            .iter()
                            .zip(centroid.iter())
                            .map(|(q, c)| q * c)
                            .sum();
                        let norm_q: f32 = query_subvector.iter().map(|q| q * q).sum::<f32>().sqrt();
                        let norm_c: f32 = centroid.iter().map(|c| c * c).sum::<f32>().sqrt();
                        1.0 - (dot / (norm_q * norm_c))
                    }
                    _ => {
                        // Fallback to Euclidean
                        query_subvector
                            .iter()
                            .zip(centroid.iter())
                            .map(|(q, c)| (q - c).powi(2))
                            .sum::<f32>()
                    }
                };
                centroid_distances.push(distance);
            }
            tables.push(centroid_distances);
        }

        Ok(Self {
            tables,
            num_subvectors,
            num_centroids,
            distance_metric,
            timestamp: std::time::Instant::now(),
            access_count: std::sync::atomic::AtomicUsize::new(1),
        })
    }

    /// Estimate the heap memory usage of this PQ distance table in bytes.
    #[allow(dead_code)]
    fn estimated_size(&self) -> usize {
        self.num_subvectors * self.num_centroids * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-FACADE-1 pin: batch entries must equal per-vector computation for
    /// ANY batch size. The former code split on should_use_batch_processing
    /// (>=32 && enable_simd) into two branches whose bodies were identical;
    /// the split (and the SIMD name) is gone. This pins the semantic contract
    /// of the collapse across the old threshold: 8 (<32) and 64 (>=32)
    /// vectors must both match compute_distance exactly.
    #[tokio::test]
    async fn batch_distances_equal_per_vector_across_old_threshold() {
        let calc = QuantizedDistanceCalculator::new(QuantizedDistanceConfig::default())
            .expect("calculator");

        let query = vec![0.5_f32; 8];
        for n in [8_usize, 64] {
            let vectors: Vec<QuantizedVectorData> = (0..n)
                .map(|i| QuantizedVectorData {
                    fp32: Some(vec![i as f32 / 100.0; 8]),
                    binary: None,
                    int8: None,
                    pq: None,
                })
                .collect();

            let batch = calc
                .compute_batch_distances(&query, &vectors, SelectedFormat::FP32)
                .await
                .expect("batch");
            assert_eq!(batch.len(), n);

            for (i, v) in vectors.iter().enumerate() {
                let single = calc
                    .compute_distance(&query, v, SelectedFormat::FP32)
                    .await
                    .expect("single");
                assert_eq!(batch[i].similarity, single.similarity, "n={n} i={i}");
            }
        }
    }

    /// TD-FACADE-1 pin: PQ-table eviction is bounded and drops WHOLE tables
    /// with correct bookkeeping (memory_usage_bytes returns to its pre-insert
    /// value). It is arbitrary-order by design now (NOT LRU) — pinned as
    /// bounded-and-consistent, not ordered, because order is unspecified.
    #[test]
    fn pq_cache_eviction_is_bounded_and_consistent() {
        let mut cache = PQDistanceCache {
            tables: std::collections::HashMap::new(),
            hits: 0,
            misses: 0,
            memory_usage_bytes: 0,
        };
        // 101 tables > the 100 bound -> one eviction round removes 10.
        for k in 0..101_u64 {
            cache.tables.insert(
                k,
                Arc::new(PQDistanceTable {
                    tables: vec![vec![0.5; 256]; 4],
                    num_subvectors: 4,
                    num_centroids: 256,
                    distance_metric: DistanceMetric::Cosine,
                    access_count: std::sync::atomic::AtomicUsize::new(0),
                    timestamp: std::time::Instant::now(),
                }),
            );
        }
        cache.memory_usage_bytes = cache.tables.values().map(|t| t.estimated_size()).sum();

        let before_len = cache.tables.len();
        QuantizedDistanceCalculator::new(QuantizedDistanceConfig::default())
            .expect("calculator")
            .evict_pq_cache_entries(&mut cache);

        assert_eq!(before_len - cache.tables.len(), 10, "exactly 10 evicted");
        let after_sum: usize = cache.tables.values().map(|t| t.estimated_size()).sum();
        assert_eq!(
            cache.memory_usage_bytes, after_sum,
            "bookkeeping must match remaining tables exactly"
        );
    }

    fn init_hardware_capabilities() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
    }

    #[test]
    fn test_fp32_distance_computation() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            init_hardware_capabilities();

            let config = QuantizedDistanceConfig::default();
            let calculator = QuantizedDistanceCalculator::new(config).unwrap();

            let query = vec![1.0, 2.0, 3.0, 4.0];
            let quantized_data = QuantizedVectorData {
                fp32: Some(vec![1.1, 2.1, 3.1, 4.1]),
                binary: Some(vec![0b10101010, 0b11001100]), // Add binary data for Binary format test
                int8: None,
                pq: None,
            };

            let result = calculator
                .compute_distance(&query, &quantized_data, SelectedFormat::FP32)
                .await
                .unwrap();

            assert!(result.similarity >= 0.0);
            assert_eq!(result.quality_estimate, 1.0);
            assert!(matches!(result.method, ComputationMethod::ExactFP32));
            assert!(result.metrics.computation_time_us > 0.0);
        });
    }

    #[test]
    fn test_binary_quantization() {
        init_hardware_capabilities();

        let config = QuantizedDistanceConfig::default();
        let calculator = QuantizedDistanceCalculator::new(config).unwrap();

        let query = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5];
        let binary = calculator.quantize_query_to_binary(&query).unwrap();

        // Should create binary representation based on median threshold
        assert_eq!(binary.len(), 1); // 8 bits = 1 byte
    }

    #[test]
    fn test_hamming_lookup_table() {
        init_hardware_capabilities();

        let lut = HammingLookupTable::new(best_simd_level() > SimdLevel::Scalar).unwrap();

        // Test known values
        assert_eq!(lut.hamming_weights[0], 0); // 0000_0000
        assert_eq!(lut.hamming_weights[1], 1); // 0000_0001
        assert_eq!(lut.hamming_weights[255], 8); // 1111_1111
        assert_eq!(lut.hamming_weights[85], 4); // 0101_0101
    }

    #[tokio::test]
    async fn test_progressive_distance_computation() {
        init_hardware_capabilities();

        let config = QuantizedDistanceConfig {
            approximation: ApproximationConfig {
                enable_progressive_refinement: true,
                quality_factor: 0.9,
                ..Default::default()
            },
            ..Default::default()
        };

        let calculator = QuantizedDistanceCalculator::new(config).unwrap();

        let query = vec![1.0; 128];
        let quantized_data = QuantizedVectorData {
            fp32: Some(vec![1.1; 128]),
            binary: Some(vec![0xFF; 16]), // 128 bits = 16 bytes
            int8: Some(Int8VectorData {
                values: vec![100; 128],
                scale: 0.01,
                zero_point: 0,
                length_renorm: None,
            }),
            pq: None,
        };

        let result = calculator
            .compute_progressive_distance(&query, &quantized_data, 0.95)
            .await
            .unwrap();

        assert!(result.quality_estimate >= 0.9);
        if let ComputationMethod::ProgressiveRefinement { stages } = result.method {
            assert!(!stages.is_empty());
        } else {
            panic!("Expected progressive refinement method");
        }
    }

    #[test]
    fn test_selected_format_options() {
        // Test that all format options are available
        let formats = [
            SelectedFormat::FP32,
            SelectedFormat::Binary,
            SelectedFormat::INT8,
            SelectedFormat::PQ,
        ];

        assert_eq!(formats.len(), 4);

        // Test format equality
        assert_eq!(SelectedFormat::FP32, SelectedFormat::FP32);
        assert_ne!(SelectedFormat::FP32, SelectedFormat::Binary);
    }

    // ------------------------------------------------------------------
    // TurboQuantVectorData (P1 — ADR-021 / TURBOQUANT_LLD_2026_05_30)
    // ------------------------------------------------------------------

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_vector_data_construction() {
        // d=1536, bit_width=2 → ceil(1536*2/8) = 384 bytes of codes.
        let codes_len = (1536 * 2usize).div_ceil(8);
        let data = TurboQuantVectorData {
            codes: vec![0u8; codes_len],
            bit_width: 2,
            scale: 1.0,
            encoded_epoch: 1,
        };
        assert_eq!(data.codes.len(), 384);
        assert_eq!(data.bit_width, 2);
        assert_eq!(data.scale, 1.0);
        assert_eq!(data.encoded_epoch, 1);

        // 4-bit: ceil(1536*4/8) = 768 bytes.
        let data4 = TurboQuantVectorData {
            codes: vec![0u8; 768],
            bit_width: 4,
            scale: 0.5,
            encoded_epoch: 12,
        };
        assert_eq!(data4.codes.len(), 768);
        assert_eq!(data4.bit_width, 4);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_vector_data_clone_preserves_fields() {
        let original = TurboQuantVectorData {
            codes: vec![0x55, 0xAA, 0xFF, 0x00],
            bit_width: 4,
            scale: std::f32::consts::PI,
            encoded_epoch: 42,
        };
        let cloned = original.clone();
        assert_eq!(cloned.codes, original.codes);
        assert_eq!(cloned.bit_width, original.bit_width);
        assert_eq!(cloned.scale.to_bits(), original.scale.to_bits());
        assert_eq!(cloned.encoded_epoch, original.encoded_epoch);
    }

    // ------------------------------------------------------------------
    // P7 — RaBitQ length renormalization for existing PQ/SQ
    // ------------------------------------------------------------------

    #[test]
    fn test_int8_default_length_renorm_is_none() {
        // Strict additivity check: existing constructors that don't opt
        // in get the legacy behaviour. Pre-P7 callers stay unaffected.
        let d = Int8VectorData {
            values: vec![1i8, 2, 3, 4],
            scale: 0.5,
            zero_point: 0,
            length_renorm: None,
        };
        assert!(d.length_renorm.is_none());
    }

    #[test]
    fn test_int8_with_length_renorm_builder() {
        let d = Int8VectorData {
            values: vec![1i8, 2, 3, 4],
            scale: 0.5,
            zero_point: 0,
            length_renorm: None,
        }
        .with_length_renorm(1.07);
        assert_eq!(d.length_renorm, Some(1.07));
    }

    #[test]
    fn test_int8_compute_length_renorm_unit_input() {
        // Vector pointing exactly at its centroid → inner = ||v||^2,
        // scale = ||v|| / ||v||^2 = 1 / ||v||.
        let v = vec![1.0f32, 0.0, 0.0, 0.0]; // ||v|| = 1
        let x_hat = vec![1.0f32, 0.0, 0.0, 0.0];
        let scale = Int8VectorData::compute_length_renorm(&v, &x_hat).unwrap();
        assert!((scale - 1.0).abs() < 1e-6, "scale = {scale}");
    }

    #[test]
    fn test_int8_compute_length_renorm_with_quantization_shrinkage() {
        // Vector v = [1, 0.5], quantized to x_hat = [0.9, 0.4]
        //   ||v|| = sqrt(1.25), <v, x_hat> = 0.9 + 0.2 = 1.1
        //   scale = sqrt(1.25) / 1.1 ≈ 1.016
        let v = vec![1.0f32, 0.5];
        let x_hat = vec![0.9f32, 0.4];
        let scale = Int8VectorData::compute_length_renorm(&v, &x_hat).unwrap();
        assert!((scale - 1.0163).abs() < 1e-3, "scale = {scale}");
    }

    #[test]
    fn test_int8_compute_length_renorm_zero_vector_returns_none() {
        let v = vec![0.0f32; 8];
        let x_hat = vec![0.0f32; 8];
        assert!(Int8VectorData::compute_length_renorm(&v, &x_hat).is_none());
    }

    #[test]
    fn test_int8_compute_length_renorm_orthogonal_returns_none() {
        let v = vec![1.0f32, 0.0];
        let x_hat = vec![0.0f32, 1.0];
        // <v, x_hat> = 0 → degenerate, no correction possible.
        assert!(Int8VectorData::compute_length_renorm(&v, &x_hat).is_none());
    }

    #[test]
    fn test_int8_compute_length_renorm_length_mismatch_returns_none() {
        let v = vec![1.0f32, 0.5, 0.3];
        let x_hat = vec![0.9f32, 0.4];
        assert!(Int8VectorData::compute_length_renorm(&v, &x_hat).is_none());
    }

    #[test]
    fn test_pq_default_length_renorm_is_none() {
        let d = PQVectorData {
            codes: vec![1u8, 2, 3],
            codebook: vec![vec![0.0; 4]; 3],
            codebook_hash: 0,
            length_renorm: None,
        };
        assert!(d.length_renorm.is_none());
    }

    #[test]
    fn test_pq_with_length_renorm_builder() {
        let d = PQVectorData {
            codes: vec![1u8, 2, 3],
            codebook: vec![vec![0.0; 4]; 3],
            codebook_hash: 0,
            length_renorm: None,
        }
        .with_length_renorm(0.94);
        assert_eq!(d.length_renorm, Some(0.94));
    }

    #[test]
    fn test_apply_length_renorm_no_op_when_none() {
        let out = apply_length_renorm(0.7, None, &DistanceMetric::Cosine);
        assert_eq!(out, 0.7);
    }

    #[test]
    fn test_apply_length_renorm_multiplies_for_cosine() {
        let out = apply_length_renorm(0.7, Some(1.05), &DistanceMetric::Cosine);
        assert!((out - 0.735).abs() < 1e-6);
    }

    #[test]
    fn test_apply_length_renorm_multiplies_for_dot_product() {
        let out = apply_length_renorm(0.7, Some(1.05), &DistanceMetric::DotProduct);
        assert!((out - 0.735).abs() < 1e-6);
    }

    #[test]
    fn test_apply_length_renorm_skips_euclidean() {
        // Euclidean is distance-shaped — multiplying by a scalar would
        // not produce the intended bias correction. Helper falls through.
        let out = apply_length_renorm(0.7, Some(1.05), &DistanceMetric::Euclidean);
        assert_eq!(out, 0.7);
    }
}
