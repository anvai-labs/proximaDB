//! Canonical quantization-level model (Slice D pre-extraction): the UnifiedQuantizationLevel
//! cluster (the rich QuantizationScheme enum + Product/Scalar/Binary/Uniform/No/Custom/Turbo
//! variant payloads + their QuantizationMethod impls). Moved DOWN from the modality vector
//! crate to foundation so both storage and the modality depend on it (clears the divergent-
//! types trap + the storage up-edges). Re-exported by proximadb-vector/src/quantization/internal_types.rs.

// Re-export the contract traits/types the cluster depends on so they flow through
// the re-export chain (a `use proximadb_quantization_model::*` — including the
// `pub use` in proximadb-vector/internal_types.rs — brings QuantizationMethod etc.
// into scope, which call sites need to resolve trait methods like .quantization_type()).
pub use proximadb_quantization_types::{
    DurableQuantState, QuantizationLifecycle, QuantizationMethod, QuantizationType,
};
use serde::{Deserialize, Serialize};

/// Unified quantization level configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnifiedQuantizationLevel {
    pub level_type: Option<QuantizationScheme>,
}

/// Quantization level types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    None(NoQuantization),
    Uniform(UniformQuantization),
    Pq(ProductQuantization),
    Scalar(ScalarQuantization),
    Binary(BinaryQuantization),
    Custom(CustomQuantization),
    /// TurboQuant — data-oblivious scalar quantizer (ADR-021).
    /// See `TURBOQUANT_LLD_2026_05_30.adoc` §"Locked Type Signatures".
    #[cfg(feature = "experimental-turboquant")]
    TurboQuant(TurboQuantization),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NoQuantization {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UniformQuantization {
    pub bits: i32,
    pub scale: Option<f32>,
    pub offset: Option<f32>,
}

impl Eq for UniformQuantization {}

impl std::hash::Hash for UniformQuantization {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
        // Hash the bits representation of the floats for consistency
        self.scale.map(|s| s.to_bits()).hash(state);
        self.offset.map(|o| o.to_bits()).hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProductQuantization {
    pub bits_per_code: i32,
    pub num_subvectors: i32,
    pub codebook_id: Option<String>,
    pub adaptive_subvectors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScalarQuantization {
    pub bits: i32,
    pub scale: f32,
    pub offset: f32,
    pub clamp_values: bool,
}

impl Eq for ScalarQuantization {}

impl std::hash::Hash for ScalarQuantization {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
        self.scale.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.clamp_values.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryQuantization {
    pub threshold: Option<f32>,
    pub sign_based: bool,
}

impl Eq for BinaryQuantization {}

impl std::hash::Hash for BinaryQuantization {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.threshold.map(|t| t.to_bits()).hash(state);
        self.sign_based.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomQuantization {
    pub type_id: String,
    pub bits_per_element: i32,
    pub config: std::collections::HashMap<String, String>,
}

impl std::hash::Hash for CustomQuantization {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.bits_per_element.hash(state);
        // Hash the config map in a deterministic way
        let mut sorted_config: Vec<_> = self.config.iter().collect();
        sorted_config.sort_by_key(|(k, _)| k.as_str());
        for (k, v) in sorted_config {
            k.hash(state);
            v.hash(state);
        }
    }
}

/// TurboQuant configuration (ADR-021, TURBOQUANT_LLD_2026_05_30 §"Locked Type Signatures").
///
/// The three fields are everything a search path needs to dispatch:
/// - `bit_width` ∈ {2, 4} for P1; {2, 3, 4} after P10.
/// - `calibration_mode`: identity vs TQ+ per-coord. Frozen after first
///   qualifying batch when TqPlus is selected. See [`CalibrationMode`].
/// - `rotation_seed`: per-collection, derived via
///   `proximadb_quantization_types::derive_rotation_seed(collection_id)`
///   on first enablement, then persisted in xCatalog (P8).
///
/// The struct itself owns no buffers — codes, scales, and TQ+ calibration
/// vectors are persisted in the `.tq` file (LLD §3) and loaded into
/// `TurboQuantVectorData` (root crate) at search time.
#[cfg(feature = "experimental-turboquant")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TurboQuantization {
    /// Bit-width per coordinate. 2 or 4 in P1.
    pub bit_width: u8,
    /// Per-coord calibration mode. See foundation's `CalibrationMode`.
    pub calibration_mode: proximadb_quantization_types::CalibrationMode,
    /// Per-collection rotation seed. Derived from collection id at first
    /// enablement, then immutable for the collection's lifetime (until an
    /// epoch bump triggers a full re-encode).
    pub rotation_seed: u64,
}

#[cfg(feature = "experimental-turboquant")]
impl TurboQuantization {
    /// Construct a TurboQuant config with identity calibration. Use when
    /// the collection is too small for TQ+ or when explicitly disabling it.
    pub fn identity(bit_width: u8, rotation_seed: u64) -> Self {
        Self {
            bit_width,
            calibration_mode: proximadb_quantization_types::CalibrationMode::Identity,
            rotation_seed,
        }
    }

    /// Construct a TurboQuant config with TQ+ calibration. The calibration
    /// itself is fit lazily on the first qualifying batch (≥ 1000 vectors).
    pub fn tq_plus(bit_width: u8, rotation_seed: u64) -> Self {
        Self {
            bit_width,
            calibration_mode: proximadb_quantization_types::CalibrationMode::TqPlus,
            rotation_seed,
        }
    }
}

#[allow(non_upper_case_globals)]
impl UnifiedQuantizationLevel {
    /// Common quantization level constants for easy access
    /// Using PascalCase for API consistency with enum variant style
    pub const Binary: Self = Self {
        level_type: Some(QuantizationScheme::Binary(BinaryQuantization {
            threshold: None,
            sign_based: false,
        })),
    };

    pub const Int8: Self = Self {
        level_type: Some(QuantizationScheme::Scalar(ScalarQuantization {
            bits: 8,
            scale: 1.0,
            offset: 0.0,
            clamp_values: true,
        })),
    };

    /// Create a PQ4 constant (requires runtime initialization due to parameter)
    pub const Pq4: Self = Self {
        level_type: Some(QuantizationScheme::Pq(ProductQuantization {
            bits_per_code: 4,
            num_subvectors: 8, // Default value
            codebook_id: None,
            adaptive_subvectors: false,
        })),
    };

    /// Create a PQ8 constant (requires runtime initialization due to parameter)
    pub const Pq8: Self = Self {
        level_type: Some(QuantizationScheme::Pq(ProductQuantization {
            bits_per_code: 8,
            num_subvectors: 8, // Default value
            codebook_id: None,
            adaptive_subvectors: false,
        })),
    };

    /// Create a PQ8 configuration (common case)
    pub fn pq8(num_subvectors: u8) -> Self {
        Self {
            level_type: Some(QuantizationScheme::Pq(ProductQuantization {
                bits_per_code: 8,
                num_subvectors: num_subvectors as i32,
                codebook_id: None,
                adaptive_subvectors: false,
            })),
        }
    }

    /// Create a PQ4 configuration (higher compression)
    pub fn pq4(num_subvectors: u8) -> Self {
        Self {
            level_type: Some(QuantizationScheme::Pq(ProductQuantization {
                bits_per_code: 4,
                num_subvectors: num_subvectors as i32,
                codebook_id: None,
                adaptive_subvectors: false,
            })),
        }
    }

    /// Create an INT8 scalar quantization
    pub fn int8() -> Self {
        Self {
            level_type: Some(QuantizationScheme::Scalar(ScalarQuantization {
                bits: 8,
                scale: 1.0,
                offset: 0.0,
                clamp_values: true,
            })),
        }
    }

    /// Create a binary quantization
    pub fn binary() -> Self {
        Self {
            level_type: Some(QuantizationScheme::Binary(BinaryQuantization {
                threshold: None,
                sign_based: false,
            })),
        }
    }

    /// Get the number of bits per element
    pub fn bits_per_element(&self) -> u32 {
        match &self.level_type {
            Some(QuantizationScheme::Pq(pq)) => pq.bits_per_code as u32,
            Some(QuantizationScheme::Scalar(sq)) => sq.bits as u32,
            Some(QuantizationScheme::Binary(_)) => 1,
            Some(QuantizationScheme::Uniform(uq)) => uq.bits as u32,
            Some(QuantizationScheme::Custom(cq)) => cq.bits_per_element as u32,
            #[cfg(feature = "experimental-turboquant")]
            Some(QuantizationScheme::TurboQuant(tq)) => tq.bit_width as u32,
            _ => 32, // Full precision
        }
    }

    /// Calculate bytes per vector based on quantization level
    pub fn bytes_per_vector(&self, dimension: usize) -> usize {
        match &self.level_type {
            Some(QuantizationScheme::Pq(pq)) => {
                let codes_per_vector = pq.num_subvectors as usize;
                let bytes_per_code = ((pq.bits_per_code + 7) / 8) as usize;
                codes_per_vector * bytes_per_code
            }
            Some(QuantizationScheme::Scalar(sq)) => dimension * ((sq.bits + 7) / 8) as usize,
            Some(QuantizationScheme::Binary(_)) => {
                dimension.div_ceil(8) // 1 bit per dimension
            }
            Some(QuantizationScheme::Uniform(uq)) => dimension * ((uq.bits + 7) / 8) as usize,
            Some(QuantizationScheme::Custom(cq)) => {
                dimension * ((cq.bits_per_element + 7) / 8) as usize
            }
            #[cfg(feature = "experimental-turboquant")]
            Some(QuantizationScheme::TurboQuant(tq)) => {
                // ceil(dim * bit_width / 8) — matches LLD §3 wire layout.
                // Plus 4 bytes per vector for the RaBitQ-style length-renorm
                // scale (stored separately in the .tq file but accounted for
                // here so callers get a single bytes-per-vector estimate).
                let bits_per_vec = dimension * tq.bit_width as usize;
                bits_per_vec.div_ceil(8) + 4
            }
            _ => dimension * 4, // Full FP32 precision
        }
    }

    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self, dimension: usize) -> f32 {
        let fp32_bytes = dimension * 4;
        let compressed_bytes = self.bytes_per_vector(dimension);
        fp32_bytes as f32 / compressed_bytes.max(1) as f32
    }

    /// Construct a TurboQuant configuration. Bit width must be 2 or 4 in P1
    /// (3-bit is deferred per LLD Q10).
    #[cfg(feature = "experimental-turboquant")]
    pub fn turboquant(bit_width: u8, rotation_seed: u64) -> Self {
        Self {
            level_type: Some(QuantizationScheme::TurboQuant(TurboQuantization::tq_plus(
                bit_width,
                rotation_seed,
            ))),
        }
    }

    /// Construct a TurboQuant configuration with identity calibration
    /// (no TQ+ per-coord adjustment).
    #[cfg(feature = "experimental-turboquant")]
    pub fn turboquant_identity(bit_width: u8, rotation_seed: u64) -> Self {
        Self {
            level_type: Some(QuantizationScheme::TurboQuant(TurboQuantization::identity(
                bit_width,
                rotation_seed,
            ))),
        }
    }
}

// ============================================================================
// QuantizationMethod trait impls
// (Phase A — Quantization Trait Convergence Plan)
// ============================================================================
//
// These impls bind every modality-side parameter struct to the foundation
// crate's `QuantizationMethod` trait. The trait is metadata-only — every
// method is `&self`, returns a small `Copy` value, and never touches the
// codebook cache or vector buffers. Routers, planners, EXPLAIN builders,
// and Prometheus label emitters call these methods instead of matching on
// the `QuantizationScheme` enum directly. See the plan document at
// `~/.claude/plans/dreamy-finding-clover.md` §"Design rationale" for the
// load-bearing reasoning, and ADR-021 §"Authority mode" for how lifecycle
// classification drives the read-time vs write-time routing decision.

impl QuantizationMethod for NoQuantization {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::None
    }
    fn bit_width(&self) -> u8 {
        // FP32 baseline — the "no quantization" path is 32-bit float.
        32
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::Identity
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        None
    }
    fn metric_label(&self) -> &'static str {
        "none"
    }
}

impl QuantizationMethod for BinaryQuantization {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Binary
    }
    fn bit_width(&self) -> u8 {
        // Binary is always 1 bit per coord.
        1
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::WriteTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        // Binary quantization has no per-collection training state: the
        // threshold (when present) is a fixed config value, not durable
        // state requiring repair. Return `None` so xCatalog stores no row.
        None
    }
    fn metric_label(&self) -> &'static str {
        "binary"
    }
}

impl QuantizationMethod for ScalarQuantization {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Scalar
    }
    fn bit_width(&self) -> u8 {
        // Clamp to u8 — the on-disk schema uses i32 for legacy reasons but
        // values are always in {1..32}.
        self.bits.clamp(1, 32) as u8
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::WriteTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        // Scalar quantization carries scale/offset as parameters, not as
        // per-collection durable state. They live alongside the codes and
        // don't need a catalog row.
        None
    }
    fn metric_label(&self) -> &'static str {
        "scalar"
    }
}

impl QuantizationMethod for UniformQuantization {
    fn quantization_type(&self) -> QuantizationType {
        // Uniform quantization is a scalar variant with optional learned
        // scale/offset. From the router's perspective it's a scalar method.
        QuantizationType::Scalar
    }
    fn bit_width(&self) -> u8 {
        self.bits.clamp(1, 32) as u8
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::WriteTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        None
    }
    fn metric_label(&self) -> &'static str {
        "uniform"
    }
}

impl QuantizationMethod for ProductQuantization {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Product
    }
    fn bit_width(&self) -> u8 {
        self.bits_per_code.clamp(1, 32) as u8
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::WriteTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        // PQ collections persist their codebook id so a restart can rebind
        // codes to the right codebook. encoded_epoch defaults to 0 here;
        // xCatalog overlays the live epoch at hydration time.
        Some(DurableQuantState {
            seed_or_codebook_id: self.codebook_id.clone(),
            calibration: None,
            encoded_epoch: 0,
        })
    }
    fn metric_label(&self) -> &'static str {
        "product"
    }
}

impl QuantizationMethod for CustomQuantization {
    fn quantization_type(&self) -> QuantizationType {
        // Custom variants are tunneled through Scalar at the router level;
        // their `type_id` is opaque to the planner and only meaningful to
        // their own encode/score implementations.
        QuantizationType::Scalar
    }
    fn bit_width(&self) -> u8 {
        self.bits_per_element.clamp(1, 32) as u8
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        QuantizationLifecycle::WriteTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        // The type_id is the closest analogue to a codebook id — preserve
        // it so a restart can pick the right impl. No calibration tunnel.
        Some(DurableQuantState {
            seed_or_codebook_id: Some(self.type_id.clone()),
            calibration: None,
            encoded_epoch: 0,
        })
    }
    fn metric_label(&self) -> &'static str {
        "custom"
    }
}

#[cfg(feature = "experimental-turboquant")]
impl QuantizationMethod for TurboQuantization {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::TurboQuant
    }
    fn bit_width(&self) -> u8 {
        self.bit_width
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        // Per ADR-021 §"Authority mode": TurboQuant codes are computed at
        // read-time from rotation_seed + frozen calibration; codes live in
        // the .tq sidecar, never in the columnar layout.
        QuantizationLifecycle::ReadTime
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        // rotation_seed + calibration_mode together let a restart
        // reconstruct the same encoding deterministically. encoded_epoch
        // is filled by the store at load time (xCatalog overlays the live
        // value at hydration — Phase E of the convergence plan).
        Some(DurableQuantState {
            seed_or_codebook_id: Some(format!("{:#x}", self.rotation_seed)),
            calibration: Some(self.calibration_mode.to_string()),
            encoded_epoch: 0,
        })
    }
    fn supports_candidate_mask(&self) -> bool {
        // TurboQuant's SIMD kernel consumes a packed u64 bitmap for the
        // block-skip fast path. Phase D AXIS adapters dispatch on this.
        true
    }
    fn metric_label(&self) -> &'static str {
        "turboquant"
    }
}

// ----------------------------------------------------------------------------
// Blanket impl on the enum + wrapper — what every router call site holds.
// ----------------------------------------------------------------------------

impl QuantizationMethod for QuantizationScheme {
    fn quantization_type(&self) -> QuantizationType {
        match self {
            Self::None(v) => v.quantization_type(),
            Self::Uniform(v) => v.quantization_type(),
            Self::Pq(v) => v.quantization_type(),
            Self::Scalar(v) => v.quantization_type(),
            Self::Binary(v) => v.quantization_type(),
            Self::Custom(v) => v.quantization_type(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.quantization_type(),
        }
    }
    fn bit_width(&self) -> u8 {
        match self {
            Self::None(v) => v.bit_width(),
            Self::Uniform(v) => v.bit_width(),
            Self::Pq(v) => v.bit_width(),
            Self::Scalar(v) => v.bit_width(),
            Self::Binary(v) => v.bit_width(),
            Self::Custom(v) => v.bit_width(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.bit_width(),
        }
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        match self {
            Self::None(v) => v.lifecycle(),
            Self::Uniform(v) => v.lifecycle(),
            Self::Pq(v) => v.lifecycle(),
            Self::Scalar(v) => v.lifecycle(),
            Self::Binary(v) => v.lifecycle(),
            Self::Custom(v) => v.lifecycle(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.lifecycle(),
        }
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        match self {
            Self::None(v) => v.durable_state(),
            Self::Uniform(v) => v.durable_state(),
            Self::Pq(v) => v.durable_state(),
            Self::Scalar(v) => v.durable_state(),
            Self::Binary(v) => v.durable_state(),
            Self::Custom(v) => v.durable_state(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.durable_state(),
        }
    }
    fn supports_candidate_mask(&self) -> bool {
        match self {
            Self::None(v) => v.supports_candidate_mask(),
            Self::Uniform(v) => v.supports_candidate_mask(),
            Self::Pq(v) => v.supports_candidate_mask(),
            Self::Scalar(v) => v.supports_candidate_mask(),
            Self::Binary(v) => v.supports_candidate_mask(),
            Self::Custom(v) => v.supports_candidate_mask(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.supports_candidate_mask(),
        }
    }
    fn metric_label(&self) -> &'static str {
        match self {
            Self::None(v) => v.metric_label(),
            Self::Uniform(v) => v.metric_label(),
            Self::Pq(v) => v.metric_label(),
            Self::Scalar(v) => v.metric_label(),
            Self::Binary(v) => v.metric_label(),
            Self::Custom(v) => v.metric_label(),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant(v) => v.metric_label(),
        }
    }
}

impl QuantizationMethod for UnifiedQuantizationLevel {
    fn quantization_type(&self) -> QuantizationType {
        // None at the wrapper level means "no quantization configured" —
        // route the same as the explicit `NoQuantization` arm.
        match self.level_type.as_ref() {
            Some(level) => level.quantization_type(),
            None => QuantizationType::None,
        }
    }
    fn bit_width(&self) -> u8 {
        match self.level_type.as_ref() {
            Some(level) => level.bit_width(),
            None => 32,
        }
    }
    fn lifecycle(&self) -> QuantizationLifecycle {
        match self.level_type.as_ref() {
            Some(level) => level.lifecycle(),
            None => QuantizationLifecycle::Identity,
        }
    }
    fn durable_state(&self) -> Option<DurableQuantState> {
        self.level_type.as_ref().and_then(|l| l.durable_state())
    }
    fn supports_candidate_mask(&self) -> bool {
        self.level_type
            .as_ref()
            .is_some_and(|l| l.supports_candidate_mask())
    }
    fn metric_label(&self) -> &'static str {
        match self.level_type.as_ref() {
            Some(level) => level.metric_label(),
            None => "none",
        }
    }
}

// ============================================================================
// Quantized data types (moved from the modality kernel — Slice D pre-extraction).
// Foundation-pure (Vec<u8> + the UnifiedQuantizationLevel above + primitive
// metadata). Shared by storage (data-type refs) + the kernel. Pre-extracting them
// here makes the upcoming StorageQuantizationEnginePort facade-FREE (the port's
// return types are these foundation types, not modality types).
// ============================================================================

/// Metadata for quantized vectors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantEngineQuantizationMetadata {
    /// Reference to codebook (for PQ)
    pub codebook_id: Option<String>,
    /// Scale factor (for scalar/uniform)
    pub scale: Option<f32>,
    /// Offset (for scalar/uniform)
    pub offset: Option<f32>,
    /// Original vector norm (useful for some metrics)
    pub norm: Option<f32>,
}

/// Backwards-compat alias for [`QuantEngineQuantizationMetadata`].
pub type QuantizationMetadata = QuantEngineQuantizationMetadata;

/// Quantized vector representation
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// The quantized data
    pub data: Vec<u8>,
    /// Quantization level used
    pub quantization_level: UnifiedQuantizationLevel,
    /// Additional metadata (scale, offset, codebook reference)
    pub metadata: QuantEngineQuantizationMetadata,
}

/// Common quantized data structure for storage
#[derive(Debug, Clone)]
pub struct StorageQuantizedData {
    /// Vector ID
    pub id: String,
    /// Primary quantization (e.g., PQ codes for ranking)
    pub primary: Option<QuantizedVector>,
    /// Filter quantization (e.g., binary sketch for filtering)
    pub filter: Option<QuantizedVector>,
    /// Fast quantization (e.g., INT8 for quick distance)
    pub fast: Option<QuantizedVector>,
    /// Original dimension
    pub dimension: usize,
    /// Metadata about quantization quality
    pub metadata: QuantizationMetadata,
}

/// Common configuration for storage engine quantization (foundation-pure data
/// struct; moved from modality-tier `proximadb-quantization-kernel`'s
/// `storage_engine` module so storage-tier crates can hold it without a
/// storage→modality up-edge — same pattern as `StorageQuantizedData` above).
/// Re-exported from the kernel crate so its public API is unchanged.
#[derive(Debug, Clone)]
pub struct StorageQuantizationConfig {
    /// Base quantization levels to use
    pub primary_level: Option<UnifiedQuantizationLevel>, // e.g., PQ8
    pub filter_level: Option<UnifiedQuantizationLevel>, // e.g., Binary
    pub fast_level: Option<UnifiedQuantizationLevel>,   // e.g., INT8

    /// Distance metric to use for quantization (affects PQ code generation)
    pub distance_metric: proximadb_distance_kernel::engine::DistanceMetric,

    /// Progressive resolution settings
    pub enable_progressive: bool,
    pub filter_threshold: f32, // Hamming distance threshold for binary filtering
    pub candidate_multiplier: usize, // How many candidates to keep at each stage

    /// Quality settings
    pub training_sample_size: usize,

    /// Resource settings
    pub memory_budget_mb: usize,
    pub enable_hardware_acceleration: bool,
}

impl Default for StorageQuantizationConfig {
    fn default() -> Self {
        Self {
            // INT8 as default primary - fast, no training required, good compression
            // PQ can be explicitly enabled in collection config when needed
            primary_level: Some(UnifiedQuantizationLevel::int8()),
            // Binary sketch for filtering (1-bit per dimension)
            filter_level: Some(UnifiedQuantizationLevel::binary()),
            // INT8 for fast approximation
            fast_level: Some(UnifiedQuantizationLevel::int8()),

            // Default to Cosine distance (most common for embeddings)
            distance_metric: proximadb_distance_kernel::engine::DistanceMetric::Cosine,

            enable_progressive: true,
            filter_threshold: 0.3,
            candidate_multiplier: 10,

            training_sample_size: 10000,

            memory_budget_mb: 1024,
            enable_hardware_acceleration: true,
        }
    }
}
