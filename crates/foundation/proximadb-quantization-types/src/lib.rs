//! # ProximaDB Quantization Types
//!
//! Foundation quantization types for ProximaDB.
//!
//! ## Purpose
//!
//! This crate provides the single source of truth for quantization types
//! across the entire ProximaDB codebase. It eliminates the proliferation of
//! duplicate quantization definitions (30+ found in audit).
//!
//! ## Types
//!
//! - [`QuantizationType`] - Standardized quantization type enum
//! - [`QuantizationLevel`] - Standardized quantization level enum
//! - [`QuantizationConfig`] - Configuration for quantization
//!
//! ## Migration
//!
//! If you're using legacy quantization types, migrate to this crate's types
//! using the provided conversion traits.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Standardized quantization type enum.
///
/// This is the single source of truth for quantization types across ProximaDB.
/// All other quantization type definitions should migrate to use this enum.
///
/// ## Variants
///
/// - `None` - No quantization
/// - `Scalar` - Scalar quantization (per-vector scalar quantization)
/// - `Product` - Product quantization (PQ)
/// - `Binary` - Binary quantization
/// - `TurboQuant` (feature `experimental-turboquant`) - Data-oblivious scalar
///   quantizer per ADR-021. Online ingest, no codebook training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    /// No quantization
    #[default]
    None,

    /// Scalar quantization
    ///
    /// Also known as: Scalar, PQScalar, Int8Quantization
    Scalar,

    /// Product quantization
    ///
    /// Also known as: PQ, ProductQuantization, OPQ
    Product,

    /// Binary quantization
    ///
    /// Also known as: Binary, BitQuantization
    Binary,

    /// TurboQuant — data-oblivious scalar quantizer (ADR-021, arXiv:2504.19874).
    /// Online ingest, no codebook training. See TURBOQUANT_HLD_2026_05_30.
    #[cfg(feature = "experimental-turboquant")]
    TurboQuant,
}

impl fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Scalar => write!(f, "scalar"),
            Self::Product => write!(f, "product"),
            Self::Binary => write!(f, "binary"),
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant => write!(f, "turboquant"),
        }
    }
}

impl QuantizationType {
    /// Create from string representation
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl FromStr for QuantizationType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "no" => Some(Self::None),
            "scalar" | "int8" | "uint8" | "scalarquantization" => Some(Self::Scalar),
            "product" | "pq" | "productquantization" => Some(Self::Product),
            "binary" | "bit" | "binaryquantization" => Some(Self::Binary),
            #[cfg(feature = "experimental-turboquant")]
            "turboquant" | "tq" | "tq+" | "turbo" => Some(Self::TurboQuant),
            _ => None,
        }
        .ok_or(())
    }
}

impl QuantizationType {
    /// Check if this quantization type uses fixed-point arithmetic
    pub fn is_fixed_point(&self) -> bool {
        #[cfg(feature = "experimental-turboquant")]
        {
            // TurboQuant codes are 2/3/4-bit integers — fixed-point.
            return matches!(self, Self::Scalar | Self::Binary | Self::TurboQuant);
        }
        #[cfg(not(feature = "experimental-turboquant"))]
        {
            matches!(self, Self::Scalar | Self::Binary)
        }
    }

    /// Check if this quantization type uses floating-point arithmetic
    pub fn is_floating_point(&self) -> bool {
        !self.is_fixed_point() && *self != Self::None
    }
}

/// Standardized quantization level enum.
///
/// This represents the bit-width or precision level for quantization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationLevel {
    /// No quantization
    #[default]
    None,

    /// 4-bit quantization
    Int4,

    /// 8-bit quantization (signed)
    Int8,

    /// 8-bit quantization (unsigned)
    UInt8,

    /// 16-bit floating point
    FP16,

    /// 32-bit floating point (no quantization)
    FP32,
}

impl fmt::Display for QuantizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Int4 => write!(f, "int4"),
            Self::Int8 => write!(f, "int8"),
            Self::UInt8 => write!(f, "uint8"),
            Self::FP16 => write!(f, "fp16"),
            Self::FP32 => write!(f, "fp32"),
        }
    }
}

impl QuantizationLevel {
    /// Create from string representation
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl FromStr for QuantizationLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "no" => Some(Self::None),
            "int4" | "4bit" => Some(Self::Int4),
            "int8" | "8bit" | "signed8" => Some(Self::Int8),
            "uint8" | "unsigned8" => Some(Self::UInt8),
            "fp16" | "float16" | "half" => Some(Self::FP16),
            "fp32" | "float32" => Some(Self::FP32),
            _ => None,
        }
        .ok_or(())
    }
}

impl QuantizationLevel {
    /// Get the bit width for this quantization level
    pub fn bit_width(&self) -> usize {
        match self {
            Self::None => 32,
            Self::Int4 => 4,
            Self::Int8 | Self::UInt8 => 8,
            Self::FP16 => 16,
            Self::FP32 => 32,
        }
    }

    /// Check if this is a signed integer type
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, Self::Int4 | Self::Int8)
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, Self::UInt8)
    }

    /// Check if this is a floating-point type
    pub fn is_floating_point(&self) -> bool {
        matches!(self, Self::FP16 | Self::FP32)
    }
}

/// Configuration for quantization
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization type
    pub quantization_type: QuantizationType,

    /// Quantization level
    pub quantization_level: QuantizationLevel,

    /// Whether to use asymmetric quantization
    pub asymmetric: bool,

    /// Whether to cache quantization results
    pub cache: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationConfig {
    /// Create a new quantization config with no quantization
    pub fn new() -> Self {
        Self {
            quantization_type: QuantizationType::None,
            quantization_level: QuantizationLevel::None,
            asymmetric: false,
            cache: false,
        }
    }

    /// Create a scalar quantization config
    pub fn scalar(level: QuantizationLevel) -> Self {
        Self {
            quantization_type: QuantizationType::Scalar,
            quantization_level: level,
            asymmetric: false,
            cache: false,
        }
    }

    /// Create a product quantization config
    pub fn product(level: QuantizationLevel) -> Self {
        Self {
            quantization_type: QuantizationType::Product,
            quantization_level: level,
            asymmetric: false,
            cache: false,
        }
    }

    /// Create a binary quantization config
    pub fn binary() -> Self {
        Self {
            quantization_type: QuantizationType::Binary,
            quantization_level: QuantizationLevel::Int4, // Binary uses 1 bit per value
            asymmetric: false,
            cache: false,
        }
    }

    /// Enable asymmetric quantization
    pub fn with_asymmetric(mut self) -> Self {
        self.asymmetric = true;
        self
    }

    /// Enable caching
    pub fn with_cache(mut self) -> Self {
        self.cache = true;
        self
    }

    /// Get the quantization type
    pub fn quantization_type(&self) -> QuantizationType {
        self.quantization_type
    }

    /// Get the quantization level
    pub fn quantization_level(&self) -> QuantizationLevel {
        self.quantization_level
    }
}

// ============================================================================
// TurboQuant types (ADR-021, TURBOQUANT_LLD_2026_05_30)
// ============================================================================
//
// Gated by `experimental-turboquant`. These types are the foundation surface
// used by the modality crate's `QuantizationLevel::TurboQuant` variant and
// the root crate's `TurboQuantVectorData` struct. They are intentionally
// minimal: bit-width, calibration mode discriminator, and per-collection
// rotation seed. All algorithm logic (Lloyd-Max, rotation, encode pipeline,
// SIMD kernels) lives in `crates/modalities/proximadb-vector/src/quantization/
// turboquant/` and lands in P2+.

/// Per-coordinate calibration mode for TurboQuant. See LLD §6 (Q6, Q7).
#[cfg(feature = "experimental-turboquant")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMode {
    /// Identity calibration — no TQ+ per-coord adjustment. Used when the first
    /// batch is below `TQPLUS_MIN_SAMPLES` (1000), or when the operator
    /// explicitly opts out. EXPLAIN surfaces `calibration_mode="identity"`.
    #[default]
    Identity,

    /// TQ+ per-coord `(shift, scale)` fit from empirical 5/95% quantiles of
    /// the first qualifying batch. Frozen after the first fit so all future
    /// adds quantize against the same target distribution. See LLD §6.
    TqPlus,
}

#[cfg(feature = "experimental-turboquant")]
impl fmt::Display for CalibrationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => write!(f, "identity"),
            Self::TqPlus => write!(f, "tq_plus"),
        }
    }
}

#[cfg(feature = "experimental-turboquant")]
impl FromStr for CalibrationMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "identity" | "none" | "off" => Ok(Self::Identity),
            "tq_plus" | "tqplus" | "tq+" | "plus" => Ok(Self::TqPlus),
            _ => Err(()),
        }
    }
}

/// Derive the per-collection TurboQuant rotation seed from a collection id.
///
/// Per LLD Q3: `seed = u64::from_le_bytes(blake3("turboquant.v1.rotation" ||
/// collection_id).as_bytes()[0..8])`. Implemented here with a simple FNV-1a
/// hash to avoid pulling blake3 into the foundation crate; P8 may swap in
/// blake3 when the xCatalog wiring lands and the dependency is paid elsewhere.
///
/// The exact hash function is part of the wire contract — changing it
/// requires re-encoding every collection's codes. Mark this as load-bearing.
#[cfg(feature = "experimental-turboquant")]
pub fn derive_rotation_seed(collection_id: &str) -> u64 {
    // FNV-1a 64-bit. Deterministic, stdlib-only.
    let prefix = b"turboquant.v1.rotation:";
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in prefix.iter().chain(collection_id.as_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ============================================================================
// QuantizationMethod trait (Phase A — Quantization Trait Convergence Plan)
// ============================================================================
//
// A metadata-only trait that collapses the 13 enum-match dispatch sites
// across the codebase (planner, router, EXPLAIN builder, Prometheus label
// emitter, lifecycle checker) into a single trait-method call.
//
// Encode / decode / score deliberately stay variant-specific — their state
// shapes diverge too far to unify cheaply (PQ needs a mutable Codebook,
// TurboQuant needs rotation_seed + frozen calibration, Binary is stateless).
// See the plan document at `~/.claude/plans/dreamy-finding-clover.md` §"Design
// rationale" for the load-bearing reasoning. Encode/decode unification is
// deferred to a future phase once a second read-time codec lands and the trait
// shape can be derived from two concrete examples rather than one.

/// Lifecycle classification for a quantization method.
///
/// Drives the routing decision between cached codes (WORM-friendly write-time
/// quantization) and on-the-fly encoding (adhoc / research workload friendly
/// read-time quantization). The progressive-search dispatcher (`Phase C`) and
/// the NOVA storage writer use this to pick between persisting codes or
/// skipping the columnar path. See ADR-021 §"Authority mode" and
/// `EMBEDDING_PRECISION_LLD_2026_05_22.adoc` Q12.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationLifecycle {
    /// Codes are encoded at write/flush/compaction time, persisted to
    /// storage, and scored read-time without re-encoding. Optimal for stable
    /// WORM corpora. Examples: Binary, Scalar/INT8, ProductQuantization.
    WriteTime,
    /// Codes are computed at read time from a durable per-collection rotation
    /// seed + frozen calibration. Optimal for adhoc / research workloads
    /// where re-encoding on every epoch bump beats persisting stale codes.
    /// Today only TurboQuant uses this lifecycle.
    ReadTime,
    /// FP32 / FP16 — no quantization applied. The "natural" arm so trait
    /// dispatch doesn't need a separate fast path.
    Identity,
}

impl fmt::Display for QuantizationLifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WriteTime => write!(f, "write_time"),
            Self::ReadTime => write!(f, "read_time"),
            Self::Identity => write!(f, "identity"),
        }
    }
}

/// Durable state every collection needs to recover its quantization at
/// restart. Filled in by the variant; `None` for stateless methods (Binary,
/// FP32). Persisted in xCatalog (`Phase E` of the convergence plan).
///
/// The string-typed `seed_or_codebook_id` and `calibration` fields are
/// deliberately opaque — they let every variant tunnel its own discriminator
/// through the same shape without forcing the catalog to learn variant
/// semantics. PQ uses `seed_or_codebook_id = Some(codebook_id)` and
/// `calibration = Some(training_epoch.to_string())`; TurboQuant uses
/// `seed_or_codebook_id = Some(format!("{:#x}", rotation_seed))` and
/// `calibration = Some(calibration_mode.to_string())`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableQuantState {
    /// Per-collection rotation seed (TurboQuant) or codebook identifier
    /// (PQ). `None` for stateless methods.
    pub seed_or_codebook_id: Option<String>,
    /// Calibration mode discriminator (TurboQuant) or training-epoch tag
    /// (PQ). `None` for stateless methods.
    pub calibration: Option<String>,
    /// Precision epoch this code was encoded under. Mismatch with the
    /// collection's current epoch triggers repair from the canonical
    /// `ProximaRecord` per `EMBEDDING_PRECISION_LLD` Q12.
    pub encoded_epoch: u64,
}

/// Metadata-only trait every quantization variant implements.
///
/// Routes, planners, EXPLAIN builders, and metric label emitters call these
/// methods instead of matching the enum directly. Encode / decode / score
/// remain per-variant — see the module-level comment for the load-bearing
/// reasoning.
///
/// Object-safe: every method takes `&self`, no associated types, no generics.
/// Both modality-side parameter structs (`BinaryQuantization`,
/// `TurboQuantization`, ...) and the enum wrappers (`QuantizationLevel`,
/// `UnifiedQuantizationLevel`) implement it via blanket-style delegation.
pub trait QuantizationMethod: Send + Sync + std::fmt::Debug {
    /// Canonical foundation-crate identifier. The one place every other
    /// dispatch site asks "what variant is this?".
    fn quantization_type(&self) -> QuantizationType;

    /// Bits per coordinate (1 for Binary, 8 for INT8, 2/4 for TurboQuant,
    /// 32 for FP32). Used by Prometheus labels, EXPLAIN, and the
    /// storage-size estimator that today lives in
    /// `UnifiedQuantizationLevel::bytes_per_vector`.
    fn bit_width(&self) -> u8;

    /// Read-time vs write-time vs identity. Drives the dispatch in
    /// `progressive_search` (`Phase C`) and the storage-engine writer in
    /// `viper/pipeline.rs` (which today skips TurboQuant blindly).
    fn lifecycle(&self) -> QuantizationLifecycle;

    /// Durable state needed to recover after process restart. `None` for
    /// stateless methods. Persisted in xCatalog (`Phase E`).
    fn durable_state(&self) -> Option<DurableQuantState>;

    /// Whether this method's scoring kernel can consume a packed bitmap
    /// allowlist directly (the TurboQuant `CandidateMaskSet` path). Default
    /// `false` — only TurboQuant overrides to `true` today. Drives the AXIS
    /// adapter dispatch (`Phase D`): when `true`, the adapter routes a
    /// `CandidateSet` straight into the kernel; when `false`, it falls back
    /// to the existing post-filter path.
    fn supports_candidate_mask(&self) -> bool {
        false
    }

    /// Stable lowercase identifier for Prometheus labels.
    /// `quantization_type().to_string()` would do this too, but pinning it
    /// here means changing label values requires a trait-method change
    /// (loud) rather than a Display impl change (silent).
    fn metric_label(&self) -> &'static str;
}

// ----------------------------------------------------------------------------
// QuantizationType inherent methods (no full trait impl — bit_width and
// durable_state need a configured method, not just a type tag).
//
// These let any code path that holds only the bare enum tag still query the
// lifecycle / metric label without re-matching. Variant structs in the
// modality crate provide the full `QuantizationMethod` impl with bit_width
// and durable_state populated.
// ----------------------------------------------------------------------------

impl QuantizationType {
    /// Lifecycle classification answerable from the type tag alone.
    pub fn lifecycle(&self) -> QuantizationLifecycle {
        match self {
            Self::None => QuantizationLifecycle::Identity,
            Self::Scalar | Self::Product | Self::Binary => QuantizationLifecycle::WriteTime,
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant => QuantizationLifecycle::ReadTime,
        }
    }

    /// Stable Prometheus label.
    pub fn metric_label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scalar => "scalar",
            Self::Product => "product",
            Self::Binary => "binary",
            #[cfg(feature = "experimental-turboquant")]
            Self::TurboQuant => "turboquant",
        }
    }

    /// Whether the canonical kernel for this type accepts a packed-bitmap
    /// allowlist. Mirrors `QuantizationMethod::supports_candidate_mask` at
    /// the bare-tag level.
    pub fn supports_candidate_mask(&self) -> bool {
        #[cfg(feature = "experimental-turboquant")]
        {
            return matches!(self, Self::TurboQuant);
        }
        #[cfg(not(feature = "experimental-turboquant"))]
        {
            false
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_default() {
        assert_eq!(QuantizationType::default(), QuantizationType::None);
    }

    #[test]
    fn test_quantization_type_display() {
        assert_eq!(QuantizationType::None.to_string(), "none");
        assert_eq!(QuantizationType::Scalar.to_string(), "scalar");
        assert_eq!(QuantizationType::Product.to_string(), "product");
        assert_eq!(QuantizationType::Binary.to_string(), "binary");
    }

    #[test]
    fn test_quantization_type_from_str() {
        assert_eq!(
            QuantizationType::from_str("none"),
            Some(QuantizationType::None)
        );
        assert_eq!(
            QuantizationType::from_str("scalar"),
            Some(QuantizationType::Scalar)
        );
        assert_eq!(
            QuantizationType::from_str("int8"),
            Some(QuantizationType::Scalar)
        );
        assert_eq!(
            QuantizationType::from_str("product"),
            Some(QuantizationType::Product)
        );
        assert_eq!(
            QuantizationType::from_str("pq"),
            Some(QuantizationType::Product)
        );
        assert_eq!(
            QuantizationType::from_str("binary"),
            Some(QuantizationType::Binary)
        );
        assert_eq!(QuantizationType::from_str("unknown"), None);
    }

    #[test]
    fn test_quantization_type_is_fixed_point() {
        assert!(!QuantizationType::None.is_fixed_point());
        assert!(QuantizationType::Scalar.is_fixed_point());
        assert!(!QuantizationType::Product.is_fixed_point());
        assert!(QuantizationType::Binary.is_fixed_point());
    }

    #[test]
    fn test_quantization_level_default() {
        assert_eq!(QuantizationLevel::default(), QuantizationLevel::None);
    }

    #[test]
    fn test_quantization_level_display() {
        assert_eq!(QuantizationLevel::None.to_string(), "none");
        assert_eq!(QuantizationLevel::Int4.to_string(), "int4");
        assert_eq!(QuantizationLevel::Int8.to_string(), "int8");
        assert_eq!(QuantizationLevel::UInt8.to_string(), "uint8");
        assert_eq!(QuantizationLevel::FP16.to_string(), "fp16");
        assert_eq!(QuantizationLevel::FP32.to_string(), "fp32");
    }

    #[test]
    fn test_quantization_level_from_str() {
        assert_eq!(
            QuantizationLevel::from_str("none"),
            Some(QuantizationLevel::None)
        );
        assert_eq!(
            QuantizationLevel::from_str("int4"),
            Some(QuantizationLevel::Int4)
        );
        assert_eq!(
            QuantizationLevel::from_str("int8"),
            Some(QuantizationLevel::Int8)
        );
        assert_eq!(
            QuantizationLevel::from_str("uint8"),
            Some(QuantizationLevel::UInt8)
        );
        assert_eq!(
            QuantizationLevel::from_str("fp16"),
            Some(QuantizationLevel::FP16)
        );
        assert_eq!(
            QuantizationLevel::from_str("fp32"),
            Some(QuantizationLevel::FP32)
        );
        assert_eq!(QuantizationLevel::from_str("unknown"), None);
    }

    #[test]
    fn test_quantization_level_bit_width() {
        assert_eq!(QuantizationLevel::None.bit_width(), 32);
        assert_eq!(QuantizationLevel::Int4.bit_width(), 4);
        assert_eq!(QuantizationLevel::Int8.bit_width(), 8);
        assert_eq!(QuantizationLevel::UInt8.bit_width(), 8);
        assert_eq!(QuantizationLevel::FP16.bit_width(), 16);
        assert_eq!(QuantizationLevel::FP32.bit_width(), 32);
    }

    #[test]
    fn test_quantization_level_type_checks() {
        assert!(QuantizationLevel::Int4.is_signed_integer());
        assert!(QuantizationLevel::Int8.is_signed_integer());
        assert!(QuantizationLevel::UInt8.is_unsigned_integer());
        assert!(!QuantizationLevel::UInt8.is_signed_integer());
        assert!(QuantizationLevel::FP16.is_floating_point());
        assert!(QuantizationLevel::FP32.is_floating_point());
        assert!(!QuantizationLevel::Int8.is_floating_point());
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type(), QuantizationType::None);
        assert_eq!(config.quantization_level(), QuantizationLevel::None);
        assert!(!config.asymmetric);
        assert!(!config.cache);
    }

    #[test]
    fn test_quantization_config_builder() {
        let config = QuantizationConfig::scalar(QuantizationLevel::Int8)
            .with_asymmetric()
            .with_cache();

        assert_eq!(config.quantization_type(), QuantizationType::Scalar);
        assert_eq!(config.quantization_level(), QuantizationLevel::Int8);
        assert!(config.asymmetric);
        assert!(config.cache);
    }

    #[test]
    fn test_quantization_config_constructors() {
        let scalar = QuantizationConfig::scalar(QuantizationLevel::Int8);
        assert_eq!(scalar.quantization_type(), QuantizationType::Scalar);
        assert_eq!(scalar.quantization_level(), QuantizationLevel::Int8);

        let product = QuantizationConfig::product(QuantizationLevel::Int4);
        assert_eq!(product.quantization_type(), QuantizationType::Product);
        assert_eq!(product.quantization_level(), QuantizationLevel::Int4);

        let binary = QuantizationConfig::binary();
        assert_eq!(binary.quantization_type(), QuantizationType::Binary);
        assert_eq!(binary.quantization_level(), QuantizationLevel::Int4);
    }

    #[test]
    fn test_quantization_serialization() {
        let qtype = QuantizationType::Scalar;
        let json = serde_json::to_string(&qtype).unwrap();
        assert_eq!(json, "\"scalar\"");

        let deserialized: QuantizationType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, QuantizationType::Scalar);

        let qlevel = QuantizationLevel::Int8;
        let json = serde_json::to_string(&qlevel).unwrap();
        assert_eq!(json, "\"int8\"");

        let deserialized: QuantizationLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, QuantizationLevel::Int8);
    }

    #[test]
    fn test_quantization_config_serialization() {
        let config = QuantizationConfig::scalar(QuantizationLevel::Int8).with_cache();
        let json = serde_json::to_string(&config).unwrap();

        let deserialized: QuantizationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.quantization_type(), QuantizationType::Scalar);
        assert!(deserialized.cache);
    }

    // ------------------------------------------------------------------
    // TurboQuant types (P1 — ADR-021 / TURBOQUANT_LLD_2026_05_30)
    // ------------------------------------------------------------------

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_type_display_and_parse() {
        assert_eq!(QuantizationType::TurboQuant.to_string(), "turboquant");
        assert_eq!(
            QuantizationType::from_str("turboquant"),
            Some(QuantizationType::TurboQuant)
        );
        assert_eq!(
            QuantizationType::from_str("tq"),
            Some(QuantizationType::TurboQuant)
        );
        assert_eq!(
            QuantizationType::from_str("TURBOQUANT"),
            Some(QuantizationType::TurboQuant)
        );
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_is_fixed_point() {
        assert!(QuantizationType::TurboQuant.is_fixed_point());
        // is_floating_point is the negation, so this must be false.
        assert!(!QuantizationType::TurboQuant.is_floating_point());
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_calibration_mode_default_is_identity() {
        assert_eq!(CalibrationMode::default(), CalibrationMode::Identity);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_calibration_mode_display_and_parse() {
        assert_eq!(CalibrationMode::Identity.to_string(), "identity");
        assert_eq!(CalibrationMode::TqPlus.to_string(), "tq_plus");
        assert_eq!(
            CalibrationMode::from_str("identity"),
            Ok(CalibrationMode::Identity)
        );
        assert_eq!(
            CalibrationMode::from_str("tq+"),
            Ok(CalibrationMode::TqPlus)
        );
        assert_eq!(CalibrationMode::from_str("nonsense"), Err(()));
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_calibration_mode_serde_snake_case() {
        let s = serde_json::to_string(&CalibrationMode::TqPlus).unwrap();
        assert_eq!(s, "\"tq_plus\"");
        let v: CalibrationMode = serde_json::from_str("\"identity\"").unwrap();
        assert_eq!(v, CalibrationMode::Identity);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_derive_rotation_seed_is_deterministic_and_collision_resistant() {
        // Determinism: same id → same seed across calls.
        let a1 = derive_rotation_seed("col-abc");
        let a2 = derive_rotation_seed("col-abc");
        assert_eq!(a1, a2);
        // Multi-tenant: different ids → different seeds. With 64-bit output,
        // birthday-paradox risk at any realistic collection count is zero;
        // for FNV-1a these short distinct inputs are practically guaranteed
        // to land on different seeds.
        let b = derive_rotation_seed("col-def");
        assert_ne!(a1, b);
        // Spot-check that an empty collection id still produces a stable
        // (non-zero) seed.
        assert_ne!(derive_rotation_seed(""), 0);
    }

    // ------------------------------------------------------------------
    // QuantizationMethod trait + QuantizationType inherent surface
    // (Phase A — Quantization Trait Convergence Plan)
    // ------------------------------------------------------------------

    #[test]
    fn quantization_lifecycle_display_round_trips_each_arm() {
        // Display must match the serde rename ("snake_case") because the
        // metric label and EXPLAIN hint both rely on the same encoding.
        assert_eq!(QuantizationLifecycle::WriteTime.to_string(), "write_time");
        assert_eq!(QuantizationLifecycle::ReadTime.to_string(), "read_time");
        assert_eq!(QuantizationLifecycle::Identity.to_string(), "identity");
    }

    #[test]
    fn quantization_lifecycle_serde_snake_case() {
        let s = serde_json::to_string(&QuantizationLifecycle::ReadTime).unwrap();
        assert_eq!(s, "\"read_time\"");
        let back: QuantizationLifecycle = serde_json::from_str("\"write_time\"").unwrap();
        assert_eq!(back, QuantizationLifecycle::WriteTime);
    }

    #[test]
    fn quantization_type_lifecycle_classifies_each_variant() {
        // The lifecycle classification is load-bearing: progressive_search
        // (Phase C) routes write-time vs read-time vs identity differently.
        assert_eq!(
            QuantizationType::None.lifecycle(),
            QuantizationLifecycle::Identity,
        );
        assert_eq!(
            QuantizationType::Scalar.lifecycle(),
            QuantizationLifecycle::WriteTime,
        );
        assert_eq!(
            QuantizationType::Product.lifecycle(),
            QuantizationLifecycle::WriteTime,
        );
        assert_eq!(
            QuantizationType::Binary.lifecycle(),
            QuantizationLifecycle::WriteTime,
        );
        #[cfg(feature = "experimental-turboquant")]
        assert_eq!(
            QuantizationType::TurboQuant.lifecycle(),
            QuantizationLifecycle::ReadTime,
        );
    }

    #[test]
    fn quantization_type_metric_label_is_stable_lowercase() {
        // Label values are wire-stable. Changing them breaks every
        // Prometheus query, so this test pins them by literal.
        assert_eq!(QuantizationType::None.metric_label(), "none");
        assert_eq!(QuantizationType::Scalar.metric_label(), "scalar");
        assert_eq!(QuantizationType::Product.metric_label(), "product");
        assert_eq!(QuantizationType::Binary.metric_label(), "binary");
        #[cfg(feature = "experimental-turboquant")]
        assert_eq!(QuantizationType::TurboQuant.metric_label(), "turboquant");
    }

    #[test]
    fn quantization_type_supports_candidate_mask_only_turboquant_today() {
        // Only TurboQuant's kernel consumes a packed-bitmask allowlist;
        // everything else falls back to post-filter. AXIS Phase D dispatches
        // on this method.
        assert!(!QuantizationType::None.supports_candidate_mask());
        assert!(!QuantizationType::Scalar.supports_candidate_mask());
        assert!(!QuantizationType::Product.supports_candidate_mask());
        assert!(!QuantizationType::Binary.supports_candidate_mask());
        #[cfg(feature = "experimental-turboquant")]
        assert!(QuantizationType::TurboQuant.supports_candidate_mask());
    }

    #[test]
    fn durable_quant_state_round_trips_through_serde() {
        // The wire shape is what xCatalog persists (Phase E). Lock the JSON
        // key names so a future struct rename doesn't silently break
        // catalog round-trips.
        let s = DurableQuantState {
            seed_or_codebook_id: Some("0xdeadbeef".to_string()),
            calibration: Some("tq_plus".to_string()),
            encoded_epoch: 7,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("seed_or_codebook_id"));
        assert!(json.contains("calibration"));
        assert!(json.contains("encoded_epoch"));
        let back: DurableQuantState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn durable_quant_state_stateless_variant_uses_nones() {
        // Binary and FP32 have no per-collection state. We still serialize
        // the struct (with nones) so the catalog row shape is uniform.
        let s = DurableQuantState {
            seed_or_codebook_id: None,
            calibration: None,
            encoded_epoch: 0,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: DurableQuantState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        assert!(back.seed_or_codebook_id.is_none());
        assert!(back.calibration.is_none());
    }

    /// Minimal `QuantizationMethod` impl used to verify the trait is
    /// object-safe and behaves correctly under dynamic dispatch. The real
    /// impls live in the modality crate alongside the variant structs.
    #[derive(Debug)]
    struct FakeMethod {
        kind: QuantizationType,
        bits: u8,
        lifecycle: QuantizationLifecycle,
        mask: bool,
    }

    impl QuantizationMethod for FakeMethod {
        fn quantization_type(&self) -> QuantizationType {
            self.kind
        }
        fn bit_width(&self) -> u8 {
            self.bits
        }
        fn lifecycle(&self) -> QuantizationLifecycle {
            self.lifecycle
        }
        fn durable_state(&self) -> Option<DurableQuantState> {
            None
        }
        fn supports_candidate_mask(&self) -> bool {
            self.mask
        }
        fn metric_label(&self) -> &'static str {
            self.kind.metric_label()
        }
    }

    #[test]
    fn quantization_method_is_object_safe_and_works_under_dyn_dispatch() {
        // If the trait grows an associated type or a generic method, this
        // test stops compiling. That's the point — object-safety is the
        // load-bearing contract for routers that hold `&dyn`.
        let m: Box<dyn QuantizationMethod> = Box::new(FakeMethod {
            kind: QuantizationType::Scalar,
            bits: 8,
            lifecycle: QuantizationLifecycle::WriteTime,
            mask: false,
        });
        assert_eq!(m.quantization_type(), QuantizationType::Scalar);
        assert_eq!(m.bit_width(), 8);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::WriteTime);
        assert!(!m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "scalar");
        assert!(m.durable_state().is_none());
    }
}
