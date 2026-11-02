//! # ProximaDB Quantization Kernel
//!
//! The pure vector-quantization kernel — SQ8 / PQ / Binary / TurboQuant
//! encode·decode math, the codebook store, the TurboQuant store registry, and the
//! in-memory quantized-storage engine. Extracted from `src/compute/quantization`
//! as step **Q3** of the quantization-kernel split
//! (`docs/12-design/QUANTIZATION_KERNEL_SPLIT_2026_06_22.adoc`).
//!
//! Depends only on `proximadb-distance-kernel` (the resolved quant→distance
//! layering), `proximadb-quantization-types` (canonical config),
//! `proximadb-vector` (SIMD kernels + TurboQuant store), and
//! `proximadb-hardware` (SIMD capability detection). No `storage`, no `core`:
//! the storage↔compute glue (codebook cache, level selection) stays in the root
//! crate at `crate::storage::compute_bridge` (step Q2).
//!
//! The root crate re-exports this kernel at the original `crate::compute::quantization`
//! path (facade in `src/compute/quantization/mod.rs`) so existing consumers are
//! unchanged.

// The moved code relied on the root crate's allow-set; mirror it here (a new crate
// does not inherit the root `#![allow]`s).
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]
#![allow(clippy::legacy_numeric_constants)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::field_reassign_with_default)]

pub mod compile_time;
pub mod hardware_accelerated;
pub mod quantization_engine;
pub mod smart_defaults;
pub mod storage_engine;
#[cfg(feature = "experimental-turboquant")]
pub mod turboquant_store_registry;
pub mod types;

// Low-dependency quantization surfaces re-exported from the vector modality.
pub use proximadb_vector::quantization::compile_time::*;
pub use proximadb_vector::quantization::smart_defaults::QuantizationSmartDefaults;

pub use quantization_engine::{
    BinaryQuantization, Codebook, CodebookData, CodebookStore, CustomQuantization,
    InMemoryCodebookStore, NoQuantization, ProductQuantization, QuantizationMetadata,
    QuantizationStrategy, QuantizedVector, ScalarQuantization, TrainingConfig,
    UnifiedQuantizationEngine, UnifiedQuantizationLevel, UniformQuantization,
};
pub use storage_engine::{StorageQuantizationConfig, StorageQuantizationEngine};
