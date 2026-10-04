//! Vector Quantization facade.
//!
//! The pure kernel was extracted to the `proximadb-quantization-kernel` crate
//! (step **Q3** of the quantization-kernel split,
//! `docs/12-design/QUANTIZATION_KERNEL_SPLIT_2026_06_22.adoc`). The storage↔compute
//! glue (codebook cache, level selection) lives in `crate::storage::compute_bridge`
//! (step **Q2**). This module re-exports both so existing
//! `crate::compute::quantization::*` consumers are unchanged.

// Kernel crate: its modules (quantization_engine, storage_engine, types, compile_time,
// hardware_accelerated, smart_defaults, and turboquant_store_registry under the
// experimental-turboquant feature) and flattened symbols, preserving the original
// `crate::compute::quantization::*` paths.
pub use proximadb_quantization_kernel::*;

// Storage↔compute glue (Q2: global_cache, selection) is NOT re-exported here —
// its consumers (the storage engines) import `crate::storage::compute_bridge::*`
// directly (ADR-094 import unification). The kernel re-export above stays: 38+
// files use the flattened `crate::compute::quantization::quantization_engine::*`
// paths.
