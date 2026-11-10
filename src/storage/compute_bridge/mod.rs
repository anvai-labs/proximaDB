//! # Compute Bridge — storage↔compute glue
//!
//! Code that sits on the *compute* side of the storage↔compute boundary but
//! legitimately depends on the storage/core layers (cache orchestrator, flush
//! parameters, …) and therefore **cannot** live in a `proximadb-*-kernel`
//! foundation crate.
//!
//! Relocated here from `src/compute/quantization/` as step **Q2** of the
//! quantization-kernel split (`docs/12-design/QUANTIZATION_KERNEL_SPLIT_2026_06_22.adoc`)
//! so the kernel modules left behind in `src/compute/quantization/` are
//! storage-free and mechanically extractable into `proximadb-quantization-kernel`
//! (step Q3).
//!
//! Consumers keep their existing `crate::compute::quantization::{global_cache,
//! selection}` import paths — `src/compute/quantization/mod.rs` re-exports these
//! modules from here.
//!
//! - [`global_cache`] — process-wide quantization codebook cache built over the
//!   storage [`CrossCacheOrchestrator`](crate::storage::cache::orchestrator::CrossCacheOrchestrator).
//! - [`selection`] — persistent-vs-stateless quantization ENGINE selection
//!   driven by the storage
//!   [`FlushParameters`](crate::storage::traits::FlushParameters).

pub mod global_cache;
pub mod selection;
