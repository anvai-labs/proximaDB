//! Compression-ops cluster, hoisted from root `storage::engines::core::ops`
//! (TD-DECOMP-81): the context-aware compression common layer, its adapter,
//! ProximaCodec tensor encoding, and SIMD config. Root keeps `pub use` shims in
//! `ops/mod.rs` so `crate::…::ops::compression_common::*` paths resolve unchanged.

pub mod compression_adapter;
pub mod compression_common;
pub mod simd_config;
