//! Engine Capabilities Trait (OCP Compliant) — re-export shim.
//!
//! The capability-descriptor subsystem (the trait + the 7 per-engine structs +
//! `CapabilityFactory` + `EngineBundle` + the `FlushThresholds` /
//! `CompactionHeuristics` config types + `ScanCapabilities`) has been hoisted
//! to the `proximadb-storage-ports` crate as a cohesive pure-metadata cluster
//! (see `crates/storage/proximadb-storage-ports/src/capabilities.rs`). It is
//! hoisted as a unit because every member depends only on the others + proto
//! (`CompressionAlgorithm`, `StorageEngine`) + `serde` + `std` — no concrete
//! engine references. Clearing it from the root lets the engine-port-traits
//! module move to its own crate (the last root-dep of `src/storage/traits/mod.rs`
//! was `CapabilityFactory`).
//!
//! This file is now a thin re-export so every existing path
//! (`crate::storage::trait_components::capabilities::Foo` and the
//! `pub use capabilities::{…}` in `trait_components/mod.rs`) resolves
//! unchanged.

pub use proximadb_storage_ports::{
    CapabilityFactory, CompactionHeuristics, EngineBundle, EngineCapabilities, FlushThresholds,
    HelixCapabilities, NovaCapabilities, RaptorCapabilities, ScanCapabilities, SstCapabilities,
    SwiftCapabilities, TstCapabilities,
};
