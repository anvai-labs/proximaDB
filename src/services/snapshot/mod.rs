//! Snapshot pin + atomic republish (Phase 8 CS/CD foundation).
//!
//! This is the foundational substrate shared by the Continuous Discovery loop
//! (F1) and External Collection (F5): pin a read-only view of a collection's
//! canonical WAL position, then atomically republish a refined snapshot by
//! driving the collection's discovery projection through the catalog freshness
//! state machine (`Updating` -> `Fresh`).
//!
//! It is pure composition over existing primitives:
//! - the global WAL manifest singleton
//!   (`crate::storage::persistence::write_ahead_log::manifest`) for the snapshot
//!   position, and
//! - `proximadb_catalog::{CatalogProjection, ProjectionFreshnessState}` plus
//!   `Catalog::replace_table` (in place — TD-CATREPLACE-1) for atomic publish.
//!
//! See `docs/12-design/PHASE8_CONTINUOUS_LOOP_HLD_LLD_2026_05_28.adoc` (F1, S0).

mod branch_ref;
mod coordinator;

pub use branch_ref::BranchRef;
pub use coordinator::{DISCOVERY_ACTIVE_PROJECTION, SnapshotPin, SnapshotPublishCoordinator};
