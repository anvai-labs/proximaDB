//! Per-collection cache policy (TD-COMPACT-2b).
//!
//! Step 1 (this module): the [`registry::CachePolicyRegistry`] resolves a
//! collection's effective region-group policy from its `cache.*` table
//! properties. Step 2 (follow-up) wires extent-scoped residency (warm/evict
//! region groups on the [`PersistentByteStore`]) to the resolved policy.

pub mod registry;
pub mod residency;

pub use registry::{CachePolicyRegistry, cache_properties_of};
pub use residency::{region_group_resident, seed_region_group};
