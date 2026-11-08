//! TD-COMPACT-2b step 2: extent-scoped residency — warm a region group of a
//! segment into the persistent L2 store under a region key.
//!
//! The cache-policy registry (step 1) resolves which groups a collection
//! wants pinned. This module makes it true: [`warm_region_group`] fetches
//! the group's byte extent from the object store (one ranged GET per group,
//! not per segment read) and stages it into the `PersistentByteStore` under
//! a `pax-region-v1/{group}/{path}` key, where the protection system pins
//! it against eviction.

use proximadb_cache::{L2Class, region_key};

/// Warm one region group of a segment into the persistent L2 store.
///
/// `group` is the region-group name from the cache policy (`index`,
/// `rerank`, `control`, etc.). `offset` and `length` are the region's byte
/// extent within the segment object (from the segment footer — the caller
/// has already parsed it). Returns whether the region was written.
///
/// The bytes are fetched with one ranged GET from the object store and
/// staged through the same temp-file mechanism as `warm_object` (never the
/// whole range in RAM).
/// Stage a region group's bytes directly from the writer's seed (TD-COMPACT-2b).
///
/// The seed's `Arc<[u8]>` IS the region's bytes — no object-store GET, no
/// extra RAM allocation, and the offsets are correct by construction (the
/// writer produced them). This is the same pattern `install_pax_cache_seed`
/// uses for the legacy invariants keys. Returns whether the region was
/// written; a put failure is logged, not swallowed.
pub async fn seed_region_group(
    group: &str,
    path: &str,
    bytes: std::sync::Arc<[u8]>,
) -> anyhow::Result<bool> {
    let Some(store) = proximadb_cache::process_store() else {
        return Ok(false);
    };
    let key = region_key(group, path);
    if store.entry_len(&key).is_some() {
        return Ok(false); // already warm
    }
    if !store.policy_covers(&key) {
        return Ok(false); // no grant covers this region group
    }
    let len = bytes.len() as u64;
    match store.put(key.clone(), L2Class::Invariants, bytes).await {
        Ok(()) => {
            tracing::debug!(key = %key, len, "region-group staged from seed");
            Ok(true)
        }
        Err(e) => {
            tracing::warn!(key = %key, len, error = %e, "region-group staging failed");
            Ok(false)
        }
    }
}

/// Whether a region group of a segment is already warm in the L2 store.
pub fn region_group_resident(group: &str, path: &str) -> bool {
    proximadb_cache::process_store()
        .is_some_and(|store| store.entry_len(&region_key(group, path)).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_key_format() {
        let key = region_key("index", "file:///data/coll/seg.pax");
        assert!(key.starts_with("pax-region-v1/index/"), "{key}");
        assert!(key.ends_with("seg.pax"), "{key}");
    }

    #[test]
    fn region_group_resident_without_store_is_false() {
        assert!(!region_group_resident("index", "file:///nonexistent"));
    }
}
