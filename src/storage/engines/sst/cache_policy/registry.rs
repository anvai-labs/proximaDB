//! The engine-side cache-policy registry (TD-COMPACT-2b step 1).
//!
//! Reads a collection's `cache.*` table properties (set via `ALTER TABLE t
//! SET ('cache.index.mode' = 'pin')`), the server's `[storage.cache_policy]`
//! defaults, and resolves the [`EffectivePolicy`] through the 2a model
//! ([`proximadb_cache::policy::resolve_cache_policy`]). The engine consults
//! this before segment discovery so the residency layer (2b step 2) can
//! warm/evict region groups per the resolved policy.
//!
//! This registry is the ONLY place that maps catalog properties → engine
//! cache behavior; no ad-hoc reads of `cache.*` keys elsewhere.

use std::collections::HashMap;
use std::sync::Arc;

use proximadb_cache::policy::{
    AdvisorMode, CollectionCacheProperties, EffectivePolicy, EngineCacheBaseline, RegionGroup,
    ResolveInput, ServerCachePolicy, StorageClassHint, resolve_cache_policy,
};

/// Process-wide singleton (the engine constructs it at boot).
pub struct CachePolicyRegistry {
    server: ServerCachePolicy,
    engine: EngineCacheBaseline,
    /// Per-collection cached resolutions (invalidated on property change).
    resolved: dashmap::DashMap<String, Arc<EffectivePolicy>>,
}

impl CachePolicyRegistry {
    pub fn new(server: ServerCachePolicy, engine: EngineCacheBaseline) -> Self {
        Self {
            server,
            engine,
            resolved: dashmap::DashMap::new(),
        }
    }
    /// Construct from the server's `[storage.cache_policy]` config section.
    /// When absent (default), uses `ServerCachePolicy::default()`.
    pub fn from_config(
        config: &crate::core::config::CoreStorageConfig,
        engine: EngineCacheBaseline,
    ) -> Self {
        Self::new(config.cache_policy.clone(), engine)
    }

    /// Resolve the effective policy for a collection from its table
    /// properties. The result is cached (per-collection) until invalidated.
    pub fn resolve(
        &self,
        collection_id: &str,
        properties: &HashMap<String, String>,
        class: StorageClassHint,
    ) -> anyhow::Result<Arc<EffectivePolicy>> {
        if let Some(cached) = self.resolved.get(collection_id) {
            return Ok(cached.clone());
        }
        let collection = CollectionCacheProperties::parse(properties).map_err(|e| {
            anyhow::anyhow!("collection {collection_id} has invalid cache properties: {e}")
        })?;
        let policy = Arc::new(resolve_cache_policy(ResolveInput {
            engine: self.engine,
            collection: &collection,
            server: &self.server,
            class,
            performance_grant: class == StorageClassHint::Performance,
            recommendations: &[],
        }));
        self.resolved
            .insert(collection_id.to_string(), policy.clone());
        Ok(policy)
    }

    /// Invalidate a collection's cached resolution (called when its
    /// properties change — an `ALTER TABLE SET` on a `cache.*` key).
    pub fn invalidate(&self, collection_id: &str) {
        self.resolved.remove(collection_id);
    }

    /// The server's advisor mode (for the /metrics provenance surface).
    pub fn advisor_mode(&self) -> AdvisorMode {
        self.server.advisor
    }

    /// A human-readable summary for diagnostics.
    pub fn describe(&self, collection_id: &str) -> Option<String> {
        self.resolved.get(collection_id).map(|policy| {
            RegionGroup::ALL
                .iter()
                .map(|g| {
                    let r = policy.group(*g);
                    format!(
                        "{}={:?}/{}:{}",
                        g.as_str(),
                        r.policy.mode,
                        format!("{:?}", r.policy.placement).as_str(),
                        r.mode_from.as_str()
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
    }
}

/// Extract only the `cache.*` properties from a collection's full property map.
pub fn cache_properties_of(properties: &HashMap<String, String>) -> HashMap<String, String> {
    properties
        .iter()
        .filter(|(k, _)| k.trim().to_ascii_lowercase().starts_with("cache."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_cache::policy::builtin_group_policy;

    fn registry() -> CachePolicyRegistry {
        CachePolicyRegistry::new(ServerCachePolicy::default(), EngineCacheBaseline::default())
    }

    #[test]
    fn resolves_and_caches_per_collection() {
        let reg = registry();
        let mut props = HashMap::new();
        props.insert("cache.index.mode".to_string(), "pin".to_string());
        props.insert("unrelated".to_string(), "value".to_string());

        let policy = reg
            .resolve("coll-1", &props, StorageClassHint::Balanced)
            .unwrap();
        assert_eq!(
            policy.group(RegionGroup::Index).policy.mode,
            proximadb_cache::policy::GroupMode::Pin
        );
        // The builtin for filter is Off; the collection didn't set it.
        assert_eq!(
            policy.group(RegionGroup::Filter).policy.mode,
            builtin_group_policy(EngineCacheBaseline::default(), RegionGroup::Filter).mode
        );

        // Cached: the same Arc comes back.
        let again = reg
            .resolve("coll-1", &props, StorageClassHint::Balanced)
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(&policy, &again));

        // Invalidation forces a re-resolve.
        reg.invalidate("coll-1");
        let fresh = reg
            .resolve("coll-1", &props, StorageClassHint::Balanced)
            .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&policy, &fresh));
    }

    #[test]
    fn invalid_cache_properties_are_an_error() {
        let reg = registry();
        let mut props = HashMap::new();
        props.insert("cache.index.mode".to_string(), "always".to_string());
        let err = reg
            .resolve("coll-2", &props, StorageClassHint::Balanced)
            .unwrap_err();
        assert!(err.to_string().contains("cache.index.mode"));
    }

    #[test]
    fn cache_properties_extraction_filters() {
        let mut props = HashMap::new();
        props.insert("cache.payload.mode".to_string(), "on_demand".to_string());
        props.insert("owner".to_string(), "system".to_string());
        props.insert("Cache.Rerank.Placement".to_string(), "ram".to_string());
        let filtered = cache_properties_of(&props);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("cache.payload.mode"));
        assert!(filtered.contains_key("Cache.Rerank.Placement"));
    }

    #[test]
    fn describe_provides_diagnostics() {
        let reg = registry();
        let mut props = HashMap::new();
        props.insert("cache.index.mode".to_string(), "pin".to_string());
        let _ = reg
            .resolve("coll-3", &props, StorageClassHint::Balanced)
            .unwrap();
        let desc = reg.describe("coll-3").unwrap();
        assert!(desc.contains("index=Pin"), "{desc}");
        assert!(desc.contains("builtin"), "{desc}");
        assert!(reg.describe("unknown").is_none());
    }
}
