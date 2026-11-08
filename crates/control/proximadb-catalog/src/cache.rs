// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Catalog metadata cache — TTL-based in-memory LRU cache for xCatalog operations.
//!
//! Caches namespace, table schema, index, and statistics lookups to reduce
//! round-trips to the backing store. Shared by all catalog backend implementations
//! via `Arc<CatalogCache>`.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::{debug, trace};

use crate::{
    CatalogIndex, CatalogNamespace, CatalogTableSchema, CatalogTableStatistics, TableIdentifier,
};

#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

impl<T: Clone> CacheEntry<T> {
    fn new(value: T) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn access(&mut self) -> &T {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        &self.value
    }
}

/// Catalog metadata cache shared by all catalog backend implementations.
///
/// TD-CAT-5: the maps are `DashMap` (lookup locks ONE shard for the LRU
/// touch, not the whole map for every reader) and the counters are plain
/// atomics (no second whole-cache lock per operation). Eviction at capacity
/// is shard-sampled — see [`CatalogCache::maybe_evict`].
pub struct CatalogCache {
    max_entries: usize,
    ttl: Duration,
    namespaces: DashMap<String, CacheEntry<CatalogNamespace>>,
    tables: DashMap<String, CacheEntry<CatalogTableSchema>>,
    indexes: DashMap<String, CacheEntry<Vec<CatalogIndex>>>,
    statistics: DashMap<String, CacheEntry<CatalogTableStatistics>>,
    // Counters. The public `CacheStats`/`get_stats()` shape is unchanged
    // (snapshotted on read); ops just no longer serialize on a second lock.
    namespace_hits: AtomicU64,
    namespace_misses: AtomicU64,
    table_hits: AtomicU64,
    table_misses: AtomicU64,
    index_hits: AtomicU64,
    index_misses: AtomicU64,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
    evictions: AtomicU64,
    invalidations: AtomicU64,
    /// Bounded ring of recently inserted keys, one per map family — the
    /// eviction sample set. Each family gets its OWN ring: a shared ring
    /// filtered by single-map membership would treat every other family's
    /// keys as "no longer resident" and drop them on each evict check,
    /// starving whichever family isn't currently dominating insert traffic
    /// (that family's `maybe_evict` would then find zero candidates while
    /// its `put_*` keeps inserting unconditionally, growing past
    /// `max_entries` until the next `evict_expired` sweep).
    namespace_ring: std::sync::Mutex<std::collections::VecDeque<String>>,
    table_ring: std::sync::Mutex<std::collections::VecDeque<String>>,
    index_ring: std::sync::Mutex<std::collections::VecDeque<String>>,
    stats_ring: std::sync::Mutex<std::collections::VecDeque<String>>,
}

/// Cache hit/miss/eviction counters.
///
/// Specialized — multi-category cache. Tracks 4 distinct cache types
/// (namespace, table, index, schema-stats) with separate hit/miss
/// counters per category. Not canonicalizable to
/// `proximadb_runtime_common::cache::CacheStats` without losing the
/// per-category breakdown that catalog observability depends on.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub namespace_hits: u64,
    pub namespace_misses: u64,
    pub table_hits: u64,
    pub table_misses: u64,
    pub index_hits: u64,
    pub index_misses: u64,
    pub stats_hits: u64,
    pub stats_misses: u64,
    pub evictions: u64,
    pub invalidations: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.namespace_hits + self.table_hits + self.index_hits + self.stats_hits;
        let misses =
            self.namespace_misses + self.table_misses + self.index_misses + self.stats_misses;
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

impl CatalogCache {
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
            namespaces: DashMap::new(),
            tables: DashMap::new(),
            indexes: DashMap::new(),
            statistics: DashMap::new(),
            namespace_hits: AtomicU64::new(0),
            namespace_misses: AtomicU64::new(0),
            table_hits: AtomicU64::new(0),
            table_misses: AtomicU64::new(0),
            index_hits: AtomicU64::new(0),
            index_misses: AtomicU64::new(0),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            namespace_ring: std::sync::Mutex::new(Default::default()),
            table_ring: std::sync::Mutex::new(Default::default()),
            index_ring: std::sync::Mutex::new(Default::default()),
            stats_ring: std::sync::Mutex::new(Default::default()),
        }
    }

    pub fn default_cache() -> Self {
        Self::new(10_000, 300)
    }

    // ---- Namespace ----

    pub fn get_namespace(&self, catalog: &str, namespace: &[String]) -> Option<CatalogNamespace> {
        let key = ns_key(catalog, namespace);
        // Hit path locks one DashMap shard for the LRU touch (was: the whole
        // map under an exclusive write lock).
        if let Some(mut e) = self.namespaces.get_mut(&key) {
            if e.is_expired(self.ttl) {
                drop(e);
                self.namespaces.remove(&key);
                self.namespace_misses.fetch_add(1, AtomicOrdering::Relaxed);
                trace!("namespace cache miss (expired): {key}");
                return None;
            }
            let value = e.access().clone();
            self.namespace_hits.fetch_add(1, AtomicOrdering::Relaxed);
            trace!("namespace cache hit: {key}");
            Some(value)
        } else {
            self.namespace_misses.fetch_add(1, AtomicOrdering::Relaxed);
            None
        }
    }

    pub fn put_namespace(&self, catalog: &str, namespace: &[String], ns: CatalogNamespace) {
        let key = ns_key(catalog, namespace);
        self.maybe_evict(&self.namespaces, &self.namespace_ring);
        self.namespaces.insert(key.clone(), CacheEntry::new(ns));
        self.track_insert(&self.namespace_ring, &key);
    }

    pub fn invalidate_namespace(&self, catalog: &str, namespace: &[String]) {
        let key = ns_key(catalog, namespace);
        self.namespaces.remove(&key);
        self.invalidations.fetch_add(1, AtomicOrdering::Relaxed);
        debug!("invalidated namespace: {key}");
    }

    // ---- Table ----

    pub fn get_table(&self, catalog: &str, id: &TableIdentifier) -> Option<CatalogTableSchema> {
        let key = tbl_key(catalog, id);
        if let Some(mut e) = self.tables.get_mut(&key) {
            if e.is_expired(self.ttl) {
                drop(e);
                self.tables.remove(&key);
                self.table_misses.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let value = e.access().clone();
            self.table_hits.fetch_add(1, AtomicOrdering::Relaxed);
            Some(value)
        } else {
            self.table_misses.fetch_add(1, AtomicOrdering::Relaxed);
            None
        }
    }

    pub fn put_table(&self, catalog: &str, id: &TableIdentifier, schema: CatalogTableSchema) {
        let key = tbl_key(catalog, id);
        self.maybe_evict(&self.tables, &self.table_ring);
        self.tables.insert(key.clone(), CacheEntry::new(schema));
        self.track_insert(&self.table_ring, &key);
    }

    pub fn invalidate_table_in_catalog(&self, catalog: &str, id: &TableIdentifier) {
        let key = tbl_key(catalog, id);
        self.tables.remove(&key);
        self.indexes.remove(&key);
        self.statistics.remove(&key);
        self.invalidations.fetch_add(1, AtomicOrdering::Relaxed);
        debug!("invalidated table {id} in catalog {catalog}");
    }

    pub fn invalidate_table(&self, id: &TableIdentifier) {
        let pattern = format!(".{}", id.to_fqn());
        self.tables.retain(|k, _| !k.ends_with(&pattern));
        self.indexes.retain(|k, _| !k.ends_with(&pattern));
        self.statistics.retain(|k, _| !k.ends_with(&pattern));
        self.invalidations.fetch_add(1, AtomicOrdering::Relaxed);
    }

    // ---- Indexes ----

    pub fn get_indexes(&self, catalog: &str, id: &TableIdentifier) -> Option<Vec<CatalogIndex>> {
        let key = tbl_key(catalog, id);
        if let Some(mut e) = self.indexes.get_mut(&key) {
            if e.is_expired(self.ttl) {
                drop(e);
                self.indexes.remove(&key);
                self.index_misses.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let value = e.access().clone();
            self.index_hits.fetch_add(1, AtomicOrdering::Relaxed);
            Some(value)
        } else {
            self.index_misses.fetch_add(1, AtomicOrdering::Relaxed);
            None
        }
    }

    pub fn put_indexes(&self, catalog: &str, id: &TableIdentifier, indexes: Vec<CatalogIndex>) {
        let key = tbl_key(catalog, id);
        self.maybe_evict(&self.indexes, &self.index_ring);
        self.indexes.insert(key.clone(), CacheEntry::new(indexes));
        self.track_insert(&self.index_ring, &key);
    }

    // ---- Statistics ----

    pub fn get_statistics(
        &self,
        catalog: &str,
        id: &TableIdentifier,
    ) -> Option<CatalogTableStatistics> {
        let key = tbl_key(catalog, id);
        if let Some(mut e) = self.statistics.get_mut(&key) {
            if e.is_expired(self.ttl) {
                drop(e);
                self.statistics.remove(&key);
                self.stats_misses.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let value = e.access().clone();
            self.stats_hits.fetch_add(1, AtomicOrdering::Relaxed);
            Some(value)
        } else {
            self.stats_misses.fetch_add(1, AtomicOrdering::Relaxed);
            None
        }
    }

    /// Cache-or-default statistics read for backends without native
    /// statistics: returns the cached value when present, otherwise caches
    /// and returns a default. Collapses the five identical per-backend stub
    /// bodies (glue/delta/iceberg/polaris/unity) onto one implementation.
    pub fn cached_default_statistics(
        &self,
        catalog: &str,
        id: &TableIdentifier,
    ) -> CatalogTableStatistics {
        if let Some(stats) = self.get_statistics(catalog, id) {
            return stats;
        }
        let stats = CatalogTableStatistics::default();
        self.put_statistics(catalog, id, stats.clone());
        stats
    }

    pub fn put_statistics(
        &self,
        catalog: &str,
        id: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) {
        let key = tbl_key(catalog, id);
        self.maybe_evict(&self.statistics, &self.stats_ring);
        self.statistics.insert(key.clone(), CacheEntry::new(stats));
        self.track_insert(&self.stats_ring, &key);
    }

    // ---- Maintenance ----

    pub fn clear(&self) {
        self.namespaces.clear();
        self.tables.clear();
        self.indexes.clear();
        self.statistics.clear();
        debug!("catalog cache cleared");
    }

    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            namespace_hits: self.namespace_hits.load(AtomicOrdering::Relaxed),
            namespace_misses: self.namespace_misses.load(AtomicOrdering::Relaxed),
            table_hits: self.table_hits.load(AtomicOrdering::Relaxed),
            table_misses: self.table_misses.load(AtomicOrdering::Relaxed),
            index_hits: self.index_hits.load(AtomicOrdering::Relaxed),
            index_misses: self.index_misses.load(AtomicOrdering::Relaxed),
            stats_hits: self.stats_hits.load(AtomicOrdering::Relaxed),
            stats_misses: self.stats_misses.load(AtomicOrdering::Relaxed),
            evictions: self.evictions.load(AtomicOrdering::Relaxed),
            invalidations: self.invalidations.load(AtomicOrdering::Relaxed),
        }
    }

    pub fn size(&self) -> usize {
        self.namespaces.len() + self.tables.len() + self.indexes.len() + self.statistics.len()
    }

    pub fn evict_expired(&self) {
        let ttl = self.ttl;
        let mut evicted = 0usize;
        {
            let before = self.namespaces.len();
            self.namespaces.retain(|_, v| !v.is_expired(ttl));
            evicted += before - self.namespaces.len();
        }
        {
            let before = self.tables.len();
            self.tables.retain(|_, v| !v.is_expired(ttl));
            evicted += before - self.tables.len();
        }
        {
            let before = self.indexes.len();
            self.indexes.retain(|_, v| !v.is_expired(ttl));
            evicted += before - self.indexes.len();
        }
        {
            let before = self.statistics.len();
            self.statistics.retain(|_, v| !v.is_expired(ttl));
            evicted += before - self.statistics.len();
        }
        if evicted > 0 {
            self.evictions
                .fetch_add(evicted as u64, AtomicOrdering::Relaxed);
        }
    }

    /// Sampled eviction (TD-CAT-5): at capacity, evict the least-recently-
    /// accessed entry among a bounded ring of the most recently inserted
    /// keys for THIS map's family — O(ring) single-shard reads, replacing
    /// the full-map `min_by_key` scan under the whole-map write lock. Victim
    /// choice relaxes from exact LRU to insert-recency-biased sampled LRU
    /// (a Redis-style approximation: an ancient cold entry can squat until
    /// TTL expires it); TTL expiry semantics are untouched.
    fn maybe_evict<T>(
        &self,
        cache: &DashMap<String, CacheEntry<T>>,
        ring: &std::sync::Mutex<std::collections::VecDeque<String>>,
    ) {
        if cache.len() < self.max_entries {
            return;
        }
        let candidate: Option<String> = {
            let mut ring = ring.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            // Drop ring entries that are no longer resident (invalidated or
            // already evicted) while scanning for the coldest victim.
            let mut victim: Option<(String, Instant)> = None;
            ring.retain(|key| match cache.get(key) {
                Some(entry) => {
                    if victim
                        .as_ref()
                        .is_none_or(|(_, coldest)| entry.last_accessed < *coldest)
                    {
                        victim = Some((key.clone(), entry.last_accessed));
                    }
                    true
                }
                None => false,
            });
            victim.map(|(key, _)| key)
        };
        if let Some(key) = candidate
            && cache.remove(&key).is_some()
        {
            self.evictions.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    /// Record an inserted key in the bounded eviction-sample ring for its family.
    fn track_insert(&self, ring: &std::sync::Mutex<std::collections::VecDeque<String>>, key: &str) {
        const RING_CAPACITY: usize = 64;
        let mut ring = ring.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if ring.len() >= RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(key.to_string());
    }
}

fn ns_key(catalog: &str, namespace: &[String]) -> String {
    format!("{}.{}", catalog, namespace.join("."))
}

fn tbl_key(catalog: &str, id: &TableIdentifier) -> String {
    format!("{}.{}", catalog, id.to_fqn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_table() {
        let cache = CatalogCache::new(100, 60);
        let id = TableIdentifier::new(vec!["db".into()], "users");
        let schema = CatalogTableSchema {
            name: "users".into(),
            ..Default::default()
        };
        cache.put_table("default", &id, schema);
        let result = cache.get_table("default", &id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "users");
    }

    #[test]
    fn cache_miss() {
        let cache = CatalogCache::new(100, 60);
        let id = TableIdentifier::new(vec!["db".into()], "missing");
        assert!(cache.get_table("default", &id).is_none());
    }

    #[test]
    fn cache_stats_hit_miss() {
        let cache = CatalogCache::new(100, 60);
        let id = TableIdentifier::new(vec!["db".into()], "t");
        let _ = cache.get_table("default", &id); // miss
        cache.put_table("default", &id, CatalogTableSchema::default());
        let _ = cache.get_table("default", &id); // hit
        let s = cache.get_stats();
        assert_eq!(s.table_misses, 1);
        assert_eq!(s.table_hits, 1);
    }

    #[test]
    fn invalidate_clears_entry() {
        let cache = CatalogCache::new(100, 60);
        let id = TableIdentifier::new(vec!["db".into()], "t");
        cache.put_table("default", &id, CatalogTableSchema::default());
        assert!(cache.get_table("default", &id).is_some());
        cache.invalidate_table_in_catalog("default", &id);
        assert!(cache.get_table("default", &id).is_none());
    }

    #[test]
    fn namespace_index_and_statistics_paths_track_hits_misses_and_invalidations() {
        let cache = CatalogCache::new(100, 60);
        let namespace = vec!["db".to_string(), "public".to_string()];
        let id = TableIdentifier::new(namespace.clone(), "users");

        assert!(cache.get_namespace("main", &namespace).is_none());
        cache.put_namespace(
            "main",
            &namespace,
            CatalogNamespace::new(namespace.clone()).with_owner("owner"),
        );
        assert_eq!(
            cache
                .get_namespace("main", &namespace)
                .unwrap()
                .owner
                .as_deref(),
            Some("owner")
        );

        assert!(cache.get_indexes("main", &id).is_none());
        cache.put_indexes(
            "main",
            &id,
            vec![CatalogIndex::new(
                "users_id_idx",
                vec!["id".to_string()],
                crate::CatalogIndexType::BTree,
            )],
        );
        assert_eq!(
            cache.get_indexes("main", &id).unwrap()[0].name,
            "users_id_idx"
        );

        assert!(cache.get_statistics("main", &id).is_none());
        cache.put_statistics(
            "main",
            &id,
            CatalogTableStatistics {
                row_count: 42,
                ..CatalogTableStatistics::default()
            },
        );
        assert_eq!(cache.get_statistics("main", &id).unwrap().row_count, 42);

        let stats = cache.get_stats();
        assert_eq!(stats.namespace_misses, 1);
        assert_eq!(stats.namespace_hits, 1);
        assert_eq!(stats.index_misses, 1);
        assert_eq!(stats.index_hits, 1);
        assert_eq!(stats.stats_misses, 1);
        assert_eq!(stats.stats_hits, 1);
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);

        cache.invalidate_namespace("main", &namespace);
        assert!(cache.get_namespace("main", &namespace).is_none());
        cache.invalidate_table(&id);
        assert!(cache.get_indexes("main", &id).is_none());
        assert!(cache.get_statistics("main", &id).is_none());
        assert!(cache.get_stats().invalidations >= 2);
    }

    /// Concurrency smoke: 8 threads hammering mixed get/put/invalidate ops
    /// must not deadlock, must leave consistent state, and must conserve
    /// counters (hits + misses == total gets) — the atomic-counter invariant
    /// the old second-RwLock stats design guaranteed implicitly. No timing
    /// assertions (CI-flaky by design).
    #[test]
    fn concurrent_mixed_ops_conserve_counters_and_do_not_deadlock() {
        use std::sync::Arc;

        let cache = Arc::new(CatalogCache::new(64, 60));
        let mut handles = Vec::new();
        const PER_THREAD: usize = 5_000;

        for t in 0..8usize {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                let shared_id = TableIdentifier::new(vec!["db".into()], "shared");
                let own_id = TableIdentifier::new(vec!["db".into()], format!("t{t}"));
                for i in 0..PER_THREAD {
                    // Shared key: contention path.
                    let _ = cache.get_table("default", &shared_id);
                    // Disjoint keys: shard-local paths.
                    cache.put_table(
                        "default",
                        &own_id,
                        CatalogTableSchema::new(format!("t{t}-{}", i % 8)),
                    );
                    let _ = cache.get_table("default", &own_id);
                    if i % 1_000 == 0 {
                        cache.invalidate_table_in_catalog("default", &own_id);
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread must not panic");
        }

        let stats = cache.get_stats();
        let total_gets = 8usize * PER_THREAD * 2;
        let counted = (stats.table_hits + stats.table_misses) as usize;
        assert_eq!(
            counted, total_gets,
            "hit+miss counters must equal the number of get_table calls"
        );
        assert!(stats.invalidations >= 8);
    }

    #[test]
    fn cache_clear_expiration_and_lru_eviction_cover_all_cache_families() {
        let cache = CatalogCache::default_cache();
        assert_eq!(cache.size(), 0);

        let id = TableIdentifier::new(vec!["db".into()], "t");
        cache.put_table("default", &id, CatalogTableSchema::new("t"));
        cache.put_indexes("default", &id, Vec::new());
        cache.put_statistics("default", &id, CatalogTableStatistics::default());
        cache.put_namespace(
            "default",
            &["db".into()],
            CatalogNamespace::new(vec!["db".into()]),
        );
        assert_eq!(cache.size(), 4);
        cache.clear();
        assert_eq!(cache.size(), 0);

        let expired = CatalogCache::new(100, 0);
        expired.put_table("default", &id, CatalogTableSchema::new("t"));
        expired.put_indexes("default", &id, Vec::new());
        expired.put_statistics("default", &id, CatalogTableStatistics::default());
        expired.put_namespace(
            "default",
            &["db".into()],
            CatalogNamespace::new(vec!["db".into()]),
        );
        std::thread::sleep(Duration::from_millis(1));
        expired.evict_expired();
        assert_eq!(expired.size(), 0);
        assert!(expired.get_stats().evictions >= 4);

        let bounded = CatalogCache::new(1, 60);
        let a = TableIdentifier::new(vec!["db".into()], "a");
        let b = TableIdentifier::new(vec!["db".into()], "b");
        bounded.put_table("default", &a, CatalogTableSchema::new("a"));
        bounded.put_table("default", &b, CatalogTableSchema::new("b"));
        assert!(bounded.get_table("default", &a).is_none());
        assert!(bounded.get_table("default", &b).is_some());
        assert_eq!(bounded.get_stats().evictions, 1);
    }

    /// Regression: a shared eviction-sample ring across families lets one
    /// family's `maybe_evict` call purge another family's keys from the ring
    /// (they're "not resident" in the map being checked), starving that
    /// other family's own eviction. Interleave puts across all four families
    /// at `max_entries == 1` each and assert every family stays at its cap —
    /// under the pre-fix shared ring, the namespace map grew to 2 entries
    /// here because the table family's evict check purged the namespace key
    /// from the ring first.
    #[test]
    fn eviction_rings_are_independent_per_family_under_interleaved_traffic() {
        let cache = CatalogCache::new(1, 60);
        let id = TableIdentifier::new(vec!["db".into()], "t");

        cache.put_namespace(
            "main",
            &["ns_a".into()],
            CatalogNamespace::new(vec!["ns_a".into()]),
        );
        cache.put_table("default", &id, CatalogTableSchema::new("tbl_a"));
        cache.put_table(
            "default",
            &TableIdentifier::new(vec!["db".into()], "t2"),
            CatalogTableSchema::new("tbl_b"),
        );
        cache.put_namespace(
            "main",
            &["ns_b".into()],
            CatalogNamespace::new(vec!["ns_b".into()]),
        );

        assert_eq!(
            cache.namespaces.len(),
            1,
            "namespace family must stay bounded at max_entries regardless of table-family traffic"
        );
        assert_eq!(cache.tables.len(), 1, "table family must stay bounded");
    }
}
