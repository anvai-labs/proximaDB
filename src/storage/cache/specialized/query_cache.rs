use crate::proto::proximadb_v1::SearchResult;
use crate::storage::cache::base::BaseCacheImpl;
use crate::storage::cache::traits::{BaseCache, CacheKey, CacheValue};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::SystemTime;

/// Query key that includes the query parameters
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryKey {
    pub collection_id: String,
    pub vector_hash: u64,
    pub k: u32,
    pub filters_hash: u64,
}

impl QueryKey {
    pub fn new(collection_id: String, vector: &[f32], k: u32, filters: Option<&str>) -> Self {
        let mut hasher = DefaultHasher::new();

        // Hash the vector
        for v in vector {
            v.to_bits().hash(&mut hasher);
        }
        let vector_hash = hasher.finish();

        // Hash the filters
        let mut hasher = DefaultHasher::new();
        if let Some(f) = filters {
            f.hash(&mut hasher);
        }
        let filters_hash = hasher.finish();

        Self {
            collection_id,
            vector_hash,
            k,
            filters_hash,
        }
    }
}

impl CacheKey for QueryKey {}

/// Cached query result with metadata
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    pub results: Vec<SearchResult>,
    pub cached_at: SystemTime,
    pub file_dependencies: Vec<String>,
    /// Canonical-WAL LSN at which this result was computed. Makes
    /// Strong-freshness reads cache-eligible: a Strong read may be served from
    /// cache iff this equals the current LSN (no write has landed since), so
    /// read-after-write still holds. `0` = "unversioned" (written by a
    /// non-LSN-aware path; never served to a Strong read via the LSN gate).
    pub computed_at_lsn: u64,
    /// Earliest TTL expiry (`valid_to_ns`, ns) among the cached records, or
    /// `None` when none of them expires. TTL expiry is time-based: it advances
    /// no LSN and triggers no invalidation, so the age and LSN checks cannot
    /// see it. Past this instant the entry is treated as a miss, otherwise a
    /// hit would serve a record every read path must hide (CLAUDE.md 16a/16c;
    /// TD-QCACHE-TTL-1). Writers compute it with
    /// `OptimizedSearchRecord::earliest_expiry_ns` before the v1 conversion,
    /// which drops the per-record expiry.
    pub min_valid_to_ns: Option<i64>,
}

impl CachedQueryResult {
    /// True once the earliest-expiring cached record has passed its TTL.
    pub fn has_expired_record(&self, now_ns: i64) -> bool {
        self.min_valid_to_ns.is_some_and(|ns| ns <= now_ns)
    }
}

/// Wall-clock now in ns (the same clock the read-path dead predicate uses).
fn now_ns() -> i64 {
    chrono::Utc::now().timestamp_nanos_opt().unwrap_or(i64::MAX)
}

impl CacheValue for CachedQueryResult {
    fn size_bytes(&self) -> usize {
        // Estimate: results + metadata
        self.results.len() * 64 + 256
    }
}

/// Specialized cache for query results with staleness detection
#[derive(Debug)]
pub struct QueryCache {
    base: BaseCacheImpl<QueryKey, CachedQueryResult>,
}

impl QueryCache {
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            base: BaseCacheImpl::new(max_memory_mb),
        }
    }

    /// Delegate put_with_hooks to base cache
    pub async fn put_with_hooks(&self, key: QueryKey, value: CachedQueryResult) {
        BaseCache::put_with_hooks(&self.base, key, value).await;
    }

    /// Delegate get_with_hooks to base cache
    pub async fn get_with_hooks(&self, key: &QueryKey) -> Option<CachedQueryResult> {
        BaseCache::get_with_hooks(&self.base, key).await
    }

    /// Get cached results if fresh
    pub async fn get_if_fresh(
        &self,
        key: &QueryKey,
        max_age_secs: u64,
    ) -> Option<Vec<SearchResult>> {
        if let Some(cached) = BaseCache::get_with_hooks(&self.base, key).await {
            let age = SystemTime::now()
                .duration_since(cached.cached_at)
                .unwrap_or_default()
                .as_secs();

            // TTL is time-based (no LSN / invalidation), so check it here.
            if age <= max_age_secs && !cached.has_expired_record(now_ns()) {
                return Some(cached.results);
            }
        }
        None
    }

    /// Like [`Self::get_if_fresh`], also returning the entry's earliest record
    /// expiry so a caller that re-caches the results (a second cache layer)
    /// can carry it forward instead of losing it.
    pub async fn get_if_fresh_with_expiry(
        &self,
        key: &QueryKey,
        max_age_secs: u64,
    ) -> Option<(Vec<SearchResult>, Option<i64>)> {
        let cached = BaseCache::get_with_hooks(&self.base, key).await?;
        let age = SystemTime::now()
            .duration_since(cached.cached_at)
            .unwrap_or_default()
            .as_secs();
        if age <= max_age_secs && !cached.has_expired_record(now_ns()) {
            Some((cached.results, cached.min_valid_to_ns))
        } else {
            None
        }
    }

    /// Get cached results as v1 if fresh (converts legacy to v1 on read)
    pub async fn get_if_fresh_v1(
        &self,
        key: &QueryKey,
        max_age_secs: u64,
    ) -> Option<Vec<crate::proto::proximadb_v1::SearchResult>> {
        if let Some(cached) = BaseCache::get_with_hooks(&self.base, key).await {
            let age = SystemTime::now()
                .duration_since(cached.cached_at)
                .unwrap_or_default()
                .as_secs();

            // TTL is time-based (no LSN / invalidation), so check it here.
            if age <= max_age_secs && !cached.has_expired_record(now_ns()) {
                return Some(
                    cached
                        .results
                        .into_iter()
                        // Results are already v1, no conversion needed
                        .collect(),
                );
            }
        }
        None
    }

    /// Strong-freshness variant of [`Self::get_if_fresh_v1`]. Serves a cached
    /// result to a Strong read ONLY when it was computed at the *current*
    /// canonical-WAL LSN — i.e. no write has advanced the LSN since, so the
    /// cached result is still read-after-write correct. `expected_lsn == 0`
    /// (LSN tracking unavailable) never hits, matching the delta-merge's
    /// "scan anyway when the LSN is unknown" guard. The TTL still applies as a
    /// backstop.
    pub async fn get_if_fresh_v1_at_lsn(
        &self,
        key: &QueryKey,
        max_age_secs: u64,
        expected_lsn: u64,
    ) -> Option<Vec<SearchResult>> {
        if expected_lsn == 0 {
            return None;
        }
        if let Some(cached) = BaseCache::get_with_hooks(&self.base, key).await {
            if cached.computed_at_lsn != expected_lsn {
                return None;
            }
            let age = SystemTime::now()
                .duration_since(cached.cached_at)
                .unwrap_or_default()
                .as_secs();
            // TTL is time-based (no LSN / invalidation), so check it here.
            if age <= max_age_secs && !cached.has_expired_record(now_ns()) {
                return Some(cached.results.into_iter().collect());
            }
        }
        None
    }

    /// Cache results with file dependencies
    pub async fn cache_with_dependencies(
        &self,
        key: QueryKey,
        results: Vec<SearchResult>,
        dependencies: Vec<String>,
        min_valid_to_ns: Option<i64>,
    ) {
        let cached = CachedQueryResult {
            results,
            cached_at: SystemTime::now(),
            file_dependencies: dependencies,
            computed_at_lsn: 0,
            min_valid_to_ns,
        };

        BaseCache::put_with_hooks(&self.base, key, cached).await;
    }

    /// Cache v1 results by converting to legacy for storage
    pub async fn cache_with_dependencies_v1(
        &self,
        key: QueryKey,
        results: Vec<crate::proto::proximadb_v1::SearchResult>,
        dependencies: Vec<String>,
        min_valid_to_ns: Option<i64>,
    ) {
        let legacy: Vec<SearchResult> = results
            .into_iter()
            // Results are already v1, no conversion needed
            .collect();
        let cached = CachedQueryResult {
            results: legacy,
            cached_at: SystemTime::now(),
            file_dependencies: dependencies,
            computed_at_lsn: 0,
            min_valid_to_ns,
        };
        BaseCache::put_with_hooks(&self.base, key, cached).await;
    }

    /// LSN-stamped variant of [`Self::cache_with_dependencies_v1`]. Records the
    /// canonical-WAL LSN the result was computed at so a later Strong read can
    /// validate freshness via [`Self::get_if_fresh_v1_at_lsn`].
    pub async fn cache_with_dependencies_v1_at_lsn(
        &self,
        key: QueryKey,
        results: Vec<SearchResult>,
        dependencies: Vec<String>,
        computed_at_lsn: u64,
        min_valid_to_ns: Option<i64>,
    ) {
        let legacy: Vec<SearchResult> = results.into_iter().collect();
        let cached = CachedQueryResult {
            results: legacy,
            cached_at: SystemTime::now(),
            file_dependencies: dependencies,
            computed_at_lsn,
            min_valid_to_ns,
        };
        BaseCache::put_with_hooks(&self.base, key, cached).await;
    }

    /// Invalidate all queries dependent on a file
    pub async fn invalidate_by_file(&self, _file_path: &str) {
        // Deferred: Implement file-based invalidation
        // This would track which queries depend on which files
    }

    /// Invalidate a specific query result
    pub async fn invalidate(&self, _key: &str) -> bool {
        // Convert string key to QueryKey if possible
        // For now, return false as we can't invalidate without proper QueryKey
        false
    }

    /// Invalidate ALL cached query results for a collection. Called on write
    /// (insert/update/delete/DDL) via the CacheInvalidationCoordinator so a
    /// write is immediately visible to subsequent reads (read-after-write).
    /// Returns the number of entries evicted.
    pub async fn invalidate_collection(&self, collection_id: &str) -> usize {
        self.base
            .remove_where(|k| k.collection_id == collection_id)
            .await
    }

    /// Resize the cache
    pub async fn resize(&self, _new_size_mb: usize) -> anyhow::Result<()> {
        // Deferred: Implement cache resizing
        Ok(())
    }

    /// Get cache metrics
    pub fn metrics(&self) -> &crate::storage::traits::UnifiedMetricsCollector {
        self.base.metrics()
    }

    /// Get the size of the cache (number of entries)
    pub async fn size(&self) -> usize {
        // Return the number of entries in the cache
        // This delegates to the base cache's size method
        self.base.size().await
    }

    /// Remove a specific entry from the cache
    pub async fn remove(&self, key: &QueryKey) -> Option<CachedQueryResult> {
        // Remove and return the cached entry if it exists
        self.base.remove(key).await
    }

    /// Remove entry by string key (for eviction system compatibility)
    /// The string should be a serialized QueryKey
    pub async fn remove_by_string(&self, key_str: &str) -> Option<CachedQueryResult> {
        // Try to deserialize the string to QueryKey
        if let Ok(key) = serde_json::from_str::<QueryKey>(key_str) {
            self.base.remove(&key).await
        } else {
            // If deserialization fails, can't remove
            None
        }
    }

    /// Get cache statistics
    pub async fn statistics(&self) -> SpecializedQueryCacheStatistics {
        // Get snapshot from unified metrics
        let snapshot = self.base.metrics().get_snapshot().await;

        SpecializedQueryCacheStatistics {
            total_entries: self.size().await,
            hit_count: snapshot.cache_hits,
            miss_count: snapshot.cache_misses,
            eviction_count: 0, // Not tracked directly in unified metrics
            memory_usage_bytes: self.base.memory_usage().await,
        }
    }

    /// Get memory usage of the cache in bytes
    pub async fn memory_usage(&self) -> usize {
        self.base.memory_usage().await
    }
}

/// Cache statistics structure
#[derive(Debug, Clone, Default)]
pub struct SpecializedQueryCacheStatistics {
    pub total_entries: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub memory_usage_bytes: usize,
}

#[cfg(test)]
mod lsn_gate_tests {
    use super::*;

    fn key(collection: &str) -> QueryKey {
        QueryKey::new(collection.to_string(), &[0.1f32, 0.2, 0.3], 5, None)
    }

    // A Strong read is cache-eligible ONLY at the LSN the entry was computed
    // at: same LSN → hit; advanced LSN (a write landed) → miss (recompute);
    // unknown LSN (0) → miss. This is the read-after-write validity predicate.
    #[tokio::test]
    async fn strong_read_hits_only_at_matching_lsn() {
        let cache = QueryCache::new(16);
        let k = key("c1");
        cache
            .cache_with_dependencies_v1_at_lsn(k.clone(), Vec::new(), Vec::new(), 42, None)
            .await;

        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 42).await.is_some());
        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 43).await.is_none());
        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 0).await.is_none());
    }

    // Entries written by a non-LSN-aware path (computed_at_lsn = 0) must never
    // be served to a Strong read, but remain valid for the plain TTL reader.
    #[tokio::test]
    async fn unversioned_entry_never_strong_hits() {
        let cache = QueryCache::new(16);
        let k = key("c2");
        cache
            .cache_with_dependencies_v1(k.clone(), Vec::new(), Vec::new(), None)
            .await;

        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 7).await.is_none());
        assert!(cache.get_if_fresh_v1(&k, 300).await.is_some());
    }

    // Read-after-write, eviction leg: a write drops the collection's entries
    // (CacheInvalidationCoordinator → invalidate_collection), so even an
    // LSN-pinned entry cannot survive a write — belt-and-suspenders alongside
    // the LSN-mismatch miss. Together these two legs are why a Strong read never
    // serves a result that predates a write.
    #[tokio::test]
    async fn write_eviction_drops_lsn_pinned_entry() {
        let cache = QueryCache::new(16);
        let k = key("c3");
        cache
            .cache_with_dependencies_v1_at_lsn(k.clone(), Vec::new(), Vec::new(), 99, None)
            .await;
        // Present at its LSN before the write.
        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 99).await.is_some());

        // A write invalidates the collection (removes by collection_id).
        assert_eq!(cache.invalidate_collection("c3").await, 1);

        // The LSN-pinned entry is gone → the next Strong read recomputes.
        assert!(cache.get_if_fresh_v1_at_lsn(&k, 300, 99).await.is_none());
    }
}

#[cfg(test)]
mod ttl_expiry_tests {
    //! TD-QCACHE-TTL-1: TTL expiry is time-based (no LSN advance, no
    //! invalidation), so every getter must refuse an entry once its
    //! earliest-expiring record has passed its TTL, including the LSN-gated
    //! Strong path at a MATCHING LSN.
    use super::*;

    fn key() -> QueryKey {
        QueryKey::new("ttl_c".to_string(), &[0.1f32, 0.2, 0.3], 5, None)
    }
    fn past() -> Option<i64> {
        Some(now_ns() - 1_000_000_000)
    }
    fn future() -> Option<i64> {
        Some(now_ns() + 3_600_000_000_000)
    }

    #[tokio::test]
    async fn expired_entry_is_a_miss_on_every_getter() {
        let cache = QueryCache::new(16);
        cache
            .cache_with_dependencies_v1_at_lsn(key(), Vec::new(), Vec::new(), 42, past())
            .await;
        assert!(
            cache.get_if_fresh(&key(), 300).await.is_none(),
            "get_if_fresh"
        );
        assert!(
            cache.get_if_fresh_v1(&key(), 300).await.is_none(),
            "get_if_fresh_v1"
        );
        assert!(
            cache
                .get_if_fresh_v1_at_lsn(&key(), 300, 42)
                .await
                .is_none(),
            "Strong read at the matching LSN must still miss an expired entry"
        );
        assert!(
            cache.get_if_fresh_with_expiry(&key(), 300).await.is_none(),
            "with_expiry"
        );
    }

    #[tokio::test]
    async fn unexpired_or_ttl_less_entry_still_hits() {
        let cache = QueryCache::new(16);
        cache
            .cache_with_dependencies_v1_at_lsn(key(), Vec::new(), Vec::new(), 42, future())
            .await;
        assert!(
            cache
                .get_if_fresh_v1_at_lsn(&key(), 300, 42)
                .await
                .is_some()
        );
        let (_, expiry) = cache
            .get_if_fresh_with_expiry(&key(), 300)
            .await
            .expect("future expiry is a hit");
        // The stored expiry travels with the hit (for a re-caching caller).
        assert!(
            expiry.is_some_and(|ns| ns > now_ns()),
            "expiry travels: {expiry:?}"
        );

        let cache = QueryCache::new(16);
        cache
            .cache_with_dependencies_v1(key(), Vec::new(), Vec::new(), None)
            .await;
        assert!(
            cache.get_if_fresh(&key(), 300).await.is_some(),
            "no TTL never expires"
        );
    }
}
