//! ADR-065 Q3 — engine-level ranged cache for coalesced-RaBitQ survivor/OID
//! byte ranges.
//!
//! ## Why this exists (co-design, first principles)
//!
//! The dominant read-path cost in coalesced ANN search is the billed
//! per-range object-store GET (`fs.read_range`) for survivor vectors (Region B
//! SQ8) and top-k OIDs (Region D) — ~14–40 ranged GETs/query at ~1 MiB each.
//! Region B/D are **immutable per segment file**, and survivors are
//! query-dependent **but repeat across hot queries** (overlapping IVF cells ⇒
//! overlapping survivor block sets ⇒ identical coalesced ranges). That is the
//! textbook "cache repeated reads of immutable data" lever, and it moves the
//! dominant cost term for a cloud DB — GET round-trips + egress — *not* CPU.
//!
//! ## Design choices (see plan: immutable-seeking-frog.md)
//!
//! - **Engine-level, not a `FileSystem` wrapper.** The engine knows the
//!   [`CacheTier`] semantics (survivors are LRU-evictable; invariants are
//!   pinned); the FS layer sees only opaque `(path, offset, length)` and cannot
//!   tier without a layering violation. This mirrors the existing
//!   [`SegmentInvariantsCache`] injection pattern.
//! - **Reuses `TenantCache`** (moka-backed, multitenant, byte-budgeted,
//!   work-conserving LRU) via its read-through `get_or_load` — no new eviction
//!   policy, no new metering.
//! - **RAM for v1.** The billed ms-latency GET dominates cost, not the cache
//!   medium, so a RAM hit wins; the SIFT1M survivor working set fits a
//!   few-hundred-MB budget. The optional persistent local-disk tier absorbs
//!   overflow and survives process restart; SSD/NVMe is recommended for latency.
//! - **Metering is free.** The cache sits *above* the FS backend: a HIT never
//!   runs the caller's loader, so `fs.read_range` is never called and the
//!   backend's `record_range_gets`/`record_bytes_read` never fire — the GET is
//!   simply not billed, and the caller records `record_get(CacheTier, …)`
//!   inside the loader so it fires exactly once per real fetch. No parallel
//!   cache hit/miss counters; the existing io_trace seam stays the single
//!   source of truth.
//! - **No invalidation for v1.** Segment files are immutable and uniquely
//!   named (`L0_{ts}_{hash}.pax`); a replaced segment has a new path ⇒ new
//!   keys ⇒ stale entries age out via the byte-budget LRU. No coherence risk.
//! - **Default-OFF.** Injected as `Option<Arc<…>>`; `None` ⇒ the read path is
//!   byte-for-byte unchanged. Opt in via env budget
//!   `PROXIMADB_SURVIVOR_CACHE_BUDGET_MB`.

use std::future::Future;
use std::sync::Arc;

use proximadb_cache::{
    CacheBudget, CacheKey, CacheKind, CacheReadOutcome, CacheScope, L2CacheStats, L2Class,
    L2ValueStore, PersistentArcBytesL2, PersistentByteStore, TenantCache,
};
use proximadb_storage_filesystem_types::{FilesystemError, FsResult};

/// The fair-sharing / isolation dimension for the survivor cache's
/// [`TenantCache`].
///
/// Data isolation is already structural: the [`CacheKey`] key string embeds the
/// full DrPath-scoped segment path (`data/{tenant}/{ns}/…`), so no two tenants
/// ever share a key. The `tenant` field here drives only the
/// *fair-sharing* dimension (per-tenant floors/ceilings under pool pressure).
/// v1 uses a single shared tenant (one elastic pool); per-tenant fair-sharing
/// is a follow-up that threads the real `tenant_id` into the search path.
/// Resolve the fair-share owner from the request trace. Missing control-plane
/// resolution uses the shared scope; aliases are never parsed or hashed into a
/// synthetic stable id. Data isolation remains structural in the `DrPath`.
fn request_cache_scope() -> CacheScope {
    crate::observability::io_trace::current_tenant_stable_id()
        .map(CacheScope::stable_tenant)
        .unwrap_or(CacheScope::Shared)
}

/// Persistent-L2 backend wrapper that forwards each exact-key probe outcome
/// into the ambient per-query io_trace (TD-IOTRACE-4). It lives at the engine
/// layer because the cache crate (foundation) must not depend on the
/// observability engine; the probe runs inside the query task, so the
/// task-local trace is in scope. Outcome semantics mirror `TenantCache`'s own
/// global counters: `Ok(Some)` = hit, `Ok(None)`/`Err` = miss (fail-open).
#[derive(Debug)]
struct TracedSurvivorL2(PersistentArcBytesL2);

#[async_trait::async_trait]
impl L2ValueStore<Arc<[u8]>> for TracedSurvivorL2 {
    async fn get(&self, key: &CacheKey) -> std::io::Result<Option<(Arc<[u8]>, u32)>> {
        let result = self.0.get(key).await;
        match &result {
            Ok(Some(_)) => crate::observability::io_trace::record_l2s(1, 0),
            Ok(None) | Err(_) => crate::observability::io_trace::record_l2s(0, 1),
        }
        result
    }

    async fn put(&self, key: &CacheKey, weight: u32, value: Arc<[u8]>) -> std::io::Result<()> {
        self.0.put(key, weight, value).await
    }

    async fn remove(&self, key: &CacheKey) -> std::io::Result<bool> {
        self.0.remove(key).await
    }

    fn resident_bytes(&self) -> u64 {
        self.0.resident_bytes()
    }
}

/// A byte-range cache for survivor (Region B SQ8) and OID (Region D) ranges of
/// immutable coalesced segments. Read-through: on a miss the caller-supplied
/// `loader` fetches the bytes; on a hit the loader never runs (so no GET is
/// issued and nothing is billed).
///
/// Backed by the multitenant, byte-budgeted, work-conserving [`TenantCache`].
pub struct SurvivorRangeCache {
    inner: Arc<TenantCache<Arc<[u8]>>>,
    l2_store: Option<Arc<PersistentByteStore>>,
    parent_l2_hits: std::sync::atomic::AtomicU64,
    parent_l2_misses: std::sync::atomic::AtomicU64,
    /// TD-CACHE-1 S2: per-tenant hot-key hit counters — the warm-set source
    /// for the shutdown manifest. Bounded (see `HOT_KEYS_CAP`); pruned by
    /// dropping the low-hit half when full. Keys are `(kind, path, off, len)`
    /// tuples, exactly what replay needs to re-issue coalesced loads.
    hot_keys: std::sync::Mutex<
        std::collections::HashMap<CacheScope, std::collections::HashMap<WarmKey, u64>>,
    >,
}

/// A replayable survivor-range identity (TD-CACHE-1 S2 manifest entry).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WarmKey {
    /// Artifact class as a stable u8 (`CacheKind::QuantizedCodes` = 0,
    /// everything else = 1) — enough to reconstruct the cache key class.
    pub k: u8,
    /// Segment path (tenant-scoped by construction — DrPath layout).
    pub p: String,
    /// Range offset/length.
    pub o: u64,
    pub l: u64,
}

/// Per-tenant cap on tracked hot keys (~100 B each ⇒ ≤ ~400 KB/tenant).
const HOT_KEYS_CAP: usize = 4096;

/// TD-RDSTRAT-12 §3: L2 namespace the exact-key write path uses (via
/// `TracedSurvivorL2`/`PersistentArcBytesL2` in [`Self::with_resolver_and_l2`]).
/// Shared by the write path and [`Self::peek_parent_residency`]'s exact-L2
/// probe so the two can never drift.
const L2_EXACT_NAMESPACE: &str = "survivor-exact";

impl SurvivorRangeCache {
    /// New cache with a `budget_bytes` byte ceiling (the shared elastic pool;
    /// also the per-tenant hard ceiling — entries beyond it bypass caching, so
    /// one collection cannot monopolize the pool).
    pub fn new(budget_bytes: u64) -> Self {
        Self::with_resolver(budget_bytes, None)
    }

    /// TD-CACHE-3 S1: construct with an optional per-tenant tier resolver
    /// (`TierPolicy::resolver` output) — the same seam the footer/index caches
    /// use. With a resolver, hot-tier tenants get admission floors /
    /// fair-share weights from the operator tier JSON; without one the pool
    /// stays uniform elastic fair share.
    ///
    /// TD-CACHE-3 S2: setting `PROXIMADB_SURVIVOR_PIN_RESERVE_FRAC` (0.0–0.5,
    /// default 0 = off) carves that fraction of the budget into the true-pin
    /// reserve — tier floors then become never-evicted-by-others pinned
    /// segments instead of admission-only preferences. Requires a resolver
    /// (floors come from the tier policy); without floors the reserve idles,
    /// so the flag is only meaningful together with `PROXIMADB_CACHE_TIERS_PATH`.
    pub fn with_resolver(
        budget_bytes: u64,
        resolver: Option<Arc<proximadb_cache::LimitsResolver>>,
    ) -> Self {
        Self::with_resolver_and_l2(budget_bytes, resolver, None)
    }

    /// Construct with the optional shared persistent byte store.
    pub fn with_resolver_and_l2(
        budget_bytes: u64,
        resolver: Option<Arc<proximadb_cache::LimitsResolver>>,
        l2_store: Option<Arc<PersistentByteStore>>,
    ) -> Self {
        let pin_frac = std::env::var("PROXIMADB_SURVIVOR_PIN_RESERVE_FRAC")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|f| *f > 0.0 && resolver.is_some())
            .unwrap_or(0.0);
        // TD-CACHE-2 S2c: cap OID/uncategorized ranges (CacheKind::Other) at a
        // fraction of the pool so OID churn cannot flush SQ8 survivor ranges
        // (QuantizedCodes — the recall-critical class — keeps ≥ the rest).
        let oid_ceiling = std::env::var("PROXIMADB_SURVIVOR_OID_CEILING_FRAC")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.3);
        let budget = CacheBudget::new(budget_bytes, budget_bytes)
            .with_high_watermark(0.9)
            .with_pin_reserve(pin_frac)
            .with_kind_ceiling(CacheKind::Other, oid_ceiling);
        let mut cache = match resolver {
            Some(r) => TenantCache::new(budget).with_limits_resolver(r),
            None => TenantCache::new(budget),
        };
        if let Some(store) = &l2_store {
            cache = cache.with_l2_backend(Arc::new(TracedSurvivorL2(PersistentArcBytesL2::new(
                store.clone(),
                L2_EXACT_NAMESPACE,
                L2Class::Survivor,
            ))));
        }
        Self {
            inner: Arc::new(cache),
            l2_store,
            parent_l2_hits: std::sync::atomic::AtomicU64::new(0),
            parent_l2_misses: std::sync::atomic::AtomicU64::new(0),
            hot_keys: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn parent_key(path: &str, off: u64, len: u64) -> String {
        format!("survivor-parent/{path}:{off}:{len}")
    }

    fn track_hot_key(&self, scope: &CacheScope, kind: CacheKind, path: &str, off: u64, len: u64) {
        if let Ok(mut hot) = self.hot_keys.lock() {
            let per_tenant = hot.entry(scope.clone()).or_default();
            if per_tenant.len() >= HOT_KEYS_CAP {
                let mut counts: Vec<u64> = per_tenant.values().copied().collect();
                counts.sort_unstable();
                let median = counts[counts.len() / 2];
                per_tenant.retain(|_, hits| *hits > median);
            }
            *per_tenant
                .entry(WarmKey {
                    k: match kind {
                        CacheKind::QuantizedCodes => 0,
                        _ => 1,
                    },
                    p: path.to_string(),
                    o: off,
                    l: len,
                })
                .or_insert(0) += 1;
        }
    }

    /// Seed a complete immutable region after its segment is atomically
    /// published. The fallback-tenant L1 entry is process-wide; the stable L2
    /// parent key lets any request tenant read exact subranges after restart.
    pub async fn seed_parent_region(&self, path: &str, off: u64, bytes: Arc<[u8]>) -> FsResult<()> {
        let len = bytes.len() as u64;
        let key = CacheKey::shared(CacheKind::QuantizedCodes, format!("{path}:{off}:{len}"));
        let weight = bytes.len().try_into().unwrap_or(u32::MAX);
        // Persist the complete region only under the range-aware parent key.
        // The exact-key TenantCache adapter would otherwise duplicate the
        // entire SQ8 region in the same L2.
        self.inner
            .insert_memory_only(key, weight, bytes.clone())
            .await;
        if let Some(store) = &self.l2_store {
            store
                .put(Self::parent_key(path, off, len), L2Class::Survivor, bytes)
                .await
                .map_err(FilesystemError::Io)?;
        }
        self.sync_prometheus();
        Ok(())
    }

    /// Stream a complete immutable parent region from an already-written local
    /// segment into persistent L2. Unlike [`Self::seed_parent_region`], this
    /// does not create a segment-sized DRAM value; demanded subranges promote
    /// lazily through [`Self::get_or_fetch_in_parent`].
    pub async fn seed_parent_region_from_file(
        &self,
        path: &str,
        off: u64,
        len: u64,
        local_path: &std::path::Path,
    ) -> FsResult<bool> {
        let Some(store) = &self.l2_store else {
            return Ok(false);
        };
        store
            .put_file_range(
                Self::parent_key(path, off, len),
                L2Class::Survivor,
                local_path,
                off,
                len,
            )
            .await
            .map_err(FilesystemError::Io)?;
        self.sync_prometheus();
        Ok(true)
    }

    /// Read through the exact-range cache, then a write-time parent-region
    /// seed, and only then the authoritative loader.
    #[allow(clippy::too_many_arguments)]
    pub async fn get_or_fetch_in_parent<F, Fut>(
        &self,
        kind: CacheKind,
        path: &str,
        off: u64,
        len: u64,
        parent_off: u64,
        parent_len: u64,
        loader: F,
    ) -> FsResult<Arc<[u8]>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = FsResult<Vec<u8>>>,
    {
        let Some(relative_off) = off.checked_sub(parent_off) else {
            return self.get_or_fetch(kind, path, off, len, loader).await;
        };
        if relative_off.saturating_add(len) > parent_len {
            return self.get_or_fetch(kind, path, off, len, loader).await;
        }
        let scope = request_cache_scope();
        self.track_hot_key(&scope, kind, path, off, len);
        let exact_key = CacheKey::with_scope(scope, kind, format!("{path}:{off}:{len}"));
        let weight = len.try_into().unwrap_or(u32::MAX);
        let parent_key = CacheKey::shared(
            CacheKind::QuantizedCodes,
            format!("{path}:{parent_off}:{parent_len}"),
        );
        let Ok(start) = usize::try_from(relative_off) else {
            return self.get_or_fetch(kind, path, off, len, loader).await;
        };
        let Ok(slice_len) = usize::try_from(len) else {
            return self.get_or_fetch(kind, path, off, len, loader).await;
        };
        if let Some((exact, outcome)) = self.inner.get_with_outcome(&exact_key).await {
            crate::observability::io_trace::record_survivor_l1s(
                u64::from(outcome == CacheReadOutcome::MemoryHit),
                u64::from(outcome != CacheReadOutcome::MemoryHit),
            );
            self.sync_prometheus();
            return Ok(exact);
        }
        let end = start.saturating_add(slice_len);
        if let Some((parent, outcome)) = self.inner.get_with_outcome(&parent_key).await
            && let Some(slice) = parent.get(start..end)
        {
            crate::observability::io_trace::record_survivor_l1s(
                u64::from(outcome == CacheReadOutcome::MemoryHit),
                u64::from(outcome != CacheReadOutcome::MemoryHit),
            );
            // A complete write-time seed already occupies the useful cache
            // slot. Do not admit every query-dependent slice as another exact
            // entry: doing so duplicates the region, evicts its parent, and
            // synchronously spills tens of MiB/query into L2.
            self.sync_prometheus();
            return Ok(Arc::from(slice));
        }
        if let Some(store) = &self.l2_store {
            let persistent_key = Self::parent_key(path, parent_off, parent_len);
            if let Ok(Some(bytes)) = store.get_range(&persistent_key, relative_off, len).await {
                self.parent_l2_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::observability::io_trace::record_l2s(1, 0);
                // Promote only the demanded range. `insert_memory_only`
                // deliberately avoids writing an exact-range duplicate
                // beside the durable parent entry.
                self.inner
                    .insert_memory_only(exact_key, weight, bytes.clone())
                    .await;
                crate::observability::io_trace::record_survivor_l1s(0, 1);
                self.sync_prometheus();
                return Ok(bytes);
            }
            self.parent_l2_misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::observability::io_trace::record_l2s(0, 1);
        }
        let result = self
            .inner
            .get_or_load(exact_key, weight, || async {
                Ok::<Arc<[u8]>, FilesystemError>(Arc::from(loader().await?))
            })
            .await;
        crate::observability::io_trace::record_survivor_l1s(0, 1);
        self.sync_prometheus();
        result.map(|(value, _)| value)
    }

    pub fn l2_stats(&self) -> L2CacheStats {
        let exact = self.inner.l2_stats();
        L2CacheStats {
            hits: exact.hits
                + self
                    .parent_l2_hits
                    .load(std::sync::atomic::Ordering::Relaxed),
            misses: exact.misses
                + self
                    .parent_l2_misses
                    .load(std::sync::atomic::Ordering::Relaxed),
            resident_bytes: exact.resident_bytes,
        }
    }

    /// TD-CACHE-2 S2d: evict every cached range of a segment file (all
    /// tenants, all kinds — the file is gone). Called by compaction after it
    /// deletes an input file; keys are `{path}:{off}:{len}`.
    pub async fn purge_path(&self, path: &str) -> usize {
        let prefix = format!("{path}:");
        let mut removed = self
            .inner
            .purge_where(|k| k.key.starts_with(prefix.as_str()))
            .await;
        if let Some(store) = &self.l2_store {
            let parent_prefix = format!("survivor-parent/{path}:");
            let quantized_marker = format!("/{}:{path}:", CacheKind::QuantizedCodes.stable_id());
            let other_marker = format!("/{}:{path}:", CacheKind::Other.stable_id());
            removed += store
                .remove_where(|key| {
                    key.starts_with(&parent_prefix)
                        || (key.starts_with("survivor-exact/")
                            && (key.contains(&quantized_marker) || key.contains(&other_marker)))
                })
                .await;
        }
        if let Ok(mut hot) = self.hot_keys.lock() {
            for keys in hot.values_mut() {
                keys.retain(|key, _| key.p != path);
            }
            hot.retain(|_, keys| !keys.is_empty());
        }
        removed
    }

    /// Remove every range below a retired collection directory while
    /// preserving sibling collections that merely share a textual prefix.
    pub async fn purge_prefix(&self, path_prefix: &str) -> usize {
        let prefix = format!("{}/", path_prefix.trim_end_matches('/'));
        let mut removed = self
            .inner
            .purge_where(|key| key.key.starts_with(&prefix))
            .await;
        if let Some(store) = &self.l2_store {
            removed += store
                .remove_where(|key| {
                    (key.starts_with("survivor-parent/") || key.starts_with("survivor-exact/"))
                        && key.contains(&prefix)
                })
                .await;
        }
        if let Ok(mut hot) = self.hot_keys.lock() {
            for keys in hot.values_mut() {
                keys.retain(|key, _| !key.p.starts_with(&prefix));
            }
            hot.retain(|_, keys| !keys.is_empty());
        }
        removed
    }

    /// Look up the byte range `[off, off+len)` of `path`; on a miss run `loader`
    /// and cache its result iff the tenant is under its byte ceiling. Returns
    /// the cached or freshly-loaded bytes.
    ///
    /// `kind` selects the artifact class — pass [`CacheKind::QuantizedCodes`]
    /// for survivor (SQ8) ranges and [`CacheKind::Other`] for OID ranges, so
    /// per-kind stats stay separated.
    ///
    /// The loader runs ONLY on a miss, so a caller that records the GET (e.g.
    /// `record_get(CacheTier::SurvivorPayload, len)`) inside the loader records
    /// it exactly once per real fetch — a hit records nothing.
    pub async fn get_or_fetch<F, Fut>(
        &self,
        kind: CacheKind,
        path: &str,
        off: u64,
        len: u64,
        loader: F,
    ) -> FsResult<Arc<[u8]>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = FsResult<Vec<u8>>>,
    {
        // TD-CACHE-3 S1: attribute the entry to the requesting tenant (ambient
        // from the per-request io_trace scope; falls back to the shared pool id
        // outside a scope). Enables per-tenant hit/miss/bytes stats and the
        // tier-resolver fair-share floors.
        let scope = request_cache_scope();
        // TD-CACHE-1 S2: count this range toward the tenant's warm set (the
        // shutdown manifest's source). Bounded map; low-hit half pruned at cap.
        self.track_hot_key(&scope, kind, path, off, len);
        let key = CacheKey::with_scope(scope, kind, format!("{path}:{off}:{len}"));
        // Ranges are ≤ a few MiB; cap the weight at u32::MAX defensively.
        let weight: u32 = len.try_into().unwrap_or(u32::MAX);
        let result = self
            .inner
            .get_or_load(key, weight, || async {
                let bytes = loader().await?;
                Ok::<Arc<[u8]>, FilesystemError>(Arc::from(bytes))
            })
            .await;
        if let Ok((_, outcome)) = &result {
            crate::observability::io_trace::record_survivor_l1s(
                u64::from(*outcome == CacheReadOutcome::MemoryHit),
                u64::from(*outcome != CacheReadOutcome::MemoryHit),
            );
        }
        self.sync_prometheus();
        result.map(|(value, _)| value)
    }

    /// TD-RDSTRAT-12 §3: side-effect-free residency probe for batch-prefetch
    /// planning — `Some(bytes)` iff a subsequent [`Self::get_or_fetch`] with
    /// these exact coordinates would be served from DRAM (pinned or L1)
    /// WITHOUT running its loader. No `track_hot_key`, no hit/miss counters,
    /// no inserts, no persistent-tier reads. Deliberately conservative: an
    /// `None` answer may still be served from a lower tier by the eventual
    /// consume call, so callers must treat every batched range as payable I/O
    /// they intended to issue anyway.
    pub async fn peek_memory_exact(
        &self,
        kind: CacheKind,
        path: &str,
        off: u64,
        len: u64,
    ) -> Option<Arc<[u8]>> {
        let scope = request_cache_scope();
        let key = CacheKey::with_scope(scope, kind, format!("{path}:{off}:{len}"));
        self.inner.peek_memory(&key).await
    }

    /// TD-IOBUDGET-2 review: exact-key residency probe WITH the L2 step —
    /// `Some(bytes)` iff a subsequent [`Self::get_or_fetch`] with these exact
    /// coordinates would be served WITHOUT running its loader, from DRAM
    /// (pinned/L1) OR from the persistent exact-key L2. Mirrors
    /// [`Self::peek_parent_residency`]'s steps 1+4: without the L2 probe a
    /// RAM-only peek classifies L1-evicted-but-L2-resident ranges as cold and
    /// re-GETs them from the object store (a warm-phase billing regression) —
    /// the wave GET's bytes are then dropped at the consume call because
    /// `get_or_fetch` serves the range from L2. Probed through the RAW
    /// `PersistentByteStore` (never the traced wrapper) so no io_trace or l2
    /// counters move; the only effect is an LRU recency tick plus a local-disk
    /// read attempt (~ms, gated-only).
    pub async fn peek_exact_residency(
        &self,
        kind: CacheKind,
        path: &str,
        off: u64,
        len: u64,
    ) -> Option<Arc<[u8]>> {
        let scope = request_cache_scope();
        let key = CacheKey::with_scope(scope, kind, format!("{path}:{off}:{len}"));
        if let Some(bytes) = self.inner.peek_memory(&key).await {
            return Some(bytes);
        }
        self.peek_exact_l2(&key, len).await
    }

    /// TD-RDSTRAT-12 §3 (round 4): the [`Self::get_or_fetch_in_parent`] mirror
    /// of [`Self::peek_memory_exact`] — `Some(bytes)` iff a subsequent
    /// `get_or_fetch_in_parent(kind, path, off, len, parent_off, parent_len)`
    /// would be served WITHOUT running its loader. Probes, in the same order
    /// as the real method and ALL side-effect-free (no `track_hot_key`, no
    /// hit/miss or `parent_l2_*` counters, no `record_l2s`, no L1 promotion —
    /// every accounting in `get_or_fetch_in_parent` lives in its body, not in
    /// the stores):
    ///
    /// 1. exact-range L1 (the `{path}:{off}:{len}` scoped key);
    /// 2. parent-region L1 slice (the write-time seed key);
    /// 3. parent-key L2 range (`survivor-parent/…`, covering seeds);
    /// 4. **exact-key L2** (`persistent_key(L2_EXACT_NAMESPACE)`) — the
    ///    clustered-loop loaders persist under exact keys via `get_or_load`'s
    ///    write-through; without this step a RAM-only peek classifies
    ///    L1-evicted-but-L2-resident ranges as cold and re-GETs them from the
    ///    object store (a warm-phase billing regression). Probed through the
    ///    RAW `PersistentByteStore` (never the traced wrapper) so no io_trace
    ///    or l2 counters move. The only effect is an LRU recency tick plus a
    ///    local-disk read (~ms per 4 MiB range, gated-only) — accepted.
    ///
    /// `None` ⇒ the loader WILL run ⇒ the caller may batch this range.
    pub async fn peek_parent_residency(
        &self,
        kind: CacheKind,
        path: &str,
        off: u64,
        len: u64,
        parent_off: u64,
        parent_len: u64,
    ) -> Option<Arc<[u8]>> {
        let scope = request_cache_scope();
        // 1. Exact-range L1 (same key get_or_fetch_in_parent builds).
        let exact_key = CacheKey::with_scope(scope.clone(), kind, format!("{path}:{off}:{len}"));
        if let Some(bytes) = self.inner.peek_memory(&exact_key).await {
            return Some(bytes);
        }
        // 2./3. Parent tiers (mirror the real method's coordinate checks).
        let Some(relative_off) = off.checked_sub(parent_off) else {
            // Outside the parent: the real method falls back to plain
            // get_or_fetch semantics — only the exact tiers apply.
            return self.peek_exact_l2(&exact_key, len).await;
        };
        if relative_off.saturating_add(len) > parent_len {
            return self.peek_exact_l2(&exact_key, len).await;
        }
        let parent_key = CacheKey::shared(
            CacheKind::QuantizedCodes,
            format!("{path}:{parent_off}:{parent_len}"),
        );
        let Ok(start) = usize::try_from(relative_off) else {
            return self.peek_exact_l2(&exact_key, len).await;
        };
        let Ok(slice_len) = usize::try_from(len) else {
            return self.peek_exact_l2(&exact_key, len).await;
        };
        let end = start.saturating_add(slice_len);
        // 2. Parent-region L1 slice.
        if let Some(parent) = self.inner.peek_memory(&parent_key).await
            && parent.get(start..end).is_some()
        {
            return Some(parent);
        }
        // 3. Parent-key L2 (covers write-time seeds). Raw store, no counters.
        if let Some(store) = &self.l2_store
            && let Ok(Some(bytes)) = store
                .get_range(
                    &Self::parent_key(path, parent_off, parent_len),
                    relative_off,
                    len,
                )
                .await
        {
            return Some(bytes);
        }
        // 4. Exact-key L2 (clustered-loop loader write-through entries).
        self.peek_exact_l2(&exact_key, len).await
    }

    /// Exact-key L2 probe through the RAW store — no `TracedSurvivorL2`
    /// io_trace recording, no L1 promotion, no counters.
    async fn peek_exact_l2(&self, exact_key: &CacheKey, len: u64) -> Option<Arc<[u8]>> {
        let store = self.l2_store.as_ref()?;
        store
            .get_range(&exact_key.persistent_key(L2_EXACT_NAMESPACE), 0, len)
            .await
            .ok()
            .flatten()
    }

    /// TD-CACHE-1 S2: the top-`k` hot ranges per tenant by hit count — the
    /// shutdown manifest payload. Cheap snapshot under the tracking lock.
    pub fn warm_set(&self, top_k: usize) -> Vec<(String, Vec<WarmKey>)> {
        let Ok(hot) = self.hot_keys.lock() else {
            return Vec::new();
        };
        hot.iter()
            .map(|(tenant, keys)| {
                let mut ranked: Vec<(&WarmKey, u64)> = keys.iter().map(|(k, h)| (k, *h)).collect();
                ranked.sort_by_key(|(_, hits)| std::cmp::Reverse(*hits));
                (
                    tenant.label(),
                    ranked
                        .into_iter()
                        .take(top_k)
                        .map(|(k, _)| k.clone())
                        .collect(),
                )
            })
            .collect()
    }

    /// TD-CACHE-1 S2: replay a warm entry — a coalesced read-through load of
    /// the range so post-restart queries hit DRAM. The caller supplies the
    /// loader (a ranged GET); a missing/compacted-away segment simply errors
    /// into a skip. Returns whether the entry loaded.
    pub async fn replay_entry<F, Fut>(&self, entry: &WarmKey, loader: F) -> bool
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = FsResult<Vec<u8>>>,
    {
        let kind = if entry.k == 0 {
            CacheKind::QuantizedCodes
        } else {
            CacheKind::Other
        };
        self.get_or_fetch(kind, &entry.p, entry.o, entry.l, loader)
            .await
            .is_ok()
    }

    /// TD-METRICS-1: mirror the `TenantCache` hit/miss/bytes atomics into the
    /// prometheus gauges after each lookup. `tenant_stats()` iterates a
    /// handful of (tenant, kind) rows — negligible next to the ranged GET the
    /// cache exists to avoid. Gauges (not counters) because the source is a
    /// running total we sample.
    fn sync_prometheus(&self) {
        /// Cardinality bound for per-tenant series (top by resident bytes).
        const SURVIVOR_TENANT_SERIES_CAP: usize = 50;
        let mut stats = self.inner.tenant_stats();
        let (mut hits, mut misses, mut bytes) = (0i64, 0i64, 0i64);
        for stat in &stats {
            hits += stat.hits as i64;
            misses += stat.misses as i64;
            bytes += stat.bytes as i64;
        }
        crate::metrics::operational_metrics::SURVIVOR_CACHE_HITS.set(hits);
        crate::metrics::operational_metrics::SURVIVOR_CACHE_MISSES.set(misses);
        crate::metrics::operational_metrics::SURVIVOR_CACHE_BYTES.set(bytes);
        crate::metrics::operational_metrics::sync_local_disk_stats("survivor", self.l2_stats());
        // TD-CACHE-3 S1: per-tenant series, bounded to the top tenants by
        // resident bytes (noisy-neighbor + hot-tier entitlement signal).
        stats.sort_by_key(|s| std::cmp::Reverse(s.bytes));
        for stat in stats.iter().take(SURVIVOR_TENANT_SERIES_CAP) {
            use crate::metrics::operational_metrics as om;
            om::SURVIVOR_CACHE_TENANT_BYTES
                .with_label_values(&[&*stat.tenant])
                .set(stat.bytes as i64);
            om::SURVIVOR_CACHE_TENANT_HITS
                .with_label_values(&[&*stat.tenant])
                .set(stat.hits as i64);
            om::SURVIVOR_CACHE_TENANT_MISSES
                .with_label_values(&[&*stat.tenant])
                .set(stat.misses as i64);
            // TD-CACHE-3 S3: enforcement metering — pinned vs entitled. The
            // billing true-up charges what is entitled; a sustained
            // pinned < entitled gap under pressure is the capacity signal to
            // move the tenant/node (residency cannot be honored).
            om::SURVIVOR_CACHE_TENANT_PINNED_BYTES
                .with_label_values(&[&*stat.tenant])
                .set(stat.pinned_bytes as i64);
            om::SURVIVOR_CACHE_TENANT_ENTITLED_BYTES
                .with_label_values(&[&*stat.tenant])
                .set(self.inner.entitlement(&stat.tenant).floor_bytes as i64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_seed_serves_subrange_from_dram_and_persistent_restart() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent L2"));
        let cache = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let parent: Arc<[u8]> = Arc::from((0u8..32).collect::<Vec<_>>());
        cache
            .seed_parent_region("seg.pax", 1_000, parent)
            .await
            .expect("seed parent");

        let loads = Arc::new(AtomicUsize::new(0));
        let loads_first = loads.clone();
        let bytes = cache
            .get_or_fetch_in_parent(
                CacheKind::QuantizedCodes,
                "seg.pax",
                1_008,
                4,
                1_000,
                32,
                move || {
                    let loads = loads_first.clone();
                    async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![99; 4])
                    }
                },
            )
            .await
            .expect("DRAM parent slice");
        assert_eq!(bytes.as_ref(), &[8, 9, 10, 11]);
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert_eq!(
            store.entry_count(),
            1,
            "parent hits must not duplicate query-dependent exact ranges in L2"
        );
        drop(cache);
        drop(store);

        let reopened =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen persistent L2"));
        let restarted =
            SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(reopened.clone()));
        let loads_restart = loads.clone();
        let bytes = restarted
            .get_or_fetch_in_parent(
                CacheKind::QuantizedCodes,
                "seg.pax",
                1_016,
                4,
                1_000,
                32,
                move || {
                    let loads = loads_restart.clone();
                    async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![99; 4])
                    }
                },
            )
            .await
            .expect("persistent parent slice");
        assert_eq!(bytes.as_ref(), &[16, 17, 18, 19]);
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert_eq!(restarted.l2_stats().hits, 1);
        assert_eq!(
            reopened.entry_count(),
            1,
            "restart promotion keeps the parent as the only persistent copy"
        );

        assert!(
            restarted.purge_path("seg.pax").await >= 1,
            "path invalidation removes DRAM and persistent entries"
        );
        drop(restarted);

        let reopened_after_purge =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen after purge"));
        let cold =
            SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(reopened_after_purge));
        let loads_after_purge = loads.clone();
        let bytes = cold
            .get_or_fetch_in_parent(
                CacheKind::QuantizedCodes,
                "seg.pax",
                1_016,
                4,
                1_000,
                32,
                move || {
                    let loads = loads_after_purge.clone();
                    async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![99; 4])
                    }
                },
            )
            .await
            .expect("loader after invalidation");
        assert_eq!(bytes.as_ref(), &[99; 4]);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    /// TD-CACHE-3 S2 e2e (survivor level): tenant A's tier floor is pinned —
    /// tenant B churning far past the pool cannot evict it. A's ranges are
    /// still served without re-running the loader after B's flood, inside
    /// each tenant's real io_trace scope (the S1 ambient route).
    #[tokio::test]
    async fn pinned_tenant_floor_survives_churner_flood() {
        // SAFETY: nextest runs process-per-test; the env var cannot leak.
        unsafe { std::env::set_var("PROXIMADB_SURVIVOR_PIN_RESERVE_FRAC", "0.5") };
        let resolver: Arc<proximadb_cache::LimitsResolver> =
            Arc::new(|tenant: &str| proximadb_cache::TenantLimits {
                floor_bytes: if tenant == "101" { 4_000 } else { 0 },
                hard_ceiling_bytes: 10_000,
                weight: 1,
            });
        let cache = SurvivorRangeCache::with_resolver(10_000, Some(resolver));

        // A loads its working set inside its tenant scope (4 × 1 KB = floor).
        crate::observability::io_trace::instrument_with_stable_tenant(
            Some("tenant-a".to_string()),
            Some(101),
            "test",
            async {
                for i in 0..4u64 {
                    cache
                        .get_or_fetch(
                            CacheKind::QuantizedCodes,
                            "a.pax",
                            i * 1_000,
                            1_000,
                            || async { Ok(vec![i as u8; 1_000]) },
                        )
                        .await
                        .unwrap();
                }
            },
        )
        .await;

        // B floods 100 KB through the 5 KB shared pool from its own scope.
        crate::observability::io_trace::instrument_with_stable_tenant(
            Some("tenant-b".to_string()),
            Some(202),
            "test",
            async {
                for i in 0..100u64 {
                    cache
                        .get_or_fetch(
                            CacheKind::QuantizedCodes,
                            "b.pax",
                            i * 1_000,
                            1_000,
                            || async { Ok(vec![0u8; 1_000]) },
                        )
                        .await
                        .unwrap();
                }
            },
        )
        .await;

        // A's floor must be served from the pinned segment: loader NOT re-run.
        crate::observability::io_trace::instrument_with_stable_tenant(
            Some("tenant-a".to_string()),
            Some(101),
            "test",
            async {
                for i in 0..4u64 {
                    let reloaded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let r2 = reloaded.clone();
                    let bytes = cache
                        .get_or_fetch(
                            CacheKind::QuantizedCodes,
                            "a.pax",
                            i * 1_000,
                            1_000,
                            move || {
                                let r = r2.clone();
                                async move {
                                    r.store(true, std::sync::atomic::Ordering::SeqCst);
                                    Ok(vec![i as u8; 1_000])
                                }
                            },
                        )
                        .await
                        .unwrap();
                    assert_eq!(bytes[0], i as u8);
                    assert!(
                        !reloaded.load(std::sync::atomic::Ordering::SeqCst),
                        "range {i}: pinned floor entry must survive B's churn"
                    );
                }
            },
        )
        .await;
    }

    /// TD-IOTRACE-4 acceptance: a query served from the persistent L2 tier
    /// records a nonzero `l2_hits` in the ambient per-query io_trace — the
    /// restart-cold DRAM tier is exactly the footer-cold-but-L2-warm state the
    /// counters exist to make observable. (Gated like the recorder itself:
    /// without the `io-trace` feature the free fns are no-ops.)
    #[cfg(feature = "io-trace")]
    #[tokio::test]
    async fn l2_probe_outcomes_reach_the_ambient_io_trace() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent L2"));
        let cache = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let parent: Arc<[u8]> = Arc::from((0u8..32).collect::<Vec<_>>());
        cache
            .seed_parent_region("seg.pax", 1_000, parent)
            .await
            .expect("seed parent");
        drop(cache);
        drop(store);

        let reopened =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("reopen persistent L2"));
        let restarted = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(reopened));
        let loads = Arc::new(AtomicUsize::new(0));
        let loads_in_scope = loads.clone();
        let snap =
            crate::observability::io_trace::instrument(Some("t".to_string()), "test", async move {
                let bytes = restarted
                    .get_or_fetch_in_parent(
                        CacheKind::QuantizedCodes,
                        "seg.pax",
                        1_008,
                        4,
                        1_000,
                        32,
                        move || {
                            let loads = loads_in_scope.clone();
                            async move {
                                loads.fetch_add(1, Ordering::SeqCst);
                                Ok(vec![99; 4])
                            }
                        },
                    )
                    .await
                    .expect("persistent parent slice");
                assert_eq!(bytes.as_ref(), &[8, 9, 10, 11]);
                crate::observability::io_trace::snapshot().expect("trace in scope")
            })
            .await;
        assert_eq!(
            loads.load(Ordering::SeqCst),
            0,
            "served from L2, not the GET"
        );
        assert!(
            snap.l2_hits >= 1,
            "parent-L2 hit must be recorded in the query trace: hits={} misses={}",
            snap.l2_hits,
            snap.l2_misses
        );
        assert_eq!(
            (snap.survivor_l1_hits, snap.survivor_l1_misses),
            (0, 1),
            "the exact-key and parent-key probes are one logical DRAM miss"
        );
    }

    /// A survivor lookup records one logical L1 outcome per requested range.
    /// The first request misses and loads; the second request hits DRAM.
    #[cfg(feature = "io-trace")]
    #[tokio::test]
    async fn dram_probe_outcomes_reach_the_ambient_io_trace() {
        let cache = SurvivorRangeCache::new(1 << 20);
        let snap =
            crate::observability::io_trace::instrument(Some("t".to_string()), "test", async move {
                cache
                    .get_or_fetch(CacheKind::QuantizedCodes, "seg.pax", 8, 4, || async {
                        Ok(vec![1, 2, 3, 4])
                    })
                    .await
                    .expect("cold load");
                cache
                    .get_or_fetch(CacheKind::QuantizedCodes, "seg.pax", 8, 4, || async {
                        panic!("warm lookup must not call loader")
                    })
                    .await
                    .expect("warm hit");
                crate::observability::io_trace::snapshot().expect("trace in scope")
            })
            .await;

        assert_eq!((snap.survivor_l1_hits, snap.survivor_l1_misses), (1, 1));
    }

    /// TD-CACHE-1 S2: warm_set ranks by hit count, caps at top_k, and
    /// replay_entry re-loads a range through the read-through path.
    #[tokio::test]
    async fn warm_set_ranks_and_replays() {
        let cache = SurvivorRangeCache::new(1024 * 1024);
        // Touch range A three times, range B once.
        for _ in 0..3 {
            cache
                .get_or_fetch(CacheKind::QuantizedCodes, "seg1.pax", 0, 8, || async {
                    Ok(vec![1u8; 8])
                })
                .await
                .unwrap();
        }
        cache
            .get_or_fetch(CacheKind::Other, "seg1.pax", 100, 4, || async {
                Ok(vec![2u8; 4])
            })
            .await
            .unwrap();

        let ws = cache.warm_set(10);
        assert_eq!(ws.len(), 1, "one (fallback) tenant tracked");
        let (_tenant, keys) = &ws[0];
        assert_eq!(keys.len(), 2);
        assert_eq!(
            (&keys[0].p[..], keys[0].o, keys[0].l, keys[0].k),
            ("seg1.pax", 0, 8, 0),
            "hottest range first"
        );
        // top_k=1 truncates to the hottest.
        let ws1 = cache.warm_set(1);
        assert_eq!(ws1[0].1.len(), 1);

        // Replay into a FRESH cache: loader runs once, then the range is hot
        // (second replay's loader must not run).
        let fresh = SurvivorRangeCache::new(1024 * 1024);
        let entry = keys[0].clone();
        assert!(
            fresh
                .replay_entry(&entry, || async { Ok(vec![9u8; 8]) })
                .await
        );
        let loaded = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let l2 = loaded.clone();
        assert!(
            fresh
                .replay_entry(&entry, move || {
                    let l = l2.clone();
                    async move {
                        l.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Ok(vec![9u8; 8])
                    }
                })
                .await
        );
        assert_eq!(
            loaded.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "replayed range must be served from cache"
        );
    }
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A hit does not re-run the loader (the GET is billed at most once per
    /// unique range), and returns identical bytes.
    #[tokio::test]
    async fn hit_skips_loader_and_returns_cached_bytes() {
        let cache = SurvivorRangeCache::new(16 * 1024 * 1024);
        let fetches = Arc::new(AtomicUsize::new(0));
        let payload = vec![42u8; 4096];

        let fetches1 = fetches.clone();
        let payload1 = payload.clone();
        let first = cache
            .get_or_fetch(CacheKind::QuantizedCodes, "seg.pax", 1024, 4096, || {
                let fetches1 = fetches1.clone();
                let payload1 = payload1.clone();
                async move {
                    fetches1.fetch_add(1, Ordering::SeqCst);
                    Ok(payload1.clone())
                }
            })
            .await
            .unwrap();

        // Same (path, off, len) — must hit the cache, NOT the loader.
        let fetches2 = fetches.clone();
        let second = cache
            .get_or_fetch(CacheKind::QuantizedCodes, "seg.pax", 1024, 4096, || {
                let fetches2 = fetches2.clone();
                async move {
                    fetches2.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![0u8; 4096]) // would be wrong if it ran
                }
            })
            .await
            .unwrap();

        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "loader ran once (miss then hit)"
        );
        assert_eq!(&*first, &payload, "miss returned the loaded bytes");
        assert_eq!(
            &*second, &payload,
            "hit returned the cached bytes, not the loader's"
        );
    }

    /// A different range key misses independently — no false sharing across keys.
    #[tokio::test]
    async fn distinct_ranges_miss_independently() {
        let cache = SurvivorRangeCache::new(16 * 1024 * 1024);
        let fetches = Arc::new(AtomicUsize::new(0));

        for (off, len) in [(0u64, 1024u64), (4096, 2048), (0, 1024)] {
            let f = fetches.clone();
            cache
                .get_or_fetch(CacheKind::QuantizedCodes, "seg.pax", off, len, || {
                    let f = f.clone();
                    async move {
                        f.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![1u8; len as usize])
                    }
                })
                .await
                .unwrap();
        }
        // (0,1024) hits on the 3rd call; the two distinct ranges miss once each.
        assert_eq!(fetches.load(Ordering::SeqCst), 2);
    }

    /// TD-RDSTRAT-12 §3: `peek_memory_exact` uses THE SAME scoped key as
    /// `get_or_fetch`, so a peek-None followed by a load makes the next peek
    /// Some — and a peek-Some guarantees the consume loader never runs (the
    /// property the batch-prefetch planner leans on).
    #[tokio::test]
    async fn peek_memory_exact_key_matches_get_or_fetch() {
        let cache = SurvivorRangeCache::new(16 * 1024 * 1024);
        assert_eq!(
            cache
                .peek_memory_exact(CacheKind::Other, "seg.pax", 0, 1024)
                .await,
            None,
            "unknown range is not resident"
        );

        let loaded: Arc<[u8]> = Arc::from(vec![7u8; 1024].as_slice());
        cache
            .get_or_fetch(CacheKind::Other, "seg.pax", 0, 1024, || async {
                Ok(loaded.to_vec())
            })
            .await
            .unwrap();

        // Resident after the load — identical coordinates, identical key.
        let resident = cache
            .peek_memory_exact(CacheKind::Other, "seg.pax", 0, 1024)
            .await;
        assert!(
            resident.is_some(),
            "peek must see the entry get_or_fetch wrote"
        );

        // A peek-verified range served via get_or_fetch never runs its loader.
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        let again = cache
            .get_or_fetch(CacheKind::Other, "seg.pax", 0, 1024, move || {
                let ran = r.clone();
                async move {
                    ran.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![0u8; 1024]) // would be wrong if it ran
                }
            })
            .await
            .unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 0, "loader skipped on residency");
        assert_eq!(&*again, resident.as_deref().unwrap());
    }

    /// TD-RDSTRAT-12 §3 round 4: `peek_parent_residency` must see EVERY tier
    /// `get_or_fetch_in_parent` can serve without its loader — exact L1,
    /// parent L1 (write-time seed), parent L2 (seed across a cache restart),
    /// and exact-key L2 (clustered-loop loader write-through after an L1
    /// eviction/restart). Dropping any probe step flips that tier to `None`
    /// and the batch planner would re-GET resident ranges.
    #[tokio::test]
    async fn peek_parent_residency_matches_get_or_fetch_in_parent_residency_tiers() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent L2"));

        // Tier: exact L1 (loader wrote the exact entry; L1 resident).
        let cache = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let payload: Arc<[u8]> = Arc::from(vec![9u8; 512].as_slice());
        cache
            .get_or_fetch_in_parent(
                CacheKind::QuantizedCodes,
                "seg.pax",
                256,
                512,
                0,
                4096,
                || {
                    let payload = payload.clone();
                    async move { Ok(payload.to_vec()) }
                },
            )
            .await
            .expect("load exact range");
        assert!(
            cache
                .peek_parent_residency(CacheKind::QuantizedCodes, "seg.pax", 256, 512, 0, 4096)
                .await
                .is_some(),
            "tier exact-L1 must be visible to the peek"
        );

        // Tier: parent L1 (write-time seed in DRAM).
        let cache2 = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let parent: Arc<[u8]> = Arc::from((0u8..64).collect::<Vec<_>>());
        cache2
            .seed_parent_region("seed.pax", 0, parent)
            .await
            .expect("seed parent");
        assert!(
            cache2
                .peek_parent_residency(CacheKind::QuantizedCodes, "seed.pax", 16, 8, 0, 64)
                .await
                .is_some(),
            "tier parent-L1 must be visible to the peek"
        );

        // Tier: parent L2 (seed persisted; a FRESH cache over the same store
        // has an empty L1, so only the persistent tier can answer).
        let cache3 = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        assert!(
            cache3
                .peek_parent_residency(CacheKind::QuantizedCodes, "seed.pax", 16, 8, 0, 64)
                .await
                .is_some(),
            "tier parent-L2 must be visible to the peek after restart"
        );

        // Tier: exact-key L2 (loader write-through; L1 empty after restart).
        let cache4 = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        assert!(
            cache4
                .peek_parent_residency(CacheKind::QuantizedCodes, "seg.pax", 256, 512, 0, 4096)
                .await
                .is_some(),
            "tier exact-L2 must be visible to the peek after restart"
        );

        // Unknown coordinates miss every tier.
        assert!(
            cache4
                .peek_parent_residency(CacheKind::QuantizedCodes, "seg.pax", 999_999, 8, 0, 4096)
                .await
                .is_none(),
            "unknown range must not be reported resident"
        );
    }

    /// TD-RDSTRAT-12 §3 round 4: the peek is SIDE-EFFECT-FREE — no hit/miss
    /// or parent_l2 counters, no io_trace l2 recordings, no L1 inserts. Any
    /// accounting added to the peek distorts hit ratios and the warm-set
    /// manifest, which the batch planner (and evidence) rely on.
    #[tokio::test]
    async fn peek_parent_residency_is_side_effect_free() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store =
            Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent L2"));
        let cache = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let parent: Arc<[u8]> = Arc::from((0u8..128).collect::<Vec<_>>());
        cache
            .seed_parent_region("seg.pax", 0, parent)
            .await
            .expect("seed parent");

        let parent_l2_hits_before = cache.parent_l2_hits.load(Ordering::SeqCst);
        let parent_l2_misses_before = cache.parent_l2_misses.load(Ordering::SeqCst);
        let l2_stats_before = cache.l2_stats();
        let inner_l2_before = cache.inner.l2_stats();

        for off in [0u64, 16, 64, 120, 4_096] {
            let _ = cache
                .peek_parent_residency(CacheKind::QuantizedCodes, "seg.pax", off, 8, 0, 128)
                .await;
        }

        assert_eq!(
            cache.parent_l2_hits.load(Ordering::SeqCst),
            parent_l2_hits_before,
            "peek must not count parent L2 hits"
        );
        assert_eq!(
            cache.parent_l2_misses.load(Ordering::SeqCst),
            parent_l2_misses_before,
            "peek must not count parent L2 misses"
        );
        assert_eq!(
            cache.l2_stats().hits,
            l2_stats_before.hits,
            "peek must not move L1 hit counters"
        );
        assert_eq!(
            cache.l2_stats().misses,
            l2_stats_before.misses,
            "peek must not move L1 miss counters"
        );
        assert_eq!(
            cache.inner.l2_stats().hits,
            inner_l2_before.hits,
            "peek must not move the exact-L2 traced counters"
        );
        assert_eq!(
            cache.inner.l2_stats().misses,
            inner_l2_before.misses,
            "peek must not move the exact-L2 traced counters"
        );
    }
}
