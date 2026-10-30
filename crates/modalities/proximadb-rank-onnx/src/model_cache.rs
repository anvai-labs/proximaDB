//! `OnnxModelCache` — refcounted, LRU-evictable session pool.
//!
//! `acquire(key) → ScorerToken` is the public read path. Tokens hold a
//! cloned `Arc<dyn ScorerSession>`; the cache holds the same `Arc`. LRU
//! eviction checks `Arc::strong_count` and *never* evicts a session
//! with outstanding tokens (matches Vespa's `OnnxModelCache::Token`).
//!
//! v1 has no built-in lazy load — `install(session)` is the caller's
//! responsibility. R-5b will add `acquire_with_loader(...)` that fetches
//! from object storage on miss.

use crate::descriptor::ModelKey;
use crate::scorer_session::ScorerSession;
use dashmap::DashMap;
use proximadb_rank_core::{RankError, RankResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Observer hook for cache events. The root-crate Prometheus glue
/// implements this to emit spec §4.10 model-cache metrics
/// (`rank_model_cache_hit_ratio`, `_size_bytes`, `_evictions_total`,
/// `_inflight_loads`). Defining the surface here keeps
/// `proximadb-rank-onnx` from depending on the root crate's
/// observability — implementations live wherever the metric
/// handles live.
///
/// All methods take `&self` and are called on the cache's hot
/// path, so impls should be lock-free / atomic. The default impl
/// is a no-op — `OnnxModelCache` without an observer pays zero
/// runtime cost.
pub trait ModelCacheObserver: Send + Sync {
    /// Called on every `acquire()`, regardless of hit / miss.
    /// `hit = true` when the entry was already resident.
    fn record_acquire(&self, _model_id: &str, _hit: bool) {}

    /// Called after `install()` completes; `total_bytes` is the
    /// post-install total resident bytes across all sessions.
    fn record_install(&self, _model_id: &str, _total_bytes: u64) {}

    /// Called once per evicted entry inside `evict_if_over_budget()`.
    /// `reason` is one of the spec §4.10 values (`budget`,
    /// `count`, `manual`, `ttl`).
    fn record_eviction(&self, _model_id: &str, _reason: &str, _freed_bytes: u64) {}

    /// Called immediately after `evict_if_over_budget()` finishes
    /// with the updated total resident bytes. Lets dashboards
    /// surface a fresh `cache_size_bytes` even when no eviction
    /// fired (e.g. after `install()` set the gauge but a parallel
    /// install grew it again).
    fn record_size(&self, _total_bytes: u64) {}

    /// Called when a cold load begins. Pair with `record_load_complete`
    /// when the loader returns (success or failure). The root-crate
    /// adapter maps this onto `rank_model_inflight_loads.inc()` so
    /// dashboards see concurrent cold loads in flight.
    fn record_load_start(&self, _model_id: &str) {}

    /// Called when a cold load ends. `ok = true` on successful install,
    /// `false` if the loader returned an error.
    fn record_load_complete(&self, _model_id: &str, _ok: bool) {}
}

/// Per-model hit/miss counters, updated atomically by the cache
/// on every `acquire()`. The observer can derive a rolling hit
/// ratio from these (or expose them directly as counters).
#[derive(Debug, Default)]
pub struct AcquireStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

/// LRU eviction strategy.
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// Evict oldest-unused sessions until total memory ≤ `budget_bytes`.
    LruByMemory { budget_bytes: usize },
    /// Evict oldest-unused sessions until cache size ≤ `max_entries`.
    LruByCount { max_entries: usize },
    /// Per-tenant budget. Tracked separately so one tenant can't squeeze
    /// out another. R-5b will implement; v1 falls back to LruByMemory.
    Tenanted { per_tenant_budget_bytes: usize },
}

/// Refcounted handle to a loaded session. While at least one token
/// exists, the cache will not evict the underlying session.
#[derive(Clone)]
pub struct ScorerToken {
    session: Arc<dyn ScorerSession>,
}

impl ScorerToken {
    pub fn session(&self) -> &dyn ScorerSession {
        self.session.as_ref()
    }
    /// Number of outstanding tokens (including this one) plus the cache
    /// entry's reference. Exposed for tests.
    pub fn refcount(&self) -> usize {
        Arc::strong_count(&self.session)
    }
}

impl std::ops::Deref for ScorerToken {
    type Target = dyn ScorerSession;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref()
    }
}

struct CacheEntry {
    session: Arc<dyn ScorerSession>,
    last_used_at_ms: AtomicI64,
}

impl CacheEntry {
    fn touch(&self) {
        self.last_used_at_ms
            .store(proximadb_kernel::time::now_unix_ms(), Ordering::Relaxed);
    }
    fn last_used(&self) -> i64 {
        self.last_used_at_ms.load(Ordering::Relaxed)
    }
}

pub struct OnnxModelCache {
    entries: DashMap<ModelKey, Arc<CacheEntry>>,
    policy: EvictionPolicy,
    observer: Option<Arc<dyn ModelCacheObserver>>,
}

impl OnnxModelCache {
    pub fn new(policy: EvictionPolicy) -> Self {
        Self {
            entries: DashMap::new(),
            policy,
            observer: None,
        }
    }

    /// Attach an observer that receives cache events (acquire,
    /// install, evict). Used by the root crate to wire spec §4.10
    /// model-cache metrics. `None` (the default) keeps the cache
    /// allocation- and atomic-lookup-free in the noop case.
    pub fn with_observer(mut self, observer: Arc<dyn ModelCacheObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Insert a loaded session and return a token holding the first
    /// outstanding reference. Replaces any prior entry for the same key
    /// (last-write-wins, matches the catalog's hot-reload semantics).
    /// Triggers an eviction pass after install.
    pub fn install(&self, session: Arc<dyn ScorerSession>) -> ScorerToken {
        let key = session.descriptor().key.clone();
        let model_id = key.model_id.clone();
        let entry = Arc::new(CacheEntry {
            session: session.clone(),
            last_used_at_ms: AtomicI64::new(proximadb_kernel::time::now_unix_ms()),
        });
        self.entries.insert(key, entry);
        let token = ScorerToken { session };
        let _ = self.evict_if_over_budget();
        if let Some(obs) = &self.observer {
            let total = self.total_memory_bytes() as u64;
            obs.record_install(&model_id, total);
            obs.record_size(total);
        }
        token
    }

    /// Acquire-or-load: returns a token if the model is cached;
    /// otherwise invokes `loader` to fetch + install + return a
    /// token. Wraps the loader call with
    /// `record_load_start`/`record_load_complete` so the
    /// `rank_model_inflight_loads` gauge sees concurrent cold loads
    /// in flight (spec §4.10). The loader is invoked synchronously
    /// here — R-5b can swap in an async variant once an async
    /// `ScorerSession` loader trait lands.
    pub fn acquire_or_load_with<F>(&self, key: &ModelKey, loader: F) -> RankResult<ScorerToken>
    where
        F: FnOnce() -> RankResult<Arc<dyn ScorerSession>>,
    {
        if let Ok(token) = self.acquire(key) {
            return Ok(token);
        }
        if let Some(obs) = &self.observer {
            obs.record_load_start(&key.model_id);
        }
        let result = loader();
        let outcome_ok = result.is_ok();
        let token_result = result.map(|session| self.install(session));
        if let Some(obs) = &self.observer {
            obs.record_load_complete(&key.model_id, outcome_ok);
        }
        token_result
    }

    /// Look up a loaded session by key. Updates last-used timestamp on
    /// hit. Returns `Err(ProfileNotFound)` on miss — R-5b will overload
    /// this with the loader path.
    pub fn acquire(&self, key: &ModelKey) -> RankResult<ScorerToken> {
        let result = match self.entries.get(key) {
            Some(r) => {
                r.touch();
                Ok(ScorerToken {
                    session: r.session.clone(),
                })
            }
            None => Err(RankError::ProfileNotFound(format!(
                "model not loaded into cache: {key}"
            ))),
        };
        if let Some(obs) = &self.observer {
            obs.record_acquire(&key.model_id, result.is_ok());
        }
        result
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sum of `memory_bytes()` across resident sessions.
    pub fn total_memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|r| r.value().session.memory_bytes())
            .sum()
    }

    /// Force an LRU pass. Returns bytes freed. Evicts in order of
    /// oldest `last_used_at`. Sessions with outstanding tokens
    /// (`Arc::strong_count > 1`) are NEVER evicted regardless of age
    /// or budget.
    pub fn evict_if_over_budget(&self) -> usize {
        let (budget_bytes, by_count_max, primary_reason) = match &self.policy {
            EvictionPolicy::LruByMemory { budget_bytes } => (Some(*budget_bytes), None, "budget"),
            EvictionPolicy::LruByCount { max_entries } => (None, Some(*max_entries), "count"),
            EvictionPolicy::Tenanted {
                per_tenant_budget_bytes,
            } => {
                // R-5b: per-tenant accounting. Fall back to a global
                // budget that's tenant_budget * 10 so things at least
                // don't grow unbounded.
                (Some(per_tenant_budget_bytes * 10), None, "budget")
            }
        };

        if budget_bytes.is_none() && by_count_max.is_none() {
            return 0;
        }

        let mut candidates: Vec<(ModelKey, i64, usize)> = self
            .entries
            .iter()
            .filter_map(|r| {
                // strong_count is at least 1 from CacheEntry.session +
                // however many tokens. We're evictable iff no token
                // holds a clone, which means strong_count == 1.
                if Arc::strong_count(&r.value().session) == 1 {
                    Some((
                        r.key().clone(),
                        r.value().last_used(),
                        r.value().session.memory_bytes(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by_key(|(_, ts, _)| *ts); // oldest first

        let mut freed = 0;
        let mut current_bytes = self.total_memory_bytes();
        let mut current_count = self.entries.len();

        for (key, _, _) in candidates {
            let should_evict = match (budget_bytes, by_count_max) {
                (Some(b), _) if current_bytes > b => true,
                (_, Some(m)) if current_count > m => true,
                _ => false,
            };
            if !should_evict {
                break;
            }
            // Double-check refcount under the remove — another thread
            // may have acquired between our scan and now.
            if let Some((_, entry)) = self
                .entries
                .remove_if(&key, |_, v| Arc::strong_count(&v.session) == 1)
            {
                let b = entry.session.memory_bytes();
                freed += b;
                current_bytes = current_bytes.saturating_sub(b);
                current_count = current_count.saturating_sub(1);
                if let Some(obs) = &self.observer {
                    obs.record_eviction(&key.model_id, primary_reason, b as u64);
                }
            }
        }
        if let Some(obs) = &self.observer {
            obs.record_size(current_bytes as u64);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{DType, ModelDescriptor, ModelFramework};
    use crate::scorer_session::MockScorerSession;

    fn descriptor(id: &str, size_bytes: u64) -> ModelDescriptor {
        ModelDescriptor {
            key: ModelKey::new(id, "1"),
            tenant: None,
            uri: format!("file:///tmp/{id}.onnx"),
            sha256: [0; 32],
            size_bytes,
            framework: ModelFramework::Onnx,
            dtype: DType::Fp32,
            input_spec: vec![],
            output_spec: vec![],
            max_batch_size: 8,
            seq: 0,
            created_at_ms: 0,
        }
    }

    fn session(id: &str, size: u64) -> Arc<dyn ScorerSession> {
        Arc::new(MockScorerSession::zeros(descriptor(id, size)))
    }

    #[test]
    fn install_then_acquire_returns_token() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        });
        let t = cache.install(session("a", 100));
        drop(t); // release the install-time token
        let acquired = cache.acquire(&ModelKey::new("a", "1")).unwrap();
        assert_eq!(acquired.descriptor().key.model_id, "a");
    }

    #[test]
    fn acquire_unknown_key_errors() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        });
        match cache.acquire(&ModelKey::new("nope", "1")) {
            Err(RankError::ProfileNotFound(_)) => {}
            Err(_) => panic!("expected ProfileNotFound, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn scorer_token_keeps_session_alive() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory { budget_bytes: 50 });
        let t = cache.install(session("a", 200)); // way over budget
        // While token is alive, eviction must NOT remove the entry
        // (refcount > 1 because cache + token both hold the Arc).
        let freed = cache.evict_if_over_budget();
        assert_eq!(freed, 0, "in-use session must not be evicted");
        assert_eq!(cache.len(), 1);
        drop(t);
        let freed = cache.evict_if_over_budget();
        assert_eq!(freed, 200, "after drop, eviction must reclaim");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn lru_evicts_oldest_first() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory { budget_bytes: 250 });
        let t1 = cache.install(session("a", 100));
        drop(t1);
        // Sleep a tick so timestamps differ.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t2 = cache.install(session("b", 100));
        drop(t2);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t3 = cache.install(session("c", 100));
        drop(t3);
        // total = 300, budget = 250 → evict the oldest (a)
        cache.evict_if_over_budget();
        assert!(cache.acquire(&ModelKey::new("a", "1")).is_err());
        assert!(cache.acquire(&ModelKey::new("b", "1")).is_ok());
        assert!(cache.acquire(&ModelKey::new("c", "1")).is_ok());
    }

    #[test]
    fn lru_eviction_respects_inuse_refcount_under_pressure() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory { budget_bytes: 50 });
        // Install two sessions, keep a token on the OLDER one.
        let pinned = cache.install(session("a", 100));
        std::thread::sleep(std::time::Duration::from_millis(2));
        let throwaway = cache.install(session("b", 100));
        drop(throwaway);
        // The oldest is "a" but it's pinned by `pinned`. Eviction must
        // evict "b" instead even though it's newer.
        cache.evict_if_over_budget();
        assert!(
            cache.acquire(&ModelKey::new("a", "1")).is_ok(),
            "pinned session must survive eviction"
        );
        assert!(
            cache.acquire(&ModelKey::new("b", "1")).is_err(),
            "unpinned session must be evicted"
        );
        drop(pinned);
    }

    #[test]
    fn lru_by_count_caps_entries() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByCount { max_entries: 2 });
        let _ = cache.install(session("a", 1));
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = cache.install(session("b", 1));
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = cache.install(session("c", 1));
        // total entries should be at most 2 after eviction.
        cache.evict_if_over_budget();
        assert!(cache.len() <= 2);
    }

    #[test]
    fn install_replaces_existing_entry() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        });
        let _ = cache.install(session("a", 100));
        let _ = cache.install(session("a", 200));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_memory_bytes(), 200);
    }

    #[test]
    fn refcount_reflects_outstanding_tokens() {
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        });
        let t1 = cache.install(session("a", 100));
        // cache holds 1 (CacheEntry.session), t1 holds 1 → 2
        assert_eq!(t1.refcount(), 2);
        let t2 = cache.acquire(&ModelKey::new("a", "1")).unwrap();
        assert_eq!(t1.refcount(), 3);
        drop(t2);
        assert_eq!(t1.refcount(), 2);
    }

    #[test]
    fn evict_with_no_budget_set_is_noop() {
        // Tenanted policy with a budget that, when multiplied by 10,
        // still exceeds the load → no eviction.
        let cache = OnnxModelCache::new(EvictionPolicy::Tenanted {
            per_tenant_budget_bytes: 1_000_000,
        });
        let _ = cache.install(session("a", 100));
        let freed = cache.evict_if_over_budget();
        assert_eq!(freed, 0);
    }

    // ---------------- ModelCacheObserver wiring ----------------

    #[derive(Default)]
    struct RecordingObserver {
        installs: AtomicU64,
        hits: AtomicU64,
        misses: AtomicU64,
        evictions: AtomicU64,
        last_size: AtomicI64,
        load_starts: AtomicU64,
        load_completes: AtomicU64,
        load_ok: AtomicU64,
    }

    impl ModelCacheObserver for RecordingObserver {
        fn record_acquire(&self, _model_id: &str, hit: bool) {
            if hit {
                self.hits.fetch_add(1, Ordering::SeqCst);
            } else {
                self.misses.fetch_add(1, Ordering::SeqCst);
            }
        }
        fn record_install(&self, _model_id: &str, _total_bytes: u64) {
            self.installs.fetch_add(1, Ordering::SeqCst);
        }
        fn record_eviction(&self, _model_id: &str, _reason: &str, _freed_bytes: u64) {
            self.evictions.fetch_add(1, Ordering::SeqCst);
        }
        fn record_size(&self, total_bytes: u64) {
            self.last_size.store(total_bytes as i64, Ordering::SeqCst);
        }
        fn record_load_start(&self, _model_id: &str) {
            self.load_starts.fetch_add(1, Ordering::SeqCst);
        }
        fn record_load_complete(&self, _model_id: &str, ok: bool) {
            self.load_completes.fetch_add(1, Ordering::SeqCst);
            if ok {
                self.load_ok.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[test]
    fn acquire_or_load_with_invokes_loader_on_miss_and_emits_load_events() {
        // First call misses the cache → loader runs → record_load_start +
        // record_load_complete(true) fire. Second call hits the cache →
        // no load events, no loader invocation.
        let obs = Arc::new(RecordingObserver::default());
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        })
        .with_observer(obs.clone());

        let invocations = Arc::new(AtomicU64::new(0));
        let key = ModelKey::new("rerank-v3", "1");

        // Cold path: cache miss → loader runs.
        let inv1 = invocations.clone();
        let t1 = cache
            .acquire_or_load_with(&key, || {
                inv1.fetch_add(1, Ordering::SeqCst);
                Ok(session("rerank-v3", 100))
            })
            .unwrap();
        drop(t1);
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert_eq!(obs.load_starts.load(Ordering::SeqCst), 1);
        assert_eq!(obs.load_completes.load(Ordering::SeqCst), 1);
        assert_eq!(obs.load_ok.load(Ordering::SeqCst), 1);

        // Warm path: cache hit → loader does not run.
        let inv2 = invocations.clone();
        let _ = cache
            .acquire_or_load_with(&key, || {
                inv2.fetch_add(1, Ordering::SeqCst);
                Ok(session("rerank-v3", 100))
            })
            .unwrap();
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            1,
            "loader must not run on cache hit"
        );
        assert_eq!(
            obs.load_starts.load(Ordering::SeqCst),
            1,
            "no new load start"
        );
    }

    #[test]
    fn acquire_or_load_with_records_load_failure() {
        // Loader returns an error → record_load_complete fires with ok=false
        // and the token result is the propagated error.
        let obs = Arc::new(RecordingObserver::default());
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory {
            budget_bytes: usize::MAX,
        })
        .with_observer(obs.clone());

        let key = ModelKey::new("ghost", "1");
        let result = cache.acquire_or_load_with(&key, || {
            Err(RankError::ProfileNotFound("simulated load failure".into()))
        });
        assert!(result.is_err());
        assert_eq!(obs.load_starts.load(Ordering::SeqCst), 1);
        assert_eq!(obs.load_completes.load(Ordering::SeqCst), 1);
        assert_eq!(obs.load_ok.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn observer_fires_on_acquire_install_evict() {
        // The observer must see install + acquire hit/miss + eviction
        // events end-to-end so root-crate Prometheus glue can emit
        // spec §4.10 model-cache metrics without instrumenting each
        // call site itself.
        let obs = Arc::new(RecordingObserver::default());
        let cache = OnnxModelCache::new(EvictionPolicy::LruByMemory { budget_bytes: 150 })
            .with_observer(obs.clone());

        // Install fires record_install + record_size.
        let t = cache.install(session("a", 100));
        drop(t);
        assert_eq!(obs.installs.load(Ordering::SeqCst), 1);
        assert_eq!(obs.last_size.load(Ordering::SeqCst), 100);

        // Acquire hit fires record_acquire(true).
        let _ = cache.acquire(&ModelKey::new("a", "1")).unwrap();
        assert_eq!(obs.hits.load(Ordering::SeqCst), 1);
        assert_eq!(obs.misses.load(Ordering::SeqCst), 0);

        // Acquire miss fires record_acquire(false).
        let _ = cache.acquire(&ModelKey::new("ghost", "1"));
        assert_eq!(obs.misses.load(Ordering::SeqCst), 1);

        // Trip the budget so eviction fires record_eviction.
        let throwaway = cache.install(session("b", 100));
        drop(throwaway);
        cache.evict_if_over_budget();
        assert!(obs.evictions.load(Ordering::SeqCst) >= 1);
    }
}
