// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! `TenantCache` — a multitenant, byte-budgeted, cache-aware primitive.
//!
//! One **global** [`moka`] cache (so capacity is pooled — economies of scale, no
//! per-tenant stranding) whose admission is moka's TinyLFU (scan-resistant and
//! frequency-aware, so a flooding tenant's cold entries are rejected *at
//! admission* and hot small-tenant entries are protected). On top of that:
//!
//! * every entry is **tenant-namespaced** by [`CacheKey`] — the isolation key;
//! * capacity is **byte-weighted** (not entry-count) via [`CachedValue::weight`];
//! * each tenant has a **soft byte ceiling** — once exceeded, that tenant's new
//!   entries bypass the cache (the value is still returned) so one tenant cannot
//!   monopolize the pool;
//! * **per-tenant stats** (hits/misses/bytes/evictions) are exposed for billing
//!   labels — this crate stays metrics-free and just returns the numbers.
//!
//! Eviction is reconciled into per-tenant byte gauges by a moka eviction
//! listener, so accounting stays correct no matter which tenant's entry moka
//! chooses to evict under global pressure.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use dashmap::DashMap;
use moka::future::Cache;
use moka::notification::RemovalCause;

mod persistent_l2;

pub use persistent_l2::{L2Class, PersistentByteStore};

/// TD-TIER-1 SC4-1: the process-wide residency-budget bridge.
///
/// The reservation registry (service layer) and the production persistent
/// L2 store (SST engine singleton) are constructed in opposite order and
/// never see each other directly; this module is their only meeting point.
/// A single cell holds the authoritative budget (the live sum of charged
/// grants across pools); setting it also arms whichever store is already
/// registered, and a store registering later picks up the current cell —
/// order-independent by construction; registration and setting share one
/// mutex so no interleaving loses an update. ONE registry per process is
/// assumed — the budget is a single number, so two registries with disjoint
/// pools would each push their own Σ and clobber the other (production
/// constructs exactly one; tests and embedded must respect the same rule).
mod residency_budget {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    use super::PersistentByteStore;

    static REGISTERED_STORE: OnceLock<Arc<PersistentByteStore>> = OnceLock::new();
    static CELL: OnceLock<Mutex<u64>> = OnceLock::new();

    fn cell() -> &'static Mutex<u64> {
        CELL.get_or_init(|| Mutex::new(0))
    }

    /// Register the process's shared persistent store (first registration
    /// wins — the SST engine's singleton). The WINNER is armed with the
    /// current budget under the cell mutex; a loser is left untouched (it
    /// would otherwise be a live store armed once and never managed again).
    pub fn register_process_store(store: Arc<PersistentByteStore>) -> bool {
        let budget = cell().lock().unwrap_or_else(|p| p.into_inner());
        let applied = REGISTERED_STORE.set(store).is_ok();
        if applied && let Some(store) = REGISTERED_STORE.get() {
            store.set_reserved_budget(*budget);
        }
        applied
    }

    /// The registered shared store, if any.
    pub fn process_store() -> Option<Arc<PersistentByteStore>> {
        REGISTERED_STORE.get().cloned()
    }

    /// Set the authoritative residency budget (live charged grant bytes
    /// across all pools) and arm the registered store, if any. Registering
    /// and setting share the cell mutex: their interleavings cannot lose an
    /// update (review SF3 — a set-store/register-store race could otherwise
    /// leave the cell and the store permanently disagreeing).
    pub fn set_process_reserved_budget(bytes: u64) {
        let mut budget = cell().lock().unwrap_or_else(|p| p.into_inner());
        *budget = bytes;
        if let Some(store) = REGISTERED_STORE.get() {
            store.set_reserved_budget(bytes);
        }
    }

    /// The current budget cell (diagnostics/tests).
    pub fn process_reserved_budget() -> u64 {
        *cell().lock().unwrap_or_else(|p| p.into_inner())
    }
}

pub use residency_budget::{
    process_reserved_budget, process_store, register_process_store, set_process_reserved_budget,
};

/// The category of a cached artifact. Drives key-namespacing and per-kind stats;
/// extend as new consumers adopt the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum CacheKind {
    /// Parsed PAX block footer / column metadata (`BlockLayout`).
    Footer,
    /// Parsed PAX segment index (block offsets).
    SegmentIndex,
    /// Quantized vector codes (SQ8 / RaBitQ) hot tier.
    QuantizedCodes,
    /// Cached SQL query result.
    QueryResult,
    /// Catalog schema / namespace metadata.
    CatalogSchema,
    /// Anything not yet categorized.
    Other,
}

/// The owner/fair-share scope of a cache entry.
///
/// Stable catalog tenant ids remain integers in memory; named scopes exist for
/// consumers whose authoritative identity is genuinely textual (for example a
/// graph name). `Shared` is for background work with no resolved tenant. Text
/// rendering is deliberately confined to metrics and persistent-key boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheScope {
    StableTenant(u64),
    Named(Arc<str>),
    Shared,
}

impl CacheScope {
    pub fn stable_tenant(tenant_stable_id: u64) -> Self {
        Self::StableTenant(tenant_stable_id)
    }

    pub fn named(scope: impl AsRef<str>) -> Self {
        Self::Named(Arc::from(scope.as_ref()))
    }

    pub fn label(&self) -> String {
        match self {
            Self::StableTenant(id) => id.to_string(),
            Self::Named(name) => name.to_string(),
            Self::Shared => "shared".to_string(),
        }
    }

    fn persistent_component(&self) -> String {
        match self {
            Self::StableTenant(id) => format!("t:{id}"),
            Self::Named(name) => format!("n:{}:{name}", name.len()),
            Self::Shared => "s".to_string(),
        }
    }
}

/// A scope-namespaced cache key. `scope` is the fair-share/accounting
/// boundary; `kind` separates artifact classes; `key` is the per-artifact
/// identifier (e.g. a segment path + block offset). Data isolation must also
/// be structural in the artifact key (for PAX this is the full `DrPath`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub scope: CacheScope,
    pub kind: CacheKind,
    pub key: Arc<str>,
}

impl CacheKey {
    pub fn with_scope(scope: CacheScope, kind: CacheKind, key: impl AsRef<str>) -> Self {
        Self {
            scope,
            kind,
            key: Arc::from(key.as_ref()),
        }
    }

    /// Construct a key for an authoritative textual scope.
    pub fn new(scope: impl AsRef<str>, kind: CacheKind, key: impl AsRef<str>) -> Self {
        Self::with_scope(CacheScope::named(scope), kind, key)
    }

    /// Construct a key for a catalog-authoritative numeric tenant id.
    pub fn for_tenant_id(tenant_stable_id: u64, kind: CacheKind, key: impl AsRef<str>) -> Self {
        Self::with_scope(CacheScope::stable_tenant(tenant_stable_id), kind, key)
    }

    /// Construct a key for background work without a resolved tenant.
    pub fn shared(kind: CacheKind, key: impl AsRef<str>) -> Self {
        Self::with_scope(CacheScope::Shared, kind, key)
    }

    /// TD-RDSTRAT-12 §3: the persistent-tier key format, exposed so engine-side
    /// residency peeks can probe the raw L2 store for entries the write path
    /// created under this exact key — without duplicating (and drifting from)
    /// the namespace/scope/kind encoding here.
    pub fn persistent_key(&self, namespace: &str) -> String {
        format!(
            "{namespace}/{}/{}:{}",
            self.scope.persistent_component(),
            self.kind.stable_id(),
            self.key
        )
    }
}

impl CacheKind {
    /// Stable discriminator used by persistent cache keys.
    pub fn stable_id(self) -> u8 {
        match self {
            Self::Footer => 0,
            Self::SegmentIndex => 1,
            Self::QuantizedCodes => 2,
            Self::QueryResult => 3,
            Self::CatalogSchema => 4,
            Self::Other => 5,
        }
    }
}

/// Async value seam behind [`TenantCache`]. L2 failures are fail-open cache
/// misses; callers retain the authoritative loader for correctness.
#[async_trait::async_trait]
pub trait L2ValueStore<V>: Send + Sync + std::fmt::Debug {
    async fn get(&self, key: &CacheKey) -> io::Result<Option<(V, u32)>>;
    async fn put(&self, key: &CacheKey, weight: u32, value: V) -> io::Result<()>;
    async fn remove(&self, key: &CacheKey) -> io::Result<bool>;
    fn resident_bytes(&self) -> u64;
}

/// Persistent L2 adapter for the raw byte values used by PAX range caches.
#[derive(Debug)]
pub struct PersistentArcBytesL2 {
    store: Arc<PersistentByteStore>,
    namespace: Arc<str>,
    class: L2Class,
}

impl PersistentArcBytesL2 {
    pub fn new(
        store: Arc<PersistentByteStore>,
        namespace: impl AsRef<str>,
        class: L2Class,
    ) -> Self {
        Self {
            store,
            namespace: Arc::from(namespace.as_ref()),
            class,
        }
    }

    pub fn backing_store(&self) -> Arc<PersistentByteStore> {
        self.store.clone()
    }
}

#[async_trait::async_trait]
impl L2ValueStore<Arc<[u8]>> for PersistentArcBytesL2 {
    async fn get(&self, key: &CacheKey) -> io::Result<Option<(Arc<[u8]>, u32)>> {
        Ok(self
            .store
            .get(&key.persistent_key(&self.namespace))
            .await?
            .map(|value| {
                let weight = value.len().try_into().unwrap_or(u32::MAX);
                (value, weight)
            }))
    }

    async fn put(&self, key: &CacheKey, _weight: u32, value: Arc<[u8]>) -> io::Result<()> {
        self.store
            .put(key.persistent_key(&self.namespace), self.class, value)
            .await
    }

    async fn remove(&self, key: &CacheKey) -> io::Result<bool> {
        Ok(self.store.remove(&key.persistent_key(&self.namespace)))
    }

    fn resident_bytes(&self) -> u64 {
        self.store.resident_bytes_for(self.class)
    }
}

/// Point-in-time persistent-tier counters for one [`TenantCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub resident_bytes: u64,
}

/// Tier that satisfied a cache read. `Loaded` means no cache tier served the
/// request and the caller-provided authoritative loader ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReadOutcome {
    MemoryHit,
    PersistentHit,
    Loaded,
}

/// A cached value carrying its byte weight (so moka's weigher budgets by bytes,
/// not entry count).
#[derive(Debug, Clone)]
struct CachedValue<V> {
    weight: u32,
    value: V,
}

/// Per-tenant elasticity limits: a guaranteed `floor`, an absolute `hard_ceiling`
/// (runaway guard), and a `weight` for the contended fair share (higher = larger
/// share). Between floor and ceiling, capacity is borrowed from the idle pool.
#[derive(Debug, Clone, Copy)]
pub struct TenantLimits {
    pub floor_bytes: u64,
    pub hard_ceiling_bytes: u64,
    pub weight: u32,
}

/// One tier's cache shares, expressed as **fractions of the global pool** so the
/// same policy scales to any `total_bytes`. `weight` drives the contended fair
/// share (higher tier = larger share); `floor_frac` is the protected working
/// set; `ceiling_frac` is the absolute runaway cap.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TierSpec {
    pub weight: u32,
    pub floor_frac: f64,
    pub ceiling_frac: f64,
}

/// JSON-loadable per-tier cache policy — a **generic, operator-supplied** config
/// schema (string-keyed tier ids; an unknown tenant tier falls back to
/// `default_tier`). OSS ships no commercial tiers: the actual tier→share data is
/// deployment config provided by the control plane (e.g. anvaiops ships a
/// `cache_tiers.json` keyed by its pricing tiers). This struct is just the Rust
/// parser + the resolver factory (the adapter that turns config into a
/// [`LimitsResolver`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TierPolicy {
    pub default_tier: String,
    pub tiers: HashMap<String, TierSpec>,
}

impl TierPolicy {
    /// Parse a tier policy from operator-supplied JSON (`cache_tiers.json`).
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Neutral OSS default: a single `"default"` tier that may use the whole pool
    /// (uniform fair share, no tier preference). Production tier shares come from
    /// operator config — OSS bakes in no commercial tiers.
    pub fn single_default() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(
            "default".into(),
            TierSpec {
                weight: 1,
                floor_frac: 0.0,
                ceiling_frac: 1.0,
            },
        );
        Self {
            default_tier: "default".into(),
            tiers,
        }
    }

    /// Whether this policy declares an explicit spec for `tier`.
    ///
    /// Lets a caller probe several spellings of the same entitlement in its own
    /// preferred order (e.g. a canonical id, then the operator's original
    /// alias) while this crate stays free of any tier vocabulary — the key
    /// space here is deliberately generic and operator-supplied.
    pub fn has_tier(&self, tier: &str) -> bool {
        self.tiers.contains_key(tier)
    }

    /// Absolute [`TenantLimits`] for `tier` at a given pool `total_bytes`
    /// (falls back to `default_tier`, then to a permissive default).
    pub fn limits(&self, tier: &str, total_bytes: u64) -> TenantLimits {
        let spec = self
            .tiers
            .get(tier)
            .or_else(|| self.tiers.get(&self.default_tier));
        match spec {
            Some(s) => TenantLimits {
                floor_bytes: (total_bytes as f64 * s.floor_frac) as u64,
                hard_ceiling_bytes: ((total_bytes as f64 * s.ceiling_frac) as u64).max(1),
                weight: s.weight.max(1),
            },
            None => TenantLimits {
                floor_bytes: 0,
                hard_ceiling_bytes: total_bytes,
                weight: 1,
            },
        }
    }

    /// Build a [`LimitsResolver`] from this policy: `tenant_id → tier`
    /// (host-supplied, e.g. catalog/billing lookup) → [`TenantLimits`].
    pub fn resolver(
        self: Arc<Self>,
        total_bytes: u64,
        tenant_to_tier: Arc<dyn Fn(&str) -> String + Send + Sync>,
    ) -> Arc<LimitsResolver> {
        Arc::new(move |tenant: &str| {
            let tier = tenant_to_tier(tenant);
            self.limits(&tier, total_bytes)
        })
    }
}

/// Pluggable per-tenant limits policy — the Strategy seam for tier-driven
/// preference. The host supplies a `tenant_id → TenantLimits` function (e.g.
/// [`TierPolicy::resolver`]); when unset, the cache falls back to the
/// `CacheBudget`'s static per-tenant map / defaults.
pub type LimitsResolver = dyn Fn(&str) -> TenantLimits + Send + Sync;

/// Cache sizing policy: a global byte pool with **work-conserving** per-tenant
/// elasticity — tenants borrow idle pool capacity up to their hard ceiling, and
/// are reclaimed toward a weighted fair share only when the pool is under
/// pressure. Nothing is pre-partitioned, so idle capacity is never stranded
/// (no fragmentation), while the hard ceiling bounds any single tenant.
#[derive(Debug, Clone)]
pub struct CacheBudget {
    /// Total bytes across all tenants (the shared pool).
    pub total_bytes: u64,
    /// Default guaranteed working set per tenant (protected from reclaim).
    pub default_floor_bytes: u64,
    /// Default absolute per-tenant cap (runaway guard).
    pub default_hard_ceiling_bytes: u64,
    /// Pool-pressure threshold as a fraction of `total_bytes`; below it tenants
    /// borrow freely (elastic), above it the fair share is enforced. Default 0.9.
    pub high_watermark_frac: f64,
    /// Per-tenant overrides (paid tiers, large tenants).
    pub per_tenant: HashMap<String, TenantLimits>,
    /// Optional time-to-live for entries (None = no expiry; rely on size).
    pub ttl: Option<Duration>,
    /// TD-CACHE-2 S2c: optional per-[`CacheKind`] admission ceilings as
    /// fractions of `total_bytes` — e.g. `Other ≤ 0.3` stops OID-range churn
    /// from flushing SQ8 survivor ranges without partitioning the pool into
    /// static budgets. A kind at/over its ceiling bypasses caching (the value
    /// is still returned); kinds without an entry stay unbounded (elastic).
    pub kind_ceilings: HashMap<CacheKind, f64>,
    /// TD-CACHE-3 S2: fraction of `total_bytes` carved out as the **pin
    /// reserve** — a side pool holding tenants' floor bytes OUTSIDE the shared
    /// moka pool, so TinyLFU (which is floor-agnostic) can never evict another
    /// tenant's guaranteed working set. 0.0 (default) disables true-pinning:
    /// floors stay admission-only. Clamped to [0.0, 0.5] so the shared pool
    /// keeps majority capacity.
    pub pin_reserve_frac: f64,
}

impl CacheBudget {
    /// A budget with a `total_bytes` pool and a default per-tenant
    /// `hard_ceiling_bytes` (absolute cap). Floor defaults to 0, high-watermark
    /// to 0.9, and the contended fair share is equal (`total / active_tenants`).
    pub fn new(total_bytes: u64, hard_ceiling_bytes: u64) -> Self {
        Self {
            total_bytes,
            default_floor_bytes: 0,
            default_hard_ceiling_bytes: hard_ceiling_bytes,
            high_watermark_frac: 0.9,
            per_tenant: HashMap::new(),
            ttl: None,
            kind_ceilings: HashMap::new(),
            pin_reserve_frac: 0.0,
        }
    }

    /// TD-CACHE-2 S2c: cap one artifact class at `frac` of the pool (see
    /// [`CacheBudget::kind_ceilings`]).
    pub fn with_kind_ceiling(mut self, kind: CacheKind, frac: f64) -> Self {
        self.kind_ceilings.insert(kind, frac.clamp(0.0, 1.0));
        self
    }

    /// TD-CACHE-3 S2: enable true-pinning by reserving `frac` of the pool for
    /// per-tenant floor segments (see [`CacheBudget::pin_reserve_frac`]).
    pub fn with_pin_reserve(mut self, frac: f64) -> Self {
        self.pin_reserve_frac = frac.clamp(0.0, 0.5);
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set the default guaranteed floor every tenant is protected up to.
    pub fn with_floor(mut self, floor_bytes: u64) -> Self {
        self.default_floor_bytes = floor_bytes;
        self
    }

    /// Override the pool-pressure threshold (fraction of `total_bytes`).
    pub fn with_high_watermark(mut self, frac: f64) -> Self {
        self.high_watermark_frac = frac;
        self
    }

    /// Override limits for a specific tenant (floor / hard ceiling / weight).
    pub fn with_tenant_limits(mut self, tenant: impl Into<String>, limits: TenantLimits) -> Self {
        self.per_tenant.insert(tenant.into(), limits);
        self
    }
}

#[derive(Default)]
struct TenantUsage {
    bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
}

/// A point-in-time per-tenant stats snapshot for metrics emission.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TenantCacheStat {
    pub tenant: String,
    pub bytes: u64,
    /// Bytes held in this tenant's pinned floor segment (subset of `bytes`).
    pub pinned_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub hit_ratio: f64,
}

/// TD-CACHE-3 S2 — the true-pin side store. Entries here live OUTSIDE the
/// shared moka pool: cross-tenant pressure cannot evict them (moka's TinyLFU
/// never sees them). Capacity discipline is two-level: a tenant may pin at
/// most its `floor_bytes` (within-tenant LRU recycling once full), and the
/// store as a whole never exceeds `reserve_bytes` (oversubscribed floors
/// degrade to admission-only — the S1 behavior — rather than stealing from
/// the shared pool).
struct PinnedEntry<V> {
    value: V,
    weight: u32,
    touch: AtomicU64,
}

struct PinnedStore<V> {
    entries: DashMap<CacheKey, PinnedEntry<V>>,
    tenant_bytes: DashMap<CacheScope, AtomicU64>,
    reserve_bytes: u64,
    used_bytes: AtomicU64,
    clock: AtomicU64,
}

/// Outcome of a pin attempt: whether the entry was pinned, and any same-tenant
/// entries recycled to make room (caller reconciles the byte gauges).
struct PinOutcome {
    pinned: bool,
    recycled: Vec<(CacheScope, CacheKind, u32)>,
}

impl<V: Clone + Send + Sync + 'static> PinnedStore<V> {
    fn new(reserve_bytes: u64) -> Self {
        Self {
            entries: DashMap::new(),
            tenant_bytes: DashMap::new(),
            reserve_bytes,
            used_bytes: AtomicU64::new(0),
            clock: AtomicU64::new(0),
        }
    }

    fn get(&self, key: &CacheKey) -> Option<V> {
        let e = self.entries.get(key)?;
        e.touch.store(
            self.clock.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Some(e.value.clone())
    }

    fn tenant_pinned(&self, scope: &CacheScope) -> u64 {
        self.tenant_bytes
            .get(scope)
            .map(|b| b.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Remove one entry, reconciling internal gauges. Returns its weight.
    fn remove(&self, key: &CacheKey) -> Option<u32> {
        let (_, e) = self.entries.remove(key)?;
        self.used_bytes
            .fetch_sub(e.weight as u64, Ordering::Relaxed);
        if let Some(b) = self.tenant_bytes.get(&key.scope) {
            b.fetch_sub(e.weight as u64, Ordering::Relaxed);
        }
        Some(e.weight)
    }

    /// The tenant's least-recently-touched pinned key (O(tenant entries); pin
    /// attempts are rare relative to gets, and a floor holds a bounded set).
    fn tenant_lru(&self, scope: &CacheScope) -> Option<CacheKey> {
        self.entries
            .iter()
            .filter(|e| e.key().scope == *scope)
            .min_by_key(|e| e.value().touch.load(Ordering::Relaxed))
            .map(|e| e.key().clone())
    }

    /// Try to pin `key` within the tenant's `floor_bytes`. Recycles the
    /// tenant's own LRU entries when the floor is full; never touches other
    /// tenants; never exceeds the global reserve.
    fn try_pin(&self, key: CacheKey, weight: u32, value: V, floor_bytes: u64) -> PinOutcome {
        let w = weight as u64;
        let mut recycled: Vec<(CacheScope, CacheKind, u32)> = Vec::new();
        if w == 0 || w > floor_bytes {
            return PinOutcome {
                pinned: false,
                recycled,
            };
        }
        // Replacing an existing pin of the same key: drop the old copy first.
        if let Some(old_w) = self.remove(&key) {
            recycled.push((key.scope.clone(), key.kind, old_w));
        }
        // Within-tenant LRU recycling until the floor fits the new entry.
        while self.tenant_pinned(&key.scope).saturating_add(w) > floor_bytes {
            let Some(lru) = self.tenant_lru(&key.scope) else {
                break;
            };
            if let Some(old_w) = self.remove(&lru) {
                recycled.push((lru.scope.clone(), lru.kind, old_w));
            } else {
                break;
            }
        }
        if self.tenant_pinned(&key.scope).saturating_add(w) > floor_bytes {
            return PinOutcome {
                pinned: false,
                recycled,
            };
        }
        // Global reserve check with rollback (tolerates racing pins).
        let prev = self.used_bytes.fetch_add(w, Ordering::Relaxed);
        if prev.saturating_add(w) > self.reserve_bytes {
            self.used_bytes.fetch_sub(w, Ordering::Relaxed);
            return PinOutcome {
                pinned: false,
                recycled,
            };
        }
        self.tenant_bytes
            .entry(key.scope.clone())
            .or_default()
            .fetch_add(w, Ordering::Relaxed);
        let touch = self.clock.fetch_add(1, Ordering::Relaxed);
        self.entries.insert(
            key,
            PinnedEntry {
                value,
                weight,
                touch: AtomicU64::new(touch),
            },
        );
        PinOutcome {
            pinned: true,
            recycled,
        }
    }
}

/// A multitenant, byte-budgeted, **work-conserving elastic** cache over `V`.
pub struct TenantCache<V: Clone + Send + Sync + 'static> {
    inner: Cache<CacheKey, CachedValue<V>>,
    usage: Arc<DashMap<CacheScope, TenantUsage>>,
    /// Live sum of admitted bytes across all tenants (pressure signal).
    global_bytes: Arc<AtomicU64>,
    total_bytes: u64,
    high_watermark: u64,
    default_floor: u64,
    default_hard_ceiling: u64,
    per_tenant: Arc<HashMap<String, TenantLimits>>,
    /// Optional tier-driven limits policy (Strategy seam); overrides the static
    /// per-tenant map when set.
    limits_resolver: Option<Arc<LimitsResolver>>,
    /// TD-CACHE-3 S2: true-pin side store (None = admission-only floors).
    pinned: Option<Arc<PinnedStore<V>>>,
    /// TD-CACHE-2 S2c: per-kind admission ceilings (bytes, pre-multiplied).
    kind_ceiling_bytes: Arc<HashMap<CacheKind, u64>>,
    /// Live per-kind admitted bytes (reconciled by the eviction listener).
    kind_bytes: Arc<DashMap<CacheKind, AtomicU64>>,
    /// Optional persistent L2. Values are written through before L1
    /// admission, so a later moka eviction is already represented on disk.
    l2: Option<Arc<dyn L2ValueStore<V>>>,
    l2_hits: Arc<AtomicU64>,
    l2_misses: Arc<AtomicU64>,
}

impl<V: Clone + Send + Sync + 'static> TenantCache<V> {
    /// Build a cache for the given byte `budget`. The eviction listener keeps the
    /// per-tenant and global byte gauges accurate under pressure.
    pub fn new(budget: CacheBudget) -> Self {
        let usage: Arc<DashMap<CacheScope, TenantUsage>> = Arc::new(DashMap::new());
        let global_bytes = Arc::new(AtomicU64::new(0));
        let listener_usage = usage.clone();
        let listener_global = global_bytes.clone();
        let kind_bytes: Arc<DashMap<CacheKind, AtomicU64>> = Arc::new(DashMap::new());
        let listener_kind = kind_bytes.clone();

        // TD-CACHE-3 S2: the pin reserve is carved OUT of the total so the
        // budget invariant holds: shared moka pool + pinned reserve = total.
        let pin_reserve =
            ((budget.total_bytes as f64) * budget.pin_reserve_frac.clamp(0.0, 0.5)) as u64;
        let shared_capacity = budget.total_bytes - pin_reserve;
        let mut builder = Cache::builder()
            .max_capacity(shared_capacity)
            .weigher(|_k: &CacheKey, v: &CachedValue<V>| v.weight)
            .eviction_listener(
                move |k: Arc<CacheKey>, v: CachedValue<V>, cause: RemovalCause| {
                    listener_global.fetch_sub(v.weight as u64, Ordering::Relaxed);
                    if let Some(b) = listener_kind.get(&k.kind) {
                        b.fetch_sub(v.weight as u64, Ordering::Relaxed);
                    }
                    if let Some(u) = listener_usage.get(&k.scope) {
                        u.bytes.fetch_sub(v.weight as u64, Ordering::Relaxed);
                        // Count true capacity/expiry evictions (not explicit removals).
                        if matches!(cause, RemovalCause::Size | RemovalCause::Expired) {
                            u.evictions.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                },
            );
        if let Some(ttl) = budget.ttl {
            builder = builder.time_to_live(ttl);
        }

        let high_watermark =
            ((budget.total_bytes as f64) * budget.high_watermark_frac.clamp(0.0, 1.0)) as u64;

        Self {
            inner: builder.build(),
            usage,
            global_bytes,
            total_bytes: budget.total_bytes,
            high_watermark,
            default_floor: budget.default_floor_bytes,
            default_hard_ceiling: budget.default_hard_ceiling_bytes,
            per_tenant: Arc::new(budget.per_tenant),
            limits_resolver: None,
            pinned: (pin_reserve > 0).then(|| Arc::new(PinnedStore::new(pin_reserve))),
            kind_ceiling_bytes: Arc::new(
                budget
                    .kind_ceilings
                    .iter()
                    .map(|(k, f)| (*k, ((budget.total_bytes as f64) * f) as u64))
                    .collect(),
            ),
            kind_bytes,
            l2: None,
            l2_hits: Arc::new(AtomicU64::new(0)),
            l2_misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Plug a tier-driven limits policy (the Strategy seam): a
    /// `tenant_id → TenantLimits` resolver, e.g. catalog tier lookup +
    /// [`TenantTier::limits`]. Overrides the static per-tenant map.
    pub fn with_limits_resolver(mut self, resolver: Arc<LimitsResolver>) -> Self {
        self.limits_resolver = Some(resolver);
        self
    }

    /// Attach a persistent L2. With no backend the historical DRAM-only path
    /// is unchanged.
    pub fn with_l2_backend(mut self, backend: Arc<dyn L2ValueStore<V>>) -> Self {
        self.l2 = Some(backend);
        self
    }

    fn usage_for(
        &self,
        scope: &CacheScope,
    ) -> dashmap::mapref::one::RefMut<'_, CacheScope, TenantUsage> {
        self.usage.entry(scope.clone()).or_default()
    }

    /// TD-CACHE-3 S3: the effective limits (entitlement) for a tenant —
    /// resolver-driven when set, static map/defaults otherwise. Public so the
    /// host can meter pinned-vs-entitled bytes for billing true-up.
    pub fn entitlement(&self, tenant: &str) -> TenantLimits {
        self.limits_for(tenant)
    }

    fn limits_for(&self, tenant: &str) -> TenantLimits {
        if let Some(resolver) = &self.limits_resolver {
            return resolver(tenant);
        }
        self.per_tenant
            .get(tenant)
            .copied()
            .unwrap_or(TenantLimits {
                floor_bytes: self.default_floor,
                hard_ceiling_bytes: self.default_hard_ceiling,
                weight: 1,
            })
    }

    /// Sum of weights over tenants currently holding bytes (for the contended
    /// weighted fair share). O(active tenants); fine for a footer/metadata cache.
    fn active_weight(&self) -> u64 {
        self.usage
            .iter()
            .filter(|e| e.value().bytes.load(Ordering::Relaxed) > 0)
            .map(|e| self.limits_for(&e.key().label()).weight.max(1) as u64)
            .sum()
    }

    /// The work-conserving elastic admission decision for `tenant` adding
    /// `weight` bytes: absolute hard-ceiling cap always; below the high watermark
    /// borrow freely from the idle pool; above it, enforce the weighted fair
    /// share (`total * w_t / Σ active w`) clamped to `[floor, hard_ceiling]`.
    /// TD-CACHE-2 S2c: kind-ceiling admission — a class at/over its share of
    /// the pool bypasses caching so it cannot flush higher-value classes.
    fn kind_admits(&self, kind: CacheKind, weight: u64) -> bool {
        let Some(ceiling) = self.kind_ceiling_bytes.get(&kind) else {
            return true;
        };
        let used = self
            .kind_bytes
            .get(&kind)
            .map(|b| b.load(Ordering::Relaxed))
            .unwrap_or(0);
        used.saturating_add(weight) <= *ceiling
    }

    fn should_admit(&self, scope: &CacheScope, weight: u64) -> bool {
        let limits = self.limits_for(&scope.label());
        let u = self.usage_for(scope).bytes.load(Ordering::Relaxed);
        // 1. absolute runaway guard — always enforced.
        if u.saturating_add(weight) > limits.hard_ceiling_bytes {
            return false;
        }
        // 2. pool not under pressure → elastic borrow up to the hard ceiling.
        let g = self.global_bytes.load(Ordering::Relaxed);
        if g.saturating_add(weight) <= self.high_watermark {
            return true;
        }
        // 3. under pressure → weighted fair share (with floor guarantee).
        let active_w = self.active_weight();
        let my_w = limits.weight.max(1) as u64;
        let fair = if active_w == 0 {
            limits.hard_ceiling_bytes
        } else {
            ((self.total_bytes as u128 * my_w as u128 / active_w as u128) as u64)
                .clamp(limits.floor_bytes, limits.hard_ceiling_bytes)
        };
        u.saturating_add(weight) <= fair
    }

    async fn lookup(&self, key: &CacheKey) -> Option<(V, CacheReadOutcome)> {
        if let Some(pinned) = &self.pinned
            && let Some(v) = pinned.get(key)
        {
            self.usage_for(&key.scope)
                .hits
                .fetch_add(1, Ordering::Relaxed);
            return Some((v, CacheReadOutcome::MemoryHit));
        }
        let result = self.inner.get(key).await;
        let u = self.usage_for(&key.scope);
        match result {
            Some(cv) => {
                u.hits.fetch_add(1, Ordering::Relaxed);
                Some((cv.value, CacheReadOutcome::MemoryHit))
            }
            None => {
                u.misses.fetch_add(1, Ordering::Relaxed);
                drop(u);
                let Some(l2) = &self.l2 else {
                    return None;
                };
                match l2.get(key).await {
                    Ok(Some((value, weight))) => {
                        self.l2_hits.fetch_add(1, Ordering::Relaxed);
                        self.insert_l1(key.clone(), weight, value.clone()).await;
                        Some((value, CacheReadOutcome::PersistentHit))
                    }
                    Ok(None) | Err(_) => {
                        self.l2_misses.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                }
            }
        }
    }

    /// Look up a value, recording a per-tenant hit or miss. Pinned entries
    /// (the tenant's floor working set) are checked before the shared pool.
    pub async fn get(&self, key: &CacheKey) -> Option<V> {
        self.lookup(key).await.map(|(value, _)| value)
    }

    /// TD-RDSTRAT-12 §3: RAM-residency probe with NO hit/miss accounting and
    /// NO inserts — answers "would `get_or_load` for this key be served from
    /// DRAM without running its loader?". Checks the pinned floor set and the
    /// shared L1 only; persistent L2 is deliberately unchecked so callers
    /// treat `None` as "a fetch may still be needed" (conservative: a resident
    /// disk-tier range could then be re-fetched once through the caller's own
    /// consume path — never double-billed at the object-store boundary twice
    /// per query, because the consumer that owns the persistent tier is the
    /// same caller). The moka recency of a probed entry is refreshed; there is
    /// no other observable effect.
    pub async fn peek_memory(&self, key: &CacheKey) -> Option<V> {
        if let Some(pinned) = &self.pinned
            && let Some(v) = pinned.get(key)
        {
            return Some(v);
        }
        self.inner.get(key).await.map(|cv| cv.value)
    }

    /// Lookup with the physical cache tier that satisfied it. Engine wrappers
    /// use this to forward truthful per-query evidence without making the
    /// foundation cache depend on an observability crate.
    pub async fn get_with_outcome(&self, key: &CacheKey) -> Option<(V, CacheReadOutcome)> {
        self.lookup(key).await
    }

    /// Get `key`, or run `loader` on miss and cache the result **iff** the
    /// tenant is under its byte ceiling (otherwise the value is returned without
    /// caching — bypass — so one tenant cannot monopolize the pool). `weight` is
    /// the value's byte size for the budget.
    pub async fn get_or_load<F, Fut, E>(
        &self,
        key: CacheKey,
        weight: u32,
        loader: F,
    ) -> Result<(V, CacheReadOutcome), E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V, E>>,
    {
        if let Some(found) = self.lookup(&key).await {
            return Ok(found);
        }
        // `lookup` already recorded the miss.
        let value = loader().await?;

        if let Some(l2) = &self.l2 {
            let _ = l2.put(&key, weight, value.clone()).await;
        }
        self.insert_l1(key, weight, value.clone()).await;
        Ok((value, CacheReadOutcome::Loaded))
    }

    /// TD-CACHE-3 S2: route an admitted insert into the tenant's pinned floor
    /// segment when true-pinning is on and the tenant has a floor. Returns
    /// true when the entry was pinned (shared-pool insert must be skipped).
    /// Recycled same-tenant entries are reconciled into the byte gauges here
    /// (they were accounted at their own admission).
    fn try_pin_admitted(&self, key: &CacheKey, weight: u32, value: &V) -> bool {
        let Some(pinned) = &self.pinned else {
            return false;
        };
        let floor = self.limits_for(&key.scope.label()).floor_bytes;
        if floor == 0 {
            return false;
        }
        let outcome = pinned.try_pin(key.clone(), weight, value.clone(), floor);
        for (tenant, kind, w) in outcome.recycled {
            self.global_bytes.fetch_sub(w as u64, Ordering::Relaxed);
            if let Some(b) = self.kind_bytes.get(&kind) {
                b.fetch_sub(w as u64, Ordering::Relaxed);
            }
            if let Some(u) = self.usage.get(&tenant) {
                u.bytes.fetch_sub(w as u64, Ordering::Relaxed);
                u.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        outcome.pinned
    }

    /// Explicitly insert/replace a value, subject to the elastic admission policy.
    pub async fn insert(&self, key: CacheKey, weight: u32, value: V) {
        if let Some(l2) = &self.l2 {
            let _ = l2.put(&key, weight, value.clone()).await;
        }
        self.insert_l1(key, weight, value).await;
    }

    /// Admit directly into DRAM without writing the attached L2.
    ///
    /// This is intentionally narrow: callers use it when the same immutable
    /// bytes are persisted under a different, range-aware parent key. Writing
    /// through the ordinary exact-key adapter as well would duplicate the
    /// complete region on disk.
    pub async fn insert_memory_only(&self, key: CacheKey, weight: u32, value: V) {
        self.insert_l1(key, weight, value).await;
    }

    async fn insert_l1(&self, key: CacheKey, weight: u32, value: V) {
        if !self.should_admit(&key.scope, weight as u64)
            || !self.kind_admits(key.kind, weight as u64)
        {
            return;
        }
        self.account_insert(&key.scope, key.kind, weight);
        if !self.try_pin_admitted(&key, weight, &value) {
            self.inner.insert(key, CachedValue { weight, value }).await;
        }
    }

    fn account_insert(&self, scope: &CacheScope, kind: CacheKind, weight: u32) {
        let u = self.usage_for(scope);
        u.bytes.fetch_add(weight as u64, Ordering::Relaxed);
        u.inserts.fetch_add(1, Ordering::Relaxed);
        self.global_bytes
            .fetch_add(weight as u64, Ordering::Relaxed);
        self.kind_bytes
            .entry(kind)
            .or_default()
            .fetch_add(weight as u64, Ordering::Relaxed);
    }

    /// TD-CACHE-2 S2d: remove every entry (shared pool AND pinned floor)
    /// whose key matches `pred`. Used by compaction to evict entries for
    /// deleted segment files — without this, dead-file entries squat in the
    /// budget until recency ages them out. Gauges are reconciled (moka via
    /// the eviction listener; pinned explicitly here).
    pub async fn purge_where(&self, pred: impl Fn(&CacheKey) -> bool) -> usize {
        let victims: Vec<CacheKey> = self
            .inner
            .iter()
            .filter(|(k, _)| pred(k))
            .map(|(k, _)| (*k).clone())
            .collect();
        let mut removed = victims.len();
        for k in &victims {
            self.inner.invalidate(k).await;
            if let Some(l2) = &self.l2 {
                let _ = l2.remove(k).await;
            }
        }
        if let Some(p) = &self.pinned {
            let pinned_victims: Vec<CacheKey> = p
                .entries
                .iter()
                .filter(|e| pred(e.key()))
                .map(|e| e.key().clone())
                .collect();
            for k in pinned_victims {
                if let Some(w) = p.remove(&k) {
                    removed += 1;
                    if let Some(l2) = &self.l2 {
                        let _ = l2.remove(&k).await;
                    }
                    self.global_bytes.fetch_sub(w as u64, Ordering::Relaxed);
                    if let Some(b) = self.kind_bytes.get(&k.kind) {
                        b.fetch_sub(w as u64, Ordering::Relaxed);
                    }
                    if let Some(u) = self.usage.get(&k.scope) {
                        u.bytes.fetch_sub(w as u64, Ordering::Relaxed);
                    }
                }
            }
        }
        removed
    }

    /// Drain moka's pending maintenance (eviction listener, etc.). Call before
    /// reading [`Self::tenant_stats`] in tests for deterministic gauges.
    pub async fn sync(&self) {
        self.inner.run_pending_tasks().await;
    }

    /// Per-tenant stats snapshot for metrics emission.
    pub fn tenant_stats(&self) -> Vec<TenantCacheStat> {
        self.usage
            .iter()
            .map(|e| {
                let u = e.value();
                let hits = u.hits.load(Ordering::Relaxed);
                let misses = u.misses.load(Ordering::Relaxed);
                let total = hits + misses;
                TenantCacheStat {
                    tenant: e.key().label(),
                    bytes: u.bytes.load(Ordering::Relaxed),
                    pinned_bytes: self
                        .pinned
                        .as_ref()
                        .map(|p| p.tenant_pinned(e.key()))
                        .unwrap_or(0),
                    hits,
                    misses,
                    inserts: u.inserts.load(Ordering::Relaxed),
                    evictions: u.evictions.load(Ordering::Relaxed),
                    hit_ratio: if total == 0 {
                        0.0
                    } else {
                        hits as f64 / total as f64
                    },
                }
            })
            .collect()
    }

    /// Current tracked byte usage for a tenant (post-`sync`).
    pub fn tenant_bytes(&self, tenant: &str) -> u64 {
        self.usage
            .get(&CacheScope::named(tenant))
            .map(|u| u.bytes.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Current tracked byte usage for a catalog-authoritative stable tenant.
    pub fn stable_tenant_bytes(&self, tenant_stable_id: u64) -> u64 {
        self.usage
            .get(&CacheScope::stable_tenant(tenant_stable_id))
            .map(|u| u.bytes.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Persistent-tier counters for metrics emission.
    pub fn l2_stats(&self) -> L2CacheStats {
        L2CacheStats {
            hits: self.l2_hits.load(Ordering::Relaxed),
            misses: self.l2_misses.load(Ordering::Relaxed),
            resident_bytes: self.l2.as_ref().map_or(0, |l2| l2.resident_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tenant: &str, k: &str) -> CacheKey {
        CacheKey::new(tenant, CacheKind::Footer, k)
    }

    // ---- TD-TIER-1 SC4-1: the residency-budget bridge ----

    /// The bridge is order-independent: setting a budget BEFORE any store is
    /// registered parks it in the cell; a late registration picks it up.
    #[tokio::test]
    async fn late_store_registration_picks_up_the_cell_budget() {
        let dir = tempfile::tempdir().expect("test tempdir");
        set_process_reserved_budget(4096);
        assert_eq!(process_reserved_budget(), 4096);
        let store = Arc::new(PersistentByteStore::open(dir.path(), 1 << 20).expect("open"));
        assert!(register_process_store(store.clone()));
        assert_eq!(
            store.reserved_budget(),
            4096,
            "a late registration applies the current cell"
        );
        // A later budget update arms the registered store immediately.
        set_process_reserved_budget(8192);
        assert_eq!(store.reserved_budget(), 8192);
    }

    #[tokio::test]
    async fn persistent_l2_survives_a_new_l1_instance() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = Arc::new(
            PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent cache"),
        );
        let first = TenantCache::new(CacheBudget::new(1024, 1024)).with_l2_backend(Arc::new(
            PersistentArcBytesL2::new(store.clone(), "survivor", L2Class::Survivor),
        ));
        let cache_key = CacheKey::new("tenant-a", CacheKind::QuantizedCodes, "segment:0:5");
        first
            .insert(cache_key.clone(), 5, Arc::from(&b"bytes"[..]))
            .await;
        drop(first);

        let second = TenantCache::new(CacheBudget::new(1024, 1024)).with_l2_backend(Arc::new(
            PersistentArcBytesL2::new(store, "survivor", L2Class::Survivor),
        ));
        assert_eq!(second.get(&cache_key).await.as_deref(), Some(&b"bytes"[..]));
        assert_eq!(second.l2_stats().hits, 1);
    }

    #[tokio::test]
    async fn stable_tenant_scope_stays_numeric_and_distinct_from_named_alias() {
        let c = TenantCache::new(CacheBudget::new(1024, 1024));
        let stable = CacheKey::for_tenant_id(42, CacheKind::Footer, "segment");
        let textual = CacheKey::new("42", CacheKind::Footer, "segment");

        assert_ne!(
            stable, textual,
            "a textual alias must not impersonate a catalog stable tenant id"
        );
        c.insert(stable.clone(), 8, 7).await;
        assert_eq!(c.get(&stable).await, Some(7));
        assert_eq!(c.get(&textual).await, None);
        assert_eq!(c.stable_tenant_bytes(42), 8);
    }

    /// TD-RDSTRAT-12 §3: `peek_memory` reports DRAM residency (L1/pinned)
    /// only — a value that would be served from persistent L2 reads as None
    /// so batch-prefetch planners conservatively count a fetch, and peeking
    /// never promotes.
    #[tokio::test]
    async fn peek_memory_is_l1_only_and_never_promotes_from_l2() {
        let dir = tempfile::tempdir().expect("test tempdir");
        let store = Arc::new(
            PersistentByteStore::open(dir.path(), 1 << 20).expect("open persistent cache"),
        );
        let first = TenantCache::new(CacheBudget::new(1024, 1024)).with_l2_backend(Arc::new(
            PersistentArcBytesL2::new(store.clone(), "survivor", L2Class::Survivor),
        ));
        let k = CacheKey::new("tenant-a", CacheKind::QuantizedCodes, "seg:0:5");
        first.insert(k.clone(), 5, Arc::from(&b"bytes"[..])).await;
        drop(first);

        let second = TenantCache::new(CacheBudget::new(1024, 1024)).with_l2_backend(Arc::new(
            PersistentArcBytesL2::new(store, "survivor", L2Class::Survivor),
        ));
        // Resident only in persistent L2: peek reports None (no promotion).
        assert_eq!(
            second.peek_memory(&k).await.as_deref(),
            None,
            "peek must not see through to the persistent tier"
        );
        // A real get promotes; afterwards peek sees it.
        assert_eq!(second.get(&k).await.as_deref(), Some(&b"bytes"[..]));
        assert_eq!(second.peek_memory(&k).await.as_deref(), Some(&b"bytes"[..]));
    }

    /// TD-CACHE-3 S2: a churning tenant CANNOT evict another tenant's pinned
    /// floor. Tiny shared pool + pin reserve; A pins its floor working set;
    /// B floods far past total capacity; every one of A's floor entries is
    /// still served (with admission-only floors, moka's TinyLFU would have
    /// reclaimed them under B's pressure).
    #[tokio::test]
    async fn pinned_floor_survives_cross_tenant_flood() {
        // total 10 KB, pin reserve 50% = 5 KB; A floor 4 KB, huge ceilings.
        let budget = CacheBudget::new(10_000, 10_000)
            .with_pin_reserve(0.5)
            .with_tenant_limits(
                "A",
                TenantLimits {
                    floor_bytes: 4_000,
                    hard_ceiling_bytes: 10_000,
                    weight: 4,
                },
            );
        let c: TenantCache<u64> = TenantCache::new(budget);
        for i in 0..4u64 {
            c.insert(key("A", &format!("hot{i}")), 1_000, i).await;
        }
        // B floods 100 KB through a 5 KB shared pool.
        for i in 0..100u64 {
            c.insert(key("B", &format!("churn{i}")), 1_000, i).await;
        }
        c.sync().await;
        for i in 0..4u64 {
            assert_eq!(
                c.get(&key("A", &format!("hot{i}"))).await,
                Some(i),
                "A's pinned floor entry hot{i} must survive B's flood"
            );
        }
        let stats = c.tenant_stats();
        let a = stats.iter().find(|s| s.tenant == "A").unwrap();
        assert_eq!(a.pinned_bytes, 4_000, "A's whole floor is pinned");
    }

    /// TD-CACHE-3 S2: a full floor recycles WITHIN the tenant (own LRU out,
    /// new entry in) — the floor is a working set, not a write-once set.
    #[tokio::test]
    async fn full_floor_recycles_within_tenant_by_lru() {
        let budget = CacheBudget::new(10_000, 10_000)
            .with_pin_reserve(0.5)
            .with_tenant_limits(
                "A",
                TenantLimits {
                    floor_bytes: 2_000,
                    hard_ceiling_bytes: 10_000,
                    weight: 1,
                },
            );
        let c: TenantCache<u64> = TenantCache::new(budget);
        c.insert(key("A", "old"), 1_000, 1).await;
        c.insert(key("A", "warm"), 1_000, 2).await;
        // Touch "warm" so "old" is the LRU pin.
        assert_eq!(c.get(&key("A", "warm")).await, Some(2));
        // Floor full (2 KB): the next pin must recycle "old", not "warm".
        c.insert(key("A", "new"), 1_000, 3).await;
        c.sync().await;
        assert_eq!(c.get(&key("A", "new")).await, Some(3), "new entry pinned");
        assert_eq!(
            c.get(&key("A", "warm")).await,
            Some(2),
            "recently-touched pin kept"
        );
        let stats = c.tenant_stats();
        let a = stats.iter().find(|s| s.tenant == "A").unwrap();
        assert_eq!(a.pinned_bytes, 2_000, "floor stays exactly full");
    }

    /// TD-CACHE-3 S2: oversubscribed floors degrade to admission-only (the
    /// entry still lands in the shared pool) — the reserve is never exceeded
    /// and no panic/starvation occurs.
    #[tokio::test]
    async fn oversubscribed_reserve_degrades_to_shared_pool() {
        // Reserve = 1 KB but the floor claims 4 KB: only 1 KB can pin.
        let budget = CacheBudget::new(10_000, 10_000)
            .with_pin_reserve(0.1)
            .with_tenant_limits(
                "A",
                TenantLimits {
                    floor_bytes: 4_000,
                    hard_ceiling_bytes: 10_000,
                    weight: 1,
                },
            );
        let c: TenantCache<u64> = TenantCache::new(budget);
        for i in 0..4u64 {
            c.insert(key("A", &format!("k{i}")), 1_000, i).await;
        }
        c.sync().await;
        // All 4 entries retrievable (pinned or shared) — nothing lost.
        for i in 0..4u64 {
            assert_eq!(c.get(&key("A", &format!("k{i}"))).await, Some(i));
        }
        let stats = c.tenant_stats();
        let a = stats.iter().find(|s| s.tenant == "A").unwrap();
        assert!(
            a.pinned_bytes <= 1_000,
            "pinned {} must not exceed the 1 KB reserve",
            a.pinned_bytes
        );
    }

    /// TD-CACHE-2 S2c: a kind at its ceiling bypasses caching (values still
    /// returned) while other kinds admit freely — OID churn cannot flush the
    /// recall-critical class.
    #[tokio::test]
    async fn kind_ceiling_caps_one_class() {
        let budget = CacheBudget::new(10_000, 10_000).with_kind_ceiling(CacheKind::Other, 0.2);
        let c: TenantCache<u64> = TenantCache::new(budget);
        // Other floods: only ~2 KB (20%) admits.
        for i in 0..50u64 {
            c.insert(
                CacheKey::new("A", CacheKind::Other, format!("o{i}")),
                500,
                i,
            )
            .await;
        }
        // QuantizedCodes admits unhindered.
        for i in 0..8u64 {
            c.insert(
                CacheKey::new("A", CacheKind::QuantizedCodes, format!("q{i}")),
                500,
                i,
            )
            .await;
        }
        c.sync().await;
        let other_hits = {
            let mut n = 0;
            for i in 0..50u64 {
                if c.get(&CacheKey::new("A", CacheKind::Other, format!("o{i}")))
                    .await
                    .is_some()
                {
                    n += 1;
                }
            }
            n
        };
        assert!(
            other_hits <= 4,
            "Other capped at 20% of 10KB = 4 x 500B entries, saw {other_hits}"
        );
        for i in 0..8u64 {
            assert_eq!(
                c.get(&CacheKey::new(
                    "A",
                    CacheKind::QuantizedCodes,
                    format!("q{i}")
                ))
                .await,
                Some(i),
                "QuantizedCodes must be unaffected by the Other flood"
            );
        }
    }

    #[tokio::test]
    async fn key_namespacing_isolates_tenants() {
        let c: TenantCache<u64> = TenantCache::new(CacheBudget::new(1 << 20, 1 << 20));
        c.insert(key("A", "seg1"), 8, 100).await;
        c.insert(key("B", "seg1"), 8, 200).await;
        c.sync().await;
        assert_eq!(c.get(&key("A", "seg1")).await, Some(100));
        assert_eq!(c.get(&key("B", "seg1")).await, Some(200));
        // Same artifact key, different tenant → different entries.
        assert_eq!(c.get(&key("A", "missing")).await, None);
    }

    #[tokio::test]
    async fn get_or_load_records_hit_miss_and_gauges() {
        let c: TenantCache<u64> = TenantCache::new(CacheBudget::new(1 << 20, 1 << 20));
        let k = key("A", "seg1");
        // miss → loads → caches
        let v = c
            .get_or_load(k.clone(), 16, || async { Ok::<u64, ()>(42) })
            .await
            .unwrap();
        assert_eq!(v, (42, CacheReadOutcome::Loaded));
        // hit → no reload
        let v2 = c
            .get_or_load(k.clone(), 16, || async { Ok::<u64, ()>(999) })
            .await
            .unwrap();
        assert_eq!(
            v2,
            (42, CacheReadOutcome::MemoryHit),
            "second call must hit cache, not reload"
        );
        c.sync().await;
        let s = c.tenant_stats();
        let a = s.iter().find(|s| s.tenant == "A").unwrap();
        assert_eq!(a.hits, 1);
        assert_eq!(a.misses, 1);
        assert_eq!(a.inserts, 1);
        assert_eq!(a.bytes, 16);
    }

    #[tokio::test]
    async fn per_tenant_ceiling_caps_one_tenant() {
        // Big global pool, tiny per-tenant ceiling.
        let c: TenantCache<u64> = TenantCache::new(CacheBudget::new(1 << 30, /*ceiling*/ 100));
        // Tenant A floods past its 100-byte ceiling.
        for i in 0..50u64 {
            c.insert(
                CacheKey::new("A", CacheKind::Footer, format!("k{i}")),
                10,
                i,
            )
            .await;
        }
        c.sync().await;
        // A is capped at/around the ceiling, NOT all 500 bytes.
        assert!(
            c.tenant_bytes("A") <= 100,
            "A bytes {} > ceiling",
            c.tenant_bytes("A")
        );
        // Tenant B (separate ceiling) is unaffected by A's flood.
        c.insert(CacheKey::new("B", CacheKind::Footer, "b1"), 10, 7)
            .await;
        c.sync().await;
        assert_eq!(
            c.get(&CacheKey::new("B", CacheKind::Footer, "b1")).await,
            Some(7)
        );
    }

    #[tokio::test]
    async fn solo_tenant_borrows_idle_pool_up_to_hard_ceiling() {
        // Elasticity: with the pool mostly idle, a solo tenant borrows well past
        // the pressure watermark, up to its hard ceiling — no stranding.
        let budget = CacheBudget::new(1000, /*hard*/ 800).with_high_watermark(0.5); // hwm=500
        let c: TenantCache<u64> = TenantCache::new(budget);
        for i in 0..200u64 {
            c.insert(
                CacheKey::new("A", CacheKind::Footer, format!("k{i}")),
                10,
                i,
            )
            .await;
        }
        c.sync().await;
        let a = c.tenant_bytes("A");
        assert!(
            a > 500,
            "solo tenant {a} did not borrow past watermark (500)"
        );
        assert!(a <= 800, "solo tenant {a} exceeded hard ceiling (800)");
    }

    #[tokio::test]
    async fn fair_share_bounds_tenants_under_pressure() {
        // Under contention, the weighted fair share (total/active) bounds every
        // tenant — no monopoly — while the pool stays fully used (no fragmentation).
        let budget = CacheBudget::new(1000, /*hard*/ 800).with_high_watermark(0.5);
        let c: TenantCache<u64> = TenantCache::new(budget);
        let tenants = ["A", "B", "C", "D"];
        // Round-robin so no tenant gets a head start.
        for round in 0..60u64 {
            for t in tenants {
                c.insert(
                    CacheKey::new(t, CacheKind::Footer, format!("k{round}")),
                    10,
                    round,
                )
                .await;
            }
        }
        c.sync().await;
        let total: u64 = tenants.iter().map(|t| c.tenant_bytes(t)).sum();
        assert!(total <= 1000 + 40, "pool over budget: {total}");
        // Fair share with 4 active ≈ 250; assert no tenant hoards (≤ ~share+slack).
        for t in tenants {
            let b = c.tenant_bytes(t);
            assert!(b <= 250 + 60, "tenant {t} hoarded {b} (> fair share)");
        }
        // And capacity is actually used (not stranded): pool well-filled.
        assert!(total >= 700, "pool under-utilized: {total}");
    }

    #[tokio::test]
    async fn byte_weighted_eviction_respects_total_budget() {
        // Global pool 100 bytes; ceiling high so the GLOBAL budget is what bites.
        let c: TenantCache<u64> = TenantCache::new(CacheBudget::new(100, 1 << 30));
        for i in 0..40u64 {
            c.insert(
                CacheKey::new("A", CacheKind::Footer, format!("k{i}")),
                10,
                i,
            )
            .await;
        }
        c.sync().await;
        // moka holds total weight ≈ max_capacity, not all 400 bytes.
        assert!(
            c.tenant_bytes("A") <= 100 + 10,
            "tracked bytes {} exceed pool",
            c.tenant_bytes("A")
        );
    }

    #[tokio::test]
    async fn json_tier_policy_gives_higher_tier_preference_under_pressure() {
        // Operator-supplied JSON tier policy (the open-core seam): a higher-tier
        // tenant wins the contended fair share. Tier ids are arbitrary strings
        // (here mimicking the control plane's pricing tiers) — OSS bakes in none.
        let total = 1600u64;
        let json = r#"{
            "default_tier": "free_trial",
            "tiers": {
                "free_trial": {"weight": 1, "floor_frac": 0.0,    "ceiling_frac": 0.0625},
                "enterprise": {"weight": 8, "floor_frac": 0.0625, "ceiling_frac": 0.5}
            }
        }"#;
        let policy = Arc::new(TierPolicy::from_json(json).unwrap());
        // Host-supplied tenant→tier authority (in prod: TenantContext.tier).
        let tenant_to_tier: Arc<dyn Fn(&str) -> String + Send + Sync> = Arc::new(|t: &str| {
            if t == "ent" {
                "enterprise".into()
            } else {
                "free_trial".into()
            }
        });
        let resolver = policy.resolver(total, tenant_to_tier);

        let budget = CacheBudget::new(total, total).with_high_watermark(0.5); // hwm=800
        let c: TenantCache<u64> = TenantCache::new(budget).with_limits_resolver(resolver);

        for round in 0..200u64 {
            for t in ["ent", "free"] {
                c.insert(
                    CacheKey::new(t, CacheKind::Footer, format!("k{round}")),
                    10,
                    round,
                )
                .await;
            }
        }
        c.sync().await;
        let ent = c.tenant_bytes("ent");
        let free = c.tenant_bytes("free");
        assert!(
            ent > free * 3,
            "enterprise {ent} not strongly preferred over free {free}"
        );
        // Free is bounded by its tier ceiling (total/16 = 100).
        assert!(free <= 100 + 20, "free {free} exceeded its tier ceiling");
        // Enterprise bounded by its tier ceiling (total/2 = 800).
        assert!(
            ent <= 800 + 20,
            "enterprise {ent} exceeded its tier ceiling"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_get_or_load_is_safe() {
        let c: Arc<TenantCache<u64>> =
            Arc::new(TenantCache::new(CacheBudget::new(1 << 20, 1 << 20)));
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let cc = c.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..100u64 {
                    let k = CacheKey::new("A", CacheKind::Footer, format!("k{}", i % 10));
                    let _ = cc
                        .get_or_load(k, 8, || async move { Ok::<u64, ()>(t + i) })
                        .await;
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        c.sync().await;
        // 10 distinct keys × 8 bytes = 80 bytes tracked (no double counting).
        assert!(c.tenant_bytes("A") <= 80, "bytes {}", c.tenant_bytes("A"));
    }
}
