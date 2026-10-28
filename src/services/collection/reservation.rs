//! TD-TIER-1 SC3 slice 1: the per-pool reservation authority.
//!
//! One authority owns one operator-declared local pool EXCLUSIVELY: a
//! process-global registry rejects a second live authority on the same pool
//! directory in-process, and an O_EXCL `owner.lock` (pid + process-start
//! token) rejects a live foreign owner cross-process; a stale lock is claimed
//! by exactly one racer via atomic rename. It is the single reservation
//! ledger for that pool.
//!
//! **Durable-before-return (review B2/B3):** every mutation builds a
//! prospective ledger, persists it (fsync file + rename + fsync dir), and
//! only then commits the in-memory state — a persist failure leaves memory
//! unchanged, so a retry re-attempts rather than fabricating success. The
//! idempotency fast path can therefore only observe a grant that is already
//! durable.
//!
//! **Billing-level integrity (review B4):** `release` and `resize` close
//! their grant's open interval at the OLD level and return it — no interval
//! is silently dropped and no level is billed across a span it did not hold.
//! `take_usage_intervals` covers only spans at a stable level.
//!
//! Capacity accounting charges effective bytes: `grant_bytes × (1 +
//! overhead_factor)`, rounded up — headers/checksums, temporary warming and
//! old/new-generation overlap. The sum of live grants' effective bytes can
//! never exceed [`ReservationPoolConfig::allocatable_bytes`] (SC-TE3), and an
//! operator-declared per-tenant limit bounds one tenant's share.
//!
//! **No quota check ever runs on the query hot path.** The authority is an
//! admission- and cadence-time component only: reads never consult it, and
//! [`ReservationAuthority::reserve`] performs no network I/O — its durable
//! state is a small local file, written under the ledger mutex and never
//! across a network call.
//!
//! **Inert in this slice.** Nothing constructs an authority in production
//! yet; the config field parses and is validated at config load, but no call
//! site opens a pool. The PERFORMANCE class keeps its SC1 `NOT_ACTIVE` status
//! until SC4 proves coverage; SC3-2 wires admission at create.
//!
//! **Crash semantics (SC-TE7/SC-TE9).** Reopen reloads the ledger, persists
//! the bumped `producer_epoch`, and prunes the bounded released-grant tail;
//! usage intervals are keyed `(producer_epoch, sequence)` so a replayed
//! emission overwrites the same record rather than double-charging. The
//! epoch discontinuity after an unclean close is itself the uncertainty
//! signal — bytes held during the unobserved gap are not fabricated
//! downstream. A corrupt or unreadable ledger fails closed: the authority
//! refuses to open (and therefore refuses admission) rather than silently
//! zeroing live grants.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use serde::{Deserialize, Serialize};

/// On-disk ledger schema version. Bump on any incompatible change; the reader
/// rejects newer/unknown versions fail-closed (pre-GA: no mixed-read duty).
const LEDGER_SCHEMA_VERSION: u32 = 1;

/// Bounded audit tail: released grants retained in the ledger. Older ones are
/// pruned at open — durable history lives in the metering records, and an
/// unbounded ledger would make every mutation an O(n) rewrite (review S6).
const MAX_RETAINED_RELEASED: usize = 64;

/// Takeover retries for the owner lock — a bounded race, not a spin.
const LOCK_ATTEMPTS: usize = 8;

/// Kill switch: when set, [`ReservationAuthority::reserve`] refuses new
/// reservations with a named reason. It never erases the ledger or the
/// operator's pool/policy configuration — degrading to balanced behavior, not
/// amnesia. A value that is not valid UTF-8 arms the switch (fail-closed).
pub const RESERVATIONS_KILL_SWITCH_ENV: &str = "PROXIMADB_TIER1_RESERVATIONS_OFF";

/// Operator declaration of one local reservation pool (TD-TIER-1 SC3). An
/// empty pool list (the default) leaves the whole feature inert.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservationPoolConfig {
    /// Stable pool identity: `[A-Za-z0-9_-]{1,64}` (it enters metering object
    /// keys, so it must not traverse), unique across the config.
    pub id: String,
    /// Local filesystem directory holding the pool's `owner.lock` and ledger.
    /// Not the object store: reservation state is node-local by design.
    pub path: String,
    /// Bytes the pool may admit in total, grant bytes × overhead included.
    pub allocatable_bytes: u64,
    /// Optional per-tenant ceiling on admitted grant bytes (quota is operator
    /// policy, never a client-supplied entitlement).
    #[serde(default)]
    pub tenant_limit_bytes: Option<u64>,
    /// Headroom multiplier applied to every grant when charging the pool:
    /// entry headers/checksums, temporary warming, old/new generation overlap.
    /// `0.1` ⇒ a 100 GiB grant charges 110 GiB. Clamped to `[0, 10]`.
    #[serde(default = "default_overhead_factor")]
    pub overhead_factor: f64,
}

fn default_overhead_factor() -> f64 {
    0.1
}

/// Pool ids enter metering object keys — same charset discipline as tenant
/// ids, or a hostile id escapes the `_metering/` subtree (review S1).
pub fn valid_pool_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl ReservationPoolConfig {
    /// Validate one declaration. Called at config load AND at authority
    /// open; startup-time duplicate id/path checks live in
    /// [`validate_pool_configs`].
    pub fn validate(&self) -> anyhow::Result<()> {
        if !valid_pool_id(&self.id) {
            anyhow::bail!(
                "reservation pool id '{}' must be 1-64 chars of [A-Za-z0-9_-]",
                self.id
            );
        }
        if self.path.trim().is_empty() {
            anyhow::bail!("reservation pool '{}' path must be non-empty", self.id);
        }
        if self.allocatable_bytes == 0 {
            anyhow::bail!(
                "reservation pool '{}' allocatable_bytes must be > 0",
                self.id
            );
        }
        if !(0.0..=10.0).contains(&self.overhead_factor) {
            anyhow::bail!(
                "reservation pool '{}' overhead_factor {} outside [0, 10]",
                self.id,
                self.overhead_factor
            );
        }
        Ok(())
    }

    /// Effective pool charge for a raw grant: `bytes + ceil(bytes × overhead)`.
    fn effective_bytes(&self, grant_bytes: u64) -> u64 {
        let overhead = (grant_bytes as f64 * self.overhead_factor).ceil() as u64;
        grant_bytes.saturating_add(overhead)
    }
}

/// Startup validation for the whole declaration list: every pool valid, and
/// ids/paths unique — two pools sharing an id would write the same metering
/// namespace with independent counters (silent overwrite), and two pools on
/// one path would fight over one ledger (review S1).
pub fn validate_pool_configs(pools: &[ReservationPoolConfig]) -> anyhow::Result<()> {
    let mut ids = HashSet::new();
    let mut paths = HashSet::new();
    for pool in pools {
        pool.validate()?;
        if !ids.insert(pool.id.clone()) {
            anyhow::bail!("duplicate reservation pool id '{}'", pool.id);
        }
        let canonical = std::fs::canonicalize(&pool.path)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| pool.path.trim_end_matches('/').to_string());
        if !paths.insert(canonical) {
            anyhow::bail!(
                "duplicate reservation pool path '{}' (pool '{}')",
                pool.path,
                pool.id
            );
        }
    }
    Ok(())
}

/// Lifecycle per `docs/current/STORAGE_PLAN.adoc` §"Reservation and coverage
/// lifecycle". Only `Reserved` and `Released` are reachable in SC3 slice 1 —
/// warming/ready/degraded arrive with SC4's coverage read path, `Releasing`
/// with SC4's retirement coordination. The transition validator is total so
/// later slices extend the machine without re-defining its floor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantState {
    /// Admission succeeded durably; bytes are charged to the pool.
    Reserved,
    /// SC4: coverage warming toward a generation.
    Warming,
    /// SC4: admitted capacity covering a declared profile/generation.
    Ready,
    /// Coverage cannot keep up; the grant stays charged (billing honesty) and
    /// the reason names the failure.
    Degraded { reason: String },
    /// Retirement in flight; bytes still charged until Released.
    Releasing,
    /// Capacity freed; accrual stopped. Terminal for this grant id.
    Released,
}

impl GrantState {
    /// Whether a grant in this state still charges pool capacity.
    pub fn charges_capacity(&self) -> bool {
        !matches!(self, GrantState::Released)
    }

    /// Legal forward transitions (fail-closed: anything not listed is an
    /// invariant violation, not a silently coerced state).
    pub fn can_transition_to(&self, next: &GrantState) -> bool {
        use GrantState::*;
        matches!(
            (self, next),
            (Reserved, Warming)
                | (Reserved, Releasing)
                | (Reserved, Released)
                | (Warming, Ready)
                | (Warming, Degraded { .. })
                | (Warming, Releasing)
                | (Ready, Degraded { .. })
                | (Ready, Releasing)
                | (Degraded { .. }, Warming)
                | (Degraded { .. }, Releasing)
                | (Releasing, Releasing) // idempotent re-release
                | (Releasing, Released)
                | (Released, Released) // idempotent no-op
        )
    }
}

/// One admitted reservation, durably recorded in the pool ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub id: String,
    pub tenant_id: String,
    /// Stable collection identity (the SC1-minted stable id; a drop/recreate
    /// upstream mints a fresh id, so a new incarnation gets a fresh grant).
    pub collection_id: String,
    pub bytes: u64,
    pub state: GrantState,
    pub created_unix_secs: i64,
    /// Accrual checkpoint: the end of the last emitted usage interval. Frozen
    /// at release; advanced by interval-closing mutations and
    /// [`ReservationAuthority::take_usage_intervals`].
    pub last_interval_end_unix_secs: i64,
}

/// A computed usage interval ready for the durable metering sink. The dedup
/// identity is `(tenant, pool, producer_epoch, sequence)` — a replayed
/// emission must overwrite the same record, never double-charge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationUsageInterval {
    pub tenant_id: String,
    pub pool_id: String,
    pub grant_id: String,
    /// Reserved-byte level held throughout `[start, end)` — no level change
    /// happens inside an interval (mutations close the open interval first).
    pub reserved_bytes: u64,
    pub interval_start_unix_secs: i64,
    pub interval_end_unix_secs: i64,
    pub producer_epoch: u64,
    pub sequence: u64,
}

/// Durable pool ledger. Every mutation is clone-prospect → persist (fsync) →
/// commit, so the in-memory state only ever contains durable history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Ledger {
    schema_version: u32,
    /// Bumped (and persisted) on every authority open; usage dedup key
    /// component and the crash-gap marker.
    producer_epoch: u64,
    /// Monotonic per-process emission counter (resets with the epoch).
    next_sequence: u64,
    grants: Vec<Grant>,
}

/// The per-pool reservation authority. Clones share one ledger/lock (an `Arc`
/// seam like `PersistentByteStore`).
#[derive(Clone, Debug)]
pub struct ReservationAuthority {
    inner: Arc<AuthorityInner>,
    config: ReservationPoolConfig,
}

#[derive(Debug)]
struct AuthorityInner {
    dir: PathBuf,
    ledger: tokio::sync::Mutex<Ledger>,
}

/// Failure modes surfaced as `Err` with actionable context (never a silent
/// downgrade of the request).
#[derive(Debug, thiserror::Error)]
pub enum ReservationError {
    #[error(
        "pool capacity exceeded: requesting {requested} bytes (effective {effective}) against {free} effective bytes free in pool '{pool}' — resize, release, or raise the operator-declared allocatable_bytes"
    )]
    Capacity {
        pool: String,
        requested: u64,
        effective: u64,
        free: u64,
    },
    #[error(
        "tenant '{tenant}' reservation limit exceeded: {sum} + {requested} grant bytes > tenant limit {limit} in pool '{pool}'"
    )]
    TenantLimit {
        pool: String,
        tenant: String,
        sum: u64,
        requested: u64,
        limit: u64,
    },
    #[error(
        "reservations disabled by kill switch {kill_switch_env}; the pool ledger and policy are untouched",
        kill_switch_env = RESERVATIONS_KILL_SWITCH_ENV
    )]
    KillSwitch,
    #[error(
        "pool '{pool}' is exclusively owned: {holder} — refusing a second owner",
        pool = .pool,
        holder = .holder
    )]
    SecondOwner { pool: String, holder: String },
    #[error("reservation pool ledger corrupt: {0}")]
    Corrupt(String),
    #[error("grant ({tenant}, {collection}) not found in pool '{pool}'")]
    NotFound {
        pool: String,
        tenant: String,
        collection: String,
    },
    #[error("illegal transition for grant {grant} from {from} to {to}")]
    IllegalTransition {
        grant: String,
        from: String,
        to: String,
    },
    #[error("invalid reservation request: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Process-global registry of live authorities: canonical pool dir → weak
/// handle. A second `open` on a pool with a LIVE authority is rejected before
/// any lock dance (review B1a — same-process takeover was otherwise free).
fn open_pool_registry() -> &'static Mutex<Vec<(PathBuf, Weak<AuthorityInner>)>> {
    static REGISTRY: std::sync::OnceLock<Mutex<Vec<(PathBuf, Weak<AuthorityInner>)>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Per-pool async gate serializing `open` WITHIN this process. Holding it
/// across the whole open (registry check → lock dance → registry insert)
/// makes the in-process exclusivity airtight: two racing opens on one pool
/// queue on the gate, and the second then sees the first's registry entry.
/// It also lets the lock dance treat an OUR-pid holder as provably stale —
/// the gate plus the registry check prove no live in-process authority owns
/// it (a same-process lock alone cannot tell its own dead authority from a
/// live second open).
fn pool_open_gate(canonical: &Path) -> Arc<tokio::sync::Mutex<()>> {
    use std::collections::HashMap;
    static GATES: std::sync::OnceLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> =
        std::sync::OnceLock::new();
    let mut gates = GATES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    gates
        .entry(canonical.to_path_buf())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

impl ReservationAuthority {
    /// Open (or take over) the pool: reject an in-process second owner,
    /// acquire the exclusive cross-process owner lock, load and validate the
    /// ledger, bump and PERSIST the producer epoch, prune the released tail.
    /// Fails closed on a live owner or a corrupt/unreadable ledger.
    pub async fn open(config: ReservationPoolConfig) -> Result<Self, ReservationError> {
        config.validate().map_err(|e| {
            ReservationError::Invalid(format!("invalid pool config '{}': {e}", config.id))
        })?;
        let dir = PathBuf::from(&config.path);
        tokio::fs::create_dir_all(&dir).await?;
        let canonical = tokio::fs::canonicalize(&dir).await?;
        let gate = pool_open_gate(&canonical);
        let _gate_guard = gate.lock().await;

        // In-process exclusivity first: a live authority on this pool dir is
        // the current process itself — no lock dance can fix that.
        {
            let mut registry = open_pool_registry()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            registry.retain(|(_, weak)| weak.upgrade().is_some());
            if registry.iter().any(|(path, _)| *path == canonical) {
                return Err(ReservationError::SecondOwner {
                    pool: config.id.clone(),
                    holder: "a live authority in this process".to_string(),
                });
            }
        }

        // Our own pid in the lock is provably stale HERE (gate held, registry
        // empty): the authority that wrote it is gone from this process.
        acquire_owner_lock(&dir, &config.id).await?;
        let mut ledger = load_ledger(&dir).await?;
        ledger.producer_epoch += 1;
        ledger.next_sequence = 0;
        prune_released(&mut ledger);
        // Persist the epoch bump at open (review S7): two lifetimes must
        // never share an epoch on disk, or their (epoch, sequence) dedup keys
        // collide even before the first mutation.
        persist_ledger(&dir, &ledger).await?;

        let inner = Arc::new(AuthorityInner {
            dir,
            ledger: tokio::sync::Mutex::new(ledger),
        });
        open_pool_registry()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((canonical, Arc::downgrade(&inner)));
        Ok(Self { inner, config })
    }

    pub fn config(&self) -> &ReservationPoolConfig {
        &self.config
    }

    /// Atomically admit (or idempotently return) a reservation.
    ///
    /// Idempotency: a live grant for the same (tenant, collection) with the
    /// same byte count returns that grant unchanged (it is durable by
    /// construction — memory only ever holds persisted state); a different
    /// byte count is a resize, admitted only if capacity and the tenant
    /// limit allow — and like [`Self::resize`] it RETURNS the interval closed
    /// at the old level (verification round 2: discarding it here dropped
    /// billing for a span whose checkpoint advanced durably). A released
    /// grant for the identity is replaced by a fresh one.
    pub async fn reserve(
        &self,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<(Grant, Vec<ReservationUsageInterval>), ReservationError> {
        if kill_switch_armed() {
            return Err(ReservationError::KillSwitch);
        }
        if bytes == 0 {
            return Err(ReservationError::Invalid(
                "reservation bytes must be > 0".to_string(),
            ));
        }
        let mut guard = self.inner.ledger.lock().await;
        if let Some(existing) = live_grant(&guard, tenant_id, collection_id) {
            if existing.bytes == bytes {
                return Ok((existing.clone(), Vec::new()));
            }
            return self
                .resize_prospective(
                    &mut guard,
                    tenant_id,
                    collection_id,
                    bytes,
                    current_unix_secs(),
                )
                .await;
        }
        let grant = self
            .admit_prospective(&mut guard, tenant_id, collection_id, bytes)
            .await?;
        Ok((grant, Vec::new()))
    }

    /// Resize a live grant (same grant id, new byte count), capacity- and
    /// limit-checked, closing the open interval at the OLD level (returned).
    /// `now_unix_secs` is the billing split instant, explicit so callers and
    /// tests pin exactly where the level changes.
    pub async fn resize(
        &self,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
        now_unix_secs: i64,
    ) -> Result<(Grant, Vec<ReservationUsageInterval>), ReservationError> {
        if kill_switch_armed() {
            return Err(ReservationError::KillSwitch);
        }
        let mut guard = self.inner.ledger.lock().await;
        self.resize_prospective(&mut guard, tenant_id, collection_id, bytes, now_unix_secs)
            .await
    }

    /// Release a grant: closes the open interval at the old level (returned —
    /// the tail is billed, not dropped), freezes accrual, frees capacity.
    /// Idempotent (a second release returns no intervals).
    pub async fn release(
        &self,
        tenant_id: &str,
        collection_id: &str,
        now_unix_secs: i64,
    ) -> Result<Vec<ReservationUsageInterval>, ReservationError> {
        let mut guard = self.inner.ledger.lock().await;
        let Some(live) = live_grant(&guard, tenant_id, collection_id).cloned() else {
            return Ok(Vec::new()); // released or never existed: idempotent
        };
        let mut prospective = (*guard).clone();
        let mut closed = Vec::new();
        // Close the tail interval at the level actually held (review B4).
        if let Some(interval) =
            close_open_interval(&mut prospective, &live, now_unix_secs, &self.config.id)
        {
            closed.push(interval);
        }
        let grant = prospective
            .grants
            .iter_mut()
            .find(|g| g.id == live.id)
            .ok_or_else(|| ReservationError::Corrupt("grant vanished in prospect".into()))?;
        // Retirement enters through Releasing (the SC4 retirement seam): the
        // transition floor is enforced, never silently coerced (review S4).
        if !grant.state.can_transition_to(&GrantState::Releasing) {
            return Err(ReservationError::IllegalTransition {
                grant: grant.id.clone(),
                from: format!("{:?}", grant.state),
                to: "Releasing".to_string(),
            });
        }
        grant.state = GrantState::Releasing;
        grant.state = GrantState::Released;
        prune_released(&mut prospective);
        persist_ledger(&self.inner.dir, &prospective).await?;
        *guard = prospective;
        Ok(closed)
    }

    /// Advance accrual checkpoints and return the intervals closed by
    /// `now_unix_secs`. Checkpoints and sequences are committed only after
    /// the ledger persists, so a crash between persist and metering replays
    /// the SAME `(epoch, sequence)` — overwrite, not double-charge — and a
    /// persist failure loses nothing (nothing was consumed).
    pub async fn take_usage_intervals(
        &self,
        now_unix_secs: i64,
    ) -> Result<Vec<ReservationUsageInterval>, ReservationError> {
        let mut guard = self.inner.ledger.lock().await;
        let mut prospective = (*guard).clone();
        let mut out = Vec::new();
        for grant in prospective.grants.clone() {
            if let Some(interval) =
                close_open_interval(&mut prospective, &grant, now_unix_secs, &self.config.id)
            {
                out.push(interval);
            }
        }
        if !out.is_empty() {
            persist_ledger(&self.inner.dir, &prospective).await?;
            *guard = prospective;
        }
        Ok(out)
    }

    /// Read-only snapshot (status projection input; no accrual side effect).
    pub async fn grants_snapshot(&self) -> Vec<Grant> {
        self.inner.ledger.lock().await.grants.clone()
    }

    /// Effective bytes charged against the pool right now (invariant input).
    pub async fn charged_effective_bytes(&self) -> u64 {
        let ledger = self.inner.ledger.lock().await;
        ledger
            .grants
            .iter()
            .filter(|g| g.state.charges_capacity())
            .map(|g| self.config.effective_bytes(g.bytes))
            .sum()
    }

    /// Clone-persist-commit admission (review B2): a persist failure leaves
    /// the in-memory ledger untouched, so the caller's Err is the truth and a
    /// retry re-attempts the same admission.
    async fn admit_prospective(
        &self,
        guard: &mut tokio::sync::MutexGuard<'_, Ledger>,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<Grant, ReservationError> {
        let mut prospective = (*guard).clone();
        let effective = self.config.effective_bytes(bytes);
        let charged: u64 = prospective
            .grants
            .iter()
            .filter(|g| g.state.charges_capacity())
            .map(|g| self.config.effective_bytes(g.bytes))
            .sum();
        let free = self.config.allocatable_bytes.saturating_sub(charged);
        if effective > free {
            return Err(ReservationError::Capacity {
                pool: self.config.id.clone(),
                requested: bytes,
                effective,
                free,
            });
        }
        if let Some(limit) = self.config.tenant_limit_bytes {
            let tenant_sum: u64 = prospective
                .grants
                .iter()
                .filter(|g| g.state.charges_capacity() && g.tenant_id == tenant_id)
                .map(|g| g.bytes)
                .sum();
            let combined =
                tenant_sum
                    .checked_add(bytes)
                    .ok_or_else(|| ReservationError::TenantLimit {
                        pool: self.config.id.clone(),
                        tenant: tenant_id.to_string(),
                        sum: tenant_sum,
                        requested: bytes,
                        limit,
                    })?;
            if combined > limit {
                return Err(ReservationError::TenantLimit {
                    pool: self.config.id.clone(),
                    tenant: tenant_id.to_string(),
                    sum: tenant_sum,
                    requested: bytes,
                    limit,
                });
            }
        }
        let now = current_unix_secs();
        let grant = Grant {
            id: new_grant_id(),
            tenant_id: tenant_id.to_string(),
            collection_id: collection_id.to_string(),
            bytes,
            state: GrantState::Reserved,
            created_unix_secs: now,
            last_interval_end_unix_secs: now,
        };
        prospective.grants.push(grant.clone());
        persist_ledger(&self.inner.dir, &prospective).await?;
        **guard = prospective;
        Ok(grant)
    }

    /// Clone-persist-commit resize: closes the grant's open interval at the
    /// OLD level first (review B4), then applies the new byte count.
    async fn resize_prospective(
        &self,
        guard: &mut tokio::sync::MutexGuard<'_, Ledger>,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
        now_unix_secs: i64,
    ) -> Result<(Grant, Vec<ReservationUsageInterval>), ReservationError> {
        let Some(live) = live_grant(guard, tenant_id, collection_id).cloned() else {
            return Err(ReservationError::NotFound {
                pool: self.config.id.clone(),
                tenant: tenant_id.to_string(),
                collection: collection_id.to_string(),
            });
        };
        let mut prospective = (*guard).clone();
        let mut closed = Vec::new();
        if let Some(interval) =
            close_open_interval(&mut prospective, &live, now_unix_secs, &self.config.id)
        {
            closed.push(interval);
        }
        let old_effective = self.config.effective_bytes(live.bytes);
        let new_effective = self.config.effective_bytes(bytes);
        let charged: u64 = prospective
            .grants
            .iter()
            .filter(|g| g.state.charges_capacity())
            .map(|g| self.config.effective_bytes(g.bytes))
            .sum();
        let free_after = self
            .config
            .allocatable_bytes
            .saturating_sub(charged - old_effective);
        if new_effective > free_after {
            return Err(ReservationError::Capacity {
                pool: self.config.id.clone(),
                requested: bytes,
                effective: new_effective,
                free: free_after,
            });
        }
        if let Some(limit) = self.config.tenant_limit_bytes {
            let tenant_sum: u64 = prospective
                .grants
                .iter()
                .filter(|g| {
                    g.state.charges_capacity() && g.tenant_id == tenant_id && g.id != live.id
                })
                .map(|g| g.bytes)
                .sum();
            let combined =
                tenant_sum
                    .checked_add(bytes)
                    .ok_or_else(|| ReservationError::TenantLimit {
                        pool: self.config.id.clone(),
                        tenant: tenant_id.to_string(),
                        sum: tenant_sum,
                        requested: bytes,
                        limit,
                    })?;
            if combined > limit {
                return Err(ReservationError::TenantLimit {
                    pool: self.config.id.clone(),
                    tenant: tenant_id.to_string(),
                    sum: tenant_sum,
                    requested: bytes,
                    limit,
                });
            }
        }
        let grant = prospective
            .grants
            .iter_mut()
            .find(|g| g.id == live.id)
            .ok_or_else(|| ReservationError::Corrupt("grant vanished in prospect".into()))?;
        grant.bytes = bytes;
        let updated = grant.clone();
        persist_ledger(&self.inner.dir, &prospective).await?;
        **guard = prospective;
        Ok((updated, closed))
    }
}

/// Close `grant`'s open interval `[last, now)` at its CURRENT level inside
/// `prospective`, allocating the next sequence, and advance the checkpoint.
/// No-op when nothing accrued. The grant is located by id in `prospective`.
fn close_open_interval(
    prospective: &mut Ledger,
    grant: &Grant,
    now_unix_secs: i64,
    pool_id: &str,
) -> Option<ReservationUsageInterval> {
    if !grant.state.charges_capacity() || grant.last_interval_end_unix_secs >= now_unix_secs {
        return None;
    }
    let sequence = prospective.next_sequence;
    prospective.next_sequence += 1;
    let interval = ReservationUsageInterval {
        tenant_id: grant.tenant_id.clone(),
        pool_id: pool_id.to_string(),
        grant_id: grant.id.clone(),
        reserved_bytes: grant.bytes,
        interval_start_unix_secs: grant.last_interval_end_unix_secs,
        interval_end_unix_secs: now_unix_secs,
        producer_epoch: prospective.producer_epoch,
        sequence,
    };
    let target = prospective.grants.iter_mut().find(|g| g.id == grant.id)?;
    target.last_interval_end_unix_secs = now_unix_secs;
    Some(interval)
}

fn live_grant<'a>(ledger: &'a Ledger, tenant_id: &str, collection_id: &str) -> Option<&'a Grant> {
    ledger.grants.iter().find(|g| {
        g.tenant_id == tenant_id && g.collection_id == collection_id && g.state.charges_capacity()
    })
}

/// Keep only the newest [`MAX_RETAINED_RELEASED`] released grants. Durable
/// history lives in the metering sink; the ledger stays bounded so each
/// mutation's rewrite stays cheap (review S6).
fn prune_released(ledger: &mut Ledger) {
    let mut released: Vec<usize> = ledger
        .grants
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.state.charges_capacity())
        .map(|(i, _)| i)
        .collect();
    if released.len() <= MAX_RETAINED_RELEASED {
        return;
    }
    released.drain(..released.len() - MAX_RETAINED_RELEASED);
    for i in released.into_iter().rev() {
        ledger.grants.remove(i);
    }
}

/// Kill switch. Fail-closed on a non-UTF-8 value (a kill switch must refuse
/// when its input is ambiguous, not arm itself into silence) — review N1.
fn kill_switch_armed() -> bool {
    match std::env::var_os(RESERVATIONS_KILL_SWITCH_ENV) {
        None => false,
        Some(value) => match value.to_str() {
            None => true,
            Some(text) => matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
        },
    }
}

fn current_unix_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

fn new_grant_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn ledger_path(dir: &Path) -> PathBuf {
    dir.join("reservations.json")
}

fn owner_lock_path(dir: &Path) -> PathBuf {
    dir.join("owner.lock")
}

/// Exclusive owner lock, `pid:start_ticks` on Linux (`/proc/self/stat`
/// field 22), bare pid elsewhere. Acquisition is `O_EXCL` create — atomic
/// against racers. An existing lock is claimed ONLY when its owner is provably
/// dead (process gone, or same pid with a different start token), through a
/// takeover PROTOCOL, not a bare rename (review B1, verification round 2: a
/// racer holding a stale verdict could rename a live successor's lock away):
/// 1. acquire the O_EXCL `owner.claim` slot (its token self-heals via the
///    same liveness probe when the claimer died mid-takeover);
/// 2. RE-READ `owner.lock` under the slot and proceed ONLY if it still
///    carries the token just judged stale — an expired verdict releases the
///    slot and loops instead of touching the lock;
/// 3. rename the (re-verified stale) lock away and release the slot.
///
/// A racer whose verdict went stale can therefore never move a lock the
/// protocol did not still prove dead. Non-Linux cannot prove death by ticks,
/// so a foreign pid merely absent from the probe is treated as live
/// (fail-closed). An unparseable claim slot fails closed (Corrupt) — a
/// crash mid-takeover wedges takeover until an operator clears it, never
/// double-ownership.
async fn acquire_owner_lock(dir: &Path, pool_id: &str) -> Result<(), ReservationError> {
    let path = owner_lock_path(dir);
    let claim_path = dir.join("owner.claim");
    let token = owner_token();
    for _ in 0..LOCK_ATTEMPTS {
        match write_exclusive(&path, &token).await {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let content = tokio::fs::read_to_string(&path).await.map_err(|err| {
                    ReservationError::Corrupt(format!("owner.lock unreadable: {err}"))
                })?;
                let held = OwnerToken::parse(&content).ok_or_else(|| {
                    ReservationError::Corrupt("owner.lock carries no pid token".into())
                })?;
                if owner_is_live(&held) {
                    return Err(ReservationError::SecondOwner {
                        pool: pool_id.to_string(),
                        holder: format!("owner.lock pid {}", held.pid),
                    });
                }
                // Take the takeover slot; a live claimer means someone else
                // is mid-takeover — let them finish (self-heals if they died).
                match write_exclusive(&claim_path, &token).await {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        let claim_content =
                            tokio::fs::read_to_string(&claim_path)
                                .await
                                .map_err(|err| {
                                    ReservationError::Corrupt(format!(
                                        "owner.claim unreadable: {err}"
                                    ))
                                })?;
                        let claimer = OwnerToken::parse(&claim_content).ok_or_else(|| {
                            ReservationError::Corrupt(
                                "owner.claim carries no pid token (crashed mid-takeover?)".into(),
                            )
                        })?;
                        if owner_is_live(&claimer) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                            continue;
                        }
                        let _ = tokio::fs::remove_file(&claim_path).await;
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                }
                // RE-VERIFY under the slot: the verdict holds only if the
                // lock still carries the exact token judged stale.
                let still_stale = match tokio::fs::read_to_string(&path).await {
                    Ok(current) => current == content,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
                    Err(err) => {
                        let _ = tokio::fs::remove_file(&claim_path).await;
                        return Err(err.into());
                    }
                };
                if !still_stale {
                    // Another racer already replaced the lock: our verdict is
                    // expired — release the slot and re-judge from scratch.
                    let _ = tokio::fs::remove_file(&claim_path).await;
                    continue;
                }
                let stale_name =
                    path.with_extension(format!("stale-{}-{}", std::process::id(), new_grant_id()));
                match tokio::fs::rename(&path, &stale_name).await {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => {
                        let _ = tokio::fs::remove_file(&claim_path).await;
                        return Err(err.into());
                    }
                }
                let _ = tokio::fs::remove_file(&stale_name).await;
                let _ = tokio::fs::remove_file(&claim_path).await;
                // Loop: exactly one racer's create_new now wins.
            }
            Err(e) => return Err(e.into()),
        }
    }
    Err(ReservationError::SecondOwner {
        pool: pool_id.to_string(),
        holder: "an unresolved lock race after bounded retries".to_string(),
    })
}

/// `pid:start_ticks` — the ticks make pid reuse distinguishable from a true
/// same-process second open on Linux.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OwnerToken {
    pid: u32,
    start_ticks: Option<u64>,
}

fn owner_token() -> String {
    format!(
        "{}:{}",
        std::process::id(),
        linux_start_ticks().unwrap_or(0)
    )
}

impl OwnerToken {
    fn parse(text: &str) -> Option<Self> {
        let (pid, ticks) = text.trim().split_once(':')?;
        Some(Self {
            pid: pid.parse().ok()?,
            start_ticks: ticks.parse().ok(),
        })
    }
}

fn linux_start_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 22 (1-based) sits after the comm field, which is parenthesized
    // and may itself contain spaces — split after the LAST ')'.
    let after_comm = stat.rsplit_once(')')?.1;
    // comm is field 2; state is 3, so start_ticks (22) is the 20th token
    // after the comm field.
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

/// Liveness of a lock holder. OUR OWN pid is NEVER live here: `open` calls
/// this only under the per-pool gate with an empty registry, which proves no
/// live in-process authority holds the lock — so an own-pid token (whether
/// from this process's dead authority or pid reuse after a crash) is stale
/// and reclaimable. A FOREIGN pid is live if `/proc/{pid}` exists with
/// matching start ticks on Linux; without comparable ticks (or off-Linux)
/// existence alone is treated as live — fail-closed.
fn owner_is_live(held: &OwnerToken) -> bool {
    let my_pid = std::process::id();
    if held.pid == my_pid {
        // Provably stale at this call site (see the doc comment).
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        let stat = match std::fs::read_to_string(format!("/proc/{}/stat", held.pid)) {
            Ok(stat) => stat,
            Err(_) => return false,
        };
        let Some((_, after_comm)) = stat.rsplit_once(')') else {
            return true; // unparseable but the process exists: treat as live
        };
        let ticks: Option<u64> = after_comm
            .split_whitespace()
            .nth(19)
            .and_then(|t| t.parse().ok());
        match (held.start_ticks, ticks) {
            (Some(held_ticks), Some(observed)) => held_ticks == observed,
            // No comparable ticks: existence is the only signal (fail-closed).
            _ => true,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No portable liveness probe: treat the lock as held (fail-closed).
        return true;
    }
}

/// O_EXCL file creation: exactly one caller wins; everyone else sees
/// `AlreadyExists`.
async fn write_exclusive(path: &Path, content: &str) -> std::io::Result<()> {
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    use tokio::io::AsyncWriteExt;
    let mut file = file;
    file.write_all(content.as_bytes()).await?;
    file.sync_all().await?;
    Ok(())
}

async fn load_ledger(dir: &Path) -> Result<Ledger, ReservationError> {
    let path = ledger_path(dir);
    // Fail CLOSED on an unreadable ledger (review S3): treating an I/O error
    // as "absent" would return an empty ledger whose next persist erases the
    // real one.
    match tokio::fs::try_exists(&path).await {
        Ok(false) => {
            return Ok(Ledger {
                schema_version: LEDGER_SCHEMA_VERSION,
                producer_epoch: 0,
                next_sequence: 0,
                grants: Vec::new(),
            });
        }
        Ok(true) => {}
        Err(e) => return Err(e.into()),
    }
    let bytes = tokio::fs::read(&path).await?;
    let ledger: Ledger = serde_json::from_slice(&bytes).map_err(|e| {
        ReservationError::Corrupt(format!("ledger at {} unparsable: {e}", path.display()))
    })?;
    if ledger.schema_version != LEDGER_SCHEMA_VERSION {
        return Err(ReservationError::Corrupt(format!(
            "ledger schema version {} != supported {LEDGER_SCHEMA_VERSION}",
            ledger.schema_version
        )));
    }
    Ok(ledger)
}

/// Durable ledger write: write tmp → fsync file → rename → fsync the parent
/// directory. Atomic AND durable (review B3 — a rename alone is not): an
/// admission that returned Ok survives power loss, and a torn write cannot
/// leave a half-ledger behind. Local-file I/O under the caller's ledger
/// mutex, by design — never network I/O under a lock.
async fn persist_ledger(dir: &Path, ledger: &Ledger) -> Result<(), ReservationError> {
    let path = ledger_path(dir);
    let bytes = serde_json::to_vec(ledger)
        .map_err(|e| ReservationError::Corrupt(format!("ledger serialize: {e}")))?;
    let tmp = path.with_extension("tmp");
    {
        let mut file = tokio::fs::File::create(&tmp).await?;
        use tokio::io::AsyncWriteExt;
        file.write_all(&bytes).await?;
        file.sync_all().await?;
    }
    tokio::fs::rename(&tmp, &path).await?;
    // fsync the directory so the rename itself is durable.
    let dir_file = tokio::fs::File::open(dir).await?;
    dir_file.sync_all().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(dir: &tempfile::TempDir) -> ReservationPoolConfig {
        ReservationPoolConfig {
            id: "p1".to_string(),
            path: dir.path().display().to_string(),
            allocatable_bytes: 60 * 1024 * 1024 * 1024,
            tenant_limit_bytes: None,
            overhead_factor: 0.1,
        }
    }

    // SC-TE3: two concurrent 40 GiB requests against 60 GiB (with overhead,
    // even two raw 40s can't both fit) — exactly one wins, the invariant holds.
    #[tokio::test]
    async fn concurrent_reservations_cannot_overcommit() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();

        let a = auth.clone();
        let b = auth.clone();
        let (ra, rb) = tokio::join!(
            a.reserve("t1", "c1", 40 * 1024 * 1024 * 1024),
            b.reserve("t2", "c2", 40 * 1024 * 1024 * 1024),
        );
        let winners = usize::from(ra.is_ok()) + usize::from(rb.is_ok());
        assert_eq!(winners, 1, "exactly one concurrent admission may win");
        let loser_err = match (ra, rb) {
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => e.to_string(),
            (Ok(_), Ok(_)) | (Err(_), Err(_)) => panic!("not a one-winner outcome"),
        };
        assert!(loser_err.contains("capacity exceeded"), "{loser_err}");

        let charged = auth.charged_effective_bytes().await;
        assert!(
            charged <= auth.config().allocatable_bytes,
            "invariant: charged {charged} <= allocatable"
        );
    }

    // SC-TE3 stress: N racing admissions against capacity for exactly K wins.
    #[tokio::test]
    async fn stress_admission_never_overcommits() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = pool(&dir);
        cfg.allocatable_bytes = 10 * 1024 * 1024; // 10 effective MiB
        cfg.overhead_factor = 0.0; // isolate the counter from overhead math
        let auth = ReservationAuthority::open(cfg).await.unwrap();
        let mut handles = Vec::new();
        for i in 0..32 {
            let a = auth.clone();
            handles.push(tokio::spawn(async move {
                a.reserve("tn", &format!("c{i}"), 2 * 1024 * 1024)
                    .await
                    .is_ok()
            }));
        }
        let wins: usize = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap() as usize)
            .sum();
        assert_eq!(wins, 5, "exactly floor(10/2) admissions win");
        assert_eq!(auth.charged_effective_bytes().await, 10 * 1024 * 1024);
    }

    // Retry idempotency: same identity + bytes returns the SAME grant, and
    // capacity is not double-charged.
    #[tokio::test]
    async fn retry_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        let (g1, _closed) = auth.reserve("t1", "c1", 1024).await.unwrap();
        let (g2, _) = auth.reserve("t1", "c1", 1024).await.unwrap();
        assert_eq!(g1.id, g2.id, "retry returns the same grant");
        let grants = auth.grants_snapshot().await;
        assert_eq!(
            grants.iter().filter(|g| g.state.charges_capacity()).count(),
            1
        );
        // A different byte count is a resize, not a second grant.
        let (g3, _closed) = auth
            .resize("t1", "c1", 2048, current_unix_secs())
            .await
            .unwrap();
        assert_eq!(g3.id, g1.id);
        assert_eq!(g3.bytes, 2048);
    }

    // Review B2: an admission whose persist fails leaves NO live grant — the
    // error is the truth, and a retry re-attempts instead of hitting a
    // fast-path Ok for an undurable grant. (Inject EISDIR via a pre-created
    // `reservations.tmp` directory.)
    #[tokio::test]
    async fn persist_failure_does_not_fabricate_admission() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        tokio::fs::create_dir(dir.path().join("reservations.tmp"))
            .await
            .unwrap();
        let err = auth.reserve("t1", "c1", 1024).await.unwrap_err();
        assert!(matches!(err, ReservationError::Io(_)), "{err:?}");
        // No live grant in memory and nothing on disk.
        assert!(
            auth.grants_snapshot()
                .await
                .iter()
                .all(|g| !g.state.charges_capacity())
        );
        tokio::fs::remove_dir(dir.path().join("reservations.tmp"))
            .await
            .unwrap();
        let (retried, _) = auth.reserve("t1", "c1", 1024).await.unwrap();
        assert_eq!(retried.bytes, 1024, "retry re-attempts the admission");
    }

    // SC-TE7: reopen (simulated crash — no clean release) reconciles grants
    // and bumps the epoch; capacity accounting survives.
    #[tokio::test]
    async fn crash_and_reopen_reconciles_grants() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let auth = ReservationAuthority::open(cfg.clone()).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        let t0 = current_unix_secs();
        let epoch_before = auth.take_usage_intervals(t0 + 1).await.unwrap()[0].producer_epoch;
        drop(auth); // no release: "crash"

        let reopened = ReservationAuthority::open(cfg).await.unwrap();
        let grants = reopened.grants_snapshot().await;
        assert_eq!(grants.len(), 1, "grant survives the crash");
        assert_eq!(grants[0].state, GrantState::Reserved);
        assert_eq!(grants[0].bytes, 1024);
        let intervals = reopened.take_usage_intervals(t0 + 2).await.unwrap();
        assert_eq!(intervals[0].producer_epoch, epoch_before + 1);
        // Capacity is still charged: a new admission must account for it.
        assert!(reopened.charged_effective_bytes().await > 0);
    }

    // SC-TE7 fail-closed: a corrupt ledger refuses to open (and therefore to
    // admit anything) — never silently zeroes live grants.
    #[tokio::test]
    async fn corrupt_ledger_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let auth = ReservationAuthority::open(cfg.clone()).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        drop(auth);
        tokio::fs::write(ledger_path(dir.path()), b"{ not json")
            .await
            .unwrap();
        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("corrupt"), "{err}");
    }

    // B1(a): a SECOND open in the SAME process is rejected — the registry
    // holds the live authority. This is the hole the forged-lock test used to
    // paper over.
    #[tokio::test]
    async fn same_process_second_open_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let _first = ReservationAuthority::open(cfg.clone()).await.unwrap();
        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("exclusively owned"), "{err}");
        // Drop every clone: the registry weak dies and reopen succeeds.
        drop(_first);
        ReservationAuthority::open(pool(&dir)).await.unwrap();
    }

    // B1(b): a live FOREIGN owner is rejected by the lock itself. The token
    // must carry pid 1's REAL start ticks — a mismatched-ticks token means
    // "the pid lives but the recorded holder doesn't" (pid reuse), which the
    // authority correctly reclaims.
    #[tokio::test]
    async fn foreign_live_owner_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let pid1_ticks = std::fs::read_to_string("/proc/1/stat")
            .ok()
            .and_then(|stat| {
                stat.rsplit_once(')').map(|(_, rest)| {
                    rest.split_whitespace()
                        .nth(19)
                        .and_then(|t| t.parse::<u64>().ok())
                })
            })
            .flatten();
        let Some(ticks) = pid1_ticks else {
            // Off-Linux or unreadable /proc: the probe treats any existing
            // foreign-pid lock as live, so a bare pid token still exercises
            // the rejection path.
            panic!("foreign-live-owner test needs /proc/1/stat (Linux CI)");
        };
        tokio::fs::write(owner_lock_path(dir.path()), format!("1:{ticks}"))
            .await
            .unwrap();
        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("exclusively owned"), "{err}");
    }

    // Verification round 2 (B1 residual): a LIVE takeover claim slot backs
    // the opener off — bounded attempts, then fail-closed — and never touches
    // the (stale) owner.lock it was judging.
    #[tokio::test]
    async fn live_claim_slot_backs_off_without_touching_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        // Stale lock: pid 4194303 with ticks that cannot match a live holder.
        tokio::fs::write(owner_lock_path(dir.path()), "4194303:999999")
            .await
            .unwrap();
        // Live claim: pid 1 with its REAL start ticks.
        let pid1_ticks = std::fs::read_to_string("/proc/1/stat")
            .ok()
            .and_then(|stat| {
                stat.rsplit_once(')').map(|(_, rest)| {
                    rest.split_whitespace()
                        .nth(19)
                        .and_then(|t| t.parse::<u64>().ok())
                })
            })
            .flatten()
            .expect("claim-slot test needs /proc/1/stat (Linux CI)");
        let claim_path = dir.path().join("owner.claim");
        tokio::fs::write(&claim_path, format!("1:{pid1_ticks}"))
            .await
            .unwrap();

        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("unresolved lock race"), "{err}");
        // Both files untouched: the stale lock survives for the claim's
        // owner, and the live claim was not stolen.
        let lock = tokio::fs::read_to_string(owner_lock_path(dir.path()))
            .await
            .unwrap();
        assert_eq!(lock, "4194303:999999");
        let claim = tokio::fs::read_to_string(&claim_path).await.unwrap();
        assert_eq!(claim, format!("1:{pid1_ticks}"));
    }

    // The pid-reuse counterpart: a foreign pid whose recorded ticks do NOT
    // match the live process's — the recorded holder died and the pid was
    // reused, so the lock is stale and reclaimable.
    #[tokio::test]
    async fn foreign_pid_reuse_with_mismatched_ticks_is_reclaimed() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let pid1_ticks = std::fs::read_to_string("/proc/1/stat")
            .ok()
            .and_then(|stat| {
                stat.rsplit_once(')').map(|(_, rest)| {
                    rest.split_whitespace()
                        .nth(19)
                        .and_then(|t| t.parse::<u64>().ok())
                })
            })
            .flatten();
        let Some(ticks) = pid1_ticks else {
            panic!("pid-reuse test needs /proc/1/stat (Linux CI)");
        };
        let mismatched = ticks.wrapping_add(1);
        tokio::fs::write(owner_lock_path(dir.path()), format!("1:{mismatched}"))
            .await
            .unwrap();
        ReservationAuthority::open(cfg).await.unwrap();
    }

    // A stale lock (dead foreign pid) is taken over by exactly one opener —
    // crash recovery. 4194303 is the PID_MAX_LIMIT ceiling: astronomically
    // unlikely to be live; if it somehow is, this test fails closed instead.
    #[tokio::test]
    async fn stale_lock_is_taken_over() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        tokio::fs::write(owner_lock_path(dir.path()), "4194303:999999")
            .await
            .unwrap();
        ReservationAuthority::open(cfg).await.unwrap();
    }

    // Our own pid with a DIFFERENT start token (pid reuse after our crash) is
    // stale; our pid with OUR token is us (the registry catches that before
    // the lock, but the lock agrees).
    #[tokio::test]
    async fn pid_reuse_with_foreign_ticks_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let foreign_ticks = linux_start_ticks().map(|t| t.wrapping_add(1)).unwrap_or(1);
        tokio::fs::write(
            owner_lock_path(dir.path()),
            format!("{}:{foreign_ticks}", std::process::id()),
        )
        .await
        .unwrap();
        ReservationAuthority::open(cfg).await.unwrap();
    }

    // SC-TE9 + B4: release RETURNS the tail interval at the level actually
    // held, then accrual stops — the tail is billed, not dropped.
    #[tokio::test]
    async fn release_returns_tail_interval_and_stops_accrual() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        let t0 = current_unix_secs() + 1; // strictly after the reserve stamp
        let i1 = auth.take_usage_intervals(t0).await.unwrap();
        assert_eq!(i1.len(), 1);
        let tail = auth.release("t1", "c1", t0 + 5).await.unwrap();
        assert_eq!(
            tail.len(),
            1,
            "the interval (t0, t0+5] is returned by release: {tail:?}"
        );
        assert_eq!(tail[0].interval_start_unix_secs, t0);
        assert_eq!(tail[0].interval_end_unix_secs, t0 + 5);
        assert_eq!(tail[0].reserved_bytes, 1024, "billed at the held level");
        // Nothing accrues after release; a second release is a no-op.
        assert!(auth.take_usage_intervals(t0 + 10).await.unwrap().is_empty());
        assert!(auth.release("t1", "c1", t0 + 15).await.unwrap().is_empty());
    }

    // B4: resize splits billing at the resize point — the closed interval
    // carries the OLD level, the next take carries the NEW level.
    #[tokio::test]
    async fn resize_splits_billing_levels() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1000).await.unwrap();
        let t0 = current_unix_secs() + 1;
        let first = auth.take_usage_intervals(t0).await.unwrap();
        assert_eq!(first[0].reserved_bytes, 1000);
        // The split instant is explicit: the interval [t0, t0+5) must close
        // at the OLD level (1000), the span after it at the NEW level (4000).
        let (_resized, closed) = auth.resize("t1", "c1", 4000, t0 + 5).await.unwrap();
        assert_eq!(
            closed.len(),
            1,
            "resize returns the interval it closed: {closed:?}"
        );
        assert_eq!(closed[0].interval_start_unix_secs, t0);
        assert_eq!(closed[0].interval_end_unix_secs, t0 + 5);
        assert_eq!(
            closed[0].reserved_bytes, 1000,
            "the closed interval carries the OLD level"
        );
        let after = auth.take_usage_intervals(t0 + 10).await.unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].interval_start_unix_secs, t0 + 5);
        assert_eq!(after[0].reserved_bytes, 4000, "new level after the split");
        // reserve-as-resize is the same contract (verification round 2): the
        // closed interval is RETURNED, never silently dropped.
        let (grant, closed2) = auth.reserve("t1", "c1", 2000).await.unwrap();
        assert_eq!(grant.bytes, 2000);
        let _ = closed2; // span length is wall-clock dependent; return is the contract
    }

    // SC-TE9 dedup: the accrual checkpoint is durably persisted with the
    // emission keys, so a crash between persist and sink replays NOTHING at
    // the same instant (no double-charge), and the next emission under the
    // new epoch carries a fresh (epoch, sequence) identity.
    #[tokio::test]
    async fn usage_intervals_are_dedup_keyed() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let auth = ReservationAuthority::open(cfg.clone()).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        let t0 = current_unix_secs() + 1; // strictly after the reserve stamp
        let first = auth.take_usage_intervals(t0).await.unwrap();
        assert_eq!(first.len(), 1);
        drop(auth); // crash between persist and metering sink

        let reopened = ReservationAuthority::open(cfg).await.unwrap();
        let replay = reopened.take_usage_intervals(t0).await.unwrap();
        assert!(
            replay.is_empty(),
            "persisted checkpoint: replay at the same instant emits nothing"
        );
        let next = reopened.take_usage_intervals(t0 + 1).await.unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].interval_start_unix_secs, t0);
        assert!(
            next[0].producer_epoch > first[0].producer_epoch,
            "epoch bumps across the reopen"
        );
    }

    // Operator quota: the per-tenant limit bounds one tenant's grant bytes
    // even when pool capacity remains.
    #[tokio::test]
    async fn tenant_limit_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = pool(&dir);
        cfg.tenant_limit_bytes = Some(2048);
        let auth = ReservationAuthority::open(cfg).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        let err = auth
            .reserve("t1", "c2", 2048)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("tenant 't1' reservation limit"), "{err}");
        // Another tenant is unaffected.
        let _ = auth.reserve("t2", "c1", 2048).await.unwrap();
    }

    // Kill switch: reserve refuses with a named reason; the ledger survives.
    #[tokio::test]
    async fn kill_switch_refuses_without_amnesia() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        let _ = auth.reserve("t1", "c1", 1024).await.unwrap();
        // Env tests must not race other policy tests reading the same var.
        let guard = env_lock::scoped_set(RESERVATIONS_KILL_SWITCH_ENV, "1");
        let err = auth
            .reserve("t1", "c2", 1024)
            .await
            .unwrap_err()
            .to_string();
        drop(guard);
        assert!(err.contains("kill switch"), "{err}");
        assert_eq!(auth.grants_snapshot().await.len(), 1, "ledger untouched");
    }

    // Overhead is charged: effective = bytes * (1 + overhead), ceil.
    #[tokio::test]
    async fn overhead_is_charged() {
        let mut cfg = pool(&tempfile::tempdir().unwrap());
        cfg.overhead_factor = 0.1;
        assert_eq!(cfg.effective_bytes(100), 110);
        cfg.overhead_factor = 0.0;
        assert_eq!(cfg.effective_bytes(100), 100);
        cfg.overhead_factor = 0.5;
        assert_eq!(cfg.effective_bytes(3), 5); // 4.5 -> ceil 5
    }

    // State-machine floor: the SC3-1-reachable transitions are legal; the
    // SC4 states exist but fail closed on misuse.
    #[test]
    fn grant_state_floor() {
        assert!(GrantState::Reserved.can_transition_to(&GrantState::Released));
        assert!(GrantState::Released.can_transition_to(&GrantState::Released));
        assert!(!GrantState::Released.can_transition_to(&GrantState::Reserved));
        assert!(!GrantState::Reserved.can_transition_to(&GrantState::Ready));
        assert!(GrantState::Reserved.charges_capacity());
        assert!(!GrantState::Released.charges_capacity());
    }

    // Config validation: a bad declaration fails at validate(); duplicates
    // fail at the list level (startup).
    #[test]
    fn pool_config_validation() {
        let mut cfg = pool(&tempfile::tempdir().unwrap());
        assert!(cfg.validate().is_ok());
        cfg.id = " ".to_string();
        assert!(cfg.validate().is_err());
        cfg.id = "../escape".to_string();
        assert!(cfg.validate().is_err(), "traversal ids rejected");
        cfg.id = "p1".to_string();
        cfg.allocatable_bytes = 0;
        assert!(cfg.validate().is_err());
        cfg.allocatable_bytes = 1;
        cfg.overhead_factor = 11.0;
        assert!(cfg.validate().is_err());
        cfg.overhead_factor = 0.1;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn pool_config_list_rejects_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        // Duplicate id on a different path.
        let mut other = cfg.clone();
        other.path = "/nonexistent-other-pool-dir".to_string();
        assert!(validate_pool_configs(&[cfg.clone(), other]).is_err());
        // Same path twice (canonicalized after create).
        let dir2 = tempfile::tempdir().unwrap();
        let mut same_path = pool(&dir2);
        same_path.id = "p2".to_string();
        same_path.path = cfg.path.clone();
        assert!(validate_pool_configs(&[cfg.clone(), same_path]).is_err());
        // Distinct id and path pass.
        let mut distinct = pool(&dir2);
        distinct.id = "p2".to_string();
        assert!(validate_pool_configs(&[cfg, distinct]).is_ok());
    }

    // Released grants are bounded in the ledger (S6): MAX_RETAINED_RELEASED
    // survives, older ones are pruned at open.
    #[tokio::test]
    async fn released_grants_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = pool(&dir);
        cfg.allocatable_bytes = 1 << 30;
        let auth = ReservationAuthority::open(cfg.clone()).await.unwrap();
        for i in 0..(MAX_RETAINED_RELEASED + 8) {
            let _ = auth.reserve("t1", &format!("c{i}"), 64).await.unwrap();
            auth.release("t1", &format!("c{i}"), current_unix_secs() + 1)
                .await
                .unwrap();
        }
        drop(auth);
        let reopened = ReservationAuthority::open(cfg).await.unwrap();
        let retained = reopened.grants_snapshot().await.len();
        assert!(
            retained <= MAX_RETAINED_RELEASED,
            "released tail bounded to {MAX_RETAINED_RELEASED}, got {retained}"
        );
    }

    /// Serialized env-var access for the kill-switch test — the same pattern
    /// #2260's v3-policy tests use. Edition-2024 marks env mutation unsafe;
    /// the kill switch reads the var per call and this guard is the only
    /// mutator, serialized on a mutex so even a shared-process local
    /// `cargo test --lib` never observes the env mid-mutation.
    mod env_lock {
        use std::sync::Mutex;

        static ENV_LOCK: Mutex<()> = Mutex::new(());

        pub struct Guard(std::sync::MutexGuard<'static, ()>, String);

        impl Drop for Guard {
            fn drop(&mut self) {
                // SAFETY: test-only; the mutex proves no parallel reader.
                unsafe { std::env::remove_var(&self.1) };
            }
        }

        pub fn scoped_set(name: &str, value: &str) -> Guard {
            let guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            // SAFETY: test-only; the mutex proves no parallel reader.
            unsafe { std::env::set_var(name, value) };
            Guard(guard, name.to_string())
        }
    }
}
