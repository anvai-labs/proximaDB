//! TD-TIER-1 SC3 slice 1: the per-pool reservation authority.
//!
//! One authority owns one operator-declared local pool EXCLUSIVELY in v1 (a
//! process-scoped lock file rejects a second owner while the first lives —
//! SC-TE8's serving-owner scoping is this lock, not a fleet allocator). It is
//! the single reservation ledger for that pool: admission is atomic under one
//! short mutex, durable BEFORE it returns (atomic tmp+rename ledger write), and
//! idempotent per (tenant, collection) — a retry returns the same grant.
//!
//! Capacity accounting charges effective bytes: `grant_bytes × (1 +
//! overhead_factor)`, rounded up — the plan's headers/checksums, warming and
//! old/new-generation headroom. The sum of live grants' effective bytes can
//! never exceed [`ReservationPoolConfig::allocatable_bytes`] (SC-TE3), and an
//! operator-declared per-tenant limit bounds one tenant's share.
//!
//! **No quota check ever runs on the query hot path.** The authority is an
//! admission- and cadence-time component only: reads never consult it, and
//! [`reserve`] performs no network I/O — its durable state is a small local
//! file, written under the ledger mutex and never across a network call.
//!
//! **Inert in this slice.** Nothing constructs an authority in production yet;
//! the config field parses and validates but no call site opens a pool. The
//! PERFORMANCE class keeps its SC1 `NOT_ACTIVE` status until SC4 proves
//! coverage; SC3-2 wires admission at create.
//!
//! **Crash semantics (SC-TE7/SC-TE9).** Reopen reloads the ledger and bumps
//! `producer_epoch`; usage intervals are keyed `(producer_epoch, sequence)` so
//! a replayed emission overwrites the same record rather than double-charging.
//! The epoch discontinuity after an unclean close is itself the uncertainty
//! signal — bytes held during the unobserved gap are not fabricated downstream.
//! A corrupt ledger fails closed: the authority refuses to open (and therefore
//! refuses admission) rather than silently zeroing live grants.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// On-disk ledger schema version. Bump on any incompatible change; the reader
/// rejects newer/unknown versions fail-closed (pre-GA: no mixed-read duty).
const LEDGER_SCHEMA_VERSION: u32 = 1;

/// Kill switch: when set, [`ReservationAuthority::reserve`] refuses new
/// reservations with a named reason. It never erases the ledger or the
/// operator's pool/policy configuration — degrading to balanced behavior, not
/// amnesia.
pub const RESERVATIONS_KILL_SWITCH_ENV: &str = "PROXIMADB_TIER1_RESERVATIONS_OFF";

/// Operator declaration of one local reservation pool (TD-TIER-1 SC3). An
/// empty pool list (the default) leaves the whole feature inert. No `Eq`:
/// `overhead_factor` is `f64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservationPoolConfig {
    /// Stable pool identity (unique across the config; path/validation key for
    /// usage records and later status projections).
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

impl ReservationPoolConfig {
    /// Validate a parsed pool: non-empty id/path, non-zero allocatable, clamped
    /// overhead. Called at config load so a bad declaration fails at startup,
    /// not at first admission.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.id.trim().is_empty() {
            anyhow::bail!("reservation pool id must be non-empty");
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

/// Lifecycle per `docs/current/STORAGE_PLAN.adoc` §"Reservation and coverage
/// lifecycle". Only `Reserved` and `Released` are reachable in SC3 slice 1 —
/// warming/ready/degraded arrive with SC4's coverage read path, `Releasing`
/// with SC4's retirement coordination. The transition validator below is total
/// so later slices extend the machine without re-defining its floor.
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
    /// SC4: retirement in flight; bytes still charged until Released.
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
    /// invariant violation, not a silent coerced state).
    pub fn can_transition_to(&self, next: &GrantState) -> bool {
        use GrantState::*;
        matches!(
            (self, next),
            (Reserved, Warming)
                | (Reserved, Released)
                | (Warming, Ready)
                | (Warming, Degraded { .. })
                | (Warming, Releasing)
                | (Ready, Degraded { .. })
                | (Ready, Releasing)
                | (Degraded { .. }, Warming)
                | (Degraded { .. }, Releasing)
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
    /// at release; advanced by [`ReservationAuthority::take_usage_intervals`].
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
    /// Reserved-byte level held throughout the interval (the level, not the
    /// integral — downstream integrates across intervals to byte-seconds).
    pub reserved_bytes: u64,
    pub interval_start_unix_secs: i64,
    pub interval_end_unix_secs: i64,
    pub producer_epoch: u64,
    pub sequence: u64,
}

/// Durable pool ledger: every mutation is written atomically (tmp + rename)
/// before the call returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Ledger {
    schema_version: u32,
    /// Bumped on every authority open; usage dedup key component.
    producer_epoch: u64,
    /// Monotonic per-process emission counter (resets with the epoch).
    next_sequence: u64,
    grants: Vec<Grant>,
}

/// The per-pool reservation authority. Clone shares one ledger/lock (an `Arc`
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
        "pool '{pool}' is exclusively owned by another live process (owner.lock pid {pid}); refusing a second owner"
    )]
    SecondOwner { pool: String, pid: u32 },
    #[error("reservation pool ledger corrupt: {0}")]
    Corrupt(String),
    #[error("grant ({tenant}, {collection}) not found in pool '{pool}'")]
    NotFound {
        pool: String,
        tenant: String,
        collection: String,
    },
    #[error("illegal transition {from:?} -> {to:?} for grant {grant}")]
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

impl ReservationAuthority {
    /// Open (or take over) the pool: acquire the exclusive owner lock, load
    /// and validate the ledger, bump the producer epoch. Fails closed on a
    /// live second owner or a corrupt ledger.
    pub async fn open(config: ReservationPoolConfig) -> Result<Self, ReservationError> {
        config.validate().map_err(|e| {
            ReservationError::Corrupt(format!("invalid pool config '{}': {e}", config.id))
        })?;
        let dir = PathBuf::from(&config.path);
        tokio::fs::create_dir_all(&dir).await?;
        acquire_owner_lock(&dir, &config.id).await?;
        let mut ledger = load_ledger(&dir).await?;
        ledger.producer_epoch += 1;
        ledger.next_sequence = 0;
        let inner = AuthorityInner {
            dir,
            ledger: tokio::sync::Mutex::new(ledger),
        };
        Ok(Self {
            inner: Arc::new(inner),
            config,
        })
    }

    pub fn config(&self) -> &ReservationPoolConfig {
        &self.config
    }

    /// Atomically admit (or idempotently return) a reservation.
    ///
    /// Idempotency: a live grant for the same (tenant, collection) with the
    /// same byte count returns that grant unchanged; a different byte count
    /// is a resize, admitted only if capacity and the tenant limit allow.
    /// A released grant for the identity is replaced by a fresh one.
    pub async fn reserve(
        &self,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<Grant, ReservationError> {
        if kill_switch_armed() {
            return Err(ReservationError::KillSwitch);
        }
        if bytes == 0 {
            return Err(ReservationError::Invalid(
                "reservation bytes must be > 0".to_string(),
            ));
        }
        let mut ledger = self.inner.ledger.lock().await;
        if let Some(existing) = live_grant(&ledger, tenant_id, collection_id) {
            if existing.bytes == bytes {
                return Ok(existing.clone());
            }
            return self
                .resize_locked(&mut ledger, tenant_id, collection_id, bytes)
                .await;
        }
        self.admit_locked(&mut ledger, tenant_id, collection_id, bytes)
            .await
    }

    /// Resize a live grant (same grant id, new byte count), capacity- and
    /// limit-checked. Shrinks always fit.
    pub async fn resize(
        &self,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<Grant, ReservationError> {
        if kill_switch_armed() {
            return Err(ReservationError::KillSwitch);
        }
        let mut ledger = self.inner.ledger.lock().await;
        self.resize_locked(&mut ledger, tenant_id, collection_id, bytes)
            .await
    }

    /// Release a grant: durably `Released`, accrual frozen (the final interval
    /// closes at release), capacity freed. Idempotent.
    pub async fn release(
        &self,
        tenant_id: &str,
        collection_id: &str,
        now_unix_secs: i64,
    ) -> Result<(), ReservationError> {
        let mut ledger = self.inner.ledger.lock().await;
        let Some(grant) = mutable_grant(&mut ledger, tenant_id, collection_id) else {
            return Ok(()); // released or never existed: idempotent no-op
        };
        grant.last_interval_end_unix_secs = grant.last_interval_end_unix_secs.max(now_unix_secs);
        grant.state = GrantState::Released;
        persist_ledger(&self.inner.dir, &ledger).await?;
        Ok(())
    }

    /// Advance accrual checkpoints and return the intervals closed by
    /// `now_unix_secs`. Each call persists the advanced checkpoints BEFORE
    /// returning, so a crash between this and the metering write replays the
    /// same `(epoch, sequence)` — overwrite, not double-charge.
    pub async fn take_usage_intervals(
        &self,
        now_unix_secs: i64,
    ) -> Result<Vec<ReservationUsageInterval>, ReservationError> {
        let mut guard = self.inner.ledger.lock().await;
        // Reborrow as a plain &mut so the per-grant loop and the sequence
        // counter are disjoint field borrows.
        let ledger: &mut Ledger = &mut guard;
        let epoch = ledger.producer_epoch;
        let mut out = Vec::new();
        for grant in &mut ledger.grants {
            let accrues =
                grant.state.charges_capacity() && grant.last_interval_end_unix_secs < now_unix_secs;
            if !accrues {
                continue;
            }
            let sequence = ledger.next_sequence;
            ledger.next_sequence += 1;
            out.push(ReservationUsageInterval {
                tenant_id: grant.tenant_id.clone(),
                pool_id: self.config.id.clone(),
                grant_id: grant.id.clone(),
                reserved_bytes: grant.bytes,
                interval_start_unix_secs: grant.last_interval_end_unix_secs,
                interval_end_unix_secs: now_unix_secs,
                producer_epoch: epoch,
                sequence,
            });
            grant.last_interval_end_unix_secs = now_unix_secs;
        }
        if !out.is_empty() {
            persist_ledger(&self.inner.dir, ledger).await?;
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

    async fn admit_locked(
        &self,
        ledger: &mut Ledger,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<Grant, ReservationError> {
        let effective = self.config.effective_bytes(bytes);
        let charged: u64 = ledger
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
            let tenant_sum: u64 = ledger
                .grants
                .iter()
                .filter(|g| g.state.charges_capacity() && g.tenant_id == tenant_id)
                .map(|g| g.bytes)
                .sum();
            if tenant_sum + bytes > limit {
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
        ledger.grants.push(grant.clone());
        persist_ledger(&self.inner.dir, ledger).await?;
        Ok(grant)
    }

    async fn resize_locked(
        &self,
        ledger: &mut Ledger,
        tenant_id: &str,
        collection_id: &str,
        bytes: u64,
    ) -> Result<Grant, ReservationError> {
        // Phase 1 (read): capture the live grant's identity; the mutable
        // borrow ends here so the capacity reads below are legal.
        let Some(grant_id) = live_grant(ledger, tenant_id, collection_id).map(|g| g.id.clone())
        else {
            return Err(ReservationError::NotFound {
                pool: self.config.id.clone(),
                tenant: tenant_id.to_string(),
                collection: collection_id.to_string(),
            });
        };
        let old_bytes = ledger
            .grants
            .iter()
            .find(|g| g.id == grant_id)
            .map(|g| g.bytes)
            .unwrap_or(0);
        // Phase 2 (checks): capacity net of the grant's own current charge,
        // plus the tenant limit excluding this grant.
        let old_effective = self.config.effective_bytes(old_bytes);
        let new_effective = self.config.effective_bytes(bytes);
        let charged: u64 = ledger
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
            let tenant_sum: u64 = ledger
                .grants
                .iter()
                .filter(|g| {
                    g.state.charges_capacity()
                        && g.tenant_id == tenant_id
                        && g.collection_id != collection_id
                })
                .map(|g| g.bytes)
                .sum();
            if tenant_sum + bytes > limit {
                return Err(ReservationError::TenantLimit {
                    pool: self.config.id.clone(),
                    tenant: tenant_id.to_string(),
                    sum: tenant_sum,
                    requested: bytes,
                    limit,
                });
            }
        }
        // Phase 3 (write): re-locate by id and commit.
        let Some(grant) = ledger
            .grants
            .iter_mut()
            .find(|g| g.id == grant_id && g.state.charges_capacity())
        else {
            return Err(ReservationError::NotFound {
                pool: self.config.id.clone(),
                tenant: tenant_id.to_string(),
                collection: collection_id.to_string(),
            });
        };
        grant.bytes = bytes;
        let updated = grant.clone();
        persist_ledger(&self.inner.dir, ledger).await?;
        Ok(updated)
    }
}

fn live_grant<'a>(ledger: &'a Ledger, tenant_id: &str, collection_id: &str) -> Option<&'a Grant> {
    ledger.grants.iter().find(|g| {
        g.tenant_id == tenant_id && g.collection_id == collection_id && g.state.charges_capacity()
    })
}

/// The LIVE (capacity-charging) grant for an identity, mutably. Historical
/// `Released` entries stay in the ledger for accrual/audit and are never
/// returned here — a released identity is "not found" to resize and an
/// idempotent no-op to release.
fn mutable_grant<'a>(
    ledger: &'a mut Ledger,
    tenant_id: &str,
    collection_id: &str,
) -> Option<&'a mut Grant> {
    let grant = ledger.grants.iter_mut().find(|g| {
        g.tenant_id == tenant_id && g.collection_id == collection_id && g.state.charges_capacity()
    })?;
    // Narrow the reborrow to the live grant only (state check above).
    Some(grant)
}

fn kill_switch_armed() -> bool {
    std::env::var_os(RESERVATIONS_KILL_SWITCH_ENV).is_some_and(|v| {
        matches!(
            v.to_str()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
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

/// Exclusive owner lock: `owner.lock` carries the owning pid. A live second
/// process is rejected; a dead owner's lock is taken over (crash recovery,
/// SC-TE7); liveness is checked through `/proc` (Linux), and on platforms
/// without it an existing lock is treated as live (fail-closed).
async fn acquire_owner_lock(dir: &Path, pool_id: &str) -> Result<u32, ReservationError> {
    let path = owner_lock_path(dir);
    match tokio::fs::try_exists(&path).await {
        Ok(false) => {}
        Ok(true) => {
            let content = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| ReservationError::Corrupt(format!("owner.lock unreadable: {e}")))?;
            let pid: u32 = content
                .trim()
                .parse()
                .map_err(|_| ReservationError::Corrupt("owner.lock carries no pid".into()))?;
            if pid != std::process::id() && process_is_alive(pid) {
                return Err(ReservationError::SecondOwner {
                    pool: pool_id.to_string(),
                    pid,
                });
            }
            // Stale (dead owner): fall through and take over.
        }
        Err(e) => return Err(e.into()),
    }
    let pid = std::process::id();
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, pid.to_string()).await?;
    tokio::fs::rename(&tmp, &path).await?;
    Ok(pid)
}

fn process_is_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No portable liveness probe: treat the lock as held (fail-closed).
        let _ = pid;
        true
    }
}

async fn load_ledger(dir: &Path) -> Result<Ledger, ReservationError> {
    let path = ledger_path(dir);
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(Ledger {
            schema_version: LEDGER_SCHEMA_VERSION,
            producer_epoch: 0,
            next_sequence: 0,
            grants: Vec::new(),
        });
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

async fn persist_ledger(dir: &Path, ledger: &Ledger) -> Result<(), ReservationError> {
    let path = ledger_path(dir);
    let bytes = serde_json::to_vec_pretty(ledger)
        .map_err(|e| ReservationError::Corrupt(format!("ledger serialize: {e}")))?;
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, &bytes).await?;
    tokio::fs::rename(&tmp, &path).await?;
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
        let g1 = auth.reserve("t1", "c1", 1024).await.unwrap();
        let g2 = auth.reserve("t1", "c1", 1024).await.unwrap();
        assert_eq!(g1.id, g2.id, "retry returns the same grant");
        let grants = auth.grants_snapshot().await;
        assert_eq!(
            grants.iter().filter(|g| g.state.charges_capacity()).count(),
            1
        );
        // A different byte count is a resize, not a second grant.
        let g3 = auth.reserve("t1", "c1", 2048).await.unwrap();
        assert_eq!(g3.id, g1.id);
        assert_eq!(g3.bytes, 2048);
    }

    // SC-TE7: reopen (simulated crash — no clean release) reconciles grants
    // and bumps the epoch; capacity accounting survives.
    #[tokio::test]
    async fn crash_and_reopen_reconciles_grants() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let auth = ReservationAuthority::open(cfg.clone()).await.unwrap();
        auth.reserve("t1", "c1", 1024).await.unwrap();
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
        auth.reserve("t1", "c1", 1024).await.unwrap();
        drop(auth);
        tokio::fs::write(ledger_path(dir.path()), b"{ not json")
            .await
            .unwrap();
        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("corrupt"), "{err}");
    }

    // Exclusive pool ownership: a live second process is rejected.
    #[tokio::test]
    async fn second_owner_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        let _first = ReservationAuthority::open(cfg.clone()).await.unwrap();
        // Same-process reopen would see its own pid; forge a foreign live pid
        // by writing the lock directly (pid 1 is alive on every Linux system).
        tokio::fs::write(owner_lock_path(dir.path()), "1")
            .await
            .unwrap();
        let err = ReservationAuthority::open(cfg).await.unwrap_err();
        assert!(err.to_string().contains("exclusively owned"), "{err}");
    }

    // A dead owner's lock is taken over (crash recovery).
    #[tokio::test]
    async fn stale_lock_is_taken_over() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = pool(&dir);
        // A pid that cannot exist on Linux (pid_max ceiling).
        tokio::fs::write(owner_lock_path(dir.path()), "4194303")
            .await
            .unwrap();
        ReservationAuthority::open(cfg).await.unwrap();
    }

    // SC-TE9: release freezes accrual — no interval is emitted past the
    // release instant; subsequent take_usage_intervals return nothing.
    // Timestamps are real-time-relative because `reserve` stamps the accrual
    // checkpoint with the wall clock.
    #[tokio::test]
    async fn release_stops_accrual() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        auth.reserve("t1", "c1", 1024).await.unwrap();
        let t0 = current_unix_secs() + 1; // strictly after the reserve stamp
        let i1 = auth.take_usage_intervals(t0).await.unwrap();
        assert_eq!(i1.len(), 1);
        auth.release("t1", "c1", t0).await.unwrap();
        let i2 = auth.take_usage_intervals(t0).await.unwrap();
        assert!(i2.is_empty(), "no accrual after release: {i2:?}");
        // Idempotent release.
        auth.release("t1", "c1", t0).await.unwrap();
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
        auth.reserve("t1", "c1", 1024).await.unwrap();
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
        assert_eq!(
            next[0].interval_start_unix_secs, t0,
            "interval starts at the persisted checkpoint"
        );
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
        auth.reserve("t1", "c1", 1024).await.unwrap();
        let err = auth
            .reserve("t1", "c2", 2048)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("tenant 't1' reservation limit"), "{err}");
        // Another tenant is unaffected.
        auth.reserve("t2", "c1", 2048).await.unwrap();
    }

    // Kill switch: reserve refuses with a named reason; the ledger survives.
    #[tokio::test]
    async fn kill_switch_refuses_without_amnesia() {
        let dir = tempfile::tempdir().unwrap();
        let auth = ReservationAuthority::open(pool(&dir)).await.unwrap();
        auth.reserve("t1", "c1", 1024).await.unwrap();
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

    // Config validation: a bad declaration fails at validate(), not at first
    // admission.
    #[test]
    fn pool_config_validation() {
        let mut cfg = pool(&tempfile::tempdir().unwrap());
        cfg.id = " ".to_string();
        assert!(cfg.validate().is_err());
        cfg.id = "p1".to_string();
        cfg.allocatable_bytes = 0;
        assert!(cfg.validate().is_err());
        cfg.allocatable_bytes = 1;
        cfg.overhead_factor = 11.0;
        assert!(cfg.validate().is_err());
        cfg.overhead_factor = 0.1;
        assert!(cfg.validate().is_ok());
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
