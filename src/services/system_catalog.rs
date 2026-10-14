//! `SystemCatalog` — the read-heavy system catalog as a [`Catalog`] implementation.
//!
//! Backs the `proximadb_catalog::Catalog` trait with the in-RAM authority
//! ([`SystemCatalogState`]) made durable by a canonical WAL
//! ([`FramedTableWalAppender`]). It is the WAL-native replacement for the
//! file-per-object `NativeCatalog`: reads are served from RAM (no `path.exists()`
//! per `table_exists`, no `read_dir` per `list_tables`, no `CatalogCache` TTL),
//! and every DDL is one durable, fsync'd WAL append folded into the in-RAM index.
//!
//! Phase 2 of the system-catalog redesign. Semantics mirror `NativeCatalog`
//! method-for-method so the live cutover (boot wires this in place of
//! `create_native_catalog`) is behaviour-preserving for all `Catalog` callers
//! (DML, introspection, REST, pgwire). It lives in the root crate because the
//! canonical-WAL substrate does, and a control-layer crate cannot depend on the
//! root crate (mirrors the `function_store` / `rank_profile_store` recipe).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;

use crate::services::catalog_snapshot_store::{
    CatalogSnapshotStore, LocalSnapshotStore, SnapshotWrite,
};
use proximadb_catalog::schema::{apply_evolution, validate_schema};
use proximadb_catalog::{
    Catalog, CatalogIndex, CatalogMlopsAssetExt, CatalogNamespace, CatalogPrimaryPod,
    CatalogSchemaEvolution, CatalogStorageLayout, CatalogTableSchema, CatalogTableStatistics,
    TableIdentifier,
};

use crate::services::record_store::TableWalAppender;
use crate::services::system_catalog_state::{CatalogDelta, SystemCatalogState};

/// Whether `candidate` is a strictly newer snapshot pointer version than the
/// `loaded` one — treating the [`NO_VERSION`] sentinel ("nothing observed yet")
/// as older than any real version. Monotonic: never reload an equal/older blob.
fn version_is_newer(loaded: u64, candidate: u64) -> bool {
    loaded == NO_VERSION || candidate > loaded
}

/// Current wall-clock milliseconds since the Unix epoch.
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Default number of committed mutations between automatic snapshots; mirrors
/// the legacy metastore's compaction threshold. Override with
/// `PROXIMADB_CATALOG_SNAPSHOT_THRESHOLD`.
const DEFAULT_SNAPSHOT_THRESHOLD: u64 = 1000;

/// Sentinel for "no object-store snapshot version observed yet" — distinct from
/// version `0`, the first real published version. Versions never reach `u64::MAX`.
const NO_VERSION: u64 = u64::MAX;

/// Outcome of a Phase 6b sinval staleness check ([`SystemCatalog::reload_if_stale`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// Coherence is not applicable — this catalog has no object-store snapshot
    /// store (an injected-appender / local single-pod catalog never tails).
    Disabled,
    /// The in-RAM cache was already current with the published snapshot.
    UpToDate,
    /// The in-RAM authority was reloaded from a newer published snapshot.
    Reloaded {
        /// Pointer version adopted.
        version: u64,
        /// Fencing generation the adopted snapshot was committed under.
        generation: u64,
        /// Whether the adopted snapshot was published by a **newer** pod (its
        /// generation outranks this pod's write generation), meaning this pod has
        /// been superseded and has stepped down to read-only.
        superseded: bool,
    },
}

/// File locations + cadence for the snapshot/compaction machinery. Present only
/// when the catalog owns an on-disk WAL (via [`SystemCatalog::open`]); the
/// injected-appender constructor ([`SystemCatalog::new`]) leaves it `None`, so
/// snapshotting is a no-op there.
struct SnapshotConfig {
    wal_path: PathBuf,
    /// Durable home for the snapshot blob — local filesystem by default
    /// ([`LocalSnapshotStore`]) or object storage ([`ObjectStoreSnapshotStore`])
    /// for `s3/gs/az` deployments (Phase 5c).
    store: Arc<dyn CatalogSnapshotStore>,
    threshold: u64,
}

/// WAL-backed read-heavy system catalog.
pub struct SystemCatalog {
    name: String,
    /// Plane this catalog serves (Phase 5 two-tier operator/account model).
    /// Defaults to [`CatalogRole::Operator`] — the single-deployment system
    /// catalog holds the whole deployment's objects until multi-account
    /// provisioning splits data-plane catalogs out per account.
    role: proximadb_catalog::CatalogRole,
    state: Arc<SystemCatalogState>,
    /// Next durable account/tenant stable id. Recovered from the WAL/snapshot
    /// state at boot; u64 lets us detect u32 exhaustion instead of wrapping.
    account_floor: AtomicU64,
    /// Next `object_id` for catalog objects (tables, columns, indexes — one
    /// shared sequence, matching `NativeCatalog::mint_object_id`). Like
    /// `account_floor`, the floor is DERIVED from recovered state rather than
    /// persisted separately, so it cannot disagree with the objects it governs.
    ///
    /// Authorization-critical: policy bindings and grants are keyed on
    /// `object_id`, so re-issuing one after a restart would grant a NEW table
    /// the permissions of a DEAD one.
    object_id_floor: AtomicU64,
    /// Next globally unique catalog object id (table/column/index). Recovered
    /// from durable schemas at boot; starts at one because zero is reserved.
    object_floor: AtomicU64,
    /// Next stable namespace/collection ids. The typed ids are narrower than
    /// object ids, so u64 floors let allocation detect exhaustion before casts.
    namespace_floor: AtomicU64,
    collection_floor: AtomicU64,
    /// Stable namespace id per (account, logical namespace). Durable schemas
    /// are the authority; this map is rebuilt from them after replay.
    namespace_ids: parking_lot::RwLock<HashMap<(String, String), u16>>,
    appender: Arc<dyn TableWalAppender>,
    /// Serializes the durable-append → in-RAM-apply pair so concurrent DDL can
    /// never interleave such that a lower-LSN mutation applies after a higher
    /// one (which the idempotent `apply_committed` would then drop). DDL is
    /// rare, so this coarse lock is free in practice and also gives
    /// read-your-writes on the committing path. Also held during checkpointing
    /// so no append races the WAL compaction.
    write_lock: tokio::sync::Mutex<()>,
    snapshot: Option<SnapshotConfig>,
    commits_since_snapshot: AtomicU64,
    /// Monotonic fencing generation this pod writes snapshots under (Phase 6a).
    /// Claimed at boot as `prior_committed_generation + 1`, so a pod that takes
    /// the catalog over outranks the one it replaced; the object-store snapshot
    /// commit is fenced on it. Single-pod / local: effectively inert (the local
    /// store never fences and each restart simply bumps it monotonically).
    generation: AtomicU64,
    /// Pointer **version** of the object-store snapshot currently reflected in
    /// [`state`](Self::state) (Phase 6b sinval key). [`NO_VERSION`] until the
    /// first published snapshot is observed. The follower poll compares the
    /// store's `latest_version()` against this to decide whether to reload; it is
    /// advanced on every successful publish (our own) and every reload (a peer's).
    loaded_version: AtomicU64,
    /// Fencing generation of the snapshot currently reflected in `state`. Equal
    /// to `generation` once we have published; raised to a peer's generation when
    /// we adopt a newer snapshot. Informational for reads + the step-down check.
    loaded_generation: AtomicU64,
    /// Read-only mode (Phase 6b): a **follower** replica (opened via
    /// [`open_follower`](Self::open_follower)) or an ex-owner that has been
    /// **superseded** by a newer-generation pod. While set, DDL is rejected and
    /// the in-RAM authority is driven solely by tailing the object-store
    /// snapshot — the bounded-staleness read replica. The single-pod default
    /// (an owner) leaves this `false`, so behaviour is unchanged.
    read_only: AtomicBool,
}

impl SystemCatalog {
    /// Construct over an in-RAM state (already replayed from the WAL by the
    /// caller) and the appender that owns the same WAL file. Snapshotting is
    /// disabled (no path bookkeeping); used by tests with an injected appender.
    pub fn new(
        name: impl Into<String>,
        state: SystemCatalogState,
        appender: Arc<dyn TableWalAppender>,
    ) -> Self {
        let (object_floor, namespace_floor, collection_floor, namespace_ids) =
            Self::identity_allocator_seed(&state);
        let account_floor = state.max_account_id().map_or(1, |id| u64::from(id) + 1);
        let object_id_floor = state.max_object_id().map_or(1, |id| id + 1);
        Self {
            name: name.into(),
            role: proximadb_catalog::CatalogRole::Operator,
            state: Arc::new(state),
            account_floor: AtomicU64::new(account_floor),
            object_id_floor: AtomicU64::new(object_id_floor),
            object_floor: AtomicU64::new(object_floor),
            namespace_floor: AtomicU64::new(namespace_floor),
            collection_floor: AtomicU64::new(collection_floor),
            namespace_ids: parking_lot::RwLock::new(namespace_ids),
            appender,
            write_lock: tokio::sync::Mutex::new(()),
            snapshot: None,
            commits_since_snapshot: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            loaded_version: AtomicU64::new(NO_VERSION),
            loaded_generation: AtomicU64::new(0),
            read_only: AtomicBool::new(false),
        }
    }

    /// Set this catalog's plane in the two-tier operator/account model
    /// (Phase 5). Default is [`CatalogRole::Operator`].
    pub fn with_role(mut self, role: proximadb_catalog::CatalogRole) -> Self {
        self.role = role;
        self
    }

    /// The plane this catalog serves.
    pub fn role(&self) -> &proximadb_catalog::CatalogRole {
        &self.role
    }

    /// Recover every transient identity allocator from the durable table
    /// schemas folded into `state`. Skipped ids are harmless; reuse is not.
    fn identity_allocator_seed(
        state: &SystemCatalogState,
    ) -> (u64, u64, u64, HashMap<(String, String), u16>) {
        let object_floor = state.max_object_id().map_or(1, |id| id.saturating_add(1));
        let mut namespace_floor = 1_u64;
        let mut collection_floor = 1_u64;
        let mut namespace_ids = HashMap::new();
        for (identifier, schema) in state.table_entries() {
            if let Some(namespace_id) = schema.stable_namespace_id {
                namespace_floor = namespace_floor.max(u64::from(namespace_id) + 1);
                if let Some(account) = state
                    .get_namespace(&identifier.namespace)
                    .and_then(|namespace| namespace.tenant_id)
                {
                    namespace_ids.insert((account, identifier.namespace.join(".")), namespace_id);
                }
            }
            if let Some(collection_id) = schema.stable_collection_id {
                collection_floor = collection_floor.max(u64::from(collection_id) + 1);
            }
        }
        (
            object_floor,
            namespace_floor,
            collection_floor,
            namespace_ids,
        )
    }

    fn mint_object_id(&self, existing: Option<u64>) -> Result<u64> {
        match existing {
            Some(id) if id > 0 => {
                let next = id
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("catalog object id space exhausted"))?;
                self.object_floor.fetch_max(next, Ordering::SeqCst);
                Ok(id)
            }
            Some(_) => Err(anyhow!("catalog object id zero is reserved")),
            None => self.allocate_bounded(&self.object_floor, u64::MAX, "catalog object"),
        }
    }

    fn allocate_bounded(&self, floor: &AtomicU64, max: u64, label: &str) -> Result<u64> {
        floor
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |id| {
                (id > 0 && id <= max).then(|| id.checked_add(1)).flatten()
            })
            .map_err(|_| anyhow!("{label} id space exhausted"))
    }

    async fn resolve_typed_triple(
        &self,
        account: &str,
        namespace_key: &str,
        existing_namespace: Option<u16>,
        existing_collection: Option<u32>,
    ) -> Result<Option<(u32, u16, u32)>> {
        // TD-CAT-7: both `Catalog` and `CatalogAuthority` expose this now (the
        // former delegates to the latter), so name the authority explicitly.
        let Some(account_id) =
            proximadb_catalog::CatalogAuthority::account_id_u32(self, account).await?
        else {
            return Ok(None);
        };
        let key = (account.to_string(), namespace_key.to_string());
        let namespace_id = if let Some(id) = existing_namespace {
            self.namespace_floor
                .fetch_max(u64::from(id) + 1, Ordering::SeqCst);
            let mut registry = self.namespace_ids.write();
            match registry.get(&key) {
                Some(current) if *current != id => {
                    return Err(anyhow!(
                        "stable namespace identity mismatch for '{}': {} != {}",
                        namespace_key,
                        current,
                        id
                    ));
                }
                Some(_) => {}
                None => {
                    registry.insert(key, id);
                }
            }
            id
        } else if let Some(id) = self.namespace_ids.read().get(&key).copied() {
            id
        } else {
            let mut registry = self.namespace_ids.write();
            if let Some(id) = registry.get(&key).copied() {
                id
            } else {
                let id = self.allocate_bounded(
                    &self.namespace_floor,
                    u64::from(u16::MAX),
                    "stable namespace",
                )? as u16;
                registry.insert(key, id);
                id
            }
        };
        let collection_id = match existing_collection {
            Some(id) => {
                self.collection_floor
                    .fetch_max(u64::from(id) + 1, Ordering::SeqCst);
                id
            }
            None => self.allocate_bounded(
                &self.collection_floor,
                u64::from(u32::MAX),
                "stable collection",
            )? as u32,
        };
        Ok(Some((account_id, namespace_id, collection_id)))
    }

    /// Open (or create) the catalog WAL at `wal_path`, restore the in-RAM
    /// authority, and return a ready catalog. Boot entry point.
    ///
    /// Restore order (bounded restart): if a durable snapshot blob exists, seed
    /// state + watermark from it, then replay only the WAL entries after the
    /// watermark (idempotent on sequence). Otherwise replay the whole WAL from
    /// empty. A snapshot that exists but fails to decode is **fatal** — after
    /// compaction the WAL alone no longer carries the pre-watermark history, so
    /// silently falling back to a WAL-only replay would lose committed DDL.
    pub async fn open(name: impl Into<String>, wal_path: impl Into<PathBuf>) -> Result<Self> {
        let wal_path = wal_path.into();
        // Default: the snapshot blob lives next to the local WAL.
        let snapshot_store = Arc::new(LocalSnapshotStore::new(wal_path.with_extension("snapshot")));
        Self::open_with_snapshot_store(name, wal_path, snapshot_store).await
    }

    /// Like [`open`](Self::open) but with an injected snapshot store. The WAL
    /// itself stays on the local filesystem at `wal_path` (object-store-native
    /// WAL durability is Phase 6); only the snapshot blob's durable home is
    /// pluggable — local fs (default) or object storage (`s3/gs/az`/`memory`)
    /// under the tenant/operator DrPath prefix (Phase 5c).
    pub async fn open_with_snapshot_store(
        name: impl Into<String>,
        wal_path: impl Into<PathBuf>,
        snapshot_store: Arc<dyn CatalogSnapshotStore>,
    ) -> Result<Self> {
        // An owner: claims the next generation and serves writes.
        Self::open_inner(name, wal_path, snapshot_store, true).await
    }

    /// Open the catalog as a **read-only follower** (Phase 6b): seed the in-RAM
    /// authority from the current object-store snapshot, but do **not** claim a
    /// higher fencing generation (a follower never takes the catalog over) and
    /// **reject DDL**. A follower tails the object-store snapshot — call
    /// [`reload_if_stale`](Self::reload_if_stale) periodically (or
    /// [`spawn_follower_poll`](Self::spawn_follower_poll)) to pull the owner's
    /// published mutations into RAM within the poll interval (bounded staleness).
    /// Reads are served from RAM exactly as on the owner; only writes differ.
    pub async fn open_follower(
        name: impl Into<String>,
        wal_path: impl Into<PathBuf>,
        snapshot_store: Arc<dyn CatalogSnapshotStore>,
    ) -> Result<Self> {
        // A follower: keeps the prior generation (no takeover) and is read-only.
        Self::open_inner(name, wal_path, snapshot_store, false).await
    }

    /// Shared boot path for owner ([`open_with_snapshot_store`](Self::open_with_snapshot_store))
    /// and follower ([`open_follower`](Self::open_follower)) catalogs.
    ///
    /// `claim_owner = true` makes this an owner: it claims `prior_generation + 1`
    /// so its fenced snapshot commits outrank the pod it replaced, and it serves
    /// writes. `claim_owner = false` makes this a read-only follower: it stays at
    /// the prior generation and rejects DDL.
    async fn open_inner(
        name: impl Into<String>,
        wal_path: impl Into<PathBuf>,
        snapshot_store: Arc<dyn CatalogSnapshotStore>,
        claim_owner: bool,
    ) -> Result<Self> {
        let wal_path = wal_path.into();
        let appender = Arc::new(crate::services::FramedTableWalAppender::open(&wal_path).await?);
        let entries = appender.read_entries().await?;

        // A snapshot that exists but fails to decode is **fatal**: after a
        // compaction the WAL alone no longer carries the pre-watermark history,
        // so silently falling back to a WAL-only replay would lose committed DDL.
        // `read` also yields the snapshot's pointer version + fencing generation.
        let (state, prior_version, prior_generation) = match snapshot_store.read().await? {
            Some(read) => {
                let state =
                    SystemCatalogState::from_snapshot_bytes(&read.bytes).with_context(|| {
                        format!("decoding catalog snapshot {}", snapshot_store.describe())
                    })?;
                // The local WAL tail (entries past the snapshot watermark) is a
                // no-op for a follower (it never appends), and the owner's own
                // post-snapshot durability for an owner.
                state.replay(&entries)?;
                (state, read.version, read.generation)
            }
            None => (
                SystemCatalogState::from_wal_entries(&entries)?,
                NO_VERSION,
                0,
            ),
        };

        // Seed the appender's sequence floor from the snapshot watermark so a
        // fresh local WAL (compacted-to-empty, or a peer's snapshot adopted into
        // a different local sequence space) cannot mint a sequence the in-RAM
        // authority has already applied (which `apply_committed` would drop). A
        // no-op when the local WAL already carries the watermark.
        appender.advance_sequence_floor(state.applied_seq());

        // Owner claims the next generation — a pod taking the catalog over
        // outranks the one it replaced, so its first fenced commit fences any
        // stale writer. For the local (unfenced) store this is inert. A follower
        // keeps the prior generation: it never publishes, so it must not outrank
        // the owner whose snapshot it tails.
        let generation = if claim_owner {
            prior_generation + 1
        } else {
            prior_generation
        };

        let threshold = std::env::var("PROXIMADB_CATALOG_SNAPSHOT_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_SNAPSHOT_THRESHOLD);

        let (object_floor, namespace_floor, collection_floor, namespace_ids) =
            Self::identity_allocator_seed(&state);
        let account_floor = state.max_account_id().map_or(1, |id| u64::from(id) + 1);
        let object_id_floor = state.max_object_id().map_or(1, |id| id + 1);
        Ok(Self {
            name: name.into(),
            role: proximadb_catalog::CatalogRole::Operator,
            state: Arc::new(state),
            account_floor: AtomicU64::new(account_floor),
            object_id_floor: AtomicU64::new(object_id_floor),
            object_floor: AtomicU64::new(object_floor),
            namespace_floor: AtomicU64::new(namespace_floor),
            collection_floor: AtomicU64::new(collection_floor),
            namespace_ids: parking_lot::RwLock::new(namespace_ids),
            appender,
            write_lock: tokio::sync::Mutex::new(()),
            snapshot: Some(SnapshotConfig {
                wal_path,
                store: snapshot_store,
                threshold,
            }),
            commits_since_snapshot: AtomicU64::new(0),
            generation: AtomicU64::new(generation),
            loaded_version: AtomicU64::new(prior_version),
            loaded_generation: AtomicU64::new(prior_generation),
            read_only: AtomicBool::new(!claim_owner),
        })
    }

    /// Durably append the deltas to the WAL, then fold them into the in-RAM
    /// authority — atomically with respect to other writers. Triggers an
    /// automatic checkpoint once enough mutations have accumulated.
    async fn commit_batch(&self, deltas: Vec<CatalogDelta>) -> Result<()> {
        // Phase 6b: a follower replica / superseded ex-owner serves reads only;
        // DDL must go to the owning pod. (Single-pod owner: never read-only.)
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "system catalog '{}' is read-only (a follower replica or superseded by a \
                 newer pod); route DDL to the owning pod",
                self.name
            ));
        }
        let _guard = self.write_lock.lock().await;
        self.commit_batch_locked(deltas).await
    }

    /// Commit with [`Self::write_lock`] already held. Account-id minting needs
    /// to check-and-insert under the same lock so two first requests for one
    /// tenant cannot receive different ids.
    async fn commit_batch_locked(&self, deltas: Vec<CatalogDelta>) -> Result<()> {
        let ops = deltas
            .iter()
            .map(|d| d.to_operation())
            .collect::<Result<Vec<_>>>()?;
        let entries = self.appender.append_operations(ops, None).await?;
        let applied = entries.len() as u64;
        for (entry, delta) in entries.into_iter().zip(deltas) {
            self.state.apply_committed(entry.sequence_number, delta);
        }
        if let Some(cfg) = &self.snapshot {
            let n = self
                .commits_since_snapshot
                .fetch_add(applied, Ordering::SeqCst)
                + applied;
            // For an object-store catalog the local WAL is only a staging log:
            // a VM/pod replacement starts with a fresh local data_dir, so the
            // snapshot is the ONLY catalog artifact that survives. Publish it
            // before acknowledging every DDL. Local catalogs retain the normal
            // threshold because their WAL itself is durable.
            if cfg.store.requires_snapshot_on_commit() || n >= cfg.threshold {
                self.checkpoint_locked(cfg).await?;
                self.commits_since_snapshot.store(0, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    /// Retry a failed object-store publication before a control-plane caller
    /// treats an already-visible account id as durable. `commit_batch_locked`
    /// applies the WAL entry before publishing the required snapshot; if that
    /// publication fails, `commits_since_snapshot` intentionally stays nonzero
    /// and the next mint-or-return call repairs it here.
    async fn publish_pending_snapshot_locked(&self) -> Result<()> {
        let Some(cfg) = &self.snapshot else {
            return Ok(());
        };
        if self.read_only.load(Ordering::SeqCst)
            || !cfg.store.requires_snapshot_on_commit()
            || self.commits_since_snapshot.load(Ordering::SeqCst) == 0
        {
            return Ok(());
        }
        self.checkpoint_locked(cfg).await?;
        self.commits_since_snapshot.store(0, Ordering::SeqCst);
        Ok(())
    }

    async fn commit(&self, delta: CatalogDelta) -> Result<()> {
        self.commit_batch(vec![delta]).await
    }

    /// Force a snapshot + WAL compaction now. No-op for injected-appender
    /// catalogs and for read-only followers (they never publish). Takes the
    /// write lock; callers must not already hold it.
    pub async fn checkpoint(&self) -> Result<()> {
        if self.read_only.load(Ordering::SeqCst) {
            return Ok(());
        }
        if let Some(cfg) = &self.snapshot {
            let _guard = self.write_lock.lock().await;
            self.checkpoint_locked(cfg).await?;
            self.commits_since_snapshot.store(0, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Snapshot the in-RAM authority durably, then compact the WAL to drop the
    /// entries the snapshot already covers. **The write lock must be held.**
    ///
    /// Crash-consistency ordering (the load-bearing invariant — a committed DDL
    /// must never be lost): the snapshot is made durable *before* the WAL is
    /// compacted. So at every crash point, recovery is correct:
    /// - before the snapshot rename: snapshot blob is the previous one (or
    ///   absent); the WAL still holds every entry → full/bounded replay, no loss.
    /// - after the snapshot rename, before/during compaction: snapshot covers
    ///   `watermark`; the WAL still holds everything; boot replays `> watermark`
    ///   (entries `<= watermark` are idempotently skipped). No loss.
    /// - after compaction: WAL holds only `> watermark`; boot seeds from the
    ///   snapshot then replays exactly those. No loss.
    async fn checkpoint_locked(&self, cfg: &SnapshotConfig) -> Result<()> {
        let watermark = self.state.applied_seq();

        // 1. Durable snapshot. The store guarantees an atomic replace (local:
        //    temp → fsync → rename; object store: a generation-fenced commit), so
        //    a reader sees the old or new blob, never a torn one.
        let generation = self.generation.load(Ordering::SeqCst);
        let bytes = self.state.to_snapshot_bytes()?;
        match cfg.store.write_atomic(generation, &bytes).await? {
            SnapshotWrite::Published { version } => {
                // Our publish IS the latest snapshot now — record its version +
                // generation so the sinval check does not mistake our own write
                // for a peer's and reload it away (read-your-writes).
                self.loaded_version.store(version, Ordering::SeqCst);
                self.loaded_generation.store(generation, Ordering::SeqCst);
            }
            SnapshotWrite::Fenced => {
                // Fenced (Phase 6a): a newer pod (higher generation) has taken
                // the catalog over. We are a stale writer — the snapshot we tried
                // to publish did NOT land, so we MUST NOT compact our local WAL.
                // Phase 6b step-down: pull the newer pod's snapshot into RAM
                // (discarding our now-doomed post-snapshot writes) and flip to
                // read-only so we stop emitting writes the fence will reject. We
                // already hold the write lock, so reload directly.
                tracing::warn!(
                    catalog = %self.name,
                    generation,
                    store = %cfg.store.describe(),
                    "catalog snapshot checkpoint was fenced by a newer pod; \
                     stepping down to read-only and reloading its snapshot"
                );
                self.reload_from_store_locked(cfg).await?;
                return Ok(());
            }
        }

        // 2. Compact the WAL: keep only entries the snapshot does not cover.
        let entries = self.appender.read_all_entries().await?;
        if entries.iter().any(|e| e.sequence_number <= watermark) {
            let kept: Vec<_> = entries
                .into_iter()
                .filter(|e| e.sequence_number > watermark)
                .collect();
            crate::services::canonical_wal::rewrite_canonical_wal(&cfg.wal_path, &kept).await?;
        }
        Ok(())
    }

    // ── Phase 6b: cross-pod cache coherence (sinval-style invalidation) ───────

    /// **Sinval staleness check.** Compare this pod's cached snapshot version
    /// against the object store's latest published version; if a newer snapshot
    /// has been published (by the owner, or by a pod that took the catalog over),
    /// lazily reload the in-RAM authority from it. This is the Postgres-`sinval`
    /// analog: the published pointer's monotonic version is the invalidation
    /// signal; a follower (or a superseded ex-owner) reloads on mismatch and is
    /// thereafter consistent within the poll interval (bounded staleness).
    ///
    /// Cheap when nothing changed: a single object-store `list` probe
    /// ([`CatalogSnapshotStore::latest_version`]) with no payload fetch. The
    /// (larger) snapshot payload is read only on a version advance. Inert for the
    /// local single-pod store (it reports no version) and for injected-appender
    /// catalogs (no snapshot store).
    pub async fn reload_if_stale(&self) -> Result<ReloadOutcome> {
        let cfg = match &self.snapshot {
            Some(cfg) => cfg,
            None => return Ok(ReloadOutcome::Disabled),
        };
        // Cheap probe first — no payload fetch.
        let latest = cfg.store.latest_version().await?;
        let loaded = self.loaded_version.load(Ordering::SeqCst);
        match latest {
            Some(v) if version_is_newer(loaded, v) => {
                // A newer snapshot exists — take the write lock so no DDL or
                // checkpoint interleaves the state swap, then reload.
                let _guard = self.write_lock.lock().await;
                self.reload_from_store_locked(cfg).await
            }
            _ => Ok(ReloadOutcome::UpToDate),
        }
    }

    /// Reload the in-RAM authority from the current object-store snapshot,
    /// **assuming the write lock is held**. Used by [`reload_if_stale`] (after it
    /// detects a version advance) and by the fenced-checkpoint step-down path.
    /// Re-checks the version under the lock so two concurrent reloaders don't
    /// double-apply, and never moves backward.
    async fn reload_from_store_locked(&self, cfg: &SnapshotConfig) -> Result<ReloadOutcome> {
        let read = match cfg.store.read().await? {
            Some(read) => read,
            // Nothing published (or it vanished) — nothing to adopt.
            None => return Ok(ReloadOutcome::UpToDate),
        };
        let loaded = self.loaded_version.load(Ordering::SeqCst);
        if !version_is_newer(loaded, read.version) {
            // Another reload already pulled this version (or newer); never reload
            // an older snapshot over a current one (monotonic reads).
            return Ok(ReloadOutcome::UpToDate);
        }

        self.state
            .load_from_snapshot_bytes(&read.bytes)
            .with_context(|| {
                format!(
                    "reloading catalog '{}' from snapshot {}",
                    self.name,
                    cfg.store.describe()
                )
            })?;
        // The snapshot may contain account ids minted by another pod after
        // this instance booted. Keep the local allocator strictly above every
        // adopted id even when the reload is same-generation (and therefore
        // does not force this instance read-only).
        if let Some(max) = self.state.max_account_id() {
            self.account_floor
                .fetch_max(u64::from(max) + 1, Ordering::SeqCst);
        }
        if let Some(max) = self.state.max_object_id() {
            let next = max
                .checked_add(1)
                .ok_or_else(|| anyhow!("catalog object-id space exhausted at {max}"))?;
            self.object_id_floor.fetch_max(next, Ordering::SeqCst);
        }
        // The adopted authority may also carry object/typed ids minted by the
        // publisher after this pod booted. Re-seed every transient allocator and
        // replace the namespace map while the catalog write lock is held, so a
        // same-generation reload cannot later reuse an adopted id.
        let (object_floor, namespace_floor, collection_floor, namespace_ids) =
            Self::identity_allocator_seed(&self.state);
        self.object_floor.fetch_max(object_floor, Ordering::SeqCst);
        self.namespace_floor
            .fetch_max(namespace_floor, Ordering::SeqCst);
        self.collection_floor
            .fetch_max(collection_floor, Ordering::SeqCst);
        *self.namespace_ids.write() = namespace_ids;
        self.loaded_version.store(read.version, Ordering::SeqCst);
        self.loaded_generation
            .store(read.generation, Ordering::SeqCst);

        // Superseded: the adopted snapshot was committed under a generation that
        // outranks this pod's write generation, i.e. a newer pod owns the
        // catalog. Step down to read-only so we stop emitting writes the fence
        // would reject (reacquiring write ownership is the Phase 7 lease).
        let superseded = read.generation > self.generation.load(Ordering::SeqCst);
        if superseded {
            self.read_only.store(true, Ordering::SeqCst);
        }
        Ok(ReloadOutcome::Reloaded {
            version: read.version,
            generation: read.generation,
            superseded,
        })
    }

    /// Spawn a background task that tails the object-store snapshot, calling
    /// [`reload_if_stale`](Self::reload_if_stale) every `interval`. This is the
    /// **follower tailer** — it pulls the owner's published DDL into a follower's
    /// (or a superseded ex-owner's) RAM within `interval` (bounded staleness).
    ///
    /// The returned [`JoinHandle`](tokio::task::JoinHandle) **owns the task**: the
    /// caller must `abort()` it on shutdown (it is a cooperative tokio task, not
    /// an OS thread, so it never blocks process exit, but a clean abort avoids a
    /// stray reload firing during teardown). Inert for catalogs without a
    /// snapshot store / version concept (each tick is a no-op `Disabled`).
    pub fn spawn_follower_poll(self: Arc<Self>, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // The first tick fires immediately; skip it so we honour `interval`.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                match self.reload_if_stale().await {
                    Ok(ReloadOutcome::Reloaded {
                        version,
                        generation,
                        superseded,
                    }) => {
                        tracing::info!(
                            catalog = %self.name,
                            version,
                            generation,
                            superseded,
                            "catalog follower reloaded in-RAM authority from a newer \
                             object-store snapshot"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(
                            catalog = %self.name,
                            error = %e,
                            "catalog follower poll failed; will retry next interval"
                        );
                    }
                }
            }
        })
    }

    /// The fencing generation this pod writes snapshots under (Phase 6a/6b).
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// The object-store snapshot pointer version currently reflected in RAM, or
    /// `None` if no published snapshot has been observed yet.
    pub fn loaded_version(&self) -> Option<u64> {
        match self.loaded_version.load(Ordering::SeqCst) {
            NO_VERSION => None,
            v => Some(v),
        }
    }

    /// Whether this catalog is a read-only replica — a follower, or an ex-owner
    /// superseded by a newer-generation pod (Phase 6b).
    pub fn is_read_only(&self) -> bool {
        self.read_only.load(Ordering::SeqCst)
    }

    /// Load a table's current schema or fail (the catalog's "table not found").
    fn require_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        self.state
            .get_table(identifier)
            .map(|arc| (*arc).clone())
            .ok_or_else(|| anyhow!("Table '{}' not found", identifier))
    }

    async fn create_namespace_inner(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
        tenant_id: Option<String>,
    ) -> Result<CatalogNamespace> {
        if self.state.namespace_exists(namespace) {
            return Err(anyhow!(
                "Namespace '{}' already exists",
                namespace.join(".")
            ));
        }
        let mut ns = CatalogNamespace::new(namespace.to_vec());
        ns.properties = properties;
        // Opaque, rename-stable server-issued id that drives physical paths
        // (DrPathBuilder); `tenant_id` records the owning tenant when created in
        // a tenant scope. Mirrors NativeCatalog::create_namespace_inner.
        ns.namespace_id = Some(format!("ns_{}", uuid::Uuid::new_v4()));
        ns.tenant_id = tenant_id;
        // TD-CAT-9: the stable catalog object_id, minted from the ONE
        // system-wide sequence that mints table object_ids (ADR-031
        // reconciliation amendment 1 — one sequence, not per-type). The default
        // backend had been the only internal catalog leaving this `None` while
        // reporting the same `catalog_type` as `NativeCatalog`, which does mint.
        //
        // Legacy namespaces persisted before this keep `object_id: None` and are
        // deliberately NOT backfilled: this catalog's state is derived from the
        // WAL, so an in-RAM backfill would be non-durable and would re-mint a
        // DIFFERENT id on every restart — churn plus a collision risk, which is
        // worse than the absence. `max_object_id` folds whatever ids exist, so
        // old and new coexist safely (mixed-read-safe, mandate #8).
        ns.object_id = Some(self.mint_object_id(None)?);
        self.commit(CatalogDelta::UpsertNamespace {
            namespace: Box::new(ns.clone()),
        })
        .await?;
        Ok(ns)
    }
}

/// TD-CAT-7: `SystemCatalog` is the DEFAULT backend and the production identity
/// authority — it owns the WAL-backed allocator and the account registry. Every
/// method below is required by the trait, so the silent `Ok(None)` inheritance
/// that left relational ABAC structurally inert (TD-AUTHZ-2) cannot recur here.
// Referenced by full path, deliberately NOT imported — see the identical note on
// `NativeCatalog`: both traits declare these names, so importing this one makes
// every concrete-type call site ambiguous.
#[async_trait]
impl proximadb_catalog::CatalogAuthority for SystemCatalog {
    async fn max_object_id(&self) -> Result<Option<u64>> {
        Ok(self.state.max_object_id())
    }
    async fn allocate_object_id(&self) -> Result<u64> {
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "system catalog '{}' is read-only; object ids must be allocated on the owning pod",
                self.name
            ));
        }
        // Reserve from the exact sequence create_table uses. The later create
        // adopts this caller-supplied id, so collection lifecycle cannot race a
        // relational DDL path through an independent process-global allocator.
        self.mint_object_id(None)
    }
    async fn mint_collection_typed_identity(
        &self,
        account: &str,
        namespace_key: &str,
    ) -> Result<Option<(u32, u16, u32)>> {
        let account = account.trim();
        if account.is_empty() {
            return Ok(None);
        }
        self.resolve_typed_triple(account, namespace_key, None, None)
            .await
    }
    /// Mint-or-return the durable tenant/account stable id through the same
    /// canonical WAL as every other SystemCatalog mutation. This is the normal
    /// server path (SystemCatalog is the default backend), so inheriting the
    /// trait's `None` implementation would leave ABAC silently unkeyed.
    async fn account_id_u32(&self, account: &str) -> Result<Option<u32>> {
        let account = account.trim();
        if account.is_empty() {
            return Ok(None);
        }
        if let Some(id) = self.state.account_id_u32(account) {
            if self
                .snapshot
                .as_ref()
                .is_some_and(|cfg| cfg.store.requires_snapshot_on_commit())
                && self.commits_since_snapshot.load(Ordering::SeqCst) > 0
                && !self.read_only.load(Ordering::SeqCst)
            {
                let _guard = self.write_lock.lock().await;
                self.publish_pending_snapshot_locked().await?;
                // A fencing reload can replace the state while repairing the
                // publication, so never return the pre-lock value blindly.
                return Ok(self.state.account_id_u32(account));
            }
            return Ok(Some(id));
        }
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "tenant '{account}' has no stable id and system catalog '{}' is read-only; \
                 mint it on the owning pod",
                self.name
            ));
        }

        // Check + allocate + durable append are one writer critical section.
        // Without the second check, concurrent first requests could return two
        // different policy keys for the same tenant.
        let _guard = self.write_lock.lock().await;
        if let Some(id) = self.state.account_id_u32(account) {
            self.publish_pending_snapshot_locked().await?;
            if self.read_only.load(Ordering::SeqCst) {
                return Ok(self.state.account_id_u32(account));
            }
            return Ok(Some(id));
        }
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "tenant '{account}' has no stable id and system catalog '{}' became read-only; \
                 mint it on the owning pod",
                self.name
            ));
        }
        let next = self.account_floor.fetch_add(1, Ordering::SeqCst);
        let stable_id = u32::try_from(next)
            .map_err(|_| anyhow!("tenant stable-id space exhausted at {next}"))?;
        self.commit_batch_locked(vec![CatalogDelta::UpsertAccount {
            account: account.to_string(),
            stable_id,
        }])
        .await?;
        Ok(Some(stable_id))
    }
    fn account_id_u32_lookup(&self, account: &str) -> Option<u32> {
        let account = account.trim();
        if account.is_empty() {
            return None;
        }
        self.state.account_id_u32(account)
    }
    async fn get_table_by_object_id(&self, object_id: u64) -> Result<Option<TableIdentifier>> {
        Ok(self.state.get_table_by_object_id(object_id))
    }
    async fn get_namespace_by_object_id(&self, object_id: u64) -> Result<Option<Vec<String>>> {
        // TD-CAT-7: this override did not exist before the split — `SystemCatalog`
        // is the DEFAULT backend and had been silently inheriting the trait's
        // `Ok(None)`, so a namespace object_id resolved to "not found" rather
        // than to its namespace. No production caller had noticed yet; the
        // compiler now requires an answer, which is the point of the split.
        Ok(self.state.get_namespace_by_object_id(object_id))
    }
}

#[async_trait]
impl Catalog for SystemCatalog {
    fn identity_authority(&self) -> Option<&dyn proximadb_catalog::CatalogAuthority> {
        Some(self)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        // Same type string as NativeCatalog so downstream behaviour/introspection
        // that keys on the catalog type is unchanged across the cutover.
        "native"
    }

    // ── Namespace operations ──────────────────────────────────────────────

    async fn create_namespace(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
    ) -> Result<CatalogNamespace> {
        self.create_namespace_inner(namespace, properties, None)
            .await
    }

    async fn create_namespace_for_tenant(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
        tenant: Option<&str>,
    ) -> Result<CatalogNamespace> {
        let tenant_id = tenant.filter(|t| !t.is_empty()).map(str::to_string);
        self.create_namespace_inner(namespace, properties, tenant_id)
            .await
    }

    async fn drop_namespace(&self, namespace: &[String], cascade: bool) -> Result<bool> {
        if !self.state.namespace_exists(namespace) {
            return Ok(false);
        }
        if !cascade && !self.state.list_tables(namespace).is_empty() {
            return Err(anyhow!(
                "Namespace '{}' is not empty. Use cascade=true to force drop.",
                namespace.join(".")
            ));
        }
        // The DropNamespace fold cascades to child tables + their statistics.
        self.commit(CatalogDelta::DropNamespace {
            levels: namespace.to_vec(),
        })
        .await?;
        Ok(true)
    }

    async fn list_namespaces(&self, parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let all = self.state.all_namespaces();
        let results = all
            .into_iter()
            .filter(|ns| match parent {
                Some(p) => ns.levels.len() == p.len() + 1 && ns.levels.starts_with(p),
                None => true,
            })
            .collect();
        Ok(results)
    }

    async fn namespace_exists(&self, namespace: &[String]) -> Result<bool> {
        Ok(self.state.namespace_exists(namespace))
    }

    async fn get_namespace(&self, namespace: &[String]) -> Result<CatalogNamespace> {
        self.state
            .get_namespace(namespace)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", namespace.join(".")))
    }

    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        removals: Vec<String>,
    ) -> Result<()> {
        let mut ns = self
            .state
            .get_namespace(namespace)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", namespace.join(".")))?;
        for (k, v) in updates {
            ns.properties.insert(k, v);
        }
        for k in removals {
            ns.properties.remove(&k);
        }
        ns.updated_at_ms = now_millis();
        self.commit(CatalogDelta::UpsertNamespace {
            namespace: Box::new(ns),
        })
        .await
    }

    // ── Table operations ──────────────────────────────────────────────────

    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        let mut schema = schema;
        proximadb_catalog::schema::normalize_identity(&mut schema);
        validate_schema(&schema)?;
        // TD-AUTHZ-2: mint object ids. ADOPT-or-allocate, matching
        // `NativeCatalog::mint_object_id`: a caller-supplied id is KEPT and
        // merely raises the floor, never overwritten — that is what preserves
        // the ADR-031 unification where a vector collection pre-sets
        // `schema.object_id` from `collection.id` before this call.
        //
        // Without this the live catalog left `object_id` None for every
        // SQL-created table, and the relational ABAC path denies on absent ids
        // (dml/mod.rs:1466) — so relational policy bindings and grants could
        // never take effect at all.
        schema.object_id = Some(self.mint_object_id(schema.object_id)?);
        for column in &mut schema.columns {
            column.object_id = Some(self.mint_object_id(column.object_id)?);
        }
        for index in &mut schema.indexes {
            index.object_id = Some(self.mint_object_id(index.object_id)?);
        }
        let table_object_id = schema
            .object_id
            .ok_or_else(|| anyhow!("system catalog failed to mint a table object id"))?;
        if let Some(existing) = self.state.get_table_by_object_id(table_object_id)
            && existing != *identifier
        {
            return Err(anyhow!(
                "object id {table_object_id} is already assigned to table '{existing}'"
            ));
        }
        if !self.state.namespace_exists(&identifier.namespace) {
            return Err(anyhow!(
                "Namespace '{}' does not exist",
                identifier.namespace.join(".")
            ));
        }
        if self.state.table_exists(identifier) {
            return Err(anyhow!("Table '{}' already exists", identifier));
        }
        schema.object_id = Some(self.mint_object_id(schema.object_id)?);
        for column in &mut schema.columns {
            column.object_id = Some(self.mint_object_id(column.object_id)?);
        }
        for index in &mut schema.indexes {
            index.object_id = Some(self.mint_object_id(index.object_id)?);
        }
        let namespace = self
            .state
            .get_namespace(&identifier.namespace)
            .ok_or_else(|| {
                anyhow!(
                    "Namespace '{}' does not exist",
                    identifier.namespace.join(".")
                )
            })?;
        if let Some(account) = namespace.tenant_id.as_deref()
            && let Some((_account_id, namespace_id, collection_id)) = self
                .resolve_typed_triple(
                    account,
                    &identifier.namespace.join("."),
                    schema.stable_namespace_id,
                    schema.stable_collection_id,
                )
                .await?
        {
            schema.stable_namespace_id = Some(namespace_id);
            schema.stable_collection_id = Some(collection_id);
        }
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema.clone()),
        })
        .await?;
        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, _purge: bool) -> Result<bool> {
        // `_purge` (physical data removal) is the storage engine's concern, not
        // the catalog's; the catalog only owns metadata. Mirrors the
        // metadata-only side of NativeCatalog::drop_table.
        if !self.state.table_exists(identifier) {
            return Ok(false);
        }
        self.commit(CatalogDelta::DropTable {
            identifier: identifier.clone(),
        })
        .await?;
        Ok(true)
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        Ok(self.state.list_tables(namespace))
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        Ok(self.state.table_exists(identifier))
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        self.require_table(identifier)
    }

    async fn rename_table(&self, from: &TableIdentifier, to: &TableIdentifier) -> Result<()> {
        let mut schema = self.require_table(from)?;
        if self.state.table_exists(to) {
            return Err(anyhow!("Table '{}' already exists", to));
        }
        schema.name = to.name.clone();
        schema.updated_at_ms = now_millis();
        // Atomic batch: drop the old key + insert the new one in one durable
        // append, so a crash between them can't leave the table doubly-present.
        self.commit_batch(vec![
            CatalogDelta::DropTable {
                identifier: from.clone(),
            },
            CatalogDelta::UpsertTable {
                identifier: to.clone(),
                schema: Box::new(schema),
            },
        ])
        .await
    }

    // ── Schema evolution ──────────────────────────────────────────────────

    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> Result<CatalogTableSchema> {
        let schema = self.require_table(identifier)?;
        let evolved = apply_evolution(&schema, &evolution)?;
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(evolved.clone()),
        })
        .await?;
        Ok(evolved)
    }

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> Result<i32> {
        Ok(self.require_table(identifier)?.schema_version)
    }

    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        version: i32,
    ) -> Result<CatalogTableSchema> {
        // Like NativeCatalog, only the current version is retained.
        let schema = self.require_table(identifier)?;
        if schema.schema_version == version {
            Ok(schema)
        } else {
            Err(anyhow!(
                "Schema version {} not found for table '{}' (current: {})",
                version,
                identifier,
                schema.schema_version
            ))
        }
    }

    // ── Index operations ──────────────────────────────────────────────────

    async fn create_index(
        &self,
        identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        let mut index = index;
        let mut schema = self.require_table(identifier)?;
        if schema.indexes.iter().any(|i| i.name == index.name) {
            return Err(anyhow!(
                "Index '{}' already exists on table '{}'",
                index.name,
                identifier
            ));
        }
        for col in &index.columns {
            if !schema.columns.iter().any(|c| &c.name == col) {
                return Err(anyhow!(
                    "Column '{}' not found in table '{}'",
                    col,
                    identifier
                ));
            }
        }
        index.object_id = Some(self.mint_object_id(index.object_id)?);
        schema.indexes.push(index.clone());
        schema.updated_at_ms = now_millis();
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema),
        })
        .await?;
        Ok(index)
    }

    async fn drop_index(&self, identifier: &TableIdentifier, index_name: &str) -> Result<bool> {
        let mut schema = self.require_table(identifier)?;
        let initial = schema.indexes.len();
        schema.indexes.retain(|i| i.name != index_name);
        if schema.indexes.len() == initial {
            return Ok(false);
        }
        schema.updated_at_ms = now_millis();
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema),
        })
        .await?;
        Ok(true)
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        Ok(self.require_table(identifier)?.indexes)
    }

    // ── Statistics ────────────────────────────────────────────────────────

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        if !self.state.table_exists(identifier) {
            return Err(anyhow!("Table '{}' not found", identifier));
        }
        Ok(self.state.get_statistics(identifier).unwrap_or_default())
    }

    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> Result<()> {
        if !self.state.table_exists(identifier) {
            return Err(anyhow!("Table '{}' not found", identifier));
        }
        self.commit(CatalogDelta::UpsertStatistics {
            identifier: identifier.clone(),
            stats: Box::new(stats),
        })
        .await
    }

    // ── Physical/publication attributes (override the error defaults) ─────

    async fn set_primary_pod(
        &self,
        identifier: &TableIdentifier,
        primary: Option<CatalogPrimaryPod>,
    ) -> Result<()> {
        let mut schema = self.require_table(identifier)?;
        schema.primary_pod = primary;
        schema.updated_at_ms = now_millis();
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema),
        })
        .await
    }

    async fn set_storage_layouts(
        &self,
        identifier: &TableIdentifier,
        layouts: Vec<CatalogStorageLayout>,
    ) -> Result<CatalogTableSchema> {
        let mut schema = self.require_table(identifier)?;
        schema.storage_layouts = layouts;
        schema.updated_at_ms = now_millis();
        self.commit(CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema.clone()),
        })
        .await?;
        Ok(schema)
    }

    async fn apply_model_registry_mutation(
        &self,
        identifier: &TableIdentifier,
        expected_revision: u64,
        mutation: proximadb_catalog::mlops::CatalogModelRegistryMutation,
    ) -> Result<CatalogTableSchema> {
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "system catalog '{}' is read-only (a follower replica or superseded by a newer \
                 pod); route model registry mutation to the owning pod",
                self.name
            ));
        }

        // The read, revision check, mutation, durable append, and in-RAM apply
        // form one compare-and-swap critical section. Taking the lock only for
        // the final write would allow two callers with the same expected
        // revision to both pass validation and overwrite one another.
        let _guard = self.write_lock.lock().await;
        if self.read_only.load(Ordering::SeqCst) {
            return Err(anyhow!(
                "system catalog '{}' became read-only; route model registry mutation to the \
                 owning pod",
                self.name
            ));
        }
        let mut schema = self.require_table(identifier)?;
        let mut asset = schema
            .mlops_asset_as_typed()?
            .ok_or_else(|| anyhow!("Catalog object '{}' is not an MLOps asset", identifier))?;
        asset
            .apply_model_mutation(expected_revision, mutation)
            .map_err(anyhow::Error::new)?;
        asset.validate().map_err(anyhow::Error::new)?;
        schema.set_mlops_asset_typed(asset);
        schema.updated_at_ms = now_millis();
        self.commit_batch_locked(vec![CatalogDelta::UpsertTable {
            identifier: identifier.clone(),
            schema: Box::new(schema.clone()),
        }])
        .await?;
        Ok(schema)
    }

    /// Take a final snapshot on graceful shutdown so the next restart replays an
    /// empty WAL tail. Best-effort: a failed checkpoint is not fatal to close
    /// (the durable WAL alone still recovers correctly).
    async fn close(&self) -> Result<()> {
        if let Err(e) = self.checkpoint().await {
            tracing::warn!(error = %e, "SystemCatalog: checkpoint on close failed (recoverable from WAL)");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::mlops::{
        CatalogMlopsAsset, CatalogModelRegistryMutation, CatalogModelUsePolicy,
    };
    use proximadb_catalog::{
        CatalogColumn, CatalogEmbeddingConfig, CatalogIndexType, CatalogMlopsAssetExt,
    };
    use proximadb_data_model::ProximaType;

    async fn catalog(dir: &std::path::Path) -> SystemCatalog {
        SystemCatalog::open("default", dir.join("catalog.wal"))
            .await
            .expect("open system catalog")
    }

    fn nslevels(levels: &[&str]) -> Vec<String> {
        levels.iter().map(|s| s.to_string()).collect()
    }

    fn vec_schema(name: &str) -> CatalogTableSchema {
        CatalogTableSchema::new(name)
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_column(CatalogColumn::new(2, "body", ProximaType::String))
            .with_primary_key(vec!["id".to_string()])
    }

    fn model_schema() -> Result<CatalogTableSchema> {
        let asset: CatalogMlopsAsset = serde_json::from_str(include_str!(
            "../../clients/python/tests/fixtures/embedding_model_xcatalog_asset.json"
        ))?;
        CatalogTableSchema::new("search-embedding").with_mlops_asset(asset)
    }

    /// TD-CAT-7: the DEFAULT backend must declare itself an identity authority.
    /// The probe is what lets a caller tell "this catalog cannot mint" from
    /// "the catalog that can, didn't" — the conflation that left relational
    /// ABAC structurally inert (TD-AUTHZ-2) without anything failing.
    #[tokio::test]
    async fn system_catalog_declares_itself_an_identity_authority() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        assert!(
            cat.identity_authority().is_some(),
            "SystemCatalog owns the WAL-backed allocator and the account registry"
        );
        Ok(())
    }

    /// TD-CAT-9: the default backend now mints a namespace `object_id` from the
    /// same system-wide sequence as tables — `NativeCatalog` always did, and the
    /// two report the same `catalog_type`, so the divergence was a trap for the
    /// first consumer to key on it.
    #[tokio::test]
    async fn namespaces_carry_a_minted_object_id() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        let levels = nslevels(&["cat9_ns"]);
        let created = cat.create_namespace(&levels, HashMap::new()).await?;
        let object_id = created
            .object_id
            .expect("an identity authority mints a namespace object_id");

        assert_eq!(
            cat.get_namespace_by_object_id(object_id).await?,
            Some(levels),
            "the reverse lookup must resolve an id this catalog itself minted"
        );
        assert_eq!(
            cat.get_namespace_by_object_id(object_id + 9_999).await?,
            None,
            "an unknown id is still None — the honest one"
        );
        Ok(())
    }

    /// TD-CAT-9: the sibling of `object_ids_are_never_reissued_across_a_restart`
    /// (TD-AUTHZ-2, below) for **namespace** ids — the half that test cannot
    /// reach, because it only ever mints tables.
    ///
    /// Minting namespace ids without folding them into the recovery floor is
    /// strictly worse than not minting at all: the floor would recover from
    /// tables only, land *below* a live namespace's id, and hand it to the next
    /// object. Object ids key policy bindings and grants, so that is a new table
    /// inheriting a dead namespace's permissions.
    ///
    /// The ordering is load-bearing — the LAST id minted before the restart is a
    /// namespace's. Neuter the namespace fold in
    /// `SystemCatalogState::max_object_id` and this fails; with the namespace
    /// created before the table it would pass either way, which is the shape of
    /// a test that proves nothing.
    #[tokio::test]
    async fn namespace_object_ids_are_never_reissued_across_a_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");

        let last_ns_object_id = {
            let cat = SystemCatalog::open("default", &wal).await?;
            let first = nslevels(&["cat9_a"]);
            cat.create_namespace(&first, HashMap::new()).await?;
            cat.create_table(&TableIdentifier::new(first, "t"), vec_schema("t"))
                .await?;
            // Minted last, so it holds the persisted high-water mark.
            cat.create_namespace(&nslevels(&["cat9_b"]), HashMap::new())
                .await?
                .object_id
                .expect("namespace object_id")
        };

        let reopened = SystemCatalog::open("default", &wal).await?;
        let fresh = reopened
            .create_table(
                &TableIdentifier::new(nslevels(&["cat9_a"]), "t2"),
                vec_schema("t2"),
            )
            .await?
            .object_id
            .expect("table object_id");

        assert!(
            fresh > last_ns_object_id,
            "object id {fresh} was re-issued at or below the persisted namespace id \
             {last_ns_object_id}; the recovery floor does not fold namespace ids"
        );
        Ok(())
    }

    /// TD-CAT-7 post-condition, end to end on the real backend: a table created
    /// through the sealed `create_table` carries the id authorization keys on.
    #[tokio::test]
    async fn created_tables_always_carry_an_object_id() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        let levels = nslevels(&["cat7_oid"]);
        cat.create_namespace(&levels, HashMap::new()).await?;

        let created = cat
            .create_table(&TableIdentifier::new(levels, "t"), vec_schema("t"))
            .await?;
        assert!(
            created.object_id.is_some(),
            "the sealed create_table must not return an unminted row"
        );
        Ok(())
    }

    #[tokio::test]
    async fn tenant_stable_ids_are_durable_and_allocator_floor_recovers() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");

        let cat = SystemCatalog::open("default", &wal).await?;
        let default_id = cat
            .account_id_u32(proximadb_tenant::DEFAULT_TENANT)
            .await?
            .expect("default tenant must mint");
        assert_eq!(default_id, 1);
        drop(cat);

        let reopened = SystemCatalog::open("default", &wal).await?;
        assert_eq!(
            reopened.account_id_u32_lookup(proximadb_tenant::DEFAULT_TENANT),
            Some(default_id),
            "WAL replay must restore the policy key"
        );
        assert_eq!(
            reopened.account_id_u32("acme").await?,
            Some(2),
            "a new tenant must mint above the replayed floor"
        );
        Ok(())
    }

    /// TD-AUTHZ-2 THE INVARIANT THAT MATTERS: an object_id is never re-issued
    /// across a restart.
    ///
    /// Policy bindings and grants are keyed on object_id, so handing a NEW table
    /// a DEAD table's id would hand it that table's permissions. The floor is
    /// derived from recovered state (`max_object_id`), so WAL replay must lift
    /// it above everything already minted.
    #[tokio::test]
    async fn object_ids_are_never_reissued_across_a_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let ns = nslevels(&["t"]);

        let first_id = {
            let cat = SystemCatalog::open("default", &wal).await?;
            cat.create_namespace(&ns, std::collections::HashMap::new())
                .await?;
            let a = cat
                .create_table(&TableIdentifier::new(ns.clone(), "a"), vec_schema("a"))
                .await?;
            let b = cat
                .create_table(&TableIdentifier::new(ns.clone(), "b"), vec_schema("b"))
                .await?;
            let (ida, idb) = (
                a.object_id.expect("table a minted"),
                b.object_id.expect("table b minted"),
            );
            assert!(idb > ida, "monotonic within a run: {idb} > {ida}");
            idb
        };

        // Restart: rebuild from the same WAL.
        let reopened = SystemCatalog::open("default", &wal).await?;
        let c = reopened
            .create_table(&TableIdentifier::new(ns.clone(), "c"), vec_schema("c"))
            .await?;
        let idc = c.object_id.expect("table c minted after restart");
        assert!(
            idc > first_id,
            "an object_id minted after restart ({idc}) must be above every id \
             minted before it ({first_id}) — re-issuing one would grant a new \
             table the policy bindings and grants of a dead one"
        );
        Ok(())
    }

    /// A caller-supplied object_id is ADOPTED, not overwritten — this is what
    /// preserves the ADR-031 unification where a vector collection pre-sets
    /// `schema.object_id` from `collection.id` before create_table runs.
    /// Adoption must also raise the floor, so the next allocation cannot collide.
    #[tokio::test]
    async fn a_caller_supplied_object_id_is_adopted_and_raises_the_floor() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let ns = nslevels(&["t"]);
        let cat = catalog(dir.path()).await;
        cat.create_namespace(&ns, std::collections::HashMap::new())
            .await?;

        let mut preset = vec_schema("preset");
        preset.object_id = Some(5_000);
        let adopted = cat
            .create_table(&TableIdentifier::new(ns.clone(), "preset"), preset)
            .await?;
        assert_eq!(
            adopted.object_id,
            Some(5_000),
            "a pre-set id must be KEPT — overwriting it splits one collection \
             into two identities and detaches any policy bound to the first"
        );

        let next = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "next"),
                vec_schema("next"),
            )
            .await?;
        assert!(
            next.object_id.expect("minted") > 5_000,
            "adoption must raise the floor so the next allocation cannot collide"
        );
        Ok(())
    }

    /// The read-back invariant, for the implementation that ACTUALLY SERVES.
    /// #1559 proved this for NativeCatalog — which the server does not register,
    /// so the invariant went unverified where it mattered.
    #[tokio::test]
    async fn object_id_survives_the_get_table_round_trip_on_system_catalog() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let ns = nslevels(&["t"]);
        let cat = catalog(dir.path()).await;
        cat.create_namespace(&ns, std::collections::HashMap::new())
            .await?;
        let ident = TableIdentifier::new(ns.clone(), "rt");
        let created = cat.create_table(&ident, vec_schema("rt")).await?;
        let minted = created.object_id.expect("create_table mints");

        let fetched = cat.get_table(&ident).await?;
        assert_eq!(
            fetched.object_id,
            Some(minted),
            "xcatalog.tables projects this value; an empty one is what \
             TD-AUTHZ-2 reports and what makes relational ABAC deny"
        );
        Ok(())
    }

    #[tokio::test]
    async fn model_asset_id_resolution_and_mutation_survive_system_catalog_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let namespace = nslevels(&["tenant-a", "mlops"]);
        let identifier = TableIdentifier::new(namespace.clone(), "search-embedding");

        let asset_id = {
            let cat = SystemCatalog::open("default", &wal).await?;
            cat.create_namespace(&namespace, HashMap::new()).await?;
            let created = cat.create_table(&identifier, model_schema()?).await?;
            let asset_id = created.object_id.context("model asset id")?;
            assert_eq!(
                cat.get_table_by_object_id(asset_id).await?,
                Some(identifier.clone())
            );

            cat.apply_model_registry_mutation(
                &identifier,
                0,
                CatalogModelRegistryMutation::set_alias("champion", 1),
            )
            .await?;
            asset_id
        };

        let reopened = SystemCatalog::open("default", &wal).await?;
        assert_eq!(
            reopened.get_table_by_object_id(asset_id).await?,
            Some(identifier.clone()),
            "stable asset lookup must be rebuilt from the WAL"
        );
        let schema = reopened.get_table(&identifier).await?;
        let CatalogMlopsAsset::EmbeddingModel(registry) = schema
            .mlops_asset_as_typed()?
            .context("typed MLOps asset")?;
        assert_eq!(registry.revision, 1);
        assert_eq!(registry.resolve_alias("champion")?.version, 1);

        let model = registry.version(1)?;
        let binding = CatalogEmbeddingConfig::pinned(
            "search-embedding",
            768,
            asset_id,
            1,
            model.contract_sha256()?,
        )?;
        let resolved = reopened
            .resolve_embedding_model_binding(&binding, &CatalogModelUsePolicy::registration_only())
            .await?;
        assert_eq!(resolved.asset_id, asset_id);
        assert_eq!(resolved.model.version, 1);
        Ok(())
    }

    #[tokio::test]
    async fn model_registry_revision_is_an_atomic_compare_and_swap_on_system_catalog() -> Result<()>
    {
        let dir = tempfile::tempdir()?;
        let namespace = nslevels(&["tenant-a", "mlops"]);
        let identifier = TableIdentifier::new(namespace.clone(), "search-embedding");
        let cat = Arc::new(catalog(dir.path()).await);
        cat.create_namespace(&namespace, HashMap::new()).await?;
        cat.create_table(&identifier, model_schema()?).await?;

        let left = cat.apply_model_registry_mutation(
            &identifier,
            0,
            CatalogModelRegistryMutation::set_alias("left", 1),
        );
        let right = cat.apply_model_registry_mutation(
            &identifier,
            0,
            CatalogModelRegistryMutation::set_alias("right", 1),
        );
        let (left, right) = tokio::join!(left, right);
        assert_ne!(
            left.is_ok(),
            right.is_ok(),
            "exactly one stale writer must win"
        );

        let schema = cat.get_table(&identifier).await?;
        let CatalogMlopsAsset::EmbeddingModel(registry) = schema
            .mlops_asset_as_typed()?
            .context("typed MLOps asset")?;
        assert_eq!(registry.revision, 1);
        assert_eq!(registry.aliases.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_first_mint_returns_one_id_for_one_tenant() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = Arc::new(catalog(dir.path()).await);
        let (left, right) = tokio::join!(cat.account_id_u32("acme"), cat.account_id_u32("acme"));
        let left = left?;
        let right = right?;
        assert_eq!(left, right);
        assert_eq!(cat.account_id_u32_lookup("acme"), left);
        Ok(())
    }

    #[tokio::test]
    async fn tenant_tables_receive_durable_total_object_and_typed_identity() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let namespace = nslevels(&["acme", "default"]);
        let first = TableIdentifier::new(namespace.clone(), "orders");
        let second = TableIdentifier::new(namespace.clone(), "customers");

        let cat = SystemCatalog::open("default", &wal).await?;
        cat.create_namespace_for_tenant(&namespace, HashMap::new(), Some("acme"))
            .await?;
        let orders = cat.create_table(&first, vec_schema("orders")).await?;
        let customers = cat.create_table(&second, vec_schema("customers")).await?;

        assert!(orders.object_id.is_some());
        assert!(
            orders
                .columns
                .iter()
                .all(|column| column.object_id.is_some())
        );
        assert_eq!(orders.stable_namespace_id, customers.stable_namespace_id);
        assert!(orders.stable_namespace_id.is_some());
        assert_ne!(orders.stable_collection_id, customers.stable_collection_id);
        let prior_max_object = cat.max_object_id().await?.expect("object ids minted");
        let stable_namespace = orders.stable_namespace_id;
        let prior_collection = customers.stable_collection_id.expect("typed collection id");
        drop(cat);

        let reopened = SystemCatalog::open("default", &wal).await?;
        let third = TableIdentifier::new(namespace, "invoices");
        let invoices = reopened
            .create_table(&third, vec_schema("invoices"))
            .await?;
        assert_eq!(invoices.stable_namespace_id, stable_namespace);
        assert!(
            invoices
                .stable_collection_id
                .is_some_and(|id| id > prior_collection)
        );
        assert!(invoices.object_id.is_some_and(|id| id > prior_max_object));
        Ok(())
    }

    #[tokio::test]
    async fn namespace_and_table_crud() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;

        let ns = cat
            .create_namespace(&nslevels(&["sales"]), HashMap::new())
            .await?;
        assert!(ns.namespace_id.is_some());
        assert!(cat.namespace_exists(&nslevels(&["sales"])).await?);

        cat.create_table(
            &TableIdentifier::new(nslevels(&["sales"]), "orders"),
            vec_schema("orders"),
        )
        .await?;
        assert!(
            cat.table_exists(&TableIdentifier::new(nslevels(&["sales"]), "orders"))
                .await?
        );
        let got = cat
            .get_table(&TableIdentifier::new(nslevels(&["sales"]), "orders"))
            .await?;
        assert_eq!(got.name, "orders");
        assert_eq!(cat.list_tables(&nslevels(&["sales"])).await?.len(), 1);

        // duplicate + missing-namespace are rejected
        assert!(
            cat.create_table(
                &TableIdentifier::new(nslevels(&["sales"]), "orders"),
                vec_schema("orders")
            )
            .await
            .is_err()
        );
        assert!(
            cat.create_table(
                &TableIdentifier::new(nslevels(&["nope"]), "x"),
                vec_schema("x")
            )
            .await
            .is_err()
        );

        assert!(
            cat.drop_table(&TableIdentifier::new(nslevels(&["sales"]), "orders"), false)
                .await?
        );
        assert!(
            !cat.table_exists(&TableIdentifier::new(nslevels(&["sales"]), "orders"))
                .await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_statistics_primary_pod_and_layouts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        cat.create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        let id = TableIdentifier::new(nslevels(&["s"]), "t");
        cat.create_table(&id, vec_schema("t")).await?;

        // index: create on a real column, reject duplicate + unknown column
        let idx = CatalogIndex::new("by_body", vec!["body".to_string()], CatalogIndexType::BTree);
        cat.create_index(&id, idx.clone()).await?;
        assert_eq!(cat.list_indexes(&id).await?.len(), 1);
        assert!(cat.create_index(&id, idx).await.is_err());
        let bad = CatalogIndex::new("bad", vec!["ghost".to_string()], CatalogIndexType::BTree);
        assert!(cat.create_index(&id, bad).await.is_err());
        assert!(cat.drop_index(&id, "by_body").await?);
        assert!(!cat.drop_index(&id, "by_body").await?);

        // statistics default then round-trip
        assert_eq!(cat.get_statistics(&id).await?.row_count, 0);
        let mut stats = CatalogTableStatistics::default();
        stats.row_count = 42;
        cat.update_statistics(&id, stats).await?;
        assert_eq!(cat.get_statistics(&id).await?.row_count, 42);

        // primary pod + storage layouts persist on the schema
        cat.set_primary_pod(
            &id,
            Some(proximadb_catalog::CatalogPrimaryPod::now(
                "pod-a",
                proximadb_catalog::CatalogPrimaryPodReason::Create,
            )),
        )
        .await?;
        assert_eq!(cat.get_table(&id).await?.primary_pod.unwrap().pod, "pod-a");

        let updated = cat
            .set_storage_layouts(&id, vec![CatalogStorageLayout::default()])
            .await?;
        assert!(!updated.storage_layouts.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rename_drop_namespace_cascade() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        cat.create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        cat.create_table(
            &TableIdentifier::new(nslevels(&["s"]), "a"),
            vec_schema("a"),
        )
        .await?;

        cat.rename_table(
            &TableIdentifier::new(nslevels(&["s"]), "a"),
            &TableIdentifier::new(nslevels(&["s"]), "b"),
        )
        .await?;
        assert!(
            !cat.table_exists(&TableIdentifier::new(nslevels(&["s"]), "a"))
                .await?
        );
        assert_eq!(
            cat.get_table(&TableIdentifier::new(nslevels(&["s"]), "b"))
                .await?
                .name,
            "b"
        );

        // non-cascade drop of a populated namespace fails; cascade succeeds
        assert!(cat.drop_namespace(&nslevels(&["s"]), false).await.is_err());
        assert!(cat.drop_namespace(&nslevels(&["s"]), true).await?);
        assert!(!cat.namespace_exists(&nslevels(&["s"])).await?);
        assert!(cat.list_tables(&nslevels(&["s"])).await?.is_empty());
        Ok(())
    }

    /// The whole point: state survives a process restart by replaying the WAL,
    /// with zero filesystem stats on the read path.
    #[tokio::test]
    async fn persists_across_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        {
            let cat = catalog(dir.path()).await;
            cat.create_namespace(&nslevels(&["db"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["db"]), "users"),
                vec_schema("users"),
            )
            .await?;
            cat.set_primary_pod(
                &TableIdentifier::new(nslevels(&["db"]), "users"),
                Some(proximadb_catalog::CatalogPrimaryPod::now(
                    "pod-x",
                    proximadb_catalog::CatalogPrimaryPodReason::Create,
                )),
            )
            .await?;
        }
        // Fresh catalog over the same WAL file = a "restart".
        let reopened = catalog(dir.path()).await;
        assert!(reopened.namespace_exists(&nslevels(&["db"])).await?);
        let users = reopened
            .get_table(&TableIdentifier::new(nslevels(&["db"]), "users"))
            .await?;
        assert_eq!(users.name, "users");
        assert_eq!(users.primary_pod.unwrap().pod, "pod-x");
        Ok(())
    }

    // ── Phase 3: snapshot + WAL compaction (bounded restart) ──────────────

    use crate::services::canonical_wal::FramedTableWalAppender;

    fn wal_path(dir: &std::path::Path) -> std::path::PathBuf {
        dir.join("catalog.wal")
    }
    fn snapshot_path(dir: &std::path::Path) -> std::path::PathBuf {
        dir.join("catalog.snapshot")
    }

    async fn wal_entry_count(dir: &std::path::Path) -> usize {
        FramedTableWalAppender::read_entries_from_path(wal_path(dir))
            .await
            .map(|e| e.len())
            .unwrap_or(0)
    }

    /// An explicit checkpoint writes a durable snapshot and compacts the WAL to
    /// drop everything the snapshot covers; later mutations land as a short
    /// tail, and a reopen restores snapshot + tail. This is the bounded-restart
    /// guarantee: the WAL stops growing without bound.
    #[tokio::test]
    async fn snapshot_compacts_wal_and_survives_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        {
            let cat = catalog(dir.path()).await;
            cat.create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t1"),
                vec_schema("t1"),
            )
            .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t2"),
                vec_schema("t2"),
            )
            .await?;
            assert_eq!(wal_entry_count(dir.path()).await, 3);

            cat.checkpoint().await?; // snapshot covers all 3; WAL compacted to empty
            assert!(snapshot_path(dir.path()).exists());
            assert_eq!(wal_entry_count(dir.path()).await, 0);

            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t3"),
                vec_schema("t3"),
            )
            .await?;
            // Only the post-snapshot tail remains on the WAL.
            assert_eq!(wal_entry_count(dir.path()).await, 1);
        }
        let reopened = catalog(dir.path()).await;
        for t in ["t1", "t2", "t3"] {
            assert!(
                reopened
                    .table_exists(&TableIdentifier::new(nslevels(&["s"]), t))
                    .await?,
                "table {t} must survive snapshot+compaction+reopen"
            );
        }
        Ok(())
    }

    /// Phase 5c — the catalog snapshot persists to **object storage** while the
    /// per-DDL WAL stays local. A reopen over the same object store restores the
    /// in-RAM authority from the object-store snapshot + the local WAL tail. This
    /// is the same bounded-restart guarantee as the local case, proving the
    /// `ObjectStoreSnapshotStore` path end-to-end (over `memory://`).
    #[tokio::test]
    async fn snapshot_to_object_store_survives_reopen() -> Result<()> {
        use crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore;
        use proximadb_object_store::ProximaObjectStore;

        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        // One backing in-memory object store shared across both opens (a fresh
        // `from_url("memory://")` would be empty — we need the same instance).
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        // Phase 6a: the object-store snapshot is a generation-fenced manifest log.
        let key = "_operator/catalog/_manifests/";

        {
            let store = Arc::new(ObjectStoreSnapshotStore::new(
                ProximaObjectStore::new(backing.clone()),
                "memory:///",
                key,
            ));
            let cat = SystemCatalog::open_with_snapshot_store("default", &wal, store).await?;
            cat.create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t1"),
                vec_schema("t1"),
            )
            .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t2"),
                vec_schema("t2"),
            )
            .await?;

            // Checkpoint pushes the snapshot to the object store and compacts the
            // local WAL to empty.
            cat.checkpoint().await?;
            assert_eq!(wal_entry_count(dir.path()).await, 0);

            // A post-snapshot DDL on an object-store catalog is acknowledged only
            // after it publishes its own snapshot (TD-OBJSTORE-2: the local WAL is
            // a staging log that does not survive VM/pod replacement, so the
            // snapshot must carry every committed DDL). That publish compacts the
            // local WAL tail back to empty — the DDL now lives in the snapshot.
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t3"),
                vec_schema("t3"),
            )
            .await?;
            assert_eq!(wal_entry_count(dir.path()).await, 0);
        }

        // Reopen with a new store handle over the SAME backing object store and
        // the same local WAL → restore = object-store snapshot + WAL tail.
        let store2 = Arc::new(ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(backing.clone()),
            "memory:///",
            key,
        ));
        let reopened = SystemCatalog::open_with_snapshot_store("default", &wal, store2).await?;
        for t in ["t1", "t2", "t3"] {
            assert!(
                reopened
                    .table_exists(&TableIdentifier::new(nslevels(&["s"]), t))
                    .await?,
                "table {t} must survive object-store snapshot + WAL tail + reopen"
            );
        }
        Ok(())
    }

    /// TD-OBJSTORE-2 regression: one collection-sized DDL batch must survive a
    /// restart onto a fresh local volume without an explicit checkpoint. Before
    /// this fix the default threshold (1,000 mutations) left these mutations
    /// only in the local catalog WAL, so a VM replacement recovered an empty
    /// catalog even though the data WAL was durable in object storage.
    #[tokio::test]
    async fn object_store_commit_survives_fresh_local_wal_without_checkpoint() -> Result<()> {
        use crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore;
        use proximadb_object_store::ProximaObjectStore;

        let first_vm = tempfile::tempdir()?;
        let replacement_vm = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let key = "_operator/catalog/_manifests/";

        {
            let store = Arc::new(ObjectStoreSnapshotStore::new(
                ProximaObjectStore::new(backing.clone()),
                "memory:///",
                key,
            ));
            let cat = SystemCatalog::open_with_snapshot_store(
                "default",
                first_vm.path().join("catalog.wal"),
                store,
            )
            .await?;
            cat.create_namespace(&nslevels(&["default"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["default"]), "durable_collection"),
                vec_schema("durable_collection"),
            )
            .await?;
            // No checkpoint and no graceful shutdown: model abrupt VM loss.
        }

        let store = Arc::new(ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(backing),
            "memory:///",
            key,
        ));
        let reopened = SystemCatalog::open_with_snapshot_store(
            "default",
            replacement_vm.path().join("catalog.wal"),
            store,
        )
        .await?;
        assert!(
            reopened
                .table_exists(&TableIdentifier::new(
                    nslevels(&["default"]),
                    "durable_collection"
                ))
                .await?,
            "catalog DDL must be present when the replacement VM has no local WAL"
        );
        Ok(())
    }

    /// Crash injection — crashed *after* the snapshot rename but *before* (or
    /// during) WAL compaction. We reproduce that exact on-disk state: take a
    /// checkpoint, then restore the pre-compaction WAL underneath the new
    /// snapshot. Recovery must replay the now-covered entries idempotently — no
    /// lost DDL, no duplicates.
    #[tokio::test]
    async fn crash_after_snapshot_before_compaction_is_idempotent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // Capture the full, uncompacted WAL (ns + t1 + t2).
        {
            let cat = catalog(dir.path()).await;
            cat.create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t1"),
                vec_schema("t1"),
            )
            .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t2"),
                vec_schema("t2"),
            )
            .await?;
        }
        let full_wal = tokio::fs::read(wal_path(dir.path())).await?;

        // Checkpoint (snapshot + compact-to-empty), then simulate the crash by
        // restoring the full WAL on top of the fresh snapshot.
        {
            let cat = catalog(dir.path()).await;
            cat.checkpoint().await?;
            assert_eq!(wal_entry_count(dir.path()).await, 0);
        }
        tokio::fs::write(wal_path(dir.path()), &full_wal).await?;
        assert_eq!(wal_entry_count(dir.path()).await, 3); // snapshot + full WAL both present

        let reopened = catalog(dir.path()).await;
        assert!(reopened.namespace_exists(&nslevels(&["s"])).await?);
        for t in ["t1", "t2"] {
            assert!(
                reopened
                    .table_exists(&TableIdentifier::new(nslevels(&["s"]), t))
                    .await?
            );
        }
        // Exactly two tables — the covered entries were skipped, not re-created.
        assert_eq!(reopened.list_tables(&nslevels(&["s"])).await?.len(), 2);
        Ok(())
    }

    /// A leftover compaction temp file (crash mid-rewrite, before the atomic
    /// rename) must be ignored: the live WAL is untouched, so recovery is
    /// unaffected.
    #[tokio::test]
    async fn leftover_compaction_temp_is_ignored() -> Result<()> {
        let dir = tempfile::tempdir()?;
        {
            let cat = catalog(dir.path()).await;
            cat.create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            cat.create_table(
                &TableIdentifier::new(nslevels(&["s"]), "t1"),
                vec_schema("t1"),
            )
            .await?;
        }
        // Garbage temp from an interrupted compaction.
        let tmp = wal_path(dir.path()).with_extension("wal-compact-tmp");
        tokio::fs::write(&tmp, b"PXWAL001-garbage-partial-frame").await?;

        let reopened = catalog(dir.path()).await;
        assert!(reopened.namespace_exists(&nslevels(&["s"])).await?);
        assert!(
            reopened
                .table_exists(&TableIdentifier::new(nslevels(&["s"]), "t1"))
                .await?
        );
        Ok(())
    }

    /// Many mutations across several checkpoints converge to the correct final
    /// state on reopen, and the WAL stays bounded (not proportional to total
    /// DDL count).
    #[tokio::test]
    async fn multiple_checkpoints_preserve_state_and_bound_wal() -> Result<()> {
        let dir = tempfile::tempdir()?;
        {
            let cat = catalog(dir.path()).await;
            cat.create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            for round in 0..5 {
                for i in 0..4 {
                    let name = format!("t_{round}_{i}");
                    cat.create_table(
                        &TableIdentifier::new(nslevels(&["s"]), &name),
                        vec_schema(&name),
                    )
                    .await?;
                }
                cat.checkpoint().await?;
            }
            // Drop half of them, checkpoint again.
            for round in 0..5 {
                cat.drop_table(
                    &TableIdentifier::new(nslevels(&["s"]), format!("t_{round}_0")),
                    false,
                )
                .await?;
            }
            cat.checkpoint().await?;
            // WAL is bounded (empty right after a checkpoint), not 25-ish entries.
            assert_eq!(wal_entry_count(dir.path()).await, 0);
        }
        let reopened = catalog(dir.path()).await;
        let tables = reopened.list_tables(&nslevels(&["s"])).await?;
        assert_eq!(tables.len(), 15); // 20 created - 5 dropped
        Ok(())
    }

    // ── Phase 6b: cross-pod cache coherence (sinval) — multi-instance harness ──
    //
    // Two (or more) in-process `SystemCatalog`s over ONE shared backing object
    // store stand in for two pods. Each has its OWN local WAL; the object-store
    // snapshot is the only shared artifact, so cross-pod visibility flows through
    // it exactly as in a real deployment.

    /// Open an owner catalog whose snapshot lives in the shared `backing` store.
    async fn open_owner(
        wal: &std::path::Path,
        backing: &Arc<dyn object_store::ObjectStore>,
    ) -> Result<SystemCatalog> {
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                "memory:///",
                "_operator/catalog/_manifests/",
            ),
        );
        SystemCatalog::open_with_snapshot_store("default", wal, store).await
    }

    /// Open a read-only follower catalog over the shared `backing` store.
    async fn open_follower(
        wal: &std::path::Path,
        backing: &Arc<dyn object_store::ObjectStore>,
    ) -> Result<SystemCatalog> {
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                "memory:///",
                "_operator/catalog/_manifests/",
            ),
        );
        SystemCatalog::open_follower("default", wal, store).await
    }

    fn tid(name: &str) -> TableIdentifier {
        TableIdentifier::new(nslevels(&["s"]), name)
    }

    /// A read-only **follower** tails the owner's published snapshot: it sees the
    /// owner's committed DDL after a `reload_if_stale`, within the staleness
    /// bound, and rejects any DDL of its own (route writes to the owner).
    #[tokio::test]
    async fn follower_tails_owner_and_rejects_writes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());

        // Owner publishes a first snapshot (ns + t1).
        let owner = open_owner(&dir.path().join("owner.wal"), &backing).await?;
        owner
            .create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        assert_eq!(owner.account_id_u32("tenant-one").await?, Some(1));
        owner.create_table(&tid("t1"), vec_schema("t1")).await?;
        owner.checkpoint().await?;

        // Follower boots from that snapshot — read-only, already sees t1.
        let follower = open_follower(&dir.path().join("follower.wal"), &backing).await?;
        assert!(follower.is_read_only());
        assert!(follower.table_exists(&tid("t1")).await?);
        assert!(!follower.table_exists(&tid("t2")).await?);

        // A follower must reject DDL — it has no write authority.
        assert!(
            follower
                .create_table(&tid("nope"), vec_schema("nope"))
                .await
                .is_err(),
            "follower must reject DDL"
        );

        // Owner commits more DDL and republishes. The follower is stale until it
        // polls; a no-publish probe is cheap and a no-op.
        owner.create_table(&tid("t2"), vec_schema("t2")).await?;
        assert_eq!(owner.account_id_u32("tenant-two").await?, Some(2));
        owner.checkpoint().await?;

        // Sinval: the follower observes the newer pointer version and reloads.
        match follower.reload_if_stale().await? {
            ReloadOutcome::Reloaded { superseded, .. } => assert!(
                !superseded,
                "a same-generation owner republish is not a takeover"
            ),
            other => panic!("expected a reload, got {other:?}"),
        }
        assert!(follower.table_exists(&tid("t2")).await?);
        assert_eq!(follower.account_id_u32_lookup("tenant-two"), Some(2));
        assert_eq!(
            follower.account_floor.load(Ordering::SeqCst),
            3,
            "snapshot reload must advance the local account allocator floor"
        );
        // Polling again with nothing new is a no-op.
        assert_eq!(follower.reload_if_stale().await?, ReloadOutcome::UpToDate);
        Ok(())
    }

    /// **Read-your-writes** on the owner: an owner's own published snapshot must
    /// not look "stale" to itself, and post-checkpoint writes that live only in
    /// RAM + the local WAL survive a sinval poll (they are never reloaded away in
    /// the absence of a newer pod).
    #[tokio::test]
    async fn owner_read_your_writes_survives_poll() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());

        let owner = open_owner(&dir.path().join("owner.wal"), &backing).await?;
        owner
            .create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        owner.create_table(&tid("t1"), vec_schema("t1")).await?;
        owner.checkpoint().await?; // publishes; loaded_version tracks our own write

        // A post-checkpoint write lives only in RAM + the local WAL (unpublished).
        owner.create_table(&tid("t2"), vec_schema("t2")).await?;

        // Polling sees our own published version — no spurious reload.
        assert_eq!(owner.reload_if_stale().await?, ReloadOutcome::UpToDate);
        assert!(!owner.is_read_only());
        // Read-your-writes: the unpublished t2 is still visible.
        assert!(owner.table_exists(&tid("t2")).await?);
        Ok(())
    }

    /// A **superseded** owner steps down on its next sinval poll: when a
    /// newer-generation pod has taken the catalog over, an old owner that has
    /// not written since the takeover learns of it lazily, reloads the newer
    /// snapshot, and flips to read-only (no lost update — the fence + step-down
    /// converge to one writer).
    ///
    /// With object-store snapshot-on-commit (TD-OBJSTORE-2) every DDL publishes,
    /// so a superseded owner fences on its *next write* rather than via a later
    /// poll — there is no "unpublished local write" window to exercise here.
    /// The poll path is therefore driven by an owner that has been silent since
    /// the takeover. (The write-time fence + discard of a doomed write is
    /// covered by `fenced_checkpoint_steps_down_and_reloads`.)
    #[tokio::test]
    async fn superseded_owner_steps_down_via_poll() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());

        // Pod A: first owner (generation 1), publishes ns + t1.
        let pod_a = open_owner(&dir.path().join("a.wal"), &backing).await?;
        pod_a
            .create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        pod_a.create_table(&tid("t1"), vec_schema("t1")).await?;
        pod_a.checkpoint().await?;
        assert_eq!(pod_a.generation(), 1);

        // Pod B: takes over (reads gen 1 → claims gen 2), publishes t2.
        let pod_b = open_owner(&dir.path().join("b.wal"), &backing).await?;
        assert_eq!(pod_b.generation(), 2);
        pod_b.create_table(&tid("t2"), vec_schema("t2")).await?;
        pod_b.checkpoint().await?;

        // Pod A has not written since the takeover, so no checkpoint-time fence
        // has fired yet — it still believes it owns the catalog.
        assert!(!pod_a.is_read_only());

        // Sinval poll on A: a higher generation owns the catalog → step down.
        match pod_a.reload_if_stale().await? {
            ReloadOutcome::Reloaded {
                generation,
                superseded,
                ..
            } => {
                assert_eq!(generation, 2);
                assert!(superseded, "a higher-generation snapshot is a takeover");
            }
            other => panic!("expected a superseding reload, got {other:?}"),
        }
        assert!(
            pod_a.is_read_only(),
            "superseded owner must step down to read-only"
        );
        // A converges to B's snapshot: sees t1 + t2.
        assert!(pod_a.table_exists(&tid("t1")).await?);
        assert!(pod_a.table_exists(&tid("t2")).await?);
        // A can no longer write.
        assert!(
            pod_a
                .create_table(&tid("t4"), vec_schema("t4"))
                .await
                .is_err()
        );
        Ok(())
    }

    /// The fenced-checkpoint **self-heal** path: even without an explicit poll, an
    /// owner that discovers it has been fenced *at checkpoint time* steps down and
    /// reloads the newer snapshot in place (it does not silently keep serving its
    /// doomed local state, and does not compact its WAL).
    #[tokio::test]
    async fn fenced_checkpoint_steps_down_and_reloads() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());

        let pod_a = open_owner(&dir.path().join("a.wal"), &backing).await?;
        pod_a
            .create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        pod_a.create_table(&tid("t1"), vec_schema("t1")).await?;
        pod_a.checkpoint().await?;

        let pod_b = open_owner(&dir.path().join("b.wal"), &backing).await?;
        pod_b.create_table(&tid("t2"), vec_schema("t2")).await?;
        pod_b.checkpoint().await?;

        // Pod A, still believing it owns the catalog, writes t3 then checkpoints —
        // the publish is FENCED (gen 1 < 2), triggering the in-place step-down.
        pod_a.create_table(&tid("t3"), vec_schema("t3")).await?;
        pod_a.checkpoint().await?;

        assert!(
            pod_a.is_read_only(),
            "a fenced owner steps down to read-only"
        );
        assert!(
            pod_a.table_exists(&tid("t2")).await?,
            "adopted B's snapshot"
        );
        assert!(
            !pod_a.table_exists(&tid("t3")).await?,
            "doomed unpublished write discarded on step-down"
        );
        Ok(())
    }

    /// Coherence is inert where it should be: an injected-appender catalog (no
    /// snapshot store) reports `Disabled`, and a local single-pod owner — whose
    /// store exposes no version — never spuriously reloads or steps down.
    #[tokio::test]
    async fn coherence_is_inert_for_local_and_injected() -> Result<()> {
        // Injected-appender catalog: no snapshot store at all → Disabled.
        let injected = SystemCatalog::new(
            "x",
            SystemCatalogState::new(),
            Arc::new(crate::services::canonical_wal::MemoryTableWalAppender::new()),
        );
        assert_eq!(injected.reload_if_stale().await?, ReloadOutcome::Disabled);

        // Local single-pod owner: the store has no version concept → the cheap
        // probe returns None and the poll is always UpToDate, never read-only.
        let dir = tempfile::tempdir()?;
        {
            let local = catalog(dir.path()).await;
            local
                .create_namespace(&nslevels(&["s"]), HashMap::new())
                .await?;
            local.create_table(&tid("t1"), vec_schema("t1")).await?;
            local.checkpoint().await?; // snapshot watermark advances; WAL → empty
            assert_eq!(wal_entry_count(dir.path()).await, 0);
        }
        // Reopen onto the empty (compacted) WAL + the snapshot: the appender's
        // file-derived sequence floor is 0, below the snapshot watermark. Seeding
        // the floor from the watermark means a fresh write still lands above it.
        let local = catalog(dir.path()).await;
        assert_eq!(local.reload_if_stale().await?, ReloadOutcome::UpToDate);
        assert!(!local.is_read_only());
        local.create_table(&tid("t2"), vec_schema("t2")).await?;
        assert!(
            local.table_exists(&tid("t2")).await?,
            "a write after a compact-to-empty reopen must apply (sequence-floor seeded)"
        );
        Ok(())
    }
}
