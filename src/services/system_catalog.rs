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

use proximadb_kernel::time::now_unix_ms;

/// Default number of committed mutations between automatic snapshots; mirrors
/// the legacy metastore's compaction threshold. Override with
/// `PROXIMADB_CATALOG_SNAPSHOT_THRESHOLD`.
const DEFAULT_SNAPSHOT_THRESHOLD: u64 = 1000;

/// Sentinel for "no object-store snapshot version observed yet" — distinct from
/// version `0`, the first real published version. Versions never reach `u64::MAX`.
const NO_VERSION: u64 = u64::MAX;

/// TD-SPINE-2 (ADR-098 D1.3) kill switch: disable the per-account sharded
/// catalog-snapshot publish/read path and revert to the legacy whole-image
/// snapshot log. Unset = sharded (the shipped default). This is the RISK
/// escape hatch the pre-GA policy allows (no backward-compat gates, but a
/// runtime off-switch for the new publish path): with it set, the catalog
/// publishes the whole image to the legacy `_manifests/` log and tails THAT
/// log, never the root. On a deployment that already published root+shards,
/// an escaping pod still ASSEMBLES the sharded image once at boot (so no
/// data is lost) and then re-seeds the legacy log; mixed ON/OFF fleets
/// diverge by design — the escape is fleet-wide, documented in TD-SPINE-2.
/// Non-memoized: `#[serial]` tests flip it mid-process.
fn sharded_catalog_snapshots_enabled() -> bool {
    !proximadb_config::env_flag_now("PROXIMADB_CATALOG_SHARDED_SNAPSHOT_DISABLE")
}

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

/// TD-SPINE-2 (ADR-098 D1.4): the stable namespace-id allocator — the durable
/// `(account, namespace) → id` registry plus the **per-account** allocation
/// floor for NEW ids. Existing ids are never renumbered; two accounts may
/// each mint id 1 (the typed id scopes an `accounts/{a}/{n}/…` path, so
/// uniqueness is per-account by design — the schema always documented it
/// that way; the allocator had been one global counter, making the u16
/// bound fleet-wide). One lock guards both maps so check-and-mint is atomic.
#[derive(Debug, Default)]
struct NamespaceIdAllocator {
    /// (account, namespace key) → stable id. Durable schemas are the
    /// authority; rebuilt from them after every replay/snapshot/shard load.
    ids: HashMap<(String, String), u16>,
    /// account → next id to mint (floor sits above every id the account ever
    /// minted, dropped ones included, via the per-account high-water).
    floors: HashMap<String, u64>,
}

impl NamespaceIdAllocator {
    /// Mint the next id for `account`, or error when the u16 space is spent.
    fn allocate(&mut self, account: &str) -> Result<u16> {
        let floor = self.floors.entry(account.to_string()).or_insert(1);
        if *floor > u64::from(u16::MAX) {
            return Err(anyhow!(
                "stable namespace id space exhausted for account '{account}'"
            ));
        }
        let id = u16::try_from(*floor)
            .map_err(|_| anyhow!("stable namespace id space exhausted for account '{account}'"))?;
        *floor += 1;
        Ok(id)
    }

    /// Raise an account's floor above `id` (adopt-or-allocate semantics).
    fn raise_floor(&mut self, account: &str, id: u16) {
        let floor = self.floors.entry(account.to_string()).or_insert(1);
        *floor = (*floor).max(u64::from(id) + 1);
    }
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
    /// Next stable collection id. The typed id is narrower than object ids,
    /// so a u64 floor lets allocation detect exhaustion before casts.
    /// (Collection ids stay a GLOBAL sequence — TD-SPINE-2 scopes only the
    /// namespace floor per account.)
    collection_floor: AtomicU64,
    /// TD-SPINE-2: stable namespace id per (account, logical namespace) +
    /// the per-account allocation floor. Durable schemas are the authority;
    /// both maps are rebuilt from them after replay/snapshot/shard loads.
    namespace_alloc: parking_lot::RwLock<NamespaceIdAllocator>,
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
    /// TD-SPINE-2: the shard-log versions this pod's in-RAM authority
    /// reflects (the local copy of the root's `account → version` index).
    /// Diffed against each published root to adopt ONLY changed shards.
    shard_versions: parking_lot::RwLock<HashMap<String, u64>>,
    /// TD-SPINE-2: accounts touched by commits since the last sharded
    /// publish — exactly the shards the next publish must write, which is
    /// what makes bytes-per-DDL O(account) instead of O(fleet).
    pending_shards: parking_lot::Mutex<std::collections::HashSet<String>>,
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
        let (object_floor, collection_floor, namespace_alloc) =
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
            collection_floor: AtomicU64::new(collection_floor),
            namespace_alloc: parking_lot::RwLock::new(namespace_alloc),
            appender,
            write_lock: tokio::sync::Mutex::new(()),
            snapshot: None,
            commits_since_snapshot: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            loaded_version: AtomicU64::new(NO_VERSION),
            loaded_generation: AtomicU64::new(0),
            read_only: AtomicBool::new(false),
            shard_versions: parking_lot::RwLock::new(HashMap::new()),
            pending_shards: parking_lot::Mutex::new(std::collections::HashSet::new()),
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
    /// TD-SPINE-2: the namespace-id floor is seeded **per account** — each
    /// account's floor sits above every id THAT account ever minted (dropped
    /// ones included, via the per-account high-water), and ids other
    /// accounts minted do not reserve anything for it. Existing ids are
    /// never renumbered; they simply stay owned by their account.
    fn identity_allocator_seed(state: &SystemCatalogState) -> (u64, u64, NamespaceIdAllocator) {
        let object_floor = state.max_object_id().map_or(1, |id| id.saturating_add(1));
        let mut collection_floor = 1_u64;
        let mut namespace_alloc = NamespaceIdAllocator::default();
        for (identifier, schema) in state.table_entries() {
            if let Some(namespace_id) = schema.stable_namespace_id {
                if let Some(account) = state
                    .get_namespace(&identifier.namespace)
                    .and_then(|namespace| namespace.tenant_id)
                {
                    namespace_alloc
                        .ids
                        .insert((account, identifier.namespace.join(".")), namespace_id);
                }
            }
            if let Some(collection_id) = schema.stable_collection_id {
                collection_floor = collection_floor.max(u64::from(collection_id) + 1);
            }
        }
        // TD-PATH-2: above every typed collection id ever minted, dropped
        // ones too (global sequence — unchanged).
        if let Some(collection_id) = state.max_collection_id_ever() {
            collection_floor = collection_floor.max(u64::from(collection_id) + 1);
        }
        // TD-SPINE-2: each account's floor sits above its own high-water
        // (live rows folded above, dropped ids from the per-account
        // high-water record that shards and WAL replay rebuild). The
        // high-water map is the authority for the floor precisely because it
        // remembers accounts with NO live namespace left — an emptied
        // account must not re-mint its retired ids after compaction.
        for account in state.ns_high_water_accounts() {
            namespace_alloc.raise_floor(
                &account,
                state.max_namespace_id_ever_for(&account).unwrap_or(0),
            );
        }
        let live_ids: Vec<(String, u16)> = namespace_alloc
            .ids
            .iter()
            .map(|((account, _key), id)| (account.clone(), *id))
            .collect();
        for (account, id) in live_ids {
            namespace_alloc.raise_floor(&account, id);
        }
        (object_floor, collection_floor, namespace_alloc)
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
            let mut alloc = self.namespace_alloc.write();
            // TD-SPINE-2: adopt-or-allocate raises the ACCOUNT's floor, not a
            // global counter — existing ids in other accounts do not reserve
            // anything here (per-account floor, new allocations only).
            alloc.raise_floor(account, id);
            match alloc.ids.get(&key) {
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
                    alloc.ids.insert(key, id);
                }
            }
            id
        } else if let Some(id) = self.namespace_alloc.read().ids.get(&key).copied() {
            id
        } else {
            let mut alloc = self.namespace_alloc.write();
            if let Some(id) = alloc.ids.get(&key).copied() {
                id
            } else {
                // TD-SPINE-2: mint from the ACCOUNT's floor (starts at 1 for
                // a fresh account — two accounts can each mint id 1; the id
                // scopes an accounts/{a}/{n}/… path, so per-account
                // uniqueness is the contract).
                let id = alloc.allocate(account)?;
                alloc.ids.insert(key, id);
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

        // TD-SPINE-2: sharded boot when the store supports it and the kill
        // switch is ON. The ROOT log's existence IS the mode — a published
        // root ⇒ root + per-account shards; no root ⇒ the legacy whole-image
        // log (the mixed-read source an owner then re-publishes as root +
        // shards on first boot). With the kill switch OFF an object-store
        // catalog still ASSEMBLES root+shards once (data preservation on
        // escape) and then re-seeds the legacy whole-image log.
        let sharded_reader =
            sharded_catalog_snapshots_enabled() && snapshot_store.supports_sharded_publish();
        // Set when the durable image came from the LEGACY whole-image log and
        // sharding is ON: the owner re-publishes it as root + shards before
        // serving, so followers converge on the sharded protocol.
        let mut needs_shard_migration = false;
        // Set when the image was assembled from root+shards but this pod will
        // publish whole-image (kill switch OFF): re-seed the legacy log once.
        let mut needs_legacy_reseed = false;

        let (state, prior_version, prior_generation, shard_versions) = if sharded_reader {
            match snapshot_store.read_root().await? {
                Some(root_read) => {
                    let (state, versions) =
                        Self::assemble_root_and_shards(&root_read, snapshot_store.as_ref()).await?;
                    state.replay(&entries)?;
                    (state, root_read.version, root_read.generation, versions)
                }
                None => match snapshot_store.read().await? {
                    Some(read) => {
                        // Mixed-read source: a whole-image snapshot written
                        // before sharding (or by an escaped pod). Decodes
                        // unchanged; the owner re-publishes it sharded below.
                        let state = SystemCatalogState::from_snapshot_bytes(&read.bytes)
                            .with_context(|| {
                                format!("decoding catalog snapshot {}", snapshot_store.describe())
                            })?;
                        state.replay(&entries)?;
                        needs_shard_migration = true;
                        (state, read.version, read.generation, HashMap::new())
                    }
                    None => (
                        SystemCatalogState::from_wal_entries(&entries)?,
                        NO_VERSION,
                        0,
                        HashMap::new(),
                    ),
                },
            }
        } else if snapshot_store.supports_sharded_publish()
            && let Some(root_read) = snapshot_store.read_root().await?
        {
            // Kill switch OFF on a store that HAD sharding on: assemble the
            // (newer) sharded image so the escape loses nothing, claim above
            // its generation, and re-seed the legacy whole-image log below.
            let (state, versions) =
                Self::assemble_root_and_shards(&root_read, snapshot_store.as_ref()).await?;
            state.replay(&entries)?;
            let legacy_generation = snapshot_store
                .read()
                .await?
                .map(|read| read.generation)
                .unwrap_or(0);
            needs_legacy_reseed = claim_owner;
            (
                state,
                NO_VERSION,
                root_read.generation.max(legacy_generation),
                versions,
            )
        } else {
            match snapshot_store.read().await? {
                Some(read) => {
                    let state = SystemCatalogState::from_snapshot_bytes(&read.bytes).with_context(
                        || format!("decoding catalog snapshot {}", snapshot_store.describe()),
                    )?;
                    state.replay(&entries)?;
                    (state, read.version, read.generation, HashMap::new())
                }
                None => (
                    SystemCatalogState::from_wal_entries(&entries)?,
                    NO_VERSION,
                    0,
                    HashMap::new(),
                ),
            }
        };

        // Seed the appender's sequence floor from the snapshot watermark so a
        // fresh local WAL (compacted-to-empty, or a peer's snapshot adopted into
        // a different local sequence space) cannot mint a sequence the in-RAM
        // authority has already applied (which `apply_committed` would drop). A
        // no-op when the local WAL already carries the watermark.
        appender.advance_sequence_floor(state.applied_seq());

        // TD-PATH-2 4c: WARN when the snapshot we just adopted is UNSTAMPED.
        // That covers two writer generations: a #2316-era high-water-aware
        // binary that simply predates stamping (its `ids_high_water` field is
        // present — the WARN is benign noise for it), and a pre-high-water
        // binary (no field, and its checkpoints compact the WAL — there the
        // floor leans on the sidecar and the WAL tail).
        if let Some(stamp) = state.loaded_writer_version()
            && proximadb_catalog::identity_writer_version_lags(stamp)
        {
            tracing::warn!(
                store = %snapshot_store.describe(),
                "adopted an unstamped catalog snapshot (writer predates \
                 writer-version stamping; if it also predates the identity \
                 high-water, the floor leans on the sidecar and the WAL tail)"
            );
        }
        // TD-PATH-2 4a: fold the durable high-water sidecar —
        // fetch_max(sidecar, snapshot.ids_high_water, WAL replay) — BEFORE the
        // allocator floors are seeded, so an older binary's field-less
        // snapshot + compacted WAL cannot lower the floor below an id this
        // deployment already minted.
        Self::adopt_identity_high_water_sidecar(&state, snapshot_store.as_ref()).await;

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

        let (object_floor, collection_floor, namespace_alloc) =
            Self::identity_allocator_seed(&state);
        let account_floor = state.max_account_id().map_or(1, |id| u64::from(id) + 1);
        let object_id_floor = state.max_object_id().map_or(1, |id| id + 1);
        let catalog = Self {
            name: name.into(),
            role: proximadb_catalog::CatalogRole::Operator,
            state: Arc::new(state),
            account_floor: AtomicU64::new(account_floor),
            object_id_floor: AtomicU64::new(object_id_floor),
            object_floor: AtomicU64::new(object_floor),
            collection_floor: AtomicU64::new(collection_floor),
            namespace_alloc: parking_lot::RwLock::new(namespace_alloc),
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
            shard_versions: parking_lot::RwLock::new(shard_versions),
            pending_shards: parking_lot::Mutex::new(std::collections::HashSet::new()),
        };
        // TD-SPINE-2 migration (owner only): the durable image came from the
        // legacy whole-image log and sharding is ON ⇒ re-publish it as root +
        // one shard per account NOW (mixed-read-safe: the legacy chain stays
        // readable; followers converge on their next reload). With the kill
        // switch OFF on a previously-sharded store, the assembled root+shard
        // image is re-seeded onto the legacy whole-image log instead, so the
        // escape never strands the sharded window's data on the root.
        if (needs_shard_migration || needs_legacy_reseed) && claim_owner {
            if let Some(cfg) = &catalog.snapshot {
                let _guard = catalog.write_lock.lock().await;
                *catalog.pending_shards.lock() = catalog.state.shard_keys();
                catalog.checkpoint_locked(cfg).await?;
                catalog.commits_since_snapshot.store(0, Ordering::SeqCst);
            }
        }
        Ok(catalog)
    }

    /// TD-SPINE-2: assemble the full in-RAM authority from a published ROOT
    /// plus every shard in its index (boot / follower assembly), returning
    /// the per-account shard-log versions observed (the local copy of the
    /// root index that reload later diffs against). A shard the root
    /// references but whose log is absent is skipped with a WARN — the root's
    /// own index still governs what that account's rows should be, and the
    /// next publish repairs it.
    async fn assemble_root_and_shards(
        root_read: &crate::services::catalog_snapshot_store::SnapshotRead,
        store: &dyn CatalogSnapshotStore,
    ) -> Result<(SystemCatalogState, HashMap<String, u64>)> {
        let (_, _, _, _, shard_index) =
            SystemCatalogState::decode_catalog_root(&root_read.bytes)
                .with_context(|| format!("decoding catalog root {}", store.describe()))?;
        let mut shards = Vec::with_capacity(shard_index.len());
        let mut versions = HashMap::with_capacity(shard_index.len());
        for (account, _root_version) in &shard_index {
            match store.read_shard(account).await? {
                Some(read) => {
                    shards.push((account.clone(), read.bytes));
                    versions.insert(account.clone(), read.version);
                }
                None => tracing::warn!(
                    account = %account,
                    store = %store.describe(),
                    "catalog root references a shard with no committed log; \
                     skipping it (the root's older version still governs)"
                ),
            }
        }
        let state = SystemCatalogState::from_root_and_shards(&root_read.bytes, &shards)?;
        Ok((state, versions))
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
        let touches_ids = deltas.iter().any(CatalogDelta::touches_ids_high_water);
        // TD-SPINE-2: capture the shard keys a `DropNamespace` retires
        // BEFORE the fold removes the namespace record (its tenant is the
        // shard attribution)...
        let mut touched: std::collections::HashSet<String> = deltas
            .iter()
            .filter_map(|delta| match delta {
                CatalogDelta::DropNamespace { levels } => {
                    let tenant = self
                        .state
                        .get_namespace(levels)
                        .and_then(|namespace| namespace.tenant_id);
                    Some(crate::services::system_catalog_state::shard_key_of_tenant(
                        tenant.as_deref(),
                    ))
                }
                _ => None,
            })
            .collect();
        // ...and the namespace paths every OTHER delta touches, resolved
        // AFTER the fold (a create-namespace + create-table batch must see
        // the namespace the same batch just created).
        let touched_levels: Vec<Vec<String>> = deltas
            .iter()
            .filter_map(|delta| match delta {
                CatalogDelta::UpsertNamespace { namespace } => Some(namespace.levels.clone()),
                CatalogDelta::UpsertTable { identifier, .. }
                | CatalogDelta::DropTable { identifier }
                | CatalogDelta::UpsertStatistics { identifier, .. } => {
                    Some(identifier.namespace.clone())
                }
                _ => None,
            })
            .collect();
        for (entry, delta) in entries.into_iter().zip(deltas) {
            self.state.apply_committed(entry.sequence_number, delta);
        }
        touched.extend(
            touched_levels
                .iter()
                .map(|levels| self.state.shard_key_of_namespace(levels)),
        );
        // TD-SPINE-2: exactly these shards (plus the always-written root) are
        // republished at the next checkpoint — bytes-per-DDL O(account).
        self.pending_shards.lock().extend(touched);
        if let Some(cfg) = &self.snapshot {
            // TD-PATH-2 4a: the mint/drop durability point. The WAL entry is
            // durable and folded; persist the sidecar so an older binary that
            // later takes over and checkpoints (compacting the WAL past this
            // commit) cannot destroy the floor's only publisher-independent
            // record. Warn-only: the WAL still carries the ids and the next
            // checkpoint retries — failing here would report an
            // already-committed DDL as failed.
            //
            // Gated on ID-BEARING deltas (review R2 MAJOR-1): an
            // `UpsertStatistics`-only commit cannot raise the high-water, and
            // DML row-count updates reach here on the ingest hot path — an
            // ungated persist cost a sidecar read + tmp-write + fsync + rename
            // per stats commit on local stores (~tripling per-commit fsync
            // latency) and, on `requires_snapshot_on_commit` object stores,
            // this GET+PUT stacked on the checkpoint's own GET+PUT. The
            // checkpoint-time write in `checkpoint_locked` stays
            // unconditional — it is the load-bearing one guarding compaction.
            if touches_ids && let Err(error) = self.persist_identity_high_water_sidecar(cfg).await {
                tracing::warn!(
                    catalog = %self.name,
                    error = %error,
                    "identity high-water sidecar write failed at the mint/drop \
                     durability point (the WAL still carries the ids; the next \
                     checkpoint retries)"
                );
            }
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

    /// TD-PATH-2 4a: fold the durable high-water sidecar into `state`
    /// (`fetch_max(sidecar, snapshot.ids_high_water, WAL replay)`) and WARN on
    /// a lagging stamp. Non-fatal by design: a missing, unreadable or corrupt
    /// sidecar only loses DROPPED-id memory (live schemas still bound the
    /// floor), and the merge-before-write in
    /// [`Self::persist_identity_high_water_sidecar`] repairs the file before
    /// the next checkpoint compacts anything.
    async fn adopt_identity_high_water_sidecar(
        state: &SystemCatalogState,
        store: &dyn CatalogSnapshotStore,
    ) {
        match store.read_identity_high_water().await {
            Ok(Some(bytes)) => match state.adopt_identity_high_water_sidecar_bytes(&bytes) {
                Ok(stamp) => {
                    if proximadb_catalog::identity_writer_version_lags(stamp) {
                        tracing::warn!(
                            store = %store.describe(),
                            stamp = ?stamp,
                            "identity high-water sidecar is unstamped (writer \
                             predates writer-version stamping; #2336-era sidecars \
                             are current but unstamped — benign. If it predates \
                             high-water awareness it may under-count minted ids) — \
                             folding it anyway"
                        );
                    }
                }
                Err(error) => tracing::warn!(
                    store = %store.describe(),
                    error = %error,
                    "identity high-water sidecar is unreadable; ignoring it"
                ),
            },
            Ok(None) => {}
            Err(error) => tracing::warn!(
                store = %store.describe(),
                error = %error,
                "identity high-water sidecar could not be read; it will be merged \
                 before the next sidecar write"
            ),
        }
    }

    /// TD-PATH-2 4a: persist the identity high-water sidecar. Merge-first: the
    /// on-disk image is folded into state so this write can never LOWER the
    /// durable floor (a missed or corrupt read at open must not regress it).
    /// An unreadable CURRENT sidecar propagates the error — writing blind
    /// could lower the floor.
    async fn persist_identity_high_water_sidecar(&self, cfg: &SnapshotConfig) -> Result<()> {
        match cfg.store.read_identity_high_water().await {
            Ok(Some(bytes)) => {
                if let Err(error) = self.state.adopt_identity_high_water_sidecar_bytes(&bytes) {
                    tracing::warn!(
                        store = %cfg.store.describe(),
                        error = %error,
                        "existing identity high-water sidecar is unreadable; replacing \
                         it with the state-derived floor"
                    );
                }
            }
            Ok(None) => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "reading the identity high-water sidecar before rewrite at {}",
                        cfg.store.describe()
                    )
                });
            }
        }
        let bytes = self.state.to_identity_high_water_sidecar_bytes()?;
        cfg.store.write_identity_high_water(&bytes).await
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
        // TD-SPINE-2: the sharded publish path (default ON; kill switch
        // `PROXIMADB_CATALOG_SHARDED_SNAPSHOT_DISABLE` reverts to the legacy
        // whole-image branch below). Local/injected stores never support the
        // sharded surface and keep the whole-image behavior.
        if sharded_catalog_snapshots_enabled() && cfg.store.supports_sharded_publish() {
            return self.checkpoint_sharded_locked(cfg).await;
        }
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

        // 1b. TD-PATH-2 4a: durable high-water sidecar BEFORE compaction. The
        //     snapshot published above carries the high-water, but a LATER
        //     older-binary publisher overwrites the snapshot with a field-less
        //     image after compacting the WAL — the sidecar is the artifact
        //     that publisher cannot touch. A failure here fails the checkpoint
        //     (the WAL is left un-compacted, still carrying the create deltas
        //     replay folds), so the floor always survives somewhere.
        //     Unconditional by design (unlike the mint-time write in
        //     `commit_batch_locked`, which is gated on id-bearing deltas):
        //     this write is what makes WAL compaction safe, whatever the
        //     commit mix since the last checkpoint was.
        self.persist_identity_high_water_sidecar(cfg).await?;

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

    /// TD-SPINE-2 (ADR-098 D1.3): the sharded checkpoint — publish ONLY the
    /// accounts touched since the last publish (each as its own
    /// generation-fenced shard manifest, O(account) bytes), then the root
    /// carrying the merged `account → shard version` index. Crash-consistent
    /// by publish order: shards first, root last. A crash between the two
    /// leaves the root naming the shards' PREDECESSORS — followers keep the
    /// older-but-consistent image and the owner's next publish (which
    /// re-serializes the same in-RAM shard image idempotently) heals it.
    /// **The write lock must be held.**
    async fn checkpoint_sharded_locked(&self, cfg: &SnapshotConfig) -> Result<()> {
        let watermark = self.state.applied_seq();
        let generation = self.generation.load(Ordering::SeqCst);

        // 1. Commit the touched shards (idempotent images of the in-RAM
        //    authority for exactly those accounts).
        let touched: Vec<String> = std::mem::take(&mut *self.pending_shards.lock())
            .into_iter()
            .collect();
        let shard_writes: Vec<(String, Vec<u8>)> = touched
            .iter()
            .map(|account| {
                Ok((
                    account.clone(),
                    self.state
                        .to_shard_bytes(account)
                        .with_context(|| format!("serializing catalog shard '{account}'"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let committed = match cfg.store.commit_shards(generation, &shard_writes).await? {
            crate::services::catalog_snapshot_store::ShardCommit::Committed(versions) => versions,
            crate::services::catalog_snapshot_store::ShardCommit::Fenced => {
                // A newer writer owns at least one shard log. Our shards did
                // not become the named tips; do NOT publish a root over
                // them. Step down exactly like a fenced whole-image
                // checkpoint (reload the winner's image, go read-only).
                tracing::warn!(
                    catalog = %self.name,
                    generation,
                    store = %cfg.store.describe(),
                    "sharded catalog checkpoint was fenced by a newer pod on a \
                     shard log; stepping down to read-only and reloading its snapshot"
                );
                self.reload_from_store_locked(cfg).await?;
                return Ok(());
            }
        };

        // 2. Commit the root with the merged index (touched shards at their
        //    NEW versions, every other account at the version this pod
        //    already reflects).
        let mut index: std::collections::BTreeMap<String, u64> =
            self.shard_versions.read().clone().into_iter().collect();
        index.extend(committed);
        let root_bytes = self.state.to_root_bytes(&index)?;
        match cfg.store.commit_root(generation, &root_bytes).await? {
            SnapshotWrite::Published { version } => {
                *self.shard_versions.write() = index.into_iter().collect();
                self.loaded_version.store(version, Ordering::SeqCst);
                self.loaded_generation.store(generation, Ordering::SeqCst);
            }
            SnapshotWrite::Fenced => {
                tracing::warn!(
                    catalog = %self.name,
                    generation,
                    store = %cfg.store.describe(),
                    "catalog root commit was fenced by a newer pod; stepping \
                     down to read-only and reloading its snapshot"
                );
                self.reload_from_store_locked(cfg).await?;
                return Ok(());
            }
        }

        // 3. TD-PATH-2 4a: the durable high-water sidecar BEFORE compaction
        //    (identical contract to the whole-image branch: the sidecar is
        //    what makes WAL compaction safe whatever the commit mix was).
        self.persist_identity_high_water_sidecar(cfg).await?;

        // 4. Compact the WAL: keep only entries the published image does not
        //    cover (identical to the whole-image branch).
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
        // TD-SPINE-2: with sharding enabled the ROOT pointer is the sinval
        // key (cheap probe, no payload). Before the first root exists (a
        // pre-migration deployment, or an owner that has not published yet)
        // the legacy whole-image log still governs — probe that instead.
        if sharded_catalog_snapshots_enabled() && cfg.store.supports_sharded_publish() {
            match cfg.store.latest_root_version().await? {
                Some(v) => {
                    let loaded = self.loaded_version.load(Ordering::SeqCst);
                    if version_is_newer(loaded, v) {
                        let _guard = self.write_lock.lock().await;
                        return self.reload_from_store_locked(cfg).await;
                    }
                    return Ok(ReloadOutcome::UpToDate);
                }
                // No root yet: fall through to the legacy whole-image probe.
                None => {}
            }
        }
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
        // TD-SPINE-2: the sharded reload — adopt the root's global halves and
        // ONLY the shards whose versions changed (never swap the whole
        // image). Falls back to the legacy whole-image reload when no root
        // has been published yet (mixed-read: pre-migration or escaped pods).
        if sharded_catalog_snapshots_enabled()
            && cfg.store.supports_sharded_publish()
            && let Some(root_read) = cfg.store.read_root().await?
        {
            return self.reload_sharded_locked(cfg, root_read).await;
        }
        self.reload_whole_image_locked(cfg).await
    }

    /// TD-SPINE-2: the sharded reload. The root's `account → shard version`
    /// index is diffed against this pod's known versions; only changed
    /// shards are fetched and adopted IN PLACE (each adoption replaces
    /// exactly that account's rows), so a DDL in one account never disturbs
    /// another account's read-your-writes window. Globals (account
    /// registry, identity high-water, watermark) come from the root.
    async fn reload_sharded_locked(
        &self,
        cfg: &SnapshotConfig,
        root_read: crate::services::catalog_snapshot_store::SnapshotRead,
    ) -> Result<ReloadOutcome> {
        let loaded = self.loaded_version.load(Ordering::SeqCst);
        if !version_is_newer(loaded, root_read.version) {
            // Another reload already pulled this version (or newer); never
            // reload an older root over a current one (monotonic reads).
            return Ok(ReloadOutcome::UpToDate);
        }
        let (_, _, writer_version, _, shard_index) =
            SystemCatalogState::decode_catalog_root(&root_read.bytes).with_context(|| {
                format!(
                    "reloading catalog '{}' from root {}",
                    self.name,
                    cfg.store.describe()
                )
            })?;
        self.state.adopt_root_bytes(&root_read.bytes)?;
        let known = self.shard_versions.read().clone();
        let mut updated = known.clone();
        for (account, root_version) in &shard_index {
            if known.get(account) == Some(root_version) {
                continue; // unchanged shard — do not even fetch it
            }
            match cfg.store.read_shard(account).await? {
                Some(read) => {
                    self.state
                        .load_shard_bytes(account, &read.bytes)
                        .with_context(|| {
                            format!(
                                "reloading catalog '{}' shard '{account}' from {}",
                                self.name,
                                cfg.store.describe()
                            )
                        })?;
                    updated.insert(account.clone(), read.version);
                }
                None => tracing::warn!(
                    catalog = %self.name,
                    account = %account,
                    store = %cfg.store.describe(),
                    "published root names a shard with no committed log; \
                     keeping the previously adopted image for that account"
                ),
            }
        }
        if proximadb_catalog::identity_writer_version_lags(writer_version) {
            tracing::warn!(
                catalog = %self.name,
                store = %cfg.store.describe(),
                "adopted an unstamped peer catalog root (writer predates \
                 writer-version stamping; if it also predates the identity \
                 high-water, the floor leans on the sidecar)"
            );
        }
        Self::adopt_identity_high_water_sidecar(&self.state, cfg.store.as_ref()).await;
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
        self.merge_namespace_allocator_seed();
        *self.shard_versions.write() = updated;
        self.loaded_version
            .store(root_read.version, Ordering::SeqCst);
        self.loaded_generation
            .store(root_read.generation, Ordering::SeqCst);
        let superseded = root_read.generation > self.generation.load(Ordering::SeqCst);
        if superseded {
            self.read_only.store(true, Ordering::SeqCst);
        }
        Ok(ReloadOutcome::Reloaded {
            version: root_read.version,
            generation: root_read.generation,
            superseded,
        })
    }

    /// TD-SPINE-2: merge the allocator seed derived from the (just-adopted)
    /// state into the transient allocators — registry rows are replaced
    /// (adopted authority wins), floors merge with max so an id this pod
    /// ever minted can never be re-issued.
    fn merge_namespace_allocator_seed(&self) {
        let (_object_floor, _collection_floor, seed) = Self::identity_allocator_seed(&self.state);
        let mut alloc = self.namespace_alloc.write();
        alloc.ids.extend(seed.ids);
        for (account, floor) in seed.floors {
            let entry = alloc.floors.entry(account).or_insert(1);
            *entry = (*entry).max(floor);
        }
    }

    /// The legacy whole-image reload (Phase 6b): swap the ENTIRE in-RAM
    /// authority from one snapshot blob. Still the path for local stores,
    /// injected stores, the kill-switch escape, and pre-migration roots.
    async fn reload_whole_image_locked(&self, cfg: &SnapshotConfig) -> Result<ReloadOutcome> {
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
        // TD-PATH-2 4a/4c: the peer's snapshot may be unstamped (an
        // #2316-era aware writer, or a pre-high-water writer with no
        // `ids_high_water`). WARN on the lagging stamp, then fold the sidecar
        // so the re-seeded floors below sit above every id this deployment
        // ever minted — not merely the live schemas the peer encoded.
        if let Some(stamp) = self.state.loaded_writer_version()
            && proximadb_catalog::identity_writer_version_lags(stamp)
        {
            tracing::warn!(
                catalog = %self.name,
                store = %cfg.store.describe(),
                "adopted an unstamped peer catalog snapshot (writer predates \
                 writer-version stamping; if it also predates the identity \
                 high-water, the floor leans on the sidecar)"
            );
        }
        Self::adopt_identity_high_water_sidecar(&self.state, cfg.store.as_ref()).await;
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
        // publisher after this pod booted. Re-seed every transient allocator
        // while the catalog write lock is held, so a same-generation reload
        // cannot later reuse an adopted id. TD-SPINE-2: the namespace
        // allocator merges per-account floors (never lowers).
        let (object_floor, collection_floor, _namespace_alloc) =
            Self::identity_allocator_seed(&self.state);
        self.object_floor.fetch_max(object_floor, Ordering::SeqCst);
        self.collection_floor
            .fetch_max(collection_floor, Ordering::SeqCst);
        self.merge_namespace_allocator_seed();
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
        ns.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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
        schema.updated_at_ms = now_unix_ms();
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

    async fn create_named(
        cat: &SystemCatalog,
        namespace: &[String],
        name: &str,
    ) -> Result<CatalogTableSchema> {
        cat.create_table(
            &TableIdentifier::new(namespace.to_vec(), name),
            vec_schema(name),
        )
        .await
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

    /// TD-PATH-2: dropping the NEWEST table and restarting must not re-issue
    /// its object_id (the floor used to be derived from live tables only, so
    /// the dropped table's id — and its grants — came back).
    #[tokio::test]
    async fn a_dropped_tables_object_id_is_not_reissued_after_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let ns = nslevels(&["t"]);
        let dropped_id = {
            let cat = SystemCatalog::open("default", &wal).await?;
            cat.create_namespace(&ns, std::collections::HashMap::new())
                .await?;
            create_named(&cat, &ns, "a").await?;
            let newest = create_named(&cat, &ns, "b").await?;
            cat.drop_table(&TableIdentifier::new(ns.clone(), "b"), false)
                .await?;
            newest.object_id.expect("table b minted")
        };
        let reopened = SystemCatalog::open("default", &wal).await?;
        let c = create_named(&reopened, &ns, "c").await?;
        let idc = c.object_id.expect("table c minted after restart");
        assert!(
            idc > dropped_id,
            "{idc} re-issues (or precedes) the dropped table's id {dropped_id}"
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

    /// TD-PATH-2 end to end: dropping the tenant table with the HIGHEST typed
    /// collection id, compacting the WAL (so its create delta is gone) and
    /// restarting must not re-issue that id — it names the dropped
    /// collection's `accounts/{a}/{n}/{c}` storage root.
    #[tokio::test]
    async fn a_dropped_typed_collection_id_is_not_reissued_after_checkpoint_and_restart()
    -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let namespace = nslevels(&["acme", "default"]);
        let dropped = {
            let cat = SystemCatalog::open("default", &wal).await?;
            cat.create_namespace_for_tenant(&namespace, HashMap::new(), Some("acme"))
                .await?;
            create_named(&cat, &namespace, "orders").await?;
            let newest = create_named(&cat, &namespace, "customers").await?;
            cat.drop_table(&TableIdentifier::new(namespace.clone(), "customers"), false)
                .await?;
            cat.checkpoint().await?;
            newest.stable_collection_id.expect("typed collection id")
        };
        let reopened = SystemCatalog::open("default", &wal).await?;
        let fresh = create_named(&reopened, &namespace, "invoices").await?;
        let minted = fresh.stable_collection_id.expect("typed collection id");
        assert!(
            minted > dropped,
            "{minted} re-issues (or precedes) the dropped collection's id {dropped}"
        );
        Ok(())
    }

    /// TD-PATH-2 4a, the older-binary publisher: an OLD (pre-high-water)
    /// binary's checkpoint publishes a snapshot WITHOUT the `ids_high_water`
    /// key and compacts the WAL past the watermark — destroying the create
    /// deltas replay would fold. Without a floor record independent of the
    /// snapshot publisher, the dropped-id memory is unrecoverable from that
    /// store and the next create re-issues the dropped collection's typed root.
    ///
    /// The durable high-water SIDECAR (beside the snapshot, never touched by a
    /// publisher that does not know it) must preserve the floor.
    #[tokio::test]
    async fn legacy_snapshot_publisher_cannot_destroy_the_dropped_id_floor() -> Result<()> {
        // The pre-TD-PATH-2 snapshot shape an older binary re-publishes: same
        // live state, no `ids_high_water`, no `writer_version`.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct PreHighWaterSnapshot {
            account_ids: HashMap<String, u32>,
            namespaces: HashMap<Vec<String>, CatalogNamespace>,
            tables: HashMap<TableIdentifier, CatalogTableSchema>,
            statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
            applied_seq: u64,
        }

        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let snapshot_path = dir.path().join("catalog.snapshot");
        let namespace = nslevels(&["legacy_pub", "default"]);
        let orders = TableIdentifier::new(namespace.clone(), "orders");

        let (dropped_coll, dropped_object) = {
            let cat = SystemCatalog::open("default", &wal).await?;
            cat.create_namespace_for_tenant(&namespace, HashMap::new(), Some("legacy_pub"))
                .await?;
            create_named(&cat, &namespace, "orders").await?;
            let newest = create_named(&cat, &namespace, "customers").await?;
            let dropped = (
                newest.stable_collection_id.expect("typed collection id"),
                newest.object_id.expect("object id"),
            );
            cat.drop_table(&TableIdentifier::new(namespace.clone(), "customers"), false)
                .await?;
            // New binary checkpoint: snapshot WITH the high-water, plus the
            // sidecar beside it.
            cat.checkpoint().await?;
            dropped
        };

        // ── Simulate the older binary taking over and checkpointing ──
        // It decodes our snapshot fine (additive fields are ignored), replays
        // an empty WAL tail, and re-publishes the same live state WITHOUT the
        // high-water key. Read the live image through a new-binary open (reads
        // only), then overwrite the snapshot with the legacy encoding.
        let (live_ns, live_schema, live_account, applied) = {
            let cat = SystemCatalog::open("default", &wal).await?;
            let ns = cat.get_namespace(&namespace).await?;
            let schema = cat.get_table(&orders).await?;
            let account = cat
                .account_id_u32_lookup("legacy_pub")
                .expect("account minted");
            (ns, schema, account, cat.state.applied_seq())
        };
        let legacy = PreHighWaterSnapshot {
            account_ids: HashMap::from([("legacy_pub".to_string(), live_account)]),
            namespaces: HashMap::from([(namespace.clone(), live_ns)]),
            tables: HashMap::from([(orders.clone(), live_schema)]),
            statistics: HashMap::new(),
            applied_seq: applied,
        };
        tokio::fs::write(&snapshot_path, rmp_serde::to_vec_named(&legacy)?).await?;
        // …and its checkpoint compacts the WAL past the watermark, so the
        // dropped collection's create delta is gone.
        crate::services::canonical_wal::rewrite_canonical_wal(&wal, &[]).await?;

        // ── New binary reopens on the legacy-published store ──
        let reopened = SystemCatalog::open("default", &wal).await?;
        // TD-PATH-2 4c: the adoption is detectable — an unstamped snapshot is
        // the WARN condition (published by a pre-high-water binary).
        assert_eq!(reopened.state.loaded_writer_version(), Some(None));
        assert!(proximadb_catalog::identity_writer_version_lags(
            reopened.state.loaded_writer_version().flatten()
        ));
        let fresh = create_named(&reopened, &namespace, "invoices").await?;
        let minted_coll = fresh.stable_collection_id.expect("typed collection id");
        let minted_object = fresh.object_id.expect("object id");
        assert!(
            minted_coll > dropped_coll,
            "typed id {minted_coll} re-issues the dropped collection's {dropped_coll}: the \
             legacy publisher destroyed the only durable record of the floor"
        );
        assert!(
            minted_object > dropped_object,
            "object id {minted_object} re-issues the dropped table's {dropped_object} \
             (its grants would transfer)"
        );

        // Contrast: a store published ONLY by this (stamped) binary adopts a
        // stamped snapshot — no WARN condition.
        reopened.checkpoint().await?;
        let reopened_again = SystemCatalog::open("default", &wal).await?;
        assert_eq!(
            reopened_again.state.loaded_writer_version(),
            Some(Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER))
        );
        assert!(!proximadb_catalog::identity_writer_version_lags(
            reopened_again.state.loaded_writer_version().flatten()
        ));
        Ok(())
    }

    /// A snapshot store whose sidecar write always fails — pins that the
    /// checkpoint-time sidecar write (TD-PATH-2 4a step 1b) is LOAD-BEARING:
    /// removing it from `checkpoint_locked`, or moving it after the WAL trim,
    /// must fail this test. (The mint-time write alone does not cover this:
    /// it is warn-only, and a reader cannot tell it succeeded.)
    struct FailingSidecarStore {
        inner: LocalSnapshotStore,
    }

    #[async_trait::async_trait]
    impl CatalogSnapshotStore for FailingSidecarStore {
        async fn read(
            &self,
        ) -> Result<Option<crate::services::catalog_snapshot_store::SnapshotRead>> {
            self.inner.read().await
        }
        async fn write_atomic(&self, generation: u64, bytes: &[u8]) -> Result<SnapshotWrite> {
            self.inner.write_atomic(generation, bytes).await
        }
        async fn write_identity_high_water(&self, _bytes: &[u8]) -> Result<()> {
            Err(anyhow!("injected sidecar-write failure (test)"))
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// TD-PATH-2 4a (review MAJOR-2): when the sidecar write fails at
    /// checkpoint time, the checkpoint must FAIL and the WAL must remain
    /// un-compacted — the create deltas are the floor's fallback record, so
    /// compaction must never destroy them without the sidecar durable.
    #[tokio::test]
    async fn failing_sidecar_write_fails_the_checkpoint_and_keeps_the_wal() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let store = Arc::new(FailingSidecarStore {
            inner: LocalSnapshotStore::new(wal.with_extension("snapshot")),
        });
        let ns = nslevels(&["sidecar_fail"]);
        let cat = SystemCatalog::open_with_snapshot_store("default", &wal, store).await?;
        cat.create_namespace(&ns, HashMap::new()).await?;
        create_named(&cat, &ns, "t").await?; // mint-time write fails (warn-only): commit still Ok

        // The checkpoint MUST fail on the sidecar write (step 1b)…
        let checkpoint = cat.checkpoint().await;
        assert!(
            checkpoint.is_err(),
            "a failed sidecar write must fail the checkpoint — otherwise the WAL \
             can be compacted with no publisher-independent floor record"
        );
        // …and the WAL must still hold the pre-watermark deltas (un-compacted).
        let appender = crate::services::FramedTableWalAppender::open(&wal).await?;
        let entries = appender.read_entries().await?;
        let watermark = cat.state.applied_seq();
        assert!(
            entries.iter().any(|e| e.sequence_number <= watermark),
            "the WAL was compacted past the watermark ({watermark}) despite the failed \
             sidecar write — the create deltas are gone with no sidecar to cover them"
        );
        // End-to-end: a plain reopen still recovers the table from that WAL.
        let reopened = SystemCatalog::open("default", &wal).await?;
        assert!(
            reopened
                .table_exists(&TableIdentifier::new(ns, "t"))
                .await?
        );
        Ok(())
    }

    /// A snapshot store that COUNTS sidecar writes (delegating everything
    /// else to a real local store), so a test can pin WHICH commits persist
    /// the mint-time sidecar.
    struct CountingSidecarStore {
        inner: LocalSnapshotStore,
        writes: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl CatalogSnapshotStore for CountingSidecarStore {
        async fn read(
            &self,
        ) -> Result<Option<crate::services::catalog_snapshot_store::SnapshotRead>> {
            self.inner.read().await
        }
        async fn write_atomic(&self, generation: u64, bytes: &[u8]) -> Result<SnapshotWrite> {
            self.inner.write_atomic(generation, bytes).await
        }
        async fn write_identity_high_water(&self, _bytes: &[u8]) -> Result<()> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// TD-PATH-2 4a (review R2 MAJOR-1): the mint-time sidecar persist is
    /// gated on ID-BEARING deltas. DML row-count updates flow through
    /// `update_statistics` as `UpsertStatistics`-only commits on the ingest
    /// hot path; they cannot raise the identity high-water, so they must not
    /// pay the sidecar round-trip (a read + tmp + fsync + rename per stats
    /// commit on local stores; a GET+PUT pair stacked on the checkpoint's own
    /// GET+PUT on `requires_snapshot_on_commit` object stores). Id-bearing
    /// commits must still persist — and the gate must be per-commit, not a
    /// sticky latch. (The checkpoint-time write stays unconditional; that
    /// half is pinned by
    /// `failing_sidecar_write_fails_the_checkpoint_and_keeps_the_wal`.)
    #[tokio::test]
    async fn stats_only_commits_skip_the_mint_time_sidecar_write() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal = dir.path().join("catalog.wal");
        let store = Arc::new(CountingSidecarStore {
            inner: LocalSnapshotStore::new(wal.with_extension("snapshot")),
            writes: std::sync::atomic::AtomicUsize::new(0),
        });
        let cat = SystemCatalog::open_with_snapshot_store("default", &wal, store.clone()).await?;
        let ns = nslevels(&["stats_gate"]);
        cat.create_namespace(&ns, HashMap::new()).await?;
        create_named(&cat, &ns, "t").await?;
        let id = TableIdentifier::new(ns.clone(), "t");
        let after_ddl = store.writes.load(Ordering::SeqCst);
        assert_eq!(
            after_ddl, 2,
            "each id-bearing commit (namespace create, table create) persists the \
             sidecar at the mint/drop durability point"
        );

        // The hot path: row-count updates are UpsertStatistics-only commits.
        // (Two fresh stats values: `update_statistics` takes them by value.)
        let mut first = CatalogTableStatistics::default();
        first.row_count = 42;
        cat.update_statistics(&id, first).await?;
        let mut second = CatalogTableStatistics::default();
        second.row_count = 84;
        cat.update_statistics(&id, second).await?;
        assert_eq!(
            store.writes.load(Ordering::SeqCst),
            after_ddl,
            "UpsertStatistics commits cannot raise the high-water and must not \
             write the sidecar (it would sit on the DML ingest hot path)"
        );

        // The gate reads the batch, not a latch: an id-bearing commit after
        // the stats-only ones persists again.
        create_named(&cat, &ns, "t2").await?;
        assert_eq!(
            store.writes.load(Ordering::SeqCst),
            after_ddl + 1,
            "an id-bearing commit after stats-only commits still persists the sidecar"
        );
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

    // ── TD-SPINE-2: per-account sharded snapshot publish / reload ───────────

    /// Records every PUT (path + byte length) and GET (path) against a real
    /// in-memory backend, so the O(account) and only-changed-shards
    /// assertions observe actual object-store traffic, not assertions about
    /// helper calls.
    #[derive(Debug, Default)]
    struct IoLedger {
        puts: std::sync::Mutex<Vec<(String, usize)>>,
        gets: std::sync::Mutex<Vec<String>>,
    }

    impl IoLedger {
        fn put_bytes_matching(&self, needle: &str) -> usize {
            self.puts
                .lock()
                .expect("puts lock")
                .iter()
                .filter(|(path, _)| path.contains(needle))
                .map(|(_, len)| *len)
                .sum()
        }

        fn put_paths_matching(&self, needle: &str) -> usize {
            self.puts
                .lock()
                .expect("puts lock")
                .iter()
                .filter(|(path, _)| path.contains(needle))
                .count()
        }

        fn get_count_matching(&self, needle: &str) -> usize {
            self.gets
                .lock()
                .expect("gets lock")
                .iter()
                .filter(|path| path.contains(needle))
                .count()
        }

        fn reset(&self) {
            self.puts.lock().expect("puts lock").clear();
            self.gets.lock().expect("gets lock").clear();
        }
    }

    #[derive(Debug)]
    struct CountingStore {
        inner: Arc<object_store::memory::InMemory>,
        ledger: Arc<IoLedger>,
    }

    impl std::fmt::Display for CountingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CountingStore")
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for CountingStore {
        async fn put_opts(
            &self,
            path: &object_store::path::Path,
            payload: object_store::PutPayload,
            options: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.ledger
                .puts
                .lock()
                .expect("puts lock")
                .push((path.to_string(), payload.len()));
            self.inner.put_opts(path, payload, options).await
        }
        async fn put_multipart_opts(
            &self,
            path: &object_store::path::Path,
            options: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(path, options).await
        }
        async fn get_opts(
            &self,
            path: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.ledger
                .gets
                .lock()
                .expect("gets lock")
                .push(path.to_string());
            self.inner.get_opts(path, options).await
        }
        fn delete_stream(
            &self,
            paths: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(paths)
        }
        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }
        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    fn hex_segment(account: &str) -> String {
        crate::storage::trait_components::path_resolver::DrPathBuilder::encode_system_catalog_shard_segment(account)
    }

    /// Guard that sets an env gate for one test and restores the prior value
    /// (or removes it) on drop — `#[serial]` keeps concurrent tests from
    /// observing the mutation.
    struct EnvGuard(&'static str, Option<String>);
    impl EnvGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let prior = std::env::var(name).ok();
            std::env::set_var(name, value);
            Self(name, prior)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.1.take() {
                Some(value) => std::env::set_var(self.0, value),
                None => std::env::remove_var(self.0),
            }
        }
    }

    async fn open_owner_counting(
        wal: &std::path::Path,
        ledger: &Arc<IoLedger>,
        backing: &Arc<object_store::memory::InMemory>,
    ) -> Result<SystemCatalog> {
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(Arc::new(CountingStore {
                    inner: backing.clone(),
                    ledger: ledger.clone(),
                })),
                "memory:///",
                crate::storage::trait_components::path_resolver::DrPathBuilder::system_catalog_manifests_subprefix(),
            ),
        );
        SystemCatalog::open_with_snapshot_store("default", wal, store).await
    }

    /// Seed two accounts of very different sizes: `big` with `big_tables`
    /// tables, `small` with one. Both accounts are minted (registry) and
    /// their namespaces are tenant-scoped, so each lands in its own shard.
    async fn seed_two_accounts(
        owner: &SystemCatalog,
        big_tables: usize,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let big_ns = nslevels(&["big", "prod"]);
        let small_ns = nslevels(&["small", "prod"]);
        owner.account_id_u32("big-tenant").await?;
        owner.account_id_u32("small-tenant").await?;
        owner
            .create_namespace_for_tenant(&big_ns, HashMap::new(), Some("big-tenant"))
            .await?;
        owner
            .create_namespace_for_tenant(&small_ns, HashMap::new(), Some("small-tenant"))
            .await?;
        for i in 0..big_tables {
            owner
                .create_table(
                    &TableIdentifier::new(big_ns.clone(), &format!("t{i:03}")),
                    vec_schema(&format!("t{i:03}")),
                )
                .await?;
        }
        owner
            .create_table(
                &TableIdentifier::new(small_ns.clone(), "only"),
                vec_schema("only"),
            )
            .await?;
        Ok((big_ns, small_ns))
    }

    /// The O(account) target (ADR-098 D1.3 / TD-SPINE-2 acceptance): a DDL
    /// in the SMALL account publishes that account's shard + the small root
    /// — measured PUT bytes stay proportional to the small account's catalog
    /// and NEVER touch the big account's shard object. Mutation control: if
    /// the publish regressed to a whole-image write, the big account's shard
    /// prefix would see zero PUTs but the legacy `_manifests/` prefix would
    /// carry the full fleet image and the byte assertions would fail.
    #[tokio::test]
    async fn ddl_publishes_only_its_accounts_shard_bytes_scale_with_account() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing = Arc::new(object_store::memory::InMemory::new());
        let ledger = Arc::new(IoLedger::default());
        let owner = open_owner_counting(&dir.path().join("owner.wal"), &ledger, &backing).await?;
        let (big_ns, small_ns) = seed_two_accounts(&owner, 40).await?;

        let big_shard_alone = owner.state.to_shard_bytes("big-tenant")?.len();
        assert!(
            big_shard_alone > 8_000,
            "the big account's shard must dominate the fixture ({big_shard_alone} bytes)"
        );
        ledger.reset();

        // One DDL in the SMALL account.
        owner
            .create_table(
                &TableIdentifier::new(small_ns.clone(), "second"),
                vec_schema("second"),
            )
            .await?;

        // The big account's shard object is untouched...
        assert_eq!(
            ledger.put_paths_matching(&format!("_shards/{}", hex_segment("big-tenant"))),
            0,
            "a small-account DDL must not republish the big account's shard"
        );
        // ...and the small account's shard IS republished.
        assert_eq!(
            ledger.put_paths_matching(&format!("_shards/{}", hex_segment("small-tenant"))),
            1,
            "the small account's shard is republished exactly once"
        );
        // The root is published once and stays small (registry-sized, not
        // table-sized).
        assert_eq!(ledger.put_paths_matching("_root/"), 1);
        assert!(
            ledger.put_bytes_matching("_root/") < big_shard_alone / 4,
            "root bytes must be registry-sized, not fleet-sized"
        );
        // Total bytes for the DDL are O(small account): far below the big
        // account's shard alone (the legacy path would rewrite ~the whole
        // fleet image through _manifests/).
        assert_eq!(ledger.put_paths_matching("_manifests/"), 0);
        let total: usize = ledger
            .puts
            .lock()
            .expect("puts lock")
            .iter()
            .map(|(_, len)| *len)
            .sum();
        assert!(
            total < big_shard_alone,
            "bytes per DDL ({total}) must be O(account), below the big account's \
             shard alone ({big_shard_alone})"
        );
        // Sanity: the DDL is durable and visible after a fresh boot.
        drop(owner);
        let reopened =
            open_owner_counting(&dir.path().join("owner.wal"), &ledger, &backing).await?;
        assert!(
            reopened
                .table_exists(&TableIdentifier::new(small_ns, "second"))
                .await?
        );
        assert!(
            reopened
                .table_exists(&TableIdentifier::new(big_ns, "t000"))
                .await?
        );
        Ok(())
    }

    /// Cross-pod reload adopts ONLY the changed shards (ADR-098 D1.3): after
    /// the owner DDLs in the small account, the follower's reload fetches
    /// the root + the small shard and NEVER GETs the big account's shard —
    /// while the big account's rows stay intact in RAM (no whole-image swap,
    /// so its read-your-writes window is undisturbed).
    #[tokio::test]
    async fn follower_reload_adopts_only_the_changed_shard() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing = Arc::new(object_store::memory::InMemory::new());
        let ledger = Arc::new(IoLedger::default());
        let owner = open_owner_counting(&dir.path().join("owner.wal"), &ledger, &backing).await?;
        let (big_ns, small_ns) = seed_two_accounts(&owner, 25).await?;

        // Follower boots from root + both shards.
        let follower =
            open_follower_counting(&dir.path().join("follower.wal"), &ledger, &backing).await?;
        assert!(
            follower
                .table_exists(&TableIdentifier::new(big_ns.clone(), "t000"))
                .await?
        );
        ledger.reset();

        // Owner DDLs in the SMALL account and republishes.
        owner
            .create_table(
                &TableIdentifier::new(small_ns.clone(), "second"),
                vec_schema("second"),
            )
            .await?;

        match follower.reload_if_stale().await? {
            ReloadOutcome::Reloaded { superseded, .. } => assert!(!superseded),
            other => panic!("expected a reload, got {other:?}"),
        }
        // The new table is visible, the big account's rows survived...
        assert!(
            follower
                .table_exists(&TableIdentifier::new(small_ns, "second"))
                .await?
        );
        assert!(
            follower
                .table_exists(&TableIdentifier::new(big_ns, "t024"))
                .await?
        );
        // ...and the reload fetched only root + the small shard: no GET of
        // the big account's shard object, no whole-image read.
        assert_eq!(
            ledger.get_count_matching(&format!("_shards/{}", hex_segment("big-tenant"))),
            0,
            "reload must not fetch the unchanged big shard"
        );
        assert_eq!(
            ledger.get_count_matching(&format!("_shards/{}", hex_segment("small-tenant"))),
            1,
            "reload fetches the changed small shard exactly once"
        );
        assert!(ledger.get_count_matching("_root/") >= 1);
        Ok(())
    }

    async fn open_follower_counting(
        wal: &std::path::Path,
        ledger: &Arc<IoLedger>,
        backing: &Arc<object_store::memory::InMemory>,
    ) -> Result<SystemCatalog> {
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(Arc::new(CountingStore {
                    inner: backing.clone(),
                    ledger: ledger.clone(),
                })),
                "memory:///",
                crate::storage::trait_components::path_resolver::DrPathBuilder::system_catalog_manifests_subprefix(),
            ),
        );
        SystemCatalog::open_follower("default", wal, store).await
    }

    /// TD-SPINE-2 migration (mixed-read): a legacy whole-image snapshot
    /// (written with the kill switch OFF, exactly as a pre-SPINE-2 binary
    /// would) boots under the new code, is re-published as root + shards,
    /// and a follower then assembles the sharded image. The legacy chain
    /// stays readable throughout.
    #[tokio::test]
    #[serial_test::serial]
    async fn legacy_whole_image_snapshot_migrates_to_root_and_shards() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let legacy_prefix = "_operator/catalog/_manifests/";

        // 1. A pre-SPINE-2 writer publishes whole-image snapshots only.
        {
            let _guard = EnvGuard::set("PROXIMADB_CATALOG_SHARDED_SNAPSHOT_DISABLE", "1");
            let store = Arc::new(
                crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                    proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                    "memory:///",
                    legacy_prefix,
                ),
            );
            let legacy = SystemCatalog::open_with_snapshot_store(
                "default",
                dir.path().join("legacy.wal"),
                store.clone(),
            )
            .await?;
            legacy
                .create_namespace_for_tenant(
                    &nslevels(&["acme", "sales"]),
                    HashMap::new(),
                    Some("acme"),
                )
                .await?;
            create_named(&legacy, &nslevels(&["acme", "sales"]), "orders").await?;
            create_named(&legacy, &nslevels(&["acme", "sales"]), "customers").await?;
            // The legacy log carries the whole image; no root exists yet.
            assert!(store.read().await?.is_some());
            assert!(store.read_root().await?.is_none());
        }

        // 2. The new binary boots (kill switch unset = sharded): it reads the
        //    legacy whole image, then re-publishes root + the acme shard.
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                "memory:///",
                legacy_prefix,
            ),
        );
        let migrated = SystemCatalog::open_with_snapshot_store(
            "default",
            dir.path().join("owner.wal"),
            store.clone(),
        )
        .await?;
        let root = store
            .read_root()
            .await?
            .expect("migration published a root");
        let (_, _, _, _, index) =
            crate::services::system_catalog_state::SystemCatalogState::decode_catalog_root(
                &root.bytes,
            )?;
        assert!(
            index.contains_key("acme"),
            "the root index names the migrated account: {index:?}"
        );
        let shard = store.read_shard("acme").await?.expect("acme shard exists");
        assert_eq!(shard.version, index["acme"]);
        // Both tables live in the shard image.
        assert!(
            migrated
                .table_exists(&TableIdentifier::new(
                    nslevels(&["acme", "sales"]),
                    "orders"
                ))
                .await?
        );

        // 3. A follower assembles root + shards and sees the migrated rows.
        let follower =
            SystemCatalog::open_follower("default", dir.path().join("follower.wal"), store.clone())
                .await?;
        assert!(
            follower
                .table_exists(&TableIdentifier::new(
                    nslevels(&["acme", "sales"]),
                    "customers"
                ))
                .await?
        );
        // And the legacy chain is still readable (mixed-read: nothing was
        // destroyed by the migration).
        let legacy_read = store.read().await?.expect("legacy log intact");
        assert!(
            crate::services::system_catalog_state::SystemCatalogState::from_snapshot_bytes(
                &legacy_read.bytes
            )?
            .table_exists(&TableIdentifier::new(
                nslevels(&["acme", "sales"]),
                "orders"
            ))
        );
        Ok(())
    }

    /// TD-SPINE-2 crash-consistency: a publish that dies between the shard
    /// commit and the root commit leaves an ORPHAN shard — a committed
    /// version the root does not name. The orphan is invisible to reloads
    /// (the ROOT version is the sinval key; a shard commit alone never
    /// moves it), and the owner's next publish heals it idempotently: it
    /// re-serializes the same in-RAM shard image onto the next slot of the
    /// log the orphan already advanced, and the follower converges without
    /// ever observing a torn account.
    #[tokio::test]
    async fn crash_between_shard_and_root_publish_is_idempotent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                "memory:///",
                "_operator/catalog/_manifests/",
            ),
        );
        let owner = SystemCatalog::open_with_snapshot_store(
            "default",
            dir.path().join("owner.wal"),
            store.clone(),
        )
        .await?;
        owner
            .create_namespace_for_tenant(
                &nslevels(&["acme", "sales"]),
                HashMap::new(),
                Some("acme"),
            )
            .await?;
        create_named(&owner, &nslevels(&["acme", "sales"]), "orders").await?;
        let root_v1 = store.read_root().await?.expect("root published").version;
        let (_, _, _, _, index_v1) =
            crate::services::system_catalog_state::SystemCatalogState::decode_catalog_root(
                &store.read_root().await?.unwrap().bytes,
            )?;

        // Follower adopts root v1.
        let follower =
            SystemCatalog::open_follower("default", dir.path().join("follower.wal"), store.clone())
                .await?;
        assert!(
            follower
                .table_exists(&TableIdentifier::new(
                    nslevels(&["acme", "sales"]),
                    "orders"
                ))
                .await?
        );

        // The crash window: a shard version commits (by the owner's own
        // serializer, same generation — exactly what a lost root PUT leaves
        // behind) but the root does not advance.
        let orphan_bytes = owner.state.to_shard_bytes("acme")?;
        let orphan_version = match store
            .commit_shards(owner.generation(), &[("acme".to_string(), orphan_bytes)])
            .await?
        {
            crate::services::catalog_snapshot_store::ShardCommit::Committed(versions) => {
                versions["acme"]
            }
            other => panic!("expected the orphan shard commit to land, got {other:?}"),
        };
        assert!(
            orphan_version > index_v1["acme"],
            "the orphan shard is ahead of the version the root names"
        );
        assert_eq!(
            store.latest_root_version().await?,
            Some(root_v1),
            "a shard-only commit never advances the root"
        );
        // The follower is unaffected: no root advance, no reload.
        assert_eq!(follower.reload_if_stale().await?, ReloadOutcome::UpToDate);

        // Heal: the owner's next DDL re-serializes its shard image (the
        // orphan's successor slot — idempotent, no conflict) and publishes
        // the root that finally names it.
        create_named(&owner, &nslevels(&["acme", "sales"]), "later").await?;
        match follower.reload_if_stale().await? {
            ReloadOutcome::Reloaded { superseded, .. } => assert!(!superseded),
            other => panic!("expected the healed publish to reload, got {other:?}"),
        }
        assert!(
            follower
                .table_exists(&TableIdentifier::new(
                    nslevels(&["acme", "sales"]),
                    "orders"
                ))
                .await?
        );
        assert!(
            follower
                .table_exists(&TableIdentifier::new(nslevels(&["acme", "sales"]), "later"))
                .await?,
            "the healed root converges the follower"
        );
        Ok(())
    }

    /// TD-SPINE-2 per-account namespace floor (ADR-098 D1.4): two accounts
    /// each mint stable namespace id 1 for their first namespace — the id
    /// scopes an `accounts/{a}/{n}/…` path, so uniqueness is per-account —
    /// and each account's subsequent ids continue from ITS OWN floor.
    #[tokio::test]
    async fn per_account_namespace_floor_two_accounts_each_mint_id_one() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let cat = catalog(dir.path()).await;
        cat.account_id_u32("acme").await?;
        cat.account_id_u32("globex").await?;

        let acme_ns = nslevels(&["acme", "sales"]);
        let globex_ns = nslevels(&["globex", "ops"]);
        cat.create_namespace_for_tenant(&acme_ns, HashMap::new(), Some("acme"))
            .await?;
        cat.create_namespace_for_tenant(&globex_ns, HashMap::new(), Some("globex"))
            .await?;
        let acme_orders = create_named(&cat, &acme_ns, "orders").await?;
        let globex_metrics = create_named(&cat, &globex_ns, "metrics").await?;
        assert_eq!(acme_orders.stable_namespace_id, Some(1));
        assert_eq!(
            globex_metrics.stable_namespace_id,
            Some(1),
            "the namespace floor is per-account: globex mints its own id 1"
        );

        // Each account's second namespace continues from its own floor.
        let acme_ns2 = nslevels(&["acme", "hr"]);
        cat.create_namespace_for_tenant(&acme_ns2, HashMap::new(), Some("acme"))
            .await?;
        let acme_people = create_named(&cat, &acme_ns2, "people").await?;
        assert_eq!(acme_people.stable_namespace_id, Some(2));
        drop(cat);

        // Existing ids are never renumbered across a restart; new allocations
        // continue per-account.
        let reopened = catalog(dir.path()).await;
        let again = reopened
            .get_table(&TableIdentifier::new(acme_ns.clone(), "orders"))
            .await?;
        assert_eq!(again.stable_namespace_id, Some(1), "existing ids unchanged");
        let globex_again = reopened
            .get_table(&TableIdentifier::new(globex_ns.clone(), "metrics"))
            .await?;
        assert_eq!(globex_again.stable_namespace_id, Some(1));
        let acme_ns3 = nslevels(&["acme", "legal"]);
        reopened
            .create_namespace_for_tenant(&acme_ns3, HashMap::new(), Some("acme"))
            .await?;
        let acme_contracts = create_named(&reopened, &acme_ns3, "contracts").await?;
        assert_eq!(
            acme_contracts.stable_namespace_id,
            Some(3),
            "acme's floor continues from its own high-water after restart"
        );
        Ok(())
    }

    /// Legacy-global ids are NOT renumbered when the per-account floor turns
    /// on: an account whose ids were minted under the old global counter
    /// keeps every one of them, and only its NEW allocations draw from its
    /// own (lower) per-account floor.
    #[tokio::test]
    async fn pre_existing_global_namespace_ids_are_never_renumbered() -> Result<()> {
        let state = crate::services::system_catalog_state::SystemCatalogState::new();
        // A legacy account whose ids were minted by the GLOBAL counter: acme
        // holds 1..5, globex holds 6 (as the old single floor would mint).
        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new({
                    let mut ns =
                        proximadb_catalog::CatalogNamespace::new(nslevels(&["acme", "sales"]));
                    ns.tenant_id = Some("acme".to_string());
                    ns
                }),
            },
        );
        for (seq, id) in [(2_u64, 1_u16), (3, 2), (4, 3), (5, 4), (6, 5)] {
            let mut schema = CatalogTableSchema::new(&format!("t{id}"));
            schema.stable_namespace_id = Some(id);
            state.apply_committed(
                seq,
                CatalogDelta::UpsertTable {
                    identifier: TableIdentifier::new(
                        nslevels(&["acme", "sales"]),
                        format!("t{id}"),
                    ),
                    schema: Box::new(schema),
                },
            );
        }
        let cat = SystemCatalog::new(
            "legacy",
            state,
            Arc::new(crate::services::canonical_wal::MemoryTableWalAppender::new()),
        );
        // The per-account floor for acme sits above ITS OWN high-water (5)…
        let acme_ns2 = nslevels(&["acme", "hr"]);
        // (namespace must exist for the typed mint to fire)
        cat.create_namespace_for_tenant(&acme_ns2, HashMap::new(), Some("acme"))
            .await?;
        let schema = vec_schema("people");
        let minted = cat
            .create_table(&TableIdentifier::new(acme_ns2, "people"), schema)
            .await?;
        assert_eq!(
            minted.stable_namespace_id,
            Some(6),
            "acme continues above its own high-water"
        );
        // …and every pre-existing id is still exactly what it was.
        for id in 1_u16..=5 {
            let table = cat
                .get_table(&TableIdentifier::new(
                    nslevels(&["acme", "sales"]),
                    format!("t{id}"),
                ))
                .await?;
            assert_eq!(table.stable_namespace_id, Some(id));
        }
        Ok(())
    }

    /// Kill-switch negative control: with
    /// `PROXIMADB_CATALOG_SHARDED_SNAPSHOT_DISABLE=1` the catalog publishes
    /// the legacy whole-image log and creates NO root or shard objects.
    /// Mutation-proven: flipping the gate's default in
    /// `sharded_catalog_snapshots_enabled` to `false` (or making the store
    /// ignore the flag) fails this test — the sharded branch fires and the
    /// root object appears.
    #[tokio::test]
    #[serial_test::serial]
    async fn kill_switch_reverts_publish_to_the_legacy_whole_image_log() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let backing: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let store = Arc::new(
            crate::services::catalog_snapshot_store::ObjectStoreSnapshotStore::new(
                proximadb_object_store::ProximaObjectStore::new(backing.clone()),
                "memory:///",
                "_operator/catalog/_manifests/",
            ),
        );
        let _guard = EnvGuard::set("PROXIMADB_CATALOG_SHARDED_SNAPSHOT_DISABLE", "1");
        let cat = SystemCatalog::open_with_snapshot_store(
            "default",
            dir.path().join("owner.wal"),
            store.clone(),
        )
        .await?;
        cat.create_namespace(&nslevels(&["s"]), HashMap::new())
            .await?;
        create_named(&cat, &nslevels(&["s"]), "t1").await?;

        // The legacy whole-image log carried the publish…
        let legacy = store.read().await?.expect("whole-image publish");
        assert!(
            crate::services::system_catalog_state::SystemCatalogState::from_snapshot_bytes(
                &legacy.bytes
            )?
            .table_exists(&tid("t1"))
        );
        // …and the sharded surface was never touched.
        assert!(store.read_root().await?.is_none());
        assert!(store.latest_root_version().await?.is_none());
        assert!(store.read_shard("_default").await?.is_none());
        Ok(())
    }
}
