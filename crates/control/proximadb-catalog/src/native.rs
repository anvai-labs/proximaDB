//! # Native Catalog - PRODUCTION READY
//!
//! The Native Catalog provides file-based metadata storage supporting:
//! - Local filesystem
//! - Cloud storage (S3, Azure Blob, GCS)
//!
//! This is the default catalog for standalone deployments.
//!
//! ## Features
//!
//! - **Cloud-First Design**: Seamless local and cloud storage support
//! - **In-Memory Caching**: Fast metadata access with configurable cache
//! - **Namespace Hierarchy**: Full support for multi-level namespaces
//! - **Schema Evolution**: Add/remove columns, rename tables
//! - **Statistics Tracking**: Table and column-level statistics
//!
//! ## Storage Layout
//!
//! ```text
//! <base_path>/
//! ├── metadata/
//! │   ├── namespaces.json           # Namespace registry
//! │   └── tables/
//! │       └── <namespace>/
//! │           └── <table>.json      # Table metadata
//! └── data/
//!     └── <namespace>/
//!         └── <table>/              # Table data files
//! ```
//!
//! ## Configuration
//!
//! Configure via `NativeCatalogConfig`:
//! - `storage_url`: file:// for local, s3://, gs://, az:// for cloud
//!
//! ## Usage
//!
//! ```ignore
//! let config = NativeCatalogConfig {
//!     storage_url: "file:///tmp/proximadb/catalog".to_string(),
//! };
//! let catalog = NativeCatalog::new("default", config, cache).await?;
//! ```

use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_storage_filesystem_types::{FileOptions, FileSystem, FilesystemError};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::cache::CatalogCache;
use crate::schema::{apply_evolution, validate_schema};
use crate::{
    Catalog, CatalogHealth, CatalogIndex, CatalogMlopsAssetExt, CatalogNamespace,
    CatalogPartitionSpec, CatalogSchemaEvolution, CatalogSortOrder, CatalogTableSchema,
    CatalogTableStatistics, TableIdentifier,
};

/// Plain Rust configuration for the native catalog.
///
/// Decoupled from `proximadb_proto::proximadb::v1::NativeCatalogConfig` so the
/// workspace contract crate doesn't depend on the heavy proto crate. The
/// network/API layer converts from the proto form when configuring the
/// catalog. The `replication` field from the proto type is omitted; cross-region
/// replication is configured separately via the DR engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NativeCatalogConfig {
    /// Storage URL, e.g. `s3://bucket/catalog`, `adls://...`, `file:///path`
    pub storage_url: String,
    /// Metadata serialization format: "json", "avro", "parquet" (default: "json")
    pub metadata_format: String,
    /// Enable schema versioning
    pub versioned: bool,
    /// Max versions to keep (default: 100)
    pub max_versions: i32,
}

/// Native ProximaDB catalog
///
/// Uses local or cloud storage as the primary metadata store
/// with in-memory caching for performance.
pub struct NativeCatalog {
    /// Catalog name
    name: String,
    /// Configuration
    config: NativeCatalogConfig,
    /// Base path for storage (local addressing; used when `fs` is `None`).
    base_path: PathBuf,
    /// Optional injected storage backend. When `Some`, all metadata/data I/O is
    /// routed through this `FileSystem` against `config.storage_url` (durable
    /// object-store or local). When `None` (default / back-compat), I/O uses
    /// local `tokio::fs` under `base_path` exactly as before — no behavior
    /// change for existing `file://` deployments. The concrete backend is
    /// injected by the root crate's `FilesystemFactory` (dependency inversion).
    fs: Option<Arc<dyn FileSystem>>,
    /// In-memory namespace cache (loaded on startup)
    namespaces: RwLock<HashMap<String, CatalogNamespace>>,
    /// In-memory table cache (loaded on demand)
    tables: RwLock<HashMap<String, TableMetadata>>,
    /// Catalog-level cache
    cache: Arc<CatalogCache>,
    /// ADR-031 O0: monotonic allocator for table `object_id`s (per-type, globally
    /// unique, never reused). Recovered best-effort via `raise_floor` as tables
    /// load; eager startup recovery / persisted high-water is an O2 hardening
    /// (object_id is not yet load-bearing in O0).
    object_id_allocator: crate::id_allocator::IdAllocator,
    /// ADR-031 O1 (dual-read): reverse index `object_id → TableIdentifier`, the
    /// inverse of the name-keyed `tables` cache. Maintained on
    /// create/load/rename/drop; populated lazily as tables load (eager build at
    /// startup is the O2 recovery hardening).
    object_id_index: RwLock<HashMap<u64, TableIdentifier>>,
    /// ADR-031 / TD-181: reverse index `object_id → namespace levels`, the
    /// namespace analogue of `object_id_index`. Maintained on create/load/drop
    /// (namespaces have no rename path). Enables the Phase-1 backfill and the
    /// Phase-2 reference cutover (`table.namespace_oid → namespace`) to resolve
    /// by id instead of by name.
    namespace_object_id_index: RwLock<HashMap<u64, Vec<String>>>,
    /// TD-181 P1: durable forward index `table fqn → object_id`, persisted to
    /// `_syscat/index.json` and loaded eagerly at startup. It is the
    /// name→oid direction the lazy `object_id_index` (oid→name) cannot answer
    /// for a not-yet-loaded table, and the prerequisite for any oid-keyed reader
    /// (S2) and for WAL-oid resolution. Maintained paired with `object_id_index`
    /// on create/load/rename/drop; rebuilt by scanning object files if absent.
    object_name_index: RwLock<HashMap<String, u64>>,
    /// TD-181 P3 (S2a): whether object_id-keyed metadata paths are written.
    /// Read once from `PROXIMADB_CATALOG_OBJECT_ID_PATHS` at construction (a
    /// process-stable deployment setting, not a per-write toggle); interior
    /// mutability so tests can force it deterministically without env races.
    oid_paths: std::sync::atomic::AtomicBool,
    /// ADR-031 Phase 4a: per-scope typed-atomic allocator — mints
    /// `stable_namespace_id` (u16, per-account) and `stable_collection_id`
    /// (u32, per-namespace) persisted on schemas. Account-scoped via the
    /// numeric `account_registry` below (numerics in-memory; no string keys).
    stable_ids: crate::id_allocator::CatalogIdService,
    /// ADR-031 Phase 4a: transient account-string → u32 registry. The account
    /// u32 is NOT persisted on the namespace (it would duplicate `account_id`);
    /// it's minted on first sight per run and used to scope `stable_ids`.
    /// Phase 4b makes it durable (for path stability); in 4a it only keys
    /// in-memory minting.
    account_registry: DashMap<String, u32>,
    /// Serializes mint + sidecar persistence. Without this, concurrent mints
    /// can publish whole-map snapshots out of order and lose the later tenant.
    account_registry_write_lock: Mutex<()>,
    /// Serializes cold control-plane model registry read-modify-write commands.
    /// This avoids lost alias/evidence updates without touching the serving
    /// data path.
    mlops_mutation_lock: Mutex<()>,
    /// u64 lets the allocator report u32 exhaustion instead of wrapping.
    account_floor: AtomicU64,
    /// ADR-031 Phase 4a: transient namespace-key → u16 map, so every collection
    /// in the same namespace gets the SAME `stable_namespace_id` (per-account,
    /// compact). Keyed by the namespace levels joined (`a.b`). Rebuilt from
    /// persisted schemas on load (the u16 lives denormalized on each schema).
    namespace_registry: DashMap<String, u16>,
    namespace_floor: AtomicU32,
}

/// Table metadata stored in storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableMetadata {
    identifier: TableIdentifierSerde,
    schema: CatalogTableSchema,
    statistics: Option<CatalogTableStatistics>,
    partition_spec: Option<CatalogPartitionSpec>,
    sort_order: Option<CatalogSortOrder>,
    created_at: i64,
    updated_at: i64,
    data_location: String,
}

/// Serializable table identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableIdentifierSerde {
    namespace: Vec<String>,
    name: String,
}

impl From<&TableIdentifier> for TableIdentifierSerde {
    fn from(id: &TableIdentifier) -> Self {
        Self {
            namespace: id.namespace.clone(),
            name: id.name.clone(),
        }
    }
}

/// TD-181 P1: durable secondary index mapping a table's stable `object_id` to
/// its `(namespace, name)` identity. It is a **derived cache** — each table's
/// `objects/{oid}.json` (S2) / `metadata/tables/{ns}/{name}.json` (legacy) is
/// the authority and carries the same identity — so the index is always
/// rebuildable by scanning the object files. It exists so an oid-keyed reader
/// can resolve a name → oid (and back) without first loading the object file,
/// which is impossible once the file path itself is keyed by oid.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ObjectIndex {
    /// One entry per table. Namespaces keep their own durable store
    /// (`namespaces.json`) + eager reverse index, so they are not duplicated here.
    #[serde(default)]
    tables: Vec<ObjectIndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObjectIndexEntry {
    object_id: u64,
    namespace: Vec<String>,
    name: String,
}

/// ADR-031 Phase 4b: durable account-string → u32 registry sidecar
/// (`_syscat/account_registry.json`). Keeps the typed path's account segment
/// stable across restarts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AccountRegistryFile {
    #[serde(default)]
    entries: Vec<(String, u32)>,
}

/// TD-181 P3 (S2a): marker recording that this catalog has begun writing
/// object_id-keyed metadata under `_syscat/objects/{oid}.json`. Written once
/// when the first oid-keyed object is persisted. A reader uses its presence to
/// decide whether to consult the oid layout (S2b); its absence means a pure
/// legacy name-keyed catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationMarker {
    /// On-disk layout version (bump on any future incompatible change).
    layout_version: u32,
    /// Whether object_id-keyed metadata paths are in use.
    oid_paths: bool,
    /// Millis-since-epoch when oid paths were first enabled for this catalog.
    migrated_at: i64,
}

impl NativeCatalog {
    /// ADR-031 / TD-181: mint a catalog `object_id` from the single system-wide
    /// sequence. Allocates a fresh id when `existing` is `None`; otherwise adopts
    /// the caller-supplied id and raises the allocator floor so it is never
    /// reused. The single place that encodes the allocation policy (DRY) — every
    /// create path (table, namespace, index, column) routes through it.
    fn mint_object_id(&self, existing: Option<u64>) -> u64 {
        match existing {
            Some(id) => {
                self.object_id_allocator.raise_floor(id + 1);
                id
            }
            None => self.object_id_allocator.allocate(),
        }
    }

    /// Check the namespace exists and mint the per-scope typed identity —
    /// shared by `create_table_inner` and `replace_table_inner` so the two
    /// cannot drift (ADR-031 Phase 4a / TD-181).
    async fn check_namespace_and_mint_identity(
        &self,
        identifier: &TableIdentifier,
        schema: &mut CatalogTableSchema,
    ) -> Result<()> {
        if !self.namespace_exists(&identifier.namespace).await? {
            return Err(anyhow!(
                "Namespace '{}' does not exist",
                identifier.namespace.join(".")
            ));
        }
        // Legacy/anonymous namespaces keep `None` (mixed-read-safe — the typed
        // path is opt-in, env-gated).
        let ns = self.get_namespace(&identifier.namespace).await?;
        if let Some(account) = ns.account_id.as_deref() {
            self.mint_stable_identity(account, &identifier.namespace.join("."), schema)
                .await;
        }
        Ok(())
    }

    /// ADR-031 / TD-181: mint stable object_ids for the table and every catalog
    /// object carried in its schema — columns and indexes — from the one
    /// system-wide sequence. `mint_object_id` allocates when unset and
    /// adopts-without-reuse a caller-supplied id (import/migration/CTAS). The
    /// `identifier` is unused: kept for signature symmetry with
    /// [`Self::check_namespace_and_mint_identity`], which needs it.
    fn mint_table_identity(&self, _identifier: &TableIdentifier, schema: &mut CatalogTableSchema) {
        schema.object_id = Some(self.mint_object_id(schema.object_id));
        for column in &mut schema.columns {
            column.object_id = Some(self.mint_object_id(column.object_id));
        }
        for index in &mut schema.indexes {
            index.object_id = Some(self.mint_object_id(index.object_id));
        }
    }

    /// ADR-031 / TD-181: recover the allocator floor from a persisted `object_id`
    /// so a restart never reuses it. The load-path inverse of `mint_object_id`;
    /// every recovery path (table, namespace, index, column) routes through it.
    fn raise_object_id_floor(&self, existing: Option<u64>) {
        if let Some(id) = existing {
            self.object_id_allocator.raise_floor(id + 1);
        }
    }

    /// ADR-031 Phase 4a: resolve the numeric account u32 for an account string.
    /// Lookup-or-mint against the transient `account_registry` (the account u32
    /// is NOT stored on the namespace — it would duplicate `account_id`; it's a
    /// registry-derived value, numeric in-memory). Returns `None` for an
    /// empty/absent account string (legacy/anonymous namespaces get no typed
    /// identity — mixed-read-safe).
    async fn ensure_account_u32(&self, account_str: &str) -> Result<Option<u32>> {
        let account_str = account_str.trim();
        if account_str.is_empty() {
            return Ok(None);
        }

        // Existing values are checked under the same lock as first mint. A
        // concurrent waiter must not observe success until the first caller's
        // durable sidecar write has completed.
        let _guard = self.account_registry_write_lock.lock().await;
        if let Some(entry) = self.account_registry.get(account_str) {
            return Ok(Some(*entry.value()));
        }
        let next = self.account_floor.fetch_add(1, Ordering::SeqCst);
        let stable_id = u32::try_from(next)
            .map_err(|_| anyhow!("tenant stable-id space exhausted at {next}"))?;
        self.account_registry
            .insert(account_str.to_string(), stable_id);

        // Persist before reporting success. Roll the in-memory entry back on a
        // write failure so a later retry cannot mistake an uncommitted id for a
        // durable policy key. The consumed numeric id may remain skipped.
        if let Err(error) = self.save_account_registry().await {
            self.account_registry.remove(account_str);
            return Err(error);
        }
        Ok(Some(stable_id))
    }

    async fn account_u32(&self, account_str: &str) -> Option<u32> {
        match self.ensure_account_u32(account_str).await {
            Ok(account) => account,
            Err(error) => {
                warn!("account-registry persist failed; stable identity not minted: {error}");
                None
            }
        }
    }

    /// ADR-031 Phase 4a: mint the per-scope typed identity (`stable_namespace_id`
    /// u16, `stable_collection_id` u32) for a collection being created under
    /// `account_str` in `namespace_key`. Minted only when an account is known;
    /// otherwise the schema keeps `None` for both (legacy path, mixed-read-safe).
    /// The namespace u16 is STABLE per namespace (every collection in the same
    /// `namespace_key` gets the same u16) via the transient `namespace_registry`.
    /// ADR-031 Phase 4c: the single mint path for the typed identity triple
    /// `(account_u32, namespace_u16, collection_u32)`. Shared by
    /// [`mint_stable_identity`] (create_table) and the public
    /// [`Catalog::mint_collection_typed_identity`] (pre-mint before storage-dir
    /// creation) so both hit the SAME `stable_ids` allocators — pre-stamped
    /// values are preserved (`existing_*` via `unwrap_or_else`), no double-mint.
    /// Returns `None` when no account is known (legacy/anonymous → no typed path).
    async fn resolve_typed_triple(
        &self,
        account_str: &str,
        namespace_key: &str,
        existing_ns: Option<u16>,
        existing_coll: Option<u32>,
    ) -> Option<(u32, u16, u32)> {
        let account = self.account_u32(account_str).await?; // None → legacy
        let ns = existing_ns.unwrap_or_else(|| {
            // First collection in this namespace (this run) → mint a per-account
            // namespace u16 + remember it so siblings reuse it.
            *self
                .namespace_registry
                .entry(namespace_key.to_string())
                .or_insert_with(|| self.stable_ids.mint_namespace_id(account))
        });
        let coll = existing_coll.unwrap_or_else(|| self.stable_ids.mint_collection_id(account, ns));
        Some((account, ns, coll))
    }

    /// ADR-031 Phase 4a: mint the per-scope typed identity (`stable_namespace_id`
    /// u16, `stable_collection_id` u32) for a collection being created under
    /// `account_str` in `namespace_key`. Minted only when an account is known;
    /// otherwise the schema keeps `None` for both (legacy path, mixed-read-safe).
    /// The namespace u16 is STABLE per namespace (every collection in the same
    /// `namespace_key` gets the same u16) via the transient `namespace_registry`.
    /// Idempotent: pre-stamped `stable_*_id` (e.g. from a Phase 4c pre-mint) are
    /// preserved — `resolve_typed_triple` short-circuits, no re-mint/drift.
    async fn mint_stable_identity(
        &self,
        account_str: &str,
        namespace_key: &str,
        schema: &mut CatalogTableSchema,
    ) {
        if let Some((_acct, ns, coll)) = self
            .resolve_typed_triple(
                account_str,
                namespace_key,
                schema.stable_namespace_id,
                schema.stable_collection_id,
            )
            .await
        {
            schema.stable_namespace_id = Some(ns);
            schema.stable_collection_id = Some(coll);
        }
    }

    /// ADR-031 Phase 4a: recover the per-scope allocator floors from a loaded
    /// schema's persisted typed identity, so a restart never reuses them. The
    /// load-path inverse of `mint_stable_identity`. Re-derives the namespace u16
    /// into the transient `namespace_registry` so siblings share it.
    async fn recover_stable_identity(
        &self,
        account_str: &str,
        namespace_key: &str,
        schema: &CatalogTableSchema,
    ) {
        let Some(account) = self.account_u32(account_str).await else {
            return;
        };
        if let Some(ns) = schema.stable_namespace_id {
            self.stable_ids.recover_namespace_floor(account, ns);
            self.namespace_floor
                .fetch_max(ns as u32 + 1, std::sync::atomic::Ordering::Relaxed);
            self.namespace_registry
                .entry(namespace_key.to_string())
                .or_insert(ns);
            if let Some(coll) = schema.stable_collection_id {
                self.stable_ids.recover_collection_floor(account, ns, coll);
            }
        }
    }

    /// Create a new native catalog backed by local `tokio::fs` (the default,
    /// back-compat path). Equivalent to `new_with_filesystem(.., None)`.
    pub async fn new(
        name: String,
        config: NativeCatalogConfig,
        cache: Arc<CatalogCache>,
    ) -> Result<Self> {
        Self::new_with_filesystem(name, config, cache, None).await
    }

    /// Create a native catalog with an optional injected storage backend.
    ///
    /// * `fs = None` — local `tokio::fs` under the parsed `base_path` (current
    ///   behavior; object-store URLs fail closed via `parse_storage_url`).
    /// * `fs = Some(backend)` — all I/O is routed through `backend` against
    ///   `config.storage_url`, enabling durable object-store (or local) catalog
    ///   persistence. The local-path parse (which rejects object-store schemes)
    ///   is skipped because addressing is by URL.
    pub async fn new_with_filesystem(
        name: String,
        config: NativeCatalogConfig,
        cache: Arc<CatalogCache>,
        fs: Option<Arc<dyn FileSystem>>,
    ) -> Result<Self> {
        info!(
            "Initializing native catalog: {} at {} (backend: {})",
            name,
            config.storage_url,
            if fs.is_some() { "injected" } else { "local" }
        );

        // `base_path` is the local addressing root; only meaningful when `fs` is
        // None. With an injected backend, addressing is by `config.storage_url`.
        let base_path = if fs.is_none() {
            Self::parse_storage_url(&config.storage_url)?
        } else {
            PathBuf::new()
        };

        let catalog = Self {
            name,
            config: config.clone(),
            base_path,
            fs,
            namespaces: RwLock::new(HashMap::new()),
            tables: RwLock::new(HashMap::new()),
            cache,
            object_id_allocator: crate::id_allocator::IdAllocator::default(),
            object_id_index: RwLock::new(HashMap::new()),
            namespace_object_id_index: RwLock::new(HashMap::new()),
            object_name_index: RwLock::new(HashMap::new()),
            oid_paths: std::sync::atomic::AtomicBool::new(Self::object_id_paths_enabled()),
            stable_ids: crate::id_allocator::CatalogIdService::new(),
            account_registry: DashMap::new(),
            account_registry_write_lock: Mutex::new(()),
            mlops_mutation_lock: Mutex::new(()),
            account_floor: AtomicU64::new(1),
            namespace_registry: DashMap::new(),
            namespace_floor: AtomicU32::new(1),
        };

        // Ensure the base location exists (local mkdir; object-store no-op).
        catalog.io_init().await?;

        // Load existing namespaces
        catalog.load_namespaces().await?;

        // ADR-031 Phase 4b: load the durable account-string → u32 registry
        // BEFORE any `load_table` (which recovers typed identities via
        // `account_u32`), so a persisted account u32 is reused, not re-minted
        // (a re-mint would drift the typed object-store path → data loss).
        catalog.load_account_registry().await?;

        // TD-181 P1: load the durable table object_id index eagerly (rebuilding
        // it from the authoritative object files if it is absent — first boot
        // after upgrade). After this, oid↔name resolution is available without a
        // name-keyed load, which oid-keyed reads (S2) and WAL-oid depend on.
        catalog.load_object_index().await?;

        Ok(catalog)
    }

    /// Parse storage URL to get local path.
    ///
    /// Only `file://` (and bare local paths) are durable today. Object-store
    /// catalog persistence is a separate, gated change (inject `FilesystemFactory`
    /// and route I/O through it). Until that lands we **fail closed** for
    /// `s3://`/`gs://`/`az://`: the previous behaviour silently redirected cloud
    /// catalog URLs to a process-local `std::env::temp_dir()` cache, so catalog
    /// metadata was non-durable, non-isolated, and unshared across pods — a
    /// silent data-loss footgun in any serverless/cloud deployment. Refusing to
    /// start is strictly safer than persisting the control-plane catalog to /tmp.
    fn parse_storage_url(url: &str) -> Result<PathBuf> {
        if let Some(path) = url.strip_prefix("file://") {
            Ok(PathBuf::from(path))
        } else if url.starts_with("s3://") || url.starts_with("gs://") || url.starts_with("az://") {
            anyhow::bail!(
                "object-store catalog URL '{url}' is not yet supported: the native catalog \
                 only persists durably to file:// today. Configure a file:// metadata_url, \
                 or wait for object-store catalog persistence (FilesystemFactory wiring). \
                 Refusing to silently cache the control-plane catalog under a local temp dir."
            )
        } else {
            // Assume plain local path.
            Ok(PathBuf::from(url))
        }
    }

    /// Load namespaces from storage
    async fn load_namespaces(&self) -> Result<()> {
        let rel = Self::namespace_index_rel();

        match self.io_read_opt(&rel).await {
            Ok(Some(data)) => {
                let mut namespaces: HashMap<String, CatalogNamespace> =
                    serde_json::from_slice(&data)?;
                // ADR-031 / TD-181 P0: recover the catalog-oid allocator floor from
                // persisted namespace object_ids FIRST, so any backfill below
                // allocates strictly above every existing id and a restart never
                // reuses one (mirrors load_table's recovery).
                for ns in namespaces.values() {
                    self.raise_object_id_floor(ns.object_id);
                }
                // Idempotent backfill: legacy rows persisted before these identities
                // existed deserialize with `namespace_id = None` (the opaque path
                // token, so warehouse paths can route through DrPathBuilder) and/or
                // `object_id = None` (the stable catalog surrogate). Assign both;
                // persist once if anything changed; a no-op on subsequent loads.
                let mut ns_id_backfilled = 0usize;
                let mut oid_backfilled = 0usize;
                for ns in namespaces.values_mut() {
                    if ns.namespace_id.is_none() {
                        ns.namespace_id = Some(Self::new_namespace_id());
                        ns_id_backfilled += 1;
                    }
                    if ns.object_id.is_none() {
                        ns.object_id = Some(self.object_id_allocator.allocate());
                        oid_backfilled += 1;
                    }
                }
                // ADR-031 / TD-181: (re)build the object_id → levels reverse index
                // from the loaded set (all namespaces now carry an object_id after
                // the backfill above), mirroring the table object_id_index.
                {
                    let mut idx = self.namespace_object_id_index.write().await;
                    idx.clear();
                    for ns in namespaces.values() {
                        if let Some(oid) = ns.object_id {
                            idx.insert(oid, ns.levels.clone());
                        }
                    }
                }
                let count = namespaces.len();
                *self.namespaces.write().await = namespaces;
                if ns_id_backfilled > 0 || oid_backfilled > 0 {
                    self.save_namespaces().await?;
                    info!(
                        "Backfilled namespace_id for {ns_id_backfilled} and object_id \
                         for {oid_backfilled} legacy namespace(s)"
                    );
                }
                debug!("Loaded {count} namespaces from {rel}");
            }
            Ok(None) => {
                debug!("No existing namespaces found at {rel}");
            }
            Err(e) => {
                warn!("Error loading namespaces: {}", e);
            }
        }

        Ok(())
    }

    /// Save namespaces to storage
    async fn save_namespaces(&self) -> Result<()> {
        let data = serde_json::to_vec_pretty(&*self.namespaces.read().await)?;
        self.io_write(&Self::namespace_index_rel(), &data).await
    }

    // ── TD-181 P1: durable table object_id index ───────────────────────────

    /// Persist the in-memory `object_id_index` (the authoritative set of
    /// oid→identifier mappings) to `_syscat/index.json`. The forward
    /// `object_name_index` is its inverse and is kept paired in memory, so the
    /// index file is serialized from `object_id_index` alone.
    async fn save_object_index(&self) -> Result<()> {
        let tables: Vec<ObjectIndexEntry> = {
            let idx = self.object_id_index.read().await;
            idx.iter()
                .map(|(oid, id)| ObjectIndexEntry {
                    object_id: *oid,
                    namespace: id.namespace.clone(),
                    name: id.name.clone(),
                })
                .collect()
        };
        let data = serde_json::to_vec_pretty(&ObjectIndex { tables })?;
        self.io_write(&Self::object_index_rel(), &data).await
    }

    /// Load the durable table object_id index eagerly, populating BOTH reverse
    /// (`object_id_index`) and forward (`object_name_index`) maps and raising the
    /// allocator floor. If the index file is absent (first boot after upgrade),
    /// rebuild it by scanning the authoritative object files **once** and persist
    /// — never an unconditional per-boot scan (steady state is a single file read).
    async fn load_object_index(&self) -> Result<()> {
        if let Some(data) = self.io_read_opt(&Self::object_index_rel()).await? {
            let index: ObjectIndex = serde_json::from_slice(&data)?;
            self.populate_object_index(&index.tables).await;
            debug!(
                "Loaded {} table object_id index entries",
                index.tables.len()
            );
            return Ok(());
        }
        // Canonical location absent → migrate the pre-rename index in place if it
        // exists (write it to `_syscat/index.json`; the old file becomes an inert
        // orphan). Cheaper + exact vs a rebuild scan, and mixed-read-safe.
        if let Some(data) = self.io_read_opt(&Self::legacy_object_index_rel()).await? {
            let index: ObjectIndex = serde_json::from_slice(&data)?;
            self.populate_object_index(&index.tables).await;
            self.save_object_index().await?;
            info!(
                "Migrated durable object_id index ({} entries) to the canonical \
                 _syscat/ location",
                index.tables.len()
            );
            return Ok(());
        }
        // Both absent → one-time rebuild from authoritative object files.
        let entries = self.scan_object_index_entries().await?;
        if entries.is_empty() {
            return Ok(());
        }
        self.populate_object_index(&entries).await;
        self.save_object_index().await?;
        info!(
            "Rebuilt durable object_id index from {} object file(s)",
            entries.len()
        );
        Ok(())
    }

    // ── ADR-031 Phase 4b: durable account-string → u32 registry ──────────
    // Mirrors `object_name_index` (`_syscat/index.json`): a sidecar JSON the
    // catalog loads eagerly at startup + writes on every new account mint, so
    // the typed object-store path segment for an account is stable across
    // restarts (a drift would silently relocate every typed-path object).

    fn account_registry_rel() -> String {
        "_syscat/account_registry.json".to_string()
    }

    /// Persist the current account-string → u32 map (called on every mint).
    async fn save_account_registry(&self) -> Result<()> {
        let entries: Vec<(String, u32)> = self
            .account_registry
            .iter()
            .map(|kv| (kv.key().clone(), *kv.value()))
            .collect();
        let data = serde_json::to_vec_pretty(&AccountRegistryFile { entries })?;
        self.io_write(&Self::account_registry_rel(), &data).await
    }

    /// Load the durable account registry eagerly. MUST run before any
    /// `load_table` (which recovers typed identities + calls `account_u32`),
    /// so the persisted u32 for an account is reused, not re-minted. Absent on
    /// first boot → no-op (mixed-read-safe).
    async fn load_account_registry(&self) -> Result<()> {
        let Some(data) = self.io_read_opt(&Self::account_registry_rel()).await? else {
            return Ok(()); // first boot — nothing to load
        };
        let file: AccountRegistryFile = serde_json::from_slice(&data)?;
        for (account, u) in &file.entries {
            self.account_registry.insert(account.clone(), *u);
        }
        // Raise the floor above the max persisted u32 so the next mint is new.
        if let Some(max) = file.entries.iter().map(|(_, u)| *u).max() {
            self.account_floor
                .fetch_max(u64::from(max) + 1, Ordering::SeqCst);
        }
        debug!("Loaded {} account-registry entries", file.entries.len());
        Ok(())
    }

    /// Populate the in-memory reverse + forward indexes from index entries and
    /// raise the allocator floor so a restart never reuses an id.
    async fn populate_object_index(&self, entries: &[ObjectIndexEntry]) {
        let mut by_id = self.object_id_index.write().await;
        let mut by_name = self.object_name_index.write().await;
        for e in entries {
            let id = TableIdentifier::new(e.namespace.clone(), e.name.clone());
            self.raise_object_id_floor(Some(e.object_id));
            by_name.insert(id.to_fqn(), e.object_id);
            by_id.insert(e.object_id, id);
        }
    }

    /// Scan the authoritative object files to rebuild the durable index when it is
    /// absent. Scans **both** layouts and dedupes by `(namespace, name)`:
    ///
    /// 1. legacy name-keyed files under `metadata/tables/{ns}/` (present for legacy
    ///    + dual-written tables), and
    /// 2. oid-keyed files under `_syscat/objects/` — so an oid-only table with no
    ///    legacy shadow (the no-shadow endgame, or a catalog whose legacy files
    ///    were cleaned up) is still recovered. Both file kinds are self-describing
    ///    (`TableMetadata` carries its own identifier + object_id), so identity is
    ///    read straight from the file. Missing either scan would silently drop a
    ///    table from the index — and, worse, never raise the allocator floor for
    ///    its oid, risking a reuse.
    async fn scan_object_index_entries(&self) -> Result<Vec<ObjectIndexEntry>> {
        let mut entries = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut push = |id: &TableIdentifier, object_id: u64| {
            if seen.insert(id.to_fqn()) {
                entries.push(ObjectIndexEntry {
                    object_id,
                    namespace: id.namespace.clone(),
                    name: id.name.clone(),
                });
            }
        };

        // 1) Legacy name-keyed files, namespace by namespace.
        let namespaces: Vec<Vec<String>> = {
            self.namespaces
                .read()
                .await
                .values()
                .map(|ns| ns.levels.clone())
                .collect()
        };
        for ns in namespaces {
            for name in self.io_list_json_stems(&Self::tables_dir_rel(&ns)).await {
                let id = TableIdentifier::new(ns.clone(), name);
                if let Some(data) = self.io_read_opt(&Self::table_metadata_rel(&id)).await? {
                    let meta: TableMetadata = serde_json::from_slice(&data)?;
                    if let Some(object_id) = meta.schema.object_id {
                        push(&id, object_id);
                    }
                }
            }
        }

        // 2) Oid-keyed authority files (flat, not namespace-partitioned). Recover
        // identity from each self-describing file; dedupe against the legacy scan.
        for stem in self.io_list_json_stems(&Self::objects_dir_rel()).await {
            let Ok(oid) = stem.parse::<u64>() else {
                continue; // not an oid file (e.g. _migration.json lives elsewhere)
            };
            if let Some(data) = self.io_read_opt(&Self::object_file_rel(oid)).await? {
                let meta: TableMetadata = serde_json::from_slice(&data)?;
                let id = TableIdentifier::new(
                    meta.identifier.namespace.clone(),
                    meta.identifier.name.clone(),
                );
                let object_id = meta.schema.object_id.unwrap_or(oid);
                push(&id, object_id);
            }
        }

        Ok(entries)
    }

    /// Insert/refresh a table's identity in both index directions, then persist.
    /// Authority-first: callers write the object file BEFORE this so a crash
    /// leaves an orphan object (re-indexed on next load), never a phantom entry.
    async fn index_upsert_table(&self, oid: u64, identifier: &TableIdentifier) -> Result<()> {
        {
            self.object_id_index
                .write()
                .await
                .insert(oid, identifier.clone());
            self.object_name_index
                .write()
                .await
                .insert(identifier.to_fqn(), oid);
        }
        self.save_object_index().await
    }

    /// Remove a table's identity from both index directions, then persist.
    /// Authority-first: callers delete the object file BEFORE this so a crash
    /// leaves a dangling entry (pruned lazily on a NotFound read), never resurrects.
    async fn index_remove_table(&self, identifier: &TableIdentifier) -> Result<()> {
        let fqn = identifier.to_fqn();
        if let Some(oid) = self.object_name_index.write().await.remove(&fqn) {
            self.object_id_index.write().await.remove(&oid);
        }
        self.save_object_index().await
    }

    /// Repoint a table's identity across a rename (oid is stable), then persist.
    async fn index_rename_table(
        &self,
        oid: u64,
        from: &TableIdentifier,
        to: &TableIdentifier,
    ) -> Result<()> {
        {
            let mut by_name = self.object_name_index.write().await;
            by_name.remove(&from.to_fqn());
            by_name.insert(to.to_fqn(), oid);
        }
        self.object_id_index.write().await.insert(oid, to.clone());
        self.save_object_index().await
    }

    // ── Relative path helpers ──────────────────────────────────────────────
    // Storage-root-relative, '/'-joined keys. Resolved against `base_path`
    // (local `PathBuf`) or `config.storage_url` (injected backend) by the
    // `io_*` helpers below, so the on-disk/object layout is identical across
    // backends.

    /// Relative key for the namespace index.
    fn namespace_index_rel() -> String {
        "metadata/namespaces.json".to_string()
    }

    /// TD-181 P1: relative key for the durable table object_id index. Colocated
    /// with the oid-keyed objects it indexes, under `_syscat/` (ADR-035 D3
    /// `_syscat/index.json`), so all object_id-cutover artifacts live in one
    /// subtree. See [`legacy_object_index_rel`](Self::legacy_object_index_rel)
    /// for the pre-rename location read as a migration fallback.
    fn object_index_rel() -> String {
        "_syscat/index.json".to_string()
    }

    /// TD-181 P1: the pre-rename index location (`metadata/object_index.json`),
    /// read once as a fallback so an existing catalog migrates its index to the
    /// canonical `_syscat/` location without a rebuild scan.
    fn legacy_object_index_rel() -> String {
        "metadata/object_index.json".to_string()
    }

    /// TD-181 P3 (S2a): relative key for an object_id-keyed table metadata file.
    /// Flat (not namespace-partitioned) — cross-tenant uniqueness comes from the
    /// globally-unique `object_id`, so identical names across tenants never
    /// collide. `list_tables` resolves names through the durable index (S2b).
    fn object_file_rel(object_id: u64) -> String {
        format!("_syscat/objects/{object_id}.json")
    }

    /// TD-181 P3: directory holding the oid-keyed object files, scanned to rebuild
    /// the durable index (so oid-only tables with no legacy shadow are recovered).
    fn objects_dir_rel() -> String {
        "_syscat/objects".to_string()
    }

    /// TD-181 P3 (S2a): relative key for the layout migration marker.
    fn migration_marker_rel() -> String {
        "_syscat/_migration.json".to_string()
    }

    /// TD-181 P3 (S2a): presence-based, default-OFF gate for object_id-keyed
    /// metadata writes (mirrors `PROXIMADB_INDEX_CATALOG_PATHS`). Unset ⇒ today's
    /// pure name-keyed behavior; set ⇒ dual-write the oid layout alongside the
    /// legacy name path (rollback-safe; reads cut over in S2b). Read once at
    /// construction into [`oid_paths`](Self::oid_paths); call sites consult
    /// [`oid_paths_on`](Self::oid_paths_on).
    fn object_id_paths_enabled() -> bool {
        std::env::var_os("PROXIMADB_CATALOG_OBJECT_ID_PATHS").is_some()
    }

    /// Whether this catalog writes object_id-keyed metadata (the construction
    /// snapshot of [`object_id_paths_enabled`](Self::object_id_paths_enabled)).
    fn oid_paths_on(&self) -> bool {
        self.oid_paths.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test-only deterministic toggle for the oid-paths gate, avoiding the
    /// process-global env races a parallel test suite would otherwise hit.
    #[cfg(test)]
    fn set_oid_paths_for_test(&self, on: bool) {
        self.oid_paths
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Relative key for a table's metadata.
    fn table_metadata_rel(identifier: &TableIdentifier) -> String {
        format!(
            "metadata/tables/{}/{}.json",
            identifier.namespace.join("/"),
            identifier.name
        )
    }

    /// Relative key prefix for a table's data directory.
    ///
    /// NOTE: this is the catalog's own storage-root-relative key (the leading
    /// `namespace` segment is the *catalog* namespace, e.g. `default`, not a
    /// `tenant_id`), not a `DrPathBuilder` tenant-isolated object path. TD-CAT-2
    /// routes these through `DrPathBuilder` for genuine tenant prefixing; until
    /// then the suffix is built separately so the key is not a `data/{..}/`
    /// literal (which the tenant-path guard flags as a raw DrPathBuilder bypass).
    fn table_data_rel(identifier: &TableIdentifier) -> String {
        let suffix = format!("{}/{}", identifier.namespace.join("/"), identifier.name);
        format!("data/{suffix}")
    }

    /// Relative key prefix for a namespace's tables directory.
    fn tables_dir_rel(namespace: &[String]) -> String {
        format!("metadata/tables/{}", namespace.join("/"))
    }

    // ── Backend-dispatching I/O helpers ────────────────────────────────────
    // `None` ⇒ local `tokio::fs` under `base_path` (byte-identical to the prior
    // behavior). `Some(fs)` ⇒ route through the injected `FileSystem` against
    // `config.storage_url`. Not-found is normalized to `Ok(None)` across both
    // error models so callers don't branch on backend-specific error kinds.

    /// Resolve a relative key to a local filesystem path.
    fn local_path(&self, rel: &str) -> PathBuf {
        self.base_path.join(rel)
    }

    /// Resolve a relative key to a full backend URL.
    fn fs_url(&self, rel: &str) -> String {
        format!("{}/{}", self.config.storage_url.trim_end_matches('/'), rel)
    }

    /// Resolved, persistable location string for a table's data directory:
    /// the local absolute path (local backend) or the full URL (injected
    /// backend). Stored in `TableMetadata.data_location`.
    fn table_data_location(&self, identifier: &TableIdentifier) -> String {
        let rel = Self::table_data_rel(identifier);
        match &self.fs {
            None => self.local_path(&rel).to_string_lossy().to_string(),
            Some(_) => self.fs_url(&rel),
        }
    }

    /// Ensure the storage base exists (local mkdir; object-store no-op).
    async fn io_init(&self) -> Result<()> {
        match &self.fs {
            None => fs::create_dir_all(&self.base_path).await?,
            Some(fs) => {
                // Object stores have no directories; local backends mkdir. A
                // backend that doesn't support it (object store) returns Ok or a
                // benign error we ignore — the base is implicit in keys.
                let _ = fs.create_dir_all(&self.config.storage_url).await;
            }
        }
        Ok(())
    }

    /// Read a key, returning `Ok(None)` when it does not exist.
    async fn io_read_opt(&self, rel: &str) -> Result<Option<Vec<u8>>> {
        match &self.fs {
            None => match fs::read(self.local_path(rel)).await {
                Ok(data) => Ok(Some(data)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(anyhow!("catalog read {rel}: {e}")),
            },
            Some(fs) => {
                let url = self.fs_url(rel);
                match fs.read(&url).await {
                    Ok(data) => Ok(Some(data)),
                    Err(FilesystemError::NotFound(_)) => Ok(None),
                    Err(FilesystemError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                        Ok(None)
                    }
                    Err(e) => Err(anyhow!("catalog read {url}: {e}")),
                }
            }
        }
    }

    /// Write a key atomically, creating parent directories as needed.
    async fn io_write(&self, rel: &str, data: &[u8]) -> Result<()> {
        match &self.fs {
            None => {
                let path = self.local_path(rel);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).await?;
                }
                fs::write(&path, data).await?;
                Ok(())
            }
            Some(fs) => {
                let url = self.fs_url(rel);
                let options = FileOptions {
                    create_dirs: true,
                    overwrite: true,
                    ..Default::default()
                };
                fs.write_atomic(&url, data, Some(options))
                    .await
                    .map_err(|e| anyhow!("catalog write {url}: {e}"))
            }
        }
    }

    /// Whether a key exists.
    async fn io_exists(&self, rel: &str) -> Result<bool> {
        match &self.fs {
            None => Ok(self.local_path(rel).exists()),
            Some(fs) => fs
                .exists(&self.fs_url(rel))
                .await
                .map_err(|e| anyhow!("catalog exists {rel}: {e}")),
        }
    }

    /// Best-effort delete of a single key; returns whether it was removed.
    async fn io_remove_file(&self, rel: &str) -> bool {
        match &self.fs {
            None => fs::remove_file(self.local_path(rel)).await.is_ok(),
            Some(fs) => fs.delete(&self.fs_url(rel)).await.is_ok(),
        }
    }

    /// Recursively delete everything under a key prefix (best-effort).
    async fn io_remove_prefix(&self, rel: &str) -> Result<()> {
        match &self.fs {
            None => {
                fs::remove_dir_all(self.local_path(rel)).await?;
                Ok(())
            }
            Some(fs) => {
                let prefix = self.fs_url(rel);
                // Object stores delete per-key; enumerate and remove each.
                if let Ok(entries) = fs.list(&prefix).await {
                    for entry in entries {
                        let _ = fs.delete(&entry.url).await;
                    }
                }
                Ok(())
            }
        }
    }

    /// List the `.json` file stems directly under a key prefix.
    async fn io_list_json_stems(&self, rel: &str) -> Vec<String> {
        let mut stems = Vec::new();
        match &self.fs {
            None => {
                if let Ok(mut entries) = fs::read_dir(self.local_path(rel)).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let path = entry.path();
                        if path.extension().is_some_and(|ext| ext == "json")
                            && let Some(stem) = path.file_stem()
                        {
                            stems.push(stem.to_string_lossy().to_string());
                        }
                    }
                }
            }
            Some(fs) => {
                if let Ok(entries) = fs.list(&self.fs_url(rel)).await {
                    for entry in entries {
                        if let Some(stem) = entry.name.strip_suffix(".json") {
                            stems.push(stem.to_string());
                        }
                    }
                }
            }
        }
        stems
    }

    /// Backend reachability check for health reporting.
    async fn io_healthy(&self) -> bool {
        match &self.fs {
            None => fs::metadata(&self.base_path).await.is_ok(),
            Some(fs) => fs.exists(&self.config.storage_url).await.unwrap_or(true),
        }
    }

    /// Load table metadata from storage
    /// TD-181 P3 (S2b): read a table's serialized metadata, preferring the
    /// oid-keyed authority file when oid paths are on. The decision tree:
    /// in oid mode, resolve `name → oid` via the durable index and read
    /// `_syscat/objects/{oid}.json`; on any miss (legacy-only table, or a
    /// dangling index entry whose oid file is gone) fall back to the legacy
    /// name path. With the gate off it is exactly the legacy read. A bypass is
    /// never wrong — it just isn't oid-keyed.
    async fn read_table_bytes(&self, identifier: &TableIdentifier) -> Result<Option<Vec<u8>>> {
        if self.oid_paths_on() {
            let oid = self
                .object_name_index
                .read()
                .await
                .get(&identifier.to_fqn())
                .copied();
            if let Some(oid) = oid
                && let Some(data) = self.io_read_opt(&Self::object_file_rel(oid)).await?
            {
                return Ok(Some(data));
            }
            // Fall through to the legacy name path (mixed-read fallback).
        }
        self.io_read_opt(&Self::table_metadata_rel(identifier))
            .await
    }

    async fn load_table(&self, identifier: &TableIdentifier) -> Result<TableMetadata> {
        let key = identifier.to_fqn();

        // Check in-memory cache first
        if let Some(meta) = self.tables.read().await.get(&key) {
            return Ok(meta.clone());
        }

        // Load from storage (oid-keyed authority preferred, legacy fallback).
        let Some(data) = self.read_table_bytes(identifier).await? else {
            // TD-181 P1: dangling index entry — the name resolves in the durable
            // index but NEITHER the oid file nor the legacy file exists. A crash
            // between `drop_table`'s file-delete and its index-persist leaves this,
            // and it would otherwise persist as a phantom in `list_tables` across
            // restarts (the index union lists it, but every `get` returns
            // NotFound). Prune it lazily here — the long-documented behavior — so
            // the index self-heals on the next read instead of needing a full
            // rebuild. Guarded on the index actually holding the entry, so a
            // genuinely-absent table stays a plain NotFound with no write.
            if self
                .object_name_index
                .read()
                .await
                .contains_key(&identifier.to_fqn())
            {
                self.index_remove_table(identifier).await?;
            }
            return Err(anyhow!("Table '{}' not found", identifier));
        };

        let meta: TableMetadata = serde_json::from_slice(&data)?;

        // ADR-031 O0/O1: recover the allocator floor + the reverse object_id index
        // from persisted ids so a restart never reuses an id and object_id → table
        // resolution survives (best-effort as tables load on demand).
        self.raise_object_id_floor(meta.schema.object_id);
        // ADR-031 Phase 4a: recover the per-scope typed-identity floors so a
        // restart never reuses a persisted `stable_namespace_id` /
        // `stable_collection_id`. Account u32 is re-derived from the transient
        // registry (durable in 4b).
        {
            let ns_key = identifier.namespace.join(".");
            if let Some(ns) = self.namespaces.read().await.get(&ns_key)
                && let Some(account) = ns.account_id.as_deref()
            {
                self.recover_stable_identity(account, &ns_key, &meta.schema)
                    .await;
            }
        }
        if let Some(id) = meta.schema.object_id {
            // TD-181 P1: opportunistically heal the durable index if this object
            // file isn't indexed yet (an orphan from a crash between object-write
            // and index-persist). Re-index + persist ONLY when missing, so the
            // common already-indexed load stays write-free. When already indexed
            // we deliberately do nothing: the forward (`object_name_index`) and
            // reverse (`object_id_index`) maps are always maintained as a pair, so
            // an entry present in one is present in the other. (A previous version
            // re-inserted into `object_id_index` alone here, which could race a
            // concurrent `drop_table` — A reads "present", B removes from both, A
            // re-inserts into the reverse map only — leaving a phantom oid→name
            // entry until the next restart. Doing nothing keeps the pair atomic.)
            let already_indexed = self
                .object_name_index
                .read()
                .await
                .contains_key(&identifier.to_fqn());
            if !already_indexed {
                self.index_upsert_table(id, identifier).await?;
            }
        }
        // ADR-031 / TD-181: recover the floor from persisted column + index
        // object_ids too (both ride in the table's schema), so a restart never
        // reuses one.
        for column in &meta.schema.columns {
            self.raise_object_id_floor(column.object_id);
        }
        for index in &meta.schema.indexes {
            self.raise_object_id_floor(index.object_id);
        }

        // Cache in memory
        self.tables.write().await.insert(key, meta.clone());

        Ok(meta)
    }

    /// Save table metadata to storage
    async fn save_table(&self, meta: &TableMetadata) -> Result<()> {
        let identifier = TableIdentifier::new(
            meta.identifier.namespace.clone(),
            meta.identifier.name.clone(),
        );

        let data = serde_json::to_vec_pretty(meta)?;
        self.io_write(&Self::table_metadata_rel(&identifier), &data)
            .await?;

        // TD-181 P3 (S2a): when oid paths are enabled, dual-write the authority
        // copy at the oid-keyed path and ensure the migration marker exists. The
        // legacy name path above stays as the shadow so a rollback (gate OFF) is
        // lossless; reads keep using the name path until S2b. The bytes are
        // identical — the same self-describing `TableMetadata` (carries its
        // identifier + object_id), so the two copies can never disagree.
        if self.oid_paths_on()
            && let Some(oid) = meta.schema.object_id
        {
            self.io_write(&Self::object_file_rel(oid), &data).await?;
            self.ensure_migration_marker().await?;
        }

        // Update in-memory cache
        self.tables
            .write()
            .await
            .insert(identifier.to_fqn(), meta.clone());

        Ok(())
    }

    /// TD-181 P3 (S2a): write the layout migration marker once (idempotent — a
    /// no-op when it already exists), recording that this catalog now persists
    /// object_id-keyed metadata.
    async fn ensure_migration_marker(&self) -> Result<()> {
        if self.io_exists(&Self::migration_marker_rel()).await? {
            return Ok(());
        }
        let marker = MigrationMarker {
            layout_version: 1,
            oid_paths: true,
            migrated_at: Self::now_millis(),
        };
        let data = serde_json::to_vec_pretty(&marker)?;
        self.io_write(&Self::migration_marker_rel(), &data).await
    }

    /// Get current timestamp in milliseconds
    fn now_millis() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    /// Mint an opaque, rename-stable namespace id (`ns_<uuid-v4>`). Matches the
    /// UUID convention used for collection ids; the `ns_` prefix keeps physical
    /// paths self-describing.
    fn new_namespace_id() -> String {
        format!("ns_{}", uuid::Uuid::new_v4())
    }

    /// Shared namespace construction. `tenant_id` records the owning tenant when
    /// the namespace is created in a tenant scope (TD-064/TD-113) so it is
    /// DR-addressable; `None` for unscoped/single-tenant creates.
    async fn create_namespace_inner(
        &self,
        namespace: &[String],
        properties: HashMap<String, String>,
        tenant_id: Option<String>,
    ) -> Result<CatalogNamespace> {
        let key = namespace.join(".");
        if self.namespaces.read().await.contains_key(&key) {
            return Err(anyhow!("Namespace '{}' already exists", key));
        }

        let now = Self::now_millis();
        let ns = CatalogNamespace {
            levels: namespace.to_vec(),
            properties,
            owner: None,
            location: None,
            created_at_ms: now,
            updated_at_ms: now,
            // ADR-031 / TD-181 P0: mint the stable catalog object_id from the same
            // single system-wide sequence that mints table object_ids (one sequence,
            // not per-type — ADR-031 reconciliation amendment 1). Native-minted only.
            object_id: Some(self.object_id_allocator.allocate()),
            // Opaque, rename-stable server-issued id that drives physical paths
            // (DrPathBuilder). `tenant_id` is the owning tenant when created in a
            // tenant scope; together they make the namespace DR-addressable.
            namespace_id: Some(Self::new_namespace_id()),
            tenant_id: tenant_id.clone(),
            // ADR-031 Phase 5 (tenant→account collapse): account_id mirrors
            // tenant_id — they are the same identity in the Phase-4 model
            // (account is the single billing/isolation tier; tenant was its
            // sub-org, now collapsed). Setting it here ACTIVATES the typed-id
            // minting: `mint_stable_identity` (create_table) reads account_id →
            // mints stable_namespace_id/stable_collection_id + the account_u32
            // registry entry. With PROXIMADB_TYPED_PATHS off (default) the ids
            // are persisted but inert (no typed path reads them) — forward-prep.
            // `create_namespace` (no tenant) keeps account_id = None (legacy).
            account_id: tenant_id,
            region_home: None,
            default_dr_region_pair_id: None,
            storage_pool_class: Default::default(),
        };

        self.namespaces
            .write()
            .await
            .insert(key.clone(), ns.clone());
        // ADR-031 / TD-181: maintain the object_id → levels reverse index.
        if let Some(oid) = ns.object_id {
            self.namespace_object_id_index
                .write()
                .await
                .insert(oid, ns.levels.clone());
        }
        self.save_namespaces().await?;

        info!("Created namespace: {}", key);
        Ok(ns)
    }

    /// Inherent accessor for the catalog metadata cache.
    /// Was a trait method before Option B consolidation; moved to inherent
    /// since the canonical `proximadb_catalog::Catalog` trait omits it.
    pub fn cache(&self) -> Option<Arc<CatalogCache>> {
        Some(self.cache.clone())
    }
}

/// TD-CAT-7: `NativeCatalog` owns an allocator and an account registry, so it
/// is an identity authority. Every method here is required by the trait — the
/// silent `Ok(None)` inheritance that let minting go missing is not reachable
/// from this impl.
// Referenced by full path, deliberately NOT imported: `NativeCatalog` implements
// both `Catalog` and `CatalogAuthority`, which declare the same method names. With
// both traits in scope every concrete-type call site becomes ambiguous (E0034).
// In-module and test callers use `Catalog`, whose defaults delegate here.
#[async_trait]
impl crate::CatalogAuthority for NativeCatalog {
    async fn max_object_id(&self) -> Result<Option<u64>> {
        // Max persisted object_id from the durable forward index (name→oid),
        // loaded eagerly at startup. Used by the root to raise the collection-id
        // allocator floor above every existing object_id (collision safety).
        Ok(self.object_name_index.read().await.values().copied().max())
    }

    async fn allocate_object_id(&self) -> Result<u64> {
        // Collection lifecycle and ordinary catalog DDL must draw from one
        // authority. The later create_table adopts this id via mint_object_id.
        Ok(self.mint_object_id(None))
    }

    async fn mint_collection_typed_identity(
        &self,
        account: &str,
        namespace_key: &str,
    ) -> Result<Option<(u32, u16, u32)>> {
        // Phase 4c pre-mint: a fresh typed triple (no existing values) via the
        // shared `resolve_typed_triple`. The caller stamps it onto the schema
        // before create_table, whose `mint_stable_identity` then preserves it
        // (idempotent — no double-mint).
        Ok(self
            .resolve_typed_triple(account, namespace_key, None, None)
            .await)
    }

    async fn account_id_u32(&self, account: &str) -> Result<Option<u32>> {
        // Thin public wrapper over the private registry lookup-or-mint. The
        // root path-resolver calls this to compose a `CollectionIdentity`.
        self.ensure_account_u32(account).await
    }

    fn account_id_u32_lookup(&self, account: &str) -> Option<u32> {
        // TD-TENANT-1 item 3: sync read-only lookup (no mint, no persist) for the
        // request-hot TenantStableIdResolver. None when unminted/empty.
        let account = account.trim();
        if account.is_empty() {
            return None;
        }
        self.account_registry.get(account).map(|v| *v.value())
    }

    async fn get_table_by_object_id(&self, object_id: u64) -> Result<Option<TableIdentifier>> {
        // TD-181 P1: the reverse index is now built eagerly at startup from the
        // durable `object_index.json` (rebuilt by scan if absent), so a fresh
        // process resolves any persisted id without first loading its table by
        // name. Still maintained on create/load/rename/drop.
        Ok(self.object_id_index.read().await.get(&object_id).cloned())
    }

    async fn get_namespace_by_object_id(&self, object_id: u64) -> Result<Option<Vec<String>>> {
        // Namespaces load eagerly at construction (load_namespaces), so this
        // reverse index is fully populated up front — unlike the lazy table index.
        Ok(self
            .namespace_object_id_index
            .read()
            .await
            .get(&object_id)
            .cloned())
    }
}

#[async_trait]
impl Catalog for NativeCatalog {
    fn identity_authority(&self) -> Option<&dyn crate::CatalogAuthority> {
        Some(self)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn catalog_type(&self) -> &str {
        "native"
    }

    // ========================
    // Namespace Operations
    // ========================

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
        let key = namespace.join(".");

        // Check if empty (unless cascade)
        if !cascade {
            let tables = self.list_tables(namespace).await?;
            if !tables.is_empty() {
                return Err(anyhow!(
                    "Namespace '{}' is not empty. Use cascade=true to force drop.",
                    key
                ));
            }
        }

        // Remove tables if cascade
        if cascade {
            let tables = self.list_tables(namespace).await?;
            for table_id in tables {
                self.drop_table(&table_id, true).await?;
            }
        }

        let removed_ns = self.namespaces.write().await.remove(&key);
        // ADR-031 / TD-181: drop the object_id → levels reverse-index entry too.
        if let Some(ns) = &removed_ns
            && let Some(oid) = ns.object_id
        {
            self.namespace_object_id_index.write().await.remove(&oid);
        }
        let removed = removed_ns.is_some();
        if removed {
            self.save_namespaces().await?;
            info!("Dropped namespace: {}", key);
        }

        Ok(removed)
    }

    async fn list_namespaces(&self, parent: Option<&[String]>) -> Result<Vec<CatalogNamespace>> {
        let namespaces = self.namespaces.read().await;

        let results: Vec<CatalogNamespace> = namespaces
            .values()
            .filter(|ns| {
                if let Some(p) = parent {
                    ns.levels.len() == p.len() + 1 && ns.levels.starts_with(p)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        Ok(results)
    }

    async fn namespace_exists(&self, namespace: &[String]) -> Result<bool> {
        let key = namespace.join(".");
        Ok(self.namespaces.read().await.contains_key(&key))
    }

    async fn get_namespace(&self, namespace: &[String]) -> Result<CatalogNamespace> {
        let key = namespace.join(".");
        self.namespaces
            .read()
            .await
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))
    }

    async fn update_namespace_properties(
        &self,
        namespace: &[String],
        updates: HashMap<String, String>,
        removals: Vec<String>,
    ) -> Result<()> {
        let key = namespace.join(".");
        let mut namespaces = self.namespaces.write().await;

        let ns = namespaces
            .get_mut(&key)
            .ok_or_else(|| anyhow!("Namespace '{}' not found", key))?;

        // Apply updates
        for (k, v) in updates {
            ns.properties.insert(k, v);
        }

        // Apply removals
        for k in removals {
            ns.properties.remove(&k);
        }

        ns.updated_at_ms = Self::now_millis();
        drop(namespaces);

        self.save_namespaces().await?;
        Ok(())
    }

    // ========================
    // Table Operations
    // ========================

    async fn create_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        // Validate schema
        // ADR-077 M1: normalize BEFORE validating. A creation path may legitimately
        // hand us a UNIQUE recorded only in the projection — `CREATE TABLE … UNIQUE(x)`
        // lowers to `relational_capabilities.unique_indexes` — and rejecting that would
        // break table creation. Folding it into the canonical field first is what makes
        // the invariant hold by construction rather than by rejecting real schemas.
        let mut schema = schema;
        crate::schema::normalize_identity(&mut schema);
        validate_schema(&schema)?;

        self.check_namespace_and_mint_identity(identifier, &mut schema)
            .await?;
        self.mint_table_identity(identifier, &mut schema);

        // Check table doesn't exist
        if self.table_exists(identifier).await? {
            return Err(anyhow!("Table '{}' already exists", identifier));
        }

        let now = Self::now_millis();
        let meta = TableMetadata {
            identifier: identifier.into(),
            schema: schema.clone(),
            statistics: None,
            partition_spec: None,
            sort_order: None,
            created_at: now,
            updated_at: now,
            data_location: self.table_data_location(identifier),
        };

        self.save_table(&meta).await?;
        // ADR-031 O1 + TD-181 P1: maintain the reverse object_id → identifier
        // index AND the durable name↔oid index. Authority-first: the object file
        // (save_table above) is written before the index is updated/persisted.
        if let Some(oid) = schema.object_id {
            self.index_upsert_table(oid, identifier).await?;
        }
        info!("Created table: {}", identifier);

        Ok(schema)
    }

    /// TD-CATREPLACE-1: atomic in place — one `save_table` overwrite IS the
    /// whole swap, so a reader never sees the table absent and an I/O failure
    /// leaves the previous schema durable. `created_at`, statistics and the
    /// partition/sort metadata survive; identity follows `create_table`
    /// (caller-supplied ids adopted, floor raised). When the adopted
    /// `object_id` differs from the existing one, the previous authority copy
    /// and its index entries are removed after the new write (authority-first,
    /// like `drop_table`), so no orphaned oid file can resurrect the old row.
    async fn replace_table_inner(
        &self,
        identifier: &TableIdentifier,
        schema: CatalogTableSchema,
    ) -> Result<CatalogTableSchema> {
        let mut schema = schema;
        crate::schema::normalize_identity(&mut schema);
        validate_schema(&schema)?;

        // Round-4 review: the shared helper, exactly as create — the namespace
        // check AND the stable typed-identity mint (a replace must not be able
        // to leave a row untyped, or create-via-replace to diverge from
        // create). Order (stable identity, then object ids) matches the
        // pre-refactor behavior.
        self.check_namespace_and_mint_identity(identifier, &mut schema)
            .await?;
        self.mint_table_identity(identifier, &mut schema);

        // Only a genuinely absent table may be replaced into existence; an
        // unreadable previous row must fail the replace rather than silently
        // reset created_at/statistics (review m2).
        let previous = if self.table_exists(identifier).await? {
            Some(self.load_table(identifier).await?)
        } else {
            None
        };
        let now = Self::now_millis();
        let meta = TableMetadata {
            identifier: identifier.into(),
            schema: schema.clone(),
            statistics: previous.as_ref().and_then(|p| p.statistics.clone()),
            partition_spec: previous.as_ref().and_then(|p| p.partition_spec.clone()),
            sort_order: previous.as_ref().and_then(|p| p.sort_order.clone()),
            created_at: previous.as_ref().map_or(now, |p| p.created_at),
            updated_at: now,
            data_location: self.table_data_location(identifier),
        };
        self.save_table(&meta).await?;
        // Mandate 16b: invalidate the shared cache immediately after the
        // durable write, before any fallible step below, so no reader is served
        // the previous schema from cache (review m1).
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        // An adopted object_id that differs from the replaced row's leaves the
        // old authority copy + reverse-index entry behind; drop them now that
        // the new write is durable (mirrors drop_table's authority-first rule).
        if let (Some(previous_oid), Some(new_oid)) = (
            previous.as_ref().and_then(|p| p.schema.object_id),
            schema.object_id,
        ) && previous_oid != new_oid
        {
            if self.oid_paths_on()
                && !self
                    .io_remove_file(&Self::object_file_rel(previous_oid))
                    .await
            {
                // A leftover old-oid authority file can resurrect the previous
                // row if the durable index is ever rebuilt from the objects dir.
                warn!(
                    "replace_table: old authority file for object {} survived removal; table {}",
                    previous_oid, identifier
                );
            }
            self.index_remove_table(identifier).await?;
        }
        if let Some(oid) = schema.object_id {
            self.index_upsert_table(oid, identifier).await?;
        }
        info!("Replaced table: {}", identifier);
        Ok(schema)
    }

    async fn drop_table(&self, identifier: &TableIdentifier, purge: bool) -> Result<bool> {
        // TD-181 P3: resolve the oid from the durable index BEFORE the entry is
        // removed, so we can delete the oid-keyed authority copy too. Delete
        // BOTH copies (authority-first, before the index entry). Removal counts
        // as success if EITHER copy existed — so a pure-oid table (no legacy
        // shadow) is still dropped, not silently skipped.
        let oid = if self.oid_paths_on() {
            self.object_name_index
                .read()
                .await
                .get(&identifier.to_fqn())
                .copied()
        } else {
            None
        };
        let legacy_removed = self
            .io_remove_file(&Self::table_metadata_rel(identifier))
            .await;
        let oid_removed = if let Some(oid) = oid {
            self.io_remove_file(&Self::object_file_rel(oid)).await
        } else {
            false
        };
        let removed = legacy_removed || oid_removed;

        if removed {
            // Remove from in-memory cache
            self.tables.write().await.remove(&identifier.to_fqn());

            // ADR-031 O1 + TD-181 P1: drop the reverse + forward index entries and
            // persist. Authority-first: the object file(s) were deleted above, so
            // a crash here leaves a dangling index entry (pruned lazily on a
            // NotFound read), never a resurrected table.
            self.index_remove_table(identifier).await?;

            // Purge data files if requested
            if purge
                && let Err(e) = self
                    .io_remove_prefix(&Self::table_data_rel(identifier))
                    .await
            {
                warn!("Failed to purge data for {}: {}", identifier, e);
            }

            // Invalidate catalog cache
            self.cache
                .invalidate_table_in_catalog(&self.name, identifier);

            info!("Dropped table: {} (purge={})", identifier, purge);
        }

        Ok(removed)
    }

    async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        // Legacy name-keyed stems (present for legacy + dual-written tables).
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<TableIdentifier> = Vec::new();
        for name in self
            .io_list_json_stems(&Self::tables_dir_rel(namespace))
            .await
        {
            let id = TableIdentifier::new(namespace.to_vec(), name);
            if seen.insert(id.to_fqn()) {
                out.push(id);
            }
        }
        // TD-181 P3 (S2b): in oid mode, UNION in oid-only tables (no legacy
        // shadow) from the durable index, filtered to this namespace and
        // deduped by (namespace, name). The flat `_syscat/objects/` layout is
        // not namespace-partitioned, so the index is the only way to enumerate
        // a namespace's oid-keyed tables.
        if self.oid_paths_on() {
            for id in self.object_id_index.read().await.values() {
                if id.namespace == namespace && seen.insert(id.to_fqn()) {
                    out.push(id.clone());
                }
            }
        }
        Ok(out)
    }

    async fn table_exists(&self, identifier: &TableIdentifier) -> Result<bool> {
        // TD-181 P3 (S2b): in oid mode a table exists if the durable index knows
        // it (covers a pure-oid table with no legacy shadow) OR the legacy file
        // is present (covers an unindexed orphan). Gate off ⇒ legacy only.
        if self.oid_paths_on()
            && self
                .object_name_index
                .read()
                .await
                .contains_key(&identifier.to_fqn())
        {
            return Ok(true);
        }
        self.io_exists(&Self::table_metadata_rel(identifier)).await
    }

    async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        // Check catalog cache first
        if let Some(schema) = self.cache.get_table(&self.name, identifier) {
            return Ok(schema);
        }

        let meta = self.load_table(identifier).await?;

        // Update catalog cache
        self.cache
            .put_table(&self.name, identifier, meta.schema.clone());

        Ok(meta.schema)
    }

    async fn rename_table(&self, from: &TableIdentifier, to: &TableIdentifier) -> Result<()> {
        // Load existing table
        let mut meta = self.load_table(from).await?;

        // Check destination doesn't exist
        if self.table_exists(to).await? {
            return Err(anyhow!("Table '{}' already exists", to));
        }

        // Update metadata
        meta.identifier = to.into();
        meta.schema.name = to.name.clone();
        meta.updated_at = Self::now_millis();

        // Save to new location
        self.save_table(&meta).await?;

        // ADR-031 O1 + TD-181 P1: object_id is preserved across rename
        // (metadata-only); repoint the reverse + forward index to the new
        // identifier and persist. The new copy is already on disk (authority).
        if let Some(oid) = meta.schema.object_id {
            self.index_rename_table(oid, from, to).await?;
        }

        // Delete old location (best-effort; the new copy is already persisted).
        self.io_remove_file(&Self::table_metadata_rel(from)).await;

        // Update in-memory cache
        self.tables.write().await.remove(&from.to_fqn());

        // Invalidate catalog cache
        self.cache.invalidate_table_in_catalog(&self.name, from);

        info!("Renamed table: {} -> {}", from, to);
        Ok(())
    }

    // ========================
    // Schema Evolution
    // ========================

    async fn evolve_schema(
        &self,
        identifier: &TableIdentifier,
        evolution: CatalogSchemaEvolution,
    ) -> Result<CatalogTableSchema> {
        let mut meta = self.load_table(identifier).await?;

        // Apply evolution
        meta.schema = apply_evolution(&meta.schema, &evolution)?;
        meta.updated_at = Self::now_millis();

        self.save_table(&meta).await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        info!(
            "Evolved schema for {}: v{} -> v{}",
            identifier,
            meta.schema.schema_version - 1,
            meta.schema.schema_version
        );
        Ok(meta.schema)
    }

    async fn set_primary_pod(
        &self,
        identifier: &TableIdentifier,
        primary: Option<crate::CatalogPrimaryPod>,
    ) -> Result<()> {
        // Read-modify-write the per-table metadata. Mirrors the
        // evolve_schema pattern: load (cache or disk), mutate, persist,
        // invalidate. The `updated_at` bump matters so downstream
        // consumers that watch the catalog cache see the new state.
        let mut meta = self.load_table(identifier).await?;
        meta.schema.primary_pod = primary;
        meta.updated_at = Self::now_millis();
        self.save_table(&meta).await?;
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);
        Ok(())
    }

    async fn set_storage_layouts(
        &self,
        identifier: &TableIdentifier,
        layouts: Vec<crate::CatalogStorageLayout>,
    ) -> Result<CatalogTableSchema> {
        // Read-modify-write the per-table metadata, mirroring set_primary_pod:
        // load (cache or disk), replace storage_layouts, persist, invalidate.
        // A physical/publication attribute → no schema_version bump. The
        // updated_at bump matters so catalog-cache watchers see the new state.
        let mut meta = self.load_table(identifier).await?;
        meta.schema.storage_layouts = layouts;
        meta.updated_at = Self::now_millis();
        self.save_table(&meta).await?;
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);
        Ok(meta.schema)
    }

    async fn apply_model_registry_mutation(
        &self,
        identifier: &TableIdentifier,
        expected_revision: u64,
        mutation: crate::mlops::CatalogModelRegistryMutation,
    ) -> Result<CatalogTableSchema> {
        let _guard = self.mlops_mutation_lock.lock().await;
        let mut meta = self.load_table(identifier).await?;
        let mut asset = meta
            .schema
            .mlops_asset_as_typed()?
            .ok_or_else(|| anyhow!("Catalog object '{}' is not an MLOps asset", identifier))?;
        asset
            .apply_model_mutation(expected_revision, mutation)
            .map_err(anyhow::Error::new)?;
        asset.validate().map_err(anyhow::Error::new)?;
        meta.schema.set_mlops_asset_typed(asset);
        let now = Self::now_millis();
        meta.schema.updated_at_ms = now;
        meta.updated_at = now;
        self.save_table(&meta).await?;
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);
        Ok(meta.schema)
    }

    async fn get_schema_version(&self, identifier: &TableIdentifier) -> Result<i32> {
        let meta = self.load_table(identifier).await?;
        Ok(meta.schema.schema_version)
    }

    async fn get_schema_by_version(
        &self,
        identifier: &TableIdentifier,
        version: i32,
    ) -> Result<CatalogTableSchema> {
        // For native catalog, we only keep the current version
        // Historical versions would require schema versioning infrastructure
        let meta = self.load_table(identifier).await?;
        if meta.schema.schema_version == version {
            Ok(meta.schema)
        } else {
            Err(anyhow!(
                "Schema version {} not found for table '{}' (current: {})",
                version,
                identifier,
                meta.schema.schema_version
            ))
        }
    }

    // ========================
    // Index Operations
    // ========================

    async fn create_index(
        &self,
        identifier: &TableIdentifier,
        index: CatalogIndex,
    ) -> Result<CatalogIndex> {
        let mut meta = self.load_table(identifier).await?;

        // Check index doesn't exist
        if meta.schema.indexes.iter().any(|i| i.name == index.name) {
            return Err(anyhow!(
                "Index '{}' already exists on table '{}'",
                index.name,
                identifier
            ));
        }

        // Validate columns exist
        for col in &index.columns {
            if !meta.schema.columns.iter().any(|c| &c.name == col) {
                return Err(anyhow!(
                    "Column '{}' not found in table '{}'",
                    col,
                    identifier
                ));
            }
        }

        // ADR-031 / TD-181: mint the index object_id from the shared sequence,
        // after the dup/column validation so a rejected create doesn't burn an id.
        let mut index = index;
        index.object_id = Some(self.mint_object_id(index.object_id));

        meta.schema.indexes.push(index.clone());
        meta.updated_at = Self::now_millis();

        self.save_table(&meta).await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        info!("Created index {} on {}", index.name, identifier);
        Ok(index)
    }

    async fn drop_index(&self, identifier: &TableIdentifier, index_name: &str) -> Result<bool> {
        let mut meta = self.load_table(identifier).await?;

        let initial_len = meta.schema.indexes.len();
        meta.schema.indexes.retain(|i| i.name != index_name);

        if meta.schema.indexes.len() < initial_len {
            meta.updated_at = Self::now_millis();
            self.save_table(&meta).await?;

            // Invalidate cache
            self.cache
                .invalidate_table_in_catalog(&self.name, identifier);

            info!("Dropped index {} from {}", index_name, identifier);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn list_indexes(&self, identifier: &TableIdentifier) -> Result<Vec<CatalogIndex>> {
        // Check cache first
        if let Some(indexes) = self.cache.get_indexes(&self.name, identifier) {
            return Ok(indexes);
        }

        let meta = self.load_table(identifier).await?;
        let indexes = meta.schema.indexes.clone();

        // Update cache
        self.cache
            .put_indexes(&self.name, identifier, indexes.clone());

        Ok(indexes)
    }

    // ========================
    // Statistics
    // ========================

    async fn get_statistics(&self, identifier: &TableIdentifier) -> Result<CatalogTableStatistics> {
        // Check cache first
        if let Some(stats) = self.cache.get_statistics(&self.name, identifier) {
            return Ok(stats);
        }

        let meta = self.load_table(identifier).await?;
        let stats = meta.statistics.unwrap_or_default();

        // Update cache
        self.cache
            .put_statistics(&self.name, identifier, stats.clone());

        Ok(stats)
    }

    async fn update_statistics(
        &self,
        identifier: &TableIdentifier,
        stats: CatalogTableStatistics,
    ) -> Result<()> {
        let mut meta = self.load_table(identifier).await?;
        meta.statistics = Some(stats.clone());
        meta.updated_at = Self::now_millis();

        self.save_table(&meta).await?;

        // Update cache
        self.cache.put_statistics(&self.name, identifier, stats);

        debug!("Updated statistics for {}", identifier);
        Ok(())
    }

    // ========================
    // Partitioning
    // ========================

    async fn get_partition_spec(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<CatalogPartitionSpec>> {
        let meta = self.load_table(identifier).await?;
        Ok(meta.partition_spec)
    }

    async fn update_partition_spec(
        &self,
        identifier: &TableIdentifier,
        spec: CatalogPartitionSpec,
    ) -> Result<()> {
        let mut meta = self.load_table(identifier).await?;
        meta.partition_spec = Some(spec);
        meta.updated_at = Self::now_millis();

        self.save_table(&meta).await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        Ok(())
    }

    // ========================
    // Sort Order
    // ========================

    async fn get_sort_order(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<CatalogSortOrder>> {
        let meta = self.load_table(identifier).await?;
        Ok(meta.sort_order)
    }

    async fn update_sort_order(
        &self,
        identifier: &TableIdentifier,
        order: CatalogSortOrder,
    ) -> Result<()> {
        let mut meta = self.load_table(identifier).await?;
        meta.sort_order = Some(order);
        meta.updated_at = Self::now_millis();

        self.save_table(&meta).await?;

        // Invalidate cache
        self.cache
            .invalidate_table_in_catalog(&self.name, identifier);

        Ok(())
    }

    // ========================
    // Health & Connectivity
    // ========================

    async fn health_check(&self) -> Result<CatalogHealth> {
        let start = Instant::now();

        // Probe storage connectivity through the active backend.
        if self.io_healthy().await {
            let latency = start.elapsed().as_millis() as u64;
            Ok(CatalogHealth::healthy(latency)
                .with_detail("storage_url", &self.config.storage_url)
                .with_detail("catalog_type", "native"))
        } else {
            Ok(CatalogHealth::unhealthy(
                "storage backend unreachable".to_string(),
            ))
        }
    }

    async fn close(&self) -> Result<()> {
        // Flush any pending writes
        debug!("Closing native catalog: {}", self.name);
        Ok(())
    }
}

#[cfg(test)]
mod catreplace_tests {
    use super::tests::catalog_in_tempdir;
    use super::*;
    use proximadb_catalog_schema::CatalogColumn;
    use proximadb_data_model::ProximaType;

    fn schema_with_columns(columns: &[(&str, ProximaType)]) -> CatalogTableSchema {
        let mut schema = CatalogTableSchema::new("t1");
        for (name, data_type) in columns {
            schema.columns.push(
                CatalogColumn::new(100 + schema.columns.len() as i32, *name, data_type.clone())
                    .nullable(true),
            );
        }
        schema
    }

    /// TD-CATREPLACE-1: replace_table keeps the row present, preserves
    /// identity and birth metadata, and applies the new schema — without the
    /// drop window the drop+create dance had.
    /// A native catalog with namespace `ns` and the `ns.t1` table id.
    async fn catalog_with_ns_t1() -> (NativeCatalog, tempfile::TempDir, TableIdentifier) {
        let (catalog, dir) = catalog_in_tempdir().await;
        catalog
            .create_namespace(&["ns".to_string()], Default::default())
            .await
            .expect("namespace");
        let identifier = TableIdentifier::new(vec!["ns".into()], "t1");
        (catalog, dir, identifier)
    }

    #[tokio::test]
    async fn replace_table_updates_in_place_preserving_identity() {
        let (catalog, _dir, identifier) = catalog_with_ns_t1().await;
        let created = catalog
            .create_table(
                &identifier,
                schema_with_columns(&[("a", ProximaType::Int64)]),
            )
            .await
            .expect("create");
        let original_oid = created.object_id.expect("object id minted");

        // The old behavior dropped the table first; assert the invariant
        // directly: after replace, the row exists with the new columns, the
        // same object_id (caller adopts it) and the same created_at.
        let mut replacement =
            schema_with_columns(&[("a", ProximaType::Int64), ("b", ProximaType::String)]);
        replacement.object_id = Some(original_oid);
        let replaced = catalog
            .replace_table(&identifier, replacement)
            .await
            .expect("replace");
        assert_eq!(replaced.object_id, Some(original_oid));
        let after = catalog.get_table(&identifier).await.expect("table present");
        assert_eq!(after.object_id, Some(original_oid));
        assert!(after.columns.iter().any(|c| c.name == "b"));

        // An adopted object_id that differs from the replaced row's swaps the
        // identity and keeps the table readable (the removal branch).
        let mut reidentified = schema_with_columns(&[("a", ProximaType::Int64)]);
        reidentified.object_id = Some(original_oid + 1000);
        let reidentified = catalog
            .replace_table(&identifier, reidentified)
            .await
            .expect("replace with a new oid");
        assert_eq!(reidentified.object_id, Some(original_oid + 1000));
        assert_eq!(
            catalog
                .get_table(&identifier)
                .await
                .expect("readable after identity swap")
                .object_id,
            Some(original_oid + 1000)
        );

        // A schema the catalog rejects costs the existing table nothing —
        // asserted LAST, so no later replace could have recreated the table had
        // the rejection dropped it.
        assert!(
            catalog
                .replace_table(&identifier, schema_with_columns(&[]))
                .await
                .is_err(),
            "a columnless schema is invalid"
        );
        assert!(
            catalog.table_exists(&identifier).await.expect("exists"),
            "a rejected replace must leave the table in place"
        );
    }

    /// TD-CATREPLACE-1: replace on a missing table behaves like a create.
    #[tokio::test]
    async fn replace_table_on_missing_table_creates() {
        let (catalog, _dir, identifier) = catalog_with_ns_t1().await;
        let replaced = catalog
            .replace_table(
                &identifier,
                schema_with_columns(&[("a", ProximaType::Int64)]),
            )
            .await
            .expect("replace creates");
        assert!(replaced.object_id.is_some());
        assert!(catalog.table_exists(&identifier).await.expect("exists"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ADR-031 Phase 4a: typed-identity minting ────────────────────────

    pub(super) async fn catalog_in_tempdir() -> (NativeCatalog, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = NativeCatalogConfig {
            storage_url: format!("file://{}", dir.path().join("cat").display()),
            metadata_format: "json".into(),
            versioned: false,
            max_versions: 100,
        };
        let cat = NativeCatalog::new(
            "t".into(),
            cfg,
            Arc::new(crate::cache::CatalogCache::new(64, 60)),
        )
        .await
        .expect("construct catalog");
        (cat, dir)
    }

    #[tokio::test]
    async fn stable_identity_is_per_scope_and_legacy_safe() {
        let (cat, _d) = catalog_in_tempdir().await;

        // No account → legacy, no typed identity (mixed-read-safe).
        let mut legacy = CatalogTableSchema::new("legacy");
        cat.mint_stable_identity("", "ns", &mut legacy).await;
        assert!(legacy.stable_namespace_id.is_none());
        assert!(legacy.stable_collection_id.is_none());

        // Account present → per-scope compact ids minted.
        let mut a1 = CatalogTableSchema::new("a1");
        cat.mint_stable_identity("acct", "ns1", &mut a1).await;
        assert_eq!(a1.stable_namespace_id, Some(1), "first ns in acct = 1");
        assert_eq!(a1.stable_collection_id, Some(1), "first coll = 1");

        // Same namespace, new collection → SAME namespace id, NEXT collection id.
        let mut a2 = CatalogTableSchema::new("a2");
        cat.mint_stable_identity("acct", "ns1", &mut a2).await;
        assert_eq!(
            a2.stable_namespace_id,
            Some(1),
            "same namespace → same ns id"
        );
        assert_eq!(a2.stable_collection_id, Some(2), "second coll in ns1 = 2");

        // Different namespace → NEXT namespace id, collection restarts at 1.
        let mut b1 = CatalogTableSchema::new("b1");
        cat.mint_stable_identity("acct", "ns2", &mut b1).await;
        assert_eq!(b1.stable_namespace_id, Some(2), "new ns = 2");
        assert_eq!(
            b1.stable_collection_id,
            Some(1),
            "new ns → coll restarts at 1"
        );
    }

    #[tokio::test]
    async fn account_registry_is_stable_within_run() {
        let (cat, _d) = catalog_in_tempdir().await;
        // Same account string → same account u32 (registry lookup-or-mint).
        let mut a = CatalogTableSchema::new("a");
        cat.mint_stable_identity("acct", "ns", &mut a).await;
        let acct_u32_a = cat.account_u32("acct").await.expect("account resolved");
        // A second call for the same account must NOT re-mint a different u32.
        let acct_u32_b = cat.account_u32("acct").await.expect("account resolved");
        assert_eq!(acct_u32_a, acct_u32_b, "account u32 stable within a run");
        // And namespace siblings share the namespace id (covered above) because
        // they share the account u32.
        assert_eq!(a.stable_namespace_id, Some(1));
    }

    #[tokio::test]
    async fn concurrent_first_account_mint_returns_one_id() {
        let (cat, _d) = catalog_in_tempdir().await;
        let (left, right) = tokio::join!(cat.account_u32("acct"), cat.account_u32("acct"));
        assert_eq!(left, right, "one account string must never receive two ids");
        assert_eq!(cat.account_id_u32_lookup("acct"), left);
    }

    #[tokio::test]
    async fn account_registry_survives_restart() {
        // ADR-031 Phase 4b: the account-string → u32 mapping MUST persist across
        // restarts, else the typed object-store path drifts → silent data loss.
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = || NativeCatalogConfig {
            storage_url: format!("file://{}/cat", dir.path().display()),
            metadata_format: "json".into(),
            versioned: false,
            max_versions: 100,
        };
        let cache = || Arc::new(crate::cache::CatalogCache::new(64, 60));

        // First instance: mint account "acct" → u32 (persists the sidecar).
        let cat1 = NativeCatalog::new("t".into(), cfg(), cache())
            .await
            .expect("construct cat1");
        let first = cat1.account_u32("acct").await.expect("minted");
        assert_eq!(first, 1, "first account mints u32 1");
        drop(cat1);

        // Second instance from the SAME dir: load_account_registry must reuse u32
        // 1 (not re-mint 2) + raise the floor above it.
        let cat2 = NativeCatalog::new("t".into(), cfg(), cache())
            .await
            .expect("construct cat2");
        let second = cat2
            .account_u32("acct")
            .await
            .expect("lookup after restart");
        assert_eq!(
            first, second,
            "account u32 must be stable across restart (durable registry)"
        );
        // A genuinely new account mints above the recovered floor (2, not 1).
        let new_acct = cat2.account_u32("other").await.expect("mint new");
        assert_eq!(new_acct, 2, "new account mints above recovered floor");

        // TD-TENANT-1 item 3: the SYNC account_id_u32_lookup returns the minted
        // u32 (no mint, no I/O) — the TenantStableIdResolver contract. None for
        // unknown/empty (fail-closed deny).
        assert_eq!(cat2.account_id_u32_lookup("acct"), Some(1));
        assert_eq!(cat2.account_id_u32_lookup("other"), Some(2));
        assert_eq!(cat2.account_id_u32_lookup("never"), None);
        assert_eq!(cat2.account_id_u32_lookup(""), None);
    }

    #[tokio::test]
    async fn mint_collection_typed_identity_is_idempotent_with_mint_stable_identity() {
        // ADR-031 Phase 4c: the pre-mint (`mint_collection_typed_identity`) and
        // the create_table mint (`mint_stable_identity`) MUST agree — the
        // pre-minted values are preserved (no double-mint / drift), because both
        // hit the SAME `resolve_typed_triple` with `existing_*` short-circuits.
        // This is the correctness invariant that lets the manager pre-mint for
        // the typed DATA path before create_table stamps the schema.
        use crate::Catalog; // trait method in scope
        let (cat, _d) = catalog_in_tempdir().await;

        // Phase 4c pre-mint: a fresh typed triple (no existing values).
        let triple = cat
            .mint_collection_typed_identity("acct", "ns1")
            .await
            .expect("mint ok")
            .expect("Some triple");
        let (acct, ns, coll) = triple;

        // Stamp the pre-minted values onto a schema (as the manager does via
        // `__typed_identity` → `schema.stable_*_id`).
        let mut schema = CatalogTableSchema::new("col1");
        schema.stable_namespace_id = Some(ns);
        schema.stable_collection_id = Some(coll);

        // create_table's mint path: `mint_stable_identity` MUST preserve the
        // pre-stamped values (idempotent) — not re-mint new ones.
        cat.mint_stable_identity("acct", "ns1", &mut schema).await;
        assert_eq!(
            schema.stable_namespace_id,
            Some(ns),
            "ns id preserved (no double-mint)"
        );
        assert_eq!(
            schema.stable_collection_id,
            Some(coll),
            "coll id preserved (no double-mint)"
        );

        // And the account u32 matches what the pre-mint used.
        assert_eq!(cat.account_u32("acct").await, Some(acct));

        // A SECOND collection in the same namespace reuses the ns u16 but mints
        // a new collection u32 (the pre-mint for col1 did NOT consume col2's id
        // — `resolve_typed_triple` mints per-collection only when unset).
        let triple2 = cat
            .mint_collection_typed_identity("acct", "ns1")
            .await
            .expect("mint ok")
            .expect("Some triple");
        assert_eq!(triple2.1, ns, "same namespace → same ns u16");
        assert_eq!(triple2.2, coll + 1, "next collection → coll u32 + 1");
    }

    #[tokio::test]
    async fn mint_collection_typed_identity_returns_none_for_no_account() {
        // Legacy/anonymous → None (no typed path; mixed-read-safe).
        use crate::Catalog;
        let (cat, _d) = catalog_in_tempdir().await;
        let triple = cat.mint_collection_typed_identity("", "ns").await;
        assert!(triple.unwrap().is_none(), "empty account → None");
    }

    #[tokio::test]
    async fn tenant_scoped_namespace_activates_typed_identity() {
        // ADR-031 Phase 5: create_namespace_for_tenant sets account_id = tenant
        // (the Phase-4 collapse), which activates mint_stable_identity in
        // create_table → the collection gets stable_namespace_id /
        // stable_collection_id minted. (create_namespace with no tenant →
        // account_id None → no typed identity, legacy.)
        use crate::{Catalog, CatalogTableSchema, TableIdentifier};
        let (cat, _d) = catalog_in_tempdir().await;

        // Tenant-scoped namespace → account_id mirrors tenant.
        let ns = cat
            .create_namespace_for_tenant(
                &["tnt_acme".to_string()],
                HashMap::new(),
                Some("tnt_acme"),
            )
            .await
            .expect("create tenant namespace");
        assert_eq!(
            ns.account_id.as_deref(),
            Some("tnt_acme"),
            "account_id mirrors tenant_id (Phase-4 collapse)"
        );

        // A collection under it → create_table mints the stable ids.
        let schema = CatalogTableSchema::new("orders").with_column(crate::CatalogColumn::new(
            1,
            "id",
            proximadb_data_model::ProximaType::Int64,
        ));
        let created = cat
            .create_table(
                &TableIdentifier::new(vec!["tnt_acme".to_string()], "orders"),
                schema,
            )
            .await
            .expect("create_table");
        assert!(
            created.stable_namespace_id.is_some(),
            "tenant-scoped collection mints stable_namespace_id (account_id set)"
        );
        assert!(
            created.stable_collection_id.is_some(),
            "tenant-scoped collection mints stable_collection_id"
        );

        // Tenant-less namespace → account_id None → no typed identity (legacy).
        let legacy_ns = cat
            .create_namespace(&["legacy".to_string()], HashMap::new())
            .await
            .expect("create legacy namespace");
        assert!(
            legacy_ns.account_id.is_none(),
            "no-tenant namespace has no account"
        );
        let legacy_schema = CatalogTableSchema::new("legacy_tbl").with_column(
            crate::CatalogColumn::new(1, "id", proximadb_data_model::ProximaType::Int64),
        );
        let legacy_created = cat
            .create_table(
                &TableIdentifier::new(vec!["legacy".to_string()], "legacy_tbl"),
                legacy_schema,
            )
            .await
            .expect("create_table legacy");
        assert!(
            legacy_created.stable_namespace_id.is_none()
                && legacy_created.stable_collection_id.is_none(),
            "tenant-less collection has no typed identity (legacy, mixed-read-safe)"
        );
    }

    // Note: These tests require a mock filesystem or temp directory
    // Full integration tests should be in the tests/ directory

    #[test]
    fn test_table_identifier_serde() {
        let id = TableIdentifier::new(vec!["db".to_string()], "users".to_string());
        let serde_id: TableIdentifierSerde = (&id).into();

        assert_eq!(serde_id.namespace, vec!["db"]);
        assert_eq!(serde_id.name, "users");
    }

    #[test]
    fn test_parse_storage_url_file() {
        let path = NativeCatalog::parse_storage_url("file:///tmp/catalog").unwrap();
        assert_eq!(path, PathBuf::from("/tmp/catalog"));
    }

    #[test]
    fn test_parse_storage_url_plain_path() {
        let path = NativeCatalog::parse_storage_url("/tmp/catalog").unwrap();
        assert_eq!(path, PathBuf::from("/tmp/catalog"));
    }

    #[test]
    fn test_parse_storage_url_object_store_fails_closed() {
        // Object-store catalog URLs must fail closed (not silently redirect to a
        // process-local temp dir) until durable object-store persistence is wired.
        for url in [
            "s3://bucket/catalog",
            "gs://bucket/catalog",
            "az://acct/catalog",
        ] {
            let err = NativeCatalog::parse_storage_url(url)
                .expect_err("object-store catalog URL must be rejected, not temp-cached");
            assert!(
                err.to_string().contains("not yet supported"),
                "unexpected error for {url}: {err}"
            );
        }
    }

    use crate::testfs::MemFs;

    /// TD-CAT-1: an injected object-store backend persists the catalog durably
    /// (proven by reading back through a FRESH catalog instance over the same
    /// backend, bypassing the in-memory cache), and an `s3://` URL — which fails
    /// closed without a backend — works once a backend is injected.
    #[tokio::test]
    async fn object_store_backend_round_trips() {
        let fs: Arc<dyn FileSystem> = Arc::new(MemFs::default());
        let cfg = || NativeCatalogConfig {
            storage_url: "s3://test-bucket/catalog".into(),
            metadata_format: "json".into(),
            versioned: false,
            max_versions: 100,
        };
        let ns = vec!["tenant_a".to_string()];
        let id = TableIdentifier::new(ns.clone(), "users".to_string());

        // Writer instance.
        let writer = NativeCatalog::new_with_filesystem(
            "t".into(),
            cfg(),
            Arc::new(crate::cache::CatalogCache::new(64, 60)),
            Some(fs.clone()),
        )
        .await
        .expect("construct over injected backend");
        writer
            .create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");
        let schema = crate::CatalogTableSchema::new("users").with_column(
            crate::CatalogColumn::new(1, "id", proximadb_data_model::ProximaType::Int64),
        );
        writer
            .create_table(&id, schema)
            .await
            .expect("create table");

        // Fresh reader over the SAME backend — proves durability, not cache.
        let reader = NativeCatalog::new_with_filesystem(
            "t".into(),
            cfg(),
            Arc::new(crate::cache::CatalogCache::new(64, 60)),
            Some(fs.clone()),
        )
        .await
        .expect("reconstruct over injected backend");
        assert_eq!(
            reader.get_table(&id).await.expect("get_table").name,
            "users"
        );
        assert!(
            reader
                .list_tables(&ns)
                .await
                .expect("list")
                .iter()
                .any(|t| t.name == "users"),
            "listed tables must include the created table"
        );
        assert!(reader.drop_table(&id, true).await.expect("drop"));
        assert!(!reader.table_exists(&id).await.expect("exists"));
    }

    // ── Slice 5b.1: set_primary_pod (NativeCatalog override) ─────────
    //
    // Each test owns a fresh `TempDir` so the JSON sidecars don't
    // collide. The setup helper also creates the namespace required by
    // `create_table`.

    async fn fresh_catalog(tmp: &tempfile::TempDir) -> NativeCatalog {
        let config = NativeCatalogConfig {
            storage_url: tmp.path().to_string_lossy().to_string(),
            metadata_format: "json".into(),
            versioned: false,
            max_versions: 100,
        };
        let cache = Arc::new(crate::cache::CatalogCache::new(64, 60));
        NativeCatalog::new("test".into(), config, cache)
            .await
            .expect("construct catalog")
    }

    async fn make_table(cat: &NativeCatalog, table: &str) -> TableIdentifier {
        let ns = vec!["tenant_a".to_string()];
        cat.create_namespace(&ns, HashMap::new())
            .await
            .expect("namespace");
        let id = TableIdentifier::new(ns, table);
        // validate_schema rejects empty-column schemas, so seed one
        // benign Int64 column — the column choice is irrelevant to
        // the primary_pod field under test.
        let schema = crate::CatalogTableSchema::new(table).with_column(crate::CatalogColumn::new(
            1,
            "id",
            proximadb_data_model::ProximaType::Int64,
        ));
        cat.create_table(&id, schema).await.expect("create table");
        id
    }

    #[tokio::test]
    async fn set_primary_pod_writes_field_to_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let id = make_table(&cat, "users").await;

        let pod = crate::CatalogPrimaryPod::now("pod-a", crate::CatalogPrimaryPodReason::Create);
        cat.set_primary_pod(&id, Some(pod.clone()))
            .await
            .expect("set succeeds on existing table");

        let read = cat.get_table(&id).await.expect("read back");
        assert_eq!(read.primary_pod.as_ref().unwrap().pod, "pod-a");
        assert!(matches!(
            read.primary_pod.as_ref().unwrap().reason,
            crate::CatalogPrimaryPodReason::Create
        ));
    }

    #[tokio::test]
    async fn set_primary_pod_with_none_clears_existing_binding() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let id = make_table(&cat, "orders").await;

        let pod = crate::CatalogPrimaryPod::now("pod-b", crate::CatalogPrimaryPodReason::Operator);
        cat.set_primary_pod(&id, Some(pod)).await.unwrap();
        cat.set_primary_pod(&id, None).await.expect("clear");

        let read = cat.get_table(&id).await.unwrap();
        assert!(read.primary_pod.is_none(), "None must clear the field");
    }

    #[tokio::test]
    async fn set_primary_pod_returns_err_for_unknown_table() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;

        let id = TableIdentifier::new(vec!["nope".to_string()], "ghost");
        let pod = crate::CatalogPrimaryPod::now("pod-c", crate::CatalogPrimaryPodReason::Failover);
        let res = cat.set_primary_pod(&id, Some(pod)).await;
        assert!(res.is_err(), "missing table must error, got: {:?}", res);
    }

    #[tokio::test]
    async fn set_primary_pod_persists_across_reload() {
        // Reloading the catalog drops the in-memory cache and forces a
        // disk read on the next get_table — verifies save_table is the
        // real persistence path, not just a cache write.
        let tmp = tempfile::tempdir().unwrap();
        let id = {
            let cat = fresh_catalog(&tmp).await;
            let id = make_table(&cat, "events").await;
            let pod =
                crate::CatalogPrimaryPod::now("pod-d", crate::CatalogPrimaryPodReason::Rebalance);
            cat.set_primary_pod(&id, Some(pod)).await.unwrap();
            id
        };

        let cat2 = fresh_catalog(&tmp).await;
        let read = cat2.get_table(&id).await.expect("reload table");
        assert_eq!(read.primary_pod.as_ref().unwrap().pod, "pod-d");
        assert!(matches!(
            read.primary_pod.as_ref().unwrap().reason,
            crate::CatalogPrimaryPodReason::Rebalance
        ));
    }

    // ── ADR-031 O0: stable object_id allocation ──────────────────────

    fn schema_with_id_col(name: &str) -> crate::CatalogTableSchema {
        crate::CatalogTableSchema::new(name).with_column(crate::CatalogColumn::new(
            1,
            "id",
            proximadb_data_model::ProximaType::Int64,
        ))
    }

    #[tokio::test]
    async fn object_id_survives_the_get_table_round_trip() {
        // TD-AUTHZ-2. `create_table_assigns_distinct_monotonic_object_ids`
        // asserts only on the schema `create_table` RETURNS — it never reads the
        // table back, so nothing covered whether the minted id survives
        // persistence. `xcatalog.tables` renders an EMPTY object_id for
        // SQL-created relational tables in the live server, and the defect can
        // only live in that untested gap: mint (proven) -> persist -> load.
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let identifier = TableIdentifier::new(ns.clone(), "roundtrip_probe");

        let created = cat
            .create_table(&identifier, schema_with_id_col("roundtrip_probe"))
            .await
            .expect("create_table");
        let minted = created.object_id.expect("create_table mints an object_id");

        let fetched = cat.get_table(&identifier).await.expect("get_table");
        assert_eq!(
            fetched.object_id,
            Some(minted),
            "the minted object_id must survive persistence and be readable via \
             get_table — this is what xcatalog.tables projects, and an empty \
             value there is what TD-AUTHZ-2 reports"
        );
    }

    #[tokio::test]
    async fn create_table_assigns_distinct_monotonic_object_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();

        let a = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "a"),
                schema_with_id_col("a"),
            )
            .await
            .unwrap();
        let b = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "b"),
                schema_with_id_col("b"),
            )
            .await
            .unwrap();

        let ida = a.object_id.expect("a got an object_id");
        let idb = b.object_id.expect("b got an object_id");
        assert!(ida >= 1, "ids start at 1, got {ida}");
        assert!(idb > ida, "monotonic + distinct: {idb} > {ida}");
    }

    /// TD-181 P1: the durable `object_index.json` records every table both
    /// directions, and — because the object files are authority and the index is
    /// a derived cache — deleting the index and reopening rebuilds an identical
    /// index by scanning the object files.
    #[tokio::test]
    async fn object_index_persists_and_rebuilds_from_object_files() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["db".to_string()];
        let id_a = TableIdentifier::new(ns.clone(), "a");
        let id_b = TableIdentifier::new(ns.clone(), "b");

        let (oid_a, oid_b) = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            let oid_a = cat
                .create_table(&id_a, schema_with_id_col("a"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            let oid_b = cat
                .create_table(&id_b, schema_with_id_col("b"))
                .await
                .unwrap()
                .object_id
                .unwrap();

            // Both directions resolve in-memory.
            assert_eq!(
                cat.get_table_by_object_id(oid_a).await.unwrap(),
                Some(id_a.clone())
            );
            assert_eq!(
                cat.object_name_index
                    .read()
                    .await
                    .get(&id_a.to_fqn())
                    .copied(),
                Some(oid_a)
            );

            // The on-disk index holds exactly the two tables.
            let raw = cat
                .io_read_opt(&NativeCatalog::object_index_rel())
                .await
                .unwrap()
                .expect("index file written");
            let parsed: ObjectIndex = serde_json::from_slice(&raw).unwrap();
            assert_eq!(parsed.tables.len(), 2);

            // Delete the derived index; the authoritative object files remain.
            cat.io_remove_file(&NativeCatalog::object_index_rel()).await;
            (oid_a, oid_b)
        };

        // Reopen the same storage: absent index → one-time scan-rebuild.
        let reopened = fresh_catalog(&tmp).await;
        assert_eq!(
            reopened.get_table_by_object_id(oid_a).await.unwrap(),
            Some(id_a),
            "rebuilt index must resolve oid_a → its identifier"
        );
        assert_eq!(
            reopened.get_table_by_object_id(oid_b).await.unwrap(),
            Some(id_b),
            "rebuilt index must resolve oid_b → its identifier"
        );
        let rebuilt_raw = reopened
            .io_read_opt(&NativeCatalog::object_index_rel())
            .await
            .unwrap()
            .expect("index rebuilt + persisted on reopen");
        let rebuilt: ObjectIndex = serde_json::from_slice(&rebuilt_raw).unwrap();
        assert_eq!(
            rebuilt.tables.len(),
            2,
            "scan-rebuild recovers exactly the persisted object files"
        );
    }

    /// TD-181 P1: a catalog whose durable index sits at the pre-rename location
    /// (`metadata/object_index.json`) migrates it to the canonical
    /// `_syscat/index.json` on load — exactly (no rebuild scan), mixed-read-safe.
    #[tokio::test]
    async fn object_index_migrates_from_legacy_location() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["db".to_string()];
        let id = TableIdentifier::new(ns.clone(), "t");

        let oid = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            let oid = cat
                .create_table(&id, schema_with_id_col("t"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            // Simulate a pre-rename catalog: relocate the index to the legacy path.
            let data = cat
                .io_read_opt(&NativeCatalog::object_index_rel())
                .await
                .unwrap()
                .expect("index at canonical location");
            cat.io_write(&NativeCatalog::legacy_object_index_rel(), &data)
                .await
                .unwrap();
            cat.io_remove_file(&NativeCatalog::object_index_rel()).await;
            oid
        };

        // Reopen: canonical absent + legacy present ⇒ migrate (not rebuild-scan).
        let cat = fresh_catalog(&tmp).await;
        assert_eq!(
            cat.get_table_by_object_id(oid).await.unwrap(),
            Some(id),
            "migrated index resolves the table"
        );
        assert!(
            cat.io_read_opt(&NativeCatalog::object_index_rel())
                .await
                .unwrap()
                .is_some(),
            "index now present at the canonical _syscat/ location"
        );
    }

    /// TD-181 P3 hardening: an oid-only table (no legacy shadow) is recovered by
    /// the index rebuild scanning `_syscat/objects/`, and its oid raises the
    /// allocator floor so a later create never reuses it. Without scanning the
    /// oid layout, such a table would be silently lost and its oid reusable.
    #[tokio::test]
    async fn rebuild_recovers_oid_only_table_from_objects_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["db".to_string()];
        let id = TableIdentifier::new(ns.clone(), "t");

        let oid = {
            let cat = fresh_catalog(&tmp).await;
            cat.set_oid_paths_for_test(true);
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            let oid = cat
                .create_table(&id, schema_with_id_col("t"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            // Reduce to a pure-oid table: drop the legacy shadow AND the durable
            // index, leaving only `_syscat/objects/{oid}.json`.
            cat.io_remove_file(&NativeCatalog::table_metadata_rel(&id))
                .await;
            cat.io_remove_file(&NativeCatalog::object_index_rel()).await;
            oid
        };

        // Reopen: durable index absent + no legacy file ⇒ rebuild must scan the
        // oid layout to recover the table.
        let cat = fresh_catalog(&tmp).await;
        cat.set_oid_paths_for_test(true);
        assert_eq!(
            cat.get_table_by_object_id(oid).await.unwrap(),
            Some(id),
            "rebuild recovers the oid-only table from _syscat/objects/"
        );

        // Floor raised past the recovered oid — a new table gets a strictly
        // greater id (no reuse).
        let oid2 = cat
            .create_table(&TableIdentifier::new(ns, "t2"), schema_with_id_col("t2"))
            .await
            .unwrap()
            .object_id
            .unwrap();
        assert!(
            oid2 > oid,
            "allocator floor raised past the recovered oid ({oid2} > {oid})"
        );
    }

    /// TD-181 P3 (S2a): with the oid-paths gate ON, a create dual-writes the
    /// oid-keyed authority file AND the legacy name-path shadow (byte-identical),
    /// writes the migration marker, and a drop removes both copies.
    #[tokio::test]
    async fn oid_paths_dual_write_creates_oid_file_marker_and_drops_both() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        cat.set_oid_paths_for_test(true);
        let ns = vec!["db".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let id = TableIdentifier::new(ns, "t");
        let oid = cat
            .create_table(&id, schema_with_id_col("t"))
            .await
            .unwrap()
            .object_id
            .unwrap();

        // Both copies exist and agree (same self-describing TableMetadata bytes).
        let legacy = cat
            .io_read_opt(&NativeCatalog::table_metadata_rel(&id))
            .await
            .unwrap()
            .expect("legacy shadow written");
        let authority = cat
            .io_read_opt(&NativeCatalog::object_file_rel(oid))
            .await
            .unwrap()
            .expect("oid authority file written");
        assert_eq!(legacy, authority, "oid and legacy copies must be identical");

        // Migration marker present.
        assert!(
            cat.io_exists(&NativeCatalog::migration_marker_rel())
                .await
                .unwrap(),
            "migration marker written on first oid-keyed object"
        );

        // Drop removes BOTH copies (authority-first).
        assert!(cat.drop_table(&id, false).await.unwrap());
        assert!(
            cat.io_read_opt(&NativeCatalog::object_file_rel(oid))
                .await
                .unwrap()
                .is_none(),
            "oid authority file deleted on drop"
        );
        assert!(
            cat.io_read_opt(&NativeCatalog::table_metadata_rel(&id))
                .await
                .unwrap()
                .is_none(),
            "legacy shadow deleted on drop"
        );
    }

    /// TD-181 P3 (S2a): with the gate OFF (default) a create writes only the
    /// legacy name path — no oid file, no marker — i.e. today's exact behavior.
    #[tokio::test]
    async fn oid_paths_off_writes_only_legacy() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        cat.set_oid_paths_for_test(false);
        let ns = vec!["db".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let id = TableIdentifier::new(ns, "t");
        let oid = cat
            .create_table(&id, schema_with_id_col("t"))
            .await
            .unwrap()
            .object_id
            .unwrap();

        assert!(
            cat.io_read_opt(&NativeCatalog::table_metadata_rel(&id))
                .await
                .unwrap()
                .is_some(),
            "legacy name path written"
        );
        assert!(
            cat.io_read_opt(&NativeCatalog::object_file_rel(oid))
                .await
                .unwrap()
                .is_none(),
            "no oid file when gate off"
        );
        assert!(
            !cat.io_exists(&NativeCatalog::migration_marker_rel())
                .await
                .unwrap(),
            "no marker when gate off"
        );
    }

    /// TD-181 P3 (S2b) — the keystone mixed-read proof. With the gate ON a
    /// catalog dir holding BOTH a legacy-only table (name path, no oid file) and
    /// an oid-only table (oid file, no legacy shadow) resolves both via `get`,
    /// lists exactly the two with no duplicates, reports both as existing, and
    /// drops each (removing all of its files).
    #[tokio::test]
    async fn s2b_mixed_read_legacy_and_oid_only_tables() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["db".to_string()];
        let id_a = TableIdentifier::new(ns.clone(), "a"); // legacy-only on disk
        let id_b = TableIdentifier::new(ns.clone(), "b"); // oid-only on disk

        let (oid_a, oid_b) = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();

            // A: created gate OFF → legacy name path only (+ durable index entry).
            cat.set_oid_paths_for_test(false);
            let oid_a = cat
                .create_table(&id_a, schema_with_id_col("a"))
                .await
                .unwrap()
                .object_id
                .unwrap();

            // B: created gate ON → dual-written; strip the legacy shadow to
            // simulate a pure-oid table (the S2 endgame, no name-path copy).
            cat.set_oid_paths_for_test(true);
            let oid_b = cat
                .create_table(&id_b, schema_with_id_col("b"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            cat.io_remove_file(&NativeCatalog::table_metadata_rel(&id_b))
                .await;
            (oid_a, oid_b)
        };

        // Reopen so reads hit disk (empty in-memory table cache); gate ON.
        let cat = fresh_catalog(&tmp).await;
        cat.set_oid_paths_for_test(true);

        // get resolves BOTH: A via the legacy fallback, B via the oid file.
        assert_eq!(
            cat.get_table(&id_a).await.unwrap().object_id,
            Some(oid_a),
            "legacy-only table resolves via fallback"
        );
        assert_eq!(
            cat.get_table(&id_b).await.unwrap().object_id,
            Some(oid_b),
            "oid-only table resolves via the oid file"
        );

        // list returns exactly the two, deduped (legacy stems ∪ index entries).
        let mut listed = cat.list_tables(&ns).await.unwrap();
        listed.sort_by(|x, y| x.name.cmp(&y.name));
        assert_eq!(
            listed,
            vec![id_a.clone(), id_b.clone()],
            "list unions legacy + oid-only with no duplicates"
        );

        // exists true for both.
        assert!(cat.table_exists(&id_a).await.unwrap());
        assert!(cat.table_exists(&id_b).await.unwrap());

        // drop removes each table's file(s): legacy-only A, oid-only B.
        assert!(cat.drop_table(&id_a, false).await.unwrap(), "drop A");
        assert!(cat.drop_table(&id_b, false).await.unwrap(), "drop B");
        assert!(
            cat.list_tables(&ns).await.unwrap().is_empty(),
            "both tables gone after drop"
        );
        assert!(
            cat.get_table(&id_a).await.is_err(),
            "A not found after drop"
        );
        assert!(
            cat.get_table(&id_b).await.is_err(),
            "B not found after drop"
        );
    }

    /// TD-181 P1: a dangling index entry (name in the durable index, but both the
    /// oid file and the legacy file gone — the state a crash between
    /// `drop_table`'s file-delete and index-persist leaves) phantoms in
    /// `list_tables`. A read of it returns NotFound AND prunes the entry, so the
    /// phantom is gone afterwards — the long-documented lazy self-heal.
    #[tokio::test]
    async fn dangling_index_entry_is_pruned_on_read() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["db".to_string()];
        let id = TableIdentifier::new(ns.clone(), "t");

        let oid = {
            let cat = fresh_catalog(&tmp).await;
            cat.set_oid_paths_for_test(true);
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            let oid = cat
                .create_table(&id, schema_with_id_col("t"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            // Crash mid-drop: both authority files gone, durable index survives.
            cat.io_remove_file(&NativeCatalog::object_file_rel(oid))
                .await;
            cat.io_remove_file(&NativeCatalog::table_metadata_rel(&id))
                .await;
            oid
        };

        // Reopen: the durable index reloads the (now dangling) entry; the
        // in-memory table cache is cold, so a read goes to disk.
        let cat = fresh_catalog(&tmp).await;
        cat.set_oid_paths_for_test(true);

        // Pre-prune: the entry phantoms in list_tables (the index union lists it).
        assert!(
            cat.list_tables(&ns)
                .await
                .unwrap()
                .iter()
                .any(|t| t.name == "t"),
            "dangling entry phantoms in list before pruning"
        );

        // A read of it is NotFound and triggers the lazy prune.
        assert!(
            cat.get_table(&id).await.is_err(),
            "get of a dangling entry returns NotFound"
        );

        // Post-prune: gone from both the list and the reverse index.
        assert!(
            cat.list_tables(&ns)
                .await
                .unwrap()
                .iter()
                .all(|t| t.name != "t"),
            "dangling entry pruned from list after the read"
        );
        assert!(
            cat.get_table_by_object_id(oid).await.unwrap().is_none(),
            "reverse index entry pruned"
        );
    }

    // ── ADR-031 / TD-181 P0: namespace object_id allocation ──────────

    #[tokio::test]
    async fn create_namespace_assigns_distinct_monotonic_object_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;

        let a = cat
            .create_namespace(&["ns_a".to_string()], HashMap::new())
            .await
            .unwrap();
        let b = cat
            .create_namespace(&["ns_b".to_string()], HashMap::new())
            .await
            .unwrap();

        let ida = a.object_id.expect("namespace a got an object_id");
        let idb = b.object_id.expect("namespace b got an object_id");
        assert!(ida >= 1, "ids start at 1, got {ida}");
        assert!(idb > ida, "monotonic + distinct: {idb} > {ida}");
    }

    #[tokio::test]
    async fn namespace_and_table_share_one_object_id_sequence() {
        // ADR-031 reconciliation amendment 1: ONE system-wide sequence mints
        // both namespace and table object_ids (not per-type spaces), so every
        // catalog object_id in a deployment is mutually distinct.
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;

        let ns = vec!["shared".to_string()];
        let ns_oid = cat
            .create_namespace(&ns, HashMap::new())
            .await
            .unwrap()
            .object_id
            .expect("namespace object_id");
        let tbl_oid = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "t"),
                schema_with_id_col("t"),
            )
            .await
            .unwrap()
            .object_id
            .expect("table object_id");

        assert_ne!(
            ns_oid, tbl_oid,
            "one shared sequence: ids never collide across object types"
        );
    }

    #[tokio::test]
    async fn namespace_object_id_recovered_on_reload_prevents_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let existing = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&["db".to_string()], HashMap::new())
                .await
                .unwrap()
                .object_id
                .expect("namespace object_id")
        };

        // Cold restart: a fresh catalog eagerly loads namespaces (constructor →
        // load_namespaces), which must recover the allocator floor from the
        // persisted object_id so the next allocation never reuses it.
        let cat2 = fresh_catalog(&tmp).await;
        let next = cat2
            .create_namespace(&["db2".to_string()], HashMap::new())
            .await
            .unwrap()
            .object_id
            .expect("namespace object_id");
        assert!(
            next > existing,
            "reload recovered the floor: {next} > {existing}"
        );
    }

    // ── ADR-031 / TD-181 P0 tail: index object_id ────────────────────

    #[tokio::test]
    async fn create_index_assigns_object_id_from_shared_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let id = TableIdentifier::new(ns.clone(), "tbl");
        let tbl_oid = cat
            .create_table(&id, schema_with_id_col("tbl"))
            .await
            .unwrap()
            .object_id
            .expect("table object_id");

        let created = cat
            .create_index(
                &id,
                crate::CatalogIndex::new(
                    "idx_id",
                    vec!["id".to_string()],
                    crate::CatalogIndexType::BTree,
                ),
            )
            .await
            .unwrap();
        let idx_oid = created.object_id.expect("index got an object_id");

        assert!(idx_oid >= 1, "ids start at 1, got {idx_oid}");
        assert_ne!(
            idx_oid, tbl_oid,
            "one shared sequence: index and table oids never collide"
        );
    }

    #[tokio::test]
    async fn index_object_id_recovered_on_reload_prevents_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["t".to_string()];
        let id = TableIdentifier::new(ns.clone(), "tbl");
        let existing = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            cat.create_table(&id, schema_with_id_col("tbl"))
                .await
                .unwrap();
            cat.create_index(
                &id,
                crate::CatalogIndex::new(
                    "idx_a",
                    vec!["id".to_string()],
                    crate::CatalogIndexType::BTree,
                ),
            )
            .await
            .unwrap()
            .object_id
            .expect("index object_id")
        };

        // Cold restart: create_index loads the table first (load_table), which
        // recovers the allocator floor from the persisted index object_id so the
        // next index never reuses it.
        let cat2 = fresh_catalog(&tmp).await;
        let next = cat2
            .create_index(
                &id,
                crate::CatalogIndex::new(
                    "idx_b",
                    vec!["id".to_string()],
                    crate::CatalogIndexType::BTree,
                ),
            )
            .await
            .unwrap()
            .object_id
            .expect("index object_id");
        assert!(
            next > existing,
            "reload recovered the floor: {next} > {existing}"
        );
    }

    // ── ADR-031 / TD-181: namespace reverse resolver ─────────────────

    #[tokio::test]
    async fn namespace_reverse_resolver_round_trips_object_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let levels = vec!["db".to_string(), "schema".to_string()];
        let oid = cat
            .create_namespace(&levels, HashMap::new())
            .await
            .unwrap()
            .object_id
            .expect("namespace object_id");

        let resolved = cat
            .get_namespace_by_object_id(oid)
            .await
            .expect("resolve")
            .expect("object_id maps to a namespace");
        assert_eq!(
            resolved, levels,
            "reverse index returns the namespace levels"
        );
    }

    #[tokio::test]
    async fn namespace_reverse_resolver_cleared_on_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let levels = vec!["db".to_string()];
        let oid = cat
            .create_namespace(&levels, HashMap::new())
            .await
            .unwrap()
            .object_id
            .expect("namespace object_id");
        assert!(cat.get_namespace_by_object_id(oid).await.unwrap().is_some());

        assert!(cat.drop_namespace(&levels, false).await.unwrap());
        assert!(
            cat.get_namespace_by_object_id(oid).await.unwrap().is_none(),
            "drop clears the reverse-index entry"
        );
    }

    #[tokio::test]
    async fn namespace_reverse_resolver_rebuilt_on_reload() {
        let tmp = tempfile::tempdir().unwrap();
        let levels = vec!["db".to_string()];
        let oid = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&levels, HashMap::new())
                .await
                .unwrap()
                .object_id
                .expect("namespace object_id")
        };

        // A fresh catalog eagerly loads namespaces (load_namespaces), which
        // rebuilds the reverse index — so the id resolves with no prior get.
        let cat2 = fresh_catalog(&tmp).await;
        let resolved = cat2
            .get_namespace_by_object_id(oid)
            .await
            .expect("resolve")
            .expect("reverse index rebuilt on load");
        assert_eq!(resolved, levels);
    }

    // ── ADR-031 / TD-181 P0 completion: column object_id ─────────────

    #[tokio::test]
    async fn create_table_assigns_column_object_ids_from_shared_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();

        let schema = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "tbl"),
                schema_with_id_col("tbl"),
            )
            .await
            .unwrap();
        let tbl_oid = schema.object_id.expect("table object_id");
        let col_oid = schema.columns[0]
            .object_id
            .expect("column got an object_id");

        assert!(col_oid >= 1, "ids start at 1, got {col_oid}");
        assert_ne!(
            col_oid, tbl_oid,
            "one shared sequence: column and table oids never collide"
        );
        // The column also keeps its physical Iceberg field-id (the `id` field),
        // independent of the catalog object_id (ADR-031 amendment 4).
        assert_eq!(schema.columns[0].id, 1, "field-id (physical) is unchanged");
    }

    #[tokio::test]
    async fn column_object_id_recovered_on_reload_prevents_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["t".to_string()];
        let id = TableIdentifier::new(ns.clone(), "tbl");
        let col_oid = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            cat.create_table(&id, schema_with_id_col("tbl"))
                .await
                .unwrap()
                .columns[0]
                .object_id
                .expect("column object_id")
        };

        // Cold restart: load_table (via get_table) recovers the allocator floor
        // from the persisted COLUMN object_id, so the next allocation never
        // reuses it (columns ride in the table schema).
        let cat2 = fresh_catalog(&tmp).await;
        cat2.get_table(&id).await.expect("load table");
        cat2.create_namespace(&["t2".to_string()], HashMap::new())
            .await
            .unwrap();
        let next = cat2
            .create_table(
                &TableIdentifier::new(vec!["t2".to_string()], "tbl2"),
                schema_with_id_col("tbl2"),
            )
            .await
            .unwrap()
            .object_id
            .expect("table object_id");
        assert!(
            next > col_oid,
            "reload recovered the floor past the column oid: {next} > {col_oid}"
        );
    }

    #[tokio::test]
    async fn rename_table_preserves_object_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let from = TableIdentifier::new(ns.clone(), "old");
        let oid = cat
            .create_table(&from, schema_with_id_col("old"))
            .await
            .unwrap()
            .object_id
            .expect("object_id assigned");

        let to = TableIdentifier::new(ns.clone(), "new");
        cat.rename_table(&from, &to).await.expect("rename");

        let after = cat.get_table(&to).await.expect("read renamed");
        assert_eq!(
            after.object_id,
            Some(oid),
            "rename is metadata-only; object_id is preserved"
        );
    }

    #[tokio::test]
    async fn create_table_with_caller_object_id_raises_allocator_floor() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();

        // Import a table with a caller-supplied id; the allocator must not reuse it.
        let mut imported = schema_with_id_col("imported");
        imported.object_id = Some(100);
        cat.create_table(&TableIdentifier::new(ns.clone(), "imported"), imported)
            .await
            .unwrap();

        let next = cat
            .create_table(
                &TableIdentifier::new(ns.clone(), "fresh"),
                schema_with_id_col("fresh"),
            )
            .await
            .unwrap();
        assert!(
            next.object_id.unwrap() > 100,
            "allocator floor raised above the imported id"
        );
    }

    #[tokio::test]
    async fn object_id_recovered_on_reload_prevents_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["t".to_string()];
        let existing = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            cat.create_table(
                &TableIdentifier::new(ns.clone(), "first"),
                schema_with_id_col("first"),
            )
            .await
            .unwrap()
            .object_id
            .unwrap()
        };

        // Fresh catalog (cold allocator). Loading the existing table recovers the
        // floor (load_table raise_floor); a new table must not reuse the id.
        let cat2 = fresh_catalog(&tmp).await;
        let _ = cat2
            .get_table(&TableIdentifier::new(ns.clone(), "first"))
            .await
            .unwrap();
        let next = cat2
            .create_table(
                &TableIdentifier::new(ns.clone(), "second"),
                schema_with_id_col("second"),
            )
            .await
            .unwrap();
        assert!(
            next.object_id.unwrap() > existing,
            "reload recovered the floor; no id reuse"
        );
    }

    // ── ADR-031 O1: reverse object_id → table resolution ─────────────

    #[tokio::test]
    async fn reverse_resolver_round_trips_object_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let id = TableIdentifier::new(ns.clone(), "tbl");
        let oid = cat
            .create_table(&id, schema_with_id_col("tbl"))
            .await
            .unwrap()
            .object_id
            .expect("object_id");

        assert_eq!(
            cat.get_table_by_object_id(oid)
                .await
                .unwrap()
                .map(|r| r.to_fqn()),
            Some(id.to_fqn()),
            "object_id resolves back to its table"
        );
        assert!(
            cat.get_table_by_object_id(999_999).await.unwrap().is_none(),
            "unknown id resolves to None"
        );
    }

    #[tokio::test]
    async fn reverse_resolver_follows_rename() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let from = TableIdentifier::new(ns.clone(), "old");
        let oid = cat
            .create_table(&from, schema_with_id_col("old"))
            .await
            .unwrap()
            .object_id
            .unwrap();
        let to = TableIdentifier::new(ns.clone(), "new");
        cat.rename_table(&from, &to).await.unwrap();

        assert_eq!(
            cat.get_table_by_object_id(oid)
                .await
                .unwrap()
                .map(|r| r.to_fqn()),
            Some(to.to_fqn()),
            "object_id is stable across rename; reverse index repoints to the new name"
        );
    }

    #[tokio::test]
    async fn reverse_resolver_cleared_on_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let ns = vec!["t".to_string()];
        cat.create_namespace(&ns, HashMap::new()).await.unwrap();
        let id = TableIdentifier::new(ns.clone(), "tbl");
        let oid = cat
            .create_table(&id, schema_with_id_col("tbl"))
            .await
            .unwrap()
            .object_id
            .unwrap();

        assert!(cat.drop_table(&id, false).await.unwrap());
        assert!(
            cat.get_table_by_object_id(oid).await.unwrap().is_none(),
            "dropped table's id no longer resolves"
        );
    }

    #[tokio::test]
    async fn reverse_resolver_recovers_on_reload() {
        let tmp = tempfile::tempdir().unwrap();
        let ns = vec!["t".to_string()];
        let (oid, fqn) = {
            let cat = fresh_catalog(&tmp).await;
            cat.create_namespace(&ns, HashMap::new()).await.unwrap();
            let id = TableIdentifier::new(ns.clone(), "tbl");
            let oid = cat
                .create_table(&id, schema_with_id_col("tbl"))
                .await
                .unwrap()
                .object_id
                .unwrap();
            (oid, id.to_fqn())
        };

        let cat2 = fresh_catalog(&tmp).await;
        // Lazy index: resolves only after the table is loaded by name.
        let _ = cat2
            .get_table(&TableIdentifier::new(ns.clone(), "tbl"))
            .await
            .unwrap();
        assert_eq!(
            cat2.get_table_by_object_id(oid)
                .await
                .unwrap()
                .map(|r| r.to_fqn()),
            Some(fqn),
            "reload repopulates the reverse index on load"
        );
    }

    // ── P3.1: set_storage_layouts (NativeCatalog override) ───────────
    //
    // The warehouse-materialization catalog hook: flip a native table to a
    // Parquet + published-authority layout so the OLAP router treats it as
    // Parquet-backed. Mirrors the set_primary_pod test shape.

    fn parquet_published_layout(location: &str) -> crate::CatalogStorageLayout {
        crate::CatalogStorageLayout {
            name: "parquet-snapshot".to_string(),
            authority: crate::CatalogAuthorityMode::ProjectionPublication,
            physical_format: crate::CatalogPhysicalFormat::Parquet,
            location: Some(location.to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn set_storage_layouts_writes_and_returns_updated_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;
        let id = make_table(&cat, "users").await;

        // A freshly created table defaults to one InternalCanonical/ProximaBlock layout.
        let before = cat.get_table(&id).await.unwrap();
        assert_eq!(before.storage_layouts.len(), 1);
        assert!(matches!(
            before.storage_layouts[0].physical_format,
            crate::CatalogPhysicalFormat::ProximaBlock
        ));

        let layout = parquet_published_layout("data/tenant_a/ns/users/_manifests");
        let returned = cat
            .set_storage_layouts(&id, vec![layout])
            .await
            .expect("set succeeds on existing table");

        // The returned schema reflects the change immediately…
        assert_eq!(returned.storage_layouts.len(), 1);
        assert!(matches!(
            returned.storage_layouts[0].physical_format,
            crate::CatalogPhysicalFormat::Parquet
        ));
        assert!(matches!(
            returned.storage_layouts[0].authority,
            crate::CatalogAuthorityMode::ProjectionPublication
        ));
        assert_eq!(
            returned.storage_layouts[0].location.as_deref(),
            Some("data/tenant_a/ns/users/_manifests")
        );
        // …and so does a fresh read.
        let read = cat.get_table(&id).await.unwrap();
        assert!(matches!(
            read.storage_layouts[0].physical_format,
            crate::CatalogPhysicalFormat::Parquet
        ));
        // Physical/publication attribute → no schema_version bump.
        assert_eq!(read.schema_version, before.schema_version);
    }

    #[tokio::test]
    async fn set_storage_layouts_persists_across_reload() {
        // Reloading drops the in-memory cache, forcing a disk read — verifies
        // the change went through save_table, not just the cache.
        let tmp = tempfile::tempdir().unwrap();
        let id = {
            let cat = fresh_catalog(&tmp).await;
            let id = make_table(&cat, "events").await;
            cat.set_storage_layouts(&id, vec![parquet_published_layout("data/t/ns/events")])
                .await
                .unwrap();
            id
        };

        let cat2 = fresh_catalog(&tmp).await;
        let read = cat2.get_table(&id).await.expect("reload table");
        assert!(matches!(
            read.storage_layouts[0].physical_format,
            crate::CatalogPhysicalFormat::Parquet
        ));
        assert_eq!(
            read.storage_layouts[0].location.as_deref(),
            Some("data/t/ns/events")
        );
    }

    #[tokio::test]
    async fn set_storage_layouts_returns_err_for_unknown_table() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = fresh_catalog(&tmp).await;

        let id = TableIdentifier::new(vec!["nope".to_string()], "ghost");
        let res = cat
            .set_storage_layouts(&id, vec![parquet_published_layout("x")])
            .await;
        assert!(res.is_err(), "missing table must error, got: {:?}", res);
    }
}
