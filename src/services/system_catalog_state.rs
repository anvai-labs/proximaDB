//! In-RAM authoritative state for the read-heavy **system catalog**.
//!
//! This is the relcache/catcache of ProximaDB: a small, in-memory authoritative
//! index of every catalog object (namespaces + tables), made durable by a
//! bounded, fsync'd canonical WAL rather than an LSM. Catalog reads (which
//! dominate the workload — every query resolves namespaces/tables) are served
//! from RAM with zero syscalls; rare DDL writes are folded in under a single
//! write lock so the secondary indexes stay mutually consistent.
//!
//! Phase 1 scope (system-catalog redesign): the in-RAM authority + the
//! WAL-delta encoding + replay-fold + snapshot (de)serialization, unit-tested in
//! isolation over a [`FramedTableWalAppender`]. It is deliberately **not** wired
//! into the metadata-backend factory yet, and does **not** implement the
//! `Catalog` trait — that live cutover is Phase 2.
//!
//! ## Why this lives in the root crate (not `proximadb-catalog`)
//! The proven canonical-WAL substrate ([`FramedTableWalAppender`] +
//! [`TableWalAppender`], mirrored from [`crate::services::function_store`]) lives
//! in the root crate, and a *control*-layer crate cannot depend on the root
//! crate. The `CanonicalOperation::CatalogMutation` variant therefore carries an
//! opaque `bytes` payload; this module owns the [`CatalogDelta`] grammar that
//! encodes/decodes it, keeping `proximadb-storage-common` decoupled from the
//! control-plane catalog types.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use proximadb_catalog::{
    CatalogNamespace, CatalogTableSchema, CatalogTableStatistics, TableIdentifier,
};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};

/// One catalog DDL delta — the unit of change folded into the in-RAM authority
/// and serialized into a `CanonicalOperation::CatalogMutation` payload.
///
/// Namespaces are keyed by their `levels` path and tables by [`TableIdentifier`]
/// (namespace path + name), matching the `Catalog` trait's addressing scheme.
///
/// `PartialEq` is intentionally not derived: `CatalogTableSchema` does not
/// implement it (it absorbs storage-plane fields that are not comparable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CatalogDelta {
    /// Mint (or replay) the durable account/tenant string -> stable u32 used by
    /// ADR-0083 paths and ABAC policy lookup. The mapping is append-only: callers
    /// serialize first mint under the SystemCatalog write lock, and replay simply
    /// restores the committed value.
    UpsertAccount { account: String, stable_id: u32 },
    /// Create or replace a namespace. Boxed to keep the enum small (the
    /// namespace carries the account/tenant/DR authority fields).
    UpsertNamespace { namespace: Box<CatalogNamespace> },
    /// Drop a namespace and all of its tables.
    DropNamespace { levels: Vec<String> },
    /// Create or replace a table/collection.
    UpsertTable {
        identifier: TableIdentifier,
        schema: Box<CatalogTableSchema>,
    },
    /// Drop a table/collection.
    DropTable { identifier: TableIdentifier },
    /// Set (replace) a table's statistics. Boxed to keep the enum small.
    UpsertStatistics {
        identifier: TableIdentifier,
        stats: Box<CatalogTableStatistics>,
    },
}

impl CatalogDelta {
    /// Human-readable routing key carried alongside the opaque payload. Lets the
    /// WAL be read as a catalog log (debugging, future routing) without decoding
    /// the bytes.
    pub fn routing_key(&self) -> String {
        match self {
            CatalogDelta::UpsertAccount { account, .. } => format!("account:{account}"),
            CatalogDelta::UpsertNamespace { namespace } => {
                format!("ns:{}", namespace.levels.join("."))
            }
            CatalogDelta::DropNamespace { levels } => format!("ns-delete:{}", levels.join(".")),
            CatalogDelta::UpsertTable { identifier, .. } => {
                format!("table:{}", identifier.to_fqn())
            }
            CatalogDelta::DropTable { identifier } => {
                format!("table-delete:{}", identifier.to_fqn())
            }
            CatalogDelta::UpsertStatistics { identifier, .. } => {
                format!("stats:{}", identifier.to_fqn())
            }
        }
    }

    /// TD-PATH-2 4a: can this delta type raise the identity high-water the
    /// sidecar carries? Every variant can mint or retire an id EXCEPT
    /// `UpsertStatistics` — it mutates only a per-table stats row (row counts,
    /// sizes), so a stats-only commit cannot move the floor. This gates the
    /// mint-time sidecar persist in `SystemCatalog::commit_batch_locked`:
    /// DML row-count updates flow through `update_statistics` as
    /// `UpsertStatistics`-only commits on the ingest hot path, and they must
    /// not pay the sidecar round-trip (a read + tmp + fsync + rename on local
    /// stores; a GET+PUT pair on `requires_snapshot_on_commit` object stores).
    /// The checkpoint-time write stays unconditional — that one is
    /// load-bearing for WAL compaction.
    pub fn touches_ids_high_water(&self) -> bool {
        !matches!(self, CatalogDelta::UpsertStatistics { .. })
    }

    /// Encode this delta as a `CanonicalOperation::CatalogMutation` ready for
    /// [`TableWalAppender::append_operations`]. The body is field-named
    /// MessagePack (so the additive-serde evolution of `CatalogTableSchema`
    /// stays compatible), framed by the TD-SPINE-1 record envelope: magic +
    /// format version + body codec. A pre-envelope binary cannot decode it
    /// (accepted pre-GA break); this binary decodes both directions by
    /// sniffing the magic.
    pub fn to_operation(&self) -> Result<CanonicalOperation> {
        let bytes = crate::services::catalog_record_envelope::to_enveloped_vec_named(self)
            .context("encoding catalog delta")?;
        Ok(CanonicalOperation::CatalogMutation {
            key: self.routing_key(),
            bytes,
        })
    }

    /// Decode a delta from a `CatalogMutation` payload. Sniffs the envelope:
    /// magic present ⇒ envelope, absent ⇒ legacy bare MessagePack (mixed-read
    /// forever); unknown envelope version/codec fails closed.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        crate::services::catalog_record_envelope::from_enveloped_slice(bytes)
            .context("decoding catalog delta")
    }
}

/// The authoritative in-RAM index. Held under one `RwLock` so a DDL that touches
/// several maps (a table insert updates `tables` *and* `ns_children`) is atomic
/// across all of them — the cross-map consistency a catalog needs, which a
/// shard-locked map (DashMap) could not provide. DDL is rare, so write-lock
/// contention is a non-issue; reads take the cheap shared lock.
#[derive(Debug, Default)]
struct CatalogInner {
    /// Durable account/tenant string -> stable u32 registry. This is the
    /// SystemCatalog equivalent of NativeCatalog's account_registry and is the
    /// single policy-lookup key authority for normal server deployments.
    account_ids: HashMap<String, u32>,
    /// Namespaces keyed by their `levels` path.
    namespaces: HashMap<Vec<String>, CatalogNamespace>,
    /// Tables keyed by full identifier. `Arc` so reads clone a pointer, not the
    /// (large, typed-field-heavy) schema.
    tables: HashMap<TableIdentifier, Arc<CatalogTableSchema>>,
    /// Secondary index: namespace path → child table names, so `list_tables`
    /// never touches the filesystem.
    ns_children: HashMap<Vec<String>, HashSet<String>>,
    /// Per-table statistics (stored separately, like NativeCatalog's
    /// `TableMetadata.statistics`, not on the schema).
    statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
    /// Derived inverse index for stable table/model asset identity. It is not
    /// persisted separately: WAL replay and snapshot load rebuild it from the
    /// authoritative schemas, so it cannot drift from `tables`.
    object_ids: HashMap<u64, TableIdentifier>,
    /// TD-SPINE-2: per-account namespace-id high-water (the highest typed
    /// namespace id ever minted in that account, dropped ones included). The
    /// namespace-id floor is **per-account** for NEW allocations; existing
    /// ids are never renumbered. Like the global `ids_high_water` it is
    /// rebuilt by replay/snapshot/shard folds, and each account's shard
    /// persists its own value so compaction cannot forget a dropped id.
    account_ns_high_water: HashMap<String, u16>,
    // Preserve field absence before live rows rebuild the per-account map.
    loaded_legacy_namespace_snapshot: bool,
    /// Highest WAL sequence number folded in (the replay watermark).
    applied_seq: u64,
    /// TD-PATH-2: the highest identity ever minted, which drops never lower.
    /// The allocator floors used to be derived only from LIVE schemas, so
    /// dropping the newest table and restarting re-issued its ids — a typed
    /// `collection_id` re-minted the dropped collection's
    /// `accounts/{a}/{n}/{c}` root, and an `object_id` handed a new table the
    /// dead one's grants. WAL replay rebuilds these from the create deltas;
    /// the snapshot carries them so compaction cannot forget a dropped id.
    ids_high_water: IdsHighWater,
    /// TD-PATH-2 4c: writer-version stamp of the snapshot this state was last
    /// decoded from. `None` = built without a snapshot (WAL replay only);
    /// `Some(None)` = adopted an UNSTAMPED snapshot, i.e. published by a
    /// pre-high-water binary — the WARN condition. Transient: never
    /// serialized, set only by [`Self::inner_from_snapshot_bytes`].
    loaded_writer_version: Option<Option<u32>>,
}

/// TD-PATH-2: monotonic identity high-water marks (see `CatalogInner`).
/// Public because the TD-SPINE-2 root decode surface hands it to the
/// catalog's boot/reload protocol.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdsHighWater {
    pub object_id: u64,
    pub collection_id: u32,
    /// Typed namespace ids (`{n}` of `accounts/{a}/{n}/{c}`), recovered from
    /// table schemas: emptying the highest namespace must not free its id.
    #[serde(default)]
    pub namespace_id: u16,
}

impl IdsHighWater {
    /// TD-PATH-2 4a: monotone merge — raise every dimension to the max of the
    /// two, never lower. The sidecar adopt path and the merge-before-write in
    /// `SystemCatalog::persist_identity_high_water_sidecar` both rely on this
    /// so a stale or partially-read artifact can never regress the floor.
    fn merge_max(&mut self, other: IdsHighWater) {
        self.object_id = self.object_id.max(other.object_id);
        self.collection_id = self.collection_id.max(other.collection_id);
        self.namespace_id = self.namespace_id.max(other.namespace_id);
    }

    fn fold_table(&mut self, schema: &CatalogTableSchema) {
        let ids = schema
            .object_id
            .into_iter()
            .chain(schema.columns.iter().filter_map(|c| c.object_id))
            .chain(schema.indexes.iter().filter_map(|i| i.object_id));
        for id in ids {
            self.object_id = self.object_id.max(id);
        }
        if let Some(id) = schema.stable_collection_id {
            self.collection_id = self.collection_id.max(id);
        }
        if let Some(id) = schema.stable_namespace_id {
            self.namespace_id = self.namespace_id.max(id);
        }
    }

    fn fold_namespace(&mut self, namespace: &CatalogNamespace) {
        if let Some(id) = namespace.object_id {
            self.object_id = self.object_id.max(id);
        }
    }
}

impl CatalogInner {
    /// TD-SPINE-2: fold a table schema's typed namespace id into the
    /// per-account high-water. The account is the namespace's tenant (the
    /// same attribution `resolve_typed_triple` mints under); a table whose
    /// namespace is absent folds into the shared default shard so the floor
    /// can never under-count.
    fn fold_table_account_ns(&mut self, namespace_levels: &[String], schema: &CatalogTableSchema) {
        if let Some(namespace_id) = schema.stable_namespace_id {
            let account = self.shard_key_of_levels(namespace_levels);
            let entry = self.account_ns_high_water.entry(account).or_insert(0);
            *entry = (*entry).max(namespace_id);
        }
    }

    /// TD-SPINE-2: the shard key of a namespace path — its namespace's
    /// tenant when recorded, else the shared default shard.
    fn shard_key_of_levels(&self, levels: &[String]) -> String {
        shard_key_of_tenant(
            self.namespaces
                .get(levels)
                .and_then(|namespace| namespace.tenant_id.as_deref()),
        )
    }

    /// TD-SPINE-2: remove every row belonging to `account`'s shard — the
    /// inverse of [`Self::apply_shard`], so adopting a shard image replaces
    /// exactly that account's rows (dropped namespaces cascade to their
    /// tables/statistics exactly like the `DropNamespace` fold).
    fn remove_shard_rows(&mut self, account: &str) {
        let doomed: Vec<Vec<String>> = self
            .namespaces
            .values()
            .filter(|namespace| shard_key_of_tenant(namespace.tenant_id.as_deref()) == account)
            .map(|namespace| namespace.levels.clone())
            .collect();
        for levels in doomed {
            self.namespaces.remove(&levels);
            if let Some(children) = self.ns_children.remove(&levels) {
                for name in children {
                    let id = TableIdentifier::new(levels.clone(), name);
                    if let Some(schema) = self.tables.remove(&id)
                        && let Some(object_id) = schema.object_id
                    {
                        self.object_ids.remove(&object_id);
                    }
                    self.statistics.remove(&id);
                }
            }
        }
        // Defensive: orphan tables (namespace record missing) live in the
        // default shard, so adopting the default shard retires them too.
        if account == DEFAULT_SHARD_KEY {
            let orphans: Vec<TableIdentifier> = self
                .tables
                .keys()
                .filter(|id| !self.namespaces.contains_key(&id.namespace))
                .cloned()
                .collect();
            for id in orphans {
                if let Some(schema) = self.tables.remove(&id)
                    && let Some(object_id) = schema.object_id
                {
                    self.object_ids.remove(&object_id);
                }
                self.statistics.remove(&id);
                if let Some(children) = self.ns_children.get_mut(&id.namespace) {
                    children.remove(&id.name);
                }
            }
        }
    }

    /// TD-SPINE-2: fold a decoded shard image into state (boot assembly and
    /// per-account reload adoption). Mirrors the snapshot-load folding so
    /// secondary indexes and the per-account namespace-id high-water stay
    /// consistent; the GLOBAL high-water also merges (it is the legacy
    /// fleet-wide record and must never regress).
    fn apply_shard(&mut self, shard: CatalogShardRecord) {
        for namespace in shard.namespaces.values() {
            self.ids_high_water.fold_namespace(namespace);
        }
        let mut account_ns = shard.namespace_id_high_water;
        for (id, schema) in &shard.tables {
            self.ids_high_water.fold_table(schema);
            if let Some(namespace_id) = schema.stable_namespace_id {
                account_ns = account_ns.max(namespace_id);
            }
            self.ns_children
                .entry(id.namespace.clone())
                .or_default()
                .insert(id.name.clone());
            if let Some(object_id) = schema.object_id {
                self.object_ids.insert(object_id, id.clone());
            }
        }
        for levels in shard.namespaces.keys() {
            self.ns_children.entry(levels.clone()).or_default();
        }
        self.namespaces.extend(shard.namespaces);
        self.tables.extend(
            shard
                .tables
                .into_iter()
                .map(|(id, schema)| (id, Arc::new(schema))),
        );
        self.statistics.extend(shard.statistics);
        let entry = self.account_ns_high_water.entry(shard.account).or_insert(0);
        *entry = (*entry).max(account_ns);
    }

    fn apply(&mut self, delta: CatalogDelta) {
        match delta {
            CatalogDelta::UpsertAccount { account, stable_id } => {
                self.account_ids.insert(account, stable_id);
            }
            CatalogDelta::UpsertNamespace { namespace } => {
                self.ids_high_water.fold_namespace(&namespace);
                let key = namespace.levels.clone();
                self.ns_children.entry(key.clone()).or_default();
                self.namespaces.insert(key, *namespace);
            }
            CatalogDelta::DropNamespace { levels } => {
                self.namespaces.remove(&levels);
                if let Some(children) = self.ns_children.remove(&levels) {
                    for name in children {
                        let id = TableIdentifier::new(levels.clone(), name);
                        if let Some(schema) = self.tables.remove(&id)
                            && let Some(object_id) = schema.object_id
                        {
                            self.object_ids.remove(&object_id);
                        }
                        self.statistics.remove(&id);
                    }
                }
            }
            CatalogDelta::UpsertTable { identifier, schema } => {
                self.ids_high_water.fold_table(&schema);
                self.fold_table_account_ns(&identifier.namespace, &schema);
                if let Some(previous) = self.tables.get(&identifier)
                    && let Some(object_id) = previous.object_id
                {
                    self.object_ids.remove(&object_id);
                }
                if let Some(object_id) = schema.object_id {
                    self.object_ids.insert(object_id, identifier.clone());
                }
                self.ns_children
                    .entry(identifier.namespace.clone())
                    .or_default()
                    .insert(identifier.name.clone());
                self.tables.insert(identifier, Arc::new(*schema));
            }
            CatalogDelta::DropTable { identifier } => {
                if let Some(schema) = self.tables.remove(&identifier)
                    && let Some(object_id) = schema.object_id
                {
                    self.object_ids.remove(&object_id);
                }
                self.statistics.remove(&identifier);
                if let Some(children) = self.ns_children.get_mut(&identifier.namespace) {
                    children.remove(&identifier.name);
                }
            }
            CatalogDelta::UpsertStatistics { identifier, stats } => {
                self.statistics.insert(identifier, *stats);
            }
        }
    }
}

/// Serializable point-in-time image of the catalog, used for the (Phase 3)
/// snapshot that bounds WAL replay. `ns_children` is omitted — it is a pure
/// derivation of `tables` and is rebuilt on load.
#[derive(Debug, Serialize, Deserialize)]
struct CatalogSnapshot {
    #[serde(default)]
    account_ids: HashMap<String, u32>,
    namespaces: HashMap<Vec<String>, CatalogNamespace>,
    tables: HashMap<TableIdentifier, CatalogTableSchema>,
    #[serde(default)]
    statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
    applied_seq: u64,
    /// TD-PATH-2: additive and optional — an older binary ignores the key, and
    /// a snapshot written before it decodes as `None` (the live schemas then
    /// bound the floor, as before).
    #[serde(default)]
    ids_high_water: Option<IdsHighWater>,
    /// TD-PATH-2 4c: writer-version stamp — identifies the publisher's
    /// identity-high-water awareness. Absent on decode = published by a
    /// pre-high-water binary (whose checkpoints compact the WAL past the
    /// watermark without preserving the dropped-id floor anywhere it names);
    /// a high-water-aware binary WARNs when it adopts such a snapshot.
    #[serde(default)]
    writer_version: Option<u32>,
    /// The per-account namespace floor, including dropped namespaces. Keep
    /// legacy field absence distinct from a present map rebuilt from live rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    account_ns_high_water: Option<HashMap<String, u16>>,
}

/// TD-SPINE-2 (ADR-098 D1.3): the shared shard key for namespaces (and their
/// tables/statistics) that carry no owning account — namespaces created
/// without a tenant scope. It is a *shard* key, not a path segment; the
/// on-storage path is hex-encoded by `DrPathBuilder`, so the reserved-looking
/// underscore is inert there.
pub const DEFAULT_SHARD_KEY: &str = "_default";

/// TD-SPINE-2 review MED-1: recover the per-account namespace-id high-water
/// map of a LEGACY image. Pre-SPINE-2 snapshots/roots carry no per-account
/// field (the map decodes empty) and every id was minted under the GLOBAL
/// fleet-wide counter — so any account that could have minted (the registry
/// plus live namespaces) is conservatively bounded by the global high-water.
/// Without this, a migration or escape-reseed boot rebuilds each account's
/// floor from LIVE rows only and can re-mint a dropped namespace's id in
/// that account, aliasing its stale `accounts/{a}/{n}/…` paths (TD-PATH-2:
/// "skipped ids are harmless; reuse is not"). A NON-empty map is a
/// post-SPINE-2 image and is trusted as-is — that is the per-account
/// minting contract (two accounts may each hold id 1).
fn recover_legacy_account_ns_high_water(
    decoded: HashMap<String, u16>,
    global_namespace_id: Option<u16>,
    candidate_accounts: std::collections::HashSet<String>,
) -> HashMap<String, u16> {
    if !decoded.is_empty() {
        return decoded;
    }
    match global_namespace_id {
        Some(global) if global > 0 => candidate_accounts
            .into_iter()
            .map(|account| (account, global))
            .collect(),
        _ => decoded,
    }
}

/// TD-SPINE-2 (ADR-098 D1.3): the shard key of a namespace — its owning
/// account (tenant) when recorded, else the shared default shard. A pure
/// function of persisted state so the publisher and every follower agree on
/// shard membership without coordination: `namespace.tenant_id` is durable
/// (it rides the namespace record in WAL deltas, snapshots and shards).
pub fn shard_key_of_tenant(tenant_id: Option<&str>) -> String {
    match tenant_id {
        Some(tenant) if !tenant.is_empty() => tenant.to_string(),
        _ => DEFAULT_SHARD_KEY.to_string(),
    }
}

/// TD-SPINE-2 (ADR-098 D1.3): the **root** record — the small global half of
/// the sharded catalog snapshot. Carries the global registries (account →
/// stable id), the fleet-wide identity high-water (object/collection ids are
/// one global sequence; the `namespace_id` member stays the legacy GLOBAL
/// record for whole-image readers and never bounds per-account floors), the
/// publisher's replay watermark, and the `account → shard log version` index
/// followers diff against. `BTreeMap` so the encoding is byte-deterministic.
#[derive(Debug, Serialize, Deserialize)]
struct CatalogRootRecord {
    account_ids: HashMap<String, u32>,
    #[serde(default)]
    ids_high_water: Option<IdsHighWater>,
    applied_seq: u64,
    /// account (shard key) → version of that account's shard manifest log
    /// this root reflects.
    shard_index: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    writer_version: Option<u32>,
    /// TD-SPINE-2 (review MED-1): the per-account namespace-id high-water
    /// map, so the root — the image a migration or escape reseed rebuilds
    /// from — preserves each account's dropped-id memory even when a shard
    /// body is missing. Merged (max) on adoption; never lowers.
    #[serde(default)]
    account_ns_high_water: HashMap<String, u16>,
}

/// TD-SPINE-2 (ADR-098 D1.3): one **account shard** — every namespace,
/// table and statistics row whose shard key is this account. Self-describing
/// (`account`) so adoption knows exactly which account's rows to replace,
/// and it carries the account's own namespace-id high-water so a dropped
/// namespace's id stays reserved across compaction.
#[derive(Debug, Serialize, Deserialize)]
struct CatalogShardRecord {
    account: String,
    namespaces: HashMap<Vec<String>, CatalogNamespace>,
    tables: HashMap<TableIdentifier, CatalogTableSchema>,
    #[serde(default)]
    statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
    #[serde(default)]
    namespace_id_high_water: u16,
    #[serde(default)]
    applied_seq: u64,
}

/// TD-PATH-2 4a: the durable identity-high-water SIDECAR image — a tiny
/// object beside the catalog snapshot, written and read independently of the
/// snapshot pointer. It exists because an older (pre-high-water) binary
/// publishing a snapshot re-encodes the live state WITHOUT the `ids_high_water`
/// key and compacts the WAL past the watermark, destroying the create deltas
/// replay would fold; the sidecar is the floor record that publisher cannot
/// touch (it never names the sidecar's path/key).
#[derive(Debug, Serialize, Deserialize)]
struct IdentityHighWaterSidecar {
    /// TD-PATH-2 4c stamp (`None` = written before the stamp existed).
    #[serde(default)]
    writer_version: Option<u32>,
    /// The floor itself (all fields always encoded by this writer).
    #[serde(default)]
    high_water: IdsHighWater,
    /// Absent on legacy sidecars. New sidecars retain the account that owns
    /// each namespace ID floor even if an old snapshot writer omits the map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    account_ns_high_water: Option<HashMap<String, u16>>,
}

/// In-RAM authoritative system-catalog state.
#[derive(Debug, Default)]
pub struct SystemCatalogState {
    inner: RwLock<CatalogInner>,
}

impl SystemCatalogState {
    /// An empty catalog (fresh deployment / known-empty WAL).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build state by folding a canonical-WAL slice. Non-catalog entries
    /// (record upserts/deletes, CDC barriers) are ignored; checkpoints carry no
    /// state here. Entries are applied in order; entries at or below the current
    /// `applied_seq` are skipped so replay is idempotent.
    pub fn from_wal_entries(entries: &[CanonicalWalEntry]) -> Result<Self> {
        let state = Self::new();
        state.replay(entries)?;
        Ok(state)
    }

    /// Fold a canonical-WAL slice into the current state (idempotent on
    /// `sequence_number`).
    pub fn replay(&self, entries: &[CanonicalWalEntry]) -> Result<()> {
        let mut inner = self.inner.write();
        for entry in entries {
            if entry.sequence_number <= inner.applied_seq {
                continue;
            }
            if let CanonicalOperation::CatalogMutation { bytes, .. } = &entry.operation {
                let delta = CatalogDelta::decode(bytes)?;
                inner.apply(delta);
                inner.applied_seq = entry.sequence_number;
            }
        }
        Ok(())
    }

    /// Apply a single decoded delta at the given WAL sequence number (used by the
    /// writer after a durable append). Skipped if already applied.
    pub fn apply_committed(&self, sequence_number: u64, delta: CatalogDelta) {
        let mut inner = self.inner.write();
        if sequence_number <= inner.applied_seq {
            return;
        }
        inner.apply(delta);
        inner.applied_seq = sequence_number;
    }

    // --- read API (pure-RAM, zero syscalls) ---

    /// Whether a table exists. Replaces `NativeCatalog`'s per-call `path.exists()`.
    pub fn table_exists(&self, identifier: &TableIdentifier) -> bool {
        self.inner.read().tables.contains_key(identifier)
    }

    /// Get a table schema by identifier (clones an `Arc`, not the schema).
    pub fn get_table(&self, identifier: &TableIdentifier) -> Option<Arc<CatalogTableSchema>> {
        self.inner.read().tables.get(identifier).cloned()
    }

    /// Resolve the rename-stable table/model asset identifier without scanning
    /// the catalog. The index is rebuilt from schemas on every recovery path.
    pub fn get_table_by_object_id(&self, object_id: u64) -> Option<TableIdentifier> {
        self.inner.read().object_ids.get(&object_id).cloned()
    }

    /// List the tables in a namespace. Replaces `NativeCatalog`'s per-call
    /// `read_dir`.
    pub fn list_tables(&self, namespace: &[String]) -> Vec<TableIdentifier> {
        let inner = self.inner.read();
        match inner.ns_children.get(namespace) {
            Some(names) => names
                .iter()
                .map(|name| TableIdentifier::new(namespace.to_vec(), name.clone()))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Whether a namespace exists.
    pub fn namespace_exists(&self, levels: &[String]) -> bool {
        self.inner.read().namespaces.contains_key(levels)
    }

    /// Get a namespace by its `levels` path.
    pub fn get_namespace(&self, levels: &[String]) -> Option<CatalogNamespace> {
        self.inner.read().namespaces.get(levels).cloned()
    }

    /// Resolve a namespace's `levels` from its stable catalog `object_id` —
    /// the namespace analogue of [`Self::get_table_by_object_id`].
    ///
    /// TD-CAT-7: a linear scan, not an index. Namespaces are few and this has
    /// no hot caller; the alternative that was in force until now — inheriting
    /// the trait's silent `Ok(None)` — answered "no such namespace" for every
    /// id, which is worse than an O(n) answer.
    pub fn get_namespace_by_object_id(&self, object_id: u64) -> Option<Vec<String>> {
        self.inner
            .read()
            .namespaces
            .values()
            .find(|ns| ns.object_id == Some(object_id))
            .map(|ns| ns.levels.clone())
    }

    /// List all namespace paths.
    pub fn list_namespaces(&self) -> Vec<Vec<String>> {
        self.inner.read().namespaces.keys().cloned().collect()
    }

    /// List all namespace objects (clones).
    pub fn all_namespaces(&self) -> Vec<CatalogNamespace> {
        self.inner.read().namespaces.values().cloned().collect()
    }

    /// Get a table's statistics, if any have been recorded.
    pub fn get_statistics(&self, identifier: &TableIdentifier) -> Option<CatalogTableStatistics> {
        self.inner.read().statistics.get(identifier).cloned()
    }

    /// Read-only hot-path lookup used by TenantStableIdResolver.
    pub fn account_id_u32(&self, account: &str) -> Option<u32> {
        self.inner.read().account_ids.get(account).copied()
    }

    /// Highest committed account id, used to recover the allocator floor after
    /// WAL/snapshot replay so ids are never reused across restarts.
    pub fn max_account_id(&self) -> Option<u32> {
        self.inner.read().account_ids.values().copied().max()
    }

    /// Snapshot the table identity rows used to recover the SystemCatalog's
    /// transient allocator floors and namespace registry after WAL replay.
    pub fn table_entries(&self) -> Vec<(TableIdentifier, Arc<CatalogTableSchema>)> {
        self.inner
            .read()
            .tables
            .iter()
            .map(|(identifier, schema)| (identifier.clone(), schema.clone()))
            .collect()
    }

    /// The highest `object_id` folded into state, across tables AND the columns
    /// and indexes they carry (they all draw from one sequence).
    ///
    /// The allocator floor is DERIVED from recovered state rather than persisted
    /// separately — the same shape as `max_account_id` above. Object ids already
    /// ride inside the persisted schemas, so WAL replay and snapshot load yield
    /// the floor for free: no second durable structure, no extra fsync, and no
    /// way for a floor record to disagree with the objects it governs.
    ///
    /// This is authorization-critical, not bookkeeping: policy bindings and
    /// grants are keyed on `object_id`, so re-issuing one after a restart would
    /// hand a NEW table the DEAD table's permissions.
    pub fn max_object_id(&self) -> Option<u64> {
        let guard = self.inner.read();
        let from_tables = guard
            .tables
            .values()
            .flat_map(|schema| {
                schema
                    .object_id
                    .into_iter()
                    .chain(schema.columns.iter().filter_map(|c| c.object_id))
                    .chain(schema.indexes.iter().filter_map(|i| i.object_id))
            })
            .max();
        // TD-CAT-9: namespaces draw from the SAME sequence, so the floor must
        // fold them too. This line and the mint in `create_namespace_inner` are
        // one change: minting without folding recovers a floor BELOW a persisted
        // namespace id after a restart and re-issues a live identity — the exact
        // hazard the paragraph above describes, only for namespaces. Neuter this
        // and `object_ids_are_never_reissued_across_a_restart` fails.
        let from_namespaces = guard
            .namespaces
            .values()
            .filter_map(|namespace| namespace.object_id)
            .max();
        // TD-PATH-2: ids of DROPPED objects stay reserved.
        let high_water = Some(guard.ids_high_water.object_id).filter(|id| *id > 0);
        from_tables.max(from_namespaces).max(high_water)
    }

    /// TD-PATH-2: the highest typed `collection_id` ever minted (live or
    /// dropped); `None` when none was. The allocator floor sits above it, so a
    /// dropped collection's typed storage root is never re-issued. The
    /// high-water already covers every live schema: every table enters state
    /// through `apply` or a snapshot load, and both fold it.
    pub fn max_collection_id_ever(&self) -> Option<u32> {
        Some(self.inner.read().ids_high_water.collection_id).filter(|id| *id > 0)
    }

    /// TD-PATH-2: the highest typed namespace id ever minted (live or emptied).
    pub fn max_namespace_id_ever(&self) -> Option<u16> {
        Some(self.inner.read().ids_high_water.namespace_id).filter(|id| *id > 0)
    }

    /// The highest WAL sequence number folded in (replay watermark / snapshot
    /// cutover LSN).
    pub fn applied_seq(&self) -> u64 {
        self.inner.read().applied_seq
    }

    // --- snapshot (de)serialization (bounds Phase 3 replay) ---

    /// Serialize the full state to a snapshot blob (envelope-framed
    /// field-named MessagePack, TD-SPINE-1).
    pub fn to_snapshot_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let snapshot = CatalogSnapshot {
            account_ids: inner.account_ids.clone(),
            namespaces: inner.namespaces.clone(),
            tables: inner
                .tables
                .iter()
                .map(|(id, schema)| (id.clone(), (**schema).clone()))
                .collect(),
            statistics: inner.statistics.clone(),
            applied_seq: inner.applied_seq,
            ids_high_water: Some(inner.ids_high_water),
            writer_version: Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER),
            account_ns_high_water: Some(inner.account_ns_high_water.clone()),
        };
        crate::services::catalog_record_envelope::to_enveloped_vec_named(&snapshot)
            .context("encoding catalog snapshot")
    }

    /// Reconstruct state from a snapshot blob, rebuilding the `ns_children`
    /// secondary index. Legacy (unframed) blobs decode unchanged.
    pub fn from_snapshot_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            inner: RwLock::new(Self::inner_from_snapshot_bytes(bytes)?),
        })
    }

    /// **Replace** the entire in-RAM authority with the image decoded from a
    /// snapshot blob, in place (the `Arc<SystemCatalogState>` identity is
    /// preserved so all holders observe the swap). This is the Phase 6b
    /// cross-pod **invalidation/reload** primitive: when a follower (or a
    /// superseded ex-owner) observes a newer object-store snapshot, it adopts
    /// that snapshot wholesale. The swap is atomic with respect to catalog reads
    /// — a concurrent reader sees the entire old image or the entire new one,
    /// never a partial blend — because it happens under the single write lock the
    /// reads take in shared mode.
    pub fn load_from_snapshot_bytes(&self, bytes: &[u8]) -> Result<()> {
        let rebuilt = Self::inner_from_snapshot_bytes(bytes)?;
        *self.inner.write() = rebuilt;
        Ok(())
    }

    /// Decode a snapshot blob into a fully-indexed [`CatalogInner`], rebuilding
    /// the `ns_children` and `object_ids` secondary indexes (which the snapshot
    /// omits as pure derivations of `tables`).
    fn inner_from_snapshot_bytes(bytes: &[u8]) -> Result<CatalogInner> {
        let snapshot: CatalogSnapshot =
            crate::services::catalog_record_envelope::from_enveloped_slice(bytes)
                .context("decoding catalog snapshot")?;
        let mut ns_children: HashMap<Vec<String>, HashSet<String>> = HashMap::new();
        // Every namespace gets an entry (so list_tables on an empty namespace
        // returns an empty set, not "missing").
        for levels in snapshot.namespaces.keys() {
            ns_children.entry(levels.clone()).or_default();
        }
        let mut tables = HashMap::with_capacity(snapshot.tables.len());
        let mut object_ids = HashMap::with_capacity(snapshot.tables.len());
        let mut ids_high_water = snapshot.ids_high_water.unwrap_or_default();
        // TD-SPINE-2 (+ review MED-1): whole-image loads rebuild the
        // per-account namespace-id high-water from BOTH the persisted
        // per-account map (dropped-id memory, carried since the review fix —
        // absent on older legacy images) and the live rows (folded below).
        let loaded_legacy_namespace_snapshot = snapshot.account_ns_high_water.is_none();
        let mut account_ns_high_water: HashMap<String, u16> = recover_legacy_account_ns_high_water(
            snapshot.account_ns_high_water.unwrap_or_default(),
            snapshot
                .ids_high_water
                .map(|high_water| high_water.namespace_id),
            snapshot
                .account_ids
                .keys()
                .cloned()
                .chain(
                    snapshot
                        .namespaces
                        .values()
                        .map(|ns| shard_key_of_tenant(ns.tenant_id.as_deref())),
                )
                .collect(),
        );
        let namespace_accounts: HashMap<Vec<String>, String> = snapshot
            .namespaces
            .iter()
            .map(|(levels, namespace)| {
                (
                    levels.clone(),
                    shard_key_of_tenant(namespace.tenant_id.as_deref()),
                )
            })
            .collect();
        for namespace in snapshot.namespaces.values() {
            ids_high_water.fold_namespace(namespace);
        }
        for (id, schema) in snapshot.tables {
            ids_high_water.fold_table(&schema);
            if let Some(namespace_id) = schema.stable_namespace_id {
                let account = namespace_accounts
                    .get(&id.namespace)
                    .cloned()
                    .unwrap_or_else(|| DEFAULT_SHARD_KEY.to_string());
                let entry = account_ns_high_water.entry(account).or_insert(0);
                *entry = (*entry).max(namespace_id);
            }
            ns_children
                .entry(id.namespace.clone())
                .or_default()
                .insert(id.name.clone());
            if let Some(object_id) = schema.object_id {
                object_ids.insert(object_id, id.clone());
            }
            tables.insert(id, Arc::new(schema));
        }
        Ok(CatalogInner {
            account_ids: snapshot.account_ids,
            namespaces: snapshot.namespaces,
            tables,
            ns_children,
            statistics: snapshot.statistics,
            object_ids,
            account_ns_high_water,
            loaded_legacy_namespace_snapshot,
            applied_seq: snapshot.applied_seq,
            ids_high_water,
            loaded_writer_version: Some(snapshot.writer_version),
        })
    }

    /// TD-PATH-2 4c: the writer-version stamp of the snapshot this state was
    /// last decoded from. `None` = the state was built without a snapshot
    /// (WAL replay only); `Some(None)` = an unstamped (pre-high-water)
    /// snapshot; `Some(Some(v))` = stamped by writer version `v`.
    pub fn loaded_writer_version(&self) -> Option<Option<u32>> {
        self.inner.read().loaded_writer_version
    }

    // --- TD-SPINE-2: sharded snapshot (root + one shard per account) -------

    /// The highest typed namespace id ever minted **in one account** (live or
    /// emptied; `None` when the account never minted one). The per-account
    /// allocator floor for NEW allocations sits above it; existing ids in
    /// other accounts are irrelevant (per-account uniqueness is the
    /// documented `accounts/{a}/{n}/{c}` model).
    pub fn max_namespace_id_ever_for(&self, account: &str) -> Option<u16> {
        self.inner
            .read()
            .account_ns_high_water
            .get(account)
            .copied()
            .filter(|id| *id > 0)
    }

    /// Every shard key with rows in state — the full account set a migration
    /// or full republish must cover.
    pub fn shard_keys(&self) -> std::collections::HashSet<String> {
        let inner = self.inner.read();
        inner
            .namespaces
            .values()
            .map(|namespace| shard_key_of_tenant(namespace.tenant_id.as_deref()))
            .collect()
    }

    /// TD-SPINE-2: the shard key of a namespace path in the current state
    /// (its namespace's tenant when recorded, else the default shard; an
    /// unknown namespace defaults too — mirroring the writer-side rule).
    pub fn shard_key_of_namespace(&self, levels: &[String]) -> String {
        self.inner.read().shard_key_of_levels(levels)
    }

    /// TD-SPINE-2: every account the per-account namespace-id high-water
    /// remembers — including accounts whose namespaces are all gone (their
    /// dropped ids must stay reserved across compaction, which is exactly
    /// why this map, not the live namespace set, seeds the allocator floors).
    pub fn ns_high_water_accounts(&self) -> std::collections::HashSet<String> {
        self.inner
            .read()
            .account_ns_high_water
            .keys()
            .cloned()
            .collect()
    }

    /// TD-SPINE-2: serialize the **root** image (global registries + id
    /// high-water + watermark + shard index), envelope-framed.
    pub fn to_root_bytes(
        &self,
        shard_index: &std::collections::BTreeMap<String, u64>,
    ) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let root = CatalogRootRecord {
            account_ids: inner.account_ids.clone(),
            ids_high_water: Some(inner.ids_high_water),
            applied_seq: inner.applied_seq,
            shard_index: shard_index.clone(),
            writer_version: Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER),
            account_ns_high_water: inner.account_ns_high_water.clone(),
        };
        crate::services::catalog_record_envelope::to_enveloped_vec_named(&root)
            .context("encoding catalog root")
    }

    /// TD-SPINE-2: serialize **one account's shard** image (its namespaces,
    /// tables, statistics and namespace-id high-water), envelope-framed.
    /// Rows are selected by shard key: a table/statistic belongs to its
    /// namespace's shard, or the default shard when the namespace is absent.
    pub fn to_shard_bytes(&self, account: &str) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let mut namespaces = HashMap::new();
        let mut member: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
        for namespace in inner.namespaces.values() {
            if shard_key_of_tenant(namespace.tenant_id.as_deref()) == account {
                member.insert(namespace.levels.clone());
                namespaces.insert(namespace.levels.clone(), namespace.clone());
            }
        }
        // A table whose namespace is missing from state (should not happen;
        // defensive) shards with the default shard, mirroring the
        // shard-key rule for absent namespaces.
        let tables: HashMap<TableIdentifier, CatalogTableSchema> = inner
            .tables
            .iter()
            .filter(|(id, _)| {
                member.contains(&id.namespace)
                    || (account == DEFAULT_SHARD_KEY
                        && !inner.namespaces.contains_key(&id.namespace))
            })
            .map(|(id, schema)| (id.clone(), (**schema).clone()))
            .collect();
        let statistics = inner
            .statistics
            .iter()
            .filter(|(id, _)| {
                member.contains(&id.namespace)
                    || (account == DEFAULT_SHARD_KEY
                        && !inner.namespaces.contains_key(&id.namespace))
            })
            .map(|(id, stats)| (id.clone(), stats.clone()))
            .collect();
        // The durable per-account high-water is the max of the folded floor
        // (which replay keeps above every live and dropped id) and whatever
        // the rows themselves carry.
        let mut namespace_id_high_water = inner.account_ns_high_water.get(account).copied();
        for schema in tables.values() {
            if let Some(id) = schema.stable_namespace_id {
                namespace_id_high_water = Some(namespace_id_high_water.unwrap_or(0).max(id));
            }
        }
        let shard = CatalogShardRecord {
            account: account.to_string(),
            namespaces,
            tables,
            statistics,
            namespace_id_high_water: namespace_id_high_water.unwrap_or(0),
            applied_seq: inner.applied_seq,
        };
        crate::services::catalog_record_envelope::to_enveloped_vec_named(&shard)
            .context("encoding catalog shard")
    }

    /// The decoded root, for the boot/reload protocol (shard-index diffing)
    /// without folding anything into state.
    pub fn decode_catalog_root(
        bytes: &[u8],
    ) -> Result<(
        HashMap<String, u32>,
        IdsHighWater,
        Option<u32>,
        u64,
        std::collections::BTreeMap<String, u64>,
    )> {
        let root: CatalogRootRecord =
            crate::services::catalog_record_envelope::from_enveloped_slice(bytes)
                .context("decoding catalog root")?;
        Ok((
            root.account_ids,
            root.ids_high_water.unwrap_or_default(),
            root.writer_version,
            root.applied_seq,
            root.shard_index,
        ))
    }

    /// TD-SPINE-2: build the full state from a root image plus every shard
    /// in its index (boot / follower assembly). The root contributes the
    /// global registries and high-water; each shard contributes its
    /// account's rows and per-account namespace-id high-water.
    pub fn from_root_and_shards(root_bytes: &[u8], shards: &[(String, Vec<u8>)]) -> Result<Self> {
        let root: CatalogRootRecord =
            crate::services::catalog_record_envelope::from_enveloped_slice(root_bytes)
                .context("decoding catalog root")?;
        let mut inner = CatalogInner {
            account_ids: root.account_ids.clone(),
            applied_seq: root.applied_seq,
            ids_high_water: root.ids_high_water.unwrap_or_default(),
            loaded_writer_version: Some(root.writer_version),
            // MED-1: the root's per-account high-water seeds dropped-id
            // memory before shards fold in (apply_shard merges max); an
            // empty map on a legacy-root-shaped body recovers from the
            // global high-water.
            account_ns_high_water: recover_legacy_account_ns_high_water(
                root.account_ns_high_water,
                root.ids_high_water
                    .map(|high_water| high_water.namespace_id),
                root.account_ids.keys().cloned().collect(),
            ),
            ..CatalogInner::default()
        };
        for (account, shard_bytes) in shards {
            let shard: CatalogShardRecord =
                crate::services::catalog_record_envelope::from_enveloped_slice(shard_bytes)
                    .with_context(|| format!("decoding catalog shard '{account}'"))?;
            if &shard.account != account {
                anyhow::bail!(
                    "catalog shard identity mismatch: log key '{account}' carries \
                     shard '{}'",
                    shard.account
                );
            }
            inner.apply_shard(shard);
        }
        Ok(Self {
            inner: RwLock::new(inner),
        })
    }

    /// TD-SPINE-2: adopt a newer **root** in place (cross-pod reload). The
    /// global registries are replaced wholesale (small, root-authoritative);
    /// the identity high-water merges monotonically (never lowers); the
    /// replay watermark advances to the publisher's.
    pub fn adopt_root_bytes(&self, root_bytes: &[u8]) -> Result<()> {
        let root: CatalogRootRecord =
            crate::services::catalog_record_envelope::from_enveloped_slice(root_bytes)
                .context("decoding catalog root")?;
        let mut inner = self.inner.write();
        inner.account_ids = root.account_ids;
        inner
            .ids_high_water
            .merge_max(root.ids_high_water.unwrap_or_default());
        inner.applied_seq = inner.applied_seq.max(root.applied_seq);
        inner.loaded_writer_version = Some(root.writer_version);
        // MED-1: fold the root's per-account high-water monotonically so a
        // reload never loses dropped-id memory.
        for (account, id) in root.account_ns_high_water {
            let entry = inner.account_ns_high_water.entry(account).or_insert(0);
            *entry = (*entry).max(id);
        }
        Ok(())
    }

    /// TD-SPINE-2: adopt one account's shard in place (cross-pod reload).
    /// Replaces EXACTLY that account's rows — every other account's rows,
    /// and the read-your-writes window for them, are untouched — and folds
    /// the shard's per-account namespace-id high-water monotonically.
    pub fn load_shard_bytes(&self, account: &str, shard_bytes: &[u8]) -> Result<()> {
        let shard: CatalogShardRecord =
            crate::services::catalog_record_envelope::from_enveloped_slice(shard_bytes)
                .with_context(|| format!("decoding catalog shard '{account}'"))?;
        if shard.account != account {
            anyhow::bail!(
                "catalog shard identity mismatch: log key '{account}' carries \
                 shard '{}'",
                shard.account
            );
        }
        let mut inner = self.inner.write();
        inner.remove_shard_rows(account);
        inner.apply_shard(shard);
        Ok(())
    }

    /// TD-PATH-2 4a: encode the durable high-water sidecar image (see
    /// [`IdentityHighWaterSidecar`]). Stamped with the current writer version
    /// and framed by the TD-SPINE-1 record envelope.
    pub fn to_identity_high_water_sidecar_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.inner.read();
        let sidecar = IdentityHighWaterSidecar {
            writer_version: Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER),
            high_water: inner.ids_high_water,
            account_ns_high_water: Some(inner.account_ns_high_water.clone()),
        };
        crate::services::catalog_record_envelope::to_enveloped_vec_named(&sidecar)
            .context("encoding identity high-water sidecar")
    }

    /// TD-PATH-2 4a: fold a sidecar image into state — monotone (`merge_max`,
    /// never lowers) — and return the sidecar's writer-version stamp so the
    /// caller can WARN on a lagging (pre-high-water) writer. Legacy
    /// (unframed) sidecar bodies decode unchanged (TD-SPINE-1 mixed-read).
    pub fn adopt_identity_high_water_sidecar_bytes(&self, bytes: &[u8]) -> Result<Option<u32>> {
        let sidecar: IdentityHighWaterSidecar =
            crate::services::catalog_record_envelope::from_enveloped_slice(bytes)
                .context("decoding identity high-water sidecar")?;
        let stamp = sidecar.writer_version;
        let mut inner = self.inner.write();
        inner.ids_high_water.merge_max(sidecar.high_water);
        if let Some(account_floors) = sidecar.account_ns_high_water {
            for (account, floor) in account_floors {
                let current = inner.account_ns_high_water.entry(account).or_default();
                *current = (*current).max(floor);
            }
        } else if inner.loaded_legacy_namespace_snapshot || inner.account_ns_high_water.is_empty() {
            // An unattributed legacy floor may belong to any account known to
            // the reconstructed catalog. Reserve it conservatively once;
            // subsequent sidecars preserve these account floors explicitly.
            let floor = inner.ids_high_water.namespace_id;
            if floor > 0 {
                let accounts: HashSet<_> = inner
                    .account_ids
                    .keys()
                    .cloned()
                    .chain(inner.account_ns_high_water.keys().cloned())
                    .chain(
                        inner
                            .namespaces
                            .values()
                            .map(|ns| shard_key_of_tenant(ns.tenant_id.as_deref())),
                    )
                    .chain(std::iter::once(DEFAULT_SHARD_KEY.to_string()))
                    .collect();
                for account in accounts {
                    let current = inner.account_ns_high_water.entry(account).or_default();
                    *current = (*current).max(floor);
                }
            }
        }
        Ok(stamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::canonical_wal::FramedTableWalAppender;
    use crate::services::record_store::TableWalAppender;

    fn ns(levels: &[&str]) -> CatalogNamespace {
        CatalogNamespace::new(levels.iter().map(|s| s.to_string()).collect())
    }

    fn tid(namespace: &[&str], name: &str) -> TableIdentifier {
        TableIdentifier::new(namespace.iter().map(|s| s.to_string()).collect(), name)
    }

    /// TD-CAT-7: `SystemCatalog` had no `get_namespace_by_object_id` override at
    /// all — the DEFAULT backend silently inherited the trait's `Ok(None)`, so a
    /// namespace object_id resolved to "no such namespace". Splitting the trait
    /// turned the omission into a compile error; this pins the answer.
    ///
    /// Teeth: make the accessor return `None` unconditionally and the `Some`
    /// arm fails.
    #[test]
    fn namespaces_resolve_from_their_object_id() {
        let state = SystemCatalogState::new();
        let mut sales = ns(&["sales"]);
        sales.object_id = Some(11);
        let mut returns = ns(&["returns"]);
        returns.object_id = Some(12);

        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(sales),
            },
        );
        state.apply_committed(
            2,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(returns),
            },
        );

        assert_eq!(
            state.get_namespace_by_object_id(11),
            Some(vec!["sales".to_string()])
        );
        assert_eq!(
            state.get_namespace_by_object_id(12),
            Some(vec!["returns".to_string()])
        );
        assert_eq!(
            state.get_namespace_by_object_id(13),
            None,
            "an unminted id stays None — the honest one"
        );
    }

    fn upsert_table_op(namespace: &[&str], name: &str) -> CanonicalOperation {
        CatalogDelta::UpsertTable {
            identifier: tid(namespace, name),
            schema: Box::new(CatalogTableSchema::new(name)),
        }
        .to_operation()
        .expect("encode upsert table")
    }

    /// Catalog mutations appended to a canonical WAL, then replayed from a fresh
    /// reopen, reconstruct the in-RAM authority exactly: namespace present, the
    /// surviving table present, the dropped table gone, `list_tables` correct,
    /// and `applied_seq` at the last folded LSN.
    #[tokio::test]
    async fn folds_catalog_mutations_from_wal() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("catalog.wal");

        {
            let appender = FramedTableWalAppender::open(&wal_path).await?;
            appender
                .append_operations(
                    vec![
                        CatalogDelta::UpsertNamespace {
                            namespace: Box::new(ns(&["sales"])),
                        }
                        .to_operation()?,
                        upsert_table_op(&["sales"], "orders"),
                        upsert_table_op(&["sales"], "returns"),
                        CatalogDelta::DropTable {
                            identifier: tid(&["sales"], "orders"),
                        }
                        .to_operation()?,
                    ],
                    None,
                )
                .await?;
        }

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let entries = reopened.read_entries().await?;
        let state = SystemCatalogState::from_wal_entries(&entries)?;

        assert!(state.namespace_exists(&["sales".to_string()]));
        assert!(!state.table_exists(&tid(&["sales"], "orders")));
        assert!(state.table_exists(&tid(&["sales"], "returns")));
        let mut tables: Vec<String> = state
            .list_tables(&["sales".to_string()])
            .into_iter()
            .map(|t| t.name)
            .collect();
        tables.sort();
        assert_eq!(tables, vec!["returns".to_string()]);
        assert_eq!(state.applied_seq(), 4);
        assert_eq!(
            state
                .get_table(&tid(&["sales"], "returns"))
                .map(|s| s.name.clone()),
            Some("returns".to_string())
        );
        Ok(())
    }

    /// TD-PATH-2: a dropped table's typed collection id and object ids stay
    /// reserved through WAL replay AND through a snapshot round trip (where
    /// its create delta is gone).
    #[test]
    fn dropped_identities_stay_reserved_through_replay_and_snapshot() -> Result<()> {
        let state = SystemCatalogState::new();
        let mut schema = CatalogTableSchema::new("newest");
        schema.object_id = Some(50);
        schema.stable_collection_id = Some(7);
        let deltas = [
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(ns(&["t"])),
            },
            CatalogDelta::UpsertTable {
                identifier: tid(&["t"], "newest"),
                schema: Box::new(schema),
            },
            CatalogDelta::DropTable {
                identifier: tid(&["t"], "newest"),
            },
        ];
        for (seq, delta) in deltas.into_iter().enumerate() {
            state.apply_committed(seq as u64 + 1, delta);
        }
        assert_eq!(state.max_collection_id_ever(), Some(7));
        assert_eq!(state.max_object_id(), Some(50));

        let restored = SystemCatalogState::from_snapshot_bytes(&state.to_snapshot_bytes()?)?;
        assert!(!restored.table_exists(&tid(&["t"], "newest")));
        assert_eq!(restored.max_collection_id_ever(), Some(7));
        assert_eq!(restored.max_object_id(), Some(50));
        Ok(())
    }

    /// TD-PATH-2 mixed-read safety: a snapshot written before the high-water
    /// existed decodes (live schemas bound the floor, as before), and an older
    /// binary decodes a new snapshot (it ignores the additive key).
    #[test]
    fn identity_high_water_is_mixed_read_safe() -> Result<()> {
        // The pre-TD-PATH-2 snapshot shape (every field is written in both
        // directions here, so the old `serde(default)`s are immaterial).
        #[derive(Serialize, Deserialize)]
        struct PreHighWaterSnapshot {
            account_ids: HashMap<String, u32>,
            namespaces: HashMap<Vec<String>, CatalogNamespace>,
            tables: HashMap<TableIdentifier, CatalogTableSchema>,
            statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
            applied_seq: u64,
        }
        let mut live = CatalogTableSchema::new("live");
        live.stable_collection_id = Some(3);
        let old = PreHighWaterSnapshot {
            account_ids: HashMap::new(),
            namespaces: HashMap::from([(vec!["t".to_string()], ns(&["t"]))]),
            tables: HashMap::from([(tid(&["t"], "live"), live)]),
            statistics: HashMap::new(),
            applied_seq: 9,
        };
        let restored = SystemCatalogState::from_snapshot_bytes(&rmp_serde::to_vec_named(&old)?)?;
        assert_eq!(restored.max_collection_id_ever(), Some(3));

        let new_bytes = restored.to_snapshot_bytes()?;
        // TD-SPINE-1: new bytes are envelope-framed. A literal pre-envelope
        // binary can no longer decode them (accepted pre-GA break, 2026-11-22
        // — the envelope's first byte `P` reads as fixint 80 to a bare rmp
        // reader). What this arm still pins is the property the envelope
        // exists to preserve: the BODY is unchanged field-named MessagePack,
        // so a reader that strips the frame (every post-SPINE-1 binary, via
        // the sniff) decodes it with the OLD struct — additive keys ignored,
        // semantics intact.
        let body = match crate::services::catalog_record_envelope::sniff_enveloped(&new_bytes)? {
            crate::services::catalog_record_envelope::Enveloped::Enveloped { body, .. } => body,
            crate::services::catalog_record_envelope::Enveloped::Legacy(body) => body,
        };
        let decoded_by_old: PreHighWaterSnapshot = rmp_serde::from_slice(body)?;
        assert_eq!(decoded_by_old.applied_seq, 9);
        assert!(
            decoded_by_old
                .namespaces
                .contains_key(&vec!["t".to_string()])
        );
        assert_eq!(
            decoded_by_old
                .tables
                .get(&tid(&["t"], "live"))
                .and_then(|schema| schema.stable_collection_id),
            Some(3)
        );
        Ok(())
    }

    /// TD-PATH-2 4c: the writer-version stamp rides the snapshot — a stamped
    /// snapshot decodes back as `Some(Some(2))` (no lag WARN), an unstamped
    /// (legacy) one as `Some(None)` (the WARN condition), and WAL-only
    /// construction is `None` (nothing adopted, nothing to WARN about).
    #[test]
    fn snapshot_writer_version_stamp_is_visible_on_adoption() -> Result<()> {
        // Legacy shape: no `writer_version` key at all.
        #[derive(Serialize, Deserialize)]
        struct UnstampedSnapshot {
            account_ids: HashMap<String, u32>,
            namespaces: HashMap<Vec<String>, CatalogNamespace>,
            tables: HashMap<TableIdentifier, CatalogTableSchema>,
            statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
            applied_seq: u64,
        }
        let legacy = UnstampedSnapshot {
            account_ids: HashMap::new(),
            namespaces: HashMap::new(),
            tables: HashMap::new(),
            statistics: HashMap::new(),
            applied_seq: 1,
        };
        let adopted = SystemCatalogState::from_snapshot_bytes(&rmp_serde::to_vec_named(&legacy)?)?;
        assert_eq!(adopted.loaded_writer_version(), Some(None));
        assert!(proximadb_catalog::identity_writer_version_lags(None));

        // Current writer: stamped, no lag.
        let stamped = SystemCatalogState::new();
        let round = SystemCatalogState::from_snapshot_bytes(&stamped.to_snapshot_bytes()?)?;
        assert_eq!(
            round.loaded_writer_version(),
            Some(Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER))
        );
        assert!(!proximadb_catalog::identity_writer_version_lags(Some(
            proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER
        )));

        // WAL-only construction: nothing was adopted.
        assert_eq!(SystemCatalogState::new().loaded_writer_version(), None);
        Ok(())
    }

    /// TD-PATH-2 4a: the sidecar round-trips the floor, adoption is monotone
    /// (a lower image never lowers it), and the sidecar carries the stamp.
    #[test]
    fn identity_high_water_sidecar_round_trips_and_adoption_is_monotone() -> Result<()> {
        let state = SystemCatalogState::new();
        let mut schema = CatalogTableSchema::new("newest");
        schema.object_id = Some(50);
        schema.stable_collection_id = Some(7);
        schema.stable_namespace_id = Some(3);
        state.apply_committed(
            1,
            CatalogDelta::UpsertTable {
                identifier: tid(&["t"], "newest"),
                schema: Box::new(schema),
            },
        );
        let bytes = state.to_identity_high_water_sidecar_bytes()?;

        // Adoption into an EMPTY state restores the floor dimensions...
        let fresh = SystemCatalogState::new();
        let stamp = fresh.adopt_identity_high_water_sidecar_bytes(&bytes)?;
        assert_eq!(
            stamp,
            Some(proximadb_catalog::IDENTITY_WRITER_VERSION_HIGH_WATER)
        );
        assert_eq!(fresh.max_collection_id_ever(), Some(7));
        assert_eq!(fresh.max_object_id(), Some(50));
        assert_eq!(fresh.max_namespace_id_ever(), Some(3));

        // ...and a STALE (older) sidecar can never lower them.
        let mut schema_low = CatalogTableSchema::new("older");
        schema_low.object_id = Some(9);
        schema_low.stable_collection_id = Some(2);
        let lagging = SystemCatalogState::new();
        lagging.apply_committed(
            1,
            CatalogDelta::UpsertTable {
                identifier: tid(&["t"], "older"),
                schema: Box::new(schema_low),
            },
        );
        let lagging_bytes = lagging.to_identity_high_water_sidecar_bytes()?;
        fresh.adopt_identity_high_water_sidecar_bytes(&lagging_bytes)?;
        assert_eq!(fresh.max_collection_id_ever(), Some(7));
        assert_eq!(fresh.max_object_id(), Some(50));
        Ok(())
    }

    /// Dropping a namespace evicts its child tables.
    #[test]
    fn drop_namespace_cascades_to_tables() {
        let state = SystemCatalogState::new();
        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(ns(&["app"])),
            },
        );
        state.apply_committed(
            2,
            CatalogDelta::UpsertTable {
                identifier: tid(&["app"], "events"),
                schema: Box::new(CatalogTableSchema::new("events")),
            },
        );
        assert!(state.table_exists(&tid(&["app"], "events")));

        state.apply_committed(
            3,
            CatalogDelta::DropNamespace {
                levels: vec!["app".to_string()],
            },
        );
        assert!(!state.namespace_exists(&["app".to_string()]));
        assert!(!state.table_exists(&tid(&["app"], "events")));
        assert!(state.list_tables(&["app".to_string()]).is_empty());
        assert_eq!(state.applied_seq(), 3);
    }

    /// Replay is idempotent: re-folding the same entries does not change state.
    #[tokio::test]
    async fn replay_is_idempotent_on_sequence_number() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("catalog.wal");
        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(vec![upsert_table_op(&["s"], "t")], None)
            .await?;
        let entries = appender.read_entries().await?;

        let state = SystemCatalogState::from_wal_entries(&entries)?;
        state.replay(&entries)?; // second fold of the same slice
        assert_eq!(state.applied_seq(), 1);
        assert!(state.table_exists(&tid(&["s"], "t")));
        assert_eq!(state.list_tables(&["s".to_string()]).len(), 1);
        Ok(())
    }

    /// Non-catalog entries interleaved in the WAL are ignored by the catalog
    /// fold and do not advance the catalog watermark.
    #[test]
    fn ignores_non_catalog_entries() -> Result<()> {
        let entries = vec![
            CanonicalWalEntry::new(1, upsert_table_op(&["s"], "t"), None),
            CanonicalWalEntry::new(
                2,
                CanonicalOperation::CdcBarrier {
                    barrier_sequence: 2,
                    events: Vec::new(),
                },
                None,
            ),
        ];
        let state = SystemCatalogState::from_wal_entries(&entries)?;
        assert!(state.table_exists(&tid(&["s"], "t")));
        // Watermark stays at the last *catalog* mutation (seq 1).
        assert_eq!(state.applied_seq(), 1);
        Ok(())
    }

    /// The snapshot blob round-trips the full read surface and the watermark.
    #[test]
    fn snapshot_round_trips() -> Result<()> {
        let state = SystemCatalogState::new();
        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(ns(&["db", "public"])),
            },
        );
        state.apply_committed(
            2,
            CatalogDelta::UpsertTable {
                identifier: tid(&["db", "public"], "users"),
                schema: Box::new(
                    CatalogTableSchema::new("users")
                        .with_distance_metric(proximadb_distance_types::DistanceMetric::Cosine),
                ),
            },
        );

        let bytes = state.to_snapshot_bytes()?;
        let restored = SystemCatalogState::from_snapshot_bytes(&bytes)?;

        assert_eq!(restored.applied_seq(), 2);
        assert!(restored.namespace_exists(&["db".to_string(), "public".to_string()]));
        let table = restored
            .get_table(&tid(&["db", "public"], "users"))
            .expect("users table present after snapshot restore");
        assert_eq!(table.name, "users");
        assert_eq!(
            table.distance_metric,
            Some(proximadb_distance_types::DistanceMetric::Cosine)
        );
        assert_eq!(
            restored
                .list_tables(&["db".to_string(), "public".to_string()])
                .len(),
            1
        );
        Ok(())
    }

    /// A torn (partially written) trailing frame is truncated on reopen, so a
    /// crash mid-append loses only the uncommitted tail; all durably-appended
    /// catalog mutations still fold cleanly.
    #[tokio::test]
    async fn torn_tail_frame_is_dropped_committed_mutations_survive() -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("catalog.wal");

        let appender = FramedTableWalAppender::open(&wal_path).await?;
        appender
            .append_operations(
                vec![
                    CatalogDelta::UpsertNamespace {
                        namespace: Box::new(ns(&["s"])),
                    }
                    .to_operation()?,
                    upsert_table_op(&["s"], "committed"),
                ],
                None,
            )
            .await?;

        // Simulate a crash mid-append: a partial frame header at the tail.
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .await?;
        file.write_all(b"PXWA").await?;
        file.flush().await?;

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let entries = reopened.read_entries().await?;
        let state = SystemCatalogState::from_wal_entries(&entries)?;

        assert!(state.table_exists(&tid(&["s"], "committed")));
        assert_eq!(state.applied_seq(), 2);
        Ok(())
    }

    // ── TD-SPINE-1: record envelope (mixed-read at the byte level) ─────────

    /// Every writer-side entry point emits the envelope, and every reader
    /// side sniffs it back off: delta, snapshot and sidecar all round-trip
    /// through the framed bytes.
    #[test]
    fn every_persisted_record_is_envelope_framed() -> Result<()> {
        use crate::services::catalog_record_envelope as env;

        let delta = CatalogDelta::UpsertNamespace {
            namespace: Box::new(ns(&["s"])),
        };
        let op_bytes = match delta.to_operation()? {
            CanonicalOperation::CatalogMutation { bytes, .. } => bytes,
            other => panic!("unexpected op {other:?}"),
        };
        assert!(op_bytes.starts_with(env::CATALOG_RECORD_MAGIC));
        match CatalogDelta::decode(&op_bytes)? {
            CatalogDelta::UpsertNamespace { .. } => {}
            other => panic!("delta round trip mismatch: {}", other.routing_key()),
        }

        let state = SystemCatalogState::new();
        state.apply_committed(
            1,
            CatalogDelta::UpsertTable {
                identifier: tid(&["s"], "t"),
                schema: Box::new(CatalogTableSchema::new("t")),
            },
        );
        let snapshot_bytes = state.to_snapshot_bytes()?;
        assert!(snapshot_bytes.starts_with(env::CATALOG_RECORD_MAGIC));
        assert!(
            SystemCatalogState::from_snapshot_bytes(&snapshot_bytes)?
                .table_exists(&tid(&["s"], "t"))
        );

        let sidecar_bytes = state.to_identity_high_water_sidecar_bytes()?;
        assert!(sidecar_bytes.starts_with(env::CATALOG_RECORD_MAGIC));
        state
            .adopt_identity_high_water_sidecar_bytes(&sidecar_bytes)
            .map(|_| ())?;
        Ok(())
    }

    /// Mixed-read, legacy→new: a snapshot and sidecar written by the
    /// pre-envelope writer (bare field-named MessagePack) decode unchanged,
    /// and the tombstone/collection ids they carry survive. Extends the
    /// `identity_high_water_is_mixed_read_safe` shape to the envelope.
    #[test]
    fn legacy_unframed_snapshot_and_sidecar_still_decode() -> Result<()> {
        // Legacy snapshot: the pre-TD-PATH-2 shape re-encoded by hand —
        // bare bytes, no envelope.
        let mut live = CatalogTableSchema::new("live");
        live.stable_collection_id = Some(3);
        #[derive(Serialize, Deserialize)]
        struct LegacySnapshot {
            account_ids: HashMap<String, u32>,
            namespaces: HashMap<Vec<String>, CatalogNamespace>,
            tables: HashMap<TableIdentifier, CatalogTableSchema>,
            statistics: HashMap<TableIdentifier, CatalogTableStatistics>,
            applied_seq: u64,
        }
        let legacy = LegacySnapshot {
            account_ids: HashMap::from([("legacy".to_string(), 7)]),
            namespaces: HashMap::from([(vec!["t".to_string()], ns(&["t"]))]),
            tables: HashMap::from([(tid(&["t"], "live"), live)]),
            statistics: HashMap::new(),
            applied_seq: 9,
        };
        let legacy_bytes = rmp_serde::to_vec_named(&legacy)?;
        let restored = SystemCatalogState::from_snapshot_bytes(&legacy_bytes)?;
        assert!(restored.table_exists(&tid(&["t"], "live")));
        assert_eq!(restored.account_id_u32("legacy"), Some(7));
        assert_eq!(restored.applied_seq(), 9);
        assert_eq!(restored.max_collection_id_ever(), Some(3));

        // Legacy sidecar: bare bytes.
        #[derive(Serialize, Deserialize)]
        struct LegacySidecar {
            #[serde(default)]
            writer_version: Option<u32>,
            #[serde(default)]
            high_water: IdsHighWater,
        }
        let legacy_sidecar = LegacySidecar {
            writer_version: None,
            high_water: IdsHighWater {
                object_id: 50,
                collection_id: 7,
                namespace_id: 3,
            },
        };
        let fresh = SystemCatalogState::new();
        fresh
            .adopt_identity_high_water_sidecar_bytes(&rmp_serde::to_vec_named(&legacy_sidecar)?)?;
        assert_eq!(fresh.max_object_id(), Some(50));
        assert_eq!(fresh.max_collection_id_ever(), Some(7));
        assert_eq!(fresh.max_namespace_id_ever(), Some(3));
        Ok(())
    }

    /// Fail-closed at the record boundary: a snapshot whose envelope names a
    /// future format version (or codec) is a typed error, not a legacy
    /// mis-decode, not a panic. Same teeth for the sidecar.
    #[test]
    fn snapshot_with_unknown_envelope_version_fails_closed() -> Result<()> {
        use crate::services::catalog_record_envelope as env;

        let state = SystemCatalogState::new();
        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(ns(&["s"])),
            },
        );
        let mut bytes = state.to_snapshot_bytes()?;
        let version_at = env::CATALOG_RECORD_MAGIC.len();
        assert_eq!(bytes[version_at], env::ENVELOPE_FORMAT_VERSION);

        bytes[version_at] = env::ENVELOPE_FORMAT_VERSION + 1;
        let err = SystemCatalogState::from_snapshot_bytes(&bytes)
            .expect_err("future envelope version must fail closed");
        assert!(format!("{err:#}").contains("format version 2"));

        bytes[version_at] = env::ENVELOPE_FORMAT_VERSION;
        bytes[version_at + 1] = env::BODY_CODEC_MSGPACK_NAMED + 3;
        let err = SystemCatalogState::from_snapshot_bytes(&bytes)
            .expect_err("unknown body codec must fail closed");
        assert!(format!("{err:#}").contains("codec 3"));

        let mut sidecar = state.to_identity_high_water_sidecar_bytes()?;
        sidecar[version_at] = env::ENVELOPE_FORMAT_VERSION + 1;
        assert!(
            state
                .adopt_identity_high_water_sidecar_bytes(&sidecar)
                .is_err()
        );
        Ok(())
    }

    /// Mixed WAL chain: entries written by a pre-envelope binary (bare
    /// MessagePack payloads) replay alongside enveloped entries from the
    /// current writer — the sniff is per-entry, so an upgraded pod folds the
    /// whole chain.
    #[tokio::test]
    async fn mixed_legacy_and_enveloped_wal_chain_replays() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let wal_path = dir.path().join("catalog.wal");
        let appender = FramedTableWalAppender::open(&wal_path).await?;

        // Legacy entry: a pre-envelope writer's payload (bare bytes).
        let legacy_delta = CatalogDelta::UpsertNamespace {
            namespace: Box::new(ns(&["legacy"])),
        };
        let legacy_op = CanonicalOperation::CatalogMutation {
            key: legacy_delta.routing_key(),
            bytes: rmp_serde::to_vec_named(&legacy_delta)?,
        };
        // Enveloped entries: the current writer.
        let enveloped_ops = vec![
            CatalogDelta::UpsertTable {
                identifier: tid(&["legacy"], "from-old"),
                schema: Box::new(CatalogTableSchema::new("from-old")),
            }
            .to_operation()?,
            CatalogDelta::UpsertTable {
                identifier: tid(&["legacy"], "from-new"),
                schema: Box::new(CatalogTableSchema::new("from-new")),
            }
            .to_operation()?,
        ];
        appender
            .append_operations(
                std::iter::once(legacy_op).chain(enveloped_ops).collect(),
                None,
            )
            .await?;

        let reopened = FramedTableWalAppender::open(&wal_path).await?;
        let entries = reopened.read_entries().await?;
        let state = SystemCatalogState::from_wal_entries(&entries)?;
        assert!(state.namespace_exists(&["legacy".to_string()]));
        assert!(state.table_exists(&tid(&["legacy"], "from-old")));
        assert!(state.table_exists(&tid(&["legacy"], "from-new")));
        assert_eq!(state.applied_seq(), 3);
        Ok(())
    }

    // ── TD-SPINE-2: sharded snapshot model ─────────────────────────────────

    fn seeded_state() -> SystemCatalogState {
        let state = SystemCatalogState::new();
        let mut seq = 0;
        let mut apply = |state: &SystemCatalogState, delta: CatalogDelta, seq: &mut u64| {
            *seq += 1;
            state.apply_committed(*seq, delta);
        };
        apply(
            &state,
            CatalogDelta::UpsertAccount {
                account: "acme".into(),
                stable_id: 1,
            },
            &mut seq,
        );
        apply(
            &state,
            CatalogDelta::UpsertAccount {
                account: "globex".into(),
                stable_id: 2,
            },
            &mut seq,
        );
        // acme: namespace + two tables (one carrying a typed ns id).
        apply(
            &state,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(with_tenant(ns(&["acme", "sales"]), Some("acme"))),
            },
            &mut seq,
        );
        let mut orders = CatalogTableSchema::new("orders");
        orders.stable_namespace_id = Some(4);
        apply(
            &state,
            CatalogDelta::UpsertTable {
                identifier: tid(&["acme", "sales"], "orders"),
                schema: Box::new(orders),
            },
            &mut seq,
        );
        apply(
            &state,
            CatalogDelta::UpsertTable {
                identifier: tid(&["acme", "sales"], "returns"),
                schema: Box::new(CatalogTableSchema::new("returns")),
            },
            &mut seq,
        );
        // globex: namespace + one table.
        apply(
            &state,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(with_tenant(ns(&["globex", "ops"]), Some("globex"))),
            },
            &mut seq,
        );
        apply(
            &state,
            CatalogDelta::UpsertTable {
                identifier: tid(&["globex", "ops"], "metrics"),
                schema: Box::new(CatalogTableSchema::new("metrics")),
            },
            &mut seq,
        );
        // A tenant-less namespace lands in the DEFAULT shard.
        apply(
            &state,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(ns(&["public"])),
            },
            &mut seq,
        );
        apply(
            &state,
            CatalogDelta::UpsertTable {
                identifier: tid(&["public"], "shared"),
                schema: Box::new(CatalogTableSchema::new("shared")),
            },
            &mut seq,
        );
        state
    }

    fn with_tenant(mut namespace: CatalogNamespace, tenant: Option<&str>) -> CatalogNamespace {
        namespace.tenant_id = tenant.map(str::to_string);
        namespace
    }

    /// Shard partitioning: every row lands in exactly one shard, keyed by
    /// its namespace's tenant; tenant-less rows share the default shard; the
    /// account registry and global high-water live in the ROOT, not shards.
    #[test]
    fn shards_partition_rows_by_account_and_root_carries_globals() -> Result<()> {
        let state = seeded_state();
        assert_eq!(
            crate::services::system_catalog_state::shard_key_of_tenant(Some("acme")),
            "acme"
        );
        assert_eq!(
            crate::services::system_catalog_state::shard_key_of_tenant(None),
            DEFAULT_SHARD_KEY
        );

        let index = std::collections::BTreeMap::from([
            ("_default".to_string(), 3),
            ("acme".to_string(), 1),
            ("globex".to_string(), 2),
        ]);
        let root_bytes = state.to_root_bytes(&index)?;
        let (accounts, high_water, _stamp, applied, decoded_index) =
            SystemCatalogState::decode_catalog_root(&root_bytes)?;
        assert_eq!(accounts.get("acme"), Some(&1));
        assert_eq!(accounts.get("globex"), Some(&2));
        assert_eq!(high_water.namespace_id, 4, "global record keeps the max");
        assert_eq!(applied, state.applied_seq());
        assert_eq!(decoded_index, index);

        // Full assembly from root + shards reproduces the whole read surface.
        let shards: Vec<(String, Vec<u8>)> = ["_default", "acme", "globex"]
            .iter()
            .map(|account| {
                (
                    (*account).to_string(),
                    state.to_shard_bytes(account).unwrap(),
                )
            })
            .collect();
        let rebuilt = SystemCatalogState::from_root_and_shards(&root_bytes, &shards)?;
        for (namespace, table) in [
            (vec!["acme".to_string(), "sales".to_string()], "orders"),
            (vec!["acme".to_string(), "sales".to_string()], "returns"),
            (vec!["globex".to_string(), "ops".to_string()], "metrics"),
            (vec!["public".to_string()], "shared"),
        ] {
            assert!(
                rebuilt.table_exists(&TableIdentifier::new(namespace, table)),
                "{table} must survive the sharded round trip"
            );
        }
        assert_eq!(rebuilt.account_id_u32("globex"), Some(2));
        assert_eq!(rebuilt.max_namespace_id_ever_for("acme"), Some(4));
        assert_eq!(rebuilt.applied_seq(), state.applied_seq());
        // A shard body carrying a different account identity is rejected.
        let mislabeled = state.to_shard_bytes("acme")?;
        assert!(
            rebuilt.load_shard_bytes("globex", &mislabeled).is_err(),
            "shard identity mismatch must fail closed"
        );
        Ok(())
    }

    /// Per-shard adoption replaces EXACTLY that account's rows: adopting an
    /// emptied acme shard removes acme's tables while globex and the default
    /// shard are untouched.
    #[test]
    fn shard_adoption_replaces_only_that_accounts_rows() -> Result<()> {
        let state = seeded_state();
        let empty_acme = SystemCatalogState::new(); // no acme rows at all
        let empty_shard_bytes = empty_acme.to_shard_bytes("acme")?;

        state.load_shard_bytes("acme", &empty_shard_bytes)?;
        assert!(
            !state.table_exists(&tid(&["acme", "sales"], "orders")),
            "acme rows are retired by the emptied shard image"
        );
        assert!(state.namespace_exists(&["public".to_string()]));
        assert!(
            state.table_exists(&tid(&["globex", "ops"], "metrics")),
            "globex rows are untouched by acme's adoption"
        );
        assert!(
            state.table_exists(&tid(&["public"], "shared")),
            "default-shard rows are untouched"
        );
        // The per-account high-water SURVIVES the emptied shard (dropped ids
        // stay reserved) — `to_shard_bytes` carries it from state.
        assert_eq!(state.max_namespace_id_ever_for("acme"), Some(4));
        Ok(())
    }

    /// The per-account namespace-id high-water remembers DROPPED ids, and
    /// only within their own account: emptying acme's highest namespace
    /// keeps acme's floor while globex mints its own ids from its own floor.
    #[test]
    fn per_account_namespace_high_water_is_scoped_and_survives_drops() -> Result<()> {
        let state = SystemCatalogState::new();
        // The namespace record is what attributes its tables to the account
        // shard (the tenant id rides the namespace, matching the mint path).
        state.apply_committed(
            1,
            CatalogDelta::UpsertNamespace {
                namespace: Box::new(with_tenant(ns(&["acme", "sales"]), Some("acme"))),
            },
        );
        let mut acme_table = CatalogTableSchema::new("orders");
        acme_table.stable_namespace_id = Some(5);
        state.apply_committed(
            2,
            CatalogDelta::UpsertTable {
                identifier: tid(&["acme", "sales"], "orders"),
                schema: Box::new(acme_table),
            },
        );
        assert_eq!(state.max_namespace_id_ever_for("acme"), Some(5));
        assert_eq!(state.max_namespace_id_ever_for("globex"), None);

        // Drop the only row carrying the id: the floor must persist.
        state.apply_committed(
            3,
            CatalogDelta::DropTable {
                identifier: tid(&["acme", "sales"], "orders"),
            },
        );
        assert_eq!(
            state.max_namespace_id_ever_for("acme"),
            Some(5),
            "a dropped namespace id stays reserved per account"
        );
        // And the reserved id survives the sharded round trip (the shard
        // body carries the per-account high-water).
        let shard = state.to_shard_bytes("acme")?;
        let round = SystemCatalogState::new();
        round.load_shard_bytes("acme", &shard)?;
        assert_eq!(round.max_namespace_id_ever_for("acme"), Some(5));
        Ok(())
    }

    #[test]
    fn fieldless_snapshot_preserves_legacy_and_attributed_sidecar_floors() -> Result<()> {
        let mut alpha = ns(&["acme", "alpha"]);
        alpha.tenant_id = Some("acme".to_string());
        let mut live = CatalogTableSchema::new("live");
        live.stable_namespace_id = Some(1);
        let snapshot = CatalogSnapshot {
            account_ids: HashMap::from([("acme".into(), 1), ("emptied".into(), 2)]),
            namespaces: HashMap::from([(alpha.levels.clone(), alpha)]),
            tables: HashMap::from([(tid(&["acme", "alpha"], "live"), live)]),
            statistics: HashMap::new(),
            applied_seq: 5,
            ids_high_water: None,
            writer_version: None,
            account_ns_high_water: None,
        };
        let bytes = rmp_serde::to_vec_named(&snapshot)?;
        let read = SystemCatalogState::from_snapshot_bytes(&bytes)?;
        assert_eq!(read.max_namespace_id_ever_for("acme"), Some(1));
        let mut sidecar = IdentityHighWaterSidecar {
            writer_version: None,
            high_water: IdsHighWater {
                namespace_id: 5,
                ..IdsHighWater::default()
            },
            account_ns_high_water: None,
        };
        let legacy_bytes = rmp_serde::to_vec_named(&sidecar)?;
        read.adopt_identity_high_water_sidecar_bytes(&legacy_bytes)?;
        assert_eq!(read.max_namespace_id_ever_for("acme"), Some(5));
        assert_eq!(read.max_namespace_id_ever_for("emptied"), Some(5));
        assert_eq!(read.max_namespace_id_ever_for(DEFAULT_SHARD_KEY), Some(5));

        let recovered = SystemCatalogState::from_snapshot_bytes(&bytes)?;
        sidecar.account_ns_high_water =
            Some(HashMap::from([("acme".into(), 3), ("emptied".into(), 5)]));
        recovered.adopt_identity_high_water_sidecar_bytes(
            &crate::services::catalog_record_envelope::to_enveloped_vec_named(&sidecar)?,
        )?;
        assert_eq!(recovered.max_namespace_id_ever_for("acme"), Some(3));
        assert_eq!(recovered.max_namespace_id_ever_for("emptied"), Some(5));
        let recorded = recovered.to_identity_high_water_sidecar_bytes()?;
        let fresh = SystemCatalogState::from_snapshot_bytes(&bytes)?;
        fresh.adopt_identity_high_water_sidecar_bytes(&recorded)?;
        assert_eq!(fresh.max_namespace_id_ever_for("acme"), Some(3));
        assert_eq!(fresh.max_namespace_id_ever_for("emptied"), Some(5));
        Ok(())
    }
}
