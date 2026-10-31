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

    /// Encode this delta as a `CanonicalOperation::CatalogMutation` ready for
    /// [`TableWalAppender::append_operations`]. Uses field-named MessagePack so
    /// the additive-serde evolution of `CatalogTableSchema` stays compatible.
    pub fn to_operation(&self) -> Result<CanonicalOperation> {
        let bytes = rmp_serde::to_vec_named(self).context("encoding catalog delta")?;
        Ok(CanonicalOperation::CatalogMutation {
            key: self.routing_key(),
            bytes,
        })
    }

    /// Decode a delta from a `CatalogMutation` payload.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes).context("decoding catalog delta")
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
}

/// TD-PATH-2: monotonic identity high-water marks (see `CatalogInner`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct IdsHighWater {
    object_id: u64,
    collection_id: u32,
}

impl IdsHighWater {
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
    }

    fn fold_namespace(&mut self, namespace: &CatalogNamespace) {
        if let Some(id) = namespace.object_id {
            self.object_id = self.object_id.max(id);
        }
    }
}

impl CatalogInner {
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
    /// dropped collection's typed storage root is never re-issued.
    pub fn max_collection_id_ever(&self) -> Option<u32> {
        let guard = self.inner.read();
        let live = guard
            .tables
            .values()
            .filter_map(|schema| schema.stable_collection_id)
            .max();
        live.max(Some(guard.ids_high_water.collection_id).filter(|id| *id > 0))
    }

    /// The highest WAL sequence number folded in (replay watermark / snapshot
    /// cutover LSN).
    pub fn applied_seq(&self) -> u64 {
        self.inner.read().applied_seq
    }

    // --- snapshot (de)serialization (bounds Phase 3 replay) ---

    /// Serialize the full state to a snapshot blob (field-named MessagePack).
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
        };
        rmp_serde::to_vec_named(&snapshot).context("encoding catalog snapshot")
    }

    /// Reconstruct state from a snapshot blob, rebuilding the `ns_children`
    /// secondary index.
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
            rmp_serde::from_slice(bytes).context("decoding catalog snapshot")?;
        let mut ns_children: HashMap<Vec<String>, HashSet<String>> = HashMap::new();
        // Every namespace gets an entry (so list_tables on an empty namespace
        // returns an empty set, not "missing").
        for levels in snapshot.namespaces.keys() {
            ns_children.entry(levels.clone()).or_default();
        }
        let mut tables = HashMap::with_capacity(snapshot.tables.len());
        let mut object_ids = HashMap::with_capacity(snapshot.tables.len());
        let mut ids_high_water = snapshot.ids_high_water.unwrap_or_default();
        for namespace in snapshot.namespaces.values() {
            ids_high_water.fold_namespace(namespace);
        }
        for (id, schema) in snapshot.tables {
            ids_high_water.fold_table(&schema);
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
            applied_seq: snapshot.applied_seq,
            ids_high_water,
        })
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
        #[derive(Serialize, Deserialize)]
        struct PreHighWaterSnapshot {
            #[serde(default)]
            account_ids: HashMap<String, u32>,
            namespaces: HashMap<Vec<String>, CatalogNamespace>,
            tables: HashMap<TableIdentifier, CatalogTableSchema>,
            #[serde(default)]
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
        let decoded_by_old: PreHighWaterSnapshot = rmp_serde::from_slice(&new_bytes)?;
        assert_eq!(decoded_by_old.applied_seq, 9);
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
}
