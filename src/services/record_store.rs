//! Canonical table/record write boundary for SQL, document, graph, vector, and
//! observability facades.
//!
//! This module is the service-layer ownership boundary for cataloged
//! `ProximaRecord` mutations. Protocol and modality facades should lower into
//! this API rather than depending on vector-specific operation names. The first
//! implementation delegates to `VectorOps` as a compatibility adapter until the
//! WAL/storage spine exposes a direct table-record writer.

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
// FA-b (Phase 4): the required read context. Only referenced on read
// primitives' `#[cfg(feature = "abac-policy")]` trailing parameter, so the
// import is gated to match — the default build never touches the type.
#[cfg(feature = "abac-policy")]
use proximadb_abac::ReadContext;
// `SystemReadReason` is constructed at internal call sites (index-build scans)
// and record-store test call sites; feature-gated like `ReadContext`.
use futures::StreamExt;
#[cfg(feature = "abac-policy")]
use proximadb_abac::SystemReadReason;
use proximadb_block_format::{BlockCompression, BlockMode};
use proximadb_catalog::{
    CatalogPhysicalFormat, CatalogStorageLayout, CatalogStorageSpecialization, CatalogTableSchema,
    CatalogWorkloadProfile,
};
use proximadb_records::chunked::{RecordChunkStream, merge_newest_wins, once_then_chunks};
use proximadb_records::{
    ProximaRecord, ProximaTreeNode, RecordKey, RecordScanOptions, RecordScanPredicate,
    RecordStorage,
};
use proximadb_storage_common::object_store_bridge::{
    BridgeObjectPath as ObjectPath, ObjectStoreBridge,
};
use proximadb_storage_common::{
    CanonicalOpenTableFormat, CanonicalOperation, CanonicalWalEntry, ProjectionDirective,
    pax_block::{PAX_SEGMENT_EXT, PaxSegmentScanner, PaxSegmentWriter, ScanPredicate},
    proxima_arrow,
    ranged_segment::RangedSegmentReader,
};

use crate::metrics::consumption_metrics::{
    object_store_egress_locality, record_kou_bytes, record_object_store_op,
};
use crate::observability::io_trace;
use crate::services::operations::VectorOps;
use crate::services::operations::batch_result::OperationMetrics;
use crate::services::operations::vectors::{RichRecordGetRequest, RichSearchResult};
use crate::storage::tenant::context::TenantContext;

/// Logical mutation kind at the canonical table-record boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableRecordMutationKind {
    /// Insert only; callers should reject duplicates before or during write.
    Insert,
    /// Insert or replace the current visible version for a key.
    Upsert,
    /// Write a new version for an existing key.
    Update,
    /// Write a tombstone or close the current visible version.
    Delete,
    /// Replace the target table snapshot with a new result set.
    OverwriteSnapshot,
    /// Replace cataloged partitions with a new result set.
    ReplacePartitions,
    /// Merge source rows into target rows by key/predicate.
    Merge,
}

/// Canonical table-record mutation.
#[derive(Debug, Clone)]
pub struct TableRecordMutation {
    /// Mutation behavior requested by the facade/planner.
    pub kind: TableRecordMutationKind,
    /// Catalog-validated canonical record envelope.
    pub record: ProximaRecord,
}

impl TableRecordMutation {
    /// Create a mutation from a canonical record.
    pub fn new(kind: TableRecordMutationKind, record: ProximaRecord) -> Self {
        Self { kind, record }
    }
}

/// One `write_mutations` call captured during a transaction, replayed by
/// [`TableRecordStore::commit_atomic`] (ADR-095 D3).
#[derive(Debug, Clone)]
pub struct StagedTableWrite {
    pub table_schema: CatalogTableSchema,
    pub mutations: Vec<TableRecordMutation>,
    pub tenant_context: Option<TenantContext>,
}

/// Point lookup request against a cataloged table/collection.
#[derive(Debug, Clone)]
pub struct TableRecordGetRequest {
    /// xCatalog table or current compatibility collection identifier.
    pub table_id: String,
    /// Logical row/record key.
    pub key: String,
    /// Whether vector embeddings should be included.
    pub include_vector: bool,
    /// Whether scalar/document props should be included.
    pub include_props: bool,
}

/// Current point-lookup result shape used by SQL DML. This remains intentionally
/// facade-neutral even though the compatibility adapter sources it from
/// `RichSearchResult` today.
pub type TableRecordGetResponse = Option<RichSearchResult>;

/// Scan request against a cataloged table/collection.
#[derive(Debug, Clone, Default)]
pub struct TableRecordScanRequest {
    /// xCatalog table or current compatibility collection identifier.
    pub table_id: String,
    /// Maximum number of records to scan. `None` means unbounded.
    pub limit: Option<usize>,
    /// Whether vector embeddings should be included.
    pub include_vector: bool,
    /// Whether scalar/document props should be included.
    pub include_props: bool,
    /// Optional structured predicate for **block/row-group pushdown** on stores
    /// that read PAX segments from object storage (e.g.
    /// [`ObjectStoreVectorRecordStore`]). When set, the store skips segment
    /// blocks the filter provably excludes before fetching their bodies; the
    /// row-exact `predicate` closure of `scan_records_filtered` is still applied
    /// on top. `None` ⇒ whole-segment read, preserving prior caller behavior.
    pub filter: Option<proximadb_filter_expression::FilterExpression>,
}

/// Scan result shape for canonical table-record reads.
pub type TableRecordScanResponse = Vec<ProximaRecord>;

/// Append-only canonical WAL boundary for table-record mutations.
///
/// This trait deliberately accepts canonical operations rather than protocol or
/// vector-specific requests. Production implementations can persist to the
/// shared WAL/log/manifest stack; tests and embedded adapters can provide
/// in-memory implementations without changing DML semantics.
#[async_trait]
pub trait TableWalAppender: Send + Sync {
    /// Append canonical operations and return the committed WAL entries with
    /// assigned sequence numbers.
    async fn append_operations(
        &self,
        operations: Vec<CanonicalOperation>,
        tenant_id: Option<String>,
    ) -> Result<Vec<CanonicalWalEntry>>;

    /// Append `operations` as ONE all-or-nothing unit (ADR-095 D3, TD-PGOLTP-1):
    /// after a crash either every entry is recovered or none is. Each operation
    /// carries its own tenant, because one transaction may span tenants'
    /// tables. Sequence numbers are contiguous and returned in input order.
    ///
    /// The default refuses rather than degrading to a non-atomic append, so a
    /// caller that needs atomicity can never silently get a partial commit.
    async fn append_operations_atomic(
        &self,
        operations: Vec<(CanonicalOperation, Option<String>)>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        let _ = operations;
        Err(anyhow!(
            "this WAL appender does not support atomic multi-operation appends"
        ))
    }

    /// Read every canonical WAL entry this appender has durably recorded, oldest first.
    /// The default returns nothing (appenders without a readable log opt out); the framed
    /// appender overrides it. Used by the CDC change-feed to surface row-level changes.
    async fn read_all_entries(&self) -> Result<Vec<CanonicalWalEntry>> {
        Ok(Vec::new())
    }

    /// Entries with `sequence_number > since` that may belong to
    /// `collection_id`, oldest first (callers still filter by collection). The
    /// default reads everything; a segmented WAL (P2.1) reads only the segments
    /// whose LSN summary can contain a match, so a change-feed read stops
    /// costing the whole log.
    async fn read_entries_since(
        &self,
        collection_id: &str,
        since: u64,
    ) -> Result<Vec<CanonicalWalEntry>> {
        let _ = collection_id;
        Ok(self
            .read_all_entries()
            .await?
            .into_iter()
            .filter(|e| e.sequence_number > since)
            .collect())
    }

    /// Highest sequence_number assigned to an entry for `collection_id`, when the
    /// appender tracks per-collection watermarks. `None` (default) = unknown ⇒ the
    /// caller must scan. Used by the TD-OLAP-17 fast-path in `read_changes_since`:
    /// if the collection's high-water has not advanced past a reader's `since_lsn`,
    /// no entry can satisfy `sequence_number > since_lsn` ⇒ empty, skipping the
    /// O(WAL-size) `read_all_entries` materialization that the OLAP delta merge
    /// otherwise pays per table, per query (the measured DF table-OPEN floor —
    /// TD-OLAP-17). The shared canonical WAL is tagged per entry, so the watermark
    /// is per-collection, not global.
    fn collection_max_sequence_for(&self, _collection_id: &str) -> Option<u64> {
        None
    }

    /// `true` when the appender maintains a watermark for EVERY collection it
    /// has ever written (on append and on recovery), so a `None` from
    /// [`collection_max_sequence_for`](Self::collection_max_sequence_for) means
    /// "this collection has no entries" rather than "unknown". Lets the change
    /// feed skip the O(WAL) decode for never-written keys — e.g. the
    /// object_id-keyed half of the ADR-031 dual-read, which otherwise decoded the
    /// whole WAL per table per OLAP query (measured: ~99.7% of DataFusion query
    /// time at TPC-DS SF 0.02).
    fn tracks_collection_high_water(&self) -> bool {
        false
    }

    /// Highest sequence number assigned so far, when the appender knows it.
    /// `None` (default) ⇒ unknown, and an in-flight write reservation then
    /// blocks WAL truncation entirely (TD-SEAM-12).
    fn last_sequence(&self) -> Option<u64> {
        None
    }

    /// First sequence still readable after WAL truncation; `0` (default) when
    /// nothing was ever truncated. A change-feed read asking for history below
    /// it fails with [`WalHistoryTruncated`] instead of returning a gap.
    ///
    /// [`WalHistoryTruncated`]: crate::services::canonical_wal::WalHistoryTruncated
    fn retained_from(&self) -> u64 {
        0
    }

    /// Pin the history after a snapshot LSN chosen from the head (see
    /// [`FramedTableWalAppender::pin_history`]). `None` (default): the
    /// appender never truncates, so there is nothing to pin.
    ///
    /// [`FramedTableWalAppender::pin_history`]: crate::services::FramedTableWalAppender::pin_history
    async fn pin_history(
        &self,
        snapshot_lsn: &(dyn Fn(u64) -> u64 + Send + Sync),
    ) -> Result<Option<(u64, crate::services::canonical_wal::WalHistoryPin)>> {
        let _ = snapshot_lsn;
        Ok(None)
    }
}

/// Writes whose WAL entries are durable but not yet applied to their partition
/// (TD-SEAM-12). Each holds a reservation at the lowest sequence it can carry
/// — the appender's `last_sequence() + 1` read BEFORE its append — so a
/// truncation floor taken while it is in flight can never pass it, even though
/// no partition shows its rows yet.
#[derive(Debug, Default)]
struct InflightWrites {
    reservations: parking_lot::Mutex<std::collections::BTreeMap<u64, usize>>,
}

/// Releases its reservation on drop, so every exit path — success, error,
/// cancellation — clears it.
struct InflightReservation<'a> {
    writes: &'a InflightWrites,
    floor: u64,
}

impl InflightWrites {
    fn reserve(&self, floor: u64) -> InflightReservation<'_> {
        *self.reservations.lock().entry(floor).or_insert(0) += 1;
        InflightReservation {
            writes: self,
            floor,
        }
    }

    fn floor(&self) -> Option<u64> {
        self.reservations.lock().keys().next().copied()
    }
}

impl Drop for InflightReservation<'_> {
    fn drop(&mut self) {
        let mut reservations = self.writes.reservations.lock();
        if let Some(count) = reservations.get_mut(&self.floor) {
            *count -= 1;
            if *count == 0 {
                reservations.remove(&self.floor);
            }
        }
    }
}

/// Physical writer route selected from xCatalog table metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableRecordStoreRoute {
    /// The general relational/HTAP/OLAP route.
    ///
    /// TD-USUB-9 renamed this from `ParquetIcebergStorage`, which asserted a
    /// behaviour that does not happen: the slot is a plain
    /// `Arc<dyn TableRecordStore>`, and the pgwire wiring fills it with the
    /// **canonical WAL store** — no Parquet is written on INSERT, and no Iceberg
    /// manifest is involved. Parquet/Iceberg publication is a separate, explicit
    /// `ALTER TABLE … MATERIALIZE` step. The route selects a *destination slot*,
    /// not a physical format, so the name now says only that.
    RelationalStorage,
    /// Specialized target: PAX block format for high-performance Vector Search / ANN.
    PaxVectorStorage,
    /// Temporary compatibility route through the old vector operations facade.
    LegacyVectorCompatibility,
}

impl TableRecordStoreRoute {
    /// Select a writer route from cataloged workload and storage metadata.
    pub fn for_schema(schema: &CatalogTableSchema) -> Self {
        match (schema.workload_profile, schema.storage_specialization) {
            (_, CatalogStorageSpecialization::VectorAnn) => {
                // Vector-specialized storage implies we want the high-performance PAX layout
                Self::PaxVectorStorage
            }
            (CatalogWorkloadProfile::Vector, _) => {
                // Legacy fallback for vector workloads that haven't migrated to the new specialized flag
                Self::LegacyVectorCompatibility
            }
            (_, CatalogStorageSpecialization::LsmWriteOptimized) => Self::LegacyVectorCompatibility,
            _ => {
                // Default for relational, HTAP, and OLAP workloads
                Self::RelationalStorage
            }
        }
    }
}

/// Result of writing canonical table-record mutations.
#[derive(Debug, Clone)]
pub struct TableRecordWriteResult {
    /// Whether the write succeeded.
    pub success: bool,
    /// Logical record IDs written.
    pub record_ids: Vec<String>,
    /// Operation metrics.
    pub metrics: OperationMetrics,
    /// Error messages for failed records.
    pub errors: Vec<String>,
    /// Optional stable error code.
    pub error_code: Option<String>,
}

impl TableRecordWriteResult {
    pub(crate) fn success(record_ids: Vec<String>) -> Self {
        Self {
            success: true,
            record_ids,
            metrics: OperationMetrics::default(),
            errors: Vec::new(),
            error_code: None,
        }
    }

    pub(crate) fn failure(error: impl Into<String>, error_code: impl Into<String>) -> Self {
        Self {
            success: false,
            record_ids: Vec::new(),
            metrics: OperationMetrics::default(),
            errors: vec![error.into()],
            error_code: Some(error_code.into()),
        }
    }

    fn from_batch_result(
        result: crate::services::operations::batch_result::BatchOperationResult,
    ) -> Self {
        Self {
            success: result.success,
            record_ids: result.vector_ids,
            metrics: result.metrics,
            errors: result.errors,
            error_code: result.error_code,
        }
    }
}

fn record_id(record: &ProximaRecord) -> String {
    if record.oid.is_empty() {
        record.local_id.clone().unwrap_or_default()
    } else {
        record.oid.clone()
    }
}

#[allow(dead_code)] // pending wiring
fn primary_layout(schema: &CatalogTableSchema) -> Option<&CatalogStorageLayout> {
    schema
        .storage_layouts
        .iter()
        .rev()
        .find(|layout| layout.name == "primary")
        .or_else(|| schema.storage_layouts.first())
}

#[allow(dead_code)] // pending wiring
fn normalize_object_path_prefix(location: &str) -> String {
    let without_scheme = location
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(location);
    without_scheme.trim_matches('/').to_string()
}

/// Encode a partition identity component (tenant id or collection key) as one
/// path segment, INJECTIVELY: `[A-Za-z0-9.-]` pass through and every other
/// byte (including `_` itself) becomes `_XX` hex. Two distinct tables must
/// never share a spill prefix — `sanitize_object_path_segment` maps `a/b` and
/// `a_b` to the same string, which here would merge their rows.
fn spill_path_segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("_{byte:02X}"));
        }
    }
    // `.` and `..` alone would be path navigation, and an empty segment would
    // collapse a level; all three get an explicit marker instead.
    match out.as_str() {
        "" | "." | ".." => format!("_{out}_"),
        _ => out,
    }
}

fn sanitize_object_path_segment(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        "table".to_string()
    } else {
        sanitized
    }
}

#[allow(clippy::expect_used)] // DrPathBuilder::build is infallible for this internally-constructed namespace
pub(crate) fn object_store_write_base_path(
    schema: &CatalogTableSchema,
    tenant_context: Option<&TenantContext>,
) -> String {
    // TODO ADR-031 Phase 4d: thread the collection's pre-minted
    // `CollectionIdentity` (from `schema.stable_namespace_id` /
    // `stable_collection_id` + the account u32) into `DrPathBuilder::build_from_identity`
    // when `typed_paths_enabled()`, so Parquet exports land at the typed
    // account-rooted path (`accounts/{base62}/…/`). Today this builds the legacy
    // string-resolved path (mixed-read-safe; env gate OFF).
    let tenant_id = tenant_context
        .map(|tc| tc.tenant_id.as_str())
        .unwrap_or("default_tenant");
    // ADR-074 S1 (C.2): source the namespace id from the identity context
    // (threaded via the `From<&MiddlewareTenantContext>` bridge), not a hardcoded
    // storage-layer constant. `None` (single-tenant / not yet threaded) falls
    // back to the path contract's DEFAULT_NAMESPACE_ID — byte-identical to the
    // prior "ns_default", so paths are unchanged (mixed-read-safe).
    let namespace_id = tenant_context
        .and_then(|tc| tc.namespace_id.clone())
        .unwrap_or_else(|| {
            crate::storage::trait_components::path_resolver::DrPathBuilder::DEFAULT_NAMESPACE_ID
                .to_string()
        });
    let mock_namespace = proximadb_catalog::CatalogNamespace::new(vec!["default".into()])
        .with_tenant(tenant_id)
        .with_namespace_id(namespace_id);

    let dr_path = crate::storage::trait_components::path_resolver::DrPathBuilder::build(
        &mock_namespace,
        &schema.name,
    )
    .expect("DrPathBuilder failed to construct valid tenant-isolated path");

    dr_path.root_prefix()
}

fn mutation_kind_label(kind: TableRecordMutationKind) -> &'static str {
    match kind {
        TableRecordMutationKind::Insert => "insert",
        TableRecordMutationKind::Upsert => "upsert",
        TableRecordMutationKind::Update => "update",
        TableRecordMutationKind::Delete => "delete",
        TableRecordMutationKind::OverwriteSnapshot => "overwrite",
        TableRecordMutationKind::ReplacePartitions => "replace-partitions",
        TableRecordMutationKind::Merge => "merge",
    }
}

/// The shared name stem for an object-store mutation object:
/// `{table}-{kind}-{nanos}` — the per-format path builders below add their
/// directory prefix and extension.
fn mutation_object_stem(schema: &CatalogTableSchema, kind: TableRecordMutationKind) -> String {
    let table = sanitize_object_path_segment(&schema.name);
    let sequence = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{table}-{}-{sequence}", mutation_kind_label(kind))
}

fn object_store_parquet_mutation_path(
    schema: &CatalogTableSchema,
    kind: TableRecordMutationKind,
    tenant_context: Option<&TenantContext>,
) -> ObjectPath {
    let base = object_store_write_base_path(schema, tenant_context);
    ObjectPath::from(format!(
        "{base}data/{}.parquet",
        mutation_object_stem(schema, kind)
    ))
}

fn object_store_pax_segment_path(
    schema: &CatalogTableSchema,
    kind: TableRecordMutationKind,
    tenant_context: Option<&TenantContext>,
) -> ObjectPath {
    let base = object_store_write_base_path(schema, tenant_context);
    ObjectPath::from(format!(
        "{base}segments/{stem}{PAX_SEGMENT_EXT}",
        stem = mutation_object_stem(schema, kind)
    ))
}

fn temp_pax_segment_path(
    schema: &CatalogTableSchema,
    kind: TableRecordMutationKind,
) -> std::path::PathBuf {
    let table = sanitize_object_path_segment(&schema.name);
    let sequence = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir()
        .join("proximadb-object-store-vector")
        .join(format!(
            "{table}-{}-{sequence}{PAX_SEGMENT_EXT}",
            mutation_kind_label(kind)
        ))
}

fn embedding_count_for_records(records: &[ProximaRecord]) -> usize {
    records
        .iter()
        .map(|record| record.embeddings.len())
        .max()
        .unwrap_or(0)
}

/// Per-`RecordBatch` row cap for object-store read-back. The bridge materializes
/// whole objects in memory, so this only bounds the Arrow batch chunking.
const OBJECT_STORE_READ_BATCH_SIZE: usize = 4096;

/// Recover the logical record key (`oid`) from a read-back record's `props`.
///
/// The schema-less `write_records_to_parquet` path does not persist `oid`
/// separately — it is the cataloged primary-key column value(s) in `props`. This
/// projects the record's leaf props into a [`CatalogRow`] and delegates to the
/// canonical [`CatalogRow::primary_key_string`] so the recovered key byte-matches
/// the one `CatalogRow::to_proxima_record` wrote (same `stable_value_string`
/// encoding + unit separator). Returns an empty string when the PK is
/// absent/non-scalar (the catalog builder errors → no fabricated key).
fn reconstruct_oid(record: &ProximaRecord, schema: &CatalogTableSchema) -> String {
    let mut values = std::collections::HashMap::new();
    for (key, node) in &record.props {
        if let ProximaTreeNode::Value(value) = node {
            values.insert(key.clone(), value.clone());
        }
    }
    proximadb_catalog::relational::CatalogRow {
        table: schema.name.clone(),
        values,
    }
    .primary_key_string(schema)
    .ok()
    .flatten()
    .unwrap_or_default()
}

/// Turn one Arrow [`RecordBatch`] back into canonical [`ProximaRecord`]s via the
/// single canonical inverse converter
/// ([`proxima_arrow::record_batch_to_proxima_records`], the inverse of
/// `proxima_records_to_record_batch`), then stamp the catalog-derived identity:
/// `oid` recovered from the primary key and `variation_id` from the table name.
fn record_batch_to_records(batch: &RecordBatch, schema: &CatalogTableSchema) -> Vec<ProximaRecord> {
    let mut records = proxima_arrow::record_batch_to_proxima_records(batch);
    for record in &mut records {
        record.oid = reconstruct_oid(record, schema);
        record.variation_id = Some(schema.name.clone());
    }
    records
}

/// List object keys under `prefix` via the canonical base-aware bridge seam
/// ([`ObjectStoreBridge::list_objects`]), keeping only those that end with
/// `suffix` (e.g. `.parquet` / `.pax`). The bridge returns paths that are
/// directly consumable by its read methods, so this is correct for both
/// empty-base and base-prefixed (cloud) deployments.
async fn list_objects_with_suffix(
    bridge: &Arc<dyn ObjectStoreBridge>,
    prefix: &ObjectPath,
    suffix: &str,
) -> Result<Vec<ObjectPath>> {
    let mut paths: Vec<ObjectPath> = bridge
        .list_objects(prefix)
        .await
        .map_err(|err| anyhow!("object-store list under '{prefix}' failed: {err}"))?
        .into_iter()
        .filter(|path| path.as_ref().ends_with(suffix))
        .collect();
    paths.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
    Ok(paths)
}

/// Canonical table-record store API.
/// One row-level change surfaced by the CDC change-feed (P2). `lsn` is the WAL sequence
/// number (monotonic), `op` is `"upsert"` or `"delete"`, `key` is the canonical OID, and
/// `props` carries the after-image (JSON) for upserts. Serializable for the REST surface.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChangeRow {
    pub lsn: u64,
    pub op: String,
    pub collection: String,
    pub key: String,
    pub props: Option<serde_json::Value>,
}

/// A table snapshot's position in the canonical WAL (ADR-025): every change
/// at or below `lsn` is visible to a scan started after the capture, and the
/// OLAP read-merge replays every change above it. While `pin` is held the WAL
/// keeps every entry above `lsn` (TD-SEAM-12); drop it once `lsn` is recorded
/// in the catalog, whose published snapshots then hold the truncation floor.
#[derive(Debug)]
pub struct SnapshotLsn {
    pub lsn: u64,
    pub pin: Option<crate::services::canonical_wal::WalHistoryPin>,
}

/// The pre-truncation snapshot LSN: the table's newest change, or `0`. Sound
/// only for a store whose WAL never truncates.
async fn snapshot_lsn_from_change_feed<S: TableRecordStore + ?Sized>(
    store: &S,
    collection_id: &str,
) -> SnapshotLsn {
    let lsn = store
        .read_changes_since(collection_id, 0)
        .await
        .map(|changes| changes.iter().map(|c| c.lsn).max().unwrap_or(0))
        .unwrap_or(0);
    SnapshotLsn { lsn, pin: None }
}

/// Build a [`ChangeRow`] for `collection_id` from a canonical WAL entry, or `None`
/// if the entry is not an upsert/delete for that collection. Shared by the
/// tenant-agnostic and tenant-scoped change-feeds so they stay in lock-step.
fn change_row_from_entry(
    entry: &proximadb_storage_common::wal_entry::CanonicalWalEntry,
    collection_id: &str,
) -> Option<ChangeRow> {
    match &entry.operation {
        CanonicalOperation::RecordUpsert {
            collection_id: c,
            record,
            ..
        } if c == collection_id => Some(ChangeRow {
            lsn: entry.sequence_number,
            op: "upsert".to_string(),
            collection: c.clone(),
            key: record.oid.clone(),
            props: serde_json::to_value(&record.props).ok(),
        }),
        CanonicalOperation::RecordDelete {
            collection_id: c,
            oid,
            ..
        } if c == collection_id => Some(ChangeRow {
            lsn: entry.sequence_number,
            op: "delete".to_string(),
            collection: c.clone(),
            key: oid.clone(),
            props: None,
        }),
        _ => None,
    }
}

#[cfg(feature = "abac-policy")]
pub mod fa_b_bypass_does_not_compile {
    //! FA-b compile-time guard (Phase 4).
    //!
    //! Asserts that a client read which omits the required `ReadContext`
    //! argument does not compile. The carrier module is cfg-gated, so the
    //! `compile_fail` doctest runs **only** when the `read_context` parameter
    //! exists; under the default (feature-off) build the parameter is absent and
    //! a 3-argument call would compile, which would make a bare `compile_fail`
    //! doctest pass *vacuously*. Gating the carrier makes the assertion
    //! meaningful.
    //!
    //! ```compile_fail
    //! use proximadb::services::record_store::TableRecordStore;
    //!
    //! // FA-b: omitting the 4th `read_context` argument → E0061. Bypass is a
    //! // compile error, not a code-review convention.
    //! fn no_read_context<S: TableRecordStore>(store: &S) {
    //!     let _ = store.get_by_key(todo!(), todo!(), todo!());
    //! }
    //! ```
}

#[async_trait]
pub trait TableRecordStore: Send + Sync {
    /// Write catalog-validated record mutations.
    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult>;

    /// Drop ALL record state for the table (the record-tier half of
    /// `DROP TABLE`; the catalog entry is the caller's concern).
    ///
    /// SQL semantics: a table recreated after a drop starts EMPTY. SQL-row
    /// record oids are the PK text by design, so without this purge a
    /// recreated table would collide with (or silently serve) the dropped
    /// rows on its first re-INSERT/scan. Restart-safety contract: an
    /// implementation that appends record operations to the canonical WAL
    /// must also append the `RecordPartitionDrop` marker so replay voids the
    /// partition too. Default: warn-and-continue — stores without drop
    /// support say so rather than silently keeping dropped rows.
    async fn drop_table_records(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        let _ = tenant_context;
        tracing::warn!(
            table = %table_schema.name,
            "record store does not support drop_table_records; a table recreated with the same name may see the dropped rows"
        );
        Ok(())
    }

    /// Make every batch durable as ONE all-or-nothing unit, then apply them in
    /// order (ADR-095 D3, TD-PGOLTP-1). Each batch is what one `write_mutations`
    /// call staged during a transaction; order matters (INSERT then UPDATE of the
    /// same key). Implementations must re-validate INSERT-absent / UPDATE-exists
    /// against current state in staged order before making anything durable.
    ///
    /// The default refuses: a store that cannot commit atomically must never
    /// silently degrade to per-batch writes.
    async fn commit_atomic(
        &self,
        batches: Vec<StagedTableWrite>,
    ) -> Result<Vec<TableRecordWriteResult>> {
        let tables: Vec<&str> = batches
            .iter()
            .map(|batch| batch.table_schema.name.as_str())
            .collect();
        Err(crate::services::txn_staging::AtomicCommitUnsupported(format!(
            "atomic transaction commit is not supported by the record store for table(s) {tables:?}"
        ))
        .into())
    }

    /// Get the current visible record for a key.
    ///
    /// `read_context` (present only under `abac-policy`) is the FA-b required
    /// authorization context: a client read resolves `AuthorizedReadContext`;
    /// an internal read (FK resolution, index build, recovery) supplies
    /// `ReadContext::system(reason, origin)`. A call that omits it does not
    /// compile when the feature is on — ABAC bypass is a compile error.
    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse>;

    /// Scan current visible records for a cataloged table.
    ///
    /// Implementations that only support point operations may return an error.
    /// Query and DML source readers use this method only after route selection
    /// chooses a native/catalog-table read path.
    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        #[cfg(feature = "abac-policy")]
        let _ = read_context;
        let _ = (request, tenant_context);
        Err(anyhow!(
            "TableRecordStore for '{}' does not support catalog-table scans yet",
            table_schema.name
        ))
    }

    /// A flush-consistent snapshot of the table's partition for the streaming
    /// Layer A provider (ADR-097 P3.1), or `Ok(None)` when this store cannot
    /// offer one — the caller then keeps its materialized scan. Taking it
    /// reads no segment.
    ///
    /// Rows are NOT filtered: the reader applies table membership, liveness
    /// and the caller's row access after resolving newest-wins.
    async fn layer_a_snapshot(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<Option<crate::services::layer_a_snapshot::LayerASnapshot>> {
        #[cfg(feature = "abac-policy")]
        let _ = read_context;
        let _ = (table_schema, tenant_context);
        Ok(None)
    }

    /// Every visible record of the table as an ordered chunk stream
    /// ([`RecordChunkStream`]: strictly ascending `oid`, each once, chunks of at
    /// most `chunk_rows`). `request.limit` is ignored. The stream owns its inputs
    /// (`'static`), so a bounded-memory consumer such as MATERIALIZE can drive it
    /// while it writes. Authorization (ABAC) stays with the caller, which applies
    /// its row filter per chunk.
    ///
    /// Default: correct but not bounded — `scan_records`, then sorted chunks.
    fn scan_records_chunked(
        self: Arc<Self>,
        table_schema: CatalogTableSchema,
        request: TableRecordScanRequest,
        chunk_rows: usize,
        tenant_context: Option<TenantContext>,
    ) -> RecordChunkStream<'static>
    where
        Self: 'static,
    {
        once_then_chunks(
            async move {
                self.scan_records(
                    &table_schema,
                    TableRecordScanRequest {
                        limit: None,
                        ..request
                    },
                    tenant_context.as_ref(),
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "scan_records_chunked"),
                )
                .await
            },
            chunk_rows,
        )
    }

    /// Scan with a row `predicate` pushed into the store (relational `WHERE`
    /// push-down). Returns up to `request.limit` records matching `predicate`;
    /// the store applies it, so callers must NOT re-filter afterward. This lets
    /// the backing store filter during iteration and stop at the limit instead
    /// of materializing the whole table.
    ///
    /// Default delegates to `scan_records` (materialize) then applies `predicate`
    /// in-memory — correct for any impl; hot stores override for early-stop.
    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let limit = request.limit.unwrap_or(usize::MAX);
        let mut req = request;
        req.limit = None;
        let mut all = self
            .scan_records(
                table_schema,
                req,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await?;
        let mut kept = 0usize;
        all.retain(|record| {
            if kept >= limit {
                return false;
            }
            let keep = predicate.is_none_or(|p| p(record));
            if keep {
                kept += 1;
            }
            keep
        });
        Ok(all)
    }

    /// TD-110 Slice C: detect a UNIQUE/PK conflict for a batch of candidate
    /// tuples against committed rows. `sets` carries one entry per unique column
    /// set with the (NULL-exempt, within-batch-deduped) candidate tuples the
    /// caller intends to insert; `primary_key` lets the PK column be read from
    /// `oid` rather than `props`.
    ///
    /// Default impl is a single short-circuiting `scan_records_filtered` per set
    /// (O(N)); index-backed stores (e.g. `DirectWalTableRecordStore`) override
    /// this with an O(1) probe.
    async fn check_unique_conflict(
        &self,
        table_schema: &CatalogTableSchema,
        table_id: &str,
        primary_key: Option<&str>,
        sets: &[UniqueCandidateSet],
        exclude_oids: &std::collections::HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<UniqueConflict>> {
        for set in sets {
            if set.candidates.is_empty() {
                continue;
            }
            let cols = set.columns.clone();
            let pk = primary_key.map(str::to_string);
            let wanted = set.candidates.clone();
            let excluded = exclude_oids.clone();
            let pred = move |existing: &ProximaRecord| {
                !excluded.contains(&existing.oid)
                    && record_unique_tuple(existing, &cols, pk.as_deref())
                        .is_some_and(|tuple| wanted.contains(&tuple))
            };
            let predicate: Option<&RecordScanPredicate<'_>> = Some(&pred);
            let hits = self
                .scan_records_filtered(
                    table_schema,
                    TableRecordScanRequest {
                        filter: None,
                        table_id: table_id.to_string(),
                        limit: Some(1),
                        include_vector: false,
                        include_props: true,
                    },
                    predicate,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(
                        SystemReadReason::ForeignKeyResolution,
                        "check_unique_conflict",
                    ),
                )
                .await?;
            if let Some(existing) = hits.first() {
                let tuple =
                    record_unique_tuple(existing, &set.columns, primary_key).unwrap_or_default();
                return Ok(Some(UniqueConflict {
                    columns: set.columns.clone(),
                    tuple,
                }));
            }
        }
        Ok(None)
    }

    /// TD-127: probe a single-column OLTP secondary index for the oids whose
    /// `column` value text is in `values`. Returns `None` when the store has no
    /// secondary index for `column` (or it is disabled) so the caller falls back
    /// to a scan; `Some(oids)` (possibly empty) when the index answered. The
    /// caller MUST still re-check each candidate against the full predicate — the
    /// index only narrows the candidate set.
    ///
    /// Default opts out (`None`); index-backed stores (e.g.
    /// `DirectWalTableRecordStore`) override this.
    async fn lookup_secondary(
        &self,
        table_schema: &CatalogTableSchema,
        column: &str,
        values: &std::collections::HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<String>>> {
        let _ = (table_schema, column, values, tenant_context);
        Ok(None)
    }

    /// CDC change-feed: return row-level changes for `collection_id` with WAL sequence
    /// number strictly greater than `since_lsn`, oldest first. The default returns nothing
    /// (stores without a readable change log opt out); the WAL-backed store overrides it.
    async fn read_changes_since(
        &self,
        collection_id: &str,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        let _ = (collection_id, since_lsn);
        Ok(Vec::new())
    }

    /// Tenant-scoped CDC change-feed: like [`read_changes_since`] but returns only
    /// changes belonging to `tenant` (matched against the WAL entry's `tenant_id`,
    /// with `None`/empty treated as the unscoped/default tenant). Required for the
    /// OLAP read-merge because the WAL `collection_id` is the bare table name (not
    /// tenant-unique), so two tenants sharing a table name would otherwise share
    /// the feed. The default returns nothing: a store that cannot filter its change
    /// log by tenant must not serve one (never the tenant-agnostic feed). The
    /// WAL-backed store and every store that routes to it override this.
    async fn read_changes_since_scoped(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        let _ = (collection_id, tenant, since_lsn);
        Ok(Vec::new())
    }

    /// Capture the snapshot LSN for MATERIALIZE before it scans the table
    /// (ADR-025). The default reads the table's newest change, which is sound
    /// only when the WAL never truncates; the WAL-backed store pins history
    /// instead, and every store that routes to it forwards this.
    async fn capture_snapshot_lsn(&self, collection_id: &str) -> Result<SnapshotLsn> {
        Ok(snapshot_lsn_from_change_feed(self, collection_id).await)
    }
}

/// One unique column set plus the candidate tuples a write intends to insert
/// (already NULL-exempt + deduped within the statement). See
/// [`TableRecordStore::check_unique_conflict`].
#[derive(Debug, Clone)]
pub struct UniqueCandidateSet {
    /// The unique constraint/index columns, in catalog order.
    pub columns: Vec<String>,
    /// Candidate tuple-reprs (from [`record_unique_tuple`]) to check.
    pub candidates: std::collections::HashSet<Vec<String>>,
}

/// A detected uniqueness violation: which column set, and the existing tuple.
#[derive(Debug, Clone)]
pub struct UniqueConflict {
    /// The violated unique constraint/index columns.
    pub columns: Vec<String>,
    /// The committed tuple that the candidate collided with.
    pub tuple: Vec<String>,
}

/// Canonical comparable-text rendering of a scalar `ProximaValue` for UNIQUE/PK
/// tuple comparison and predicate evaluation. Shared by the record store and
/// `DmlService` so the value seen at write time (index maintenance) matches the
/// value seen at check time.
///
/// ADR-031 O2: master gate for keying the WAL `collection_id` by the stable
/// `object_id` (default OFF). Mixed-read-safe — readers dual-read (name OR id), so
/// this rolls out per-writer/per-deployment without a flag-day cutover.
fn wal_object_id_keying_enabled() -> bool {
    std::env::var("PROXIMADB_WAL_OBJECT_ID_KEY")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
}

/// The WAL `collection_id` for a table: the stable `object_id` (decimal text) when
/// `object_id_keying` is on and the table carries one, else the bare name (legacy).
/// Pure (gate passed in) so it unit-tests without touching the process env.
fn wal_collection_id(table_schema: &CatalogTableSchema, object_id_keying: bool) -> String {
    if object_id_keying && let Some(oid) = table_schema.object_id {
        return oid.to_string();
    }
    table_schema.name.clone()
}

/// ADR-031 O3: the in-memory record partition / index key(s) for a table, kept in
/// LOCKSTEP with the WAL `collection_id` (same gate) so recovery — which routes
/// replayed entries by their stamped `collection_id` — lands records in the same
/// partition the live path reads. Returns `(primary, legacy)`:
/// - gate ON and the table carries an `object_id` ⇒ `primary` = the oid (decimal,
///   == [`wal_collection_id`]), `legacy` = `Some(name)`. Records written before the
///   flip live under the name key; the read path dual-reads and writes
///   migrate-on-touch (memtable deletes don't tombstone, so a stale legacy copy
///   would otherwise resurrect). Mixed-read-safe across the flip.
/// - otherwise ⇒ `(name, None)` — today's behavior, byte-identical, zero overhead.
///
/// `primary` always equals `wal_collection_id(table_schema, gate)`, so the memtable,
/// its indexes, and the WAL are keyed identically within a gate state.
pub(crate) fn memtable_partition_keys(
    table_schema: &CatalogTableSchema,
) -> (String, Option<String>) {
    let keying = wal_object_id_keying_enabled();
    let primary = wal_collection_id(table_schema, keying);
    let legacy = if keying && primary != table_schema.name {
        Some(table_schema.name.clone())
    } else {
        None
    };
    (primary, legacy)
}

/// Merge a primary + legacy partition scan during an oid cutover: the primary
/// (post-flip) partition wins on `oid` collisions, then the union is truncated to
/// `limit`. With the gate off there is no legacy partition and this is never called.
fn merge_scan_primary_wins(
    mut primary: Vec<ProximaRecord>,
    legacy: Vec<ProximaRecord>,
    limit: Option<usize>,
) -> Vec<ProximaRecord> {
    let seen: std::collections::HashSet<String> = primary.iter().map(|r| r.oid.clone()).collect();
    for record in legacy {
        if !seen.contains(&record.oid) {
            primary.push(record);
        }
    }
    if let Some(limit) = limit {
        primary.truncate(limit);
    }
    primary
}

pub(crate) fn proxima_value_to_unique_text(value: &proximadb_data_model::ProximaValue) -> String {
    use proximadb_data_model::ProximaValue;
    match value {
        ProximaValue::Boolean(value) => {
            if *value {
                "t".to_string()
            } else {
                "f".to_string()
            }
        }
        ProximaValue::Int8(value) => value.to_string(),
        ProximaValue::Int16(value) => value.to_string(),
        ProximaValue::Int32(value) => value.to_string(),
        ProximaValue::Int64(value) => value.to_string(),
        ProximaValue::UInt8(value) => value.to_string(),
        ProximaValue::UInt16(value) => value.to_string(),
        ProximaValue::UInt32(value) => value.to_string(),
        ProximaValue::UInt64(value) => value.to_string(),
        ProximaValue::Float16(value) => value.to_string(),
        ProximaValue::Float32(value) => value.to_string(),
        ProximaValue::Float64(value) => value.to_string(),
        ProximaValue::Decimal(value) => value.clone(),
        ProximaValue::String(value) | ProximaValue::Symbol(value) => value.clone(),
        ProximaValue::DenseVector(values) => values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
        ProximaValue::Null => String::new(),
        other => format!("{other:?}"),
    }
}

/// Render a record's value tuple for `columns` as comparable text for UNIQUE/PK
/// checks. Returns `None` when ANY column is NULL or absent — SQL UNIQUE permits
/// multiple NULL tuples, so such rows are exempt. The primary key is read from
/// `oid` (it is not stored in `props`); other columns must be scalar `props`.
pub(crate) fn record_unique_tuple(
    record: &ProximaRecord,
    columns: &[String],
    primary_key: Option<&str>,
) -> Option<Vec<String>> {
    use proximadb_data_model::ProximaValue;
    let mut tuple = Vec::with_capacity(columns.len());
    for column in columns {
        if primary_key.is_some_and(|pk| column.eq_ignore_ascii_case(pk)) {
            tuple.push(record.oid.clone());
            continue;
        }
        match record.props.get(column) {
            Some(ProximaTreeNode::Value(ProximaValue::Null)) | None => return None,
            Some(ProximaTreeNode::Value(value)) => {
                tuple.push(proxima_value_to_unique_text(value));
            }
            // Non-scalar (nested tree / array) — not a scalar unique key.
            Some(_) => return None,
        }
    }
    Some(tuple)
}

/// The column sets carrying a UNIQUE guarantee for a table — cataloged unique
/// indexes plus inline `UNIQUE (...)` constraints. Shared by `DmlService`
/// (candidate construction) and `DirectWalTableRecordStore` (index maintenance)
/// so both agree on exactly which sets are enforced. (TD-110 Slice C.)
pub(crate) fn schema_unique_column_sets(table_schema: &CatalogTableSchema) -> Vec<Vec<String>> {
    let mut sets: Vec<Vec<String>> = Vec::new();
    for index in &table_schema.relational_capabilities.unique_indexes {
        if !index.columns.is_empty() {
            sets.push(index.columns.clone());
        }
    }
    for constraint in &table_schema.relational_capabilities.constraints {
        if let proximadb_catalog::ColumnConstraint::Unique { columns } = constraint
            && !columns.is_empty()
        {
            sets.push(columns.clone());
        }
    }
    sets
}

/// The single-column primary key for a table (explicit, else conventional
/// `id`/`record_id`). The PK value lives in `oid`, not `props` — see
/// [`record_unique_tuple`].
pub(crate) fn schema_primary_key_column(table_schema: &CatalogTableSchema) -> Option<String> {
    table_schema.primary_key.first().cloned().or_else(|| {
        table_schema
            .columns
            .iter()
            .find(|column| column.name == "id" || column.name == "record_id")
            .map(|column| column.name.clone())
    })
}

/// All primary-key columns (composite-aware) — the `Vec` form of
/// [`schema_primary_key_column`]. Declared multi-column PKs are returned in
/// catalog order; otherwise the single id/record_id fallback is used. Shared so
/// the DML FK-existence/cascade paths and this store agree on the parent PK.
pub(crate) fn schema_primary_key_columns(table_schema: &CatalogTableSchema) -> Vec<String> {
    if !table_schema.primary_key.is_empty() {
        table_schema.primary_key.clone()
    } else {
        schema_primary_key_column(table_schema)
            .into_iter()
            .collect()
    }
}

/// Index-eligible scalar column types for an OLTP secondary (hash-equality)
/// index. Floats/decimals are excluded — equality on a float is semantically
/// fragile and `f32`/`f64` shortest-round-trip text can diverge for the same
/// decimal, which would make the probe text miss the indexed text — as are
/// non-scalar, temporal, and binary types; those columns fall back to the scan
/// path. (TD-127.)
fn is_secondary_indexable_type(data_type: &proximadb_data_model::ProximaType) -> bool {
    use proximadb_data_model::ProximaType;
    matches!(
        data_type,
        ProximaType::Boolean
            | ProximaType::Int8
            | ProximaType::Int16
            | ProximaType::Int32
            | ProximaType::Int64
            | ProximaType::UInt8
            | ProximaType::UInt16
            | ProximaType::UInt32
            | ProximaType::UInt64
            | ProximaType::String
            | ProximaType::Symbol
    )
}

/// The single-column non-unique secondary indexes declared on a table,
/// restricted to index-eligible scalar columns. The motivating consumer is the
/// code-graph workload (look up symbols by `name`/`file`, neither the PK). Unit
/// #1 indexes single columns only (composite is a follow-on). Shared by
/// `DmlService` (probe extraction) and `DirectWalTableRecordStore` (index
/// build/maintenance) so both agree on exactly which columns are indexed.
/// (TD-127.)
pub(crate) fn schema_secondary_index_columns(table_schema: &CatalogTableSchema) -> Vec<String> {
    let eligible: std::collections::HashMap<&str, &proximadb_data_model::ProximaType> =
        table_schema
            .columns
            .iter()
            .map(|column| (column.name.as_str(), &column.data_type))
            .collect();
    let mut columns: Vec<String> = Vec::new();
    for index in &table_schema.relational_capabilities.secondary_indexes {
        let [column] = index.columns.as_slice() else {
            continue; // single-column hash indexes only in unit #1
        };
        if columns.iter().any(|existing| existing == column) {
            continue;
        }
        if eligible
            .get(column.as_str())
            .is_some_and(|data_type| is_secondary_indexable_type(data_type))
        {
            columns.push(column.clone());
        }
    }
    columns
}

/// Render a record's scalar value for `column` as comparable text for the
/// secondary index — `None` when NULL/absent/non-scalar. Reuses
/// [`record_unique_tuple`] (with no primary key, so the value is read from
/// `props`) so the indexed text and the query-side probe text derive
/// identically through [`proxima_value_to_unique_text`]. (TD-127.)
fn record_secondary_text(record: &ProximaRecord, column: &str) -> Option<String> {
    let columns = [column.to_string()];
    record_unique_tuple(record, &columns, None).map(|mut tuple| tuple.remove(0))
}

/// TD-127 kill-switch: `PROXIMADB_SECONDARY_INDEX_DISABLE` forces the OLTP
/// secondary-index build/maintain/probe off (scan fallback), mirroring the
/// scan-index escape hatch (`PROXIMADB_SCAN_INDEX_DISABLE`).
fn secondary_index_disabled() -> bool {
    std::env::var_os("PROXIMADB_SECONDARY_INDEX_DISABLE").is_some()
}

/// Build the per-set candidate tuples for `records`, rejecting a tuple that
/// repeats within this statement (NULL tuples exempt). Shared by the INSERT /
/// UPDATE enforcement in `DmlService` and the INSERT-SELECT native executor so
/// every write path applies UNIQUE the same way. (TD-110.)
pub(crate) fn build_unique_candidate_sets(
    table_schema: &CatalogTableSchema,
    records: &[ProximaRecord],
    primary_key: Option<&str>,
) -> Result<Vec<UniqueCandidateSet>> {
    let mut candidate_sets = Vec::new();
    for columns in schema_unique_column_sets(table_schema) {
        let mut candidates: std::collections::HashSet<Vec<String>> =
            std::collections::HashSet::new();
        for record in records {
            let Some(tuple) = record_unique_tuple(record, &columns, primary_key) else {
                continue; // NULL/absent in the tuple → exempt
            };
            if !candidates.insert(tuple.clone()) {
                return Err(anyhow!(
                    "duplicate key value violates unique constraint on ({}) for table '{}': ({}) appears more than once in this statement",
                    columns.join(", "),
                    table_schema.name,
                    tuple.join(", ")
                ));
            }
        }
        if !candidates.is_empty() {
            candidate_sets.push(UniqueCandidateSet {
                columns,
                candidates,
            });
        }
    }
    Ok(candidate_sets)
}

/// xCatalog-routed table-record store.
///
/// The router makes the migration rule explicit: DML chooses a writer from
/// table/catalog definitions. Until the direct canonical writer exists, both
/// routes can delegate to the compatibility adapter, but callers no longer
/// depend on vector-specific APIs or naming.
pub struct CatalogRoutingTableRecordStore {
    iceberg_store: Arc<dyn TableRecordStore>,
    vector_store: Arc<dyn TableRecordStore>,
    legacy_vector_store: Arc<dyn TableRecordStore>,
}

impl CatalogRoutingTableRecordStore {
    /// Build a router with explicit object store and legacy implementations.
    pub fn new(
        iceberg_store: Arc<dyn TableRecordStore>,
        vector_store: Arc<dyn TableRecordStore>,
        legacy_vector_store: Arc<dyn TableRecordStore>,
    ) -> Self {
        Self {
            iceberg_store,
            vector_store,
            legacy_vector_store,
        }
    }

    /// Build the current migration router. The old vector adapter backs all
    /// routes until the direct object store writers are fully wired.
    pub fn with_vector_compatibility(vector_ops: Arc<VectorOps>) -> Self {
        let compatibility = Arc::new(VectorOpsTableRecordStore::new(vector_ops));
        Self::new(compatibility.clone(), compatibility.clone(), compatibility)
    }

    fn store_for_schema(&self, schema: &CatalogTableSchema) -> &Arc<dyn TableRecordStore> {
        match TableRecordStoreRoute::for_schema(schema) {
            TableRecordStoreRoute::RelationalStorage => &self.iceberg_store,
            TableRecordStoreRoute::PaxVectorStorage => &self.vector_store,
            TableRecordStoreRoute::LegacyVectorCompatibility => &self.legacy_vector_store,
        }
    }
}

#[async_trait]
impl TableRecordStore for CatalogRoutingTableRecordStore {
    async fn drop_table_records(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        self.store_for_schema(table_schema)
            .drop_table_records(table_schema, tenant_context)
            .await
    }

    /// One atomic unit needs one log: delegate only when every batch routes to
    /// the same underlying store; otherwise refuse (never split a commit).
    async fn commit_atomic(
        &self,
        batches: Vec<StagedTableWrite>,
    ) -> Result<Vec<TableRecordWriteResult>> {
        let Some(first) = batches.first() else {
            return Ok(Vec::new());
        };
        let target = self.store_for_schema(&first.table_schema).clone();
        // Compare data addresses: `Arc::ptr_eq` on `dyn` also compares vtables.
        // (`usize`, not a raw pointer, so the future stays `Send`.)
        let address = |store: &Arc<dyn TableRecordStore>| Arc::as_ptr(store) as *const () as usize;
        let target_address = address(&target);
        if let Some(other) = batches
            .iter()
            .find(|batch| address(self.store_for_schema(&batch.table_schema)) != target_address)
        {
            return Err(crate::services::txn_staging::AtomicCommitUnsupported(format!(
                "atomic transaction commit spans storage routes ('{}' and '{}' use different stores)",
                first.table_schema.name, other.table_schema.name
            ))
            .into());
        }
        target.commit_atomic(batches).await
    }

    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        self.store_for_schema(table_schema)
            .write_mutations(table_schema, mutations, tenant_context)
            .await
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        self.store_for_schema(table_schema)
            .get_by_key(
                table_schema,
                request,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        self.store_for_schema(table_schema)
            .scan_records(
                table_schema,
                request,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await
    }

    async fn layer_a_snapshot(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<Option<crate::services::layer_a_snapshot::LayerASnapshot>> {
        self.store_for_schema(table_schema)
            .layer_a_snapshot(
                table_schema,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await
    }

    fn scan_records_chunked(
        self: Arc<Self>,
        table_schema: CatalogTableSchema,
        request: TableRecordScanRequest,
        chunk_rows: usize,
        tenant_context: Option<TenantContext>,
    ) -> RecordChunkStream<'static> {
        self.store_for_schema(&table_schema)
            .clone()
            .scan_records_chunked(table_schema, request, chunk_rows, tenant_context)
    }

    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        self.store_for_schema(table_schema)
            .scan_records_filtered(
                table_schema,
                request,
                predicate,
                tenant_context,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await
    }

    /// TD-127: forward the secondary-index probe to the schema's routed store, so
    /// an index-backed route (the WAL-backed native store) is actually consulted
    /// instead of falling through to the `None` trait default (which would force
    /// every reader behind the router onto the scan path).
    async fn lookup_secondary(
        &self,
        table_schema: &CatalogTableSchema,
        column: &str,
        values: &std::collections::HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<String>>> {
        self.store_for_schema(table_schema)
            .lookup_secondary(table_schema, column, values, tenant_context)
            .await
    }

    /// CDC change-feed: relational changes live in the relational (iceberg) route — the
    /// WAL-backed store — so delegate there. Without this override the routing store would
    /// fall through to the empty trait default, hiding pgwire writes from the change-feed.
    async fn read_changes_since(
        &self,
        collection_id: &str,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        self.iceberg_store
            .read_changes_since(collection_id, since_lsn)
            .await
    }

    /// Forward the tenant-scoped feed too: the trait default would fall back to the
    /// tenant-agnostic `read_changes_since`, leaking another tenant's changes into
    /// the OLAP read-merge whenever this router is the store in use.
    async fn read_changes_since_scoped(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        self.iceberg_store
            .read_changes_since_scoped(collection_id, tenant, since_lsn)
            .await
    }

    async fn capture_snapshot_lsn(&self, collection_id: &str) -> Result<SnapshotLsn> {
        self.iceberg_store.capture_snapshot_lsn(collection_id).await
    }
}

/// Compatibility implementation backed by `VectorOps`.
///
/// This adapter is intentionally narrow: it lets pgwire and other facades
/// depend on `TableRecordStore` immediately while the underlying WAL/storage
/// writer is extracted from the vector service.
pub struct VectorOpsTableRecordStore {
    vector_ops: Arc<VectorOps>,
}

impl VectorOpsTableRecordStore {
    /// Wrap the current vector operations service as a table-record store.
    pub fn new(vector_ops: Arc<VectorOps>) -> Self {
        Self { vector_ops }
    }
}

#[async_trait]
impl TableRecordStore for VectorOpsTableRecordStore {
    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let records = mutations
            .into_iter()
            .map(|mutation| mutation.record)
            .collect::<Vec<_>>();
        let result = self
            .vector_ops
            .insert_records_with_tenant_context(&table_schema.name, records, tenant_context)
            .await?;
        Ok(TableRecordWriteResult::from_batch_result(result))
    }

    async fn get_by_key(
        &self,
        _table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        self.vector_ops
            .get_record_with_tenant_context(
                RichRecordGetRequest {
                    collection_id: request.table_id,
                    record_id: request.key,
                    include_vector: request.include_vector,
                    include_props: request.include_props,
                },
                tenant_context,
                proximadb_runtime::PortIdentity::anonymous(),
            )
            .await
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        self.vector_ops
            .scan_records_with_tenant_context(
                &table_schema.name,
                request.limit,
                request.include_vector,
                request.include_props,
                tenant_context,
            )
            .await
    }
}

/// Direct canonical table-record implementation over the shared
/// `proximadb_records::RecordStorage` contract.
///
/// This is the service-layer adapter for the target relational/document/graph
/// durable spine. It intentionally has no VectorOps dependency. Concrete WAL,
/// PAX, row-family, and projection stores can implement `RecordStorage` and be
/// exposed to SQL, REST/gRPC, Arrow Flight, and embedded facades through this
/// adapter.
pub struct RecordStorageTableRecordStore {
    storage: Arc<dyn RecordStorage>,
}

/// The scan options for a table scan request: the request's limit pushed
/// down, and the tenant's partition selected when a tenant context is
/// present — shared prologue of `RecordStorageTableRecordStore`'s scan
/// methods.
fn table_scan_options(
    request: &TableRecordScanRequest,
    tenant_context: Option<&TenantContext>,
) -> RecordScanOptions {
    let mut options = request
        .limit
        .map(RecordScanOptions::limit)
        .unwrap_or_else(RecordScanOptions::unbounded);
    if let Some(tenant_context) = tenant_context {
        options = options.with_tenant_id(tenant_context.tenant_id.clone());
    }
    options
}

/// Apply a scan request's limit and include flags to already-materialized
/// records — shared tail of the object-store `scan_records` impls (truncate
/// to the limit, then strip vectors and/or props as requested).
fn truncate_and_project_scan(records: &mut Vec<ProximaRecord>, request: &TableRecordScanRequest) {
    if let Some(limit) = request.limit {
        records.truncate(limit);
    }
    if !request.include_vector {
        for record in records.iter_mut() {
            record.embeddings.clear();
        }
    }
    if !request.include_props {
        for record in records.iter_mut() {
            record.props.clear();
        }
    }
}

impl RecordStorageTableRecordStore {
    /// Wrap a canonical record storage implementation.
    pub fn new(storage: Arc<dyn RecordStorage>) -> Self {
        Self { storage }
    }

    /// The ordered chunk scan of one record partition, with the same table
    /// membership (`variation_id`) and vector/prop stripping as `scan_records`.
    fn partition_chunks(
        storage: Arc<dyn RecordStorage>,
        table_name: String,
        request: &TableRecordScanRequest,
        chunk_rows: usize,
        tenant_id: Option<String>,
    ) -> RecordChunkStream<'static> {
        let mut options = RecordScanOptions::unbounded();
        if let Some(tenant_id) = tenant_id {
            options = options.with_tenant_id(tenant_id);
        }
        let (include_vector, include_props) = (request.include_vector, request.include_props);
        storage
            .scan_record_chunks(options, chunk_rows)
            .filter_map(move |chunk| {
                let chunk = chunk.map(|mut records| {
                    records.retain(|record| {
                        record
                            .variation_id
                            .as_deref()
                            .is_none_or(|variation| variation == table_name)
                    });
                    for record in &mut records {
                        if !include_vector {
                            record.embeddings.clear();
                        }
                        if !include_props {
                            record.props.clear();
                        }
                    }
                    records
                });
                futures::future::ready(match chunk {
                    Ok(records) if records.is_empty() => None,
                    other => Some(other),
                })
            })
            .boxed()
    }
}

#[async_trait]
impl TableRecordStore for RecordStorageTableRecordStore {
    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        _tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let mut record_ids = Vec::with_capacity(mutations.len());

        for mutation in mutations {
            let kind = mutation.kind;
            let record = mutation.record;
            let key = RecordKey::from(&record);

            match kind {
                TableRecordMutationKind::Insert => {
                    if self.storage.get_record(&key).await?.is_some() {
                        return Ok(TableRecordWriteResult::failure(
                            format!(
                                "Record '{}' already exists in table '{}'",
                                record.oid, table_schema.name
                            ),
                            "INSERT_CONFLICT",
                        ));
                    }
                    let written = self.storage.upsert_record(record).await?;
                    record_ids.push(written.oid);
                }
                TableRecordMutationKind::Upsert | TableRecordMutationKind::Update => {
                    let written = self.storage.upsert_record(record).await?;
                    record_ids.push(written.oid);
                }
                TableRecordMutationKind::Delete => {
                    self.storage.delete_record(&key).await?;
                    record_ids.push(record.oid);
                }
                TableRecordMutationKind::OverwriteSnapshot
                | TableRecordMutationKind::ReplacePartitions
                | TableRecordMutationKind::Merge => {
                    return Err(anyhow!(
                        "Mutation kind {:?} for table '{}' requires snapshot/merge commit support",
                        kind,
                        table_schema.name
                    ));
                }
            }
        }

        Ok(TableRecordWriteResult::success(record_ids))
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        _tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        let record = self
            .storage
            .get_record(&RecordKey::new(request.key.clone()))
            .await?;
        Ok(record.map(|record| {
            proxima_record_to_get_response(
                record,
                table_schema,
                request.include_vector,
                request.include_props,
            )
        }))
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let options = table_scan_options(&request, tenant_context);
        let mut records = self.storage.scan_records_with_options(options).await?;
        records.retain(|record| {
            record
                .variation_id
                .as_deref()
                .map(|variation| variation == table_schema.name)
                .unwrap_or(true)
        });
        truncate_and_project_scan(&mut records, &request);
        Ok(records)
    }

    fn scan_records_chunked(
        self: Arc<Self>,
        table_schema: CatalogTableSchema,
        request: TableRecordScanRequest,
        chunk_rows: usize,
        tenant_context: Option<TenantContext>,
    ) -> RecordChunkStream<'static> {
        Self::partition_chunks(
            self.storage.clone(),
            table_schema.name,
            &request,
            chunk_rows,
            tenant_context.map(|t| t.tenant_id),
        )
    }

    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let options = table_scan_options(&request, tenant_context);

        // Fold the table-membership (variation_id) check INTO the pushed-down
        // predicate so the store's early-stop counts only fully-matching rows —
        // a post-scan retain would under-return against the limit.
        let table_name = table_schema.name.as_str();
        let combined = |record: &ProximaRecord| {
            let belongs = record
                .variation_id
                .as_deref()
                .map(|variation| variation == table_name)
                .unwrap_or(true);
            belongs && predicate.is_none_or(|p| p(record))
        };
        let mut records = self
            .storage
            .scan_records_filtered(options, Some(&combined))
            .await?;

        truncate_and_project_scan(&mut records, &request);
        Ok(records)
    }
}

/// Direct Proxima-authoritative table writer.
///
/// This is the native OLTP/HTAP commit route for mutable Proxima-owned tables:
/// append canonical WAL operations first, then apply the visible record state
/// to the canonical `RecordStorage` row/delta spine. Layer-2 projections such
/// as PAX stripes, columnar blocks, HNSW, JSON, graph topology, and open-format
/// manifests are driven from WAL `ProjectionDirective`s and remain rebuildable.
/// TD-110 Slice C: in-memory UNIQUE/PK index for one unique column set —
/// `tuple-repr → owning oids`. A set may transiently hold >1 oid only across the
/// pre-existing uniqueness TOCTOU (also present in the scan path); steady state
/// is one oid per tuple.
#[derive(Default)]
struct UniqueSetIndex {
    columns: Vec<String>,
    tuple_to_oids: std::collections::HashMap<Vec<String>, std::collections::HashSet<String>>,
}

/// Per-table UNIQUE/PK index. Each oid self-tracks its current per-set tuple
/// (`oid_tuples`) so an update or delete can remove the OLD tuples without
/// re-reading storage. Built lazily on first check (scanning the WAL-rebuilt
/// current state) and maintained incrementally on every subsequent write.
#[derive(Default)]
struct TableUniqueIndex {
    /// One per unique column set, in `schema_unique_column_sets` order.
    sets: Vec<UniqueSetIndex>,
    /// oid → its current per-set tuple (`None` = NULL-exempt for that set).
    oid_tuples: std::collections::HashMap<String, Vec<Option<Vec<String>>>>,
}

impl TableUniqueIndex {
    fn with_sets(set_columns: &[Vec<String>]) -> Self {
        Self {
            sets: set_columns
                .iter()
                .map(|columns| UniqueSetIndex {
                    columns: columns.clone(),
                    tuple_to_oids: std::collections::HashMap::new(),
                })
                .collect(),
            oid_tuples: std::collections::HashMap::new(),
        }
    }

    /// Insert/update `record`: drop its previous per-set tuples (if any) then add
    /// its current ones. Uniform across INSERT/UPSERT/UPDATE.
    fn upsert(&mut self, record: &ProximaRecord, primary_key: Option<&str>) {
        self.remove_oid_tuples(&record.oid);
        let mut per_set = Vec::with_capacity(self.sets.len());
        for set in &mut self.sets {
            let tuple = record_unique_tuple(record, &set.columns, primary_key);
            if let Some(tuple) = &tuple {
                set.tuple_to_oids
                    .entry(tuple.clone())
                    .or_default()
                    .insert(record.oid.clone());
            }
            per_set.push(tuple);
        }
        self.oid_tuples.insert(record.oid.clone(), per_set);
    }

    /// Remove `oid` entirely (DELETE).
    fn delete(&mut self, oid: &str) {
        self.remove_oid_tuples(oid);
        self.oid_tuples.remove(oid);
    }

    /// Detach `oid`'s currently-indexed tuples from `tuple_to_oids` (shared by
    /// upsert's replace and delete). Leaves `oid_tuples[oid]` for the caller.
    fn remove_oid_tuples(&mut self, oid: &str) {
        let Some(previous) = self.oid_tuples.get(oid).cloned() else {
            return;
        };
        for (set, tuple) in self.sets.iter_mut().zip(previous.iter()) {
            if let Some(tuple) = tuple
                && let Some(oids) = set.tuple_to_oids.get_mut(tuple)
            {
                oids.remove(oid);
                if oids.is_empty() {
                    set.tuple_to_oids.remove(tuple);
                }
            }
        }
    }

    /// First candidate tuple in `candidates` that already exists for the set
    /// matching `columns`, owned by some oid NOT in `exclude_oids`, if any.
    /// `exclude_oids` lets an UPDATE ignore the rows it is itself rewriting (so a
    /// row keeping or vacating its own unique value is not a self-conflict).
    fn conflict(
        &self,
        columns: &[String],
        candidates: &std::collections::HashSet<Vec<String>>,
        exclude_oids: &std::collections::HashSet<String>,
    ) -> Option<Vec<String>> {
        let set = self.sets.iter().find(|set| set.columns == columns)?;
        candidates
            .iter()
            .find(|candidate| {
                set.tuple_to_oids
                    .get(*candidate)
                    .is_some_and(|owners| owners.iter().any(|oid| !exclude_oids.contains(oid)))
            })
            .cloned()
    }
}

/// TD-127: one column's hash secondary index — `value-text → owning oids`.
#[derive(Default)]
struct ColumnSecondaryIndex {
    column: String,
    value_to_oids: std::collections::HashMap<String, std::collections::HashSet<String>>,
}

/// Per-table non-unique secondary index over single columns. Each oid
/// self-tracks its current per-column value text (`oid_values`) so an update or
/// delete drops the OLD value without re-reading storage — the same maintenance
/// shape as [`TableUniqueIndex`]. Built lazily on first probe (scanning the
/// WAL-rebuilt current state) and maintained incrementally on every write.
#[derive(Default)]
struct TableSecondaryIndex {
    /// One per indexed column, in `schema_secondary_index_columns` order.
    columns: Vec<ColumnSecondaryIndex>,
    /// oid → its current per-column value text (`None` = NULL/absent for that
    /// column, so it is not indexed).
    oid_values: std::collections::HashMap<String, Vec<Option<String>>>,
}

impl TableSecondaryIndex {
    fn with_columns(columns: &[String]) -> Self {
        Self {
            columns: columns
                .iter()
                .map(|column| ColumnSecondaryIndex {
                    column: column.clone(),
                    value_to_oids: std::collections::HashMap::new(),
                })
                .collect(),
            oid_values: std::collections::HashMap::new(),
        }
    }

    /// Insert/update `record`: drop its previous per-column values (if any) then
    /// add its current ones. Uniform across INSERT/UPSERT/UPDATE.
    fn upsert(&mut self, record: &ProximaRecord) {
        self.remove_oid(&record.oid);
        let mut per_column = Vec::with_capacity(self.columns.len());
        for column in &mut self.columns {
            let text = record_secondary_text(record, &column.column);
            if let Some(text) = &text {
                column
                    .value_to_oids
                    .entry(text.clone())
                    .or_default()
                    .insert(record.oid.clone());
            }
            per_column.push(text);
        }
        self.oid_values.insert(record.oid.clone(), per_column);
    }

    /// Remove `oid` entirely (DELETE).
    fn delete(&mut self, oid: &str) {
        self.remove_oid(oid);
        self.oid_values.remove(oid);
    }

    /// Detach `oid`'s currently-indexed values from `value_to_oids` (shared by
    /// upsert's replace and delete). Leaves `oid_values[oid]` for the caller.
    fn remove_oid(&mut self, oid: &str) {
        let Some(previous) = self.oid_values.get(oid).cloned() else {
            return;
        };
        for (column, value) in self.columns.iter_mut().zip(previous.iter()) {
            if let Some(value) = value
                && let Some(oids) = column.value_to_oids.get_mut(value)
            {
                oids.remove(oid);
                if oids.is_empty() {
                    column.value_to_oids.remove(value);
                }
            }
        }
    }

    /// Union of oids whose `column` value text is in `values`. `None` when the
    /// column is not indexed by this table index (caller scans); `Some` (possibly
    /// empty) when the column is indexed.
    fn probe(
        &self,
        column: &str,
        values: &std::collections::HashSet<String>,
    ) -> Option<std::collections::HashSet<String>> {
        let index = self.columns.iter().find(|c| c.column == column)?;
        let mut oids = std::collections::HashSet::new();
        for value in values {
            if let Some(owners) = index.value_to_oids.get(value) {
                oids.extend(owners.iter().cloned());
            }
        }
        Some(oids)
    }
}

/// Catalog schemas the spill partitions encode with, keyed like the partitions
/// (`(tenant, collection key)`). Filled by `partitions_for` on every write —
/// the one place the schema is in hand — and read by a partition at each flush
/// through its [`SpillSchemaSource`](crate::services::record_spill::SpillSchemaSource).
///
/// Boot seeding fills a second, per-collection map from the catalog before WAL
/// replay, so a partition replay creates encodes with its schema before any
/// write arrives. The catalog does not record which tenant wrote a table, and
/// two tenants may share a table name, so a collection key that maps to
/// different schemas is recorded as ambiguous and gets no fallback.
#[derive(Default)]
struct SchemaHints {
    by_partition:
        parking_lot::RwLock<std::collections::HashMap<(String, String), Arc<CatalogTableSchema>>>,
    /// Seeded at boot; `None` = ambiguous (several distinct schemas).
    by_collection:
        parking_lot::RwLock<std::collections::HashMap<String, Option<Arc<CatalogTableSchema>>>>,
}

fn same_columns(a: &CatalogTableSchema, b: &CatalogTableSchema) -> bool {
    a.columns.len() == b.columns.len()
        && a.columns.iter().zip(&b.columns).all(|(x, y)| {
            x.name == y.name && x.data_type == y.data_type && x.is_deleted == y.is_deleted
        })
}

impl SchemaHints {
    /// Record `schema` for the partition, replacing a stale one (an `ALTER
    /// TABLE`); an unchanged schema costs a read-lock and a column compare.
    fn note(&self, tenant_id: &str, collection: &str, schema: &CatalogTableSchema) {
        let key = (tenant_id.to_string(), collection.to_string());
        if self
            .by_partition
            .read()
            .get(&key)
            .is_some_and(|held| same_columns(held, schema))
        {
            return;
        }
        self.by_partition
            .write()
            .insert(key, Arc::new(schema.clone()));
    }

    /// Boot seeding: record `schema` as the fallback for `collection`, or mark
    /// the key ambiguous if a different schema already claims it.
    fn seed(&self, collection: &str, schema: &CatalogTableSchema) {
        let mut by_collection = self.by_collection.write();
        match by_collection.get_mut(collection) {
            None => {
                by_collection.insert(collection.to_string(), Some(Arc::new(schema.clone())));
            }
            Some(slot) => {
                if slot
                    .as_ref()
                    .is_some_and(|held| !same_columns(held, schema))
                {
                    *slot = None;
                }
            }
        }
    }

    fn get(&self, tenant_id: &str, collection: &str) -> Option<CatalogTableSchema> {
        if let Some(schema) = self
            .by_partition
            .read()
            .get(&(tenant_id.to_string(), collection.to_string()))
        {
            return Some((**schema).clone());
        }
        self.by_collection
            .read()
            .get(collection)
            .and_then(|slot| slot.as_ref())
            .map(|schema| (**schema).clone())
    }
}

pub struct DirectWalTableRecordStore {
    /// Per-(tenant_id, collection) record partitions, created on demand via
    /// `storage_factory`. TD-064: tenant + collection isolation is STRUCTURAL —
    /// selecting the partition by the catalog-resolved (tenant, collection)
    /// identity replaces per-record tenant/`variation_id` filtering on the hot
    /// path, and scopes oid point-lookups/insert-conflicts per (tenant, table).
    /// The empty tenant id (`""`) is just one more tenant key (single-tenant).
    partitions:
        parking_lot::RwLock<std::collections::HashMap<(String, String), Arc<dyn RecordStorage>>>,
    /// Factory for a fresh per-partition record store (default: in-memory
    /// memtable), given the partition's `(tenant_id, collection)` identity.
    ///
    /// TD-USUB-1 slice 2a: a DURABLE partition store must derive its own object
    /// path, and `DrPathBuilder` needs the partition identity to do that — which
    /// [`Self::partition`] has in hand and used to discard. Passing it is the
    /// stated prerequisite for wiring `SpillRecordStorage`.
    storage_factory: Arc<dyn Fn(&str, &str) -> Arc<dyn RecordStorage> + Send + Sync>,
    /// True when every partition key routes to ONE shared backing store
    /// ([`Self::new`]) rather than a per-partition one ([`Self::new_partitioned`]).
    ///
    /// Recorded at construction instead of probed. The previous check called the
    /// factory purely to `Arc::ptr_eq` its product against a stored partition —
    /// which only worked because every factory was side-effect-free. A factory
    /// that builds a durable store performs I/O and claims a path, so calling it
    /// speculatively to throw the result away is no longer merely wasteful.
    shared_storage: bool,
    wal_appender: Arc<dyn TableWalAppender>,
    /// Appended-but-not-yet-applied writes (TD-SEAM-12); see [`InflightWrites`].
    inflight: InflightWrites,
    /// TD-110 Slice C: UNIQUE/PK index keyed by `(tenant_id, collection)` so a
    /// table's UNIQUE/PK enforcement is per-tenant. Presence of a key == "index
    /// built". Lazily built on first `check_unique_conflict`, then maintained on
    /// every `write_mutations`.
    unique_index:
        parking_lot::RwLock<std::collections::HashMap<(String, String), TableUniqueIndex>>,
    /// TD-127: non-unique OLTP secondary index keyed by `(tenant_id, collection)`
    /// so a table's secondary indexes are per-tenant, mirroring `unique_index`.
    /// Presence of a key == "index built". Lazily built on first
    /// `lookup_secondary`, then maintained on every `write_mutations`.
    secondary_index:
        parking_lot::RwLock<std::collections::HashMap<(String, String), TableSecondaryIndex>>,
    /// ADR-095 review F2: set when an atomic commit's group frame became durable
    /// but applying it to memory failed. The in-memory state is then a partial
    /// transaction, so every read and write is refused until a restart replays
    /// the WAL — never serve a half-applied commit.
    poisoned: std::sync::atomic::AtomicBool,
    /// Set by [`Self::new_spilling`]: the catalog schemas its spill partitions
    /// encode with. `None` for stores whose partitions never encode a segment.
    schema_hints: Option<Arc<SchemaHints>>,
    /// Set by [`Self::new_spilling`]: the filesystem its spill partitions write
    /// segments with, which a [`Self::layer_a_snapshot`] reader must read them
    /// through. `None` for stores whose partitions never write a segment.
    segment_filesystem: Option<Arc<dyn crate::storage::persistence::filesystem::FileSystem>>,
}

/// Synthetic collections whose rows a service store appends straight to the
/// canonical WAL and replays itself: never table data.
fn is_store_owned_collection(collection_id: &str) -> bool {
    [
        crate::services::branch_base_store::BRANCH_BASES_COLLECTION_ID,
        crate::services::FUNCTIONS_COLLECTION_ID,
        crate::services::RANK_PROFILES_COLLECTION_ID,
    ]
    .contains(&collection_id)
}

impl DirectWalTableRecordStore {
    /// TD-OLAP-17 fast path: no entry for `collection_id` can exceed `since_lsn`
    /// when its watermark is at/below it, or when the appender tracks every
    /// collection and has none for this one.
    fn change_feed_is_empty(&self, collection_id: &str, since_lsn: u64) -> bool {
        match self.wal_appender.collection_max_sequence_for(collection_id) {
            Some(max) => max <= since_lsn,
            None => self.wal_appender.tracks_collection_high_water(),
        }
    }

    /// Create a direct writer that routes every `(tenant, collection)` partition
    /// to the single supplied `storage`. This is the non-isolated shape used by
    /// single-tenant unit tests and callers that intentionally share one store;
    /// production multi-tenant paths use [`Self::new_partitioned`].
    pub fn new(storage: Arc<dyn RecordStorage>, wal_appender: Arc<dyn TableWalAppender>) -> Self {
        Self::with_storage_factory_inner(
            wal_appender,
            Arc::new(move |_tenant_id, _collection| storage.clone()),
            true,
        )
    }

    /// Create a direct writer with per-(tenant, collection) partitions backed by
    /// in-memory memtables created on demand — the isolated production shape.
    pub fn new_partitioned(wal_appender: Arc<dyn TableWalAppender>) -> Self {
        Self::with_storage_factory(
            wal_appender,
            Arc::new(|_tenant_id, _collection| {
                Arc::new(crate::services::MemtableRecordStorage::new()) as Arc<dyn RecordStorage>
            }),
        )
    }

    /// Create a direct writer with per-(tenant, collection) partitions that spill
    /// to durable Parquet segments once resident rows exceed the configured bound
    /// (TD-USUB-1 slice 2a).
    ///
    /// **Off means the legacy TYPE.** With
    /// `PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT` off (`0`) this builds
    /// [`MemtableRecordStorage`] — the same store [`Self::new_partitioned`] builds —
    /// rather than a spill store configured never to flush. Mandate #8: with the
    /// marker absent, the shipped path is byte-identical to the legacy one, not
    /// merely equivalent-looking.
    ///
    /// `segment_base` maps a partition's `(tenant_id, collection)` to the object
    /// prefix its segments are written under. Callers build it with
    /// `DrPathBuilder`; this store never constructs a raw path
    /// (`object_store_write_base_path` is the relational precedent).
    ///
    /// **The derived path is within-run only.** `SpillRecordStorage` tracks its
    /// segment list in memory, and WAL replay — which is what repopulates a
    /// partition after a restart — reaches [`Self::partition`] carrying only the
    /// WAL entry's `tenant_id` and `collection_id`. `CanonicalWalEntry` has no
    /// namespace field, so recovery **cannot** re-derive a namespace-scoped
    /// prefix. That is why slice 2b (WAL truncation) requires catalog-registered
    /// segments first: truncation may only reference a segment whose location is
    /// recorded somewhere recovery can read it. See TD-USUB-1.
    /// The production constructor (TD-SEAM-3 / TD-USUB-1 item iii): with the
    /// spill gate (`PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT`) off (`0`) this is
    /// exactly [`Self::new_partitioned`]; on (the default), partitions spill to
    /// local Parquet segments under `spill_root`, beside the canonical WAL they
    /// recover with — `{spill_root}/{tenant}/{collection}`, each component
    /// encoded injectively ([`spill_path_segment`]).
    ///
    /// Local, not object storage, for now: the WAL itself is a local file, so
    /// segments next to it are durable exactly as long as it is, and recovery
    /// needs no catalog lookup to find them. Object-store placement waits for
    /// catalog registration of the prefix (TD-USUB-1 item ii).
    pub async fn open_for_server(
        wal_appender: Arc<dyn TableWalAppender>,
        spill_root: std::path::PathBuf,
    ) -> Result<Self> {
        if crate::services::record_spill::configured_flush_threshold().is_none() {
            // Spill turned off over a root an earlier run spilled into, AFTER
            // the WAL was truncated: rows flushed there may be in no retained
            // WAL entry, so a memtable store would serve a silent subset.
            // Refuse. With the history intact, replay rebuilds every row in
            // memory and turning the spill off is safe.
            if wal_appender.retained_from() > 1
                && crate::services::storage_format::spill_root_has_state(&spill_root)
            {
                anyhow::bail!(
                    "record spill is off ({}=0) but {} holds spill segments from an earlier \
                     run and the WAL below them was truncated, so their rows are in no \
                     retained WAL entry; turn the spill back on",
                    crate::services::record_spill::SPILL_MAX_RESIDENT_ENV,
                    spill_root.display()
                );
            }
            return Ok(Self::new_partitioned(wal_appender));
        }
        // An empty root would put segments at `/{tenant}/{collection}`.
        if spill_root.as_os_str().is_empty() {
            anyhow::bail!("record spill: the spill gate is set but no spill root was configured");
        }
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(
                crate::storage::persistence::filesystem::local::LocalConfig::default(),
            )
            .await
            .map_err(|e| anyhow!("record spill: local filesystem init failed: {e}"))?,
        );
        let root = spill_root
            .to_string_lossy()
            .trim_end_matches('/')
            .to_string();
        tracing::info!(
            spill_root = %root,
            "relational record spill enabled: partitions flush to local Parquet segments"
        );
        Ok(Self::new_spilling(
            wal_appender,
            filesystem,
            Arc::new(move |tenant_id: &str, collection: &str| {
                format!(
                    "{root}/{}/{}",
                    spill_path_segment(tenant_id),
                    spill_path_segment(collection)
                )
            }),
        ))
    }

    /// Flush every live partition's write buffer to durable segments
    /// ([`RecordStore::flush`]; a no-op for a memtable partition). Called on
    /// graceful shutdown AFTER the servers stop, so no write races it.
    ///
    /// Every partition is attempted even if one fails; the error names each
    /// failure. A failed flush loses nothing — the rows stay resident and the
    /// WAL, which is never truncated past an unflushed row, replays them.
    pub async fn flush_partitions(&self) -> Result<usize> {
        let partitions: Vec<((String, String), Arc<dyn RecordStorage>)> = self
            .partitions
            .read()
            .iter()
            .map(|(key, partition)| (key.clone(), partition.clone()))
            .collect();
        let mut failures = Vec::new();
        for ((tenant_id, collection), partition) in &partitions {
            if let Err(error) = partition.flush().await {
                failures.push(format!(
                    "(tenant '{tenant_id}', collection '{collection}'): {error:#}"
                ));
            }
        }
        if failures.is_empty() {
            Ok(partitions.len())
        } else {
            Err(anyhow!(
                "flushing {} of {} record partition(s) failed: {}",
                failures.len(),
                partitions.len(),
                failures.join("; ")
            ))
        }
    }

    pub fn new_spilling(
        wal_appender: Arc<dyn TableWalAppender>,
        filesystem: Arc<dyn crate::storage::persistence::filesystem::FileSystem>,
        segment_base: Arc<dyn Fn(&str, &str) -> String + Send + Sync>,
    ) -> Self {
        let hints = Arc::new(SchemaHints::default());
        let factory_hints = hints.clone();
        let segment_filesystem = filesystem.clone();
        let mut store = Self::with_storage_factory(
            wal_appender,
            Arc::new(move |tenant_id, collection| {
                // Each spill partition encodes with its table's catalog schema,
                // looked up at flush time so a later `ALTER TABLE` applies.
                let schema_source = {
                    let hints = factory_hints.clone();
                    let (tenant_id, collection) = (tenant_id.to_string(), collection.to_string());
                    crate::services::record_spill::SpillSchemaSource::new(move || {
                        hints.get(&tenant_id, &collection)
                    })
                };
                match crate::services::record_spill::configured_flush_threshold() {
                    // The untenanted partition (`tenant_key(None)`, the normal
                    // key on a single-tenant server) spills too: it persists its
                    // deletes stamped `""`, like its own rows, so its heap is
                    // bounded and its tombstones leave memory. See
                    // `SpillRecordStorage::untenanted` for why that is safe for
                    // exactly this partition.
                    Some(threshold) if tenant_id.is_empty() => Arc::new(
                        crate::services::SpillRecordStorage::with_flush_threshold(
                            filesystem.clone(),
                            segment_base(tenant_id, collection),
                            Some(threshold),
                        )
                        .untenanted()
                        .with_schema_source(schema_source),
                    )
                        as Arc<dyn RecordStorage>,
                    Some(threshold) => {
                        Arc::new(
                            crate::services::SpillRecordStorage::with_flush_threshold(
                                filesystem.clone(),
                                segment_base(tenant_id, collection),
                                Some(threshold),
                            )
                            // Required for a tombstone-only flush to stamp its
                            // synthesised record; the store refuses to persist a
                            // tombstone without it.
                            .with_tenant(tenant_id)
                            .with_schema_source(schema_source),
                        ) as Arc<dyn RecordStorage>
                    }
                    None => Arc::new(crate::services::MemtableRecordStorage::new())
                        as Arc<dyn RecordStorage>,
                }
            }),
        );
        store.schema_hints = Some(hints);
        store.segment_filesystem = Some(segment_filesystem);
        store
    }

    /// Create a direct writer with a custom per-partition storage factory. The
    /// factory receives the partition's `(tenant_id, collection)` identity so a
    /// durable store can derive its own path.
    ///
    /// # Invariant the caller must uphold
    ///
    /// **The factory MUST return a distinct store per `(tenant_id, collection)`.**
    /// A factory that hands back one shared `Arc` for every partition — what
    /// [`Self::new`] does internally — breaks `drop_table_records`: it would take
    /// the purge branch, append a `RecordPartitionDrop` WAL marker, evict the map
    /// entry, and then the next access would hand back that same shared store
    /// with every "dropped" row still in it. Silently, and the useless marker
    /// replays on every restart.
    ///
    /// [`Self::new`] is the supported way to share one store; it records that
    /// shape so `drop_table_records` can decline the purge and warn instead. This
    /// constructor cannot detect it — the old `Arc::ptr_eq` probe could, but only
    /// by calling the factory speculatively, which a durable factory must not be
    /// subjected to (it performs I/O and claims an object path).
    pub fn with_storage_factory(
        wal_appender: Arc<dyn TableWalAppender>,
        storage_factory: Arc<dyn Fn(&str, &str) -> Arc<dyn RecordStorage> + Send + Sync>,
    ) -> Self {
        Self::with_storage_factory_inner(wal_appender, storage_factory, false)
    }

    fn with_storage_factory_inner(
        wal_appender: Arc<dyn TableWalAppender>,
        storage_factory: Arc<dyn Fn(&str, &str) -> Arc<dyn RecordStorage> + Send + Sync>,
        shared_storage: bool,
    ) -> Self {
        Self {
            partitions: parking_lot::RwLock::new(std::collections::HashMap::new()),
            storage_factory,
            shared_storage,
            wal_appender,
            inflight: InflightWrites::default(),
            unique_index: parking_lot::RwLock::new(std::collections::HashMap::new()),
            secondary_index: parking_lot::RwLock::new(std::collections::HashMap::new()),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            schema_hints: None,
            segment_filesystem: None,
        }
    }

    /// Reserve the lowest sequence the next append can be assigned, for the
    /// whole append-then-apply span (TD-SEAM-12). An appender that cannot say
    /// its last sequence reserves `0`, which blocks truncation outright.
    fn reserve_inflight(&self) -> InflightReservation<'_> {
        let floor = self
            .wal_appender
            .last_sequence()
            .map_or(0, |last| last.saturating_add(1));
        self.inflight.reserve(floor)
    }

    /// The lowest sequence any in-flight write (appended, not yet applied) may
    /// hold; `None` when no write is in flight. One input to the WAL
    /// truncation floor, alongside [`Self::min_unflushed_lsn`] (TD-SEAM-12).
    pub fn inflight_floor(&self) -> Option<u64> {
        self.inflight.floor()
    }

    /// Seed the spill partitions' catalog schemas before WAL replay (see
    /// `SchemaHints`), so a partition replay creates encodes typed columns
    /// instead of holding those rows resident until the first write. Returns
    /// how many collection keys were seeded; a no-op for stores whose
    /// partitions never encode a segment.
    pub fn seed_schema_hints<'a>(
        &self,
        schemas: impl IntoIterator<Item = &'a CatalogTableSchema>,
    ) -> usize {
        let Some(hints) = &self.schema_hints else {
            return 0;
        };
        let mut seeded = 0;
        for schema in schemas {
            let (primary, legacy) = memtable_partition_keys(schema);
            for key in std::iter::once(primary).chain(legacy) {
                hints.seed(&key, schema);
                seeded += 1;
            }
        }
        seeded
    }

    /// Fail loudly when the change feed is asked for history the WAL no longer
    /// holds, instead of returning the retained suffix as if it were complete.
    fn ensure_history_retained(&self, since_lsn: u64) -> Result<()> {
        let retained_from = self.wal_appender.retained_from();
        if retained_from > since_lsn.saturating_add(1) {
            return Err(crate::services::canonical_wal::WalHistoryTruncated {
                since: since_lsn,
                retained_from,
            }
            .into());
        }
        Ok(())
    }

    /// Refuse when a durable commit could not be applied (see `poisoned`).
    fn ensure_healthy(&self) -> Result<()> {
        if self.poisoned.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(anyhow!(
                "relational record store is poisoned: a committed transaction could not be \
                 applied in memory; restart the server to replay the WAL"
            ));
        }
        Ok(())
    }

    /// Resolve the tenant scope key from an optional tenant context.
    fn tenant_key(tenant_context: Option<&TenantContext>) -> String {
        tenant_context
            .map(|tenant| tenant.tenant_id.clone())
            .unwrap_or_default()
    }

    /// Select (creating on demand) the record partition for `(tenant_id, collection)`.
    fn partition(&self, tenant_id: &str, collection: &str) -> Arc<dyn RecordStorage> {
        let key = (tenant_id.to_string(), collection.to_string());
        if let Some(partition) = self.partitions.read().get(&key) {
            return partition.clone();
        }
        self.partitions
            .write()
            .entry(key)
            .or_insert_with(|| (self.storage_factory)(tenant_id, collection))
            .clone()
    }

    /// The lowest canonical-WAL LSN the relational record tier still needs: the
    /// minimum of every live partition's [`RecordStore::min_unflushed_lsn`].
    /// One input to WAL truncation (TD-SEAM-2 / TD-USUB-1 slice 2b); the other
    /// WAL consumers (change-feed readers, graph checkpoints) bound it too.
    ///
    /// * `Ok(None)`: no partition holds an unflushed row.
    /// * `Ok(Some(lsn))`: entries below `lsn` are durable elsewhere on this
    ///   tier's account.
    /// * `Err`: do not truncate. Any partition refusing makes the whole tier
    ///   refuse; there is no partial answer. A plain memtable partition refuses
    ///   (the trait default), so with the spill gate off — every partition a
    ///   memtable — this always refuses unless no partition exists.
    ///
    /// Shared-storage mode ([`Self::new`]) refuses unconditionally: its one
    /// backing store is not a per-partition durable store.
    ///
    /// **What this does NOT cover: writes in flight.** The write path appends
    /// to the WAL first and applies to the partition after, so an entry that is
    /// durable in the WAL but not yet resident in any partition is invisible
    /// here. A truncation must therefore also be bounded by the lowest LSN any
    /// in-flight write can hold (a reservation taken before its append) and by
    /// a WAL position read BEFORE this call. That reservation is P2.2's to add,
    /// alongside the truncation that needs it; this answer alone is not a safe
    /// truncation point.
    ///
    /// The partition list is snapshotted first, so the read lock is not held
    /// across the per-partition awaits.
    pub async fn min_unflushed_lsn(&self) -> Result<Option<u64>> {
        if self.shared_storage {
            anyhow::bail!(
                "shared-storage record store: one backing store for every partition is not \
                 durable per partition, so the WAL cannot be truncated on its account"
            );
        }
        let partitions: Vec<((String, String), Arc<dyn RecordStorage>)> = self
            .partitions
            .read()
            .iter()
            .map(|(key, partition)| (key.clone(), partition.clone()))
            .collect();
        let mut floor: Option<u64> = None;
        for ((tenant_id, collection), partition) in partitions {
            let partition_floor = partition.min_unflushed_lsn().await.with_context(|| {
                format!("record partition (tenant '{tenant_id}', collection '{collection}') refuses WAL truncation")
            })?;
            floor = match (floor, partition_floor) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        Ok(floor)
    }

    /// Remove ALL in-RAM state for one `(tenant, collection)` partition: the
    /// partition itself (recreated empty on demand) and any lazily built
    /// UNIQUE/secondary index for it. The WAL marker that keeps this
    /// restart-safe is appended by [`Self::drop_table_records`].
    fn drop_partition_state(&self, tenant_id: &str, collection_id: &str) {
        self.partitions
            .write()
            .remove(&(tenant_id.to_string(), collection_id.to_string()));
        let index_key = (tenant_id.to_string(), collection_id.to_string());
        self.unique_index.write().remove(&index_key);
        self.secondary_index.write().remove(&index_key);
    }

    /// Apply already-durable mutations to the memtable partitions and the
    /// incremental UNIQUE/secondary indexes (shared by `write_mutations` and
    /// `commit_atomic`). `lsns[i]` is the WAL sequence of `storage_actions[i]`;
    /// `None` = the positional mapping could not be trusted.
    async fn apply_storage_actions(
        &self,
        table_schema: &CatalogTableSchema,
        partition: &Arc<dyn RecordStorage>,
        legacy_partition: Option<&Arc<dyn RecordStorage>>,
        index_key: &(String, String),
        storage_actions: Vec<(TableRecordMutationKind, ProximaRecord)>,
        lsns: Option<&[u64]>,
    ) -> Result<Vec<String>> {
        // TD-110 Slice C: maintain the UNIQUE/PK index incrementally — but only
        // once it has been built (first `check_unique_conflict`). Until then the
        // lazy build captures these writes from current state, so skipping is
        // safe; checking once avoids per-write work for tables never probed.
        let index_primary_key = schema_primary_key_column(table_schema);
        let maintain_index = !schema_unique_column_sets(table_schema).is_empty()
            && self.unique_index.read().contains_key(index_key);
        // TD-127: maintain the secondary index on the same once-built basis as the
        // UNIQUE/PK index — until first `lookup_secondary` builds it, the lazy
        // build captures these writes from current state, so skipping is safe.
        let maintain_secondary = !secondary_index_disabled()
            && !schema_secondary_index_columns(table_schema).is_empty()
            && self.secondary_index.read().contains_key(index_key);
        let maintain_any = maintain_index || maintain_secondary;

        let mut record_ids = Vec::with_capacity(storage_actions.len());
        for (idx, (kind, record)) in storage_actions.into_iter().enumerate() {
            // `None` when the positional mapping could not be trusted (above), in
            // which case the store gets no LSN and reports "cannot truncate"
            // rather than a minimum derived from a partial set.
            let lsn = lsns.as_ref().and_then(|v| v.get(idx).copied());
            let key = RecordKey::from(&record);
            match kind {
                TableRecordMutationKind::Insert
                | TableRecordMutationKind::Upsert
                | TableRecordMutationKind::Update => {
                    if maintain_any {
                        let written = match lsn {
                            Some(lsn) => {
                                partition.upsert_record_at_lsn(record.clone(), lsn).await?
                            }
                            None => partition.upsert_record(record.clone()).await?,
                        };
                        record_ids.push(written.oid);
                        if maintain_index
                            && let Some(index) = self.unique_index.write().get_mut(index_key)
                        {
                            index.upsert(&record, index_primary_key.as_deref());
                        }
                        if maintain_secondary
                            && let Some(index) = self.secondary_index.write().get_mut(index_key)
                        {
                            index.upsert(&record);
                        }
                    } else {
                        let written = match lsn {
                            Some(lsn) => partition.upsert_record_at_lsn(record, lsn).await?,
                            None => partition.upsert_record(record).await?,
                        };
                        record_ids.push(written.oid);
                    }
                    // ADR-031 O3 migrate-on-write: drain any pre-flip copy from the
                    // legacy name-keyed partition so it can't resurrect on a later
                    // dual-read (memtable deletes don't tombstone). No-op when absent.
                    if let Some(legacy) = &legacy_partition {
                        legacy.delete_record(&key).await?;
                    }
                }
                TableRecordMutationKind::Delete => {
                    match lsn {
                        Some(lsn) => partition.delete_record_at_lsn(&key, lsn).await?,
                        None => partition.delete_record(&key).await?,
                    };
                    // Delete from BOTH partitions during the cutover window. The
                    // legacy copy's delete is not this WAL entry's (replay routes
                    // it to the primary), so it stays untracked.
                    if let Some(legacy) = &legacy_partition {
                        legacy.delete_record(&key).await?;
                    }
                    if maintain_index
                        && let Some(index) = self.unique_index.write().get_mut(index_key)
                    {
                        index.delete(&record.oid);
                    }
                    if maintain_secondary
                        && let Some(index) = self.secondary_index.write().get_mut(index_key)
                    {
                        index.delete(&record.oid);
                    }
                    record_ids.push(record.oid);
                }
                TableRecordMutationKind::OverwriteSnapshot
                | TableRecordMutationKind::ReplacePartitions
                | TableRecordMutationKind::Merge => {
                    return Err(anyhow!(
                        "mutation kind {kind:?} reached the apply step for table '{}'; only row mutations are applied here",
                        table_schema.name
                    ));
                }
            }
        }
        Ok(record_ids)
    }

    /// ADR-031 O3: the primary record partition for a table, plus — during an oid
    /// cutover with pre-flip data resident — the legacy name-keyed partition, so the
    /// hot path is mixed-read-safe (dual-read + migrate-on-write). With the gate off,
    /// `legacy` is `None` and this is exactly [`Self::partition`] by name.
    fn partitions_for(
        &self,
        tenant_id: &str,
        table_schema: &CatalogTableSchema,
    ) -> (Arc<dyn RecordStorage>, Option<Arc<dyn RecordStorage>>) {
        let (primary, legacy) = memtable_partition_keys(table_schema);
        if let Some(hints) = &self.schema_hints {
            hints.note(tenant_id, &primary, table_schema);
            if let Some(legacy) = &legacy {
                hints.note(tenant_id, legacy, table_schema);
            }
        }
        let primary_partition = self.partition(tenant_id, &primary);
        let legacy_partition = legacy
            .map(|key| self.partition(tenant_id, &key))
            // A shared-storage store (`new`) routes every key to the same backing
            // Arc; there is then no distinct legacy partition to dual-read/migrate
            // (and migrate-on-write would otherwise delete what we just wrote).
            .filter(|legacy| !Arc::ptr_eq(legacy, &primary_partition));
        (primary_partition, legacy_partition)
    }

    /// Replay canonical WAL entries into the correct `(tenant, table)` partitions
    /// on recovery, routing by the entry's `tenant_id` + the operation's
    /// `collection_id`. Reuses the per-store `RecordStore` point ops.
    pub async fn replay_wal_entries<I>(
        &self,
        entries: I,
    ) -> Result<proximadb_records::RecordRecoverySummary>
    where
        I: IntoIterator<Item = CanonicalWalEntry>,
    {
        let mut summary = proximadb_records::RecordRecoverySummary::default();
        for entry in entries {
            // TD-SEAM-12: rows a service store keeps in the WAL itself belong
            // to that store, not to a table; replayed here they would sit
            // unflushed in a partition of their own and cap the truncation
            // floor (the live path never puts them in a partition either).
            if let CanonicalOperation::RecordUpsert { collection_id, .. }
            | CanonicalOperation::RecordDelete { collection_id, .. } = &entry.operation
                && is_store_owned_collection(collection_id)
            {
                continue;
            }
            let tenant_id = entry.tenant_id.clone().unwrap_or_default();
            // The replayed entry carries its own LSN, and it is the SAME LSN the
            // live write path recorded. Passing it keeps a spill-backed partition
            // able to compute a truncation point after a restart — replay through
            // plain `upsert_record` would leave every recovered row resident but
            // untracked, so `min_unflushed_lsn` would fail closed forever and 2b
            // could never make progress on a restarted node.
            let lsn = entry.sequence_number;
            match entry.operation {
                CanonicalOperation::RecordUpsert {
                    collection_id,
                    record,
                    ..
                } => {
                    self.partition(&tenant_id, &collection_id)
                        .upsert_record_at_lsn(*record, lsn)
                        .await?;
                    summary.upserts_replayed += 1;
                }
                CanonicalOperation::RecordDelete {
                    collection_id, oid, ..
                } => {
                    self.partition(&tenant_id, &collection_id)
                        .delete_record_at_lsn(&RecordKey::new(oid), lsn)
                        .await?;
                    summary.deletes_replayed += 1;
                }
                CanonicalOperation::RecordPartitionDrop { collection_id } => {
                    // DROP TABLE: purge the partition's durable objects, then void
                    // it (+ its lazily built index state). Upserts replayed after
                    // this entry re-seed the partition — a table recreated
                    // post-drop starts empty.
                    //
                    // The purge mirrors the live `drop_table_records` (TD-USUB-20).
                    // Without it, a spill partition that crossed its flush
                    // threshold earlier in THIS replay has written the dropped
                    // rows into segments; voiding the map entry leaves them on
                    // disk, and the next access's discovery adopts them into the
                    // re-created table. Safe under WAL truncation: truncation only
                    // discards entries below `min_unflushed_lsn`, so a drop that
                    // replay still sees is at or above it, and every post-drop row
                    // a purge here deletes is still in the WAL after it and
                    // replays back. A failure propagates: serving the dropped rows
                    // would be silently wrong (mandate #1).
                    //
                    // Shared-storage mode cannot purge one partition (every key is
                    // one backing store); the live path declines there too.
                    if !self.shared_storage {
                        self.partition(&tenant_id, &collection_id)
                            .purge_durable_objects()
                            .await
                            .with_context(|| {
                                format!(
                                    "replaying RecordPartitionDrop for '{collection_id}': \
                                     purging durable objects"
                                )
                            })?;
                    }
                    self.drop_partition_state(&tenant_id, &collection_id);
                }
                // Checkpoints, CDC barriers, and system-catalog mutations carry
                // no record state for this partitioned store to replay.
                CanonicalOperation::Checkpoint(_)
                | CanonicalOperation::CdcBarrier { .. }
                | CanonicalOperation::CatalogMutation { .. } => {}
            }
        }
        Ok(summary)
    }

    /// Build the UNIQUE/PK index for `(tenant_id, table_schema)` if not already
    /// built, by scanning the tenant's current visible state (WAL recovery has
    /// Scan the table's visible records across BOTH the current partition
    /// and (TD-127 O3) any pre-flip legacy name-keyed partition, the current
    /// partition winning on conflict — shared by the unique/secondary index
    /// builders below.
    async fn scan_visible_records_for_index(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_id: &str,
        collection_id: &str,
        legacy_key: &Option<String>,
    ) -> Result<Vec<ProximaRecord>> {
        let scan_request = TableRecordScanRequest {
            filter: None,
            table_id: collection_id.to_string(),
            limit: None,
            include_vector: false,
            include_props: true,
        };
        let mut existing =
            RecordStorageTableRecordStore::new(self.partition(tenant_id, collection_id))
                .scan_records(
                    table_schema,
                    scan_request.clone(),
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(
                        SystemReadReason::IndexBuild,
                        "record_store::build_unique_index",
                    ),
                )
                .await?;
        if let Some(legacy) = legacy_key {
            let legacy_recs = RecordStorageTableRecordStore::new(self.partition(tenant_id, legacy))
                .scan_records(
                    table_schema,
                    scan_request,
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::IndexBuild, "record_store::build_index"),
                )
                .await?;
            existing = merge_scan_primary_wins(existing, legacy_recs, None);
        }
        Ok(existing)
    }

    /// rebuilt it). A no-op when the table has no UNIQUE/PK sets.
    async fn ensure_unique_index_built(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_id: &str,
    ) -> Result<()> {
        let (collection_id, legacy_key) = memtable_partition_keys(table_schema);
        let index_key = (tenant_id.to_string(), collection_id.clone());
        if self.unique_index.read().contains_key(&index_key) {
            return Ok(());
        }
        let set_columns = schema_unique_column_sets(table_schema);
        if set_columns.is_empty() {
            return Ok(());
        }
        let primary_key = schema_primary_key_column(table_schema);
        let existing = self
            .scan_visible_records_for_index(table_schema, tenant_id, &collection_id, &legacy_key)
            .await?;
        let mut index = TableUniqueIndex::with_sets(&set_columns);
        for record in &existing {
            index.upsert(record, primary_key.as_deref());
        }
        // Double-checked insert: keep an index another writer built meanwhile.
        self.unique_index.write().entry(index_key).or_insert(index);
        Ok(())
    }

    /// TD-127: build the secondary index for `(tenant_id, table_schema)` if not
    /// already built, by scanning the tenant's current visible state (WAL
    /// recovery has rebuilt it). A no-op when the table declares no eligible
    /// secondary-index columns or the kill-switch is set.
    async fn ensure_secondary_index_built(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_id: &str,
    ) -> Result<()> {
        if secondary_index_disabled() {
            return Ok(());
        }
        let (collection_id, legacy_key) = memtable_partition_keys(table_schema);
        let index_key = (tenant_id.to_string(), collection_id.clone());
        if self.secondary_index.read().contains_key(&index_key) {
            return Ok(());
        }
        let columns = schema_secondary_index_columns(table_schema);
        if columns.is_empty() {
            return Ok(());
        }
        let existing = self
            .scan_visible_records_for_index(table_schema, tenant_id, &collection_id, &legacy_key)
            .await?;
        let mut index = TableSecondaryIndex::with_columns(&columns);
        for record in &existing {
            index.upsert(record);
        }
        // Double-checked insert: keep an index another writer built meanwhile.
        self.secondary_index
            .write()
            .entry(index_key)
            .or_insert(index);
        Ok(())
    }
}

#[async_trait]
impl TableRecordStore for DirectWalTableRecordStore {
    /// Record-tier DROP TABLE: purge the table's `(tenant, collection)`
    /// partition (+ legacy cutover alias, + lazily built index state) and
    /// append the `RecordPartitionDrop` WAL marker FIRST so replay after a
    /// restart voids the partition too (crash between marker and purge is
    /// the fail-safe direction: the marker replays, the partition resets).
    async fn drop_table_records(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<()> {
        let tenant_scope = Self::tenant_key(tenant_context);
        let (collection_id, legacy_key) = memtable_partition_keys(table_schema);

        // Shared-storage mode (`new`): every partition key routes to ONE
        // backing Arc, so a partition-scoped purge is impossible. Fail soft (the
        // catalog drop already happened); that shape is documented non-isolated
        // (unit tests only).
        //
        // Read from the flag recorded at construction, not by calling the factory
        // for an `Arc::ptr_eq` comparison: a factory that builds a DURABLE store
        // does I/O and claims an object path, so probing it would create a
        // partition store on every DROP TABLE purely to discard it.
        if self.shared_storage {
            tracing::warn!(
                table = %table_schema.name,
                "DirectWalTableRecordStore in shared-storage mode cannot purge one table's records; a table recreated with the same name may see the dropped rows"
            );
            return Ok(());
        }

        // Restart-safety first: the marker must be durable before the live
        // purge, or a crash in between replays the dropped rows back.
        let tenant_id = tenant_context.map(|tenant| tenant.tenant_id.clone());
        let mut operations = vec![CanonicalOperation::RecordPartitionDrop {
            collection_id: collection_id.clone(),
        }];
        if let Some(legacy) = &legacy_key {
            operations.push(CanonicalOperation::RecordPartitionDrop {
                collection_id: legacy.clone(),
            });
        }
        // Held until the purge and state drop below have run (TD-SEAM-12).
        let _inflight = self.reserve_inflight();
        self.wal_appender
            .append_operations(operations, tenant_id)
            .await?;

        // Delete the partition's durable objects BEFORE releasing the in-memory
        // handle, and surface a failure rather than logging it.
        //
        // Order matters: `drop_partition_state` removes the map entry, and for a
        // store that owns objects that is NOT deletion — it is losing the handle
        // to data that stays in the bucket. Purging first means a failure leaves
        // the partition addressable, so a retried DROP can finish the job.
        //
        // A no-op for the in-memory default (nothing to own). The purge is
        // idempotent, so a partition created on demand here purely to be dropped
        // is harmless.
        //
        // The error is returned, but do NOT read that as "a failed purge fails the
        // DROP": the caller cannot make it fatal. `DropTable::execute` commits the
        // CATALOG drop before calling this, so by the time a purge failure is
        // known the table is already gone — failing the statement would report
        // failure for a table that really is dropped. It therefore records the
        // failure loudly (`error!` + `proximadb_record_tier_purge_failures_total`)
        // and continues. Returning the error here is still right: it is the
        // caller's decision to make, and the context travels with it.
        //
        // The consequence is that a purge failure orphans objects permanently
        // (nothing references them afterwards, so nothing retries). That gap needs
        // an orphan reaper and is the first Open item on TD-USUB-1 — it is not
        // closed by this change.
        //
        // `partitions_for` already drops a legacy partition that is the same `Arc`
        // as the primary, so a shared-storage store is not purged twice.
        let (primary_partition, legacy_partition) =
            self.partitions_for(&tenant_scope, table_schema);
        for partition in std::iter::once(primary_partition).chain(legacy_partition) {
            partition.purge_durable_objects().await.with_context(|| {
                format!(
                    "dropping table '{}': purging durable objects for partition '{collection_id}'",
                    table_schema.name
                )
            })?;
        }

        self.drop_partition_state(&tenant_scope, &collection_id);
        if let Some(legacy) = &legacy_key {
            self.drop_partition_state(&tenant_scope, legacy);
        }
        tracing::info!(
            table = %table_schema.name,
            collection = %collection_id,
            "🗑️ dropped record partition (+ WAL marker) for table"
        );
        Ok(())
    }

    /// The snapshot LSN is the WAL head, lowered below any write that is
    /// appended but not yet applied (a scan could miss it, so the read-merge
    /// must replay it). Chosen under the WAL's append lock and pinned there,
    /// so no truncation can slip between the capture and the pin. Falls back
    /// to the table's newest change for an appender that never truncates.
    async fn capture_snapshot_lsn(&self, collection_id: &str) -> Result<SnapshotLsn> {
        let inflight = &self.inflight;
        let below_inflight = |head: u64| {
            inflight
                .floor()
                .map_or(head, |floor| head.min(floor.saturating_sub(1)))
        };
        match self.wal_appender.pin_history(&below_inflight).await? {
            Some((lsn, pin)) => Ok(SnapshotLsn {
                lsn,
                pin: Some(pin),
            }),
            None => Ok(snapshot_lsn_from_change_feed(self, collection_id).await),
        }
    }

    /// CDC change-feed over the canonical WAL: surface every RecordUpsert/RecordDelete for
    /// `collection_id` (the table name) with sequence number > `since_lsn`, oldest first.
    /// Tenant-agnostic (all tenants) — see [`read_changes_since_scoped`] for the
    /// tenant-isolated feed the OLAP read-merge uses.
    async fn read_changes_since(
        &self,
        collection_id: &str,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        // TD-OLAP-17 fast-path: skip the O(WAL) scan when the collection's
        // high-water hasn't advanced past `since_lsn` (no entry can satisfy
        // `sequence_number > since_lsn`). `None` ⇒ appender doesn't track ⇒ scan.
        self.ensure_history_retained(since_lsn)?;
        if self.change_feed_is_empty(collection_id, since_lsn) {
            return Ok(Vec::new());
        }
        let entries = self
            .wal_appender
            .read_entries_since(collection_id, since_lsn)
            .await?;
        let mut out: Vec<ChangeRow> = entries
            .iter()
            .filter(|e| e.sequence_number > since_lsn)
            .filter_map(|e| change_row_from_entry(e, collection_id))
            .collect();
        out.sort_by_key(|r| r.lsn);
        Ok(out)
    }

    /// Tenant-isolated change-feed: like [`read_changes_since`] but keeps only
    /// entries whose `tenant_id` matches `tenant` (None/"" = the unscoped tenant).
    /// The WAL `collection_id` is the bare table name (not tenant-unique), so this
    /// match is what isolates two tenants that share a table name.
    async fn read_changes_since_scoped(
        &self,
        collection_id: &str,
        tenant: Option<&str>,
        since_lsn: u64,
    ) -> Result<Vec<ChangeRow>> {
        // TD-OLAP-17 fast-path (tenant-agnostic): the collection high-water is a
        // sufficient condition for emptiness across ALL tenants — if no entry for
        // this collection exceeds `since_lsn`, the tenant filter cannot add any.
        self.ensure_history_retained(since_lsn)?;
        if self.change_feed_is_empty(collection_id, since_lsn) {
            return Ok(Vec::new());
        }
        let entries = self
            .wal_appender
            .read_entries_since(collection_id, since_lsn)
            .await?;
        let want = tenant.filter(|t| !t.is_empty());
        let mut out: Vec<ChangeRow> = entries
            .iter()
            .filter(|e| e.sequence_number > since_lsn)
            .filter(|e| e.tenant_id.as_deref().filter(|t| !t.is_empty()) == want)
            .filter_map(|e| change_row_from_entry(e, collection_id))
            .collect();
        out.sort_by_key(|r| r.lsn);
        Ok(out)
    }

    async fn write_mutations(
        &self,
        table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        self.ensure_healthy()?;
        let mut operations = Vec::with_capacity(mutations.len());
        let mut storage_actions = Vec::with_capacity(mutations.len());
        let projections = projection_directives_for_schema(table_schema);
        // TD-064: structural per-(tenant, collection) partition selection.
        let tenant_scope = Self::tenant_key(tenant_context);
        // ADR-031 O2/O3: the WAL `collection_id` is the stable `object_id` (decimal)
        // once the cutover gate is on and the table carries one; otherwise the bare
        // name (legacy). The memtable partition + indexes key by the SAME id
        // (lockstep), and during the mixed window writes also touch the legacy
        // name-keyed partition (migrate-on-write) so a pre-flip copy can't resurrect
        // (memtable deletes don't tombstone). Readers dual-read — mixed-read-safe.
        let (collection_id, legacy_key) = memtable_partition_keys(table_schema);
        let partition = self.partition(&tenant_scope, &collection_id);
        let legacy_partition = legacy_key
            .as_ref()
            .map(|key| self.partition(&tenant_scope, key))
            // Shared-storage stores route every key to one Arc — no distinct legacy
            // partition, and migrate-on-write must NOT delete what we just wrote.
            .filter(|legacy| !Arc::ptr_eq(legacy, &partition));
        let index_key = (tenant_scope.clone(), collection_id.clone());

        for mutation in mutations {
            let kind = mutation.kind;
            let record = mutation.record;
            let key = RecordKey::from(&record);

            match kind {
                TableRecordMutationKind::Insert => {
                    let exists = partition.get_record(&key).await?.is_some()
                        || match &legacy_partition {
                            Some(legacy) => legacy.get_record(&key).await?.is_some(),
                            None => false,
                        };
                    if exists {
                        return Ok(TableRecordWriteResult::failure(
                            format!(
                                "Record '{}' already exists in table '{}'",
                                record.oid, table_schema.name
                            ),
                            "INSERT_CONFLICT",
                        ));
                    }
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.clone(),
                        record: Box::new(record.clone()),
                        projections: projections.clone(),
                    });
                    storage_actions.push((kind, record));
                }
                TableRecordMutationKind::Update => {
                    let exists = partition.get_record(&key).await?.is_some()
                        || match &legacy_partition {
                            Some(legacy) => legacy.get_record(&key).await?.is_some(),
                            None => false,
                        };
                    if !exists {
                        return Ok(TableRecordWriteResult::failure(
                            format!(
                                "Record '{}' does not exist in table '{}'",
                                record.oid, table_schema.name
                            ),
                            "UPDATE_NOT_FOUND",
                        ));
                    }
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.clone(),
                        record: Box::new(record.clone()),
                        projections: projections.clone(),
                    });
                    storage_actions.push((kind, record));
                }
                TableRecordMutationKind::Upsert => {
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.clone(),
                        record: Box::new(record.clone()),
                        projections: projections.clone(),
                    });
                    storage_actions.push((kind, record));
                }
                TableRecordMutationKind::Delete => {
                    operations.push(CanonicalOperation::RecordDelete {
                        collection_id: collection_id.clone(),
                        oid: record.oid.clone(),
                        projections: projections.clone(),
                    });
                    storage_actions.push((kind, record));
                }
                TableRecordMutationKind::OverwriteSnapshot
                | TableRecordMutationKind::ReplacePartitions
                | TableRecordMutationKind::Merge => {
                    return Err(anyhow!(
                        "Mutation kind {:?} for table '{}' requires snapshot/merge commit support",
                        kind,
                        table_schema.name
                    ));
                }
            }
        }

        let tenant_id = tenant_context.map(|tenant| tenant.tenant_id.clone());
        // Bind the result: each entry's `sequence_number` is the durable WAL LSN
        // of the corresponding operation, and until TD-USUB-1's 2b work it was
        // discarded here. `operations` and `storage_actions` are pushed in
        // lockstep below, and `append_operations` returns its entries in input
        // order (built by `map`/`collect` over the input, after `sync_data`), so
        // index i of this vec is the LSN for `storage_actions[i]`.
        // Held until the entries are applied to their partitions (TD-SEAM-12).
        let _inflight = self.reserve_inflight();
        let wal_entries = self
            .wal_appender
            .append_operations(operations, tenant_id)
            .await?;
        // Defensive, not decorative: a future appender that reorders or coalesces
        // would make the positional mapping silently wrong — and a wrong LSN here
        // becomes a wrong truncation point later, i.e. data loss. Mismatched
        // lengths therefore disable LSN tracking rather than guess.
        let lsns: Option<Vec<u64>> = if wal_entries.len() == storage_actions.len() {
            Some(wal_entries.iter().map(|e| e.sequence_number).collect())
        } else {
            tracing::warn!(
                entries = wal_entries.len(),
                actions = storage_actions.len(),
                "WAL entry count does not match storage actions; skipping LSN association \
                 (truncation will fail closed rather than use a guessed mapping)"
            );
            None
        };

        let record_ids = self
            .apply_storage_actions(
                table_schema,
                &partition,
                legacy_partition.as_ref(),
                &index_key,
                storage_actions,
                lsns.as_deref(),
            )
            .await?;

        Ok(TableRecordWriteResult::success(record_ids))
    }

    /// ADR-095 D3 / TD-PGOLTP-1: make a whole transaction durable as ONE WAL
    /// group frame, then apply it in staged order.
    ///
    /// The staged statements were validated against base ∪ staged under the
    /// tables' op-locks, but writers that do not take op-locks (INSERT…SELECT,
    /// DDL, graph merge-back) can still change a key, and `RecordUpsert` has no
    /// conflict semantics. So INSERT-absent / UPDATE-exists is re-checked here
    /// against current partitions, in staged order with a local overlay, BEFORE
    /// anything is durable; a mismatch aborts the whole commit (40001).
    async fn commit_atomic(
        &self,
        batches: Vec<StagedTableWrite>,
    ) -> Result<Vec<TableRecordWriteResult>> {
        self.ensure_healthy()?;
        use crate::services::txn_staging::{CommitOutcomeUnknown, TxnSerializationConflict};

        struct PlannedBatch {
            table_schema: CatalogTableSchema,
            partition: Arc<dyn RecordStorage>,
            legacy_partition: Option<Arc<dyn RecordStorage>>,
            index_key: (String, String),
            storage_actions: Vec<(TableRecordMutationKind, ProximaRecord)>,
        }

        let mut operations: Vec<(CanonicalOperation, Option<String>)> = Vec::new();
        let mut planned: Vec<PlannedBatch> = Vec::with_capacity(batches.len());
        // (tenant, collection, oid) -> present after the mutations planned so far.
        let mut overlay: std::collections::HashMap<(String, String, String), bool> =
            std::collections::HashMap::new();

        for batch in batches {
            let tenant_context = batch.tenant_context.as_ref();
            let tenant_scope = Self::tenant_key(tenant_context);
            let tenant_id = tenant_context.map(|tenant| tenant.tenant_id.clone());
            let projections = projection_directives_for_schema(&batch.table_schema);
            let (collection_id, _) = memtable_partition_keys(&batch.table_schema);
            let (partition, legacy_partition) =
                self.partitions_for(&tenant_scope, &batch.table_schema);
            let mut storage_actions = Vec::with_capacity(batch.mutations.len());

            for mutation in batch.mutations {
                let record = mutation.record;
                let overlay_key = (
                    tenant_scope.clone(),
                    collection_id.clone(),
                    record.oid.clone(),
                );
                let present = match overlay.get(&overlay_key) {
                    Some(present) => *present,
                    None => {
                        let key = RecordKey::from(&record);
                        partition.get_record(&key).await?.is_some()
                            || match &legacy_partition {
                                Some(legacy) => legacy.get_record(&key).await?.is_some(),
                                None => false,
                            }
                    }
                };
                match mutation.kind {
                    TableRecordMutationKind::Insert if present => {
                        return Err(TxnSerializationConflict(format!(
                            "row '{}' of table '{}' was inserted concurrently",
                            record.oid, batch.table_schema.name
                        ))
                        .into());
                    }
                    TableRecordMutationKind::Update if !present => {
                        return Err(TxnSerializationConflict(format!(
                            "row '{}' of table '{}' was deleted concurrently",
                            record.oid, batch.table_schema.name
                        ))
                        .into());
                    }
                    TableRecordMutationKind::Insert
                    | TableRecordMutationKind::Update
                    | TableRecordMutationKind::Upsert => {
                        overlay.insert(overlay_key, true);
                        operations.push((
                            CanonicalOperation::RecordUpsert {
                                collection_id: collection_id.clone(),
                                record: Box::new(record.clone()),
                                projections: projections.clone(),
                            },
                            tenant_id.clone(),
                        ));
                    }
                    TableRecordMutationKind::Delete => {
                        overlay.insert(overlay_key, false);
                        operations.push((
                            CanonicalOperation::RecordDelete {
                                collection_id: collection_id.clone(),
                                oid: record.oid.clone(),
                                projections: projections.clone(),
                            },
                            tenant_id.clone(),
                        ));
                    }
                    TableRecordMutationKind::OverwriteSnapshot
                    | TableRecordMutationKind::ReplacePartitions
                    | TableRecordMutationKind::Merge => {
                        return Err(anyhow!(
                            "mutation kind {:?} for table '{}' cannot be committed atomically",
                            mutation.kind,
                            batch.table_schema.name
                        ));
                    }
                }
                storage_actions.push((mutation.kind, record));
            }

            planned.push(PlannedBatch {
                index_key: (tenant_scope, collection_id),
                table_schema: batch.table_schema,
                partition,
                legacy_partition,
                storage_actions,
            });
        }

        let expected = operations.len();
        // Held until the batch is applied (TD-SEAM-12).
        let _inflight = self.reserve_inflight();
        let entries = match self.wal_appender.append_operations_atomic(operations).await {
            Ok(entries) => entries,
            Err(err)
                if err.chain().any(|cause| {
                    cause
                        .downcast_ref::<crate::services::canonical_wal::WalAppendIndeterminate>()
                        .is_some()
                }) =>
            {
                return Err(CommitOutcomeUnknown(format!("{err:#}")).into());
            }
            Err(err) => return Err(err),
        };

        // Durable from here on: any failure is "outcome unknown", never a
        // plain error the client could read as "not committed".
        if entries.len() != expected {
            return Err(CommitOutcomeUnknown(format!(
                "atomic append returned {} entries for {} operations",
                entries.len(),
                expected
            ))
            .into());
        }
        let lsns: Vec<u64> = entries.iter().map(|entry| entry.sequence_number).collect();
        let mut results = Vec::with_capacity(planned.len());
        let mut offset = 0usize;
        for batch in planned {
            let count = batch.storage_actions.len();
            let batch_lsns = &lsns[offset..offset + count];
            offset += count;
            let record_ids = self
                .apply_storage_actions(
                    &batch.table_schema,
                    &batch.partition,
                    batch.legacy_partition.as_ref(),
                    &batch.index_key,
                    batch.storage_actions,
                    Some(batch_lsns),
                )
                .await
                .map_err(|err| {
                    // Durable but only partly applied: refuse to serve it.
                    self.poisoned
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    tracing::error!(
                        error = %format!("{err:#}"),
                        "applying a durable atomic commit failed; record store poisoned until restart"
                    );
                    CommitOutcomeUnknown(format!("{err:#}"))
                })?;
            results.push(TableRecordWriteResult::success(record_ids));
        }
        Ok(results)
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        self.ensure_healthy()?;
        // Pass `None`: the partition already scopes the tenant structurally, so
        // no per-record tenant filter is needed (TD-064).
        let tenant = Self::tenant_key(tenant_context);
        let (primary, legacy) = self.partitions_for(&tenant, table_schema);
        let Some(legacy) = legacy else {
            return RecordStorageTableRecordStore::new(primary)
                .get_by_key(
                    table_schema,
                    request,
                    None,
                    #[cfg(feature = "abac-policy")]
                    read_context,
                )
                .await;
        };
        // O3 cutover window: the primary (post-flip) partition wins; fall back to
        // the legacy name-keyed partition for records not yet migrated-on-write.
        let primary_res = RecordStorageTableRecordStore::new(primary)
            .get_by_key(
                table_schema,
                request.clone(),
                None,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await?;
        if primary_res.is_some() {
            return Ok(primary_res);
        }
        RecordStorageTableRecordStore::new(legacy)
            .get_by_key(
                table_schema,
                request,
                None,
                #[cfg(feature = "abac-policy")]
                read_context,
            )
            .await
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        self.ensure_healthy()?;
        let tenant = Self::tenant_key(tenant_context);
        let (primary, legacy) = self.partitions_for(&tenant, table_schema);
        let Some(legacy) = legacy else {
            return RecordStorageTableRecordStore::new(primary)
                .scan_records(
                    table_schema,
                    request,
                    None,
                    #[cfg(feature = "abac-policy")]
                    _read_context,
                )
                .await;
        };
        let limit = request.limit;
        let primary_recs = RecordStorageTableRecordStore::new(primary)
            .scan_records(
                table_schema,
                request.clone(),
                None,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await?;
        let legacy_recs = RecordStorageTableRecordStore::new(legacy)
            .scan_records(
                table_schema,
                request,
                None,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await?;
        Ok(merge_scan_primary_wins(primary_recs, legacy_recs, limit))
    }

    /// The primary partition's [`RecordScanView`] and the filesystem its
    /// segments were written with.
    ///
    /// Declines (`Ok(None)`) while the pre-flip legacy partition still holds a
    /// live row: the materialized scan merges it primary-wins over LIVE rows,
    /// which is not a layer order a snapshot can express. Also declines when
    /// there are segments but no filesystem to read them through (a custom
    /// storage factory) — never a snapshot that silently omits them.
    async fn layer_a_snapshot(
        &self,
        table_schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<Option<crate::services::layer_a_snapshot::LayerASnapshot>> {
        self.ensure_healthy()?;
        let tenant = Self::tenant_key(tenant_context);
        let (primary, legacy) = self.partitions_for(&tenant, table_schema);
        if let Some(legacy) = legacy
            && !legacy.scan_records(1).await?.is_empty()
        {
            return Ok(None);
        }
        let view = primary.scan_view().await?;
        if !view.segments.is_empty() && self.segment_filesystem.is_none() {
            return Ok(None);
        }
        Ok(Some(crate::services::layer_a_snapshot::LayerASnapshot {
            view,
            filesystem: self.segment_filesystem.clone(),
            table_name: table_schema.name.clone(),
        }))
    }

    /// Primary-over-legacy, like `scan_records`: during an oid-keying cutover the
    /// legacy (name-keyed) partition is merged under the primary, and the primary
    /// copy of an oid wins. Both partitions stream in oid order, so the merge holds
    /// one chunk of each.
    fn scan_records_chunked(
        self: Arc<Self>,
        table_schema: CatalogTableSchema,
        request: TableRecordScanRequest,
        chunk_rows: usize,
        tenant_context: Option<TenantContext>,
    ) -> RecordChunkStream<'static> {
        if let Err(e) = self.ensure_healthy() {
            return futures::stream::once(async move { Err(e) }).boxed();
        }
        let tenant = Self::tenant_key(tenant_context.as_ref());
        let (primary, legacy) = self.partitions_for(&tenant, &table_schema);
        let name = table_schema.name;
        // The partition is already the tenant's: no per-record tenant filter, as in
        // `scan_records` (which passes `None`).
        let primary = RecordStorageTableRecordStore::partition_chunks(
            primary,
            name.clone(),
            &request,
            chunk_rows,
            None,
        );
        match legacy {
            None => primary,
            Some(legacy) => merge_newest_wins(
                vec![
                    RecordStorageTableRecordStore::partition_chunks(
                        legacy, name, &request, chunk_rows, None,
                    ),
                    primary,
                ],
                chunk_rows,
            ),
        }
    }

    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        self.ensure_healthy()?;
        let tenant = Self::tenant_key(tenant_context);
        let (primary, legacy) = self.partitions_for(&tenant, table_schema);
        let Some(legacy) = legacy else {
            return RecordStorageTableRecordStore::new(primary)
                .scan_records_filtered(
                    table_schema,
                    request,
                    predicate,
                    None,
                    #[cfg(feature = "abac-policy")]
                    _read_context,
                )
                .await;
        };
        let limit = request.limit;
        let primary_recs = RecordStorageTableRecordStore::new(primary)
            .scan_records_filtered(
                table_schema,
                request.clone(),
                predicate,
                None,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await?;
        let legacy_recs = RecordStorageTableRecordStore::new(legacy)
            .scan_records_filtered(
                table_schema,
                request,
                predicate,
                None,
                #[cfg(feature = "abac-policy")]
                _read_context,
            )
            .await?;
        Ok(merge_scan_primary_wins(primary_recs, legacy_recs, limit))
    }

    /// TD-110 Slice C: O(1) index-backed override of the default scan. Builds the
    /// per-table index on first use (from WAL-recovered current state), then
    /// probes candidate tuples directly.
    async fn check_unique_conflict(
        &self,
        table_schema: &CatalogTableSchema,
        _table_id: &str,
        _primary_key: Option<&str>,
        sets: &[UniqueCandidateSet],
        exclude_oids: &std::collections::HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<UniqueConflict>> {
        self.ensure_healthy()?;
        let tenant_scope = Self::tenant_key(tenant_context);
        self.ensure_unique_index_built(table_schema, &tenant_scope)
            .await?;
        let index_key = (tenant_scope, memtable_partition_keys(table_schema).0);
        let index = self.unique_index.read();
        let Some(table_index) = index.get(&index_key) else {
            return Ok(None); // table has no UNIQUE/PK sets
        };
        for set in sets {
            if let Some(tuple) = table_index.conflict(&set.columns, &set.candidates, exclude_oids) {
                return Ok(Some(UniqueConflict {
                    columns: set.columns.clone(),
                    tuple,
                }));
            }
        }
        Ok(None)
    }

    /// TD-127: index-backed secondary lookup. Builds the per-table index on first
    /// use (from current state), then probes for candidate oids. Returns `None`
    /// (scan fallback) when `column` is not an indexed column or the kill-switch
    /// is set.
    async fn lookup_secondary(
        &self,
        table_schema: &CatalogTableSchema,
        column: &str,
        values: &std::collections::HashSet<String>,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Option<Vec<String>>> {
        self.ensure_healthy()?;
        if secondary_index_disabled() {
            return Ok(None);
        }
        if !schema_secondary_index_columns(table_schema)
            .iter()
            .any(|indexed| indexed == column)
        {
            return Ok(None);
        }
        let tenant_scope = Self::tenant_key(tenant_context);
        self.ensure_secondary_index_built(table_schema, &tenant_scope)
            .await?;
        let index_key = (tenant_scope, memtable_partition_keys(table_schema).0);
        let index = self.secondary_index.read();
        let Some(table_index) = index.get(&index_key) else {
            return Ok(None);
        };
        Ok(table_index
            .probe(column, values)
            .map(|oids| oids.into_iter().collect()))
    }
}

fn projection_directives_for_schema(schema: &CatalogTableSchema) -> Vec<ProjectionDirective> {
    match schema.storage_specialization {
        CatalogStorageSpecialization::GenericRelational
        | CatalogStorageSpecialization::PaxRowFamily
        | CatalogStorageSpecialization::PaxOltp
        | CatalogStorageSpecialization::PaxOlap
        | CatalogStorageSpecialization::ColumnarAnalytics => {
            vec![ProjectionDirective::ColumnarVariation {
                collection_id: schema.name.clone(),
                fields: schema
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect(),
            }]
        }
        CatalogStorageSpecialization::VectorAnn => schema
            .columns
            .iter()
            .filter(|column| {
                matches!(
                    column.data_type,
                    proximadb_data_model::ProximaType::DenseVector { .. }
                )
            })
            .map(|column| ProjectionDirective::HnswIndex {
                collection_id: schema.name.clone(),
                embedding_field: column.name.clone(),
            })
            .collect(),
        CatalogStorageSpecialization::DocumentJson => {
            vec![ProjectionDirective::DocumentJsonPathIndex {
                collection_id: schema.name.clone(),
                indexed_paths: Vec::new(),
            }]
        }
        CatalogStorageSpecialization::GraphTopology => {
            vec![ProjectionDirective::CsrRebuild {
                graph_id: schema.name.clone(),
            }]
        }
        CatalogStorageSpecialization::ObservabilityTimeSeries => schema
            .columns
            .iter()
            .find(|column| column.name == "trace_id")
            .map(|trace_column| {
                vec![ProjectionDirective::ObservabilityTraceIndex {
                    collection_id: schema.name.clone(),
                    service_name: None,
                    trace_id_field: trace_column.name.clone(),
                    span_id_field: "span_id".to_string(),
                }]
            })
            .unwrap_or_default(),
        CatalogStorageSpecialization::ExternalOpenTable => schema
            .storage_layouts
            .iter()
            .find_map(|layout| canonical_open_table_format(&layout.physical_format))
            .map(|format| {
                vec![ProjectionDirective::OpenFormatManifest {
                    namespace: "default".to_string(),
                    table_name: schema.name.clone(),
                    format,
                }]
            })
            .unwrap_or_default(),
        CatalogStorageSpecialization::LsmWriteOptimized => Vec::new(),
    }
}

fn canonical_open_table_format(format: &CatalogPhysicalFormat) -> Option<CanonicalOpenTableFormat> {
    match format {
        CatalogPhysicalFormat::Iceberg => Some(CanonicalOpenTableFormat::Iceberg),
        CatalogPhysicalFormat::Delta => Some(CanonicalOpenTableFormat::Delta),
        CatalogPhysicalFormat::Hudi => Some(CanonicalOpenTableFormat::Hudi),
        _ => None,
    }
}

pub(crate) fn proxima_record_to_get_response(
    record: ProximaRecord,
    _table_schema: &CatalogTableSchema,
    include_vector: bool,
    include_props: bool,
) -> RichSearchResult {
    let vector = if include_vector {
        record
            .embeddings
            .first()
            .map(|embedding| embedding.values.to_fp32_owned())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let props = if include_props {
        record
            .props
            .iter()
            .filter_map(|(key, node)| match node {
                ProximaTreeNode::Value(value) => Some((key.clone(), value.clone())),
                ProximaTreeNode::Object(_) => None,
            })
            .collect()
    } else {
        Default::default()
    };

    RichSearchResult {
        id: if record.oid.is_empty() {
            "unknown".to_string()
        } else {
            record.oid
        },
        score: 1.0,
        similarity: None,
        vector,
        props,
        version: if record.record_version == 0 {
            None
        } else {
            Some(record.record_version as u32)
        },
        timestamp: if record.created_at_ns == 0 {
            None
        } else {
            Some(record.created_at_ns / 1_000_000)
        },
        source: record.origin,
    }
}

#[cfg(test)]
mod tests {

    /// Forgotten tempdir root for a spill test: `(root, base)` — the dir is
    /// deliberately leaked (test-hygiene: unique per test, never shared), and
    /// `base` is its string form for path closures.
    fn spill_root() -> (std::path::PathBuf, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let base = root.to_string_lossy().to_string();
        std::mem::forget(dir);
        (root, base)
    }

    /// A default local filesystem handle for the spill harness.
    async fn local_fs() -> Arc<crate::storage::persistence::filesystem::local::LocalFileSystem> {
        Arc::new(
            crate::storage::persistence::filesystem::local::LocalFileSystem::new(
                crate::storage::persistence::filesystem::local::LocalConfig::default(),
            )
            .await
            .expect("local filesystem"),
        )
    }

    /// `n` upsert WAL entries (`r{i}` oids, LSN 1..=n) for `collection`,
    /// stamped `tenant` on the entry (the record stays unstamped — the
    /// tenant-ingestion path derives it from the entry).
    fn upsert_entries(n: u64, collection: &str, tenant: &str) -> Vec<CanonicalWalEntry> {
        (0..n)
            .map(|i| {
                CanonicalWalEntry::new(
                    i + 1,
                    CanonicalOperation::RecordUpsert {
                        collection_id: collection.to_string(),
                        record: Box::new(ProximaRecord {
                            oid: format!("r{i}"),
                            variation_id: Some(collection.to_string()),
                            ..Default::default()
                        }),
                        projections: vec![],
                    },
                    Some(tenant.to_string()),
                )
            })
            .collect()
    }

    /// The standard spill-path resolver: `{base}/{tenant}/{collection}`.
    fn tenant_path_resolver(base: String) -> Arc<dyn Fn(&str, &str) -> String + Send + Sync> {
        Arc::new(move |tenant_id: &str, collection: &str| {
            format!("{base}/{tenant_id}/{collection}")
        })
    }
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_schema::Schema as ArrowSchema;
    use futures::stream::BoxStream;
    use proximadb_catalog::{CatalogColumn, CatalogStorageLayout};
    use proximadb_data_model::ProximaValue;

    /// ADR-031 O2: the WAL `collection_id` is the stable `object_id` (decimal) when
    /// keying is on AND the table carries one, else the bare name (legacy). This is
    /// what makes the cutover mixed-read-safe + rename-safe.
    #[test]
    fn wal_collection_id_prefers_object_id_when_keying_on() {
        let mut schema = CatalogTableSchema::new("orders");
        // No object_id yet → always the name, regardless of the gate.
        assert_eq!(wal_collection_id(&schema, false), "orders");
        assert_eq!(
            wal_collection_id(&schema, true),
            "orders",
            "no object_id → name even when keying is on"
        );
        // With an object_id: keying off → name; keying on → the decimal id.
        schema.object_id = Some(42);
        assert_eq!(
            wal_collection_id(&schema, false),
            "orders",
            "keying off → name"
        );
        assert_eq!(
            wal_collection_id(&schema, true),
            "42",
            "keying on + object_id → stable id, never the mutable name"
        );
    }

    /// ADR-031 O3: with the WAL/memtable oid gate ON, the in-memory partition is
    /// keyed by the stable object_id, but a pre-flip record recovered into the
    /// legacy name-keyed partition stays visible (dual-read), migrates to the oid
    /// partition on write, and is deleted from BOTH partitions on delete (memtable
    /// deletes don't tombstone, so a stale legacy copy must not resurrect).
    /// Process-isolated under nextest, so the env gate does not leak across tests.
    #[tokio::test]
    async fn memtable_oid_cutover_dual_reads_and_migrates_legacy_partition() {
        unsafe { std::env::set_var("PROXIMADB_WAL_OBJECT_ID_KEY", "1") };

        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_partitioned(wal);
        let mut schema = CatalogTableSchema::new("orders");
        schema.object_id = Some(123);

        let get = |oid: &str| TableRecordGetRequest {
            table_id: "orders".to_string(),
            key: oid.to_string(),
            include_vector: false,
            include_props: true,
        };
        let scan = || TableRecordScanRequest {
            filter: None,
            table_id: "orders".to_string(),
            limit: None,
            include_vector: false,
            include_props: true,
        };

        // Seed a PRE-FLIP record into the legacy name-keyed partition via WAL replay
        // (a name-stamped entry written before the gate flip).
        let legacy_rec = ProximaRecord {
            oid: "r1".to_string(),
            variation_id: Some("orders".to_string()),
            ..Default::default()
        };
        store
            .replay_wal_entries(vec![CanonicalWalEntry::new(
                1,
                CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(legacy_rec),
                    projections: vec![],
                },
                None,
            )])
            .await
            .unwrap();

        // Dual-read: under the oid gate, get + scan still see the legacy record.
        assert!(
            store
                .get_by_key(
                    &schema,
                    get("r1"),
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .unwrap()
                .is_some(),
            "legacy-partition record visible via dual-read"
        );
        assert_eq!(
            store
                .scan_records(
                    &schema,
                    scan(),
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests",),
                )
                .await
                .unwrap()
                .len(),
            1
        );

        // Migrate-on-write: an Update moves r1 into the oid partition and drains the
        // legacy copy — exactly one visible record, the migrated (v2) version.
        let updated = ProximaRecord {
            oid: "r1".to_string(),
            variation_id: Some("orders".to_string()),
            record_version: 2,
            ..Default::default()
        };
        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Update,
                    updated,
                )],
                None,
            )
            .await
            .unwrap();
        let after = store
            .scan_records(
                &schema,
                scan(),
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();
        assert_eq!(
            after.len(),
            1,
            "no duplicate across legacy + oid partitions"
        );
        assert_eq!(after[0].record_version, 2, "migrated primary copy wins");

        // Delete drains BOTH partitions — no legacy resurrection on the next read.
        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Delete,
                    after[0].clone(),
                )],
                None,
            )
            .await
            .unwrap();
        assert!(
            store
                .get_by_key(
                    &schema,
                    get("r1"),
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .unwrap()
                .is_none(),
            "deleted from both partitions — legacy copy did not resurrect"
        );

        unsafe { std::env::remove_var("PROXIMADB_WAL_OBJECT_ID_KEY") };
    }
    use proximadb_data_model::{ProximaType, VectorElement};
    use proximadb_kernel::error::StorageError;
    use proximadb_records::{EmbeddingCell, EmbeddingValues, RecordScan, RecordStore};
    use proximadb_storage_common::object_store_bridge::{
        BridgeInMemoryObjectStore as InMemory, BridgeObjectStore,
    };
    use proximadb_storage_common::pax_block::SEGMENT_MAGIC;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, RwLock};

    #[test]
    fn table_record_route_uses_catalog_storage_specialization() {
        let cases_parquet = [
            ("generic", CatalogStorageSpecialization::GenericRelational),
            ("columnar", CatalogStorageSpecialization::ColumnarAnalytics),
            ("document", CatalogStorageSpecialization::DocumentJson),
            ("graph", CatalogStorageSpecialization::GraphTopology),
        ];
        for (label, spec) in cases_parquet {
            let schema = CatalogTableSchema::new(label)
                .with_workload_profile(CatalogWorkloadProfile::Olap)
                .with_storage_specialization(spec);
            assert_eq!(
                TableRecordStoreRoute::for_schema(&schema),
                TableRecordStoreRoute::RelationalStorage,
                "{label} must route to RelationalStorage"
            );
        }

        let cases_pax_vector = [
            ("pax_row_family", CatalogStorageSpecialization::PaxRowFamily),
            ("pax_oltp", CatalogStorageSpecialization::PaxOltp),
            ("pax_olap", CatalogStorageSpecialization::PaxOlap),
            ("vector_ann", CatalogStorageSpecialization::VectorAnn),
        ];
        for (label, spec) in cases_pax_vector {
            let schema = CatalogTableSchema::new(label)
                .with_workload_profile(CatalogWorkloadProfile::Vector)
                .with_storage_specialization(spec);
            let expected = if spec == CatalogStorageSpecialization::VectorAnn {
                TableRecordStoreRoute::PaxVectorStorage
            } else {
                TableRecordStoreRoute::LegacyVectorCompatibility
            };
            assert_eq!(
                TableRecordStoreRoute::for_schema(&schema),
                expected,
                "{label} must route to {expected:?}"
            );
        }

        let legacy_schema = CatalogTableSchema::new("legacy_lsm")
            .with_workload_profile(CatalogWorkloadProfile::Htap)
            .with_storage_specialization(CatalogStorageSpecialization::LsmWriteOptimized);
        assert_eq!(
            TableRecordStoreRoute::for_schema(&legacy_schema),
            TableRecordStoreRoute::LegacyVectorCompatibility
        );
    }

    #[derive(Default)]
    struct MemoryRecordStorage {
        records: RwLock<HashMap<String, ProximaRecord>>,
    }

    #[async_trait]
    impl RecordStore for MemoryRecordStorage {
        async fn upsert_record(&self, record: ProximaRecord) -> Result<ProximaRecord> {
            self.records
                .write()
                .expect("memory storage write lock")
                .insert(record.oid.clone(), record.clone());
            Ok(record)
        }

        async fn get_record(&self, key: &RecordKey) -> Result<Option<ProximaRecord>> {
            Ok(self
                .records
                .read()
                .expect("memory storage read lock")
                .get(&key.oid)
                .cloned())
        }

        async fn delete_record(&self, key: &RecordKey) -> Result<bool> {
            Ok(self
                .records
                .write()
                .expect("memory storage write lock")
                .remove(&key.oid)
                .is_some())
        }
    }

    #[derive(Default)]
    struct RecordingWalAppender {
        next_sequence: AtomicU64,
        entries: Mutex<Vec<CanonicalWalEntry>>,
    }

    struct CapturingObjectStoreBridge {
        store: Arc<dyn BridgeObjectStore>,
        writes: Mutex<Vec<(ObjectPath, Vec<String>)>>,
        segments: Mutex<Vec<(ObjectPath, Vec<u8>)>>,
    }

    impl CapturingObjectStoreBridge {
        fn new() -> Self {
            Self {
                store: Arc::new(InMemory::new()),
                writes: Mutex::new(Vec::new()),
                segments: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ObjectStoreBridge for CapturingObjectStoreBridge {
        fn inner_store(&self) -> Arc<dyn BridgeObjectStore> {
            self.store.clone()
        }

        async fn read_parquet_batches(
            &self,
            _path: &ObjectPath,
            _schema: Arc<ArrowSchema>,
            _batch_size: usize,
            _tenant_id: Option<&str>,
        ) -> std::result::Result<
            BoxStream<'static, std::result::Result<RecordBatch, StorageError>>,
            StorageError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }

        async fn write_records_to_parquet(
            &self,
            path: &ObjectPath,
            records: &[ProximaRecord],
            _tenant_id: Option<&str>,
        ) -> std::result::Result<(), StorageError> {
            self.writes.lock().unwrap().push((
                path.clone(),
                records.iter().map(|record| record.oid.clone()).collect(),
            ));
            Ok(())
        }

        async fn fetch_vector_segment(
            &self,
            _path: &ObjectPath,
            _tenant_id: Option<&str>,
        ) -> std::result::Result<Vec<u8>, StorageError> {
            Ok(Vec::new())
        }

        async fn persist_vector_segment(
            &self,
            path: &ObjectPath,
            data: &[u8],
            _tenant_id: Option<&str>,
        ) -> std::result::Result<(), StorageError> {
            self.segments
                .lock()
                .unwrap()
                .push((path.clone(), data.to_vec()));
            Ok(())
        }

        async fn latest_manifest_version(
            &self,
            _manifest_prefix: &str,
        ) -> std::result::Result<Option<u64>, StorageError> {
            Ok(None)
        }

        async fn publish_snapshot(
            &self,
            _data_prefix: &ObjectPath,
            _manifest_prefix: &str,
            _parent: Option<u64>,
        ) -> std::result::Result<
            proximadb_storage_common::object_store_bridge::CommitOutcome,
            StorageError,
        > {
            use proximadb_storage_common::object_store_bridge::CommitOutcome;
            Ok(CommitOutcome::Committed(0))
        }
    }

    #[async_trait]
    impl TableWalAppender for RecordingWalAppender {
        async fn append_operations(
            &self,
            operations: Vec<CanonicalOperation>,
            tenant_id: Option<String>,
        ) -> Result<Vec<CanonicalWalEntry>> {
            let entries = operations
                .into_iter()
                .map(|operation| {
                    let sequence_number = self.next_sequence.fetch_add(1, Ordering::SeqCst) + 1;
                    CanonicalWalEntry::new(sequence_number, operation, tenant_id.clone())
                })
                .collect::<Vec<_>>();
            self.entries
                .lock()
                .expect("recording WAL append lock")
                .extend(entries.clone());
            Ok(entries)
        }
    }

    #[async_trait]
    impl RecordScan for MemoryRecordStorage {
        async fn scan_records(&self, limit: usize) -> Result<Vec<ProximaRecord>> {
            Ok(self
                .records
                .read()
                .expect("memory storage read lock")
                .values()
                .take(limit)
                .cloned()
                .collect())
        }
    }

    #[tokio::test]
    async fn record_storage_scan_records_filtered_pushes_predicate_and_table_membership() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let store = RecordStorageTableRecordStore::new(storage.clone());
        let schema = CatalogTableSchema::new("orders");

        for (oid, variation, version) in [
            ("keep-1", "orders", 1u64),
            ("keep-2", "orders", 2),
            ("skip-pred", "orders", 3), // belongs to table but fails predicate
            ("skip-table", "other", 1), // matches predicate but is another table
        ] {
            storage
                .upsert_record(ProximaRecord {
                    oid: oid.to_string(),
                    variation_id: Some(variation.to_string()),
                    record_version: version,
                    ..Default::default()
                })
                .await
                .unwrap();
        }

        // Predicate (the "WHERE"): oid starts with "keep". The store must apply
        // BOTH table membership (variation_id) AND the predicate.
        let pred = |r: &ProximaRecord| r.oid.starts_with("keep");
        let got = store
            .scan_records_filtered(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "orders".to_string(),
                    limit: None,
                    include_vector: true,
                    include_props: true,
                },
                Some(&pred),
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();

        let mut oids: Vec<_> = got.iter().map(|r| r.oid.clone()).collect();
        oids.sort();
        assert_eq!(
            oids,
            vec!["keep-1".to_string(), "keep-2".to_string()],
            "only rows that belong to the table AND match the predicate"
        );
    }

    /// TD-CONV-2: DROP TABLE must purge the record tier so a table recreated
    /// with the same name starts EMPTY. SQL-row record oids are the PK text, so
    /// without the purge the recreated table's first re-INSERT hits
    /// INSERT_CONFLICT against the dropped rows (or scans silently serve them).
    /// Restart-safety: the appended `RecordPartitionDrop` marker must void the
    /// partition on replay too.
    #[tokio::test]
    async fn drop_table_records_purges_partition_and_wal_marker_replays_the_drop() {
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_partitioned(wal.clone());
        let schema = CatalogTableSchema::new("region");
        let insert = |oid: &str| {
            TableRecordMutation::new(
                TableRecordMutationKind::Insert,
                ProximaRecord {
                    oid: oid.to_string(),
                    local_id: Some(oid.to_string()),
                    variation_id: Some("region".to_string()),
                    ..Default::default()
                },
            )
        };

        // First insert of PK '0' succeeds.
        let first = store
            .write_mutations(&schema, vec![insert("0")], None)
            .await
            .unwrap();
        assert!(first.success);

        // DROP TABLE purges the record tier…
        store.drop_table_records(&schema, None).await.unwrap();

        // …the drop marker is durable in the WAL…
        let entries = wal
            .entries
            .lock()
            .expect("recording WAL append lock")
            .clone();
        assert!(
            entries.iter().any(|e| matches!(
                &e.operation,
                CanonicalOperation::RecordPartitionDrop { collection_id } if collection_id == "region"
            )),
            "drop marker must be appended"
        );

        // …the recreated table's first re-INSERT of the same PK succeeds…
        let reinsert = store
            .write_mutations(&schema, vec![insert("0")], None)
            .await
            .unwrap();
        assert!(
            reinsert.success,
            "recreated table must not see the dropped rows: {:?}",
            reinsert.error_code
        );

        // …and replaying [upsert, drop] into a FRESH store leaves the
        // partition void (restart-safe: no resurrection).
        let replay_wal = Arc::new(RecordingWalAppender::default());
        let replayed = DirectWalTableRecordStore::new_partitioned(replay_wal);
        replayed.replay_wal_entries(entries).await.unwrap();
        let resurrect = replayed
            .write_mutations(&schema, vec![insert("0")], None)
            .await
            .unwrap();
        assert!(
            resurrect.success,
            "replayed drop must void the partition: {:?}",
            resurrect.error_code
        );
    }

    #[tokio::test]
    async fn record_storage_table_store_writes_reads_scans_and_deletes() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let store = RecordStorageTableRecordStore::new(storage);
        let schema = CatalogTableSchema::new("orders");
        let record = ProximaRecord {
            oid: "o1".to_string(),
            local_id: Some("o1".to_string()),
            variation_id: Some("orders".to_string()),
            ..Default::default()
        };

        let inserted = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record.clone(),
                )],
                None,
            )
            .await
            .unwrap();
        assert!(inserted.success);
        assert_eq!(inserted.record_ids, vec!["o1"]);

        let duplicate = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record.clone(),
                )],
                None,
            )
            .await
            .unwrap();
        assert!(!duplicate.success);
        assert_eq!(duplicate.error_code.as_deref(), Some("INSERT_CONFLICT"));

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "orders".to_string(),
                    key: "o1".to_string(),
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("record should exist");
        assert_eq!(fetched.id, "o1");

        let scanned = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "orders".to_string(),
                    limit: Some(10),
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();
        assert_eq!(scanned.len(), 1);

        let deleted = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Delete,
                    record,
                )],
                None,
            )
            .await
            .unwrap();
        assert!(deleted.success);
        assert!(
            store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: "orders".to_string(),
                        key: "o1".to_string(),
                        include_vector: true,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn direct_wal_table_store_appends_canonical_wal_before_storage_state() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage.clone(), wal.clone());
        let schema = CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(2, "amount", ProximaType::Float64))
            .with_storage_specialization(CatalogStorageSpecialization::PaxRowFamily);
        let record = ProximaRecord {
            oid: "o1".to_string(),
            local_id: Some("o1".to_string()),
            variation_id: Some("orders".to_string()),
            ..Default::default()
        };

        let inserted = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record.clone(),
                )],
                None,
            )
            .await
            .unwrap();

        assert!(inserted.success);
        assert_eq!(inserted.record_ids, vec!["o1"]);
        assert!(
            storage
                .get_record(&RecordKey::new("o1"))
                .await
                .unwrap()
                .is_some()
        );

        let entries = wal.entries.lock().expect("recording WAL read lock");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].sequence_number, 1);
        match &entries[0].operation {
            CanonicalOperation::RecordUpsert {
                collection_id,
                record,
                projections,
            } => {
                assert_eq!(collection_id, "orders");
                assert_eq!(record.oid, "o1");
                assert_eq!(projections.len(), 1);
                match &projections[0] {
                    ProjectionDirective::ColumnarVariation {
                        collection_id,
                        fields,
                    } => {
                        assert_eq!(collection_id, "orders");
                        assert_eq!(fields, &vec!["id".to_string(), "amount".to_string()]);
                    }
                    other => panic!("expected columnar/PAX projection directive, got {other:?}"),
                }
            }
            other => panic!("expected upsert WAL entry, got {other:?}"),
        }
    }

    /// TD-OLAP-17: `read_changes_since` skips the O(WAL) `read_all_entries` scan
    /// when the appender's per-collection high-water has not advanced past
    /// `since_lsn`. This is the fast-path that removes the DF table-OPEN floor
    /// (the per-table-per-query `changed_oids_since` scan). `MemoryTableWalAppender`
    /// tracks the high-water on append; `RecordingWalAppender` inherits the default
    /// (`None` ⇒ scan), so this test uses the memory appender.
    #[tokio::test]
    async fn read_changes_since_skips_scan_via_collection_high_water() {
        use crate::services::canonical_wal::MemoryTableWalAppender;

        let wal = Arc::new(MemoryTableWalAppender::new());
        let upsert = |coll: &str| CanonicalOperation::RecordUpsert {
            collection_id: coll.to_string(),
            record: Box::new(ProximaRecord::default()),
            projections: vec![],
        };
        // t1 → seq 1, 2 (high-water 2); t2 → seq 3 (high-water 3).
        wal.append_operations(vec![upsert("t1")], None)
            .await
            .unwrap();
        wal.append_operations(vec![upsert("t1")], None)
            .await
            .unwrap();
        wal.append_operations(vec![upsert("t2")], None)
            .await
            .unwrap();

        let store = DirectWalTableRecordStore::new_partitioned(wal);

        // Fast-path: high-water (2) <= since_lsn 2 ⇒ empty, no scan.
        assert!(store.read_changes_since("t1", 2).await.unwrap().is_empty());
        // Scan: high-water (2) > since_lsn 1 ⇒ the seq-2 entry.
        assert_eq!(store.read_changes_since("t1", 1).await.unwrap().len(), 1);
        // Fast-path for t2.
        assert!(store.read_changes_since("t2", 3).await.unwrap().is_empty());
        assert_eq!(store.read_changes_since("t2", 2).await.unwrap().len(), 1);
        // Unknown collection ⇒ no high-water ⇒ scan ⇒ empty.
        assert!(
            store
                .read_changes_since("never-written", 0)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn direct_wal_table_store_rejects_duplicate_insert_without_wal_append() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage, wal.clone());
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let record = ProximaRecord {
            oid: "o1".to_string(),
            variation_id: Some("orders".to_string()),
            ..Default::default()
        };

        assert!(
            store
                .write_mutations(
                    &schema,
                    vec![TableRecordMutation::new(
                        TableRecordMutationKind::Insert,
                        record.clone(),
                    )],
                    None,
                )
                .await
                .unwrap()
                .success
        );
        let duplicate = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record,
                )],
                None,
            )
            .await
            .unwrap();

        assert!(!duplicate.success);
        assert_eq!(duplicate.error_code.as_deref(), Some("INSERT_CONFLICT"));
        assert_eq!(
            wal.entries.lock().expect("recording WAL read lock").len(),
            1
        );
    }

    // ── TD-127: OLTP secondary index (build / probe / IN-list / maintenance) ──────

    /// Build a `code_symbol`-shaped schema declaring non-unique secondary indexes
    /// on `name` and `file` (the code-graph lookup columns) plus an unindexed
    /// `lang` column, PK `oid`.
    fn secondary_index_schema() -> CatalogTableSchema {
        use proximadb_catalog::{CatalogIndex, CatalogIndexType, RelationalCapabilities};
        CatalogTableSchema::new("code_symbol")
            .with_column(CatalogColumn::new(1, "oid", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(2, "name", ProximaType::String))
            .with_column(CatalogColumn::new(3, "file", ProximaType::String))
            .with_column(CatalogColumn::new(4, "lang", ProximaType::String))
            .with_primary_key(vec!["oid".to_string()])
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp)
            .with_relational_capabilities(RelationalCapabilities {
                primary_key: vec!["oid".to_string()],
                secondary_indexes: vec![
                    CatalogIndex::new(
                        "sym_name_idx",
                        vec!["name".to_string()],
                        CatalogIndexType::Hash,
                    ),
                    CatalogIndex::new(
                        "sym_file_idx",
                        vec!["file".to_string()],
                        CatalogIndexType::Hash,
                    ),
                ],
                ..Default::default()
            })
    }

    fn symbol_record(oid: &str, name: &str, file: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.to_string(),
            local_id: Some(oid.to_string()),
            variation_id: Some("code_symbol".to_string()),
            props: proximadb_records::ProximaTree::from([
                (
                    "name".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(name.to_string())),
                ),
                (
                    "file".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(file.to_string())),
                ),
            ]),
            ..Default::default()
        }
    }

    async fn write_one(
        store: &DirectWalTableRecordStore,
        schema: &CatalogTableSchema,
        kind: TableRecordMutationKind,
        record: ProximaRecord,
    ) {
        let result = store
            .write_mutations(schema, vec![TableRecordMutation::new(kind, record)], None)
            .await
            .expect("write_mutations");
        assert!(result.success, "write failed: {:?}", result.error_code);
    }

    /// Probe `column IN values` and return the matching oids, sorted.
    async fn probe_sorted(
        store: &DirectWalTableRecordStore,
        schema: &CatalogTableSchema,
        column: &str,
        values: &[&str],
    ) -> Option<Vec<String>> {
        let set: std::collections::HashSet<String> = values.iter().map(|v| v.to_string()).collect();
        store
            .lookup_secondary(schema, column, &set, None)
            .await
            .expect("lookup_secondary")
            .map(|mut oids| {
                oids.sort();
                oids
            })
    }

    #[tokio::test]
    async fn secondary_index_probes_equality_and_in_list() {
        let store = DirectWalTableRecordStore::new(
            Arc::new(MemoryRecordStorage::default()),
            Arc::new(RecordingWalAppender::default()),
        );
        let schema = secondary_index_schema();
        for (oid, name, file) in [
            ("s1", "parse", "a.rs"),
            ("s2", "parse", "b.rs"), // same name in another file
            ("s3", "emit", "b.rs"),
            ("s4", "emit", "c.rs"),
        ] {
            write_one(
                &store,
                &schema,
                TableRecordMutationKind::Insert,
                symbol_record(oid, name, file),
            )
            .await;
        }

        // Equality on the indexed `name` column → both `parse` symbols.
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["parse"]).await,
            Some(vec!["s1".to_string(), "s2".to_string()])
        );
        // IN-list on the indexed `file` column → union of a.rs + c.rs.
        assert_eq!(
            probe_sorted(&store, &schema, "file", &["a.rs", "c.rs"]).await,
            Some(vec!["s1".to_string(), "s4".to_string()])
        );
        // A value with no rows → an empty (but present) answer, NOT a scan fallback.
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["missing"]).await,
            Some(vec![])
        );
        // An unindexed column → `None` so the caller scans.
        assert_eq!(probe_sorted(&store, &schema, "lang", &["rust"]).await, None);
    }

    #[tokio::test]
    async fn secondary_index_maintained_on_update_and_delete() {
        let store = DirectWalTableRecordStore::new(
            Arc::new(MemoryRecordStorage::default()),
            Arc::new(RecordingWalAppender::default()),
        );
        let schema = secondary_index_schema();
        write_one(
            &store,
            &schema,
            TableRecordMutationKind::Insert,
            symbol_record("s1", "parse", "a.rs"),
        )
        .await;
        // First probe builds the index from current state.
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["parse"]).await,
            Some(vec!["s1".to_string()])
        );

        // UPDATE the indexed value: the OLD value must stop matching, the NEW one start.
        write_one(
            &store,
            &schema,
            TableRecordMutationKind::Update,
            symbol_record("s1", "lex", "a.rs"),
        )
        .await;
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["parse"]).await,
            Some(vec![]),
            "old indexed value detached on update"
        );
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["lex"]).await,
            Some(vec!["s1".to_string()]),
            "new indexed value attached on update"
        );

        // DELETE removes the oid from the index entirely.
        write_one(
            &store,
            &schema,
            TableRecordMutationKind::Delete,
            symbol_record("s1", "lex", "a.rs"),
        )
        .await;
        assert_eq!(
            probe_sorted(&store, &schema, "name", &["lex"]).await,
            Some(vec![])
        );
    }

    #[tokio::test]
    async fn secondary_index_kill_switch_disables_probe() {
        // SAFETY: single-threaded test; set + remove the process env around the probe.
        unsafe { std::env::set_var("PROXIMADB_SECONDARY_INDEX_DISABLE", "1") };
        let store = DirectWalTableRecordStore::new(
            Arc::new(MemoryRecordStorage::default()),
            Arc::new(RecordingWalAppender::default()),
        );
        let schema = secondary_index_schema();
        write_one(
            &store,
            &schema,
            TableRecordMutationKind::Insert,
            symbol_record("s1", "parse", "a.rs"),
        )
        .await;
        // Kill-switch on → `None` (scan fallback), even for an indexed column.
        let result = probe_sorted(&store, &schema, "name", &["parse"]).await;
        unsafe { std::env::remove_var("PROXIMADB_SECONDARY_INDEX_DISABLE") };
        assert_eq!(result, None, "kill-switch forces the scan fallback");
    }

    // ── Retirement gate #2: prior-version closure and tombstone visibility ─────────
    //
    // These tests verify that below all protocol facades, the canonical write path
    // (DirectWalTableRecordStore) correctly hides deleted records from scans and
    // point-lookups, replaces old field values after an update, and allows re-insert
    // after delete without prior-version interference.

    #[tokio::test]
    async fn tombstone_is_invisible_to_scan_after_delete() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage, wal);
        let schema = CatalogTableSchema::new("events")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);

        for id in ["e1", "e2", "e3"] {
            store
                .write_mutations(
                    &schema,
                    vec![TableRecordMutation::new(
                        TableRecordMutationKind::Upsert,
                        ProximaRecord {
                            oid: id.to_string(),
                            variation_id: Some("events".to_string()),
                            ..Default::default()
                        },
                    )],
                    None,
                )
                .await
                .unwrap();
        }

        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Delete,
                    ProximaRecord {
                        oid: "e2".to_string(),
                        variation_id: Some("events".to_string()),
                        ..Default::default()
                    },
                )],
                None,
            )
            .await
            .unwrap();

        let rows = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "events".to_string(),
                    limit: Some(10),
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();

        let ids: Vec<&str> = rows.iter().map(|r| r.oid.as_str()).collect();
        assert!(
            !ids.contains(&"e2"),
            "tombstoned record e2 must not appear in scan; got: {ids:?}"
        );
        assert_eq!(rows.len(), 2, "only e1 and e3 should remain; got: {ids:?}");
    }

    #[tokio::test]
    async fn prior_version_is_closed_after_update() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage, wal);
        let schema = CatalogTableSchema::new("users")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);

        let original = ProximaRecord {
            oid: "u1".to_string(),
            variation_id: Some("users".to_string()),
            local_id: Some("alice".to_string()),
            props: [(
                "display_name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("alice".to_string())),
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    original,
                )],
                None,
            )
            .await
            .unwrap();

        let updated = ProximaRecord {
            oid: "u1".to_string(),
            variation_id: Some("users".to_string()),
            local_id: Some("alice-renamed".to_string()),
            props: [(
                "display_name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("alice-renamed".to_string())),
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let result = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Update,
                    updated,
                )],
                None,
            )
            .await
            .unwrap();
        assert!(result.success, "update must succeed");

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "users".to_string(),
                    key: "u1".to_string(),
                    include_vector: false,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("record must exist after update");

        assert_eq!(
            fetched.props.get("display_name"),
            Some(&ProximaValue::String("alice-renamed".to_string())),
            "prior version (alice) must be closed; only updated version visible"
        );
    }

    #[tokio::test]
    async fn re_insert_succeeds_after_delete_no_prior_version_interference() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage, wal);
        let schema = CatalogTableSchema::new("items")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);

        let record = ProximaRecord {
            oid: "i1".to_string(),
            variation_id: Some("items".to_string()),
            local_id: Some("first-generation".to_string()),
            ..Default::default()
        };
        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record,
                )],
                None,
            )
            .await
            .unwrap();

        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Delete,
                    ProximaRecord {
                        oid: "i1".to_string(),
                        variation_id: Some("items".to_string()),
                        ..Default::default()
                    },
                )],
                None,
            )
            .await
            .unwrap();

        let re_insert = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    ProximaRecord {
                        oid: "i1".to_string(),
                        variation_id: Some("items".to_string()),
                        local_id: Some("second-generation".to_string()),
                        props: [(
                            "generation".to_string(),
                            ProximaTreeNode::Value(ProximaValue::String(
                                "second-generation".to_string(),
                            )),
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )],
                None,
            )
            .await
            .unwrap();
        assert!(
            re_insert.success,
            "re-insert after delete must succeed without INSERT_CONFLICT"
        );

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "items".to_string(),
                    key: "i1".to_string(),
                    include_vector: false,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("re-inserted record must be visible");
        assert_eq!(
            fetched.props.get("generation"),
            Some(&ProximaValue::String("second-generation".to_string())),
            "re-inserted record must carry second-generation data, not prior-version ghost"
        );
    }

    #[tokio::test]
    async fn vector_bearing_pax_table_writes_canonical_without_vector_ops() {
        // Gate 4: vector-bearing relational tables write embeddings into the canonical
        // WAL path without VectorOps being the mutation owner.
        use proximadb_records::EmbeddingCell;

        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage.clone(), wal.clone());
        let schema = CatalogTableSchema::new("products")
            .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(
                2,
                "embedding",
                ProximaType::DenseVector {
                    element: VectorElement::Float32,
                    dim: 0,
                },
            ))
            .with_storage_specialization(CatalogStorageSpecialization::PaxRowFamily);

        let record = ProximaRecord {
            oid: "p1".to_string(),
            variation_id: Some("products".to_string()),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            ..Default::default()
        };

        let result = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Insert,
                    record,
                )],
                None,
            )
            .await
            .unwrap();
        assert!(result.success, "vector-bearing insert must succeed");

        // Scope the WAL lock guard so it provably drops before the `get_by_key`
        // await below (clippy::await_holding_lock): a std Mutex held across an
        // await would block the runtime if contended. The block end drops the
        // guard; no explicit `drop` needed.
        {
            let entries = wal.entries.lock().expect("recording WAL read lock");
            assert_eq!(entries.len(), 1);
            match &entries[0].operation {
                CanonicalOperation::RecordUpsert { record, .. } => {
                    assert_eq!(record.embeddings.len(), 1);
                    assert_eq!(
                        record.embeddings[0].values,
                        proximadb_records::EmbeddingValues::Fp32(vec![0.1_f32, 0.2, 0.3, 0.4])
                    );
                }
                other => panic!("expected RecordUpsert WAL entry, got {other:?}"),
            }
        }

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "products".to_string(),
                    key: "p1".to_string(),
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("inserted record must be visible");
        assert_eq!(fetched.vector, vec![0.1_f32, 0.2, 0.3, 0.4]);
    }

    #[tokio::test]
    async fn direct_wal_table_store_delete_writes_canonical_delete_entry() {
        let storage = Arc::new(MemoryRecordStorage::default());
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new(storage.clone(), wal.clone());
        let schema = CatalogTableSchema::new("orders")
            .with_storage_specialization(CatalogStorageSpecialization::PaxOltp);
        let record = ProximaRecord {
            oid: "o1".to_string(),
            variation_id: Some("orders".to_string()),
            ..Default::default()
        };

        store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Upsert,
                    record.clone(),
                )],
                None,
            )
            .await
            .unwrap();
        let deleted = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Delete,
                    record,
                )],
                None,
            )
            .await
            .unwrap();

        assert!(deleted.success);
        assert!(
            storage
                .get_record(&RecordKey::new("o1"))
                .await
                .unwrap()
                .is_none()
        );
        let entries = wal.entries.lock().expect("recording WAL read lock");
        assert_eq!(entries.len(), 2);
        match &entries[1].operation {
            CanonicalOperation::RecordDelete {
                collection_id, oid, ..
            } => {
                assert_eq!(collection_id, "orders");
                assert_eq!(oid, "o1");
            }
            other => panic!("expected delete WAL entry, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn object_store_iceberg_record_store_writes_parquet_via_bridge() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        let store = ObjectStoreIcebergRecordStore::new(bridge.clone());
        let schema = CatalogTableSchema::new("orders").with_storage_layout(
            CatalogStorageLayout::projection_publication(
                "primary",
                CatalogPhysicalFormat::Iceberg,
                "warehouse/orders",
            ),
        );
        let records = vec![
            ProximaRecord {
                oid: "o1".to_string(),
                variation_id: Some("orders".to_string()),
                ..Default::default()
            },
            ProximaRecord {
                oid: "o2".to_string(),
                variation_id: Some("orders".to_string()),
                ..Default::default()
            },
        ];

        let result = store
            .write_mutations(
                &schema,
                records
                    .into_iter()
                    .map(|record| TableRecordMutation::new(TableRecordMutationKind::Insert, record))
                    .collect(),
                None,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.record_ids, vec!["o1".to_string(), "o2".to_string()]);
        let writes = bridge.writes.lock().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].1, vec!["o1".to_string(), "o2".to_string()]);
        assert!(
            writes[0]
                .0
                .as_ref()
                .starts_with("data/default_tenant/ns_default/orders/data/orders-insert-")
        );
        assert!(writes[0].0.as_ref().ends_with(".parquet"));
    }

    #[tokio::test]
    async fn object_store_vector_record_store_persists_pax_segment_via_bridge() {
        let bridge = Arc::new(CapturingObjectStoreBridge::new());
        let store = ObjectStoreVectorRecordStore::new(bridge.clone());
        let schema = CatalogTableSchema::new("vectors")
            .with_workload_profile(CatalogWorkloadProfile::Vector)
            .with_storage_specialization(CatalogStorageSpecialization::VectorAnn);
        let records = vec![ProximaRecord {
            oid: "v1".to_string(),
            variation_id: Some("vectors".to_string()),
            ..Default::default()
        }];

        let result = store
            .write_mutations(
                &schema,
                records
                    .into_iter()
                    .map(|record| TableRecordMutation::new(TableRecordMutationKind::Upsert, record))
                    .collect(),
                None,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.record_ids, vec!["v1".to_string()]);
        assert!(bridge.writes.lock().unwrap().is_empty());
        let segments = bridge.segments.lock().unwrap();
        assert_eq!(segments.len(), 1);
        assert!(
            segments[0]
                .0
                .as_ref()
                .starts_with("data/default_tenant/ns_default/vectors/segments/vectors-upsert-")
        );
        assert!(segments[0].0.as_ref().ends_with(PAX_SEGMENT_EXT));
        assert!(
            segments[0].1.ends_with(SEGMENT_MAGIC),
            "persisted bytes must be a PAX segment"
        );
    }

    fn props_record(oid: &str, table: &str, props: Vec<(&str, ProximaValue)>) -> ProximaRecord {
        let mut tree = HashMap::new();
        for (key, value) in props {
            tree.insert(key.to_string(), ProximaTreeNode::Value(value));
        }
        ProximaRecord {
            oid: oid.to_string(),
            variation_id: Some(table.to_string()),
            props: tree,
            ..Default::default()
        }
    }

    /// F3 relational round trip: write records through `ObjectStoreIcebergRecordStore`
    /// over a real `IcebergObjectStoreBridge` (in-memory object store) and read
    /// them back via the Parquet read path — props + dense vector must survive,
    /// and the primary key must be recovered as the record `oid`.
    #[tokio::test]
    async fn iceberg_record_store_round_trips_through_object_store() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;

        let bridge: Arc<dyn ObjectStoreBridge> =
            Arc::new(IcebergObjectStoreBridge::from_url("memory://").unwrap());
        let store = ObjectStoreIcebergRecordStore::new(bridge);
        let mut schema = CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "id", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(2, "amount", ProximaType::Int64));
        schema.primary_key = vec!["id".to_string()];

        let mut r0 = props_record(
            "o1",
            "orders",
            vec![
                ("id", ProximaValue::String("o1".into())),
                ("amount", ProximaValue::Int64(30)),
            ],
        );
        r0.embeddings
            .push(EmbeddingCell::new_fp32("", "", 3, vec![1.0, 2.0, 3.0]));
        let r1 = props_record(
            "o2",
            "orders",
            vec![
                ("id", ProximaValue::String("o2".into())),
                ("amount", ProximaValue::Int64(41)),
            ],
        );

        let written = store
            .write_mutations(
                &schema,
                vec![
                    TableRecordMutation::new(TableRecordMutationKind::Insert, r0),
                    TableRecordMutation::new(TableRecordMutationKind::Insert, r1),
                ],
                None,
            )
            .await
            .unwrap();
        assert!(written.success);

        let mut scanned = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "orders".to_string(),
                    limit: None,
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();
        scanned.sort_by(|a, b| a.oid.cmp(&b.oid));
        assert_eq!(scanned.len(), 2, "both written records must read back");
        assert_eq!(scanned[0].oid, "o1", "primary key recovered as oid");
        assert_eq!(scanned[1].oid, "o2");
        assert_eq!(
            proximadb_records::tree_get(&scanned[0].props, "amount"),
            Some(&ProximaValue::Int64(30))
        );
        assert_eq!(
            scanned[0]
                .embeddings
                .first()
                .map(|e| e.values.to_fp32_owned()),
            Some(vec![1.0, 2.0, 3.0]),
            "dense vector must round-trip through Parquet"
        );

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "orders".to_string(),
                    key: "o2".to_string(),
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("get_by_key must find the persisted record");
        assert_eq!(fetched.id, "o2");
        assert_eq!(fetched.props.get("amount"), Some(&ProximaValue::Int64(41)));
    }

    /// F3 vector round trip: write a vector-bearing record through
    /// `ObjectStoreVectorRecordStore` over a real `IcebergObjectStoreBridge` and
    /// read it back via the PAX segment fetch path — `oid` and the dense
    /// embedding (the vector store's canonical payload) must survive.
    #[tokio::test]
    async fn vector_record_store_round_trips_pax_segment_through_object_store() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;

        let bridge: Arc<dyn ObjectStoreBridge> =
            Arc::new(IcebergObjectStoreBridge::from_url("memory://").unwrap());
        let store = ObjectStoreVectorRecordStore::new(bridge);
        let schema = CatalogTableSchema::new("vectors")
            .with_workload_profile(CatalogWorkloadProfile::Vector)
            .with_storage_specialization(CatalogStorageSpecialization::VectorAnn);

        let mut props = HashMap::new();
        props.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("books".into())),
        );
        let record = ProximaRecord {
            oid: "v1".to_string(),
            variation_id: Some("vectors".to_string()),
            created_at_ns: 1_700_000_000_000_000_000,
            props,
            embeddings: vec![EmbeddingCell::new_fp32(
                "",
                "dense",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            ..Default::default()
        };

        let written = store
            .write_mutations(
                &schema,
                vec![TableRecordMutation::new(
                    TableRecordMutationKind::Upsert,
                    record,
                )],
                None,
            )
            .await
            .unwrap();
        assert!(written.success);

        let scanned = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "vectors".to_string(),
                    limit: None,
                    include_vector: true,
                    include_props: true,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap();
        assert_eq!(scanned.len(), 1, "the persisted record must read back");
        assert_eq!(scanned[0].oid, "v1", "oid must round-trip through PAX");
        // PAX v2 stores dense vectors with SQ8 scalar quantization (lossy, 4x), so
        // the embedding reconstructs within one quantization step rather than bit-exactly.
        let got = scanned[0]
            .embeddings
            .first()
            .map(|e| e.values.to_fp32_owned())
            .expect("dense embedding must round-trip through the PAX segment");
        let expected = [0.1_f32, 0.2, 0.3, 0.4];
        assert_eq!(got.len(), expected.len(), "embedding dim must round-trip");
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() <= 0.02,
                "dense embedding must round-trip within SQ8 tolerance: got {got:?} vs {expected:?}"
            );
        }
        // Phase B: props + timestamps now round-trip (was oid+embedding only).
        assert_eq!(
            proximadb_records::tree_get(&scanned[0].props, "category"),
            Some(&ProximaValue::String("books".into())),
            "props must round-trip through the PAX segment"
        );
        assert_eq!(
            scanned[0].created_at_ns, 1_700_000_000_000_000_000,
            "created_at must round-trip through the PAX segment"
        );

        let fetched = store
            .get_by_key(
                &schema,
                TableRecordGetRequest {
                    table_id: "vectors".to_string(),
                    key: "v1".to_string(),
                    include_vector: true,
                    include_props: false,
                },
                None,
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .unwrap()
            .expect("get_by_key must find the persisted vector record");
        // SQ8-quantized reconstruction: compare within one quantization step.
        let expected = [0.1_f32, 0.2, 0.3, 0.4];
        assert_eq!(
            fetched.vector.len(),
            expected.len(),
            "vector dim must round-trip"
        );
        for (g, e) in fetched.vector.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() <= 0.02,
                "get_by_key vector must round-trip within SQ8 tolerance: got {:?} vs {expected:?}",
                fetched.vector
            );
        }
    }

    /// P-Trace (TD-DOC-CONV-2 / ADR-054): MEASURE the pre-shred document I/O baseline. A
    /// representative document corpus (rich, varied `props` — the metadata a hybrid-shred format
    /// would promote to typed columns) is persisted as ONE PAX segment, then each read op runs
    /// inside an `io_trace::scope` so the task-local trace is captured. This records the co-design
    /// baseline the shredding + pushdown work (ADR-054 P-Shred/P-Pushdown) will improve: TODAY every
    /// document read — point-get AND filtered scan alike — FETCHES THE WHOLE SEGMENT (no pushdown,
    /// no ranged reads), and the write records its segment bytes (the new `put_pax` hook).
    /// `range_gets` is gated behind the `io-trace` feature; `bytes_read`/`bytes_written` are always-on.
    #[tokio::test]
    async fn doc_iotrace_baseline_preshred() {
        use proximadb_iceberg_engine::IcebergObjectStoreBridge;

        let bridge: Arc<dyn ObjectStoreBridge> =
            Arc::new(IcebergObjectStoreBridge::from_url("memory://").unwrap());
        let store = ObjectStoreVectorRecordStore::new(bridge);
        let schema = CatalogTableSchema::new("docs")
            .with_workload_profile(CatalogWorkloadProfile::Vector)
            .with_storage_specialization(CatalogStorageSpecialization::VectorAnn);

        const N: usize = 500;
        let mut muts = Vec::with_capacity(N);
        for i in 0..N {
            let mut props = HashMap::new();
            props.insert(
                "title".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!(
                    "Document number {i} — a representative title with several words"
                ))),
            );
            props.insert(
                "status".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(
                    if i % 3 == 0 { "active" } else { "archived" }.to_string(),
                )),
            );
            props.insert(
                "day".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int64((i % 365) as i64)),
            );
            props.insert(
                "author".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("author_{}", i % 50))),
            );
            props.insert(
                "body".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("lorem ipsum dolor sit ".repeat(16))),
            );
            let record = ProximaRecord {
                oid: format!("doc-{i}"),
                variation_id: Some("docs".to_string()),
                created_at_ns: 1_700_000_000_000_000_000 + i as i64,
                props,
                embeddings: vec![EmbeddingCell::new_fp32(
                    "",
                    "dense",
                    8,
                    (0..8).map(|d| ((i + d) as f32).sin()).collect(),
                )],
                ..Default::default()
            };
            muts.push(TableRecordMutation::new(
                TableRecordMutationKind::Upsert,
                record,
            ));
        }

        // WRITE — the whole-segment flush; captures bytes_written (the new put_pax hook).
        let write_snap = io_trace::scope(async {
            let w = store
                .write_mutations(&schema, muts, None)
                .await
                .expect("write_mutations");
            assert!(w.success);
            io_trace::snapshot().expect("snapshot in scope")
        })
        .await;

        // POINT-GET — one record, but the WHOLE segment is fetched (fetch_pax).
        let get_snap = io_trace::scope(async {
            let got = store
                .get_by_key(
                    &schema,
                    TableRecordGetRequest {
                        table_id: "docs".to_string(),
                        key: "doc-250".to_string(),
                        include_vector: false,
                        include_props: true,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .expect("get_by_key");
            assert!(got.is_some(), "point-get must find the record");
            io_trace::snapshot().expect("snapshot in scope")
        })
        .await;

        // FILTERED SCAN — the query_documents baseline; filter is NOT pushed down, so the whole
        // segment is fetched and filtered in memory.
        let scan_snap = io_trace::scope(async {
            let scanned = store
                .scan_records(
                    &schema,
                    TableRecordScanRequest {
                        table_id: "docs".to_string(),
                        limit: None,
                        include_vector: false,
                        include_props: true,
                        filter: None,
                    },
                    None,
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .expect("scan_records");
            assert_eq!(scanned.len(), N, "all docs read back");
            io_trace::snapshot().expect("snapshot in scope")
        })
        .await;

        eprintln!("BAKE_METRIC doc_trace.corpus_docs={N}");
        eprintln!(
            "BAKE_METRIC doc_trace.write.bytes_written={} put_ops={}",
            write_snap.bytes_written, write_snap.put_ops
        );
        eprintln!(
            "BAKE_METRIC doc_trace.point_get.bytes_read={} get_ops={} range_gets={}",
            get_snap.bytes_read, get_snap.get_ops, get_snap.range_gets
        );
        eprintln!(
            "BAKE_METRIC doc_trace.filtered_scan.bytes_read={} get_ops={} range_gets={}",
            scan_snap.bytes_read, scan_snap.get_ops, scan_snap.range_gets
        );
        // The pre-shred baseline: a point-get fetches the SAME whole segment a full scan does.
        eprintln!(
            "BAKE_METRIC doc_trace.point_get_reads_whole_segment={}",
            get_snap.bytes_read == scan_snap.bytes_read && get_snap.bytes_read > 0
        );

        assert!(
            write_snap.bytes_written > 0,
            "flush must record bytes_written (the new put_pax hook)"
        );
        assert!(
            get_snap.bytes_read > 0,
            "point-get must record bytes_read (fetch_pax)"
        );
        assert!(
            scan_snap.bytes_read > 0,
            "filtered scan must record bytes_read (fetch_pax)"
        );
    }

    /// TD-USUB-1 slice 2a wiring: the factory must RECEIVE the partition
    /// identity. `partition()` always had `(tenant_id, collection)` in hand and
    /// discarded it, which is exactly why a durable partition store — which must
    /// derive its own `DrPathBuilder` prefix — could not be wired at all.
    ///
    /// Asserts the identity that actually arrives, not merely that the factory
    /// ran: a factory called with the wrong scope would still produce a working
    /// store, and would silently co-mingle two tenants' segments under one path.
    #[tokio::test]
    async fn storage_factory_receives_the_partition_identity() {
        let seen: Arc<parking_lot::Mutex<Vec<(String, String)>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let recorder = seen.clone();

        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::with_storage_factory(
            wal,
            Arc::new(move |tenant_id: &str, collection: &str| {
                recorder
                    .lock()
                    .push((tenant_id.to_string(), collection.to_string()));
                Arc::new(crate::services::MemtableRecordStorage::new()) as Arc<dyn RecordStorage>
            }),
        );

        // Seed through WAL replay deliberately: recovery is the path that reaches
        // `partition()` carrying ONLY the entry's tenant_id + collection_id, and
        // it is the path a durable store's identity has to survive.
        store
            .replay_wal_entries(vec![CanonicalWalEntry::new(
                1,
                CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(ProximaRecord {
                        oid: "r1".to_string(),
                        variation_id: Some("orders".to_string()),
                        ..Default::default()
                    }),
                    projections: vec![],
                },
                Some("tenant-a".to_string()),
            )])
            .await
            .expect("replay");

        let calls = seen.lock().clone();
        assert!(
            calls.contains(&("tenant-a".to_string(), "orders".to_string())),
            "the factory must be handed the partition's (tenant, collection); got {calls:?}"
        );
    }

    /// Mandate #8: with the gate off the path must be the LEGACY one.
    ///
    /// The observable form of that claim is that nothing is written to object
    /// storage at all — not that a spill store was constructed with flushing
    /// disabled. `new_spilling` builds a `MemtableRecordStorage` when the gate is
    /// absent precisely so the default path runs the same code it always did,
    /// rather than different code that currently behaves the same.
    #[tokio::test]
    async fn spilling_store_writes_no_segments_when_the_gate_is_off() {
        unsafe { std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "0") };

        let (root, base) = spill_root();
        let fs = local_fs().await;

        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_spilling(wal, fs, tenant_path_resolver(base));

        let entries = upsert_entries(32, "orders", "tenant-a");
        store.replay_wal_entries(entries).await.expect("replay");

        // `expect`, not `unwrap_or(false)`: an unreadable root must fail the test,
        // not read as "wrote nothing" and pass it.
        let wrote_anything = std::fs::read_dir(&root)
            .expect("read spill root")
            .next()
            .is_some();
        assert!(
            !wrote_anything,
            "gate off must write no segments; found entries under {}",
            root.display()
        );
        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// ADR-097 P3.1: a spilling store's Layer A snapshot names the segments its
    /// partition flushed and the filesystem that wrote them, with the unflushed
    /// rows resident — and together they hold every row, none read twice.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn layer_a_snapshot_lists_flushed_segments_and_resident_rows() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }
        let (_, base) = spill_root();
        let fs = local_fs().await;
        let store = DirectWalTableRecordStore::new_spilling(
            Arc::new(RecordingWalAppender::default()),
            fs,
            tenant_path_resolver(base),
        );
        let entries: Vec<CanonicalWalEntry> = (0..6)
            .map(|i| {
                CanonicalWalEntry::new(
                    i + 1,
                    CanonicalOperation::RecordUpsert {
                        collection_id: "orders".to_string(),
                        record: Box::new(ProximaRecord {
                            oid: format!("r{i}"),
                            tenant_id: "tenant-a".to_string(),
                            variation_id: Some("orders".to_string()),
                            ..Default::default()
                        }),
                        projections: vec![],
                    },
                    Some("tenant-a".to_string()),
                )
            })
            .collect();
        store.replay_wal_entries(entries).await.expect("replay");
        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };

        let snapshot = store
            .layer_a_snapshot(
                &CatalogTableSchema::new("orders"),
                Some(&TenantContext::for_tenant_id("tenant-a")),
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .expect("snapshot")
            .expect("a spilling store offers a snapshot");
        assert_eq!(snapshot.table_name, "orders");
        assert!(
            !snapshot.view.segments.is_empty(),
            "rows past the threshold flushed"
        );
        let filesystem = snapshot
            .filesystem
            .clone()
            .expect("segments come with their filesystem");

        let mut oids: Vec<String> = snapshot
            .view
            .resident
            .iter()
            .map(|r| r.oid.clone())
            .collect();
        for path in &snapshot.view.segments {
            let bytes = filesystem.read(path).await.expect("segment");
            let rows = crate::storage::engines::sst::segment_format::read_segment_records(
                &bytes,
                &[],
                &[],
                None,
            )
            .expect("decode");
            oids.extend(rows.into_iter().map(|r| r.oid));
        }
        oids.sort();
        assert_eq!(oids, ["r0", "r1", "r2", "r3", "r4", "r5"]);
    }

    /// The UNTENANTED partition spills too, and persists its deletes.
    ///
    /// `tenant_key` and the replay path are both `unwrap_or_default()`, so `""`
    /// reaches the factory — and it is the normal key on a single-tenant
    /// server. It used to get an unbounded memtable, because a spill store
    /// without a tenant could not persist a tombstone. The factory now builds it
    /// `untenanted()`: its rows and its tombstones both carry `""`, so the heap
    /// is bounded and a delete leaves memory like any other.
    ///
    /// Asserted by OBSERVABLE effect: past the threshold a segment exists under
    /// the root, every row is still served, a delete is suppressed, and after a
    /// flush nothing constrains WAL truncation.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    /// Boot seeding reaches a partition WAL replay creates, before any write,
    /// so its DECIMAL column is typed from the catalog. A collection key two
    /// different schemas claim gets no fallback (the catalog cannot say which
    /// tenant owns which) — its rows are never encoded under a guessed schema;
    /// they flush through inference plus the residual, still exact.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn boot_seeded_schemas_reach_replayed_partitions_unless_ambiguous() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "100");
        }
        let (_, base) = spill_root();
        let fs = local_fs().await;
        let store = DirectWalTableRecordStore::new_spilling(
            Arc::new(RecordingWalAppender::default()),
            fs,
            tenant_path_resolver(base),
        );
        let decimal = |table: &str| {
            CatalogTableSchema::new(table).with_column(proximadb_catalog::CatalogColumn::new(
                1,
                "price",
                ProximaType::Decimal {
                    precision: 10,
                    scale: 2,
                },
            ))
        };
        let other_shape = CatalogTableSchema::new("shared").with_column(
            proximadb_catalog::CatalogColumn::new(1, "price", ProximaType::String),
        );
        let schemas = [decimal("orders"), decimal("shared"), other_shape];
        assert!(store.seed_schema_hints(&schemas) >= 3);

        let priced = |lsn: u64, table: &str| {
            let mut record = ProximaRecord {
                oid: format!("{table}-{lsn}"),
                tenant_id: "tenant-a".to_string(),
                ..Default::default()
            };
            record.props.insert(
                "price".to_string(),
                proximadb_records::ProximaTreeNode::Value(ProximaValue::Decimal("1.50".into())),
            );
            CanonicalWalEntry::new(
                lsn,
                CanonicalOperation::RecordUpsert {
                    collection_id: table.to_string(),
                    record: Box::new(record),
                    projections: vec![],
                },
                Some("tenant-a".to_string()),
            )
        };
        store
            .replay_wal_entries(vec![priced(1, "orders"), priced(2, "shared")])
            .await
            .expect("replay");
        store.flush_partitions().await.expect("flush");

        let orders = store.partition("tenant-a", "orders");
        assert_eq!(
            RecordStore::min_unflushed_lsn(orders.as_ref()).await.ok(),
            Some(None),
            "the seeded schema typed the replayed DECIMAL rows, so they flushed"
        );
        let hints = store.schema_hints.as_ref().expect("spilling store");
        assert!(
            hints.get("tenant-a", "orders").is_some(),
            "the unambiguous key has a fallback"
        );
        assert!(
            hints.get("tenant-a", "shared").is_none(),
            "an ambiguous key gets no fallback, so no schema is guessed"
        );
        let shared = store.partition("tenant-a", "shared");
        assert_eq!(
            RecordStore::min_unflushed_lsn(shared.as_ref()).await.ok(),
            Some(None),
            "it still flushes, through inference plus the residual"
        );
        let back = shared
            .get_record(&RecordKey::new("shared-2".to_string()))
            .await
            .expect("read")
            .expect("row");
        assert_eq!(
            back.props.get("price"),
            Some(&proximadb_records::ProximaTreeNode::Value(
                ProximaValue::Decimal("1.50".into())
            )),
            "exact"
        );

        unsafe {
            std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV);
        }
    }

    /// The production factory encodes with the table's catalog schema: the
    /// schema `partitions_for` sees on a write reaches the spill partition, so a
    /// DECIMAL column — which inference cannot type — is flushed rather than
    /// held resident.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn the_spilling_factory_encodes_with_the_catalog_schema() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "2");
        }
        let (_, base) = spill_root();
        let fs = local_fs().await;
        let store = DirectWalTableRecordStore::new_spilling(
            Arc::new(RecordingWalAppender::default()),
            fs,
            tenant_path_resolver(base),
        );
        let schema =
            CatalogTableSchema::new("orders").with_column(proximadb_catalog::CatalogColumn::new(
                1,
                "price",
                ProximaType::Decimal {
                    precision: 10,
                    scale: 2,
                },
            ));

        let (partition, _) = store.partitions_for("tenant-a", &schema);
        for i in 0..3u64 {
            let mut row = ProximaRecord {
                oid: format!("o{i}"),
                tenant_id: "tenant-a".to_string(),
                ..Default::default()
            };
            row.props.insert(
                "price".to_string(),
                proximadb_records::ProximaTreeNode::Value(ProximaValue::Decimal(format!("{i}.50"))),
            );
            partition
                .upsert_record_at_lsn(row, i + 1)
                .await
                .expect("upsert");
        }
        RecordStore::flush(partition.as_ref()).await.expect("flush");
        assert_eq!(
            RecordStore::min_unflushed_lsn(partition.as_ref())
                .await
                .ok(),
            Some(None),
            "every DECIMAL row was written — none held for want of a schema"
        );

        unsafe {
            std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV);
        }
    }

    #[tokio::test]
    async fn an_empty_tenant_partition_spills_and_persists_deletes() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "2");
        }

        let (root, base) = spill_root();
        let fs = local_fs().await;
        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_spilling(
            wal,
            fs,
            Arc::new(move |tenant_id: &str, collection: &str| {
                format!("{base}/untenanted{tenant_id}/{collection}")
            }),
        );

        // No tenant context at all — this is what `tenant_key(None)` produces.
        let partition = store.partition("", "orders");
        for i in 0..8u64 {
            partition
                .upsert_record_at_lsn(
                    ProximaRecord {
                        oid: format!("o{i}"),
                        tenant_id: String::new(),
                        ..Default::default()
                    },
                    i + 1,
                )
                .await
                .expect("upsert must succeed for an untenanted partition");
        }
        assert!(
            std::fs::read_dir(root.join("untenanted").join("orders"))
                .expect("the untenanted partition spilled")
                .next()
                .is_some(),
            "past the threshold the untenanted partition must write a segment"
        );

        partition
            .delete_record(&RecordKey::new("o0".to_string()))
            .await
            .expect("delete");
        RecordStore::flush(partition.as_ref()).await.expect("flush");
        assert_eq!(
            RecordScan::scan_records(partition.as_ref(), 100)
                .await
                .expect("scan")
                .len(),
            7,
            "every row but the deleted one is served"
        );
        assert_eq!(
            RecordStore::min_unflushed_lsn(partition.as_ref())
                .await
                .ok(),
            Some(None),
            "the delete is durable: nothing constrains truncation"
        );

        unsafe {
            std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV);
        }
    }

    /// The PRODUCTION wiring of the tenant is exercised, not just the store's own
    /// unit tests.
    ///
    /// Persisting a tombstone refuses without a non-empty tenant, and the only
    /// thing that satisfies that in production is `.with_tenant(tenant_id)` in
    /// `new_spilling`'s factory closure. Remove that one call and every
    /// `record_spill` unit test still passes — they construct the store directly
    /// — while the first post-flush DELETE would make every subsequent flush
    /// error, and (because `upsert_record` flushes at the threshold) every
    /// subsequent upsert with it.
    ///
    /// This is the class this module's own comments already name: a wiring line
    /// whose absence no test notices. So it is asserted THROUGH `new_spilling`.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn spilling_partition_persists_a_tombstone_through_the_production_wiring() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }

        let (root, base) = spill_root();
        let fs = local_fs().await;

        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_spilling(wal, fs, tenant_path_resolver(base));

        let upsert = |i: u64| {
            CanonicalWalEntry::new(
                i + 1,
                CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(ProximaRecord {
                        oid: format!("r{i}"),
                        variation_id: Some("orders".to_string()),
                        ..Default::default()
                    }),
                    projections: vec![],
                },
                Some("tenant-a".to_string()),
            )
        };

        // Eight upserts at threshold 4 flush at least once, so segments exist and
        // the DELETE below records a tombstone rather than a plain removal.
        let mut entries: Vec<CanonicalWalEntry> = (0..8).map(upsert).collect();
        entries.push(CanonicalWalEntry::new(
            9,
            CanonicalOperation::RecordDelete {
                collection_id: "orders".to_string(),
                oid: "r0".to_string(),
                projections: vec![],
            },
            Some("tenant-a".to_string()),
        ));
        // More upserts, so a flush runs WITH that tombstone pending — the flush
        // that refuses if the tenant was never plumbed.
        entries.extend((10..18).map(upsert));

        store
            .replay_wal_entries(entries)
            .await
            .expect("replay must succeed");

        // THE wiring assertion, and it has to be this one.
        //
        // Neither "replay succeeded" nor "r0 is absent" distinguishes a plumbed
        // tenant from an unplumbed one, because `flush` does not fail without a
        // tenant — it warns, drops the tombstones from its snapshot, writes the
        // residents and returns Ok, and the un-persisted tombstone still
        // suppresses `r0` from memory. (Round 2 added this test when the flush DID
        // return `Err`; round 3 changed that response for good reasons and
        // silently invalidated the evidence, which went unnoticed for two rounds.)
        //
        // What does distinguish them is DURABILITY: with the tenant, the tombstone
        // reaches a segment and is cleared post-commit, so the floor lies above
        // the delete's LSN (9) — only later upserts may still be resident.
        // Without it the tombstone stays in memory at LSN 9 and pins the floor
        // there. Asserted by VALUE rather than `is_ok()` so a refusal arm cannot
        // silently stand in for the right answer.
        let floor = RecordStore::min_unflushed_lsn(store.partition("tenant-a", "orders").as_ref())
            .await
            .expect("every replayed write carries its LSN");
        assert!(
            floor.is_none_or(|lsn| lsn > 9),
            "the tombstone must be DURABLE — i.e. the production factory supplied a tenant; \
             floor {floor:?}"
        );

        // And the deleted row stays gone through the read-merge.
        let schema = CatalogTableSchema::new("orders");
        let scanned = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "orders".to_string(),
                    limit: None,
                    include_vector: false,
                    include_props: true,
                },
                Some(&TenantContext::for_tenant_id("tenant-a")),
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .expect("scan");
        assert!(
            !scanned.iter().any(|r| r.oid == "r0"),
            "the deleted row must not come back after its tombstone was persisted"
        );

        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// TD-USUB-20: replaying `insert × N → RecordPartitionDrop → insert` into a
    /// spilling store must leave the re-created table holding ONLY the post-drop
    /// row, with no pre-drop segment left under its prefix.
    ///
    /// The N inserts cross the flush threshold during replay, so the dropped
    /// rows reach segments. Before the fix the replay drop arm only voided the
    /// map entry; the next access rebuilt the partition over the same prefix,
    /// discovery adopted those segments, and the dropped rows came back.
    /// Mutation check: removing the purge from the replay drop arm fails both
    /// assertions. A second replay over the same prefix (the next boot) must
    /// agree.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn replayed_drop_purges_segments_written_earlier_in_the_replay() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }

        let (root, base) = spill_root();
        let fs = local_fs().await;
        let build = |fs: Arc<crate::storage::persistence::filesystem::local::LocalFileSystem>| {
            let base = base.clone();
            DirectWalTableRecordStore::new_spilling(
                Arc::new(RecordingWalAppender::default()),
                fs,
                Arc::new(move |tenant_id: &str, collection: &str| {
                    format!("{base}/{tenant_id}/{collection}")
                }),
            )
        };
        let tenant = Some("tenant-a".to_string());
        let upsert = |lsn: u64, oid: &str| {
            CanonicalWalEntry::new(
                lsn,
                CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(ProximaRecord {
                        oid: oid.to_string(),
                        variation_id: Some("orders".to_string()),
                        ..Default::default()
                    }),
                    projections: vec![],
                },
                tenant.clone(),
            )
        };
        let mut entries: Vec<CanonicalWalEntry> = (0..8u64)
            .map(|i| upsert(i + 1, &format!("old{i}")))
            .collect();
        entries.push(CanonicalWalEntry::new(
            9,
            CanonicalOperation::RecordPartitionDrop {
                collection_id: "orders".to_string(),
            },
            tenant.clone(),
        ));
        entries.push(upsert(10, "new0"));

        let segment_files = || -> Vec<String> {
            std::fs::read_dir(root.join("tenant-a").join("orders"))
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().to_string())
                        .filter(|n| n.starts_with("spill-"))
                        .collect()
                })
                .unwrap_or_default()
        };
        async fn scan_oids(store: &DirectWalTableRecordStore) -> Vec<String> {
            let mut oids: Vec<String> = store
                .scan_records(
                    &CatalogTableSchema::new("orders"),
                    TableRecordScanRequest {
                        filter: None,
                        table_id: "orders".to_string(),
                        limit: None,
                        include_vector: false,
                        include_props: true,
                    },
                    Some(&TenantContext::for_tenant_id("tenant-a")),
                    #[cfg(feature = "abac-policy")]
                    &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
                )
                .await
                .expect("scan")
                .into_iter()
                .map(|r| r.oid)
                .collect();
            oids.sort();
            oids
        }

        for boot in 0..2 {
            let store = build(fs.clone());
            store
                .replay_wal_entries(entries.clone())
                .await
                .expect("replay must succeed");
            assert_eq!(
                scan_oids(&store).await,
                ["new0"],
                "boot {boot}: the re-created table must hold only the post-drop row"
            );
            assert!(
                segment_files().is_empty(),
                "boot {boot}: no pre-drop segment may survive the replayed drop: {:?}",
                segment_files()
            );
        }

        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// The record tier's truncation floor refuses for every non-durable shape:
    /// a memtable partition (the trait default) and shared-storage mode. An
    /// empty partitioned store has no constraint.
    #[tokio::test]
    async fn record_tier_floor_refuses_for_memtable_and_shared_storage() {
        let partitioned =
            DirectWalTableRecordStore::new_partitioned(Arc::new(RecordingWalAppender::default()));
        assert_eq!(
            partitioned.min_unflushed_lsn().await.ok(),
            Some(None),
            "no partition, no constraint"
        );
        partitioned
            .replay_wal_entries(vec![CanonicalWalEntry::new(
                1,
                CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(ProximaRecord {
                        oid: "r1".to_string(),
                        ..Default::default()
                    }),
                    projections: vec![],
                },
                Some("tenant-a".to_string()),
            )])
            .await
            .expect("replay");
        let err = partitioned
            .min_unflushed_lsn()
            .await
            .expect_err("a memtable partition holds its rows only in memory");
        assert!(
            format!("{err:#}").contains("collection 'orders'"),
            "{err:#}"
        );

        let shared = DirectWalTableRecordStore::new(
            Arc::new(crate::services::MemtableRecordStorage::new()),
            Arc::new(RecordingWalAppender::default()),
        );
        assert!(
            shared.min_unflushed_lsn().await.is_err(),
            "shared storage refuses unconditionally"
        );
    }

    /// With the spill gate on, the tier floor is the minimum over spill
    /// partitions' resident LSNs — the untenanted (`""`) partition included,
    /// since it spills too — and ONE refusing partition (a memtable) makes the
    /// whole tier refuse.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn record_tier_floor_is_the_minimum_over_spill_partitions() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }
        let (_, base) = spill_root();
        let fs = local_fs().await;
        let store_fs = fs.clone();
        let refusing_base = format!("{base}/mixed");
        let store = DirectWalTableRecordStore::new_spilling(
            Arc::new(RecordingWalAppender::default()),
            fs,
            tenant_path_resolver(base),
        );
        let upsert = |lsn: u64, tenant: &str, collection: &str| {
            CanonicalWalEntry::new(
                lsn,
                CanonicalOperation::RecordUpsert {
                    collection_id: collection.to_string(),
                    record: Box::new(ProximaRecord {
                        oid: format!("r{lsn}"),
                        ..Default::default()
                    }),
                    projections: vec![],
                },
                Some(tenant.to_string()),
            )
        };

        // `orders`: LSNs 1..=6 at threshold 4 — 1..=4 flush, 5 and 6 stay
        // resident. `items`: LSNs 7 and 8, both resident.
        let mut entries: Vec<CanonicalWalEntry> = (1..=6)
            .map(|lsn| upsert(lsn, "tenant-a", "orders"))
            .collect();
        entries.extend((7..=8).map(|lsn| upsert(lsn, "tenant-a", "items")));
        store.replay_wal_entries(entries).await.expect("replay");
        assert_eq!(
            store.min_unflushed_lsn().await.ok(),
            Some(Some(5)),
            "the minimum resident LSN across spill partitions"
        );

        // The untenanted partition spills like any other, so it reports its
        // resident LSN instead of refusing; the minimum is unchanged.
        store
            .replay_wal_entries(vec![upsert(9, "", "orders")])
            .await
            .expect("replay");
        assert_eq!(
            store.min_unflushed_lsn().await.ok(),
            Some(Some(5)),
            "an untenanted spill partition answers, it does not refuse"
        );

        // A memtable partition has no durable copy of its rows, so it refuses —
        // and one refusal makes the whole tier refuse.
        let spill_fs = store_fs.clone();
        let spill_base = refusing_base.clone();
        let mixed = DirectWalTableRecordStore::with_storage_factory(
            Arc::new(RecordingWalAppender::default()),
            Arc::new(move |tenant_id: &str, collection: &str| {
                if collection == "legacy" {
                    Arc::new(crate::services::MemtableRecordStorage::new())
                        as Arc<dyn RecordStorage>
                } else {
                    Arc::new(
                        crate::services::SpillRecordStorage::with_flush_threshold(
                            spill_fs.clone(),
                            format!("{spill_base}/{tenant_id}/{collection}"),
                            Some(4),
                        )
                        .with_tenant(tenant_id),
                    ) as Arc<dyn RecordStorage>
                }
            }),
        );
        mixed
            .replay_wal_entries(vec![
                upsert(1, "tenant-a", "orders"),
                upsert(2, "tenant-a", "legacy"),
            ])
            .await
            .expect("replay");
        assert!(
            mixed.min_unflushed_lsn().await.is_err(),
            "one refusing partition makes the whole tier refuse"
        );

        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    #[test]
    fn spill_path_segment_is_injective_and_never_navigates() {
        let samples = [
            "a/b", "a_b", "a_2Fb", "A.b-c", "", ".", "..", "tenant a", "é",
        ];
        let encoded: Vec<String> = samples.iter().map(|s| spill_path_segment(s)).collect();
        let unique: std::collections::HashSet<&String> = encoded.iter().collect();
        assert_eq!(unique.len(), samples.len(), "{encoded:?}");
        for e in &encoded {
            assert!(
                !e.contains('/') && !e.is_empty() && e != "." && e != "..",
                "{e}"
            );
        }
        assert_eq!(spill_path_segment("orders-2026.v1"), "orders-2026.v1");
    }

    /// With the gate off (`0`) the production constructor is the legacy
    /// memtable store and writes nothing under the spill root.
    #[tokio::test]
    async fn open_for_server_with_the_gate_off_is_the_partitioned_memtable() {
        unsafe { std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "0") };
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("record-spill");
        let store = DirectWalTableRecordStore::open_for_server(
            Arc::new(RecordingWalAppender::default()),
            root.clone(),
        )
        .await
        .expect("open");
        store
            .replay_wal_entries((1..=20u64).map(|lsn| {
                CanonicalWalEntry::new(
                    lsn,
                    CanonicalOperation::RecordUpsert {
                        collection_id: "orders".to_string(),
                        record: Box::new(ProximaRecord {
                            oid: format!("r{lsn}"),
                            ..Default::default()
                        }),
                        projections: vec![],
                    },
                    Some("tenant-a".to_string()),
                )
            }))
            .await
            .expect("replay");
        assert_eq!(store.flush_partitions().await.expect("flush"), 1);
        assert!(
            !root.exists(),
            "no spill object may be written with the gate off"
        );
        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// Turning the spill off over a root an earlier run spilled into refuses:
    /// those rows may be in no retained WAL entry, so a memtable store would
    /// serve a silent subset.
    #[tokio::test]
    async fn open_for_server_refuses_to_turn_the_spill_off_over_spilled_state() {
        unsafe { std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "0") };
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("record-spill");
        let partition = root.join("tenant-a").join("orders");
        std::fs::create_dir_all(&partition).expect("spill dir");
        std::fs::write(partition.join("spill-0000000000000001.parquet"), b"x").expect("segment");
        let wal = Arc::new(
            crate::services::FramedTableWalAppender::open(dir.path().join("canonical-records.wal"))
                .await
                .expect("wal"),
        );
        wal.enable_segment_rotation(1);
        for i in 0..3 {
            wal.append_operations(
                vec![CanonicalOperation::RecordUpsert {
                    collection_id: "orders".to_string(),
                    record: Box::new(ProximaRecord {
                        oid: format!("r{i}"),
                        ..Default::default()
                    }),
                    projections: vec![],
                }],
                None,
            )
            .await
            .expect("append");
        }

        // History intact: replay rebuilds every row in memory, so off is safe.
        DirectWalTableRecordStore::open_for_server(wal.clone(), root.clone())
            .await
            .expect("an untruncated WAL may turn the spill off");

        // History truncated: the segments may hold the only copy. Refuse.
        wal.truncate_below(u64::MAX).await.expect("truncate");
        let err = DirectWalTableRecordStore::open_for_server(wal, root)
            .await
            .err()
            .expect("spilled state over a truncated WAL with the gate off must refuse");
        assert!(
            format!("{err:#}").contains("turn the spill back on"),
            "{err:#}"
        );
        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// With the gate set, the production constructor spills under
    /// `{root}/{tenant}/{collection}` and `flush_partitions` makes the resident
    /// tail durable, so the tier floor then has no constraint.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn open_for_server_with_the_gate_spills_under_the_root_and_flushes_on_shutdown() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("record-spill");
        let store = DirectWalTableRecordStore::open_for_server(
            Arc::new(RecordingWalAppender::default()),
            root.clone(),
        )
        .await
        .expect("open");
        // 6 rows at threshold 4: 4 flush on the way, 2 stay resident.
        store
            .replay_wal_entries((1..=6u64).map(|lsn| {
                CanonicalWalEntry::new(
                    lsn,
                    CanonicalOperation::RecordUpsert {
                        collection_id: "orders/eu".to_string(),
                        record: Box::new(ProximaRecord {
                            oid: format!("r{lsn}"),
                            ..Default::default()
                        }),
                        projections: vec![],
                    },
                    Some("tenant-a".to_string()),
                )
            }))
            .await
            .expect("replay");
        let partition_dir = root.join("tenant-a").join("orders_2Feu");
        let segments = |dir: &std::path::Path| -> usize {
            std::fs::read_dir(dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with("spill-"))
                        .count()
                })
                .unwrap_or(0)
        };
        assert_eq!(
            segments(&partition_dir),
            1,
            "one threshold flush during replay"
        );

        store.flush_partitions().await.expect("shutdown flush");
        assert_eq!(
            segments(&partition_dir),
            2,
            "the resident tail is now durable"
        );

        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// The wiring proof: with the gate ON, a partition really is spill-backed —
    /// it flushes to a durable segment AND still serves every row through the
    /// ordinary `TableRecordStore` read path.
    ///
    /// Asserted as the OBSERVABLE effect (a segment exists on disk, and the scan
    /// still returns all rows) rather than by inspecting the partition's type.
    /// Both halves matter: a flush that loses rows and a "flush" that never
    /// happened would each pass one half alone.
    ///
    /// Process-isolated under nextest, so the env gate does not leak.
    #[tokio::test]
    async fn spilling_partition_flushes_and_still_serves_every_row() {
        unsafe {
            std::env::set_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV, "4");
        }

        let (root, base) = spill_root();
        let fs = local_fs().await;

        let wal = Arc::new(RecordingWalAppender::default());
        let store = DirectWalTableRecordStore::new_spilling(wal, fs, tenant_path_resolver(base));

        const ROWS: u64 = 20;
        let entries = upsert_entries(ROWS, "orders", "tenant-a");
        store.replay_wal_entries(entries).await.expect("replay");

        // Half 1: rows above the threshold actually left memory for a segment.
        let segment_count = walkdir_files(&root);
        assert!(
            segment_count > 0,
            "threshold 4 with {ROWS} rows must have flushed at least one segment under {}",
            root.display()
        );

        // Half 2: the read-merge serves flushed + resident rows as one set.
        let schema = CatalogTableSchema::new("orders");
        let scanned = store
            .scan_records(
                &schema,
                TableRecordScanRequest {
                    filter: None,
                    table_id: "orders".to_string(),
                    limit: None,
                    include_vector: false,
                    include_props: true,
                },
                Some(&TenantContext::for_tenant_id("tenant-a")),
                #[cfg(feature = "abac-policy")]
                &ReadContext::system(SystemReadReason::Statistics, "record_store::tests"),
            )
            .await
            .expect("scan");
        assert_eq!(
            scanned.len() as u64,
            ROWS,
            "spilled rows must stay visible through the read-merge; segments={segment_count}"
        );

        // Remove the gate, matching this file's existing precedent
        // (`memtable_oid_cutover_…`). nextest isolates by process, but under
        // plain `cargo test` a leaked gate races the sibling gate-unset test in
        // this same module and would make it flush.
        unsafe { std::env::remove_var(crate::services::record_spill::SPILL_MAX_RESIDENT_ENV) };
    }

    /// Count regular files under a directory tree (segments land in per-partition
    /// subdirectories, so a flat `read_dir` would miss them).
    fn walkdir_files(root: &std::path::Path) -> usize {
        let mut count = 0;
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    count += 1;
                }
            }
        }
        count
    }

    /// Phase D: the recovered `oid` byte-matches the catalog's canonical
    /// composite-key encoding (`CatalogRow::primary_key_string`), not a divergent
    /// local join — so a read-back record carries the same oid the write path
    /// (`CatalogRow::to_proxima_record`) produced.
    #[test]
    fn reconstruct_oid_matches_catalog_canonical_composite_key() {
        let mut schema = CatalogTableSchema::new("orders")
            .with_column(CatalogColumn::new(1, "region", ProximaType::String).nullable(false))
            .with_column(CatalogColumn::new(2, "id", ProximaType::Int64).nullable(false));
        schema.primary_key = vec!["region".to_string(), "id".to_string()];

        let mut props = HashMap::new();
        props.insert(
            "region".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("us".into())),
        );
        props.insert(
            "id".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(7)),
        );
        let record = ProximaRecord {
            props,
            ..Default::default()
        };

        let mut values = HashMap::new();
        values.insert("region".to_string(), ProximaValue::String("us".into()));
        values.insert("id".to_string(), ProximaValue::Int64(7));
        let canonical = proximadb_catalog::relational::CatalogRow {
            table: "orders".to_string(),
            values,
        }
        .primary_key_string(&schema)
        .unwrap()
        .expect("composite PK must produce a key");

        assert_eq!(
            reconstruct_oid(&record, &schema),
            canonical,
            "recovered oid must match CatalogRow::primary_key_string"
        );
    }
}

/// OLAP publication target: Parquet files and Iceberg metadata over object storage.
///
/// This path is for analytical/open-format publications and external-authority
/// tables. It is not the direct write authority for PostgreSQL-style mutable
/// OLTP tables; those commit through the WAL/row-delta path first.
pub struct ObjectStoreIcebergRecordStore {
    bridge: Arc<dyn ObjectStoreBridge>,
}

/// Unpack a mutation batch into (records, ids, kind), short-circuiting an
/// empty batch to a successful no-op result — the shared opening of the
/// object-store `write_mutations` impls.
enum UnpackedMutations {
    Empty(TableRecordWriteResult),
    Batch {
        records: Vec<ProximaRecord>,
        record_ids: Vec<String>,
        kind: TableRecordMutationKind,
    },
}

fn unpack_mutations(mutations: &[TableRecordMutation]) -> UnpackedMutations {
    let records = mutations
        .iter()
        .map(|mutation| mutation.record.clone())
        .collect::<Vec<_>>();
    let record_ids = records.iter().map(record_id).collect::<Vec<_>>();

    if records.is_empty() {
        return UnpackedMutations::Empty(TableRecordWriteResult::success(record_ids));
    }

    let kind = mutations
        .first()
        .map(|mutation| mutation.kind)
        .unwrap_or(TableRecordMutationKind::Insert);
    UnpackedMutations::Batch {
        records,
        record_ids,
        kind,
    }
}

impl ObjectStoreIcebergRecordStore {
    pub fn new(bridge: Arc<dyn ObjectStoreBridge>) -> Self {
        Self { bridge }
    }

    /// Read every current record for `schema` by listing the Parquet data
    /// objects the write path produced under the table's `data/` prefix and
    /// decoding each via the bridge's `read_parquet_batches`.
    ///
    /// This is a full-scan leaf read: every data object is listed and decoded.
    /// The advisory schema is `empty()` because the bridge's v1 read returns the
    /// file's batches as written (the Parquet file embeds its own schema) and the
    /// canonical reverse converter is self-describing. The heavy read
    /// optimizations — catalog-authoritative projection/coercion, Iceberg
    /// manifest + row-group pruning, predicate pushdown, and true streaming/range
    /// reads — are deferred to F5/P5 and layer behind this same `ObjectStoreBridge`
    /// seam (see `iceberg-engine/src/lib.rs` v1-scope notes; routing these leaf
    /// reads through DataFusion via `ComputeScheduler` is P1/P5).
    async fn read_all_records(
        &self,
        schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<ProximaRecord>> {
        let base = object_store_write_base_path(schema, tenant_context);
        let prefix = ObjectPath::from(format!("{base}data"));
        let parquet_paths = list_objects_with_suffix(&self.bridge, &prefix, ".parquet").await?;

        let tenant_id = tenant_context.map(|tc| tc.tenant_id.as_str());
        record_object_store_op(tenant_id, "list_parquet");

        let mut records = Vec::new();

        for path in parquet_paths {
            record_object_store_op(tenant_id, "read_parquet");
            let mut stream = self
                .bridge
                .read_parquet_batches(
                    &path,
                    Arc::new(ArrowSchema::empty()),
                    OBJECT_STORE_READ_BATCH_SIZE,
                    tenant_id,
                )
                .await
                .map_err(|err| {
                    anyhow!("ObjectStoreIcebergRecordStore failed to read '{path}': {err}")
                })?;
            while let Some(batch) = stream.next().await {
                let batch = batch.map_err(|err| {
                    anyhow!(
                        "ObjectStoreIcebergRecordStore failed to decode batch from '{path}': {err}"
                    )
                })?;
                records.extend(record_batch_to_records(&batch, schema));
            }
        }
        Ok(records)
    }
}

#[async_trait]
impl TableRecordStore for ObjectStoreIcebergRecordStore {
    async fn write_mutations(
        &self,
        schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        _tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let (records, record_ids, kind) = match unpack_mutations(&mutations) {
            UnpackedMutations::Empty(result) => return Ok(result),
            UnpackedMutations::Batch {
                records,
                record_ids,
                kind,
            } => (records, record_ids, kind),
        };
        let path = object_store_parquet_mutation_path(schema, kind, _tenant_context);
        let tenant_id = _tenant_context.map(|tc| tc.tenant_id.as_str());
        record_object_store_op(tenant_id, "write_parquet");
        self.bridge
            .write_records_to_parquet(&path, &records, tenant_id)
            .await
            .map_err(|err| {
                anyhow!(
                    "ObjectStoreIcebergRecordStore failed to write '{}' to '{}': {err}",
                    schema.name,
                    path
                )
            })?;

        Ok(TableRecordWriteResult {
            success: true,
            record_ids,
            metrics: OperationMetrics::default(),
            errors: vec![],
            error_code: None,
        })
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        _tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        let records = self.read_all_records(table_schema, _tenant_context).await?;
        let found = records.into_iter().find(|record| record.oid == request.key);
        Ok(found.map(|record| {
            proxima_record_to_get_response(
                record,
                table_schema,
                request.include_vector,
                request.include_props,
            )
        }))
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        _tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let mut records = self.read_all_records(table_schema, _tenant_context).await?;
        truncate_and_project_scan(&mut records, &request);
        Ok(records)
    }
}

/// Specialized Vector/ANN Target: PAX block formats.
/// Process-global multitenant footer/index caches for the PAX v2 ranged read
/// path. The cache is inherently a process singleton; `SharedServices` calls
/// [`init_segment_caches`] once at boot, and every `ObjectStoreVectorRecordStore`
/// auto-picks them up — no threading through every constructor.
static GLOBAL_SEGMENT_CACHES: std::sync::OnceLock<(
    Arc<proximadb_storage_common::ranged_segment::FooterCache>,
    Arc<proximadb_storage_common::ranged_segment::SegmentIndexCache>,
)> = std::sync::OnceLock::new();

/// Initialize the global footer/index caches from `budget` (idempotent — first
/// call wins). Call once during server boot.
///
/// `limits_resolver` is the open-core injection seam (Dependency Inversion): OSS
/// passes `None` → uniform elastic fair share. An enterprise/control-plane boot
/// path supplies a [`proximadb_cache::LimitsResolver`] (built from an
/// operator `cache_tiers.json` [`proximadb_cache::TierPolicy`] + a tenant→tier
/// authority such as `TenantContext.tier`) to get tier-weighted preference —
/// without any commercial policy baked into the OSS engine.
pub fn init_segment_caches(
    budget: proximadb_cache::CacheBudget,
    limits_resolver: Option<Arc<proximadb_cache::LimitsResolver>>,
) {
    let mut footer = proximadb_storage_common::ranged_segment::FooterCache::new(budget.clone());
    let mut index = proximadb_storage_common::ranged_segment::SegmentIndexCache::new(budget);
    if let Some(resolver) = limits_resolver {
        footer = footer.with_limits_resolver(resolver.clone());
        index = index.with_limits_resolver(resolver);
    }
    let _ = GLOBAL_SEGMENT_CACHES.set((Arc::new(footer), Arc::new(index)));
}

/// Per-tenant stats snapshot for the global footer cache (for metrics emission).
/// Empty when caches are not initialized.
pub fn segment_cache_tenant_stats() -> Vec<proximadb_cache::TenantCacheStat> {
    GLOBAL_SEGMENT_CACHES
        .get()
        .map(|(fc, _)| fc.tenant_stats())
        .unwrap_or_default()
}

/// The global segment-index cache, if initialised (TD-USUB-8 slice 1).
///
/// Exposed so the DataFusion PAX lane can skip its index reads too — it built
/// its own read sequence and never used `RangedSegmentReader::open_with_cache`,
/// so the cache existed but served exactly one caller.
pub fn segment_index_cache()
-> Option<Arc<proximadb_storage_common::ranged_segment::SegmentIndexCache>> {
    segment_caches().1
}

fn segment_caches() -> (
    Option<Arc<proximadb_storage_common::ranged_segment::FooterCache>>,
    Option<Arc<proximadb_storage_common::ranged_segment::SegmentIndexCache>>,
) {
    match GLOBAL_SEGMENT_CACHES.get() {
        Some((fc, ic)) => (Some(fc.clone()), Some(ic.clone())),
        None => (None, None),
    }
}

/// Process-global tenant→tier map (the *authority* hook for the open-core cache
/// tier policy). OSS leaves it empty (→ uniform fair share); the auth/control
/// plane populates it from a request tier claim via [`set_tenant_tier`], and the
/// cache `LimitsResolver` (built from an operator `cache_tiers.json`) reads it.
/// Commercial tier *data* stays out of OSS — this only carries opaque tier ids.
static TENANT_TIERS: std::sync::LazyLock<dashmap::DashMap<String, StampedTier>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// A stamped tier claim held in **both** vocabularies (TD-TENANT-3 S3).
///
/// Before this, the registry held only the raw client string, and the four
/// consumers read it two incompatible ways: three matched it as a literal key
/// against the operator's `cache_tiers.json`, while the cost path parsed it
/// alias-aware via [`Tier::from_claim`]. So `pro` and `tier3` — the *same*
/// entitlement — resolved identically for cost multipliers but landed in
/// different cache buckets, and one of them silently missed.
#[derive(Debug, Clone)]
pub struct StampedTier {
    /// The claim resolved to a canonical tier. Normalizing here (rather than at
    /// each read) also bounds the registry's value space to the known tier set,
    /// so an unrecognized client claim no longer persists verbatim.
    pub canonical: crate::catalog::tenant_tier::Tier,
    /// The operator's original spelling, preserved so an alias-keyed
    /// `cache_tiers.json` keeps binding without a flag day.
    pub raw: String,
}

/// Record (or update) a tenant's tier id — called by the auth layer from a
/// request claim/header. No-op semantics for unknown tenants (just inserts).
///
/// TD-TENANT-3 S3: normalization happens **once, here**, so every downstream
/// reader sees one vocabulary. An unrecognized claim resolves to the default
/// tier (fail-safe, unchanged in effect — it previously missed every policy
/// key and fell back to `default_tier` anyway).
pub fn set_tenant_tier(tenant_id: impl Into<String>, tier: impl Into<String>) {
    let raw = tier.into();
    let canonical = crate::catalog::tenant_tier::Tier::from_claim(&raw)
        .unwrap_or_else(crate::catalog::tenant_tier::default_tier);
    TENANT_TIERS.insert(tenant_id.into(), StampedTier { canonical, raw });
}

/// The tier id as the control plane spelled it, if one was stamped.
///
/// This is the *operator* vocabulary — use [`tenant_tier_policy_key`] to look a
/// tenant up in a `cache_tiers.json`, or [`tenant_tier_resolved`] for the
/// canonical [`Tier`].
pub fn tenant_tier(tenant_id: &str) -> Option<String> {
    TENANT_TIERS.get(tenant_id).map(|r| r.raw.clone())
}

/// The key to use for a tenant against an operator tier policy, probing the
/// **canonical id first** and falling back to the operator's original spelling
/// (TD-TENANT-3 S3).
///
/// `declares` reports whether the policy has an explicit spec for a key (e.g.
/// `|k| policy.has_tier(k)`). The two-probe order is what makes the
/// normalization mixed-read-safe: a `cache_tiers.json` keyed by canonical ids
/// and one keyed by commercial aliases both bind, so no operator config has to
/// change in lockstep with this engine change.
///
/// `None` when the tenant has no stamp or the policy declares neither
/// spelling — the caller then applies the policy's own default tier.
pub fn tenant_tier_policy_key(tenant_id: &str, declares: impl Fn(&str) -> bool) -> Option<String> {
    let stamped = TENANT_TIERS.get(tenant_id)?;
    let canonical = stamped.canonical.id();
    if declares(canonical) {
        return Some(canonical.to_string());
    }
    declares(&stamped.raw).then(|| stamped.raw.clone())
}

/// Resolve a tenant to its [`Tier`] from the header-fed registry above — the
/// co-design C5 tenant→tier bridge. Parses the stamped claim via
/// [`Tier::from_claim`] (alias-aware) and falls back to the configured default
/// tier when no claim was stamped or it is unrecognized. This converges the
/// header-fed tier source (the cache `LimitsResolver`) with the route cost
/// model's tier multiplier — one tier registry, not two. Installed at startup
/// into `tenant_tier::set_tenant_tier_resolver` (see `database.rs`).
pub fn tenant_tier_resolved(tenant_id: &str) -> crate::catalog::tenant_tier::Tier {
    // TD-TENANT-3 S3: the claim was already parsed at the stamp, so this is a
    // read, not a re-parse — and it can no longer disagree with what the cache
    // path resolves for the same tenant.
    TENANT_TIERS
        .get(tenant_id)
        .map(|stamped| stamped.canonical)
        .unwrap_or_else(crate::catalog::tenant_tier::default_tier)
}

#[cfg(test)]
mod tenant_tier_bridge_tests {
    use super::*;
    use crate::catalog::tenant_tier::{Tier, default_tier};

    #[test]
    fn resolves_stamped_tier_else_default() {
        // Unique tenant ids so this never races sibling tests on the global map.
        let unknown = "bridge-test-unknown-tenant";
        assert_eq!(tenant_tier_resolved(unknown), default_tier());

        let t = "bridge-test-pro-tenant";
        set_tenant_tier(t, "pro"); // control-plane stamped X-Tenant-Tier: pro
        assert_eq!(tenant_tier_resolved(t), Tier::Tier3);

        // An unrecognized claim falls back to the default tier (fail-safe).
        let bad = "bridge-test-bad-claim-tenant";
        set_tenant_tier(bad, "not-a-tier");
        assert_eq!(tenant_tier_resolved(bad), default_tier());
    }

    /// TD-TENANT-3 S3: the defect this workstream closes — two spellings of one
    /// entitlement must resolve to the SAME cache bucket, not two.
    #[test]
    fn alias_and_canonical_spellings_converge_on_one_policy_key() {
        let alias_tenant = "vocab-test-alias-tenant";
        let canonical_tenant = "vocab-test-canonical-tenant";
        set_tenant_tier(alias_tenant, "pro"); // commercial alias
        set_tenant_tier(canonical_tenant, "tier3"); // canonical id

        // Both stamps resolve to the same tier...
        assert_eq!(tenant_tier_resolved(alias_tenant), Tier::Tier3);
        assert_eq!(tenant_tier_resolved(canonical_tenant), Tier::Tier3);

        // ...and against a canonically-keyed policy they now agree, where
        // before the fix `pro` missed every key and fell back to the default.
        let canonical_policy = |key: &str| key == "tier3";
        assert_eq!(
            tenant_tier_policy_key(alias_tenant, canonical_policy),
            Some("tier3".to_string())
        );
        assert_eq!(
            tenant_tier_policy_key(canonical_tenant, canonical_policy),
            Some("tier3".to_string())
        );
    }

    /// Mixed-read safety: an operator `cache_tiers.json` keyed by commercial
    /// aliases must keep binding, so normalization needs no config flag day.
    #[test]
    fn alias_keyed_operator_policy_still_binds_via_the_raw_probe() {
        let tenant = "vocab-test-alias-keyed-policy-tenant";
        set_tenant_tier(tenant, "pro");

        // Policy declares ONLY the alias — the canonical probe misses, the raw
        // probe hits.
        let alias_policy = |key: &str| key == "pro";
        assert_eq!(
            tenant_tier_policy_key(tenant, alias_policy),
            Some("pro".to_string())
        );

        // A policy declaring neither yields None, so the caller applies its own
        // default tier (unchanged behavior).
        assert_eq!(tenant_tier_policy_key(tenant, |_| false), None);
    }

    /// The canonical probe is preferred when a policy declares BOTH spellings,
    /// so two tenants on one entitlement can never split across buckets.
    #[test]
    fn canonical_probe_wins_when_a_policy_declares_both_spellings() {
        let tenant = "vocab-test-both-spellings-tenant";
        set_tenant_tier(tenant, "enterprise");
        let both = |key: &str| key == "tier5" || key == "enterprise";
        assert_eq!(
            tenant_tier_policy_key(tenant, both),
            Some("tier5".to_string())
        );
    }

    /// An unrecognized claim must not persist verbatim as a policy key — it
    /// normalizes to the default tier at the stamp.
    #[test]
    fn unrecognized_claim_does_not_reach_the_policy_as_a_raw_key() {
        let tenant = "vocab-test-garbage-claim-tenant";
        set_tenant_tier(tenant, "wat-is-this");
        assert_eq!(tenant_tier_resolved(tenant), default_tier());
        // The garbage string is retained for diagnostics but only binds if the
        // operator literally declared it.
        assert_eq!(
            tenant_tier_policy_key(tenant, |key| key == default_tier().id()),
            Some(default_tier().id().to_string())
        );
    }
}

pub struct ObjectStoreVectorRecordStore {
    bridge: Arc<dyn ObjectStoreBridge>,
    footer_cache: Option<Arc<proximadb_storage_common::ranged_segment::FooterCache>>,
    index_cache: Option<Arc<proximadb_storage_common::ranged_segment::SegmentIndexCache>>,
}

impl ObjectStoreVectorRecordStore {
    pub fn new(bridge: Arc<dyn ObjectStoreBridge>) -> Self {
        // Auto-pick up the process-global caches if SharedServices initialized them.
        let (footer_cache, index_cache) = segment_caches();
        Self {
            bridge,
            footer_cache,
            index_cache,
        }
    }

    /// Read every current record for `schema` by listing the PAX segment objects
    /// the write path produced under the table's `segments/` prefix and decoding
    /// each via the bridge's `fetch_vector_segment`.
    async fn read_all_records(
        &self,
        schema: &CatalogTableSchema,
        tenant_context: Option<&TenantContext>,
    ) -> Result<Vec<ProximaRecord>> {
        let base = object_store_write_base_path(schema, tenant_context);
        let prefix = ObjectPath::from(format!("{base}segments"));
        let segment_paths =
            list_objects_with_suffix(&self.bridge, &prefix, PAX_SEGMENT_EXT).await?;
        let tenant_id = tenant_context.map(|tc| tc.tenant_id.as_str());
        record_object_store_op(tenant_id, "list_pax");
        io_trace::record_op_str("list_pax");

        let mut records = Vec::new();
        for path in segment_paths {
            record_object_store_op(tenant_id, "fetch_pax");
            io_trace::record_op_str("fetch_pax");
            let bytes = self
                .bridge
                .fetch_vector_segment(&path, tenant_id)
                .await
                .map_err(|err| {
                    anyhow!("ObjectStoreVectorRecordStore failed to fetch '{path}': {err}")
                })?;
            io_trace::record_bytes_read(bytes.len() as u64);
            // KOU read-egress (Dimension 2): the fetched bytes left object storage
            // for compute. Metered per-(tenant, locality, direction); only
            // chargeable (cross-region/-cloud/on-prem/internet) localities feed the
            // cost model's egress term — same-region/AZ reads are free.
            record_kou_bytes(
                tenant_id,
                object_store_egress_locality(),
                "read",
                bytes.len() as u64,
            );
            records.extend(pax_segment_to_records(bytes, schema)?);
        }
        Ok(records)
    }
}

/// Reconstruct full [`ProximaRecord`]s from a PAX segment via the canonical
/// segment decoder ([`PaxSegmentScanner::read_records`], the inverse of
/// `PaxSegmentWriter::add_record`). Props, labels, edges, timestamps, and the
/// dense embedding all round-trip.
///
/// Embedding model ids and promoted user-column keys are not persisted
/// positionally by the format, so we pass empty slices (best-effort `model_N`
/// defaults); deriving them from the catalog schema is a follow-up. `variation_id`
/// (also not a canonical PAX column) is restamped from the table name.
fn pax_segment_to_records(
    bytes: Vec<u8>,
    schema: &CatalogTableSchema,
) -> Result<Vec<ProximaRecord>> {
    let mut scanner = PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default())
        .map_err(|err| anyhow!("ObjectStoreVectorRecordStore failed to open PAX segment: {err}"))?;
    let mut records = scanner.read_records(&[], &[], None).map_err(|err| {
        anyhow!("ObjectStoreVectorRecordStore failed to decode PAX segment: {err}")
    })?;
    for record in &mut records {
        record.variation_id = Some(schema.name.clone());
    }
    Ok(records)
}

#[async_trait]
impl TableRecordStore for ObjectStoreVectorRecordStore {
    async fn write_mutations(
        &self,
        schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        _tenant_context: Option<&TenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let (records, record_ids, kind) = match unpack_mutations(&mutations) {
            UnpackedMutations::Empty(result) => return Ok(result),
            UnpackedMutations::Batch {
                records,
                record_ids,
                kind,
            } => (records, record_ids, kind),
        };
        let object_path = object_store_pax_segment_path(schema, kind, _tenant_context);
        let local_path = temp_pax_segment_path(schema, kind);
        // P-Shred (ADR-055): resolve the shred-list from the catalog's props-auto-promotion
        // policy — each promoted prop key maps to a typed user-column (by name → id). Sorted by
        // column id for deterministic stripe layout. Empty ⇒ no shredding (byte-for-byte today's
        // output). The msgpack PROPS tail stays authoritative; these columns are a pruning index.
        let mut shred_spec: Vec<proximadb_block_format::ShredColumn> = schema
            .props_auto_promotion
            .promoted_keys
            .iter()
            .filter_map(|(prop_key, col_name)| {
                schema
                    .columns
                    .iter()
                    .find(|c| &c.name == col_name)
                    // TD-USUB-6: carry the EXACT declared type, not just the id.
                    // This path resolves against `CatalogTableSchema`, so it knows
                    // `Int32` from `Int64` and `Timestamp(Micros)` from
                    // `Timestamp(Millis)` — the only paths that do can ever make a
                    // typed stripe authoritative.
                    .map(|c| {
                        proximadb_block_format::ShredColumn::typed(
                            prop_key.clone(),
                            c.id,
                            c.data_type.clone(),
                        )
                    })
            })
            .collect();
        shred_spec.sort_by_key(|c| c.column_id);
        let mut writer = PaxSegmentWriter::new(
            &local_path,
            BlockMode::Pax,
            BlockCompression::None,
            &schema.name,
            0,
            embedding_count_for_records(&records),
            None,
        )
        .with_shred_columns(shred_spec);
        for record in &records {
            writer.add_record(record).map_err(|err| {
                anyhow!(
                    "ObjectStoreVectorRecordStore failed to encode '{}' as PAX: {err}",
                    schema.name
                )
            })?;
        }
        let segment_meta = writer.finish().map_err(|err| {
            anyhow!(
                "ObjectStoreVectorRecordStore failed to finish PAX segment for '{}': {err}",
                schema.name
            )
        })?;
        let bytes = std::fs::read(&segment_meta.path).map_err(|err| {
            anyhow!(
                "ObjectStoreVectorRecordStore failed to read staged PAX segment '{}': {err}",
                segment_meta.path.display()
            )
        })?;
        let remove_result = std::fs::remove_file(&segment_meta.path);

        let tenant_id = _tenant_context.map(|tc| tc.tenant_id.as_str());
        // Per-query WRITE trace: the PAX segment PUT is the object-store write boundary. Mirrors the
        // read side's `record_bytes_read` in `read_all_records`, so a flush's bytes are observable
        // to the co-design cost model (no-op outside an `io_trace::scope`).
        io_trace::record_op_str("put_pax");
        io_trace::record_bytes_written(bytes.len() as u64);
        self.bridge
            .persist_vector_segment(&object_path, &bytes, tenant_id)
            .await
            .map_err(|err| {
                anyhow!(
                    "ObjectStoreVectorRecordStore failed to persist '{}' to '{}': {err}",
                    schema.name,
                    object_path
                )
            })?;
        if let Err(err) = remove_result {
            tracing::debug!(
                "failed to remove staged PAX segment '{}': {}",
                segment_meta.path.display(),
                err
            );
        }

        Ok(TableRecordWriteResult {
            success: true,
            record_ids,
            metrics: OperationMetrics::default(),
            errors: vec![],
            error_code: None,
        })
    }

    async fn get_by_key(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordGetRequest,
        _tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordGetResponse> {
        let records = self.read_all_records(table_schema, _tenant_context).await?;
        let found = records.into_iter().find(|record| record.oid == request.key);
        Ok(found.map(|record| {
            proxima_record_to_get_response(
                record,
                table_schema,
                request.include_vector,
                request.include_props,
            )
        }))
    }

    async fn scan_records(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        _tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let mut records = self.read_all_records(table_schema, _tenant_context).await?;
        truncate_and_project_scan(&mut records, &request);
        Ok(records)
    }
    /// Override: when a structured `request.filter` is present, read segments via
    /// footer-first **ranged** reads and skip whole blocks the filter provably
    /// excludes (predicate pushdown) before fetching their bodies. The row-exact
    /// `predicate` closure is still applied on top (block pruning is coarse), and
    /// the scan stops at `request.limit`. With no structured filter, this falls
    /// back to the default materialize-then-filter behavior.
    async fn scan_records_filtered(
        &self,
        table_schema: &CatalogTableSchema,
        request: TableRecordScanRequest,
        predicate: Option<&RecordScanPredicate<'_>>,
        tenant_context: Option<&TenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &ReadContext,
    ) -> Result<TableRecordScanResponse> {
        let Some(filter) = request.filter.clone() else {
            let limit = request.limit.unwrap_or(usize::MAX);
            let mut req = request;
            req.limit = None;
            let mut all = self
                .scan_records(
                    table_schema,
                    req,
                    tenant_context,
                    #[cfg(feature = "abac-policy")]
                    _read_context,
                )
                .await?;
            let mut kept = 0usize;
            all.retain(|record| {
                if kept >= limit {
                    return false;
                }
                let keep = predicate.is_none_or(|p| p(record));
                if keep {
                    kept += 1;
                }
                keep
            });
            return Ok(all);
        };

        let limit = request.limit.unwrap_or(usize::MAX);
        let base = object_store_write_base_path(table_schema, tenant_context);
        let prefix = ObjectPath::from(format!("{base}segments"));
        let segment_paths =
            list_objects_with_suffix(&self.bridge, &prefix, PAX_SEGMENT_EXT).await?;
        let tenant_id = tenant_context.map(|tc| tc.tenant_id.as_str());
        record_object_store_op(tenant_id, "list_pax");
        io_trace::record_op_str("list_pax");
        let field_to_col: &(dyn Fn(&str) -> Option<i32> + Sync) =
            &crate::storage::engines::sst::segment_format::pax_field_to_col;

        let mut out = Vec::new();
        'segments: for path in segment_paths {
            record_object_store_op(tenant_id, "fetch_pax_ranged");
            io_trace::record_op_str("fetch_pax_ranged");
            let reader = RangedSegmentReader::open_with_cache(
                self.bridge.as_ref(),
                path.clone(),
                tenant_id,
                self.footer_cache.clone(),
                self.index_cache.clone(),
            )
            .await
            .map_err(|err| anyhow!("ranged open '{path}': {err}"))?;
            let recs = reader
                .read_records_pruned(&filter, field_to_col, &[], &[])
                .await
                .map_err(|err| anyhow!("ranged pruned read '{path}': {err}"))?;
            // Forward this open's physical read accounting into the per-query
            // I/O trace (co-design C0: object-store bytes + footer-cache
            // effectiveness, Dimensions 1 & 3).
            let st = reader.read_stats();
            io_trace::record_bytes_read(st.bytes_read);
            io_trace::record_range_gets(st.range_gets);
            io_trace::record_footers(st.footer_hits, st.footer_misses);
            // KOU read-egress (Dimension 2): ranged-read bytes that left object storage.
            record_kou_bytes(
                tenant_id,
                object_store_egress_locality(),
                "read",
                st.bytes_read,
            );
            for mut record in recs {
                record.variation_id = Some(table_schema.name.clone());
                if predicate.is_none_or(|p| p(&record)) {
                    if !request.include_vector {
                        record.embeddings.clear();
                    }
                    if !request.include_props {
                        record.props.clear();
                    }
                    out.push(record);
                    if out.len() >= limit {
                        break 'segments;
                    }
                }
            }
        }
        Ok(out)
    }
}
