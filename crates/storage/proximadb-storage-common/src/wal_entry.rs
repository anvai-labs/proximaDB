//! Canonical WAL entry types for durability, recovery, and CDC consolidation.
//!
//! Every modality (document, graph node/edge, vector, observability) writes
//! WAL entries as `CanonicalOperation::RecordUpsert` carrying a boxed
//! `ProximaRecord`.
//! Recovery replays these entries and calls `ProjectionRebuilder` to reconstruct
//! modality-specific projections without those projections being an independent
//! durable source of truth.
//!
//! Spec reference:
//! `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc` Phase 5.

use proximadb_records::ProximaRecord;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Canonical WAL entry
// ---------------------------------------------------------------------------

/// The canonical WAL entry written by every modality.
///
/// Document writes, graph node/edge writes, vector writes, and observability
/// writes all funnel through this shape. Modality-specific projection state
/// (adjacency tables, JSON path indexes, HNSW, CSR) is rebuilt from the
/// `projections` field during recovery rather than being persisted as a
/// separate durable truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalWalEntry {
    /// Monotonically increasing WAL sequence number (log sequence number / LSN).
    pub sequence_number: u64,
    /// Wall-clock time at which the entry was written (ms since Unix epoch).
    pub timestamp_ms: u64,
    /// The canonical operation to apply.
    pub operation: CanonicalOperation,
    /// Optional tenant scope; `None` for single-tenant deployments.
    pub tenant_id: Option<String>,
}

impl CanonicalWalEntry {
    pub fn new(
        sequence_number: u64,
        operation: CanonicalOperation,
        tenant_id: Option<String>,
    ) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            sequence_number,
            timestamp_ms,
            operation,
            tenant_id,
        }
    }

    pub fn is_checkpoint(&self) -> bool {
        matches!(self.operation, CanonicalOperation::Checkpoint(_))
    }
}

// ---------------------------------------------------------------------------
// Canonical operation enum
// ---------------------------------------------------------------------------

/// Operation carried by a `CanonicalWalEntry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalOperation {
    /// Insert or update any record (document, node, edge, vector, observation).
    ///
    /// The `projections` list tells recovery which physical projections to
    /// rebuild or update when replaying this entry.
    RecordUpsert {
        collection_id: String,
        record: Box<ProximaRecord>,
        projections: Vec<ProjectionDirective>,
    },
    /// Delete a record by canonical OID.
    ///
    /// The `projections` list tells recovery which projections to invalidate.
    RecordDelete {
        collection_id: String,
        /// Canonical object ID (`ProximaRecord::oid`).
        oid: String,
        projections: Vec<ProjectionDirective>,
    },
    /// Void ALL record state for one `(tenant, collection_id)` partition from
    /// this point on (DROP TABLE / DROP COLLECTION).
    ///
    /// Live stores purge the partition when appending this entry; recovery
    /// replay must do the same when it folds it, so a table recreated after
    /// the drop starts empty instead of resurrecting the dropped rows (the
    /// record oid for SQL rows is the PK text by design, so a recreated table
    /// would otherwise collide with — or silently serve — the dropped rows).
    /// Upserts replayed AFTER this entry re-seed the partition normally.
    RecordPartitionDrop {
        /// The dropped partition's collection key (the same id the upserts
        /// were routed under — stable object id or legacy bare name).
        collection_id: String,
    },
    /// Durable checkpoint. Recovery can start from the latest `Checkpoint`
    /// and replay only subsequent entries rather than the full log.
    Checkpoint(SnapshotManifest),
    /// CDC publish barrier. Marks the highest sequence number up to which
    /// CDC consumers have been notified. Replaying past this entry re-emits
    /// CDC events to catch up late subscribers.
    CdcBarrier {
        /// Sequence number of the last CDC-published entry.
        barrier_sequence: u64,
        /// Events that were published up to `barrier_sequence`.
        events: Vec<CdcRecordEvent>,
    },
    /// System-catalog DDL mutation (read-heavy system-catalog redesign).
    ///
    /// Carries an opaque, control-plane-encoded routing `key` plus the
    /// serialized catalog delta `bytes`, so that this storage-layer crate stays
    /// decoupled from the control-plane catalog types (storage sits *below*
    /// control in the workspace layering). The system catalog's in-RAM
    /// authority (`SystemCatalogState`, root crate) owns the key grammar and
    /// decodes `bytes` when folding the WAL on recovery. Catalog mutations do
    /// not drive record/projection replay, so record-recovery consumers treat
    /// this variant as a no-op.
    CatalogMutation {
        /// Control-plane routing key, e.g. `"table:<catalog>/<ns>/<name>"`,
        /// `"table-delete:<catalog>/<ns>/<name>"`, or `"ns:<catalog>/<name>"`.
        /// Opaque to storage-common.
        key: String,
        /// Serialized catalog delta (`rmp_serde` of the typed mutation).
        /// Opaque bytes; decoded by the system catalog.
        bytes: Vec<u8>,
    },
}

// ---------------------------------------------------------------------------
// Projection directives
// ---------------------------------------------------------------------------

/// Describes one physical projection that must be rebuilt when replaying a
/// canonical WAL entry.
///
/// Directives travel alongside the authoritative record so that recovery
/// knows exactly which projections to touch without hard-coded dispatch on
/// `source_model` or collection naming conventions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProjectionDirective {
    /// Rebuild the JSON path index for the given collection.
    DocumentJsonPathIndex {
        collection_id: String,
        /// JSON path expressions that are indexed (e.g. `"$.tags[*]"`).
        indexed_paths: Vec<String>,
    },
    /// Rebuild the full-text search index for specific fields.
    FullTextIndex {
        collection_id: String,
        indexed_fields: Vec<String>,
    },
    /// Update one row in the graph adjacency table.
    ///
    /// Each upsert of a node or edge generates one directive per affected
    /// adjacency row so that recovery can apply fine-grained updates without
    /// scanning the whole table.
    AdjacencyTableRow {
        graph_id: String,
        node_oid: String,
        /// Edge references originating from or terminating at `node_oid`.
        edge_refs: Vec<EdgeRef>,
    },
    /// Schedule a CSR re-materialisation for the graph.
    ///
    /// This is a coarse directive: recovery marks the CSR stale and a
    /// background task performs the actual rebuild. Write-heavy workloads
    /// accumulate many individual `AdjacencyTableRow` directives and
    /// coalesce them into a single `CsrRebuild` when the write rate drops
    /// below the CSR-staleness threshold.
    CsrRebuild { graph_id: String },
    /// Update the HNSW index for the given embedding field.
    HnswIndex {
        collection_id: String,
        embedding_field: String,
    },
    /// Update the columnar variation projection for a set of fields.
    ColumnarVariation {
        collection_id: String,
        fields: Vec<String>,
    },
    /// Rebuild an observability trace/service-correlation index.
    ///
    /// Trace indexes are Layer 2 projections over canonical log/span/metric
    /// records. They must not introduce independent WAL or retention semantics.
    ObservabilityTraceIndex {
        collection_id: String,
        service_name: Option<String>,
        trace_id_field: String,
        span_id_field: String,
    },
    /// Rebuild a retained time-series rollup projection.
    ///
    /// The rollup is authoritative only when materialized back as canonical
    /// aggregate records by policy. Otherwise it remains rebuildable from WAL
    /// and canonical observability records.
    TimeSeriesRollup {
        collection_id: String,
        metric_name: String,
        window_ms: u64,
        dimensions: Vec<String>,
    },
    /// Refresh an open-format manifest projection such as Iceberg REST catalog
    /// metadata.
    ///
    /// This directive models Layer 3 catalog facade refresh. Ownership remains
    /// with xCatalog + ProximaRecord unless the catalog marks the table as
    /// explicitly external-authoritative.
    OpenFormatManifest {
        namespace: String,
        table_name: String,
        format: OpenTableFormat,
    },
}

/// Open table format whose manifest/protocol facade can be refreshed from
/// canonical records.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpenTableFormat {
    Iceberg,
    Delta,
    Hudi,
}

/// A lightweight edge reference stored inside `AdjacencyTableRow`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeRef {
    pub src_oid: String,
    pub dst_oid: String,
    pub edge_type: String,
    pub weight: Option<f64>,
}

// ---------------------------------------------------------------------------
// Snapshot manifest (checkpoint)
// ---------------------------------------------------------------------------

/// Snapshot manifest captured inside a `Checkpoint` WAL entry.
///
/// Recovery starts from the latest checkpoint, checks `projection_freshness`
/// to skip projections already up-to-date, then replays only the WAL entries
/// after `sequence_number`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    /// WAL LSN at which the snapshot was taken.
    pub sequence_number: u64,
    /// Wall-clock time of the snapshot (ms since Unix epoch).
    pub timestamp_ms: u64,
    /// Collections present in this snapshot.
    pub collection_ids: Vec<String>,
    /// Per-projection freshness metadata so recovery can skip projections
    /// whose `last_rebuilt_lsn >= checkpoint.sequence_number`.
    pub projection_freshness: Vec<ProjectionFreshness>,
}

/// Records the last WAL LSN at which a named projection was successfully rebuilt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFreshness {
    /// Human-readable projection name (e.g. `"doc_json_path:my_collection"`).
    pub projection_name: String,
    /// WAL sequence number of the last successful rebuild.
    pub last_rebuilt_lsn: u64,
    /// Whether the projection is currently stale and requires a rebuild.
    pub stale: bool,
}

// ---------------------------------------------------------------------------
// CDC canonical event
// ---------------------------------------------------------------------------

/// Canonical CDC event published when a record changes.
///
/// CDC consumers receive `CdcRecordEvent` and derive modality-specific logical
/// views from it; they do not receive raw `DocumentRecord` or graph-specific
/// change structs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcRecordEvent {
    /// Unique event identifier (UUID recommended).
    pub event_id: String,
    /// WAL sequence number that produced this event.
    pub sequence_number: u64,
    /// Wall-clock time of the event (ms since Unix epoch).
    pub timestamp_ms: u64,
    pub collection_id: String,
    pub operation: CdcOperation,
    /// Canonical record state after the change (`None` for `Delete`).
    pub record_after: Option<ProximaRecord>,
    /// Modality-specific logical views derived from the canonical record.
    ///
    /// These are pre-computed at publish time so consumers can subscribe
    /// to just the view they care about without re-implementing projection logic.
    pub logical_views: Vec<CdcLogicalView>,
}

/// Type of CDC operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CdcOperation {
    Insert,
    Update,
    Delete,
}

/// A derived logical view attached to a `CdcRecordEvent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CdcLogicalView {
    /// Document facade view of the record change.
    DocumentChange {
        collection_id: String,
        document_json: serde_json::Value,
    },
    /// Graph node facade view.
    GraphNodeChange {
        graph_id: String,
        node_json: serde_json::Value,
    },
    /// Graph edge facade view.
    GraphEdgeChange {
        graph_id: String,
        edge_json: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Projection rebuilder trait
// ---------------------------------------------------------------------------

/// Callback interface called by `recover_from_canonical_wal` for each
/// `ProjectionDirective` encountered during log replay.
///
/// Production implementations wire into the document index, graph adjacency
/// table, HNSW manager, and CSR materialiser. The in-memory test double
/// records which directives were applied.
pub trait ProjectionRebuilder {
    fn apply_directive(
        &mut self,
        record: Option<&ProximaRecord>,
        directive: &ProjectionDirective,
    ) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

/// Summary produced by `recover_from_canonical_wal`.
#[derive(Debug, Default)]
pub struct RecoveryResult {
    /// Number of `RecordUpsert` entries replayed.
    pub upserts_replayed: u64,
    /// Number of `RecordDelete` entries replayed.
    pub deletes_replayed: u64,
    /// Number of projection directives applied.
    pub directives_applied: u64,
    /// Number of directives that returned an error (non-fatal; logged).
    pub directive_errors: u64,
    /// Sequence number of the latest entry replayed.
    pub last_sequence: u64,
}

/// Replay a slice of canonical WAL entries, firing each `ProjectionDirective`
/// through `rebuilder`.
///
/// Entries before `start_after_lsn` are skipped so callers can resume from a
/// checkpoint LSN without replaying the full history.
pub fn recover_from_canonical_wal<R: ProjectionRebuilder>(
    entries: &[CanonicalWalEntry],
    rebuilder: &mut R,
    start_after_lsn: u64,
) -> RecoveryResult {
    let mut result = RecoveryResult::default();

    for entry in entries {
        if entry.sequence_number <= start_after_lsn {
            continue;
        }
        result.last_sequence = entry.sequence_number;

        match &entry.operation {
            CanonicalOperation::RecordUpsert {
                record,
                projections,
                ..
            } => {
                result.upserts_replayed += 1;
                for directive in projections {
                    match rebuilder.apply_directive(Some(record.as_ref()), directive) {
                        Ok(()) => result.directives_applied += 1,
                        Err(_) => result.directive_errors += 1,
                    }
                }
            }
            CanonicalOperation::RecordDelete {
                oid: _,
                projections,
                ..
            } => {
                result.deletes_replayed += 1;
                for directive in projections {
                    match rebuilder.apply_directive(None, directive) {
                        Ok(()) => result.directives_applied += 1,
                        Err(_) => result.directive_errors += 1,
                    }
                }
            }
            // A partition drop voids whole-partition state; projection state for
            // the partition is reset by the record-store replay (which owns the
            // partitions), not per-directive here.
            CanonicalOperation::RecordPartitionDrop { .. } => {}
            // Checkpoints, CDC barriers, and catalog mutations do not trigger
            // record/projection rebuilds; the caller uses the checkpoint LSN to
            // scope the replay window, and catalog mutations are folded by the
            // system catalog's own in-RAM authority, not by record recovery.
            CanonicalOperation::Checkpoint(_)
            | CanonicalOperation::CdcBarrier { .. }
            | CanonicalOperation::CatalogMutation { .. } => {}
        }
    }

    result
}

/// Find the highest-LSN `Checkpoint` entry in the log.
///
/// Recovery should call this first, then pass the returned `sequence_number`
/// as `start_after_lsn` to `recover_from_canonical_wal`.
pub fn latest_checkpoint(entries: &[CanonicalWalEntry]) -> Option<&SnapshotManifest> {
    entries.iter().rev().find_map(|e| match &e.operation {
        CanonicalOperation::Checkpoint(m) => Some(m),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Branch / LSN filter (T3.1 Slice 3 — 2026-05-26)
// ---------------------------------------------------------------------------

/// Filter a canonical WAL slice down to `RecordUpsert` entries that belong
/// to a specific branch and fall within an LSN range.
///
/// Used by the graph branch merge orchestrator (ADR-012) to extract one
/// branch's mutations from a shared WAL stream before passing them through
/// the per-branch event lists that `walk_diff` (in `src/graph/merge.rs`)
/// consumes.
///
/// **Scope (T3.1 Slice 3):**
///
/// * Only `CanonicalOperation::RecordUpsert` entries are considered.
/// * The record's `branch_id` MUST equal `Some(branch_id)` — entries with
///   `branch_id: None` are EXCLUDED. (A record with no branch tag has not
///   been tagged for any branch; the orchestrator treats those as "main"
///   via a separate code path, not via this filter.)
/// * `RecordDelete` entries are ALWAYS excluded. They lack a full record
///   and therefore have no branch metadata; the orchestrator tracks
///   deletes via separate context.
/// * `Checkpoint` and `CdcBarrier` entries are ALWAYS excluded. They are
///   recovery / publish markers, not branch-scoped data mutations.
///
/// Returns references into the input slice — caller owns lifetimes. The
/// function is `O(n)` over the input.
///
/// # Example
///
/// ```ignore
/// // Get all RecordUpserts on branch "feature-x" with LSN > merge_base_lsn
/// // and LSN <= current_head_lsn.
/// let entries = filter_wal_by_branch_lsn(
///     &all_entries,
///     "feature-x",
///     (merge_base_lsn + 1)..=current_head_lsn,
/// );
/// ```
pub fn filter_wal_by_branch_lsn<'a, R>(
    entries: &'a [CanonicalWalEntry],
    branch_id: &str,
    lsn_range: R,
) -> Vec<&'a CanonicalWalEntry>
where
    R: std::ops::RangeBounds<u64>,
{
    entries
        .iter()
        .filter(|entry| lsn_range.contains(&entry.sequence_number))
        .filter(|entry| match &entry.operation {
            CanonicalOperation::RecordUpsert { record, .. } => {
                record.branch_id.as_deref() == Some(branch_id)
            }
            _ => false,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Merge-base discovery (T3.1 Slice 4 — 2026-05-26)
// ---------------------------------------------------------------------------

/// Find the LSN at which two branches diverged from their common ancestor.
///
/// The merge base is defined as
/// `min(first_lsn(branch_a), first_lsn(branch_b)) - 1` — the LSN
/// immediately before either branch's first `RecordUpsert` with the
/// given `branch_id`. The orchestrator (ADR-012 §3 step 1, T3.1 Slice 5)
/// passes this LSN to [`filter_wal_by_branch_lsn`] and then to
/// `walk_diff` (in `src/graph/merge.rs`) to scope the diff window.
///
/// **Assumption (T3.1 Slice 4):** both branches share a common ancestor
/// — they were forked from the same parent (typically "main") and have
/// mutated independently since. Asymmetric forks (branch_b forked from
/// branch_a, not from the parent) are out of scope for this slice; the
/// orchestrator (Slice 5+) is responsible for resolving ancestry and
/// passing the correct branch pair.
///
/// Returns `None` if either branch has no `RecordUpsert` entries in the
/// log (i.e. that branch is empty — there is nothing to merge).
///
/// Walks the input slice twice in the worst case (once per branch);
/// `O(n)` over `entries`. Acceptable for v0.2; future slices can switch
/// to a single-pass implementation if profiles show a hot path.
pub fn find_merge_base_lsn(
    entries: &[CanonicalWalEntry],
    branch_a: &str,
    branch_b: &str,
) -> Option<u64> {
    let first_a = first_lsn_for_branch(entries, branch_a)?;
    let first_b = first_lsn_for_branch(entries, branch_b)?;
    Some(first_a.min(first_b).saturating_sub(1))
}

/// Helper: smallest sequence number among `RecordUpsert` entries
/// belonging to `branch_id`. Returns `None` if no such entries exist.
fn first_lsn_for_branch(entries: &[CanonicalWalEntry], branch_id: &str) -> Option<u64> {
    entries
        .iter()
        .filter_map(|entry| match &entry.operation {
            CanonicalOperation::RecordUpsert { record, .. }
                if record.branch_id.as_deref() == Some(branch_id) =>
            {
                Some(entry.sequence_number)
            }
            _ => None,
        })
        .min()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::ProximaRecord;

    // ── test double ──────────────────────────────────────────────────────

    #[derive(Default)]
    struct RecordingRebuilder {
        applied: Vec<(Option<String>, ProjectionDirective)>,
        fail_variant: Option<std::mem::Discriminant<ProjectionDirective>>,
    }

    impl ProjectionRebuilder for RecordingRebuilder {
        fn apply_directive(
            &mut self,
            record: Option<&ProximaRecord>,
            directive: &ProjectionDirective,
        ) -> Result<(), String> {
            if let Some(fail) = self.fail_variant
                && std::mem::discriminant(directive) == fail
            {
                return Err("injected failure".into());
            }
            let oid = record.map(|r| r.oid.clone());
            self.applied.push((oid, directive.clone()));
            Ok(())
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_record(oid: &str) -> ProximaRecord {
        ProximaRecord {
            oid: oid.into(),
            ..Default::default()
        }
    }

    fn upsert_entry(seq: u64, oid: &str, dirs: Vec<ProjectionDirective>) -> CanonicalWalEntry {
        CanonicalWalEntry::new(
            seq,
            CanonicalOperation::RecordUpsert {
                collection_id: "col".into(),
                record: Box::new(make_record(oid)),
                projections: dirs,
            },
            None,
        )
    }

    fn delete_entry(seq: u64, oid: &str, dirs: Vec<ProjectionDirective>) -> CanonicalWalEntry {
        CanonicalWalEntry::new(
            seq,
            CanonicalOperation::RecordDelete {
                collection_id: "col".into(),
                oid: oid.into(),
                projections: dirs,
            },
            None,
        )
    }

    fn checkpoint_entry(seq: u64, collections: Vec<String>) -> CanonicalWalEntry {
        let manifest = SnapshotManifest {
            sequence_number: seq,
            timestamp_ms: 0,
            collection_ids: collections,
            projection_freshness: vec![],
        };
        CanonicalWalEntry::new(seq, CanonicalOperation::Checkpoint(manifest), None)
    }

    // ── CanonicalWalEntry construction ───────────────────────────────────

    #[test]
    fn new_entry_has_correct_sequence_and_tenant() {
        let entry = CanonicalWalEntry::new(
            42,
            CanonicalOperation::RecordUpsert {
                collection_id: "docs".into(),
                record: Box::new(make_record("r1")),
                projections: vec![],
            },
            Some("acme".into()),
        );
        assert_eq!(entry.sequence_number, 42);
        assert_eq!(entry.tenant_id.as_deref(), Some("acme"));
        assert!(!entry.is_checkpoint());
    }

    #[test]
    fn checkpoint_entry_is_identified() {
        let e = checkpoint_entry(10, vec!["docs".into()]);
        assert!(e.is_checkpoint());
    }

    // ── recovery: basic replay ────────────────────────────────────────────

    #[test]
    fn recovery_replays_upserts_and_fires_directives() {
        let dir = ProjectionDirective::DocumentJsonPathIndex {
            collection_id: "docs".into(),
            indexed_paths: vec!["$.title".into()],
        };
        let entries = vec![upsert_entry(1, "r1", vec![dir.clone()])];
        let mut rb = RecordingRebuilder::default();

        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.upserts_replayed, 1);
        assert_eq!(result.directives_applied, 1);
        assert_eq!(result.last_sequence, 1);
        assert_eq!(rb.applied.len(), 1);
        assert_eq!(rb.applied[0].1, dir);
        assert_eq!(rb.applied[0].0.as_deref(), Some("r1"));
    }

    #[test]
    fn recovery_replays_deletes_with_no_record() {
        let dir = ProjectionDirective::AdjacencyTableRow {
            graph_id: "g1".into(),
            node_oid: "n1".into(),
            edge_refs: vec![],
        };
        let entries = vec![delete_entry(1, "n1", vec![dir])];
        let mut rb = RecordingRebuilder::default();

        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.deletes_replayed, 1);
        assert_eq!(result.directives_applied, 1);
        // Delete directives pass None as the record.
        assert_eq!(rb.applied[0].0, None);
    }

    #[test]
    fn recovery_skips_entries_at_or_before_start_lsn() {
        let entries = vec![
            upsert_entry(1, "r1", vec![]),
            upsert_entry(2, "r2", vec![]),
            upsert_entry(3, "r3", vec![]),
        ];
        let mut rb = RecordingRebuilder::default();

        let result = recover_from_canonical_wal(&entries, &mut rb, 2);

        // Only entry 3 replayed.
        assert_eq!(result.upserts_replayed, 1);
        assert_eq!(result.last_sequence, 3);
    }

    #[test]
    fn recovery_skips_checkpoint_and_cdc_barrier_entries() {
        let cdc = CanonicalWalEntry::new(
            2,
            CanonicalOperation::CdcBarrier {
                barrier_sequence: 1,
                events: vec![],
            },
            None,
        );
        let entries = vec![
            upsert_entry(1, "r1", vec![]),
            cdc,
            checkpoint_entry(3, vec![]),
            upsert_entry(4, "r4", vec![]),
        ];
        let mut rb = RecordingRebuilder::default();

        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.upserts_replayed, 2);
        assert_eq!(result.deletes_replayed, 0);
        assert_eq!(result.last_sequence, 4);
    }

    #[test]
    fn recovery_counts_directive_errors_without_stopping() {
        let good = ProjectionDirective::FullTextIndex {
            collection_id: "c".into(),
            indexed_fields: vec!["body".into()],
        };
        let bad = ProjectionDirective::CsrRebuild {
            graph_id: "g".into(),
        };
        let entries = vec![upsert_entry(1, "r1", vec![good.clone(), bad.clone()])];

        let mut rb = RecordingRebuilder {
            fail_variant: Some(std::mem::discriminant(&bad)),
            ..Default::default()
        };
        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.directives_applied, 1);
        assert_eq!(result.directive_errors, 1);
        // Recovery did not abort.
        assert_eq!(result.upserts_replayed, 1);
    }

    // ── latest_checkpoint ────────────────────────────────────────────────

    #[test]
    fn latest_checkpoint_finds_highest_lsn_checkpoint() {
        let entries = vec![
            checkpoint_entry(5, vec!["a".into()]),
            upsert_entry(6, "r6", vec![]),
            checkpoint_entry(10, vec!["a".into(), "b".into()]),
            upsert_entry(11, "r11", vec![]),
        ];
        let cp = latest_checkpoint(&entries).expect("should find a checkpoint");
        assert_eq!(cp.sequence_number, 10);
        assert_eq!(cp.collection_ids.len(), 2);
    }

    #[test]
    fn latest_checkpoint_returns_none_when_no_checkpoint() {
        let entries = vec![upsert_entry(1, "r1", vec![])];
        assert!(latest_checkpoint(&entries).is_none());
    }

    // ── ProjectionDirective variants ──────────────────────────────────────

    #[test]
    fn adjacency_table_directive_carries_edge_refs() {
        let directive = ProjectionDirective::AdjacencyTableRow {
            graph_id: "social".into(),
            node_oid: "alice".into(),
            edge_refs: vec![
                EdgeRef {
                    src_oid: "alice".into(),
                    dst_oid: "bob".into(),
                    edge_type: "FOLLOWS".into(),
                    weight: Some(1.0),
                },
                EdgeRef {
                    src_oid: "carol".into(),
                    dst_oid: "alice".into(),
                    edge_type: "FOLLOWS".into(),
                    weight: None,
                },
            ],
        };
        if let ProjectionDirective::AdjacencyTableRow { edge_refs, .. } = &directive {
            assert_eq!(edge_refs.len(), 2);
            assert!(edge_refs[0].weight.is_some());
            assert!(edge_refs[1].weight.is_none());
        } else {
            panic!("wrong variant");
        }
    }

    // ── SnapshotManifest freshness ────────────────────────────────────────

    #[test]
    fn snapshot_manifest_freshness_tracks_stale_projections() {
        let manifest = SnapshotManifest {
            sequence_number: 100,
            timestamp_ms: 0,
            collection_ids: vec!["docs".into()],
            projection_freshness: vec![
                ProjectionFreshness {
                    projection_name: "doc_json_path:docs".into(),
                    last_rebuilt_lsn: 100,
                    stale: false,
                },
                ProjectionFreshness {
                    projection_name: "fts:docs".into(),
                    last_rebuilt_lsn: 80,
                    stale: true,
                },
            ],
        };
        let stale: Vec<_> = manifest
            .projection_freshness
            .iter()
            .filter(|p| p.stale)
            .collect();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].projection_name, "fts:docs");
    }

    // ── CDC canonical event ───────────────────────────────────────────────

    #[test]
    fn cdc_event_carries_logical_views() {
        let event = CdcRecordEvent {
            event_id: "evt-1".into(),
            sequence_number: 7,
            timestamp_ms: 0,
            collection_id: "docs".into(),
            operation: CdcOperation::Insert,
            record_after: Some(make_record("doc-1")),
            logical_views: vec![
                CdcLogicalView::DocumentChange {
                    collection_id: "docs".into(),
                    document_json: serde_json::json!({"title": "hello"}),
                },
                CdcLogicalView::GraphNodeChange {
                    graph_id: "kg".into(),
                    node_json: serde_json::json!({"id": "doc-1"}),
                },
            ],
        };
        assert_eq!(event.operation, CdcOperation::Insert);
        assert!(event.record_after.is_some());
        assert_eq!(event.logical_views.len(), 2);
    }

    #[test]
    fn cdc_delete_event_has_no_record_after() {
        let event = CdcRecordEvent {
            event_id: "evt-2".into(),
            sequence_number: 8,
            timestamp_ms: 0,
            collection_id: "nodes".into(),
            operation: CdcOperation::Delete,
            record_after: None,
            logical_views: vec![],
        };
        assert_eq!(event.operation, CdcOperation::Delete);
        assert!(event.record_after.is_none());
    }

    // ── crash/recovery: checkpoint-scoped replay ──────────────────────────

    #[test]
    fn crash_recovery_resumes_from_checkpoint_lsn() {
        // Simulate a log with an earlier checkpoint and entries after it.
        let dir = ProjectionDirective::HnswIndex {
            collection_id: "vecs".into(),
            embedding_field: "embedding".into(),
        };
        let entries = vec![
            upsert_entry(1, "pre-cp-1", vec![dir.clone()]),
            upsert_entry(2, "pre-cp-2", vec![dir.clone()]),
            checkpoint_entry(3, vec!["vecs".into()]),
            upsert_entry(4, "post-cp-1", vec![dir.clone()]),
            upsert_entry(5, "post-cp-2", vec![dir.clone()]),
        ];

        // Recovery finds the checkpoint at LSN 3, then replays only after it.
        let checkpoint_lsn = latest_checkpoint(&entries)
            .map(|m| m.sequence_number)
            .unwrap_or(0);
        assert_eq!(checkpoint_lsn, 3);

        let mut rb = RecordingRebuilder::default();
        let result = recover_from_canonical_wal(&entries, &mut rb, checkpoint_lsn);

        // Only entries 4 and 5 replayed.
        assert_eq!(result.upserts_replayed, 2);
        assert_eq!(result.directives_applied, 2);
        let replayed_oids: Vec<_> = rb
            .applied
            .iter()
            .filter_map(|(oid, _)| oid.as_deref())
            .collect();
        assert!(replayed_oids.contains(&"post-cp-1"));
        assert!(replayed_oids.contains(&"post-cp-2"));
        assert!(!replayed_oids.contains(&"pre-cp-1"));
    }

    #[test]
    fn crash_recovery_graph_adjacency_directives_all_fired() {
        let edge_dir = ProjectionDirective::AdjacencyTableRow {
            graph_id: "kg".into(),
            node_oid: "alice".into(),
            edge_refs: vec![EdgeRef {
                src_oid: "alice".into(),
                dst_oid: "bob".into(),
                edge_type: "KNOWS".into(),
                weight: None,
            }],
        };
        let csr_dir = ProjectionDirective::CsrRebuild {
            graph_id: "kg".into(),
        };

        let entries = vec![
            upsert_entry(1, "alice", vec![edge_dir.clone()]),
            upsert_entry(2, "bob", vec![edge_dir.clone(), csr_dir.clone()]),
        ];
        let mut rb = RecordingRebuilder::default();
        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.directives_applied, 3);
        let csr_count = rb
            .applied
            .iter()
            .filter(|(_, d)| matches!(d, ProjectionDirective::CsrRebuild { .. }))
            .count();
        assert_eq!(csr_count, 1);
    }

    #[test]
    fn crash_recovery_observability_projection_directives_are_typed() {
        let trace_dir = ProjectionDirective::ObservabilityTraceIndex {
            collection_id: "otel_spans".into(),
            service_name: Some("checkout".into()),
            trace_id_field: "trace_id".into(),
            span_id_field: "span_id".into(),
        };
        let rollup_dir = ProjectionDirective::TimeSeriesRollup {
            collection_id: "metrics".into(),
            metric_name: "http.server.duration".into(),
            window_ms: 60_000,
            dimensions: vec!["service.name".into()],
        };

        let entries = vec![upsert_entry(
            1,
            "span-1",
            vec![trace_dir.clone(), rollup_dir.clone()],
        )];
        let mut rb = RecordingRebuilder::default();
        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.directives_applied, 2);
        assert!(rb.applied.iter().any(|(_, directive)| matches!(
            directive,
            ProjectionDirective::ObservabilityTraceIndex { .. }
        )));
        assert!(rb.applied.iter().any(|(_, directive)| matches!(
            directive,
            ProjectionDirective::TimeSeriesRollup { .. }
        )));
    }

    #[test]
    fn crash_recovery_open_format_manifest_directive_is_typed() {
        let manifest_dir = ProjectionDirective::OpenFormatManifest {
            namespace: "main".into(),
            table_name: "events".into(),
            format: OpenTableFormat::Iceberg,
        };

        let entries = vec![upsert_entry(1, "evt-1", vec![manifest_dir])];
        let mut rb = RecordingRebuilder::default();
        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.directives_applied, 1);
        assert!(matches!(
            rb.applied.first().map(|(_, directive)| directive),
            Some(ProjectionDirective::OpenFormatManifest {
                format: OpenTableFormat::Iceberg,
                ..
            })
        ));
    }

    #[test]
    fn projection_corruption_repaired_by_full_replay_from_zero() {
        // Simulate projection corruption: ignore the checkpoint and replay
        // everything from LSN 0 to get a clean rebuild.
        let dir = ProjectionDirective::DocumentJsonPathIndex {
            collection_id: "docs".into(),
            indexed_paths: vec!["$.tags".into()],
        };
        let entries = vec![
            upsert_entry(1, "d1", vec![dir.clone()]),
            upsert_entry(2, "d2", vec![dir.clone()]),
            checkpoint_entry(3, vec!["docs".into()]),
            upsert_entry(4, "d3", vec![dir.clone()]),
        ];
        let mut rb = RecordingRebuilder::default();
        // Full replay (start_after_lsn = 0) rebuilds even pre-checkpoint entries.
        let result = recover_from_canonical_wal(&entries, &mut rb, 0);

        assert_eq!(result.upserts_replayed, 3);
        assert_eq!(result.directives_applied, 3);
    }

    // ── T3.1 Slice 3 — filter_wal_by_branch_lsn tests ──────────────────────

    fn branch_upsert(seq: u64, oid: &str, branch: Option<&str>) -> CanonicalWalEntry {
        let mut record = make_record(oid);
        record.branch_id = branch.map(String::from);
        CanonicalWalEntry::new(
            seq,
            CanonicalOperation::RecordUpsert {
                collection_id: "col".into(),
                record: Box::new(record),
                projections: vec![],
            },
            None,
        )
    }

    #[test]
    fn filter_empty_input_returns_empty() {
        let out = filter_wal_by_branch_lsn(&[], "any", 0..u64::MAX);
        assert!(out.is_empty());
    }

    #[test]
    fn filter_lsn_range_drops_pre_range_entries() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            branch_upsert(5, "b", Some("dev")),
            branch_upsert(10, "c", Some("dev")),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 5..=10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sequence_number, 5);
        assert_eq!(out[1].sequence_number, 10);
    }

    #[test]
    fn filter_lsn_range_drops_post_range_entries() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            branch_upsert(5, "b", Some("dev")),
            branch_upsert(10, "c", Some("dev")),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sequence_number, 1);
    }

    #[test]
    fn filter_keeps_only_matching_branch_id() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            branch_upsert(2, "b", Some("prod")),
            branch_upsert(3, "c", Some("dev")),
            branch_upsert(4, "d", Some("feature-x")),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..u64::MAX);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sequence_number, 1);
        assert_eq!(out[1].sequence_number, 3);
    }

    #[test]
    fn filter_excludes_records_with_no_branch_id() {
        // Records with branch_id == None should be excluded from any
        // specific-branch filter. They are not part of "any" branch.
        let entries = vec![
            branch_upsert(1, "a", None),
            branch_upsert(2, "b", Some("dev")),
            branch_upsert(3, "c", None),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..u64::MAX);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sequence_number, 2);
    }

    #[test]
    fn filter_excludes_record_delete_entries() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            delete_entry(2, "b", vec![]),
            branch_upsert(3, "c", Some("dev")),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..u64::MAX);
        // RecordDelete excluded regardless of LSN/branch.
        assert_eq!(out.len(), 2);
        assert!(matches!(
            out[0].operation,
            CanonicalOperation::RecordUpsert { .. }
        ));
    }

    #[test]
    fn filter_excludes_checkpoint_entries() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            checkpoint_entry(2, vec!["col".into()]),
            branch_upsert(3, "c", Some("dev")),
        ];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..u64::MAX);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sequence_number, 1);
        assert_eq!(out[1].sequence_number, 3);
    }

    #[test]
    fn filter_excludes_cdc_barrier_entries() {
        let cdc = CanonicalWalEntry::new(
            5,
            CanonicalOperation::CdcBarrier {
                barrier_sequence: 4,
                events: vec![],
            },
            None,
        );
        let entries = vec![branch_upsert(1, "a", Some("dev")), cdc];
        let out = filter_wal_by_branch_lsn(&entries, "dev", 0..u64::MAX);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sequence_number, 1);
    }

    #[test]
    fn filter_combines_lsn_and_branch_filters() {
        let entries = vec![
            branch_upsert(1, "a", Some("dev")),
            branch_upsert(2, "b", Some("prod")),
            branch_upsert(3, "c", Some("dev")),
            branch_upsert(4, "d", Some("dev")),
            delete_entry(5, "e", vec![]),
            branch_upsert(6, "f", Some("dev")),
        ];
        // dev branch entries in LSN range [2, 4]: should return seq 3, 4.
        let out = filter_wal_by_branch_lsn(&entries, "dev", 2..=4);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].sequence_number, 3);
        assert_eq!(out[1].sequence_number, 4);
    }

    // ── T3.1 Slice 4 — find_merge_base_lsn tests ───────────────────────────

    #[test]
    fn find_merge_base_returns_none_for_empty_input() {
        assert_eq!(find_merge_base_lsn(&[], "a", "b"), None);
    }

    #[test]
    fn find_merge_base_returns_none_when_branch_a_missing() {
        let entries = vec![branch_upsert(3, "x", Some("b"))];
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), None);
    }

    #[test]
    fn find_merge_base_returns_none_when_branch_b_missing() {
        let entries = vec![branch_upsert(3, "x", Some("a"))];
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), None);
    }

    #[test]
    fn find_merge_base_returns_first_lsn_minus_one_when_both_present() {
        let entries = vec![
            branch_upsert(5, "x", Some("a")),
            branch_upsert(7, "y", Some("b")),
        ];
        // min(5, 7) - 1 = 4
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(4));
    }

    #[test]
    fn find_merge_base_picks_min_when_branches_first_at_different_lsns() {
        let entries = vec![
            branch_upsert(10, "x", Some("a")),
            branch_upsert(3, "y", Some("b")),
            branch_upsert(12, "z", Some("a")),
            branch_upsert(8, "w", Some("b")),
        ];
        // first_lsn("a") = 10, first_lsn("b") = 3 → min = 3 → merge_base = 2.
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(2));
    }

    #[test]
    fn find_merge_base_returns_zero_when_earliest_first_lsn_is_one() {
        let entries = vec![
            branch_upsert(1, "x", Some("a")),
            branch_upsert(5, "y", Some("b")),
        ];
        // min(1, 5) - 1 = 0 via saturating_sub.
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(0));
    }

    #[test]
    fn find_merge_base_returns_zero_when_earliest_first_lsn_is_zero() {
        let entries = vec![
            branch_upsert(0, "x", Some("a")),
            branch_upsert(5, "y", Some("b")),
        ];
        // saturating_sub(1) on 0 = 0.
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(0));
    }

    #[test]
    fn find_merge_base_ignores_non_upsert_entries() {
        // Only RecordUpsert carries branch_id. Checkpoint, RecordDelete,
        // CdcBarrier — even at lower LSNs — must not be considered.
        let cdc = CanonicalWalEntry::new(
            1,
            CanonicalOperation::CdcBarrier {
                barrier_sequence: 0,
                events: vec![],
            },
            None,
        );
        let entries = vec![
            cdc,
            checkpoint_entry(2, vec!["col".into()]),
            delete_entry(3, "victim", vec![]),
            branch_upsert(5, "x", Some("a")),
            branch_upsert(7, "y", Some("b")),
        ];
        // Only LSNs 5 and 7 count. min(5, 7) - 1 = 4.
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(4));
    }

    #[test]
    fn find_merge_base_works_when_branches_first_at_same_lsn() {
        // Forked simultaneously at LSN 5 — both branches' first record
        // is at the same sequence number.
        let entries = vec![
            branch_upsert(5, "x", Some("a")),
            branch_upsert(5, "y", Some("b")),
        ];
        // min(5, 5) - 1 = 4.
        assert_eq!(find_merge_base_lsn(&entries, "a", "b"), Some(4));
    }
}
