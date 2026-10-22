//! Graph branch merge semantics (ADR-012).
//!
//! Defines CRDT-based conflict resolution policies for graph mutations across
//! concurrent branches. All durable graph truth flows through canonical
//! `ProximaRecord` WAL commits; this module provides the semantic contracts
//! that govern how conflicting mutations from two branches are reconciled before
//! the merged WAL fragment is written.
//!
//! ## Merge protocol (ADR-012 §3)
//!
//! 1. Identify the common base WAL position for the two branches.
//! 2. Classify each mutation in the diff against the table below.
//! 3. Resolve conflicts using the per-class policy.
//! 4. Commit the resolved mutation set as a single WAL fragment.
//! 5. Trigger incremental projection rebuild (HNSW, CSR) for affected records.
//!
//! ## Per-class policies
//!
//! | Mutation class      | Policy                        |
//! |---------------------|-------------------------------|
//! | Node upsert         | Last-Write-Wins (LWW by WAL ts) |
//! | Node delete         | Delete-wins (2P-Set semantics) |
//! | Edge upsert         | LWW by WAL ts                 |
//! | Edge delete         | Delete-wins (2P-Set semantics) |
//! | Embedding update    | LWW by WAL ts                 |
//! | Label set           | Add-Wins Set union             |
//! | Props key           | LWW per key                   |

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

use crate::services::record_store::TableWalAppender;

/// Mutation class used to select the merge policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationClass {
    NodeUpsert,
    NodeDelete,
    EdgeUpsert,
    EdgeDelete,
    EmbeddingUpdate,
    LabelSet,
    PropsKey,
}

impl MutationClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NodeUpsert => "node_upsert",
            Self::NodeDelete => "node_delete",
            Self::EdgeUpsert => "edge_upsert",
            Self::EdgeDelete => "edge_delete",
            Self::EmbeddingUpdate => "embedding_update",
            Self::LabelSet => "label_set",
            Self::PropsKey => "props_key",
        }
    }
}

/// CRDT merge policy applied to a conflicting mutation pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergePolicy {
    /// Accept the mutation with the higher WAL timestamp; discard the other.
    LastWriteWins,
    /// A delete on either branch wins over an upsert on the other.
    DeleteWins,
    /// Union of label/tag sets from both branches; no label is lost.
    AddWinsSetUnion,
    /// LWW applied independently per props key rather than per record.
    LastWriteWinsPerKey,
}

impl MergePolicy {
    pub fn for_class(class: MutationClass) -> Self {
        match class {
            MutationClass::NodeUpsert => Self::LastWriteWins,
            MutationClass::NodeDelete => Self::DeleteWins,
            MutationClass::EdgeUpsert => Self::LastWriteWins,
            MutationClass::EdgeDelete => Self::DeleteWins,
            MutationClass::EmbeddingUpdate => Self::LastWriteWins,
            MutationClass::LabelSet => Self::AddWinsSetUnion,
            MutationClass::PropsKey => Self::LastWriteWinsPerKey,
        }
    }
}

/// A conflict detected between two branches for a single record or key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    /// Canonical record ID (node or edge).
    pub record_id: String,
    /// Mutation class that produced the conflict.
    pub mutation_class: MutationClass,
    /// WAL timestamp on the left (base) branch (microseconds since epoch).
    pub left_wal_ts_us: i64,
    /// WAL timestamp on the right (incoming) branch.
    pub right_wal_ts_us: i64,
    /// Policy that will be applied.
    pub policy: MergePolicy,
}

impl MergeConflict {
    pub fn new(
        record_id: impl Into<String>,
        mutation_class: MutationClass,
        left_wal_ts_us: i64,
        right_wal_ts_us: i64,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            policy: MergePolicy::for_class(mutation_class),
            mutation_class,
            left_wal_ts_us,
            right_wal_ts_us,
        }
    }

    /// Returns true when the right (incoming) branch wins under LWW.
    pub fn right_wins_lww(&self) -> bool {
        self.right_wal_ts_us > self.left_wal_ts_us
    }
}

/// Outcome of resolving a single [`MergeConflict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeOutcome {
    /// Accept the left (base) branch mutation; discard the right.
    KeepLeft,
    /// Accept the right (incoming) branch mutation; discard the left.
    KeepRight,
    /// Both branches are deleted; the record is absent after merge.
    BothDeleted,
    /// Union of label sets from both branches is the resolved value.
    UnionLabels,
}

/// Resolution for one conflict, pairing the conflict descriptor with its outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMergeResolution {
    pub conflict: MergeConflict,
    pub outcome: MergeOutcome,
}

impl GraphMergeResolution {
    /// Apply ADR-012 policy to produce a resolution for the given conflict.
    pub fn resolve(conflict: MergeConflict) -> Self {
        let outcome = match conflict.policy {
            MergePolicy::LastWriteWins | MergePolicy::LastWriteWinsPerKey => {
                if conflict.right_wins_lww() {
                    MergeOutcome::KeepRight
                } else {
                    MergeOutcome::KeepLeft
                }
            }
            MergePolicy::DeleteWins => MergeOutcome::BothDeleted,
            MergePolicy::AddWinsSetUnion => MergeOutcome::UnionLabels,
        };
        Self { conflict, outcome }
    }
}

/// Convenience: resolve a batch of conflicts.
pub fn resolve_batch(conflicts: Vec<MergeConflict>) -> Vec<GraphMergeResolution> {
    conflicts
        .into_iter()
        .map(GraphMergeResolution::resolve)
        .collect()
}

/// ADR-012 Add-Wins label union: merge two label vectors, deduplicate, preserve order.
///
/// Labels from both sides are preserved (Add-Wins Set Union). Neither side's labels
/// are removed. Use this when updating a node whose label set may have been
/// independently modified in concurrent branches.
pub fn merge_labels(existing: &[String], incoming: &[String]) -> Vec<String> {
    let mut merged = existing.to_vec();
    let existing_set: std::collections::HashSet<&str> =
        existing.iter().map(String::as_str).collect();
    for label in incoming {
        if !existing_set.contains(label.as_str()) {
            merged.push(label.clone());
        }
    }
    merged
}

// ──────────────────────────────────────────────────────────────────────────────
// T3.1 Slice 1 (2026-05-26) — WAL diff walker (pure logic).
//
// The full merge protocol from ADR-012 §3 is:
//   1. Identify merge base (the WAL position where two branches diverged).
//   2. Walk the WAL diff between merge base and each branch HEAD.
//   3. Classify each mutation into one of the 7 `MutationClass` variants.
//   4. Resolve conflicts per `MergePolicy::for_class` (already implemented).
//   5. Write merged records through the canonical WAL.
//
// This slice lands steps 3-4 at the pure-logic boundary. Steps 1-2 + 5
// originally required `branch_id` plumbing into `ProximaRecord` and integration
// with the canonical WAL iteration API. Those landed in follow-up slices; this
// module now consumes branch-tagged canonical WAL entries directly.
//
// The caller is responsible for extracting per-branch mutation events into
// the input lists. Branch identity is implicit: "which list."
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal mutation event consumed by the WAL diff walker.
///
/// Constructed by the caller (typically by walking canonical WAL entries for
/// one branch). The walker compares events across two such lists and emits
/// `MergeConflict` entries for OIDs touched on both sides.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MutationEvent {
    /// Canonical record OID (node or edge).
    pub record_oid: String,
    /// Classification of the mutation (already determined by the caller; in
    /// future slices this comes from `classify_mutation` below applied to the
    /// base + branch versions of the record).
    pub mutation_class: MutationClass,
    /// WAL timestamp in microseconds since epoch.
    pub wal_ts_us: i64,
}

impl MutationEvent {
    pub fn new(
        record_oid: impl Into<String>,
        mutation_class: MutationClass,
        wal_ts_us: i64,
    ) -> Self {
        Self {
            record_oid: record_oid.into(),
            mutation_class,
            wal_ts_us,
        }
    }
}

/// Walk two ordered lists of mutation events (one per branch) and produce
/// `MergeConflict` entries for every record OID touched on BOTH sides.
///
/// For each side, the event with the latest `wal_ts_us` is the
/// representative — earlier events are superseded. The mutation class is
/// taken from the left side; if the right side disagrees on class, that
/// indicates an upstream classifier inconsistency and the right side's
/// class is preserved on the resulting `MergeConflict` (the caller can
/// reconcile if needed — but in normal use both sides should classify the
/// same record OID into the same class).
///
/// OIDs that appear on only one side are NOT emitted as conflicts; they're
/// clean unilateral mutations that the orchestrator applies directly.
pub fn walk_diff(left: &[MutationEvent], right: &[MutationEvent]) -> Vec<MergeConflict> {
    use std::collections::HashMap;

    // Build "latest event per OID" maps for each side. `O(n)` per side.
    let mut left_latest: HashMap<&str, &MutationEvent> = HashMap::new();
    for ev in left {
        let entry = left_latest.entry(ev.record_oid.as_str()).or_insert(ev);
        if ev.wal_ts_us > entry.wal_ts_us {
            *entry = ev;
        }
    }
    let mut right_latest: HashMap<&str, &MutationEvent> = HashMap::new();
    for ev in right {
        let entry = right_latest.entry(ev.record_oid.as_str()).or_insert(ev);
        if ev.wal_ts_us > entry.wal_ts_us {
            *entry = ev;
        }
    }

    // For OIDs in BOTH maps, emit a conflict. Prefer the left side's class
    // (in agreement runs this matches the right side's class anyway; in
    // disagreement runs the orchestrator should reconcile upstream).
    let mut conflicts: Vec<MergeConflict> = left_latest
        .iter()
        .filter_map(|(oid, l)| {
            right_latest
                .get(oid)
                .map(|r| MergeConflict::new(*oid, l.mutation_class, l.wal_ts_us, r.wal_ts_us))
        })
        .collect();

    // Deterministic ordering — by OID — so callers (and tests) can compare
    // outputs without depending on HashMap iteration order.
    conflicts.sort_by(|a, b| a.record_id.cmp(&b.record_id));
    conflicts
}

/// Classify a record mutation across base + branch_a + branch_b versions.
///
/// Implements the precedence rules from ADR-012 §3 (codified during Phase 1
/// audit). When base is `None`, treat as new-record creation on the
/// branch(es). When any branch is `None`, only the existing branch's
/// classification matters.
///
/// Precedence (top wins):
///   1. Tombstone on either branch → `NodeDelete` / `EdgeDelete`.
///   2. Pure embedding change → `EmbeddingUpdate`.
///   3. Pure label change → `LabelSet`.
///   4. Any props change → `PropsKey`.
///   5. Else, if any version is an edge → `EdgeUpsert`.
///   6. Default → `NodeUpsert`.
pub fn classify_mutation(
    base: Option<&proximadb_records::ProximaRecord>,
    branch_a: Option<&proximadb_records::ProximaRecord>,
    branch_b: Option<&proximadb_records::ProximaRecord>,
) -> MutationClass {
    let is_tombstone = |r: Option<&proximadb_records::ProximaRecord>| -> bool {
        r.is_some_and(|rec| {
            rec.valid_to_ns.is_some_and(|vt| vt <= 0)
                && rec.embeddings.is_empty()
                && rec.origin.as_deref() == Some("delete")
        })
    };
    let is_edge = |r: Option<&proximadb_records::ProximaRecord>| -> bool {
        r.is_some_and(|rec| rec.edge.is_some())
    };
    let embeddings_changed = |old: Option<&proximadb_records::ProximaRecord>,
                              new: Option<&proximadb_records::ProximaRecord>|
     -> bool {
        match (old, new) {
            (Some(o), Some(n)) => o.embeddings != n.embeddings,
            _ => false,
        }
    };
    let labels_changed = |old: Option<&proximadb_records::ProximaRecord>,
                          new: Option<&proximadb_records::ProximaRecord>|
     -> bool {
        match (old, new) {
            (Some(o), Some(n)) => o.labels != n.labels,
            _ => false,
        }
    };
    let props_changed = |old: Option<&proximadb_records::ProximaRecord>,
                         new: Option<&proximadb_records::ProximaRecord>|
     -> bool {
        match (old, new) {
            (Some(o), Some(n)) => o.props != n.props,
            _ => false,
        }
    };

    // 1. Delete-wins: tombstone on either branch.
    if is_tombstone(branch_a) || is_tombstone(branch_b) {
        if is_edge(base) || is_edge(branch_a) || is_edge(branch_b) {
            return MutationClass::EdgeDelete;
        }
        return MutationClass::NodeDelete;
    }

    let a_emb = embeddings_changed(base, branch_a);
    let b_emb = embeddings_changed(base, branch_b);
    let a_lbl = labels_changed(base, branch_a);
    let b_lbl = labels_changed(base, branch_b);
    let a_props = props_changed(base, branch_a);
    let b_props = props_changed(base, branch_b);

    // 2. Pure embedding change (no other mutations on either branch).
    if (a_emb || b_emb) && !a_lbl && !b_lbl && !a_props && !b_props {
        return MutationClass::EmbeddingUpdate;
    }

    // 3. Pure label change.
    if (a_lbl || b_lbl) && !a_emb && !b_emb && !a_props && !b_props {
        return MutationClass::LabelSet;
    }

    // 4. Any props change wins next.
    if a_props || b_props {
        return MutationClass::PropsKey;
    }

    // 5. Edge upsert default if any version is an edge.
    if is_edge(base) || is_edge(branch_a) || is_edge(branch_b) {
        return MutationClass::EdgeUpsert;
    }

    // 6. Default — node upsert.
    MutationClass::NodeUpsert
}

// ──────────────────────────────────────────────────────────────────────────────
// T3.1 Slice 5 (2026-05-26) — GraphBranchMerger orchestrator.
//
// Composes Slices 1+3+4 into the merge protocol from ADR-012 §3 (steps 1-5).
// Step 6 (write merged records back through the canonical WAL) is deferred to
// Slice 6 — this slice is a pure orchestrator that returns a `MergeReport`
// for the caller to act on.
//
//   1. Identify merge base       → find_merge_base_lsn         (Slice 4)
//   2. Filter per-branch diff    → filter_wal_by_branch_lsn    (Slice 3)
//   3. Classify mutations        → entry_to_mutation_event     (NEW here)
//   4. Walk diff for conflicts   → walk_diff                   (Slice 1)
//   5. Resolve conflicts         → resolve_batch               (Slice 1)
// ──────────────────────────────────────────────────────────────────────────────

/// Outcome of a merge operation (T3.1 Slice 5).
///
/// Returned by [`merge_branches`]. The caller is responsible for what to do
/// next — typically: inspect `conflicts` and `resolutions`, then write the
/// resolved records back through the canonical WAL (Slice 6+). This slice
/// is deliberately a read-only orchestrator.
#[derive(Debug, Clone)]
pub struct MergeReport {
    /// LSN at which the two branches diverged (per `find_merge_base_lsn`).
    pub merge_base_lsn: u64,
    /// `MutationEvent` derived from the left branch's WAL slice (LSN > merge_base).
    pub left_events: Vec<MutationEvent>,
    /// `MutationEvent` derived from the right branch's WAL slice.
    pub right_events: Vec<MutationEvent>,
    /// Conflicts detected by `walk_diff` over the two event lists.
    pub conflicts: Vec<MergeConflict>,
    /// Resolutions per the policy table in ADR-012 §3.
    pub resolutions: Vec<GraphMergeResolution>,
}

/// Compose the merge protocol from ADR-012 §3 (steps 1-5; step 6 is the
/// caller's job).
///
/// Returns `None` when either branch has no `RecordUpsert` entries in the
/// WAL — there is nothing to merge in that case.
///
/// Conflict classification reconstructs the latest base, left, and right
/// record states from the supplied WAL slice and applies [`classify_mutation`]
/// to the pairwise conflict. This lets the report distinguish
/// `EmbeddingUpdate`, `LabelSet`, and `PropsKey` conflicts instead of treating
/// every non-delete node mutation as a generic `NodeUpsert`.
pub fn merge_branches(
    wal: &[proximadb_storage_common::CanonicalWalEntry],
    branch_a: &str,
    branch_b: &str,
) -> Option<MergeReport> {
    let merge_base_lsn = proximadb_storage_common::find_merge_base_lsn(wal, branch_a, branch_b)?;
    let base_records = latest_base_records(wal, ..=merge_base_lsn);
    Some(merge_from_base(
        wal,
        merge_base_lsn,
        base_records,
        branch_a,
        branch_b,
    ))
}

/// The main-line base of one branched `(tenant, collection)` as of WAL
/// sequence [`Self::through`]: the latest untagged upsert per oid at or below
/// it (TD-SEAM-12 hazard 2). Kept so a merge still has its base once the
/// history below `through` is truncated: the base at a merge base `M >=
/// through` is this image plus the untagged upserts in `(through, M]`, which
/// is what [`merge_branches`] computes from the full history.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MergeBaseImage {
    /// Highest sequence folded in.
    pub through: u64,
    /// oid → latest untagged upsert at or below `through`.
    pub records: HashMap<String, proximadb_records::ProximaRecord>,
}

impl MergeBaseImage {
    /// Fold the untagged upserts of `entries` in `(self.through, through]`
    /// into the image and advance it to `through` (never backwards).
    /// `entries` are one tenant's entries of one collection, in any order.
    pub fn advance(
        &mut self,
        entries: &[proximadb_storage_common::CanonicalWalEntry],
        through: u64,
    ) {
        if through < self.through {
            return;
        }
        for (oid, record) in latest_base_records(entries, after_through(self.through, through)) {
            self.records.insert(oid, record.clone());
        }
        self.through = through;
    }
}

/// [`merge_branches_with_base`] cannot use the image it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeBaseError {
    /// The image folds in main-line writes after the branches diverged, so the
    /// base they diverged from cannot be recovered from it.
    ImageAboveMergeBase { through: u64, merge_base_lsn: u64 },
}

impl std::fmt::Display for MergeBaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageAboveMergeBase {
                through,
                merge_base_lsn,
            } => write!(
                f,
                "the merge-base image runs through sequence {through}, past the merge base \
                 {merge_base_lsn}"
            ),
        }
    }
}

impl std::error::Error for MergeBaseError {}

/// [`merge_branches`] over a WAL whose history at or below `base.through` may
/// be gone: the main-line base comes from `base`, plus the untagged upserts
/// of `wal` in `(base.through, merge base]`. Entries of `wal` at or below
/// `base.through` are ignored. With the image of the truncated prefix, the
/// report equals [`merge_branches`] over the untruncated history.
pub fn merge_branches_with_base(
    wal: &[proximadb_storage_common::CanonicalWalEntry],
    base: &MergeBaseImage,
    branch_a: &str,
    branch_b: &str,
) -> std::result::Result<Option<MergeReport>, MergeBaseError> {
    let Some(merge_base_lsn) =
        proximadb_storage_common::find_merge_base_lsn(wal, branch_a, branch_b)
    else {
        return Ok(None);
    };
    if base.through > merge_base_lsn {
        return Err(MergeBaseError::ImageAboveMergeBase {
            through: base.through,
            merge_base_lsn,
        });
    }
    let mut base_records: HashMap<String, &proximadb_records::ProximaRecord> = base
        .records
        .iter()
        .map(|(oid, record)| (oid.clone(), record))
        .collect();
    base_records.extend(latest_base_records(
        wal,
        after_through(base.through, merge_base_lsn),
    ));
    Ok(Some(merge_from_base(
        wal,
        merge_base_lsn,
        base_records,
        branch_a,
        branch_b,
    )))
}

/// Sequences above an image's `through` up to `upto`. An image through `0`
/// has folded nothing yet, so the range then starts unbounded.
fn after_through(through: u64, upto: u64) -> (std::ops::Bound<u64>, std::ops::Bound<u64>) {
    let start = if through == 0 {
        std::ops::Bound::Unbounded
    } else {
        std::ops::Bound::Excluded(through)
    };
    (start, std::ops::Bound::Included(upto))
}

/// Steps 2–5 of the merge protocol, given the merge base and its records.
fn merge_from_base(
    wal: &[proximadb_storage_common::CanonicalWalEntry],
    merge_base_lsn: u64,
    base_records: HashMap<String, &proximadb_records::ProximaRecord>,
    branch_a: &str,
    branch_b: &str,
) -> MergeReport {
    let diff_range = merge_base_lsn.saturating_add(1)..;

    let left_entries =
        proximadb_storage_common::filter_wal_by_branch_lsn(wal, branch_a, diff_range.clone());
    let right_entries =
        proximadb_storage_common::filter_wal_by_branch_lsn(wal, branch_b, diff_range);

    let left_events: Vec<MutationEvent> = left_entries
        .iter()
        .filter_map(|e| entry_to_mutation_event_with_base(e, &base_records))
        .collect();
    let right_events: Vec<MutationEvent> = right_entries
        .iter()
        .filter_map(|e| entry_to_mutation_event_with_base(e, &base_records))
        .collect();

    let mut conflicts = walk_diff(&left_events, &right_events);
    refine_conflicts_with_pairwise_records(
        &mut conflicts,
        &left_entries,
        &right_entries,
        &base_records,
    );
    let resolutions = resolve_batch(conflicts.clone());

    MergeReport {
        merge_base_lsn,
        left_events,
        right_events,
        conflicts,
        resolutions,
    }
}

/// Per-entry heuristic classification: takes a single canonical WAL entry
/// and produces a `MutationEvent` if the entry is a `RecordUpsert`.
///
/// Returns `None` for non-upsert entries (`RecordDelete`, `Checkpoint`,
/// `CdcBarrier`); the orchestrator handles those via separate context.
///
/// Class mapping:
///   - `valid_to_ns <= 0 && embeddings.is_empty() && origin == "delete"`
///     → tombstone → `NodeDelete` (or `EdgeDelete` if `edge.is_some`)
///   - `edge.is_some()` → `EdgeUpsert`
///   - otherwise → `NodeUpsert`
///
/// Timestamp: `entry.timestamp_ms * 1000` (ms → us), saturating on overflow.
#[cfg(test)]
fn entry_to_mutation_event(
    entry: &proximadb_storage_common::CanonicalWalEntry,
) -> Option<MutationEvent> {
    entry_to_mutation_event_with_base(entry, &std::collections::HashMap::new())
}

fn entry_to_mutation_event_with_base(
    entry: &proximadb_storage_common::CanonicalWalEntry,
    base_records: &std::collections::HashMap<String, &proximadb_records::ProximaRecord>,
) -> Option<MutationEvent> {
    use proximadb_storage_common::CanonicalOperation;
    match &entry.operation {
        CanonicalOperation::RecordUpsert { record, .. } => {
            let base = base_records.get(record.oid.as_str()).copied();
            let class = classify_mutation(base, Some(record), None);
            let wal_ts_us = (entry.timestamp_ms as i64).saturating_mul(1_000);
            Some(MutationEvent::new(record.oid.clone(), class, wal_ts_us))
        }
        _ => None,
    }
}

/// Latest untagged (main-line) upsert per oid among the entries whose
/// sequence lies in `seqs`.
fn latest_base_records(
    wal: &[proximadb_storage_common::CanonicalWalEntry],
    seqs: impl std::ops::RangeBounds<u64>,
) -> std::collections::HashMap<String, &proximadb_records::ProximaRecord> {
    use proximadb_storage_common::CanonicalOperation;

    let mut latest: std::collections::HashMap<String, (u64, &proximadb_records::ProximaRecord)> =
        std::collections::HashMap::new();
    for entry in wal {
        if !seqs.contains(&entry.sequence_number) {
            continue;
        }
        if let CanonicalOperation::RecordUpsert { record, .. } = &entry.operation {
            if record.branch_id.is_some() {
                continue;
            }
            let replace = latest
                .get(record.oid.as_str())
                .is_none_or(|(seq, _)| entry.sequence_number >= *seq);
            if replace {
                latest.insert(record.oid.clone(), (entry.sequence_number, record.as_ref()));
            }
        }
    }

    latest
        .into_iter()
        .map(|(oid, (_, record))| (oid, record))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct LatestBranchRecord<'a> {
    record: &'a proximadb_records::ProximaRecord,
    wal_ts_us: i64,
    sequence_number: u64,
}

fn latest_branch_records<'a>(
    entries: &[&'a proximadb_storage_common::CanonicalWalEntry],
) -> std::collections::HashMap<&'a str, LatestBranchRecord<'a>> {
    use proximadb_storage_common::CanonicalOperation;

    let mut latest: std::collections::HashMap<&'a str, LatestBranchRecord<'a>> =
        std::collections::HashMap::new();
    for entry in entries {
        if let CanonicalOperation::RecordUpsert { record, .. } = &entry.operation {
            let wal_ts_us = (entry.timestamp_ms as i64).saturating_mul(1_000);
            let candidate = LatestBranchRecord {
                record,
                wal_ts_us,
                sequence_number: entry.sequence_number,
            };
            let replace = latest.get(record.oid.as_str()).is_none_or(|current| {
                wal_ts_us > current.wal_ts_us
                    || (wal_ts_us == current.wal_ts_us
                        && entry.sequence_number >= current.sequence_number)
            });
            if replace {
                latest.insert(record.oid.as_str(), candidate);
            }
        }
    }
    latest
}

fn refine_conflicts_with_pairwise_records(
    conflicts: &mut [MergeConflict],
    left_entries: &[&proximadb_storage_common::CanonicalWalEntry],
    right_entries: &[&proximadb_storage_common::CanonicalWalEntry],
    base_records: &std::collections::HashMap<String, &proximadb_records::ProximaRecord>,
) {
    let left_records = latest_branch_records(left_entries);
    let right_records = latest_branch_records(right_entries);

    for conflict in conflicts {
        let Some(left) = left_records.get(conflict.record_id.as_str()) else {
            continue;
        };
        let Some(right) = right_records.get(conflict.record_id.as_str()) else {
            continue;
        };
        let base = base_records.get(conflict.record_id.as_str()).copied();
        let mutation_class = classify_mutation(base, Some(left.record), Some(right.record));
        *conflict = MergeConflict::new(
            conflict.record_id.clone(),
            mutation_class,
            left.wal_ts_us,
            right.wal_ts_us,
        );
    }
}

/// Result of a write-back operation, containing the written WAL entries.
#[derive(Debug, Clone)]
pub struct WriteBackResult {
    /// The WAL entries that were written.
    pub written_entries: Vec<proximadb_storage_common::CanonicalWalEntry>,
    /// The LSN of the first written entry.
    pub first_lsn: u64,
    /// The LSN of the last written entry.
    pub last_lsn: u64,
}

/// Apply merge resolutions and write the merged records to the canonical WAL.
///
/// This is step 6 of the ADR-012 merge protocol. It takes:
/// - The original WAL entries (for fetching records to merge)
/// - The merge report (containing resolutions)
/// - The WAL path for writing
/// - The branch names and collection ID
///
/// Returns `Ok(None)` if there's nothing to write (all records were deleted).
pub async fn write_back_merge(
    wal: &[proximadb_storage_common::CanonicalWalEntry],
    report: &MergeReport,
    wal_path: &Path,
    collection_id: &str,
    branch_a: &str,
    branch_b: &str,
    tenant_id: Option<String>,
) -> Result<Option<WriteBackResult>> {
    use proximadb_storage_common::CanonicalOperation;

    let diff_range = report.merge_base_lsn.saturating_add(1)..;
    let left_entries =
        proximadb_storage_common::filter_wal_by_branch_lsn(wal, branch_a, diff_range.clone());
    let right_entries =
        proximadb_storage_common::filter_wal_by_branch_lsn(wal, branch_b, diff_range);

    // Build OID → latest record maps for each branch
    let left_records = latest_branch_records_for_write(&left_entries);
    let right_records = latest_branch_records_for_write(&right_entries);

    // Track which OIDs have been handled (conflicts or unilateral)
    let mut handled_oids: HashSet<String> = HashSet::new();

    // Tag every merged record with origin = "branch_merge:<a>:<b>" so the
    // canonical WAL identifies the write as a branch-merge resolution (per
    // ADR-012 §3 step 6 / T3.1 Slice 6 plan spec).
    let merge_origin = format!("branch_merge:{}:{}", branch_a, branch_b);
    let stamp_origin = |record: &proximadb_records::ProximaRecord| {
        let mut cloned = record.clone();
        cloned.origin = Some(merge_origin.clone());
        Box::new(cloned)
    };

    // Build the merged operations
    let mut operations = Vec::new();

    // First, handle all conflicts with their resolutions
    for resolution in &report.resolutions {
        let oid = &resolution.conflict.record_id;
        handled_oids.insert(oid.clone());

        match resolution.outcome {
            MergeOutcome::KeepLeft => {
                if let Some((record, _)) = left_records.get(oid.as_str()) {
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.to_string(),
                        record: stamp_origin(record),
                        projections: vec![],
                    });
                }
            }
            MergeOutcome::KeepRight => {
                if let Some((record, _)) = right_records.get(oid.as_str()) {
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.to_string(),
                        record: stamp_origin(record),
                        projections: vec![],
                    });
                }
            }
            MergeOutcome::BothDeleted => {
                // Write a tombstone delete operation
                operations.push(CanonicalOperation::RecordDelete {
                    collection_id: collection_id.to_string(),
                    oid: oid.clone(),
                    projections: vec![],
                });
            }
            MergeOutcome::UnionLabels => {
                // Merge label sets from both branches
                if let (Some((left_rec, _)), Some((right_rec, _))) = (
                    left_records.get(oid.as_str()),
                    right_records.get(oid.as_str()),
                ) {
                    let mut merged_rec = (*left_rec).clone();
                    let left_labels: Vec<String> = left_rec.labels.iter().cloned().collect();
                    let right_labels: Vec<String> = right_rec.labels.iter().cloned().collect();
                    merged_rec.labels = proximadb_records::LabelSet::from(merge_labels(
                        &left_labels,
                        &right_labels,
                    ));
                    merged_rec.origin = Some(merge_origin.clone());
                    operations.push(CanonicalOperation::RecordUpsert {
                        collection_id: collection_id.to_string(),
                        record: Box::new(merged_rec),
                        projections: vec![],
                    });
                }
            }
        }
    }

    // Second, handle unilateral mutations (OIDs that appear on only one side)
    for (oid, (record, _seq)) in &left_records {
        if !handled_oids.contains(*oid) {
            handled_oids.insert(oid.to_string());
            operations.push(CanonicalOperation::RecordUpsert {
                collection_id: collection_id.to_string(),
                record: stamp_origin(record),
                projections: vec![],
            });
        }
    }
    for (oid, (record, _seq)) in &right_records {
        if !handled_oids.contains(*oid) {
            handled_oids.insert(oid.to_string());
            operations.push(CanonicalOperation::RecordUpsert {
                collection_id: collection_id.to_string(),
                record: stamp_origin(record),
                projections: vec![],
            });
        }
    }

    if operations.is_empty() {
        return Ok(None);
    }

    // Write through the process's ONE appender for this file. Opening a
    // second one gave it its own sequence counter: the shared appender then
    // reissued the merge's sequence numbers, and never saw the merge entries
    // in its per-collection high-water marks (the change-feed fast path could
    // skip them).
    let appender = crate::services::FramedTableWalAppender::open_shared(wal_path)
        .await
        .context("opening WAL appender for merge write-back")?;

    let written_entries = appender
        .append_operations(operations, tenant_id)
        .await
        .context("writing merged operations to WAL")?;

    let first_lsn = written_entries
        .first()
        .map(|e| e.sequence_number)
        .unwrap_or(0);
    let last_lsn = written_entries
        .last()
        .map(|e| e.sequence_number)
        .unwrap_or(0);

    Ok(Some(WriteBackResult {
        written_entries,
        first_lsn,
        last_lsn,
    }))
}

/// Helper to get the latest record per OID from a slice of WAL entries.
/// Returns (record, sequence_number) tuples.
fn latest_branch_records_for_write<'a>(
    entries: &'a [&'a proximadb_storage_common::CanonicalWalEntry],
) -> HashMap<&'a str, (&'a proximadb_records::ProximaRecord, u64)> {
    use proximadb_storage_common::CanonicalOperation;

    let mut latest: HashMap<&'a str, (&'a proximadb_records::ProximaRecord, u64)> = HashMap::new();
    for entry in entries {
        if let CanonicalOperation::RecordUpsert { record, .. } = &entry.operation {
            let candidate = (record.as_ref(), entry.sequence_number);
            let replace = latest
                .get(record.oid.as_str())
                .is_none_or(|&(_, prev_seq)| entry.sequence_number > prev_seq);
            if replace {
                latest.insert(record.oid.as_str(), candidate);
            }
        }
    }
    latest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_delete_always_wins() {
        let conflict = MergeConflict::new("node-1", MutationClass::NodeDelete, 100, 50);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::BothDeleted);
    }

    #[test]
    fn node_upsert_lww_right_wins() {
        let conflict = MergeConflict::new("node-2", MutationClass::NodeUpsert, 100, 200);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::KeepRight);
    }

    #[test]
    fn node_upsert_lww_left_wins() {
        let conflict = MergeConflict::new("node-3", MutationClass::NodeUpsert, 300, 200);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::KeepLeft);
    }

    #[test]
    fn label_set_always_unions() {
        let conflict = MergeConflict::new("node-4", MutationClass::LabelSet, 100, 50);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::UnionLabels);
    }

    #[test]
    fn embedding_update_lww() {
        let conflict = MergeConflict::new("node-5", MutationClass::EmbeddingUpdate, 500, 400);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::KeepLeft);
    }

    #[test]
    fn edge_delete_wins_over_upsert() {
        let conflict = MergeConflict::new("edge-1", MutationClass::EdgeDelete, 50, 200);
        let resolution = GraphMergeResolution::resolve(conflict);
        assert_eq!(resolution.outcome, MergeOutcome::BothDeleted);
    }

    #[test]
    fn resolve_batch_processes_all() {
        let conflicts = vec![
            MergeConflict::new("a", MutationClass::NodeUpsert, 1, 2),
            MergeConflict::new("b", MutationClass::NodeDelete, 1, 2),
            MergeConflict::new("c", MutationClass::LabelSet, 1, 2),
        ];
        let resolutions = resolve_batch(conflicts);
        assert_eq!(resolutions.len(), 3);
        assert_eq!(resolutions[0].outcome, MergeOutcome::KeepRight);
        assert_eq!(resolutions[1].outcome, MergeOutcome::BothDeleted);
        assert_eq!(resolutions[2].outcome, MergeOutcome::UnionLabels);
    }

    #[test]
    fn merge_policy_table_matches_adr_012() {
        assert_eq!(
            MergePolicy::for_class(MutationClass::NodeUpsert),
            MergePolicy::LastWriteWins
        );
        assert_eq!(
            MergePolicy::for_class(MutationClass::NodeDelete),
            MergePolicy::DeleteWins
        );
        assert_eq!(
            MergePolicy::for_class(MutationClass::EdgeDelete),
            MergePolicy::DeleteWins
        );
        assert_eq!(
            MergePolicy::for_class(MutationClass::LabelSet),
            MergePolicy::AddWinsSetUnion
        );
        assert_eq!(
            MergePolicy::for_class(MutationClass::PropsKey),
            MergePolicy::LastWriteWinsPerKey
        );
    }

    // ── T3.1 Slice 1 — walk_diff tests ─────────────────────────────────────

    fn ev(oid: &str, class: MutationClass, ts: i64) -> MutationEvent {
        MutationEvent::new(oid, class, ts)
    }

    #[test]
    fn walk_diff_empty_inputs_yields_no_conflicts() {
        let conflicts = walk_diff(&[], &[]);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn walk_diff_disjoint_oids_yields_no_conflicts() {
        let left = vec![
            ev("a", MutationClass::NodeUpsert, 100),
            ev("b", MutationClass::NodeUpsert, 100),
        ];
        let right = vec![
            ev("c", MutationClass::NodeUpsert, 100),
            ev("d", MutationClass::NodeUpsert, 100),
        ];
        let conflicts = walk_diff(&left, &right);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn walk_diff_overlapping_oid_yields_conflict_with_correct_timestamps() {
        let left = vec![ev("shared", MutationClass::NodeUpsert, 100)];
        let right = vec![ev("shared", MutationClass::NodeUpsert, 200)];
        let conflicts = walk_diff(&left, &right);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].record_id, "shared");
        assert_eq!(conflicts[0].left_wal_ts_us, 100);
        assert_eq!(conflicts[0].right_wal_ts_us, 200);
        // Policy comes from the existing primitive — verify LWW for NodeUpsert.
        assert_eq!(conflicts[0].policy, MergePolicy::LastWriteWins);
    }

    #[test]
    fn walk_diff_picks_latest_event_per_oid_per_side() {
        let left = vec![
            ev("x", MutationClass::NodeUpsert, 100),
            ev("x", MutationClass::NodeUpsert, 250), // latest on left
            ev("x", MutationClass::NodeUpsert, 200),
        ];
        let right = vec![
            ev("x", MutationClass::NodeUpsert, 50),
            ev("x", MutationClass::NodeUpsert, 175), // latest on right
        ];
        let conflicts = walk_diff(&left, &right);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].left_wal_ts_us, 250);
        assert_eq!(conflicts[0].right_wal_ts_us, 175);
    }

    #[test]
    fn walk_diff_delete_on_one_side_resolves_to_delete_wins() {
        let left = vec![ev("doomed", MutationClass::NodeDelete, 100)];
        let right = vec![ev("doomed", MutationClass::NodeUpsert, 500)];
        let conflicts = walk_diff(&left, &right);
        assert_eq!(conflicts.len(), 1);
        // Class is taken from left (NodeDelete); existing primitive routes
        // through DeleteWins policy regardless of timestamp ordering.
        assert_eq!(conflicts[0].mutation_class, MutationClass::NodeDelete);
        let resolution = GraphMergeResolution::resolve(conflicts[0].clone());
        assert_eq!(resolution.outcome, MergeOutcome::BothDeleted);
    }

    #[test]
    fn walk_diff_results_are_deterministically_ordered_by_oid() {
        let left = vec![
            ev("z", MutationClass::NodeUpsert, 100),
            ev("a", MutationClass::NodeUpsert, 100),
            ev("m", MutationClass::NodeUpsert, 100),
        ];
        let right = vec![
            ev("z", MutationClass::NodeUpsert, 200),
            ev("a", MutationClass::NodeUpsert, 200),
            ev("m", MutationClass::NodeUpsert, 200),
        ];
        let conflicts = walk_diff(&left, &right);
        let oids: Vec<&str> = conflicts.iter().map(|c| c.record_id.as_str()).collect();
        assert_eq!(oids, vec!["a", "m", "z"]);
    }

    // ── T3.1 Slice 1 — classify_mutation tests ─────────────────────────────

    fn rec_default() -> proximadb_records::ProximaRecord {
        proximadb_records::ProximaRecord::default()
    }

    fn rec_tombstone() -> proximadb_records::ProximaRecord {
        let mut r = rec_default();
        r.valid_to_ns = Some(0);
        r.embeddings = Vec::new();
        r.origin = Some("delete".to_string());
        r
    }

    fn rec_edge() -> proximadb_records::ProximaRecord {
        let mut r = rec_default();
        r.edge = Some(proximadb_records::EdgeShape {
            source_id: "src".to_string(),
            target_id: "dst".to_string(),
            edge_type: "rel".to_string(),
            weight: None,
        });
        r
    }

    #[test]
    fn classify_tombstone_on_branch_a_returns_node_delete() {
        let base = rec_default();
        let a = rec_tombstone();
        let b = rec_default();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::NodeDelete
        );
    }

    #[test]
    fn classify_tombstone_on_edge_returns_edge_delete() {
        let base = rec_edge();
        let mut a = rec_tombstone();
        a.edge = Some(proximadb_records::EdgeShape {
            source_id: "src".to_string(),
            target_id: "dst".to_string(),
            edge_type: "rel".to_string(),
            weight: None,
        });
        let b = rec_edge();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::EdgeDelete
        );
    }

    #[test]
    fn classify_default_returns_node_upsert() {
        let base = rec_default();
        let a = rec_default();
        let b = rec_default();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::NodeUpsert
        );
    }

    #[test]
    fn classify_edge_with_no_field_changes_returns_edge_upsert() {
        let base = rec_edge();
        let a = rec_edge();
        let b = rec_edge();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::EdgeUpsert
        );
    }

    #[test]
    fn classify_label_only_change_returns_label_set() {
        let base = rec_default();
        let mut a = rec_default();
        a.labels = proximadb_records::LabelSet::from(vec!["new".to_string()]);
        let b = rec_default();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::LabelSet
        );
    }

    #[test]
    fn classify_props_change_returns_props_key() {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::ProximaTreeNode;
        let base = rec_default();
        let mut a = rec_default();
        a.props.insert(
            "k".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("v".to_string())),
        );
        let b = rec_default();
        assert_eq!(
            classify_mutation(Some(&base), Some(&a), Some(&b)),
            MutationClass::PropsKey
        );
    }

    // ── T3.1 Slice 5 — merge_branches orchestrator tests ───────────────────

    fn upsert_canonical_entry(
        seq: u64,
        oid: &str,
        branch: Option<&str>,
        ts_ms: u64,
    ) -> proximadb_storage_common::CanonicalWalEntry {
        let mut record = proximadb_records::ProximaRecord::default();
        record.oid = oid.to_string();
        record.branch_id = branch.map(String::from);
        proximadb_storage_common::CanonicalWalEntry {
            sequence_number: seq,
            timestamp_ms: ts_ms,
            operation: proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id: "col".to_string(),
                record: Box::new(record),
                projections: vec![],
            },
            tenant_id: None,
        }
    }

    fn upsert_record_canonical_entry(
        seq: u64,
        ts_ms: u64,
        record: proximadb_records::ProximaRecord,
    ) -> proximadb_storage_common::CanonicalWalEntry {
        proximadb_storage_common::CanonicalWalEntry {
            sequence_number: seq,
            timestamp_ms: ts_ms,
            operation: proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id: "col".to_string(),
                record: Box::new(record),
                projections: vec![],
            },
            tenant_id: None,
        }
    }

    fn branch_record(oid: &str, branch: Option<&str>) -> proximadb_records::ProximaRecord {
        let mut record = proximadb_records::ProximaRecord::default();
        record.oid = oid.to_string();
        record.branch_id = branch.map(String::from);
        record
    }

    fn fp32_embedding(values: &[f32]) -> proximadb_records::EmbeddingCell {
        proximadb_records::EmbeddingCell::new_fp32(
            "test-model",
            "dense_vector",
            values.len() as u32,
            values.to_vec(),
        )
    }

    fn tombstone_canonical_entry(
        seq: u64,
        oid: &str,
        branch: Option<&str>,
        ts_ms: u64,
    ) -> proximadb_storage_common::CanonicalWalEntry {
        let mut record = proximadb_records::ProximaRecord::default();
        record.oid = oid.to_string();
        record.branch_id = branch.map(String::from);
        record.valid_to_ns = Some(0);
        record.embeddings = Vec::new();
        record.origin = Some("delete".to_string());
        proximadb_storage_common::CanonicalWalEntry {
            sequence_number: seq,
            timestamp_ms: ts_ms,
            operation: proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id: "col".to_string(),
                record: Box::new(record),
                projections: vec![],
            },
            tenant_id: None,
        }
    }

    #[test]
    fn merge_branches_returns_none_for_empty_wal() {
        assert!(merge_branches(&[], "a", "b").is_none());
    }

    #[test]
    fn merge_branches_returns_none_when_branch_a_has_no_entries() {
        let wal = vec![upsert_canonical_entry(5, "x", Some("b"), 100)];
        assert!(merge_branches(&wal, "a", "b").is_none());
    }

    #[test]
    fn merge_branches_returns_report_with_correct_merge_base() {
        let wal = vec![
            upsert_canonical_entry(5, "x", Some("a"), 100),
            upsert_canonical_entry(7, "y", Some("b"), 200),
        ];
        let report = merge_branches(&wal, "a", "b").expect("both branches present");
        // min(5, 7) - 1 = 4
        assert_eq!(report.merge_base_lsn, 4);
        assert_eq!(report.left_events.len(), 1);
        assert_eq!(report.right_events.len(), 1);
        assert_eq!(report.left_events[0].record_oid, "x");
        assert_eq!(report.right_events[0].record_oid, "y");
    }

    #[test]
    fn merge_branches_finds_no_conflicts_for_disjoint_oids() {
        let wal = vec![
            upsert_canonical_entry(5, "x", Some("a"), 100),
            upsert_canonical_entry(7, "y", Some("b"), 200),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert!(report.conflicts.is_empty());
        assert!(report.resolutions.is_empty());
    }

    #[test]
    fn merge_branches_finds_conflict_for_overlapping_oid() {
        // Both branches touch OID "shared" — should produce a conflict.
        let wal = vec![
            upsert_canonical_entry(5, "shared", Some("a"), 100),
            upsert_canonical_entry(7, "shared", Some("b"), 200),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(report.conflicts[0].record_id, "shared");
        assert_eq!(
            report.conflicts[0].mutation_class,
            MutationClass::NodeUpsert
        );
        // NodeUpsert → LastWriteWins → right (ts=200_000) wins over left (ts=100_000).
        assert_eq!(report.resolutions.len(), 1);
        assert_eq!(report.resolutions[0].outcome, MergeOutcome::KeepRight);
    }

    #[test]
    fn merge_branches_classifies_edge_record_as_edge_upsert() {
        let mut edge_record = proximadb_records::ProximaRecord::default();
        edge_record.oid = "edge-1".to_string();
        edge_record.branch_id = Some("a".to_string());
        edge_record.edge = Some(proximadb_records::EdgeShape {
            source_id: "src".to_string(),
            target_id: "dst".to_string(),
            edge_type: "rel".to_string(),
            weight: None,
        });
        let entry = proximadb_storage_common::CanonicalWalEntry {
            sequence_number: 5,
            timestamp_ms: 100,
            operation: proximadb_storage_common::CanonicalOperation::RecordUpsert {
                collection_id: "col".to_string(),
                record: Box::new(edge_record),
                projections: vec![],
            },
            tenant_id: None,
        };
        let wal = vec![entry, upsert_canonical_entry(7, "node-1", Some("b"), 200)];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(
            report.left_events[0].mutation_class,
            MutationClass::EdgeUpsert
        );
    }

    #[test]
    fn merge_branches_classifies_tombstone_as_node_delete() {
        let wal = vec![
            tombstone_canonical_entry(5, "doomed", Some("a"), 100),
            upsert_canonical_entry(7, "fresh", Some("b"), 200),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(
            report.left_events[0].mutation_class,
            MutationClass::NodeDelete
        );
    }

    #[test]
    fn merge_branches_pairwise_classifies_label_conflict() {
        let mut base = branch_record("shared", None);
        base.labels = proximadb_records::LabelSet::from(vec!["base".to_string()]);
        let mut left = branch_record("shared", Some("a"));
        left.labels =
            proximadb_records::LabelSet::from(vec!["base".to_string(), "left".to_string()]);
        let mut right = branch_record("shared", Some("b"));
        right.labels =
            proximadb_records::LabelSet::from(vec!["base".to_string(), "right".to_string()]);

        let wal = vec![
            upsert_record_canonical_entry(3, 50, base),
            upsert_record_canonical_entry(5, 100, left),
            upsert_record_canonical_entry(7, 200, right),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(
            report.left_events[0].mutation_class,
            MutationClass::LabelSet
        );
        assert_eq!(
            report.right_events[0].mutation_class,
            MutationClass::LabelSet
        );
        assert_eq!(report.conflicts[0].mutation_class, MutationClass::LabelSet);
        assert_eq!(report.resolutions[0].outcome, MergeOutcome::UnionLabels);
    }

    #[test]
    fn merge_branches_pairwise_classifies_props_conflict() {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::ProximaTreeNode;

        let base = branch_record("shared", None);
        let mut left = branch_record("shared", Some("a"));
        left.props.insert(
            "name".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("left".to_string())),
        );
        let mut right = branch_record("shared", Some("b"));
        right.props.insert(
            "name".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("right".to_string())),
        );

        let wal = vec![
            upsert_record_canonical_entry(3, 50, base),
            upsert_record_canonical_entry(5, 100, left),
            upsert_record_canonical_entry(7, 200, right),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(
            report.left_events[0].mutation_class,
            MutationClass::PropsKey
        );
        assert_eq!(
            report.right_events[0].mutation_class,
            MutationClass::PropsKey
        );
        assert_eq!(report.conflicts[0].mutation_class, MutationClass::PropsKey);
        assert_eq!(report.resolutions[0].outcome, MergeOutcome::KeepRight);
    }

    #[test]
    fn merge_branches_pairwise_classifies_embedding_conflict() {
        let mut base = branch_record("shared", None);
        base.embeddings = vec![fp32_embedding(&[0.0, 0.0])];
        let mut left = branch_record("shared", Some("a"));
        left.embeddings = vec![fp32_embedding(&[1.0, 0.0])];
        let mut right = branch_record("shared", Some("b"));
        right.embeddings = vec![fp32_embedding(&[0.0, 1.0])];

        let wal = vec![
            upsert_record_canonical_entry(3, 50, base),
            upsert_record_canonical_entry(5, 300, left),
            upsert_record_canonical_entry(7, 200, right),
        ];
        let report = merge_branches(&wal, "a", "b").unwrap();
        assert_eq!(
            report.left_events[0].mutation_class,
            MutationClass::EmbeddingUpdate
        );
        assert_eq!(
            report.right_events[0].mutation_class,
            MutationClass::EmbeddingUpdate
        );
        assert_eq!(
            report.conflicts[0].mutation_class,
            MutationClass::EmbeddingUpdate
        );
        assert_eq!(report.resolutions[0].outcome, MergeOutcome::KeepLeft);
    }

    #[test]
    fn entry_to_mutation_event_returns_none_for_non_upsert_entries() {
        // Checkpoint, CdcBarrier, and RecordDelete entries don't carry a
        // record we can classify. They yield None from the helper and are
        // implicitly skipped by `filter_map` in merge_branches.
        let checkpoint = proximadb_storage_common::CanonicalWalEntry {
            sequence_number: 3,
            timestamp_ms: 50,
            operation: proximadb_storage_common::CanonicalOperation::Checkpoint(
                proximadb_storage_common::SnapshotManifest {
                    sequence_number: 3,
                    timestamp_ms: 50,
                    collection_ids: vec!["col".into()],
                    projection_freshness: vec![],
                },
            ),
            tenant_id: None,
        };
        let delete = proximadb_storage_common::CanonicalWalEntry {
            sequence_number: 4,
            timestamp_ms: 75,
            operation: proximadb_storage_common::CanonicalOperation::RecordDelete {
                collection_id: "col".into(),
                oid: "victim".into(),
                projections: vec![],
            },
            tenant_id: None,
        };
        assert!(entry_to_mutation_event(&checkpoint).is_none());
        assert!(entry_to_mutation_event(&delete).is_none());
    }

    // ── T3.1 Slice 6 — write_back_merge tests ─────────────────────────────────────

    #[tokio::test]
    async fn write_back_merge_handles_keep_left_resolution() {
        use proximadb_records::ProximaTreeNode;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Create WAL entries for a conflict
        let mut left_rec = branch_record("shared", Some("a"));
        left_rec.props.insert(
            "side".to_string(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "left".to_string(),
            )),
        );
        let mut right_rec = branch_record("shared", Some("b"));
        right_rec.props.insert(
            "side".to_string(),
            ProximaTreeNode::Value(proximadb_data_model::ProximaValue::String(
                "right".to_string(),
            )),
        );

        let wal = vec![
            upsert_canonical_entry(3, "base", None, 50),
            upsert_record_canonical_entry(5, 100, left_rec),
            upsert_record_canonical_entry(7, 200, right_rec),
        ];

        let report = merge_branches(&wal, "a", "b").unwrap();

        // Write back should keep the left (earlier timestamp) record due to LWW
        let result = write_back_merge(&wal, &report, &wal_path, "col", "a", "b", None)
            .await
            .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.written_entries.len(), 1);
        assert_eq!(result.first_lsn, 1);
        assert_eq!(result.last_lsn, 1);

        // Verify the written operation contains the left record
        match &result.written_entries[0].operation {
            proximadb_storage_common::CanonicalOperation::RecordUpsert { record, .. } => {
                assert_eq!(record.oid, "shared");
                assert!(record.props.contains_key("side"));
                assert_eq!(record.origin.as_deref(), Some("branch_merge:a:b"));
            }
            _ => panic!("Expected RecordUpsert"),
        }
    }

    #[tokio::test]
    async fn write_back_merge_stamps_origin_on_unilateral_records() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Disjoint OIDs → unilateral path on both sides
        let wal = vec![
            upsert_canonical_entry(3, "left-only", Some("a"), 100),
            upsert_canonical_entry(5, "right-only", Some("b"), 200),
        ];

        let report = merge_branches(&wal, "a", "b").unwrap();
        let result = write_back_merge(&wal, &report, &wal_path, "col", "a", "b", None)
            .await
            .unwrap()
            .expect("write_back_merge should return Some");

        assert_eq!(result.written_entries.len(), 2);
        for entry in &result.written_entries {
            match &entry.operation {
                proximadb_storage_common::CanonicalOperation::RecordUpsert { record, .. } => {
                    assert_eq!(
                        record.origin.as_deref(),
                        Some("branch_merge:a:b"),
                        "every merged record must carry the branch-merge origin"
                    );
                }
                other => panic!("expected RecordUpsert, got {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn write_back_merge_handles_both_deleted_resolution() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Create WAL entries where one side is a tombstone
        let mut left_rec = branch_record("shared", Some("a"));
        left_rec.valid_to_ns = Some(0);
        left_rec.embeddings = Vec::new();
        left_rec.origin = Some("delete".to_string());
        let right_rec = branch_record("shared", Some("b"));

        let wal = vec![
            upsert_record_canonical_entry(3, 50, rec_default()),
            upsert_record_canonical_entry(5, 100, left_rec),
            upsert_record_canonical_entry(7, 200, right_rec),
        ];

        let report = merge_branches(&wal, "a", "b").unwrap();

        let result = write_back_merge(&wal, &report, &wal_path, "col", "a", "b", None)
            .await
            .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.written_entries.len(), 1);

        // Should write a delete operation
        match &result.written_entries[0].operation {
            proximadb_storage_common::CanonicalOperation::RecordDelete { oid, .. } => {
                assert_eq!(oid, "shared");
            }
            _ => panic!("Expected RecordDelete"),
        }
    }

    #[tokio::test]
    async fn write_back_merge_handles_union_labels_resolution() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Create WAL entries for label conflict
        let base = branch_record("shared", None);
        let mut left = branch_record("shared", Some("a"));
        left.labels =
            proximadb_records::LabelSet::from(vec!["base".to_string(), "left".to_string()]);
        let mut right = branch_record("shared", Some("b"));
        right.labels =
            proximadb_records::LabelSet::from(vec!["base".to_string(), "right".to_string()]);

        let wal = vec![
            upsert_record_canonical_entry(3, 50, base),
            upsert_record_canonical_entry(5, 100, left),
            upsert_record_canonical_entry(7, 200, right),
        ];

        let report = merge_branches(&wal, "a", "b").unwrap();

        let result = write_back_merge(&wal, &report, &wal_path, "col", "a", "b", None)
            .await
            .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.written_entries.len(), 1);

        // Verify merged labels contain all unique labels
        match &result.written_entries[0].operation {
            proximadb_storage_common::CanonicalOperation::RecordUpsert { record, .. } => {
                assert_eq!(record.oid, "shared");
                let labels: Vec<&str> = record.labels.iter().map(|s| s.as_str()).collect();
                assert_eq!(labels.len(), 3);
                assert!(labels.contains(&"base"));
                assert!(labels.contains(&"left"));
                assert!(labels.contains(&"right"));
            }
            _ => panic!("Expected RecordUpsert"),
        }
    }

    #[tokio::test]
    async fn write_back_merge_handles_unilateral_mutations() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Create WAL entries with no conflicts (disjoint OIDs)
        let wal = vec![
            upsert_canonical_entry(3, "left-only", Some("a"), 100),
            upsert_canonical_entry(5, "right-only", Some("b"), 200),
        ];

        let report = merge_branches(&wal, "a", "b").unwrap();

        let result = write_back_merge(&wal, &report, &wal_path, "col", "a", "b", None)
            .await
            .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        // Both unilateral records should be written
        assert_eq!(result.written_entries.len(), 2);
    }

    #[tokio::test]
    async fn write_back_merge_returns_none_for_empty_operations() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let wal_path = temp_dir.path().join("test.wal");

        // Empty WAL
        let result = write_back_merge(
            &[],
            &MergeReport {
                merge_base_lsn: 0,
                left_events: vec![],
                right_events: vec![],
                conflicts: vec![],
                resolutions: vec![],
            },
            &wal_path,
            "col",
            "a",
            "b",
            None,
        )
        .await
        .unwrap();

        assert!(result.is_none());
    }

    // ── TD-SEAM-12 hazard 2 — merge-base images ────────────────────────────

    /// Main-line history that the base classification depends on (props,
    /// labels and embeddings change on main before the branches diverge), a
    /// main write after the divergence, and both branches touching every oid.
    fn diverging_history() -> Vec<proximadb_storage_common::CanonicalWalEntry> {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::ProximaTreeNode;
        let named = |oid: &str, branch: Option<&str>, name: &str| {
            let mut record = branch_record(oid, branch);
            record.props.insert(
                "name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(name.to_string())),
            );
            record
        };
        let labelled = |oid: &str, branch: Option<&str>, labels: &[&str]| {
            let mut record = branch_record(oid, branch);
            record.labels = proximadb_records::LabelSet::from(
                labels.iter().map(|l| l.to_string()).collect::<Vec<_>>(),
            );
            record
        };
        let mut embedded = named("s1", None, "v1");
        embedded.embeddings = vec![fp32_embedding(&[0.0, 1.0])];
        let mut s2_embedded = labelled("s2", Some("b"), &["x", "y"]);
        s2_embedded.embeddings = vec![fp32_embedding(&[1.0, 1.0])];
        vec![
            upsert_record_canonical_entry(1, 10, named("s1", None, "v0")),
            upsert_record_canonical_entry(2, 20, labelled("s2", None, &["x"])),
            upsert_record_canonical_entry(3, 30, embedded),
            upsert_record_canonical_entry(4, 40, named("s3", None, "v0")),
            upsert_record_canonical_entry(5, 50, labelled("s2", None, &["x", "y"])),
            upsert_record_canonical_entry(6, 60, named("s1", Some("a"), "left")),
            upsert_record_canonical_entry(7, 70, named("s3", None, "main-after")),
            upsert_record_canonical_entry(8, 80, named("s1", Some("b"), "right")),
            upsert_record_canonical_entry(9, 90, labelled("s2", Some("a"), &["x", "y", "z"])),
            upsert_record_canonical_entry(10, 100, s2_embedded),
            upsert_record_canonical_entry(11, 110, named("s3", Some("b"), "v0")),
            upsert_record_canonical_entry(12, 120, named("s3", Some("a"), "v2")),
        ]
    }

    fn above(
        wal: &[proximadb_storage_common::CanonicalWalEntry],
        through: u64,
    ) -> Vec<proximadb_storage_common::CanonicalWalEntry> {
        wal.iter()
            .filter(|e| e.sequence_number > through)
            .cloned()
            .collect()
    }

    #[test]
    fn base_image_plus_suffix_equals_full_history() {
        let wal = diverging_history();
        let full = merge_branches(&wal, "a", "b").expect("both branches present");
        assert_eq!(full.merge_base_lsn, 5);
        assert!(!full.conflicts.is_empty());

        for through in 0..=full.merge_base_lsn {
            let mut image = MergeBaseImage::default();
            image.advance(&wal, through);
            // Truncate everything the image covers.
            let retained = above(&wal, through);
            let from_image = merge_branches_with_base(&retained, &image, "a", "b")
                .expect("image at or below the merge base")
                .expect("both branches retained");
            assert_eq!(
                format!("{from_image:?}"),
                format!("{full:?}"),
                "image through {through} + retained suffix must equal the full history"
            );
            // Entries the image already covers are ignored, not double-counted.
            let untruncated = merge_branches_with_base(&wal, &image, "a", "b")
                .unwrap()
                .unwrap();
            assert_eq!(format!("{untruncated:?}"), format!("{full:?}"));
        }

        // Without the image, the truncated history merges against the wrong base.
        let truncated_only = merge_branches(&above(&wal, 5), "a", "b").unwrap();
        assert_ne!(format!("{truncated_only:?}"), format!("{full:?}"));
    }

    #[test]
    fn base_image_advances_incrementally() {
        let wal = diverging_history();
        let mut stepped = MergeBaseImage::default();
        stepped.advance(&wal, 2);
        // Only the history the image has not folded is needed to advance it.
        stepped.advance(&above(&wal, 2), 5);
        stepped.advance(&wal, 3); // never backwards
        let mut direct = MergeBaseImage::default();
        direct.advance(&wal, 5);
        assert_eq!(stepped, direct);
        assert_eq!(
            direct.records.len(),
            3,
            "s1, s2, s3 — branch writes excluded"
        );
    }

    #[test]
    fn base_image_past_the_merge_base_is_refused() {
        let wal = diverging_history();
        let mut image = MergeBaseImage::default();
        image.advance(&wal, 6);
        assert_eq!(
            merge_branches_with_base(&wal, &image, "a", "b").unwrap_err(),
            MergeBaseError::ImageAboveMergeBase {
                through: 6,
                merge_base_lsn: 5
            }
        );
        assert!(
            merge_branches_with_base(&wal, &image, "a", "missing")
                .unwrap()
                .is_none()
        );
    }
}
