// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! OLAP read-time delta merge (ADR-025, relational cold path).
//!
//! `ALTER TABLE … MATERIALIZE` writes an atomic, already-dead-filtered Parquet
//! snapshot of a table at a point in time (`snapshot_lsn`). Any `DELETE` /
//! `UPDATE` / `INSERT` that lands **after** that snapshot stays in the
//! authoritative record store (memtable + canonical WAL + PAX) and is *not*
//! reflected in the cold Parquet base. Without reconciliation an OLAP `SELECT`
//! routed to DataFusion over the Parquet base returns stale rows — the gap this
//! module closes.
//!
//! The reconciliation is a **read-time merge** keyed by the canonical `oid`:
//!
//! * `suppress` — the set of oids that changed since `snapshot_lsn`, taken from
//!   the canonical-WAL change-feed (`TableRecordStore::read_changes_since`).
//!   Deletes, updates, and post-snapshot inserts all land here.
//! * `appends` — the **current live** row for each changed oid, read back from
//!   the record store (already dead-record-filtered). A deleted/expired oid has
//!   no live row, so it simply does not appear in `appends`.
//!
//! The merged result is every base row whose oid is **not** suppressed and which
//! is not itself TTL-dead, followed by the appends. This is the in-memory
//! deletion vector of ADR-025 done WAL-first: the suppress-set is, by
//! construction, a rebuildable projection of the canonical WAL (ADR-020 — never a
//! parallel durable authority), and re-`MATERIALIZE` plays the role of the N3
//! compaction that physically drops the dead rows.
//!
//! This module is the pure, I/O-free core (so it unit-tests without the
//! `datafusion-integration` feature). The Arrow/Parquet wiring that feeds it the
//! base rows, suppress-set, and append rows lives in
//! [`crate::query::execution::datafusion_engine`].

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use proximadb_catalog::CatalogTableSchema;
use proximadb_records::{ProximaRecord, is_record_dead};

/// One row of the cold Parquet base snapshot, tagged with the identity and
/// liveness metadata the merge needs. `payload` is the opaque per-row value the
/// caller carries through unchanged (e.g. an Arrow row index, or — in tests — a
/// scalar). `valid_to_ns` is the row's MVCC upper bound when known (PR4+ sidecar
/// lineage); `None` means "unknown", which the canonical predicate treats as
/// live (passive-TTL is then eventual-until-rematerialize — a documented
/// first-cut limitation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseRow<R> {
    /// Canonical record identity. For the relational path this is the
    /// PK-derived oid (recomputed from the base row's primary-key column via the
    /// same canonicalization the writer used), so it matches the change-feed key.
    pub oid: String,
    /// MVCC upper bound, if the base carries it. `None` = unknown (treated live).
    pub valid_to_ns: Option<i64>,
    /// Opaque row payload carried through the merge unchanged.
    pub payload: R,
}

impl<R> BaseRow<R> {
    /// Construct a base row.
    pub fn new(oid: impl Into<String>, valid_to_ns: Option<i64>, payload: R) -> Self {
        Self {
            oid: oid.into(),
            valid_to_ns,
            payload,
        }
    }
}

/// Merge a cold OLAP base snapshot with the authoritative post-snapshot delta.
///
/// * `base` — rows read from the materialized Parquet snapshot, each tagged with
///   its canonical `oid` and (optionally) `valid_to_ns`.
/// * `suppress` — oids that changed since the snapshot (deletes + updates + new
///   inserts); their base copies are stale and must be dropped.
/// * `appends` — the current live row for each changed oid (already
///   dead-filtered by the record store). Deleted/expired oids contribute none.
/// * `now_ns` — wall clock for the canonical dead-record predicate.
///
/// A base row survives iff its oid is **not** suppressed and it is not TTL-dead.
/// Output order is the surviving base rows (in their original order) followed by
/// the appends — deterministic, which keeps merge-on/merge-off result sets
/// comparable in tests.
///
/// This resolves every cross-boundary case with one rule: base-row delete
/// (suppressed, no append → gone); base-row update incl. PK change (old oid
/// suppressed, new oid appended); post-snapshot insert (suppressed since it is a
/// new upsert, appended as live); insert-then-delete after snapshot (suppressed,
/// no live append → gone, the OLAP analog of invariant 16d); re-insert of a
/// deleted PK (base suppressed, current appended).
pub fn merge_base_with_delta<R>(
    base: Vec<BaseRow<R>>,
    suppress: &HashSet<String>,
    appends: Vec<R>,
    now_ns: i64,
) -> Vec<R> {
    let mut out = Vec::with_capacity(base.len() + appends.len());
    for row in base {
        if suppress.contains(&row.oid) {
            continue;
        }
        if is_record_dead(row.valid_to_ns, now_ns) {
            continue;
        }
        out.push(row.payload);
    }
    out.extend(appends);
    out
}

/// Authoritative post-snapshot delta source for the OLAP read-merge.
///
/// Implemented by `DmlService` over the canonical WAL + record store. Kept as a
/// narrow, Arrow-free trait so the query-execution layer carries it on
/// [`QueryExecutionContext`](super::engine::QueryExecutionContext) without
/// depending on the services layer or on `datafusion-integration`.
#[async_trait]
pub trait OlapDeltaSource: Send + Sync {
    /// Canonical oids that changed (upsert OR delete) for `table` strictly after
    /// `snapshot_lsn`, from the canonical-WAL change-feed. Order is irrelevant;
    /// the merge dedups into a set.
    async fn changed_oids_since(
        &self,
        table: &str,
        snapshot_lsn: u64,
        tenant: Option<&str>,
    ) -> anyhow::Result<Vec<String>>;

    /// The table's catalog schema plus the **current live** records for `oids`,
    /// each built with every column materialized in `props` (the exact shape the
    /// warehouse materializer writes), so the caller can feed them straight to
    /// `proxima_records_to_record_batch`. Deleted/expired/absent oids contribute
    /// no record (the record store applies the canonical dead-record predicate).
    async fn current_records(
        &self,
        table: &str,
        oids: &[String],
        tenant: Option<&str>,
    ) -> anyhow::Result<(CatalogTableSchema, Vec<ProximaRecord>)>;
}

/// Per-table OLAP delta-merge parameters resolved from the catalog at query time.
#[derive(Clone, Debug)]
pub struct OlapDeltaTable {
    /// WAL high-water LSN the cold Parquet base was snapshotted at (recorded in
    /// the storage layout `properties` by `MATERIALIZE`).
    pub snapshot_lsn: u64,
    /// Primary-key column whose value canonicalizes to the record `oid`; used to
    /// recompute each base row's oid for suppress-set membership. Tables without
    /// a single-column PK are not eligible (keyless heaps fall back to the bare
    /// Parquet read — a documented first-cut limitation).
    pub pk_column: String,
    /// The Iceberg metadata version the same `parquet-snapshot` layout records
    /// for its publication. Only our publishers (MATERIALIZE's flip,
    /// `advance_snapshot_lsn`, the live-delete flip) write it, so a base opened
    /// at exactly this version IS our own publication — complete apart from
    /// the delta at `snapshot_lsn`, its deletion vectors ordered against the
    /// WAL. `None` when the layout records no version: never eligible.
    pub recorded_metadata_version: Option<u64>,
}

/// Per-query wiring for the OLAP read-merge: the authoritative delta source plus
/// the per-table parameters for every parquet-backed table that opted in. Absent
/// (`QueryExecutionContext::olap_delta == None`) ⇒ legacy bare-Parquet reads.
#[derive(Clone)]
pub struct OlapDeltaConfig {
    /// Authoritative post-snapshot delta source (the `DmlService`).
    pub source: Arc<dyn OlapDeltaSource>,
    /// Eligible tables keyed by normalized table key (see
    /// [`normalize_table_key`](super::engine::normalize_table_key)).
    pub tables: HashMap<String, OlapDeltaTable>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suppress(oids: &[&str]) -> HashSet<String> {
        oids.iter().map(|s| (*s).to_string()).collect()
    }

    /// A1: base {1,2,3} merged with delta {delete 2, update 3→3', insert 4}.
    /// suppress = {2,3,4} (every changed oid); appends = current live = {3',4}.
    /// Expect {1, 3', 4} — locks delete/update/insert and ordering.
    #[test]
    fn merge_parquet_base_with_delta_pure() {
        let base = vec![
            BaseRow::new("1", None, "1"),
            BaseRow::new("2", None, "2"),
            BaseRow::new("3", None, "3"),
        ];
        let merged =
            merge_base_with_delta(base, &suppress(&["2", "3", "4"]), vec!["3'", "4"], 1_000);
        assert_eq!(merged, vec!["1", "3'", "4"]);
    }

    /// A2 core: COUNT is the merged cardinality, never the base footer count.
    /// base has 3 rows; after delete 2 + insert 4 the live count is 3, not 4.
    #[test]
    fn merged_count_excludes_deleted_includes_inserted() {
        let base = vec![
            BaseRow::new("1", None, 10),
            BaseRow::new("2", None, 20),
            BaseRow::new("3", None, 30),
        ];
        let merged = merge_base_with_delta(base, &suppress(&["2", "4"]), vec![40], 1_000);
        assert_eq!(merged.len(), 3); // {1, 3, 4}
        assert_eq!(merged, vec![10, 30, 40]);
    }

    /// UPDATE that changes the primary key: old oid suppressed, new oid appended.
    #[test]
    fn update_changing_pk_suppresses_old_appends_new() {
        let base = vec![BaseRow::new("old", None, "old-row")];
        // PK change emits delete(old) + insert(new) in the WAL → both oids change.
        let merged =
            merge_base_with_delta(base, &suppress(&["old", "new"]), vec!["new-row"], 1_000);
        assert_eq!(merged, vec!["new-row"]);
    }

    /// Insert-then-delete after the snapshot: oid is suppressed (it changed) but
    /// has no live append → it must be absent. OLAP analog of invariant 16d.
    #[test]
    fn insert_then_delete_after_materialize_absent() {
        let base = vec![BaseRow::new("a", None, "a")];
        // "b" was inserted then deleted post-snapshot: suppressed, no live append.
        let merged = merge_base_with_delta(base, &suppress(&["b"]), vec![], 1_000);
        assert_eq!(merged, vec!["a"]);
    }

    /// Passive TTL on a base row that carries its `valid_to_ns` (PR4+ lineage):
    /// expired rows drop even with no WAL op, via the canonical predicate.
    #[test]
    fn ttl_expired_base_row_dropped_when_valid_to_ns_known() {
        let base = vec![
            BaseRow::new("live", Some(0), "x"),      // tombstone marker → dead
            BaseRow::new("expired", Some(500), "y"), // valid_to < now → dead
            BaseRow::new("ok", Some(5_000), "z"),    // valid_to > now → live
            BaseRow::new("unknown", None, "w"),      // unknown → treated live
        ];
        let merged = merge_base_with_delta(base, &HashSet::new(), vec![], 1_000);
        assert_eq!(merged, vec!["z", "w"]);
    }

    /// Empty delta (nothing changed since the snapshot) returns the base intact.
    #[test]
    fn empty_delta_returns_base() {
        let base = vec![BaseRow::new("1", None, 1), BaseRow::new("2", None, 2)];
        let merged = merge_base_with_delta(base, &HashSet::new(), vec![], 1_000);
        assert_eq!(merged, vec![1, 2]);
    }
}
