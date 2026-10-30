//! Phase 5 Slice 5.4 — Merge WAL/memtable delta with directory-routed
//! search results.
//!
//! The strong-consistency read path produces two candidate sets per
//! query:
//!
//! 1. `directory_results` — records selected by the
//!    [`VectorObjectEconomyDirectory`](crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectory)-routed
//!    search over committed-and-flushed SST blocks. These are visible
//!    up to the directory's `freshness_watermark_lsn`.
//! 2. `delta_results` — records returned by
//!    [`scan_wal_delta_if_needed`](crate::services::operations::vectors::VectorOperationsService::scan_wal_delta_if_needed)
//!    for the WAL/memtable region committed after the watermark.
//!    Tombstones in this set are marked with an empty `vector` plus a
//!    set `expires_at`.
//!
//! [`merge_delta_with_directory_results`] combines the two with these
//! rules:
//!
//! * **The freshest copy of an OID wins** ([`resolve_freshest`]): later
//!   `updated_at` first; on an equal timestamp a tombstone wins (mandate
//!   16d — tombstones carry `record_version` 0, so versions are compared
//!   only after), then the higher version, then the delta. The delta
//!   usually holds the newer version, but NOT always: flush retains
//!   tombstone-only batches (TD-USUB-1), so a stale delta tombstone can be
//!   older than a flushed re-insert.
//! * **Dead winners suppress.** A winning tombstone or TTL-expired copy
//!   removes the OID from the merged set and is never returned.
//! * **Top-k truncation comes last.** After resolution and dead
//!   filtering, results are re-sorted by score (descending) and
//!   truncated to `top_k`.
//!
//! [`reconcile_against_wal_state`] is the two-stage search's merge: it drops a
//! persisted/indexed candidate whose OID has a live unflushed version (that
//! version is authoritative even when the query skipped it as non-matching),
//! and lets a candidate beat a dead unflushed version only if strictly newer.
//!
//! The function is a pure data transform: no I/O, no concurrency, no
//! filesystem. Fully unit-testable.

use std::collections::HashMap;

use crate::core::search::results::OptimizedSearchRecord;

/// True when this record is a WAL-emitted tombstone marker.
///
/// The WAL's `search_unflushed_vectors` returns tombstones with
/// `vector = None` and `expires_at = Some(t)` where `t` is the
/// expiration timestamp (already validated to be in the past at the
/// time of emission). Other empty-vector results (e.g. records
/// returned with `include_vectors=false`) carry no `expires_at`, so
/// the combined check is the right distinguishing signal for the
/// merge step.
fn is_wal_tombstone(record: &OptimizedSearchRecord) -> bool {
    // Canonical tombstone marker: valid_to_ns == Some(0). (The earlier
    // `vector.is_none() && expires_at.is_some()` heuristic broke under
    // `include_vectors=false`, where every delta record has vector == None,
    // and relied on the unit-muddled expires_at.) valid_to_ns is the
    // ns-accurate source of truth.
    record.valid_to_ns == Some(0)
}

/// Merge WAL/memtable delta candidates with directory-routed candidates
/// into a single top-k result list with MVCC semantics.
///
/// See module-level docs for the rules. The function does not mutate
/// either input vector beyond consuming them.
pub fn merge_delta_with_directory_results(
    delta_results: Vec<OptimizedSearchRecord>,
    directory_results: Vec<OptimizedSearchRecord>,
    top_k: usize,
) -> Vec<OptimizedSearchRecord> {
    if top_k == 0 {
        return Vec::new();
    }
    let now_ns = proximadb_kernel::time::now_unix_ns();

    // Resolve every OID to its FRESHEST copy (the delta wins exact ties, and
    // a tombstone wins an exact tie within the delta — mandate 16d). The old
    // rule "a delta tombstone suppresses every copy" let a STALE retained
    // tombstone (flush keeps tombstone-only batches, TD-USUB-1) hide a newer
    // flushed re-insert in the directory results.
    let resolved = resolve_freshest(
        delta_results
            .into_iter()
            .map(|r| (CandidateSource::Wal, r))
            .chain(
                directory_results
                    .into_iter()
                    .map(|r| (CandidateSource::Storage, r)),
            ),
    );

    // Dead winners (tombstones, TTL-expired) suppress their OID and are never
    // returned — canonical `is_record_dead`, never `expires_at`.
    let mut merged: Vec<OptimizedSearchRecord> = resolved
        .into_values()
        .map(|(_, record)| record)
        .filter(|record| {
            !is_wal_tombstone(record)
                && !proximadb_records::is_record_dead(record.valid_to_ns, now_ns)
        })
        .collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    merged.truncate(top_k);
    merged
}

/// Where a search candidate came from. The discriminant order is the
/// tie-break priority on an EXACT freshness tie (lower wins): unflushed WAL,
/// then the in-memory AXIS index, then persisted storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateSource {
    Wal = 0,
    Axis = 1,
    Storage = 2,
}

fn supersedes(
    candidate: &(CandidateSource, OptimizedSearchRecord),
    current: &(CandidateSource, OptimizedSearchRecord),
) -> bool {
    let (cu, ru) = (
        candidate.1.updated_at.unwrap_or(i64::MIN),
        current.1.updated_at.unwrap_or(i64::MIN),
    );
    if cu != ru {
        return cu > ru;
    }
    // Same timestamp: a tombstone wins BEFORE versions are compared (mandate
    // 16d). Tombstones carry record_version 0 while live versions can carry
    // >= 1 (SQL UPDATE, gRPC client versions), so a version-first tie-break
    // let a same-instant live copy beat its own delete.
    let (ct, rt) = (is_wal_tombstone(&candidate.1), is_wal_tombstone(&current.1));
    if ct != rt {
        return ct;
    }
    let (cv, rv) = (
        candidate.1.version.unwrap_or(0),
        current.1.version.unwrap_or(0),
    );
    if cv != rv {
        return cv > rv;
    }
    candidate.0 < current.0
}

/// Resolve every OID to its FRESHEST candidate across sources.
///
/// Replaces the historical source-priority merge (WAL > AXIS > storage, with
/// "a dead WAL copy beats a live one"). That rule was wrong in two directions:
/// a STALE retained WAL tombstone (flush keeps tombstone-only batches,
/// TD-USUB-1) hid a NEWER flushed re-insert, and a delete-then-reinsert in one
/// unflushed delta stayed deleted. Dead filtering happens AFTER this, never
/// before (the PR #2063 rule: pick the winner, then dead-filter).
pub fn resolve_freshest(
    candidates: impl IntoIterator<Item = (CandidateSource, OptimizedSearchRecord)>,
) -> HashMap<String, (CandidateSource, OptimizedSearchRecord)> {
    let mut best: HashMap<String, (CandidateSource, OptimizedSearchRecord)> = HashMap::new();
    for candidate in candidates {
        match best.get(&candidate.1.id) {
            Some(current) if !supersedes(&candidate, current) => {}
            _ => {
                best.insert(candidate.1.id.clone(), candidate);
            }
        }
    }
    best
}

/// The state of one OID's FRESHEST unflushed version, from the same WAL
/// snapshot the WAL leg scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalOidState {
    /// A live version: authoritative over every persisted / indexed copy
    /// (the get-by-id rule). If it matched the query and made the WAL leg's
    /// cut it is already a WAL hit; otherwise the OID is not a valid result.
    Live,
    /// A tombstone or TTL-expired version written at `updated_at_ms`. A
    /// persisted copy survives only if it is STRICTLY newer — a stale retained
    /// tombstone (flush keeps tombstone-only batches, TD-USUB-1) must not hide
    /// a newer flushed re-insert; an equal-millisecond copy loses to the delete.
    Dead { updated_at_ms: i64 },
}

/// Merge WAL hits with the other sources' candidates, reconciling each
/// non-WAL candidate against its OID's unflushed state.
///
/// `wal_hits` are the WAL leg's live, matching, freshest versions; markers
/// for dead versions (tombstones and TTL-expired) are skipped — the state
/// lookup carries their suppression. Every other
/// candidate is first resolved to its freshest copy across AXIS/storage, then:
/// * no unflushed version of its OID → kept;
/// * a live unflushed version → dropped (the WAL copy is authoritative: it is
///   either already a WAL hit, failed the predicate, or was outside the WAL
///   leg's top-k over distinct current matches, so cannot be in the global
///   top-k);
/// * a dead unflushed version → kept only if strictly newer (ms) than it. A
///   candidate without version facts can never prove that and is dropped —
///   the historical "WAL beats storage" outcome.
pub fn reconcile_against_wal_state(
    wal_hits: Vec<OptimizedSearchRecord>,
    others: impl IntoIterator<Item = (CandidateSource, OptimizedSearchRecord)>,
    wal_state: &dyn Fn(&str) -> Option<WalOidState>,
    now_ns: i64,
) -> HashMap<String, OptimizedSearchRecord> {
    let mut merged: HashMap<String, OptimizedSearchRecord> = HashMap::new();
    for hit in wal_hits {
        // Dead-version markers (tombstones AND TTL-expired) are never results;
        // the state lookup already carries their suppression.
        if is_wal_tombstone(&hit) || proximadb_records::is_record_dead(hit.valid_to_ns, now_ns) {
            continue;
        }
        merged.insert(hit.id.clone(), hit);
    }
    for (oid, (_source, candidate)) in resolve_freshest(others) {
        if merged.contains_key(&oid) {
            continue;
        }
        let keep = match wal_state(&oid) {
            None => true,
            Some(WalOidState::Live) => false,
            Some(WalOidState::Dead { updated_at_ms }) => {
                candidate.updated_at.is_some_and(|u| u > updated_at_ms)
            }
        };
        if keep {
            merged.insert(oid, candidate);
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::search::results::OptimizedSearchRecord;

    fn live(id: &str, score: f32) -> OptimizedSearchRecord {
        OptimizedSearchRecord {
            id: id.to_string(),
            vector_id: Some(id.to_string()),
            score,
            similarity: Some(score),
            ..Default::default()
        }
    }

    fn tombstone(id: &str) -> OptimizedSearchRecord {
        OptimizedSearchRecord {
            id: id.to_string(),
            vector_id: Some(id.to_string()),
            score: 0.0,
            similarity: Some(0.0),
            vector: None,
            // Canonical tombstone marker (valid_to_ns == Some(0)); expires_at
            // is display-only.
            valid_to_ns: Some(0),
            expires_at: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn empty_delta_returns_directory_results_unchanged_after_truncation() {
        let directory = vec![live("a", 0.9), live("b", 0.8), live("c", 0.7)];
        let merged = merge_delta_with_directory_results(vec![], directory, 10);

        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id, "a");
        assert_eq!(merged[1].id, "b");
        assert_eq!(merged[2].id, "c");
    }

    #[test]
    fn delta_only_oid_is_included_in_merged_results() {
        let directory = vec![live("a", 0.9), live("b", 0.8)];
        let delta = vec![live("z", 0.95)];
        let merged = merge_delta_with_directory_results(delta, directory, 10);

        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged[0].id, "z",
            "delta record with highest score wins position 0"
        );
        assert_eq!(merged[1].id, "a");
        assert_eq!(merged[2].id, "b");
    }

    #[test]
    fn same_oid_in_both_sets_delta_version_wins() {
        // Directory has a stale "a" with score 0.5; delta has the same
        // OID with a refreshed score 0.95. Delta must win.
        let directory = vec![live("a", 0.5), live("b", 0.8)];
        let delta = vec![live("a", 0.95)];
        let merged = merge_delta_with_directory_results(delta, directory, 10);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "a");
        assert_eq!(merged[0].score, 0.95, "delta-wins: refreshed score visible");
        assert_eq!(merged[1].id, "b");
    }

    #[test]
    fn tombstone_in_delta_suppresses_directory_result() {
        let directory = vec![live("a", 0.9), live("b", 0.8)];
        let delta = vec![tombstone("a")];
        let merged = merge_delta_with_directory_results(delta, directory, 10);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "b", "tombstoned OID removed from output");
    }

    #[test]
    fn tombstone_in_delta_suppresses_live_copy_in_same_delta() {
        // Insert-then-delete before any flush: both the live record and its
        // tombstone are unflushed, so both surface in the WAL delta. The
        // tombstone must suppress the live copy sitting beside it in the
        // same delta — not just the flushed/directory copy.
        let directory: Vec<OptimizedSearchRecord> = vec![];
        let delta = vec![
            live("a", 0.9), // live copy, inserted
            tombstone("a"), // tombstone, same OID, newer
            live("b", 0.8),
        ];
        let merged = merge_delta_with_directory_results(delta, directory, 10);

        assert_eq!(
            merged.len(),
            1,
            "live + tombstone for the same OID in the delta collapses to nothing"
        );
        assert_eq!(merged[0].id, "b");
    }

    #[test]
    fn mixed_directory_delta_tombstone_yields_expected_count() {
        // 3 directory + 2 delta (one live override + one new OID) + 1
        // tombstone for an unrelated directory OID. Expected output:
        //   - "a" suppressed by tombstone
        //   - "b" overridden by delta
        //   - "c" passes through from directory
        //   - "d" new from delta
        // => 3 final results.
        let directory = vec![live("a", 0.9), live("b", 0.5), live("c", 0.7)];
        let delta = vec![live("b", 0.95), live("d", 0.85), tombstone("a")];
        let merged = merge_delta_with_directory_results(delta, directory, 10);

        assert_eq!(merged.len(), 3);
        let ids: Vec<&str> = merged.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"a"), "tombstoned a suppressed");
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert!(ids.contains(&"d"));
        // Score order: b(0.95) > d(0.85) > c(0.7)
        assert_eq!(merged[0].id, "b");
        assert_eq!(merged[1].id, "d");
        assert_eq!(merged[2].id, "c");
    }

    #[test]
    fn top_k_truncates_after_merge_and_sort() {
        let directory = vec![live("a", 0.9), live("b", 0.8), live("c", 0.7)];
        let delta = vec![live("z", 0.95)];
        let merged = merge_delta_with_directory_results(delta, directory, 2);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "z");
        assert_eq!(merged[1].id, "a");
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let directory = vec![live("a", 0.9)];
        let delta = vec![live("z", 0.95)];
        let merged = merge_delta_with_directory_results(delta, directory, 0);
        assert!(merged.is_empty());
    }

    #[test]
    fn ties_break_by_id_for_determinism() {
        let directory = vec![live("a", 0.5), live("b", 0.5)];
        let merged_run1 = merge_delta_with_directory_results(vec![], directory.clone(), 10);
        let merged_run2 = merge_delta_with_directory_results(vec![], directory, 10);

        assert_eq!(merged_run1[0].id, "a");
        assert_eq!(merged_run1[1].id, "b");
        // Two independent runs must produce the same order.
        assert_eq!(
            merged_run1.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            merged_run2.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
        );
    }

    fn versioned(id: &str, score: f32, updated_at: i64, dead: bool) -> OptimizedSearchRecord {
        OptimizedSearchRecord {
            id: id.to_string(),
            vector_id: Some(id.to_string()),
            score,
            similarity: Some(score),
            updated_at: Some(updated_at),
            version: Some(1),
            valid_to_ns: if dead { Some(0) } else { None },
            ..Default::default()
        }
    }

    /// R7: a STALE delta tombstone (t=200) must not hide a NEWER flushed
    /// re-insert (t=300) — flush retains tombstone-only batches.
    #[test]
    fn stale_delta_tombstone_does_not_hide_newer_flushed_reinsert() {
        let delta = vec![versioned("x", 0.0, 200, true)];
        let directory = vec![versioned("x", 0.9, 300, false)];
        let merged = merge_delta_with_directory_results(delta, directory, 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, "x");
        assert_eq!(merged[0].updated_at, Some(300));
    }

    /// The common case still holds: a NEWER delta tombstone suppresses the
    /// older flushed copy.
    #[test]
    fn newer_delta_tombstone_suppresses_older_flushed_copy() {
        let delta = vec![versioned("x", 0.0, 300, true)];
        let directory = vec![versioned("x", 0.9, 200, false)];
        assert!(merge_delta_with_directory_results(delta, directory, 10).is_empty());
    }

    /// Mandate 16d: insert and delete both unflushed with the SAME timestamp
    /// — the tombstone wins the tie and the OID stays deleted.
    #[test]
    fn same_instant_insert_then_delete_stays_deleted() {
        let delta = vec![
            versioned("x", 0.9, 200, false),
            versioned("x", 0.0, 200, true),
        ];
        assert!(merge_delta_with_directory_results(delta, vec![], 10).is_empty());
        let delta_rev = vec![
            versioned("x", 0.0, 200, true),
            versioned("x", 0.9, 200, false),
        ];
        assert!(merge_delta_with_directory_results(delta_rev, vec![], 10).is_empty());
    }

    /// Delete then re-insert, both unflushed: the newer re-insert is visible
    /// (the old "dead wins" rule hid it).
    #[test]
    fn delete_then_reinsert_in_one_delta_is_visible() {
        let delta = vec![
            versioned("x", 0.0, 200, true),
            versioned("x", 0.8, 300, false),
        ];
        let merged = merge_delta_with_directory_results(delta, vec![], 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].updated_at, Some(300));
    }

    fn with_version(mut r: OptimizedSearchRecord, version: u32) -> OptimizedSearchRecord {
        r.version = Some(version);
        r
    }

    /// Review MAJOR-1: production tombstones carry record_version 0, while a
    /// SQL UPDATE / gRPC client can stamp version >= 1. A same-millisecond
    /// delete must still win — version must never outrank the tombstone.
    #[test]
    fn same_millisecond_version0_tombstone_beats_version2_live_copy() {
        let tomb = with_version(versioned("x", 0.0, 500, true), 0);
        let live = with_version(versioned("x", 0.9, 500, false), 2);
        for delta in [vec![tomb.clone(), live.clone()], vec![live, tomb]] {
            assert!(
                merge_delta_with_directory_results(delta, vec![], 10).is_empty(),
                "a same-ms delete must stay deleted regardless of versions"
            );
        }
    }

    /// R2: a LIVE unflushed version supersedes the persisted copy even when the
    /// query skipped it (non-matching) — the stale copy must not be returned.
    #[test]
    fn live_unflushed_version_supersedes_persisted_copy() {
        let others = vec![
            (
                CandidateSource::Storage,
                versioned("moved", 0.9, 100, false),
            ),
            (CandidateSource::Storage, versioned("stay", 0.8, 100, false)),
        ];
        let state = |oid: &str| (oid == "moved").then_some(WalOidState::Live);
        let merged = reconcile_against_wal_state(vec![], others, &state, 0);
        let mut kept: Vec<&str> = merged.keys().map(String::as_str).collect();
        kept.sort();
        assert_eq!(kept, vec!["stay"]);
    }

    /// R7: a persisted copy NEWER than a dead unflushed version (a stale
    /// retained tombstone) wins; an equal-ms or older copy loses.
    #[test]
    fn persisted_copy_beats_only_an_older_dead_unflushed_version() {
        let others = vec![
            (
                CandidateSource::Storage,
                versioned("reborn", 0.9, 300, false),
            ),
            (CandidateSource::Storage, versioned("tied", 0.8, 250, false)),
            (CandidateSource::Storage, versioned("gone", 0.7, 100, false)),
        ];
        let state = |_: &str| Some(WalOidState::Dead { updated_at_ms: 250 });
        let merged = reconcile_against_wal_state(vec![], others, &state, 0);
        let kept: Vec<&str> = merged.keys().map(String::as_str).collect();
        assert_eq!(kept, vec!["reborn"]);
    }

    /// Review MAJOR-2: an unversioned candidate (an AXIS hit) with NO
    /// unflushed version of its OID is kept; it is dropped only when an
    /// unflushed version exists (live, or dead with no proof of being newer).
    #[test]
    fn unversioned_candidate_kept_unless_its_oid_is_unflushed() {
        let others = vec![
            (CandidateSource::Axis, live("indexed_only", 0.9)),
            (CandidateSource::Axis, live("has_live_wal", 0.8)),
            (CandidateSource::Axis, live("has_dead_wal", 0.7)),
        ];
        let state = |oid: &str| match oid {
            "has_live_wal" => Some(WalOidState::Live),
            "has_dead_wal" => Some(WalOidState::Dead { updated_at_ms: 1 }),
            _ => None,
        };
        let merged = reconcile_against_wal_state(vec![], others, &state, 0);
        let kept: Vec<&str> = merged.keys().map(String::as_str).collect();
        assert_eq!(kept, vec!["indexed_only"]);
    }

    /// WAL hits are kept as-is and win over any other copy of the same OID;
    /// WAL tombstone markers are not returned.
    #[test]
    fn wal_hits_win_and_markers_are_not_returned() {
        let wal_hits = vec![
            versioned("w", 0.9, 400, false),
            versioned("t", 0.0, 400, true),
        ];
        let others = vec![(CandidateSource::Storage, versioned("w", 0.95, 100, false))];
        let state = |oid: &str| match oid {
            "w" => Some(WalOidState::Live),
            "t" => Some(WalOidState::Dead { updated_at_ms: 400 }),
            _ => None,
        };
        let merged = reconcile_against_wal_state(wal_hits, others, &state, 0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged["w"].updated_at, Some(400), "the WAL copy wins");
    }
}
