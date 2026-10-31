//! O1: resolve OIDs and rehydrate records after the cross-segment merge.
//!
//! Each coalesced segment's cascade returns its top-k′ rows. The engine merge
//! (`BoundedPriorityQueue`, score-only) keeps k′ of the S×k′, so resolving
//! OIDs and rehydrating records per segment, before the merge, paid for
//! (S−1)×k′ rows that were then discarded. Here a cascade returns only
//! `(position, distance)`; this table hands out placeholder records with the
//! same scores, the unchanged merge picks the winners, and only the winners
//! are rehydrated — grouped per segment, in parallel. The OID comes from the
//! rehydrated record.
//!
//! The merge sees exactly the scores it saw before, in the same insertion
//! order, so it selects the same rows. Any failure to resolve a winner is a
//! [`DeferredResolveError`]; the caller then re-runs the search on the
//! per-segment path, so a placeholder never reaches a caller.
//!
//! Debugging: `RUST_LOG=proximadb::sst::deferred=debug` logs, per query, the
//! candidates deferred, the winners and segments resolved with timings, and
//! every failure. The `proximadb_pax_deferred_*` counters meter the same in
//! production (candidates − winners = rows never rehydrated).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::join_all;
use proximadb_distance_kernel::DistanceMetric;
use tracing::debug;

use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::engines::sst::SstEngine;
use crate::storage::engines::sst::centroid_directory_reader::DirectoryChoice;
use crate::storage::engines::sst::segment_format::CascadeHit;

/// The tracing target of the deferred path.
pub(crate) const TRACE_TARGET: &str = "proximadb::sst::deferred";

/// O1 kill switch: `PROXIMADB_PAX_READ_DEFERRED_REHYDRATE=0|off|false|no`
/// resolves OIDs per segment before the merge, as before.
pub(crate) fn deferred_rehydrate_enabled() -> bool {
    crate::storage::engines::sst::block_cluster::env_kill_switch_off(
        "PROXIMADB_PAX_READ_DEFERRED_REHYDRATE",
    )
}

/// A fresh deferred table when O1 is on (`None` otherwise), logging the
/// query's deferral mode.
pub(crate) fn maybe_new(
    collection_id: &str,
    k: usize,
) -> Option<std::sync::Arc<DeferredRehydrate>> {
    let table = deferred_rehydrate_enabled().then(|| std::sync::Arc::new(DeferredRehydrate::new()));
    tracing::debug!(
        target: TRACE_TARGET,
        collection_id,
        deferred = table.is_some(),
        k,
        "vector search: deferred rehydrate mode"
    );
    table
}

/// Whether `error` is a resolve failure a deferred attempt may retry on.
pub(crate) fn retry_due(retryable: bool, error: &anyhow::Error) -> bool {
    retryable && error.downcast_ref::<DeferredResolveError>().is_some()
}

/// Meter and log a deferred retry (a persistently bad block then shows up as
/// a climbing counter and doubled latency, not silence).
pub(crate) fn note_retry(collection_id: &str, error: &anyhow::Error) {
    crate::storage::engines::sst::metrics::PAX_DEFERRED_REHYDRATE_RETRIES.inc();
    tracing::warn!(
        target: TRACE_TARGET,
        collection_id,
        error = %error,
        "deferred rehydrate failed; re-running per segment"
    );
}

/// A deferred winner could not be resolved; the search must be re-run on the
/// per-segment path.
#[derive(Debug)]
pub(crate) struct DeferredResolveError(pub String);

impl std::fmt::Display for DeferredResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "deferred rehydrate: {}", self.0)
    }
}

impl std::error::Error for DeferredResolveError {}

/// One query's deferred cascade candidates.
pub(crate) struct DeferredRehydrate {
    /// Per-query placeholder-id prefix. It starts with NUL and carries a
    /// process-unique counter, and every placeholder must resolve to a table
    /// entry, so a real record id is never mistaken for one.
    prefix: String,
    hits: Mutex<Vec<(String, u32)>>,
}

impl DeferredRehydrate {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            prefix: format!(
                "\u{0}pax-deferred-{}:",
                NEXT.fetch_add(1, Ordering::Relaxed)
            ),
            hits: Mutex::new(Vec::new()),
        }
    }

    /// A placeholder record for one cascade hit: the score the per-segment
    /// path would give it, and an id naming its table entry.
    pub(crate) fn placeholder(
        &self,
        path: &str,
        hit: &CascadeHit,
        metric: &DistanceMetric,
    ) -> OptimizedSearchRecord {
        let mut hits = self.hits.lock().unwrap_or_else(|p| p.into_inner());
        let index = hits.len();
        hits.push((path.to_string(), hit.position));
        let sim = OptimizedSearchRecord::standardized_distance_to_similarity(hit.distance, metric);
        let mut record = OptimizedSearchRecord::new(format!("{}{index}", self.prefix), sim);
        record.similarity = Some(sim);
        record
    }

    fn entry(&self, id: &str, hits: &[(String, u32)]) -> Option<Result<(String, u32), ()>> {
        let index = id.strip_prefix(&self.prefix)?;
        Some(
            index
                .parse::<usize>()
                .ok()
                .and_then(|i| hits.get(i).cloned())
                .ok_or(()),
        )
    }

    /// Rehydrate the merged winners that are placeholders, in place. Records
    /// from non-cascade paths pass through unchanged. Each segment rehydrates
    /// inside its centroid-directory scope when the directory planned it, so
    /// it reuses the directory-decoded header and footer exactly as the
    /// per-segment path does (no cold header/footer GETs).
    pub(crate) async fn resolve(
        &self,
        engine: &SstEngine,
        mut records: Vec<OptimizedSearchRecord>,
        directory_cells: &HashMap<String, Arc<DirectoryChoice>>,
    ) -> Result<Vec<OptimizedSearchRecord>, DeferredResolveError> {
        let started = std::time::Instant::now();
        // `resolve` runs once per table, so the table is moved out, not copied.
        let hits = std::mem::take(&mut *self.hits.lock().unwrap_or_else(|p| p.into_inner()));
        crate::storage::engines::sst::metrics::PAX_DEFERRED_SEARCHES.inc();
        crate::storage::engines::sst::metrics::PAX_DEFERRED_CANDIDATES.inc_by(hits.len() as u64);
        let mut by_path: BTreeMap<String, Vec<(usize, u32)>> = BTreeMap::new();
        for (slot, record) in records.iter().enumerate() {
            match self.entry(&record.id, &hits) {
                None => {}
                Some(Ok((path, position))) => {
                    by_path.entry(path).or_default().push((slot, position))
                }
                Some(Err(())) => {
                    return Err(DeferredResolveError(format!(
                        "placeholder {:?} has no candidate",
                        record.id
                    )));
                }
            }
        }
        let winners: usize = by_path.values().map(Vec::len).sum();
        debug!(
            target: TRACE_TARGET,
            candidates = hits.len(),
            merged = records.len(),
            winners,
            segments = by_path.len(),
            "deferred rehydrate: resolving merged winners"
        );
        let reads = by_path.into_iter().map(|(path, slots)| async move {
            let segment_started = std::time::Instant::now();
            let fs = engine
                .filesystem()
                .get_filesystem(&path)
                .map_err(|e| DeferredResolveError(format!("opening {path}: {e}")))?;
            let positions: Vec<u32> = slots.iter().map(|(_, position)| *position).collect();
            let read = crate::storage::engines::sst::segment_format::read_records_by_positions(
                fs.as_ref(),
                &path,
                &positions,
                &[],
                &[],
                None,
                engine.segment_invariants_cache.as_deref(),
            );
            let rehydrated = crate::storage::engines::sst::segment_format::scope_directory_cells(
                directory_cells.get(&path).cloned(),
                read,
            )
            .await
            .map_err(|e| DeferredResolveError(format!("rehydrating {path}: {e}")))?;
            debug!(
                target: TRACE_TARGET,
                path = %path,
                winners = slots.len(),
                elapsed_us = segment_started.elapsed().as_micros() as u64,
                "deferred rehydrate: segment resolved"
            );
            Ok::<_, DeferredResolveError>((path, slots, rehydrated))
        });
        for read in join_all(reads).await {
            let (path, slots, rehydrated) = read?;
            if rehydrated.len() != slots.len() {
                return Err(DeferredResolveError(format!(
                    "{path}: {} records for {} positions",
                    rehydrated.len(),
                    slots.len()
                )));
            }
            for ((slot, position), full) in slots.into_iter().zip(rehydrated) {
                let full = full.ok_or_else(|| {
                    DeferredResolveError(format!("{path}: no record at position {position}"))
                })?;
                // A row decodes an undecodable OID stripe as "": the
                // per-segment path rejected such a row ("produced no
                // identity"), so this path does too.
                if full.oid.is_empty() {
                    return Err(DeferredResolveError(format!(
                        "{path}: row at position {position} has no OID"
                    )));
                }
                let record = records.get_mut(slot).ok_or_else(|| {
                    DeferredResolveError(format!("merged slot {slot} out of range"))
                })?;
                record.id = full.oid.clone();
                record.metadata =
                    proximadb_records::conversions::proxima_tree_to_value_map(&full.props);
                record.vector_id = Some(full.oid.clone());
                super::stamp_persisted_version(record, &full);
            }
        }
        if let Some(left) = records.iter().find(|r| r.id.starts_with(&self.prefix)) {
            return Err(DeferredResolveError(format!(
                "placeholder {:?} left unresolved",
                left.id
            )));
        }
        crate::storage::engines::sst::metrics::PAX_DEFERRED_WINNERS.inc_by(winners as u64);
        debug!(
            target: TRACE_TARGET,
            winners,
            skipped_rehydrates = hits.len().saturating_sub(winners),
            elapsed_us = started.elapsed().as_micros() as u64,
            "deferred rehydrate: done"
        );
        Ok(records)
    }
}
