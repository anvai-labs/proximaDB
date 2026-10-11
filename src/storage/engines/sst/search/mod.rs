/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! SST Engine Search Module
//!
//! Contains search operations and coordination logic for the SST engine.
//! This module implements the three-stage filtering pipeline:
//! 1. Bloom filter stage - eliminate non-matching SST files
//! 2. Row filter stage - filter records within SST files
//! 3. Vector stage - compute distances for remaining candidates
//!
//! The module provides:
//! - Main unified search implementation
//! - Direct search fallback for simple queries
//! - Search coordination and optimization
//! - File discovery and routing logic

pub mod coordinator;
pub mod operations;
pub mod optimizer;

use anyhow::Result;
use futures::future::join_all;

/// TD-RDSTRAT-13 PR-B: records the cascade umbrella wall on drop, so every
/// exit path (returns, `?`) attributes. Pairs with the four stage timers.
struct UmbrellaTimer {
    started: std::time::Instant,
}

impl UmbrellaTimer {
    fn new(started: std::time::Instant) -> Self {
        Self { started }
    }
}

impl Drop for UmbrellaTimer {
    fn drop(&mut self) {
        let us = self.started.elapsed().as_micros() as u64;
        crate::observability::io_trace::record_cascade_total_us(us);
        crate::storage::engines::sst::metrics::record_cascade_total_us(us);
    }
}

/// TD-RDSTRAT-13 C3: a discovered segment plus the size the listing already
/// reported. `size_bytes` lets the read path skip the per-segment HEAD (one
/// billed metadata op per segment per query); `None` means the backend did
/// not report a size and the reader must stat.
#[derive(Clone)]
pub(crate) struct DiscoveredSstFile {
    pub url: String,
    pub size_bytes: Option<u64>,
}

/// TD-SEARCH-2 S2: process-wide in-flight cold-scan counter — the input to
/// the adaptive morsel degree. Incremented per `fallback_to_direct_search`
/// (the segment-scan path; memtable/index serves don't burn scan CPU).
pub static INFLIGHT_SCANS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// RAII guard for [`INFLIGHT_SCANS`].
pub struct ScanGuard;

impl ScanGuard {
    pub fn enter() -> Self {
        INFLIGHT_SCANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ScanGuard
    }
}

impl Drop for ScanGuard {
    fn drop(&mut self) {
        INFLIGHT_SCANS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// TD-SEARCH-2 S2: the adaptive intra-file morsel degree —
/// `clamp(cores / inflight_scans, 1, cores)`. A lone cold query spreads its
/// RaBitQ rank across every core (minimum latency); at high concurrency each
/// query degrades toward sequential (maximum throughput, no oversubscription:
/// total CPU workers ≈ cores regardless of load).
///
/// `PROXIMADB_SEARCH_MORSEL_DEGREE`: unset/`0` = adaptive, `1` = off
/// (sequential rank), `n` = fixed n workers (clamped to cores).
pub fn morsel_degree() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if let Ok(v) = std::env::var("PROXIMADB_SEARCH_MORSEL_DEGREE")
        && let Ok(n) = v.trim().parse::<usize>()
        && n > 0
    {
        return n.min(cores);
    }
    let inflight = INFLIGHT_SCANS
        .load(std::sync::atomic::Ordering::Relaxed)
        .max(1);
    // Measured posture (2026-07-25, 1M/3-segment cold A/B sweeps): the lone-
    // query win is stable and reproducible — c=1 cold 5.8→10.4 QPS, mean
    // 174→96 ms (all cores on one rank) — while every mid/high-concurrency
    // configuration tried (cores/inflight, cores/(inflight+1)) was inside the
    // box's ±30% run-to-run throughput variance, with regressions observed in
    // matched pairs. So the default engages morsels ONLY when this scan is
    // alone in flight; any concurrency runs the sequential (baseline) rank —
    // zero throughput risk. Operators can force a fixed degree via
    // `PROXIMADB_SEARCH_MORSEL_DEGREE` for latency-first deployments.
    if inflight == 1 { cores } else { 1 }
}

/// TD-SEARCH-2: resolve the inter-file search parallelism degree.
///
/// Config field `search_parallel_files` (from `[storage.optimization]` in TOML):
/// - `0` (default) → 50% of CPU cores (wise default — leaves cores for flush/compaction/gRPC)
/// - `1` → sequential (no parallelism; for debugging)
/// - `n > 1` → exactly n parallel workers
///
/// Hot-path override: `PROXIMADB_SEARCH_PARALLEL_FILES` env var takes precedence
/// over the config field — operators can tune without a restart.
fn resolve_search_parallelism(config_value: u16) -> u16 {
    // Env override (hot-path tuning)
    if let Ok(v) = std::env::var("PROXIMADB_SEARCH_PARALLEL_FILES")
        && let Ok(n) = v.trim().parse::<u16>()
    {
        return resolve_parallel_degree(n);
    }
    let base = resolve_parallel_degree(config_value);
    // S2b: when auto (config_value == 0), divide by in-flight queries so
    // concurrent searches don't oversubscribe. Explicit n > 0 is honored.
    if config_value == 0 {
        adaptive_degree(
            base,
            IN_FLIGHT_SEARCHES.load(std::sync::atomic::Ordering::Relaxed),
        )
    } else {
        base
    }
}

fn resolve_parallel_degree(requested: u16) -> u16 {
    let cores = std::thread::available_parallelism()
        .map(|n| u16::try_from(n.get()).unwrap_or(u16::MAX))
        .unwrap_or(4);
    // Advisory ceiling: 2× logical cores (I/O-bound search oversubscription —
    // workers wait on object-store GETs, so extra workers use idle CPU).
    let ceiling = cores.saturating_mul(2);
    match requested {
        0 => (cores / 2).max(1),            // 0 = half the cores (wise default)
        _ => requested.min(ceiling).max(1), // clamp to [1, 2×cores]
    }
}
// TD-SEARCH-2 S2b: process-global count of concurrent searches. The auto
// parallel degree divides by this (cores/2 / in_flight) so concurrent queries
// don't oversubscribe. Explicit config/env values bypass the adaptive cap.
static IN_FLIGHT_SEARCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// S2b: adaptive degree formula - pure for unit-testing. Divides base by the
/// in-flight count, floored at 1.
fn adaptive_degree(base: u16, in_flight: usize) -> u16 {
    (base / (in_flight.max(1) as u16)).max(1)
}

/// S2b: RAII guard - increments on acquire, decrements on drop. Lives for the
/// search scope so every exit path (return, ?, unwind) decrements correctly.
struct InFlightSearchGuard;
impl InFlightSearchGuard {
    fn acquire() -> Self {
        IN_FLIGHT_SEARCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}
impl Drop for InFlightSearchGuard {
    fn drop(&mut self) {
        IN_FLIGHT_SEARCHES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

use std::collections::HashMap;
use tracing::{debug, info, trace, warn}; // TD-SEARCH-2: concurrent inter-file scan

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::engines::core::formats::arrow_block::ArrowBlockReader;
use crate::storage::engines::sst::{SstEngine, SstError};
use crate::storage::traits::{StorageQueryContext, UnifiedStorageFormat};
use proximadb_distance_kernel::DistanceMetric;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_index_traits::{
    IndexFilterOperator, IndexHybridQuery, IndexMetadataFilter, IndexSearchEffort, IndexVectorQuery,
};

pub use coordinator::SearchCoordinator;
pub use operations::SearchOperations;
pub use optimizer::SearchOptimizer;

/// TD-112: per-(process, collection) in-flight guard so lazy rebuild-from-SST
/// does not stampede when several cold queries race. It is not a completion
/// cache; `registered_vector_count` is the durable warm signal. A rebuild permit
/// clears this set on drop so cancellation/panic cannot leave a stale guard.
static AXIS_REBUILD_GUARD: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<String>>,
> = std::sync::OnceLock::new();

fn axis_rebuild_guard() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    AXIS_REBUILD_GUARD.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

struct AxisRebuildPermit {
    collection_id: String,
}

impl Drop for AxisRebuildPermit {
    fn drop(&mut self) {
        if let Some(guard) = AXIS_REBUILD_GUARD.get() {
            let mut in_flight = guard
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            in_flight.remove(&self.collection_id);
        }
    }
}

fn try_begin_axis_rebuild(collection_id: &str) -> Option<AxisRebuildPermit> {
    let mut in_flight = axis_rebuild_guard()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !in_flight.insert(collection_id.to_string()) {
        return None;
    }
    Some(AxisRebuildPermit {
        collection_id: collection_id.to_string(),
    })
}

/// ADR-030 / TD-158: attribute an SST vector-search query's compute time to KRU
/// on **any** exit path (the search has several early returns — exact-scan,
/// orchestrated, direct). On drop it records the elapsed millis to the active
/// per-query I/O trace under the given engine label; `record_compute_ms` no-ops
/// outside an `io_trace` scope, so internal/test callers are unaffected.
struct ComputeMsGuard {
    engine: &'static str,
    started: std::time::Instant,
}

impl ComputeMsGuard {
    fn new(engine: &'static str) -> Self {
        Self {
            engine,
            started: std::time::Instant::now(),
        }
    }
}

impl Drop for ComputeMsGuard {
    fn drop(&mut self) {
        crate::observability::io_trace::record_compute_ms(
            self.engine,
            self.started.elapsed().as_millis() as u64,
        );
    }
}

/// TD-165: the exact-vs-approximate route is a *cost* decision, and for a cloud DB
/// the dominant term is bytes read from object storage, not vectors or CPU. An
/// `Adaptive` search takes the exact brute-force segment scan when the whole scan
/// fits one efficient ranged GET (`N · dim · 4 bytes ≤` this budget) — cheaper than
/// rebuilding a cold in-memory index, which would read the same bytes anyway, and
/// 100% recall. Above the budget the persisted/approximate index is used. 64 MiB is
/// a conservative single-ranged-read size; full storage-class-aware tuning (object
/// store round-trips vs local NVMe bandwidth) is config-driven follow-up. See
/// `docs/12-design/EXACT_VS_ANN_ROUTING_COST_MODEL_2026_06_26.adoc`.
// TODO(serverless follow-up): source this from config per storage class.
const EXACT_SCAN_MAX_BYTES: usize = 64 * 1024 * 1024;

/// ADR-028: resolve the collection's `index_policy` into the exact-scan decision
/// inputs for the SST adaptive route. Returns `(byte_budget, pin_exact)`:
///
/// * `pin_exact` — the owner set `mode = "exact"`, so the adaptive route must
///   scan exactly at any N (the 100%-recall SLA).
/// * `byte_budget` — a non-zero `byte_budget` override, else the storage-class
///   default `EXACT_SCAN_MAX_BYTES`.
///
/// Pure + total so it is unit-testable without a live collection. Query-time
/// precedence (per-query `SearchMode` over policy) is enforced by the caller: this
/// only refines the `Adaptive` (no explicit per-query intent) arm.
fn resolve_exact_budget(policy: Option<&crate::proto::proximadb_v1::IndexPolicy>) -> (usize, bool) {
    match policy {
        Some(p) => {
            let pin_exact = p.mode.trim().eq_ignore_ascii_case("exact");
            let budget = if p.byte_budget > 0 {
                p.byte_budget as usize
            } else {
                EXACT_SCAN_MAX_BYTES
            };
            (budget, pin_exact)
        }
        None => (EXACT_SCAN_MAX_BYTES, false),
    }
}

impl SstEngine {
    /// TD-165: best-effort total vector count across a collection's segments, read
    /// cheaply from each segment header (8-byte prefix + the bincode header — no
    /// data blocks). Feeds the small-collection exact-recall gate. Returns 0 on any
    /// error or for non-`SST1` segments (Arrow/PAX), so the gate then falls through
    /// to the normal approximate/orchestrated path.
    async fn segment_vector_count(&self, storage_url: &str) -> usize {
        let files = match self.discover_sstable_files(storage_url).await {
            Ok(files) => files,
            Err(_) => return 0,
        };
        let mut total = 0usize;
        for file_path in &files {
            if let Ok(Some(entry_count)) = self.legacy_sst_entry_count(&file_path.url).await {
                total = total.saturating_add(entry_count);
            }
        }
        total
    }

    /// Read the count carried by a legacy `SST1` header.
    ///
    /// Current PAX files expose no useful count through this legacy header, so
    /// their durable suffix is enough to return `None` without a paid magic
    /// probe. Unknown/non-PAX paths still sniff the header for mixed-format
    /// safety.
    async fn legacy_sst_entry_count(&self, file_path: &str) -> Result<Option<usize>> {
        use crate::storage::engines::sst::SstableHeader;

        if file_path.ends_with(".pax") {
            return Ok(None);
        }
        let fs = self.filesystem().get_filesystem(file_path)?;
        let prefix = fs.read_range(file_path, 0, 8).await?;
        if prefix.len() < 8 || &prefix[0..4] != b"SST1" {
            return Ok(None);
        }
        let header_len = u32::from_le_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]) as u64;
        let header_data = fs.read_range(file_path, 8, header_len).await?;
        let header: SstableHeader = bincode::deserialize(&header_data)?;
        Ok(Some(header.entry_count as usize))
    }

    /// TD-165: exact brute-force search over the collection's segment(s), bypassing
    /// any approximate index. Forces `prune_config.force_exact` so neither the
    /// centroid file-pruning nor the Z-order block-pruning can drop the true NN —
    /// the same guarantee the index-less embedded path already provides.
    async fn execute_exact_segment_scan(
        &self,
        ctx: &StorageQueryContext,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
        filter_expression: Option<&FilterExpression>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let exact_params = Self::force_exact_params(ctx.search_params.as_ref());
        let exact_ctx = StorageQueryContext {
            search_params: std::sync::Arc::new(exact_params),
            collection: ctx.collection.clone(),
            metadata: ctx.metadata.clone(),
            user_context: ctx.user_context.clone(),
            tenant_context: ctx.tenant_context.clone(),
        };
        self.fallback_to_direct_search(
            &exact_ctx,
            collection_id,
            storage_url,
            query_vector,
            k,
            distance_metric,
            filter_expression,
            true, // include_vectors
            true, // include_metadata
        )
        .await
    }

    /// Build the effective exact-search contract from caller parameters.
    /// `want_exact_search` may resolve Adaptive to exact; both the mode-driven
    /// file pruner and the block/cascade gates must see that same resolution.
    fn force_exact_params(
        params: &crate::core::search::SearchParams,
    ) -> crate::core::search::SearchParams {
        let mut exact = params.clone();
        exact.search_mode = crate::core::search::SearchMode::Exact;
        exact.block_prune.force_exact = true;
        exact
    }

    /// PAX RaBitQ→SQ8 cascade attempt for a `.pax` segment — the co-designed
    /// approximate read path (P3 C.2: metadata pre-prune → RaBitQ candidate rank
    /// → SQ8 rerank, full f32 never decoded). Returns `Ok(Some)` when the cascade
    /// served, or `Ok(None)`/`Err` when it doesn't apply (segment isn't PAX, has no
    /// RaBitQ-coded embedding, or a transient I/O error). The caller falls back to
    /// the generic materialize-and-score scan in all of those cases, so this is
    /// additive and mixed-read-safe.
    ///
    /// **Metric gate.** Serves only Euclidean collections — the cascade's rerank is
    /// L2-validated (recall 0.932 @ N=100k). Other metrics stay on the generic scan
    /// until the cascade is metric-generalized + re-validated (follow-up).
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "cold-deletion-vectors"), allow(unused_variables))]
    async fn try_pax_cascade(
        &self,
        sstable_path: &str,
        query_vector: &[f32],
        filter_expression: Option<&FilterExpression>,
        k: usize,
        distance_metric: DistanceMetric,
        collection_id: &str,
        collection_root: &str,
        snapshot_lsn: u64,
        known_size: Option<u64>,
    ) -> anyhow::Result<Option<Vec<OptimizedSearchRecord>>> {
        // TD-RDSTRAT-13 PR-B: umbrella wall over the whole cascade — the four
        // stage timers (probe rank, Region-B, Region-D, rehydrate) decompose
        // this span; their sum vs this value names any unattributed remainder.
        let cascade_started = std::time::Instant::now();
        let _cascade_umbrella = UmbrellaTimer::new(cascade_started);
        use proximadb_block_format::RankMetric;
        // Map the query metric to a cascade rank metric. Euclidean + Cosine
        // carry the SIFT-ratchet recall validation; DotProduct is also routed
        // (`should_try_pax_cascade`) and mapped, but its filtered-recall
        // validation is the TD-FPRUNE-1 filtered-ratchet follow-up — exotic
        // metrics stay on the generic scan.
        let Some(rank_metric) = (match distance_metric {
            DistanceMetric::Euclidean => Some(RankMetric::L2),
            DistanceMetric::Cosine => Some(RankMetric::Cosine),
            DistanceMetric::DotProduct => Some(RankMetric::DotProduct),
            _ => None,
        }) else {
            return Ok(None);
        };
        let fs = self
            .filesystem()
            .get_filesystem(sstable_path)
            .map_err(|e| anyhow::anyhow!("opening PAX segment {sstable_path}: {e}"))?;

        // ADR-062 / TD-RDSTRAT-6: a coalesced-RaBitQ segment takes the
        // scan-then-rerank path — one RaBitQ-region GET (keep=100% rank) + one
        // footer GET + a few coalesced survivor-block GETs. Detected by the 4 B
        // `SEG_HEADER_MAGIC` head; a legacy segment (PBLK head) falls through to
        // the in-block RaBitQ cascade below (mixed-read). Any I/O error / `None`
        // also falls through (safe degradation, never an incorrect result).
        //
        // Filtered queries: with the ADR-089 P1 gate ON, a Stage-F metadata
        // pre-scan builds a row allow-set and the cascade ranks ONLY matching
        // rows (row-accurate — same `evaluate_filter_proxima` semantics as the
        // exact path). Gate OFF (default), a non-coalesced segment, or any
        // stage-F failure routes the filtered query past the cascade to the
        // exact materialize-and-rank path below, exactly as before.
        {
            use crate::storage::engines::sst::segment_format::{
                pax_filtered_cascade_enabled, pax_filtered_row_allow,
                rabitq_search_segment_coalesced_allowed,
            };
            // ADR-089 / TD-FPRUNE-1 P1: a filtered query builds a predicate
            // row allow-set from the Region-D metadata (Stage F) and runs the
            // cascade restricted to matching rows — instead of declining into
            // the whole-object exact scan. Default-OFF env gate; any stage-F
            // failure or a non-coalesced segment falls back to the exact path
            // (fail-safe, never an incorrect result).
            let row_allow = match filter_expression {
                Some(filter) if pax_filtered_cascade_enabled() => {
                    match pax_filtered_row_allow(fs.as_ref(), sstable_path, filter).await {
                        Ok(Some((allow, _stats))) if allow.is_empty() => {
                            // Provably zero matching rows in THIS segment —
                            // an empty per-file result (other segments/WAL
                            // still contribute via the caller's merge).
                            return Ok(Some(Vec::new()));
                        }
                        Ok(Some((allow, _stats))) => Some(allow),
                        Ok(None) => None, // non-coalesced → exact fallback
                        Err(e) => {
                            tracing::warn!(
                                "stage-F row allow-set failed for {sstable_path} \
                                 (falling back to exact scan): {e}"
                            );
                            None
                        }
                    }
                }
                _ => None,
            };
            // ADR-065: call the coalesced path directly — it reads the 56 B
            // header-prefix internally (cached) + returns Ok(None) for a non-
            // coalesced segment, so no separate 4 B magic-detection GET is needed
            // (collapses the redundant offset-0 prefix read the FS trace found).
            if filter_expression.is_none() || row_allow.is_some() {
                let coalesced_hits = rabitq_search_segment_coalesced_allowed(
                    fs.as_ref(),
                    sstable_path,
                    query_vector,
                    k,
                    rank_metric,
                    self.segment_invariants_cache.as_deref(),
                    self.survivor_cache.as_deref(),
                    row_allow.as_ref(),
                    known_size,
                )
                .await?;
                if let Some(hits) = coalesced_hits {
                    // TD-DELVEC-1 WI-4 slice 2: merge-on-read on the RaBitQ ANN
                    // path — warm this segment's deletion vector, then drop hits
                    // whose row position (`CascadeHit::position`, the global row
                    // index the coalesced scan scored = the space the DV keys on)
                    // is deleted as of the scan's snapshot LSN. A cold delete is
                    // invisible on the cascade path too. Best-effort: no DV store
                    // or a load failure ⇒ no skipping (degraded, not a query
                    // failure). Under the default build this filter is cfg'd out.
                    #[cfg(feature = "cold-deletion-vectors")]
                    let hits = if let Some(dv) = self.deletion_vector_store.as_ref() {
                        let _ = dv.load(sstable_path).await;
                        let mut kept = Vec::with_capacity(hits.len());
                        for h in hits {
                            if !dv
                                .is_deleted_as_of(sstable_path, h.position, snapshot_lsn)
                                .await
                            {
                                kept.push(h);
                            }
                        }
                        kept
                    } else {
                        hits
                    };
                    // TD-FPRUNE-1 top-k rehydration: the exact path this cascade
                    // replaces returns full record metadata (props); the cascade
                    // ranks from codes and knows only id+score+vector. Fetch the
                    // k winners' records (position-selective, bounded by the hit
                    // set) so the filtered response keeps the same shape as the
                    // exact path. Best-effort: a rehydrate failure degrades to
                    // metadata-less hits (the pre-flip cascade behavior), never a
                    // query failure.
                    let positions: Vec<u32> = hits.iter().map(|h| h.position).collect();
                    let rehydrated =
                        crate::storage::engines::sst::segment_format::read_records_by_positions(
                            fs.as_ref(),
                            sstable_path,
                            &positions,
                            &[],
                            &[],
                            None,
                            self.segment_invariants_cache.as_deref(),
                        )
                        .await
                        .unwrap_or_else(|e| {
                            tracing::warn!(
                                "cascade top-k rehydration failed for {sstable_path} \
                                 (hits return without metadata): {e}"
                            );
                            vec![None; positions.len()]
                        });
                    let records = hits
                        .into_iter()
                        .zip(rehydrated)
                        .map(|(h, rec)| {
                            let sim = OptimizedSearchRecord::standardized_distance_to_similarity(
                                h.distance,
                                &distance_metric,
                            );
                            let mut r = OptimizedSearchRecord::new(h.oid, sim);
                            // `new` leaves `similarity: None`, and the response
                            // boundary displays `similarity.unwrap_or(0.0)` — set
                            // it so cascade hits don't render as score 0.0.
                            r.similarity = Some(sim);
                            if let Some(v) = h.vector {
                                r = r.add_vector(v);
                            }
                            if let Some(full) = rec {
                                r.metadata =
                                    proximadb_records::conversions::proxima_tree_to_value_map(
                                        &full.props,
                                    );
                                r.vector_id = Some(full.oid.clone());
                            }
                            r
                        })
                        .collect();
                    crate::observability::io_trace::record_vector_ann_proof();
                    return Ok(Some(records));
                }
            }
        }

        // The legacy in-block RaBitQ cascade (whole-read / striped / centroid-prune)
        // was superseded by coalesced scan-then-rerank (ADR-062 / TD-RDSTRAT-6) and
        // removed. A non-coalesced RaBitQ segment (none exist after the #1024
        // default-on flip) or a filtered query falls through to the caller's exact
        // materialize-and-rank path (`search_pax_file_exact`) — safe degradation,
        // never an incorrect result.
        let _ = (collection_id, collection_root);
        Ok(None)
    }

    // pax_rabitq_pool_for_top_k moved to segment_format.rs (PR2: adaptive M,
    // now scales with the segment's row count N via region.n_rows()).

    /// Main unified search implementation with orchestration
    ///
    /// This is the primary search entry point that implements intelligent
    /// search routing and the three-stage filtering pipeline.
    pub async fn search_vectors_unified(
        &self,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let _search_start = std::time::Instant::now();
        // ADR-030 / TD-158: attribute this query's SST compute to KRU on any exit
        // path (records the elapsed time to the active io_trace on drop).
        let _compute_guard = ComputeMsGuard::new("sst");

        // Track metadata access for cache optimization
        if let Some(orch) = self.orchestrator() {
            (**orch).pattern_tracker().track_access_async(
                format!("{}::sst::metadata", ctx.collection_id()),
                crate::storage::cache::orchestrator::CacheType::Metadata,
            );
        }

        // Extract search parameters from context
        let collection_id = ctx.collection_id();
        let storage_url = ctx
            .collection_storage_path()
            .ok_or_else(|| SstError::InvalidArgument("No storage URL in context".into()))?;
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| SstError::InvalidArgument("No query vector in context".into()))?;
        let k = ctx.top_k();
        let distance_metric = ctx.distance_metric();
        let filter_expression = ctx.search_params.filter_expression.as_ref();
        let cache_policy = self.vector_cache_policy();

        info!(
            "🚀 SST: Starting unified search for collection {} with {} dimensions",
            collection_id,
            query_vector.len()
        );

        // TD-165: route exact-vs-approximate by the query's `SearchMode` intent —
        // the orchestrated index path below otherwise ignores it, so an `Exact`
        // query (the default) silently received approximate results, and a cold
        // HNSW rebuild was observed excluding the true NN from its candidate pool
        // entirely (f32-rerank cannot recover a candidate that was never surfaced).
        //   - `Exact`        → exact segment scan (honor the 100% contract).
        //   - `Approximate`  → orchestrated index (caller accepted the recall trade).
        //   - `Adaptive{t}`  → exact when the scan fits one ranged GET (cost gate)
        //                      and within the per-query count cap `t`, else index.
        // The cost gate is dim-aware (bytes = N·dim·4), not a flat vector count.
        // `segment_vector_count` is read only for `Adaptive`, and only the exact arm
        // is taken here — which also skips the multi-second cold AXIS rebuild below.
        // See `docs/12-design/EXACT_VS_ANN_ROUTING_COST_MODEL_2026_06_26.adoc`.
        // ADR-028 precedence: an explicit per-query `SearchMode` (Exact/Approximate)
        // wins outright; the default `Adaptive` arm defers to the collection
        // `index_policy` (mode=exact pin, or a byte_budget override) and then the
        // cost-derived auto-default.
        let want_exact = self
            .want_exact_search(ctx, query_vector, &storage_url)
            .await;
        if want_exact {
            info!(
                "🎯 SST: exact segment scan for collection {} (SearchMode honored; cost-gated) — guaranteed recall (TD-165)",
                collection_id
            );
            let result = self
                .execute_exact_segment_scan(
                    ctx,
                    collection_id,
                    &storage_url,
                    query_vector,
                    k,
                    distance_metric,
                    filter_expression,
                )
                .await;
            if result.is_ok() {
                Self::record_completed_vector_access(
                    ctx,
                    &storage_url,
                    query_vector.len(),
                    k,
                    filter_expression.is_some(),
                    crate::observability::io_trace::VectorAccessPath::Exact,
                    cache_policy,
                );
            }
            return result;
        }

        let ann_proofs_before = crate::observability::io_trace::vector_ann_proof_count();

        // Determine search strategy based on context
        // Co-design search routing: the collection's index_configs is authoritative.
        // - Empty index_configs → use_axis_indexes=false → co-designed PAX scan
        //   (RaBitQ + SQ8 + A0 coarse-probe from object storage). The segment IS
        //   the index; no in-memory HNSW/IVF is built or queried.
        // - HNSW/IVF in index_configs → use_axis_indexes=true → AXIS in-memory
        //   search (hot/streaming, low-latency path).
        //
        // The global AXIS manager being registered (has_axis_manager=true) does NOT
        // force AXIS on collections that didn't ask for it — that was the pre-fix
        // bug (OR logic → AXIS intercepted every collection → 8.6GB RSS + 1.76%
        // recall for co-design collections whose PAX scan was never exercised).
        let use_orchestration = self.use_orchestrated_search(ctx);

        if use_orchestration {
            debug!("🔍 SST: AXIS manager is available for HNSW/IVF search");
            // TD-112: if the in-memory AXIS index is absent (e.g. after a
            // restart), rebuild it from the durable SST segments before
            // searching, so post-flush recall does not silently degrade to a
            // brute-force segment scan.
            self.ensure_axis_index_from_sst(collection_id, &storage_url)
                .await;
        }

        let result = if use_orchestration {
            // Use advanced orchestration when available
            self.execute_orchestrated_search(
                ctx,
                collection_id,
                &storage_url,
                query_vector,
                k,
                distance_metric,
                filter_expression,
            )
            .await
        } else {
            // Use direct search for simple queries
            self.execute_direct_search(
                ctx,
                collection_id,
                &storage_url,
                query_vector,
                k,
                distance_metric,
                filter_expression,
            )
            .await
        };
        if result.is_ok()
            && crate::observability::io_trace::vector_ann_proof_count() > ann_proofs_before
        {
            Self::record_completed_vector_access(
                ctx,
                &storage_url,
                query_vector.len(),
                k,
                filter_expression.is_some(),
                crate::observability::io_trace::VectorAccessPath::Ann,
                cache_policy,
            );
        }
        result
    }

    /// TD-112: lazily rebuild a collection's AXIS index from its durable SST
    /// segments when the in-memory index is absent.
    ///
    /// AXIS indexes are in-memory; after a restart they are empty and — for HNSW,
    /// or IVF flushed in sub-train-threshold batches — are not persisted, so
    /// post-flush search would silently degrade to a brute-force segment scan.
    /// This reads the flushed records back from the segments and re-indexes them
    /// through the same `handle_flushed_vectors` hook the flush path uses,
    /// covering every index type (unlike the IVF-only persist+cold-load path).
    /// Best-effort and guarded so at most one potentially expensive rebuild runs
    /// for a collection at a time.
    async fn ensure_axis_index_from_sst(&self, collection_id: &str, storage_url: &str) {
        let Some(axis) = self.axis_manager() else {
            return;
        };
        // Already COVERED by the AXIS store (warm / cold-loaded / rebuilt)?
        // Nothing to do. (We key on the store rather than HNSW/IVF presence,
        // since those structures are built lazily and aren't a reliable signal
        // for small collections.)
        //
        // Issue #1126 (L3): a bare `> 0` check here conflated "warm" with
        // "has ANY vectors". WAL replay after a restart seeds the AXIS store
        // with just the replayed records, which skipped this rebuild and made
        // the orchestrated cold search return ONLY that subset — flushed-SST
        // records were invisible to search while GET-by-id still found them.
        // Rebuild whenever the durable segments hold more records than the
        // store; when the durable total is uncountable (unknown format),
        // rebuild anyway — an extra rebuild is cheap, invisible data is not.
        let registered = axis.registered_vector_count(collection_id).await;
        if registered > 0 {
            let covered = match self.discover_sstable_files(storage_url).await {
                Ok(files) if !files.is_empty() => {
                    match self
                        .durable_vector_count_for_rebuild(
                            &files.iter().map(|f| f.url.clone()).collect::<Vec<_>>(),
                        )
                        .await
                    {
                        Some(durable) => registered as u64 >= durable,
                        None => false, // unknown → rebuild
                    }
                }
                // No durable segments (or unlistable): nothing to rebuild from.
                _ => true,
            };
            if covered {
                return;
            }
        }
        // Attempt at most one rebuild per collection concurrently. Attempts are
        // not sticky; the permit drop removes them so a later query can retry
        // after transient filesystem/index errors or explicit AXIS drops.
        let Some(_rebuild_permit) = try_begin_axis_rebuild(collection_id) else {
            return;
        };

        let files = match self.discover_sstable_files(storage_url).await {
            Ok(files) if !files.is_empty() => files,
            _ => return,
        };

        // M1-3 cold-read recall: if EVERY durable segment is a RaBitQ-PAX segment
        // WITHOUT an f32 tier (the default write format), rebuilding AXIS from it
        // would index COARSE RaBitQ-reconstructed vectors (block-format reader.rs:
        // "RaBitQ is a search representation; reconstruction is coarse") → ~0.46
        // recall — WORSE than letting the cold search fall through to the RaBitQ
        // cascade (~0.93, which ranks the RaBitQ codes properly via
        // `try_pax_cascade`). Skip the lossy rebuild; the dispatch fallback
        // reaches the cascade for `.pax` Euclidean/Cosine. Any exact-readable
        // segment (RawF32/SQ8/RaBitQ-with-tier/`.sst`/`.arrow`) breaks the
        // `all()` check and rebuilds normally (exact AXIS). See
        // `pax_segment_is_coarse_rabitq_without_f32_tier`.
        if self
            .all_segments_coarse_rabitq_pax_without_f32_tier(
                &files.iter().map(|f| f.url.clone()).collect::<Vec<_>>(),
            )
            .await
        {
            tracing::info!(
                "M1-3 cold-read: RaBitQ-PAX without f32 tier for '{collection_id}' ({} segments) \
                 — skipping coarse AXIS rebuild; cold reads use the RaBitQ cascade",
                files.len()
            );
            return;
        }

        // Read all records from the durable SST segments via the storage trait's
        // `read_all_records` (UnifiedSSTReader::read_batch). The per-segment
        // `read_segment_records` / `read_all_records_for_compaction` path returned
        // EMPTY for a single ProximaBlocks segment (its `apply_strategy` produced
        // 0 data blocks from the minimal compaction context), which left the AXIS
        // store unrepopulated after an index-store loss (TD-184).
        let records = match self
            .read_all_records(collection_id, Some(storage_url))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    "TD-112 rebuild: read_all_records failed for '{collection_id}': {e}"
                );
                return;
            }
        };
        if records.is_empty() {
            return;
        }

        let count = records.len();
        match axis
            .handle_flushed_vectors(
                collection_id,
                records,
                files.iter().map(|f| f.url.clone()).collect::<Vec<_>>(),
            )
            .await
        {
            Ok(()) => {
                tracing::info!(
                    "TD-112: rebuilt AXIS index for '{collection_id}' from {count} vectors across {} segments",
                    files.len()
                );
            }
            Err(e) => tracing::warn!(
                "TD-112 rebuild: AXIS rebuild-from-SST failed for '{collection_id}': {e}"
            ),
        }
    }

    /// True iff EVERY file in `files` is a `.pax` segment whose `EMBED_BASE` is
    /// RaBitQ-coded with no f32 tier (i.e. no exact data — an AXIS rebuild from
    /// it would be coarse). Used by [`ensure_axis_index_from_sst`] to skip the
    /// lossy rebuild for default RaBitQ-PAX collections so cold reads use the
    /// RaBitQ cascade. A non-`.pax` file, a read error, or any exact-readable
    /// segment (RawF32/SQ8/RaBitQ-with-tier) returns `false` (→ rebuild).
    /// Short-circuits on the first non-coarse segment. `files` is non-empty when
    /// called from the rebuild path.
    /// Issue #1126 (L3): total record count across a collection's durable
    /// segments, for the AXIS rebuild coverage check ONLY. `Some(total)` when
    /// every file is countable (`SST1` header or `.pax` block scan);
    /// `None` when any file is an unknown/uncountable format — the caller then
    /// errs toward rebuilding. Kept separate from the TD-165
    /// `segment_vector_count` gate, whose 0-on-unknown semantics ("fall through
    /// to the approximate path") must not change.
    async fn durable_vector_count_for_rebuild(&self, files: &[String]) -> Option<u64> {
        use crate::storage::engines::sst::SstableHeader;
        let mut total: u64 = 0;
        for file_path in files {
            let fs = self.filesystem().get_filesystem(file_path).ok()?;
            if file_path.ends_with(".pax") {
                // Segments are the flush unit and small relative to query cost;
                // a full read + index parse is the same cost class as the
                // coarse-RaBitQ probe below.
                let bytes = fs.read(file_path).await.ok()?;
                use proximadb_storage_common::pax_block::{PaxSegmentScanner, ScanPredicate};
                let mut scanner =
                    PaxSegmentScanner::from_bytes(bytes, ScanPredicate::default()).ok()?;
                while let Some(block) = scanner.next_block() {
                    total += block.row_count() as u64;
                }
            } else {
                let prefix = fs.read_range(file_path, 0, 8).await.ok()?;
                if prefix.len() < 8 || &prefix[0..4] != b"SST1" {
                    return None; // unknown format (e.g. .arrow) → rebuild
                }
                let header_len =
                    u32::from_le_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]) as u64;
                let header_data = fs.read_range(file_path, 8, header_len).await.ok()?;
                let header = bincode::deserialize::<SstableHeader>(&header_data).ok()?;
                total += header.entry_count;
            }
        }
        Some(total)
    }

    async fn all_segments_coarse_rabitq_pax_without_f32_tier(&self, files: &[String]) -> bool {
        use crate::storage::engines::sst::segment_format::pax_segment_is_coarse_rabitq_without_f32_tier;
        for file in files {
            if !file.ends_with(".pax") {
                return false; // legacy `.sst` / `.arrow` → exact-readable → rebuild
            }
            let Ok(fs) = self.filesystem().get_filesystem(file) else {
                return false; // can't open → don't skip; let the rebuild try
            };
            let Ok(bytes) = fs.read(file).await else {
                return false;
            };
            if !pax_segment_is_coarse_rabitq_without_f32_tier(&bytes) {
                return false; // RawF32/SQ8/RaBitQ-with-tier → rebuild exact
            }
        }
        !files.is_empty()
    }

    /// Read all records from one SST segment, dispatching on the engine's block
    /// format (an engine writes a single format, so no per-file detection).
    async fn read_segment_records(
        &self,
        file: &str,
        format: crate::storage::engines::sst::block_format::BlockFormat,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        use crate::storage::engines::sst::block_format::BlockFormat;
        match format {
            BlockFormat::ArrowBlock => {
                // Cloud URLs download to a scratch file for the path-based reader
                // (defect-6 read class); local paths pass through.
                let seg = crate::storage::engines::sst::staged_write::LocalizedSegment::fetch(
                    &self.filesystem_port(),
                    file,
                )
                .await?;
                let reader = ArrowBlockReader::open(seg.path())
                    .map_err(|e| anyhow::anyhow!("open arrow segment {file}: {e}"))?;
                reader
                    .read_all()
                    .map_err(|e| anyhow::anyhow!("read arrow segment {file}: {e}"))
            }
            BlockFormat::ProximaBlocks => {
                self.sstable_reader()
                    .read_all_records_for_compaction(&[file.to_string()])
                    .await
            }
            BlockFormat::PaxBlock => {
                // P3: read a PAX segment via the mixed-format primitive (magic-detected,
                // reuses PaxSegmentScanner). Best-effort schema keys (empty) for now.
                let bytes = crate::storage::engines::sst::staged_write::read_object_bytes(
                    &self.filesystem_port(),
                    file,
                )
                .await
                .map_err(|e| anyhow::anyhow!("read pax segment {file}: {e}"))?;
                // tenant_ctx None: the AXIS rebuild indexes vectors per-collection and
                // does not consume record.tenant_id. (Deriving the owning tenant from
                // the DrPathBuilder segment path is a follow-up for when the
                // tenant-column-drop flag is enabled end-to-end.)
                crate::storage::engines::sst::segment_format::read_segment_records(
                    &bytes,
                    &[],
                    &[],
                    None,
                )
            }
        }
    }

    /// Execute orchestrated search with intelligent routing
    ///
    /// This method uses the AXIS manager for HNSW/IVF-based approximate search
    /// when available, falling back to direct search otherwise.
    async fn execute_orchestrated_search(
        &self,
        ctx: &StorageQueryContext,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
        filter_expression: Option<&FilterExpression>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        info!("🎯 SST: Using orchestrated search strategy");

        // Check if AXIS manager is available for HNSW/IVF search
        if let Some(axis_manager) = self.axis_manager() {
            info!(
                "🔗 SST: AXIS manager available, attempting HNSW index search for collection {}",
                collection_id
            );

            // Convert filter expression to index metadata filters
            let index_filters = Self::convert_filter_to_index(filter_expression);

            // Build hybrid query for the index trait
            let hybrid_query = IndexHybridQuery {
                collection_id: collection_id.to_string(),
                vector_query: Some(IndexVectorQuery::Dense {
                    vector: query_vector.to_vec(),
                    similarity_threshold: 0.0,
                }),
                metadata_filters: index_filters,
                id_filters: Vec::new(),
                top_k: k,
                include_expired: false,
                // Thread the accuracy-vs-latency knob into the index query so the
                // warm HNSW/IVF path honors `exact`/`approximate`/`approximate:N`
                // (mapped to HNSW `ef` / IVF `nprobe`). `None` ⇒ index default.
                search_effort: ctx
                    .search_params
                    .search_mode
                    .to_search_effort()
                    .map(|effort| match effort {
                        crate::core::search::SearchEffort::Exact => IndexSearchEffort::Exact,
                        crate::core::search::SearchEffort::Approximate { hint } => {
                            IndexSearchEffort::Approximate { hint }
                        }
                    }),
            };

            // Execute AXIS query (HNSW or IVF based on index type).
            //
            // Phase timer surfaces what fraction of the warm-path
            // latency is the actual AXIS walk vs the surrounding
            // dispatch (HybridQuery construction above, result
            // conversion below, outer get-or-create-via-OnceLock).
            // Empirically (10K × 128d cosine) the AXIS walk itself
            // is ~1 ms; the rest is dispatch + conversion overhead.
            let axis_start = std::time::Instant::now();
            match axis_manager.query(hybrid_query).await {
                Ok(axis_results) => {
                    let axis_us = axis_start.elapsed().as_micros() as u64;
                    info!(
                        "✅ SST: AXIS HNSW search completed in {:?} - found {} candidates",
                        std::time::Duration::from_micros(axis_us),
                        axis_results.results.len()
                    );
                    tracing::info!(
                        target: "sst_warm_phase",
                        phase = "axis_query",
                        elapsed_us = axis_us,
                        n_results = axis_results.results.len(),
                        "phase done"
                    );
                    tracing::info!(
                        target: "axis_diag",
                        site = "execute_orchestrated_search.result",
                        n_results = axis_results.results.len(),
                        top1_id = axis_results.results.first().map(|r| r.vector_id.as_str()).unwrap_or(""),
                        top1_raw_similarity = ?axis_results.results.first().map(|r| r.similarity),
                        "AXIS query returned — values are raw ScoredResult.similarity, no further normalization is applied before returning to caller"
                    );

                    // Convert AXIS results to OptimizedSearchRecord
                    let convert_start = std::time::Instant::now();
                    let results: Vec<OptimizedSearchRecord> = axis_results
                        .results
                        .into_iter()
                        .take(k)
                        .map(|scored| OptimizedSearchRecord {
                            id: scored.vector_id.to_string(),
                            vector_id: Some(scored.vector_id.to_string()),
                            score: scored.similarity,
                            similarity: Some(scored.similarity),
                            vector: None, // AXIS doesn't return vectors by default
                            ..Default::default()
                        })
                        .collect();
                    let convert_us = convert_start.elapsed().as_micros() as u64;
                    tracing::info!(
                        target: "sst_warm_phase",
                        phase = "axis_result_convert",
                        elapsed_us = convert_us,
                        n_results = results.len(),
                        "phase done"
                    );

                    // If we need vectors or got fewer results, optionally refine with SST lookup
                    if results.is_empty() {
                        info!("⚠️ SST: AXIS returned no results, falling back to direct search");
                        return self
                            .fallback_to_direct_search(
                                ctx,
                                collection_id,
                                storage_url,
                                query_vector,
                                k,
                                distance_metric,
                                filter_expression,
                                true,
                                true,
                            )
                            .await;
                    }

                    crate::observability::io_trace::record_vector_ann_proof();
                    return Ok(results);
                }
                Err(e) => {
                    warn!(
                        "⚠️ SST: AXIS query failed ({}), falling back to direct search",
                        e
                    );
                }
            }
        } else {
            debug!("🔍 SST: AXIS manager not available, using direct search");
        }

        // Fall back to direct search if AXIS is unavailable or failed
        self.fallback_to_direct_search(
            ctx,
            collection_id,
            storage_url,
            query_vector,
            k,
            distance_metric,
            filter_expression,
            true, // include_vectors
            true, // include_metadata
        )
        .await
    }

    /// Convert FilterExpression to AXIS MetadataFilter format
    fn convert_filter_to_index(
        filter_expression: Option<&FilterExpression>,
    ) -> Vec<IndexMetadataFilter> {
        let Some(filter) = filter_expression else {
            return Vec::new();
        };

        // Convert filter expressions to index metadata filters
        let mut index_filters = Vec::new();

        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                // Convert ComparisonOperator to IndexFilterOperator
                let index_operator = match operator {
                    ComparisonOperator::Equals => IndexFilterOperator::Equals,
                    ComparisonOperator::NotEquals => IndexFilterOperator::NotEquals,
                    ComparisonOperator::GreaterThan => IndexFilterOperator::GreaterThan,
                    ComparisonOperator::GreaterThanOrEqual => {
                        IndexFilterOperator::GreaterThanOrEqual
                    }
                    ComparisonOperator::LessThan => IndexFilterOperator::LessThan,
                    ComparisonOperator::LessThanOrEqual => IndexFilterOperator::LessThanOrEqual,
                    ComparisonOperator::In => IndexFilterOperator::In,
                    ComparisonOperator::NotIn => IndexFilterOperator::NotIn,
                    ComparisonOperator::Contains => IndexFilterOperator::Contains,
                    ComparisonOperator::StartsWith => IndexFilterOperator::StartsWith,
                    ComparisonOperator::EndsWith => IndexFilterOperator::EndsWith,
                    ComparisonOperator::Like => IndexFilterOperator::Like,
                    ComparisonOperator::Between => IndexFilterOperator::Between,
                    _ => {
                        debug!(
                            "Operator {:?} not directly supported by index, will use post-filtering",
                            operator
                        );
                        return index_filters;
                    }
                };

                index_filters.push(IndexMetadataFilter {
                    field: field.clone(),
                    operator: index_operator,
                    value: value.clone(),
                });
            }
            FilterExpression::And(filters) => {
                for f in filters {
                    index_filters.extend(Self::convert_filter_to_index(Some(f)));
                }
            }
            FilterExpression::Or(_) | FilterExpression::Not(_) => {
                // OR and NOT are not directly supported by the index, will use post-filtering
                debug!("OR/NOT filters not supported by index, will use post-filtering");
            }
        }

        index_filters
    }

    /// Execute direct search without orchestration
    async fn execute_direct_search(
        &self,
        ctx: &StorageQueryContext,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
        filter_expression: Option<&FilterExpression>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        info!("🔍 SST: Using direct search strategy");

        self.fallback_to_direct_search(
            ctx,
            collection_id,
            storage_url,
            query_vector,
            k,
            distance_metric,
            filter_expression,
            true, // include_vectors
            true, // include_metadata
        )
        .await
    }

    /// Fallback direct search implementation
    ///
    /// This method implements a simplified but efficient search that:
    /// 1. Discovers relevant SSTable files
    /// 2. Searches each file using the unified reader
    /// 3. Combines and ranks results
    pub async fn fallback_to_direct_search(
        &self,
        ctx: &StorageQueryContext,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
        filter_expression: Option<&FilterExpression>,
        include_vectors: bool,
        include_metadata: bool,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        tracing::debug!(
            "[SST] Starting direct search for collection {}, storage_url: {}",
            collection_id,
            storage_url
        );

        // TD-DELVEC-1 WI-4: capture the scan's snapshot LSN once per direct-scan
        // invocation for merge-on-read deletion-vector filtering in
        // `search_pax_file_exact`. Captured at this convergence point — every route
        // that reaches a `.pax` exact read funnels through here (exact segment scan,
        // orchestrated AXIS fallback, and plain direct search) — so cold deletes are
        // hidden uniformly across all of them. A later capture is strictly correct
        // for slice 1's "hide deletes" goal (a larger LSN surfaces more deletes to
        // the filter). Falls back to u64::MAX (see all deletes) when no freshness
        // source is wired.
        #[cfg(feature = "cold-deletion-vectors")]
        let snapshot_lsn = match self.freshness_lsn_source_ref() {
            Some(src) => src.current_lsn(collection_id).await,
            None => u64::MAX,
        };
        #[cfg(not(feature = "cold-deletion-vectors"))]
        let snapshot_lsn: u64 = u64::MAX;

        let mut all_candidates = Vec::new();

        // Phase 7.2 warm-path profiling: emit per-phase elapsed
        // timings under the dedicated `sst_warm_phase` target so a
        // bench-side tracing layer can aggregate without touching
        // unrelated logs. These calls cost one `Instant::now()` per
        // phase boundary — negligible at search granularity.
        let discovery_start = std::time::Instant::now();

        // Discover SSTable files for this collection with optional centroid pruning
        // When SearchMode is Approximate, uses centroid-based IVF-style optimization
        tracing::debug!(storage_url = %storage_url, "Discovering SSTable files");
        let search_mode = &ctx.search_params.search_mode;

        // **Honor SearchMode::Exact (2026-05-30)**: when the caller
        // asked for an exact search, force `prune_config.force_exact = true`
        // so the sqrt-based centroid block pruning doesn't silently
        // drop recall. Without this override, an Exact search at 100K
        // (where the SST has ≥100 blocks) keeps only `sqrt(num_blocks)`
        // blocks via centroid distance — measured 5% recall vs true
        // brute force on random data. Customers asking for `Exact` get
        // approximate results otherwise.
        //
        // Reconciled + measured 2026-06-28 (TD-096 S1): the 5% was this
        // Exact-pre-override path; with the override, Exact no longer
        // collapses. Approximate recall at scale (sqrt pruning keeps
        // sqrt(num_blocks) blocks — see the block-skip unit tests in
        // block_pruning.rs) is measured in
        // docs/_internal/status/TD_096_S1_RECALL_RECONCILIATION_2026_06_28.adoc.
        //
        // For `SearchMode::Approximate` and `SearchMode::Adaptive` the
        // caller's `block_prune` config flows through unchanged.
        let prune_config_owned;
        let prune_config: &crate::core::search::BlockPruneConfig =
            if matches!(search_mode, crate::core::search::SearchMode::Exact)
                && !ctx.search_params.block_prune.force_exact
            {
                prune_config_owned = crate::core::search::BlockPruneConfig {
                    force_exact: true,
                    ..ctx.search_params.block_prune.clone()
                };
                &prune_config_owned
            } else {
                &ctx.search_params.block_prune
            };
        let sstable_files = self
            .discover_sstable_files_with_centroid_pruning(
                storage_url,
                query_vector,
                distance_metric,
                search_mode,
                prune_config,
            )
            .await?;
        let discovery_us = discovery_start.elapsed().as_micros() as u64;
        tracing::info!(
            target: "sst_warm_phase",
            phase = "discovery",
            elapsed_us = discovery_us,
            file_count = sstable_files.len(),
            "phase done"
        );
        tracing::debug!(
            "[SST] Discovered {} SSTable files (search_mode={:?})",
            sstable_files.len(),
            search_mode
        );
        for (i, file) in sstable_files.iter().enumerate() {
            tracing::trace!(index = i, file = %file.url, "Discovered SSTable file");
        }

        debug!(
            "🔍 SST: Found {} SSTable files for collection {}",
            sstable_files.len(),
            collection_id
        );

        // Search each SSTable file with block-level pruning. Reuse the
        // SearchMode::Exact-aware prune_config from above so the
        // per-file scan also honors `force_exact` when the caller
        // asked for an exact search.
        // TD-SEARCH-2 S2: count this scan in-flight for the lifetime of the
        // per-file work — the adaptive morsel degree divides cores by it.
        let _scan_guard = ScanGuard::enter();
        let scan_start = std::time::Instant::now();

        // TD-SEARCH-2: inter-file parallel search. The degree is config/env-driven:
        //   search_parallel_files = 0 → 50% of CPU cores (wise default)
        //   search_parallel_files = 1 → sequential
        //   search_parallel_files = N → exactly N workers
        // Hot-path override: PROXIMADB_SEARCH_PARALLEL_FILES env takes precedence.
        let file_count = u16::try_from(sstable_files.len()).unwrap_or(u16::MAX);
        let parallel_degree = resolve_search_parallelism(self.config().search_parallel_files)
            .min(file_count)
            .max(1);

        let enable_parallel = parallel_degree > 1;

        if enable_parallel {
            tracing::info!(
                file_count = sstable_files.len(),
                parallel_degree,
                "TD-SEARCH-2: parallel inter-file scan"
            );

            // Build per-file futures (borrow &self — no 'static needed).
            let file_futures = sstable_files.iter().map(|discovered| {
                let sstable_path = discovered.url.as_str();
                let known_size = discovered.size_bytes;
                let filter_owned = filter_expression.cloned();
                async move {
                    let result: Result<Vec<OptimizedSearchRecord>, String> = async {
                        // PAX cascade
                        let pax_cascade: Option<Vec<OptimizedSearchRecord>> =
                            if Self::should_try_pax_cascade(
                                sstable_path,
                                distance_metric,
                                prune_config,
                            ) {
                                match self
                                    .try_pax_cascade(
                                        sstable_path,
                                        query_vector,
                                        filter_expression,
                                        k,
                                        distance_metric,
                                        collection_id,
                                        storage_url,
                                        snapshot_lsn,
                                        known_size,
                                    )
                                    .await
                                {
                                    Ok(Some(records)) => Some(records),
                                    Ok(None) => None,
                                    Err(e) => {
                                        warn!(
                                            file = sstable_path,
                                            error = %e,
                                            "PAX cascade unavailable; falling back to generic scan"
                                        );
                                        None
                                    }
                                }
                            } else {
                                None
                            };

                        let search_result = if let Some(records) = pax_cascade {
                            Ok(records)
                        } else if sstable_path.ends_with(".arrow") {
                            self.search_arrow_file(
                                sstable_path,
                                query_vector,
                                filter_owned.clone(),
                                k,
                                distance_metric,
                            )
                            .await
                        } else if sstable_path.ends_with(".pax") {
                            self.search_pax_file_exact(
                                sstable_path,
                                query_vector,
                                filter_owned.clone(),
                                k,
                                distance_metric,
                                snapshot_lsn,
                                prune_config.force_exact,
                            )
                            .await
                        } else {
                            self.sstable_reader()
                                .search_with_filter_and_pruning(
                                    sstable_path,
                                    query_vector,
                                    filter_owned.clone(),
                                    k,
                                    distance_metric,
                                    Some(&*ctx.collection),
                                    prune_config,
                                )
                                .await
                        };

                        search_result.map_err(|e| format!("{sstable_path}: {e}"))
                    }
                    .await;
                    (sstable_path, result)
                }
            });

            let results = join_all(file_futures).await;
            for (path, result) in results {
                match result {
                    Ok(file_results) => {
                        debug!(
                            file = path,
                            n = file_results.len(),
                            "SST parallel per-file result"
                        );
                        all_candidates.extend(file_results);
                    }
                    Err(e) => {
                        Self::handle_per_file_scan_failure(path, &e, prune_config.force_exact)?
                    }
                }
            }
        } else {
            // Sequential fallback (single file or flag explicitly off)
            for (file_idx, discovered) in sstable_files.iter().enumerate() {
                let sstable_path = discovered.url.as_str();
                let known_size = discovered.size_bytes;
                trace!(
                    "SST: Searching file [{}/{}]: {} (force_exact={})",
                    file_idx + 1,
                    sstable_files.len(),
                    sstable_path,
                    prune_config.force_exact
                );

                // PAX RaBitQ→SQ8 cascade (PAX Phase 2 read-side wiring): try it first
                // for `.pax` segments under a validated metric (Euclidean or Cosine).
                // The generic dispatch below handles every other case — `.arrow`, legacy
                // `.sst`, AND `.pax` under Dot/other metrics or any cascade miss
                // (not-PAX / no RaBitQ / error) — so this is additive and mixed-read-safe.
                let pax_cascade: Option<Vec<OptimizedSearchRecord>> =
                    if Self::should_try_pax_cascade(sstable_path, distance_metric, prune_config) {
                        match self
                            .try_pax_cascade(
                                sstable_path,
                                query_vector,
                                filter_expression,
                                k,
                                distance_metric,
                                collection_id,
                                storage_url,
                                snapshot_lsn,
                                known_size,
                            )
                            .await
                        {
                            Ok(Some(records)) => {
                                debug!(
                                    file = %sstable_path,
                                    n = records.len(),
                                    "SST per-file result source=pax_cascade"
                                );
                                Some(records)
                            }
                            Ok(None) => None,
                            Err(e) => {
                                warn!(
                                    file = %sstable_path,
                                    error = %e,
                                    "PAX cascade unavailable; falling back to generic scan"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                // Dispatch based on file format (Arrow vs ProximaBlocks); the PAX
                // cascade short-circuits above when it applies.
                let search_result = if let Some(records) = pax_cascade {
                    Ok(records)
                } else if sstable_path.ends_with(".arrow") {
                    // Use ArrowBlockReader for Arrow format files
                    self.search_arrow_file(
                        sstable_path,
                        query_vector,
                        filter_expression.cloned(),
                        k, // Use exact k
                        distance_metric,
                    )
                    .await
                } else if sstable_path.ends_with(".pax") {
                    // A `.pax` segment the RaBitQ cascade did not cover (non-L2/Cosine
                    // metric, non-RaBitQ quant, or a cascade miss/error). Exact
                    // materialize-and-rank via the mixed-format reader so `.pax` is
                    // searchable under every metric/quant — this is what makes the PAX
                    // write-default flip safe (otherwise the ProximaBlocks-only
                    // `sstable_reader` below would fail to decode a `.pax` file).
                    self.search_pax_file_exact(
                        sstable_path,
                        query_vector,
                        filter_expression.cloned(),
                        k,
                        distance_metric,
                        snapshot_lsn,
                        prune_config.force_exact,
                    )
                    .await
                } else {
                    // Use SSTable reader for ProximaBlocks format
                    // Choose execution strategy based on flags (TD-041, TD-039, TD-031)
                    let use_parallel_morsels =
                        ctx.search_params.enable_parallel_morsels.unwrap_or(false);
                    let use_vectorized = ctx
                        .search_params
                        .enable_vectorized_execution
                        .unwrap_or(false);
                    let use_pipeline = ctx.search_params.enable_pipeline_execution.unwrap_or(false);

                    if use_pipeline {
                        trace!("SST: Using pipeline-based execution path (TD-031)");
                        self.sstable_reader()
                            .search_with_pipeline_execution(
                                sstable_path,
                                query_vector,
                                filter_expression.cloned(),
                                k, // Use exact k
                                distance_metric,
                                Some(&*ctx.collection),
                                prune_config,
                            )
                            .await
                    } else if use_parallel_morsels {
                        trace!("SST: Using parallel morsel execution path (TD-039)");
                        self.sstable_reader()
                            .search_with_filter_parallel_morsels(
                                sstable_path,
                                query_vector,
                                filter_expression.cloned(),
                                k, // Use exact k
                                distance_metric,
                                Some(&*ctx.collection),
                                prune_config,
                                None, // Use default worker count (CPU cores)
                            )
                            .await
                    } else if use_vectorized {
                        trace!("SST: Using vectorized execution path (TD-041)");
                        self.sstable_reader()
                            .search_with_filter_vectorized(
                                sstable_path,
                                query_vector,
                                filter_expression.cloned(),
                                k, // Use exact k
                                distance_metric,
                                Some(&*ctx.collection),
                                prune_config,
                            )
                            .await
                    } else {
                        trace!("SST: Using scalar execution path");
                        self.sstable_reader()
                            .search_with_filter_and_pruning(
                                sstable_path,
                                query_vector,
                                filter_expression.cloned(),
                                k, // Use exact k
                                distance_metric,
                                Some(&*ctx.collection), // Pass collection for type-safe metadata deserialization
                                prune_config, // Pass block pruning config for Z-order/centroid pruning
                            )
                            .await
                    }
                };

                match search_result {
                    Ok(results) => {
                        debug!(
                            file = %sstable_path,
                            n = results.len(),
                            "SST per-file result (post-dispatch)"
                        );
                        all_candidates.extend(results);
                    }
                    Err(e) => {
                        Self::handle_per_file_scan_failure(
                            sstable_path,
                            &e,
                            prune_config.force_exact,
                        )?;
                    }
                }
            } // end sequential fallback
        } // end if enable_parallel / else

        let scan_us = scan_start.elapsed().as_micros() as u64;
        let candidate_count_before_merge = all_candidates.len();
        tracing::info!(
            target: "sst_warm_phase",
            phase = "per_file_scan",
            elapsed_us = scan_us,
            file_count = sstable_files.len(),
            candidate_count = candidate_count_before_merge,
            "phase done"
        );

        // Use bounded priority queue for efficient top-k selection
        let merge_start = std::time::Instant::now();
        let mut priority_queue = BoundedPriorityQueue::new(k);

        // Insert all candidates into bounded queue
        for candidate in all_candidates {
            priority_queue.try_insert(candidate);
        }

        // Get sorted results from bounded queue
        let mut all_candidates = priority_queue.into_sorted_vec();
        let merge_us = merge_start.elapsed().as_micros() as u64;
        tracing::info!(
            target: "sst_warm_phase",
            phase = "topk_merge",
            elapsed_us = merge_us,
            candidate_count = candidate_count_before_merge,
            "phase done"
        );
        tracing::debug!(candidate_count = all_candidates.len(), "Before filtering");

        // Filter results based on include flags
        let filter_start = std::time::Instant::now();
        self.filter_search_results(&mut all_candidates, include_vectors, include_metadata);
        let filter_us = filter_start.elapsed().as_micros() as u64;
        tracing::info!(
            target: "sst_warm_phase",
            phase = "result_filter",
            elapsed_us = filter_us,
            "phase done"
        );
        tracing::debug!(filtered_count = all_candidates.len(), "After filtering");

        info!(
            "🏁 SST: Direct search completed - Collection: {}, Results: {}/{}",
            collection_id,
            all_candidates.len(),
            k
        );

        Ok(all_candidates)
    }

    /// TD-SEARCH-2 S2: the exact-vs-approximate routing decision, extracted
    /// from `search_vectors_unified` so the multi-core `_arc` entry reuses the
    /// identical logic (no behavior drift between the two entry points).
    async fn want_exact_search(
        &self,
        ctx: &StorageQueryContext,
        query_vector: &[f32],
        storage_url: &str,
    ) -> bool {
        use crate::core::search::SearchMode;
        match &ctx.search_params.search_mode {
            SearchMode::Approximate { .. } => false,
            SearchMode::Exact => true,
            SearchMode::Adaptive { threshold } => {
                let policy = ctx
                    .collection
                    .config
                    .as_ref()
                    .and_then(|c| c.index_policy.as_ref());
                let (byte_budget, pin_exact) = resolve_exact_budget(policy);
                if pin_exact {
                    // Owner pinned exact — always brute-force, any N.
                    true
                } else {
                    let count = self.segment_vector_count(storage_url).await;
                    let dim = query_vector.len().max(1);
                    let scan_bytes = count.saturating_mul(dim).saturating_mul(4);
                    count > 0 && scan_bytes <= byte_budget && count <= *threshold
                }
            }
        }
    }

    /// Emit one successful, physically attributable vector access. The
    /// requested mode remains separate from `actual_path`: Adaptive and
    /// Approximate may fall back to exact, and such ambiguous executions must
    /// never warm an ANN cost cell without an engagement proof.
    fn vector_cache_policy(&self) -> crate::observability::io_trace::VectorCachePolicy {
        crate::observability::io_trace::VectorCachePolicy::from_tiers(
            self.segment_invariants_cache.is_some(),
            self.survivor_cache.is_some(),
        )
    }

    fn record_completed_vector_access(
        ctx: &StorageQueryContext,
        storage_url: &str,
        dimensions: usize,
        top_k: usize,
        has_filter: bool,
        actual_path: crate::observability::io_trace::VectorAccessPath,
        cache_policy: crate::observability::io_trace::VectorCachePolicy,
    ) {
        use crate::core::search::SearchMode;
        use crate::observability::io_trace::{
            VectorAccessTrace, VectorSearchIntent, VectorStorageScope,
        };

        let requested_mode = match &ctx.search_params.search_mode {
            SearchMode::Exact => VectorSearchIntent::Exact,
            SearchMode::Approximate { .. } => VectorSearchIntent::Approximate,
            SearchMode::Adaptive { .. } => VectorSearchIntent::Adaptive,
        };
        crate::observability::io_trace::record_vector_access(VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: dimensions as u64,
            top_k: top_k as u64,
            has_filter,
            requested_mode,
            actual_path,
            storage_scope: VectorStorageScope::from_storage_url(storage_url),
            cache_policy,
        });
    }

    /// TD-SEARCH-2 S2: whether to use the AXIS in-memory orchestration path
    /// (extracted; reused by `_arc`). `use_axis_indexes` is authoritative — a
    /// registered global AXIS manager does NOT force AXIS on co-designed
    /// collections (ADR-070).
    fn use_orchestrated_search(&self, ctx: &StorageQueryContext) -> bool {
        ctx.metadata.use_axis_indexes && self.axis_manager().is_some()
    }

    /// TD-SEARCH-2 S2: per-file scan with the full format dispatch (PAX
    /// cascade → Arrow → exact-PAX → ProximaBlocks reader with
    /// pipeline/morsel/vectorized/scalar). A `&self` helper so the multi-core
    /// `tokio::spawn` path can call it via a cloned `Arc<SstEngine>` (`&*arc`)
    /// — `tokio::spawn` needs `'static`, which a borrowing `&self` closure is not.
    async fn scan_single_file(
        &self,
        sstable_path: &str,
        collection: &proximadb_proto::proximadb_v1::Collection,
        query_vector: &[f32],
        filter_expression: Option<&FilterExpression>,
        k: usize,
        distance_metric: DistanceMetric,
        collection_id: &str,
        storage_url: &str,
        prune_config: &crate::core::search::BlockPruneConfig,
        use_pipeline: bool,
        use_parallel_morsels: bool,
        use_vectorized: bool,
        known_size: Option<u64>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // TD-DELVEC-1 WI-4: capture the scan's snapshot LSN once per single-file
        // scan (the parallel/arc path's per-file entry) for merge-on-read filtering
        // in `search_pax_file_exact`. See `fallback_to_direct_search` for the
        // rationale; u64::MAX (see all deletes) when no freshness source is wired.
        #[cfg(feature = "cold-deletion-vectors")]
        let snapshot_lsn = match self.freshness_lsn_source_ref() {
            Some(src) => src.current_lsn(collection_id).await,
            None => u64::MAX,
        };
        #[cfg(not(feature = "cold-deletion-vectors"))]
        let snapshot_lsn: u64 = u64::MAX;

        // PAX RaBitQ→SQ8 cascade first for `.pax` under a validated metric.
        let pax_cascade: Option<Vec<OptimizedSearchRecord>> =
            if Self::should_try_pax_cascade(sstable_path, distance_metric, prune_config) {
                match self
                    .try_pax_cascade(
                        sstable_path,
                        query_vector,
                        filter_expression,
                        k,
                        distance_metric,
                        collection_id,
                        storage_url,
                        snapshot_lsn,
                        known_size,
                    )
                    .await
                {
                    Ok(Some(records)) => Some(records),
                    Ok(None) => None,
                    Err(e) => {
                        warn!(
                            file = sstable_path,
                            error = %e,
                            "PAX cascade unavailable; falling back to generic scan"
                        );
                        None
                    }
                }
            } else {
                None
            };

        if let Some(records) = pax_cascade {
            return Ok(records);
        }

        // Generic dispatch by extension / execution flag.
        if sstable_path.ends_with(".arrow") {
            self.search_arrow_file(
                sstable_path,
                query_vector,
                filter_expression.cloned(),
                k,
                distance_metric,
            )
            .await
        } else if sstable_path.ends_with(".pax") {
            self.search_pax_file_exact(
                sstable_path,
                query_vector,
                filter_expression.cloned(),
                k,
                distance_metric,
                snapshot_lsn,
                prune_config.force_exact,
            )
            .await
        } else if use_pipeline {
            self.sstable_reader()
                .search_with_pipeline_execution(
                    sstable_path,
                    query_vector,
                    filter_expression.cloned(),
                    k,
                    distance_metric,
                    Some(collection),
                    prune_config,
                )
                .await
        } else if use_parallel_morsels {
            self.sstable_reader()
                .search_with_filter_parallel_morsels(
                    sstable_path,
                    query_vector,
                    filter_expression.cloned(),
                    k,
                    distance_metric,
                    Some(collection),
                    prune_config,
                    None,
                )
                .await
        } else if use_vectorized {
            self.sstable_reader()
                .search_with_filter_vectorized(
                    sstable_path,
                    query_vector,
                    filter_expression.cloned(),
                    k,
                    distance_metric,
                    Some(collection),
                    prune_config,
                )
                .await
        } else {
            self.sstable_reader()
                .search_with_filter_and_pruning(
                    sstable_path,
                    query_vector,
                    filter_expression.cloned(),
                    k,
                    distance_metric,
                    Some(collection),
                    prune_config,
                )
                .await
        }
    }

    /// Central exactness gate for the approximate PAX cascade. Both direct
    /// implementations call this helper so sequential and multi-core scans
    /// cannot drift on the correctness contract.
    fn should_try_pax_cascade(
        sstable_path: &str,
        distance_metric: DistanceMetric,
        prune_config: &crate::core::search::BlockPruneConfig,
    ) -> bool {
        !prune_config.force_exact
            && sstable_path.ends_with(".pax")
            && matches!(
                distance_metric,
                DistanceMetric::Euclidean | DistanceMetric::Cosine | DistanceMetric::DotProduct
            )
    }

    /// Preserve the caller's completeness contract at the per-file boundary.
    /// Approximate scans retain the historical best-effort behavior, but an
    /// exact scan cannot silently omit an unreadable or semantically ineligible
    /// segment and still claim complete results.
    fn handle_per_file_scan_failure(
        sstable_path: &str,
        error: &dyn std::fmt::Display,
        require_complete_scan: bool,
    ) -> Result<()> {
        if require_complete_scan {
            anyhow::bail!("SST exact scan failed for segment '{sstable_path}': {error}");
        }
        warn!(
            file = sstable_path,
            error = %error,
            "SST approximate per-file scan failed (best-effort)"
        );
        Ok(())
    }

    /// TD-SEARCH-2 S2: multi-core direct search. Same semantics as
    /// `fallback_to_direct_search` but each per-file scan runs on its own tokio
    /// worker via `tokio::spawn`, gated by a per-query `Semaphore(degree)`, so a
    /// single query uses `degree` cores for the CPU-bound per-file work. Arc
    /// receiver: each task clones `Arc<SstEngine>` + owned inputs and calls
    /// `scan_single_file` via `&*arc`. Recall-neutral (independent per-file
    /// scans + order-independent `BoundedPriorityQueue` merge). `degree == 1` /
    /// single file stays sequential (no spawn overhead).
    pub async fn fallback_to_direct_search_arc(
        self: std::sync::Arc<Self>,
        ctx: &StorageQueryContext,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
        filter_expression: Option<&FilterExpression>,
        include_vectors: bool,
        include_metadata: bool,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // --- discover (mirrors fallback_to_direct_search) ---
        let search_mode = &ctx.search_params.search_mode;
        let prune_config_owned;
        let prune_config: &crate::core::search::BlockPruneConfig =
            if matches!(search_mode, crate::core::search::SearchMode::Exact)
                && !ctx.search_params.block_prune.force_exact
            {
                prune_config_owned = crate::core::search::BlockPruneConfig {
                    force_exact: true,
                    ..ctx.search_params.block_prune.clone()
                };
                &prune_config_owned
            } else {
                &ctx.search_params.block_prune
            };
        let sstable_files = self
            .discover_sstable_files_with_centroid_pruning(
                storage_url,
                query_vector,
                distance_metric,
                search_mode,
                prune_config,
            )
            .await?;

        let file_count = u16::try_from(sstable_files.len()).unwrap_or(u16::MAX);
        let degree = resolve_search_parallelism(self.config().search_parallel_files)
            .min(file_count)
            .max(1);

        let use_pipeline = ctx.search_params.enable_pipeline_execution.unwrap_or(false);
        let use_parallel_morsels = ctx.search_params.enable_parallel_morsels.unwrap_or(false);
        let use_vectorized = ctx
            .search_params
            .enable_vectorized_execution
            .unwrap_or(false);

        // One clone per query; Arc::clone per task.
        let query_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(query_vector);
        let collection = std::sync::Arc::clone(&ctx.collection);
        let filter_owned = filter_expression.cloned();
        // `tokio::task_local!` does not propagate through the per-file spawns.
        // Capture once and rebind in each child so physical I/O and ANN proof
        // counters remain part of this query's measured cost.
        let trace_handle = crate::observability::io_trace::current_handle();

        let mut all_candidates = Vec::new();

        if degree > 1 {
            tracing::info!(
                file_count = sstable_files.len(),
                parallel_degree = degree,
                "TD-SEARCH-2 S2: multi-core (tokio::spawn) inter-file scan"
            );
            // Cap in-flight scans at `degree` so file_count >> cores does not
            // oversubscribe; the runtime worker pool backpressures cross-query.
            let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(degree as usize));
            let mut handles: Vec<(
                String,
                tokio::task::JoinHandle<Result<Vec<OptimizedSearchRecord>>>,
            )> = Vec::with_capacity(sstable_files.len());
            for discovered in sstable_files {
                let DiscoveredSstFile {
                    url: path,
                    size_bytes: known_size,
                } = discovered;
                // Gate concurrency: await a permit before spawning (bound at degree).
                let permit = std::sync::Arc::clone(&sem).acquire_owned().await?;
                let engine = std::sync::Arc::clone(&self);
                let collection = std::sync::Arc::clone(&collection);
                let query = std::sync::Arc::clone(&query_arc);
                let filter = filter_owned.clone();
                let cid = collection_id.to_string();
                let url = storage_url.to_string();
                let prune = prune_config.clone();
                let trace = trace_handle.clone();
                let scan_path = path.clone();
                handles.push((
                    path,
                    tokio::spawn(async move {
                        let _permit = permit; // released on task drop
                        // `engine` is `Arc<SstEngine>`; method-call auto-derefs to `&self`.
                        let scan = engine.scan_single_file(
                            &scan_path,
                            &collection,
                            &query[..],
                            filter.as_ref(),
                            k,
                            distance_metric,
                            &cid,
                            &url,
                            &prune,
                            use_pipeline,
                            use_parallel_morsels,
                            use_vectorized,
                            known_size,
                        );
                        if let Some(trace) = trace {
                            crate::observability::io_trace::scope_with_handle(trace, scan).await
                        } else {
                            scan.await
                        }
                    }),
                ));
            }
            // Approximate scans remain best-effort on per-file I/O errors.
            // Exact scans fail closed because omitting one segment violates the
            // completeness contract. JoinError is always fatal: a panic is a
            // logic bug, not transient I/O.
            for (path, h) in handles {
                match h.await {
                    Ok(Ok(recs)) => all_candidates.extend(recs),
                    Ok(Err(e)) => {
                        Self::handle_per_file_scan_failure(&path, &e, prune_config.force_exact)?
                    }
                    Err(join_err) => {
                        return Err(anyhow::anyhow!(
                            "SST S2: per-file scan task panicked: {join_err}"
                        ));
                    }
                }
            }
        } else {
            // degree == 1 / single file: sequential, no spawn overhead.
            for discovered in &sstable_files {
                match self
                    .scan_single_file(
                        &discovered.url,
                        &collection,
                        &query_arc[..],
                        filter_owned.as_ref(),
                        k,
                        distance_metric,
                        collection_id,
                        storage_url,
                        prune_config,
                        use_pipeline,
                        use_parallel_morsels,
                        use_vectorized,
                        discovered.size_bytes,
                    )
                    .await
                {
                    Ok(recs) => all_candidates.extend(recs),
                    Err(e) => Self::handle_per_file_scan_failure(
                        &discovered.url,
                        &e,
                        prune_config.force_exact,
                    )?,
                }
            }
        }

        // --- merge + finalize (mirrors fallback_to_direct_search) ---
        let mut priority_queue = BoundedPriorityQueue::new(k);
        for candidate in all_candidates {
            priority_queue.try_insert(candidate);
        }
        let mut all_candidates = priority_queue.into_sorted_vec();
        self.filter_search_results(&mut all_candidates, include_vectors, include_metadata);
        Ok(all_candidates)
    }

    /// TD-SEARCH-2 S2: Arc-receiver production entry. Same routing as
    /// `search_vectors_unified` (reuses `want_exact_search` +
    /// `use_orchestrated_search`), but the direct-search branch dispatches to
    /// the multi-core `fallback_to_direct_search_arc`. Exact + orchestrated
    /// branches delegate to the existing `&self` methods (no spawn needed:
    /// exact is a single segment; orchestrated is in-memory AXIS).
    pub async fn search_vectors_unified_arc(
        self: std::sync::Arc<Self>,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let _compute_guard = ComputeMsGuard::new("sst");
        let _in_flight_guard = InFlightSearchGuard::acquire(); // S2b: adaptive degree

        if let Some(orch) = self.orchestrator() {
            (**orch).pattern_tracker().track_access_async(
                format!("{}::sst::metadata", ctx.collection_id()),
                crate::storage::cache::orchestrator::CacheType::Metadata,
            );
        }

        let collection_id = ctx.collection_id();
        let storage_url = ctx
            .collection_storage_path()
            .ok_or_else(|| SstError::InvalidArgument("No storage URL in context".into()))?;
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| SstError::InvalidArgument("No query vector in context".into()))?;
        let k = ctx.top_k();
        let distance_metric = ctx.distance_metric();
        let filter_expression = ctx.search_params.filter_expression.as_ref();
        let cache_policy = self.vector_cache_policy();

        if self
            .want_exact_search(ctx, query_vector, &storage_url)
            .await
        {
            let result = self
                .execute_exact_segment_scan(
                    ctx,
                    collection_id,
                    &storage_url,
                    query_vector,
                    k,
                    distance_metric,
                    filter_expression,
                )
                .await;
            if result.is_ok() {
                Self::record_completed_vector_access(
                    ctx,
                    &storage_url,
                    query_vector.len(),
                    k,
                    filter_expression.is_some(),
                    crate::observability::io_trace::VectorAccessPath::Exact,
                    cache_policy,
                );
            }
            return result;
        }

        let ann_proofs_before = crate::observability::io_trace::vector_ann_proof_count();

        let use_orchestration = self.use_orchestrated_search(ctx);
        if use_orchestration {
            self.ensure_axis_index_from_sst(collection_id, &storage_url)
                .await;
        }
        let result = if use_orchestration {
            self.execute_orchestrated_search(
                ctx,
                collection_id,
                &storage_url,
                query_vector,
                k,
                distance_metric,
                filter_expression,
            )
            .await
        } else {
            // Multi-core direct search (S2).
            self.fallback_to_direct_search_arc(
                ctx,
                collection_id,
                &storage_url,
                query_vector,
                k,
                distance_metric,
                filter_expression,
                true,
                true,
            )
            .await
        };
        if result.is_ok()
            && crate::observability::io_trace::vector_ann_proof_count() > ann_proofs_before
        {
            Self::record_completed_vector_access(
                ctx,
                &storage_url,
                query_vector.len(),
                k,
                filter_expression.is_some(),
                crate::observability::io_trace::VectorAccessPath::Ann,
                cache_policy,
            );
        }
        result
    }

    /// Discover SSTable files with optional centroid-based pruning (LanceDB-inspired IVF optimization)
    ///
    /// When `search_mode` is Approximate, this method:
    /// 1. Loads headers from all SST files to get centroids
    /// 2. Computes distance from query to each centroid
    /// 3. Returns only the top nprobe files (closest centroids to query)
    /// 4. This can skip 80-90% of files for large datasets
    async fn discover_sstable_files_with_centroid_pruning(
        &self,
        storage_url: &str,
        query_vector: &[f32],
        distance_metric: proximadb_distance_kernel::DistanceMetric,
        search_mode: &crate::core::search::SearchMode,
        prune_config: &crate::core::search::BlockPruneConfig, // [AGENT_FIX] New parameter
    ) -> Result<Vec<DiscoveredSstFile>> {
        use crate::core::search::SearchMode;

        // First get all files
        let all_files = self.discover_sstable_files(storage_url).await?;

        // [AGENT_FIX] Use the configured min_keep value instead of a hardcoded number.
        let min_keep = prune_config.min_keep.max(1);

        // OPTIMIZATION: Skip file-level pruning for small datasets where overhead exceeds benefit.
        // Loading centroids from each file header and computing distances has significant I/O
        // and CPU overhead. Only worth it when we have many files to prune.
        use crate::storage::engines::core::constants::pruning;
        if all_files.len() < pruning::MIN_FILES_FOR_PRUNING {
            tracing::debug!(
                "SST file pruning skipped: {} files < {} threshold (overhead would exceed benefit)",
                all_files.len(),
                pruning::MIN_FILES_FOR_PRUNING
            );
            return Ok(all_files);
        }

        // For exact mode or small datasets (<= min_keep), search all files
        if matches!(search_mode, SearchMode::Exact) || all_files.len() <= min_keep {
            if !matches!(search_mode, SearchMode::Exact) {
                tracing::warn!(
                    "Fewer than or equal to `min_keep` ({}) SST files, forcing exact search.",
                    min_keep
                );
            }
            return Ok(all_files);
        }

        // Only Approximate and large-dataset Adaptive searches should use centroid pruning.
        if !matches!(
            search_mode,
            SearchMode::Approximate { .. } | SearchMode::Adaptive { .. }
        ) {
            tracing::warn!(
                "Search mode is neither 'Approximate' nor 'Adaptive', forcing exact search by returning all files."
            );
            return Ok(all_files);
        }

        if let SearchMode::Adaptive { threshold } = search_mode {
            let estimated_dataset_size = all_files.len() * 1000;
            if estimated_dataset_size < *threshold {
                tracing::debug!(
                    "Adaptive SST pruning skipped: estimated dataset size {} < threshold {}",
                    estimated_dataset_size,
                    threshold
                );
                return Ok(all_files);
            }
        }

        // Calculate effective nprobe based on search mode and number of files
        let nprobe = search_mode.effective_nprobe(all_files.len(), all_files.len() * 1000); // Estimate 1000 vectors per file

        // If nprobe >= number of files, search all
        if nprobe >= all_files.len() {
            return Ok(all_files);
        }

        // Load headers and compute centroid distances
        let mut file_distances: Vec<(String, f32)> = Vec::new();

        for file_path in &all_files {
            match self.load_sst_header_centroid(&file_path.url).await {
                Ok(Some((centroid, _max_distance_to_centroid))) => {
                    if centroid.len() == query_vector.len() {
                        // Compute distance from query to file centroid
                        let distance = self.compute_centroid_distance(
                            query_vector,
                            &centroid,
                            distance_metric,
                        );
                        file_distances.push((file_path.url.clone(), distance));
                    } else {
                        // Dimension mismatch - include file anyway
                        file_distances.push((file_path.url.clone(), 0.0));
                    }
                }
                Ok(None) => {
                    // No centroid - include file anyway (for backwards compatibility)
                    file_distances.push((file_path.url.clone(), 0.0));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load centroid from {}: {}, including anyway",
                        file_path.url,
                        e
                    );
                    file_distances.push((file_path.url.clone(), 0.0));
                }
            }
        }

        // Sort by distance (ascending - closest first for similarity search)
        file_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top nprobe files (carrying the list-time sizes through).
        let selected_urls: std::collections::HashSet<&str> = file_distances
            .iter()
            .take(nprobe)
            .map(|(path, _)| path.as_str())
            .collect();
        let selected_files: Vec<DiscoveredSstFile> = all_files
            .iter()
            .filter(|f| selected_urls.contains(f.url.as_str()))
            .cloned()
            .collect();

        if selected_files.len() < all_files.len() {
            crate::observability::io_trace::record_vector_ann_proof();
        }

        debug!(
            "🎯 SST Centroid pruning: selected {}/{} files (nprobe={})",
            selected_files.len(),
            all_files.len(),
            nprobe
        );

        Ok(selected_files)
    }

    /// Load centroid from SST header for partition-aware search
    async fn load_sst_header_centroid(&self, file_path: &str) -> Result<Option<(Vec<f32>, f32)>> {
        use crate::storage::engines::sst::SstableHeader;

        // Discovery already classifies current PAX segments by their durable
        // `.pax` suffix. PAX has its partitioning model in Region A0, not in a
        // legacy SST1 header centroid, so probing its first 8 bytes can only
        // return `None`. Avoid one paid ranged GET per PAX segment per query.
        // Unknown and legacy paths still take the magic-sniff path below,
        // preserving mixed-format reads.
        if file_path.ends_with(".pax") {
            return Ok(None);
        }

        let fs = self.filesystem().get_filesystem(file_path)?;

        // Read just the first part of the file to get header
        // Format: SST1 (4 bytes) + header_len (4 bytes) + header data
        let header_prefix = fs.read_range(file_path, 0, 8).await?;

        // Verify magic — a coalesced segment (PXH1, ADR-065) has no legacy SST1
        // centroid; return None cleanly (no error) so partition-aware search skips it.
        if &header_prefix[0..4] != b"SST1" {
            return Ok(None);
        }

        let header_len = u32::from_le_bytes([
            header_prefix[4],
            header_prefix[5],
            header_prefix[6],
            header_prefix[7],
        ]) as usize;

        // Read header data
        let header_data = fs.read_range(file_path, 8, header_len as u64).await?;

        // Deserialize header
        let header: SstableHeader = bincode::deserialize(&header_data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize SST header: {}", e))?;

        // Return centroid and max_distance if available
        if let Some(centroid) = header.centroid {
            let max_dist = header.max_distance_to_centroid.unwrap_or(f32::MAX);
            Ok(Some((centroid, max_dist)))
        } else {
            Ok(None)
        }
    }

    /// Compute distance from query to centroid
    fn compute_centroid_distance(
        &self,
        query: &[f32],
        centroid: &[f32],
        metric: proximadb_distance_kernel::DistanceMetric,
    ) -> f32 {
        use proximadb_distance_kernel::DistanceMetric;

        match metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0f32;
                for (q_val, c_val) in query.iter().zip(centroid.iter()) {
                    let diff = q_val - c_val;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            DistanceMetric::Cosine | DistanceMetric::DotProduct => {
                // For cosine/IP, we want to maximize similarity
                // Return 1 - cosine_similarity as "distance"
                let mut dot = 0.0f32;
                let mut norm_q = 0.0f32;
                let mut norm_c = 0.0f32;
                for (q_val, c_val) in query.iter().zip(centroid.iter()) {
                    dot += q_val * c_val;
                    norm_q += q_val * q_val;
                    norm_c += c_val * c_val;
                }
                let denom = (norm_q * norm_c).sqrt();
                if denom > 0.0 {
                    1.0 - (dot / denom)
                } else {
                    1.0
                }
            }
            _ => {
                // Default to Euclidean for other metrics
                let mut sum = 0.0f32;
                for (q_val, c_val) in query.iter().zip(centroid.iter()) {
                    let diff = q_val - c_val;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
        }
    }

    /// Discover SSTable files for a collection.
    ///
    /// `pub(crate)` so the flush path can reuse it for the L0 compaction
    /// trigger (TD-114) without duplicating segment discovery.
    pub(crate) async fn discover_sstable_files(
        &self,
        storage_url: &str,
    ) -> Result<Vec<DiscoveredSstFile>> {
        tracing::debug!(
            "[SST] discover_sstable_files called with storage_url: {}",
            storage_url
        );

        let mut files = Vec::new();

        // storage_url is already the correct data directory path from collection_storage_path()
        // No need to parse and reconstruct - use it directly
        let data_url = storage_url;

        // List files in the collection directory
        let fs = self.filesystem().get_filesystem(data_url)?;
        tracing::debug!("[SST] Got filesystem for data_url: {}", data_url);

        // Handle case where directory doesn't exist yet (e.g., before first flush)
        let entries = match fs.list(data_url).await {
            Ok(entries) => {
                tracing::debug!("[SST] Found {} entries in {}", entries.len(), data_url);
                entries
            }
            Err(e) if e.to_string().contains("No such file or directory") => {
                tracing::warn!("[SST] Directory doesn't exist yet: {}", data_url);
                return Ok(files);
            }
            Err(e) => {
                tracing::error!("[SST] Failed to list directory {}: {:?}", data_url, e);
                return Err(anyhow::anyhow!(
                    "Failed to list directory {}: {}",
                    data_url,
                    e
                ));
            }
        };

        // TD-COMPACT-13: the data dir is a reconciler-sweep target (its
        // retire-pending obligations drain on the worker), and any input whose
        // delete failed is filtered from the live set here — superseded
        // segments stop serving the moment the obligation is recorded.
        crate::storage::engines::sst::retirement_ledger::register_collection_dir(
            &std::path::PathBuf::from(data_url),
        );
        let pending = crate::storage::engines::sst::retirement_ledger::pending_paths();
        for entry in entries {
            tracing::trace!(
                "[SST] Examining entry: name={}, url={}, is_dir={}",
                entry.name,
                entry.url,
                entry.metadata.is_directory
            );
            if !entry.metadata.is_directory && pending.contains(&entry.url) {
                tracing::debug!(
                    url = %entry.url,
                    "TD-COMPACT-13: filtering pending-retirement segment from discovery"
                );
                continue;
            }
            if !entry.metadata.is_directory
                && (entry.name.ends_with(".sst")
                    || entry.name.ends_with(".arrow")
                    || entry.name.ends_with(".pax"))
            {
                // TD-RDSTRAT-13 C3: carry the list-time object size so the
                // search path can skip the per-segment HEAD. `0` means the
                // backend did not report a size — record `None` and let the
                // reader fall back to a real stat; never guess.
                files.push(DiscoveredSstFile {
                    size_bytes: (entry.metadata.size > 0).then_some(entry.metadata.size),
                    url: entry.url,
                });
                tracing::debug!("[SST] Found data file: {}", entry.name);
            }
        }

        tracing::debug!(
            "[SST] Discovered {} .sst files in {}",
            files.len(),
            data_url
        );
        Ok(files)
    }

    /// Parse storage URL to extract base URL and collection ID
    #[allow(dead_code)]
    #[allow(dead_code)]
    fn parse_storage_url(&self, storage_url: &str) -> Result<(String, String)> {
        // Fallback: assume storage_url is base_url/collection_id format
        if let Some(last_slash) = storage_url.rfind('/') {
            let base = &storage_url[..last_slash];
            let collection = &storage_url[last_slash + 1..];
            Ok((base.to_string(), collection.to_string()))
        } else {
            Err(
                SstError::InvalidArgument(format!("Invalid storage URL format: {}", storage_url))
                    .into(),
            )
        }
    }

    /// Filter search results based on include flags
    #[expect(clippy::ptr_arg)] // Accepting &mut Vec for API compatibility
    fn filter_search_results(
        &self,
        results: &mut Vec<OptimizedSearchRecord>,
        include_vectors: bool,
        include_metadata: bool,
    ) {
        if !include_vectors {
            for result in results.iter_mut() {
                result.vector = None;
            }
        }

        if !include_metadata {
            for result in results.iter_mut() {
                result.metadata = HashMap::new();
            }
        }
    }

    /// List SSTable files for search in a specific directory
    pub async fn list_sstable_files_for_search(&self, data_dir: &str) -> Result<Vec<String>> {
        let mut sstable_files = Vec::new();

        // Use filesystem to list files directly
        if let Ok(mut entries) = tokio::fs::read_dir(data_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                if let Some(name) = entry.file_name().to_str()
                    && (name.ends_with(".sst")
                        || name.ends_with(".arrow")
                        || name.ends_with(".pax"))
                {
                    sstable_files.push(format!("{}/{}", data_dir, name));
                }
            }
        }

        debug!(
            "📋 Listed {} SSTable files in {}",
            sstable_files.len(),
            data_dir
        );
        Ok(sstable_files)
    }

    /// Search within an Arrow format file
    ///
    /// Uses ArrowBlockReader to read and search through Arrow IPC files,
    /// providing the same interface as SSTable searches for seamless integration.
    async fn search_arrow_file(
        &self,
        arrow_path: &str,
        query_vector: &[f32],
        _filter_expression: Option<FilterExpression>,
        limit: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        use proximadb_distance_kernel::engine::{SimilarityResult, UnifiedDistanceCompute};
        use std::sync::Arc;

        debug!("🏹 Searching Arrow file: {}", arrow_path);

        // Cloud URLs download to a scratch file for the path-based reader
        // (defect-6 read class); local paths pass through.
        let seg = crate::storage::engines::sst::staged_write::LocalizedSegment::fetch(
            &self.filesystem_port(),
            arrow_path,
        )
        .await?;

        // Open the Arrow file reader
        let reader = ArrowBlockReader::open(seg.path())
            .map_err(|e| anyhow::anyhow!("Failed to open Arrow file {}: {}", arrow_path, e))?;

        // Read all records from the Arrow file
        let records = reader
            .read_all()
            .map_err(|e| anyhow::anyhow!("Failed to read Arrow file {}: {}", arrow_path, e))?;

        trace!("🏹 Arrow file contains {} records", records.len());

        // Create distance computer with the specified metric
        let distance_computer = UnifiedDistanceCompute::new(distance_metric);

        // Score all records
        // Note: Metadata filtering for Arrow files is simplified - for full filter support,
        // use ProximaBlocks format which has optimized filter evaluation
        let mut candidates: Vec<OptimizedSearchRecord> =
            Vec::with_capacity(records.len().min(limit));

        for record in &records {
            let Some(embedding) = record.embeddings.first() else {
                continue;
            };
            if embedding.values.is_empty() {
                continue;
            }

            // Compute raw distance
            let raw_distance = distance_computer.distance(query_vector, &embedding.as_fp32_cow());

            // Use SimilarityResult to get normalized_score (higher = more similar)
            // This ensures consistency with the rest of the codebase and BoundedPriorityQueue
            let similarity_result = SimilarityResult::new(raw_distance, distance_metric);

            // OptimizedSearchRecord uses canonical ProximaValue metadata internally.
            let metadata = proximadb_records::conversions::proxima_tree_to_value_map(&record.props);

            candidates.push(OptimizedSearchRecord {
                id: record.oid.clone(),
                vector_id: Some(record.oid.clone()),
                score: similarity_result.normalized_score, // Use normalized_score (higher = better)
                similarity: Some(similarity_result.normalized_score),
                vector: Some(Arc::new(embedding.values.to_fp32_owned())),
                metadata,
                ..Default::default()
            });
        }

        // Sort by score descending (higher normalized_score = more similar = better)
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);

        debug!("🏹 Arrow search found {} candidates", candidates.len());
        Ok(candidates)
    }

    /// Exact materialize-and-rank search over a `.pax` segment that the RaBitQ
    /// cascade did not cover — a non-L2/Cosine metric (Dot / inner-product /
    /// exotic), a non-RaBitQ quant (RawF32 / SQ8), or a cascade miss/error. The
    /// segment is decoded back to `ProximaRecord`s via the mixed-format reader
    /// (`segment_format::read_segment_records`, magic-detected) and ranked by
    /// exhaustive distance for the requested metric, so a `.pax` file is searchable
    /// under EVERY metric and quant for approximate/adaptive callers. An explicit
    /// exact request additionally requires raw-f32 authority and fails loudly
    /// before ranking when the segment is lossy. This is the property that makes the PAX
    /// write-default flip safe: the L2/Cosine fast path still takes the RaBitQ
    /// cascade; everything else falls here instead of hitting the
    /// ProximaBlocks-only `sstable_reader` (which cannot decode `.pax`). Recall
    /// is exact for `RawF32` or an exact-f32 tier; lossy `RaBitQ`/`SQ8` values
    /// remain valid only for callers that accepted approximate semantics.
    #[cfg_attr(not(feature = "cold-deletion-vectors"), allow(unused_variables))]
    async fn search_pax_file_exact(
        &self,
        pax_path: &str,
        query_vector: &[f32],
        filter_expression: Option<FilterExpression>,
        limit: usize,
        distance_metric: DistanceMetric,
        snapshot_lsn: u64,
        require_exact_authority: bool,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        use proximadb_distance_kernel::engine::{SimilarityResult, UnifiedDistanceCompute};
        use std::sync::Arc;

        debug!("📦 Exact PAX scan: {}", pax_path);

        let bytes = crate::storage::engines::sst::staged_write::read_object_bytes(
            &self.filesystem_port(),
            pax_path,
        )
        .await
        .map_err(|e| anyhow::anyhow!("read pax segment {pax_path}: {e}"))?;
        if require_exact_authority
            && !crate::storage::engines::sst::segment_format::pax_segment_has_exact_vector_authority(
                &bytes,
            )
        {
            anyhow::bail!(
                "exact vector search requires authoritative raw-f32 values, but PAX segment \
                 '{pax_path}' is lossy; enable the collection's pax_f32_tier or use \
                 SearchMode::Approximate"
            );
        }
        let records = crate::storage::engines::sst::segment_format::read_segment_records(
            &bytes,
            &[],
            &[],
            None,
        )
        .map_err(|e| anyhow::anyhow!("decode pax segment {pax_path}: {e}"))?;
        trace!(
            "📦 PAX segment {} decoded to {} records",
            pax_path,
            records.len()
        );

        // TD-DELVEC-1 WI-4: warm this segment's deletion vector (lazy-load the
        // `.dv`) so merge-on-read can skip deleted positions below. Best-effort —
        // a load failure just means no skipping for this segment (the tombstone
        // still provides read-coherence via is_record_dead until the DV is the
        // sole mechanism).
        #[cfg(feature = "cold-deletion-vectors")]
        if let Some(dv) = &self.deletion_vector_store {
            let _ = dv.load(pax_path).await;
        }

        let distance_computer = UnifiedDistanceCompute::new(distance_metric);
        let mut candidates: Vec<OptimizedSearchRecord> =
            Vec::with_capacity(records.len().min(limit));

        for (pos, record) in records.iter().enumerate() {
            // TD-DELVEC-1 WI-4: merge-on-read — skip positions whose deletion
            // vector bit is set as of the scan's snapshot LSN (captured by the
            // caller at the direct-scan convergence point). A cold delete is
            // invisible here.
            #[cfg(feature = "cold-deletion-vectors")]
            if let Some(dv) = &self.deletion_vector_store
                && dv
                    .is_deleted_as_of(pax_path, pos as u32, snapshot_lsn)
                    .await
            {
                continue;
            }

            // Apply the metadata filter at record level (canonical props), mirroring
            // the ProximaBlocks scan path — PAX is the default format, so filter
            // support here is required, not optional (the Arrow path skips it).
            if let Some(filter_expr) = &filter_expression
                && !crate::core::search::sql_value_filter::evaluate_filter_proxima(
                    filter_expr,
                    &record.props,
                )
            {
                continue;
            }

            let Some(embedding) = record.embeddings.first() else {
                continue;
            };
            if embedding.values.is_empty() {
                continue;
            }

            let raw_distance = distance_computer.distance(query_vector, &embedding.as_fp32_cow());
            let similarity_result = SimilarityResult::new(raw_distance, distance_metric);
            let metadata = proximadb_records::conversions::proxima_tree_to_value_map(&record.props);

            candidates.push(OptimizedSearchRecord {
                id: record.oid.clone(),
                vector_id: Some(record.oid.clone()),
                score: similarity_result.normalized_score,
                similarity: Some(similarity_result.normalized_score),
                vector: Some(Arc::new(embedding.values.to_fp32_owned())),
                metadata,
                ..Default::default()
            });
        }

        // Sort by score descending (higher normalized_score = more similar = better).
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);

        debug!("📦 Exact PAX scan found {} candidates", candidates.len());
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_data_model::ProximaValue;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

    /// TD-SEARCH-2 S2b: the adaptive degree formula divides the base degree
    /// by the in-flight query count, floored at 1 (never 0).
    #[test]
    fn s2b_adaptive_degree_divides_by_in_flight() {
        assert_eq!(adaptive_degree(4, 1), 4, "1 in-flight: full degree");
        assert_eq!(adaptive_degree(4, 2), 2, "2 in-flight: halved");
        assert_eq!(adaptive_degree(4, 4), 1, "4 in-flight: quartered");
        assert_eq!(adaptive_degree(4, 8), 1, "8 in-flight: floored at 1");
        assert_eq!(adaptive_degree(4, 0), 4, "0 in-flight: treated as 1");
    }

    /// TD-SEARCH-2 S2: multi-core (`tokio::spawn`) inter-file search is
    /// recall-neutral vs sequential. Flushes 3 batches → 3 files (no compaction),
    /// then runs the SAME query through `search_vectors_unified_arc` at degree=1
    /// (sequential) and degree=4 (multi-core spawn) and asserts the returned
    /// id-sets agree. Process-per-test isolation (nextest) makes the env-set safe.
    #[tokio::test]
    async fn s2_multicore_arc_matches_sequential() {
        use crate::core::search::SearchParams;
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, StorageAssignment, StorageConfig,
        };
        use crate::storage::persistence::filesystem::FilesystemConfig;
        use crate::storage::traits::{FlushParameters, StorageQueryContext, StorageQueryMetadata};
        use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path().to_str().unwrap().to_string();
        std::mem::forget(temp_dir); // keep dir for the engine's lifetime
        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(format!("file://{}", base));
        let filesystem = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());

        let mut sst_config = SstConfig::default();
        sst_config.block_format = "ArrowBlock".to_string();
        sst_config.compaction_threshold = 100; // keep 3 files (no compaction) → degree > 1
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());
        let engine = SstEngine::new_with_config(sst_config, filesystem, distance_compute)
            .await
            .unwrap();

        let dim = 64usize;
        let cid = "s2_multicore_test";
        let collection = Collection {
            id: cid.to_string(),
            config: Some(CollectionConfig {
                name: cid.to_string(),
                dimension: dim as u32,
                storage_config: Some(StorageConfig::default()),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                primary_path: base.clone(),
                base_location: base.clone(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mk_rec = |idx: usize| {
            let id = format!("b_{}", idx);
            let ts_ns = (idx as i64).saturating_mul(1_000_000);
            ProximaRecord {
                oid: id.clone(),
                local_id: Some(id),
                created_at_ns: ts_ns,
                updated_at_ns: ts_ns,
                record_version: 1,
                embeddings: vec![EmbeddingCell {
                    model_id: "test".to_string(),
                    modality: "dense_vector".to_string(),
                    dim: dim as u32,
                    values: EmbeddingValues::Fp32(
                        (0..dim)
                            .map(|j| ((idx as f32) * 0.1 + (j as f32) * 0.01).sin())
                            .collect(),
                    ),
                    ..Default::default()
                }],
                ..ProximaRecord::default()
            }
        };

        // Flush 3 batches → 3 files.
        for b in 0..3u32 {
            let start = (b as usize) * 25;
            let recs: Vec<ProximaRecord> = (start..start + 25).map(mk_rec).collect();
            let params = FlushParameters {
                collection_id: Some(cid.to_string()),
                vector_records: recs,
                force: true,
                synchronous: true,
                collection_config: Some(collection.clone()),
                ..Default::default()
            };
            let r = engine.do_flush(&params).await.unwrap();
            assert!(r.success, "flush batch {} failed", b);
        }

        // Query vector = idx 0's pattern (guaranteed present); SearchParams.vector is Vec<f32>.
        let query: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.01).sin()).collect();
        let mk_ctx = || StorageQueryContext {
            search_params: Arc::new(SearchParams {
                vector: Some(query.clone()),
                top_k: Some(10),
                filters: None,
                filter_expression: None,
                ..Default::default()
            }),
            collection: Arc::new(collection.clone()),
            metadata: StorageQueryMetadata {
                collection_id: cid.to_string(),
                ..Default::default()
            },
            user_context: None,
            tenant_context: None,
        };

        let engine_arc = Arc::new(engine);

        // degree=1 (sequential — exercises fallback_to_direct_search_arc's else branch).
        // SAFETY: nextest runs each test in its own process; no other thread is
        // reading this env var concurrently. Sets the inter-file parallel degree.
        unsafe { std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "1") };
        let ids_seq: std::collections::HashSet<String> = engine_arc
            .clone()
            .search_vectors_unified_arc(&mk_ctx())
            .await
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();

        // degree=4 (multi-core — exercises the tokio::spawn + Semaphore path).
        // SAFETY: same as above — process-per-test isolation.
        unsafe { std::env::set_var("PROXIMADB_SEARCH_PARALLEL_FILES", "4") };
        let ids_par: std::collections::HashSet<String> = engine_arc
            .search_vectors_unified_arc(&mk_ctx())
            .await
            .unwrap()
            .iter()
            .map(|r| r.id.clone())
            .collect();

        assert!(!ids_seq.is_empty(), "should find the query's own vector");
        assert_eq!(
            ids_seq, ids_par,
            "multi-core (degree=4) must return the same id-set as sequential (degree=1)"
        );
    }

    /// ADR-030 / TD-158: the SST `ComputeMsGuard` records elapsed compute to the
    /// active per-query I/O trace on drop, under the engine label — so the
    /// billing observer can attribute KRU to "sst". No-op outside a scope.
    #[tokio::test]
    async fn compute_ms_guard_records_to_active_io_trace() {
        use crate::observability::io_trace;
        let snap = io_trace::scope(async {
            {
                let _g = ComputeMsGuard::new("sst");
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            } // guard drops here → records elapsed into the active trace
            io_trace::snapshot()
        })
        .await
        .expect("snapshot inside an active io_trace scope");
        assert!(
            snap.compute_ms.contains_key("sst"),
            "guard must record compute under the 'sst' engine label, got {:?}",
            snap.compute_ms
        );
        assert!(
            snap.total_compute_ms() >= 1,
            "elapsed compute must be >= 1ms after a 2ms sleep, got {}",
            snap.total_compute_ms()
        );
    }

    #[tokio::test]
    async fn test_parse_storage_url() {
        let engine = create_test_engine().await;

        // Test valid storage URL
        let (base, collection) = engine
            .parse_storage_url("file:///data/collections/test_collection")
            .unwrap();
        assert_eq!(base, "file:///data/collections");
        assert_eq!(collection, "test_collection");

        // Test invalid storage URL
        assert!(engine.parse_storage_url("invalid_url").is_err());
    }

    #[tokio::test]
    async fn pax_centroid_classification_does_not_probe_object() {
        let engine = create_test_engine().await;
        let missing_pax = "file:///definitely-missing/segment.pax";
        assert!(
            engine
                .load_sst_header_centroid(missing_pax)
                .await
                .unwrap()
                .is_none(),
            "PAX carries its partition model in A0 and must not pay an SST1 magic GET"
        );

        let missing_legacy = "file:///definitely-missing/segment.sst";
        assert!(
            engine
                .load_sst_header_centroid(missing_legacy)
                .await
                .is_err(),
            "unknown/legacy paths must retain the mixed-format magic sniff"
        );
    }

    #[tokio::test]
    async fn pax_count_classification_does_not_probe_object() {
        let engine = create_test_engine().await;
        let missing_pax = "file:///definitely-missing/segment.pax";
        assert!(
            engine
                .legacy_sst_entry_count(missing_pax)
                .await
                .unwrap()
                .is_none(),
            "PAX has no legacy SST1 count and must not pay a magic GET"
        );

        let missing_legacy = "file:///definitely-missing/segment.sst";
        assert!(
            engine.legacy_sst_entry_count(missing_legacy).await.is_err(),
            "unknown/legacy count paths must retain the mixed-format magic sniff"
        );
    }

    #[tokio::test]
    async fn test_filter_search_results() {
        let engine = create_test_engine().await;
        let mut results = vec![
            create_test_search_result("id1", vec![1.0, 2.0], 0.5),
            create_test_search_result("id2", vec![3.0, 4.0], 0.3),
        ];

        // Test removing vectors
        engine.filter_search_results(&mut results, false, true);
        assert!(results[0].vector.is_none());
        assert!(results[1].vector.is_none());

        // Test removing metadata
        let mut results = vec![create_test_search_result("id1", vec![1.0, 2.0], 0.5)];
        engine.filter_search_results(&mut results, true, false);
        assert!(results[0].metadata.is_empty());
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }

    fn create_test_search_result(id: &str, values: Vec<f32>, score: f32) -> OptimizedSearchRecord {
        let mut record = OptimizedSearchRecord::default();
        record.id = id.to_string();
        record.score = score;
        record.vector = Some(Arc::new(values));
        record.metadata = {
            let mut metadata = HashMap::new();
            metadata.insert(
                "test_key".to_string(),
                ProximaValue::String("test_value".to_string()),
            );
            metadata
        };
        record
    }

    // ---- ADR-028 index_policy → exact-scan budget resolution ----

    use crate::proto::proximadb_v1::IndexPolicy;

    #[test]
    fn resolve_budget_none_uses_storage_class_default() {
        let (budget, pin_exact) = resolve_exact_budget(None);
        assert_eq!(budget, EXACT_SCAN_MAX_BYTES);
        assert!(!pin_exact);
    }

    #[test]
    fn resolve_budget_mode_exact_pins_exact() {
        let p = IndexPolicy {
            mode: "Exact".to_string(),
            ..Default::default()
        };
        let (_, pin_exact) = resolve_exact_budget(Some(&p));
        assert!(pin_exact, "mode=exact must pin exact regardless of N");
    }

    #[test]
    fn resolve_budget_nonzero_override_wins() {
        let p = IndexPolicy {
            mode: "auto".to_string(),
            byte_budget: 8 * 1024 * 1024,
            ..Default::default()
        };
        let (budget, pin_exact) = resolve_exact_budget(Some(&p));
        assert_eq!(budget, 8 * 1024 * 1024);
        assert!(!pin_exact);
    }

    #[test]
    fn resolve_budget_zero_override_falls_back_to_default() {
        let p = IndexPolicy {
            mode: "auto".to_string(),
            byte_budget: 0,
            ..Default::default()
        };
        let (budget, _) = resolve_exact_budget(Some(&p));
        assert_eq!(budget, EXACT_SCAN_MAX_BYTES);
    }

    #[test]
    fn exact_prune_contract_blocks_every_pax_cascade_entry() {
        let approximate = crate::core::search::BlockPruneConfig::default();
        assert!(SstEngine::should_try_pax_cascade(
            "segment.pax",
            DistanceMetric::Euclidean,
            &approximate,
        ));

        let exact = crate::core::search::BlockPruneConfig {
            force_exact: true,
            ..Default::default()
        };
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ] {
            assert!(
                !SstEngine::should_try_pax_cascade("segment.pax", metric, &exact),
                "force_exact must dominate every ANN-compatible metric"
            );
        }
    }

    #[test]
    fn adaptive_route_resolved_exact_rewrites_every_pruning_control() {
        let caller = crate::core::search::SearchParams {
            search_mode: crate::core::search::SearchMode::Adaptive { threshold: 10_000 },
            block_prune: crate::core::search::BlockPruneConfig {
                force_exact: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let effective = SstEngine::force_exact_params(&caller);
        assert!(matches!(
            effective.search_mode,
            crate::core::search::SearchMode::Exact
        ));
        assert!(effective.block_prune.force_exact);
        assert!(matches!(
            caller.search_mode,
            crate::core::search::SearchMode::Adaptive { .. }
        ));
        assert!(
            !caller.block_prune.force_exact,
            "caller intent stays immutable"
        );
    }
}
