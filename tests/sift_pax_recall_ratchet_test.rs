//! WS8 — real-dataset PAX RaBitQ→SQ8 recall ratchet on SIFT1M.
//!
//! The existing PAX recall ratchet (`pax-recall-ratchet`, qa-gate) runs on a
//! synthetic DIM=64 LCG corpus and explicitly flags itself as NOT representative
//! of production (clustered / high-dim / N≥1M). The default-on flip (Phase F) is
//! GATED on a real-dataset recall@10 ≥ 0.90 artifact — this test produces it.
//!
//! It mirrors `pax_cascade_prod_search_test` (Euclidean `SstEngine`, PAX env on,
//! `search_vectors_unified` dispatch) but swaps:
//!   - synthetic vectors → the real SIFT1M corpus (TEXMEX `.fvecs`, 128-d), and
//!   - brute-force ground truth → SIFT's provided `.ivecs` neighbours (full 1M),
//!     or a brute-force oracle over the inserted subset (N < 1M floor).
//!
//! Format choice: plain-binary `.fvecs`/`.ivecs` (no native dep), deliberately
//! avoiding the `hdf5` crate's libhdf5 build friction across CI platforms (the
//! plan's mandated evaluation). Parsed with std only (little-endian).
//!
//! Dataset (env `PROXIMADB_SIFT_DATASET_DIR`):
//!   - `sift_base.fvecs`       (1 000 000 × 128)  — insert set
//!   - `sift_query.fvecs`      (10 000 × 128)     — query set
//!   - `sift_groundtruth.ivecs`(10 000 × 100)     — true neighbours (full 1M only)
//!
//! Download (qa-gate `sift-pax-recall` job or local; TEXMEX archive MD5
//! `b23d1b3b2ee8469d819b61ca900ef0ed`):
//! ```text
//! curl -fL -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//! ```
//!
//! Env knobs:
//!   PROXIMADB_SIFT_DATASET_DIR   dir holding the three files (unset → SKIP)
//!   PROXIMADB_SIFT_N             insert only the first N base vectors (subset /
//!                                CI floor); unset → full 1M + provided GT
//!   PROXIMADB_SIFT_QUERIES       cap the query count (default 1000)
//!   PROXIMADB_SIFT_RECALL_FLOOR  ratchet threshold (default 0.90)
//!   PROXIMADB_SIFT_COALESCED_BYTE_BUDGET optional bytes/query ceiling for the
//!                                coalesced end-to-end eval (unset = report only)
//!   PROXIMADB_RECALL_DATASET_REQUIRED fail instead of skip when corpus is absent
//!   PROXIMADB_OBJECT_STORE_URL optional registered cloud-emulator/real-cloud
//!                                base; unset keeps the paired cohort local
//!   PROXIMADB_SEGMENT_INVARIANTS_CACHE_MB / PROXIMADB_SURVIVOR_CACHE_BUDGET_MB
//!                                explicit nonzero values opt the paired gate
//!                                into those existing warm tiers; unset/0 keeps
//!                                its controlled cache-disabled baseline
//!
//! nextest isolates each test in its own process, so the PAX env vars set here
//! don't leak. `set_var` is `unsafe` (edition 2024).

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchMode, SearchParams};
use proximadb::observability::io_trace::{
    VectorAccessPath, VectorCachePolicy, VectorSearchIntent, VectorStorageScope,
};
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
};
use proximadb::query::route_cost_model::{VectorCacheOutcome, vector_access_shape};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::engines::sst::segment_format::{
    CacheTier, SegmentInvariantsCache, drain_get_trace, rabitq_search_segment_coalesced,
    write_pax_segment_compacted,
};
use proximadb::storage::engines::sst::survivor_range_cache::SurvivorRangeCache;
use proximadb::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use proximadb_block_format::{RankMetric, VectorQuant};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tempfile::TempDir;

const DIMENSION: usize = 128;
const TOP_K: usize = 10;
const BATCH_SIZE: usize = 20_000;
const DEFAULT_QUERIES: usize = 1000;
/// Bound the exact side of the paired access-path evidence across every SIFT
/// scale. This yields 100 pairs at N=10k, 30 at N=100k, and 3 at N=1M: enough
/// to cross the route-cost model's three-sample warmup without making the
/// full-corpus recall ratchet pay an unbounded brute-force tax.
const MAX_PAIRED_VECTOR_COMPARISONS: usize = 3_000_000;
const MIN_PAIRED_SAMPLES: usize = 3;

fn directory_file_bytes(path: &Path) -> std::io::Result<u64> {
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            bytes = bytes.saturating_add(directory_file_bytes(&entry.path())?);
        } else if metadata.is_file() {
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// TEXMEX .fvecs / .ivecs loader (std-only, little-endian — no native dep).
// ---------------------------------------------------------------------------

/// Read up to `limit` records from an `.fvecs`/`.ivecs` file. Each record is
/// `[count: i32 LE][count × T LE]` where T is f32 for fvecs. `None` = all.
fn read_vec_records_f32(path: &Path, limit: Option<usize>) -> std::io::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let dim = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let need = dim.checked_mul(4).expect("fvecs dim overflow");
        if pos + need > bytes.len() {
            break;
        }
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            let off = pos + j * 4;
            v.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
        pos += need;
        out.push(v);
        if matches!(limit, Some(n) if out.len() >= n) {
            break;
        }
    }
    Ok(out)
}

/// Read an `.ivecs` file of integer records → `Vec<Vec<u32>>` (SIFT ground-truth
/// neighbour indices are non-negative).
fn read_vec_records_u32(path: &Path, limit: Option<usize>) -> std::io::Result<Vec<Vec<u32>>> {
    let bytes = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let dim = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let need = dim.checked_mul(4).expect("ivecs dim overflow");
        if pos + need > bytes.len() {
            break;
        }
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            let off = pos + j * 4;
            v.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as u32);
        }
        pos += need;
        out.push(v);
        if matches!(limit, Some(n) if out.len() >= n) {
            break;
        }
    }
    Ok(out)
}

fn dataset_path(filename: &str) -> Option<PathBuf> {
    std::env::var("PROXIMADB_SIFT_DATASET_DIR").ok().map(|d| {
        let mut p = PathBuf::from(d);
        p.push(filename);
        p
    })
}

fn dataset_required() -> bool {
    std::env::var("PROXIMADB_RECALL_DATASET_REQUIRED")
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
}

fn paired_storage_base(local_base: &Path, configured: Option<&str>, run_id: &str) -> String {
    match configured {
        Some(base) => {
            let base = base.trim();
            assert!(
                !base.is_empty(),
                "PROXIMADB_OBJECT_STORE_URL must be unset or a non-empty storage URL"
            );
            format!("{}/td-xmodal-4-paired/{run_id}", base.trim_end_matches('/'))
        }
        None => local_base.to_string_lossy().into_owned(),
    }
}

fn explicit_cache_budget_bytes(configured: Option<&str>, gate: &str) -> Option<u64> {
    let raw = configured?;
    let mb = raw
        .trim()
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("{gate} must be an unsigned MiB value, got {raw:?}"));
    if mb == 0 {
        None
    } else {
        Some(
            mb.checked_mul(1024 * 1024)
                .unwrap_or_else(|| panic!("{gate} MiB value overflows bytes: {mb}")),
        )
    }
}

/// Squared L2 distance (order-preserving — fine for ranking).
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force exact top-`k` neighbour ids over `base` for `query` (subset GT).
fn brute_force_topk(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<String> {
    let mut sims: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq(query, v)))
        .collect();
    if sims.len() > k {
        sims.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
        sims.truncate(k);
    }
    sims.into_iter().map(|(i, _)| format!("v{i}")).collect()
}

/// Build exact top-k sets for a prefix subset without doing a full sort per
/// query. SIFT's provided top-100 row is still exact after filtering to ids in
/// the subset whenever it contains at least `TOP_K` such ids. The remaining
/// rows fall back to a parallel brute-force linear selection.
fn exact_ground_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt_path: Option<&Path>,
) -> (Vec<std::collections::HashSet<String>>, usize) {
    let provided = gt_path
        .filter(|path| path.exists())
        .map(|path| read_vec_records_u32(path, Some(queries.len())).expect("read gt"))
        .unwrap_or_default();
    let fallback_count = AtomicUsize::new(0);
    let result = queries
        .par_iter()
        .enumerate()
        .map(|(qi, query)| {
            if let Some(row) = provided.get(qi) {
                let filtered: Vec<u32> = row
                    .iter()
                    .copied()
                    .filter(|id| (*id as usize) < base.len())
                    .take(TOP_K)
                    .collect();
                if filtered.len() == TOP_K {
                    return filtered.into_iter().map(vid).collect();
                }
            }
            fallback_count.fetch_add(1, Ordering::Relaxed);
            brute_force_topk(base, query, TOP_K).into_iter().collect()
        })
        .collect();
    (result, fallback_count.load(Ordering::Relaxed))
}

fn vid(i: u32) -> String {
    format!("v{i}")
}

/// Mint a catalog-style decimal collection object id (ADR-075:
/// `CollectionObjectId = u64`).
///
/// The SST flush and compaction-admission paths are fail-closed on a decimal id
/// (`FlushParams::get_collection_object_id` /
/// `CompactionParams::get_collection_object_id`, in
/// `crates/storage/proximadb-storage-traits/src/types.rs`), so a bare-engine
/// test that drives either must supply one — the human name belongs in
/// `CollectionConfig.name`. Do NOT hash the name here: that boundary's own doc
/// comment warns it would create "a second, collision-prone identity domain that
/// cannot be joined reliably with catalog, MVCC, deletion-vector, or compaction
/// state". Mirrors the production catalog's monotonic mint without standing up
/// the catalog, as `tests/sst_ivf2_compaction_route_proof_test.rs` already does.
fn next_object_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed).to_string()
}

fn collection(id: &str, base_location: String) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: DIMENSION as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn vector_record(i: u32, v: Vec<f32>) -> VectorRecord {
    VectorRecord {
        id: vid(i),
        vector: v,
        metadata: HashMap::new(),
        version: Some(1),
        timestamp: Some(i as i64),
        updated_at: None,
        expires_at: None,
        source: None,
    }
}

async fn flush_batch(engine: &SstEngine, collection: &Collection, batch: Vec<VectorRecord>) {
    let n = batch.len();
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: batch.into_iter().map(Into::into).collect(),
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: None,
        trigger_compaction: false,
        batch_ids: vec![],
        collection_config: Some(collection.clone()),
        estimated_size: 0,
    };
    let result = engine.do_flush(&params).await.expect("flush succeeds");
    assert!(result.success, "flush should succeed");
    assert!(
        result.entries_flushed.unwrap_or(0) > 0,
        "flush should write vectors (batch of {n})"
    );
}

async fn search_topk_with_mode(
    engine: &SstEngine,
    collection: &Collection,
    query: Vec<f32>,
    search_mode: SearchMode,
) -> Vec<String> {
    let ctx = StorageQueryContext {
        search_params: Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            search_mode,
            block_prune: BlockPruneConfig {
                radius_k: 0.0,
                force_exact: false,
                mode: BlockPruneMode::Ratio,
                ratio: 1.0,
                min_keep: 1,
                max_keep: 0,
                min_blocks_override: Some(0),
            },
            ..Default::default()
        }),
        collection: Arc::new(collection.clone()),
        metadata: StorageQueryMetadata {
            collection_id: collection.id.clone(),
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };
    engine
        .search_vectors_unified(&ctx)
        .await
        .expect("search succeeds")
        .into_iter()
        .map(|r| r.id)
        .collect()
}

async fn search_topk(engine: &SstEngine, collection: &Collection, query: Vec<f32>) -> Vec<String> {
    search_topk_with_mode(engine, collection, query, SearchMode::default()).await
}

#[derive(Debug)]
struct MeasuredSearch {
    ids: Vec<String>,
    elapsed_us: u64,
    snapshot: proximadb::observability::io_trace::IoTraceSnapshot,
}

#[derive(Debug, Default)]
struct OutcomeMeasurements {
    elapsed_us: Vec<u64>,
    get_ops: u64,
    range_gets: u64,
    bytes_read: u64,
    compute_ms: u64,
    footer_hits: u64,
    footer_misses: u64,
    survivor_l1_hits: u64,
    survivor_l1_misses: u64,
    l2_hits: u64,
    l2_misses: u64,
}

impl OutcomeMeasurements {
    fn latency_min_median_max(&self) -> (u64, u64, u64) {
        let mut elapsed_us = self.elapsed_us.clone();
        elapsed_us.sort_unstable();
        (
            elapsed_us.first().copied().unwrap_or_default(),
            percentile_us(&elapsed_us, 50),
            elapsed_us.last().copied().unwrap_or_default(),
        )
    }
}

fn record_outcome_measurement(
    cohorts: &mut BTreeMap<&'static str, OutcomeMeasurements>,
    elapsed_us: u64,
    snapshot: &proximadb::observability::io_trace::IoTraceSnapshot,
) {
    let cohort = cohorts
        .entry(VectorCacheOutcome::from_snapshot(snapshot).as_str())
        .or_default();
    cohort.elapsed_us.push(elapsed_us);
    cohort.get_ops = cohort.get_ops.saturating_add(snapshot.get_ops);
    cohort.range_gets = cohort.range_gets.saturating_add(snapshot.range_gets);
    cohort.bytes_read = cohort.bytes_read.saturating_add(snapshot.bytes_read);
    cohort.compute_ms = cohort
        .compute_ms
        .saturating_add(snapshot.total_compute_ms());
    cohort.footer_hits = cohort.footer_hits.saturating_add(snapshot.footer_hits);
    cohort.footer_misses = cohort.footer_misses.saturating_add(snapshot.footer_misses);
    cohort.survivor_l1_hits = cohort
        .survivor_l1_hits
        .saturating_add(snapshot.survivor_l1_hits);
    cohort.survivor_l1_misses = cohort
        .survivor_l1_misses
        .saturating_add(snapshot.survivor_l1_misses);
    cohort.l2_hits = cohort.l2_hits.saturating_add(snapshot.l2_hits);
    cohort.l2_misses = cohort.l2_misses.saturating_add(snapshot.l2_misses);
}

fn report_outcome_measurements(
    arm: &str,
    cohorts: &mut BTreeMap<&'static str, OutcomeMeasurements>,
) {
    for (outcome, cohort) in cohorts {
        let samples = cohort.elapsed_us.len();
        let denominator = samples as f64;
        let (min_us, median_us, max_us) = cohort.latency_min_median_max();
        eprintln!(
            "SIFT {arm} cache-outcome evidence: outcome={outcome}, samples={samples}, \
             latency min/median/max={min_us}/{median_us}/{max_us} us; \
             GET/range-GET/bytes/compute-ms per sample=\
             {:.2}/{:.2}/{:.0}/{:.2}; footer hit/miss={:.2}/{:.2}, \
             survivor-L1 hit/miss={:.2}/{:.2}, L2 hit/miss={:.2}/{:.2}",
            cohort.get_ops as f64 / denominator,
            cohort.range_gets as f64 / denominator,
            cohort.bytes_read as f64 / denominator,
            cohort.compute_ms as f64 / denominator,
            cohort.footer_hits as f64 / denominator,
            cohort.footer_misses as f64 / denominator,
            cohort.survivor_l1_hits as f64 / denominator,
            cohort.survivor_l1_misses as f64 / denominator,
            cohort.l2_hits as f64 / denominator,
            cohort.l2_misses as f64 / denominator,
        );
    }
}

async fn measured_search(
    engine: &SstEngine,
    collection: &Collection,
    query: Vec<f32>,
    search_mode: SearchMode,
) -> MeasuredSearch {
    let started = Instant::now();
    let (ids, snapshot) = proximadb::observability::io_trace::scope(async {
        let ids = search_topk_with_mode(engine, collection, query, search_mode).await;
        let snapshot = proximadb::observability::io_trace::snapshot()
            .expect("io_trace scope active during measured vector search");
        (ids, snapshot)
    })
    .await;
    MeasuredSearch {
        ids,
        elapsed_us: started.elapsed().as_micros() as u64,
        snapshot,
    }
}

fn assert_single_vector_access(
    measured: &MeasuredSearch,
    expected_intent: VectorSearchIntent,
    expected_path: VectorAccessPath,
    expected_storage_scope: VectorStorageScope,
    expected_cache_policy: VectorCachePolicy,
) {
    assert_eq!(
        measured.snapshot.vector_accesses.len(),
        1,
        "each ratchet query must report one physical vector access"
    );
    let access = &measured.snapshot.vector_accesses[0];
    assert_eq!(
        access.requested_mode, expected_intent,
        "ratchet caller intent must be explicit and attributable"
    );
    assert_eq!(
        access.actual_path, expected_path,
        "ratchet must exercise the requested physical access path"
    );
    assert_eq!(access.engine, "sst");
    assert_eq!(access.dimensions, DIMENSION as u64);
    assert_eq!(access.top_k, TOP_K as u64);
    assert!(!access.has_filter);
    assert_eq!(
        access.storage_scope, expected_storage_scope,
        "the SIFT fixture must attribute the scope derived from its storage URL"
    );
    assert_eq!(
        access.cache_policy, expected_cache_policy,
        "the SIFT fixture must stamp the cache policy visible before routing"
    );
}

/// WS8 real-dataset ratchet: PAX RaBitQ→SQ8 cascade recall@10 on SIFT1M.
#[tokio::test]
async fn sift_pax_cascade_recall_at_10_ratchet() {
    // Post-flip the RG layout is the write default; this arm pins the LEGACY
    // block framing via the kill-switch so the A/B contrast stays measurable.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        // A paired exact/ANN cost sample is admissible only when the exact arm
        // ranks authoritative inserted vectors, not lossy SQ8 reconstructions.
        std::env::set_var("PROXIMADB_PAX_F32_TIER", "1");
        std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0");
    }
    run_sift_ratchet("baseline").await;
}

/// TD-PAXRG-1 Phase G: the SAME ratchet with the row-group Region D layout
/// ON (`PROXIMADB_PAX_WRITE_RG_LAYOUT=1`) — Regions A/B are byte-identical so
/// recall must hold (read-path plumbing integrity at scale: footer-index
/// ranged reads, RG zone pruning, OID-chunk top-k, Region C exact rerank).
/// Also under f32 tier ⇒ the Region C hoist is exercised end-to-end.
#[tokio::test]
async fn sift_pax_cascade_recall_at_10_ratchet_rg_layout() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_PAX_F32_TIER", "1");
        std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "1");
    }
    run_sift_ratchet("rg_layout").await;
}

async fn run_sift_ratchet(label: &str) {
    let base_path = match dataset_path("sift_base.fvecs") {
        Some(p) => p,
        None => {
            assert!(
                !dataset_required(),
                "PROXIMADB_SIFT_DATASET_DIR is required by this recall gate"
            );
            eprintln!(
                "skip: PROXIMADB_SIFT_DATASET_DIR unset — SIFT1M ratchet needs the TEXMEX corpus"
            );
            return;
        }
    };
    if !base_path.exists() {
        assert!(
            !dataset_required(),
            "required SIFT dataset is missing: {base_path:?}"
        );
        eprintln!("skip: {base_path:?} not found — download SIFT1M (.fvecs) to enable");
        return;
    }

    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let subset_n: Option<usize> = std::env::var("PROXIMADB_SIFT_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0);
    let max_queries: usize = std::env::var("PROXIMADB_SIFT_QUERIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_QUERIES);
    let floor: f64 = std::env::var("PROXIMADB_SIFT_RECALL_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| (0.0..=1.0).contains(f))
        .unwrap_or(0.90);

    let temp_dir = TempDir::new().unwrap();
    let object_store_url = std::env::var("PROXIMADB_OBJECT_STORE_URL").ok();
    let storage_base = paired_storage_base(
        temp_dir.path(),
        object_store_url.as_deref(),
        &uuid::Uuid::new_v4().simple().to_string(),
    );
    let expected_storage_scope = VectorStorageScope::from_storage_url(&storage_base);
    assert_ne!(
        expected_storage_scope,
        VectorStorageScope::Unknown,
        "paired SIFT evidence requires a known local or remote storage URL: {storage_base}"
    );
    let collection = collection(&format!("sift_pax_ratchet_{label}"), storage_base);
    // Unset cache-budget gates preserve the controlled cache-free baseline.
    // Explicit existing budgets opt this same harness into the corresponding
    // route-time policy; no parallel benchmark or new env gate is needed.
    let invariants_raw = std::env::var("PROXIMADB_SEGMENT_INVARIANTS_CACHE_MB").ok();
    let survivor_raw = std::env::var("PROXIMADB_SURVIVOR_CACHE_BUDGET_MB").ok();
    let invariants_budget = explicit_cache_budget_bytes(
        invariants_raw.as_deref(),
        "PROXIMADB_SEGMENT_INVARIANTS_CACHE_MB",
    );
    let survivor_budget = explicit_cache_budget_bytes(
        survivor_raw.as_deref(),
        "PROXIMADB_SURVIVOR_CACHE_BUDGET_MB",
    );
    let expected_cache_policy =
        VectorCachePolicy::from_tiers(invariants_budget.is_some(), survivor_budget.is_some());
    let engine = SstEngine::new().await.unwrap().with_warm_tier_caches(
        invariants_budget.map(|bytes| {
            Arc::new(SegmentInvariantsCache::new(
                usize::try_from(bytes).expect("invariants cache budget fits usize"),
            ))
        }),
        survivor_budget.map(|bytes| Arc::new(SurvivorRangeCache::new(bytes))),
    );

    // --- Load base (subset or full) ---------------------------------------------------
    eprintln!("loading base vectors ({})", {
        subset_n
            .map(|n| format!("subset N={n}"))
            .unwrap_or_else(|| "full 1M".to_string())
    });
    let base = read_vec_records_f32(&base_path, subset_n).expect("read sift_base.fvecs");
    let n = base.len();
    assert!(n >= TOP_K, "need at least {TOP_K} base vectors, got {n}");

    // --- Insert + flush in batches ----------------------------------------------------
    let mut batch: Vec<VectorRecord> = Vec::with_capacity(BATCH_SIZE);
    for (i, v) in base.iter().enumerate() {
        batch.push(vector_record(i as u32, v.clone()));
        if batch.len() == BATCH_SIZE {
            let b = std::mem::take(&mut batch);
            flush_batch(&engine, &collection, b).await;
        }
    }
    if !batch.is_empty() {
        flush_batch(&engine, &collection, batch).await;
    }
    eprintln!("flushed {n} base vectors");

    // --- Load queries -----------------------------------------------------------------
    let query_path = dataset_path("sift_query.fvecs");
    if dataset_required() {
        assert!(
            query_path.as_ref().is_some_and(|path| path.exists()),
            "required SIFT query corpus is missing: {query_path:?}"
        );
    }
    let queries: Vec<Vec<f32>> = match query_path.as_ref().filter(|p| p.exists()) {
        Some(p) => read_vec_records_f32(p, Some(max_queries)).expect("read sift_query.fvecs"),
        None => {
            // Fall back to the first base vectors as queries (degenerate but lets
            // the ratchet run without the query file; recall is then self-match).
            base.iter().take(max_queries.min(n)).cloned().collect()
        }
    };
    let qcount = queries.len();
    assert!(qcount > 0, "need at least one SIFT query");

    // --- Ground truth: filtered provided top-100 plus exact subset fallback ------------
    let gt_path = dataset_path("sift_groundtruth.ivecs");
    if dataset_required() {
        assert!(
            gt_path.as_ref().is_some_and(|path| path.exists()),
            "required SIFT ground-truth corpus is missing: {gt_path:?}"
        );
    }
    let (ground_truth, brute_force_rows) = exact_ground_truth(&base, &queries, gt_path.as_deref());
    eprintln!(
        "ground truth rows={qcount}: provided/filterable={}, brute-force={brute_force_rows}",
        qcount - brute_force_rows
    );
    // base no longer needed for GT in the provided-GT path; drop to free memory
    // before the query loop on the 1M run.
    drop(base);

    // --- Search + recall + bounded exact/ANN paired evidence --------------------------
    let paired_queries = qcount.min(
        (MAX_PAIRED_VECTOR_COMPARISONS / n.max(1))
            .max(MIN_PAIRED_SAMPLES)
            .min(qcount),
    );
    let mut recall_sum = 0.0f64;
    let mut measured = 0usize;
    let mut exact_us = Vec::with_capacity(paired_queries);
    let mut ann_us = Vec::with_capacity(paired_queries);
    let mut exact_gets = 0u64;
    let mut ann_gets = 0u64;
    let mut exact_range_gets = 0u64;
    let mut ann_range_gets = 0u64;
    let mut exact_bytes = 0u64;
    let mut ann_bytes = 0u64;
    let mut exact_compute_ms = 0u64;
    let mut ann_compute_ms = 0u64;
    let mut comparable_regime_pairs: HashMap<String, usize> = HashMap::new();
    let mut completed_outcome_pairs: BTreeMap<String, usize> = BTreeMap::new();
    let mut exact_outcome_measurements = BTreeMap::new();
    let mut ann_outcome_measurements = BTreeMap::new();
    for (qi, query) in queries.iter().enumerate() {
        let paired = qi < paired_queries;
        let (ann, exact) = if paired && qi.is_multiple_of(2) {
            let exact =
                measured_search(&engine, &collection, query.clone(), SearchMode::Exact).await;
            let ann = measured_search(
                &engine,
                &collection,
                query.clone(),
                SearchMode::Approximate { nprobe: None },
            )
            .await;
            (ann, Some(exact))
        } else if paired {
            let ann = measured_search(
                &engine,
                &collection,
                query.clone(),
                SearchMode::Approximate { nprobe: None },
            )
            .await;
            let exact =
                measured_search(&engine, &collection, query.clone(), SearchMode::Exact).await;
            (ann, Some(exact))
        } else {
            (
                measured_search(
                    &engine,
                    &collection,
                    query.clone(),
                    SearchMode::Approximate { nprobe: None },
                )
                .await,
                None,
            )
        };

        assert_single_vector_access(
            &ann,
            VectorSearchIntent::Approximate,
            VectorAccessPath::Ann,
            expected_storage_scope,
            expected_cache_policy,
        );
        if let Some(exact) = exact {
            assert_single_vector_access(
                &exact,
                VectorSearchIntent::Exact,
                VectorAccessPath::Exact,
                expected_storage_scope,
                expected_cache_policy,
            );
            let exact_shape = vector_access_shape(&exact.snapshot.vector_accesses[0]);
            let ann_shape = vector_access_shape(&ann.snapshot.vector_accesses[0]);
            let exact_cache = VectorCacheOutcome::from_snapshot(&exact.snapshot);
            let ann_cache = VectorCacheOutcome::from_snapshot(&ann.snapshot);
            record_outcome_measurement(
                &mut exact_outcome_measurements,
                exact.elapsed_us,
                &exact.snapshot,
            );
            record_outcome_measurement(
                &mut ann_outcome_measurements,
                ann.elapsed_us,
                &ann.snapshot,
            );
            *completed_outcome_pairs
                .entry(format!(
                    "exact={}/ann={}",
                    exact_cache.as_str(),
                    ann_cache.as_str()
                ))
                .or_default() += 1;
            let known_scope = exact.snapshot.vector_accesses[0].storage_scope
                != VectorStorageScope::Unknown
                && ann.snapshot.vector_accesses[0].storage_scope != VectorStorageScope::Unknown;
            if exact_shape == ann_shape
                && exact_cache.is_admissible()
                && ann_cache.is_admissible()
                && known_scope
            {
                *comparable_regime_pairs.entry(exact_shape).or_default() += 1;
            } else {
                eprintln!(
                    "  cohort mismatch query {qi}: exact={exact_shape}/outcome={}, \
                     ANN={ann_shape}/outcome={}",
                    exact_cache.as_str(),
                    ann_cache.as_str()
                );
            }
            let exact_ids: std::collections::HashSet<String> =
                exact.ids.iter().take(TOP_K).cloned().collect();
            assert_eq!(
                exact_ids.intersection(&ground_truth[qi]).count(),
                TOP_K,
                "exact SIFT top-k must match the ground-truth set for query {qi}"
            );
            exact_us.push(exact.elapsed_us);
            assert!(
                exact.snapshot.get_ops > 0 && exact.snapshot.bytes_read > 0,
                "exact PAX scan must report its whole-segment physical read"
            );
            exact_gets = exact_gets.saturating_add(exact.snapshot.get_ops);
            exact_range_gets = exact_range_gets.saturating_add(exact.snapshot.range_gets);
            exact_bytes = exact_bytes.saturating_add(exact.snapshot.bytes_read);
            exact_compute_ms = exact_compute_ms.saturating_add(exact.snapshot.total_compute_ms());
            ann_us.push(ann.elapsed_us);
            ann_gets = ann_gets.saturating_add(ann.snapshot.get_ops);
            ann_range_gets = ann_range_gets.saturating_add(ann.snapshot.range_gets);
            ann_bytes = ann_bytes.saturating_add(ann.snapshot.bytes_read);
            ann_compute_ms = ann_compute_ms.saturating_add(ann.snapshot.total_compute_ms());
        }

        if ann.ids.is_empty() {
            eprintln!("  warn: query {qi} returned no results");
            continue;
        }
        let got_ids: std::collections::HashSet<String> = ann.ids.into_iter().take(TOP_K).collect();
        let gt_ids = &ground_truth[qi];
        let overlap = got_ids.intersection(gt_ids).count();
        recall_sum += overlap as f64 / TOP_K as f64;
        measured += 1;
    }
    assert!(measured > 0, "no queries succeeded — cannot report recall");
    let recall = recall_sum / measured as f64;
    eprintln!(
        "SIFT PAX cascade recall@{TOP_K} [{label}] = {recall:.4} over {measured} queries \
         (N={n}, floor={floor}, brute-force-GT rows={brute_force_rows})",
    );
    exact_us.sort_unstable();
    ann_us.sort_unstable();
    let exact_p50_us = percentile_us(&exact_us, 50);
    let exact_p95_us = percentile_us(&exact_us, 95);
    let ann_p50_us = percentile_us(&ann_us, 50);
    let ann_p95_us = percentile_us(&ann_us, 95);
    eprintln!(
        "SIFT paired exact/ANN evidence: pairs={paired_queries}, N={n}, dim={DIMENSION}, \
         top_k={TOP_K}; exact p50/p95={exact_p50_us}/{exact_p95_us} us, \
         ANN p50/p95={ann_p50_us}/{ann_p95_us} us; \
         exact GET/range-GET/bytes/compute-ms per pair={:.2}/{:.2}/{:.0}/{:.2}, \
         ANN GET/range-GET/bytes/compute-ms per pair={:.2}/{:.2}/{:.0}/{:.2}",
        exact_gets as f64 / paired_queries as f64,
        exact_range_gets as f64 / paired_queries as f64,
        exact_bytes as f64 / paired_queries as f64,
        exact_compute_ms as f64 / paired_queries as f64,
        ann_gets as f64 / paired_queries as f64,
        ann_range_gets as f64 / paired_queries as f64,
        ann_bytes as f64 / paired_queries as f64,
        ann_compute_ms as f64 / paired_queries as f64,
    );
    report_outcome_measurements("exact", &mut exact_outcome_measurements);
    report_outcome_measurements("ANN", &mut ann_outcome_measurements);
    let (admitted_regime, admitted_pairs) = comparable_regime_pairs
        .iter()
        .max_by_key(|(_, pairs)| **pairs)
        .map(|(regime, pairs)| (regime.as_str(), *pairs))
        .unwrap_or(("none", 0));
    eprintln!("SIFT comparable regime cohort: regime={admitted_regime}, pairs={admitted_pairs}");
    for (outcomes, pairs) in completed_outcome_pairs {
        eprintln!("SIFT completed cache outcomes: {outcomes}, pairs={pairs}");
    }
    assert!(
        admitted_pairs >= MIN_PAIRED_SAMPLES,
        "exact/ANN evidence must contain at least {MIN_PAIRED_SAMPLES} pairs from the same known storage/cache policy with admissible arm outcomes; best cohort={admitted_regime} ({admitted_pairs})"
    );
    assert!(
        recall >= floor,
        "SIFT PAX cascade recall@{TOP_K} = {recall:.4} < {floor} floor (N={n}) — \
         default-on flip (Phase F) is BLOCKED until the cascade is tuned (pool/REFINE via \
         PROXIMADB_PAX_RABITQ_POOL_MULT / _MIN)"
    );
}

// ---------------------------------------------------------------------------
// Loader unit tests — synthetic .fvecs/.ivecs round-trip (no dataset needed).
// ---------------------------------------------------------------------------

#[test]
fn paired_storage_base_uses_local_fixture_when_object_store_is_unset() {
    let temp_dir = TempDir::new().expect("tempdir");
    assert_eq!(
        paired_storage_base(temp_dir.path(), None, "run-a"),
        temp_dir.path().to_string_lossy()
    );
}

#[test]
fn paired_storage_base_isolates_remote_measurement_prefix() {
    let temp_dir = TempDir::new().expect("tempdir");
    let base = paired_storage_base(
        temp_dir.path(),
        Some("az://proximadb-test/vector-evidence/"),
        "run-a",
    );
    assert_eq!(
        base,
        "az://proximadb-test/vector-evidence/td-xmodal-4-paired/run-a"
    );
    assert_eq!(
        VectorStorageScope::from_storage_url(&base),
        VectorStorageScope::Remote
    );
}

#[test]
fn paired_cache_policy_uses_existing_explicit_tier_budgets() {
    let policy = |invariants, survivor| {
        VectorCachePolicy::from_tiers(
            explicit_cache_budget_bytes(invariants, "invariants").is_some(),
            explicit_cache_budget_bytes(survivor, "survivor").is_some(),
        )
    };
    assert_eq!(policy(None, None), VectorCachePolicy::Disabled);
    assert_eq!(policy(Some("64"), None), VectorCachePolicy::InvariantsOnly);
    assert_eq!(policy(None, Some("64")), VectorCachePolicy::SurvivorOnly);
    assert_eq!(policy(Some("64"), Some("64")), VectorCachePolicy::Full);
    assert_eq!(policy(Some("0"), Some("0")), VectorCachePolicy::Disabled);
}

#[test]
fn cache_outcome_measurements_remain_stratified() {
    let cold = proximadb::observability::io_trace::IoTraceSnapshot {
        get_ops: 3,
        range_gets: 2,
        bytes_read: 30,
        survivor_l1_misses: 4,
        ..Default::default()
    };
    let mixed = proximadb::observability::io_trace::IoTraceSnapshot {
        get_ops: 1,
        range_gets: 1,
        bytes_read: 10,
        survivor_l1_hits: 5,
        survivor_l1_misses: 1,
        ..Default::default()
    };
    let mut cohorts = BTreeMap::new();

    record_outcome_measurement(&mut cohorts, 300, &cold);
    record_outcome_measurement(&mut cohorts, 200, &mixed);
    record_outcome_measurement(&mut cohorts, 100, &cold);

    let cold = cohorts.get("cold").expect("cold cohort");
    assert_eq!(cold.elapsed_us, vec![300, 100]);
    assert_eq!(cold.get_ops, 6);
    assert_eq!(cold.bytes_read, 60);
    assert_eq!(cold.survivor_l1_misses, 8);
    assert_eq!(cold.latency_min_median_max(), (100, 100, 300));

    let mixed = cohorts.get("mixed").expect("mixed cohort");
    assert_eq!(mixed.elapsed_us, vec![200]);
    assert_eq!(mixed.get_ops, 1);
    assert_eq!(mixed.survivor_l1_hits, 5);
    assert_eq!(mixed.survivor_l1_misses, 1);
}

#[test]
fn fvecs_round_trip_parses_dim_and_values() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("synth.fvecs");
    // 3 vectors of dim 4, little-endian.
    let mut bytes = Vec::<u8>::new();
    for v in [
        [1.0f32, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [-1.0, -2.0, -3.0, -4.0],
    ] {
        bytes.extend_from_slice(&4i32.to_le_bytes());
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&path, &bytes).unwrap();

    let got = read_vec_records_f32(&path, None).unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(got[1], vec![10.0, 20.0, 30.0, 40.0]);
    assert_eq!(got[2], vec![-1.0, -2.0, -3.0, -4.0]);

    // limit honored
    let got2 = read_vec_records_f32(&path, Some(2)).unwrap();
    assert_eq!(got2.len(), 2);
}

#[test]
fn ivecs_round_trip_parses_neighbour_indices() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("synth.ivecs");
    let mut bytes = Vec::<u8>::new();
    // 2 rows of 3 indices each.
    for row in [[7i32, 3, 99], [0, 1, 2]] {
        bytes.extend_from_slice(&3i32.to_le_bytes());
        for x in row {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&path, &bytes).unwrap();

    let got = read_vec_records_u32(&path, None).unwrap();
    assert_eq!(got, vec![vec![7, 3, 99], vec![0, 1, 2]]);
}

/// ADR-062 / TD-RDSTRAT-6 PR1 eval: coalesced-RaBitQ **scan-then-rerank** recall
/// + ranged-GET cost on SIFT1M. The two reviewer PR1 ratchets:
///   (a) IVF flush opt-in ON (`PROXIMADB_PAX_FLUSH_CLUSTER=ivf`) — rerank
///       coalescing assumes `cluster_order_pca_ivf` survivor locality.
///   (b) a SINGLE-FILE collection (one flush batch → one segment) isolates the
///       per-file "~1 RaBitQ GET" win, NOT conflated with the PR2 GET-budget
///       compaction (the flush→compaction scheduler is unwired).
/// Expects recall@10 ≈ 0.99 at the keep=100% RaBitQ scan AND range_gets/query ≪
/// today's ~370. `#[ignore]` (needs SIFT1M); record into BENCHMARK_EVIDENCE.toml.
#[tokio::test]
#[ignore = "SIFT1M eval — set PROXIMADB_SIFT_DATASET_DIR + run with --ignored"]
async fn sift_coalesced_rabitq_scan_rerank_eval() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "1");
        // IVF flush opt-in: default OFF (ADR-065 — the decisive no-IVF run proved
        // the coalesced layout needs no cluster_order_pca_ivf ordering; the sign-bit
        // bootstrap is the production flush path, recall 0.985, GETs 40, flush 35 s).
        // Set PROXIMADB_SIFT_IVF_FLUSH=1 to re-enable IVF-at-flush for A/B.
        if std::env::var("PROXIMADB_SIFT_IVF_FLUSH").ok().as_deref() == Some("1") {
            std::env::set_var("PROXIMADB_PAX_FLUSH_CLUSTER", "ivf");
        }
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1"); // the layout under test
        std::env::set_var("PROXIMADB_COUNT_FS_IO", "1");
    }

    let base_path = match dataset_path("sift_base.fvecs") {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!(
                "skip: PROXIMADB_SIFT_DATASET_DIR unset/missing — coalesced eval needs SIFT1M"
            );
            return;
        }
    };
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let subset_n: Option<usize> = std::env::var("PROXIMADB_SIFT_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0);
    let max_queries: usize = std::env::var("PROXIMADB_SIFT_QUERIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_QUERIES);
    let recall_floor: f64 = std::env::var("PROXIMADB_SIFT_COALESCED_RECALL_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| *f > 0.0)
        .unwrap_or(0.95);
    let get_budget: u64 = std::env::var("PROXIMADB_SIFT_COALESCED_GET_BUDGET")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(60);
    let byte_budget: Option<u64> = std::env::var("PROXIMADB_SIFT_COALESCED_BYTE_BUDGET")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|bytes| *bytes > 0);

    let base = read_vec_records_f32(&base_path, subset_n).expect("read sift_base.fvecs");
    let n = base.len();
    assert!(n >= TOP_K, "need at least {TOP_K} base vectors, got {n}");

    let query_path = dataset_path("sift_query.fvecs");
    let queries: Vec<Vec<f32>> = match query_path.as_ref().filter(|p| p.exists()) {
        Some(p) => read_vec_records_f32(p, Some(max_queries)).expect("read sift_query.fvecs"),
        None => base.iter().take(max_queries.min(n)).cloned().collect(),
    };
    let qcount = queries.len();
    let gt_path = dataset_path("sift_groundtruth.ivecs");
    let use_provided_gt = subset_n.is_none() && gt_path.as_ref().is_some_and(|p| p.exists());
    let ground_truth: Vec<std::collections::HashSet<String>> = if use_provided_gt {
        let gt = read_vec_records_u32(gt_path.as_ref().unwrap(), Some(qcount)).expect("read gt");
        gt.into_iter()
            .map(|row| row.into_iter().take(TOP_K).map(vid).collect())
            .collect()
    } else {
        queries
            .iter()
            .map(|q| brute_force_topk(&base, q, TOP_K).into_iter().collect())
            .collect()
    };

    // (b) SINGLE-FILE collection: flush the entire base in ONE batch → one segment
    // file, isolating the per-file "~1 RaBitQ GET" win from the PR2 GET-budget
    // compaction (the flush→compaction scheduler is unwired).
    let temp_dir = TempDir::new().unwrap();
    let collection = collection(
        "sift_coalesced_eval",
        temp_dir.path().to_string_lossy().into_owned(),
    );
    let engine = SstEngine::new().await.unwrap()
        .with_directory_cache(Arc::new(
            proximadb::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache::new(),
        ));
    // ADR-065 Q3: opt-in ranged survivor/OID cache. Set
    // PROXIMADB_SURVIVOR_CACHE_BUDGET_MB (default unset → uncached baseline) to
    // measure the GET/bytes win on the repeated-query working set.
    let survivor_cache = match std::env::var("PROXIMADB_SURVIVOR_CACHE_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|b| *b > 0)
    {
        Some(mb) => Some(Arc::new(
            proximadb::storage::engines::sst::survivor_range_cache::SurvivorRangeCache::new(
                mb * 1024 * 1024,
            ),
        )),
        None => None,
    };
    let engine = engine.with_warm_tier_caches(
        Some(Arc::new(
            proximadb::storage::engines::sst::segment_format::SegmentInvariantsCache::new(
                64 * 1024 * 1024,
            ),
        )),
        survivor_cache,
    );
    let batch: Vec<VectorRecord> = base
        .iter()
        .enumerate()
        .map(|(i, v)| vector_record(i as u32, v.clone()))
        .collect();
    let flush_started = Instant::now();
    flush_batch(&engine, &collection, batch).await;
    let flush_ms = flush_started.elapsed().as_millis();
    let persisted_bytes = directory_file_bytes(temp_dir.path()).expect("measure persisted bytes");
    eprintln!("[coalesced] single-file collection: N={n} vectors flushed in one batch");

    // Search all queries in ONE io_trace scope; range_gets accumulates over the loop.
    let (recall, measured, snap, mut query_us) = proximadb::observability::io_trace::scope(async {
        let mut recall_sum = 0.0f64;
        let mut measured = 0usize;
        let mut query_us = Vec::with_capacity(queries.len());
        for (qi, query) in queries.iter().enumerate() {
            let query_started = Instant::now();
            let got = search_topk(&engine, &collection, query.clone()).await;
            query_us.push(query_started.elapsed().as_micros() as u64);
            if got.is_empty() {
                continue;
            }
            let got_ids: std::collections::HashSet<String> = got.into_iter().take(TOP_K).collect();
            recall_sum += got_ids.intersection(&ground_truth[qi]).count() as f64 / TOP_K as f64;
            measured += 1;
        }
        let recall = if measured > 0 {
            recall_sum / measured as f64
        } else {
            0.0
        };
        let snap = proximadb::observability::io_trace::snapshot().expect("io_trace scope active");
        (recall, measured, snap, query_us)
    })
    .await;
    query_us.sort_unstable();
    let mean_query_us = if query_us.is_empty() {
        0
    } else {
        query_us.iter().sum::<u64>() / query_us.len() as u64
    };
    let p95_query_us = query_us
        .get(query_us.len().saturating_sub(1) * 95 / 100)
        .copied()
        .unwrap_or(0);

    let per_q_gets = if measured > 0 {
        snap.range_gets / measured as u64
    } else {
        0
    };
    let per_q_bytes = if measured > 0 {
        snap.bytes_read / measured as u64
    } else {
        0
    };
    eprintln!(
        "=== ADR-062 coalesced scan-then-rerank (N={n}, {qcount} queries, top-{TOP_K}, IVF ON, single-file) ==="
    );
    eprintln!(
        "  recall@{TOP_K} = {recall:.4}  range_gets/query = {per_q_gets}  bytes_read/query = {per_q_bytes}  (measured={measured}, total_gets={})",
        snap.range_gets
    );
    eprintln!(
        "  persisted_bytes = {persisted_bytes}  persisted_bytes/row = {:.2}  flush_ms = {flush_ms}  mean/p95_query_us = {mean_query_us}/{p95_query_us}",
        persisted_bytes as f64 / n as f64
    );
    eprintln!("  vs legacy block-prune ~370 GETs/query — target: < {get_budget} (≪ 370)");

    // ADR-065 cache-co-design: per-tier GET-size trace. GET count = same-region
    // billing metric; GET sizes = latency (NIC + decode). Each read tagged by
    // CacheTier (InvariantIndex = Region A; InvariantMeta = header/footer/params;
    // SurvivorPayload = Region B ranges; ResultPayload = Region D OIDs).
    let gets = proximadb::storage::engines::sst::segment_format::drain_get_trace();
    if !gets.is_empty() {
        use proximadb::storage::engines::sst::segment_format::CacheTier;
        eprintln!("  per-tier GET trace:");
        for tier in [
            CacheTier::InvariantIndex,
            CacheTier::InvariantMeta,
            CacheTier::SurvivorPayload,
            CacheTier::ResultPayload,
        ] {
            let mut sizes: Vec<u64> = gets
                .iter()
                .filter(|(t, _)| *t == tier)
                .map(|(_, s)| *s)
                .collect();
            if sizes.is_empty() {
                continue;
            }
            sizes.sort_unstable();
            let n = sizes.len();
            let sum: u64 = sizes.iter().sum();
            let p50 = sizes[n / 2];
            let p99 = sizes[(n * 99).saturating_sub(1).min(n - 1)];
            eprintln!(
                "    {:>6}: n={:<5} total={:>7.1} MB  min={:>9}  p50={:>9}  p99={:>9}  max={:>9} B",
                tier.label(),
                n,
                sum as f64 / 1e6,
                sizes[0],
                p50,
                p99,
                sizes[n - 1]
            );
        }
    }

    assert!(
        measured > 0,
        "no queries measured — search returned nothing"
    );
    assert!(
        recall >= recall_floor,
        "coalesced recall@{TOP_K} = {recall:.4} < floor {recall_floor}"
    );
    assert!(
        per_q_gets < get_budget,
        "coalesced range_gets/query = {per_q_gets} >= budget {get_budget} (not ≪ 370)"
    );
    if let Some(byte_budget) = byte_budget {
        assert!(
            per_q_bytes < byte_budget,
            "coalesced bytes_read/query = {per_q_bytes} >= budget {byte_budget}"
        );
    }
}

#[derive(Debug)]
struct Ivf2BakeMetrics {
    recall: f64,
    measured: usize,
    snapshot: proximadb::observability::io_trace::IoTraceSnapshot,
    p50_us: u64,
    p95_us: u64,
    get_trace: Vec<(CacheTier, u64)>,
}

fn ivf2_record(row: usize, vector: &[f32]) -> ProximaRecord {
    ProximaRecord {
        oid: vid(row as u32),
        created_at_ns: 1_000 + row as i64,
        updated_at_ns: 1_000 + row as i64,
        record_version: 1,
        embeddings: vec![EmbeddingCell {
            model_id: "sift1m".into(),
            modality: "dense_vector".into(),
            dim: vector.len() as u32,
            values: EmbeddingValues::Fp32(vector.to_vec()),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn csv_usize_env(name: &str, defaults: &[usize]) -> Vec<usize> {
    let values = std::env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| defaults.to_vec());
    let mut values = values;
    values.sort_unstable();
    values.dedup();
    values
}

fn percentile_us(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() - 1) * percentile.min(100) / 100]
}

fn write_ivf2_geometry(
    root: &Path,
    base: &[Vec<f32>],
    file_count: usize,
    v3: bool,
) -> Vec<PathBuf> {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
        std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "1");
        if v3 {
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        } else {
            std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        }
    }
    let rows_per_file = base.len().div_ceil(file_count);
    let target_block =
        proximadb_storage_common::iops_budget::IopsBudget::CLOUD.target_block_bytes() as usize;
    let mut paths = Vec::new();
    for file_index in 0..file_count {
        let begin = file_index * rows_per_file;
        if begin >= base.len() {
            break;
        }
        let end = (begin + rows_per_file).min(base.len());
        let records: Vec<ProximaRecord> = base[begin..end]
            .iter()
            .enumerate()
            .map(|(local, vector)| ivf2_record(begin + local, vector))
            .collect();
        let path = root.join(format!(
            "{}-{file_index:03}.pax",
            if v3 { "v3" } else { "pxh1" }
        ));
        write_pax_segment_compacted(
            &path,
            &records,
            "sift_ivf2_prc1",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(target_block),
            None, // destination_url
        )
        .expect("write compacted bakeoff segment");
        paths.push(path);
    }
    assert_eq!(paths.len(), file_count.min(base.len()));
    paths
}

async fn search_ivf2_geometry(
    fs: &LocalFileSystem,
    paths: &[PathBuf],
    query: &[f32],
    invariants: Option<&SegmentInvariantsCache>,
    survivors: Option<&SurvivorRangeCache>,
) -> Vec<String> {
    let mut hits = Vec::new();
    for path in paths {
        let path = path.to_str().expect("utf8 bakeoff path");
        let segment_hits = rabitq_search_segment_coalesced(
            fs,
            path,
            query,
            TOP_K,
            RankMetric::L2,
            invariants,
            survivors,
        )
        .await
        .expect("coalesced segment search")
        .expect("bakeoff segment is coalesced");
        hits.extend(segment_hits);
    }
    hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    hits.truncate(TOP_K);
    hits.into_iter().map(|hit| hit.oid).collect()
}

async fn run_ivf2_bake_arm(
    fs: &LocalFileSystem,
    paths: &[PathBuf],
    queries: &[Vec<f32>],
    ground_truth: &[std::collections::HashSet<String>],
    warm: bool,
) -> Ivf2BakeMetrics {
    let _ = drain_get_trace();
    let invariants = warm.then(|| SegmentInvariantsCache::new(512 * 1024 * 1024));
    let survivors = warm.then(|| SurvivorRangeCache::new(512 * 1024 * 1024));
    if warm {
        for query in queries {
            let _ = search_ivf2_geometry(fs, paths, query, invariants.as_ref(), survivors.as_ref())
                .await;
        }
        let _ = drain_get_trace();
    }
    let (recall, measured, snapshot, mut query_us) =
        proximadb::observability::io_trace::scope(async {
            let mut recall_sum = 0.0;
            let mut measured = 0usize;
            let mut query_us = Vec::with_capacity(queries.len());
            for (query_index, query) in queries.iter().enumerate() {
                let started = Instant::now();
                let got =
                    search_ivf2_geometry(fs, paths, query, invariants.as_ref(), survivors.as_ref())
                        .await;
                query_us.push(started.elapsed().as_micros() as u64);
                if got.is_empty() {
                    continue;
                }
                let got: std::collections::HashSet<String> = got.into_iter().collect();
                recall_sum +=
                    got.intersection(&ground_truth[query_index]).count() as f64 / TOP_K as f64;
                measured += 1;
            }
            let recall = if measured == 0 {
                0.0
            } else {
                recall_sum / measured as f64
            };
            let snapshot = proximadb::observability::io_trace::snapshot()
                .expect("bakeoff io_trace scope active");
            (recall, measured, snapshot, query_us)
        })
        .await;
    let get_trace = drain_get_trace();
    query_us.sort_unstable();
    Ivf2BakeMetrics {
        recall,
        measured,
        snapshot,
        p50_us: percentile_us(&query_us, 50),
        p95_us: percentile_us(&query_us, 95),
        get_trace,
    }
}

fn ivf2_stage(metrics: &Ivf2BakeMetrics, tier: CacheTier) -> (u64, u64) {
    metrics
        .get_trace
        .iter()
        .filter(|(observed, _)| *observed == tier)
        .fold((0, 0), |(gets, bytes), (_, size)| (gets + 1, bytes + size))
}

fn ivf2_read_cost(metrics: &Ivf2BakeMetrics) -> f64 {
    let q = metrics.measured.max(1) as f64;
    let gets = metrics.snapshot.range_gets as f64 / q;
    let mib = metrics.snapshot.bytes_read as f64 / q / (1024.0 * 1024.0);
    20.0 * gets + 5.0 * mib
}

fn ivf2_region_a_bytes(metrics: &Ivf2BakeMetrics) -> u64 {
    ivf2_stage(metrics, CacheTier::InvariantIndex).1 + ivf2_stage(metrics, CacheTier::ProbeIndex).1
}

fn env_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

unsafe fn set_ivf2_layout_proof_arm(prefix_bytes: Option<u64>, gap: Option<u64>, split: bool) {
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP");
        std::env::remove_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE");
        std::env::remove_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE");
        if let Some(bytes) = prefix_bytes {
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", bytes.to_string());
        }
        if let Some(bytes) = gap {
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_GAP", bytes.to_string());
            std::env::set_var("PROXIMADB_PAX_VECTOR_COALESCE_RANGE", "4194304");
        }
        if split {
            std::env::set_var("PROXIMADB_PAX_SPLIT_PROBE_META_CACHE", "1");
        }
    }
}

fn print_ivf2_bake_metric(
    files: usize,
    layout: &str,
    temperature: &str,
    nprobe: usize,
    metrics: &Ivf2BakeMetrics,
) {
    let q = metrics.measured.max(1) as u64;
    let (control_gets, control_bytes) = ivf2_stage(metrics, CacheTier::SearchControl);
    let (meta_gets, meta_bytes) = ivf2_stage(metrics, CacheTier::InvariantMeta);
    let (full_a_gets, full_a_bytes) = ivf2_stage(metrics, CacheTier::InvariantIndex);
    let (probe_a_gets, probe_a_bytes) = ivf2_stage(metrics, CacheTier::ProbeIndex);
    let (region_b_gets, region_b_bytes) = ivf2_stage(metrics, CacheTier::SurvivorPayload);
    let (region_d_gets, region_d_bytes) = ivf2_stage(metrics, CacheTier::ResultPayload);
    eprintln!(
        "BAKE_METRIC rdstrat8_prc1 files={files} layout={layout} temperature={temperature} nprobe={nprobe} recall_at_10={:.4} queries={} range_gets_per_query={} bytes_read_per_query={} region_a_bytes_per_query={} region_b_bytes_per_query={} ivf_cells_total_per_query={} ivf_cells_probed_per_query={} probed_rows_per_query={} fetch_rounds_per_query={} whole_region_fallback={} p50_us={} p95_us={} control_gets_per_query={} control_bytes_per_query={} meta_gets_per_query={} meta_bytes_per_query={} full_a_gets_per_query={} full_a_bytes_per_query={} probe_a_gets_per_query={} probe_a_bytes_per_query={} region_b_gets_per_query={} traced_region_b_bytes_per_query={} region_d_gets_per_query={} region_d_bytes_per_query={}",
        metrics.recall,
        metrics.measured,
        metrics.snapshot.range_gets / q,
        metrics.snapshot.bytes_read / q,
        (full_a_bytes + probe_a_bytes) / q,
        region_b_bytes / q,
        metrics.snapshot.ivf_cells_total / q,
        metrics.snapshot.ivf_cells_probed / q,
        metrics.snapshot.ivf_probed_rows / q,
        metrics.snapshot.ivf_fetch_rounds / q,
        metrics.snapshot.ivf_whole_region_fallback,
        metrics.p50_us,
        metrics.p95_us,
        control_gets / q,
        control_bytes / q,
        meta_gets / q,
        meta_bytes / q,
        full_a_gets / q,
        full_a_bytes / q,
        probe_a_gets / q,
        probe_a_bytes / q,
        region_b_gets / q,
        region_b_bytes / q,
        region_d_gets / q,
        region_d_bytes / q,
    );
}

/// TD-RDSTRAT-8 PR-C1 release-profile bakeoff. The control and treatment use
/// the same compaction writer, PCA/IVF row order, file partitions, SQ8/RaBitQ
/// tiers, queries, and ground truth. The only layout difference is PXH1 versus
/// v3/A0; each v3 arm changes only `nprobe`.
#[tokio::test]
#[ignore = "SIFT1M PR-C1 eval — set PROXIMADB_SIFT_DATASET_DIR + run release profile"]
async fn sift_ivf2_probe_release_bakeoff_eval() {
    let base_path = match dataset_path("sift_base.fvecs") {
        Some(path) if path.exists() => path,
        _ if dataset_required() => panic!("SIFT1M dataset required but missing"),
        _ => {
            eprintln!("skip: PROXIMADB_SIFT_DATASET_DIR unset/missing");
            return;
        }
    };
    let base = read_vec_records_f32(&base_path, None).expect("read SIFT1M base");
    assert_eq!(base.len(), 1_000_000, "PR-C1 headline requires full SIFT1M");
    let query_path = dataset_path("sift_query.fvecs").expect("SIFT query path");
    let queries = read_vec_records_f32(&query_path, Some(100)).expect("read 100 SIFT queries");
    assert_eq!(queries.len(), 100, "PR-C1 headline requires 100 queries");
    let gt_path = dataset_path("sift_groundtruth.ivecs").expect("SIFT ground-truth path");
    let truth = read_vec_records_u32(&gt_path, Some(queries.len())).expect("read SIFT truth");
    let ground_truth: Vec<std::collections::HashSet<String>> = truth
        .into_iter()
        .map(|row| row.into_iter().take(TOP_K).map(vid).collect())
        .collect();
    // TD-IOBUDGET-2 (product decision 2026-09-03): 0.98 -> 0.975 for the
    // economical default geometry (k=122 file://: bytes -51%, GETs -13%,
    // recall 0.9794). Historical TD numbers keep their original contexts.
    let recall_floor = 0.975;
    let file_counts = csv_usize_env("PROXIMADB_SIFT_IVF2_FILE_COUNTS", &[1, 4]);
    let nprobes = csv_usize_env("PROXIMADB_SIFT_IVF2_NPROBE_SWEEP", &[4, 8, 16, 32]);
    let layout_proof = env_enabled("PROXIMADB_SIFT_IVF2_LAYOUT_PROOF");
    let proof_prefix = std::env::var("PROXIMADB_SIFT_IVF2_PREFIX_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64 * 1024);
    let proof_gaps = csv_usize_env("PROXIMADB_SIFT_IVF2_A_GAP_SWEEP", &[1024 * 1024]);
    let fs = LocalFileSystem::new_with_encryption(LocalConfig::default(), None)
        .await
        .expect("local filesystem");

    for file_count in file_counts {
        let dir = TempDir::new().expect("bakeoff tempdir");
        let pxh1 = write_ivf2_geometry(dir.path(), &base, file_count, false);
        let v3 = write_ivf2_geometry(dir.path(), &base, file_count, true);

        unsafe {
            std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
            std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
            set_ivf2_layout_proof_arm(None, None, false);
            std::env::set_var("PROXIMADB_TRACE_PAX_STAGES", "1");
        }
        let control_cold = run_ivf2_bake_arm(&fs, &pxh1, &queries, &ground_truth, false).await;
        let control_warm = run_ivf2_bake_arm(&fs, &pxh1, &queries, &ground_truth, true).await;
        print_ivf2_bake_metric(file_count, "pxh1", "cold", 0, &control_cold);
        print_ivf2_bake_metric(file_count, "pxh1", "warm", 0, &control_warm);
        assert!(
            control_cold.recall >= recall_floor,
            "PXH1 control recall {:.4} < {recall_floor} for {file_count} files",
            control_cold.recall
        );

        let mut bytes_qualifying = false;
        let mut joint_get_qualifying = false;
        for nprobe in &nprobes {
            unsafe {
                set_ivf2_layout_proof_arm(None, None, false);
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_NPROBE", nprobe.to_string());
            }
            let probe_cold = run_ivf2_bake_arm(&fs, &v3, &queries, &ground_truth, false).await;
            let probe_warm = run_ivf2_bake_arm(&fs, &v3, &queries, &ground_truth, true).await;
            print_ivf2_bake_metric(file_count, "v3_probe", "cold", *nprobe, &probe_cold);
            print_ivf2_bake_metric(file_count, "v3_probe", "warm", *nprobe, &probe_warm);
            assert!(
                probe_cold.snapshot.ivf_cells_probed > 0,
                "probe did not engage for files={file_count}, nprobe={nprobe}"
            );
            assert_eq!(
                probe_cold.snapshot.ivf_whole_region_fallback, 0,
                "probe silently fell back for files={file_count}, nprobe={nprobe}"
            );
            let q = probe_cold.measured.max(1) as u64;
            let cq = control_cold.measured.max(1) as u64;
            let clears_bytes = probe_cold.recall >= recall_floor
                && probe_cold.snapshot.bytes_read / q < control_cold.snapshot.bytes_read / cq
                && ivf2_region_a_bytes(&probe_cold) / q < ivf2_region_a_bytes(&control_cold) / cq;
            bytes_qualifying |= clears_bytes;
            joint_get_qualifying |= clears_bytes
                && probe_cold.snapshot.range_gets / q < control_cold.snapshot.range_gets / cq;

            if layout_proof {
                let mut arms: Vec<(String, Option<u64>, Option<u64>, bool)> = vec![
                    ("split_meta_cache".to_string(), None, None, true),
                    (
                        format!("prefix_{proof_prefix}"),
                        Some(proof_prefix),
                        None,
                        false,
                    ),
                ];
                for gap in &proof_gaps {
                    arms.push((format!("a_gap_{gap}"), None, Some(*gap as u64), false));
                    arms.push((
                        format!("combined_gap_{gap}"),
                        Some(proof_prefix),
                        Some(*gap as u64),
                        true,
                    ));
                }
                for (label, prefix, gap, split) in arms {
                    unsafe {
                        set_ivf2_layout_proof_arm(prefix, gap, split);
                    }
                    let candidate_cold =
                        run_ivf2_bake_arm(&fs, &v3, &queries, &ground_truth, false).await;
                    let candidate_warm =
                        run_ivf2_bake_arm(&fs, &v3, &queries, &ground_truth, true).await;
                    print_ivf2_bake_metric(
                        file_count,
                        &format!("v3_{label}"),
                        "cold",
                        *nprobe,
                        &candidate_cold,
                    );
                    print_ivf2_bake_metric(
                        file_count,
                        &format!("v3_{label}"),
                        "warm",
                        *nprobe,
                        &candidate_warm,
                    );
                    assert!(
                        (candidate_cold.recall - probe_cold.recall).abs() < f64::EPSILON,
                        "layout-only arm {label} changed cold recall"
                    );
                    assert!(
                        (candidate_warm.recall - probe_warm.recall).abs() < f64::EPSILON,
                        "layout-only arm {label} changed warm recall"
                    );
                    let base_cold_cost = ivf2_read_cost(&probe_cold);
                    let base_warm_cost = ivf2_read_cost(&probe_warm);
                    let candidate_cold_cost = ivf2_read_cost(&candidate_cold);
                    let candidate_warm_cost = ivf2_read_cost(&candidate_warm);
                    eprintln!(
                        "PROOF_GATE rdstrat8_prc2 files={file_count} nprobe={nprobe} arm={label} cold_cost={candidate_cold_cost:.3} baseline_cold_cost={base_cold_cost:.3} cold_accept={} warm_cost={candidate_warm_cost:.3} baseline_warm_cost={base_warm_cost:.3} warm_accept={}",
                        candidate_cold_cost < base_cold_cost,
                        candidate_warm_cost < base_warm_cost,
                    );
                }
                unsafe {
                    set_ivf2_layout_proof_arm(None, None, false);
                }
            }
        }
        eprintln!(
            "BAKE_GATE rdstrat8_prc1 files={file_count} bytes_gate={bytes_qualifying} joint_get_gate={joint_get_qualifying}"
        );
        assert!(
            bytes_qualifying,
            "no v3 nprobe arm cleared recall/byte gates for {file_count} files"
        );
    }

    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
        std::env::remove_var("PROXIMADB_PAX_READ_COARSE_NPROBE");
        std::env::remove_var("PROXIMADB_TRACE_PAX_STAGES");
        set_ivf2_layout_proof_arm(None, None, false);
    }
}

/// TD-RDSTRAT-8 PR-B SIFT-scale recall/GET gate for the two-level IVF coarse
/// probe. This drives v3 through the production compaction trigger, then
/// compares probe ON with the whole-region scan over the same segment.
#[ignore = "SIFT eval — set PROXIMADB_SIFT_DATASET_DIR + run with --ignored"]
#[tokio::test]
async fn sift_ivf2_coarse_probe_recall_ratchet() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_PAX_BLOCK_CLUSTER", "1");
        std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
        std::env::set_var("PROXIMADB_COUNT_FS_IO", "1");
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
        std::env::set_var(
            "PROXIMADB_IVF_K",
            &std::env::var("PROXIMADB_SIFT_IVF_K").unwrap_or_else(|_| "64".to_string()),
        );
        std::env::set_var("PROXIMADB_TRACE_GETS", "1");
    }

    let base_path = match dataset_path("sift_base.fvecs") {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!("skip: PROXIMADB_SIFT_DATASET_DIR unset/missing — IVF2 gate needs SIFT");
            return;
        }
    };
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let subset_n: usize = std::env::var("PROXIMADB_SIFT_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(100_000);
    let max_queries: usize = std::env::var("PROXIMADB_SIFT_QUERIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(100.min(DEFAULT_QUERIES));
    let recall_drop: f64 = std::env::var("PROXIMADB_SIFT_IVF2_RECALL_DROP")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| *f > 0.0)
        .unwrap_or(0.10);

    let base = read_vec_records_f32(&base_path, Some(subset_n)).expect("read sift_base.fvecs");
    let n = base.len();
    assert!(n >= TOP_K, "need at least {TOP_K} base vectors, got {n}");
    let query_path = dataset_path("sift_query.fvecs");
    let queries: Vec<Vec<f32>> = match query_path.as_ref().filter(|p| p.exists()) {
        Some(p) => read_vec_records_f32(p, Some(max_queries)).expect("read sift_query.fvecs"),
        None => base.iter().take(max_queries.min(n)).cloned().collect(),
    };
    let qcount = queries.len();
    let ground_truth: Vec<std::collections::HashSet<String>> = queries
        .iter()
        .map(|q| {
            brute_force_topk(&base, q, TOP_K)
                .into_iter()
                .take(TOP_K)
                .collect()
        })
        .collect();

    let temp_dir = TempDir::new().unwrap();
    let collection = Collection {
        id: next_object_id(),
        config: Some(CollectionConfig {
            name: "sift_ivf2_gate".to_string(),
            dimension: DIMENSION as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            tags: vec!["workload_profile:append".into(), "l0_threshold:2".into()],
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let engine = SstEngine::new().await.unwrap();

    let half = n / 2;
    let batch_a: Vec<VectorRecord> = base[..half]
        .iter()
        .enumerate()
        .map(|(i, v)| vector_record(i as u32, v.clone()))
        .collect();
    let batch_b: Vec<VectorRecord> = base[half..]
        .iter()
        .enumerate()
        .map(|(i, v)| vector_record((half + i) as u32, v.clone()))
        .collect();
    flush_batch(&engine, &collection, batch_a).await;
    let params_b = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: batch_b.into_iter().map(Into::into).collect(),
        force: true,
        synchronous: true,
        collection_config: Some(collection.clone()),
        ..Default::default()
    };
    let second = engine.do_flush(&params_b).await.expect("second flush");
    assert!(second.success, "second flush must succeed");
    assert!(
        second
            .engine_metrics
            .get("compaction_ran")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "armed compaction must run inline on the 2nd flush (compaction_error: {:?})",
        second.compaction_error
    );

    async fn measure(
        engine: &SstEngine,
        collection: &Collection,
        queries: &[Vec<f32>],
        ground_truth: &[std::collections::HashSet<String>],
        probe_on: bool,
    ) -> (f64, u64, u64, usize) {
        unsafe {
            if probe_on {
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "1");
            } else {
                std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
            }
        }
        let _ = proximadb::storage::engines::sst::segment_format::drain_get_trace();
        let _ = proximadb::storage::engines::sst::segment_format::drain_probe_trace();
        let (recall, measured, snap, _us) = proximadb::observability::io_trace::scope(async {
            let mut recall_sum = 0.0f64;
            let mut measured = 0usize;
            for (qi, query) in queries.iter().enumerate() {
                let got = search_topk(engine, collection, query.clone()).await;
                if got.is_empty() {
                    continue;
                }
                let got_ids: std::collections::HashSet<String> =
                    got.into_iter().take(TOP_K).collect();
                recall_sum += got_ids.intersection(&ground_truth[qi]).count() as f64 / TOP_K as f64;
                measured += 1;
            }
            let recall = if measured > 0 {
                recall_sum / measured as f64
            } else {
                0.0
            };
            let snap =
                proximadb::observability::io_trace::snapshot().expect("io_trace scope active");
            (recall, measured, snap, Vec::<u64>::new())
        })
        .await;
        let per_q_gets = if measured > 0 {
            snap.range_gets / measured as u64
        } else {
            0
        };
        let per_q_bytes = if measured > 0 {
            snap.bytes_read / measured as u64
        } else {
            0
        };
        (recall, per_q_gets, per_q_bytes, measured)
    }

    let (recall_base, gets_base, bytes_base, measured_base) =
        measure(&engine, &collection, &queries, &ground_truth, false).await;
    let (recall_probe, gets_probe, bytes_probe, measured_probe) =
        measure(&engine, &collection, &queries, &ground_truth, true).await;
    let probe_trace = proximadb::storage::engines::sst::segment_format::drain_probe_trace();

    eprintln!(
        "=== TD-RDSTRAT-8 IVF2 coarse probe (N={n}, {qcount} queries, top-{TOP_K}, IVF_K from env) ===\n  \
         baseline (probe OFF): recall@{TOP_K}={recall_base:.4}, range_gets/q={gets_base}, bytes/q={bytes_base}\n  \
         probe    (ON):        recall@{TOP_K}={recall_probe:.4}, range_gets/q={gets_probe}, bytes/q={bytes_probe}, \
         probes_recorded={}",
        probe_trace.len()
    );
    if !probe_trace.is_empty() {
        let cells_total: u64 = probe_trace.iter().map(|t| t.0).sum();
        let cells_probed: u64 = probe_trace.iter().map(|t| t.1).sum();
        let fetch_rounds: u64 = probe_trace.iter().map(|t| t.3).sum();
        eprintln!(
            "  probe engagement: Σcells_total={cells_total} Σcells_probed={cells_probed} \
             Σfetch_rounds={fetch_rounds} over {} queries",
            probe_trace.len()
        );
    }

    assert!(
        measured_base > 0 && measured_probe > 0,
        "no queries measured"
    );
    // A non-empty trace is NOT evidence that the probe is doing its job: measured
    // at the default N=100_000 it engages on 17 of 1000 queries (1.7%) and
    // `!is_empty()` certifies that happily. The remaining 983 take the
    // whole-Region-A fallback — the GET budget the probe exists to avoid — so a
    // presence check would let a probe that is effectively off pass as engaged.
    // Assert a proportion instead. See TD-IVF2ENGAGE-1 for why this currently
    // fails at N=100_000 while passing at N=20_000.
    let engaged = probe_trace.len();
    let engagement_floor = (qcount + 1) / 2; // > 50% of measured queries
    assert!(
        engaged >= engagement_floor,
        "coarse probe engaged on only {engaged}/{qcount} queries (floor {engagement_floor});          the rest fell back to the whole-Region-A scan — see TD-IVF2ENGAGE-1"
    );
    assert!(
        recall_probe >= recall_base - recall_drop,
        "probe recall@{TOP_K} = {recall_probe:.4} dropped > {recall_drop} vs baseline {recall_base:.4}"
    );
    assert!(
        bytes_probe < bytes_base,
        "probe bytes/q = {bytes_probe} must read fewer than baseline {bytes_base}"
    );
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_WRITE_A0_TRAIN");
        std::env::set_var("PROXIMADB_PAX_READ_COARSE_PROBE", "0");
        std::env::remove_var("PROXIMADB_TRACE_GETS");
    }
}

// ============================================================================
// TD-FPRUNE-1 flip evidence — the FILTERED leg (adversarial-review finding).
//
// Zero tests exercised the GATED DISPATCH before this: every filtered-cascade
// test called `pax_filtered_row_allow` / `rabitq_search_segment_coalesced_allowed`
// directly, bypassing the `search/mod.rs` gate, so a default-ON flip had no CI
// coverage. This leg runs the real `search_vectors_unified` WITH a filter and
// `PROXIMADB_PAX_FILTERED_CASCADE=1` (+ footer stats at write) and asserts:
//   (a) filtered recall@10 vs FILTERED brute-force ground truth >= floor,
//   (b) every returned id is in the filtered partition (predicate holds
//       through the full dispatch + cascade allow-set stack),
//   (c) every hit carries its metadata (the top-k rehydration — without it
//       the cascade silently returns id+score only, downgrading the exact
//       path's with-metadata response shape).
// The unfiltered arms above and their `!has_filter` invariant are untouched.
// ============================================================================

const FILTERED_PARTITIONS: u32 = 8;

fn partition_tag(i: u32) -> String {
    format!("p{}", i % FILTERED_PARTITIONS)
}

async fn run_sift_filtered_cascade_ratchet() {
    use proximadb::proto::proximadb_v1::sql_value;
    let base_path = match dataset_path("sift_base.fvecs") {
        Some(p) => p,
        None => {
            assert!(
                !dataset_required(),
                "required SIFT corpus is missing (PROXIMADB_RECALL_DATASET_REQUIRED=1)"
            );
            eprintln!("skipping filtered ratchet: no SIFT dataset");
            return;
        }
    };
    let floor = std::env::var("PROXIMADB_SIFT_RECALL_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.90);

    let n_filter = std::env::var("PROXIMADB_SIFT_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10_000);
    let max_queries = std::env::var("PROXIMADB_SIFT_QUERIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100);

    let engine = SstEngine::new().await.unwrap();
    let tmp = TempDir::new().unwrap();
    let storage_base = format!("file://{}", tmp.path().join("sift_filtered").display());
    let collection = collection("sift_filtered_ratchet", storage_base);

    // Ingest with a deterministic partition tag in metadata (→ record props).
    let base = read_vec_records_f32(&base_path, Some(n_filter)).expect("read sift_base.fvecs");
    let n = base.len();
    let mut batch: Vec<VectorRecord> = Vec::with_capacity(BATCH_SIZE);
    for (i, v) in base.iter().enumerate() {
        let i = i as u32;
        let mut r = vector_record(i, v.clone());
        r.metadata.insert(
            "partition".to_string(),
            proximadb::proto::proximadb_v1::SqlValue {
                value: Some(sql_value::Value::StringValue(partition_tag(i))),
            },
        );
        batch.push(r);
        if batch.len() == BATCH_SIZE {
            let b = std::mem::take(&mut batch);
            flush_batch(&engine, &collection, b).await;
        }
    }
    if !batch.is_empty() {
        flush_batch(&engine, &collection, batch).await;
    }
    eprintln!("filtered ratchet: flushed {n} tagged base vectors");

    let query_path = dataset_path("sift_query.fvecs").expect("query corpus");
    let queries: Vec<Vec<f32>> =
        read_vec_records_f32(&query_path, Some(max_queries)).expect("read sift_query.fvecs");

    let mut recall_sum = 0.0f64;
    let mut measured = 0usize;
    for (qi, query) in queries.iter().enumerate() {
        let p = partition_tag(qi as u32);
        // Filtered brute-force GT: top-TOP_K among ONLY partition-p rows.
        let mut candidates: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .filter(|(i, _)| partition_tag(*i as u32) == p)
            .map(|(i, v)| {
                let d: f32 = v
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i, d)
            })
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: Vec<String> = candidates
            .iter()
            .take(TOP_K)
            .map(|(i, _)| vid(*i as u32))
            .collect();

        // The REAL dispatch, with the filter the gate keys on.
        let ctx = StorageQueryContext {
            search_params: Arc::new(SearchParams {
                query_vectors: Some(vec![query.clone()]),
                top_k: Some(TOP_K as u16),
                distance_metric: Some(DistanceMetric::Euclidean),
                search_mode: SearchMode::Approximate { nprobe: None },
                filter_expression: Some(
                    proximadb_filter_expression::FilterExpression::Comparison {
                        field: "partition".to_string(),
                        operator: proximadb_filter_expression::ComparisonOperator::Equals,
                        value: serde_json::json!(p),
                    },
                ),
                block_prune: BlockPruneConfig {
                    radius_k: 0.0,
                    force_exact: false,
                    mode: BlockPruneMode::Ratio,
                    ratio: 1.0,
                    min_keep: 1,
                    max_keep: 0,
                    min_blocks_override: Some(0),
                },
                ..Default::default()
            }),
            collection: Arc::new(collection.clone()),
            metadata: StorageQueryMetadata {
                collection_id: collection.id.clone(),
                ..Default::default()
            },
            user_context: None,
            tenant_context: None,
        };
        let results = engine
            .search_vectors_unified(&ctx)
            .await
            .expect("filtered search through the unified dispatch must succeed");
        assert!(!results.is_empty(), "partition {p} must yield hits");

        let gt_set: std::collections::HashSet<&str> = gt.iter().map(|s| s.as_str()).collect();
        let mut hits_in_gt = 0usize;
        for r in &results {
            // (b) predicate: the hit's partition is the filtered one.
            let idx: u32 =
                r.id.trim_start_matches('v')
                    .parse()
                    .expect("hit id encodes its base index");
            assert_eq!(
                partition_tag(idx),
                p,
                "hit {} leaked across the filter boundary",
                r.id
            );
            // (c) rehydration: metadata came back with the hit.
            let tag = match r.metadata.get("partition") {
                Some(proximadb_data_model::ProximaValue::String(s)) => s.as_str(),
                other => panic!("partition metadata must rehydrate as a string, got {other:?}"),
            };
            assert_eq!(
                tag, p,
                "cascade hit must carry its rehydrated partition metadata"
            );
            if gt_set.contains(r.id.as_str()) {
                hits_in_gt += 1;
            }
        }
        recall_sum += hits_in_gt as f64 / TOP_K as f64;
        measured += 1;
    }
    let recall = recall_sum / measured.max(1) as f64;
    eprintln!(
        "SIFT FILTERED cascade recall@10 = {recall:.4} over {measured} queries \
         (N={n}, floor={floor}, partitions={FILTERED_PARTITIONS})"
    );
    assert!(
        recall >= floor,
        "filtered cascade recall {recall:.4} below floor {floor} — the default-ON \
         flip is blocked until this holds"
    );
}

/// The filtered leg of the ratchet — see the module comment above. Pins BOTH
/// gates ON (the flip configuration) plus the same write geometry as the
/// baseline arm; removes them at the end so the unfiltered arms are unaffected
/// under nextest's process-per-test isolation and tolerate plain `cargo test`.
#[tokio::test]
async fn sift_pax_filtered_cascade_recall_ratchet() {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_PAX_F32_TIER", "1");
        std::env::set_var("PROXIMADB_PAX_WRITE_RG_LAYOUT", "0");
        // The cascade gate is default-ON since the 2026-08-31 flip — this leg
        // runs the shipped configuration; only footer stats (still default-OFF)
        // is pinned so the write side carries the shred field map.
        std::env::set_var("PROXIMADB_PAX_FOOTER_STATS", "1");
    }
    run_sift_filtered_cascade_ratchet().await;
    unsafe {
        std::env::remove_var("PROXIMADB_PAX_FOOTER_STATS");
    }
}
