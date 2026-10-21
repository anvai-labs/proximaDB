/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

//! Warm-path profile bench — phase-by-phase breakdown of a single
//! search query.
//!
//! The e2e bench (`bench_vector_object_economy_e2e.rs`) measures
//! end-to-end warm latency but doesn't say WHERE the time goes. This
//! bench captures per-phase timings via the
//! `target: "sst_warm_phase"` tracing events emitted by
//! `SstEngine::fallback_to_direct_search` and reports per-phase
//! percentile distributions.
//!
//! Phases:
//!
//! * `discovery` — listing SSTable files (+ optional centroid prune)
//! * `per_file_scan` — sum across all files of the actual ANN scan
//! * `topk_merge` — bounded priority queue merge across file results
//! * `result_filter` — final include-vectors/include-metadata pass
//!
//! Run:
//!
//! ```bash
//! BENCH_VECTORS=100000 BENCH_DIM=768 BENCH_QUERIES=50 \
//!   cargo bench --bench bench_warm_path_profile
//! ```

use proximadb::compute::distance_computation::{DistanceMetric, UnifiedDistanceCompute};
use proximadb::core::search::{BlockPruneConfig, SearchMode, SearchParams};
use proximadb::proto::proximadb_v1::{Collection, CollectionConfig};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageEngine};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tempfile::TempDir;
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;

// ────────────────────────────────────────────────────────────────────────
// Custom layer: capture `sst_warm_phase` events into a sample store
// ────────────────────────────────────────────────────────────────────────

/// Per-phase samples in microseconds. Keyed by the `phase` field on
/// the tracing event (e.g. "discovery", "per_file_scan").
#[derive(Default)]
struct PhaseStore {
    samples: HashMap<String, Vec<u64>>,
}

impl PhaseStore {
    fn record(&mut self, phase: String, elapsed_us: u64) {
        self.samples.entry(phase).or_default().push(elapsed_us);
    }

    fn drain_and_collect(&mut self) -> HashMap<String, Vec<u64>> {
        std::mem::take(&mut self.samples)
    }
}

/// Tracing layer that captures events emitted under the
/// `sst_warm_phase` target. Extracts the `phase` (str) and
/// `elapsed_us` (u64) fields and stores them.
struct WarmPhaseLayer {
    store: Arc<Mutex<PhaseStore>>,
}

/// Visitor that pulls `phase` + `elapsed_us` out of an event.
struct PhaseVisitor {
    phase: Option<String>,
    elapsed_us: Option<u64>,
}

impl tracing::field::Visit for PhaseVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "phase" {
            self.phase = Some(value.to_string());
        }
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if field.name() == "elapsed_us" {
            self.elapsed_us = Some(value);
        }
    }

    // The remaining types are required by the trait but unused for
    // this bench. tracing's `info!(phase = "x")` literals become
    // `record_str`; integer literals become `record_u64` (or i64).
    fn record_debug(&mut self, _f: &tracing::field::Field, _v: &dyn std::fmt::Debug) {}
    fn record_i64(&mut self, _f: &tracing::field::Field, _v: i64) {}
    fn record_bool(&mut self, _f: &tracing::field::Field, _v: bool) {}
}

impl<S> Layer<S> for WarmPhaseLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let target = event.metadata().target();
        // Diagnostic stream: AXIS decision points and HMGI config
        // dumps. Printed to stderr verbatim with a formatted dump of
        // the event fields. Independent of the phase-timing store.
        if target == "axis_diag" {
            let mut visitor = DiagVisitor::default();
            event.record(&mut visitor);
            eprintln!("[axis_diag] {}", visitor.into_summary());
            return;
        }
        if target != "sst_warm_phase" {
            return;
        }
        let mut visitor = PhaseVisitor {
            phase: None,
            elapsed_us: None,
        };
        event.record(&mut visitor);
        if let (Some(phase), Some(elapsed_us)) = (visitor.phase, visitor.elapsed_us)
            && let Ok(mut store) = self.store.lock()
        {
            store.record(phase, elapsed_us);
        }
    }
}

/// Visitor that flattens any tracing event into a sorted key=value
/// dump. Used for axis_diag events so the bench operator can see what
/// each decision-point logged without ad-hoc per-event parsers.
#[derive(Default)]
struct DiagVisitor {
    fields: Vec<(String, String)>,
}

impl DiagVisitor {
    fn into_summary(self) -> String {
        let mut entries = self.fields;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl tracing::field::Visit for DiagVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.fields
            .push((field.name().to_string(), format!("{:?}", value)));
    }
}

// ────────────────────────────────────────────────────────────────────────
// Bench config (mirrors the e2e bench)
// ────────────────────────────────────────────────────────────────────────

struct BenchConfig {
    vector_count: usize,
    dimension: usize,
    top_k: usize,
    warm_runs: usize,
    pre_warm_runs: usize,
    /// When true, use `SearchMode::Approximate { nprobe: None }`
    /// (auto-calc sqrt). Default false (Exact) preserves the
    /// baseline behaviour the e2e bench measures.
    approx_mode: bool,
    /// When true, sets `SearchParams::enable_vectorized_execution`,
    /// routing the SST reader through its SIMD vectorized path
    /// (TD-041). Default false uses the scalar path.
    vectorized: bool,
    /// When true, construct an `AxisManager` with default HNSW
    /// config, register it globally so the SstEngine picks it up,
    /// and insert all records into AXIS after flush. This routes
    /// queries through `execute_orchestrated_search` →
    /// `axis_manager.query(...)` (HNSW) instead of the linear
    /// fallback. Setup time grows by the HNSW build cost
    /// (~O(n log n) NEON distance computes for build).
    ///
    /// Kept as a legacy boolean for backward compatibility; the
    /// `index` field is the new richer surface that subsumes it.
    /// When `index = Flat`, axis is forced off regardless.
    axis: bool,
    /// Which index type to exercise (`BENCH_INDEX`). Drives the
    /// AxisManager wiring + strategy selection:
    ///   * `flat`  — no AXIS registered; engine falls back to direct
    ///               brute-force scan (always 100% recall, slowest).
    ///   * `hnsw`  — AXIS registered, default collection-scoped HNSW
    ///               (post-`b3985b59c` default). Inserts route through
    ///               `insert_into_hnsw` which honors the collection
    ///               metric.
    ///   * `hmgi`  — AXIS registered; `enable_hmgi` called BEFORE
    ///               inserts so insertion routes through HMGI
    ///               partitions. Multi-modality partitioning over
    ///               HNSW (see arXiv:2510.10123).
    ///   * `ivf`   — AXIS registered; `update_collection_strategy`
    ///               injects an IVF spec so inserts route through
    ///               `insert_into_ivf`. Coarse-quantizer + per-cell
    ///               probe.
    index: IndexType,
    /// Distance metric for the collection. Configurable via
    /// `BENCH_METRIC=cosine|euclidean|dotproduct`. Default: cosine
    /// (matches the SaaS default the product ships with).
    ///
    /// The bench drives the SAME metric through both the exact path
    /// (engine's `UnifiedDistanceCompute`) and the AXIS path (via
    /// collection config + shared_collection_cache wiring). This
    /// makes per-metric QA validation possible — recall mismatch
    /// then localises to a specific (metric, algorithm) pair instead
    /// of a global "AXIS is broken" verdict.
    metric: DistanceMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexType {
    Flat,
    Hnsw,
    Hmgi,
    Ivf,
}

impl IndexType {
    fn label(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Hnsw => "hnsw",
            Self::Hmgi => "hmgi",
            Self::Ivf => "ivf",
        }
    }

    fn use_axis(&self) -> bool {
        !matches!(self, Self::Flat)
    }
}

fn parse_index(s: Option<&str>) -> IndexType {
    match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("hnsw") => IndexType::Hnsw,
        Some("flat") | Some("brute") | Some("brute_force") => IndexType::Flat,
        Some("hmgi") => IndexType::Hmgi,
        Some("ivf") => IndexType::Ivf,
        Some(other) => {
            eprintln!(
                "[bench warn] unknown BENCH_INDEX={:?}, falling back to hnsw",
                other
            );
            IndexType::Hnsw
        }
    }
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            vector_count: env_usize("BENCH_VECTORS", 100_000),
            dimension: env_usize("BENCH_DIM", 768),
            top_k: env_usize("BENCH_TOPK", 10),
            warm_runs: env_usize("BENCH_QUERIES", 50),
            // One pre-warm pass to populate caches the same way the
            // e2e bench does — turbopuffer-style "warm" is reached
            // after the first query.
            pre_warm_runs: 1,
            approx_mode: env_bool("BENCH_APPROX", false),
            vectorized: env_bool("BENCH_VECTORIZED", false),
            // If BENCH_INDEX is supplied, it wins over the legacy
            // BENCH_AXIS bool — axis is implied by the index type
            // (Flat → no axis; everything else → axis).
            axis: match std::env::var("BENCH_INDEX") {
                Ok(s) => parse_index(Some(s.as_str())).use_axis(),
                Err(_) => env_bool("BENCH_AXIS", false),
            },
            metric: parse_metric(std::env::var("BENCH_METRIC").ok().as_deref()),
            index: parse_index(std::env::var("BENCH_INDEX").ok().as_deref()),
        }
    }
}

/// Resolve `BENCH_METRIC` to a `DistanceMetric`. Defaults to Cosine
/// to match the SaaS default. Unknown values fall back to Cosine
/// with a stderr warning (so a typo doesn't silently change semantics).
fn parse_metric(s: Option<&str>) -> DistanceMetric {
    match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("cosine") => DistanceMetric::Cosine,
        Some("euclidean") | Some("l2") => DistanceMetric::Euclidean,
        Some("dotproduct") | Some("dot") | Some("ip") | Some("innerproduct") => {
            DistanceMetric::DotProduct
        }
        Some("manhattan") | Some("l1") => DistanceMetric::Manhattan,
        Some(other) => {
            eprintln!(
                "[bench warn] unknown BENCH_METRIC={:?}, falling back to Cosine",
                other
            );
            DistanceMetric::Cosine
        }
    }
}

/// Proto enum value the collection config carries for this metric.
/// Mirrors `proximadb-vector::distance::conversion::internal_distance_to_proto`
/// but kept inline so the bench compiles without depending on that
/// crate's surface directly.
fn metric_proto_code(m: DistanceMetric) -> i32 {
    match m {
        DistanceMetric::Cosine => 1,
        DistanceMetric::Euclidean => 2,
        DistanceMetric::DotProduct => 3,
        DistanceMetric::Hamming => 4,
        DistanceMetric::Manhattan => 5,
        _ => 1, // unspecified-ish; bench only exercises the common ones
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ────────────────────────────────────────────────────────────────────────
// Setup
// ────────────────────────────────────────────────────────────────────────

struct SetupResult {
    warm_engine: Arc<SstEngine>,
    warm_collection: Arc<Collection>,
    insert_ms: u128,
    flush_ms: u128,
    /// Time spent building the AXIS HNSW index post-flush, or 0
    /// when `BENCH_AXIS` is not set.
    axis_build_ms: u128,
    _temp_dir: TempDir,
}

impl SetupResult {
    async fn run(cfg: &BenchConfig) -> Self {
        let temp_dir = TempDir::new().expect("tempdir");
        let collection = make_collection(&temp_dir, cfg.dimension, cfg.metric);
        let vectors = synthetic_records(&collection.id, cfg.vector_count, cfg.dimension);

        // AXIS wiring: gated by `cfg.axis` (driven by BENCH_INDEX
        // when supplied, falling back to BENCH_AXIS). Builds the
        // AxisManager and registers it globally *before* engine
        // construction so `get_sst_axis_manager()` picks it up
        // during `SstEngine::new_with_config`.
        //
        // **Metric correctness**: AxisManager defaults to DotProduct
        // when it can't find the collection's distance metric (see
        // manager.rs:830). Without a shared_collection_cache, AXIS
        // and the exact path would use different metrics → recall
        // measurement is meaningless. Inject a minimal cache so AXIS
        // reads the same metric the collection was constructed with.
        //
        // Per-index-type branching also happens here:
        //   * Flat:  AXIS not registered (use_axis() returns false).
        //   * HNSW:  default — `insert_into_hnsw` runs.
        //   * HMGI:  explicit `enable_hmgi` call before any insert
        //            so `is_hmgi_enabled == true` and inserts route
        //            through HMGI partitions.
        //   * IVF:   explicit `update_collection_strategy` with an
        //            IVF spec so `insert_dense_vector_index`
        //            dispatches to `insert_into_ivf`.
        if cfg.axis {
            let mut axis_manager_inner =
                proximadb::index::AxisManager::new(proximadb::index::AxisConfig::default())
                    .await
                    .expect("axis manager");

            let shared_cache = std::sync::Arc::new(dashmap::DashMap::new());
            shared_cache.insert(
                collection.id.clone(),
                std::sync::Arc::new(collection.clone()),
            );
            axis_manager_inner.set_shared_collection_cache(shared_cache);

            // Per-index-type strategy injection BEFORE wrapping in Arc.
            // ensure_collection_strategy creates a default strategy
            // lazily on first insert if none exists; we override
            // here so the chosen algorithm is used from the start.
            match cfg.index {
                IndexType::Hmgi => {
                    use proximadb::index::axis::management::manager as axis_mgr;
                    use proximadb::index::axis::types::{
                        Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
                    };
                    // We must drive enable_hmgi via the SAME instance
                    // we'll later retrieve via get_sst_axis_manager,
                    // so wrap in Arc first.
                    let axis_arc = std::sync::Arc::new(axis_manager_inner);
                    let oid: u64 = {
                        use std::hash::{Hash, Hasher};
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        collection.id.hash(&mut h);
                        h.finish()
                    };
                    axis_arc
                        .enable_hmgi(&collection.id, None, oid)
                        .await
                        .expect("enable_hmgi");

                    // **HMGI ef plumbing**: previously the HMGI cell
                    // only called enable_hmgi and let
                    // ensure_collection_strategy → recommend_strategy
                    // pick the default (AxisHnswConfig::default().ef
                    // = 50). That meant BENCH_HNSW_EF was ignored on
                    // the HMGI path — measured at 10K, HMGI partitions
                    // logged `ef_search=50` even when BENCH_HNSW_EF
                    // was set to 500. Now we install an explicit
                    // HNSW spec so insert_hmgi's
                    // get_or_create_partition gets the right ef
                    // (the spec flows through via insert_hmgi's
                    // `Option<&IndexAlgorithm>` parameter committed
                    // in 476dc951a).
                    let ef_search = env_usize("BENCH_HNSW_EF", 100) as u32;
                    let m = env_usize("BENCH_HNSW_M", 16) as u32;
                    let ef_construction = env_usize("BENCH_HNSW_EF_CONSTRUCTION", 200) as u32;
                    let hnsw_spec = IndexSpecification::new(
                        Data::DenseVector {
                            dimension: cfg.dimension,
                        },
                        IndexAlgorithm::HNSW {
                            m,
                            ef_construction,
                            ef_search,
                            max_elements: 1_000_000,
                        },
                    );
                    let strategy = IndexSelectionStrategy {
                        indexes: vec![hnsw_spec],
                        routing_rules: vec![],
                    };
                    axis_arc
                        .update_collection_strategy(&collection.id, strategy)
                        .await
                        .expect("update_collection_strategy HMGI");

                    let _ = axis_mgr::AxisManager::new; // silence unused-import lint
                    proximadb::storage::engines::sst::core::set_sst_axis_manager(axis_arc);
                }
                IndexType::Ivf => {
                    use proximadb::index::axis::types::{
                        Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
                    };
                    // Adaptive IVF sizing (textbook: nlist = sqrt(N),
                    // nprobe = sqrt(nlist) for ~95% recall trade-off):
                    //
                    //   N=10K   → nlist=100, nprobe=10
                    //   N=25K   → nlist=158, nprobe=12
                    //   N=100K  → nlist=316, nprobe=17
                    //   N=1M    → nlist=1000, nprobe=31
                    //
                    // Floored at nlist=100 (sub-1K collections don't
                    // benefit from finer partitioning and k-means
                    // can't train with fewer clusters than samples).
                    // Override via BENCH_IVF_NLIST / BENCH_IVF_NPROBE.
                    let sqrt_n = (cfg.vector_count as f64).sqrt() as u32;
                    let auto_nlist = sqrt_n.max(100);
                    let auto_nprobe = (auto_nlist as f64).sqrt().ceil() as u32;
                    let nlist = env_usize("BENCH_IVF_NLIST", auto_nlist as usize) as u32;
                    let nprobe = env_usize("BENCH_IVF_NPROBE", auto_nprobe as usize) as u32;
                    let ivf_spec = IndexSpecification::new(
                        Data::DenseVector {
                            dimension: cfg.dimension,
                        },
                        IndexAlgorithm::IVF {
                            nlist,
                            nprobe,
                            quantizer: None,
                        },
                    );
                    let strategy = IndexSelectionStrategy {
                        indexes: vec![ivf_spec],
                        routing_rules: vec![],
                    };
                    let axis_arc = std::sync::Arc::new(axis_manager_inner);
                    axis_arc
                        .update_collection_strategy(&collection.id, strategy)
                        .await
                        .expect("update_collection_strategy IVF");
                    proximadb::storage::engines::sst::core::set_sst_axis_manager(axis_arc);
                }
                IndexType::Hnsw => {
                    // Force an explicit HNSW strategy. WITHOUT this,
                    // the adaptive engine's `recommend_strategy` for
                    // a dense-vector collection returns IVF and the
                    // bench silently exercises IVF instead of HNSW.
                    //
                    // `BENCH_HNSW_EF` (default 100) sets the search
                    // beam width. Strategy spec → AxisHnswConfig.ef
                    // wiring is end-to-end since the
                    // `IndexAlgorithm::HNSW.ef_search`-plumbing
                    // commit; no env override into HNSW code needed.
                    use proximadb::index::axis::types::{
                        Data, IndexAlgorithm, IndexSelectionStrategy, IndexSpecification,
                    };
                    let ef_search = env_usize("BENCH_HNSW_EF", 100) as u32;
                    let m = env_usize("BENCH_HNSW_M", 16) as u32;
                    let ef_construction = env_usize("BENCH_HNSW_EF_CONSTRUCTION", 200) as u32;
                    let hnsw_spec = IndexSpecification::new(
                        Data::DenseVector {
                            dimension: cfg.dimension,
                        },
                        IndexAlgorithm::HNSW {
                            m,
                            ef_construction,
                            ef_search,
                            max_elements: 1_000_000,
                        },
                    );
                    let strategy = IndexSelectionStrategy {
                        indexes: vec![hnsw_spec],
                        routing_rules: vec![],
                    };
                    let axis_arc = std::sync::Arc::new(axis_manager_inner);
                    axis_arc
                        .update_collection_strategy(&collection.id, strategy)
                        .await
                        .expect("update_collection_strategy HNSW");
                    proximadb::storage::engines::sst::core::set_sst_axis_manager(axis_arc);
                }
                IndexType::Flat => {
                    // Flat never reaches this branch (cfg.axis = false).
                    proximadb::storage::engines::sst::core::set_sst_axis_manager(
                        std::sync::Arc::new(axis_manager_inner),
                    );
                }
            }
        }

        let insert_start = Instant::now();
        let fs = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("filesystem factory"),
        );
        let dist = Arc::new(UnifiedDistanceCompute::default());
        let engine = SstEngine::new_with_config(Default::default(), fs.clone(), dist.clone())
            .await
            .expect("sst engine");
        let flush_start = Instant::now();
        let insert_ms = (flush_start - insert_start).as_millis();

        let params = FlushParameters {
            collection_id: Some(collection.id.clone()),
            collection_config: Some(collection.clone()),
            vector_records: vectors.clone(),
            force: true,
            synchronous: true,
            ..Default::default()
        };
        engine.flush(params).await.expect("flush");
        let flush_ms = flush_start.elapsed().as_millis();

        // Slice B: build the HNSW after flush so the AXIS index
        // mirrors the on-disk data. The engine's
        // `execute_orchestrated_search` will route warm queries
        // through `axis_manager.query(...)` once records are
        // indexed.
        let axis_build_ms = if cfg.axis {
            let build_start = Instant::now();
            if let Some(axis_manager) =
                proximadb::storage::engines::sst::core::get_sst_axis_manager()
            {
                for record in &vectors {
                    axis_manager
                        .insert_record(&collection.id, record)
                        .await
                        .expect("axis insert");
                }
            }
            build_start.elapsed().as_millis()
        } else {
            0
        };

        Self {
            warm_engine: Arc::new(engine),
            warm_collection: Arc::new(collection),
            insert_ms,
            flush_ms,
            axis_build_ms,
            _temp_dir: temp_dir,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = BenchConfig::from_env();

    // Install the warm-phase capturing layer BEFORE setup so any
    // events emitted during setup (we don't expect any, but doesn't
    // hurt) are still in scope. We reset the store after setup to
    // ensure only timed queries contribute to the report.
    let store = Arc::new(Mutex::new(PhaseStore::default()));
    let layer = WarmPhaseLayer {
        store: store.clone(),
    };
    Registry::default().with(layer).init();

    println!("================================================================================");
    println!("   ProximaDB — Warm-Path Profile (per-phase breakdown)");
    println!("================================================================================");
    println!();
    println!("Configuration:");
    println!("  vectors:    {}", cfg.vector_count);
    println!("  dimension:  {}", cfg.dimension);
    println!("  top_k:      {}", cfg.top_k);
    println!("  warm_runs:  {}", cfg.warm_runs);
    println!(
        "  mode:       {}",
        if cfg.approx_mode {
            "Approximate (nprobe=auto)"
        } else {
            "Exact (default)"
        }
    );
    println!(
        "  vectorized: {} (enable_vectorized_execution)",
        cfg.vectorized
    );
    println!("  index:      {} (BENCH_INDEX)", cfg.index.label());
    println!(
        "  axis:       {} (derived from BENCH_INDEX/BENCH_AXIS)",
        cfg.axis
    );
    println!("  metric:     {:?} (BENCH_METRIC)", cfg.metric);
    println!();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        let setup = SetupResult::run(&cfg).await;
        if cfg.axis {
            println!(
                "Setup: insert {} ms, flush {} ms, axis_build {} ms",
                setup.insert_ms, setup.flush_ms, setup.axis_build_ms
            );
        } else {
            println!(
                "Setup: insert {} ms, flush {} ms",
                setup.insert_ms, setup.flush_ms
            );
        }

        // Pre-warm + reset
        for _ in 0..cfg.pre_warm_runs {
            run_one_query(&setup, &cfg).await;
        }
        if let Ok(mut s) = store.lock() {
            s.drain_and_collect();
        }

        // Recall measurement pass — only meaningful when AXIS is on
        // (otherwise both paths converge on the same direct scan and
        // recall is trivially 1.0). 20 queries × top_k is the sample
        // size; small enough that the ~62 ms-per-exact-query cost is
        // bounded at ~1.5 s. Larger samples reduce variance but the
        // mean/min/max we report stabilizes quickly.
        let recall: Option<RecallStats> = if cfg.axis {
            let recall_samples = 20usize;
            println!();
            println!(
                "📐 Measuring recall@{} over {} queries",
                cfg.top_k, recall_samples
            );
            let stats = measure_recall(&setup, &cfg, recall_samples).await;
            Some(stats)
        } else {
            None
        };

        // **Important** — drain the phase store after the recall
        // pass. Recall measurement calls `fallback_to_direct_search`
        // (which fires `discovery` / `per_file_scan` / `topk_merge`
        // / `result_filter` events) and `search_vectors_unified`
        // (which fires `axis_query` / `axis_result_convert` when
        // AXIS is wired). If we don't drain, the warm-pass report
        // mixes both populations and the per-phase means are
        // misleading.
        if let Ok(mut s) = store.lock() {
            s.drain_and_collect();
        }

        println!();
        println!(
            "📊 Capturing per-phase timings across {} warm queries",
            cfg.warm_runs
        );

        let mut total_us = Vec::with_capacity(cfg.warm_runs);
        for _ in 0..cfg.warm_runs {
            let t = run_one_query(&setup, &cfg).await;
            total_us.push(t);
        }

        let captured = store.lock().unwrap().drain_and_collect();
        print_report(&cfg, &total_us, &captured);
        if let Some(stats) = recall {
            print_recall_report(&cfg, &stats);
        }
    });
}

// ────────────────────────────────────────────────────────────────────────
// Recall measurement
// ────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct RecallStats {
    /// ID-overlap recall@k per query — fraction of the AXIS top_k
    /// IDs that ALSO appear in the **default-flat** top_k.
    /// Strictest measure; sensitive to ties.
    id_overlap: Vec<f64>,
    /// Score-threshold recall@k per query — fraction of the AXIS
    /// top_k whose true metric score is >= default-flat's k-th best
    /// score. Independent of how AXIS reports score units.
    score_threshold: Vec<f64>,
    /// **Shadow recall**: default-flat top_k vs true brute-force
    /// (force_exact=true, no block pruning). When this is < 1.0 it
    /// means default-flat itself is missing records the true exact
    /// search would have returned — i.e. the bench's regular "exact"
    /// baseline is approximate at this scale. Disclosed so callers
    /// can multiply through to get axis_recall vs true ground truth.
    flat_default_vs_force_exact: Vec<f64>,
    /// **Shadow recall**: AXIS top_k vs true brute-force. This is
    /// the absolute recall number — what the customer would observe
    /// vs a no-pruning baseline. = id_overlap × flat_default_vs_force_exact
    /// approximately, with cross-term corrections.
    axis_vs_force_exact: Vec<f64>,
    /// How many of the queries returned identical top_k id sets
    /// (vs default-flat).
    perfect_matches: usize,
}

/// True "higher = better" score between query and record under the
/// configured metric. Independent of how either path reports its
/// score, so recall measurements are not corrupted by score-unit
/// normalization disparities.
fn true_score(metric: DistanceMetric, query: &[f32], v: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for j in 0..query.len() {
                dot += query[j] * v[j];
                na += query[j] * query[j];
                nb += v[j] * v[j];
            }
            if na == 0.0 || nb == 0.0 {
                return f32::NEG_INFINITY;
            }
            dot / (na.sqrt() * nb.sqrt())
        }
        DistanceMetric::DotProduct => {
            let mut dot = 0.0f32;
            for j in 0..query.len() {
                dot += query[j] * v[j];
            }
            dot
        }
        DistanceMetric::Euclidean => {
            // L2 distance — invert so higher = more similar.
            let mut sq = 0.0f32;
            for j in 0..query.len() {
                let d = query[j] - v[j];
                sq += d * d;
            }
            -sq.sqrt()
        }
        DistanceMetric::Manhattan => {
            let mut s = 0.0f32;
            for j in 0..query.len() {
                s += (query[j] - v[j]).abs();
            }
            -s
        }
        _ => {
            // For exotic metrics, fall back to cosine — the bench
            // surfaces this via the stderr warning in parse_metric.
            true_score(DistanceMetric::Cosine, query, v)
        }
    }
}

/// For each of `n_queries` random queries, runs both the exact path
/// (`fallback_to_direct_search`, brute force) and the AXIS HNSW path
/// (`search_vectors_unified` orchestrated). Computes the set
/// intersection of returned IDs and reports recall@k.
async fn measure_recall(setup: &SetupResult, cfg: &BenchConfig, n_queries: usize) -> RecallStats {
    let collection_id = setup.warm_collection.id.clone();
    // search_vectors_unified resolves the storage URL via
    // `ctx.collection_storage_path()` which appends `{collection_id}/data`
    // to the base location. We must do the same when calling
    // fallback_to_direct_search directly, otherwise its file discovery
    // finds zero SST files in the bare base_location.
    let storage_url = {
        let probe_params = Arc::new(SearchParams {
            vector: Some(vec![0.0; cfg.dimension]),
            top_k: Some(1),
            ..Default::default()
        });
        let probe_ctx = StorageQueryContext::new(probe_params, Arc::clone(&setup.warm_collection));
        probe_ctx
            .collection_storage_path()
            .expect("collection storage path resolves")
    };
    let distance_metric = cfg.metric;

    // Capture the original records by id so we can re-derive any
    // candidate's vector for ground-truth manual cosine. This is
    // tiny (10K × 768 fp32 ≈ 30 MB at our bench scale).
    use std::collections::HashMap;
    let record_lookup: HashMap<String, Vec<f32>> = (0..cfg.vector_count)
        .map(|i| {
            let oid = format!("{}_{:08}", setup.warm_collection.id, i);
            let v: Vec<f32> = (0..cfg.dimension)
                .map(|j| pseudo_random_f32(i as u64 + 1, j))
                .collect();
            (oid, v)
        })
        .collect();

    let mut id_overlap = Vec::with_capacity(n_queries);
    let mut score_threshold = Vec::with_capacity(n_queries);
    let mut flat_default_vs_force_exact = Vec::with_capacity(n_queries);
    let mut axis_vs_force_exact = Vec::with_capacity(n_queries);
    let mut perfect_matches = 0usize;

    // BlockPruneConfig with force_exact=true bypasses the centroid +
    // sqrt-based block pruning that the default flat path uses at
    // scale (~100+ blocks). This is the "true" brute-force baseline
    // we compare both default-flat and AXIS against, so the recall
    // numbers we report are absolute (vs no-pruning ground truth)
    // and not just relative to ProximaDB's approximate flat path.
    let force_exact_block_prune = BlockPruneConfig {
        force_exact: true,
        ..Default::default()
    };

    for q_idx in 0..n_queries {
        // Vary the query per iteration so we don't measure recall on
        // the same vector 20 times. Use the same pseudo-random
        // generator as the records but with seeds that don't match
        // any record (records use seeds 1..=count; queries use
        // 10_000_000 + q_idx). This gives discriminative top-K
        // results — neighbours are random scattered points, not
        // a near-continuous sinusoid where many records tie.
        let query_vector: Vec<f32> = (0..cfg.dimension)
            .map(|j| pseudo_random_f32(10_000_000 + q_idx as u64, j))
            .collect();

        // **Default-flat baseline** (what the bench has always
        // compared against): SearchMode::Exact with the default
        // BlockPruneConfig. At small N this is true brute-force; at
        // 100K+ where flat has ≥100 blocks, the default config
        // applies sqrt-based centroid pruning and skips most blocks.
        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector.clone()),
            top_k: Some(cfg.top_k as u16),
            search_mode: SearchMode::Exact,
            ..Default::default()
        });
        let exact_ctx =
            StorageQueryContext::new(search_params.clone(), Arc::clone(&setup.warm_collection));
        let exact = setup
            .warm_engine
            .fallback_to_direct_search(
                &exact_ctx,
                &collection_id,
                &storage_url,
                &query_vector,
                cfg.top_k,
                distance_metric,
                None,
                true,
                true,
            )
            .await
            .expect("exact search");
        let exact_ids: std::collections::HashSet<String> =
            exact.iter().map(|r| r.id.clone()).collect();

        // **Force-exact shadow baseline** (true brute-force, no
        // block pruning): same call as `exact` above but with a
        // BlockPruneConfig that disables centroid + sqrt pruning.
        // Used to measure how much recall the default-flat path is
        // costing at scale.
        let force_exact_params = Arc::new(SearchParams {
            vector: Some(query_vector.clone()),
            top_k: Some(cfg.top_k as u16),
            search_mode: SearchMode::Exact,
            block_prune: force_exact_block_prune.clone(),
            ..Default::default()
        });
        let force_exact_ctx =
            StorageQueryContext::new(force_exact_params, Arc::clone(&setup.warm_collection));
        let force_exact = setup
            .warm_engine
            .fallback_to_direct_search(
                &force_exact_ctx,
                &collection_id,
                &storage_url,
                &query_vector,
                cfg.top_k,
                distance_metric,
                None,
                true,
                true,
            )
            .await
            .expect("force_exact search");
        let force_exact_ids: std::collections::HashSet<String> =
            force_exact.iter().map(|r| r.id.clone()).collect();

        // AXIS HNSW: same query, but routed through
        // execute_orchestrated_search.
        let axis_ctx = StorageQueryContext::new(search_params, Arc::clone(&setup.warm_collection));
        let axis = setup
            .warm_engine
            .search_vectors_unified(&axis_ctx)
            .await
            .expect("axis search");
        let axis_ids: std::collections::HashSet<String> =
            axis.iter().map(|r| r.id.clone()).collect();

        // Diagnostic: dump the first query's exact + axis IDs to
        // catch ID-format mismatches early (e.g. exact returns oid
        // strings but axis returns numeric vector_ids).
        if q_idx == 0 {
            // Helper: true cosine sim between query and one of our
            // synthetic records, re-derived from the bench seed.
            let true_cosine = |id: &str| -> f32 {
                let v = match record_lookup.get(id) {
                    Some(v) => v,
                    None => return f32::NAN,
                };
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for j in 0..cfg.dimension {
                    dot += query_vector[j] * v[j];
                    na += query_vector[j] * query_vector[j];
                    nb += v[j] * v[j];
                }
                dot / (na.sqrt() * nb.sqrt())
            };
            eprintln!(
                "[recall debug] q0 exact (top {}):  id / reported_score / true_cosine",
                exact.len().min(5)
            );
            for r in exact.iter().take(5) {
                eprintln!(
                    "    {} reported={:.6} true_cos={:.6}",
                    r.id,
                    r.score,
                    true_cosine(&r.id)
                );
            }
            eprintln!(
                "[recall debug] q0 axis  (top {}):  id / reported_score / true_cosine",
                axis.len().min(5)
            );
            for r in axis.iter().take(5) {
                eprintln!(
                    "    {} reported={:.6} true_cos={:.6}",
                    r.id,
                    r.score,
                    true_cosine(&r.id)
                );
            }
        }

        let intersection = exact_ids.intersection(&axis_ids).count();
        let overlap_recall = intersection as f64 / cfg.top_k as f64;
        id_overlap.push(overlap_recall);
        if intersection == cfg.top_k {
            perfect_matches += 1;
        }

        // Shadow comparison 1: default-flat top_k vs true brute-force.
        // When < 1.0, default-flat is missing records the true exact
        // would have returned — its pruning is costing recall.
        let flat_vs_fe = exact_ids.intersection(&force_exact_ids).count() as f64 / cfg.top_k as f64;
        flat_default_vs_force_exact.push(flat_vs_fe);

        // Shadow comparison 2: AXIS top_k vs true brute-force. This
        // is the absolute recall — what a customer would observe.
        let axis_vs_fe = axis_ids.intersection(&force_exact_ids).count() as f64 / cfg.top_k as f64;
        axis_vs_force_exact.push(axis_vs_fe);

        // Score-threshold recall: independent of how AXIS reports
        // similarity. Compute the TRUE score (per the configured
        // metric, "higher = better" normalization) for every record
        // in both result sets. The exact path's worst (k-th) true
        // score becomes the bar — AXIS records that meet or beat it
        // count as recall hits.
        let mut exact_true_scores: Vec<f32> = exact
            .iter()
            .filter_map(|r| {
                record_lookup
                    .get(&r.id)
                    .map(|v| true_score(cfg.metric, &query_vector, v))
            })
            .collect();
        exact_true_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = exact_true_scores
            .get(cfg.top_k - 1)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        let axis_above_threshold: usize = axis
            .iter()
            .filter_map(|r| record_lookup.get(&r.id))
            .map(|v| true_score(cfg.metric, &query_vector, v))
            .filter(|s| *s >= threshold)
            .count();
        let threshold_recall = axis_above_threshold as f64 / cfg.top_k as f64;
        score_threshold.push(threshold_recall);
    }

    RecallStats {
        id_overlap,
        score_threshold,
        flat_default_vs_force_exact,
        axis_vs_force_exact,
        perfect_matches,
    }
}

fn print_recall_report(cfg: &BenchConfig, stats: &RecallStats) {
    let n = stats.id_overlap.len();
    if n == 0 {
        return;
    }
    let summarize = |xs: &[f64]| -> (f64, f64, f64) {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, min, max)
    };
    let (id_mean, id_min, id_max) = summarize(&stats.id_overlap);
    let (sc_mean, sc_min, sc_max) = summarize(&stats.score_threshold);
    let (flat_fe_mean, flat_fe_min, flat_fe_max) = summarize(&stats.flat_default_vs_force_exact);
    let (axis_fe_mean, axis_fe_min, axis_fe_max) = summarize(&stats.axis_vs_force_exact);

    println!();
    println!("================================================================================");
    println!("   Recall@{}  (metric={:?})", cfg.top_k, cfg.metric);
    println!("================================================================================");
    println!();
    println!(
        "  ID-overlap recall (AXIS vs default-flat — what the bench has historically reported):"
    );
    println!(
        "      mean: {:.3}   min: {:.3}   max: {:.3}",
        id_mean, id_min, id_max
    );
    println!();
    println!("  Shadow vs true brute-force (force_exact=true baseline):");
    println!("      flat default vs force_exact:");
    println!(
        "          mean: {:.3}   min: {:.3}   max: {:.3}    (1.0 = pruning costs no recall)",
        flat_fe_mean, flat_fe_min, flat_fe_max
    );
    println!("      AXIS vs force_exact (ABSOLUTE recall — customer-observed):");
    println!(
        "          mean: {:.3}   min: {:.3}   max: {:.3}",
        axis_fe_mean, axis_fe_min, axis_fe_max
    );
    println!("  Score-threshold recall (AXIS records whose true metric score >=");
    println!("      exact's k-th best score / k):");
    println!(
        "      mean: {:.3}   min: {:.3}   max: {:.3}",
        sc_mean, sc_min, sc_max
    );
    println!(
        "  perfect (identical id sets): {}/{} queries",
        stats.perfect_matches, n
    );
    if sc_mean >= 0.95 {
        println!("  status:  ✅ score-threshold recall in turbopuffer-class band (>= 0.95)");
    } else if sc_mean >= 0.90 {
        println!("  status:  ⚠️  score-threshold recall acceptable but below 95%");
    } else {
        println!(
            "  status:  ❌ score-threshold recall below 90% — AXIS not finding good candidates"
        );
    }
    if id_mean < sc_mean - 0.05 {
        println!(
            "  note:    id-overlap < score-threshold by >5% — AXIS is finding *as-good* records"
        );
        println!("           but not the SAME records (score ties / index ordering variance).");
    }
}

async fn run_one_query(setup: &SetupResult, cfg: &BenchConfig) -> u64 {
    let query_vector = synthetic_query(cfg.dimension);
    let search_mode = if cfg.approx_mode {
        SearchMode::Approximate { nprobe: None }
    } else {
        SearchMode::Exact
    };
    let search_params = Arc::new(SearchParams {
        vector: Some(query_vector),
        top_k: Some(cfg.top_k as u16),
        search_mode,
        enable_vectorized_execution: if cfg.vectorized { Some(true) } else { None },
        ..Default::default()
    });
    let ctx = StorageQueryContext::new(search_params, Arc::clone(&setup.warm_collection));
    let start = Instant::now();
    let _ = setup
        .warm_engine
        .search_vectors_unified(&ctx)
        .await
        .expect("search");
    start.elapsed().as_micros() as u64
}

// ────────────────────────────────────────────────────────────────────────
// Report
// ────────────────────────────────────────────────────────────────────────

fn percentile(samples: &[u64], p: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u64> = samples.to_vec();
    sorted.sort_unstable();
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    (samples.iter().sum::<u64>() / samples.len() as u64) as u64
}

fn print_report(cfg: &BenchConfig, total_us: &[u64], captured: &HashMap<String, Vec<u64>>) {
    println!();
    println!("================================================================================");
    println!("   Results");
    println!("================================================================================");
    println!();
    println!("Total query latency (warm, includes all phases + outer dispatch):");
    print_dist("  total", total_us);
    println!();

    // Print captured phases in a consistent order matching the
    // search pipeline. Unknown phases (added later by source code)
    // appear at the end.
    //
    // The first 4 are the direct/flat fallback path
    // (`fallback_to_direct_search` in `sst/search/mod.rs`).
    // `axis_query` and `axis_result_convert` are the AXIS-path
    // phases (`execute_orchestrated_search`). For a given run only
    // one set fires — flat populates the first 4, AXIS-backed
    // indexes populate the latter 2.
    let canonical = [
        "discovery",
        "per_file_scan",
        "topk_merge",
        "result_filter",
        "axis_query",
        "axis_result_convert",
    ];
    println!(
        "Per-phase breakdown (microseconds, {} samples / phase):",
        cfg.warm_runs
    );
    println!(
        "  {:<18} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "phase", "mean", "p50", "p99", "max", "% of mean total"
    );
    let total_mean = mean(total_us);

    for name in canonical.iter() {
        if let Some(samples) = captured.get(*name) {
            let m = mean(samples);
            let p50 = percentile(samples, 50.0);
            let p99 = percentile(samples, 99.0);
            let max = *samples.iter().max().unwrap_or(&0);
            let pct = if total_mean > 0 {
                (m as f64 / total_mean as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "  {:<18} {:>10} {:>10} {:>10} {:>10} {:>7.1}%",
                name, m, p50, p99, max, pct
            );
        } else {
            println!("  {:<18} {:>10}", name, "(no samples)");
        }
    }
    // Any extra phases not in the canonical list
    for (name, samples) in captured.iter() {
        if canonical.iter().any(|c| *c == name.as_str()) {
            continue;
        }
        let m = mean(samples);
        let p50 = percentile(samples, 50.0);
        let p99 = percentile(samples, 99.0);
        let max = *samples.iter().max().unwrap_or(&0);
        let pct = if total_mean > 0 {
            (m as f64 / total_mean as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<18} {:>10} {:>10} {:>10} {:>10} {:>7.1}%",
            name, m, p50, p99, max, pct
        );
    }

    // Sum of phase means vs total mean — the gap is overhead
    // outside instrumented phases (collection lookup, query-vector
    // validation, dispatch, etc.).
    let phase_total_mean: u64 = canonical
        .iter()
        .filter_map(|n| captured.get(*n).map(|s| mean(s)))
        .sum();
    let gap = total_mean.saturating_sub(phase_total_mean);
    println!();
    println!(
        "Outside-instrumented overhead:  {} us  ({:.1}% of mean total)",
        gap,
        if total_mean > 0 {
            (gap as f64 / total_mean as f64) * 100.0
        } else {
            0.0
        }
    );
}

fn print_dist(label: &str, samples: &[u64]) {
    println!(
        "  {:<18}  mean={} us  p50={} us  p99={} us  max={} us",
        label,
        mean(samples),
        percentile(samples, 50.0),
        percentile(samples, 99.0),
        samples.iter().max().copied().unwrap_or(0),
    );
}

// ────────────────────────────────────────────────────────────────────────
// Synthetic data (mirrors e2e bench)
// ────────────────────────────────────────────────────────────────────────

fn make_collection(temp_dir: &TempDir, dim: usize, metric: DistanceMetric) -> Collection {
    Collection {
        id: "bench_warm_profile".to_string(),
        config: Some(CollectionConfig {
            dimension: dim as u32,
            distance_metric: Some(metric_proto_code(metric)),
            ..Default::default()
        }),
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Deterministic pseudo-random f32 in [-1, 1] from a `(seed, j)` pair.
/// Uses a splitmix64-style hash so consecutive (i, j) pairs are
/// uncorrelated — the synthetic vectors look like white-noise points
/// instead of the smooth sinusoid the older bench used. Smooth
/// vectors produce many near-tied cosine scores, which makes top-K
/// non-deterministic between two correct implementations.
fn pseudo_random_f32(seed: u64, j: usize) -> f32 {
    let mut x = seed.wrapping_add(j as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    // Map u32 → [-1, 1)
    let u = (x as u32) as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}

fn synthetic_records(
    collection_id: &str,
    count: usize,
    dim: usize,
) -> Vec<proximadb_records::ProximaRecord> {
    (0..count)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("{}_{:08}", collection_id, i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32("default", "vector", dim as u32, (0..dim)
                        .map(|j| pseudo_random_f32(i as u64 + 1, j))
                        .collect(),)],
            props: proximadb_records::ProximaTree::new(),
            record_version: 1,
            created_at_ns: 0,
            updated_at_ns: 0,
            ..Default::default()
        })
        .collect()
}

fn synthetic_query(dim: usize) -> Vec<f32> {
    // Use a fixed query seed (0) that doesn't match any record seed
    // (records use i+1, starting at i=0 → seed=1). Combined with the
    // splitmix64 mixing in pseudo_random_f32, this yields a query
    // vector uncorrelated with any single record's vector — so
    // cosine top-K is genuinely informative.
    (0..dim).map(|j| pseudo_random_f32(0, j)).collect()
}
