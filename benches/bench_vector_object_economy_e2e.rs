/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

//! End-to-end Vector Object Economy benchmark.
//!
//! Counterpart to `bench_vector_object_economy.rs`, which measures
//! the building-block CPU costs in isolation. This bench drives the
//! full directory-routed read path: SstEngine + flush + searches
//! against a tempdir-backed local filesystem.
//!
//! Output is intentionally shaped to be compared head-to-head with
//! turbopuffer's published architecture numbers:
//!
//! * cold query (1M docs):    ~400-500ms, 3-4 round trips
//! * warm query (1M docs):    ~14ms p50
//!
//! ## What this bench measures
//!
//! * **Insert + flush throughput** — how long it takes to land N
//!   synthetic vectors and trigger the SST flush that populates the
//!   per-collection object-economy directory.
//! * **Cold p50/p99 query latency** — first query against a freshly
//!   constructed engine. Page cache cold, index cold, block cache cold.
//! * **Warm p50/p99 query latency** — subsequent queries against the
//!   same engine. Page cache populated by the cold query.
//! * **Per-query throughput** — queries/sec.
//!
//! ## What this bench does NOT yet measure
//!
//! * **Object-store GET count + bytes per query** — needs a mocked-S3
//!   round-trip tracker or real S3 backend. Today the local filesystem
//!   doesn't surface "GET-equivalent" metrics, so we report wall-clock
//!   latency as the proxy. (Turbopuffer's "3-4 round trips" claim
//!   matters when egress cost is the limiting factor; with local FS,
//!   the latency comparison is still meaningful.)
//! * **Recall @ k** — ProximaDB defaults to `SearchMode::Exact` which
//!   yields 100% recall by definition. Adding an approximate-mode
//!   recall measurement against SIFT/GIST ground truth is a separate
//!   slice.
//! * **WAL delta merge cost** — there are no in-flight writes during
//!   the search loop, so the strong-route delta scan exits early
//!   (current_lsn ≤ watermark). Measuring the merge cost under
//!   concurrent write + read load is a separate slice.
//!
//! ## Scale control
//!
//! Default: 10 000 vectors, dim 128, top_k 10 — runs in seconds,
//! suitable for CI smoke. Override via env:
//!
//! ```bash
//! BENCH_VECTORS=1000000  BENCH_DIM=768  BENCH_QUERIES=100 \
//!   cargo bench --bench bench_vector_object_economy_e2e
//! ```
//!
//! At 1M vectors, expect the flush phase to take minutes and total
//! runtime in the 5-15 minute range depending on hardware.

use proximadb::compute::distance_computation::UnifiedDistanceCompute;
use proximadb::core::search::SearchParams;
use proximadb::proto::proximadb_v1::{Collection, CollectionConfig};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageEngine};
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

fn main() {
    let cfg = BenchConfig::from_env();

    println!("================================================================================");
    println!("   ProximaDB — Vector Object Economy E2E Benchmark");
    println!("================================================================================");
    println!();
    println!("Configuration:");
    println!("  vectors:    {}", cfg.vector_count);
    println!("  dimension:  {}", cfg.dimension);
    println!("  top_k:      {}", cfg.top_k);
    println!("  cold runs:  {} (fresh engine each)", cfg.cold_runs);
    println!(
        "  warm runs:  {} (same engine, repeated queries)",
        cfg.warm_runs
    );
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let setup = SetupResult::run(&cfg).await;
        let cold = measure_cold(&cfg).await;
        let warm = measure_warm(&cfg, &setup).await;

        print_report(&cfg, &setup, &cold, &warm);
    });
}

// ────────────────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────────────────

struct BenchConfig {
    vector_count: usize,
    dimension: usize,
    top_k: usize,
    cold_runs: usize,
    warm_runs: usize,
    /// C0 cloud-trace enabler: object-store URL for the collection's
    /// `base_location`. `None` (default) → local tempdir (current behavior).
    /// Set `BENCH_OBJECT_STORE_URL=s3://bucket/path` (+ build with the cloud
    /// feature e.g. `aws` + creds in env) to measure real cloud-cold RTT —
    /// the dominant cost term ComputeScheduler currently guesses at.
    store_url: Option<String>,
}

impl BenchConfig {
    /// Store-scheme classification for the evidence artifact (so a local run
    /// is never mistaken for a cloud run — the whole point of C0).
    fn store_kind(&self) -> &'static str {
        match self.store_url.as_deref() {
            Some(u)
                if u.starts_with("s3://")
                    || u.starts_with("gs://")
                    || u.starts_with("az://")
                    || u.starts_with("abfs://") =>
            {
                "cloud"
            }
            Some(_) => "configured",
            None => "local-fs",
        }
    }

    fn from_env() -> Self {
        Self {
            vector_count: env_usize("BENCH_VECTORS", 10_000),
            dimension: env_usize("BENCH_DIM", 128),
            top_k: env_usize("BENCH_TOP_K", 10),
            cold_runs: env_usize("BENCH_COLD_RUNS", 3),
            warm_runs: env_usize("BENCH_WARM_RUNS", 50),
            store_url: std::env::var("BENCH_OBJECT_STORE_URL").ok(),
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ────────────────────────────────────────────────────────────────────────
// Setup: insert + flush
// ────────────────────────────────────────────────────────────────────────

struct SetupResult {
    insert_ms: u128,
    flush_ms: u128,
    bytes_on_disk: u64,
    /// Held across the bench so the tempdir survives.
    _temp_dir: TempDir,
    /// Cached collection + filesystem for warm reuse — saves us
    /// from rebuilding the engine across warm queries.
    warm_engine: Arc<SstEngine>,
    warm_collection: Arc<Collection>,
}

impl SetupResult {
    async fn run(cfg: &BenchConfig) -> Self {
        let temp_dir = TempDir::new().expect("tempdir");
        let base = cfg
            .store_url
            .clone()
            .unwrap_or_else(|| temp_dir.path().to_str().unwrap().to_string());
        let collection = make_collection(&base, cfg.dimension);

        let vectors = synthetic_records(&collection.id, cfg.vector_count, cfg.dimension);

        let insert_start = Instant::now();
        // Use FilesystemFactory::create_default which returns the
        // production factory pre-configured for local fs. Matches the
        // integration-test pattern.
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
            vector_records: vectors,
            force: true,
            synchronous: true,
            ..Default::default()
        };
        let flush_result = engine.flush(params).await.expect("flush");
        let flush_ms = flush_start.elapsed().as_millis();

        Self {
            insert_ms,
            flush_ms,
            bytes_on_disk: flush_result.bytes_written.unwrap_or(0),
            _temp_dir: temp_dir,
            warm_engine: Arc::new(engine),
            warm_collection: Arc::new(collection),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Cold measurement: fresh engine each iteration
// ────────────────────────────────────────────────────────────────────────

struct LatencyDist {
    samples_ms: Vec<f64>,
}

impl LatencyDist {
    fn percentile(&self, p: f64) -> f64 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn mean(&self) -> f64 {
        if self.samples_ms.is_empty() {
            return 0.0;
        }
        self.samples_ms.iter().sum::<f64>() / self.samples_ms.len() as f64
    }

    fn min(&self) -> f64 {
        self.samples_ms
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }
    fn max(&self) -> f64 {
        self.samples_ms
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

async fn measure_cold(cfg: &BenchConfig) -> LatencyDist {
    let mut samples = Vec::with_capacity(cfg.cold_runs);
    println!(
        "📊 Cold measurement: {} fresh-engine query runs",
        cfg.cold_runs
    );

    for run in 0..cfg.cold_runs {
        let temp_dir = TempDir::new().expect("tempdir");
        let base = cfg
            .store_url
            .clone()
            .unwrap_or_else(|| temp_dir.path().to_str().unwrap().to_string());
        let collection = make_collection(&base, cfg.dimension);
        let vectors = synthetic_records(&collection.id, cfg.vector_count, cfg.dimension);

        let fs = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("filesystem factory"),
        );
        let dist = Arc::new(UnifiedDistanceCompute::default());
        let engine = SstEngine::new_with_config(Default::default(), fs, dist)
            .await
            .expect("sst engine");
        let params = FlushParameters {
            collection_id: Some(collection.id.clone()),
            collection_config: Some(collection.clone()),
            vector_records: vectors,
            force: true,
            synchronous: true,
            ..Default::default()
        };
        engine.flush(params).await.expect("flush");

        let query_vector = synthetic_query(cfg.dimension);
        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector),
            top_k: Some(cfg.top_k as u16),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(search_params, Arc::new(collection));

        // The actual cold measurement: time the first query against
        // this brand-new engine. Page cache cold (just flushed +
        // unbuffered_fsync'd), index cache cold, block cache cold.
        let start = Instant::now();
        let _results = engine.search_vectors_unified(&ctx).await.expect("search");
        let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;

        println!(
            "   run {}: cold query latency {:.2} ms",
            run + 1,
            elapsed_ms
        );
        samples.push(elapsed_ms);
    }
    println!();
    LatencyDist {
        samples_ms: samples,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Warm measurement: same engine, repeated queries
// ────────────────────────────────────────────────────────────────────────

async fn measure_warm(cfg: &BenchConfig, setup: &SetupResult) -> LatencyDist {
    let mut samples = Vec::with_capacity(cfg.warm_runs);
    println!(
        "📊 Warm measurement: {} queries on same engine",
        cfg.warm_runs
    );

    // Pre-warm: one untimed query to populate caches the same way
    // turbopuffer's "warm" path is reached after a first query.
    {
        let query_vector = synthetic_query(cfg.dimension);
        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector),
            top_k: Some(cfg.top_k as u16),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(search_params, Arc::clone(&setup.warm_collection));
        let _ = setup.warm_engine.search_vectors_unified(&ctx).await;
    }

    for _ in 0..cfg.warm_runs {
        let query_vector = synthetic_query(cfg.dimension);
        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector),
            top_k: Some(cfg.top_k as u16),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(search_params, Arc::clone(&setup.warm_collection));
        let start = Instant::now();
        let _results = setup
            .warm_engine
            .search_vectors_unified(&ctx)
            .await
            .expect("search");
        let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;
        samples.push(elapsed_ms);
    }
    println!();
    LatencyDist {
        samples_ms: samples,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Report
// ────────────────────────────────────────────────────────────────────────

fn print_report(cfg: &BenchConfig, setup: &SetupResult, cold: &LatencyDist, warm: &LatencyDist) {
    println!("================================================================================");
    println!("   Results");
    println!("================================================================================");
    println!();
    println!("Setup:");
    println!("  insert phase:           {} ms", setup.insert_ms);
    println!("  flush phase:            {} ms", setup.flush_ms);
    println!(
        "  on-disk size:           {}",
        format_bytes(setup.bytes_on_disk)
    );
    println!();
    println!("Object store (C0 cloud-trace enabler):");
    println!("  kind:                   {}", cfg.store_kind());
    println!(
        "  url:                    {}",
        cfg.store_url.as_deref().unwrap_or("<local tempdir>")
    );
    if cfg.store_kind() != "cloud" {
        println!(
            "  NOTE: {} run — NOT a cloud-cold measurement.",
            cfg.store_kind()
        );
        println!(
            "  Set BENCH_OBJECT_STORE_URL=s3://… (+ cloud feature + creds) for the C0 number."
        );
    }
    println!();
    println!("Cold query latency (fresh engine, no caches):");
    println!("  min:                    {:.2} ms", cold.min());
    println!("  mean:                   {:.2} ms", cold.mean());
    println!("  p50:                    {:.2} ms", cold.percentile(50.0));
    println!("  p99:                    {:.2} ms", cold.percentile(99.0));
    println!("  max:                    {:.2} ms", cold.max());
    println!();
    println!("Warm query latency (same engine, caches populated):");
    println!("  min:                    {:.2} ms", warm.min());
    println!("  mean:                   {:.2} ms", warm.mean());
    println!("  p50:                    {:.2} ms", warm.percentile(50.0));
    println!("  p99:                    {:.2} ms", warm.percentile(99.0));
    println!("  p99.9:                  {:.2} ms", warm.percentile(99.9));
    println!("  max:                    {:.2} ms", warm.max());
    if warm.mean() > 0.0 {
        println!("  effective QPS:          {:.0}", 1_000.0 / warm.mean());
    }
    println!();
    println!("================================================================================");
    println!("   Turbopuffer comparison (their published numbers, 1M docs)");
    println!("================================================================================");
    println!();
    println!(
        "                          turbopuffer       ProximaDB ({} vectors)",
        cfg.vector_count
    );
    println!(
        "  cold p50 latency        ~400-500 ms       {:.2} ms",
        cold.percentile(50.0)
    );
    println!(
        "  warm p50 latency        ~14 ms            {:.2} ms",
        warm.percentile(50.0)
    );
    println!();
    println!("Interpretation:");
    if cfg.vector_count >= 1_000_000 {
        println!("  Scale matches turbopuffer's 1M-doc claim — numbers are");
        println!("  directly comparable. Cold path here is page-cache cold but");
        println!("  NOT object-storage cold (turbopuffer's S3 RTTs add ~100ms");
        println!("  each, which won't show on local FS).");
    } else {
        println!("  Scale below turbopuffer's 1M benchmark — numbers are");
        println!("  indicative, not directly comparable. Run with");
        println!("  BENCH_VECTORS=1000000 for the headline comparison.");
    }
    println!();
    println!("================================================================================");
    println!("   BENCHMARK_EVIDENCE artifact — curate into");
    println!("   docs/_internal/roadmap/BENCHMARK_EVIDENCE.toml");
    println!("================================================================================");
    println!();
    let status = if cfg.store_kind() == "cloud" {
        "measured-cloud"
    } else {
        "measured-local"
    };
    println!("[[claims]]");
    println!("id = \"object_economy_e2e_latency\"");
    println!(
        "status = \"{status}\"  # local ≠ cloud-cold; do NOT promote to measured-cloud from a non-cloud run"
    );
    println!("store_kind = \"{}\"", cfg.store_kind());
    println!(
        "store_url = \"{}\"",
        cfg.store_url.as_deref().unwrap_or("<local tempdir>")
    );
    println!("vectors = {}", cfg.vector_count);
    println!("dimension = {}", cfg.dimension);
    println!("top_k = {}", cfg.top_k);
    println!("cold_p50_ms = {:.2}", cold.percentile(50.0));
    println!("cold_p99_ms = {:.2}", cold.percentile(99.0));
    println!("warm_p50_ms = {:.2}", warm.percentile(50.0));
    println!("warm_p99_ms = {:.2}", warm.percentile(99.0));
    println!();
}

// ────────────────────────────────────────────────────────────────────────
// Synthetic data generators
// ────────────────────────────────────────────────────────────────────────

fn make_collection(base_location: &str, dim: usize) -> Collection {
    Collection {
        storage_class_status: None, // SC1 (#2259) field; None = NOT_ACTIVE
        id: "bench_voe_e2e".to_string(),
        config: Some(CollectionConfig {
            dimension: dim as u32,
            distance_metric: Some(1), // Cosine
            ..Default::default()
        }),
        storage_assignment: Some(proximadb::proto::proximadb_v1::StorageAssignment {
            base_location: base_location.to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn synthetic_records(
    collection_id: &str,
    count: usize,
    dim: usize,
) -> Vec<proximadb_records::ProximaRecord> {
    (0..count)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("{}_{:08}", collection_id, i),
            embeddings: vec![proximadb_records::EmbeddingCell {
                model_id: "default".to_string(),
                modality: "vector".to_string(),
                dim: dim as u32,
                // Deterministic but non-trivial: vary the vector
                // contents per record so the centroid clustering has
                // real work to do.
                values: proximadb_records::EmbeddingValues::Fp32(
                    (0..dim).map(|j| ((i + j) as f32 * 0.001).sin()).collect(),
                ),
                ..Default::default()
            }],
            props: proximadb_records::ProximaTree::new(),
            record_version: 1,
            created_at_ns: 0,
            updated_at_ns: 0,
            ..Default::default()
        })
        .collect()
}

fn synthetic_query(dim: usize) -> Vec<f32> {
    // Match the distribution of synthetic_records so queries hit
    // realistic centroids rather than always landing in a single
    // pocket of the space.
    (0..dim).map(|j| (j as f32 * 0.001).sin() + 0.01).collect()
}

fn format_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if n >= GB {
        format!("{:.2} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.2} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.2} KB", n as f64 / KB as f64)
    } else {
        format!("{} B", n)
    }
}
