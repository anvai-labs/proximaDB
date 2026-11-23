//! Per-search GET-count benchmark — SST vs HELIX (TD-096 S2 / S1.5).
//!
//! Resolves the open question TD-096 S1 left: *is there a material SST-vs-HELIX
//! GET/byte gap on the production Approximate path?* S1 refuted the recall
//! collapse (100% at 10K/100K) and downgraded TD-096 to guidance *pending this
//! measurement*. This bench wraps the shared filesystem seam (via the
//! `PROXIMADB_COUNT_FS_IO` env gate) and reports the per-search read-operation
//! count + bytes for each backend, so the S2 route-disclosure decision rests on
//! evidence, not assertion.
//!
//! Run with:
//! ```bash
//! cargo bench -p proximadb-embedded --bench bench_25_search_get_count
//! # heavier / HELIX:
//! PROXIMADB_BENCH_ENGINE=helix PROXIMADB_BENCH_VECTOR_COUNT=100000 \
//!   cargo bench -p proximadb-embedded --bench bench_25_search_get_count
//! ```
//! (`PROXIMADB_COUNT_FS_IO` is set by the bench itself.)

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use proximadb_storage_filesystem_types::counting::global_counters;
use tempfile::TempDir;

const DIMENSION: usize = 128;
const TOP_K: usize = 100;
const NUM_QUERIES: usize = 20;
const BATCH_SIZE: usize = 10_000;
const DEFAULT_VECTOR_COUNT: usize = 10_000;
const SEED: u64 = 0xA11C_E5EE;

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn generate_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SimpleRng::new(seed);
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dimension).map(|_| rng.next_f32()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect()
}

fn env_count() -> usize {
    std::env::var("PROXIMADB_BENCH_VECTOR_COUNT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 100)
        .unwrap_or(DEFAULT_VECTOR_COUNT)
}

fn wait_for_search_ready(
    db: &EmbeddedProximaDB,
    collection: &str,
    query: &[f32],
    max_k: usize,
    search_mode: &str,
) {
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut last_error = None;
    while Instant::now() < deadline {
        match db.search_with_mode(collection, query.to_vec(), max_k, None, Some(search_mode)) {
            Ok(results) if !results.is_empty() => return,
            Ok(_) => last_error = Some("search returned no rows".to_string()),
            Err(err) => last_error = Some(err.to_string()),
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "ERROR: search did not become ready within 15s: {}",
        last_error.unwrap_or_else(|| "no probe result".to_string())
    );
    std::process::exit(1);
}

fn main() {
    // Initialize a tracing subscriber so the search-path stage logs (Stage
    // 1/2/3 counts, AXIS route) surface under RUST_LOG — for root-cause
    // diagnosis of the GET count. Best-effort (no-op if a subscriber exists).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .try_init();

    // Enable the GET-count instrumentation on the filesystem seam (default OFF
    // in production). Edition 2024: set_var is unsafe.
    unsafe {
        std::env::set_var("PROXIMADB_COUNT_FS_IO", "1");
    }

    let count = env_count();
    let engine = std::env::var("PROXIMADB_BENCH_ENGINE").unwrap_or_else(|_| "sst".to_string());
    // Default approximate (the production path). Set PROXIMADB_BENCH_SEARCH_MODE=exact
    // to verify the instrumentation catches the flat-scan SST reads (Exact reads
    // all partitions via the FileSystem).
    let search_mode =
        std::env::var("PROXIMADB_BENCH_SEARCH_MODE").unwrap_or_else(|_| "approximate".to_string());
    // Cold path (TD-096 S2): drop+reopen after flush with the cache disabled so
    // searches read from disk (removes the warm-cache + WAL confound that masked
    // the warm measurement). Set PROXIMADB_BENCH_COLD=1 to enable.
    let cold = std::env::var("PROXIMADB_BENCH_COLD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let counters = global_counters();

    println!("\n{}", "=".repeat(72));
    println!(
        "  SEARCH GET-COUNT BENCHMARK — {count} vectors, dim {DIMENSION}, engine {engine}, mode {search_mode}, cold={cold}, top_k {TOP_K}"
    );
    println!("{}", "=".repeat(72));

    let vectors = generate_vectors(count, DIMENSION, SEED);
    let queries = generate_vectors(NUM_QUERIES, DIMENSION, SEED ^ 0x9E37_79B9);

    let tmp = TempDir::new().expect("tempdir");
    let data_path = tmp.path().to_str().expect("tmp path").to_string();
    let config = EmbeddedConfig::for_benchmarks(&data_path);
    let db = EmbeddedProximaDB::new(config).expect("embedded db");
    let collection = format!("getcount_{engine}");
    db.create_collection(&collection, DIMENSION as u32, Some(&engine))
        .expect("create collection");

    for start in (0..count).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(count);
        let ids: Vec<String> = (start..end).map(|i| format!("vec_{i}")).collect();
        let batch: Vec<Vec<f32>> = vectors[start..end].to_vec();
        db.insert(&collection, ids, batch, None)
            .expect("insert batch");
    }
    db.flush()
        .unwrap_or_else(|e| println!("  warn: flush: {e}"));
    wait_for_search_ready(&db, &collection, &queries[0], TOP_K, &search_mode);

    // Cold path (TD-096 S2): drop the warm db — clearing the in-memory WAL
    // memtable + block cache that confounded the warm measurement — and reopen
    // from the flushed data with the cache disabled. Searches then read from the
    // FileSystem-backed SST files so CountingFileSystem measures real disk GETs.
    let db = if cold {
        drop(db);
        // Reset BEFORE reopen so the recovery/index-load reads — the real
        // cold-start I/O; the engine is otherwise in-memory, so warm per-query
        // GETs are ~0 — are counted.
        counters.reset();
        let reopen_config = EmbeddedConfig::for_benchmarks_cold(&data_path);
        EmbeddedProximaDB::new(reopen_config).expect("reopen db (cold)")
    } else {
        wait_for_search_ready(&db, &collection, &queries[0], TOP_K, &search_mode);
        // Warm: reset after the probe so only measured searches count.
        counters.reset();
        db
    };

    let mut measured = 0usize;
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(queries.len());
    for query in &queries {
        let t0 = std::time::Instant::now();
        let ok = db
            .search_with_mode(&collection, query.clone(), TOP_K, None, Some(&search_mode))
            .is_ok();
        let lat_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ok {
            measured += 1;
            latencies_ms.push(lat_ms);
        }
    }

    if measured == 0 {
        eprintln!("ERROR: no successful queries — cannot report GET count");
        std::process::exit(1);
    }

    let full = counters.full_reads.load(Ordering::Relaxed);
    let range = counters.range_reads.load(Ordering::Relaxed);
    let batched = counters.batched_range_reads.load(Ordering::Relaxed);
    let bytes = counters.bytes_read.load(Ordering::Relaxed);
    let total_gets = full + range + batched;
    let gets_per_query = total_gets as f64 / measured as f64;
    let bytes_per_query = bytes as f64 / measured as f64;

    println!("\n  Results ({measured} queries, approximate):");
    println!("    total GETs        {total_gets}   ({gets_per_query:.1} / query)");
    println!("      full reads      {full}");
    println!("      range reads     {range}");
    println!("      batched ranges  {batched}");
    println!(
        "    bytes read        {bytes}   ({bytes_per_query:.0} / query, {:.2} MiB/query)",
        bytes_per_query / (1024.0 * 1024.0)
    );
    // Performance (TD-096 S2): per-query latency. In cold mode the FIRST query
    // includes the index load, so it is reported separately (cold-start latency).
    // Accuracy (recall) is NOT re-measured here — the cold path loads the SAME
    // data as the warm path, so recall is S1's 100% (load strategy does not
    // change result correctness); see bench_24_recall_at_k_engine for recall.
    let avg_lat = latencies_ms.iter().sum::<f64>() / measured as f64;
    let cold_start_lat = latencies_ms.first().copied().unwrap_or(0.0);
    let mut sorted = latencies_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99 = sorted[((sorted.len() as f64 * 0.99) as usize).min(sorted.len().saturating_sub(1))];
    println!("    avg latency       {avg_lat:.2} ms/query");
    println!("    cold-start (#1)   {cold_start_lat:.2} ms");
    println!("    p99 latency       {p99:.2} ms/query");
    println!("{}", "=".repeat(72));
}
