//! Real engine Recall@k benchmark — index-vs-ground-truth.
//!
//! Unlike `bench_23_recall_at_k` (which times the recall *computation* over
//! synthetic + SIMULATED approximate results), this builds a real ANN index via
//! the embedded engine, runs real searches, and computes Recall@k against an
//! exact brute-force oracle — i.e. it measures the engine's ACTUAL recall.
//!
//! Recall is a set-intersection correctness metric (independent of CPU/timing),
//! so the recall numbers are trustworthy even on a busy machine; the reported
//! latency is indicative only.
//!
//! Run with:
//! ```bash
//! cargo bench -p proximadb-embedded --bench bench_24_recall_at_k_engine
//! # heavier / different engine:
//! PROXIMADB_BENCH_VECTOR_COUNT=50000 PROXIMADB_BENCH_ENGINE=helix \
//!   cargo bench -p proximadb-embedded --bench bench_24_recall_at_k_engine
//! ```
//!
//! Adapted from the `tests/all_engines_50k_benchmark.rs` harness (same embedded
//! API), but focused on a single engine and Recall@{1,10,100} for the evidence
//! ledger's `recall_at_10` claim.

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const DIMENSION: usize = 128;
const KS: [usize; 3] = [1, 10, 100];
const DEFAULT_VECTOR_COUNT: usize = 10_000;
const NUM_QUERIES: usize = 20;
const BATCH_SIZE: usize = 10_000;
const SEED: u64 = 0xA11C_E5EE;

/// Deterministic LCG (matches the all_engines harness) so runs are reproducible.
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

/// Normalized random vectors (so dot product == cosine similarity).
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

/// Exact nearest neighbours by brute force (dot product on normalized vectors =
/// cosine). Returns the ORDERED top-`k` ids per query (so a prefix gives the
/// exact top-k' for any k' <= k).
fn exact_topk(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<String>> {
    queries
        .iter()
        .map(|query| {
            let mut sims: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.iter()
                .take(k)
                .map(|(i, _)| format!("vec_{i}"))
                .collect()
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
    let count = env_count();
    let engine = std::env::var("PROXIMADB_BENCH_ENGINE").unwrap_or_else(|_| "sst".to_string());
    // Default to APPROXIMATE so we measure the engine's actual ANN recall.
    // Exact mode searches all partitions (100% recall by construction) and would
    // make this bench meaningless. Override with PROXIMADB_BENCH_SEARCH_MODE.
    let search_mode =
        std::env::var("PROXIMADB_BENCH_SEARCH_MODE").unwrap_or_else(|_| "approximate".to_string());
    // The set of k values to report Recall@k for. By default {1,10,100}. The
    // single search runs at `max(ks)`, so with `approximate:N` the HNSW `ef`
    // is floored at `max(ks)` — masking low-`ef` effects on small k. Set
    // PROXIMADB_BENCH_TOPK=K to search at exactly K and report only Recall@K,
    // which isolates the recall/latency curve at low effort (ef can drop to K).
    let ks: Vec<usize> = match std::env::var("PROXIMADB_BENCH_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|k| *k >= 1)
    {
        Some(k) => vec![k],
        None => KS.to_vec(),
    };
    let max_k = *ks.iter().max().expect("non-empty ks");

    println!("\n{}", "=".repeat(72));
    println!(
        "  ENGINE RECALL@k BENCHMARK — {count} vectors, dim {DIMENSION}, engine {engine}, mode {search_mode}, top_k {max_k}"
    );
    println!("{}", "=".repeat(72));

    let vectors = generate_vectors(count, DIMENSION, SEED);
    let queries = generate_vectors(NUM_QUERIES, DIMENSION, SEED ^ 0x9E37_79B9);
    let exact = exact_topk(&vectors, &queries, max_k);

    let tmp = TempDir::new().expect("tempdir");
    let config = EmbeddedConfig::for_benchmarks(tmp.path().to_str().expect("tmp path"));
    let db = EmbeddedProximaDB::new(config).expect("embedded db");
    let collection = format!("recall_{engine}");
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
    wait_for_search_ready(&db, &collection, &queries[0], max_k, &search_mode);

    // Recall@k accumulators + latency.
    let mut recall_sums = vec![0.0f64; ks.len()];
    let mut latencies_ms = Vec::with_capacity(NUM_QUERIES);
    let mut measured = 0usize;
    for (q_idx, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = match db.search_with_mode(
            &collection,
            query.clone(),
            max_k,
            None,
            Some(&search_mode),
        ) {
            Ok(r) => r,
            Err(e) => {
                println!("  warn: query {q_idx} failed: {e}");
                continue;
            }
        };
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        measured += 1;
        let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        for (ki, &k) in ks.iter().enumerate() {
            let exact_k: HashSet<&String> = exact[q_idx].iter().take(k).collect();
            let got_k: HashSet<&String> = result_ids.iter().take(k).collect();
            // Recall@k vs an exact oracle of k neighbours.
            recall_sums[ki] += exact_k.intersection(&got_k).count() as f64 / k as f64;
        }
    }

    if measured == 0 {
        eprintln!("ERROR: no successful queries — cannot report recall");
        std::process::exit(1);
    }

    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let avg_latency = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    let p99 =
        latencies_ms[((latencies_ms.len() as f64 * 0.99) as usize).min(latencies_ms.len() - 1)];

    println!("\n  Results ({measured} queries, exact brute-force oracle):");
    for (ki, &k) in ks.iter().enumerate() {
        println!(
            "    Recall@{k:<4} {:.2}%",
            recall_sums[ki] / measured as f64 * 100.0
        );
    }
    println!("    avg search latency  {avg_latency:.2} ms (indicative)");
    println!("    p99 search latency  {p99:.2} ms (indicative)");
    println!("{}", "=".repeat(72));
}
