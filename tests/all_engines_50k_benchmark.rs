//! Multi-engine vector benchmark (resource intensive)
//!
//! Comprehensive benchmark testing all storage engines at higher scale.
//! Measures insert performance, flush time, search latency, and recall quality.
//!
//! Run with:
//! ```bash
//! # Default workload (10K vectors)
//! cargo test --test all_engines_50k_benchmark -- --nocapture --test-threads=1 --ignored
//!
//! # Heavier workload (50K vectors)
//! PROXIMADB_BENCH_VECTOR_COUNT=50000 cargo test --test all_engines_50k_benchmark -- --nocapture --test-threads=1 --ignored
//! ```

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// Test configuration
const DEFAULT_VECTOR_COUNT: usize = 10_000;
const DIMENSION: usize = 128;
const TOP_K: usize = 10;
const NUM_QUERIES: usize = 10;
const BATCH_SIZE: usize = 10_000;

fn benchmark_engines() -> Vec<&'static str> {
    let mut engines = vec!["sst", "helix", "nova"]; // VIPER deprecated (ADR-093)
    if cfg!(feature = "experimental-engines") {
        engines.push("swift");
    }
    engines
}

fn is_engine_disabled_error(error: &str) -> bool {
    error.contains("disabled in default builds") || error.contains("experimental and disabled")
}

fn benchmark_vector_count() -> usize {
    std::env::var("PROXIMADB_BENCH_VECTOR_COUNT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1_000)
        .unwrap_or(DEFAULT_VECTOR_COUNT)
}

/// Simple LCG random number generator for deterministic results
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

/// Generate normalized random vectors
fn generate_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SimpleRng::new(seed);
    let mut vectors = Vec::with_capacity(count);

    for _ in 0..count {
        let mut v: Vec<f32> = (0..dimension).map(|_| rng.next_f32()).collect();
        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        vectors.push(v);
    }
    vectors
}

/// Compute exact nearest neighbors using brute force
fn compute_exact_neighbors(
    vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    top_k: usize,
) -> Vec<HashSet<String>> {
    query_vectors
        .iter()
        .map(|query| {
            let mut similarities: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                    (i, dot)
                })
                .collect();
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            similarities
                .iter()
                .take(top_k)
                .map(|(i, _)| format!("vec_{}", i))
                .collect()
        })
        .collect()
}

/// Benchmark result for a single engine
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchmarkResult {
    engine: String,
    vector_count: usize,
    insert_time_secs: f64,
    insert_qps: f64,
    flush_time_secs: f64,
    avg_search_latency_ms: f64,
    p99_search_latency_ms: f64,
    avg_recall: f64,
    rating: &'static str,
    error: Option<String>,
}

/// Run benchmark for a single engine
fn benchmark_engine(
    engine: &str,
    temp_dir: &TempDir,
    vector_count: usize,
    vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    exact_neighbors: &[HashSet<String>],
) -> BenchmarkResult {
    let collection_name = format!("bench_{}_{}", vector_count, engine);

    println!("\n{}", "=".repeat(60));
    println!(
        "  ENGINE: {} - {} VECTORS",
        engine.to_uppercase(),
        vector_count
    );
    println!("{}", "=".repeat(60));

    // Create embedded database config
    let config = EmbeddedConfig::for_benchmarks(temp_dir.path().to_str().unwrap());

    // Create embedded database
    let db = match EmbeddedProximaDB::new(config) {
        Ok(db) => db,
        Err(e) => {
            return BenchmarkResult {
                engine: engine.to_string(),
                vector_count,
                insert_time_secs: 0.0,
                insert_qps: 0.0,
                flush_time_secs: 0.0,
                avg_search_latency_ms: 0.0,
                p99_search_latency_ms: 0.0,
                avg_recall: 0.0,
                rating: "ERROR",
                error: Some(format!("Failed to create DB: {}", e)),
            };
        }
    };

    // Create collection
    println!("  Creating collection...");
    if let Err(e) = db.create_collection(&collection_name, DIMENSION as u32, Some(engine)) {
        return BenchmarkResult {
            engine: engine.to_string(),
            vector_count,
            insert_time_secs: 0.0,
            insert_qps: 0.0,
            flush_time_secs: 0.0,
            avg_search_latency_ms: 0.0,
            p99_search_latency_ms: 0.0,
            avg_recall: 0.0,
            rating: "ERROR",
            error: Some(format!("Failed to create collection: {}", e)),
        };
    }

    // Insert vectors in batches
    println!(
        "  Inserting {} vectors in batches of {}...",
        vector_count, BATCH_SIZE
    );
    let insert_start = Instant::now();

    for batch_start in (0..vector_count).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(vector_count);
        let batch_ids: Vec<String> = (batch_start..batch_end)
            .map(|i| format!("vec_{}", i))
            .collect();
        let batch_vectors: Vec<Vec<f32>> = vectors[batch_start..batch_end].to_vec();

        if let Err(e) = db.insert(&collection_name, batch_ids, batch_vectors, None) {
            return BenchmarkResult {
                engine: engine.to_string(),
                vector_count,
                insert_time_secs: 0.0,
                insert_qps: 0.0,
                flush_time_secs: 0.0,
                avg_search_latency_ms: 0.0,
                p99_search_latency_ms: 0.0,
                avg_recall: 0.0,
                rating: "ERROR",
                error: Some(format!("Failed to insert batch: {}", e)),
            };
        }
        println!(
            "    Inserted batch {}/{}",
            batch_start / BATCH_SIZE + 1,
            vector_count.div_ceil(BATCH_SIZE)
        );
    }

    let insert_time = insert_start.elapsed().as_secs_f64();
    let insert_qps = vector_count as f64 / insert_time;
    println!(
        "  Insert completed: {:.2}s ({:.0} vec/s)",
        insert_time, insert_qps
    );

    // Flush
    println!("  Flushing to disk...");
    let flush_start = Instant::now();
    if let Err(e) = db.flush() {
        println!("  Warning: Flush failed: {}", e);
    }
    let flush_time = flush_start.elapsed().as_secs_f64();
    println!("  Flush completed: {:.2}s", flush_time);

    // Wait for async indexing
    println!("  Waiting for async index building (5s)...");
    std::thread::sleep(Duration::from_secs(5));

    // Run search queries
    println!("  Running {} search queries...", NUM_QUERIES);
    let mut latencies = Vec::with_capacity(NUM_QUERIES);
    let mut recalls = Vec::with_capacity(NUM_QUERIES);

    for (i, query) in query_vectors.iter().enumerate() {
        let search_start = Instant::now();
        let results = match db.search(&collection_name, query.clone(), TOP_K, None) {
            Ok(r) => r,
            Err(e) => {
                println!("  Warning: Search query {} failed: {}", i, e);
                continue;
            }
        };
        let latency_ms = search_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency_ms);

        // Calculate recall
        let result_ids: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();
        let recall = exact_neighbors[i].intersection(&result_ids).count() as f64 / TOP_K as f64;
        recalls.push(recall);
    }

    // Calculate statistics
    let avg_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };

    let p99_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = ((latencies.len() as f64 * 0.99) as usize).min(latencies.len() - 1);
        latencies[p99_idx]
    };

    let avg_recall = if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    };

    // Calculate rating based on latency/recall tradeoff
    // Expected latency: 1ms + 5ms * log10(vector_count)
    let expected_latency = 1.0 + 5.0 * (vector_count as f64).log10();
    let latency_ratio = avg_latency / expected_latency;

    let rating = if latency_ratio <= 0.5 && avg_recall >= 0.9 {
        "EXCELLENT"
    } else if latency_ratio <= 1.0 && avg_recall >= 0.8 {
        "GOOD"
    } else if latency_ratio <= 2.0 && avg_recall >= 0.7 {
        "ACCEPTABLE"
    } else {
        "POOR"
    };

    println!("\n  Results:");
    println!("    Recall@{}: {:.1}%", TOP_K, avg_recall * 100.0);
    println!("    Avg Latency: {:.2}ms", avg_latency);
    println!("    P99 Latency: {:.2}ms", p99_latency);
    println!("    Expected Latency: {:.2}ms", expected_latency);
    println!("    Latency Ratio: {:.2}x", latency_ratio);
    println!("    Rating: {}", rating);

    BenchmarkResult {
        engine: engine.to_string(),
        vector_count,
        insert_time_secs: insert_time,
        insert_qps,
        flush_time_secs: flush_time,
        avg_search_latency_ms: avg_latency,
        p99_search_latency_ms: p99_latency,
        avg_recall,
        rating,
        error: None,
    }
}

/// Main benchmark test - all 6 engines with configurable vector count
#[test]
#[ignore = "resource-intensive benchmark; run manually with --ignored"]
fn test_all_engines_50k_benchmark() {
    let vector_count = benchmark_vector_count();

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "MULTI-ENGINE VECTOR BENCHMARK");
    println!("{}", "=".repeat(80));
    println!("\nConfiguration:");
    println!("  Vectors: {}", vector_count);
    println!("  Dimension: {}", DIMENSION);
    println!("  Top-K: {}", TOP_K);
    println!("  Queries: {}", NUM_QUERIES);
    println!("  Env override: PROXIMADB_BENCH_VECTOR_COUNT");

    let engines = benchmark_engines();

    let mut results = Vec::new();
    let vectors = generate_vectors(vector_count, DIMENSION, 42);
    let query_vectors = generate_vectors(NUM_QUERIES, DIMENSION, 123);
    let exact_neighbors = compute_exact_neighbors(&vectors, &query_vectors, TOP_K);

    for engine in engines {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let result = benchmark_engine(
            engine,
            &temp_dir,
            vector_count,
            &vectors,
            &query_vectors,
            &exact_neighbors,
        );
        results.push(result);
    }

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "Engine", "Vectors", "Recall@10", "Avg Latency", "Insert QPS", "Flush(s)", "Rating"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        if r.error.is_some() {
            println!(
                "{:<10} {:>10} {:>10} {:>12} {:>12} {:>10} ERROR",
                r.engine.to_uppercase(),
                r.vector_count,
                "-",
                "-",
                "-",
                "-"
            );
        } else {
            println!(
                "{:<10} {:>10} {:>9.1}% {:>10.2}ms {:>11.0}/s {:>9.1}s {:>10}",
                r.engine.to_uppercase(),
                r.vector_count,
                r.avg_recall * 100.0,
                r.avg_search_latency_ms,
                r.insert_qps,
                r.flush_time_secs,
                r.rating
            );
        }
    }

    // Ranking by latency
    println!("\n{:-^80}", " LATENCY RANKING ");
    let mut sorted_by_latency: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    sorted_by_latency.sort_by(|a, b| {
        a.avg_search_latency_ms
            .partial_cmp(&b.avg_search_latency_ms)
            .unwrap()
    });
    for (i, r) in sorted_by_latency.iter().enumerate() {
        println!(
            "  {}. {}: {:.1}ms avg, {:.0}% recall",
            i + 1,
            r.engine.to_uppercase(),
            r.avg_search_latency_ms,
            r.avg_recall * 100.0
        );
    }

    // Ranking by recall
    println!("\n{:-^80}", " RECALL RANKING ");
    let mut sorted_by_recall: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    sorted_by_recall.sort_by(|a, b| b.avg_recall.partial_cmp(&a.avg_recall).unwrap());
    for (i, r) in sorted_by_recall.iter().enumerate() {
        println!(
            "  {}. {}: {:.1}% recall, {:.1}ms latency",
            i + 1,
            r.engine.to_uppercase(),
            r.avg_recall * 100.0,
            r.avg_search_latency_ms
        );
    }

    // Ranking by insert performance
    println!("\n{:-^80}", " INSERT PERFORMANCE RANKING ");
    let mut sorted_by_insert: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();
    sorted_by_insert.sort_by(|a, b| b.insert_qps.partial_cmp(&a.insert_qps).unwrap());
    for (i, r) in sorted_by_insert.iter().enumerate() {
        println!(
            "  {}. {}: {:.0} vec/s, {:.1}s flush",
            i + 1,
            r.engine.to_uppercase(),
            r.insert_qps,
            r.flush_time_secs
        );
    }

    println!("\n{}", "=".repeat(80));

    // Assert that at least some engines succeeded
    let successful_count = results.iter().filter(|r| r.error.is_none()).count();
    assert!(
        successful_count >= 4,
        "Expected at least 4 engines to succeed, got {}",
        successful_count
    );

    // Assert reasonable recall for successful engines
    for r in results.iter().filter(|r| r.error.is_none()) {
        assert!(
            r.avg_recall >= 0.5,
            "Engine {} has low recall: {:.1}%",
            r.engine,
            r.avg_recall * 100.0
        );
    }
}

/// Quick smoke test with fewer vectors (for CI)
#[test]
fn test_all_engines_quick_smoke_test() {
    const QUICK_VECTOR_COUNT: usize = 500;
    const QUICK_TOP_K: usize = 5;

    println!("\n{}", "=".repeat(60));
    println!("{:^60}", "QUICK SMOKE TEST - ALL ENGINES");
    println!("{}", "=".repeat(60));

    let engines = benchmark_engines();

    let vectors = generate_vectors(QUICK_VECTOR_COUNT, DIMENSION, 42);
    let query = generate_vectors(1, DIMENSION, 123)[0].clone();
    let mut successful_engines = 0usize;

    for engine in engines {
        println!("\n  Testing {}...", engine.to_uppercase());

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = EmbeddedConfig::for_benchmarks(temp_dir.path().to_str().unwrap());
        let db = EmbeddedProximaDB::new(config).expect("Failed to create DB");

        let collection = format!("smoke_{}", engine);
        if let Err(e) = db.create_collection(&collection, DIMENSION as u32, Some(engine)) {
            let msg = e.to_string();
            if is_engine_disabled_error(&msg) {
                println!("    Skipping {} ({})", engine.to_uppercase(), msg);
                continue;
            }
            panic!("Failed to create collection for {}: {}", engine, msg);
        }

        let ids: Vec<String> = (0..QUICK_VECTOR_COUNT).map(|i| format!("v{}", i)).collect();
        if let Err(e) = db.insert(&collection, ids, vectors.clone(), None) {
            let msg = e.to_string();
            if is_engine_disabled_error(&msg) {
                println!("    Skipping {} ({})", engine.to_uppercase(), msg);
                continue;
            }
            panic!("Failed to insert for {}: {}", engine, msg);
        }

        db.flush().ok();
        std::thread::sleep(Duration::from_millis(300));

        let results = match db.search(&collection, query.clone(), QUICK_TOP_K, None) {
            Ok(results) => results,
            Err(e) => {
                let msg = e.to_string();
                if is_engine_disabled_error(&msg) {
                    println!("    Skipping {} ({})", engine.to_uppercase(), msg);
                    continue;
                }
                panic!("Failed to search for {}: {}", engine, msg);
            }
        };

        assert!(!results.is_empty(), "{} returned no results", engine);
        successful_engines += 1;
        println!(
            "    {} returned {} results",
            engine.to_uppercase(),
            results.len()
        );
    }

    assert!(
        successful_engines >= 4,
        "Expected at least 4 engines to pass smoke test, got {}",
        successful_engines
    );

    println!("\n{:-^60}", " ALL ENGINES PASSED ");
}
