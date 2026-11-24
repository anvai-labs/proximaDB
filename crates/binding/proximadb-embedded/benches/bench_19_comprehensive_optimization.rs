// Comprehensive Optimization Benchmark
//
// Tests all optimization permutations systematically using the embedded API:
// - the selectable storage engines (SST, HELIX, NOVA, …)
// - 3 search modes (Exact, Approximate, Adaptive)
// - 2 distance metrics (Cosine, Euclidean)
//
// This benchmark uses the embedded database API for simplicity and stability.

#[path = "../../../../benches/common/benchmark_utils.rs"]
mod benchmark_utils;
use benchmark_utils::print_system_info;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB, StorageLocationConfig};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ============================================================================
// CONFIGURATION DIMENSIONS
// ============================================================================

/// Storage engines to benchmark
#[cfg(feature = "experimental-engines")]
const ENGINES: &[&str] = &["sst", "helix", "swift", "nova", "raptor"];
#[cfg(not(feature = "experimental-engines"))]
const ENGINES: &[&str] = &["sst", "helix", "nova"];

/// Search modes: "exact", "approximate:N", "adaptive:N"
const SEARCH_MODES: &[&str] = &["exact", "approximate:10", "adaptive:5000"];

// ============================================================================
// BENCHMARK CONSTANTS
// ============================================================================

const DIMENSION: usize = 768;
const TOP_K: usize = 10;
const NUM_QUERIES: usize = 10;

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkConfig {
    engine: String,
    search_mode: String,
    vector_count: usize,
    dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    config: BenchmarkConfig,
    insert_qps: f64,
    search_qps: f64,
    latency_p50_us: f64,
    latency_p99_us: f64,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate test vectors with controlled clustering
fn generate_test_vectors(count: usize, dimension: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let num_clusters = 10;

    // Create cluster centers
    let cluster_centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            let mut center: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = center.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut center {
                    *val /= norm;
                }
            }
            center
        })
        .collect();

    // Generate vectors around cluster centers
    let mut ids = Vec::with_capacity(count);
    let mut vectors = Vec::with_capacity(count);

    for i in 0..count {
        let cluster_idx = i % num_clusters;
        let center = &cluster_centers[cluster_idx];

        let mut vector: Vec<f32> = center
            .iter()
            .map(|&c| c + rng.gen_range(-0.3..0.3))
            .collect();

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut vector {
                *val /= norm;
            }
        }

        ids.push(format!("vec_{}", i));
        vectors.push(vector);
    }

    (ids, vectors)
}

/// Generate query vectors
fn generate_query_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(123);

    (0..count)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut vec {
                    *val /= norm;
                }
            }
            vec
        })
        .collect()
}

/// Run a single benchmark configuration
fn run_benchmark(engine: &str, search_mode: &str, vector_count: usize) -> BenchmarkResult {
    use uuid::Uuid;

    let collection_name = format!("bench_{}", Uuid::new_v4().to_string()[..8].to_string());
    let data_path = format!("/tmp/proximadb-comprehensive-bench/{}", collection_name);

    // Ensure parent directory exists
    let _ = std::fs::create_dir_all("/tmp/proximadb-comprehensive-bench");

    // Create embedded database
    let config = EmbeddedConfig {
        storage_locations: vec![StorageLocationConfig {
            path: data_path.clone(),
            weight: 1,
            tags: vec!["benchmark".to_string()],
        }],
        metadata_path: format!("{}/metadata", data_path),
        cache_size_mb: 64,
        default_engine: engine.to_string(),
        enable_wal: false, // Disable WAL for benchmark speed
        access_mode: proximadb_embedded::AccessMode::Exclusive,
        node_id: Some("benchmark-node".to_string()),
        wal_sync_mode: "batch".to_string(),
        block_prune_mode: "sqrt".to_string(),
        block_prune_ratio: 0.2,
        block_prune_min_keep: 1,
        block_prune_max_keep: 0,
        enable_rl_planner: true, // Enable RL planner for optimization benchmarks
        rl_policy_path: None,    // Use default path
        ..Default::default()
    };

    let db = EmbeddedProximaDB::new(config).unwrap();

    // Create collection
    db.create_collection(&collection_name, DIMENSION as u32, Some(engine))
        .unwrap();

    // Generate test data
    let (ids, vectors) = generate_test_vectors(vector_count, DIMENSION);
    let queries = generate_query_vectors(NUM_QUERIES, DIMENSION);

    // Measure insert performance
    let insert_start = Instant::now();
    db.insert(&collection_name, ids, vectors, None).unwrap();
    db.flush().unwrap();
    let insert_duration = insert_start.elapsed();
    let insert_qps = vector_count as f64 / insert_duration.as_secs_f64();

    // Measure search performance
    let mut search_latencies = Vec::with_capacity(queries.len());

    for query in &queries {
        let search_start = Instant::now();
        let _results = db
            .search_with_mode(
                &collection_name,
                query.clone(),
                TOP_K,
                None,
                Some(search_mode),
            )
            .unwrap();
        search_latencies.push(search_start.elapsed());
    }

    // Calculate metrics
    let total_search_time: Duration = search_latencies.iter().sum();
    let search_qps = queries.len() as f64 / total_search_time.as_secs_f64();

    search_latencies.sort();
    let latency_p50_us = if !search_latencies.is_empty() {
        search_latencies[search_latencies.len() / 2].as_micros() as f64
    } else {
        0.0
    };
    let latency_p99_us = if !search_latencies.is_empty() {
        search_latencies[(search_latencies.len() * 99) / 100].as_micros() as f64
    } else {
        0.0
    };

    // Cleanup - drop db first, then remove directory
    drop(db);
    let _ = std::fs::remove_dir_all(&data_path);

    BenchmarkResult {
        config: BenchmarkConfig {
            engine: engine.to_string(),
            search_mode: search_mode.to_string(),
            vector_count,
            dimension: DIMENSION,
        },
        insert_qps,
        search_qps,
        latency_p50_us,
        latency_p99_us,
    }
}

// ============================================================================
// SETUP HELPER — build a loaded DB once, reuse across b.iter() samples
// ============================================================================

/// Holds a pre-populated DB for search-only benchmarks.
/// Dropping this struct cleans up the temp directory.
struct SearchState {
    db: EmbeddedProximaDB,
    collection_name: String,
    queries: Vec<Vec<f32>>,
    data_path: String,
}

impl Drop for SearchState {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.data_path);
    }
}

/// Build a DB with `vector_count` pre-loaded vectors; return it ready for search.
/// Call this OUTSIDE b.iter() to avoid per-sample DB lifecycle overhead.
fn setup_search_db(engine: &str, vector_count: usize) -> SearchState {
    use uuid::Uuid;
    let collection_name = format!("bench_{}", &Uuid::new_v4().to_string()[..8]);
    let data_path = format!("/tmp/proximadb-comprehensive-bench/{}", collection_name);
    let _ = std::fs::create_dir_all("/tmp/proximadb-comprehensive-bench");

    let config = EmbeddedConfig {
        storage_locations: vec![StorageLocationConfig {
            path: data_path.clone(),
            weight: 1,
            tags: vec!["benchmark".to_string()],
        }],
        metadata_path: format!("{}/metadata", data_path),
        cache_size_mb: 64,
        default_engine: engine.to_string(),
        enable_wal: false,
        access_mode: proximadb_embedded::AccessMode::Exclusive,
        node_id: Some("benchmark-node".to_string()),
        wal_sync_mode: "batch".to_string(),
        block_prune_mode: "sqrt".to_string(),
        block_prune_ratio: 0.2,
        block_prune_min_keep: 1,
        block_prune_max_keep: 0,
        enable_rl_planner: false, // disabled for search-only benchmark
        rl_policy_path: None,
        ..Default::default()
    };

    let db = EmbeddedProximaDB::new(config).unwrap();
    db.create_collection(&collection_name, DIMENSION as u32, Some(engine))
        .unwrap();

    let (ids, vectors) = generate_test_vectors(vector_count, DIMENSION);
    db.insert(&collection_name, ids, vectors, None).unwrap();
    db.flush().unwrap();

    let queries = generate_query_vectors(NUM_QUERIES, DIMENSION);

    SearchState {
        db,
        collection_name,
        queries,
        data_path,
    }
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

/// Benchmark 1: Engine comparison (100 vectors, exact search).
/// DB is built once per engine outside b.iter() to avoid per-iteration allocation churn.
fn benchmark_engines_1k(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Engines (100 vectors)");

    let mut group = c.benchmark_group("engines_1k");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &engine in ENGINES {
        // Build DB once — expensive setup, not measured
        let state = setup_search_db(engine, 100);
        let query = state.queries[0].clone();

        group.bench_with_input(BenchmarkId::from_parameter(engine), &engine, |b, _| {
            b.iter(|| {
                let results = state.db.search_with_mode(
                    &state.collection_name,
                    query.clone(),
                    TOP_K,
                    None,
                    Some("exact"),
                );
                black_box(results)
            });
        });
        // state drops here, cleaning up the temp directory
    }

    group.finish();
}

/// Benchmark 2: Engine comparison (1K vectors, search-only latency).
/// DB is built once per engine outside b.iter() to avoid allocation churn.
fn benchmark_engines_10k(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Engines (1K vectors, search latency)");

    let mut group = c.benchmark_group("engines_10k");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &engine in ENGINES {
        // Build DB once — expensive setup, not measured
        let state = setup_search_db(engine, 1_000);
        let query = state.queries[0].clone();

        group.bench_with_input(BenchmarkId::from_parameter(engine), &engine, |b, _| {
            b.iter(|| {
                let results = state.db.search_with_mode(
                    &state.collection_name,
                    query.clone(),
                    TOP_K,
                    None,
                    Some("exact"),
                );
                black_box(results)
            });
        });
        // state drops here, cleaning up the temp directory
    }

    group.finish();
}

/// Benchmark 3: Search mode comparison (SST engine, search-only latency).
/// SST DB is built once per mode outside b.iter().
fn benchmark_search_modes_sst(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Search Modes SST (search latency)");

    let mut group = c.benchmark_group("search_modes_sst");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for &mode in SEARCH_MODES {
        let state = setup_search_db("sst", 500);
        let query = state.queries[0].clone();
        let mode_name = mode.replace(":", "_");

        group.bench_with_input(
            BenchmarkId::from_parameter(&mode_name),
            &mode,
            |b, &mode| {
                b.iter(|| {
                    let results = state.db.search_with_mode(
                        &state.collection_name,
                        query.clone(),
                        TOP_K,
                        None,
                        Some(mode),
                    );
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 4: Search mode comparison (HELIX engine, search-only latency).
/// HELIX DB is built once per mode outside b.iter().
fn benchmark_search_modes_helix(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Search Modes HELIX (search latency)");

    let mut group = c.benchmark_group("search_modes_helix");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for &mode in SEARCH_MODES {
        let state = setup_search_db("helix", 500);
        let query = state.queries[0].clone();
        let mode_name = mode.replace(":", "_");

        group.bench_with_input(
            BenchmarkId::from_parameter(&mode_name),
            &mode,
            |b, &mode| {
                b.iter(|| {
                    let results = state.db.search_with_mode(
                        &state.collection_name,
                        query.clone(),
                        TOP_K,
                        None,
                        Some(mode),
                    );
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 5: Full matrix — key engines × search modes (500 vectors, search-only latency).
/// Each DB is built once per engine outside b.iter(); only search is measured.
fn benchmark_full_matrix(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Full Matrix (search latency)");

    let mut group = c.benchmark_group("full_matrix");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    let key_engines = &["sst", "helix", "nova"];
    let key_modes = &["exact", "approximate:10"];

    for &engine in key_engines {
        // One DB per engine, shared across all modes for that engine
        let state = setup_search_db(engine, 500);
        let query = state.queries[0].clone();

        for &mode in key_modes {
            let mode_name = mode.replace(":", "_");
            let bench_id = format!("{}/{}", engine, mode_name);

            group.bench_with_input(BenchmarkId::from_parameter(&bench_id), &mode, |b, &mode| {
                b.iter(|| {
                    let results = state.db.search_with_mode(
                        &state.collection_name,
                        query.clone(),
                        TOP_K,
                        None,
                        Some(mode),
                    );
                    black_box(results)
                });
            });
        }
        // state drops here, temp dir cleaned up
    }

    group.finish();
}

/// Quick sanity check benchmark
fn benchmark_quick_sanity(c: &mut Criterion) {
    print_system_info("Comprehensive Optimization - Quick Sanity Check");

    let mut group = c.benchmark_group("quick_sanity");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Just test SST with 100 vectors to verify the benchmark works
    group.bench_function("sst_1k_exact", |b| {
        b.iter(|| {
            let result = run_benchmark("sst", "exact", 100);
            eprintln!(
                "SST 1K: insert={:.0} QPS, search={:.0} QPS, p50={:.0}µs",
                result.insert_qps, result.search_qps, result.latency_p50_us
            );
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    name = quick;
    config = Criterion::default().sample_size(10);
    targets = benchmark_quick_sanity
);

criterion_group!(
    name = engines;
    config = Criterion::default().sample_size(10);
    targets = benchmark_engines_1k, benchmark_engines_10k
);

criterion_group!(
    name = search_modes;
    config = Criterion::default().sample_size(10);
    targets = benchmark_search_modes_sst, benchmark_search_modes_helix
);

criterion_group!(
    name = full_matrix;
    config = Criterion::default().sample_size(10);
    targets = benchmark_full_matrix
);

criterion_main!(quick, engines, search_modes, full_matrix);
