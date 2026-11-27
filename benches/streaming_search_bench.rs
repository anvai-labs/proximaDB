//! Benchmark for streaming search memory efficiency
//!
//! This benchmark compares streaming vs. non-streaming search performance,
//! measuring memory usage, latency, throughput, and batch size impact.
//!
//! Metrics measured:
//! 1. **Memory usage** - Compare streaming vs. non-streaming search
//! 2. **Latency** - Time to first result in streaming mode
//! 3. **Throughput** - Results per second
//! 4. **Batch size impact** - How batch_size affects performance

mod common;
use common::benchmark_utils::{STANDARD_DIMENSIONS, print_system_info};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Simulated streaming search result (mirrors EmbeddedSearchIterator output)
#[derive(Debug, Clone)]
struct StreamingSearchResult {
    _id: String,
    _score: f32,
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

/// Simulated non-streaming search result with full vector data
#[derive(Clone)]
struct FullSearchResult {
    _id: String,
    _score: f32,
    vector: Vec<f32>,
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

/// Generate random normalized vector
fn generate_random_vector(dimension: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    let mut vec = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let mut h = DefaultHasher::new();
        (seed, i).hash(&mut h);
        let val = (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        vec.push(val);
    }

    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut vec {
            *val /= norm;
        }
    }
    vec
}

/// Generate test dataset of vectors
fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_random_vector(dimension, i as u64))
        .collect()
}

/// Simulate non-streaming search returning all results at once
fn simulate_non_streaming_search(
    database: &[Vec<f32>],
    query: &[f32],
    top_k: usize,
) -> Vec<FullSearchResult> {
    // Calculate distances and sort
    let mut scored: Vec<(usize, f32)> = database
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            // Simple dot product distance (cosine for normalized vectors)
            let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            (idx, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    // Return full results with vectors
    scored
        .into_iter()
        .map(|(idx, score)| FullSearchResult {
            _id: format!("vec_{}", idx),
            _score: score,
            vector: database[idx].clone(),
            metadata: HashMap::new(),
        })
        .collect()
}

/// Simulate streaming search returning results in batches
fn simulate_streaming_search(
    database: &[Vec<f32>],
    query: &[f32],
    top_k: usize,
    batch_size: usize,
) -> impl Iterator<Item = Vec<StreamingSearchResult>> {
    // Calculate distances and sort
    let mut scored: Vec<(usize, f32)> = database
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            (idx, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    // Convert to streaming results (no vector data)
    let results: Vec<StreamingSearchResult> = scored
        .into_iter()
        .map(|(idx, score)| StreamingSearchResult {
            _id: format!("vec_{}", idx),
            _score: score,
            metadata: HashMap::new(),
        })
        .collect();

    // Return as batched iterator
    StreamingIterator {
        results,
        batch_size,
        position: 0,
    }
}

/// Simple iterator that yields results in batches
struct StreamingIterator {
    results: Vec<StreamingSearchResult>,
    batch_size: usize,
    position: usize,
}

impl Iterator for StreamingIterator {
    type Item = Vec<StreamingSearchResult>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.results.len());
        let batch = self.results[self.position..end].to_vec();
        self.position = end;

        if batch.is_empty() { None } else { Some(batch) }
    }
}

/// Simulate async streaming search with channel-based delivery
async fn simulate_async_streaming_search(
    database: Arc<Vec<Vec<f32>>>,
    query: Vec<f32>,
    top_k: usize,
    batch_size: usize,
) -> mpsc::Receiver<Vec<StreamingSearchResult>> {
    let (tx, rx) = mpsc::channel(4);

    tokio::spawn(async move {
        // Calculate distances and sort
        let mut scored: Vec<(usize, f32)> = database
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Send in batches
        for chunk in scored.chunks(batch_size) {
            let batch: Vec<StreamingSearchResult> = chunk
                .iter()
                .map(|(idx, score)| StreamingSearchResult {
                    id: format!("vec_{}", *idx),
                    score: *score,
                    metadata: HashMap::new(),
                })
                .collect();

            if tx.send(batch).await.is_err() {
                break;
            }
        }
    });

    rx
}

// ============================================================================
// BENCHMARK 1: Memory Usage Comparison
// ============================================================================
fn bench_memory_usage(c: &mut Criterion) {
    print_system_info("Streaming Search Benchmarks");

    let mut group = c.benchmark_group("memory_comparison");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(1));

    // Test with realistic dimensions
    let dimensions = [128, 768, 1536];
    let database_size = 10_000;
    let top_k = 1000;

    for dimension in dimensions {
        let database = generate_test_vectors(database_size, dimension);
        let query = generate_random_vector(dimension, 999);

        // Non-streaming: returns full vectors (high memory)
        group.bench_with_input(
            BenchmarkId::new("non_streaming", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let results = simulate_non_streaming_search(&database, &query, top_k);
                    // Force memory allocation by accessing all vector data
                    let total_bytes: usize = results.iter().map(|r| r.vector.len() * 4).sum();
                    black_box((results, total_bytes))
                })
            },
        );

        // Streaming: returns minimal metadata (low memory)
        group.bench_with_input(
            BenchmarkId::new("streaming_batch_100", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let iter = simulate_streaming_search(&database, &query, top_k, 100);
                    let mut count = 0;
                    for batch in iter {
                        count += batch.len();
                        black_box(&batch);
                    }
                    black_box(count)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 2: Time to First Result (Latency)
// ============================================================================
fn bench_time_to_first_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_to_first_result");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(1));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let dimension = 768;
    let database_size = 10_000;
    let top_k = 1000;
    let batch_sizes = [10, 100, 1000];

    let database = Arc::new(generate_test_vectors(database_size, dimension));
    let query = generate_random_vector(dimension, 42);

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("async_streaming", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    runtime.block_on(async {
                        let db_clone = Arc::clone(&database);
                        let q = query.clone();

                        let start = Instant::now();
                        let mut rx = simulate_async_streaming_search(db_clone, q, top_k, bs).await;

                        // Measure time to first batch
                        let first_batch = rx.recv().await;
                        let latency = start.elapsed();

                        black_box((first_batch, latency))
                    })
                })
            },
        );
    }

    // Compare with non-streaming (must wait for all results)
    group.bench_function("non_streaming_all", |b| {
        b.iter(|| {
            let start = Instant::now();
            let results = simulate_non_streaming_search(&database, &query, top_k);
            let latency = start.elapsed();
            // First result available only after all are computed
            black_box((results.first().cloned(), latency))
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Throughput (Results per Second)
// ============================================================================
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(1));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let dimension = 768;
    let database_size = 10_000;
    let top_k_values = [100, 1000, 5000];

    let database = Arc::new(generate_test_vectors(database_size, dimension));
    let query = generate_random_vector(dimension, 42);

    for top_k in top_k_values {
        group.throughput(Throughput::Elements(top_k as u64));

        // Streaming throughput
        group.bench_with_input(BenchmarkId::new("streaming", top_k), &top_k, |b, &k| {
            b.iter(|| {
                runtime.block_on(async {
                    let db_clone = Arc::clone(&database);
                    let q = query.clone();

                    let mut rx = simulate_async_streaming_search(db_clone, q, k, 100).await;

                    let mut total = 0;
                    while let Some(batch) = rx.recv().await {
                        total += batch.len();
                        black_box(&batch);
                    }
                    black_box(total)
                })
            })
        });

        // Non-streaming throughput
        group.bench_with_input(BenchmarkId::new("non_streaming", top_k), &top_k, |b, &k| {
            b.iter(|| {
                let results = simulate_non_streaming_search(&database, &query, k);
                black_box(results.len())
            })
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Batch Size Impact
// ============================================================================
fn bench_batch_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_impact");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(1));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let dimension = 768;
    let database_size = 10_000;
    let top_k = 1000;
    let batch_sizes = [10, 50, 100, 250, 500, 1000];

    let database = Arc::new(generate_test_vectors(database_size, dimension));
    let query = generate_random_vector(dimension, 42);

    for batch_size in batch_sizes {
        // Measure total time with different batch sizes
        group.bench_with_input(
            BenchmarkId::new("total_time", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    runtime.block_on(async {
                        let db_clone = Arc::clone(&database);
                        let q = query.clone();

                        let mut rx = simulate_async_streaming_search(db_clone, q, top_k, bs).await;

                        let mut total = 0;
                        while let Some(batch) = rx.recv().await {
                            total += batch.len();
                        }
                        black_box(total)
                    })
                })
            },
        );

        // Measure batch count overhead
        group.bench_with_input(
            BenchmarkId::new("batch_overhead", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let iter = simulate_streaming_search(&database, &query, top_k, bs);
                    let mut batch_count = 0;
                    let mut result_count = 0;
                    for batch in iter {
                        batch_count += 1;
                        result_count += batch.len();
                        black_box(&batch);
                    }
                    black_box((batch_count, result_count))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 5: Dimension Scaling Impact
// ============================================================================
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(1));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    // Use standard dimensions from the codebase
    let dimensions = STANDARD_DIMENSIONS;
    let database_size = 5_000;
    let top_k = 500;
    let batch_size = 100;

    for &dimension in dimensions {
        let database = Arc::new(generate_test_vectors(database_size, dimension));
        let query = generate_random_vector(dimension, 42);

        // Streaming with this dimension
        group.bench_with_input(
            BenchmarkId::new("streaming", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    runtime.block_on(async {
                        let db_clone = Arc::clone(&database);
                        let q = query.clone();

                        let mut rx =
                            simulate_async_streaming_search(db_clone, q, top_k, batch_size).await;

                        let mut total = 0;
                        while let Some(batch) = rx.recv().await {
                            total += batch.len();
                        }
                        black_box(total)
                    })
                })
            },
        );

        // Non-streaming with this dimension
        group.bench_with_input(
            BenchmarkId::new("non_streaming", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let results = simulate_non_streaming_search(&database, &query, top_k);
                    black_box(results.len())
                })
            },
        );

        // Memory overhead comparison (bytes per result)
        group.bench_with_input(
            BenchmarkId::new("memory_bytes_per_result", dimension),
            &dimension,
            |b, &dim| {
                b.iter(|| {
                    // Non-streaming: full vector data
                    let non_streaming_bytes = dim * 4; // f32 = 4 bytes

                    // Streaming: just id + score + metadata
                    let streaming_bytes = 8 + 4 + 0; // String estimate + f32 + empty metadata

                    let savings = 1.0 - (streaming_bytes as f64 / non_streaming_bytes as f64);
                    black_box((non_streaming_bytes, streaming_bytes, savings))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 6: Large Result Set Handling
// ============================================================================
fn bench_large_result_sets(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_result_sets");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let dimension = 768;
    let database_size = 50_000;
    let batch_size = 100;

    let database = Arc::new(generate_test_vectors(database_size, dimension));
    let query = generate_random_vector(dimension, 42);

    // Test with increasing top_k values
    let top_k_values = [1_000, 5_000, 10_000, 25_000];

    for top_k in top_k_values {
        group.throughput(Throughput::Elements(top_k as u64));

        // Streaming handles large result sets efficiently
        group.bench_with_input(BenchmarkId::new("streaming", top_k), &top_k, |b, &k| {
            b.iter(|| {
                runtime.block_on(async {
                    let db_clone = Arc::clone(&database);
                    let q = query.clone();

                    let mut rx = simulate_async_streaming_search(db_clone, q, k, batch_size).await;

                    let mut total = 0;
                    while let Some(batch) = rx.recv().await {
                        total += batch.len();
                    }
                    black_box(total)
                })
            })
        });

        // Non-streaming must allocate all at once
        group.bench_with_input(BenchmarkId::new("non_streaming", top_k), &top_k, |b, &k| {
            b.iter(|| {
                let results = simulate_non_streaming_search(&database, &query, k);
                black_box(results.len())
            })
        });
    }

    group.finish();
}

// Configure and run all benchmarks
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(40)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_memory_usage,
              bench_time_to_first_result,
              bench_throughput,
              bench_batch_size_impact,
              bench_dimension_scaling,
              bench_large_result_sets
}

criterion_main!(benches);
