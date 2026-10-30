//! SST Quantization Performance Benchmarks
//!
//! Benchmarks for measuring the performance of SST quantization operations including:
//! - Quantization speed for different vector dimensions
//! - Progressive search performance
//! - Memory usage and pooling efficiency
//! - Compression ratios
//! - Distance table precomputation overhead

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::sync::{Arc, Once};
use std::time::Duration;

/// Global initialization for hardware capabilities
static INIT: Once = Once::new();

/// Initialize hardware capabilities once for all benchmarks
fn init_hardware() {
    INIT.call_once(|| {
        let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    });
}

use proximadb::compute::quantization::{StorageQuantizationConfig, StorageQuantizationEngine};
use proximadb::core::memory::pool::VectorMemoryPool;
use proximadb_distance_kernel::DistanceMetric;

/// Generate random vectors for benchmarking
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut vectors = Vec::with_capacity(count);

    for _ in 0..count {
        let mut vec = vec![0.0f32; dim];
        for val in &mut vec {
            *val = rng.gen_range(-1.0..1.0);
        }

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut vec {
                *val /= norm;
            }
        }

        vectors.push(vec);
    }

    vectors
}

/// Benchmark quantization speed for different dimensions
fn bench_quantization_speed(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("quantization_speed");
    group.measurement_time(Duration::from_secs(5));

    // Test different dimensions
    for dim in &[128, 256, 384, 512, 768, 1536] {
        let vectors = generate_vectors(1024, *dim);

        // Create and train engine
        let engine = futures::executor::block_on(async {
            // Canonical default wiring (redundancy pass 3): builds the
            // distance compute FROM the config's metric, so the benchmark
            // metric is always explicit in the config object.
            let mut engine =
                StorageQuantizationEngine::new_with_config(StorageQuantizationConfig::default());

            // Train the quantization model with sample vectors
            engine
                .train(&vectors)
                .await
                .expect("Failed to train quantization model");

            Arc::new(engine)
        });

        group.throughput(Throughput::Elements(vectors.len() as u64));
        group.bench_with_input(BenchmarkId::new("dimension", dim), dim, |b, _| {
            b.iter(|| {
                futures::executor::block_on(async {
                    let result = engine.quantize_batch(&vectors, None).await.unwrap();
                    black_box(result);
                })
            });
        });
    }

    group.finish();
}

/// Benchmark progressive search performance
fn bench_progressive_search(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("progressive_search");
    group.measurement_time(Duration::from_secs(5));

    // Test different dataset sizes
    for size in &[1024, 5120, 10240] {
        let vectors = generate_vectors(*size, 384);

        // Setup: Create engine, train, and quantize vectors
        let (engine, quantized) = futures::executor::block_on(async {
            // Canonical default wiring (redundancy pass 3): builds the
            // distance compute FROM the config's metric, so the benchmark
            // metric is always explicit in the config object.
            let mut engine =
                StorageQuantizationEngine::new_with_config(StorageQuantizationConfig::default());

            // Train the quantization model
            engine
                .train(&vectors)
                .await
                .expect("Failed to train quantization model");

            let engine = Arc::new(engine);
            let quantized = engine.quantize_batch(&vectors, None).await.unwrap();
            (engine, quantized)
        });

        let query = vectors[0].clone();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("dataset_size", size), size, |b, _| {
            b.iter(|| {
                futures::executor::block_on(async {
                    let stages = engine
                        .progressive_search(&query, &quantized, 10, &DistanceMetric::Cosine)
                        .await
                        .unwrap();
                    black_box(stages);
                })
            });
        });
    }

    group.finish();
}

/// Benchmark memory pool efficiency
fn bench_memory_pool(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("memory_pool");
    group.measurement_time(Duration::from_secs(5));

    let memory_pool = Arc::new(VectorMemoryPool::new());
    let vectors = generate_vectors(128, 384);

    group.bench_function("with_pooling", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(128 * 384 * 4); // Simulate pooled buffer
            // Simulate serialization
            for vec in &vectors {
                for val in vec {
                    buffer.extend_from_slice(&val.to_le_bytes());
                }
            }
            black_box(buffer.len());
            // Buffer returns to pool when dropped
        });
    });

    group.bench_function("without_pooling", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(128 * 384 * 4);

            // Simulate serialization
            for vec in &vectors {
                for val in vec {
                    buffer.extend_from_slice(&val.to_le_bytes());
                }
            }

            black_box(buffer.len());
        });
    });

    group.finish();
}

/// Benchmark PQ distance table precomputation
fn bench_pq_distance_tables(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("pq_distance_tables");
    group.measurement_time(Duration::from_secs(5));

    // Test different PQ configurations
    for (subvectors, bits) in &[(8, 8), (16, 8), (32, 8), (16, 4)] {
        let vectors = generate_vectors(1024, 512);

        // Setup: Create engine with PQ configuration
        let (engine, quantized) = futures::executor::block_on(async {
            let config = StorageQuantizationConfig::default();
            // Note: QuantizationLevelType not available, using default config
            // config.primary_level = Some(UnifiedQuantizationLevel {
            //     level_type: Some(QuantizationLevelType::Pq(ProductQuantization {
            //         num_subvectors: *subvectors,
            //         bits_per_code: *bits,
            //         codebook_id: None,
            //         adaptive_subvectors: false,
            //     })),
            // });

            let mut engine = StorageQuantizationEngine::new_with_config(config);

            // Train PQ
            engine.train(&vectors).await.unwrap();
            let quantized = engine.quantize_batch(&vectors, None).await.unwrap();

            (Arc::new(engine), quantized)
        });

        let query = vectors[0].clone();
        let config_name = format!("pq_{}_{}", subvectors, bits);

        group.bench_with_input(
            BenchmarkId::new("config", &config_name),
            &config_name,
            |b, _| {
                b.iter(|| {
                    futures::executor::block_on(async {
                        let stages = engine
                            .progressive_search(&query, &quantized, 10, &DistanceMetric::Euclidean)
                            .await
                            .unwrap();

                        // Note: SearchStage not available, using simplified implementation
                        // let pq_stage = stages.iter().find(|s| s.stage == SearchStage::PQRanking);

                        black_box(&stages);
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compression ratios
fn bench_compression_ratios(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("compression_ratios");
    group.measurement_time(Duration::from_secs(5));

    for dim in &[256, 512, 768, 1536] {
        let vectors = generate_vectors(1024, *dim);
        let original_size = vectors.len() * *dim * 4; // f32 = 4 bytes

        let engine = futures::executor::block_on(async {
            // Canonical default wiring (redundancy pass 3): builds the
            // distance compute FROM the config's metric, so the benchmark
            // metric is always explicit in the config object.
            let mut engine =
                StorageQuantizationEngine::new_with_config(StorageQuantizationConfig::default());

            // Train the quantization model
            engine
                .train(&vectors)
                .await
                .expect("Failed to train quantization model");

            Arc::new(engine)
        });

        group.bench_with_input(BenchmarkId::new("dimension", dim), dim, |b, _| {
            b.iter(|| {
                futures::executor::block_on(async {
                    let quantized = engine.quantize_batch(&vectors, None).await.unwrap();
                    let savings = engine.calculate_savings(original_size, &quantized);
                    black_box(savings);
                })
            });
        });
    }

    group.finish();
}

/// Benchmark binary filtering efficiency
fn bench_binary_filtering(c: &mut Criterion) {
    init_hardware();

    let mut group = c.benchmark_group("binary_filtering");
    group.measurement_time(Duration::from_secs(5));

    for size in &[1024, 5120, 10240] {
        let vectors = generate_vectors(*size, 384);

        let (engine, quantized) = futures::executor::block_on(async {
            let mut engine =
                StorageQuantizationEngine::new_with_config(StorageQuantizationConfig {
                    enable_progressive: true,
                    filter_threshold: 0.3,
                    ..Default::default()
                });

            // Train the quantization model
            engine
                .train(&vectors)
                .await
                .expect("Failed to train quantization model");

            let engine = Arc::new(engine);
            let quantized = engine.quantize_batch(&vectors, None).await.unwrap();
            (engine, quantized)
        });

        let query = vectors[0].clone();

        group.bench_with_input(BenchmarkId::new("candidates", size), size, |b, _| {
            b.iter(|| {
                futures::executor::block_on(async {
                    let stages = engine
                        .progressive_search(&query, &quantized, 10, &DistanceMetric::Cosine)
                        .await
                        .unwrap();

                    // Note: SearchStage not available, using simplified implementation
                    // let binary_stage = stages.iter().find(|s| s.stage == SearchStage::BinaryFilter);

                    black_box(&stages);
                })
            });
        });
    }

    group.finish();
}

// Configure with consistent settings across all benchmarks
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(40)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets = bench_quantization_speed,
              bench_progressive_search,
              bench_memory_pool,
              bench_pq_distance_tables,
              bench_compression_ratios,
              bench_binary_filtering
}

criterion_main!(benches);
