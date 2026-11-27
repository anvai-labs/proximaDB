// Criterion benchmarks for Hybrid Search fusion strategies
//
// Measures performance of different fusion algorithms:
// - Reciprocal Rank Fusion (RRF)
// - Weighted Linear Combination
// - Borda Count
// - CombSUM, CombMIN, CombMAX
// - Condorcet
// - Dempster-Shafer
// - Adaptive
// - Projection Fusion B5 (arXiv:2604.13728)
//
// The `fusion_rrf_vs_projection` group is the head-to-head comparison referenced
// in TD-044: it measures the latency tradeoff between the relevance-default RRF
// (paper-reported nDCG@10 = 0.828 on TREC-COVID, the best of 6 configs) and the
// Projection variant B5 (paper-reported as faster with greater diversity, but
// not a relevance upgrade). Use this to validate the speed/diversity tradeoff
// claim on synthetic data before running the full quality benchmark on a
// labeled IR dataset.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use proximadb::core::search::hybrid::{
    BM25Result, FusionStrategy, HybridFusionEngine, VectorResult,
};

fn create_test_bm25_results(count: usize) -> Vec<BM25Result> {
    (0..count)
        .map(|i| BM25Result {
            doc_id: format!("doc_{}", i),
            score: 1.0 - (i as f64 * 0.01),
            highlights: None,
            metadata: std::collections::HashMap::new(),
        })
        .collect()
}

fn create_test_vector_results(count: usize) -> Vec<VectorResult> {
    (0..count)
        .map(|i| VectorResult {
            doc_id: format!("doc_{}", i),
            score: 0.9 - (i as f64 * 0.01),
            distance: i as f64 * 0.01,
            metadata: std::collections::HashMap::new(),
        })
        .collect()
}

fn bench_rrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_rrf");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_weighted_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_weighted_linear");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::WeightedLinear {
                alpha: 0.5,
                bm25_normalize: true,
                vector_normalize: true,
            });
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_borda_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_borda_count");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_comb_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_comb_sum");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::CombSum);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_comb_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_comb_min");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::CombMin);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_comb_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_comb_max");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::CombMax);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_condorcet(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_condorcet");

    // Condorcet is O(n²), so use smaller sizes
    for size in [10, 20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::Condorcet);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_dempster_shafer(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_dempster_shafer");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.5 });
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_adaptive");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_projection");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = HybridFusionEngine::new(FusionStrategy::Projection { alpha: 0.5 });
            let bm25_results = create_test_bm25_results(size);
            let vector_results = create_test_vector_results(size);

            b.iter(|| {
                black_box(
                    engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

/// Head-to-head latency comparison: RRF vs Projection on identical inputs.
///
/// Each iteration fuses the same BM25 + vector result sets with both strategies
/// so the relative cost is comparable directly. Report goes in TD-044 closure.
#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_rrf_vs_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_rrf_vs_projection");

    for size in [50, 100, 500, 1000].iter() {
        let bm25_results = create_test_bm25_results(*size);
        let vector_results = create_test_vector_results(*size);

        let rrf_engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });
        let proj_engine = HybridFusionEngine::new(FusionStrategy::Projection { alpha: 0.5 });

        group.bench_with_input(BenchmarkId::new("rrf", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    rrf_engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("projection", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    proj_engine
                        .fuse(
                            black_box(bm25_results.clone()),
                            black_box(vector_results.clone()),
                        )
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

#[expect(deprecated, reason = "Measures the legacy fusion strategy")]
fn bench_overlap_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_overlap_scenarios");

    // High overlap scenario (90% overlap)
    group.bench_function("high_overlap_90pct", |b| {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);
        let bm25_results: Vec<BM25Result> = (0..100)
            .map(|i| BM25Result {
                doc_id: format!("doc_{}", i % 10), // 10 unique docs, 90% overlap
                score: 1.0 - (i as f64 * 0.01),
                highlights: None,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        let vector_results: Vec<VectorResult> = (0..100)
            .map(|i| VectorResult {
                doc_id: format!("doc_{}", i % 10),
                score: 0.9 - (i as f64 * 0.01),
                distance: i as f64 * 0.01,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        b.iter(|| {
            black_box(
                engine
                    .fuse(
                        black_box(bm25_results.clone()),
                        black_box(vector_results.clone()),
                    )
                    .unwrap(),
            )
        });
    });

    // Low overlap scenario (10% overlap)
    group.bench_function("low_overlap_10pct", |b| {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);
        let bm25_results: Vec<BM25Result> = (0..100)
            .map(|i| BM25Result {
                doc_id: format!("bm25_{}", i), // All unique
                score: 1.0 - (i as f64 * 0.01),
                highlights: None,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        let vector_results: Vec<VectorResult> = (0..100)
            .map(|i| VectorResult {
                doc_id: format!("vector_{}", i), // All unique
                score: 0.9 - (i as f64 * 0.01),
                distance: i as f64 * 0.01,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        b.iter(|| {
            black_box(
                engine
                    .fuse(
                        black_box(bm25_results.clone()),
                        black_box(vector_results.clone()),
                    )
                    .unwrap(),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rrf,
    bench_weighted_linear,
    bench_borda_count,
    bench_comb_sum,
    bench_comb_min,
    bench_comb_max,
    bench_condorcet,
    bench_dempster_shafer,
    bench_adaptive,
    bench_projection,
    bench_rrf_vs_projection,
    bench_overlap_scenarios,
);

criterion_main!(benches);
