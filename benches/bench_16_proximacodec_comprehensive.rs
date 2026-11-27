// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive ProximaCodec Benchmark: Baseline vs SIMD vs GPU
//!
//! Auto-detects platform and benchmarks all available acceleration variants:
//! - Baseline (pure Rust)
//! - SIMD (NEON on ARM, AVX2/AVX-512 on x86)
//! - GPU (Metal on macOS, CUDA on Linux, ROCm on AMD, OpenCL fallback)

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Instant;

// ProximaCodec unified API
use proximadb::storage::engines::core::ops::proximacodec::{ProximaCodec, types::ProximaScheme};

// Baseline implementations (for direct comparisons)
use proximadb::storage::engines::core::ops::proximacodec::impls::baseline::functions::{
    delta, double_delta, frame_of_ref, raw, zigzag,
};

// SIMD implementations (for direct comparisons)
use proximadb::storage::engines::core::ops::proximacodec::simd;

// GPU implementations (for direct comparisons)
#[cfg(all(feature = "metal", target_os = "macos"))]
use proximadb::storage::engines::core::ops::proximacodec::impls::gpu::kernels::metal;

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
use proximadb::storage::engines::core::ops::proximacodec::impls::gpu::kernels::cuda;

// ============================================================================
// Test Data Generators
// ============================================================================

/// Generate normalized embeddings data (values in [-1, 1])
/// For realistic embedding scenarios: num_vectors × dimension
fn generate_normalized(num_vectors: usize, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            let state = RandomState::new();
            let mut hasher = state.build_hasher();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            ((hash % 2000) as f32 / 1000.0) - 1.0 // Range: [-1.0, 1.0]
        })
        .collect()
}

/// Generate sinusoidal/periodic data
fn generate_sinusoidal(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            let freq1 = (i as f32 * 0.1).sin() * 50.0;
            let freq2 = (i as f32 * 0.05).cos() * 30.0;
            let freq3 = (i as f32 * 0.02).sin() * 20.0;
            100.0 + freq1 + freq2 + freq3
        })
        .collect()
}

/// Generate time-series test data (trending with noise)
fn generate_time_series(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            let baseline = 100.0;
            let trend = i as f32 * 0.05;
            let noise = (i as f32 * 0.1).sin() * 2.0;
            baseline + trend + noise
        })
        .collect()
}

/// Generate sequential data (monotonic increase)
fn generate_sequential(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    (0..size).map(|i| i as f32 * 0.1).collect()
}

/// Generate random data (high entropy)
fn generate_random(num_vectors: usize, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            let state = RandomState::new();
            let mut hasher = state.build_hasher();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            (hash % 10000) as f32 / 10.0
        })
        .collect()
}

/// Generate sparse data (mostly zeros with occasional non-zero values)
fn generate_sparse(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            if i % 100 == 0 {
                (i as f32 / 100.0) + 1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Generate constant data (all same value)
fn generate_constant(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    vec![42.0; size]
}

/// Generate clustered data (values grouped in distinct clusters)
fn generate_clustered(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let size = num_vectors * dimension;
    (0..size)
        .map(|i| {
            let cluster = (i / 100) % 5;
            let base = (cluster as f32) * 50.0;
            let variation = ((i % 100) as f32) * 0.5;
            base + variation
        })
        .collect()
}

// ============================================================================
// Platform Detection
// ============================================================================

fn get_platform_info() -> String {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;

    if cfg!(all(feature = "metal", target_os = "macos")) {
        return format!("{} {} (Metal GPU)", os, arch);
    }
    if cfg!(all(
        feature = "gpu",
        target_os = "linux",
        target_arch = "x86_64"
    )) {
        return format!("{} {} (CUDA GPU)", os, arch);
    }
    if cfg!(target_arch = "aarch64") {
        return format!("{} {} (NEON SIMD)", os, arch);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return format!("{} {} (AVX-512 SIMD)", os, arch);
        } else if std::is_x86_feature_detected!("avx2") {
            return format!("{} {} (AVX2 SIMD)", os, arch);
        }
    }

    format!("{} {} (Baseline)", os, arch)
}

// ============================================================================
// Raw (Identity) Encoding Benchmarks
// ============================================================================

fn bench_raw_all_variants(c: &mut Criterion) {
    println!("\n🎯 Platform: {}", get_platform_info());
    println!("Testing Raw (identity) encoding - baseline for normalized embeddings\n");

    let mut group = c.benchmark_group("raw_comprehensive");

    for column_size in [256, 1024].iter() {
        group.throughput(Throughput::Elements(*column_size as u64));

        // Use normalized data (the key use case for Raw encoding)
        let values = generate_normalized(*column_size, 1);
        let benchmark_id = format!("{}vec", column_size);

        // Baseline (only variant - Raw is identity encoding, no SIMD/GPU acceleration needed)
        group.bench_with_input(
            BenchmarkId::new("baseline", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = raw::encode_f32(black_box(vals)).unwrap();
                    let _decoded = raw::decode_f32(&encoded).unwrap();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// DoubleDelta Benchmarks
// ============================================================================

fn bench_double_delta_all_variants(c: &mut Criterion) {
    println!("\n🎯 Platform: {}", get_platform_info());
    println!("Testing column encoding (1 dimension across N vectors)\n");

    let mut group = c.benchmark_group("double_delta_comprehensive");

    // Test column sizes (number of vectors per column)
    for column_size in [256, 1024].iter() {
        group.throughput(Throughput::Elements(*column_size as u64));
        let values = generate_time_series(*column_size, 1);
        let benchmark_id = format!("{}vec", column_size);

        // Baseline (pure Rust)
        group.bench_with_input(
            BenchmarkId::new("baseline", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = double_delta::encode_f32(black_box(vals)).unwrap();
                    let _decoded = double_delta::decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // SIMD (NEON/AVX2/AVX-512)
        group.bench_with_input(
            BenchmarkId::new("simd", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = simd::simd_double_delta_encode_f32(black_box(vals)).unwrap();
                    let _decoded =
                        simd::simd_double_delta_decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // Metal GPU (macOS only)
        #[cfg(all(feature = "metal", target_os = "macos"))]
        group.bench_with_input(
            BenchmarkId::new("metal_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = metal::metal_double_delta_encode_f32(black_box(vals)).unwrap();
                    let _decoded =
                        metal::metal_double_delta_decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // CUDA GPU (Linux NVIDIA)
        #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("cuda_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = cuda::cuda_double_delta_encode_f32(black_box(vals)).unwrap();
                    let _decoded =
                        cuda::cuda_double_delta_decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Delta Encoding Benchmarks
// ============================================================================

fn bench_delta_all_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_comprehensive");

    for column_size in [256, 1024].iter() {
        group.throughput(Throughput::Elements(*column_size as u64));
        let values = generate_time_series(*column_size, 1);
        let benchmark_id = format!("{}vec", column_size);
        let base = 100.0f32;

        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = delta::encode_f32(black_box(vals), base as i64).unwrap();
                    let _decoded = delta::decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // SIMD
        group.bench_with_input(
            BenchmarkId::new("simd", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = simd::simd_delta_encode_f32(black_box(vals), base).unwrap();
                    let _decoded = simd::simd_delta_decode_f32(&encoded, base).unwrap();
                })
            },
        );

        // Metal GPU
        #[cfg(all(feature = "metal", target_os = "macos"))]
        group.bench_with_input(
            BenchmarkId::new("metal_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = metal::metal_delta_encode_f32(black_box(vals), base).unwrap();
                    let _decoded = metal::metal_delta_decode_f32(&encoded, base).unwrap();
                })
            },
        );

        // CUDA GPU
        #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("cuda_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = cuda::cuda_delta_encode_f32(black_box(vals), base).unwrap();
                    let _decoded = cuda::cuda_delta_decode_f32(&encoded, base).unwrap();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Frame-of-Reference Benchmarks
// ============================================================================

fn bench_frame_of_reference_all_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_of_reference_comprehensive");

    for column_size in [256, 1024].iter() {
        group.throughput(Throughput::Elements(*column_size as u64));
        let values = generate_time_series(*column_size, 1);
        let benchmark_id = format!("{}vec", column_size);
        let reference = 100i64;
        let bits = 16u8;

        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = frame_of_ref::encode_f32(black_box(vals), reference).unwrap();
                    let _decoded = frame_of_ref::decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // SIMD
        group.bench_with_input(
            BenchmarkId::new("simd", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded =
                        simd::simd_frame_of_reference_encode_f32(black_box(vals), reference, bits)
                            .unwrap();
                    let _decoded = simd::simd_frame_of_reference_decode_f32(
                        &encoded,
                        reference,
                        bits,
                        vals.len(),
                    )
                    .unwrap();
                })
            },
        );

        // Metal GPU
        #[cfg(all(feature = "metal", target_os = "macos"))]
        group.bench_with_input(
            BenchmarkId::new("metal_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = metal::metal_frame_of_reference_encode_f32(
                        black_box(vals),
                        reference,
                        bits,
                    )
                    .unwrap();
                    let _decoded = metal::metal_frame_of_reference_decode_f32(
                        &encoded,
                        reference,
                        bits,
                        vals.len(),
                    )
                    .unwrap();
                })
            },
        );

        // CUDA GPU
        #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("cuda_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded =
                        cuda::cuda_frame_of_reference_encode_f32(black_box(vals), reference, bits)
                            .unwrap();
                    let _decoded = cuda::cuda_frame_of_reference_decode_f32(
                        &encoded,
                        reference,
                        bits,
                        vals.len(),
                    )
                    .unwrap();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Zigzag Encoding Benchmarks
// ============================================================================

fn bench_zigzag_all_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("zigzag_comprehensive");

    for column_size in [256, 1024].iter() {
        group.throughput(Throughput::Elements(*column_size as u64));
        let values = generate_time_series(*column_size, 1);
        let benchmark_id = format!("{}vec", column_size);
        let bits = 16u8;

        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = zigzag::encode_f32(black_box(vals), bits).unwrap();
                    let _decoded = zigzag::decode_f32(&encoded, vals.len()).unwrap();
                })
            },
        );

        // SIMD
        group.bench_with_input(
            BenchmarkId::new("simd", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = simd::simd_zigzag_encode_f32(black_box(vals), bits).unwrap();
                    let _decoded =
                        simd::simd_zigzag_decode_f32(&encoded, bits, vals.len()).unwrap();
                })
            },
        );

        // Metal GPU
        #[cfg(all(feature = "metal", target_os = "macos"))]
        group.bench_with_input(
            BenchmarkId::new("metal_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = metal::metal_zigzag_encode_f32(black_box(vals), bits).unwrap();
                    let _decoded =
                        metal::metal_zigzag_decode_f32(&encoded, bits, vals.len()).unwrap();
                })
            },
        );

        // CUDA GPU
        #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
        group.bench_with_input(
            BenchmarkId::new("cuda_gpu", &benchmark_id),
            &values,
            |b, vals| {
                b.iter(|| {
                    let encoded = cuda::cuda_zigzag_encode_f32(black_box(vals), bits).unwrap();
                    let _decoded =
                        cuda::cuda_zigzag_decode_f32(&encoded, bits, vals.len()).unwrap();
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Summary Benchmark (Large Batch Comparison)
// ============================================================================

fn bench_summary_large_batch(c: &mut Criterion) {
    // Test column size: 1024 values (1 dimension across 1024 vectors)
    let column_size: usize = 1024;

    let mut group = c.benchmark_group("summary_large_batch");
    group.throughput(Throughput::Elements(column_size as u64));

    let values = generate_time_series(column_size, 1);

    println!(
        "\n📊 Summary: Column with {} values (1 dimension across {} vectors)",
        column_size, column_size
    );
    println!("Platform: {}\n", get_platform_info());

    // DoubleDelta variants
    group.bench_function("double_delta_baseline", |b| {
        b.iter(|| {
            let encoded = double_delta::encode_f32(black_box(&values)).unwrap();
            let _decoded = double_delta::decode_f32(&encoded, values.len()).unwrap();
        })
    });

    group.bench_function("double_delta_simd", |b| {
        b.iter(|| {
            let encoded = simd::simd_double_delta_encode_f32(black_box(&values)).unwrap();
            let _decoded = simd::simd_double_delta_decode_f32(&encoded, values.len()).unwrap();
        })
    });

    #[cfg(all(feature = "metal", target_os = "macos"))]
    group.bench_function("double_delta_metal", |b| {
        b.iter(|| {
            let encoded = metal::metal_double_delta_encode_f32(black_box(&values)).unwrap();
            let _decoded = metal::metal_double_delta_decode_f32(&encoded, values.len()).unwrap();
        })
    });

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    group.bench_function("double_delta_cuda", |b| {
        b.iter(|| {
            let encoded = cuda::cuda_double_delta_encode_f32(black_box(&values)).unwrap();
            let _decoded = cuda::cuda_double_delta_decode_f32(&encoded, values.len()).unwrap();
        })
    });

    group.finish();
}

// ============================================================================
// Comprehensive Compression Analysis
// ============================================================================

struct CompressionMetrics {
    scheme_name: &'static str,
    pattern_name: &'static str,
    original_bytes: usize,
    encoded_bytes: usize,
    compression_ratio: f64,
    encode_time_us: f64,
    decode_time_us: f64,
}

/// Analyze compression ratios for ALL encoding schemes across ALL data patterns
fn bench_compression_analysis(c: &mut Criterion) {
    println!(
        "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                    ProximaCodec Comprehensive Compression Analysis                                           ║"
    );
    println!(
        "║         Testing Column Encoding (1 dimension across N vectors)                                               ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let codec = ProximaCodec::global();

    // Test column sizes (number of vectors per column)
    let column_size = 1024; // Number of vectors (rows in column)

    println!(
        "Testing: Single column with {} values (1 dimension across {} vectors)\n",
        column_size, column_size
    );

    let mut all_results: Vec<CompressionMetrics> = Vec::new();

    // Data pattern generators - now generate single columns
    let patterns: Vec<(&str, Box<dyn Fn(usize) -> Vec<f32>>)> = vec![
        (
            "Normalized",
            Box::new(|size| generate_normalized(size, 1).into_iter().collect()),
        ),
        (
            "Sinusoidal",
            Box::new(|size| generate_sinusoidal(size, 1).into_iter().collect()),
        ),
        (
            "Time-Series",
            Box::new(|size| generate_time_series(size, 1).into_iter().collect()),
        ),
        (
            "Sequential",
            Box::new(|size| generate_sequential(size, 1).into_iter().collect()),
        ),
        (
            "Random",
            Box::new(|size| generate_random(size, 1).into_iter().collect()),
        ),
        (
            "Sparse",
            Box::new(|size| generate_sparse(size, 1).into_iter().collect()),
        ),
        (
            "Constant",
            Box::new(|size| generate_constant(size, 1).into_iter().collect()),
        ),
        (
            "Clustered",
            Box::new(|size| generate_clustered(size, 1).into_iter().collect()),
        ),
    ];

    // Encoding schemes to test
    // Only lossless encoding schemes (perfect roundtrip for f32)
    let schemes: Vec<(&'static str, ProximaScheme)> = vec![
        ("Raw", ProximaScheme::Raw), // Identity encoding (baseline)
        ("Delta", ProximaScheme::Delta { base: 0 }),
        (
            "DoubleDelta",
            ProximaScheme::DoubleDelta {
                first_value: 0,
                first_delta: 0,
            },
        ),
        ("BitPacked-32", ProximaScheme::BitPacked { bits: 32 }), // Lossless for f32 (full precision)
        (
            "PForDelta",
            ProximaScheme::PForDelta {
                majority_bits: 20,
                base: 0,
            },
        ),
        ("Simple8b", ProximaScheme::Simple8b),
        ("VByte", ProximaScheme::VByte),
        ("SparseBitmap", ProximaScheme::SparseBitmap),
        ("SparseCOO", ProximaScheme::SparseCOO),
        ("Dictionary", ProximaScheme::Dictionary),
        ("RunLength", ProximaScheme::RunLength),
    ];

    // Test all combinations
    for (pattern_name, gen_fn) in &patterns {
        println!(
            "\n═══ {} Data Pattern ({} floats = {} bytes) ═══",
            pattern_name,
            column_size,
            column_size * 4
        );

        let values = gen_fn(column_size);
        let original_bytes = values.len() * std::mem::size_of::<f32>();

        for (scheme_name, scheme) in &schemes {
            // Measure encode time
            let start = Instant::now();
            let encoded_result = codec.encode(&values, scheme.clone());
            let encode_time = start.elapsed().as_secs_f64() * 1_000_000.0; // Convert to µs

            if let Ok(encoded) = encoded_result {
                // Measure decode time
                let start = Instant::now();
                let decoded_result = codec.decode(&encoded);
                let decode_time = start.elapsed().as_secs_f64() * 1_000_000.0;

                if decoded_result.is_ok() {
                    let compression_ratio = 1.0 - (encoded.len() as f64 / original_bytes as f64);

                    all_results.push(CompressionMetrics {
                        scheme_name,
                        pattern_name,
                        original_bytes,
                        encoded_bytes: encoded.len(),
                        compression_ratio,
                        encode_time_us: encode_time,
                        decode_time_us: decode_time,
                    });
                } else {
                    println!("  ❌ {}: Decode failed", scheme_name);
                }
            } else {
                println!("  ⏭️  {}: Not supported for this data pattern", scheme_name);
            }
        }
    }

    // Print comprehensive summary table
    print_compression_summary(&all_results);

    // Run a minimal benchmark to satisfy Criterion requirements
    let mut group = c.benchmark_group("compression_analysis");
    group.bench_function("analysis_complete", |b| b.iter(|| {}));
    group.finish();
}

fn print_compression_summary(results: &[CompressionMetrics]) {
    println!(
        "\n\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                            COMPREHENSIVE COMPRESSION SUMMARY TABLE                                           ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Group by pattern
    let patterns: Vec<&str> = results
        .iter()
        .map(|r| r.pattern_name)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for pattern in &patterns {
        println!("\n═══ {} Data Pattern ═══", pattern);
        println!(
            "{:<20} {:>12} {:>12} {:>15} {:>15} {:>15}",
            "Scheme", "Original", "Encoded", "Compression", "Encode (µs)", "Decode (µs)"
        );
        println!("{}", "─".repeat(105));

        let mut pattern_results: Vec<_> = results
            .iter()
            .filter(|r| r.pattern_name == *pattern)
            .collect();

        // Sort by compression ratio (best first)
        pattern_results.sort_by(|a, b| {
            b.compression_ratio
                .partial_cmp(&a.compression_ratio)
                .unwrap()
        });

        for result in pattern_results {
            println!(
                "{:<20} {:>12} {:>12} {:>14.1}% {:>15.2} {:>15.2}",
                result.scheme_name,
                format_bytes(result.original_bytes),
                format_bytes(result.encoded_bytes),
                result.compression_ratio * 100.0,
                result.encode_time_us,
                result.decode_time_us
            );
        }
    }

    // Best schemes per pattern (weighted: 20% encode + 40% decode + 40% compression)
    println!(
        "\n\n═══ 🏆 BEST SCHEME PER DATA PATTERN (20% Encode + 40% Decode + 40% Compression) ═══\n"
    );
    println!(
        "{:<20} {:<24} {:>15} {:>13} {:>13} {:>10}",
        "Data Pattern", "Recommended Scheme", "Compression", "Encode (µs)", "Decode (µs)", "Score"
    );
    println!("{}", "─".repeat(110));

    for pattern in &patterns {
        // Only consider schemes with positive compression (no expansion)
        let pattern_results: Vec<_> = results
            .iter()
            .filter(|r| r.pattern_name == *pattern && r.compression_ratio > 0.0)
            .collect();

        if pattern_results.is_empty() {
            println!(
                "{:<20} {:<20} {:>15} {:>13} {:>13} {:>10}",
                pattern, "None (all expand)", "N/A", "N/A", "N/A", "N/A"
            );
            continue;
        }

        // Find min times for normalization
        let min_encode_time = pattern_results
            .iter()
            .map(|r| r.encode_time_us)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let min_decode_time = pattern_results
            .iter()
            .map(|r| r.decode_time_us)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        // Calculate composite score: 40% decode + 40% compression + 20% encode
        let mut scored_results: Vec<_> = pattern_results
            .iter()
            .map(|r| {
                let compression_score = r.compression_ratio; // Already filtered for > 0
                let encode_speed_score = (min_encode_time / r.encode_time_us).min(1.0);
                let decode_speed_score = (min_decode_time / r.decode_time_us).min(1.0);
                let composite_score = (decode_speed_score * 0.4)
                    + (compression_score * 0.4)
                    + (encode_speed_score * 0.2);
                (r, composite_score)
            })
            .collect();

        // Sort by composite score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if let Some((best, _score)) = scored_results.first() {
            // Check if Delta is within 5% compression of best and prefer it for speed
            let best_compression = pattern_results
                .iter()
                .max_by(|a, b| {
                    a.compression_ratio
                        .partial_cmp(&b.compression_ratio)
                        .unwrap()
                })
                .unwrap();

            let delta_result = pattern_results.iter().find(|r| r.scheme_name == "Delta");

            let chosen = if let Some(delta) = delta_result {
                // If Delta is within 5% of best compression, prefer Delta for speed
                if (best_compression.compression_ratio - delta.compression_ratio).abs() <= 0.05
                    && delta.encode_time_us < best.encode_time_us
                {
                    delta
                } else {
                    best
                }
            } else {
                best
            };

            // Recalculate score for chosen scheme
            let compression_score = chosen.compression_ratio;
            let encode_speed_score = (min_encode_time / chosen.encode_time_us).min(1.0);
            let decode_speed_score = (min_decode_time / chosen.decode_time_us).min(1.0);
            let final_score =
                (decode_speed_score * 0.4) + (compression_score * 0.4) + (encode_speed_score * 0.2);

            println!(
                "{:<20} {:<20} {:>14.1}% {:>13.2} {:>13.2} {:>10.3}",
                pattern,
                chosen.scheme_name,
                chosen.compression_ratio * 100.0,
                chosen.encode_time_us,
                chosen.decode_time_us,
                final_score
            );
        }
    }

    println!(
        "\n═══════════════════════════════════════════════════════════════════════════════════\n"
    );
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = comprehensive_benches;
    config = Criterion::default()
        .sample_size(100)  // Max 100 iterations
        .warm_up_time(std::time::Duration::from_secs(1))  // 1s warmup
        .measurement_time(std::time::Duration::from_secs(5));  // 5s max execution
    targets =
        bench_compression_analysis,
        bench_raw_all_variants,
        bench_double_delta_all_variants,
        bench_delta_all_variants,
        bench_frame_of_reference_all_variants,
        bench_zigzag_all_variants,
        bench_summary_large_batch
);

criterion_main!(comprehensive_benches);
