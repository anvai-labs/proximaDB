//! FP16 Centroid Quantization Tests
//!
//! Validates FP16 centroid quantization implementation across SST, SWIFT, and HELIX engines.
//!
//! Tests cover:
//! - FP32 ↔ FP16 conversion accuracy
//! - Round-trip precision loss
//! - Distance computation accuracy with FP16 centroids
//! - Storage savings verification
//! - Backward compatibility (FP32 fallback)

use proximadb::storage::engines::sst::{fp16_to_fp32, fp32_to_fp16};

#[test]
fn test_fp16_conversion_accuracy() {
    // Test conversion accuracy for typical centroid values
    let test_vectors = vec![
        vec![0.0, 1.0, -1.0, 0.5, -0.5],    // Simple values
        vec![0.001, 0.999, -0.001, -0.999], // Near boundaries
        vec![1.5, -1.5, 2.0, -2.0, 3.0],    // Larger magnitudes
        vec![1e-3, 1e-2, 1e-1, 1e0, 1e1],   // Different scales
    ];

    for original in test_vectors {
        // Convert FP32 → FP16 → FP32
        let fp16 = fp32_to_fp16(&original);
        let reconstructed = fp16_to_fp32(&fp16);

        assert_eq!(original.len(), reconstructed.len());

        // Check that conversion error is < 0.1% for each dimension
        for (i, (&orig, &recon)) in original.iter().zip(reconstructed.iter()).enumerate() {
            let abs_error = (orig - recon).abs();
            let rel_error = if orig.abs() > 1e-6 {
                abs_error / orig.abs()
            } else {
                abs_error
            };

            assert!(
                rel_error < 0.001 || abs_error < 0.001,
                "Dimension {}: FP16 conversion error too large. Original: {}, Reconstructed: {}, Rel Error: {:.4}%",
                i,
                orig,
                recon,
                rel_error * 100.0
            );
        }
    }
}

#[test]
fn test_fp16_storage_reduction() {
    // Verify 50% storage reduction
    let dimension = 128;
    let fp32_vector: Vec<f32> = (0..dimension).map(|i| (i as f32) / 128.0).collect();

    let fp16_vector = fp32_to_fp16(&fp32_vector);

    // FP32: 4 bytes per element, FP16: 2 bytes per element
    let fp32_bytes = fp32_vector.len() * std::mem::size_of::<f32>();
    let fp16_bytes = fp16_vector.len() * std::mem::size_of::<u16>();

    assert_eq!(fp32_bytes, dimension * 4);
    assert_eq!(fp16_bytes, dimension * 2);
    assert_eq!(
        fp16_bytes,
        fp32_bytes / 2,
        "FP16 should be 50% of FP32 storage"
    );
}

#[test]
fn test_fp16_distance_accuracy() {
    // Test that distance computations with FP16 centroids maintain accuracy
    let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let centroid_fp32 = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85];

    // Compute distance with FP32
    let distance_fp32 = euclidean_distance(&query, &centroid_fp32);

    // Convert centroid to FP16 and back
    let centroid_fp16 = fp32_to_fp16(&centroid_fp32);
    let centroid_fp32_reconstructed = fp16_to_fp32(&centroid_fp16);

    // Compute distance with reconstructed FP32 (from FP16)
    let distance_fp16_path = euclidean_distance(&query, &centroid_fp32_reconstructed);

    // Distance error should be < 0.1%
    let error = (distance_fp32 - distance_fp16_path).abs();
    let relative_error = error / distance_fp32;

    println!("FP32 distance: {:.6}", distance_fp32);
    println!("FP16 path distance: {:.6}", distance_fp16_path);
    println!("Absolute error: {:.6}", error);
    println!("Relative error: {:.4}%", relative_error * 100.0);

    assert!(
        relative_error < 0.001,
        "Distance computation error with FP16 centroids exceeds 0.1%: {:.4}%",
        relative_error * 100.0
    );
}

// Recall fixture for centroid block selection. A single random query made this
// test flaky: with top-10 the old `>= 0.99` bar meant exactly 10/10, so one
// near-tie flipped by fp16 rounding at rank 10 failed CI. The fixture is now
// seeded and scores the MEAN recall@10 over many queries against the ANN
// target, with a negative control proving the bar can fail.
const RECALL_CENTROIDS: usize = 100;
const RECALL_DIMENSION: usize = 128;
const RECALL_QUERIES: usize = 1000;
const RECALL_TOP_K: usize = 10;
const RECALL_SEED: u64 = 0x5EED_F016;
/// ANN recall target for centroid selection (mean recall@10). Measured
/// 2026-11-26 (`measure_fp16_recall_margin`): fp16 scores min 0.9988 /
/// mean 0.9995 over 200 seeds, so the bar has a wide margin.
const RECALL_TARGET: f64 = 0.975;
/// Low fp16 mantissa bits dropped by the negative control (fp16 keeps 10).
/// The smallest count that fails the bar: dropping 5 scores 0.9824 (passes),
/// dropping 6 scores 0.9668 (fails).
const COARSE_DROPPED_MANTISSA_BITS: u32 = 6;

fn top_k_ids(query: &[f32], centroids: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(idx, centroid)| (idx, euclidean_distance(query, centroid)))
        .collect();
    distances.sort_by(|a, b| a.1.total_cmp(&b.1));
    distances.iter().take(k).map(|(idx, _)| *idx).collect()
}

fn fp16_round_trip(values: &[f32]) -> Vec<f32> {
    fp16_to_fp32(&fp32_to_fp16(values))
}

/// fp16 round trip with the low `bits` mantissa bits zeroed: a deliberately
/// coarser conversion the recall bar must reject.
fn coarse_round_trip(values: &[f32], bits: u32) -> Vec<f32> {
    let mask: u16 = !((1u16 << bits) - 1);
    let coarse: Vec<u16> = fp32_to_fp16(values).iter().map(|h| h & mask).collect();
    fp16_to_fp32(&coarse)
}

/// Mean recall@10 of `convert`ed centroids against the fp32 ground truth over
/// `RECALL_QUERIES` seeded queries, plus the number of queries below 10/10.
fn mean_recall_at_k(seed: u64, convert: &dyn Fn(&[f32]) -> Vec<f32>) -> (f64, usize) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..RECALL_CENTROIDS)
        .map(|_| {
            (0..RECALL_DIMENSION)
                .map(|_| rng.gen_range(0.0..1.0))
                .collect()
        })
        .collect();
    let converted: Vec<Vec<f32>> = centroids.iter().map(|c| convert(c)).collect();

    let mut hits = 0usize;
    let mut imperfect = 0usize;
    for _ in 0..RECALL_QUERIES {
        let query: Vec<f32> = (0..RECALL_DIMENSION)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        let truth = top_k_ids(&query, &centroids, RECALL_TOP_K);
        let got = top_k_ids(&query, &converted, RECALL_TOP_K);
        let found = truth.iter().filter(|id| got.contains(id)).count();
        hits += found;
        if found < RECALL_TOP_K {
            imperfect += 1;
        }
    }
    (
        hits as f64 / (RECALL_QUERIES * RECALL_TOP_K) as f64,
        imperfect,
    )
}

#[test]
fn test_fp16_recall_preservation() {
    let (recall, imperfect) = mean_recall_at_k(RECALL_SEED, &fp16_round_trip);
    println!(
        "fp16 mean recall@{RECALL_TOP_K} over {RECALL_QUERIES} queries: {recall:.4} \
         ({imperfect} queries below {RECALL_TOP_K}/{RECALL_TOP_K})"
    );
    assert!(
        recall >= RECALL_TARGET,
        "fp16 centroid mean recall@{RECALL_TOP_K} {recall:.4} is below the ANN target {RECALL_TARGET}"
    );

    // Negative control: a coarser conversion must fail the same bar, or the
    // bar proves nothing.
    let (coarse_recall, _) = mean_recall_at_k(RECALL_SEED, &|values| {
        coarse_round_trip(values, COARSE_DROPPED_MANTISSA_BITS)
    });
    println!(
        "coarse ({COARSE_DROPPED_MANTISSA_BITS} mantissa bits dropped) mean recall@{RECALL_TOP_K}: {coarse_recall:.4}"
    );
    assert!(
        coarse_recall < RECALL_TARGET,
        "negative control passed the bar ({coarse_recall:.4} >= {RECALL_TARGET}); the bar has no teeth"
    );
}

/// Margin measurement behind `RECALL_TARGET` and `COARSE_DROPPED_MANTISSA_BITS`:
/// `cargo nextest run --test fp16_centroid_test --run-ignored only --no-capture`.
#[test]
#[ignore = "measurement only"]
fn measure_fp16_recall_margin() {
    let seeds = 200u64;
    let mut min = f64::MAX;
    let mut sum = 0.0;
    for seed in 0..seeds {
        let (recall, _) = mean_recall_at_k(seed, &fp16_round_trip);
        min = min.min(recall);
        sum += recall;
    }
    println!(
        "fp16 over {seeds} seeds: min {min:.4}, mean {:.4}",
        sum / seeds as f64
    );
    for bits in 1..=8u32 {
        let (recall, imperfect) =
            mean_recall_at_k(RECALL_SEED, &|values| coarse_round_trip(values, bits));
        println!(
            "drop {bits} mantissa bits: mean recall {recall:.4}, {imperfect} imperfect queries"
        );
    }
}

#[test]
fn test_fp16_edge_cases() {
    // Test edge cases: zero, very small, very large values
    let edge_cases = vec![
        vec![0.0, 0.0, 0.0, 0.0],         // All zeros
        vec![1e-6, 1e-5, 1e-4, 1e-3],     // Very small positive
        vec![-1e-6, -1e-5, -1e-4, -1e-3], // Very small negative
        vec![10.0, 20.0, 30.0, 40.0],     // Large values
        vec![-10.0, -20.0, -30.0, -40.0], // Large negative values
    ];

    for original in edge_cases {
        let fp16 = fp32_to_fp16(&original);
        let reconstructed = fp16_to_fp32(&fp16);

        // For edge cases, allow slightly larger errors but ensure no NaN/Inf
        for &val in &reconstructed {
            assert!(
                val.is_finite(),
                "FP16 conversion produced non-finite value: {}",
                val
            );
        }
    }
}

// Helper function for Euclidean distance
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod sst_integration {

    #[test]
    fn test_sst_index_entry_fp16_serialization() {
        // This test requires access to SST internals
        // TODO: Add test for IndexEntry serialization/deserialization with FP16
        // Verify that:
        // 1. IndexEntry with centroid_fp16 = Some(...) serializes correctly
        // 2. Deserialization reconstructs centroid_fp16
        // 3. Old files without centroid_fp16 deserialize with centroid_fp16 = None
    }
}

#[cfg(test)]
mod swift_integration {

    #[test]
    fn test_swift_superblock_fp16() {
        // This test requires access to SWIFT internals
        // TODO: Add test for SuperBlock with FP16 centroids
        // Verify that:
        // 1. SuperBlock search prefers FP16 centroid when available
        // 2. Falls back to FP32 when FP16 is None
        // 3. Distance computations are within 0.1% error
    }
}
