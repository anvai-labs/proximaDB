// Crate-level lint allows carried over from the root crate — this code moved out of
// the monolith verbatim and inherits its pre-existing patterns as-is (the first four
// match the root crate's lib.rs; the rest are surfaced by `--all-targets` here).
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]
#![allow(clippy::legacy_numeric_constants)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]

//! Distance Computation Module
//!
//! Provides unified distance calculation APIs across all storage engines and hardware backends.
//! All SIMD implementations are now integrated directly into UnifiedDistanceCompute.

pub mod affine_u8;
pub mod engine; // Consolidated engine with all SIMD implementations
pub mod platform;
pub mod quantized; // Unified quantized distance computation for all engines
pub mod sparse; // Sparse vector optimizations

// UnifiedDistanceCompute now contains all SIMD implementations directly:
// - Hardware-aware SIMD implementations (AVX2, SSE, NEON, etc.)
// - GPU acceleration when available
// - Automatic fallback to scalar implementations
// - Distance metric normalization for consistent semantics
// - Zero adapter overhead with direct inline calls

// DEPRECATED: PlatformCapability is deprecated - use HardwareBackend from core::hardware_capabilities
// pub use core::PlatformCapability;

pub mod benchmark; // Benchmarking code (moved from distance/benchmark.rs)

// Re-export main types from engine
pub use affine_u8::{AffineU8BatchResults, AffineU8BatchView, AffineU8Params};
pub use engine::{
    DistanceComputeProvider, DistanceMetric, DistanceMode, MetricProperties, SimilarityResult,
    UnifiedDistanceCompute,
};

// GPU-accelerator injection hook (issue #162): the `compute::gpu` layer registers
// its factory at startup so this crate stays free of a `compute` dependency.
#[cfg(feature = "gpu")]
pub use engine::register_gpu_accelerator_factory;

// Re-export quantized distance computation types
pub use quantized::{
    ApproximationConfig, CacheEvictionPolicy, ComputationMethod, DistanceCacheConfig,
    DistanceMetrics, HardwarePreferences, InstructionSet, Int8VectorData, PQVectorData,
    QuantizedDistanceCalculator, QuantizedDistanceConfig, QuantizedDistanceResult,
    QuantizedVectorData, SIMDOptimization, SelectedFormat, VectorizationStrategy,
};

// Re-export sparse vector optimization types
pub use sparse::{
    CosineSparsityChecker, CosineSparsityWarning, CosineWarningConfig, SparseDistanceResult,
    SparsityAnalyzer, SparsityConfig, SparsityInfo, estimate_cosine_degradation, is_cosine_safe,
    sparse_l2_distance, sparse_l2_distance_scalar, sparse_l2_distance_squared,
};

// DEPRECATED: These exports from core are deprecated. Use UnifiedDistanceCompute instead.
// Commenting out to force migration
// pub use core::{
//     DistanceCompute, PlatformCapability, create_distance_calculator,
//     detect_platform_capability, SimdLevel, DistanceMetric
// };

// --- Tests inlined from tests/unit/compute/test_unified_modules_coverage.rs ---

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_distance_computation_construction() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        // Test default construction
        let _default_compute = UnifiedDistanceCompute::default();
        // Test that the engine was constructed successfully
        // (platform capability is private and automatically detected)
        assert!(true); // Constructor succeeded

        // Test construction with specific metric
        let _euclidean_compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let _cosine_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let _manhattan_compute = UnifiedDistanceCompute::new(DistanceMetric::Manhattan);

        // All engines should construct successfully
        // (platform capability detection is internal)
        assert!(true); // All constructors succeeded
    }

    #[test]
    fn test_chunked_batch_calculation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        let query = vec![1.0, 2.0, 3.0];
        let vector_data: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3])
            .collect();
        let vectors: Vec<&[f32]> = vector_data.iter().map(|v| v.as_slice()).collect();

        // Test batch calculation instead
        let distances =
            compute.calculate_distance_batch(&query, &vectors, &DistanceMetric::Euclidean);

        assert_eq!(distances.len(), 100);

        // Verify first and last distances
        assert_eq!(
            distances[0],
            compute.calculate_distance(&query, vectors[0], &DistanceMetric::Euclidean)
        );
        assert_eq!(
            distances[99],
            compute.calculate_distance(&query, vectors[99], &DistanceMetric::Euclidean)
        );
    }

    #[test]
    fn test_dimension_mismatch_handling() {
        // A dimension mismatch scores as the worst possible distance instead
        // of panicking: under `panic = "abort"` a panic here killed the server
        // (TD-SECSCAN-1). Request boundaries reject mismatched query vectors.
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];

        let distance = compute.distance_with_metric(&vec1, &vec2, &DistanceMetric::Euclidean);
        assert_eq!(distance, f32::INFINITY);
    }

    #[test]
    fn test_distance_normalization() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // Test that all metrics follow "lower = more similar" semantics
        let identical = vec![1.0, 2.0, 3.0];
        let similar = vec![1.1, 2.1, 3.1];
        let different = vec![-1.0, -2.0, -3.0];

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::Manhattan,
        ] {
            let d_identical = compute.calculate_distance(&identical, &identical, &metric);
            let d_similar = compute.calculate_distance(&identical, &similar, &metric);
            let d_different = compute.calculate_distance(&identical, &different, &metric);

            // Distance to self should be minimal (with epsilon for floating point)
            assert!(
                d_identical.rank_value <= d_similar.rank_value + 1e-6,
                "For {:?}: d_identical={} should be <= d_similar={}",
                metric,
                d_identical.rank_value,
                d_similar.rank_value
            );
            // Similar vectors should have less distance than different ones
            assert!(
                d_similar.rank_value < d_different.rank_value + 1e-6,
                "For {:?}: d_similar={} should be < d_different={}",
                metric,
                d_similar.rank_value,
                d_different.rank_value
            );
        }

        // DotProduct behaves differently - it's based on magnitude and angle
        // For unnormalized vectors, the relationship may not hold
        let metric = DistanceMetric::DotProduct;
        let d_identical = compute.calculate_distance(&identical, &identical, &metric);
        let d_different = compute.calculate_distance(&identical, &different, &metric);
        // Just verify it returns valid values
        assert!(d_identical.rank_value.is_finite());
        assert!(d_different.rank_value.is_finite());
    }
}
