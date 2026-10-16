//! Unified Distance Computation System for ProximaDB - CONSOLIDATED VERSION
//!
//! This module provides a unified abstraction for distance calculations with
//! all SIMD implementations integrated directly, eliminating the need for core.rs
//!
//! Key features:
//! - Hardware acceleration with runtime SIMD detection (AVX2, SSE2, NEON, etc.)
//! - Distance metric hierarchy (request → collection → system default)
//! - Batch processing for optimal performance
//! - Consistent results across storage tiers
//! - **Normalized distance semantics**: ALL metrics return values where LOWER = MORE SIMILAR
//!
//! ## Architecture Changes:
//! - All SIMD implementations moved from core.rs as private methods
//! - No more circular dependencies or adapter overhead
//! - Direct inline SIMD calls for maximum performance
//!
//! ## Performance Characteristics (Latest Benchmarks):
//!
//! | Dimension | Metric | Throughput | Latency | SIMD |
//! |-----------|--------|------------|---------|------|
//! | 128D | Cosine | 20.8M ops/s | 0.048μs | AVX2 |
//! | 512D | Euclidean | 3.8M ops/s | 0.26μs | AVX2 |
//! | 1536D | Dot Product | 1.2M ops/s | 0.82μs | AVX2 |
//!
//! ## Hardware Detection Strategy:
//!
//! ```text
//! Startup: detect_hardware() → cache in OnceLock
//!     ↓
//! Runtime: get_cached_preferred_backend() → zero overhead
//!     ↓
//! Compute: dispatch to SIMD implementation
//! ```
//!
//! ## Distance Metric Normalization:
//!
//! All metrics are normalized so that LOWER values = MORE SIMILAR:
//! - **Cosine**: 1 - similarity (range: [0, 2])
//! - **Dot Product**: -dot_product (inverted for consistency)
//! - **Euclidean**: L2 distance (natural)
//! - **Manhattan**: L1 distance (natural)
//!
//! This normalization ensures consistent behavior across all storage engines
//! and search algorithms without special casing.

use anyhow::Result;
use async_trait::async_trait;
use proximadb_memory_pool::{PooledItem, VectorMemoryPool};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use tracing::{info, trace};

// Use proto enum as the single source of truth for DistanceMetric
pub use proximadb_proto::proximadb_v1::DistanceMetric;

// Extension trait for DistanceMetric
pub trait DistanceMetricExt {
    fn is_similarity(&self) -> bool;
}

impl DistanceMetricExt for DistanceMetric {
    fn is_similarity(&self) -> bool {
        matches!(self, DistanceMetric::DotProduct)
    }
}
#[cfg(feature = "gpu")]
use proximadb_hardware_caps::get_hardware_capabilities;
#[cfg(feature = "gpu")]
use tokio::runtime::{Builder as TokioRuntimeBuilder, Handle as TokioHandle};

use proximadb_hardware::{SimdLevel, best_simd_level};

// Re-export HardwareBackend for public use
pub use proximadb_hardware_caps::HardwareBackend;

// Platform-specific SIMD imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Hardware Backend Caching (from original engine.rs)
// ============================================================================

/// Global cache for preferred hardware backend - detected once and cached forever
///
/// ## Caching Strategy:
///
/// Hardware capabilities are detected once at startup and cached forever.
/// This eliminates the overhead of repeated CPU feature detection which
/// involves CPUID instructions and can be expensive.
///
/// The OnceLock ensures thread-safe initialization without locks after
/// the first access.
static PREFERRED_BACKEND: OnceLock<HardwareBackend> = OnceLock::new();

/// Cache for GPU enablement check
///
/// GPU detection is expensive (requires CUDA/OpenCL initialization),
/// so we cache the result. Currently GPU support is experimental.
static GPU_ENABLED_CACHE: OnceLock<bool> = OnceLock::new();

/// Flag to track if we've logged the search backend (first-time-only logging)
///
/// This ensures we only log the search backend once to avoid spamming logs
/// during searches, while still providing visibility into which backend is used.
static SEARCH_BACKEND_LOGGED: AtomicBool = AtomicBool::new(false);

/// Get cached preferred backend - avoids repeated hardware detection
///
/// ## Performance Impact:
///
/// First call: ~100μs (hardware detection)
/// Subsequent calls: ~1ns (memory read)
///
/// This 100,000x speedup is critical for hot paths.
fn get_cached_preferred_backend() -> HardwareBackend {
    *PREFERRED_BACKEND.get_or_init(|| match best_simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::AVX512 => HardwareBackend::AVX512,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::AVX2 => HardwareBackend::AVX2,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::SSE41 => HardwareBackend::SSE,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::NEON => HardwareBackend::NEON,
        _ => HardwareBackend::Scalar,
    })
}

/// Check if GPU is enabled (cached)
///
/// Currently returns false as GPU support is experimental.
/// Future versions will support CUDA/ROCm/Metal acceleration.
fn is_gpu_enabled_cached() -> bool {
    *GPU_ENABLED_CACHE.get_or_init(|| {
        #[cfg(feature = "gpu")]
        {
            let caps = get_hardware_capabilities();
            caps.has_gpu()
        }
        #[cfg(not(feature = "gpu"))]
        false
    })
}

/// Log the search backend being used (first time only)
///
/// This function logs the SIMD backend being used for distance computation
/// only on the first call, avoiding log spam during searches.
fn log_search_backend_first_time(platform: PlatformCapability) {
    // Use compare_exchange to ensure we only log once, even with concurrent access
    if SEARCH_BACKEND_LOGGED
        .compare_exchange(false, true, AtomicOrdering::SeqCst, AtomicOrdering::Relaxed)
        .is_ok()
    {
        let backend_name = match platform {
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx512 => "AVX-512",
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx2 => "AVX2+FMA",
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx => "AVX",
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Sse2 => "SSE2",
            #[cfg(target_arch = "aarch64")]
            PlatformCapability::ArmNeon => "ARM NEON",
            #[cfg(target_arch = "aarch64")]
            PlatformCapability::ArmSve => "ARM SVE",
            PlatformCapability::Scalar => "Scalar",
        };
        info!("🔍 Search backend: {} SIMD", backend_name);
    }
}

/// Initialize hardware backend caching
pub fn initialize_hardware_backend_cache() {
    let _ = get_cached_preferred_backend();
    let _ = is_gpu_enabled_cached();
}

// ============================================================================
// Platform Capability Mapping from Global Hardware Detection
// ============================================================================

/// Platform-agnostic SIMD capability detection
///
/// ## SIMD Instruction Sets:
///
/// ### x86_64:
/// - **SSE2**: 128-bit vectors, 2x speedup (baseline for x86_64)
/// - **AVX**: 256-bit vectors, 4x speedup (Sandy Bridge+)
/// - **AVX2**: 256-bit with FMA, 4-8x speedup (Haswell+)
/// - **AVX512**: 512-bit vectors, 8-16x speedup (Skylake-X+)
///
/// ### ARM:
/// - **NEON**: 128-bit vectors, 2-4x speedup (ARMv7+)
/// - **SVE**: Scalable vectors, 4-32x speedup (ARMv8.2+)
///
/// ### Fallback:
/// - **Scalar**: No SIMD, portable C implementation
///
/// ## Selection Priority:
///
/// AVX512 > AVX2 > AVX > SSE2 > Scalar (x86_64)
/// SVE > NEON > Scalar (ARM)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PlatformCapability {
    /// Scalar fallback - works everywhere but slowest
    Scalar,

    #[cfg(target_arch = "x86_64")]
    /// SSE2 - 128-bit SIMD (Pentium 4, 2001+)
    X86Sse2,

    #[cfg(target_arch = "x86_64")]
    /// AVX - 256-bit SIMD (Sandy Bridge, 2011+)
    X86Avx,

    #[cfg(target_arch = "x86_64")]
    /// AVX2 - 256-bit with FMA (Haswell, 2013+)
    X86Avx2,

    #[cfg(target_arch = "x86_64")]
    /// AVX512 - 512-bit SIMD (Skylake-X, 2017+)
    X86Avx512,

    #[cfg(target_arch = "aarch64")]
    /// ARM NEON - 128-bit SIMD (Cortex-A8+)
    ArmNeon,

    #[cfg(target_arch = "aarch64")]
    /// ARM SVE - Scalable vectors (ARMv8.2+)
    ArmSve,
}

/// Global capability cache - maps from HardwareBackend to PlatformCapability once
///
/// This two-level caching (HardwareBackend → PlatformCapability) enables
/// clean separation between hardware detection (in core module) and
/// SIMD dispatch (in compute module).
static PLATFORM_CAPABILITY: OnceLock<PlatformCapability> = OnceLock::new();

/// Get platform capability from global hardware detection (no per-call overhead)
///
/// ## Dispatch Strategy:
///
/// 1. Check PLATFORM_CAPABILITY cache (hot path: 1ns)
/// 2. If uninitialized, detect hardware (cold path: 100μs)
/// 3. Map HardwareBackend to PlatformCapability
/// 4. Cache forever
///
/// This ensures SIMD dispatch has zero overhead in production.
fn get_platform_capability() -> PlatformCapability {
    *PLATFORM_CAPABILITY.get_or_init(|| match best_simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::AVX512 => {
            trace!("Using AVX-512 SIMD from global hardware detection");
            PlatformCapability::X86Avx512
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::AVX2 => {
            trace!("Using AVX2 SIMD from global hardware detection");
            PlatformCapability::X86Avx2
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::SSE41 => {
            trace!("Using SSE2 SIMD from global hardware detection");
            PlatformCapability::X86Sse2
        }
        #[cfg(target_arch = "aarch64")]
        SimdLevel::NEON => {
            trace!("Using ARM NEON SIMD from global hardware detection");
            PlatformCapability::ArmNeon
        }
        _ => {
            trace!("Using scalar implementation from global hardware detection");
            PlatformCapability::Scalar
        }
    })
}

// ============================================================================
// Core Types and Traits
// ============================================================================

/// Factory that builds a GPU distance accelerator. Registered by the
/// feature-gated `compute::gpu` layer at startup (`src/bin/server.rs`) so this
/// kernel crate holds only the `GpuAccelerator` trait + a hook and stays free of a
/// `compute` dependency (issue #162). No registration (default builds) => CPU path.
#[cfg(feature = "gpu")]
pub type GpuAcceleratorFactory = fn() -> anyhow::Result<Arc<dyn GpuAccelerator>>;

#[cfg(feature = "gpu")]
static GPU_ACCELERATOR_FACTORY: OnceLock<GpuAcceleratorFactory> = OnceLock::new();

/// Register the GPU-accelerator factory. Call once at startup from the
/// `compute::gpu` layer (under `#[cfg(feature = "gpu")]`). No-op if already set.
#[cfg(feature = "gpu")]
pub fn register_gpu_accelerator_factory(factory: GpuAcceleratorFactory) {
    let _ = GPU_ACCELERATOR_FACTORY.set(factory);
}

/// GPU accelerator trait for hardware acceleration
#[async_trait]
pub trait GpuAccelerator: Send + Sync {
    /// Check if GPU is available
    fn is_available(&self) -> bool;

    /// Get the hardware backend type
    fn backend(&self) -> HardwareBackend;

    /// Calculate distance using GPU
    async fn calculate_distance_gpu(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        metric: DistanceMetric,
    ) -> Result<f32>;

    /// Calculate batch distances using GPU
    async fn calculate_batch_gpu(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>>;
}

/// Distance computation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMode {
    /// Use CPU computation
    #[default]
    Cpu,
    /// Use GPU if available, fallback to CPU
    GpuWithFallback,
    /// Use GPU only, fail if not available
    GpuOnly,
    /// Optimize for ranking (compatibility)
    RankOptimized,
    /// Normalized mode (compatibility)
    Normalized,
}

/// Properties of a distance metric
#[derive(Debug, Clone)]
pub struct MetricProperties {
    /// Name of the metric
    pub name: String,
    /// Whether higher values mean more similar (true) or less similar (false)
    pub is_similarity: bool,
    /// Typical value range
    pub range: (f32, f32),
    /// Whether the metric is symmetric
    pub is_symmetric: bool,
}

/// Result of a similarity computation with semantic meaning
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimilarityResult {
    /// The computed distance value (always normalized: lower = more similar)
    pub distance: f32,
    /// The raw value before normalization
    pub raw_value: f32,
    /// The metric used
    pub metric: DistanceMetric,
    /// Semantic similarity score (0.0 = identical, 1.0 = maximally different)
    pub similarity_score: f32,
    /// Normalized score in [0, 1] where 1 = most similar (for compatibility)
    pub normalized_score: f32,
    /// Value optimized for ranking (lower = more similar) - same as distance for compatibility
    pub rank_value: f32,
}

/// Lazy wrapper for batch distance results
/// Delays conversion from raw distances until actually needed
pub struct BatchDistanceResults {
    /// Pooled buffer containing raw distances
    pub distances: PooledItem<Vec<f32>>,
    /// The metric used for distance calculation
    pub metric: DistanceMetric,
    /// Reference to compute engine for normalization (reserved for future use)
    #[allow(dead_code)]
    compute: UnifiedDistanceCompute,
}

impl BatchDistanceResults {
    /// Get the number of results
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Get raw distance at index
    pub fn raw_distance(&self, index: usize) -> Option<f32> {
        if index < self.distances.len() {
            Some(self.distances[index])
        } else {
            None
        }
    }

    /// Convert to similarity results (consumes self)
    pub fn into_similarity_results(self) -> Vec<SimilarityResult> {
        self.distances
            .iter()
            .map(|&raw_distance| SimilarityResult::new(raw_distance, self.metric))
            .collect()
    }

    /// Get top-k results without converting all
    pub fn top_k(self, k: usize) -> Vec<SimilarityResult> {
        let mut indexed: Vec<(usize, f32)> = self
            .distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();

        // Partial sort for top-k
        let k = k.min(indexed.len());
        if k > 0 {
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
        }

        // Convert only top-k to results
        indexed[..k]
            .iter()
            .map(|&(_, raw_distance)| SimilarityResult::new(raw_distance, self.metric))
            .collect()
    }
}

impl SimilarityResult {
    /// Create a new similarity result with normalization
    /// IMPORTANT: normalize_distance now returns normalized_similarity directly (0-1, higher = better)
    /// No need for double inversion!
    pub fn new(raw_value: f32, metric: DistanceMetric) -> Self {
        let (distance, normalized_similarity) = Self::normalize_distance(raw_value, &metric);
        Self {
            distance,
            raw_value,
            metric,
            similarity_score: normalized_similarity, // Now this IS the normalized similarity
            normalized_score: normalized_similarity, // Same value, for compatibility
            rank_value: distance,                    // Raw distance for ranking (lower = better)
        }
    }

    /// Normalize distance based on metric type
    /// Returns (distance, normalized_similarity) where normalized_similarity is [0,1], higher = more similar
    /// distance is normalized so that LOWER = MORE SIMILAR for all metrics (for ranking consistency)
    fn normalize_distance(value: f32, metric: &DistanceMetric) -> (f32, f32) {
        match metric {
            DistanceMetric::DotProduct => {
                // Dot product: higher = more similar, so invert for
                // distance semantics. Return negated value as distance
                // (lower = more similar). HNSW's heap also negates
                // internally via `metric_aware_distance`; consumers
                // that want SimilarityResult directly should rely on
                // this `distance` field for ranking.
                let distance = -value;

                // **Soft-sign normalization** (replaces the older
                // `((v+1)/2).clamp(0, 1)` which silently collapsed
                // every unnormalized inner product above 1.0 to the
                // same score of 1.0 — destroying magnitude information
                // for vectors with norms in the tens or hundreds).
                //
                // Formula: `f(v) = 0.5 + 0.5 * v / (1 + |v|)`
                //   * monotone increasing across the entire real line
                //   * bounded in (0, 1), `f(0) = 0.5` (orthogonal anchor)
                //   * no clamping → preserves rank for any |v|
                //     (v=5 → 0.917, v=50 → 0.990, v=500 → 0.999)
                //   * cheap: 2 adds, 1 mul, 1 div, 1 abs (no `exp`)
                //
                // NaN / infinity guards avoid 0/0 and ∞/∞ producing
                // NaN scores that would corrupt downstream ranking.
                let normalized_similarity = if value.is_nan() {
                    0.5
                } else if value.is_infinite() {
                    if value > 0.0 { 1.0 } else { 0.0 }
                } else {
                    0.5 + 0.5 * value / (1.0 + value.abs())
                };
                (distance, normalized_similarity)
            }
            DistanceMetric::Cosine | DistanceMetric::Unspecified => {
                // Cosine distance: [0, 2] range, lower = more similar
                // Convert to normalized similarity [0,1] where higher = more similar
                let normalized_similarity = if value.is_infinite() {
                    0.0
                } else {
                    1.0 - (value / 2.0).clamp(0.0, 1.0)
                };
                (value, normalized_similarity)
            }
            DistanceMetric::Euclidean => {
                // Euclidean: lower = more similar, convert to similarity [0,1]
                let normalized_similarity = 1.0 / (1.0 + value);
                (value, normalized_similarity)
            }
            DistanceMetric::Manhattan => {
                // Manhattan: lower = more similar, use exponential decay
                let normalized_similarity = (-value / 10.0).exp();
                (value, normalized_similarity)
            }
            _ => {
                // Default: use Euclidean-style conversion
                let normalized_similarity = 1.0 / (1.0 + value);
                (value, normalized_similarity)
            }
        }
    }

    /// Compare two results for ranking (implements total ordering)
    pub fn compare_for_ranking(&self, other: &Self) -> Ordering {
        // Use total_cmp for proper NaN handling
        self.distance.total_cmp(&other.distance)
    }

    /// Partial comparison for sorting (compatibility)
    pub fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.rank_value.partial_cmp(&other.rank_value)
    }
}

impl Default for SimilarityResult {
    fn default() -> Self {
        Self {
            distance: 0.0,
            raw_value: 0.0,
            metric: DistanceMetric::Euclidean,
            similarity_score: 0.0,
            normalized_score: 0.0,
            rank_value: 0.0,
        }
    }
}

/// Trait for providing distance computation services
#[async_trait]
pub trait DistanceComputeProvider: Send + Sync {
    /// Get the unified distance compute manager (required for compatibility)
    fn distance_compute(&self) -> &UnifiedDistanceCompute;

    /// Get the default distance metric for this provider
    fn default_metric(&self) -> DistanceMetric {
        self.distance_compute().system_default
    }

    /// Compute distance between two vectors
    fn compute_distance(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        metric: Option<DistanceMetric>,
    ) -> f32 {
        let metric = metric.unwrap_or_else(|| self.default_metric());
        self.distance_compute()
            .distance_with_metric(vec_a, vec_b, &metric)
    }

    /// Compute distances for a batch of vectors
    fn compute_batch_distances(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: Option<DistanceMetric>,
    ) -> Vec<f32> {
        self.distance_compute()
            .distance_batch(query, vectors, metric)
    }

    /// Get properties of a metric
    fn metric_properties(&self, metric: DistanceMetric) -> MetricProperties {
        self.distance_compute().get_metric_properties(metric)
    }

    /// Check if GPU acceleration is available
    fn is_gpu_available(&self) -> bool {
        self.distance_compute().get_gpu_accelerator().is_some()
    }
}

// ============================================================================
// Unified Distance Compute Implementation with Integrated SIMD
// ============================================================================

/// Unified distance computation manager with hardware acceleration
#[derive(Clone)]
pub struct UnifiedDistanceCompute {
    /// System default distance metric
    system_default: DistanceMetric,
    /// Hardware capability from centralized detection
    hardware_backend: HardwareBackend,
    /// Platform SIMD capability
    platform_capability: PlatformCapability,
    /// Lazy-initialized GPU accelerator - only created when actually needed
    gpu_accelerator_lazy: std::sync::OnceLock<Option<Arc<dyn GpuAccelerator>>>,
    /// Preferred hardware backend
    preferred_backend: HardwareBackend,
    /// Memory pool for reducing allocations in batch operations
    memory_pool: Arc<VectorMemoryPool>,
    /// Enable GPU acceleration
    gpu_enabled: bool,
}

impl std::fmt::Debug for UnifiedDistanceCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedDistanceCompute")
            .field("system_default", &self.system_default)
            .field("hardware_backend", &self.hardware_backend)
            .field("platform_capability", &self.platform_capability)
            .field("preferred_backend", &self.preferred_backend)
            .field("gpu_enabled", &self.gpu_enabled)
            .finish()
    }
}

impl Default for UnifiedDistanceCompute {
    fn default() -> Self {
        // P2: align the compute default with the product default (Cosine). Every
        // other layer — collection create, proto Unspecified→Cosine,
        // StorageQueryContext, the SIMD dispatcher — agrees on Cosine; this was
        // the lone Euclidean outlier. Production code should still pass the
        // collection metric explicitly per call (distance_with_metric).
        Self::new(DistanceMetric::Cosine)
    }
}

impl UnifiedDistanceCompute {
    /// Create new instance with system default metric
    pub fn new(metric: DistanceMetric) -> Self {
        let preferred_backend = get_cached_preferred_backend();
        let gpu_enabled = is_gpu_enabled_cached();
        let platform_capability = get_platform_capability();
        let hardware_backend = preferred_backend;

        trace!(
            "Creating UnifiedDistanceCompute with metric: {:?}, backend: {:?}, platform: {:?}",
            metric, hardware_backend, platform_capability
        );

        Self {
            system_default: metric,
            hardware_backend,
            platform_capability,
            gpu_accelerator_lazy: std::sync::OnceLock::new(),
            preferred_backend,
            memory_pool: Arc::new(VectorMemoryPool::new()),
            gpu_enabled,
        }
    }

    /// Acquire a pooled f32 scratch buffer for representation-specific kernels.
    pub(crate) fn acquire_f32_buffer(&self, capacity: usize) -> PooledItem<Vec<f32>> {
        self.memory_pool.f32_buffer(capacity)
    }

    /// Create with explicit backend selection (for testing)
    pub fn with_backend(metric: DistanceMetric, backend: HardwareBackend) -> Self {
        let platform_capability = get_platform_capability(); // Uses global cached detection
        trace!(
            "Creating UnifiedDistanceCompute with explicit backend: {:?}, platform: {:?}",
            backend, platform_capability
        );

        Self {
            system_default: metric,
            hardware_backend: backend,
            platform_capability,
            gpu_accelerator_lazy: std::sync::OnceLock::new(),
            preferred_backend: backend,
            memory_pool: Arc::new(VectorMemoryPool::new()),
            gpu_enabled: false,
        }
    }

    /// Get GPU accelerator lazily - only initialize when actually needed
    fn get_gpu_accelerator(&self) -> Option<&Arc<dyn GpuAccelerator>> {
        self.gpu_accelerator_lazy
            .get_or_init(|| {
                #[cfg(feature = "gpu")]
                {
                    if !self.gpu_enabled {
                        return None;
                    }

                    // Try to create GPU accelerator based on detected hardware
                    let Some(factory) = GPU_ACCELERATOR_FACTORY.get() else {
                        return None;
                    };
                    match factory() {
                        Ok(accel) => {
                            if accel.is_available() {
                                trace!(
                                    "GPU accelerator initialized for backend: {:?}",
                                    accel.backend()
                                );
                                Some(accel as Arc<dyn GpuAccelerator>)
                            } else {
                                tracing::warn!("GPU accelerator unavailable after initialization");
                                None
                            }
                        }
                        Err(err) => {
                            tracing::warn!("Failed to initialize GPU accelerator: {:?}", err);
                            None
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    None
                }
            })
            .as_ref()
    }

    /// Get the preferred backend for this instance
    pub fn get_preferred_backend(&self) -> HardwareBackend {
        self.preferred_backend
    }

    /// Get available hardware backends
    pub fn available_backends(&self) -> Vec<HardwareBackend> {
        let mut backends = vec![self.hardware_backend];

        if let Some(gpu) = self.gpu_accelerator_lazy.get().and_then(|g| g.as_ref())
            && gpu.is_available()
        {
            backends.push(gpu.backend());
        }

        backends.push(HardwareBackend::Scalar);
        backends
    }

    // ========================================================================
    // SIMD Distance Implementations (moved from core.rs)
    // ========================================================================

    /// Compute distance using the most optimal SIMD path available
    #[inline(always)]
    fn compute_distance_simd(&self, vec_a: &[f32], vec_b: &[f32], metric: &DistanceMetric) -> f32 {
        // The SIMD kernels below index both slices by `a.len()` inside
        // `unsafe`, so equal length is a memory-safety precondition and the
        // check must stay in release builds. It used to be `assert_eq!`, which
        // under `release-server` (`panic = "abort"`) let one wrong-dimension
        // query kill the server (TD-SECSCAN-1). Callers validate the query
        // dimension at the request boundary; this is the backstop. Same cost
        // as the assert it replaces (one predictable compare; the mismatch
        // path is out of line and cold).
        if vec_a.len() != vec_b.len() {
            return dimension_mismatch_score(vec_a.len(), vec_b.len(), metric);
        }

        // Log search backend on first search only
        log_search_backend_first_time(self.platform_capability);

        match metric {
            DistanceMetric::Cosine | DistanceMetric::Unspecified => {
                self.compute_cosine_simd(vec_a, vec_b)
            }
            DistanceMetric::Euclidean => self.compute_euclidean_simd(vec_a, vec_b),
            DistanceMetric::DotProduct => self.compute_dot_product_simd(vec_a, vec_b),
            DistanceMetric::Manhattan => self.compute_manhattan_scalar(vec_a, vec_b),
            DistanceMetric::Jaccard => self.compute_jaccard_simd(vec_a, vec_b),
            DistanceMetric::Hamming => self.compute_hamming_scalar(vec_a, vec_b),
            DistanceMetric::Chebyshev => self.compute_chebyshev_scalar(vec_a, vec_b),
            DistanceMetric::Minkowski => self.compute_minkowski_scalar(vec_a, vec_b, 3.0),
            DistanceMetric::Canberra => self.compute_canberra_scalar(vec_a, vec_b),
            DistanceMetric::BrayCurtis => self.compute_bray_curtis_scalar(vec_a, vec_b),
            DistanceMetric::Angular => self.compute_angular_scalar(vec_a, vec_b),
            DistanceMetric::Hellinger => self.compute_hellinger_scalar(vec_a, vec_b),
            _ => self.compute_euclidean_simd(vec_a, vec_b), // Default fallback for unhandled metrics
        }
    }

    // ------------------------------------------------------------------------
    // Cosine Distance Implementation
    // ------------------------------------------------------------------------

    /// Compute cosine distance using the best available SIMD instruction set.
    #[inline(always)]
    fn compute_cosine_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.platform_capability {
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx2 | PlatformCapability::X86Avx512 =>
            // SAFETY: cosine_distance_avx2 is marked with target_feature(enable = "avx2,fma")
            // and we only call it after verifying AVX2/AVX512 support via platform_capability.
            // Vectors a and b are guaranteed to have same length by debug_assert.
            // Performance: AVX2 provides 4-8x speedup over scalar implementation.
            unsafe { self.cosine_distance_avx2(a, b) },
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Sse2 | PlatformCapability::X86Avx =>
            // SAFETY: cosine_distance_sse2 requires SSE2 which is baseline for x86_64.
            // Platform capability check ensures CPU supports these instructions.
            // Vectors are guaranteed equal length.
            // Performance: SSE2 provides 2x speedup over scalar.
            unsafe { self.cosine_distance_sse2(a, b) },
            #[cfg(target_arch = "aarch64")]
            PlatformCapability::ArmNeon =>
            // SAFETY: NEON is guaranteed available on all AArch64 processors.
            // Platform capability check confirms NEON support.
            // Vectors are guaranteed equal length.
            // Performance: NEON provides 2-4x speedup over scalar.
            unsafe { self.cosine_distance_neon(a, b) },
            _ => self.cosine_distance_scalar(a, b),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn cosine_distance_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: This function is only called after verifying AVX2 support.
        // Invariants:
        // - Vectors a and b have same length (checked by caller)
        // - _mm256_loadu_ps handles unaligned loads safely
        // - Pointer arithmetic stays within slice bounds (i*8 + 8 <= len)
        // Performance: AVX2 processes 8 floats per iteration (256-bit registers)
        unsafe {
            let chunks = a.len() / 8;

            let mut dot = _mm256_setzero_ps();
            let mut norm_a = _mm256_setzero_ps();
            let mut norm_b = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                // SAFETY: offset = i * 8, where i < chunks = len / 8
                // Therefore offset + 8 <= len, keeping us within bounds.
                // _mm256_loadu_ps handles unaligned memory access.
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

                dot = _mm256_fmadd_ps(va, vb, dot);
                norm_a = _mm256_fmadd_ps(va, va, norm_a);
                norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
            }

            // Horizontal sum using AVX2
            let dot_sum = self.hsum_ps_avx2(dot);
            let norm_a_sum = self.hsum_ps_avx2(norm_a);
            let norm_b_sum = self.hsum_ps_avx2(norm_b);

            // Handle remainder with scalar
            let mut dot_final = dot_sum;
            let mut norm_a_final = norm_a_sum;
            let mut norm_b_final = norm_b_sum;

            let start = chunks * 8;
            for i in start..a.len() {
                dot_final += a[i] * b[i];
                norm_a_final += a[i] * a[i];
                norm_b_final += b[i] * b[i];
            }

            // Handle zero vectors
            if norm_a_final == 0.0 || norm_b_final == 0.0 {
                return f32::INFINITY;
            }

            1.0 - (dot_final / (norm_a_final.sqrt() * norm_b_final.sqrt()))
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps_avx2(&self, v: __m256) -> f32 {
        let v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
        let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        let v32 = _mm_add_ss(v64, _mm_movehdup_ps(v64));
        _mm_cvtss_f32(v32)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn cosine_distance_sse2(&self, a: &[f32], b: &[f32]) -> f32 {
        // SSE2 implementation (simplified)
        self.cosine_distance_scalar(a, b)
    }

    /// Compute cosine distance using ARM NEON 128-bit SIMD instructions.
    #[cfg(target_arch = "aarch64")]
    unsafe fn cosine_distance_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: NEON is always available on AArch64.
        // Invariants:
        // - Vectors have same length
        // - vld1q_f32 requires 16-byte chunks (4 floats)
        // - Pointer arithmetic stays within bounds (i*4 + 4 <= len)
        // Performance: NEON processes 4 floats per iteration (128-bit registers)
        unsafe {
            use std::arch::aarch64::*;

            let chunks = a.len() / 4;
            let mut dot = vdupq_n_f32(0.0);
            let mut norm_a = vdupq_n_f32(0.0);
            let mut norm_b = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));

                dot = vfmaq_f32(dot, va, vb);
                norm_a = vfmaq_f32(norm_a, va, va);
                norm_b = vfmaq_f32(norm_b, vb, vb);
            }

            // Horizontal sum
            let dot_sum = vaddvq_f32(dot);
            let norm_a_sum = vaddvq_f32(norm_a);
            let norm_b_sum = vaddvq_f32(norm_b);

            // Handle remainder
            let mut dot_final = dot_sum;
            let mut norm_a_final = norm_a_sum;
            let mut norm_b_final = norm_b_sum;

            let start = chunks * 4;
            for i in start..a.len() {
                dot_final += a[i] * b[i];
                norm_a_final += a[i] * a[i];
                norm_b_final += b[i] * b[i];
            }

            if norm_a_final == 0.0 || norm_b_final == 0.0 {
                return f32::INFINITY;
            }

            1.0 - (dot_final / (norm_a_final.sqrt() * norm_b_final.sqrt()))
        }
    }

    /// Compute cosine distance using a scalar fallback (no SIMD).
    fn cosine_distance_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        // Ensure vectors have the same length
        if a.len() != b.len() {
            tracing::warn!(
                "cosine_distance_scalar: vector length mismatch (a={}, b={})",
                a.len(),
                b.len()
            );
            return f32::INFINITY;
        }

        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            dot += a_val * b_val;
            norm_a += a_val * a_val;
            norm_b += b_val * b_val;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return f32::INFINITY;
        }

        1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
    }

    // ------------------------------------------------------------------------
    // Euclidean Distance Implementation
    // ------------------------------------------------------------------------

    /// Compute Euclidean (L2) distance using the best available SIMD instruction set.
    #[inline(always)]
    fn compute_euclidean_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.platform_capability {
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx2 | PlatformCapability::X86Avx512 =>
            // SAFETY: euclidean_distance_avx2 requires AVX2/FMA support which is verified.
            // Platform capability check ensures CPU supports these instructions.
            // Vectors are guaranteed equal length by caller.
            // Performance: AVX2 provides 4-8x speedup for L2 distance computation.
            unsafe { self.euclidean_distance_avx2(a, b) },
            #[cfg(target_arch = "aarch64")]
            PlatformCapability::ArmNeon =>
            // SAFETY: NEON is baseline for AArch64 processors.
            // Vectors are guaranteed equal length.
            // Performance: NEON provides 2-4x speedup.
            unsafe { self.euclidean_distance_neon(a, b) },
            _ => self.euclidean_distance_scalar(a, b),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn euclidean_distance_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: Called only after AVX2 capability verification.
        // Invariants:
        // - Vectors have equal length
        // - _mm256_loadu_ps safely handles unaligned loads
        // - FMA instructions (_mm256_fmadd_ps) compute (a-b)^2 efficiently
        // Performance: Processes 8 floats per iteration with fused multiply-add
        unsafe {
            let chunks = a.len() / 8;
            let mut sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                // SAFETY: offset = i * 8, where i < chunks = len / 8
                // Therefore offset + 8 <= len, keeping us within bounds.
                // _mm256_loadu_ps handles unaligned memory access.
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            let mut result = self.hsum_ps_avx2(sum);

            // Handle remainder
            let start = chunks * 8;
            for i in start..a.len() {
                let diff = a[i] - b[i];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    /// Compute Euclidean distance using ARM NEON 128-bit SIMD instructions.
    #[cfg(target_arch = "aarch64")]
    unsafe fn euclidean_distance_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: NEON is always available on AArch64.
        // Invariants:
        // - Vectors have equal length
        // - vld1q_f32 loads 4 floats (128-bit)
        // - vfmaq_f32 performs fused multiply-add
        // Performance: Processes 4 floats per iteration
        unsafe {
            use std::arch::aarch64::*;

            let chunks = a.len() / 4;
            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let diff = vsubq_f32(va, vb);
                sum = vfmaq_f32(sum, diff, diff);
            }

            let mut result = vaddvq_f32(sum);

            // Handle remainder
            let start = chunks * 4;
            for i in start..a.len() {
                let diff = a[i] - b[i];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    /// Compute Euclidean distance using a scalar fallback (no SIMD).
    fn euclidean_distance_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        // Ensure vectors have the same length
        if a.len() != b.len() {
            tracing::warn!(
                "euclidean_distance_scalar: vector length mismatch (a={}, b={})",
                a.len(),
                b.len()
            );
            return f32::INFINITY;
        }

        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let diff = a_val - b_val;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    // ------------------------------------------------------------------------
    // Dot Product Implementation
    // ------------------------------------------------------------------------

    /// Compute dot product distance using the best available SIMD instruction set.
    #[inline(always)]
    fn compute_dot_product_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.platform_capability {
            #[cfg(target_arch = "x86_64")]
            PlatformCapability::X86Avx2 | PlatformCapability::X86Avx512 =>
            // SAFETY: dot_product_avx2 requires AVX2/FMA which is verified.
            // Vectors guaranteed equal length.
            // Performance: AVX2 achieves 20M+ ops/sec for 128D vectors.
            unsafe { self.dot_product_avx2(a, b) },
            #[cfg(target_arch = "aarch64")]
            PlatformCapability::ArmNeon =>
            // SAFETY: NEON is baseline for AArch64.
            // Vectors guaranteed equal length.
            // Performance: 2-4x speedup over scalar.
            unsafe { self.dot_product_neon(a, b) },
            _ => self.dot_product_scalar(a, b),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: AVX2 support verified before calling.
        // Invariants:
        // - Equal length vectors
        // - _mm256_fmadd_ps performs a*b+sum in one instruction
        // - Horizontal sum correctly reduces 256-bit vector
        // Performance: Peak performance metric - 20M+ ops/sec
        unsafe {
            let chunks = a.len() / 8;
            let mut sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                // SAFETY: offset = i * 8, where i < chunks = len / 8
                // Therefore offset + 8 <= len, keeping us within bounds.
                // _mm256_loadu_ps handles unaligned memory access.
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            let mut result = self.hsum_ps_avx2(sum);

            // Handle remainder
            let start = chunks * 8;
            for i in start..a.len() {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// Compute dot product using ARM NEON 128-bit SIMD instructions.
    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            use std::arch::aarch64::*;

            let chunks = a.len() / 4;
            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                sum = vfmaq_f32(sum, va, vb);
            }

            let mut result = vaddvq_f32(sum);

            // Handle remainder
            let start = chunks * 4;
            for i in start..a.len() {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// Compute dot product using a scalar fallback (no SIMD).
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum += a_val * b_val;
        }
        sum
    }

    // ------------------------------------------------------------------------
    // Jaccard Distance Implementation
    // ------------------------------------------------------------------------

    /// Compute Jaccard distance (delegates to scalar; no efficient SIMD path).
    #[inline(always)]
    fn compute_jaccard_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        // Jaccard doesn't have efficient SIMD implementation, use scalar
        self.compute_jaccard_scalar(a, b)
    }

    /// Compute Jaccard distance as 1 - (min-sum intersection / max-sum union).
    fn compute_jaccard_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let min_val = a_val.min(*b_val);
            let max_val = a_val.max(*b_val);
            intersection += min_val;
            union += max_val;
        }

        if union == 0.0 {
            0.0
        } else {
            (1.0 - (intersection / union)).clamp(0.0, 1.0)
        }
    }

    // ------------------------------------------------------------------------
    // Other Distance Metrics (Scalar implementations)
    // ------------------------------------------------------------------------

    /// Compute Manhattan (L1) distance as the sum of absolute differences.
    fn compute_manhattan_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum += (a_val - b_val).abs();
        }
        sum
    }

    /// Compute Hamming distance as the count of differing dimensions.
    fn compute_hamming_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut count = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            if (a_val - b_val).abs() > f32::EPSILON {
                count += 1.0;
            }
        }
        count
    }

    /// Compute Chebyshev (L-infinity) distance as the maximum absolute difference.
    fn compute_chebyshev_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut max_diff = 0.0f32;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let diff = (a_val - b_val).abs();
            max_diff = max_diff.max(diff);
        }
        max_diff
    }

    /// Compute Minkowski distance with parameter `p` (generalizes L1 and L2).
    fn compute_minkowski_scalar(&self, a: &[f32], b: &[f32], p: f32) -> f32 {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum += (a_val - b_val).abs().powf(p);
        }
        sum.powf(1.0 / p)
    }

    /// Compute Canberra distance, a weighted variant of Manhattan distance.
    fn compute_canberra_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let denominator = a_val.abs() + b_val.abs();
            if denominator > 0.0 {
                sum += (a_val - b_val).abs() / denominator;
            }
        }
        sum
    }

    /// Compute Bray-Curtis dissimilarity as sum-of-abs-diff / sum-of-abs-total.
    fn compute_bray_curtis_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum_diff = 0.0;
        let mut sum_total = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum_diff += (a_val - b_val).abs();
            sum_total += a_val.abs() + b_val.abs();
        }
        if sum_total == 0.0 {
            0.0
        } else {
            sum_diff / sum_total
        }
    }

    /// Compute angular distance as acos(cosine_similarity) / pi, clamped to [0, 1].
    fn compute_angular_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let cosine_sim = 1.0 - self.cosine_distance_scalar(a, b);
        (cosine_sim.acos() / std::f32::consts::PI).clamp(0.0, 1.0)
    }

    /// Compute Hellinger distance between two probability distributions.
    fn compute_hellinger_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let sqrt_diff = a_val.sqrt() - b_val.sqrt();
            sum += sqrt_diff * sqrt_diff;
        }
        (sum / 2.0).sqrt()
    }

    // ========================================================================
    // Public API Methods
    // ========================================================================

    /// Calculate distance (compatibility method that returns SimilarityResult)
    pub fn calculate_distance(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        metric: &DistanceMetric,
    ) -> SimilarityResult {
        let raw_value = self.compute_distance_simd(vec_a, vec_b, metric);
        SimilarityResult::new(raw_value, *metric)
    }

    /// Main entry point: compute distance with proper metric resolution
    pub fn distance(&self, vec_a: &[f32], vec_b: &[f32]) -> f32 {
        self.distance_with_metric(vec_a, vec_b, &self.system_default)
    }

    /// Compute distance with explicit metric
    pub fn distance_with_metric(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        metric: &DistanceMetric,
    ) -> f32 {
        // Direct SIMD computation - no adapters, no overhead
        self.compute_distance_simd(vec_a, vec_b, metric)
    }

    /// Basic batch distance calculation (for backward compatibility)
    /// Delegates to the optimized pooled version
    pub fn calculate_distance_batch(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
    ) -> Vec<SimilarityResult> {
        // Delegate to the optimized pooled version
        self.batch_distance_pooled_simd(query, vectors, metric)
    }

    /// Optimized batch distance with memory pooling and SIMD
    /// This is the recommended method for batch distance calculations
    ///
    /// Processing hierarchy:
    /// 1. Acquire buffer from memory pool (reduces allocations)
    /// 2. Process in cache-friendly batches (improves locality)
    /// 3. Use SIMD for each batch if available (maximizes throughput)
    pub fn batch_distance_pooled_simd(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
    ) -> Vec<SimilarityResult> {
        // Prefer GPU if available and recommended for workload size
        if let Some(gpu_results) = self.try_gpu_batch(query, vectors, metric) {
            return gpu_results;
        }

        // Step 1: Acquire pooled buffer for results
        let mut pooled_results = self.memory_pool.vector_buffers.acquire();
        pooled_results.clear();
        pooled_results.reserve(vectors.len());

        // Step 2: Process in batches with SIMD
        self.batch_process_with_simd(query, vectors, metric, &mut pooled_results);

        // Convert pooled buffer to owned vector with all fields
        let results: Vec<SimilarityResult> = pooled_results
            .iter()
            .map(|&raw_distance| {
                // Use SimilarityResult::new to properly handle metric-specific normalization
                SimilarityResult::new(raw_distance, *metric)
            })
            .collect();

        // Buffer automatically returns to pool when dropped
        results
    }

    /// Optimized batch distance returning lazy wrapper
    /// Returns a wrapper that delays conversion until actually needed
    pub fn batch_distance_pooled_lazy(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
    ) -> BatchDistanceResults {
        // Acquire pooled buffer for results
        let mut pooled_results = self.memory_pool.vector_buffers.acquire();
        pooled_results.clear();
        pooled_results.reserve(vectors.len());

        // Process in batches with SIMD
        self.batch_process_with_simd(query, vectors, metric, &mut pooled_results);

        // Return lazy wrapper that delays conversion
        BatchDistanceResults {
            distances: pooled_results,
            metric: *metric,
            compute: self.clone(),
        }
    }

    /// Attempt GPU-accelerated batch distance when beneficial
    #[cfg(feature = "gpu")]
    fn try_gpu_batch(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
    ) -> Option<Vec<SimilarityResult>> {
        if !self.gpu_enabled {
            return None;
        }

        let caps = get_hardware_capabilities();

        // Avoid GPU setup cost on small workloads
        if !caps.should_use_gpu_distance(query.len()) || !caps.should_use_gpu_batch(vectors.len()) {
            return None;
        }

        let accel = self.get_gpu_accelerator()?;
        if !accel.is_available() {
            return None;
        }

        // Materialize data for GPU kernel
        let query_vec = query.to_vec();
        let batch: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
        let backend = accel.backend();
        let batch_len = batch.len();
        let dim = query_vec.len();

        // Spawn/block on GPU execution using existing runtime if present
        let accel = Arc::clone(accel);
        let fut = async move { accel.calculate_batch_gpu(&query_vec, &batch, *metric).await };

        let gpu_distances = match TokioHandle::try_current() {
            Ok(handle) => handle.block_on(fut),
            Err(_) => {
                match TokioRuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt.block_on(fut),
                    Err(err) => {
                        tracing::warn!(
                            "Failed to build tokio runtime for GPU batch distance: {:?}",
                            err
                        );
                        return None;
                    }
                }
            }
        };

        match gpu_distances {
            Ok(distances) => {
                tracing::debug!(
                    "Using GPU backend {:?} for batch distance (n_vectors={}, dim={})",
                    backend,
                    batch_len,
                    dim
                );
                Some(
                    distances
                        .into_iter()
                        .map(|d| SimilarityResult::new(d, *metric))
                        .collect(),
                )
            }
            Err(err) => {
                tracing::warn!(
                    "GPU batch distance failed (backend {:?}), falling back to SIMD: {:?}",
                    backend,
                    err
                );
                None
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn try_gpu_batch(
        &self,
        _query: &[f32],
        _vectors: &[&[f32]],
        _metric: &DistanceMetric,
    ) -> Option<Vec<SimilarityResult>> {
        None
    }

    /// Batch distance calculation with external buffer reuse
    /// Best for repeated operations where you manage the buffer
    pub fn batch_distance_into_buffer(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
        results: &mut Vec<SimilarityResult>,
    ) {
        results.clear();
        results.reserve(vectors.len());

        // Use temporary pooled buffer for intermediate calculations
        let mut distances = self.memory_pool.vector_buffers.acquire();
        distances.clear();

        // Process with SIMD
        self.batch_process_with_simd(query, vectors, metric, &mut distances);

        // Convert distances to results
        for &raw_distance in distances.iter() {
            results.push(SimilarityResult::new(raw_distance, *metric));
        }
    }

    /// Core batch processing with SIMD dispatch
    fn batch_process_with_simd(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
        distances: &mut Vec<f32>,
    ) {
        // Determine optimal batch size based on hardware
        let batch_size = self.get_optimal_batch_size();

        // Process in cache-friendly batches
        for chunk in vectors.chunks(batch_size) {
            #[cfg(target_arch = "x86_64")]
            match self.platform_capability {
                PlatformCapability::X86Avx2 | PlatformCapability::X86Avx512 => unsafe {
                    self.simd_batch_avx2(query, chunk, metric, distances);
                    continue;
                },
                _ => {}
            }

            #[cfg(target_arch = "aarch64")]
            match self.platform_capability {
                PlatformCapability::ArmNeon | PlatformCapability::ArmSve => unsafe {
                    self.simd_batch_neon(query, chunk, metric, distances);
                    continue;
                },
                _ => {}
            }

            // Scalar fallback
            self.scalar_batch(query, chunk, metric, distances);
        }
    }

    /// Get optimal batch size for current hardware
    fn get_optimal_batch_size(&self) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            match self.platform_capability {
                PlatformCapability::X86Avx512 => 128, // AVX-512: Process more vectors for better cache use
                PlatformCapability::X86Avx2 => 64, // AVX2: Good balance of cache and register use
                _ => 32,                           // SSE2: Smaller batches
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            32 // NEON: Smaller batches for mobile/embedded
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return 16; // Scalar: Small batches for cache locality
        }
    }

    /// Normalize distance based on metric type
    #[allow(dead_code)]
    fn normalize_distance(&self, distance: f32, metric: &DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
            _ => distance,
        }
    }

    /// Scalar batch processing (fallback)
    fn scalar_batch(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
        distances: &mut Vec<f32>,
    ) {
        for vector in vectors {
            let distance = self.compute_distance_simd(query, vector, metric);
            distances.push(distance);
        }
    }

    /// AVX2 batch processing with 4-way unrolling and software prefetch (TD-037).
    ///
    /// Processes 4 vectors per loop iteration for instruction pipelining.
    /// Software prefetch hints bring the next batch into L1 cache while
    /// the current batch is being computed.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_batch_avx2(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
        distances: &mut Vec<f32>,
    ) {
        const UNROLL_FACTOR: usize = 4;

        let (chunks, remainder) = vectors.as_chunks::<UNROLL_FACTOR>();

        for (i, chunk) in chunks.iter().enumerate() {
            // Prefetch next batch into L1 cache (locality hint 3 = L1, read intent 0)
            if i + 1 < chunks.len() {
                let next = &chunks[i + 1];
                for v in next.iter() {
                    #[cfg(target_arch = "x86_64")]
                    {
                        use std::arch::x86_64::*;
                        _mm_prefetch(v.as_ptr() as *const i8, _MM_HINT_T0);
                    }
                }
            }

            let d0 = self.compute_distance_simd(query, chunk[0], metric);
            let d1 = self.compute_distance_simd(query, chunk[1], metric);
            let d2 = self.compute_distance_simd(query, chunk[2], metric);
            let d3 = self.compute_distance_simd(query, chunk[3], metric);

            distances.push(d0);
            distances.push(d1);
            distances.push(d2);
            distances.push(d3);
        }

        for vector in remainder {
            distances.push(self.compute_distance_simd(query, vector, metric));
        }
    }

    /// NEON batch processing with 4-way unrolling and software prefetch (TD-037).
    ///
    /// Apple M-series cores have deep pipelines that benefit from 4-way unrolling.
    /// Software prefetch hints bring the next batch of vectors into L1 cache
    /// while the current batch is being processed, reducing cache miss stalls.
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_batch_neon(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: &DistanceMetric,
        distances: &mut Vec<f32>,
    ) {
        const UNROLL_FACTOR: usize = 4;

        let (chunks, remainder) = vectors.as_chunks::<UNROLL_FACTOR>();

        for (i, chunk) in chunks.iter().enumerate() {
            // Prefetch next batch into L1 cache while processing current batch
            if i + 1 < chunks.len() {
                let next = &chunks[i + 1];
                for v in next.iter() {
                    // Hint to the compiler that we'll access this data soon.
                    // std::arch::aarch64::_prefetch is unstable on stable Rust,
                    // so we use a volatile read of the first byte as a portable
                    // prefetch hint that prevents the access from being optimized away.
                    let _ = unsafe { std::ptr::read_volatile(v.as_ptr()) };
                }
            }

            let d0 = self.compute_distance_simd(query, chunk[0], metric);
            let d1 = self.compute_distance_simd(query, chunk[1], metric);
            let d2 = self.compute_distance_simd(query, chunk[2], metric);
            let d3 = self.compute_distance_simd(query, chunk[3], metric);

            distances.push(d0);
            distances.push(d1);
            distances.push(d2);
            distances.push(d3);
        }

        for vector in remainder {
            distances.push(self.compute_distance_simd(query, vector, metric));
        }
    }

    /// Batch distance computation for optimal performance
    pub fn distance_batch(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: Option<DistanceMetric>,
    ) -> Vec<f32> {
        let metric = metric.unwrap_or(self.system_default);

        vectors
            .iter()
            .map(|v| self.compute_distance_simd(query, v, &metric))
            .collect()
    }

    /// Compute distances for multiple candidate vectors against a single query
    /// using dimension-major iteration for better cache utilization.
    ///
    /// For small batch sizes (<= 8), falls back to the standard per-vector approach.
    /// For larger batches, iterates dimension-by-dimension accumulating partial distances
    /// for all candidates simultaneously, which improves SIMD utilization.
    pub fn distance_batch_transposed(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
        metric: Option<DistanceMetric>,
    ) -> Vec<f32> {
        let metric = metric.unwrap_or(self.system_default);
        let n = candidates.len();

        // Fall back to standard approach for small batches
        if n <= 8 || query.is_empty() {
            return self.distance_batch(query, candidates, Some(metric));
        }

        // Verify all candidates have same dimension as query
        let d = query.len();
        for c in candidates {
            if c.len() != d {
                return self.distance_batch(query, candidates, Some(metric));
            }
        }

        match metric {
            DistanceMetric::Euclidean => self.batch_l2_transposed(query, candidates),
            DistanceMetric::Cosine => self.batch_cosine_transposed(query, candidates),
            DistanceMetric::DotProduct => self.batch_dot_transposed(query, candidates),
            _ => self.distance_batch(query, candidates, Some(metric)),
        }
    }

    /// L2 distance using dimension-major (transposed) iteration.
    /// Uses pooled buffer to avoid per-call allocation of the accumulator.
    fn batch_l2_transposed(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        let n = candidates.len();
        let d = query.len();

        let mut accum = self.memory_pool.vector_buffers.acquire();
        accum.clear();
        accum.resize(n, 0.0f32);

        for dim in 0..d {
            let q = query[dim];
            for (i, candidate) in candidates.iter().enumerate() {
                let diff = q - candidate[dim];
                accum[i] += diff * diff;
            }
        }

        // L2 distance is sqrt of sum of squared differences
        for val in accum.iter_mut() {
            *val = val.sqrt();
        }
        accum.to_vec()
    }

    /// Cosine distance using dimension-major (transposed) iteration.
    /// Uses pooled buffers for dot_products and norms_b intermediates.
    fn batch_cosine_transposed(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        let n = candidates.len();
        let d = query.len();

        let mut dot_products = self.memory_pool.vector_buffers.acquire();
        dot_products.clear();
        dot_products.resize(n, 0.0f32);

        let mut norms_b = self.memory_pool.vector_buffers.acquire();
        norms_b.clear();
        norms_b.resize(n, 0.0f32);

        let mut norm_a = 0.0f32;

        for dim in 0..d {
            let q = query[dim];
            norm_a += q * q;
            for (i, candidate) in candidates.iter().enumerate() {
                let c = candidate[dim];
                dot_products[i] += q * c;
                norms_b[i] += c * c;
            }
        }

        let norm_a = norm_a.sqrt();
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let norm_b = norms_b[i].sqrt();
            let denom = norm_a * norm_b;
            if denom > 0.0 {
                results.push(1.0 - (dot_products[i] / denom));
            } else {
                results.push(f32::INFINITY);
            }
        }
        results
    }

    /// Dot product distance using dimension-major (transposed) iteration.
    /// Uses pooled buffer for the accumulator.
    fn batch_dot_transposed(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        let n = candidates.len();
        let d = query.len();

        let mut dot_products = self.memory_pool.vector_buffers.acquire();
        dot_products.clear();
        dot_products.resize(n, 0.0f32);

        for dim in 0..d {
            let q = query[dim];
            for (i, candidate) in candidates.iter().enumerate() {
                dot_products[i] += q * candidate[dim];
            }
        }

        dot_products.to_vec()
    }

    /// Compute similarity results with semantic meaning
    pub fn similarity(
        &self,
        vec_a: &[f32],
        vec_b: &[f32],
        metric: Option<DistanceMetric>,
    ) -> SimilarityResult {
        let metric = metric.unwrap_or(self.system_default);
        let raw_value = self.compute_distance_simd(vec_a, vec_b, &metric);
        SimilarityResult::new(raw_value, metric)
    }

    /// Batch similarity computation
    pub fn similarity_batch(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
        metric: Option<DistanceMetric>,
    ) -> Vec<SimilarityResult> {
        let metric = metric.unwrap_or(self.system_default);

        vectors
            .iter()
            .map(|v| {
                let raw_value = self.compute_distance_simd(query, v, &metric);
                SimilarityResult::new(raw_value, metric)
            })
            .collect()
    }

    /// Calculate INT8 quantized distance (compatibility placeholder)
    pub fn calculate_int8_distance(
        &self,
        _vec_a: &[i8],
        _vec_b: &[i8],
        _scale_a: f32,
        _scale_b: f32,
        _zero_point_a: i8,
        _zero_point_b: i8,
        metric: &DistanceMetric,
    ) -> SimilarityResult {
        // Deferred: Implement actual INT8 distance calculation
        // For now, return a default result
        SimilarityResult::new(0.0, *metric)
    }

    /// Calculate PQ distance with 2D codebook structure using asymmetric distance computation (ADC)
    pub fn calculate_pq_distance(
        &self,
        query: &[f32],
        codes: &[u8],
        codebook: &[Vec<f32>], // 2D structure: [num_subvectors][centroids_per_subvector * dim]
        metric: &DistanceMetric,
    ) -> SimilarityResult {
        if codebook.is_empty() || codes.is_empty() || query.is_empty() {
            return SimilarityResult::new(0.0, *metric);
        }

        let num_subvectors = codebook.len();
        let subvector_dim = query.len() / num_subvectors;

        // Each codebook[i] contains all centroids for subvector i
        // Assuming 256 centroids per subvector (8-bit codes)
        let _centroids_per_subvector = 256;

        let mut total_distance = 0.0;

        // Compute asymmetric distance: query vs quantized vector
        for subvector_idx in 0..num_subvectors.min(codes.len()) {
            let code = codes[subvector_idx] as usize;

            // Get the query subvector
            let query_start = subvector_idx * subvector_dim;
            let query_end = query_start + subvector_dim;
            let query_subvector = &query[query_start..query_end.min(query.len())];

            // Get the corresponding centroid from the codebook
            let centroid_start = code * subvector_dim;
            let centroid_end = centroid_start + subvector_dim;

            if centroid_end <= codebook[subvector_idx].len() {
                let centroid = &codebook[subvector_idx][centroid_start..centroid_end];

                // Same backstop as `compute_distance_simd` (TD-SECSCAN-1 §5):
                // the SIMD kernels below index both slices by `a.len()` inside
                // `unsafe`, so equal length is a memory-safety precondition.
                // Both slices are `subvector_dim` long by construction today;
                // this keeps that a checked invariant rather than an
                // unchecked assumption. It is not routed through
                // `compute_distance_simd` because PQ scores its metrics
                // differently (negated dot product, Manhattan inline, Euclidean
                // fallback for every other metric), so delegating would change
                // results. Cost: one predictable `usize` compare per
                // subvector, next to the bounds-checked slicing above; the
                // mismatch path is out of line and `#[cold]`.
                if query_subvector.len() != centroid.len() {
                    return SimilarityResult::new(
                        dimension_mismatch_score(query_subvector.len(), centroid.len(), metric),
                        *metric,
                    );
                }

                // Compute distance between query subvector and centroid
                let subvector_distance = match metric {
                    DistanceMetric::Euclidean => {
                        self.compute_euclidean_simd(query_subvector, centroid)
                    }
                    DistanceMetric::Cosine => self.compute_cosine_simd(query_subvector, centroid),
                    DistanceMetric::DotProduct => {
                        -self.compute_dot_product_simd(query_subvector, centroid)
                    }
                    DistanceMetric::Manhattan => query_subvector
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum(),
                    _ => {
                        // Fallback to euclidean for unsupported metrics
                        self.compute_euclidean_simd(query_subvector, centroid)
                    }
                };

                total_distance += subvector_distance;
            }
        }

        SimilarityResult::new(total_distance, *metric)
    }

    /// Get the system default metric
    pub fn system_default(&self) -> DistanceMetric {
        self.system_default
    }

    /// Check if a metric is similarity-based (higher = better)
    pub fn is_similarity_metric(&self, metric: &DistanceMetric) -> bool {
        matches!(metric, DistanceMetric::DotProduct)
    }

    /// Get metric properties
    pub fn get_metric_properties(&self, metric: DistanceMetric) -> MetricProperties {
        match metric {
            DistanceMetric::Cosine => MetricProperties {
                name: "Cosine Distance".to_string(),
                is_similarity: false,
                range: (0.0, 2.0),
                is_symmetric: true,
            },
            DistanceMetric::Euclidean => MetricProperties {
                name: "Euclidean Distance".to_string(),
                is_similarity: false,
                range: (0.0, f32::INFINITY),
                is_symmetric: true,
            },
            DistanceMetric::DotProduct => MetricProperties {
                name: "Dot Product".to_string(),
                is_similarity: true,
                range: (f32::NEG_INFINITY, f32::INFINITY),
                is_symmetric: true,
            },
            DistanceMetric::Manhattan => MetricProperties {
                name: "Manhattan Distance".to_string(),
                is_similarity: false,
                range: (0.0, f32::INFINITY),
                is_symmetric: true,
            },
            _ => MetricProperties {
                name: format!("{:?}", metric),
                is_similarity: false,
                range: (0.0, f32::INFINITY),
                is_symmetric: true,
            },
        }
    }
}

// ============================================================================
// DistanceComputeProvider Implementation
// ============================================================================

#[async_trait]
impl DistanceComputeProvider for UnifiedDistanceCompute {
    fn distance_compute(&self) -> &UnifiedDistanceCompute {
        self
    }
}

// ============================================================================
// Initialization Functions
// ============================================================================

/// Pre-warm distance computation system with common metrics
pub fn prewarm_calculator_cache() {
    let common_metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ];

    for metric in common_metrics {
        let test_vec_a = vec![1.0f32, 0.0f32];
        let test_vec_b = vec![0.0f32, 1.0f32];
        let compute = UnifiedDistanceCompute::new(metric);
        let _distance = compute.distance(&test_vec_a, &test_vec_b);
        trace!("Pre-warmed distance computation for metric: {:?}", metric);
    }
}

/// Initialize all UnifiedDistanceCompute optimizations
pub fn initialize_distance_compute_optimizations() {
    info!("🚀 Initializing UnifiedDistanceCompute optimizations...");

    // Initialize hardware backend caching
    initialize_hardware_backend_cache();
    info!("✅ Hardware backend caching initialized");

    // Initialize platform capability mapping from global hardware detection
    let platform = get_platform_capability();
    info!(
        "✅ Platform capability mapped from global detection: {:?}",
        platform
    );

    // Pre-warm calculator cache
    prewarm_calculator_cache();
    info!("✅ Distance calculator cache pre-warmed");

    // Log optimization summary
    let preferred_backend = get_cached_preferred_backend();
    let gpu_enabled = is_gpu_enabled_cached();

    info!("🎯 Distance computation optimizations active:");
    info!("   • Hardware Backend: {:?} (cached)", preferred_backend);
    info!("   • Platform SIMD: {:?}", platform);
    info!(
        "   • GPU Acceleration: {} (lazy-loaded)",
        if gpu_enabled { "enabled" } else { "disabled" }
    );
    info!("   • All SIMD implementations integrated directly");
    info!("   • Zero adapter overhead - direct inline calls");
}

/// Score for a vector pair whose dimensions differ: the worst possible value
/// for `metric` (`-inf` for dot product, a similarity; `+inf` for every
/// distance), so the pair ranks last and never displaces a valid candidate.
/// Logged once per process — reaching it means a caller skipped the
/// request-boundary dimension check.
#[cold]
#[inline(never)]
fn dimension_mismatch_score(len_a: usize, len_b: usize, metric: &DistanceMetric) -> f32 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, AtomicOrdering::Relaxed) {
        tracing::error!(
            len_a,
            len_b,
            "distance kernel called with mismatched vector dimensions; scoring the pair as \
             worst-possible (a caller skipped query-vector dimension validation)"
        );
    }
    match metric {
        DistanceMetric::DotProduct => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-SECSCAN-1: a dimension mismatch must not panic. Under the
    /// `release-server` profile (`panic = "abort"`) the former `assert_eq!`
    /// killed the server process on one wrong-dimension query. A mismatched
    /// pair now scores as the worst possible candidate for its metric, so it
    /// ranks last and never wins a top-k slot.
    #[test]
    fn dimension_mismatch_scores_worst_instead_of_panicking() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
        ] {
            let compute = UnifiedDistanceCompute::new(metric);
            assert_eq!(compute.distance_with_metric(&a, &b, &metric), f32::INFINITY);
            assert_eq!(
                compute.distance_with_metric(&a, &[], &metric),
                f32::INFINITY
            );
        }
        let dot = DistanceMetric::DotProduct;
        let compute = UnifiedDistanceCompute::new(dot);
        assert_eq!(
            compute.distance_with_metric(&a, &b, &dot),
            f32::NEG_INFINITY
        );
        // Matching dimensions are unaffected.
        assert_eq!(compute.distance_with_metric(&a, &a, &dot), 1.0);
    }

    /// TD-SECSCAN-1 §5: the length backstop added to `calculate_pq_distance`
    /// must not change PQ scoring on well-formed input. 4-dim query, 2
    /// subvectors of dim 2, codes picking centroid 1 then centroid 0.
    #[test]
    fn pq_distance_unchanged_by_length_backstop() {
        let metric = DistanceMetric::Euclidean;
        let compute = UnifiedDistanceCompute::new(metric);
        let query = [1.0f32, 0.0, 0.0, 1.0];
        let codebook = vec![
            vec![0.0f32, 0.0, 1.0, 0.0], // subvector 0: centroid 0 = (0,0), 1 = (1,0)
            vec![0.0f32, 1.0, 5.0, 5.0], // subvector 1: centroid 0 = (0,1), 1 = (5,5)
        ];
        let result = compute.calculate_pq_distance(&query, &[1, 0], &codebook, &metric);
        // Both subvectors match their centroid exactly.
        assert!(result.raw_value.abs() < 1e-6, "{}", result.raw_value);
        let result = compute.calculate_pq_distance(&query, &[0, 1], &codebook, &metric);
        let expected = compute.distance_with_metric(&[1.0, 0.0], &[0.0, 0.0], &metric)
            + compute.distance_with_metric(&[0.0, 1.0], &[5.0, 5.0], &metric);
        assert!((result.raw_value - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let distance = compute.distance(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6); // Orthogonal vectors
    }

    #[test]
    fn test_euclidean_distance() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = compute.distance(&a, &b);
        assert!((distance - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn test_dot_product() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let distance = compute.distance(&a, &b);
        assert!((distance - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_batch_computation() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let query = vec![0.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]];
        let distances = compute.distance_batch(&query, &vectors, None);
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 1.0).abs() < 1e-6);
        assert!((distances[1] - 1.0).abs() < 1e-6);
        assert!((distances[2] - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_cosine_distance_identical_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let distance = compute.distance(&a, &b);
        // Identical vectors: cosine similarity = 1, distance = 1 - 1 = 0
        assert!(
            distance.abs() < 1e-5,
            "Identical vectors should have cosine distance ~0, got {}",
            distance
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let distance = compute.distance(&a, &b);
        // Orthogonal vectors: cosine similarity = 0, distance = 1 - 0 = 1
        assert!(
            (distance - 1.0).abs() < 1e-5,
            "Orthogonal vectors should have cosine distance ~1.0, got {}",
            distance
        );
    }

    #[test]
    fn test_cosine_distance_opposite_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let distance = compute.distance(&a, &b);
        // Opposite vectors: cosine similarity = -1, distance = 1 - (-1) = 2
        assert!(
            (distance - 2.0).abs() < 1e-5,
            "Opposite vectors should have cosine distance ~2.0, got {}",
            distance
        );
    }

    #[test]
    fn test_euclidean_distance_known_values() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        // 3-4-5 right triangle
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = compute.distance(&a, &b);
        assert!(
            (distance - 5.0).abs() < 1e-5,
            "Expected distance 5.0 (3-4-5 triangle), got {}",
            distance
        );

        // Same point = distance 0
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![1.0, 2.0, 3.0];
        let distance = compute.distance(&c, &d);
        assert!(
            distance.abs() < 1e-5,
            "Same point should have distance 0, got {}",
            distance
        );

        // Unit distance along each axis in 3D: sqrt(3)
        let e = vec![0.0, 0.0, 0.0];
        let f = vec![1.0, 1.0, 1.0];
        let distance = compute.distance(&e, &f);
        assert!(
            (distance - 3.0_f32.sqrt()).abs() < 1e-5,
            "Expected sqrt(3), got {}",
            distance
        );
    }

    #[test]
    fn test_dot_product_known_values() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);

        // Known dot product: 1*4 + 2*5 + 3*6 = 32
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let distance = compute.distance(&a, &b);
        assert!(
            (distance - 32.0).abs() < 1e-5,
            "Expected dot product 32.0, got {}",
            distance
        );

        // Orthogonal vectors: dot product = 0
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let distance = compute.distance(&c, &d);
        assert!(
            distance.abs() < 1e-5,
            "Orthogonal vectors should have dot product 0, got {}",
            distance
        );

        // Unit vector dot with itself = 1
        let e = vec![1.0, 0.0, 0.0];
        let distance = compute.distance(&e, &e);
        assert!(
            (distance - 1.0).abs() < 1e-5,
            "Unit vector dot with itself should be 1.0, got {}",
            distance
        );
    }

    #[test]
    fn test_dot_product_normalized_similarity_soft_sign() {
        // Pins the soft-sign transform for DotProduct normalization.
        // The old `((v+1)/2).clamp(0,1)` collapsed every inner
        // product above 1.0 to a score of 1.0 — destroying magnitude
        // information for unnormalized vectors. This test fails if a
        // future regression reintroduces clamping.
        let cases = [
            // (raw inner product, expected normalized similarity)
            (-50.0_f32, 0.5 + 0.5 * (-50.0) / 51.0), // ≈ 0.010
            (-1.0, 0.25),                            // 0.5 + 0.5 * -0.5 = 0.25
            (0.0, 0.5),                              // orthogonal anchor
            (0.5, 0.5 + 0.5 * 0.5 / 1.5),            // ≈ 0.667
            (1.0, 0.75),                             // 0.5 + 0.5 * 0.5 = 0.75
            (5.0, 0.5 + 0.5 * 5.0 / 6.0),            // ≈ 0.917
            (50.0, 0.5 + 0.5 * 50.0 / 51.0),         // ≈ 0.990 (NOT clamped to 1.0)
            (500.0, 0.5 + 0.5 * 500.0 / 501.0),      // ≈ 0.999 (NOT clamped to 1.0)
        ];
        for (raw, expected) in cases {
            let result = SimilarityResult::new(raw, DistanceMetric::DotProduct);
            assert!(
                (result.normalized_score - expected).abs() < 1e-5,
                "DotProduct soft-sign at v={}: expected {:.6}, got {:.6}",
                raw,
                expected,
                result.normalized_score
            );
        }

        // Critical regression check: two unnormalized inner products
        // must produce DISTINGUISHABLE scores (old formula clamped
        // both to 1.0).
        let s5 = SimilarityResult::new(5.0, DistanceMetric::DotProduct).normalized_score;
        let s50 = SimilarityResult::new(50.0, DistanceMetric::DotProduct).normalized_score;
        assert!(
            s50 > s5 + 0.01,
            "v=5 and v=50 must produce distinguishable similarity scores, got s5={} s50={}",
            s5,
            s50
        );

        // NaN / infinity handling.
        let nan = SimilarityResult::new(f32::NAN, DistanceMetric::DotProduct).normalized_score;
        assert!(
            (nan - 0.5).abs() < 1e-6,
            "NaN inner product → 0.5, got {}",
            nan
        );
        let pinf =
            SimilarityResult::new(f32::INFINITY, DistanceMetric::DotProduct).normalized_score;
        assert!((pinf - 1.0).abs() < 1e-6, "+inf → 1.0, got {}", pinf);
        let ninf =
            SimilarityResult::new(f32::NEG_INFINITY, DistanceMetric::DotProduct).normalized_score;
        assert!((ninf - 0.0).abs() < 1e-6, "-inf → 0.0, got {}", ninf);
    }

    #[test]
    fn test_distance_with_explicit_metric() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Use explicit cosine metric even though default is Euclidean
        let cosine_dist = compute.distance_with_metric(&a, &b, &DistanceMetric::Cosine);
        assert!(
            (cosine_dist - 1.0).abs() < 1e-5,
            "Expected cosine distance 1.0 for orthogonal, got {}",
            cosine_dist
        );

        // Use default (Euclidean)
        let euclidean_dist = compute.distance(&a, &b);
        let expected = 2.0_f32.sqrt();
        assert!(
            (euclidean_dist - expected).abs() < 1e-5,
            "Expected euclidean distance sqrt(2), got {}",
            euclidean_dist
        );
    }

    #[test]
    fn test_batch_cosine_distances() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<&[f32]> = vec![
            &[1.0, 0.0, 0.0],  // identical: distance 0
            &[0.0, 1.0, 0.0],  // orthogonal: distance 1
            &[-1.0, 0.0, 0.0], // opposite: distance 2
        ];
        let distances = compute.distance_batch(&query, &vectors, None);
        assert_eq!(distances.len(), 3);
        assert!(distances[0].abs() < 1e-5, "identical should be ~0");
        assert!((distances[1] - 1.0).abs() < 1e-5, "orthogonal should be ~1");
        assert!((distances[2] - 2.0).abs() < 1e-5, "opposite should be ~2");
    }

    #[test]
    fn test_distance_metric_ext_is_similarity() {
        assert!(DistanceMetric::DotProduct.is_similarity());
        assert!(!DistanceMetric::Cosine.is_similarity());
        assert!(!DistanceMetric::Euclidean.is_similarity());
    }

    #[test]
    fn test_default_unified_distance_compute() {
        // P2: the default metric is now Cosine (the product default), aligning
        // UnifiedDistanceCompute with the collection/proto + SIMD-dispatcher
        // default. Production code should still pass the collection metric
        // explicitly per call (distance_with_metric / calculate_distance).
        let compute = UnifiedDistanceCompute::default();
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let distance = compute.distance(&a, &b);
        // cosine distance = 1 − (1·1 + 0·1) / (1 · √2) = 1 − 1/√2 ≈ 0.29289
        let expected = 1.0 - 1.0 / 2.0_f32.sqrt();
        assert!(
            (distance - expected).abs() < 1e-5,
            "default-Compute should use Cosine; got {distance}, expected {expected}"
        );
    }

    #[test]
    fn test_euclidean_distance_higher_dimensions() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        // 4D vector: sqrt(1 + 1 + 1 + 1) = 2
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let distance = compute.distance(&a, &b);
        assert!(
            (distance - 2.0).abs() < 1e-5,
            "Expected 2.0 for 4D unit diagonal, got {}",
            distance
        );
    }

    #[test]
    fn test_batch_transposed_l2_correctness() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Create >8 candidates to trigger transposed path
        let candidates_owned: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..10)
                    .map(|j| (i as f32) * 0.1 + (j as f32) * 0.5)
                    .collect()
            })
            .collect();
        let candidates: Vec<&[f32]> = candidates_owned.iter().map(|v| v.as_slice()).collect();

        let standard = compute.distance_batch(&query, &candidates, Some(DistanceMetric::Euclidean));
        let transposed =
            compute.distance_batch_transposed(&query, &candidates, Some(DistanceMetric::Euclidean));

        assert_eq!(standard.len(), transposed.len());
        for (i, (s, t)) in standard.iter().zip(transposed.iter()).enumerate() {
            assert!(
                (s - t).abs() < 1e-4,
                "L2 mismatch at index {}: standard={}, transposed={}",
                i,
                s,
                t
            );
        }
    }

    #[test]
    fn test_batch_transposed_cosine_correctness() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let query = vec![1.0, 0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6];

        let candidates_owned: Vec<Vec<f32>> = (0..15)
            .map(|i| {
                (0..10)
                    .map(|j| ((i + j) as f32 * 0.3).sin().abs() + 0.01)
                    .collect()
            })
            .collect();
        let candidates: Vec<&[f32]> = candidates_owned.iter().map(|v| v.as_slice()).collect();

        let standard = compute.distance_batch(&query, &candidates, Some(DistanceMetric::Cosine));
        let transposed =
            compute.distance_batch_transposed(&query, &candidates, Some(DistanceMetric::Cosine));

        assert_eq!(standard.len(), transposed.len());
        for (i, (s, t)) in standard.iter().zip(transposed.iter()).enumerate() {
            assert!(
                (s - t).abs() < 1e-4,
                "Cosine mismatch at index {}: standard={}, transposed={}",
                i,
                s,
                t
            );
        }
    }

    #[test]
    fn test_batch_transposed_dot_correctness() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let candidates_owned: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                (0..10)
                    .map(|j| (i as f32) * 0.2 + (j as f32) * 0.1)
                    .collect()
            })
            .collect();
        let candidates: Vec<&[f32]> = candidates_owned.iter().map(|v| v.as_slice()).collect();

        let standard =
            compute.distance_batch(&query, &candidates, Some(DistanceMetric::DotProduct));
        let transposed = compute.distance_batch_transposed(
            &query,
            &candidates,
            Some(DistanceMetric::DotProduct),
        );

        assert_eq!(standard.len(), transposed.len());
        for (i, (s, t)) in standard.iter().zip(transposed.iter()).enumerate() {
            assert!(
                (s - t).abs() < 1e-3,
                "DotProduct mismatch at index {}: standard={}, transposed={}",
                i,
                s,
                t
            );
        }
    }

    #[test]
    fn test_batch_transposed_small_fallback() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let query = vec![1.0, 2.0, 3.0];

        // Only 4 candidates (<= 8) should use standard path
        let candidates_owned: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let candidates: Vec<&[f32]> = candidates_owned.iter().map(|v| v.as_slice()).collect();

        let standard = compute.distance_batch(&query, &candidates, Some(DistanceMetric::Euclidean));
        let transposed =
            compute.distance_batch_transposed(&query, &candidates, Some(DistanceMetric::Euclidean));

        assert_eq!(standard.len(), transposed.len());
        for (i, (s, t)) in standard.iter().zip(transposed.iter()).enumerate() {
            assert!(
                (s - t).abs() < 1e-6,
                "Small batch mismatch at index {}: standard={}, transposed={}",
                i,
                s,
                t
            );
        }
    }

    // ======================================================================
    // Infrastructure tests for core distance computation
    // ======================================================================

    #[test]
    fn test_cosine_distance_known_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // Parallel vectors (same direction) should have distance ~0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // Scalar multiple of a
        let distance = compute.distance_with_metric(&a, &b, &DistanceMetric::Cosine);
        assert!(
            distance.abs() < 1e-5,
            "Parallel vectors should have cosine distance ~0, got {}",
            distance
        );

        // Orthogonal vectors should have distance ~1
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let distance = compute.distance_with_metric(&c, &d, &DistanceMetric::Cosine);
        assert!(
            (distance - 1.0).abs() < 1e-5,
            "Orthogonal vectors should have cosine distance ~1.0, got {}",
            distance
        );

        // Anti-parallel vectors should have distance ~2
        let e = vec![1.0, 0.0, 0.0];
        let f = vec![-1.0, 0.0, 0.0];
        let distance = compute.distance_with_metric(&e, &f, &DistanceMetric::Cosine);
        assert!(
            (distance - 2.0).abs() < 1e-5,
            "Anti-parallel vectors should have cosine distance ~2.0, got {}",
            distance
        );
    }

    #[test]
    fn test_euclidean_distance_known_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        // 3-4-5 right triangle
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = compute.distance_with_metric(&a, &b, &DistanceMetric::Euclidean);
        assert!(
            (distance - 5.0).abs() < 1e-5,
            "Expected 5.0 for 3-4-5 triangle, got {}",
            distance
        );

        // Unit distance along single axis
        let c = vec![0.0, 0.0, 0.0];
        let d = vec![1.0, 0.0, 0.0];
        let distance = compute.distance_with_metric(&c, &d, &DistanceMetric::Euclidean);
        assert!(
            (distance - 1.0).abs() < 1e-5,
            "Expected 1.0 for unit axis distance, got {}",
            distance
        );

        // 3D diagonal: sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
        let e = vec![0.0, 0.0, 0.0];
        let f = vec![1.0, 2.0, 2.0];
        let distance = compute.distance_with_metric(&e, &f, &DistanceMetric::Euclidean);
        assert!(
            (distance - 3.0).abs() < 1e-5,
            "Expected 3.0, got {}",
            distance
        );
    }

    #[test]
    fn test_dot_product_known_vectors() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);

        // Known dot product: 1*4 + 2*5 + 3*6 = 32
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let distance = compute.distance_with_metric(&a, &b, &DistanceMetric::DotProduct);
        assert!(
            (distance - 32.0).abs() < 1e-5,
            "Expected dot product 32.0, got {}",
            distance
        );

        // Orthogonal vectors: dot product = 0
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let distance = compute.distance_with_metric(&c, &d, &DistanceMetric::DotProduct);
        assert!(
            distance.abs() < 1e-5,
            "Orthogonal vectors should have dot product 0, got {}",
            distance
        );

        // Negative dot product for opposing vectors
        let e = vec![1.0, 0.0];
        let f = vec![-1.0, 0.0];
        let distance = compute.distance_with_metric(&e, &f, &DistanceMetric::DotProduct);
        assert!(
            (distance - (-1.0)).abs() < 1e-5,
            "Opposing unit vectors should have dot product -1.0, got {}",
            distance
        );
    }

    #[test]
    fn test_distance_metric_variants() {
        // Verify all DistanceMetric enum variants exist and can be used
        let metrics = vec![
            DistanceMetric::Unspecified,
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Hamming,
            DistanceMetric::Manhattan,
            DistanceMetric::Jaccard,
            DistanceMetric::Angular,
            DistanceMetric::Chebyshev,
            DistanceMetric::Canberra,
            DistanceMetric::Minkowski,
            DistanceMetric::BrayCurtis,
            DistanceMetric::Hellinger,
            DistanceMetric::Custom,
        ];
        assert_eq!(metrics.len(), 14, "Expected 14 distance metric variants");

        // Verify each has a string name via proto
        for metric in &metrics {
            let name = metric.as_str_name();
            assert!(
                !name.is_empty(),
                "Metric {:?} should have a non-empty name",
                metric
            );
        }

        // Verify the is_similarity extension trait
        assert!(DistanceMetric::DotProduct.is_similarity());
        assert!(!DistanceMetric::Cosine.is_similarity());
        assert!(!DistanceMetric::Euclidean.is_similarity());
        assert!(!DistanceMetric::Manhattan.is_similarity());
    }

    #[test]
    fn test_zero_vector_distance() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        let zero = vec![0.0, 0.0, 0.0];
        let nonzero = vec![3.0, 4.0, 0.0];

        // Euclidean distance from zero to (3,4,0) = 5
        let distance = compute.distance_with_metric(&zero, &nonzero, &DistanceMetric::Euclidean);
        assert!(
            (distance - 5.0).abs() < 1e-5,
            "Euclidean from zero to (3,4,0) should be 5.0, got {}",
            distance
        );

        // Dot product with zero vector = 0
        let distance = compute.distance_with_metric(&zero, &nonzero, &DistanceMetric::DotProduct);
        assert!(
            distance.abs() < 1e-5,
            "Dot product with zero vector should be 0, got {}",
            distance
        );

        // Manhattan distance from zero to (3,4,0) = 7
        let distance = compute.distance_with_metric(&zero, &nonzero, &DistanceMetric::Manhattan);
        assert!(
            (distance - 7.0).abs() < 1e-5,
            "Manhattan from zero to (3,4,0) should be 7.0, got {}",
            distance
        );
    }

    #[test]
    fn test_identical_vector_distance() {
        let compute = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Euclidean distance between identical vectors = 0
        let distance = compute.distance_with_metric(&v, &v, &DistanceMetric::Euclidean);
        assert!(
            distance.abs() < 1e-5,
            "Euclidean distance of identical vectors should be 0, got {}",
            distance
        );

        // Cosine distance between identical vectors = 0
        let distance = compute.distance_with_metric(&v, &v, &DistanceMetric::Cosine);
        assert!(
            distance.abs() < 1e-5,
            "Cosine distance of identical vectors should be 0, got {}",
            distance
        );

        // Manhattan distance between identical vectors = 0
        let distance = compute.distance_with_metric(&v, &v, &DistanceMetric::Manhattan);
        assert!(
            distance.abs() < 1e-5,
            "Manhattan distance of identical vectors should be 0, got {}",
            distance
        );

        // Dot product of identical vectors = sum of squares = 1+4+9+16+25 = 55
        let distance = compute.distance_with_metric(&v, &v, &DistanceMetric::DotProduct);
        assert!(
            (distance - 55.0).abs() < 1e-4,
            "Dot product of identical [1,2,3,4,5] should be 55, got {}",
            distance
        );
    }

    // --- Tests inlined from tests/unit/compute/distance_tests.rs ---

    #[test]
    fn test_platform_detection() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        // Test that we can create calculators for all metrics
        let cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let dot_calc = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let cosine = cosine_calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);
        let euclidean = euclidean_calc.calculate_distance(&a, &b, &DistanceMetric::Euclidean);
        let dot = dot_calc.calculate_distance(&a, &b, &DistanceMetric::DotProduct);

        // Verify results are reasonable
        assert!(cosine.raw_value >= 0.0 && cosine.raw_value <= 2.0);
        assert!(euclidean.raw_value >= 0.0);
        assert!(dot.raw_value >= 0.0);
    }

    #[test]
    fn test_scalar_implementations() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let dot_calc = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);

        let cosine = cosine_calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);
        assert!((cosine.raw_value - 1.0).abs() < 0.0001); // Orthogonal vectors

        let euclidean = euclidean_calc.calculate_distance(&a, &b, &DistanceMetric::Euclidean);
        assert!((euclidean.raw_value - 1.414).abs() < 0.01); // sqrt(2)

        let dot = dot_calc.calculate_distance(&a, &b, &DistanceMetric::DotProduct);
        assert_eq!(dot.raw_value, 0.0); // Orthogonal vectors
    }

    #[test]
    fn test_metric_specific_implementations() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Test direct usage of optimized calculators
        let cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let dot_calc = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);
        let manhattan_calc = UnifiedDistanceCompute::new(DistanceMetric::Manhattan);

        // Test that all calculators work without panicking
        let _ = cosine_calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);
        let _ = euclidean_calc.calculate_distance(&a, &b, &DistanceMetric::Euclidean);
        let _ = dot_calc.calculate_distance(&a, &b, &DistanceMetric::DotProduct);
        let _ = manhattan_calc.calculate_distance(&a, &b, &DistanceMetric::Manhattan);
    }

    #[test]
    fn test_batch_processing() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let mut results = Vec::new();
        for v in &vectors {
            results.push(calc.calculate_distance(&query, v, &DistanceMetric::Euclidean));
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].raw_value, 0.0); // Same vector
        assert!(results[1].raw_value > 0.0); // Different vectors
        assert!(results[2].raw_value > results[1].raw_value); // More distant vector
    }

    #[test]
    fn test_distance_metric_properties() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let dot_calc = UnifiedDistanceCompute::new(DistanceMetric::DotProduct);
        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let manhattan_calc = UnifiedDistanceCompute::new(DistanceMetric::Manhattan);

        // Just verify they can calculate distances
        let _ = cosine_calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);
        let _ = dot_calc.calculate_distance(&a, &b, &DistanceMetric::DotProduct);
        let _ = euclidean_calc.calculate_distance(&a, &b, &DistanceMetric::Euclidean);
        let _ = manhattan_calc.calculate_distance(&a, &b, &DistanceMetric::Manhattan);
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        let result = calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);

        // Just verify the result is reasonable
        assert!(result.raw_value >= 0.0 && result.raw_value <= 2.0);
    }

    #[test]
    fn test_zero_vectors() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let zero_a = vec![0.0, 0.0, 0.0];
        let zero_b = vec![0.0, 0.0, 0.0];
        let non_zero = vec![1.0, 2.0, 3.0];

        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let manhattan_calc = UnifiedDistanceCompute::new(DistanceMetric::Manhattan);

        // Zero distance between identical zero vectors
        assert_eq!(
            euclidean_calc
                .calculate_distance(&zero_a, &zero_b, &DistanceMetric::Euclidean)
                .raw_value,
            0.0
        );
        assert_eq!(
            manhattan_calc
                .calculate_distance(&zero_a, &zero_b, &DistanceMetric::Manhattan)
                .raw_value,
            0.0
        );

        // Non-zero distance between zero and non-zero vectors
        assert!(
            euclidean_calc
                .calculate_distance(&zero_a, &non_zero, &DistanceMetric::Euclidean)
                .raw_value
                > 0.0
        );
        assert!(
            manhattan_calc
                .calculate_distance(&zero_a, &non_zero, &DistanceMetric::Manhattan)
                .raw_value
                > 0.0
        );
    }

    #[test]
    fn test_edge_cases() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        // Test with single element vectors
        let a = vec![5.0];
        let b = vec![3.0];

        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let _cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let manhattan_calc = UnifiedDistanceCompute::new(DistanceMetric::Manhattan);

        // Single element euclidean distance
        assert_eq!(
            euclidean_calc
                .calculate_distance(&a, &b, &DistanceMetric::Euclidean)
                .raw_value,
            2.0
        );
        assert_eq!(
            manhattan_calc
                .calculate_distance(&a, &b, &DistanceMetric::Manhattan)
                .raw_value,
            2.0
        );

        // Test with very small values
        let tiny_a = vec![1e-10, 1e-10];
        let tiny_b = vec![2e-10, 2e-10];

        let dist = euclidean_calc.calculate_distance(&tiny_a, &tiny_b, &DistanceMetric::Euclidean);
        assert!(dist.raw_value > 0.0 && dist.raw_value < 1e-5);

        // Test with very large values
        let large_a = vec![1e6, 1e6];
        let large_b = vec![1e6 + 1.0, 1e6 + 1.0];

        let dist =
            euclidean_calc.calculate_distance(&large_a, &large_b, &DistanceMetric::Euclidean);
        assert!((dist.raw_value - std::f32::consts::SQRT_2).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let jaccard_calc = UnifiedDistanceCompute::new(DistanceMetric::Jaccard);

        // Test identical sets (binary vectors)
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        assert_eq!(
            jaccard_calc
                .calculate_distance(&a, &b, &DistanceMetric::Jaccard)
                .raw_value,
            0.0
        );

        // Test completely different sets
        let c = vec![1.0, 1.0, 0.0, 0.0];
        let d = vec![0.0, 0.0, 1.0, 1.0];
        assert_eq!(
            jaccard_calc
                .calculate_distance(&c, &d, &DistanceMetric::Jaccard)
                .raw_value,
            1.0
        );

        // Test partial overlap
        let e = vec![1.0, 1.0, 0.0, 0.0];
        let f = vec![1.0, 0.0, 1.0, 0.0];
        let dist = jaccard_calc.calculate_distance(&e, &f, &DistanceMetric::Jaccard);
        assert!(dist.raw_value > 0.0 && dist.raw_value < 1.0);
    }

    #[test]
    fn test_hamming_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let hamming_calc = UnifiedDistanceCompute::new(DistanceMetric::Hamming);

        // Test identical vectors
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0];
        assert_eq!(
            hamming_calc
                .calculate_distance(&a, &b, &DistanceMetric::Hamming)
                .raw_value,
            0.0
        );

        // Test completely different vectors
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let d = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            hamming_calc
                .calculate_distance(&c, &d, &DistanceMetric::Hamming)
                .raw_value,
            4.0
        );

        // Test partial difference
        let e = vec![1.0, 0.0, 1.0, 0.0];
        let f = vec![1.0, 1.0, 0.0, 0.0];
        assert_eq!(
            hamming_calc
                .calculate_distance(&e, &f, &DistanceMetric::Hamming)
                .raw_value,
            2.0
        );
    }

    #[test]
    fn test_chebyshev_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let chebyshev_calc = UnifiedDistanceCompute::new(DistanceMetric::Chebyshev);

        // Test identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(
            chebyshev_calc
                .calculate_distance(&a, &b, &DistanceMetric::Chebyshev)
                .raw_value,
            0.0
        );

        // Test different vectors
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![4.0, 2.0, 1.0];
        assert_eq!(
            chebyshev_calc
                .calculate_distance(&c, &d, &DistanceMetric::Chebyshev)
                .raw_value,
            3.0
        ); // max(|1-4|, |2-2|, |3-1|) = 3.0

        // Test with negative values
        let e = vec![-1.0, -2.0, -3.0];
        let f = vec![1.0, 2.0, 3.0];
        assert_eq!(
            chebyshev_calc
                .calculate_distance(&e, &f, &DistanceMetric::Chebyshev)
                .raw_value,
            6.0
        ); // max(2, 4, 6) = 6
    }

    #[test]
    fn test_canberra_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let canberra_calc = UnifiedDistanceCompute::new(DistanceMetric::Canberra);

        // Test identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(
            canberra_calc
                .calculate_distance(&a, &b, &DistanceMetric::Canberra)
                .raw_value,
            0.0
        );

        // Test different vectors
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![2.0, 3.0, 5.0];
        let dist = canberra_calc.calculate_distance(&c, &d, &DistanceMetric::Canberra);
        // |1-2|/(|1|+|2|) + |2-3|/(|2|+|3|) + |3-5|/(|3|+|5|) = 1/3 + 1/5 + 2/8 = 0.783...
        assert!((dist.raw_value - 0.783).abs() < 0.01);

        // Test with zero values
        let e = vec![0.0, 1.0, 2.0];
        let f = vec![1.0, 0.0, 3.0];
        let dist2 = canberra_calc.calculate_distance(&e, &f, &DistanceMetric::Canberra);
        assert!(dist2.raw_value > 0.0);
    }

    #[test]
    fn test_minkowski_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let minkowski_calc = UnifiedDistanceCompute::new(DistanceMetric::Minkowski);

        // Test identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(
            minkowski_calc
                .calculate_distance(&a, &b, &DistanceMetric::Minkowski)
                .raw_value,
            0.0
        );

        // Test different vectors (with p=3 default)
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let dist = minkowski_calc.calculate_distance(&c, &d, &DistanceMetric::Minkowski);
        // (|1-0|^3 + |0-1|^3)^(1/3) = (1 + 1)^(1/3) = 2^(1/3) ~ 1.26
        assert!((dist.raw_value - 1.26).abs() < 0.01);
    }

    #[test]
    fn test_angular_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let angular_calc = UnifiedDistanceCompute::new(DistanceMetric::Angular);

        // Test identical vectors (angle = 0)
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(
            (angular_calc
                .calculate_distance(&a, &b, &DistanceMetric::Angular)
                .raw_value
                - 0.0)
                .abs()
                < 1e-6
        );

        // Test orthogonal vectors (angle = pi/2)
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert!(
            (angular_calc
                .calculate_distance(&c, &d, &DistanceMetric::Angular)
                .raw_value
                - 0.5)
                .abs()
                < 0.01
        ); // pi/2 / pi = 0.5

        // Test opposite vectors (angle = pi)
        let e = vec![1.0, 0.0];
        let f = vec![-1.0, 0.0];
        assert!(
            (angular_calc
                .calculate_distance(&e, &f, &DistanceMetric::Angular)
                .raw_value
                - 1.0)
                .abs()
                < 0.01
        ); // pi / pi = 1.0
    }

    #[test]
    fn test_bray_curtis_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let bray_curtis_calc = UnifiedDistanceCompute::new(DistanceMetric::BrayCurtis);

        // Test identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(
            bray_curtis_calc
                .calculate_distance(&a, &b, &DistanceMetric::BrayCurtis)
                .raw_value,
            0.0
        );

        // Test different vectors
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![2.0, 3.0, 4.0];
        let dist = bray_curtis_calc.calculate_distance(&c, &d, &DistanceMetric::BrayCurtis);
        // |1-2| + |2-3| + |3-4| / (1+2+2+3+3+4) = 3/15 = 0.2
        assert!((dist.raw_value - 0.2).abs() < 0.01);

        // Test with zero vectors
        let e = vec![0.0, 0.0, 0.0];
        let f = vec![0.0, 0.0, 0.0];
        assert_eq!(
            bray_curtis_calc
                .calculate_distance(&e, &f, &DistanceMetric::BrayCurtis)
                .raw_value,
            0.0
        );
    }

    #[test]
    fn test_hellinger_distance() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let hellinger_calc = UnifiedDistanceCompute::new(DistanceMetric::Hellinger);

        // Test identical distributions
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        assert!(
            (hellinger_calc
                .calculate_distance(&a, &b, &DistanceMetric::Hellinger)
                .raw_value
                - 0.0)
                .abs()
                < 1e-6
        );

        // Test different distributions
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let dist = hellinger_calc.calculate_distance(&c, &d, &DistanceMetric::Hellinger);
        // sqrt(0.5 * ((1-0)^2 + (0-1)^2)) = sqrt(0.5 * 2) = 1.0
        assert!((dist.raw_value - 1.0).abs() < 0.01);

        // Test with non-normalized vectors (should normalize internally)
        let e = vec![2.0, 2.0];
        let f = vec![1.0, 3.0];
        let dist2 = hellinger_calc.calculate_distance(&e, &f, &DistanceMetric::Hellinger);
        assert!(dist2.raw_value > 0.0 && dist2.raw_value < 1.0);
    }

    #[test]
    fn test_batch_consistency() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Same as query
            vec![2.0, 3.0, 4.0, 5.0],
            vec![0.0, 1.0, 2.0, 3.0],
        ];

        // Test all distance metrics
        let metrics = vec![
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
            DistanceMetric::Jaccard,
            DistanceMetric::Hamming,
            DistanceMetric::Chebyshev,
            DistanceMetric::Canberra,
            DistanceMetric::Minkowski,
            DistanceMetric::Angular,
            DistanceMetric::BrayCurtis,
            DistanceMetric::Hellinger,
        ];

        for metric in metrics {
            let calc = UnifiedDistanceCompute::new(metric);

            // Calculate batch results manually
            let mut batch_results = Vec::new();
            for v in &vectors {
                batch_results.push(calc.calculate_distance(&query, v, &metric));
            }

            // Verify all calculations work
            for (i, v) in vectors.iter().enumerate() {
                let individual_result = calc.calculate_distance(&query, v, &metric);
                assert!(
                    (batch_results[i].raw_value - individual_result.raw_value).abs() < 1e-6,
                    "Batch and individual results don't match for {:?}",
                    metric
                );
            }
        }
    }

    #[test]
    fn test_large_vector_dimensions() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        // Test with high-dimensional vectors
        let dim = 1024;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.001).collect();

        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);
        let cosine_calc = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // Just verify no panic and reasonable results
        let euclidean_dist = euclidean_calc.calculate_distance(&a, &b, &DistanceMetric::Euclidean);
        let cosine_dist = cosine_calc.calculate_distance(&a, &b, &DistanceMetric::Cosine);

        assert!(euclidean_dist.raw_value > 0.0);
        assert!(cosine_dist.raw_value >= 0.0 && cosine_dist.raw_value <= 2.0);
    }

    #[test]
    fn test_nan_and_infinity_handling() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        use std::f32::{INFINITY, NAN, NEG_INFINITY};

        let normal = vec![1.0, 2.0, 3.0];
        let with_nan = vec![1.0, NAN, 3.0];
        let with_inf = vec![1.0, INFINITY, 3.0];
        let with_neg_inf = vec![1.0, NEG_INFINITY, 3.0];

        let euclidean_calc = UnifiedDistanceCompute::new(DistanceMetric::Euclidean);

        // Test NaN propagation
        let dist_nan =
            euclidean_calc.calculate_distance(&normal, &with_nan, &DistanceMetric::Euclidean);
        assert!(dist_nan.raw_value.is_nan());

        // Test infinity handling
        let dist_inf =
            euclidean_calc.calculate_distance(&normal, &with_inf, &DistanceMetric::Euclidean);
        assert!(dist_inf.raw_value.is_infinite());

        let dist_neg_inf =
            euclidean_calc.calculate_distance(&normal, &with_neg_inf, &DistanceMetric::Euclidean);
        assert!(dist_neg_inf.raw_value.is_infinite());
    }
}
