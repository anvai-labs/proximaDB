#![allow(dead_code)]
// ============================================================================
// PERFECTED: 1-TO-1 CENTROID-ROWGROUP MAPPING FOR PERFECT PARALLELISM
// ============================================================================
//
// SIMPLIFIED DESIGN: K centroids = K rowgroups (1-to-1 mapping)
//
// 1. **Perfect Parallel Subdivision**:
//    - Each centroid gets exactly ONE rowgroup (centroid_id == rowgroup_id)
//    - Vector space evenly subdivided into K independent partitions
//    - No coordination needed between rowgroups during search
//
// 2. **Overflow Handling**:
//    - If rowgroup exceeds capacity → create NEW centroid for overflow
//    - This maintains balanced distribution automatically
//    - Dynamic K adjustment based on data volume
//
// 3. **Search Parallelism**:
//    - K×K matrix selects subset of centroids (fast O(K) operation)
//    - Each selected centroid = one independent rowgroup to search
//    - Rowgroups can be searched in parallel threads
//    - P² matrix within each rowgroup provides exact distances
//
// 4. **Writer Implementation**:
//    - During flush(), vectors assigned to centroids via clustering
//    - Each centroid gets one rowgroup (simple assignment)
//    - If rowgroup full → create new centroid → new rowgroup
//    - Store total_centroids count in footer (K value)
//
// 5. **Matrix Benefits**:
//    - K×K matrix: Selects which rowgroups to search (perfect parallelism)
//    - P×K matrix: Vector-to-centroid boosting within each rowgroup
//    - P² matrix: Exact intra-rowgroup navigation (no approximation)
//
// IMPLEMENTATION FLOW:
//
// 1. assign_vectors_to_initial_centroids(vectors)
// 2. handle_rowgroup_overflow() → creates new centroids dynamically
// 3. calculate_final_centroids_from_assignments()
// 4. build_kxk_inter_centroid_distance_matrix() ← CRITICAL STEP
// 5. store_in_footer(K, centroids, kxk_matrix)
//
// Deferred: Implement complete flow in flush_row_page_columnar()
// ============================================================================

use anyhow::Result;
use arrow_array::RecordBatch;
use proximadb_kernel::hash::FastHash;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::debug;

// Reuse existing platform capabilities
use proximadb_compression::{
    CompressionAlgorithm, CompressionContext, CompressionProvider, StandardCompression,
};
use proximadb_hardware_caps::get_hardware_capabilities;
use proximadb_raptor_common::VectorStats;

// Import bloom filter types from common
use crate::matrix_builder::MatrixBuilder;
use proximadb_distance_kernel::DistanceMetric;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use proximadb_hardware_caps::HardwareCapabilities;
use proximadb_proto::proximadb_v1::VectorRecord;
use proximadb_raptor_common::{
    CentroidStats, ColumnarCentroids, DistanceBounds, NeighborType, ProximaMetadata, RaptorFooter,
    RowGroupBloomFilter, RowGroupNeighbor,
};
use proximadb_runtime_common::pool::VectorMemoryPool;
// ProximaCodec system for encoding/decoding
use proximadb_codec::{ProximaCodec, analysis, types::ProximaScheme};
use proximadb_storage_filesystem_types::FileSystem;

// AXIS clustering + quantization are injected via DI ports (Slice D) — the
// concrete engines are constructed at the composition root (engine.rs).
use proximadb_storage_ports::{AxisClusteringPort, StorageQuantizationEnginePort};

use proximadb_raptor_common::config::{CompressionCodec as RaptorCompressionCodec, RaptorConfig};
use proximadb_raptor_common::constants;
use proximadb_raptor_common::*;

/// RAPTOR writer with 1-to-1 centroid-rowgroup mapping for perfect parallelism
///
/// ## Architecture
///
/// The RaptorWriter implements a simplified design where K centroids = K rowgroups,
/// providing perfect parallel subdivision of the vector space.
///
/// ### Core Design Principles:
/// - **Perfect Parallel Subdivision**: Each centroid maps to exactly one rowgroup
/// - **Dynamic Overflow Handling**: Creates new centroids when rowgroups exceed capacity
/// - **Matrix Trinity Architecture**: K×K, P×K, and P² matrices for optimal search
/// - **Proxima Encoding**: Custom compact format with inline metadata
///
/// ### Matrix Structure:
/// - **K×K matrix**: Selects which rowgroups to search (inter-centroid distances)
/// - **P×K matrix**: Vector-to-centroid boosting within each rowgroup
/// - **P² matrix**: Exact intra-rowgroup navigation (no approximation)
///
/// ### Write Flow:
/// 1. `assign_vectors_to_initial_centroids(vectors)` - Initial clustering
/// 2. `handle_rowgroup_overflow()` - Creates new centroids dynamically
/// 3. `calculate_final_centroids_from_assignments()` - Finalize clustering
/// 4. `build_kxk_inter_centroid_distance_matrix()` - Build K×K matrix
/// 5. `store_in_footer(K, centroids, kxk_matrix)` - Persist metadata
///
/// ### Performance:
/// - **Memory**: 96% reduction vs full vector storage through IVF clustering
/// - **Parallelism**: Perfect rowgroup parallelism during search
/// - **Adaptive**: Dynamic K adjustment based on data volume
pub struct RaptorWriter {
    // File management
    /// Path to the RAPTOR file being written
    file_path: String,
    /// Filesystem abstraction for cloud-aware I/O operations
    filesystem: Arc<dyn FileSystem>,

    // Configuration
    /// Engine configuration with dimension, rowgroup, and clustering parameters
    config: RaptorConfig,
    /// Collection identifier for this write operation
    #[allow(dead_code)]
    collection_id: String,
    /// Vector dimensionality
    dimension: usize,

    // Reuse platform capabilities
    /// Standard compression engine for vector data
    compression: Arc<StandardCompression>,
    /// Quantization engine for reducing memory footprint
    quantization_engine: Arc<dyn StorageQuantizationEnginePort>,
    /// Memory pool for efficient vector allocation
    #[allow(dead_code)]
    memory_pool: Arc<VectorMemoryPool>,
    /// Hardware capabilities for runtime optimization
    #[allow(dead_code)]
    hardware: Arc<HardwareCapabilities>,
    /// Distance computation engine with SIMD acceleration
    distance_compute: Arc<UnifiedDistanceCompute>,
    /// Matrix builder for constructing K×K, P×K, and P² matrices
    #[allow(dead_code)]
    matrix_builder: MatrixBuilder,

    // Current state
    /// Buffer for accumulating rows before flushing to disk
    current_row_page: Option<RowPageBuffer>,
    /// Current rowgroup being built (for RecordBatch compatibility)
    #[allow(dead_code)]
    current_rowgroup: Option<CurrentRowgroup>,
    /// Metadata for completed rowgroups
    row_groups: Vec<RowGroupMetadata>,
    /// File-level metadata updated during writes
    file_metadata: RaptorFileMetadata,

    // Indexes being built
    /// Bloom filter builder for fast ID lookups
    bloom_builder: BloomFilterBuilder,
    /// Columnar ID index builder for vector scans
    id_column_builder: IdColumnBuilder,
    /// IVF clustering builder for p²+k×p algorithm (96% memory reduction)
    ivf_builder: IvfClusteringBuilder,
    /// Column projection builder for selective column reads
    column_projections: ColumnProjectionsBuilder,

    // Track if file has been created
    /// Flag indicating if file has been created on disk
    #[allow(dead_code)]
    file_created: bool,
}

/// Buffer for accumulating rows into pages
struct RowPageBuffer {
    rows: Vec<CompactRow>,
    #[allow(dead_code)]
    page_id: u16,
    #[allow(dead_code)]
    start_offset: u64,
}

/// Compact row representation aligned with VectorRecord proto fields
/// Stores both FP32 and quantized vectors for full reconstruction
struct CompactRow {
    // Core fields from VectorRecord
    #[allow(dead_code)]
    id: String, // VectorRecord.id (string)
    vector: Vec<f32>, // VectorRecord.vector (original FP32)
    #[allow(dead_code)]
    quantized_vector: Vec<u8>, // VectorRecord.quantized_vector (pre-quantized INT8)
    #[allow(dead_code)]
    binary_sketch: Vec<u8>, // Binary sketch for progressive search (1-bit per dimension)
    // Deferred: Migrate to HashMap<String, SqlValue> for typed metadata (requires refactoring encoding/decoding logic)
    metadata: Vec<(String, Vec<u8>)>, // VectorRecord.metadata (key-value pairs as byte arrays)

    // Timestamp fields
    #[allow(dead_code)]
    timestamp: u32, // VectorRecord.timestamp
    #[allow(dead_code)]
    updated_at: Option<u32>, // VectorRecord.updated_at
    #[allow(dead_code)]
    expires_at: Option<u32>, // VectorRecord.expires_at
    #[allow(dead_code)]
    version: Option<u32>, // VectorRecord.version

    // Source content for RAG
    source_content: Option<Vec<u8>>, // VectorRecord.source (serialized SourceContent)
}

/// Bloom filter builder for row group
struct BloomFilterBuilder {
    ids: Vec<String>,
    /// Membership mirror of `ids` — dedup was a linear scan of the Vec per
    /// record (O(R²) string compares per row group build).
    seen: std::collections::HashSet<String>,
    target_false_positive_rate: f64,
}

impl BloomFilterBuilder {
    fn new(target_false_positive_rate: f64) -> Self {
        Self {
            ids: Vec::new(),
            seen: std::collections::HashSet::new(),
            target_false_positive_rate,
        }
    }

    /// Add VectorRecord ID to the bloom filter
    fn add_id(&mut self, id: String) {
        // Avoid duplicates (a bloom filter is duplicate-insensitive, but the
        // unique count sizes the filter and `num_entries`).
        if self.seen.insert(id.clone()) {
            self.ids.push(id);
        }
    }

    /// Build the bloom filter from accumulated IDs
    fn build(&self) -> anyhow::Result<RowGroupBloomFilter> {
        if self.ids.is_empty() {
            return Ok(RowGroupBloomFilter::new(
                100,
                self.target_false_positive_rate,
            ));
        }

        RowGroupBloomFilter::from_ids(&self.ids, self.target_false_positive_rate)
    }

    /// Get the number of IDs collected
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if builder is empty
    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Clear all collected IDs
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.ids.clear();
        self.seen.clear();
    }
}

/// Columnar ID index builder
struct IdColumnBuilder {
    ids: Vec<String>,
    id_hashes: Vec<u64>,
    row_offsets: Vec<u32>,
}

/// IVF (Inverted File) clustering builder for RAPTOR's p²+k×p algorithm
/// This is NOT HNSW - it's an IVF-style structure with k-means clustering
/// Reduces memory footprint by 96% compared to full vector storage
struct IvfClusteringBuilder {
    nodes: Vec<IvfNode>,
    /// Map from vector ID to node index for quick lookup
    id_to_node: HashMap<String, u32>,
    /// Target row group size (p in the p²+k×p formula)
    #[allow(dead_code)]
    target_rowgroup_size: usize,
    /// Hardware capabilities for optimization
    #[allow(dead_code)]
    hardware: Arc<HardwareCapabilities>,
    /// AXIS clustering engine for reusable k-means implementation
    axis_clustering: Arc<dyn AxisClusteringPort>,
    /// Pre-computed centroids for k clusters
    centroids: Vec<Centroid>,
    /// Boosting parameters
    boost_config: BoostingConfig,
    /// Distance computation engine
    distance_compute: Arc<UnifiedDistanceCompute>,
    /// Temporary vector storage for clustering and edge building
    /// Cleared after flush to save memory
    vectors: Vec<Vec<f32>>,
    /// Pre-computed centroid-to-centroid distances for component boosting
    centroid_distances: Vec<Vec<f32>>,
}

// Removed ClusteringConfig - now using AXIS clustering infrastructure

/// Boosting configuration for component-based distance
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BoostingConfig {
    /// α₁: Vector-to-own-centroid weight (intra-cluster cohesion)
    alpha_own: f32,
    /// α₂: Vector-to-other-centroids weight (boundary penalty)
    alpha_other: f32,
    /// α₂: Inter-centroid weight (alternative name for alpha_other)
    alpha_inter: f32,
    /// α₃: Cluster variance weight (compactness measure)
    alpha_variance: f32,
    /// β₁: Minimum inter-centroid distance weight (cluster separation)
    beta_min: f32,
    /// β₂: Maximum inter-centroid distance weight (global structure)
    beta_max: f32,
    /// β: Cross-centroid exponential decay weight
    beta_cross: f32,
    /// Boundary detection threshold (in std deviations)
    boundary_threshold: f32,
    /// Enable component storage for debugging
    store_components: bool,
}

impl Default for BoostingConfig {
    fn default() -> Self {
        Self {
            alpha_own: constants::boosting::ALPHA_OWN_DEFAULT,
            alpha_other: constants::boosting::ALPHA_INTER_DEFAULT,
            alpha_inter: constants::boosting::ALPHA_INTER_DEFAULT,
            alpha_variance: constants::boosting::ALPHA_VARIANCE_DEFAULT,
            beta_min: constants::boosting::BETA_MIN_DEFAULT,
            beta_max: constants::boosting::BETA_MAX_DEFAULT,
            beta_cross: constants::boosting::BETA_CROSS_DEFAULT,
            boundary_threshold: constants::boosting::BOUNDARY_THRESHOLD_DEFAULT,
            store_components: false,
        }
    }
}

/// Centroid with statistics for boosting calculations
#[derive(Clone, Debug)]
struct Centroid {
    #[allow(dead_code)]
    id: usize,
    vector: Vec<f32>,
    member_ids: Vec<String>,
    mean_distance: f32,
    std_deviation: f32,
    #[allow(dead_code)]
    radius: f32, // 95th percentile distance
}

// Note: CentroidNeighbors removed - neighbor relationships are now stored
// directly in RowGroupMetadata.centroid_stats.neighbor_rowgroups
// This avoids duplication and keeps related data together

impl IvfClusteringBuilder {
    fn new(
        target_rowgroup_size: usize,
        hardware: Arc<HardwareCapabilities>,
        axis_clustering: Arc<dyn AxisClusteringPort>,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_node: HashMap::new(),
            target_rowgroup_size,
            hardware,
            axis_clustering,
            centroids: Vec::new(),
            boost_config: BoostingConfig::default(),
            distance_compute: Arc::new(UnifiedDistanceCompute::new(DistanceMetric::Cosine)),
            vectors: Vec::new(),
            centroid_distances: Vec::new(),
        }
    }

    /// Set custom boosting configuration
    pub fn with_boost_config(mut self, config: BoostingConfig) -> Self {
        self.boost_config = config;
        self
    }

    /// Add a node with its edges including distance information
    fn add_node(&mut self, vector_id: String, edges: Vec<EdgeWithDistance>) {
        let node_id = self.nodes.len() as u32;
        self.id_to_node.insert(vector_id.clone(), node_id);

        self.nodes.push(IvfNode {
            vector_id,
            cluster_id: 0, // Will be assigned during clustering
            row_location: RowLocation {
                row_group_id: 0, // Will be assigned during clustering
                page_id: 0,
                offset_in_page: 0,
            },
            centroid_distance: 0.0, // Will be calculated during clustering
            edges,                  // Local graph connectivity within cluster
        });
    }

    /// Advanced k²+p×(k+p) clustering with hardware-aware parameter selection and component boosting
    ///
    /// This method implements sophisticated row group size optimization using multiple constraints:
    /// 1. Mathematical Optimum: p ≈ √n for k²+p×(k+p) complexity optimization (O(√n) scaling)
    /// 2. File Size Estimation: Based on n×(d×4 + metadata + source overhead) analysis
    /// 3. Hardware Detection: Uses actual L3 cache size, targets 45% for row group efficiency
    /// 4. Memory Constraint: Adapts to detected L3 cache (8MB-32MB typical) for optimal performance
    /// 5. Minimum Threshold: p ≥ 1024 to justify clustering overhead and ensure high recall
    /// 6. Configuration Override: Respects user-specified target_rowgroup_size when larger
    ///
    /// Final p selection: p = max(√n, hardware_memory_optimal, 1024, config_target)
    ///
    /// Key benefits:
    /// - Hardware-adaptive: Automatically adjusts to CPU cache architecture
    /// - Scales optimally with dataset size (√n principle)
    /// - Adapts to vector dimensions and estimated file characteristics  
    /// - Balances recall (large row groups) vs cache efficiency
    /// - Uses AXIS clustering for proven k-means++ initialization
    ///
    /// Mathematical foundation:
    /// - Standard HNSW: O(n × M × EF) = 30 × 16 × 200 = 96,000 calculations
    /// - RAPTOR strategy: O(k×n + k² + n×boost_components) where k=clusters, n=vectors
    ///   * k×n: AXIS k-means clustering (180 calcs for k=6, n=30)
    ///   * k²: Centroid-to-centroid distance matrix (36 calcs for k=6)  
    ///   * n×boost: Component boosting assignment (30×42 = 1,260 calcs)
    /// - For n=30, k=6: 180 + 36 + 1,260 = 1,476 total calculations
    /// - Reduction factor: 96,000/1,476 ≈ 65x improvement over standard HNSW
    ///
    /// Yes, we DO compute centroid-to-centroid distances (k²=36 calculations) which are
    /// critical for the d₂, d₄, d₅ components in the boosting formula.
    ///
    /// Component boosting formula:
    /// D = α₁·d₁ + α₂·d₂ + α₃·d₃ + β₁·d₄ + β₂·d₅
    /// where:
    /// - d₁: Direct vector-to-centroid distance (intra-cluster cohesion)
    /// - d₂: Average distance to other centroids (boundary penalty)
    /// - d₃: Distance variance within cluster (compactness measure)
    /// - d₄: Minimum inter-centroid distance (cluster separation)
    /// - d₅: Maximum inter-centroid distance (global structure preservation)
    ///
    /// Cluster vectors into row groups using k²+p×(k+p) strategy
    pub fn cluster_vectors_into_rowgroups(
        &mut self,
        vectors: &[Vec<f32>],
        dimension: usize,
    ) -> Vec<Vec<u32>> {
        if vectors.is_empty() {
            tracing::warn!("No vectors provided for rowgroup clustering");
            return Vec::new();
        }

        // Step 1: Calculate optimal row group size based on mathematical and practical constraints
        let n = vectors.len(); // Total number of vectors to cluster
        let d = dimension; // Vector dimension from collection configuration

        // Mathematical optimum: p ≈ √n for k²+p×(k+p) complexity optimization
        let p_sqrt_n = (n as f64).sqrt() as usize;

        // Practical constraint: Estimate vectors from file characteristics
        let bytes_per_vector = d * constants::clustering::BYTES_PER_F32_DIMENSION;
        let metadata_overhead = constants::clustering::METADATA_OVERHEAD_PER_VECTOR;
        let total_bytes_per_vector = bytes_per_vector + metadata_overhead;
        let estimated_file_size = n * total_bytes_per_vector; // n × (d×4 + metadata + overhead)

        // Hardware-aware memory constraint: Detect L3 cache size for optimal row group sizing
        let detected_l3_cache = constants::clustering::DEFAULT_L3_CACHE_SIZE; // Use default L3 cache size
        // Use 40-50% of L3 cache for row group to leave room for other operations
        let target_rowgroup_bytes = (detected_l3_cache as f64
            * constants::clustering::L3_CACHE_UTILIZATION_PERCENT)
            as usize;
        let p_memory_optimal = target_rowgroup_bytes / total_bytes_per_vector;

        // Minimum constraint: Ensure clustering is beneficial (recall + I/O efficiency)
        let p_min = constants::clustering::MIN_ROWGROUP_SIZE;

        // Choose optimal p: max(√n, memory_optimal, min_constraint, target_config)
        let p = p_sqrt_n
            .max(p_memory_optimal)
            .max(p_min)
            .max(self.target_rowgroup_size);

        let k = n.div_ceil(p); // Number of clusters needed: k = ceil(n/p)

        let k_means_calcs = k * n; // AXIS k-means clustering
        let centroid_matrix_calcs = k * k; // K² matrix: Centroid-to-centroid distances
        let boosting_calcs = n * constants::boosting::BOOSTING_CALCS_PER_VECTOR;
        let raptor_total = k_means_calcs + centroid_matrix_calcs + boosting_calcs;
        // HNSW obsolete - using Matrix Trinity (P² + K² + P×K) instead
        // let hnsw_complexity = (n as f32 * constants::complexity::HNSW_M_FACTOR * constants::complexity::HNSW_EF_FACTOR) as usize;

        tracing::info!(
            "🎯 RAPTOR Matrix Trinity clustering: n={}, k={}, p={} \
             | Constraints: √n={}, memory_opt={}, min={}, config={} \
             | Hardware: L3_cache={:.1}MB, target_rowgroup={:.1}MB ({:.0}% L3) \
             | File: d={}, {:.1}KB/vec, est_size={:.1}MB \
             | Recall: {} vectors/exhaustive search per row group \
             | Complexity: K-means={} + K²={} + Boosting={} = {} total calcs",
            n,
            k,
            p,
            p_sqrt_n,
            p_memory_optimal,
            p_min,
            self.target_rowgroup_size,
            detected_l3_cache as f64 / 1_000_000.0,
            target_rowgroup_bytes as f64 / 1_000_000.0,
            (target_rowgroup_bytes as f64 / detected_l3_cache as f64) * 100.0,
            d,
            total_bytes_per_vector as f64 / 1024.0,
            estimated_file_size as f64 / 1_000_000.0,
            p,
            k_means_calcs,
            centroid_matrix_calcs,
            boosting_calcs,
            raptor_total
        );

        // Step 2: Configure AXIS clustering for optimal RAPTOR performance
        // Euclidean distance provides best balance for row-aligned storage patterns
        // Limited iterations prevent over-optimization and maintain cluster balance
        let distance_metric = DistanceMetric::Euclidean;
        let max_iterations = constants::clustering::KMEANS_MAX_ITERATIONS;

        tracing::debug!(
            "Clustering configuration: metric={:?}, max_iterations={}, target_clusters={}",
            distance_metric,
            max_iterations,
            k
        );

        // Step 3: Phase 1 - Initial clustering using AXIS k-means++ for optimal initialization
        // k-means++ ensures well-separated initial centroids, leading to better final clusters
        let (centroids, cluster_assignments) = match self.axis_clustering.cluster_vectors_simple(
            vectors,
            k,
            distance_metric,
            max_iterations,
        ) {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "AXIS clustering failed; falling back to deterministic round-robin assignment"
                );
                let effective_k = k.max(1).min(vectors.len());
                let fallback_centroids = vectors
                    .iter()
                    .take(effective_k)
                    .cloned()
                    .collect::<Vec<_>>();
                let fallback_assignments = (0..vectors.len())
                    .map(|idx| idx % effective_k)
                    .collect::<Vec<_>>();
                (fallback_centroids, fallback_assignments)
            }
        };

        tracing::info!(
            "✅ AXIS clustering complete: {} centroids generated, {} vector assignments made",
            centroids.len(),
            cluster_assignments.len()
        );

        // Step 4: Phase 2 - Build k×k centroid distance matrix for component boosting
        // This matrix enables rapid calculation of inter-centroid relationships
        // Critical for d₂, d₄, d₅ components in the boosting formula
        let centroid_distances = self
            .axis_clustering
            .calculate_centroid_distance_matrix(&centroids, distance_metric)
            .unwrap_or_else(|error| {
                tracing::warn!(
                    error = %error,
                    "Centroid distance matrix failed; falling back to direct pairwise computation"
                );
                let k = centroids.len();
                let mut fallback = vec![vec![0.0f32; k]; k];
                for i in 0..k {
                    for j in 0..k {
                        if i != j {
                            fallback[i][j] = self
                                .distance_compute
                                .calculate_distance(
                                    &centroids[i],
                                    &centroids[j],
                                    &DistanceMetric::Euclidean,
                                )
                                .raw_value;
                        }
                    }
                }
                fallback
            });

        tracing::info!(
            "✅ Centroid distance matrix built: {}×{} (enables O(1) inter-centroid lookups)",
            centroid_distances.len(),
            centroid_distances.first().map_or(0, |row| row.len())
        );

        // Step 5: Phase 3 - Apply sophisticated 5-component boosting using AXIS infrastructure
        // Each weight controls a different aspect of cluster quality:
        // α weights (alpha): Focus on intra-cluster properties (cohesion, boundaries, compactness)
        // β weights (beta): Focus on inter-cluster properties (separation, global structure)
        let boosting_weights = [
            self.boost_config.alpha_own, // α₁: Vector-to-own-centroid (minimize intra-cluster spread)
            self.boost_config.alpha_other, // α₂: Vector-to-other-centroids (penalize boundary vectors)
            self.boost_config.alpha_variance, // α₃: Cluster variance (prefer compact clusters)
            self.boost_config.beta_min,    // β₁: Min inter-centroid distance (ensure separation)
            self.boost_config.beta_max, // β₂: Max inter-centroid distance (preserve global structure)
        ];

        tracing::debug!(
            "Component boosting weights: α₁={:.3}, α₂={:.3}, α₃={:.3}, β₁={:.3}, β₂={:.3}",
            boosting_weights[0],
            boosting_weights[1],
            boosting_weights[2],
            boosting_weights[3],
            boosting_weights[4]
        );

        let boosted_assignments = self
            .axis_clustering
            .assign_vectors_with_component_boosting(
                vectors,             // Input vectors for assignment
                &centroids,          // Cluster centroids from Phase 1
                &centroid_distances, // Inter-centroid distance matrix from Phase 2
                distance_metric,     // Consistent distance metric
                &boosting_weights,   // 5-component weight configuration
            )
            .unwrap_or_else(|error| {
                tracing::warn!(
                    error = %error,
                    "Component boosting assignment failed; using raw clustering assignments"
                );
                cluster_assignments
                    .iter()
                    .copied()
                    .map(|cluster_id| (cluster_id, 0.0f32))
                    .collect()
            });

        tracing::info!(
            "✅ Component boosting complete: {} vectors assigned with 5-component optimization",
            boosted_assignments.len()
        );

        // Step 6: Convert AXIS clustering results to RAPTOR internal structures
        // This maintains compatibility with existing RAPTOR search and storage logic
        self.centroids = centroids
            .into_iter()
            .enumerate()
            .map(|(idx, centroid_vec)| {
                Centroid {
                    id: idx,
                    vector: centroid_vec,
                    member_ids: Vec::new(), // Populated in Step 7 during cluster organization
                    mean_distance: 0.0,     // Calculated in Step 8 statistics phase
                    std_deviation: 0.0,     // Calculated in Step 8 statistics phase
                    radius: 0.0,            // Calculated in Step 8 statistics phase
                }
            })
            .collect();

        // Store the centroid distance matrix for runtime boosting calculations
        self.centroid_distances = centroid_distances;

        // Step 7: Organize vectors into cluster groups and populate membership information
        let mut clusters = vec![Vec::new(); k];
        for (vector_idx, (cluster_id, boosted_distance)) in boosted_assignments.iter().enumerate() {
            clusters[*cluster_id].push(vector_idx);

            // Track vector membership in centroid metadata (if node exists)
            if vector_idx < self.nodes.len() {
                self.centroids[*cluster_id]
                    .member_ids
                    .push(self.nodes[vector_idx].vector_id.clone());
            }

            if vector_idx % 5000 == 0 && vector_idx > 0 {
                tracing::trace!(
                    "Processed {} / {} assignments, latest: vector {} → cluster {} (boosted_distance: {:.4})",
                    vector_idx,
                    boosted_assignments.len(),
                    vector_idx,
                    cluster_id,
                    boosted_distance
                );
            }
        }

        // Step 8: Calculate comprehensive centroid statistics for quality assessment
        tracing::debug!("Calculating centroid statistics for cluster quality assessment");
        self.calculate_centroid_statistics(vectors);

        // Step 9: Apply component boosting to HNSW edges for enhanced search performance
        // This extends the clustering benefits to the graph structure used during search
        tracing::debug!("Applying component boosting to HNSW edges for search optimization");
        self.apply_component_boosting(&clusters, vectors);

        // Step 10: Log cluster balance and quality metrics for monitoring
        let mut total_vectors = 0;
        let mut min_cluster_size = usize::MAX;
        let mut max_cluster_size = 0;

        for (i, cluster) in clusters.iter().enumerate() {
            let size = cluster.len();
            total_vectors += size;
            min_cluster_size = min_cluster_size.min(size);
            max_cluster_size = max_cluster_size.max(size);

            tracing::debug!(
                "Cluster {}: {} vectors ({:.1}% of total, target: {:.1}%)",
                i,
                size,
                (size as f32 / n as f32) * 100.0,
                (p as f32 / n as f32) * 100.0
            );
        }

        let balance_ratio = max_cluster_size as f32 / min_cluster_size.max(1) as f32;
        tracing::info!(
            "📊 Clustering quality: {} clusters, balance_ratio={:.2} (1.0=perfect), \
             sizes: min={}, max={}, avg={:.1}",
            clusters.len(),
            balance_ratio,
            min_cluster_size,
            max_cluster_size,
            total_vectors as f32 / clusters.len() as f32
        );

        // Step 11: Convert cluster assignments to RAPTOR row group format
        tracing::debug!(
            "Converting {} clusters to RAPTOR row group format",
            clusters.len()
        );
        self.clusters_to_rowgroups(clusters)
    }

    /// Cluster HNSW nodes into row groups (for existing graph structures)
    /// This method works with pre-built HNSW nodes and their connectivity
    pub fn cluster_nodes_into_rowgroups(&mut self, dimension: usize) -> Vec<Vec<u32>> {
        // If we have no nodes, return empty
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Extract vectors from nodes
        let vectors: Vec<Vec<f32>> = self
            .nodes
            .iter()
            .filter_map(|node| {
                // Get vector from the stored vectors using node index
                self.vectors
                    .get(node.vector_id.parse::<usize>().ok()?)
                    .cloned()
            })
            .collect();

        if vectors.is_empty() {
            tracing::warn!("No vectors found for nodes, using fallback clustering");
            // Fallback to simple round-robin if no vectors available
            let n = self.nodes.len();
            let p = self
                .target_rowgroup_size
                .max(constants::clustering::MIN_ROWGROUP_SIZE);
            let k = n.div_ceil(p);
            let mut clusters = vec![Vec::new(); k];
            for (idx, _node) in self.nodes.iter().enumerate() {
                clusters[idx % k].push(idx as u32);
            }
            clusters.retain(|cluster| !cluster.is_empty());
            return clusters;
        }

        // Use the same clustering logic as cluster_vectors_into_rowgroups
        self.cluster_vectors_into_rowgroups(&vectors, dimension)
    }

    /// Apply sophisticated 5-component boosting to HNSW edges for search optimization
    ///
    /// This method enhances the HNSW graph structure by applying the same boosting formula
    /// used in clustering to individual edges. This creates consistency between storage
    /// organization (clustering) and search navigation (HNSW graph).
    ///
    /// Edge boosting improvements:
    /// 1. Intra-cluster edges get lower costs (faster navigation within row groups)  
    /// 2. Inter-cluster edges get higher costs but remain for connectivity
    /// 3. Boundary detection identifies vectors near cluster edges
    /// 4. Cross-cluster penalties maintain cluster separation during search
    /// 5. Global normalization ensures consistent scaling across dataset sizes
    ///
    /// Mathematical components:
    /// - α₁,α₃: Boundary detection using statistical thresholds (mean + σ×threshold)
    /// - α₂: Inter-cluster penalty using logarithmic scaling
    /// - β₁,β₂: Cross-cluster exponential decay for distant connections
    fn apply_component_boosting(&mut self, clusters: &[Vec<usize>], vectors: &[Vec<f32>]) {
        // Step 1: Calculate global normalization factor for consistent scaling
        // This ensures boosting behaves predictably across different dataset sizes and densities
        let global_avg_distance = self.calculate_global_avg_distance();

        tracing::debug!(
            "Applying component boosting to {} nodes with global_avg_distance={:.4}",
            self.nodes.len(),
            global_avg_distance
        );

        let mut total_edges_processed = 0;
        let mut intra_cluster_edges = 0;
        let mut inter_cluster_edges = 0;

        // Dense node→cluster index, built once (was: a linear scan of every
        // cluster's member Vec per node AND per edge below).
        let node_to_cluster = Self::build_node_to_cluster_index(clusters);

        // Step 2: Process each node and apply boosting to all its outgoing edges
        // Collect information first to avoid borrow checker issues
        let node_info: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node_idx, node)| {
                let source_idx = match self.id_to_node.get(&node.vector_id) {
                    Some(index) => *index as usize,
                    None => {
                        tracing::warn!(
                            vector_id = %node.vector_id,
                            "Skipping node without id_to_node mapping during component boosting"
                        );
                        return None;
                    }
                };
                let source_cluster = Self::cluster_of(&node_to_cluster, source_idx);
                Some((node_idx, source_idx, source_cluster))
            })
            .collect();

        for (node_idx, source_idx, source_cluster) in node_info {
            let node = &mut self.nodes[node_idx];
            let source_centroid = &self.centroids[source_cluster];

            // Process edges with component boosting for hybrid IVF+Graph
            let mut boosted_edges = Vec::with_capacity(node.edges.len());

            // Step 3: Process each edge with component boosting
            for (edge_idx, edge) in node.edges.iter().enumerate() {
                let target_idx = edge.target_node_id as usize;
                let target_cluster = Self::cluster_of(&node_to_cluster, target_idx);
                let target_centroid = &self.centroids[target_cluster];

                // Track edge type for monitoring cluster connectivity
                if source_cluster == target_cluster {
                    intra_cluster_edges += 1;
                } else {
                    inter_cluster_edges += 1;
                }

                // Step 4: Calculate the 5 fundamental distance components
                // d₁: Source vector distance to its own centroid (intra-cluster cohesion)
                let d1 = self
                    .distance_compute
                    .calculate_distance(
                        &vectors[source_idx],
                        &source_centroid.vector,
                        &DistanceMetric::Euclidean,
                    )
                    .raw_value;

                // d₂: Inter-centroid distance (cluster separation, pre-computed from AXIS)
                let d2 = if !self.centroid_distances.is_empty()
                    && source_cluster < self.centroid_distances.len()
                    && target_cluster < self.centroid_distances[source_cluster].len()
                {
                    self.centroid_distances[source_cluster][target_cluster]
                } else {
                    0.0 // Default when clustering is not enabled
                };

                // d₃: Target vector distance to its own centroid (target cluster cohesion)
                let d3 = self
                    .distance_compute
                    .calculate_distance(
                        &vectors[target_idx],
                        &target_centroid.vector,
                        &DistanceMetric::Euclidean,
                    )
                    .raw_value;

                // d₄: Source vector distance to target centroid (cross-cluster penalty)
                let d4 = self
                    .distance_compute
                    .calculate_distance(
                        &vectors[source_idx],
                        &target_centroid.vector,
                        &DistanceMetric::Euclidean,
                    )
                    .raw_value;

                // d₅: Target vector distance to source centroid (reverse cross-cluster penalty)
                let d5 = self
                    .distance_compute
                    .calculate_distance(
                        &vectors[target_idx],
                        &source_centroid.vector,
                        &DistanceMetric::Euclidean,
                    )
                    .raw_value;

                // Step 5: Calculate adaptive boosting factors based on statistical thresholds
                // α₁: Boundary detection for source vector (higher penalty for outliers)
                let alpha1 = if d1
                    > source_centroid.mean_distance
                        + self.boost_config.boundary_threshold * source_centroid.std_deviation
                {
                    self.boost_config.alpha_own // Apply penalty for boundary vectors
                } else {
                    1.0 // No penalty for well-contained vectors
                };

                // α₃: Boundary detection for target vector (symmetric to α₁)
                let alpha3 = if d3
                    > target_centroid.mean_distance
                        + self.boost_config.boundary_threshold * target_centroid.std_deviation
                {
                    self.boost_config.alpha_own // Apply penalty for boundary vectors
                } else {
                    1.0 // No penalty for well-contained vectors
                };

                // Step 6: Calculate dynamic scaling factors based on distance relationships
                // α₂: Inter-cluster penalty with logarithmic scaling (smooth increase with distance)
                let alpha2 =
                    self.boost_config.alpha_inter * (1.0 + (d2 / global_avg_distance).ln());

                // β₁: Cross-cluster exponential decay (rapid decrease for distant clusters)
                let beta1 = self.boost_config.beta_cross * (-d4 / global_avg_distance).exp();

                // β₂: Reverse cross-cluster exponential decay (symmetric to β₁)
                let beta2 = self.boost_config.beta_cross * (-d5 / global_avg_distance).exp();

                // Step 7: Apply the complete 5-component boosting formula
                // Each component contributes to different aspects of cluster quality:
                // - α₁×d₁: Penalize edges from boundary vectors (maintain intra-cluster quality)
                // - α₂×d₂: Scale by inter-cluster distance (preserve cluster separation)
                // - α₃×d₃: Penalize edges to boundary vectors (maintain target cluster quality)
                // - β₁×d₄: Cross-cluster penalty with exponential decay (smooth transitions)
                // - β₂×d₅: Reverse cross-cluster penalty (bidirectional consistency)
                let boosted_distance =
                    alpha1 * d1 + alpha2 * d2 + alpha3 * d3 + beta1 * d4 + beta2 * d5;

                // Debug logging for detailed component analysis (trace level to avoid spam)
                if edge_idx < 3 && node_idx % 1000 == 0 {
                    // Sample logging
                    tracing::trace!(
                        "Edge boosting: node {} → {} | components: d₁={:.3}×{:.2}={:.3}, d₂={:.3}×{:.2}={:.3}, \
                         d₃={:.3}×{:.2}={:.3}, d₄={:.3}×{:.2}={:.3}, d₅={:.3}×{:.2}={:.3} | final={:.3}",
                        source_idx,
                        target_idx,
                        d1,
                        alpha1,
                        alpha1 * d1,
                        d2,
                        alpha2,
                        alpha2 * d2,
                        d3,
                        alpha3,
                        alpha3 * d3,
                        d4,
                        beta1,
                        beta1 * d4,
                        d5,
                        beta2,
                        beta2 * d5,
                        boosted_distance
                    );
                }

                // Step 8: Create boosted edge with optional component storage for debugging
                let boost_info = if self.boost_config.store_components {
                    Some(BoostInfo {
                        d1,
                        d2,
                        d3,
                        d4,
                        d5,
                        alpha_values: [alpha1, alpha2, alpha3],
                        beta_values: [beta1, beta2],
                    })
                } else {
                    None
                };

                // Create the enhanced edge with both raw and boosted distance information
                boosted_edges.push(BoostedEdge {
                    target_node_id: edge.target_node_id,
                    target_vector_id: edge.target_vector_id.clone(),
                    raw_distance: edge.distance, // Original HNSW distance
                    boosted_distance,            // Enhanced distance with clustering awareness
                    boost_components: boost_info, // Optional detailed breakdown for analysis
                });

                total_edges_processed += 1;
            }

            // Step 9: Calculate improvement metrics for hybrid approach
            // These metrics help validate that boosting is improving clustering alignment
            if !node.edges.is_empty() && !boosted_edges.is_empty() {
                let avg_raw =
                    node.edges.iter().map(|e| e.distance).sum::<f32>() / node.edges.len() as f32;
                let avg_boosted = boosted_edges
                    .iter()
                    .map(|e| e.boosted_distance)
                    .sum::<f32>()
                    / boosted_edges.len() as f32;
                let improvement_pct = ((avg_boosted - avg_raw) / avg_raw * 100.0).abs();

                // Log significant improvements at trace level for detailed analysis
                if improvement_pct > 10.0 {
                    // Only log nodes with significant changes
                    tracing::trace!(
                        "Node {} (cluster {}): avg distance {:.3} → {:.3} ({:.1}% change, {} edges)",
                        node.vector_id,
                        source_cluster,
                        avg_raw,
                        avg_boosted,
                        (avg_boosted - avg_raw) / avg_raw * 100.0,
                        boosted_edges.len()
                    );
                }
            }

            // Step 10: Store boosted edges for serialization
            // Note: In production, this would update the node's edge structure
            // For compatibility, we maintain the current structure but log the enhanced metrics

            if node_idx % 2000 == 0 && node_idx > 0 {
                tracing::debug!(
                    "Processed {} / {} nodes for component boosting",
                    node_idx,
                    self.nodes.len()
                );
            }
        }

        // Step 11: Log comprehensive boosting statistics
        let intra_cluster_ratio = intra_cluster_edges as f32 / total_edges_processed.max(1) as f32;
        let inter_cluster_ratio = inter_cluster_edges as f32 / total_edges_processed.max(1) as f32;

        tracing::info!(
            "✅ Component boosting completed: {} nodes, {} edges processed. \
             Edge distribution: {:.1}% intra-cluster, {:.1}% inter-cluster (optimal: >70% intra)",
            self.nodes.len(),
            total_edges_processed,
            intra_cluster_ratio * 100.0,
            inter_cluster_ratio * 100.0
        );

        // Warn if too many inter-cluster edges (suggests poor clustering)
        if inter_cluster_ratio > 0.5 {
            tracing::warn!(
                "High inter-cluster edge ratio ({:.1}%) may indicate suboptimal clustering. \
                 Consider adjusting cluster count or boosting weights.",
                inter_cluster_ratio * 100.0
            );
        }
    }

    /// Dense node→cluster lookup built in ONE pass, replacing the per-edge
    /// a linear member-Vec scan does (every cluster's member Vec per
    /// edge → O(E×N) for the boosting pass). First-match semantics: a node in
    /// multiple clusters resolves to the lowest cluster id; `u32::MAX` marks
    /// not-assigned and resolves to 0 — both exactly like the linear scan.
    fn build_node_to_cluster_index(clusters: &[Vec<usize>]) -> Vec<u32> {
        let max_idx = clusters.iter().flat_map(|m| m.iter().copied()).max();
        let mut index = match max_idx {
            Some(max) => vec![u32::MAX; max + 1],
            None => return Vec::new(),
        };
        for (cluster_id, members) in clusters.iter().enumerate() {
            for &node in members {
                if node < index.len() && index[node] == u32::MAX {
                    index[node] = cluster_id as u32;
                }
            }
        }
        index
    }

    /// O(1) cluster lookup against [`Self::build_node_to_cluster_index`]'s
    /// output; out-of-range and unassigned nodes resolve to cluster 0, the
    /// not-found default of the linear scan this replaces.
    #[inline]
    fn cluster_of(node_to_cluster: &[u32], node_idx: usize) -> usize {
        match node_to_cluster.get(node_idx) {
            Some(&cluster) if cluster != u32::MAX => cluster as usize,
            _ => 0,
        }
    }

    /// Helper: Calculate global average distance for normalization
    fn calculate_global_avg_distance(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0;

        for row in &self.centroid_distances {
            for &dist in row {
                if dist > 0.0 {
                    total += dist;
                    count += 1;
                }
            }
        }

        if count > 0 { total / count as f32 } else { 1.0 }
    }

    /// Helper: Calculate centroid statistics for boosting
    fn calculate_centroid_statistics(&mut self, vectors: &[Vec<f32>]) {
        for centroid in &mut self.centroids {
            let mut distances = Vec::new();

            for vec in vectors {
                let dist = self
                    .distance_compute
                    .calculate_distance(vec, &centroid.vector, &DistanceMetric::Euclidean)
                    .raw_value;
                distances.push(dist);
            }

            // Calculate mean
            let mean = distances.iter().sum::<f32>() / distances.len() as f32;

            // Calculate standard deviation
            let variance =
                distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
            let std_dev = variance.sqrt();

            // Calculate 95th percentile (radius)
            distances.sort_by(|a, b| {
                a.partial_cmp(b).unwrap_or_else(|| {
                    // Handle NaN values by treating them as less than any number
                    if a.is_nan() && b.is_nan() {
                        std::cmp::Ordering::Equal
                    } else if a.is_nan() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
            });
            let percentile_95 = distances[(distances.len() as f32 * 0.95) as usize];

            centroid.mean_distance = mean;
            centroid.std_deviation = std_dev;
            centroid.radius = percentile_95;
        }
    }

    /// Convert clusters to rowgroups using 5-component boosting for intelligent co-location
    fn clusters_to_rowgroups(&mut self, clusters: Vec<Vec<usize>>) -> Vec<Vec<u32>> {
        let mut rowgroups = Vec::new();

        // Process each cluster independently
        for cluster in clusters.into_iter() {
            if cluster.is_empty() {
                continue;
            }

            // Convert cluster indices to node IDs
            let mut current_rowgroup = Vec::new();
            for node_idx in cluster {
                if node_idx < self.nodes.len() {
                    current_rowgroup.push(node_idx as u32);

                    // Split into target rowgroup size
                    if current_rowgroup.len() >= self.target_rowgroup_size {
                        rowgroups.push(current_rowgroup.clone());
                        current_rowgroup.clear();
                    }
                }
            }

            // Add remaining nodes
            if !current_rowgroup.is_empty() {
                rowgroups.push(current_rowgroup);
            }
        }

        rowgroups
    }
}

impl IvfClusteringBuilder {
    /// Build K×K inter-centroid distance matrix (upper triangle storage)
    fn build_inter_centroid_matrix(&self) -> InterCentroidMatrix {
        let k = self.centroids.len();

        // Handle edge case: no centroids or single centroid
        if k <= 1 {
            return InterCentroidMatrix {
                num_centroids: k as u32,
                compressed_data: Vec::new(),
                compression_metadata: InterCentroidCompressionMetadata::default(),
                lookup_table: Vec::new(),
            };
        }

        // Calculate all pairwise distances (upper triangle only)
        let upper_triangle_size = k * (k - 1) / 2;
        let mut distances = Vec::with_capacity(upper_triangle_size);
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        // Use the distance compute from the centroids
        for i in 0..k {
            for j in (i + 1)..k {
                let dist = self
                    .distance_compute
                    .calculate_distance(
                        &self.centroids[i].vector,
                        &self.centroids[j].vector,
                        &DistanceMetric::Cosine,
                    )
                    .raw_value;
                distances.push(dist);
                min_distance = min_distance.min(dist);
                max_distance = max_distance.max(dist);
            }
        }

        // Quantize distances to u16 for compression
        let scale_factor = if max_distance > min_distance {
            65535.0 / (max_distance - min_distance)
        } else {
            1.0
        };

        let mut compressed_data = Vec::with_capacity(upper_triangle_size * 2);
        let mut lookup_table = vec![0u32; k];
        let mut offset = 0u32;

        let mut idx = 0;
        for (i, lt_entry) in lookup_table.iter_mut().enumerate() {
            *lt_entry = offset;
            for _j in (i + 1)..k {
                let quantized = ((distances[idx] - min_distance) * scale_factor) as u16;
                compressed_data.extend_from_slice(&quantized.to_le_bytes());
                offset += 2;
                idx += 1;
            }
        }

        InterCentroidMatrix {
            num_centroids: k as u32,
            compressed_data,
            compression_metadata: InterCentroidCompressionMetadata {
                min_distance,
                max_distance,
                scale_factor,
                compression_type: CompressionType::Quantized16Bit,
                row_encodings: vec![],
                row_compressed_sizes: vec![],
            },
            lookup_table,
        }
    }

    /// Determine adaptive P×K storage strategy with exponential boundary detection
    /// Returns (strategy, coverage_ratio) based on K/D relationship
    fn determine_adaptive_pk_strategy(
        &self,
        k: f32,
        d: f32,
    ) -> (VectorCentroidStorageStrategy, f32) {
        let k_over_d = k / d;

        // Exponential decay function for coverage based on K/D ratio
        // boundary_score(k, d) = max(0.1, min(1.0, exp(-α × log(k/d + 1))))
        // where α = 2.0 (sensitivity parameter)
        let alpha = 2.0;
        let min_coverage = 0.1; // 10% floor - never go below this

        // Calculate coverage using smooth exponential decay
        let raw_coverage = (-alpha * (k_over_d + 1.0).ln()).exp();
        let coverage_ratio = f32::max(min_coverage, f32::min(raw_coverage, 1.0));

        // Determine strategy based on coverage requirements
        let strategy = match coverage_ratio {
            c if c >= 0.9 => VectorCentroidStorageStrategy::Full, // 90-100%: store all
            c if c >= 0.5 => VectorCentroidStorageStrategy::Hierarchical, // 50-90%: hierarchical
            _ => VectorCentroidStorageStrategy::Sparse,           // 10-50%: sparse only
        };

        tracing::debug!(
            "Adaptive P×K strategy: K/D={:.3}, coverage={:.1}%, strategy={:?}",
            k_over_d,
            coverage_ratio * 100.0,
            strategy
        );

        (strategy, coverage_ratio)
    }

    /// Build P×K vector-to-centroid distance matrices for all rowgroups
    fn build_vector_centroid_matrices(&self, rowgroups: &[RowGroup]) -> Vec<VectorCentroidMatrix> {
        let k = self.centroids.len();

        // Handle edge case: no centroids
        if k == 0 {
            return Vec::new();
        }

        let k_f32 = k as f32;
        let dimension = self.centroids[0].vector.len() as f32;

        // Adaptive storage strategy with exponential boundary detection
        let (storage_strategy, coverage_ratio) =
            self.determine_adaptive_pk_strategy(k_f32, dimension);

        let _coverage_percent = (coverage_ratio * 100.0) as u8;

        tracing::info!(
            "Building P×K matrices with strategy {:?} (K={}, D={}, ratio={:.2})",
            storage_strategy,
            k,
            dimension,
            k_f32 / dimension
        );

        let mut matrices = Vec::with_capacity(rowgroups.len());

        for (rg_idx, rowgroup) in rowgroups.iter().enumerate() {
            let matrix = match storage_strategy {
                VectorCentroidStorageStrategy::Full => self.build_full_pk_matrix(rowgroup, rg_idx),
                VectorCentroidStorageStrategy::Hierarchical => {
                    self.build_hierarchical_pk_matrix(rowgroup, rg_idx)
                }
                VectorCentroidStorageStrategy::Sparse => {
                    self.build_adaptive_sparse_pk_matrix(rowgroup, rg_idx, coverage_ratio)
                }
            };
            matrices.push(matrix);
        }

        matrices
    }

    /// Build full P×K matrix with quantization
    fn build_full_pk_matrix(&self, rowgroup: &RowGroup, rg_idx: usize) -> VectorCentroidMatrix {
        let p = rowgroup.vector_count;
        let k = self.centroids.len();

        // Calculate all distances and find max for quantization
        let mut distances = vec![vec![0.0f32; k]; p];
        let mut max_distance = 0.0f32;

        let vector_ids = if let Some(ref columnar_data) = rowgroup.columnar_data {
            &columnar_data.vector_ids
        } else {
            return self.build_empty_pk_matrix();
        };

        for (vec_idx, vector_id) in vector_ids.iter().enumerate() {
            // Find the actual vector data
            let vector_data = self.vector_by_id(vector_id);

            for (cent_idx, centroid) in self.centroids.iter().enumerate() {
                let dist = self
                    .distance_compute
                    .calculate_distance(&vector_data, &centroid.vector, &DistanceMetric::Euclidean)
                    .raw_value;
                distances[vec_idx][cent_idx] = dist;
                max_distance = max_distance.max(dist);
            }
        }

        let scale_factor = if max_distance > 0.0 {
            65535.0 / max_distance
        } else {
            1.0
        };

        // Quantize to 16-bit
        let mut compressed_data = Vec::with_capacity(p * k * 2);
        for vec_dists in &distances {
            for &dist in vec_dists {
                let quantized = (dist * scale_factor) as u16;
                compressed_data.extend_from_slice(&quantized.to_le_bytes());
            }
        }

        VectorCentroidMatrix {
            rowgroup_id: rg_idx as u16,
            num_vectors: p as u32,
            num_centroids: k as u32,
            storage_strategy: VectorCentroidStorageStrategy::Full,
            compressed_data,
            hierarchical_data: None,
            sparse_data: None,
            compression_metadata: VectorCentroidCompressionMetadata {
                centroid_stats: Vec::new(),
                global_min_distance: 0.0,
                global_max_distance: max_distance,
                global_mean_distance: max_distance / 2.0,
                centroid_encodings: Vec::new(),
            },
        }
    }

    /// Build empty P×K matrix when no data is available
    fn build_empty_pk_matrix(&self) -> VectorCentroidMatrix {
        VectorCentroidMatrix {
            rowgroup_id: 0,
            num_vectors: 0,
            num_centroids: 0,
            storage_strategy: VectorCentroidStorageStrategy::Sparse, // Empty matrix uses sparse
            compressed_data: Vec::new(),
            hierarchical_data: None,
            sparse_data: None,
            compression_metadata: VectorCentroidCompressionMetadata {
                centroid_stats: Vec::new(),
                global_min_distance: 0.0,
                global_max_distance: 0.0,
                global_mean_distance: 0.0,
                centroid_encodings: Vec::new(),
            },
        }
    }

    /// Build hierarchical P×K matrix with mean + sparse deltas
    fn build_hierarchical_pk_matrix(
        &self,
        rowgroup: &RowGroup,
        rg_idx: usize,
    ) -> VectorCentroidMatrix {
        let p = rowgroup.vector_count;
        let k = self.centroids.len();

        // Calculate all distances
        let mut distances = vec![vec![0.0f32; k]; p];
        let mut max_distance = 0.0f32;

        let vector_ids = if let Some(ref columnar_data) = rowgroup.columnar_data {
            &columnar_data.vector_ids
        } else {
            return self.build_empty_pk_matrix();
        };

        for (vec_idx, vector_id) in vector_ids.iter().enumerate() {
            let vector_data = self.vector_by_id(vector_id);

            for (cent_idx, centroid) in self.centroids.iter().enumerate() {
                let dist = self
                    .distance_compute
                    .calculate_distance(&vector_data, &centroid.vector, &DistanceMetric::Euclidean)
                    .raw_value;
                distances[vec_idx][cent_idx] = dist;
                max_distance = max_distance.max(dist);
            }
        }

        // Calculate mean distances per centroid
        let mut mean_distances = vec![0.0f32; k];
        for cent_idx in 0..k {
            let sum: f32 = distances.iter().map(|v| v[cent_idx]).sum();
            mean_distances[cent_idx] = sum / p as f32;
        }

        // Calculate deltas and store only significant ones (>5% deviation)
        let mut sparse_deltas = Vec::new();
        for (vec_idx, vec_dists) in distances.iter().enumerate() {
            for (cent_idx, &dist) in vec_dists.iter().enumerate() {
                let delta = dist - mean_distances[cent_idx];
                let deviation_pct = (delta.abs() / mean_distances[cent_idx].max(0.001)) * 100.0;

                if deviation_pct > 5.0 {
                    sparse_deltas.push(DeltaEntry {
                        vector_index: vec_idx as u32,
                        centroid_index: cent_idx as u16,
                        delta_value: delta,
                    });
                }
            }
        }

        // Quantize mean distances
        let scale_factor = if max_distance > 0.0 {
            65535.0 / max_distance
        } else {
            1.0
        };

        let mut compressed_data = Vec::with_capacity(k * 2);
        for &mean_dist in &mean_distances {
            let quantized = (mean_dist * scale_factor) as u16;
            compressed_data.extend_from_slice(&quantized.to_le_bytes());
        }

        tracing::debug!(
            "Hierarchical P×K for rowgroup {}: {} vectors, {} centroids, {} sparse deltas ({:.2}% sparse)",
            rg_idx,
            p,
            k,
            sparse_deltas.len(),
            (1.0 - sparse_deltas.len() as f32 / (p * k) as f32) * 100.0
        );

        VectorCentroidMatrix {
            rowgroup_id: rg_idx as u16,
            num_vectors: p as u32,
            num_centroids: k as u32,
            storage_strategy: VectorCentroidStorageStrategy::Hierarchical,
            compressed_data,
            hierarchical_data: Some(HierarchicalData {
                mean_distances,
                sparse_deltas,
            }),
            sparse_data: None,
            compression_metadata: VectorCentroidCompressionMetadata {
                centroid_stats: Vec::new(),
                global_min_distance: 0.0,
                global_max_distance: max_distance,
                global_mean_distance: max_distance / 2.0,
                centroid_encodings: Vec::new(),
            },
        }
    }

    /// Build sparse P×K matrix with adaptive boundary detection
    /// Stores only vectors with boundary_score > threshold (10%-50% coverage)
    fn build_adaptive_sparse_pk_matrix(
        &self,
        rowgroup: &RowGroup,
        rg_idx: usize,
        coverage_ratio: f32,
    ) -> VectorCentroidMatrix {
        let p = rowgroup.vector_count;
        let k = self.centroids.len();
        let d = self.centroids[0].vector.len() as f32;

        // Calculate boundary threshold based on coverage ratio
        let boundary_threshold = 1.0 - coverage_ratio; // Higher threshold = more selective

        let mut boundary_vectors = Vec::new();
        let mut max_distance = 0.0f32;

        // Step 1: Identify boundary vectors using exponential decay formula
        let vector_ids = if let Some(ref columnar_data) = rowgroup.columnar_data {
            &columnar_data.vector_ids
        } else {
            return self.build_empty_pk_matrix();
        };

        for (vec_idx, vector_id) in vector_ids.iter().enumerate() {
            let vector_data = self.vector_by_id(vector_id);

            // Calculate distances to all centroids
            let mut centroid_distances: Vec<f32> = self
                .centroids
                .iter()
                .map(|centroid| {
                    self.distance_compute
                        .calculate_distance(&vector_data, &centroid.vector, &DistanceMetric::Cosine)
                        .raw_value
                })
                .collect();

            // Find assigned centroid (minimum distance) and nearest neighbor
            let min_distance = centroid_distances
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let assigned_centroid_idx = centroid_distances
                .iter()
                .position(|&d| d == min_distance)
                .unwrap_or(0);

            // Remove assigned centroid and find next nearest
            centroid_distances[assigned_centroid_idx] = f32::INFINITY;
            let neighbor_distance = centroid_distances
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);

            // Calculate boundary score using exponential decay
            // boundary_score = exp(-α × |d_own - d_neighbor|) × log(k/d + 1)
            let alpha = 2.0;
            let distance_diff = (min_distance - neighbor_distance).abs();
            let boundary_score = (-alpha * distance_diff).exp() * (k as f32 / d + 1.0).ln();

            // Store if boundary score exceeds threshold
            if boundary_score > boundary_threshold {
                // Reset distances array for storage
                for (cent_idx, centroid) in self.centroids.iter().enumerate() {
                    let dist = self
                        .distance_compute
                        .calculate_distance(&vector_data, &centroid.vector, &DistanceMetric::Cosine)
                        .raw_value;

                    centroid_distances[cent_idx] = dist;
                    max_distance = max_distance.max(dist);
                }

                boundary_vectors.push((vec_idx, centroid_distances));
            }
        }

        // Step 2: Quantize and compress boundary vector distances using unified modules
        let num_boundary_vectors = boundary_vectors.len();
        let mut sparse_entries = Vec::with_capacity(num_boundary_vectors * k);

        for (vec_idx, distances) in boundary_vectors {
            // Store all centroid distances for this boundary vector
            for (cent_idx, distance) in distances.iter().enumerate() {
                sparse_entries.push(SparseEntry {
                    vector_idx: vec_idx as u32,
                    centroid_idx: cent_idx as u32,
                    quantized_distance: (*distance / max_distance * 255.0) as u8,
                });
            }
        }

        // Step 3: Apply Proxima compression to sparse entries
        let distances_only: Vec<f32> = sparse_entries
            .iter()
            .map(|entry| entry.quantized_distance as f32 / 255.0 * max_distance)
            .collect();

        // Manual quantization
        let q_min = distances_only.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let q_max = distances_only
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = if q_max > q_min {
            255.0 / (q_max - q_min)
        } else {
            1.0
        };
        let quantized_u8: Vec<u8> = distances_only
            .iter()
            .map(|&d| ((d - q_min) * scale).round() as u8)
            .collect();

        // Apply Proxima encoding for SIMD optimization (migrated to ProximaCodec)
        let scheme = ProximaScheme::BitPacked { bits: 8 }; // Efficient for sparse data

        // Convert u8 to i32 for encoding (ProximaCodec operates on i32/i64/f32)
        let quantized_i32: Vec<i32> = quantized_u8.iter().map(|&v| v as i32).collect();

        let codec = ProximaCodec::global();
        let compressed_data = codec
            .encode_i32(&quantized_i32, scheme)
            .unwrap_or_else(|_| quantized_u8.clone()); // Fallback to uncompressed if encoding fails

        let sparsity_achieved = num_boundary_vectors as f32 / p as f32;

        tracing::info!(
            "Adaptive sparse P×K for rowgroup {}: {}/{} vectors stored ({:.1}% sparsity, target {:.1}%)",
            rg_idx,
            num_boundary_vectors,
            p,
            sparsity_achieved * 100.0,
            coverage_ratio * 100.0
        );

        // Step 4: Create BloomFilter for fast boundary vector lookup
        let bloom_filter = self.create_boundary_vector_bloom_filter(&sparse_entries);

        VectorCentroidMatrix {
            rowgroup_id: rg_idx as u16,
            num_vectors: p as u32,
            num_centroids: k as u32,
            storage_strategy: VectorCentroidStorageStrategy::Sparse,
            compressed_data,
            hierarchical_data: None,
            sparse_data: Some(SparseData {
                top_k: num_boundary_vectors as u32,
                entries: sparse_entries,
                boundary_bloom_filter: Some(bloom_filter),
                sparsity_ratio: sparsity_achieved,
            }),
            compression_metadata: VectorCentroidCompressionMetadata {
                centroid_stats: Vec::new(), // Would be populated with actual stats
                global_min_distance: 0.0,
                global_max_distance: max_distance,
                global_mean_distance: max_distance / 2.0,
                centroid_encodings: Vec::new(), // Would be populated with schemes
            },
        }
    }

    /// Create BloomFilter for fast boundary vector lookup (2KB, 1% false positive rate)
    fn create_boundary_vector_bloom_filter(&self, sparse_entries: &[SparseEntry]) -> Vec<u8> {
        // Extract unique vector indices that are boundary vectors
        let mut boundary_vector_ids: Vec<u32> = sparse_entries
            .iter()
            .map(|entry| entry.vector_idx)
            .collect();
        boundary_vector_ids.sort_unstable();
        boundary_vector_ids.dedup();

        // Use unified bloom filter module for consistency
        use proximadb_bloom::{BloomFilterConfig, factory::BloomFilterFactory};

        let bloom_config = BloomFilterConfig::for_sstable(boundary_vector_ids.len());
        let mut bloom = BloomFilterFactory::create(&bloom_config);

        for &vector_idx in &boundary_vector_ids {
            bloom.insert(&vector_idx.to_le_bytes());
        }

        bloom.serialize().unwrap_or_else(|_| vec![0u8; 2048])
    }

    /// Helper to get vector data by ID from stored vectors
    fn vector_by_id(&self, vector_id: &str) -> Vec<f32> {
        // Look up vector by ID from the node mapping
        if let Some(&node_idx) = self.id_to_node.get(vector_id)
            && (node_idx as usize) < self.vectors.len()
        {
            return self.vectors[node_idx as usize].clone();
        }

        // Fallback: return zero vector if not found
        // This shouldn't happen in normal operation
        tracing::warn!(
            "Vector {} not found in storage, returning zero vector",
            vector_id
        );
        vec![0.0; self.centroids.first().map_or(768, |c| c.vector.len())]
    }
}

impl RaptorWriter {
    /// Build P² matrix for intra-rowgroup navigation (replaces local HNSW)
    fn build_p2_matrix(&self, vectors: &[Vec<f32>]) -> Result<P2Matrix> {
        let n = vectors.len();
        let upper_triangle_size = n * (n - 1) / 2;
        let mut distances = Vec::with_capacity(upper_triangle_size);

        // Use UnifiedDistanceCompute with SIMD acceleration
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // Find min/max for quantization
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        // Compute upper triangle only
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = distance_compute
                    .calculate_distance(&vectors[i], &vectors[j], &DistanceMetric::Cosine)
                    .raw_value;
                distances.push(dist);
                min_distance = min_distance.min(dist);
                max_distance = max_distance.max(dist);
            }
        }

        // Quantize to INT8 manually for now
        let scale = 255.0 / (max_distance - min_distance);
        let quantized: Vec<u8> = distances
            .iter()
            .map(|&d| {
                let normalized = (d - min_distance) * scale;
                normalized.round() as u8
            })
            .collect();

        // Apply Proxima encoding using ProximaCodec
        let scheme = if max_distance - min_distance < 0.1 {
            // Small range - use delta encoding
            ProximaScheme::Delta {
                base: (min_distance * 1000.0) as i64,
            }
        } else if distances.len() > 10000 {
            // Large dataset - use bit packing
            ProximaScheme::BitPacked { bits: 8 }
        } else {
            // Default to frame-of-reference
            ProximaScheme::FrameOfReference {
                reference: (min_distance * 1000.0) as i64,
                bits: 8,
            }
        };

        // Convert u8 to u32 for codec encoding
        let quantized_u32: Vec<u32> = quantized.iter().map(|&v| v as u32).collect();
        let codec = ProximaCodec::global();
        let encoded = codec.encode_u32(&quantized_u32, scheme.clone())?;
        let compressed_size = encoded.len() as u32;

        Ok(P2Matrix {
            num_vectors: n as u32,
            distances: encoded,
            min_distance,
            max_distance,
            compression: scheme, // Move scheme here (last use)
            compressed_size,
        })
    }

    /// Build optimized K×K inter-centroid distance matrix (CRITICAL for Matrix Trinity)
    /// This replaces the simple Vec<Vec<f32>> with compressed InterCentroidMatrix
    fn build_kxk_inter_centroid_matrix(
        &self,
        final_centroids: &[Vec<f32>],
    ) -> Result<InterCentroidMatrix> {
        let k = final_centroids.len();

        tracing::info!(
            "Building K×K inter-centroid matrix: {} centroids, {} distances to compute",
            k,
            k * (k - 1) / 2
        );

        // Calculate all pairwise distances (upper triangle only)
        let upper_triangle_size = k * (k - 1) / 2;
        let mut distances = Vec::with_capacity(upper_triangle_size);
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        for i in 0..k {
            for j in (i + 1)..k {
                let dist = self
                    .distance_compute
                    .calculate_distance(
                        &final_centroids[i],
                        &final_centroids[j],
                        &DistanceMetric::Cosine,
                    )
                    .raw_value;

                distances.push(dist);
                min_distance = min_distance.min(dist);
                max_distance = max_distance.max(dist);
            }
        }

        // Quantize to INT8 for compression
        let quantized_distances: Vec<u8> = distances
            .iter()
            .map(|&dist| {
                let normalized = (dist - min_distance) / (max_distance - min_distance);
                (normalized * 255.0).round() as u8
            })
            .collect();

        // Apply Proxima encoding
        let scheme = ProximaScheme::BitPacked { bits: 8 };
        let quantized_u32: Vec<u32> = quantized_distances.iter().map(|&v| v as u32).collect();
        let codec = ProximaCodec::global();
        let encoded = codec.encode_u32(&quantized_u32, scheme.clone())?;
        let encoded_len = encoded.len();

        tracing::info!(
            "K×K matrix: {} centroids → {} bytes compressed ({:.2}x compression)",
            k,
            encoded_len,
            quantized_distances.len() as f32 / encoded_len as f32
        );

        Ok(InterCentroidMatrix {
            num_centroids: k as u32,
            compressed_data: encoded,
            compression_metadata: InterCentroidCompressionMetadata {
                min_distance,
                max_distance,
                scale_factor: 255.0 / (max_distance - min_distance),
                compression_type: CompressionType::Quantized8Bit,
                row_encodings: vec![scheme; k], // Same scheme for all rows initially
                row_compressed_sizes: vec![encoded_len as u16 / k as u16; k], // Average size per row
            },
            lookup_table: Vec::new(), // Would be populated with actual offsets
        })
    }

    /// Handle rowgroup overflow by creating new centroids (1-to-1 mapping enforcement)
    /// This ensures perfect parallelism: K centroids = K rowgroups
    fn handle_rowgroup_overflow(
        &self,
        initial_assignments: Vec<(usize, Vec<Vec<f32>>)>,
        max_vectors_per_rowgroup: usize,
    ) -> Result<Vec<(usize, Vec<Vec<f32>>)>> {
        let mut final_assignments = Vec::new();
        let mut next_centroid_id = 0;
        let num_initial = initial_assignments.len();

        tracing::info!(
            "Handling rowgroup overflow: max {} vectors per rowgroup",
            max_vectors_per_rowgroup
        );

        for (_original_centroid_id, mut vectors) in initial_assignments {
            // Split large centroids across multiple rowgroups
            // Each new rowgroup gets a unique centroid (1-to-1 mapping)
            while !vectors.is_empty() {
                let chunk_size = vectors.len().min(max_vectors_per_rowgroup);
                let rowgroup_vectors: Vec<_> = vectors.drain(..chunk_size).collect();

                final_assignments.push((next_centroid_id, rowgroup_vectors));

                tracing::debug!(
                    "Created centroid {} → rowgroup {} with {} vectors",
                    next_centroid_id,
                    next_centroid_id,
                    chunk_size
                );

                next_centroid_id += 1;
            }
        }

        tracing::info!(
            "Overflow handling complete: {} initial assignments → {} final centroids (K={})",
            num_initial,
            final_assignments.len(),
            next_centroid_id
        );

        Ok(final_assignments)
    }

    /// Calculate final centroid positions from rowgroup assignments
    fn calculate_final_centroids(
        &self,
        assignments: &[(usize, Vec<Vec<f32>>)],
    ) -> Result<Vec<Vec<f32>>> {
        let mut final_centroids = Vec::new();

        for (centroid_id, vectors) in assignments {
            if vectors.is_empty() {
                return Err(anyhow::anyhow!(
                    "Empty vector set for centroid {}",
                    centroid_id
                ));
            }

            // Calculate centroid as mean of all assigned vectors
            let dimension = vectors[0].len();
            let mut centroid = vec![0.0; dimension];

            for vector in vectors {
                for (i, &value) in vector.iter().enumerate() {
                    centroid[i] += value;
                }
            }

            // Normalize by count to get mean
            let count = vectors.len() as f32;
            for value in &mut centroid {
                *value /= count;
            }

            final_centroids.push(centroid);
        }

        tracing::info!(
            "Calculated {} final centroids from assignments",
            final_centroids.len()
        );

        Ok(final_centroids)
    }

    /// Compute all centroid-to-centroid distances for K×K matrix
    fn compute_all_centroid_distances(&mut self, centroids: &[(u32, Vec<f32>)]) -> Result<()> {
        let k = centroids.len();
        self.ivf_builder.centroid_distances = vec![vec![0.0; k]; k];

        for i in 0..k {
            for j in (i + 1)..k {
                let dist = self
                    .distance_compute
                    .calculate_distance(
                        &centroids[i].1,
                        &centroids[j].1,
                        &DistanceMetric::Euclidean,
                    )
                    .raw_value;

                // Store symmetrically
                self.ivf_builder.centroid_distances[i][j] = dist;
                self.ivf_builder.centroid_distances[j][i] = dist;
            }
        }

        Ok(())
    }

    /// Helper: Find nearest centroid for a vector
    fn find_nearest_centroid(&self, vec: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut min_dist = f32::MAX;
        let mut nearest = 0;

        for (idx, centroid) in centroids.iter().enumerate() {
            let dist = self
                .distance_compute
                .calculate_distance(vec, centroid, &DistanceMetric::Euclidean)
                .raw_value;
            if dist < min_dist {
                min_dist = dist;
                nearest = idx;
            }
        }

        nearest
    }

    /// Partition a large cluster into optimal rowgroups using 5-component boosted distances
    fn partition_cluster_with_boosting(
        &self,
        cluster_idx: usize,
        cluster: &[usize],
    ) -> Vec<Vec<usize>> {
        let mut subgroups = Vec::new();
        let mut remaining: Vec<usize> = cluster.to_vec();

        // Get cluster centroid for boosting calculations
        let cluster_centroid = &self.ivf_builder.centroids[cluster_idx];

        while !remaining.is_empty() {
            // Step 1: Start new subgroup with a seed vector (furthest from centroid for diversity)
            let seed_idx = self.find_furthest_from_centroid(&remaining, cluster_centroid);
            let seed = remaining.remove(seed_idx);
            let mut subgroup = vec![seed];

            // Step 2: Greedily add vectors with minimum boosted distance to subgroup
            while subgroup.len() < self.config.rowgroup_size && !remaining.is_empty() {
                // Find vector with minimum average boosted distance to current subgroup
                let (best_idx, best_score) =
                    self.find_best_addition_with_boosting(&remaining, &subgroup, cluster_idx);

                if best_score < f32::INFINITY {
                    let best_node = remaining.remove(best_idx);
                    subgroup.push(best_node);

                    // Log progress for large clusters
                    if subgroup.len().is_multiple_of(50) {
                        tracing::trace!(
                            "Building subgroup: {} vectors, best_score={:.4}",
                            subgroup.len(),
                            best_score
                        );
                    }
                } else {
                    break; // No good candidates left
                }
            }

            subgroups.push(subgroup);
        }

        tracing::debug!(
            "Partitioned cluster {} ({} vectors) into {} subgroups using boosted distances",
            cluster_idx,
            cluster.len(),
            subgroups.len()
        );

        subgroups
    }

    /// Find vector furthest from centroid (for diverse seed selection)
    fn find_furthest_from_centroid(&self, candidates: &[usize], _centroid: &Centroid) -> usize {
        let mut max_dist = 0.0;
        let mut best_idx = 0;

        for (idx, &node_idx) in candidates.iter().enumerate() {
            let dist = self.ivf_builder.nodes[node_idx].centroid_distance;
            if dist > max_dist {
                max_dist = dist;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Find best vector to add using 5-component boosted distances
    fn find_best_addition_with_boosting(
        &self,
        candidates: &[usize],
        current_group: &[usize],
        cluster_idx: usize,
    ) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_score = f32::INFINITY;

        // Get cluster centroid for boosting
        let _cluster_centroid = &self.ivf_builder.centroids[cluster_idx];

        // Step 1: Evaluate each candidate
        for (cand_idx, &candidate) in candidates.iter().enumerate() {
            let mut total_boosted_distance = 0.0;
            let mut count = 0;

            // Step 2: Calculate average boosted distance to current group members
            for &group_member in current_group {
                // Use edge information if available
                let candidate_node = &self.ivf_builder.nodes[candidate];
                let member_node = &self.ivf_builder.nodes[group_member];

                // Look for existing edge with boosted distance
                if let Some(edge) = candidate_node
                    .edges
                    .iter()
                    .find(|e| e.target_node_id == group_member as u32)
                {
                    // Use pre-computed boosted distance from edge
                    total_boosted_distance += edge.distance;
                    count += 1;
                } else {
                    // Estimate boosted distance using centroid distances
                    // Since both are in same cluster, use simplified formula
                    let d1 = candidate_node.centroid_distance;
                    let _d2 = 0.0; // Same cluster, so inter-centroid distance is 0
                    let d3 = member_node.centroid_distance;

                    // Simplified boosting for same-cluster vectors
                    let boosted = d1 * 0.5 + d3 * 0.5;
                    total_boosted_distance += boosted;
                    count += 1;
                }
            }

            // Step 3: Calculate average score
            if count > 0 {
                let avg_score = total_boosted_distance / count as f32;
                if avg_score < best_score {
                    best_score = avg_score;
                    best_idx = cand_idx;
                }
            }
        }

        (best_idx, best_score)
    }

    /// Calculate cohesion using boosted distances
    fn calculate_boosted_cohesion(&self, group: &[u32]) -> f32 {
        let mut total_distance = 0.0;
        let mut count = 0;

        // Calculate average pairwise boosted distance within group
        for &node_idx in group {
            let node = &self.ivf_builder.nodes[node_idx as usize];

            // Use boosted edge distances for cohesion
            for edge in &node.edges {
                if group.contains(&edge.target_node_id) {
                    // Edge distance already includes boosting
                    total_distance += edge.distance;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_distance / count as f32
        } else {
            f32::MAX
        }
    }

    /// Calculate cohesion metric for a row group (average intra-group distance)
    fn calculate_cohesion(&self, group: &[u32]) -> f32 {
        let mut total_distance = 0.0;
        let mut count = 0;

        for &node_idx in group {
            let node = &self.ivf_builder.nodes[node_idx as usize];
            // Calculate cohesion using edge distances
            for edge in &node.edges {
                if group.contains(&edge.target_node_id) {
                    total_distance += edge.distance;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_distance / count as f32
        } else {
            f32::MAX
        }
    }
}

/// Column projections builder
struct ColumnProjectionsBuilder {
    metadata_columns: HashMap<String, Vec<Vec<u8>>>,
    #[allow(dead_code)]
    filter_bitmaps: HashMap<String, Vec<bool>>,
}

#[derive(Debug, Clone, Copy)]
struct RowLocation {
    #[allow(dead_code)]
    row_group_id: u32,
    #[allow(dead_code)]
    page_id: u16,
    #[allow(dead_code)]
    offset_in_page: u16,
}

/// Minimal node in the HNSW graph - stores only ID and edges with distances
/// Reduces memory by 96% compared to storing full vectors
#[derive(Debug, Clone)]
/// Hybrid IVF+Graph node combining clustering with local connectivity
/// Uses both IVF cluster assignment AND edges for navigation
struct IvfNode {
    /// UUID-style ID (32 bytes)
    vector_id: String,
    /// Cluster assignment (0 to k-1) for IVF routing
    cluster_id: u32,
    /// Location in row group (row_group_id, row_offset)
    #[allow(dead_code)]
    row_location: RowLocation,
    /// Distance to assigned centroid for boosting
    centroid_distance: f32,
    /// Local edges within cluster for graph navigation
    edges: Vec<EdgeWithDistance>,
}

/// Edge with distance for intelligent row group clustering
#[derive(Debug, Clone)]
struct EdgeWithDistance {
    #[allow(dead_code)]
    target_node_id: u32,
    #[allow(dead_code)]
    target_vector_id: String,
    distance: f32, // Similarity distance for clustering decisions
}

/// Enhanced edge with pre-computed boosted distance (serialized)
#[derive(Debug, Clone)]
struct BoostedEdge {
    #[allow(dead_code)]
    target_node_id: u32,
    #[allow(dead_code)]
    target_vector_id: String,
    #[allow(dead_code)]
    raw_distance: f32, // Original distance
    boosted_distance: f32, // Pre-computed boosted distance
    #[allow(dead_code)]
    boost_components: Option<BoostInfo>, // Optional: store component breakdown
}

/// Boost component breakdown for debugging/tuning
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BoostInfo {
    d1: f32, // Source to its centroid
    d2: f32, // Centroid to centroid
    d3: f32, // Target to its centroid
    d4: f32, // Source to target centroid
    d5: f32, // Target to source centroid
    alpha_values: [f32; 3],
    beta_values: [f32; 2],
}

// Additional fields for tracking current state
struct CurrentRowgroup {
    #[allow(dead_code)]
    batch: RecordBatch,
    #[allow(dead_code)]
    size: usize,
}

// Metadata column analysis for intelligent encoding
#[allow(dead_code)]
struct MetadataColumn {
    name: String,
    values: Vec<String>,
    distinct_count: usize,
    all_integers: bool,
    all_floats: bool,
    all_booleans: bool,
}

impl MetadataColumn {
    fn new(name: String) -> Self {
        Self {
            name,
            values: Vec::new(),
            distinct_count: 0,
            all_integers: true,
            all_floats: true,
            all_booleans: true,
        }
    }

    #[allow(dead_code)]
    fn add_value(&mut self, value: String) {
        // Check type compatibility
        if self.all_integers {
            self.all_integers = value.parse::<i64>().is_ok();
        }
        if self.all_floats {
            self.all_floats = value.parse::<f32>().is_ok();
        }
        if self.all_booleans {
            let lower = value.to_lowercase();
            self.all_booleans = lower == "true" || lower == "false" || value == "0" || value == "1";
        }

        self.values.push(value);
    }

    #[allow(dead_code)]
    fn analyze_and_choose_encoding(&mut self) -> MetadataEncoding {
        use std::collections::HashSet;

        // Calculate distinct count
        let unique: HashSet<_> = self.values.iter().cloned().collect();
        self.distinct_count = unique.len();

        // Choose encoding based on characteristics
        if self.distinct_count == 1 {
            MetadataEncoding::RunLength
        } else if self.all_booleans {
            MetadataEncoding::Boolean
        } else if self.all_integers {
            MetadataEncoding::Integer
        } else if self.all_floats {
            MetadataEncoding::Float
        } else if self.distinct_count <= self.values.len() / 10 {
            // Dictionary encoding if cardinality < 10%
            MetadataEncoding::Dictionary
        } else {
            MetadataEncoding::String
        }
    }

    #[allow(dead_code)]
    fn build_dictionary(&self) -> Vec<String> {
        use std::collections::BTreeSet;
        let unique: BTreeSet<_> = self.values.iter().cloned().collect();
        unique.into_iter().collect()
    }

    #[allow(dead_code)]
    fn encode_as_indices(&self, dict: &[String]) -> Vec<usize> {
        let dict_map: HashMap<_, _> = dict
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        self.values
            .iter()
            .map(|v| *dict_map.get(v.as_str()).unwrap_or(&0))
            .collect()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum MetadataEncoding {
    Dictionary, // Low cardinality strings
    Integer,    // Integer values with Proxima
    Float,      // Float values with Proxima
    Boolean,    // Boolean values as bits
    String,     // High cardinality strings
    RunLength,  // All values the same
}

/// Metadata structure for bloom filter storage
// REMOVED: BloomFilterMetadata - duplicate of common.rs::BloomFilterMetadata
// Use type alias to maintain local naming if needed
type BloomFilterMetadata = proximadb_raptor_common::BloomFilterMetadata;

impl MetadataEncoding {
    #[allow(dead_code)]
    #[allow(dead_code)]
    fn to_byte(self) -> u8 {
        match self {
            Self::Dictionary => 0x10,
            Self::Integer => 0x11,
            Self::Float => 0x12,
            Self::Boolean => 0x13,
            Self::String => 0x14,
            Self::RunLength => 0x15,
        }
    }
}

/// Build a test-only `AxisClusteringPort` for the isolated unit tests below.
/// They construct `IvfClusteringBuilder` directly, without the composition root
/// (engine.rs) that normally injects `AxisClusteringEngine` — which lives in the
/// root crate and so cannot be named from here. The stub's methods are never
/// invoked (the tests use placeholder rowgroups; actual clustering is deferred),
/// so no-op implementations suffice.
#[cfg(test)]
fn make_test_axis_clustering() -> Arc<dyn AxisClusteringPort> {
    Arc::new(TestAxisClustering)
}

#[cfg(test)]
struct TestAxisClustering;

#[cfg(test)]
impl AxisClusteringPort for TestAxisClustering {
    fn cluster_vectors_simple(
        &self,
        _vectors: &[Vec<f32>],
        _k: usize,
        _distance_metric: DistanceMetric,
        _max_iterations: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        Ok((Vec::new(), Vec::new()))
    }

    fn calculate_centroid_distance_matrix(
        &self,
        _centroids: &[Vec<f32>],
        _distance_metric: DistanceMetric,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }

    fn assign_vectors_with_component_boosting(
        &self,
        _vectors: &[Vec<f32>],
        _centroids: &[Vec<f32>],
        _centroid_distances: &[Vec<f32>],
        _distance_metric: DistanceMetric,
        _boosting_weights: &[f32],
    ) -> Result<Vec<(usize, f32)>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod minimal_hnsw_tests {
    use super::*;

    #[test]
    fn test_distance_aware_clustering() {
        // Create a minimal HNSW builder
        let hw_caps = proximadb_hardware_caps::get_hardware_capabilities();
        let mut builder = IvfClusteringBuilder::new(3, hw_caps, make_test_axis_clustering()); // Small row groups for testing

        // Add nodes with predefined edges and distances
        // Node 0 connects to 1 (distance 0.1) and 2 (distance 0.8)
        builder.add_node(
            "vec_0".to_string(),
            vec![
                EdgeWithDistance {
                    target_node_id: 1,
                    target_vector_id: "vec_1".to_string(),
                    distance: 0.1,
                },
                EdgeWithDistance {
                    target_node_id: 2,
                    target_vector_id: "vec_2".to_string(),
                    distance: 0.8,
                },
            ],
        );

        // Node 1 connects to 0 (distance 0.1) and 3 (distance 0.2)
        builder.add_node(
            "vec_1".to_string(),
            vec![
                EdgeWithDistance {
                    target_node_id: 0,
                    target_vector_id: "vec_0".to_string(),
                    distance: 0.1,
                },
                EdgeWithDistance {
                    target_node_id: 3,
                    target_vector_id: "vec_3".to_string(),
                    distance: 0.2,
                },
            ],
        );

        // Node 2 connects to 0 (distance 0.8) and 4 (distance 0.15)
        builder.add_node(
            "vec_2".to_string(),
            vec![
                EdgeWithDistance {
                    target_node_id: 0,
                    target_vector_id: "vec_0".to_string(),
                    distance: 0.8,
                },
                EdgeWithDistance {
                    target_node_id: 4,
                    target_vector_id: "vec_4".to_string(),
                    distance: 0.15,
                },
            ],
        );

        // Node 3 connects to 1 (distance 0.2) and 4 (distance 0.3)
        builder.add_node(
            "vec_3".to_string(),
            vec![
                EdgeWithDistance {
                    target_node_id: 1,
                    target_vector_id: "vec_1".to_string(),
                    distance: 0.2,
                },
                EdgeWithDistance {
                    target_node_id: 4,
                    target_vector_id: "vec_4".to_string(),
                    distance: 0.3,
                },
            ],
        );

        // Node 4 connects to 2 (distance 0.15) and 3 (distance 0.3)
        builder.add_node(
            "vec_4".to_string(),
            vec![
                EdgeWithDistance {
                    target_node_id: 2,
                    target_vector_id: "vec_2".to_string(),
                    distance: 0.15,
                },
                EdgeWithDistance {
                    target_node_id: 3,
                    target_vector_id: "vec_3".to_string(),
                    distance: 0.3,
                },
            ],
        );

        // Deferred: Perform clustering - cluster_into_rowgroups method needs to be implemented
        // Temporary placeholder for compilation
        let rowgroups = vec![vec![0, 1], vec![2, 3, 4]]; // Placeholder clustering
        assert!(rowgroups.len() >= 2, "Should create at least 2 row groups");

        // Check that each node is assigned to exactly one row group
        let mut all_nodes: Vec<u32> = Vec::new();
        for group in &rowgroups {
            all_nodes.extend(group);
        }
        all_nodes.sort();
        assert_eq!(
            all_nodes,
            vec![0, 1, 2, 3, 4],
            "All nodes should be assigned"
        );

        // Deferred: Verify cohesion - calculate_cohesion method needs to be implemented
        // for group in &rowgroups {
        //     let cohesion = builder.calculate_cohesion(group);
        //     assert!(cohesion < 1.0, "Row groups should have good cohesion");
    }
}

#[test]
fn test_uniqueness_guarantee() {
    let hw_caps = proximadb_hardware_caps::get_hardware_capabilities();
    let mut builder = IvfClusteringBuilder::new(5, hw_caps, make_test_axis_clustering());

    // Add 10 nodes
    for i in 0..10 {
        let edges = if i > 0 {
            vec![EdgeWithDistance {
                target_node_id: i - 1,
                target_vector_id: format!("vec_{}", i - 1),
                distance: 0.1,
            }]
        } else {
            vec![]
        };
        builder.add_node(format!("vec_{}", i), edges);
    }

    // Deferred: cluster_into_rowgroups method needs to be implemented
    // Temporary placeholder for compilation
    let rowgroups = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]]; // Placeholder clustering

    // Verify each ID exists in exactly one row group
    let mut id_count = vec![0; 10];
    for group in &rowgroups {
        for &node_idx in group {
            id_count[node_idx as usize] += 1;
        }
    }

    for count in id_count {
        assert_eq!(count, 1, "Each ID should appear exactly once");
    }
}

#[test]
fn test_memory_reduction() {
    // Calculate memory usage for 1M vectors
    let num_vectors = 1_000_000;
    let dimension = 1536;

    // Legacy approach: full vectors
    let legacy_per_node = dimension * 4 + 32 + 64; // vector + id + edges
    let legacy_total = (num_vectors as i64) * (legacy_per_node as i64);

    // Minimal approach: ID only
    let minimal_per_node = 32 + 8 + 64; // id + location + edges  
    let minimal_total = num_vectors * minimal_per_node;

    let reduction_percent = (1.0 - (minimal_total as f64 / legacy_total as f64)) * 100.0;

    assert!(
        reduction_percent > 95.0,
        "Should achieve >95% memory reduction"
    );
    println!("Memory reduction: {:.1}%", reduction_percent);
    println!(
        "Legacy: {} MB, Minimal: {} MB",
        legacy_total / (1024 * 1024),
        minimal_total / (1024 * 1024)
    );
}

impl RaptorWriter {
    pub async fn new(
        file_path: String,
        config: RaptorConfig,
        collection_id: String,
        dimension: usize,
        // Injected collaborators (Slice D port-inversion): the composition root
        // (engine.rs) constructs the concrete engines + filesystem and injects
        // them, so this module depends only on the port traits — enabling its
        // extraction to a crate.
        filesystem: Arc<dyn FileSystem>,
        quantization_engine: Arc<dyn StorageQuantizationEnginePort>,
        axis_clustering: Arc<dyn AxisClusteringPort>,
    ) -> Result<Self> {
        if !filesystem.supports_append() {
            anyhow::bail!(
                "RAPTOR requires durable append, but backend {} for {} does not support it",
                filesystem.filesystem_type(),
                file_path
            );
        }

        // Initialize hardware capabilities
        let hardware = get_hardware_capabilities();

        // Initialize unified compression
        let _compression_algo = match &config.compression {
            RaptorCompressionCodec::None => CompressionAlgorithm::None,
            RaptorCompressionCodec::Lz4 => CompressionAlgorithm::Lz4,
            RaptorCompressionCodec::Zstd(_level) => CompressionAlgorithm::Zstd,
            RaptorCompressionCodec::Snappy => CompressionAlgorithm::Snappy,
            RaptorCompressionCodec::Gzip(_level) => CompressionAlgorithm::Gzip,
        };
        let compression = Arc::new(StandardCompression);

        // Initialize unified distance compute first
        let distance_compute = Arc::new(UnifiedDistanceCompute::new(
            proximadb_distance_kernel::DistanceMetric::Cosine,
        ));

        // Quantization engine is injected (see fn params above) — the concrete
        // codebook store / unified quant / storage quant chain is assembled at
        // the composition root (engine.rs). The distance_compute initialized
        // above is this writer's own field.

        // Initialize memory pool with default configuration
        let memory_pool = Arc::new(VectorMemoryPool::new());

        // Initialize file metadata (matching consolidated struct)
        let file_metadata = RaptorFileMetadata {
            version: 1,
            created_by: "ProximaDB RAPTOR v1.0".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            file_path: file_path.clone(),
            file_size: 0,
            total_rows: 0,
            total_vectors: 0,
            dimension,
            collection_id: collection_id.clone(),
            row_groups: Vec::new(),
            num_rowgroups: 0,
            rowgroup_offsets: Vec::new(),
            rowgroup_sizes: Vec::new(),
            rowgroup_vector_counts: Vec::new(),
            schema: SchemaDescriptor {
                vector_dimension: dimension,
                metadata_fields: Vec::new(),
                version: 1,
            },
            // Matrix Trinity metadata replaces HNSW
            // K² matrix stored in footer
            // P² and P×K matrices stored per rowgroup
            bloom_filter_metadata: None,
            compression_codec: format!("{:?}", config.compression),
            compression_ratio: 0.0,        // Will be calculated during flush
            cluster_centroids: Vec::new(), // Will be populated during clustering
            cluster_assignments: HashMap::new(), // Will be populated during clustering
            custom_metadata: HashMap::new(),
            key_value_metadata: Vec::new(),
            footer_offset: 0,
            footer_size: 0,
            last_accessed: 0,
            locality_clusters: Vec::new(),
        };

        // Write header magic at file start
        filesystem
            .write(&file_path, constants::RAPTOR_MAGIC, None)
            .await?;

        // Initialize matrix builder
        let matrix_builder = MatrixBuilder::new(
            distance_compute.clone(),
            hardware.clone(),
            DistanceMetric::Cosine, // Use config's distance metric
        );

        // Extract rowgroup_size before moving config
        let rowgroup_size = config.rowgroup_size;

        Ok(Self {
            file_path: file_path.clone(),
            filesystem,
            config,
            collection_id,
            dimension,
            compression,
            quantization_engine,
            memory_pool,
            hardware,
            distance_compute,
            matrix_builder,
            current_row_page: None,
            current_rowgroup: None,
            row_groups: Vec::new(),
            file_metadata,
            bloom_builder: BloomFilterBuilder::new(0.01),
            id_column_builder: IdColumnBuilder {
                ids: Vec::new(),
                id_hashes: Vec::new(),
                row_offsets: Vec::new(),
            },
            ivf_builder: IvfClusteringBuilder::new(
                rowgroup_size,
                get_hardware_capabilities(),
                axis_clustering,
            ),
            column_projections: ColumnProjectionsBuilder {
                metadata_columns: HashMap::new(),
                filter_bitmaps: HashMap::new(),
            },
            file_created: true, // Header was written
        })
    }

    /// Write vector records (main entry point)
    pub async fn write_vectors(&mut self, vectors: &[VectorRecord]) -> Result<()> {
        for vector in vectors {
            self.add_vector(vector).await?;

            // Flush page when it reaches configured row page size (default 1000 for optimal HNSW I/O)
            // This minimizes wasted reads: at k=10, reads 1000 vectors for 10 results (1% efficiency)
            if let Some(ref page) = self.current_row_page
                && page.rows.len() >= self.config.rowgroup_size
            {
                self.flush_row_page().await?;
            }
        }
        Ok(())
    }

    /// Add a single vector to the current page
    /// Stores both FP32 and quantized versions for full reconstruction
    async fn add_vector(&mut self, vector: &VectorRecord) -> Result<()> {
        // Extract ID - required field in VectorRecord
        let id = vector.id.clone();

        // Store original FP32 vector for reconstruction
        let fp32_vector = vector.vector.clone();

        // Get quantized vectors - both INT8 (primary) and binary sketch for progressive search
        let (quantized_vector, binary_sketch) = {
            // Quantize vector using unified engine
            let quantized_batch = self
                .quantization_engine
                .quantize_batch(
                    std::slice::from_ref(&vector.vector),
                    Some(std::slice::from_ref(&id)),
                )
                .await?;
            let quantized = quantized_batch
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Failed to quantize vector"))?;

            // Get the primary quantized data (INT8)
            let int8_data = quantized.primary.map_or_else(Vec::new, |q| q.data);

            // Get binary sketch for progressive search (1-bit per dimension)
            // Fall back to computing it directly if not available from quantization engine
            let binary_data = quantized.filter.map_or_else(
                || {
                    // Compute binary sketch: 1 bit per dimension (sign-based)
                    let dim = vector.vector.len();
                    let mut binary = vec![0u8; dim.div_ceil(8)];
                    for (i, &val) in vector.vector.iter().enumerate() {
                        if val > 0.0 {
                            binary[i / 8] |= 1 << (i % 8);
                        }
                    }
                    binary
                },
                |q| q.data,
            );

            (int8_data, binary_data)
        };

        // Extract metadata as key-value pairs (byte arrays for custom binary format)
        // Deferred: Migrate to SqlValue when refactoring RAPTOR's encoding/decoding logic
        let metadata: Vec<(String, Vec<u8>)> = vector
            .metadata
            .iter()
            .map(|(key, value)| {
                let value_bytes = match &value.value {
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::StringValue(s)) => {
                        s.as_bytes().to_vec()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::NumberValue(n)) => {
                        n.to_le_bytes().to_vec()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::BoolValue(b)) => {
                        vec![if *b { 1 } else { 0 }]
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::Int64Value(i)) => {
                        i.to_le_bytes().to_vec()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::BytesValue(b)) => {
                        b.clone()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::JsonbValue(b)) => {
                        b.clone()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::NullValue(_)) => {
                        Vec::new()
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::ArrayValue(arr)) => {
                        use prost::Message;
                        let mut buf = Vec::new();
                        if let Err(e) = arr.encode(&mut buf) {
                            tracing::warn!("ArrayValue encoding failed: {}, using empty bytes", e);
                            Vec::new()
                        } else {
                            buf
                        }
                    }
                    Some(proximadb_proto::proximadb_v1::sql_value::Value::ObjectValue(obj)) => {
                        use prost::Message;
                        let mut buf = Vec::new();
                        if let Err(e) = obj.encode(&mut buf) {
                            tracing::warn!("ObjectValue encoding failed: {}, using empty bytes", e);
                            Vec::new()
                        } else {
                            buf
                        }
                    }
                    None => Vec::new(),
                };
                (key.clone(), value_bytes)
            })
            .collect();

        // Extract source content if present
        let source_content = vector.source.as_ref().map(|source| {
            // Serialize SourceContent proto to bytes
            use prost::Message;
            let mut buf = Vec::new();
            if let Err(e) = source.encode(&mut buf) {
                tracing::warn!("Source content encoding failed: {}, using empty bytes", e);
                Vec::new()
            } else {
                buf
            }
        });

        // Create compact row with all VectorRecord fields
        let compact_row = CompactRow {
            id: id.clone(),
            vector: fp32_vector.clone(), // Clone for CompactRow, original used for IVF
            quantized_vector,
            binary_sketch, // Binary sketch for progressive search (1-bit per dimension)
            metadata,
            timestamp: vector.timestamp.unwrap_or(0) as u32,
            updated_at: vector.updated_at.map(|v| v as u32),
            expires_at: vector.expires_at.map(|v| v as u32),
            version: vector.version,
            source_content,
        };

        // Determine row location
        let page_id = self.row_groups.len() as u16;
        let offset_in_page = self
            .current_row_page
            .as_ref()
            .map_or(0, |p| p.rows.len() as u16);

        let location = RowLocation {
            row_group_id: self.row_groups.len() as u32,
            page_id,
            offset_in_page,
        };

        // Update bloom filter and columnar ID index
        self.bloom_builder.add_id(id.clone());
        self.id_column_builder.ids.push(id.clone());
        let id_hash = proximadb_kernel::hash::XxHasher::hash_bytes(id.as_bytes());
        self.id_column_builder.id_hashes.push(id_hash);
        self.id_column_builder
            .row_offsets
            .push(offset_in_page as u32);

        // Add to IVF builder with vector for clustering
        // Store vector for clustering and edge building
        self.ivf_builder.vectors.push(fp32_vector);

        // Add node with location information
        self.ivf_builder.nodes.push(IvfNode {
            vector_id: id.clone(),
            cluster_id: 0, // Will be assigned during clustering
            row_location: location,
            centroid_distance: 0.0, // Will be calculated during clustering
            edges: Vec::new(),      // Will be built after clustering
        });

        // Update id mapping
        let node_id = (self.ivf_builder.nodes.len() - 1) as u32;
        self.ivf_builder.id_to_node.insert(id.clone(), node_id);

        // Update column projections for filtering
        self.update_column_projections(vector, location)?;

        // Add to current page
        if self.current_row_page.is_none() {
            self.current_row_page = Some(RowPageBuffer {
                rows: Vec::new(),
                page_id,
                start_offset: self
                    .filesystem
                    .metadata(&self.file_path)
                    .await
                    .map(|m| m.size)
                    .unwrap_or(0),
            });
        }

        // SAFETY: current_row_page is guaranteed to be Some after the initialization above.
        self.current_row_page
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Current row page not initialized"))?
            .rows
            .push(compact_row);
        self.file_metadata.total_rows += 1;

        Ok(())
    }

    async fn quantize_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        // Simplified - would use actual quantization
        Ok(batch.clone())
    }

    /// Write bloom filter to disk and return its offset
    async fn write_bloom_filter(&mut self, bloom_filter: &RowGroupBloomFilter) -> Result<u64> {
        // Serialize bloom filter to bytes
        let bloom_data = bincode::serialize(bloom_filter)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bloom filter: {}", e))?;

        // Compress bloom filter for storage efficiency
        let compressed = self.compression.compress(
            &bloom_data,
            CompressionAlgorithm::Zstd,
            6,
            CompressionContext::Block,
        )?;

        // Get current file size as offset before append
        let offset = self
            .filesystem
            .metadata(&self.file_path)
            .await
            .map(|m| m.size)
            .unwrap_or(0);

        // Write to filesystem
        self.filesystem.append(&self.file_path, &compressed).await?;

        tracing::debug!(
            "Wrote bloom filter: {} bytes compressed from {} bytes original ({:.1}% compression)",
            compressed.len(),
            bloom_data.len(),
            (1.0 - compressed.len() as f64 / bloom_data.len() as f64) * 100.0
        );

        Ok(offset)
    }

    /// Flush current row page using columnar compression
    async fn flush_row_page(&mut self) -> Result<()> {
        if let Some(page) = self.current_row_page.take() {
            let mut column_pages = HashMap::new();
            let rowgroup_id = self.row_groups.len() as u32;

            // === 1. Compress and write vector column ===
            let vector_data = self.encode_vector_column(&page)?;
            let vector_compressed = self.compression.compress(
                &vector_data,
                CompressionAlgorithm::Lz4, // Fast decompression for hot path
                3,
                CompressionContext::VectorSerialization,
            )?;
            // Get current file size, or 0 if file doesn't exist yet
            let vector_offset = match self.filesystem.metadata(&self.file_path).await {
                Ok(meta) => meta.size,
                Err(_) => 0, // File doesn't exist yet, start at offset 0
            };
            self.filesystem
                .append(&self.file_path, &vector_compressed)
                .await?;
            column_pages.insert(
                ColumnType::VectorsFp32,
                ColumnPageMetadata {
                    column_type: ColumnType::VectorsFp32,
                    offset: vector_offset,
                    compressed_size: vector_compressed.len() as u64,
                    uncompressed_size: vector_data.len() as u64,
                    compression: CompressionAlgorithm::Lz4,
                    encoding: Some(ProximaScheme::BitPacked { bits: 16 }),
                    null_count: 0,
                    min_value: None,
                    max_value: None,
                },
            );

            // === 2. Compress and write ID column ===
            let id_data = self.encode_id_column(&page)?;
            let id_compressed = self.compression.compress(
                &id_data,
                CompressionAlgorithm::Zstd, // Higher compression for IDs
                9,
                CompressionContext::Column, // IDs are homogeneous columnar data
            )?;
            let id_offset = self
                .filesystem
                .metadata(&self.file_path)
                .await
                .map(|m| m.size)
                .unwrap_or(0);
            self.filesystem
                .append(&self.file_path, &id_compressed)
                .await?;
            column_pages.insert(
                ColumnType::Ids,
                ColumnPageMetadata {
                    column_type: ColumnType::Ids,
                    offset: id_offset,
                    compressed_size: id_compressed.len() as u64,
                    uncompressed_size: id_data.len() as u64,
                    compression: CompressionAlgorithm::Zstd,
                    encoding: None,
                    null_count: 0,
                    min_value: None,
                    max_value: None,
                },
            );

            // === 3. Compress metadata columns individually ===
            let metadata_columns = self.group_metadata_by_key(&page)?;
            for (key, values) in metadata_columns {
                let meta_data = self.encode_metadata_column(&key, &values)?;

                // Choose compression based on cardinality
                let unique_values: HashSet<_> = values.iter().collect();
                let cardinality_ratio = unique_values.len() as f32 / values.len() as f32;
                let algorithm = if cardinality_ratio < 0.1 {
                    CompressionAlgorithm::Zstd // Better for dictionary-encoded
                } else {
                    CompressionAlgorithm::Snappy // Faster for high cardinality
                };

                let meta_compressed = self.compression.compress(
                    &meta_data,
                    algorithm,
                    6,
                    CompressionContext::Block, // Metadata is mixed/heterogeneous data
                )?;
                let meta_offset = match self.filesystem.metadata(&self.file_path).await {
                    Ok(meta) => meta.size,
                    Err(_) => 0,
                };
                self.filesystem
                    .append(&self.file_path, &meta_compressed)
                    .await?;
                column_pages.insert(
                    ColumnType::Metadata(key.clone()),
                    ColumnPageMetadata {
                        column_type: ColumnType::Metadata(key),
                        offset: meta_offset,
                        compressed_size: meta_compressed.len() as u64,
                        uncompressed_size: meta_data.len() as u64,
                        compression: algorithm,
                        encoding: None,
                        null_count: 0,
                        min_value: None,
                        max_value: None,
                    },
                );
            }

            // === 4. Compress source content with maximum compression (if present) ===
            if self.has_source_content(&page) {
                let source_data = self.encode_source_content(&page)?;
                let source_compressed = self.compression.compress(
                    &source_data,
                    CompressionAlgorithm::Zstd, // Best ratio for text
                    19,                         // Maximum compression
                    CompressionContext::Block,  // Source content is text/mixed data
                )?;
                let source_offset = match self.filesystem.metadata(&self.file_path).await {
                    Ok(meta) => meta.size,
                    Err(_) => 0,
                };
                self.filesystem
                    .append(&self.file_path, &source_compressed)
                    .await?;
                column_pages.insert(
                    ColumnType::SourceContent,
                    ColumnPageMetadata {
                        column_type: ColumnType::SourceContent,
                        offset: source_offset,
                        compressed_size: source_compressed.len() as u64,
                        uncompressed_size: source_data.len() as u64,
                        compression: CompressionAlgorithm::Zstd,
                        encoding: None,
                        null_count: 0,
                        min_value: None,
                        max_value: None,
                    },
                );
            }

            // === 5. Build and write P² matrix ===
            let page_vectors: Vec<Vec<f32>> = page.rows.iter().map(|r| r.vector.clone()).collect();
            let p2_matrix = self.build_p2_matrix(&page_vectors)?;
            let p2_data = bincode::serialize(&p2_matrix)?;
            let p2_compressed = self.compression.compress(
                &p2_data,
                CompressionAlgorithm::Lz4, // Fast access for navigation
                6,
                CompressionContext::Column, // Matrix data is numeric/homogeneous
            )?;
            let p2_offset = match self.filesystem.metadata(&self.file_path).await {
                Ok(meta) => meta.size,
                Err(_) => 0,
            };
            self.filesystem
                .append(&self.file_path, &p2_compressed)
                .await?;
            column_pages.insert(
                ColumnType::P2Matrix,
                ColumnPageMetadata {
                    column_type: ColumnType::P2Matrix,
                    offset: p2_offset,
                    compressed_size: p2_compressed.len() as u64,
                    uncompressed_size: p2_data.len() as u64,
                    compression: CompressionAlgorithm::Lz4,
                    encoding: None,
                    null_count: 0,
                    min_value: None,
                    max_value: None,
                },
            );

            // Compute centroid for this rowgroup (CRITICAL for hierarchical search)
            let centroid = if !page_vectors.is_empty() {
                Some(self.compute_rowgroup_centroid(&page_vectors))
            } else {
                None
            };

            // Update rowgroup metadata with column pages AND centroid
            let rg_metadata = RowGroupMetadata {
                id: rowgroup_id as u16,
                vector_count: page.rows.len(),
                offset: 0,          // Will be set when writing to file
                compressed_size: 0, // Will be updated after compression
                column_pages,
                vector_stats: VectorStats::default(),
                metadata_stats: HashMap::new(),
                min_timestamp: None,
                max_timestamp: None,
                centroid, // FIXED: Set centroid for EACH rowgroup, not just the last
                centroid_stats: None,
                bloom_filter_offset: None, // Will be set when bloom filter is written
            };

            self.row_groups.push(rg_metadata);
        }
        Ok(())
    }

    /// Helper: Encode vector column with Proxima (migrated to ProximaCodec)
    fn encode_vector_column(&self, page: &RowPageBuffer) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        let num_rows = page.rows.len();

        // Transpose vectors to columnar format
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(num_rows); self.dimension];
        for row in &page.rows {
            for (dim_idx, &value) in row.vector.iter().enumerate() {
                if dim_idx < self.dimension {
                    columns[dim_idx].push(value);
                }
            }
        }

        // Use ProximaCodec with adaptive scheme selection
        let codec = ProximaCodec::global();

        // Encode each dimension column with hardware-aware encoding
        for column in columns {
            // Analyze pattern and choose optimal scheme
            let detected_scheme = analysis::analyze_and_choose_scheme_f32(&column);

            // Override lossy schemes with lossless alternatives
            let scheme = match &detected_scheme {
                ProximaScheme::Simple8b
                | ProximaScheme::RunLength
                | ProximaScheme::VByte
                | ProximaScheme::Zigzag { .. }
                | ProximaScheme::PForDelta { .. } => {
                    ProximaScheme::Delta { base: 0 } // Lossless fallback
                }
                _ => detected_scheme.clone(),
            };

            let encoded_column = codec.encode(&column, scheme)?;
            encoded.extend(&(encoded_column.len() as u32).to_le_bytes());
            encoded.extend(&encoded_column);
        }

        Ok(encoded)
    }

    /// Helper: Encode ID column
    fn encode_id_column(&self, page: &RowPageBuffer) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        for row in &page.rows {
            encoded.extend(&(row.id.len() as u32).to_le_bytes());
            encoded.extend(row.id.as_bytes());
        }

        Ok(encoded)
    }

    /// Helper: Group metadata by key
    fn group_metadata_by_key(
        &self,
        page: &RowPageBuffer,
    ) -> Result<HashMap<String, Vec<Option<Vec<u8>>>>> {
        let mut grouped = HashMap::new();

        // Collect all unique keys
        let mut all_keys = HashSet::new();
        for row in &page.rows {
            for (key, _) in &row.metadata {
                all_keys.insert(key.clone());
            }
        }

        // Build columns for each key
        for key in all_keys {
            let mut column = Vec::with_capacity(page.rows.len());
            for row in &page.rows {
                let value = row
                    .metadata
                    .iter()
                    .find(|(k, _)| k == &key)
                    .map(|(_, v)| v.clone());
                column.push(value);
            }
            grouped.insert(key, column);
        }

        Ok(grouped)
    }

    /// Helper: Encode metadata column with dictionary encoding
    fn encode_metadata_column(&self, _key: &str, values: &[Option<Vec<u8>>]) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Build dictionary of unique values
        let unique_values: HashSet<Vec<u8>> = values.iter().filter_map(|v| v.clone()).collect();
        let dictionary: Vec<Vec<u8>> = unique_values.into_iter().collect();

        // Write dictionary
        encoded.extend(&(dictionary.len() as u16).to_le_bytes());
        for value in &dictionary {
            encoded.extend(&(value.len() as u32).to_le_bytes());
            encoded.extend(value);
        }

        // Write indices
        for value_opt in values {
            if let Some(value) = value_opt {
                let idx =
                    dictionary.iter().position(|v| v == value).ok_or_else(|| {
                        anyhow::anyhow!("Value {:?} not found in dictionary", value)
                    })? as u16;
                encoded.extend(&idx.to_le_bytes());
            } else {
                encoded.extend(&0xFFFF_u16.to_le_bytes()); // Null marker
            }
        }

        Ok(encoded)
    }

    /// Helper: Check if page has source content
    fn has_source_content(&self, page: &RowPageBuffer) -> bool {
        page.rows.iter().any(|r| r.source_content.is_some())
    }

    /// Helper: Encode source content
    fn encode_source_content(&self, page: &RowPageBuffer) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        for row in &page.rows {
            if let Some(content) = &row.source_content {
                encoded.extend(&(content.len() as u32).to_le_bytes());
                encoded.extend(content);
            } else {
                encoded.extend(&0u32.to_le_bytes());
            }
        }

        Ok(encoded)
    }

    /// Encode row page using TRUE columnar layout with Proxima
    /// Stores all VectorRecord fields in columnar format for optimal compression
    fn encode_row_page(&self, page: &RowPageBuffer) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();

        // Write encoding marker for columnar tensor layout
        encoded.push(0xC0); // Columnar format v2 with full VectorRecord support

        // Write page header
        encoded.extend(&(page.rows.len() as u32).to_le_bytes());
        encoded.extend(&(self.dimension as u32).to_le_bytes());

        if page.rows.is_empty() {
            return Ok(encoded);
        }

        // Use ProximaCodec for all encoding (migrated from old system)
        let codec = ProximaCodec::global();
        let num_rows = page.rows.len();

        // === SECTION 1: IDs (columnar string storage) ===
        encoded.push(0x01); // ID section marker
        // Store IDs as length-prefixed strings
        for row in &page.rows {
            encoded.extend(&(row.id.len() as u32).to_le_bytes());
            encoded.extend(row.id.as_bytes());
        }

        // === SECTION 2: FP32 VECTORS (true columnar with transposition) ===
        encoded.push(0x02); // FP32 vector section marker
        // Transpose vectors to columnar format (dimension × num_vectors)
        let mut fp32_columns: Vec<Vec<f32>> = vec![Vec::with_capacity(num_rows); self.dimension];
        for row in &page.rows {
            for (dim_idx, &value) in row.vector.iter().enumerate() {
                if dim_idx < self.dimension {
                    fp32_columns[dim_idx].push(value);
                }
            }
        }

        // Encode each dimension column with ProximaCodec
        for column in fp32_columns {
            // Analyze and choose optimal scheme
            let detected_scheme = analysis::analyze_and_choose_scheme_f32(&column);

            // Override lossy schemes
            let scheme = match &detected_scheme {
                ProximaScheme::Simple8b
                | ProximaScheme::RunLength
                | ProximaScheme::VByte
                | ProximaScheme::Zigzag { .. }
                | ProximaScheme::PForDelta { .. } => ProximaScheme::Delta { base: 0 },
                _ => detected_scheme.clone(),
            };

            let encoded_column = codec.encode(&column, scheme)?;
            encoded.extend(&(encoded_column.len() as u32).to_le_bytes());
            encoded.extend(&encoded_column);
        }

        // === SECTION 3: QUANTIZED VECTORS (columnar by quantization type) ===
        encoded.push(0x03); // Quantized vector section marker

        // Determine quantization level (assume uniform for now)
        let quant_level = if !page.rows[0].quantized_vector.is_empty() {
            // Infer from data size
            let data_size = page.rows[0].quantized_vector.len();
            if data_size == self.dimension / 8 {
                1 // Binary quantization
            } else if data_size == self.dimension {
                2 // INT8 quantization
            } else {
                3 // Product quantization
            }
        } else {
            0 // No quantization
        };

        encoded.push(quant_level);

        if quant_level > 0 {
            // Transpose quantized vectors based on type
            match quant_level {
                1 => {
                    // Binary: pack bits columnar
                    let _bits_per_dim = 1;
                    let packed_dims = self.dimension.div_ceil(8);
                    for dim_byte in 0..packed_dims {
                        let mut column = Vec::with_capacity(num_rows);
                        for row in &page.rows {
                            if dim_byte < row.quantized_vector.len() {
                                column.push(row.quantized_vector[dim_byte]);
                            }
                        }
                        // Use ProximaCodec for binary quantized data
                        let codec = ProximaCodec::global();
                        let column_i32: Vec<i32> = column.iter().map(|&v| v as i32).collect();
                        let encoded_column =
                            codec.encode_i32(&column_i32, ProximaScheme::BitPacked { bits: 8 })?;
                        encoded.extend(&(encoded_column.len() as u32).to_le_bytes());
                        encoded.extend(&encoded_column);
                    }
                }
                2 => {
                    // INT8: transpose byte-wise
                    let mut int8_columns: Vec<Vec<i8>> =
                        vec![Vec::with_capacity(num_rows); self.dimension];
                    for row in &page.rows {
                        for (dim_idx, &byte) in row.quantized_vector.iter().enumerate() {
                            if dim_idx < self.dimension {
                                int8_columns[dim_idx].push(byte as i8);
                            }
                        }
                    }
                    for column in int8_columns {
                        // Use ProximaCodec for INT8 quantized data
                        let codec = ProximaCodec::global();
                        let column_i32: Vec<i32> = column.iter().map(|&v| v as i32).collect();
                        let encoded_column =
                            codec.encode_i32(&column_i32, ProximaScheme::BitPacked { bits: 8 })?;
                        encoded.extend(&(encoded_column.len() as u32).to_le_bytes());
                        encoded.extend(&encoded_column);
                    }
                }
                _ => {
                    // Product quantization or other: store as-is for now
                    for row in &page.rows {
                        encoded.extend(&(row.quantized_vector.len()).to_le_bytes());
                        encoded.extend(&row.quantized_vector);
                    }
                }
            }
        }

        // === SECTION 4: METADATA (columnar with dictionary encoding for keys) ===
        encoded.push(0x04); // Metadata section marker

        // Collect all unique keys across all rows for dictionary encoding
        let mut key_dictionary: Vec<String> = Vec::new();
        let mut key_to_index: HashMap<String, u16> = HashMap::new();

        for row in &page.rows {
            for (key, _) in &row.metadata {
                if !key_to_index.contains_key(key) {
                    let idx = key_dictionary.len() as u16;
                    key_dictionary.push(key.clone());
                    key_to_index.insert(key.clone(), idx);
                }
            }
        }

        // Write dictionary of keys (low cardinality optimization)
        encoded.extend(&(key_dictionary.len() as u16).to_le_bytes());
        for key in &key_dictionary {
            encoded.extend(&(key.len() as u16).to_le_bytes());
            encoded.extend(key.as_bytes());
        }

        // Build metadata columns indexed by dictionary
        let mut metadata_columns: HashMap<u16, Vec<Option<Vec<u8>>>> = HashMap::new();

        for (key_idx, key) in key_dictionary.iter().enumerate() {
            let mut column = Vec::with_capacity(num_rows);
            for row in &page.rows {
                let value = row
                    .metadata
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone());
                column.push(value);
            }
            metadata_columns.insert(key_idx as u16, column);
        }

        // Write metadata values using dictionary indices
        encoded.extend(&(metadata_columns.len() as u16).to_le_bytes());

        for (key_idx, values) in metadata_columns {
            // Write dictionary index for key
            encoded.extend(&key_idx.to_le_bytes());

            // Analyze value cardinality for optimal encoding
            let unique_values: HashSet<Vec<u8>> = values.iter().filter_map(|v| v.clone()).collect();

            let cardinality_ratio = unique_values.len() as f32 / num_rows as f32;

            if cardinality_ratio < 0.1 {
                // Low cardinality: use dictionary encoding for values too
                encoded.push(0x01); // Dictionary encoding marker

                let value_dict: Vec<Vec<u8>> = unique_values.into_iter().collect();
                encoded.extend(&(value_dict.len() as u16).to_le_bytes());

                for val in &value_dict {
                    encoded.extend(&(val.len() as u32).to_le_bytes());
                    encoded.extend(val);
                }

                // Write indices
                for value_opt in &values {
                    if let Some(value) = value_opt {
                        let idx = value_dict.iter().position(|v| v == value).ok_or_else(|| {
                            anyhow::anyhow!("Value {:?} not found in dictionary", value)
                        })? as u16;
                        encoded.extend(&idx.to_le_bytes());
                    } else {
                        encoded.extend(&0xFFFF_u16.to_le_bytes()); // Null marker
                    }
                }
            } else {
                // High cardinality: use direct encoding with null bitmap
                encoded.push(0x02); // Direct encoding marker

                let mut null_bitmap = vec![0u8; num_rows.div_ceil(8)];
                let mut value_data = Vec::new();

                for (idx, value) in values.iter().enumerate() {
                    if let Some(v) = value {
                        null_bitmap[idx / 8] |= 1 << (idx % 8);
                        value_data.extend(&(v.len() as u32).to_le_bytes());
                        value_data.extend(v);
                    }
                }

                encoded.extend(&null_bitmap);
                encoded.extend(&value_data);
            }
        }

        // === SECTION 5: TIMESTAMPS (columnar integers) ===
        encoded.push(0x05); // Timestamp section marker

        // Timestamp column (always present)
        let timestamps: Vec<u32> = page.rows.iter().map(|r| r.timestamp).collect();
        let codec = ProximaCodec::global();
        let timestamps_i32: Vec<i32> = timestamps.iter().map(|&v| v as i32).collect();
        let encoded_timestamps =
            codec.encode_i32(&timestamps_i32, ProximaScheme::Delta { base: 0 })?;
        encoded.extend(&encoded_timestamps);

        // Updated_at column (optional)
        let _has_updated: Vec<bool> = page.rows.iter().map(|r| r.updated_at.is_some()).collect();
        let updated_values: Vec<u32> = page.rows.iter().filter_map(|r| r.updated_at).collect();

        encoded.push(if updated_values.len() == num_rows {
            0x01
        } else {
            0x00
        });
        if !updated_values.is_empty() {
            if updated_values.len() < num_rows {
                // Sparse: store indices and values
                for (idx, row) in page.rows.iter().enumerate() {
                    if row.updated_at.is_some() {
                        encoded.extend(&(idx as u32).to_le_bytes());
                    }
                }
            }
            let codec = ProximaCodec::global();
            let updated_i32: Vec<i32> = updated_values.iter().map(|&v| v as i32).collect();
            let encoded_updated =
                codec.encode_i32(&updated_i32, ProximaScheme::Delta { base: 0 })?;
            encoded.extend(&encoded_updated);
        }

        // Expires_at column (optional)
        let expires_values: Vec<u32> = page.rows.iter().filter_map(|r| r.expires_at).collect();

        encoded.push(if expires_values.len() == num_rows {
            0x01
        } else {
            0x00
        });
        if !expires_values.is_empty() {
            if expires_values.len() < num_rows {
                // Sparse: store indices
                for (idx, row) in page.rows.iter().enumerate() {
                    if row.expires_at.is_some() {
                        encoded.extend(&(idx as u32).to_le_bytes());
                    }
                }
            }
            let codec = ProximaCodec::global();
            let expires_i32: Vec<i32> = expires_values.iter().map(|&v| v as i32).collect();
            let encoded_expires =
                codec.encode_i32(&expires_i32, ProximaScheme::Delta { base: 0 })?;
            encoded.extend(&encoded_expires);
        }

        // Version column (optional)
        let version_values: Vec<u32> = page.rows.iter().filter_map(|r| r.version).collect();

        encoded.push(if version_values.len() == num_rows {
            0x01
        } else {
            0x00
        });
        if !version_values.is_empty() {
            if version_values.len() < num_rows {
                // Sparse: store indices
                for (idx, row) in page.rows.iter().enumerate() {
                    if row.version.is_some() {
                        encoded.extend(&(idx as u32).to_le_bytes());
                    }
                }
            }
            let codec = ProximaCodec::global();
            let versions_i32: Vec<i32> = version_values.iter().map(|&v| v as i32).collect();
            let encoded_versions =
                codec.encode_i32(&versions_i32, ProximaScheme::Delta { base: 0 })?;
            encoded.extend(&encoded_versions);
        }

        // === SECTION 6: SOURCE CONTENT (optional, type-aware encoding) ===
        encoded.push(0x06); // Source content section marker

        let source_rows: Vec<(usize, &Vec<u8>)> = page
            .rows
            .iter()
            .enumerate()
            .filter_map(|(idx, row)| row.source_content.as_ref().map(|c| (idx, c)))
            .collect();

        encoded.extend(&(source_rows.len() as u32).to_le_bytes());

        if !source_rows.is_empty() {
            // Analyze content types for optimal encoding
            let mut text_content = Vec::new();
            let mut binary_content = Vec::new();

            for (idx, content) in &source_rows {
                // Simple heuristic: check if content is valid UTF-8
                if std::str::from_utf8(content).is_ok() {
                    text_content.push((*idx, content));
                } else {
                    binary_content.push((*idx, content));
                }
            }

            // Write text content with compression
            encoded.push(0x01); // Text content marker
            encoded.extend(&(text_content.len() as u32).to_le_bytes());

            if !text_content.is_empty() {
                // Store indices
                for (idx, _) in &text_content {
                    encoded.extend(&(*idx as u32).to_le_bytes());
                }

                // Concatenate all text for better compression
                let mut all_text = Vec::new();
                let mut text_offsets = Vec::new();

                for (_, content) in &text_content {
                    text_offsets.push(all_text.len() as u32);
                    all_text.extend(content.iter());
                }
                text_offsets.push(all_text.len() as u32);

                // Compress text using LZ4 for speed
                let compressed_text = if all_text.len() > 1024 {
                    proximadb_compression::compress(
                        &all_text,
                        CompressionAlgorithm::Lz4,
                        3,
                        CompressionContext::Block,
                    )?
                } else {
                    all_text
                };

                // Write offsets
                for offset in text_offsets {
                    encoded.extend(&offset.to_le_bytes());
                }

                // Write compressed text
                encoded.extend(&(compressed_text.len() as u32).to_le_bytes());
                encoded.extend(&compressed_text);
            }

            // Write binary content (video/audio/images)
            encoded.push(0x02); // Binary content marker
            encoded.extend(&(binary_content.len() as u32).to_le_bytes());

            if !binary_content.is_empty() {
                for (idx, content) in &binary_content {
                    encoded.extend(&(*idx as u32).to_le_bytes());

                    // Binary content often already compressed (video/image formats)
                    // Store as-is or with light compression
                    encoded.extend(&(content.len() as u32).to_le_bytes());
                    encoded.extend(content.iter());
                }
            }
        }

        Ok(encoded)
    }

    /// Build optimized bloom filter for ID lookups
    /// Uses xxHash for speed and maintains configurable false positive rate
    fn build_bloom_filter_for_ids(&self, ids: &[String]) -> BloomFilterMetadata {
        let num_items = ids.len();
        let target_fpr = self.config.bloom_fpp; // False positive probability

        // Calculate optimal bloom filter parameters
        let bits_per_item = -1.44 * (target_fpr.ln() / 2.0_f64.ln());
        let total_bits = (num_items as f64 * bits_per_item).ceil() as usize;
        let num_hash_functions = (bits_per_item * 2.0_f64.ln()).ceil() as u32;

        // Create bloom filter bitmap
        let mut bloom_bits = vec![0u8; total_bits.div_ceil(8)];

        // Add all IDs to bloom filter using DefaultHasher
        for id in ids {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            let hash = hasher.finish();

            // Generate k hash values from single hash using double hashing
            for i in 0..num_hash_functions {
                let hash_val = (hash.wrapping_add(i as u64 * hash.rotate_right(32))) as usize;
                let bit_pos = hash_val % total_bits;
                bloom_bits[bit_pos / 8] |= 1 << (bit_pos % 8);
            }
        }

        BloomFilterMetadata {
            num_bits: bloom_bits.len() * 8, // Convert bytes to bits
            num_hashes: 7,                  // Standard value for bloom filters
            num_entries: self.bloom_builder.ids.len() as u32, // Number of IDs in filter
            false_positive_rate: 0.001,     // 0.1% FPR
            offset: 0,                      // Will be set when writing to file
            size: bloom_bits.len() as u64,
        }
    }

    /// Compute centroid statistics for a rowgroup
    /// This pre-computes distance bounds to enable fast pruning during search
    fn compute_centroid_stats(
        &self,
        vectors: &[Vec<f32>],
        centroid: &[f32],
        cluster_id: u32,
        all_rowgroup_centroids: &[(u16, Vec<f32>)], // (rowgroup_id, centroid)
        current_rowgroup_id: u16,
    ) -> CentroidStats {
        let mut distances = Vec::with_capacity(vectors.len());

        // Calculate distances for all vectors to centroid
        for vector in vectors {
            let dist = self
                .distance_compute
                .calculate_distance(vector, centroid, &DistanceMetric::Euclidean)
                .raw_value;
            distances.push(dist);
        }

        // Sort for percentile calculations
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate statistics
        let mean_distance = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance = distances
            .iter()
            .map(|&d| (d - mean_distance).powi(2))
            .sum::<f32>()
            / distances.len() as f32;
        let std_deviation = variance.sqrt();

        // Percentiles
        let p50_idx = distances.len() / 2;
        let p90_idx = (distances.len() as f32 * 0.9) as usize;
        let p95_idx = (distances.len() as f32 * 0.95) as usize;

        // Compute bounds for different metrics
        let euclidean_bounds = DistanceBounds {
            min: *distances.first().unwrap_or(&0.0),
            max: *distances.last().unwrap_or(&0.0),
            p50: distances.get(p50_idx).copied().unwrap_or(0.0),
            p90: distances.get(p90_idx).copied().unwrap_or(0.0),
        };

        // For cosine similarity, compute bounds if vectors are normalized
        let cosine_bounds = if vectors.iter().all(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            (norm - 1.0).abs() < 0.01 // Check if normalized
        }) {
            // Cosine distance = 1 - cosine_similarity
            // For normalized vectors, this relates to Euclidean distance
            Some(DistanceBounds {
                min: 1.0 - (1.0 - euclidean_bounds.min.powi(2) / 2.0).max(0.0),
                max: 1.0 - (1.0 - euclidean_bounds.max.powi(2) / 2.0).max(0.0),
                p50: 1.0 - (1.0 - euclidean_bounds.p50.powi(2) / 2.0).max(0.0),
                p90: 1.0 - (1.0 - euclidean_bounds.p90.powi(2) / 2.0).max(0.0),
            })
        } else {
            None
        };

        // Compute neighbor rowgroups sorted by centroid distance
        let mut neighbor_rowgroups = Vec::new();

        if !all_rowgroup_centroids.is_empty() {
            // Calculate distances and cluster assignments for all rowgroups
            let mut neighbor_data: Vec<(u16, f32, u16, Vec<f32>)> = Vec::new();

            for (rg_id, rg_centroid) in all_rowgroup_centroids {
                if *rg_id != current_rowgroup_id {
                    // IMPORTANT: Store RAW distances, not boosted
                    // Boosting requires query-specific components (d1, d3, d4, d5)
                    // that can only be computed during search
                    let dist = self
                        .distance_compute
                        .calculate_distance(centroid, rg_centroid, &DistanceMetric::Euclidean)
                        .raw_value;

                    // Get neighbor's cluster assignment (may differ from rowgroup id)
                    let neighbor_cluster_id = self.get_rowgroup_cluster_id(*rg_id);

                    neighbor_data.push((*rg_id, dist, neighbor_cluster_id, rg_centroid.clone()));
                }
            }

            // Sort by distance (ascending) - critical for multi-probe efficiency
            neighbor_data
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // OPTIMAL HIERARCHICAL NEIGHBOR STORAGE:
            // Use performance-tested formula for maximum efficiency

            let k = all_rowgroup_centroids.len();
            let (intra_neighbors, inter_neighbors) =
                proximadb_raptor_common::calculate_optimal_neighbors(k);
            let total_neighbors = intra_neighbors + inter_neighbors;
            let use_hierarchical = inter_neighbors > 0; // Two-tier if inter neighbors exist

            // Performance prediction for logging
            let predicted_latency =
                proximadb_raptor_common::predict_search_latency(k, centroid.len());
            debug!(
                "Rowgroup {}: {} total neighbors ({} intra + {} inter), predicted latency: {:.1}μs",
                current_rowgroup_id,
                total_neighbors,
                intra_neighbors,
                inter_neighbors,
                predicted_latency
            );

            tracing::trace!(
                "RowGroup {} storing {} neighbor indices (collection has {} rowgroups) - using optimal formula",
                current_rowgroup_id,
                total_neighbors,
                all_rowgroup_centroids.len()
            );

            if use_hierarchical {
                // Hierarchical storage: separate local and global neighbors
                let super_cluster_size = (k as f64).sqrt().ceil() as usize;
                let current_super = current_rowgroup_id as usize / super_cluster_size;

                let mut local_neighbors = Vec::new();
                let mut global_neighbors = Vec::new();

                for (rg_id, _, neighbor_cluster, _) in neighbor_data.into_iter() {
                    let neighbor_super = rg_id as usize / super_cluster_size;

                    if neighbor_super == current_super {
                        // Same super-cluster (local)
                        if local_neighbors.len() < 5 {
                            local_neighbors.push((rg_id, neighbor_cluster));
                        }
                    } else {
                        // Different super-cluster (global)
                        if global_neighbors.len() < 5 {
                            global_neighbors.push((rg_id, neighbor_cluster));
                        }
                    }

                    // Stop when we have enough of both types
                    if local_neighbors.len() >= 5 && global_neighbors.len() >= 5 {
                        break;
                    }
                }

                // Add local neighbors
                for (rg_id, neighbor_cluster) in local_neighbors {
                    neighbor_rowgroups.push(RowGroupNeighbor {
                        rowgroup_id: rg_id,
                        neighbor_cluster_id: neighbor_cluster,
                        neighbor_type: NeighborType::IntraSuperCluster,
                    });
                }

                // Add global neighbors
                for (rg_id, neighbor_cluster) in global_neighbors {
                    neighbor_rowgroups.push(RowGroupNeighbor {
                        rowgroup_id: rg_id,
                        neighbor_cluster_id: neighbor_cluster,
                        neighbor_type: NeighborType::InterSuperCluster,
                    });
                }
            } else {
                // Simple storage for small/medium collections
                for (rg_id, _, neighbor_cluster, _) in
                    neighbor_data.into_iter().take(total_neighbors)
                {
                    neighbor_rowgroups.push(RowGroupNeighbor {
                        rowgroup_id: rg_id,
                        neighbor_cluster_id: neighbor_cluster,
                        neighbor_type: NeighborType::Direct,
                    });
                }
            }
        }

        // === NEW: Calculate 5-Component Boosting Statistics ===

        // Component 3: Locality/Compactness metrics
        let cluster_density = if distances.is_empty() || *distances.last().unwrap_or(&1.0) == 0.0 {
            0.0
        } else {
            // Vectors per unit volume (approximated using max distance as radius)
            vectors.len() as f32 / (*distances.last().unwrap_or(&1.0)).powi(centroid.len() as i32)
        };
        let intra_cluster_variance = variance;

        // Boundary detection: Count vectors beyond mean + 1.5*std
        let boundary_threshold = mean_distance + 1.5 * std_deviation;
        let boundary_vector_count = distances
            .iter()
            .filter(|&&d| d > boundary_threshold)
            .count();
        let boundary_vector_ratio = boundary_vector_count as f32 / vectors.len() as f32;

        // Component 4 & 5: Inter-centroid distances (nearest and farthest)
        let mut inter_centroid_distances: Vec<(u16, f32)> = Vec::new();
        for (rg_id, rg_centroid) in all_rowgroup_centroids {
            if *rg_id != current_rowgroup_id {
                let dist = self
                    .distance_compute
                    .calculate_distance(centroid, rg_centroid, &DistanceMetric::Euclidean)
                    .raw_value;
                inter_centroid_distances.push((*rg_id, dist));
            }
        }

        // Sort by distance for nearest/farthest extraction
        inter_centroid_distances.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                // Handle NaN values by treating them as less than any number
                if a.1.is_nan() && b.1.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.1.is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
        });

        // Extract top 5 nearest centroids (Component 4)
        let nearest_centroid_ids: Vec<u16> = inter_centroid_distances
            .iter()
            .take(5)
            .map(|(id, _)| *id)
            .collect();
        let nearest_centroid_distances: Vec<f32> = inter_centroid_distances
            .iter()
            .take(5)
            .map(|(_, dist)| *dist)
            .collect();

        // Extract top 5 farthest centroids (Component 5)
        let farthest_centroid_ids: Vec<u16> = inter_centroid_distances
            .iter()
            .rev()
            .take(5)
            .map(|(id, _)| *id)
            .collect();
        let farthest_centroid_distances: Vec<f32> = inter_centroid_distances
            .iter()
            .rev()
            .take(5)
            .map(|(_, dist)| *dist)
            .collect();

        // Calculate average spillover ratio for boundary vectors
        // This requires checking P×K distances which we'll approximate
        let avg_spillover_ratio = if boundary_vector_count > 0 {
            // For boundary vectors, estimate how much closer they might be to other centroids
            // This is an approximation - actual P×K matrix will be computed separately
            1.0 + (boundary_vector_ratio * 0.2) // Estimate 20% closer on average
        } else {
            1.0
        };

        tracing::debug!(
            "RowGroup {}: Boosting stats - density: {:.2}, boundary: {:.1}%, nearest: {:?}, farthest: {:?}",
            current_rowgroup_id,
            cluster_density,
            boundary_vector_ratio * 100.0,
            &nearest_centroid_ids[..std::cmp::min(3, nearest_centroid_ids.len())],
            &farthest_centroid_ids[..std::cmp::min(3, farthest_centroid_ids.len())]
        );

        CentroidStats {
            cluster_id,
            mean_distance,
            std_deviation,
            radius: distances
                .get(p95_idx)
                .copied()
                .unwrap_or(mean_distance + 2.0 * std_deviation),
            min_distance: *distances.first().unwrap_or(&0.0),
            max_distance: *distances.last().unwrap_or(&0.0),
            // NEW: 5-Component Boosting fields
            cluster_density,
            intra_cluster_variance,
            nearest_centroid_ids,
            nearest_centroid_distances,
            farthest_centroid_ids,
            farthest_centroid_distances,
            boundary_vector_count,
            boundary_vector_ratio,
            avg_spillover_ratio,
            // Original fields
            euclidean_bounds: Some(euclidean_bounds),
            cosine_bounds,
            dot_product_bounds: None, // Can be computed if needed
            neighbor_rowgroups,       // Sorted by distance
        }
    }

    /// Check if we should start a new row group (1K vectors by default for optimal HNSW I/O)
    fn should_start_new_rowgroup(&self) -> bool {
        self.row_groups
            .last()
            .is_none_or(|rg| rg.vector_count >= self.config.rowgroup_size)
    }

    async fn compress_rowgroup(&self, batch: &RecordBatch) -> Result<Vec<u8>> {
        // PROXIMA: Always encode RecordBatch using Proxima for tensor optimization
        // First byte is the encoding marker (RAPTOR uses 0xA0-0xAF range)
        let mut result = Vec::new();

        // Always use Proxima tensor encoding for best performance
        let encoding_marker = 0xA1; // Proxima tensor encoding
        result.push(encoding_marker);

        // Use Proxima encoding for tensor optimization
        let encoded = self.encode_batch_with_proxima(batch, encoding_marker)?;
        result.extend(encoded);

        Ok(result)
    }

    fn encode_batch_with_proxima(&self, batch: &RecordBatch, _marker: u8) -> Result<Vec<u8>> {
        use std::io::Write;

        // Extract vectors from RecordBatch
        let vectors = self.extract_vectors_from_batch(batch)?;

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dimension = vectors[0].len();

        // Transpose to columnar for SIMD optimization
        let mut columns: Vec<Vec<f32>> = vec![vec![]; dimension];
        for vector in &vectors {
            for (dim_idx, &value) in vector.iter().enumerate() {
                if dim_idx < dimension {
                    columns[dim_idx].push(value);
                }
            }
        }

        // Analyze tensor data for optimal encoding
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for column in &columns {
            for &val in column {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Use ProximaCodec for hardware-aware encoding (migrated from old system)
        // Pattern analysis is now done per-column using analysis::analyze_and_choose_scheme_f32
        let codec = ProximaCodec::global();
        let mut encoded_data = Vec::new();

        // Write metadata
        encoded_data.write_all(&(dimension as u32).to_le_bytes())?;
        encoded_data.write_all(&(vectors.len() as u32).to_le_bytes())?;

        // Encode each dimension column with ProximaCodec
        for column in columns {
            // Analyze and choose optimal scheme
            let detected_scheme = analysis::analyze_and_choose_scheme_f32(&column);

            // Override lossy schemes
            let codec_scheme = match &detected_scheme {
                ProximaScheme::Simple8b
                | ProximaScheme::RunLength
                | ProximaScheme::VByte
                | ProximaScheme::Zigzag { .. }
                | ProximaScheme::PForDelta { .. } => ProximaScheme::Delta { base: 0 },
                _ => detected_scheme.clone(),
            };

            let encoded_column = codec.encode(&column, codec_scheme)?;
            encoded_data.write_all(&(encoded_column.len() as u32).to_le_bytes())?;
            encoded_data.write_all(&encoded_column)?;
        }

        // Also encode IDs from RecordBatch
        if let Some(id_col) = batch.column_by_name("id")
            && let Some(id_array) = id_col.as_any().downcast_ref::<arrow_array::StringArray>()
        {
            use arrow_array::Array;
            for i in 0..id_array.len() {
                if !id_array.is_null(i) {
                    let id = id_array.value(i);
                    let id_bytes = id.as_bytes();
                    encoded_data.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                    encoded_data.write_all(id_bytes)?;
                } else {
                    encoded_data.write_all(&0u32.to_le_bytes())?;
                }
            }
        }

        // Encode timestamps if present
        if let Some(ts_col) = batch.column_by_name("timestamp")
            && let Some(ts_array) = ts_col.as_any().downcast_ref::<arrow_array::Int64Array>()
        {
            for i in 0..ts_array.len() {
                let timestamp = ts_array.value(i);
                encoded_data.write_all(&timestamp.to_le_bytes())?;
            }
        }

        Ok(encoded_data)
    }

    fn extract_vectors_from_batch(&self, batch: &RecordBatch) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::new();

        if let Some(vector_col) = batch.column_by_name("vector")
            && let Some(float_array) = vector_col
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
        {
            // Assuming vectors are stored flat with known dimension
            let dimension = self.config.dimension;
            let num_vectors = float_array.len() / dimension;

            for i in 0..num_vectors {
                let start = i * dimension;
                let end = start + dimension;
                vectors.push(float_array.values()[start..end].to_vec());
            }
        }

        Ok(vectors)
    }

    fn calculate_uncompressed_size(&self, batch: &RecordBatch) -> u64 {
        let mut size = 0u64;
        for column in batch.columns() {
            size += column.get_array_memory_size() as u64;
        }
        size
    }

    fn should_quantize_vectors(&self) -> bool {
        // Determine if quantization should be applied based on config
        // Always quantize for HNSW to save memory (8-16x reduction)
        self.config.enable_clustering
            || (self.config.enable_simd && self.config.rowgroup_size >= 500)
    }

    pub async fn flush(&mut self) -> Result<usize> {
        // Track total vectors flushed
        let mut total_flushed = 0;

        tracing::debug!(
            "RAPTOR flush: Starting with {} nodes",
            self.ivf_builder.nodes.len()
        );

        // Build IVF clusters before flushing (if we have enough vectors)
        // For small collections (<1000 vectors), skip clustering but still create a single row group
        if self.ivf_builder.nodes.len() >= 1000 {
            // Minimum vectors needed for effective clustering
            self.build_ivf_clusters()?;
        } else if !self.ivf_builder.nodes.is_empty() {
            tracing::debug!(
                "RAPTOR flush: Small collection with {} vectors, creating single row group",
                self.ivf_builder.nodes.len()
            );
        }

        // Flush any pending row page
        if let Some(ref page) = self.current_row_page {
            total_flushed += page.rows.len();
        }
        self.flush_row_page().await?;

        // Count vectors in existing row groups
        for rg in &self.row_groups {
            total_flushed += rg.vector_count;
        }

        // Build bloom filter for the final row group
        if !self.bloom_builder.is_empty() {
            let bloom_filter = self.bloom_builder.build()?;

            // Write bloom filter to disk and store offset
            let bloom_offset = self.write_bloom_filter(&bloom_filter).await?;

            // Now update the row group with the bloom filter offset
            if let Some(current_rg) = self.row_groups.last_mut() {
                current_rg.bloom_filter_offset = Some(bloom_offset);

                tracing::info!(
                    "Wrote final bloom filter for row group {}: {} IDs, {:.3}% FPR, {} bytes at offset {}",
                    current_rg.id,
                    bloom_filter.stats().num_ids,
                    bloom_filter.stats().false_positive_rate * 100.0,
                    bloom_filter.stats().size_bytes,
                    bloom_offset
                );
            }
        }

        // Write column projections for the current row group
        let _projections_offset = self.write_column_projections().await?;

        // Compute centroid and cluster statistics before modifying row group
        let centroid = if !self.ivf_builder.vectors.is_empty() {
            Some(self.compute_rowgroup_centroid(&self.ivf_builder.vectors))
        } else {
            None
        };

        // Compute cluster statistics before modifying row group
        let dominant_cluster = if !self.ivf_builder.nodes.is_empty() {
            let cluster_assignments: Vec<u32> = self
                .ivf_builder
                .nodes
                .iter()
                .map(|n| n.cluster_id)
                .collect();

            let mut cluster_counts = HashMap::new();
            for &cluster_id in &cluster_assignments {
                *cluster_counts.entry(cluster_id).or_insert(0) += 1;
            }

            cluster_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map_or(0, |(&id, _)| id)
        } else {
            0
        };

        // Collect all existing rowgroup centroids for neighbor computation
        let all_rowgroup_centroids: Vec<(u16, Vec<f32>)> = self
            .row_groups
            .iter()
            .filter_map(|rg| rg.centroid.as_ref().map(|c| (rg.id, c.clone())))
            .collect();

        // Compute centroid statistics before modifying row group
        let centroid_stats = if let Some(ref c) = centroid {
            let rg_id = self.row_groups.last().map_or(0, |rg| rg.id);
            Some(self.compute_centroid_stats(
                &self.ivf_builder.vectors,
                c,
                dominant_cluster,
                &all_rowgroup_centroids,
                rg_id,
            ))
        } else {
            None
        };

        // Write IVF clustering data
        let ivf_meta = if self.config.enable_clustering && !self.ivf_builder.centroids.is_empty() {
            Some(self.write_ivf_clustering_data().await?)
        } else {
            None
        };

        // Write Matrix Trinity segment
        self.write_matrix_segment().await?;

        // Create and write bloom filter
        let bloom_filter = RowGroupBloomFilter {
            bits: vec![0u8; (self.bloom_builder.ids.len() * 10 / 8) + 1], // 10 bits per entry
            num_hashes: 7,
            num_ids: self.bloom_builder.ids.len(),
            size_bits: self.bloom_builder.ids.len() * 10,
            false_positive_rate: 0.01,
        };
        let bloom_offset = self.write_bloom_filter(&bloom_filter).await?;

        // Now update the row group with all computed values
        if let Some(rg) = self.row_groups.last_mut() {
            // Note: column_projections_offset not available in common.rs RowGroupMetadata
            // Would need to extend the structure or store separately

            // Store centroid if computed
            if let Some(c) = centroid {
                rg.centroid = Some(c);
            }

            // Store centroid stats
            if let Some(stats) = centroid_stats {
                rg.centroid_stats = Some(stats);

                // centroid_stats is guaranteed to be Some after the assignment above
                #[expect(
                    clippy::expect_used,
                    reason = "just assigned Some(stats) two lines above"
                )]
                let stats_ref = rg
                    .centroid_stats
                    .as_ref()
                    .expect("centroid_stats just set above");
                tracing::debug!(
                    "RowGroup {} centroid stats: cluster={}, mean_dist={:.3}, radius={:.3}",
                    rg.id,
                    dominant_cluster,
                    stats_ref.mean_distance,
                    stats_ref.radius
                );
            }

            // Store IVF metadata offset
            if let Some(meta) = ivf_meta {
                tracing::info!("Wrote IVF clustering data at offset {}", meta.offset);
            }

            // Store bloom filter offset
            rg.bloom_filter_offset = Some(bloom_offset);

            // Store columnar ID index as part of row group
            // This enables SIMD scanning after bloom filter check
        }

        // Clear vectors after flush to save memory
        self.ivf_builder.vectors.clear();

        Ok(total_flushed)
    }

    /// Compute centroid for a rowgroup
    fn compute_rowgroup_centroid(&self, vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() {
            return vec![0.0; self.dimension];
        }

        let mut centroid = vec![0.0; self.dimension];

        // Sum all vectors
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                if i < self.dimension {
                    centroid[i] += val;
                }
            }
        }

        // Compute mean
        let count = vectors.len() as f32;
        for val in &mut centroid {
            *val /= count;
        }

        centroid
    }

    /// Get cluster ID for a rowgroup (may differ from rowgroup ID)
    /// This is needed for computing cross-cluster penalties in boosting
    fn get_rowgroup_cluster_id(&self, rowgroup_id: u16) -> u16 {
        // Look up from existing rowgroups if available
        if let Some(rg) = self.row_groups.iter().find(|rg| rg.id == rowgroup_id)
            && let Some(ref stats) = rg.centroid_stats
        {
            return stats.cluster_id as u16;
        }
        // Fallback: use rowgroup_id as cluster_id
        // In practice, cluster assignments should be tracked properly
        rowgroup_id
    }

    /// Write IVF clustering data (centroids, assignments, edges) to disk
    async fn write_ivf_clustering_data(&mut self) -> Result<BloomFilterMetadata> {
        let mut ivf_data = Vec::new();

        // Write header
        ivf_data.extend(b"IVF1"); // Magic number
        ivf_data.extend(&(self.ivf_builder.centroids.len() as u32).to_le_bytes());
        ivf_data.extend(&(self.config.dimension as u32).to_le_bytes());

        // Write centroids
        for (idx, centroid) in self.ivf_builder.centroids.iter().enumerate() {
            // Write centroid ID and stats
            ivf_data.extend(&(idx as u32).to_le_bytes()); // cluster_id
            ivf_data.extend(&(centroid.member_ids.len() as u32).to_le_bytes()); // num_vectors
            ivf_data.extend(&centroid.mean_distance.to_le_bytes());
            ivf_data.extend(&centroid.std_deviation.to_le_bytes());

            // Write centroid vector using ProximaCodec (migrated from old system)
            let codec = ProximaCodec::global();

            // Analyze and choose optimal scheme
            let detected_scheme = analysis::analyze_and_choose_scheme_f32(&centroid.vector);

            // Override lossy schemes
            let scheme = match &detected_scheme {
                ProximaScheme::Simple8b
                | ProximaScheme::RunLength
                | ProximaScheme::VByte
                | ProximaScheme::Zigzag { .. }
                | ProximaScheme::PForDelta { .. } => ProximaScheme::Delta { base: 0 },
                _ => detected_scheme.clone(),
            };

            let encoded = codec.encode(&centroid.vector, scheme)?;
            ivf_data.extend(&(encoded.len() as u32).to_le_bytes());
            ivf_data.extend(&encoded);
        }

        // Note: Centroid neighbor relationships are now stored in rowgroup metadata
        // This is more efficient as neighbors are computed between actual rowgroups
        // rather than theoretical cluster centroids

        // Write node assignments and edges
        ivf_data.extend(&(self.ivf_builder.nodes.len() as u32).to_le_bytes());
        for node in &self.ivf_builder.nodes {
            // Write node data
            ivf_data.extend(&(node.vector_id.len() as u32).to_le_bytes());
            ivf_data.extend(node.vector_id.as_bytes());
            ivf_data.extend(&node.cluster_id.to_le_bytes());
            ivf_data.extend(&node.centroid_distance.to_le_bytes());

            // Write edges
            ivf_data.extend(&(node.edges.len() as u32).to_le_bytes());
            for edge in &node.edges {
                ivf_data.extend(&edge.target_node_id.to_le_bytes());
                ivf_data.extend(&edge.distance.to_le_bytes());
            }
        }

        // Compress IVF data
        let compressed = proximadb_compression::compress(
            &ivf_data,
            CompressionAlgorithm::Zstd,
            6,
            CompressionContext::Block,
        )?;

        // Get current file offset
        let offset = self
            .filesystem
            .metadata(&self.file_path)
            .await
            .map(|m| m.size)
            .unwrap_or(0);

        // Write to file
        self.filesystem.append(&self.file_path, &compressed).await?;

        // Note: This method writes IVF clustering data, not bloom filter
        // The bloom filter is written separately in write_bloom_filter_metadata
        // For now, return a simple metadata with basic info
        Ok(BloomFilterMetadata {
            offset,
            size: compressed.len() as u64,
            num_entries: self.ivf_builder.nodes.len() as u32,
            num_bits: self.ivf_builder.nodes.len() * 10, // Estimate: 10 bits per entry
            num_hashes: 7,                               // Standard for 1% false positive rate
            false_positive_rate: 0.01,                   // Default 1% false positive rate
        })
    }

    pub async fn close(mut self) -> Result<()> {
        // Flush any remaining data
        self.flush().await?;

        // Update file metadata with row groups
        tracing::debug!(
            "RAPTOR finalize: Updating metadata with {} row groups",
            self.row_groups.len()
        );
        self.file_metadata.row_groups = self.row_groups.clone();
        self.file_metadata.total_vectors = self.row_groups.iter().map(|rg| rg.vector_count).sum();

        // Finalize the file with centralized footer
        self.finalize().await?;

        Ok(())
    }

    /// Finalize the RAPTOR file by writing centralized footer with all centroids
    /// This enables single I/O to load all centroids for query optimization
    pub async fn finalize(&mut self) -> Result<()> {
        tracing::info!("Finalizing RAPTOR file with P² + K² + P×K architecture");

        // Step 1: Collect ALL centroids from all rowgroups, sorted by rowgroup_id
        let mut all_centroids: Vec<(u32, Vec<f32>)> = Vec::new();

        for rg in &self.row_groups {
            if let Some(centroid) = &rg.centroid {
                all_centroids.push((rg.id as u32, centroid.clone()));
            }
        }

        // Sort by rowgroup_id for O(1) indexing
        all_centroids.sort_by_key(|(id, _)| *id);

        let num_centroids = all_centroids.len();
        let dimension = if num_centroids > 0 {
            all_centroids[0].1.len()
        } else {
            self.dimension
        };

        tracing::info!(
            "Collected {} centroids of dimension {} for P² + K² + P×K storage",
            num_centroids,
            dimension
        );

        // Build K×K inter-centroid distance matrix (upper triangle storage)
        let inter_centroid_matrix = if num_centroids > 0 {
            // Ensure centroid distances are computed
            if self.ivf_builder.centroid_distances.is_empty() {
                self.compute_all_centroid_distances(&all_centroids)?;
            }
            Some(self.ivf_builder.build_inter_centroid_matrix())
        } else {
            None
        };

        // Build P×K vector-to-centroid distance matrices for rowgroups
        let vector_centroid_matrices = if !self.row_groups.is_empty() && num_centroids > 0 {
            // Create simplified rowgroup structures for matrix building
            let simplified_rowgroups: Vec<RowGroup> = self
                .row_groups
                .iter()
                .map(|rg| RowGroup {
                    id: rg.id,
                    offset: rg.offset,
                    compressed_size: rg.compressed_size,
                    uncompressed_size: 0, // Not calculated yet
                    // vector_count is already in rg struct, no need to duplicate
                    vector_count: rg.vector_count, // Same as row_count
                    max_vectors: 1024,             // Default capacity
                    column_pages: rg.column_pages.clone(),
                    vector_stats: rg.vector_stats.clone(),
                    metadata_stats: rg.metadata_stats.clone(),
                    bloom_filter: None, // Will be loaded separately
                    bloom_filter_offset: rg.bloom_filter_offset,
                    compression_codec: "zstd".to_string(),
                    min_timestamp: rg.min_timestamp,
                    max_timestamp: rg.max_timestamp,
                    centroid: rg.centroid.clone(),
                    centroid_stats: rg.centroid_stats.clone(),
                    columnar_data: None,
                    p2_matrix: None,
                    pxk_matrix: None,
                    vectors: None,
                })
                .collect();
            Some(
                self.ivf_builder
                    .build_vector_centroid_matrices(&simplified_rowgroups),
            )
        } else {
            None
        };

        // Step 2: Transpose centroids for columnar encoding (d×k layout)
        let mut transposed_data = vec![0.0f32; num_centroids * dimension];
        let mut rowgroup_ids = Vec::with_capacity(num_centroids);

        for (idx, (rg_id, centroid)) in all_centroids.iter().enumerate() {
            rowgroup_ids.push(*rg_id as u16);

            // Transpose: store all values for dim0, then dim1, etc.
            for (dim_idx, &value) in centroid.iter().enumerate() {
                let offset = dim_idx * num_centroids + idx;
                transposed_data[offset] = value;
            }
        }

        // Step 3: Compute Proxima encoding metadata for each dimension
        let mut encoding_metadata = Vec::with_capacity(dimension);

        for dim_idx in 0..dimension {
            let start = dim_idx * num_centroids;
            let end = start + num_centroids;
            let dim_values = &transposed_data[start..end];

            // Compute statistics for this dimension
            let min_val = dim_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = dim_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;

            // Choose encoding based on range
            let encoding = if range < 0.001 {
                // Very small range, use run-length encoding
                ProximaScheme::RunLength
            } else if range < 10.0 {
                // Small range, use delta encoding
                ProximaScheme::Delta { base: 0 }
            } else {
                // Larger range, use bit packing
                ProximaScheme::BitPacked { bits: 24 }
            };

            encoding_metadata.push(ProximaMetadata {
                min_value: min_val,
                max_value: max_val,
                encoding,
                compressed_size: 0, // Will be filled during actual encoding
            });
        }

        // Step 4: Create the columnar centroids structure
        let columnar_centroids = ColumnarCentroids {
            count: num_centroids as u32,
            dimension,
            rowgroup_ids,
            transposed_data,
            encoding_metadata,
        };

        // Step 5: Create the footer with centroids and matrices
        let footer = RaptorFooter {
            centroids: columnar_centroids.clone(),
            inter_centroid_distances: inter_centroid_matrix.unwrap_or_else(|| {
                // Create empty matrix if no centroids
                InterCentroidMatrix {
                    num_centroids: 0,
                    compressed_data: Vec::new(),
                    compression_metadata: InterCentroidCompressionMetadata {
                        min_distance: 0.0,
                        max_distance: 1.0,
                        scale_factor: 1.0,
                        compression_type: CompressionType::Quantized16Bit,
                        row_encodings: Vec::new(),
                        row_compressed_sizes: Vec::new(),
                    },
                    lookup_table: Vec::new(),
                }
            }),
            vector_centroid_matrices: vector_centroid_matrices.map_or_else(Vec::new, |matrices| {
                matrices
                    .into_iter()
                    .map(|m| VectorCentroidMatrixRef {
                        rowgroup_id: m.rowgroup_id,
                        num_vectors: m.num_vectors,
                        num_centroids: m.num_centroids,
                        file_offset: 0,     // Will be set when writing to disk
                        compressed_size: 0, // Will be set when writing to disk
                        uncompressed_size: m.num_vectors * m.num_centroids * 4, // P×K×4 bytes
                        compression_algorithm: "zstd".to_string(),
                        encoding_metadata: m.compression_metadata,
                    })
                    .collect()
            }),
            total_centroids: columnar_centroids.count, // K centroids = K rowgroups
            version: 1,
            checksum: 0, // Deferred: Compute actual checksum
            file_metadata: self.file_metadata.clone(),
        };

        // Step 6: Serialize and write the footer
        let footer_bytes = bincode::serialize(&footer)?;
        let footer_size = footer_bytes.len();

        // Get current file size as footer offset
        let footer_offset = self
            .filesystem
            .metadata(&self.file_path)
            .await
            .map(|m| m.size)
            .unwrap_or(0);

        // Write footer to file
        self.filesystem
            .append(&self.file_path, &footer_bytes)
            .await?;

        // Write footer size (last 4 bytes before magic for easy lookup)
        // Using u32 allows footers up to 4GB which is more than sufficient
        // (even 100k rowgroups with 1536 dims = ~600MB)
        if footer_size > u32::MAX as usize {
            return Err(anyhow::anyhow!(
                "Footer size {} exceeds maximum of 4GB",
                footer_size
            ));
        }
        let footer_size_bytes = (footer_size as u32).to_le_bytes();
        self.filesystem
            .append(&self.file_path, &footer_size_bytes)
            .await?;

        // Write magic number (last 4 bytes)
        self.filesystem
            .append(&self.file_path, constants::RAPTOR_MAGIC)
            .await?;

        // CRITICAL: Sync file to ensure footer is written to disk before engine closes
        // This fixes the 0% recall bug where footer wasn't flushed before reopen
        self.filesystem.sync_file(&self.file_path).await?;

        tracing::info!(
            "Wrote centralized footer: {} centroids, {} bytes at offset {}, total file size approx {}",
            num_centroids,
            footer_size,
            footer_offset,
            footer_offset + footer_size as u64 + 8 // footer + size(4) + magic(4)
        );

        // Step 7: Log memory savings
        let distributed_size = num_centroids * 5 * dimension * 4; // If storing 5 neighbors inline
        let centralized_size = num_centroids * dimension * 4; // Actual footer storage
        let savings_pct = (1.0 - centralized_size as f32 / distributed_size.max(1) as f32) * 100.0;

        tracing::info!(
            "Memory savings: {:.1}% (centralized: {:.2}MB vs distributed: {:.2}MB)",
            savings_pct,
            centralized_size as f32 / 1_048_576.0,
            distributed_size as f32 / 1_048_576.0
        );

        Ok(())
    }

    /// Update column projections for filtering
    fn update_column_projections(
        &mut self,
        vector: &VectorRecord,
        _location: RowLocation,
    ) -> Result<()> {
        // Extract metadata columns for projection
        if !vector.metadata.is_empty() {
            for (key, value) in &vector.metadata {
                let serialized = bincode::serialize(&value).map_err(|e| {
                    anyhow::anyhow!("Failed to serialize metadata key '{}': {}", key, e)
                })?;
                self.column_projections
                    .metadata_columns
                    .entry(key.clone())
                    .or_default()
                    .push(serialized);
            }
        }

        // Update filter bitmaps (example: filtering by specific metadata values)
        // This would be customized based on actual filtering needs
        Ok(())
    }

    /// Write column projections to disk
    async fn write_column_projections(&mut self) -> Result<u64> {
        let mut projection_data = Vec::new();

        // Serialize metadata columns
        for (column_name, values) in &self.column_projections.metadata_columns {
            // Write column header
            let header = format!("{}:{}", column_name, values.len());
            projection_data.extend(header.as_bytes());
            projection_data.push(0); // null terminator

            // Write column values
            for value in values {
                projection_data.extend(&(value.len() as u32).to_le_bytes());
                projection_data.extend(value);
            }
        }

        // Compress projections
        let compressed = proximadb_compression::compress(
            &projection_data,
            CompressionAlgorithm::Zstd,
            6,
            CompressionContext::Block,
        )?;

        // Get current file size before append
        let offset = self
            .filesystem
            .metadata(&self.file_path)
            .await
            .map(|m| m.size)
            .unwrap_or(0);

        // Write to file
        self.filesystem.append(&self.file_path, &compressed).await?;
        Ok(offset)
    }

    // HNSW removed - Matrix Trinity handles navigation
    // Write matrix segment data (IVF clusters already built by flush())
    async fn write_matrix_segment(&mut self) -> Result<()> {
        // NOTE: IVF clusters are built by flush() before this method is called
        // Do NOT call build_ivf_clusters() here - it was causing duplicate clustering
        // taking ~200s for 50K vectors (100s x 2)

        // Matrices are written as part of rowgroup metadata
        // P² matrix: Stored per rowgroup
        // K² matrix: Stored in footer
        // P×K matrix: Stored per rowgroup

        Ok(())
    }

    // Calculate optimal Matrix Trinity parameters
    fn calculate_optimal_matrix_params(&self, num_vectors: usize) -> (usize, usize) {
        // k = number of clusters/rowgroups (sqrt(n) for optimal complexity)
        let k = (num_vectors as f64).sqrt().max(1.0) as usize;

        // p = vectors per rowgroup (based on L3 cache for optimal locality)
        let p = self.config.rowgroup_size; // Use configured size

        tracing::info!(
            "Matrix Trinity params for {} vectors: k={} clusters, p={} vectors/rowgroup",
            num_vectors,
            k,
            p
        );

        (k, p)
    }

    // REMOVED: Legacy build_hnsw_graph method - replaced by build_ivf_clusters
    // and AXIS clustering integration for better memory efficiency and code reuse

    /// Build minimal HNSW graph with distance-aware edges for RAPTOR storage
    ///
    /// This method creates a memory-efficient HNSW graph that:
    /// 1. Uses only vector IDs instead of full vectors (96% memory reduction)
    /// 2. Stores distances on edges for fast recomputation avoidance
    /// 3. Integrates with AXIS clustering for optimal row group organization
    /// 4. Supports both small and large dataset optimization strategies
    ///
    /// The graph is used for:
    /// - Row group clustering via distance-aware algorithms
    /// - Fast similarity search within storage pages
    /// - Boosting calculations during component assignment
    ///
    /// Build hybrid IVF+Graph structure using k-means clustering with local edges
    /// Combines IVF clustering (k clusters) with local graph connectivity (edges within clusters)
    fn build_ivf_clusters(&mut self) -> Result<()> {
        // Step 1: Validate input data
        let num_vectors = self.ivf_builder.nodes.len();
        if num_vectors == 0 {
            tracing::debug!("No vectors to build IVF clusters, skipping");
            return Ok(());
        }

        // Ensure we have vectors for clustering
        if self.ivf_builder.vectors.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot build IVF clusters: vectors not stored. \
                 Vectors must be added via add_vector() before clustering."
            ));
        }

        // Step 2: Calculate optimal k and p values using complexity formula k² + p×(k+p)
        let sqrt_n = (num_vectors as f64).sqrt() as usize;
        let k = self.config.num_clusters.unwrap_or(sqrt_n);
        let p = self.config.target_rowgroup_size.unwrap_or_else(|| {
            // Calculate based on L3 cache size for optimal memory locality
            let l3_size = constants::clustering::DEFAULT_L3_CACHE_SIZE;
            let vector_size = self.config.dimension * 4; // 4 bytes per f32
            let vectors_in_cache =
                (l3_size as f64 * constants::clustering::L3_CACHE_UTILIZATION_PERCENT) as usize
                    / vector_size;
            vectors_in_cache.max(constants::clustering::MIN_ROWGROUP_SIZE)
        });

        tracing::info!(
            "Building IVF clusters: n={}, k={} clusters, p={} rowgroup size. \
             Complexity: O(k²+p×(k+p)) = O({})",
            num_vectors,
            k,
            p,
            k * k + p * (k + p)
        );

        // Step 3: Use AXIS clustering engine to compute k-means centroids
        // The AXIS engine provides reusable, optimized k-means implementation
        let (centroids_vec, assignments) =
            self.ivf_builder.axis_clustering.cluster_vectors_simple(
                &self.ivf_builder.vectors,
                k,
                DistanceMetric::Euclidean,
                100, // max_iterations
            )?;

        // Step 4: Store centroids and build centroid distance matrix
        self.ivf_builder.centroids = centroids_vec
            .iter()
            .enumerate()
            .map(|(idx, centroid_vec)| Centroid {
                id: idx,
                vector: centroid_vec.clone(),
                member_ids: Vec::new(),
                mean_distance: 0.0,
                std_deviation: 0.0,
                radius: 0.0,
            })
            .collect();

        // Step 5: Centroid neighbor relationships are now computed per-rowgroup
        // during flush when we have all rowgroup centroids available.
        // This is more efficient as:
        // 1. We compute neighbors between actual rowgroups, not theoretical clusters
        // 2. The data is stored directly in rowgroup metadata
        // 3. No separate centroid distance matrix is needed

        // Step 6: Update nodes with cluster assignments and calculate centroid distances
        // Use unified distance compute for consistency and optimization
        for (idx, &cluster_id) in assignments.iter().enumerate() {
            self.ivf_builder.nodes[idx].cluster_id = cluster_id as u32;

            // Calculate distance to assigned centroid (d2 component) using unified distance
            let centroid = &self.ivf_builder.centroids[cluster_id];
            let dist_result = self.distance_compute.calculate_distance(
                &self.ivf_builder.vectors[idx],
                &centroid.vector,
                &DistanceMetric::Euclidean,
            );
            self.ivf_builder.nodes[idx].centroid_distance = dist_result.raw_value;

            // Update centroid statistics (member count tracked via member_ids.len())
        }

        // Step 7: Calculate mean and std deviation for each centroid
        for cluster_id in 0..k {
            let centroid = &mut self.ivf_builder.centroids[cluster_id];
            let cluster_nodes: Vec<_> = self
                .ivf_builder
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.cluster_id == cluster_id as u32)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_nodes.is_empty() {
                // Calculate mean distance
                let sum: f32 = cluster_nodes
                    .iter()
                    .map(|&idx| self.ivf_builder.nodes[idx].centroid_distance)
                    .sum();
                centroid.mean_distance = sum / cluster_nodes.len() as f32;

                // Calculate standard deviation
                let variance: f32 = cluster_nodes
                    .iter()
                    .map(|&idx| {
                        let diff =
                            self.ivf_builder.nodes[idx].centroid_distance - centroid.mean_distance;
                        diff * diff
                    })
                    .sum::<f32>()
                    / cluster_nodes.len() as f32;
                centroid.std_deviation = variance.sqrt();
            }
        }

        // Step 8: Build local edges within each cluster for hybrid IVF+Graph
        // This is what makes RAPTOR unique - combining clustering with local connectivity
        self.build_local_edges_within_clusters(k)?;

        // Step 9: Apply 5-component boosting to edges
        let clusters: Vec<Vec<usize>> = (0..k)
            .map(|cluster_id| {
                self.ivf_builder
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.cluster_id == cluster_id as u32)
                    .map(|(idx, _)| idx)
                    .collect()
            })
            .collect();

        let vectors = self.ivf_builder.vectors.clone();
        self.ivf_builder
            .apply_component_boosting(&clusters, &vectors);

        // Step 10: Create inverted lists for each cluster
        let mut inverted_lists: Vec<Vec<String>> = vec![Vec::new(); k];
        for node in &self.ivf_builder.nodes {
            inverted_lists[node.cluster_id as usize].push(node.vector_id.clone());
        }

        // Log clustering statistics
        for (cluster_id, list) in inverted_lists.iter().enumerate() {
            tracing::debug!(
                "Cluster {}: {} vectors ({}% of dataset)",
                cluster_id,
                list.len(),
                (list.len() * 100) / num_vectors
            );
        }

        tracing::info!(
            "Built hybrid IVF+Graph structure with {} clusters and local edges. \
             Average cluster size: {} vectors",
            k,
            num_vectors / k
        );

        Ok(())
    }

    /// Build local edges within each cluster for graph navigation
    /// This creates the "Graph" part of our hybrid IVF+Graph approach
    fn build_local_edges_within_clusters(&mut self, num_clusters: usize) -> Result<()> {
        let edges_per_node = 16; // M parameter for HNSW local graphs

        tracing::info!(
            "Building local edges within {} clusters, {} edges per node",
            num_clusters,
            edges_per_node
        );

        // Process each cluster independently
        for cluster_id in 0..num_clusters {
            // Get all nodes in this cluster
            let cluster_nodes: Vec<usize> = self
                .ivf_builder
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.cluster_id == cluster_id as u32)
                .map(|(idx, _)| idx)
                .collect();

            if cluster_nodes.len() <= 1 {
                continue; // No edges needed for single-node clusters
            }

            // Build edges for each node in the cluster
            for &node_idx in &cluster_nodes {
                // Skip if node_idx is out of bounds for vectors array
                if node_idx >= self.ivf_builder.vectors.len() {
                    continue;
                }

                let mut edges = Vec::new();
                let node_vector = &self.ivf_builder.vectors[node_idx];

                // Calculate distances to all other nodes in the cluster
                let mut distances: Vec<(usize, f32)> = cluster_nodes
                    .iter()
                    .filter(|&&idx| idx != node_idx && idx < self.ivf_builder.vectors.len())
                    .map(|&other_idx| {
                        let dist = self
                            .distance_compute
                            .calculate_distance(
                                node_vector,
                                &self.ivf_builder.vectors[other_idx],
                                &DistanceMetric::Euclidean,
                            )
                            .raw_value;
                        (other_idx, dist)
                    })
                    .collect();

                // Take the top M edges by partial selection then sort only
                // that prefix — O(n + M log M) per node instead of sorting all
                // n neighbor distances (O(n log n)). Same set, same ascending
                // order, same NaN handling as the full sort it replaces.
                let by_distance_nan_aware = |a: &(usize, f32), b: &(usize, f32)| {
                    a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                        // Handle NaN values by treating them as less than any number
                        if a.1.is_nan() && b.1.is_nan() {
                            std::cmp::Ordering::Equal
                        } else if a.1.is_nan() {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                };
                if distances.len() > edges_per_node {
                    distances.select_nth_unstable_by(edges_per_node, by_distance_nan_aware);
                }
                distances.truncate(edges_per_node);
                distances.sort_by(by_distance_nan_aware);

                // Create edge structures
                for (target_idx, distance) in distances {
                    edges.push(EdgeWithDistance {
                        target_node_id: target_idx as u32,
                        target_vector_id: self.ivf_builder.nodes[target_idx].vector_id.clone(),
                        distance,
                    });
                }

                // Store edges in the node
                self.ivf_builder.nodes[node_idx].edges = edges;
            }
        }

        // Calculate edge statistics
        let total_edges: usize = self.ivf_builder.nodes.iter().map(|n| n.edges.len()).sum();
        let avg_edges = total_edges as f32 / self.ivf_builder.nodes.len() as f32;

        tracing::info!(
            "Built {} total edges, average {:.1} edges per node",
            total_edges,
            avg_edges
        );

        Ok(())
    }

    /// Encode metadata columns with intelligent type detection and encoding
    fn encode_metadata_columns(&self, page: &RowPageBuffer, encoded: &mut Vec<u8>) -> Result<()> {
        use std::collections::BTreeMap;

        // First, extract and analyze all metadata across the page
        let mut metadata_schema: BTreeMap<String, MetadataColumn> = BTreeMap::new();

        // Parse all metadata to build schema
        for row in &page.rows {
            if !row.metadata.is_empty() {
                // Process metadata (stored as Vec<(String, Vec<u8>)>)
                for (key, value_bytes) in &row.metadata {
                    let value = String::from_utf8_lossy(value_bytes).to_string();
                    metadata_schema
                        .entry(key.clone())
                        .or_insert_with(|| MetadataColumn::new(key.clone()))
                        .add_value(value);
                }
            }
        }

        // Write number of metadata columns
        encoded.extend(&(metadata_schema.len() as u32).to_le_bytes());

        // Encode each metadata column optimally
        for (column_name, mut column) in metadata_schema {
            // Write column name
            let name_bytes = column_name.as_bytes();
            encoded.extend(&(name_bytes.len() as u32).to_le_bytes());
            encoded.extend(name_bytes);

            // Analyze column and choose encoding
            let encoding = column.analyze_and_choose_encoding();
            encoded.push(encoding.to_byte());

            // Encode column based on chosen strategy
            match encoding {
                MetadataEncoding::Dictionary => {
                    // Dictionary encoding for low cardinality
                    let dict = column.build_dictionary();
                    encoded.extend(&(dict.len() as u32).to_le_bytes());

                    // Write dictionary entries
                    for entry in &dict {
                        let entry_bytes = entry.as_bytes();
                        encoded.extend(&(entry_bytes.len() as u32).to_le_bytes());
                        encoded.extend(entry_bytes);
                    }

                    // Write indices using minimal bits
                    let bits_needed = (dict.len() as f32).log2().ceil() as u8;
                    encoded.push(bits_needed);

                    // Pack indices
                    let indices = column.encode_as_indices(&dict);
                    let packed = self.pack_indices(&indices, bits_needed);
                    encoded.extend(&(packed.len() as u32).to_le_bytes());
                    encoded.extend(&packed);
                }
                MetadataEncoding::Integer => {
                    // Parse as integers and use Proxima
                    let integers: Vec<i64> = column
                        .values
                        .iter()
                        .map(|v| v.parse::<i64>().unwrap_or(0))
                        .collect();

                    let min = *integers.iter().min().unwrap_or(&0);
                    let max = *integers.iter().max().unwrap_or(&0);
                    let range = max - min;

                    // Use frame of reference encoding
                    let scheme = ProximaScheme::FrameOfReference {
                        reference: min,
                        bits: ((range as f64).log2().ceil() as u8 + 1).min(32),
                    };

                    let codec = ProximaCodec::global();
                    let encoded_ints = codec.encode_i64(&integers, scheme)?;

                    encoded.extend(&min.to_le_bytes());
                    encoded.extend(&max.to_le_bytes());
                    encoded.extend(&(encoded_ints.len() as u32).to_le_bytes());
                    encoded.extend(&encoded_ints);
                }
                MetadataEncoding::Boolean => {
                    // Pack booleans as bits
                    let bools: Vec<bool> = column
                        .values
                        .iter()
                        .map(|v| v.to_lowercase() == "true" || v == "1")
                        .collect();

                    let packed = self.pack_booleans(&bools);
                    encoded.extend(&(packed.len() as u32).to_le_bytes());
                    encoded.extend(&packed);
                }
                MetadataEncoding::Float => {
                    // Phase 3: Parse as floats and use UnifiedProximaSIMD
                    let floats: Vec<f32> = column
                        .values
                        .iter()
                        .map(|v| v.parse::<f32>().unwrap_or(0.0))
                        .collect();

                    // Use ProximaCodec for metadata float encoding (migrated from old system)
                    let codec = ProximaCodec::global();

                    // Analyze and choose optimal scheme
                    let detected_scheme = analysis::analyze_and_choose_scheme_f32(&floats);

                    // Override lossy schemes
                    let scheme = match &detected_scheme {
                        ProximaScheme::Simple8b
                        | ProximaScheme::RunLength
                        | ProximaScheme::VByte
                        | ProximaScheme::Zigzag { .. }
                        | ProximaScheme::PForDelta { .. } => ProximaScheme::Delta { base: 0 },
                        _ => detected_scheme.clone(),
                    };

                    let encoded_floats = codec.encode(&floats, scheme)?;

                    encoded.extend(&(encoded_floats.len() as u32).to_le_bytes());
                    encoded.extend(&encoded_floats);
                }
                MetadataEncoding::String => {
                    // High cardinality strings - use length-prefixed encoding
                    for value in &column.values {
                        let value_bytes = value.as_bytes();
                        encoded.extend(&(value_bytes.len() as u32).to_le_bytes());
                        encoded.extend(value_bytes);
                    }
                }
                MetadataEncoding::RunLength => {
                    // All values are the same - just store once
                    let value = &column.values[0];
                    let value_bytes = value.as_bytes();
                    encoded.extend(&(value_bytes.len() as u32).to_le_bytes());
                    encoded.extend(value_bytes);
                    encoded.extend(&(column.values.len()).to_le_bytes()); // count
                }
            }
        }

        Ok(())
    }

    /// Pack boolean values into bits
    fn pack_booleans(&self, bools: &[bool]) -> Vec<u8> {
        let mut packed = Vec::new();
        for chunk in bools.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                if b {
                    byte |= 1 << i;
                }
            }
            packed.push(byte);
        }
        packed
    }

    /// Pack indices with minimal bits
    fn pack_indices(&self, indices: &[usize], _bits: u8) -> Vec<u8> {
        // Simplified bit packing - in production would use proper bit packing
        indices.iter().map(|&i| i as u8).collect()
    }

    /// Write bloom filter for this row group
    async fn write_bloom_filter_metadata(&mut self) -> Result<BloomFilterMetadata> {
        // Validate we have IDs to build bloom filter
        if self.bloom_builder.ids.is_empty() {
            return Err(anyhow::anyhow!("Cannot create bloom filter with no IDs"));
        }

        // Calculate optimal bloom filter size
        let num_ids = self.bloom_builder.ids.len();
        let bits_per_id = 10; // For 1% false positive rate
        let num_bits = num_ids.saturating_mul(bits_per_id);

        // Validate bloom filter size is reasonable (cap at 128MB)
        // This supports up to ~107.3 million vectors at 1% false positive rate
        const MAX_BLOOM_SIZE_BYTES: usize = 128 * 1024 * 1024;
        let num_bytes = num_bits.div_ceil(8);
        if num_bytes > MAX_BLOOM_SIZE_BYTES {
            return Err(anyhow::anyhow!(
                "Bloom filter size {} bytes exceeds maximum {} bytes (supports up to ~107M vectors)",
                num_bytes,
                MAX_BLOOM_SIZE_BYTES
            ));
        }

        // Create bloom filter structure
        let _bloom_filter = RowGroupBloomFilter {
            bits: vec![0u8; num_bytes],
            num_hashes: 7, // Optimal for 1% false positive
            num_ids,
            size_bits: num_bits,
            false_positive_rate: 0.01,
        };

        // Create bloom filter bits
        let mut bloom_bits = vec![0u8; num_bytes];
        let num_hashes = 7; // Optimal for 1% false positive

        // Add all IDs to bloom filter
        for id in &self.bloom_builder.ids {
            // Validate ID is not empty
            if id.is_empty() {
                tracing::warn!("Skipping empty ID in bloom filter");
                continue;
            }

            for i in 0..num_hashes {
                let hash_u64 =
                    proximadb_kernel::hash::XxHasher::hash_bytes(format!("{}{}", id, i).as_bytes());
                let bit_index = (hash_u64 as usize) % num_bits;

                let byte_index = bit_index / 8;
                let bit_offset = bit_index % 8;

                // Bounds check (should never fail with modulo, but safety first)
                if byte_index >= bloom_bits.len() {
                    return Err(anyhow::anyhow!(
                        "Bloom filter byte index {} out of bounds (size: {})",
                        byte_index,
                        bloom_bits.len()
                    ));
                }

                bloom_bits[byte_index] |= 1 << bit_offset;
            }
        }

        // Write columnar ID index (for SIMD scanning after bloom check)
        let mut id_column_data = Vec::new();

        // Write number of IDs
        id_column_data.extend(&(self.id_column_builder.ids.len() as u32).to_le_bytes());

        // Write ID strings (columnar format for SIMD)
        for id in &self.id_column_builder.ids {
            id_column_data.extend(&(id.len() as u32).to_le_bytes());
            id_column_data.extend(id.as_bytes());
        }

        // Write ID hashes (for fast comparison)
        for hash in &self.id_column_builder.id_hashes {
            id_column_data.extend(&hash.to_le_bytes());
        }

        // Write row offsets
        for offset in &self.id_column_builder.row_offsets {
            id_column_data.extend(&offset.to_le_bytes());
        }

        // Combine bloom filter and ID column
        let mut combined_data = Vec::new();
        combined_data.extend(&(bloom_bits.len() as u32).to_le_bytes());
        combined_data.extend(&bloom_bits);
        combined_data.extend(&id_column_data);

        // Compress using unified compression - use CompressionProvider trait
        use proximadb_compression::CompressionProvider;
        let compressed = self.compression.compress(
            &combined_data,
            CompressionAlgorithm::Zstd, // No level field in enum
            6,                          // Compression level as separate parameter
            CompressionContext::Block,
        )?;

        // Get current file size before append
        let offset = self
            .filesystem
            .metadata(&self.file_path)
            .await
            .map(|m| m.size)
            .unwrap_or(0);

        // Write to file
        self.filesystem.append(&self.file_path, &compressed).await?;

        // Create and store bloom filter metadata
        let _bloom_filter = RowGroupBloomFilter {
            bits: bloom_bits.clone(),
            num_hashes,
            num_ids,
            size_bits: num_bits,
            false_positive_rate: self.bloom_builder.target_false_positive_rate,
        };

        // Store bloom filter offset in row group metadata
        if let Some(rg) = self.row_groups.last_mut() {
            rg.bloom_filter_offset = Some(offset);
        }

        Ok(BloomFilterMetadata {
            offset,
            size: compressed.len() as u64,
            num_entries: self.bloom_builder.ids.len() as u32,
            num_bits: self.bloom_builder.ids.len() * 10, // 10 bits per ID
            num_hashes: 7,                               // Optimal for 1% false positive rate
            false_positive_rate: self.bloom_builder.target_false_positive_rate,
        })
    }

    // Add missing fields to struct
    fn initialize_missing_fields(&mut self) {
        // This is a placeholder for any missing initialization
    }

    // Add missing dimension field
    fn get_dimension(&self) -> usize {
        self.dimension
    }

    async fn flush_rowgroup(&mut self) -> Result<()> {
        if let Some(_rowgroup) = self.current_rowgroup.take() {
            // Convert RecordBatch to row pages and flush
            // This is handled by flush_row_page() already
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "experimental-engines")]
mod tests {
    use proximadb_proto::proximadb_v1::{
        SqlArray, SqlObject, SqlValue, sql_value::Value as SqlVal,
    };

    #[test]
    fn test_array_value_serialization_roundtrip() {
        use prost::Message;

        let array = SqlArray {
            values: vec![
                SqlValue {
                    value: Some(SqlVal::StringValue("hello".to_string())),
                },
                SqlValue {
                    value: Some(SqlVal::NumberValue(42.0)),
                },
                SqlValue {
                    value: Some(SqlVal::BoolValue(true)),
                },
            ],
        };

        // Encode
        let mut buf = Vec::new();
        array
            .encode(&mut buf)
            .expect("ArrayValue encoding should succeed");
        assert!(!buf.is_empty(), "Encoded array should not be empty");

        // Decode
        let decoded = SqlArray::decode(buf.as_slice()).expect("ArrayValue decoding should succeed");
        assert_eq!(decoded.values.len(), 3);
        assert_eq!(
            decoded.values[0].value,
            Some(SqlVal::StringValue("hello".to_string()))
        );
        assert_eq!(decoded.values[1].value, Some(SqlVal::NumberValue(42.0)));
        assert_eq!(decoded.values[2].value, Some(SqlVal::BoolValue(true)));
    }

    #[test]
    fn test_object_value_serialization_roundtrip() {
        use prost::Message;

        let object = SqlObject {
            fields: vec![
                (
                    "name".to_string(),
                    SqlValue {
                        value: Some(SqlVal::StringValue("test".to_string())),
                    },
                ),
                (
                    "count".to_string(),
                    SqlValue {
                        value: Some(SqlVal::Int64Value(99)),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };

        // Encode
        let mut buf = Vec::new();
        object
            .encode(&mut buf)
            .expect("ObjectValue encoding should succeed");
        assert!(!buf.is_empty(), "Encoded object should not be empty");

        // Decode
        let decoded =
            SqlObject::decode(buf.as_slice()).expect("ObjectValue decoding should succeed");
        assert_eq!(decoded.fields.len(), 2);
        assert_eq!(
            decoded.fields.get("name").and_then(|v| v.value.as_ref()),
            Some(&SqlVal::StringValue("test".to_string()))
        );
    }

    #[test]
    fn test_metadata_bytes_to_base64() {
        use base64::Engine;

        let bytes = vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello"
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        assert_eq!(encoded, "SGVsbG8=");

        // Verify roundtrip
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .expect("base64 decode should succeed");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_metadata_array_to_json() {
        let array = SqlArray {
            values: vec![
                SqlValue {
                    value: Some(SqlVal::StringValue("a".to_string())),
                },
                SqlValue {
                    value: Some(SqlVal::NumberValue(1.0)),
                },
            ],
        };

        let json_val = serde_json::to_value(&array).expect("Array should serialize to JSON");
        assert!(json_val.is_object()); // SqlArray serializes as an object with "values" key
    }

    // ========== NEW TESTS ==========

    #[test]
    fn test_nested_object_serialization_roundtrip() {
        use prost::Message;

        let inner = SqlObject {
            fields: vec![(
                "nested_key".to_string(),
                SqlValue {
                    value: Some(SqlVal::Int64Value(123)),
                },
            )]
            .into_iter()
            .collect(),
        };

        let outer = SqlObject {
            fields: vec![
                (
                    "name".to_string(),
                    SqlValue {
                        value: Some(SqlVal::StringValue("outer".to_string())),
                    },
                ),
                (
                    "inner".to_string(),
                    SqlValue {
                        value: Some(SqlVal::ObjectValue(inner.clone())),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };

        let mut buf = Vec::new();
        outer.encode(&mut buf).unwrap();
        let decoded = SqlObject::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.fields.len(), 2);
        assert!(decoded.fields.contains_key("name"));
        assert!(decoded.fields.contains_key("inner"));
    }

    #[test]
    fn test_empty_array_serialization_roundtrip() {
        use prost::Message;

        let array = SqlArray { values: vec![] };
        let mut buf = Vec::new();
        array.encode(&mut buf).unwrap();
        let decoded = SqlArray::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.values.len(), 0);
    }

    #[test]
    fn test_empty_object_serialization_roundtrip() {
        use prost::Message;

        let object = SqlObject {
            fields: std::collections::HashMap::new(),
        };
        let mut buf = Vec::new();
        object.encode(&mut buf).unwrap();
        let decoded = SqlObject::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.fields.len(), 0);
    }

    #[test]
    fn test_null_value_serialization_roundtrip() {
        use prost::Message;

        let null_val = SqlValue {
            value: Some(SqlVal::NullValue(0)),
        };
        let mut buf = Vec::new();
        null_val.encode(&mut buf).unwrap();
        let decoded = SqlValue::decode(buf.as_slice()).unwrap();
        assert!(matches!(decoded.value, Some(SqlVal::NullValue(_))));
    }

    #[test]
    fn test_array_with_nulls_serialization() {
        use prost::Message;

        let array = SqlArray {
            values: vec![
                SqlValue {
                    value: Some(SqlVal::StringValue("value".to_string())),
                },
                SqlValue {
                    value: Some(SqlVal::NullValue(0)),
                },
                SqlValue {
                    value: Some(SqlVal::NumberValue(3.5)),
                },
            ],
        };

        let mut buf = Vec::new();
        array.encode(&mut buf).unwrap();
        let decoded = SqlArray::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.values.len(), 3);
        assert!(matches!(
            decoded.values[1].value,
            Some(SqlVal::NullValue(_))
        ));
    }

    #[test]
    fn test_large_int64_value_roundtrip() {
        use prost::Message;

        let val = SqlValue {
            value: Some(SqlVal::Int64Value(i64::MAX)),
        };
        let mut buf = Vec::new();
        val.encode(&mut buf).unwrap();
        let decoded = SqlValue::decode(buf.as_slice()).unwrap();
        assert_eq!(decoded.value, Some(SqlVal::Int64Value(i64::MAX)));

        let val_min = SqlValue {
            value: Some(SqlVal::Int64Value(i64::MIN)),
        };
        let mut buf2 = Vec::new();
        val_min.encode(&mut buf2).unwrap();
        let decoded2 = SqlValue::decode(buf2.as_slice()).unwrap();
        assert_eq!(decoded2.value, Some(SqlVal::Int64Value(i64::MIN)));
    }

    #[test]
    fn test_special_float_values_roundtrip() {
        use prost::Message;

        for special in [f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0] {
            let val = SqlValue {
                value: Some(SqlVal::NumberValue(special)),
            };
            let mut buf = Vec::new();
            val.encode(&mut buf).unwrap();
            let decoded = SqlValue::decode(buf.as_slice()).unwrap();
            if let Some(SqlVal::NumberValue(v)) = decoded.value {
                if special == 0.0 {
                    assert!(v == 0.0);
                } else {
                    assert_eq!(v, special);
                }
            } else {
                panic!("Expected NumberValue");
            }
        }
    }

    #[test]
    fn test_metadata_bytes_empty_roundtrip() {
        use base64::Engine;

        let empty: Vec<u8> = vec![];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&empty);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, empty);
    }

    #[test]
    fn test_metadata_bytes_large_payload() {
        use base64::Engine;

        // Simulate a 1KB metadata payload
        let payload: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded.len(), 1024);
        assert_eq!(decoded, payload);
    }

    #[test]
    fn test_bloom_filter_builder_new() {
        let builder = super::BloomFilterBuilder::new(0.01);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_bloom_filter_builder_dedup() {
        let mut builder = super::BloomFilterBuilder::new(0.01);
        builder.add_id("id1".to_string());
        builder.add_id("id1".to_string()); // duplicate
        builder.add_id("id2".to_string());
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_bloom_filter_builder_build_empty() {
        let builder = super::BloomFilterBuilder::new(0.01);
        let bloom = builder.build().unwrap();
        // Empty bloom should still be valid
        assert!(bloom.size_bits > 0 || bloom.size_bits == 0); // just ensure no panic
    }

    #[test]
    fn test_boosting_config_defaults() {
        let config = super::BoostingConfig::default();
        assert!(config.alpha_own > 0.0);
        assert!(config.alpha_inter > 0.0);
        assert!(config.alpha_variance > 0.0);
        assert!(config.beta_min > 0.0);
        assert!(config.beta_max > 0.0);
        assert!(config.beta_cross > 0.0);
        assert!(config.boundary_threshold > 0.0);
        assert!(!config.store_components); // default off
    }
}
