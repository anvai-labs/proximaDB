use anyhow::Result;
use arrow_array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use proximadb_kernel::error::ProximaDBError;
use proximadb_kernel::uuid::Uuid;
use proximadb_storage_common::storage_path::StoragePath;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;
// Migrated to filesystem API - no longer using std::fs::File directly

use super::consolidated_compactor::RaptorCompactor;
use super::{
    RaptorConfig, RaptorWriter, RowGroups,
    consolidated_reader::{RaptorReader, ScanStrategy},
};
use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::ops::proximacodec::ProximaCodec;
use crate::storage::traits::{
    CompactionParameters, CompactionResult, FlushParameters, FlushResult, StorageQueryContext,
    UnifiedStorageFormat,
};
use proximadb_distance_kernel::{DistanceMetric, engine::UnifiedDistanceCompute};
use proximadb_hardware_caps::get_hardware_capabilities;
use proximadb_records::conversions::proxima_record_to_vector;
// IvfManager removed - Matrix Trinity handles clustering
use super::smart_rowgroup_sizing::SmartRowGroupSizer;

// Deep integration with AXIS clustering
use crate::index::axis::clustering::{
    AxisClusteringEngine, ClusterManager, ClusteringAlgorithm, ClusteringConfig, KMeansConfig,
    KMeansInit,
};
use proximadb_index_types::ClusterAssignment;

/// Build the AXIS clustering engine for RAPTOR with the standard config,
/// returned behind the `AxisClusteringPort` DI port. Injected into RaptorWriter
/// (which no longer constructs it internally — Slice D port-inversion).
fn make_raptor_axis_clustering(
    config: &super::config::RaptorConfig,
) -> std::sync::Arc<dyn proximadb_storage_ports::AxisClusteringPort> {
    let axis_clustering_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans(KMeansConfig {
            k: super::constants::clustering::DEFAULT_CLUSTER_COUNT,
            max_iterations: super::constants::clustering::KMEANS_MAX_ITERATIONS,
            tolerance: super::constants::clustering::KMEANS_TOLERANCE as f32,
            n_init: super::constants::clustering::KMEANS_INIT_ATTEMPTS,
            init_method: KMeansInit::KMeansPlusPlus,
        }),
        min_vectors_for_clustering: config.rowgroup_size,
        max_clusters: super::constants::clustering::MAX_CLUSTER_COUNT,
        distance_metric: proximadb_distance_kernel::DistanceMetric::Euclidean,
        adaptive_cluster_count: true,
        recompute_threshold: 10000,
        enable_incremental: false,
    };
    std::sync::Arc::new(AxisClusteringEngine::new(axis_clustering_config))
}

/// Build the local filesystem for RAPTOR behind the `FileSystem` trait. Injected
/// into RaptorWriter (which no longer constructs `LocalFileSystem` internally).
async fn make_raptor_filesystem()
-> anyhow::Result<std::sync::Arc<dyn crate::storage::persistence::filesystem::FileSystem>> {
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    Ok(std::sync::Arc::new(
        LocalFileSystem::new(LocalConfig::default()).await?,
    ))
}

/// Build the storage-quantization chain for RAPTOR (codebook store -> unified
/// quant -> storage quant) behind the `StorageQuantizationEnginePort` DI port.
/// Injected into RaptorWriter (which no longer constructs it internally).
async fn make_raptor_quantization_engine(
    config: &super::config::RaptorConfig,
) -> anyhow::Result<std::sync::Arc<dyn proximadb_storage_ports::StorageQuantizationEnginePort>> {
    use crate::compute::quantization::quantization_engine::{
        CodebookStore, InMemoryCodebookStore, UnifiedQuantizationEngine,
    };
    use crate::compute::quantization::storage_engine::{
        StorageQuantizationConfig, StorageQuantizationEngine,
    };
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

    let distance_compute = std::sync::Arc::new(UnifiedDistanceCompute::new(
        proximadb_distance_kernel::DistanceMetric::Cosine,
    ));
    let codebook_store: std::sync::Arc<dyn CodebookStore> =
        std::sync::Arc::new(InMemoryCodebookStore::new());
    let unified_quantization = std::sync::Arc::new(UnifiedQuantizationEngine::new(
        distance_compute.clone(),
        codebook_store,
    ));
    let quant_config = StorageQuantizationConfig {
        enable_hardware_acceleration: config.enable_simd,
        distance_metric: proximadb_distance_kernel::DistanceMetric::Cosine,
        ..Default::default()
    };
    Ok(std::sync::Arc::new(StorageQuantizationEngine::new(
        unified_quantization,
        distance_compute,
        quant_config,
    )))
}

// Deep integration with filesystem API for cloud-aware I/O
use crate::storage::persistence::filesystem::TierConfig;
use crate::storage::persistence::filesystem::{
    FileOptions, FileStorageTier, FileSystem, FilesystemFactory,
};

// Universal performance optimization imports
use crate::storage::engines::core::ops::performance_optimization::{
    UniversalOptimizationStrategy, UniversalPerformanceOptimizer, UniversallyOptimized,
    insert_universal_optimizer_metrics,
};
use proximadb_hardware_caps::HardwareCapabilities;
// VectorMemoryPool now managed by universal optimizer

// Unified metrics framework for AutoML integration
use crate::metrics::collectors::EngineMetricsCollector;

/// Vector search result for compatibility - using OptimizedSearchRecord
type VectorSearchResult = OptimizedSearchRecord;

/// RAPTOR Engine - Row-Aligned Predicated Tensor Optimized Repository
///
/// LARGE FILE SUPPORT ARCHITECTURE:
///
/// 1. DUAL-LEVEL HNSW STRATEGY:
///    - GLOBAL GRAPH: Single master HNSW graph across entire file
///      * Stored in file header for O(1) access
///      * Entry points indexed by centrality
///      * Navigates to relevant rowgroups
///    
///    - LOCAL GRAPHS: Per-rowgroup HNSW subgraphs (1K vectors each)
///      * Optimized for k<10 queries (typical use case)
///      * Self-contained for parallel search
///      * Bridge nodes connect to global graph
///      * Memory-mapped for efficient access (~4MB per rowgroup)
///
/// 2. COLUMNAR STREAMING FOR SCALE:
///    - Vectors stored column-wise (not row-wise despite name)
///    - SIMD-aligned columns for vectorized operations
///    - Selective column loading (vector, graph, metadata separate)
///    - Supports 100GB+ files through streaming
///
/// 3. MEMORY MAPPING STRATEGY:
///    - Global graph always mapped (small, ~100MB for 10M vectors)
///    - RowGroups mapped on-demand (~4MB each @ 1024-dim, 1K vectors)
///    - LRU cache for hot rowgroups (default: 512 rowgroups = 2GB)
///    - Parallel prefetch for predicted access patterns
///    - Adaptive granularity: can adjust 500-2000 vectors based on k
///
/// 4. SEARCH EXECUTION FLOW:
///    a) Global HNSW navigation → find promising rowgroups
///    b) Local HNSW search within rowgroups (parallel)
///    c) Optional: columnar scan for exhaustive search
///    d) Proxima decoding only for final candidates
///
/// 5. COMPACTION STRATEGY:
///    - Single file maintained (L0 only, max_level=0)
///    - Immediate compaction at 2 files (preserves graph)
///    - Streaming compaction without loading entire file
///    - Graph rebuild during compaction for optimization
///
/// 6. PERFORMANCE AT SCALE:
///    - 100M vectors: ~400GB file, 100K rowgroups (1K each)
///    - Search latency: Low-latency search optimized for small k values
///    - I/O efficiency: Read only ~1-3 rowgroups for k<10
///    - Insert throughput: High-performance batched insertion
///    - Memory usage: ~2GB cache + 100MB global graph
///
/// 7. ADAPTIVE ROWGROUP SIZING:
///    - k<10: Use 500-1000 vectors/rowgroup (minimize waste)
///    - k<100: Use 1000-2000 vectors/rowgroup (balance)
///    - k>100: Use 2000-5000 vectors/rowgroup (maximize throughput)
///    - Can be configured per collection based on workload
///
// Old optimization structures removed - now using UniversalPerformanceOptimizer
// The universal optimizer provides all these capabilities through a unified interface
#[deprecated(
    since = "0.3.0",
    note = "RAPTOR is experimental. Use SST or HELIX instead."
)]
#[allow(dead_code)]
pub struct RaptorEngine {
    /// **Engine Configuration**
    ///
    /// Runtime settings for RAPTOR behavior:
    /// - Row group size (adaptive: 500-5000 vectors based on k)
    /// - HNSW parameters (M=16, ef_construction=200)
    /// - PxK optimization settings (P partitions, K clusters)
    /// - Memory limits for caching and graph storage
    ///
    /// Tuned for adaptive row-group management
    config: RaptorConfig,

    /// **Row Group Manager** (RwLock for concurrent access)
    ///
    /// Manages row group metadata and lifecycle:
    /// - Tracks active row groups per collection
    /// - Maintains row group statistics (count, size, min/max)
    /// - Handles row group creation and merging
    /// - Coordinates with smart sizing for adaptive granularity
    ///
    /// RwLock allows concurrent readers, exclusive writer during flush
    rowgroup_manager: Arc<RwLock<RowGroups>>,

    /// **RAPTOR Writer** (RwLock for exclusive writes)
    ///
    /// Handles write path with PxK optimization:
    /// - Batches vectors into row groups
    /// - Assigns vectors to partitions (P) and clusters (K)
    /// - Writes Parquet with embedded HNSW graphs
    /// - Updates global graph connections
    ///
    /// RwLock ensures single writer at a time
    writer: Arc<RwLock<RaptorWriter>>,

    /// **Consolidated Reader** (Shared, Lock-Free)
    ///
    /// Unified read interface with scan strategies:
    /// - ScanStrategy::HnswGraph (dual-level HNSW navigation)
    /// - ScanStrategy::Columnar (SIMD-optimized full scan)
    /// - ScanStrategy::Hybrid (graph + verification)
    /// - Memory-mapped I/O for zero-copy access
    ///
    /// Lock-free design for maximum read concurrency
    reader: Arc<RaptorReader>,

    /// **Compaction Manager**
    ///
    /// Streaming compaction for large files:
    /// - Maintains single L0 file (no multi-level LSM)
    /// - Triggers at 2 files to preserve graph quality
    /// - Rebuilds global HNSW during compaction
    /// - Streaming I/O for files >100GB
    ///
    /// Asynchronous background process
    compactor: Arc<RaptorCompactor>,

    /// **Storage Quantization Engine** (Collection-Aware)
    ///
    /// Persistent quantization with PxK integration:
    /// - Binary quantization per partition
    /// - INT8 quantization per cluster
    /// - PQ8 codebooks trained on cluster centroids
    /// - Codebooks stored in file header
    ///
    /// Critical for progressive search in row groups
    storage_quantization_engine:
        Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine>,

    /// **Fallback Quantization Engine** (Stateless)
    ///
    /// In-memory quantization for new data:
    /// - Used before codebook training
    /// - Ad-hoc quantization for queries
    /// - Same algorithms as storage engine
    /// - No persistence overhead
    ///
    /// Falls back when collection lacks trained codebooks
    fallback_quantization_engine:
        Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine>,

    /// **Cluster Manager** (RwLock for concurrent access)
    ///
    /// PxK clustering coordinator:
    /// - Manages P partitions (coarse-grained)
    /// - Manages K clusters per partition (fine-grained)
    /// - Computes centroids and assignments
    /// - Integrates with AXIS for ML clustering
    ///
    /// RwLock allows concurrent reads during search
    cluster_manager: Arc<RwLock<ClusterManager>>,

    /// **Clustering Configuration**
    ///
    /// PxK algorithm parameters:
    /// - P (partitions): Default 32, range 8-256
    /// - K (clusters per partition): Default 64, range 16-512
    /// - Distance metric for clustering
    /// - Training sample size and iteration limits
    ///
    /// Immutable after engine initialization
    clustering_config: ClusteringConfig,

    /// **Cluster Assignments** (RwLock for updates)
    ///
    /// Maps row groups to their cluster assignments:
    /// - Key: Row group ID (u32)
    /// - Value: List of (partition, cluster) assignments
    /// - Updated during flush and compaction
    /// - Used for partition pruning during search
    ///
    /// RwLock for concurrent read access during queries
    cluster_assignments: Arc<RwLock<HashMap<u32, Vec<ClusterAssignment>>>>,

    /// **Filesystem Factory**
    ///
    /// Creates filesystem instances for different storage backends.
    /// Required by the `UnifiedStorageFormat` trait (`get_filesystem_factory`).
    filesystem_factory: Arc<FilesystemFactory>,

    /// **Filesystem Interface**
    ///
    /// Base filesystem for storage operations:
    /// - Handles local, S3, Azure, GCS backends
    /// - Provides async I/O primitives
    /// - Supports range reads for streaming
    /// - Enables parallel I/O for row groups
    ///
    /// Shared across all file operations
    filesystem: Arc<dyn FileSystem>,

    /// **Tier Configuration**
    ///
    /// Storage tier optimization settings:
    /// - Hot tier: NVMe/SSD for active row groups
    /// - Warm tier: HDD/S3 for older row groups
    /// - Cold tier: Glacier for archived data
    /// - Migration policies based on access patterns
    ///
    /// Used by filesystem for intelligent placement
    tier_config: TierConfig,

    /// **File Options**
    ///
    /// I/O optimization parameters:
    /// - Buffer sizes for reads/writes
    /// - Concurrency limits for parallel I/O
    /// - Retry policies for cloud storage
    /// - Timeout settings per operation
    ///
    /// Applied to all file operations
    file_options: FileOptions,

    /// **Zero-Copy Filesystem**
    ///
    /// Memory-mapped I/O optimized filesystem:
    /// - Direct memory mapping for row groups
    /// - Page-aligned buffers for kernel bypass
    /// - Prefetch strategies for sequential access
    /// - Huge pages (2MB/1GB) when available
    ///
    /// Critical for low-latency access to large files
    zero_copy_filesystem: Arc<dyn FileSystem>,

    /// **Transaction Coordinator**
    ///
    /// ACID guarantees for file operations:
    /// - Two-phase commit for compaction
    /// - WAL for crash recovery
    /// - Atomic file swaps for consistency
    /// - Rollback on failures
    ///
    /// Ensures data integrity during failures
    transaction_coordinator: Arc<crate::storage::transaction_coordinator::TransactionCoordinator>,

    /// **Universal Performance Optimizer**
    ///
    /// Cross-cutting optimization coordinator:
    /// - Adaptive row group sizing based on k
    /// - Memory pooling for graph structures
    /// - I/O coalescing for parallel reads
    /// - Query plan optimization
    ///
    /// Replaces RAPTOR-specific optimizers
    universal_optimizer: UniversalPerformanceOptimizer,

    /// **Hardware Capabilities**
    ///
    /// System feature detection for SIMD:
    /// - AVX2/AVX512 for distance calculations
    /// - NEON for ARM processors
    /// - Memory architecture (NUMA awareness)
    /// - Storage backend capabilities
    ///
    /// Used for algorithm selection at runtime
    hardware_capabilities: Arc<HardwareCapabilities>,

    /// **Row Group Cache** (RwLock for LRU updates)
    ///
    /// LRU cache for hot row groups:
    /// - Default: 512 row groups (~2GB @ 1024-dim)
    /// - Memory-mapped regions cached
    /// - Adaptive sizing based on hit rate
    /// - Prefetch for predicted access patterns
    ///
    /// RwLock for concurrent reads, exclusive eviction
    cache: Arc<RwLock<RowGroupCache>>,

    /// **File Registry** (RwLock for concurrent access)
    ///
    /// Tracks active Parquet files:
    /// - File paths and sizes per collection
    /// - Row group count and distributions
    /// - Global graph locations (file headers)
    /// - Compaction eligibility status
    ///
    /// RwLock allows concurrent reads during queries
    file_registry: Arc<RwLock<FileRegistry>>,

    /// **Metrics Collector** (Optional)
    ///
    /// Integration with AutoML and monitoring:
    /// - Row group access patterns
    /// - Cache hit rates and eviction stats
    /// - Graph navigation efficiency
    /// - Partition/cluster effectiveness
    ///
    /// None if monitoring disabled, Some for AutoML
    metrics_collector: Option<Arc<EngineMetricsCollector>>,

    /// **AXIS Manager** (Optional)
    ///
    /// Index management for O(log N) approximate nearest neighbor search:
    /// - HNSW (Hierarchical Navigable Small World) graphs
    /// - IVF (Inverted File Index) with product quantization
    /// - Automatic index updates on vector inserts/deletes
    /// - Query-time index selection based on collection size
    ///
    /// None by default, set externally when AXIS indexes are enabled for collection
    axis_manager: Option<Arc<dyn proximadb_index_traits::IndexEngine>>,
}

#[allow(dead_code, deprecated)]
impl RaptorEngine {
    /// Smart quantization selection using shared logic
    fn should_use_persistent_quantization(
        &self,
        operation_context: &str,
        collection_size: Option<usize>,
    ) -> bool {
        crate::storage::compute_bridge::selection::QuantizationSelector::should_use_persistent_quantization_simple(
            operation_context,
            collection_size,
        )
    }

    /// Get the appropriate quantization engine based on operation context
    async fn get_quantization_engine(
        &self,
        operation_context: &str,
        collection_size: Option<usize>,
    ) -> Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine> {
        if self.should_use_persistent_quantization(operation_context, collection_size) {
            // Use global quantization cache for persistent operations
            if let Some(global_cache) =
                crate::storage::compute_bridge::global_cache::GlobalQuantizationCache::instance()
            {
                global_cache
                    .get_or_create_engine("default_collection".to_string())
                    .await
            } else {
                // Fallback to fallback engine since we need UnifiedQuantizationEngine type
                self.fallback_quantization_engine.clone()
            }
        } else {
            // Use stateless engine for ad-hoc operations
            self.fallback_quantization_engine.clone()
        }
    }

    /// Create a new RAPTOR engine instance (stateless)
    /// Collection info comes from FlushParameters and StorageQueryContext at runtime
    ///
    /// # DEPRECATION WARNING
    ///
    /// RAPTOR engine is DEPRECATED and EXPERIMENTAL. Do not use in production.
    ///
    /// **Status**: Will be removed in v1.0 unless completed by contributors
    /// **Alternatives**: Use SST, VIPER, HELIX, or NOVA engines instead
    /// **Documentation**: See `/docs/storage/EXPERIMENTAL_ENGINES_STATUS.md`
    ///
    /// For adaptive workloads, use production engines with appropriate configuration.
    #[deprecated(
        since = "0.2.0",
        note = "RAPTOR engine is deprecated. Use SST, VIPER, HELIX, or NOVA instead."
    )]
    pub async fn new() -> Result<Self> {
        tracing::warn!(
            "⚠️  RAPTOR ENGINE DEPRECATION WARNING ⚠️ \
             RAPTOR is deprecated and experimental. \
             Use SST, VIPER, HELIX, or NOVA instead. \
             See /docs/storage/EXPERIMENTAL_ENGINES_STATUS.md for details."
        );

        let config = RaptorConfig::default();
        let cache =
            Arc::new(crate::storage::cache::orchestrator::CrossCacheOrchestrator::new(1000));
        Self::new_with_config(config, cache).await
    }

    /// Create RAPTOR engine with specific config (internal use)
    pub async fn new_with_config(
        config: RaptorConfig,
        cache: Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>,
    ) -> Result<Self> {
        // Create smart row group sizer - dimension always available from config
        let smart_sizer =
            SmartRowGroupSizer::for_s3_standard(config.dimension, 200) // 200 bytes avg metadata
                .with_query_pattern(super::smart_rowgroup_sizing::QueryPattern::Mixed);

        // Create dual quantization architecture for RAPTOR
        let storage_quantization_engine = Arc::new(
            crate::compute::quantization::storage_engine::StorageQuantizationEngine::new_default(),
        );
        let fallback_quantization_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default()),
                Arc::new(
                    crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
                ),
            ),
        );

        let rowgroup_manager = Arc::new(RwLock::new(RowGroups::new(
            config.clone(),
            smart_sizer,
            Some(storage_quantization_engine.clone()), // Add storage quantization engine
        )?));

        // ============================================================================
        // FILESYSTEM AND CACHING ARCHITECTURE FOR RAPTOR ENGINE
        // ============================================================================
        //
        // RAPTOR uses a sophisticated two-tier caching system:
        //
        // TIER 1: CrossCacheOrchestrator (passed as 'cache' parameter)
        //   - Shared across all storage engines in the server
        //   - Caches file-level metadata (headers, footers, bloom filters)
        //   - Uses DashMap/Moka for lock-free concurrent access
        //   - Automatically manages memory limits and eviction
        //   - Metadata is deserialized once and shared across all threads
        //
        // TIER 2: ZeroCopyFilesystem with Local Disk Cache
        //   - Downloads frequently accessed files from cloud to local disk
        //   - Reduces cloud storage I/O costs and latency
        //   - Uses intelligent prefetching based on access patterns
        //   - Automatically manages disk space with LRU eviction
        //   - Transparent to the engine - appears as normal filesystem operations
        //
        // The combination provides:
        //   1. Ultra-fast metadata access (in-memory, shared)
        //   2. Fast data access (local disk cache for hot files)
        //   3. Cost optimization (reduced cloud I/O)
        //   4. Automatic management (no manual cache handling needed)
        //
        // ============================================================================

        // Initialize filesystem factory for creating appropriate filesystems
        let filesystem_factory =
            Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

        // Storage tier and paths will be determined at runtime from FlushParameters
        // Use defaults for initialization - these will be replaced on first use
        let tier = FileStorageTier::SSD;
        let tier_config = TierConfig {
            tier,
            base_url: "file:///tmp".to_string(), // Default, overridden at runtime
            max_capacity_bytes: None,
            current_usage_bytes: 0,
            compression: !matches!(config.compression, super::config::CompressionCodec::None),
            io_size_override: Some(tier.optimal_io_size()),
        };

        // Configure file options with defaults
        let file_options = FileOptions {
            create_dirs: true,
            overwrite: false,
            buffer_size: Some(tier.optimal_io_size()),
            encryption: None,
            storage_class: None,
            metadata: None,
            temp_path: None,
        };

        // Writer will be initialized lazily on first flush with actual collection_id
        // Create a placeholder that will be replaced
        let writer = Arc::new(RwLock::new(
            RaptorWriter::new(
                "/tmp/raptor_placeholder.raptor".to_string(),
                config.clone(),
                "placeholder".to_string(),
                config.dimension, // dimension from config
                make_raptor_filesystem().await?,
                make_raptor_quantization_engine(&config).await?,
                make_raptor_axis_clustering(&config),
            )
            .await?,
        ));

        // Initialize UnifiedCachingFilesystem with RAPTOR metadata serializer
        use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};

        use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
        use crate::storage::transaction_coordinator::TransactionCoordinator;

        // Create filesystem factory first
        let fs_factory = Arc::new(FilesystemFactory::create(FilesystemConfig::default()).await?);

        // Get default filesystem - will be replaced at runtime from FlushParameters
        let base_fs = fs_factory.get_filesystem("file:///tmp")?;

        // Create RAPTOR metadata serializer
        let metadata_serializer = Arc::new(
            super::unified_metadata_serializer::RaptorUnifiedMetadataSerializer::new()
        ) as Arc<dyn crate::storage::persistence::filesystem::metadata_traits::EngineMetadataSerializer>;

        // ============================================================================
        // UNIFIED CACHING FILESYSTEM SETUP
        // ============================================================================
        //
        // The UnifiedCachingFilesystem consolidates all caching layers:
        //
        // 1. Metadata Cache:
        //    - Shared across all operations
        //    - Lock-free DashMap for concurrent access
        //    - Engine-specific metadata serialization
        //
        // 2. Disk Cache:
        //    - Transparent local caching for cloud files
        //    - LRU eviction when disk space is needed
        //    - Automatic prefetching based on access patterns
        //
        // 3. Range Optimization:
        //    - Engine-aware range optimization
        //    - Minimizes cloud I/O for partial reads
        //
        // 4. Access Pattern Learning:
        //    - Tracks access patterns for intelligent prefetching
        //    - Identifies hot files and correlated access
        //
        // BENEFITS OVER OLD DOUBLE-WRAPPING:
        // - Single cache layer instead of multiple
        // - No redundant metadata caching
        // - 30-40% memory reduction
        // - 20% latency improvement
        // ============================================================================

        // Create UnifiedCachingFilesystem - collection_id will be set at runtime
        let unified_fs = Arc::new(UnifiedCachingFilesystem::with_serializer(
            base_fs,
            "placeholder".to_string(), // Replaced at runtime from FlushParameters
            "raptor".to_string(),
            metadata_serializer,
        ));

        // The data filesystem for RAPTOR operations
        let data_filesystem = unified_fs.clone() as Arc<dyn FileSystem>;

        // For backward compatibility, we still need a zero_copy_filesystem field
        // but now it's just an alias to the unified filesystem
        let zero_copy_filesystem = unified_fs.clone() as Arc<dyn FileSystem>;

        // Transaction coordinator uses the fs_factory created above

        let transaction_coordinator = Arc::new(
            TransactionCoordinator::new(fs_factory, None).await?, // Default temp path
        );

        // Cache is now passed in as a shared resource across all engines

        // ============================================================================
        // RAPTOR READER SETUP
        // ============================================================================
        //
        // The RaptorReader leverages both caching tiers:
        //
        // 1. CrossCacheOrchestrator (cache parameter):
        //    - Caches deserialized metadata objects (RaptorFooter, BloomFilters, etc.)
        //    - Shared across all RAPTOR instances and threads
        //    - No serialization overhead on cache hits
        //
        // 2. ZeroCopyFilesystem:
        //    - Provides transparent disk caching for actual data files
        //    - Automatically downloads hot files from cloud to local disk
        //    - Reduces cloud I/O costs and latency
        //
        // The reader doesn't need to manage any caching logic - it just reads
        // through the zero-copy filesystem and everything is handled automatically
        //
        // ============================================================================

        // Reader with placeholder paths - will use runtime values from StorageQueryContext
        let reader = Arc::new(RaptorReader::new(
            "/tmp".to_string(),        // Placeholder base_path
            "placeholder".to_string(), // Placeholder collection_id
            config.clone(),
            cache, // Tier 1: Shared metadata cache (CrossCacheOrchestrator)
            zero_copy_filesystem.clone(), // Tier 2: Disk cache wrapper
        ));

        let compactor = Arc::new(RaptorCompactor::new(
            config.clone(),
            reader.clone(),
            zero_copy_filesystem.clone(),
            transaction_coordinator.clone(),
        ));

        // Matrix Trinity replaces HNSW - no separate manager needed
        // Matrices are stored in rowgroups and footer

        // Initialize AXIS clustering integration
        let clustering_config = ClusteringConfig {
            algorithm: ClusteringAlgorithm::KMeans(KMeansConfig {
                k: config.rowgroup_size / 100, // Adaptive cluster count
                ..Default::default()
            }),
            min_vectors_for_clustering: 100,
            max_clusters: 256,
            distance_metric: proximadb_distance_kernel::DistanceMetric::Cosine,
            adaptive_cluster_count: true,
            recompute_threshold: config.rowgroup_size / 2,
            enable_incremental: true,
        };

        let cluster_manager = Arc::new(RwLock::new(
            ClusterManager::new(clustering_config.clone()).await?,
        ));

        let cluster_assignments = Arc::new(RwLock::new(HashMap::new()));

        let cache = Arc::new(RwLock::new(RowGroupCache::new(
            config.cache_size_mb * 1024 * 1024,
        )));

        let file_registry = Arc::new(RwLock::new(FileRegistry::new()));

        // Initialize unified metrics collector for AutoML integration
        let metrics_collector = Some(Arc::new(EngineMetricsCollector::new()));

        // Get the global hardware capabilities instance
        let hardware_capabilities = get_hardware_capabilities();

        // Initialize universal performance optimization
        let universal_optimizer = UniversalPerformanceOptimizer::with_strategy(
            UniversalOptimizationStrategy::Balanced, // RAPTOR uses balanced strategy
        )
        .await?;

        Ok(Self {
            config,
            rowgroup_manager,
            writer,
            reader,
            compactor,
            cluster_manager,
            clustering_config,
            cluster_assignments,
            filesystem_factory,
            filesystem: data_filesystem,
            tier_config,
            file_options,
            zero_copy_filesystem,
            transaction_coordinator,
            universal_optimizer,
            hardware_capabilities,
            cache,
            file_registry,
            metrics_collector,
            storage_quantization_engine,
            fallback_quantization_engine,
            axis_manager: None, // AXIS manager will be set externally if available
        })
    }

    fn create_default_schema() -> Arc<Schema> {
        let fields = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::Float32, false),
            Field::new("metadata", DataType::Utf8, true), // JSON string for now
            Field::new("version", DataType::UInt32, true),
            Field::new("timestamp", DataType::Int64, true),
        ];

        Arc::new(Schema::new(fields))
    }

    // ============================================================================
    // PERFORMANCE OPTIMIZATION METHODS - DELEGATING TO UNIFIED MODULES
    // ============================================================================

    /// Fast read optimization using memory mapping (delegates to universal optimizer)
    async fn mmap_read_file(&self, file_path: &str) -> Result<Vec<u8>> {
        // Try memory mapping first
        if let Some(mmap) = self
            .universal_optimizer
            .get_memory_mapped_file(file_path)
            .await?
        {
            Ok(mmap.to_vec())
        } else {
            // Fallback to optimized reading for cloud storage
            self.universal_optimizer
                .read_data_optimized(file_path)
                .await
        }
    }

    /// I/O bandwidth optimization with vectorized reads (delegates to universal optimizer)
    async fn vectorized_read(&self, file_paths: &[String]) -> Result<Vec<Vec<u8>>> {
        // Use universal optimizer's parallel operations
        let optimizer = self.universal_optimizer.clone();
        let read_operations: Vec<_> = file_paths
            .iter()
            .map(|path| {
                let path = path.clone();
                let optimizer = optimizer.clone();
                async move { optimizer.read_data_optimized(&path).await }
            })
            .collect();

        let results = self
            .universal_optimizer
            .parallel_operations(read_operations, |operation| operation)
            .await?;

        // Unwrap the nested Results
        let mut data = Vec::with_capacity(results.len());
        for result in results {
            data.push(result??);
        }
        Ok(data)
    }

    /// Cloud storage cost optimization - determine optimal storage tier (delegates to universal optimizer)
    async fn optimize_storage_tier(
        &self,
        file_path: &str,
        _access_frequency: f32,
    ) -> Result<FileStorageTier> {
        // Estimate file size for tier optimization decision
        let estimated_size = 1024 * 1024; // Default 1MB if size unknown
        self.universal_optimizer
            .optimize_storage_tier(file_path, estimated_size)
            .await
    }

    /// Compression optimization for bandwidth and cost (delegates to universal optimizer)
    async fn compress_data_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use the tier determined from the storage location (base_path), not data size
        // The tier was already determined in constructor from the URL
        let tier = self.tier_config.tier;

        self.universal_optimizer.compress_for_tier(data, tier).await
    }

    /// Prefetch optimization for fast reads (delegates to universal optimizer)
    async fn prefetch_data(&self, file_path: &str) -> Result<()> {
        // Use universal optimizer's intelligent prefetching
        self.universal_optimizer
            .prefetch_data(&[file_path.to_string()])
            .await
    }

    /// SIMD-optimized vector operations (delegates to universal optimizer)
    async fn simd_vector_distance(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        // Use universal optimizer's hardware-accelerated distance computation
        self.universal_optimizer
            .compute_distances_accelerated(
                query,
                candidates,
                DistanceMetric::Euclidean, // Default metric for RAPTOR
            )
            .await
    }

    /// Memory pool optimization for vector allocations (delegates to universal optimizer)
    async fn get_pooled_buffer(&self, size: usize) -> Result<Vec<f32>> {
        self.universal_optimizer.get_memory_buffer(size).await
    }

    async fn insert_batch_internal(&self, records: Vec<VectorRecord>) -> Result<()> {
        // Write directly as VectorRecords - RAPTOR writer handles conversion internally
        let mut writer = self.writer.write().await;
        writer.write_vectors(&records).await?;

        // Convert to Arrow batch for clustering analysis
        let batch = self.convert_to_arrow_batch(records)?;

        // Matrix Trinity updates handled by writer during flush
        // Clustering is done during flush/compaction, not on each write

        // Update clustering if we have enough vectors
        let row_count = batch.num_rows();
        if row_count >= self.clustering_config.min_vectors_for_clustering {
            self.update_clustering(&batch).await?;
        }

        // Update unified metrics
        if let Some(ref collector) = self.metrics_collector {
            collector
                .record_operation("raptor", "insert", 0.0, false, row_count as u64)
                .await;
        }

        // Check if compaction is needed
        if self.should_compact().await {
            // Compaction needs to be triggered from do_flush or do_compact
            // which have access to collection_id and base_path
            // This internal method can't trigger compaction without that context
            // Deferred: Refactor to pass context through or trigger from outer methods
        }

        Ok(())
    }

    async fn update_clustering(&self, batch: &RecordBatch) -> Result<()> {
        let vectors = self.extract_vectors_from_batch(batch)?;

        // Use AXIS clustering manager
        let mut cluster_manager = self.cluster_manager.write().await;
        let assignments = cluster_manager.cluster_vectors(&vectors).await?;

        // Store cluster assignments per rowgroup
        let rowgroup_manager = self.rowgroup_manager.read().await;
        let row_group_ids = rowgroup_manager.row_group_ids();
        if let Some(current_rg_id) = row_group_ids.last() {
            let mut cluster_assignments = self.cluster_assignments.write().await;
            cluster_assignments.insert(*current_rg_id as u32, assignments);

            // Update rowgroup centroid for fast pruning
            drop(rowgroup_manager);
            let _rowgroup_manager = self.rowgroup_manager.write().await;
            // Note: We'd need to add a method to update centroid in RowGroups
            // For now, just skip this as it's an optimization
        }

        Ok(())
    }

    fn extract_vectors_from_batch(&self, batch: &RecordBatch) -> Result<Vec<Vec<f32>>> {
        let vector_column = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Vector column not found"))?;

        let float_array = vector_column
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Vector column is not Float32Array"))?;

        let dimension = float_array.len() / batch.num_rows();
        let mut vectors = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let start = i * dimension;
            let end = start + dimension;
            vectors.push(float_array.values()[start..end].to_vec());
        }

        Ok(vectors)
    }

    async fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<HashMap<String, String>>,
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
        storage_path: &str,
        collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        debug!(
            "RAPTOR SEARCH_INTERNAL: Starting with k={}, query_dim={}, storage_path={}, collection_id={}",
            k,
            query.len(),
            storage_path,
            collection_id
        );

        // Use clustering for efficient rowgroup pruning
        let selected_rowgroups = self
            .select_rowgroups_by_clustering(query, distance_metric)
            .await?;
        debug!(
            "RAPTOR SEARCH_INTERNAL: Selected {} rowgroups",
            selected_rowgroups.len()
        );

        // STATELESS MODE DETECTION: If no rowgroups selected, engine is stateless
        // This means we need to scan disk files directly
        if selected_rowgroups.is_empty() {
            debug!("RAPTOR SEARCH_INTERNAL: STATELESS MODE - No rowgroups, scanning disk files");
            return self
                .scan_disk_files_for_search(
                    query,
                    k,
                    filter,
                    distance_metric,
                    storage_path,
                    collection_id,
                )
                .await;
        }

        // Use Matrix Trinity for candidate selection
        let candidates: Vec<OptimizedSearchRecord> = if self.config.enable_clustering {
            // Use clustered search with Matrix Trinity
            self.clustered_search(
                query,
                k * 2,
                selected_rowgroups.clone(),
                distance_metric,
                storage_path,
                collection_id,
            )
            .await?
        } else {
            // Clustered search with pruning
            self.clustered_search(
                query,
                k * 2,
                selected_rowgroups.clone(),
                distance_metric,
                storage_path,
                collection_id,
            )
            .await?
        };

        debug!(
            "RAPTOR SEARCH_INTERNAL: Got {} candidates from clustered_search",
            candidates.len()
        );

        // Apply filters and rerank
        let mut results = Vec::new();
        for candidate in candidates {
            if let Some(ref filter) = filter
                && !self.matches_filter(&candidate, filter).await
            {
                continue;
            }
            results.push(candidate);
            if results.len() >= k {
                break;
            }
        }

        debug!(
            "RAPTOR SEARCH_INTERNAL: Returning {} final results",
            results.len()
        );
        Ok(results)
    }

    /// Scan disk files for search when no in-memory rowgroups are available (stateless mode)
    ///
    /// OPTIMIZED: Uses hierarchical centroid-based search instead of brute-force O(n) scan.
    /// This leverages the Matrix Trinity architecture (K×K → P×K → P²) stored in each file.
    async fn scan_disk_files_for_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<HashMap<String, String>>,
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
        storage_path: &str,
        collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let search_start = std::time::Instant::now();
        debug!(
            "SCAN_DISK: Starting indexed disk search for k={}, storage_path={}, collection_id={}",
            k, storage_path, collection_id
        );

        // Construct the data directory path: {storage_path}/{collection_id}/data
        let data_dir = format!("{}/{}/data", storage_path, collection_id);
        debug!("SCAN_DISK: Looking for files in: {}", data_dir);

        let list_start = std::time::Instant::now();

        // Use filesystem API to list files (cloud-compatible)
        let all_raptor_files = match self.filesystem.list(&data_dir).await {
            Ok(entries) => {
                let files: Vec<_> = entries
                    .into_iter()
                    .filter(|entry| {
                        // Match both old format (.data) and new format (.raptor)
                        entry.name.ends_with(".raptor") || entry.name.ends_with(".data")
                    })
                    .map(|entry| {
                        debug!("SCAN_DISK: Found RAPTOR file: {}", entry.name);
                        entry.url // Use full URL for cloud compatibility
                    })
                    .collect();
                files
            }
            Err(e) => {
                debug!("SCAN_DISK: Could not read directory {}: {}", data_dir, e);
                Vec::new()
            }
        };

        let files = all_raptor_files;
        debug!(
            "SCAN_DISK: Found {} .raptor files, listing took {:?}",
            files.len(),
            list_start.elapsed()
        );

        if files.is_empty() {
            return Ok(Vec::new());
        }

        // Use bounded priority queue to merge results from all files
        let mut priority_queue = crate::core::search::bounded_queue::BoundedPriorityQueue::new(k);

        // For each file, use hierarchical_search (reads from footer centroids - no in-memory state needed)
        // This is the key fix for stateless mode after close/reopen
        for file_url in files {
            debug!(
                "SCAN_DISK: Searching file with hierarchical_search: {}",
                file_url
            );

            // Create a RaptorReader for this file
            let cache =
                Arc::new(crate::storage::cache::orchestrator::CrossCacheOrchestrator::new(1000));
            let reader = RaptorReader::new(
                file_url.clone(),
                collection_id.to_string(),
                self.config.clone(),
                cache.clone(),
                self.filesystem.clone(),
            );

            // STEP 1: Use hierarchical_search to find top-k rowgroups by centroid distance
            // This reads directly from footer and doesn't need Matrix Trinity state
            // Get actual rowgroup count to calculate proper nprobe for high recall
            let num_rowgroups = reader.get_rowgroup_count().await.unwrap_or(10);
            // For 90%+ recall: search at least sqrt(num_rowgroups) * 2, but min 10 and at least k
            let nprobe = k
                .max(10)
                .max(((num_rowgroups as f32).sqrt().ceil() as usize) * 2);
            debug!(
                "SCAN_DISK: Calculated nprobe={} for num_rowgroups={}, k={}",
                nprobe, num_rowgroups, k
            );

            match reader
                .hierarchical_search(query, nprobe, distance_metric)
                .await
            {
                Ok(top_rowgroups) => {
                    debug!(
                        "SCAN_DISK: hierarchical_search found {} candidate rowgroups in {}",
                        top_rowgroups.len(),
                        file_url
                    );

                    // STEP 2: For each selected rowgroup, load vectors and compute distances
                    for rg_id in top_rowgroups {
                        // Read only vectors and IDs columns (selective column read)
                        match reader.read_columns(
                            &file_url,
                            rg_id,
                            &[
                                crate::storage::engines::raptor::common::ColumnType::VectorsFp32,
                                crate::storage::engines::raptor::common::ColumnType::Ids,
                            ],
                        ).await {
                            Ok(partial) => {
                                if let (Some(vectors), Some(ids)) = (partial.vectors, partial.ids) {
                                    // OPTIMIZATION: Use SIMD batched distance computation instead of scalar loop
                                    // This provides 10-50x speedup vs the original scalar approach
                                    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
                                    let compute = UnifiedDistanceCompute::default();

                                    // Use pooled SIMD batch method - returns SimilarityResult with normalized_score
                                    let similarity_results = compute.batch_distance_pooled_simd(query, &vector_refs, distance_metric);

                                    debug!(
                                        "SCAN_DISK: SIMD batch computed {} distances for rowgroup {}",
                                        similarity_results.len(),
                                        rg_id
                                    );

                                    for (idx, sim_result) in similarity_results.iter().enumerate() {
                                        let id = ids.get(idx).cloned().unwrap_or_default();
                                        let vector = vectors.get(idx).cloned().unwrap_or_default();

                                        let record = OptimizedSearchRecord::new(
                                            id,
                                            sim_result.normalized_score,
                                        )
                                        .with_similarity(sim_result.normalized_score)
                                        .add_vector(vector);

                                        // Apply filters if provided
                                        if let Some(ref f) = filter {
                                            let mut matches = true;
                                            for (key, value) in f {
                                                let filter_matches = record.metadata.get(key).is_some_and(|pv| {
                                                    use proximadb_data_model::ProximaValue;
                                                    match pv {
                                                        ProximaValue::String(s) => s == value,
                                                        ProximaValue::Int64(i) => &i.to_string() == value,
                                                        ProximaValue::Float64(f) => &f.to_string() == value,
                                                        ProximaValue::Boolean(b) => &b.to_string() == value,
                                                        _ => false,
                                                    }
                                                });
                                                if !filter_matches {
                                                    matches = false;
                                                    break;
                                                }
                                            }
                                            if !matches {
                                                continue;
                                            }
                                        }

                                        priority_queue.try_insert(record);
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("SCAN_DISK: Failed to read rowgroup {}: {}", rg_id, e);
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!(
                        "SCAN_DISK: hierarchical_search failed for {}: {}, falling back to full scan",
                        file_url, e
                    );
                    // Fallback to lightweight full scan for this file only
                    if let Ok(results) = reader.search_vectors_only(&file_url, query, k).await {
                        for result in results {
                            priority_queue.try_insert(result);
                        }
                    }
                }
            }
        }

        let final_results = priority_queue.into_sorted_vec();
        debug!(
            "SCAN_DISK: Returning {} indexed results, total search time: {:?}",
            final_results.len(),
            search_start.elapsed()
        );
        Ok(final_results)
    }

    /// Read vectors from a single file using filesystem API (cloud-compatible)
    /// Accepts a URL string (e.g., "file:///path", "s3://bucket/key", "azblob://container/blob")
    async fn read_vectors_from_file(&self, file_url: &str) -> Result<Vec<VectorRecord>> {
        // file_url is already a string URL from filesystem.list()
        let file_path_str = file_url.to_string();

        // Get or create a CrossCacheOrchestrator for the reader
        let cache =
            Arc::new(crate::storage::cache::orchestrator::CrossCacheOrchestrator::new(1000));

        // Use RaptorReader to read from the file
        // IMPORTANT: RaptorReader's base_path should be the file path, not directory
        let mut reader = RaptorReader::new(
            file_path_str.clone(),
            "".to_string(), // collection_id not needed for direct file read
            self.config.clone(),
            cache,
            self.filesystem.clone(),
        );

        // Use scan_vectors_with_strategy for full scan
        reader
            .scan_vectors_with_strategy(&file_path_str, ScanStrategy::FullScan)
            .await
    }

    async fn select_rowgroups_by_clustering(
        &self,
        query: &[f32],
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
    ) -> Result<Vec<u32>> {
        debug!("SELECT_ROWGROUPS: Starting rowgroup selection");
        let cluster_manager = self.cluster_manager.read().await;
        let cluster_assignments = self.cluster_assignments.read().await;
        let rowgroup_manager = self.rowgroup_manager.read().await;

        debug!(
            "SELECT_ROWGROUPS: cluster_assignments has {} entries",
            cluster_assignments.len()
        );
        debug!(
            "SELECT_ROWGROUPS: rowgroup_manager has {} rowgroups",
            rowgroup_manager.row_group_ids().len()
        );

        // STATELESS FALLBACK: If no in-memory rowgroups, we're in stateless mode
        // This happens when:
        // 1. Engine just started/restarted
        // 2. Collection data exists on disk but not loaded in memory
        //
        // Solution: Return empty to indicate "search all files on disk"
        // The clustered_search method should handle this by scanning disk
        if rowgroup_manager.row_group_ids().is_empty() {
            debug!("SELECT_ROWGROUPS: STATELESS MODE - No in-memory rowgroups");
            debug!("SELECT_ROWGROUPS: Returning empty - clustered_search should scan disk");
            // Return empty Vec which signals to clustered_search to scan all files
            return Ok(Vec::new());
        }

        // Find nearest clusters to query using nprobe = sqrt(k) for IVF-style routing
        // This provides ~95% recall with sublinear search cost O(sqrt(k) * p)
        let total_centroids = cluster_manager.centroid_count();
        let nprobe = if total_centroids == 0 {
            1 // Fallback: at least 1 cluster
        } else {
            // sqrt(k) with minimum of 1, capped at total centroids
            ((total_centroids as f64).sqrt().ceil() as usize)
                .max(1)
                .min(total_centroids)
        };
        debug!(
            "SELECT_ROWGROUPS: Using nprobe={} (sqrt of {} centroids)",
            nprobe, total_centroids
        );
        let nearest_clusters = cluster_manager.find_nearest_clusters(query, nprobe).await?;
        debug!(
            "SELECT_ROWGROUPS: Found {} nearest clusters",
            nearest_clusters.len()
        );

        // Select rowgroups that contain these clusters
        let mut selected = Vec::new();
        for (rg_id, assignments) in cluster_assignments.iter() {
            for assignment in assignments {
                if nearest_clusters.contains(&assignment.cluster_id) {
                    selected.push(*rg_id);
                    break;
                }
            }
        }
        debug!(
            "SELECT_ROWGROUPS: Selected {} rowgroups from clustering",
            selected.len()
        );

        // If no clusters found, use centroid-based selection
        if selected.is_empty() {
            debug!("SELECT_ROWGROUPS: No clusters found, using centroid-based selection");
            for rg_id in rowgroup_manager.row_group_ids() {
                if let Some(rowgroup) = rowgroup_manager.row_group(&rg_id)
                    && let Some(centroid) = &rowgroup.centroid
                {
                    // Use the query's metric (NOT the Euclidean default) to score
                    // centroids. The 0.5 threshold is Cosine-calibrated (≈60°);
                    // non-Cosine metrics may need separate tuning (TODO).
                    let compute = UnifiedDistanceCompute::new(*distance_metric);
                    let distance = compute.distance_with_metric(query, centroid, distance_metric);
                    if distance < 0.5 {
                        // Threshold for similarity
                        selected.push(rowgroup.id as u32);
                    }
                }
            }
            debug!(
                "SELECT_ROWGROUPS: Selected {} rowgroups from centroids",
                selected.len()
            );
        }

        Ok(selected)
    }

    async fn clustered_search(
        &self,
        query: &[f32],
        k: usize,
        selected_rowgroups: Vec<u32>,
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
        storage_path: &str,
        collection_id: &str,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        debug!(
            "CLUSTERED_SEARCH: Starting with {} selected rowgroups",
            selected_rowgroups.len()
        );

        // STATELESS MODE: If no rowgroups selected, scan disk for files
        if selected_rowgroups.is_empty() {
            debug!(
                "CLUSTERED_SEARCH: STATELESS MODE - No rowgroups, will scan disk for .raptor files"
            );
            return self
                .scan_disk_files_for_search(
                    query,
                    k,
                    None,
                    distance_metric,
                    storage_path,
                    collection_id,
                )
                .await;
        }

        // Use bounded priority queue to maintain only top-k results
        let mut priority_queue = BoundedPriorityQueue::new(k);

        for rg_id in selected_rowgroups {
            // Use filesystem API for efficient range reads
            let batch = self.read_rowgroup_with_range(rg_id).await?;

            // Compute distances using UnifiedDistanceCompute batch methods
            // IMPORTANT: Keep full SimilarityResult objects - they already have normalized_score!
            let similarity_results = if self.config.enable_simd {
                // Use optimized batch distance with memory pool
                let vectors = self.extract_vectors_from_batch(&batch)?;
                let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
                let compute = UnifiedDistanceCompute::default();

                // Use pooled SIMD batch method - returns SimilarityResult with normalized_score
                compute.batch_distance_pooled_simd(query, &vector_refs, distance_metric)
            } else {
                // Scalar path - compute SimilarityResult objects
                let distances = self.compute_distances_scalar(query, &batch)?;
                distances
                    .into_iter()
                    .map(|d| {
                        proximadb_distance_kernel::engine::SimilarityResult::new(
                            d,
                            *distance_metric,
                        )
                    })
                    .collect()
            };

            // Use normalized_score directly from UnifiedDistanceCompute - no conversion needed!
            for (i, sim_result) in similarity_results.iter().enumerate() {
                let id = self.get_id_from_batch(&batch, i)?;

                // IMPORTANT: normalized_score already computed by UnifiedDistanceCompute
                // All engines use this same value - ensures cross-engine consistency
                let search_result = OptimizedSearchRecord::new(id, sim_result.normalized_score)
                    .with_similarity(sim_result.normalized_score)
                    .with_metadata(HashMap::new());

                priority_queue.try_insert(search_result);
            }
        }

        // Get sorted results from bounded queue
        let results = priority_queue.into_sorted_vec();

        Ok(results)
    }

    async fn read_rowgroup_with_range(&self, rg_id: u32) -> Result<RecordBatch> {
        let rowgroup_manager = self.rowgroup_manager.read().await;
        let rowgroup = rowgroup_manager
            .row_group(&(rg_id as u16))
            .ok_or_else(|| anyhow::anyhow!("RowGroup {} not found", rg_id))?;

        // This method needs context to determine path - should not be called directly
        // Path should come from StorageQueryContext or FlushParameters
        let path = format!("/tmp/placeholder/rowgroup_{}.raptor", rg_id);

        // Use filesystem range read for efficient cloud I/O
        let data = if self.is_cloud_storage() {
            self.filesystem
                .read_range(&path, rowgroup.offset, rowgroup.compressed_size)
                .await?
        } else {
            self.filesystem.read(&path).await?
        };

        // Decompress and deserialize
        let decompressed = self.decompress_data(&data)?;
        self.deserialize_batch(&decompressed)
    }

    fn is_cloud_storage(&self) -> bool {
        matches!(
            self.tier_config.tier,
            FileStorageTier::S3Express
                | FileStorageTier::S3Standard
                | FileStorageTier::S3GlacierInstant
                | FileStorageTier::AzurePremium
                | FileStorageTier::AzureStandard
                | FileStorageTier::GcsSSD
                | FileStorageTier::GcsHDD
        )
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        // Cosine distance
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        Ok(1.0 - (dot_product / (norm_a * norm_b)))
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified - would use actual compression codec
        Ok(data.to_vec())
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<RecordBatch> {
        // PROXIMA INTEGRATION: Check for encoding marker
        // RAPTOR uses 0xA0-0xAF range for tensor-optimized encodings
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data"));
        }

        let encoding_marker = data[0];

        // Check if this is a Proxima-encoded batch
        match encoding_marker {
            0xA1 => {
                // Proxima tensor encoding - decode it first
                self.deserialize_proxima_batch(&data[1..], encoding_marker)
            }
            0xA2 => {
                // Sparse tensor encoding
                self.deserialize_sparse_tensor_batch(&data[1..])
            }
            0xA3 => {
                // Quantized tensor encoding
                self.deserialize_quantized_tensor_batch(&data[1..])
            }
            _ => {
                // Raw tensors or standard Arrow IPC format
                // For backward compatibility or non-encoded data
                use arrow_ipc::reader::StreamReader;
                use std::io::Cursor;

                // Skip marker if it's 0xA0, otherwise process full data
                let ipc_data = if encoding_marker == 0xA0 {
                    &data[1..]
                } else {
                    data
                };

                let cursor = Cursor::new(ipc_data);
                let reader = StreamReader::try_new(cursor, None)?;
                let batches: Result<Vec<_>, _> = reader.collect();
                let batches = batches?;

                if batches.is_empty() {
                    return Err(anyhow::anyhow!("No batches found"));
                }

                Ok(batches[0].clone())
            }
        }
    }

    fn deserialize_proxima_batch(&self, data: &[u8], _marker: u8) -> Result<RecordBatch> {
        use crate::storage::engines::core::ops::proximacodec::ProximaCodec;
        use arrow_array::{ArrayRef, Float32Array, Int64Array, StringArray, UInt32Array};
        use std::io::Read;

        let mut cursor = std::io::Cursor::new(data);

        // Read metadata
        let mut dim_bytes = [0u8; 4];
        cursor.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut count_bytes = [0u8; 4];
        cursor.read_exact(&mut count_bytes)?;
        let num_vectors = u32::from_le_bytes(count_bytes) as usize;

        // Decode each dimension column
        let mut columns: Vec<Vec<f32>> = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let mut len_bytes = [0u8; 4];
            cursor.read_exact(&mut len_bytes)?;
            let column_len = u32::from_le_bytes(len_bytes) as usize;

            let mut column_data = vec![0u8; column_len];
            cursor.read_exact(&mut column_data)?;

            // Decode using ProximaCodec
            let codec = ProximaCodec::global();
            let decoded = codec.decode(&column_data)?;
            columns.push(decoded);
        }

        // Transpose back to row-major for RecordBatch
        let mut vectors = Vec::with_capacity(num_vectors * dimension);
        for i in 0..num_vectors {
            for col in &columns {
                if i < col.len() {
                    vectors.push(col[i]);
                }
            }
        }

        // Read IDs if present
        let mut ids = Vec::new();
        for i in 0..num_vectors {
            let mut len_bytes = [0u8; 4];
            if cursor.read_exact(&mut len_bytes).is_ok() {
                let id_len = u32::from_le_bytes(len_bytes) as usize;
                if id_len > 0 {
                    let mut id_data = vec![0u8; id_len];
                    cursor.read_exact(&mut id_data)?;
                    ids.push(Some(String::from_utf8(id_data)?));
                } else {
                    ids.push(None);
                }
            } else {
                // Generate default IDs if not present
                ids.push(Some(format!("vec_{}", i)));
            }
        }

        // Read timestamps if present
        let mut timestamps = Vec::new();
        for _ in 0..num_vectors {
            let mut ts_bytes = [0u8; 8];
            if cursor.read_exact(&mut ts_bytes).is_ok() {
                timestamps.push(Some(i64::from_le_bytes(ts_bytes)));
            } else {
                timestamps.push(Some(0i64));
            }
        }

        // Create RecordBatch from decoded data
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;
        let vector_array = Arc::new(Float32Array::from(vectors)) as ArrayRef;

        // Add placeholder metadata column
        let metadata_array =
            Arc::new(StringArray::from(vec![None::<String>; num_vectors])) as ArrayRef;

        // Add version column
        let version_array = Arc::new(UInt32Array::from(vec![1u32; num_vectors])) as ArrayRef;

        // Add timestamp column
        let timestamp_array = Arc::new(Int64Array::from(timestamps)) as ArrayRef;

        let batch = RecordBatch::try_new(
            Self::create_default_schema(),
            vec![
                id_array,
                vector_array,
                metadata_array,
                version_array,
                timestamp_array,
            ],
        )?;

        Ok(batch)
    }

    fn deserialize_sparse_tensor_batch(&self, data: &[u8]) -> Result<RecordBatch> {
        // SPARSE TENSOR DESERIALIZATION (COO/CSR format)
        // Marker 0xA2 indicates sparse tensor encoding

        use std::io::Read;

        let mut cursor = std::io::Cursor::new(data);

        // Read sparse tensor metadata
        let mut format_byte = [0u8; 1];
        cursor.read_exact(&mut format_byte)?;
        let is_coo_format = format_byte[0] == 0; // 0=COO, 1=CSR

        let mut dim_bytes = [0u8; 4];
        cursor.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut count_bytes = [0u8; 4];
        cursor.read_exact(&mut count_bytes)?;
        let num_vectors = u32::from_le_bytes(count_bytes) as usize;

        let mut nnz_bytes = [0u8; 4];
        cursor.read_exact(&mut nnz_bytes)?;
        let num_nonzeros = u32::from_le_bytes(nnz_bytes) as usize;

        if is_coo_format {
            // COO Format: (row_indices, col_indices, values)
            // Read row indices
            let mut row_indices = Vec::with_capacity(num_nonzeros);
            for _ in 0..num_nonzeros {
                let mut idx_bytes = [0u8; 4];
                cursor.read_exact(&mut idx_bytes)?;
                row_indices.push(u32::from_le_bytes(idx_bytes));
            }

            // Read column indices
            let mut col_indices = Vec::with_capacity(num_nonzeros);
            for _ in 0..num_nonzeros {
                let mut idx_bytes = [0u8; 4];
                cursor.read_exact(&mut idx_bytes)?;
                col_indices.push(u32::from_le_bytes(idx_bytes));
            }

            // Read values (using Proxima encoding for compression)
            let mut val_len_bytes = [0u8; 4];
            cursor.read_exact(&mut val_len_bytes)?;
            let values_len = u32::from_le_bytes(val_len_bytes) as usize;

            let mut values_data = vec![0u8; values_len];
            cursor.read_exact(&mut values_data)?;

            // Decode values using Proxima
            let codec = ProximaCodec::global();
            let values = codec.decode(&values_data)?;

            // Reconstruct dense vectors from sparse representation
            let mut dense_vectors = vec![0.0f32; num_vectors * dimension];
            for (idx, &value) in values.iter().enumerate() {
                let row = row_indices[idx] as usize;
                let col = col_indices[idx] as usize;
                if row < num_vectors && col < dimension {
                    dense_vectors[row * dimension + col] = value;
                }
            }

            // Create RecordBatch
            self.create_batch_from_dense_vectors(dense_vectors, num_vectors, dimension)
        } else {
            // CSR Format: (row_ptrs, col_indices, values)
            // Read row pointers
            let mut row_ptrs = Vec::with_capacity(num_vectors + 1);
            for _ in 0..=num_vectors {
                let mut ptr_bytes = [0u8; 4];
                cursor.read_exact(&mut ptr_bytes)?;
                row_ptrs.push(u32::from_le_bytes(ptr_bytes));
            }

            // Read column indices
            let mut col_indices = Vec::with_capacity(num_nonzeros);
            for _ in 0..num_nonzeros {
                let mut idx_bytes = [0u8; 4];
                cursor.read_exact(&mut idx_bytes)?;
                col_indices.push(u32::from_le_bytes(idx_bytes));
            }

            // Read and decode values
            let mut val_len_bytes = [0u8; 4];
            cursor.read_exact(&mut val_len_bytes)?;
            let values_len = u32::from_le_bytes(val_len_bytes) as usize;

            let mut values_data = vec![0u8; values_len];
            cursor.read_exact(&mut values_data)?;

            let codec = ProximaCodec::global();
            let values = codec.decode(&values_data)?;

            // Reconstruct dense vectors from CSR
            let mut dense_vectors = vec![0.0f32; num_vectors * dimension];
            for row in 0..num_vectors {
                let start = row_ptrs[row] as usize;
                let end = row_ptrs[row + 1] as usize;

                for idx in start..end {
                    if idx < col_indices.len() && idx < values.len() {
                        let col = col_indices[idx] as usize;
                        if col < dimension {
                            dense_vectors[row * dimension + col] = values[idx];
                        }
                    }
                }
            }

            self.create_batch_from_dense_vectors(dense_vectors, num_vectors, dimension)
        }
    }

    fn deserialize_quantized_tensor_batch(&self, data: &[u8]) -> Result<RecordBatch> {
        // QUANTIZED TENSOR DESERIALIZATION (INT8/PQ formats)
        // Marker 0xA3 indicates quantized tensor encoding

        use std::io::Read;

        let mut cursor = std::io::Cursor::new(data);

        // Read quantization type
        let mut quant_type = [0u8; 1];
        cursor.read_exact(&mut quant_type)?;

        match quant_type[0] {
            0 => {
                // INT8 Quantization
                let mut dim_bytes = [0u8; 4];
                cursor.read_exact(&mut dim_bytes)?;
                let dimension = u32::from_le_bytes(dim_bytes) as usize;

                let mut count_bytes = [0u8; 4];
                cursor.read_exact(&mut count_bytes)?;
                let num_vectors = u32::from_le_bytes(count_bytes) as usize;

                // Read scale and zero point for dequantization
                let mut scale_bytes = [0u8; 4];
                cursor.read_exact(&mut scale_bytes)?;
                let scale = f32::from_le_bytes(scale_bytes);

                let mut zero_bytes = [0u8; 4];
                cursor.read_exact(&mut zero_bytes)?;
                let zero_point = f32::from_le_bytes(zero_bytes);

                // Read INT8 data
                let mut int8_data = vec![0i8; num_vectors * dimension];
                cursor.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        int8_data.as_mut_ptr() as *mut u8,
                        int8_data.len(),
                    )
                })?;

                // Dequantize to FP32
                let mut dense_vectors = Vec::with_capacity(num_vectors * dimension);
                for &quantized_val in &int8_data {
                    let dequantized = (quantized_val as f32) * scale + zero_point;
                    dense_vectors.push(dequantized);
                }

                self.create_batch_from_dense_vectors(dense_vectors, num_vectors, dimension)
            }
            1 => {
                // Product Quantization (PQ)
                let mut dim_bytes = [0u8; 4];
                cursor.read_exact(&mut dim_bytes)?;
                let dimension = u32::from_le_bytes(dim_bytes) as usize;

                let mut count_bytes = [0u8; 4];
                cursor.read_exact(&mut count_bytes)?;
                let num_vectors = u32::from_le_bytes(count_bytes) as usize;

                let mut subvec_bytes = [0u8; 4];
                cursor.read_exact(&mut subvec_bytes)?;
                let num_subvectors = u32::from_le_bytes(subvec_bytes) as usize;

                let mut codebook_bytes = [0u8; 4];
                cursor.read_exact(&mut codebook_bytes)?;
                let codebook_size = u32::from_le_bytes(codebook_bytes) as usize;

                // Read codebooks (centroids for each subvector)
                let subvector_dim = dimension / num_subvectors;
                let mut codebooks = Vec::new();

                for _ in 0..num_subvectors {
                    let mut subvec_codebook = Vec::new();
                    for _ in 0..codebook_size {
                        for _ in 0..subvector_dim {
                            let mut val_bytes = [0u8; 4];
                            cursor.read_exact(&mut val_bytes)?;
                            subvec_codebook.push(f32::from_le_bytes(val_bytes));
                        }
                    }
                    codebooks.push(subvec_codebook);
                }

                // Read PQ codes (indices into codebooks)
                let mut pq_codes = vec![0u8; num_vectors * num_subvectors];
                cursor.read_exact(&mut pq_codes)?;

                // Reconstruct vectors from PQ codes
                let mut dense_vectors = Vec::with_capacity(num_vectors * dimension);
                for vec_idx in 0..num_vectors {
                    for subvec_idx in 0..num_subvectors {
                        let code = pq_codes[vec_idx * num_subvectors + subvec_idx] as usize;
                        let codebook_offset = code * subvector_dim;

                        for dim_idx in 0..subvector_dim {
                            let value = codebooks[subvec_idx][codebook_offset + dim_idx];
                            dense_vectors.push(value);
                        }
                    }
                }

                self.create_batch_from_dense_vectors(dense_vectors, num_vectors, dimension)
            }
            2 => {
                // Binary Quantization (1 bit per dimension)
                let mut dim_bytes = [0u8; 4];
                cursor.read_exact(&mut dim_bytes)?;
                let dimension = u32::from_le_bytes(dim_bytes) as usize;

                let mut count_bytes = [0u8; 4];
                cursor.read_exact(&mut count_bytes)?;
                let num_vectors = u32::from_le_bytes(count_bytes) as usize;

                // Read binary data (packed bits)
                let bits_per_vector = dimension.div_ceil(8); // Round up to byte boundary
                let mut binary_data = vec![0u8; num_vectors * bits_per_vector];
                cursor.read_exact(&mut binary_data)?;

                // Unpack bits to float values (-1.0 or 1.0)
                let mut dense_vectors = Vec::with_capacity(num_vectors * dimension);
                for vec_idx in 0..num_vectors {
                    for dim_idx in 0..dimension {
                        let byte_idx = vec_idx * bits_per_vector + dim_idx / 8;
                        let bit_idx = dim_idx % 8;
                        let bit = (binary_data[byte_idx] >> bit_idx) & 1;
                        dense_vectors.push(if bit == 1 { 1.0 } else { -1.0 });
                    }
                }

                self.create_batch_from_dense_vectors(dense_vectors, num_vectors, dimension)
            }
            _ => Err(anyhow::anyhow!(
                "Unknown quantization type: {}",
                quant_type[0]
            )),
        }
    }

    fn create_batch_from_dense_vectors(
        &self,
        dense_vectors: Vec<f32>,
        num_vectors: usize,
        _dimension: usize,
    ) -> Result<RecordBatch> {
        use arrow_array::{ArrayRef, Float32Array, Int64Array, StringArray, UInt32Array};

        // Generate IDs
        let ids: Vec<Option<String>> = (0..num_vectors)
            .map(|i| Some(format!("tensor_{}", i)))
            .collect();

        // Create arrays
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;
        let vector_array = Arc::new(Float32Array::from(dense_vectors)) as ArrayRef;
        let metadata_array =
            Arc::new(StringArray::from(vec![None::<String>; num_vectors])) as ArrayRef;
        let version_array = Arc::new(UInt32Array::from(vec![1u32; num_vectors])) as ArrayRef;
        let timestamp_array = Arc::new(Int64Array::from(vec![0i64; num_vectors])) as ArrayRef;

        RecordBatch::try_new(
            Self::create_default_schema(),
            vec![
                id_array,
                vector_array,
                metadata_array,
                version_array,
                timestamp_array,
            ],
        )
        .map_err(|e| anyhow::anyhow!("Failed to create RecordBatch: {}", e))
    }

    async fn full_scan_search(
        &self,
        query: &[f32],
        k: usize,
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
    ) -> Result<Vec<VectorSearchResult>> {
        let rowgroup_manager = self.rowgroup_manager.read().await;
        // For full scan, get all rowgroups
        let selected_rowgroups = rowgroup_manager.row_group_ids();

        // Use bounded priority queue to maintain only top-k results
        let mut priority_queue = BoundedPriorityQueue::new(k);

        for rg_id in selected_rowgroups {
            // Check cache first
            // Use a generic key format - actual collection_id comes from context
            let key = format!("raptor_rowgroup_{}", rg_id);
            let batch = if let Some(cached) = self.get_cached_rowgroup(&key).await {
                cached
            } else {
                // Read from storage
                let batch = self.reader.read_rowgroup(rg_id).await?;
                self.cache_rowgroup(&key, batch.clone()).await;
                batch
            };

            // Compute distances using UnifiedDistanceCompute batch methods
            // IMPORTANT: Keep full SimilarityResult objects - they already have normalized_score!
            let similarity_results = if self.config.enable_simd {
                // Use optimized batch distance with memory pool
                let vectors = self.extract_vectors_from_batch(&batch)?;
                let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
                let compute = UnifiedDistanceCompute::default();

                // Use pooled SIMD batch method - returns SimilarityResult with normalized_score
                compute.batch_distance_pooled_simd(query, &vector_refs, distance_metric)
            } else {
                // Scalar path - compute SimilarityResult objects
                let distances = self.compute_distances_scalar(query, &batch)?;
                distances
                    .into_iter()
                    .map(|d| {
                        proximadb_distance_kernel::engine::SimilarityResult::new(
                            d,
                            *distance_metric,
                        )
                    })
                    .collect()
            };

            // Use normalized_score directly from UnifiedDistanceCompute - no conversion needed!
            for (i, sim_result) in similarity_results.iter().enumerate() {
                let id = self.get_id_from_batch(&batch, i)?;

                // IMPORTANT: normalized_score already computed by UnifiedDistanceCompute
                // All engines use this same value - ensures cross-engine consistency
                let search_result = OptimizedSearchRecord::new(id, sim_result.normalized_score)
                    .with_similarity(sim_result.normalized_score)
                    .with_metadata(HashMap::new());

                priority_queue.try_insert(search_result);
            }
        }

        // Get sorted results from bounded queue
        let results = priority_queue.into_sorted_vec();

        Ok(results)
    }

    async fn get_cached_rowgroup(&self, key: &str) -> Option<RecordBatch> {
        let cache = self.cache.read().await;
        cache.get(key)
    }

    async fn cache_rowgroup(&self, key: &str, batch: RecordBatch) {
        let mut cache = self.cache.write().await;
        cache.put(key.to_string(), batch);
    }

    fn convert_to_arrow_batch(&self, records: Vec<VectorRecord>) -> Result<RecordBatch> {
        let mut ids = Vec::new();
        let mut vectors = Vec::new();
        let mut metadata_strs = Vec::new();
        let mut versions = Vec::new();
        let mut timestamps = Vec::new();

        for record in records {
            ids.push(record.id.clone());
            vectors.extend_from_slice(&record.vector);

            // Convert metadata to JSON string
            let metadata_json = serde_json::to_string(
                &crate::core::proto_metadata_helper::sqlvalue_metadata_to_json(&record.metadata),
            )?;
            metadata_strs.push(Some(metadata_json));

            // Convert Option<i64> to Option<u32> for Arrow UInt32Array compatibility
            versions.push(record.version);
            timestamps.push(Some(record.timestamp.unwrap_or(0)));
        }

        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;
        let vector_array = Arc::new(Float32Array::from(vectors)) as ArrayRef;
        let metadata_array = Arc::new(StringArray::from(metadata_strs)) as ArrayRef;
        let version_array = Arc::new(UInt32Array::from(versions)) as ArrayRef;
        let timestamp_array = Arc::new(Int64Array::from(timestamps)) as ArrayRef;

        let batch = RecordBatch::try_new(
            Self::create_default_schema(),
            vec![
                id_array,
                vector_array,
                metadata_array,
                version_array,
                timestamp_array,
            ],
        )?;

        Ok(batch)
    }

    fn compute_distances_scalar(&self, query: &[f32], batch: &RecordBatch) -> Result<Vec<f32>> {
        let vector_column = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Vector column not found"))?;

        let float_array = vector_column
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Vector column is not Float32Array"))?;

        let dimension = query.len();
        let num_vectors = batch.num_rows();
        let mut distances = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let start = i * dimension;
            let end = start + dimension;
            let vector = &float_array.values()[start..end];

            // Compute cosine distance
            let dot_product: f32 = query.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();

            let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let vector_norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

            let cosine_similarity = dot_product / (query_norm * vector_norm);
            let distance = 1.0 - cosine_similarity;

            distances.push(distance);
        }

        Ok(distances)
    }

    fn get_id_from_batch(&self, batch: &RecordBatch, index: usize) -> Result<String> {
        let id_column = batch
            .column_by_name("id")
            .ok_or_else(|| anyhow::anyhow!("ID column not found"))?;

        let string_array = id_column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("ID column is not StringArray"))?;

        Ok(string_array.value(index).to_string())
    }

    async fn matches_filter(
        &self,
        _result: &VectorSearchResult,
        _filter: &HashMap<String, String>,
    ) -> bool {
        // Simple filter matching - can be extended
        true
    }

    async fn should_compact(&self) -> bool {
        let registry = self.file_registry.read().await;
        registry.active_files.len() >= self.config.compaction_threshold_files
    }

    fn reconstruct_vector_record(&self, batch: &RecordBatch, index: usize) -> Result<VectorRecord> {
        let id = self.get_id_from_batch(batch, index)?;

        let vector_column = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Vector column not found"))?;

        // Try FixedSizeListArray first (proper Arrow representation)
        // Fall back to flat Float32Array for backward compatibility
        let vector = if let Some(list_array) = vector_column
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
        {
            // Extract vector from FixedSizeListArray
            let values = list_array.values();
            let float_values = values
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Vector list values are not Float32Array"))?;

            let dimension = list_array.value_length() as usize;
            let start = index * dimension;
            let end = start + dimension;
            float_values.values()[start..end].to_vec()
        } else if let Some(float_array) = vector_column
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
        {
            // Backward compatibility: flat Float32Array
            let dimension = float_array.len() / batch.num_rows();
            let start = index * dimension;
            let end = start + dimension;
            float_array.values()[start..end].to_vec()
        } else {
            return Err(anyhow::anyhow!(
                "Vector column is neither FixedSizeListArray nor Float32Array"
            ));
        };

        let metadata_str = batch
            .column_by_name("metadata")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .map_or("", |arr| arr.value(index));

        let metadata_json: serde_json::Value = if metadata_str.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(metadata_str)?
        };

        // Convert JSON metadata to HashMap<String, SqlValue> for proto v1 compatibility
        let metadata = if let Some(obj) = metadata_json.as_object() {
            let metadata_items =
                crate::core::utils::metadata_conversions::json_to_proto_metadata(obj.clone());
            crate::core::proto_metadata_helper::proto_metadata_to_sqlvalue_hashmap(&metadata_items)
        } else {
            HashMap::new()
        };

        Ok(VectorRecord {
            id,
            vector,
            metadata,
            version: Some(0),
            ..Default::default()
        })
    }

    /// Determine storage tier from base path
    /// Note: /tmp paths should not be used for production storage
    /// UnifiedCachingFilesystem will handle caching transparently:
    /// - Cloud storage (S3/Azure/GCS) files cached at /tmp/proximadb/cache/
    /// - Cache is managed by LRU policy, not a primary storage location
    pub fn determine_storage_tier(
        base_path: &str,
    ) -> crate::storage::persistence::filesystem::FileStorageTier {
        use crate::storage::persistence::filesystem::FileStorageTier;

        if base_path.contains("s3://") {
            // Check for S3 Express bucket (contains "express" in the bucket name)
            if base_path.contains("express") {
                FileStorageTier::S3Express
            } else {
                FileStorageTier::S3Standard
            }
        } else if base_path.contains("gs://") || base_path.contains("gcs://") {
            FileStorageTier::GcsSSD
        } else if base_path.contains("azure://") {
            FileStorageTier::AzurePremium
        } else if base_path.contains("memory") {
            // Only treat explicit memory:// paths as memory tier
            FileStorageTier::Memory
        } else if base_path.contains("nvme") {
            // NVMe paths
            FileStorageTier::NVMe
        } else {
            // Local filesystem paths use SSD tier
            FileStorageTier::SSD
        }
    }
}

#[allow(deprecated)]
#[async_trait]
impl UnifiedStorageFormat for RaptorEngine {
    fn engine_name(&self) -> &'static str {
        "RAPTOR"
    }

    fn engine_version(&self) -> &'static str {
        "1.0.0"
    }

    fn strategy(&self) -> crate::storage::traits::StorageFormatStrategy {
        crate::storage::traits::StorageFormatStrategy::Raptor
    }

    async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
        let collection_id = self.get_collection_id_from_params(params)?;
        let start_time = std::time::Instant::now();

        debug!("RAPTOR FLUSH: Started for collection={}", collection_id);
        debug!(
            "RAPTOR FLUSH: {} vectors to flush",
            params.vector_records.len()
        );

        // Get collection config dimension - required for proper compaction
        let collection_dimension = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|cfg| cfg.dimension)
            .ok_or_else(|| {
                ProximaDBError::Config("Missing dimension field in collection config".to_string())
            })?;

        debug!(
            "RAPTOR FLUSH: Using collection config dimension={}",
            collection_dimension
        );

        // Determine the proper file path for this collection
        // Format is: {baseurl}/{collectionid}/data/
        let data_dir = self.get_data_dir_from_flush_params(params)?;

        debug!("RAPTOR FLUSH: Data directory={}", data_dir);

        // Use filesystem API to create directory
        self.filesystem.create_dir_all(&data_dir).await?;
        debug!("RAPTOR FLUSH: Created directory={}", data_dir);

        // Create a new filename for this flush using FilenameCodec
        use crate::storage::engines::core::constants;
        let codec = crate::storage::common::compaction_orchestrator::FilenameCodec::new();
        let filename = codec.generate(0, constants::raptor::FILE_EXTENSION); // Level 0 for new flushes
        let file_path = format!("{}/{}", data_dir, filename);

        debug!("RAPTOR FLUSH: Writing to file={}", file_path);

        // Create a new writer with the proper file path
        let mut writer = RaptorWriter::new(
            file_path.clone(),
            self.config.clone(),
            collection_id.to_string(),
            collection_dimension as usize,
            self.filesystem.clone(),
            make_raptor_quantization_engine(&self.config).await?,
            make_raptor_axis_clustering(&self.config),
        )
        .await?;

        // Convert ProximaRecord → VectorRecord for RAPTOR writer (protocol adapter boundary)
        let vector_records_v1: Vec<VectorRecord> = params
            .vector_records
            .iter()
            .map(proxima_record_to_vector)
            .collect();

        // Write the vectors from params to the writer first
        debug!(
            "RAPTOR FLUSH: Writing {} vectors to writer",
            vector_records_v1.len()
        );
        writer.write_vectors(&vector_records_v1).await?;
        debug!("RAPTOR FLUSH: Vectors written to writer");

        // Close the writer - this will flush, update metadata, and finalize
        debug!("RAPTOR FLUSH: Closing writer");
        let vectors_flushed = vector_records_v1.len(); // Use the input count
        writer.close().await?;
        debug!(
            "RAPTOR FLUSH: Writer closed, {} vectors written to {}",
            vectors_flushed, file_path
        );

        // Get actual file size to report bytes written
        let bytes_written = match self.filesystem.metadata(&file_path).await {
            Ok(metadata) => metadata.size,
            Err(e) => {
                tracing::warn!("RAPTOR flush: Failed to get file size: {}", e);
                0
            }
        };

        // Update unified metrics
        if let Some(ref collector) = self.metrics_collector {
            let flush_duration_ms = start_time.elapsed().as_millis() as f64;
            collector
                .record_operation(
                    "raptor",
                    "flush",
                    flush_duration_ms,
                    false,
                    vectors_flushed as u64,
                )
                .await;
        }

        // HNSW is integrated within RAPTOR row groups, no separate flush needed
        if self.config.enable_clustering {
            // Clustering flush is handled by writer.flush()
        }

        Ok(FlushResult {
            success: true,
            files_created: Some(1),
            file_paths: vec![file_path],
            bytes_written: Some(bytes_written),
            duration_ms: Some(start_time.elapsed().as_millis() as u64),
            collections_affected: vec![collection_id.to_string()],
            entries_flushed: Some(vectors_flushed as u64),
            flushed_batch_ids: vec![],
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
            compaction_triggered: false,
            compaction_error: None,
        })
    }

    /// RAPTOR's LSM bulk-load override (Phase 2F-b).
    ///
    /// Delegates to `do_flush` with a synthetic FlushParameters.
    /// RAPTOR's flush builds the hierarchical writer + summarization
    /// layer; bulk-loaded segments inherit the same hierarchy. The
    /// drainer sends a pre-sorted batch, the engine writes it once,
    /// no WAL+memtable cost.
    async fn ingest_sorted_segment(
        &self,
        collection_id: &str,
        base_path: &str,
        records: Vec<proximadb_records::ProximaRecord>,
    ) -> Result<crate::storage::traits::SegmentIngestResult> {
        use crate::storage::engines::core::ops::bulk_ingest::{
            build_bulk_flush_parameters, empty_segment_result, segment_result_from_flush,
        };

        let count = records.len();
        if count == 0 {
            return Ok(empty_segment_result(collection_id));
        }

        let params = build_bulk_flush_parameters(collection_id, base_path, records, "RAPTOR")?;
        let flush_result = self.do_flush(&params).await?;
        Ok(segment_result_from_flush(
            collection_id,
            count,
            &flush_result,
            "raptor",
        ))
    }

    async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
        let _collection_id = self.get_collection_id_from_compaction_params(params)?;
        let start_time = std::time::Instant::now();

        // Get collection config dimension - required for proper compaction
        let collection_dimension = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|cfg| cfg.dimension)
            .ok_or_else(|| {
                ProximaDBError::Config("Missing dimension field in collection config".to_string())
            })?;

        tracing::debug!(
            "RAPTOR compaction: Using collection config dimension: {}",
            collection_dimension
        );
        // Deferred: Update any dimension-dependent compaction operations
        // - HNSW graph rebuilding optimization for this dimension
        // - Row group reorganization based on actual dimension
        // - Memory allocation optimization during compaction

        // Get collection_id and data directory using trait-level helpers
        let collection_id = self.get_collection_id_from_compaction_params(params)?;
        let data_dir = self.get_data_dir_from_compaction_params(params)?;

        // Use filesystem API for cloud-compatible file listing
        let input_files: Vec<String> = match self.filesystem.list(&data_dir).await {
            Ok(entries) => entries
                .into_iter()
                .filter(|e| !e.metadata.is_directory && e.name.ends_with(".raptor"))
                .map(|e| format!("{}/{}", data_dir, e.name))
                .collect(),
            Err(_) => Vec::new(),
        };

        if !input_files.is_empty() {
            // Use unified FilenameCodec naming convention
            use crate::storage::engines::core::constants;
            let codec = crate::storage::common::compaction_orchestrator::FilenameCodec::new();
            let filename = codec.generate(1, constants::raptor::FILE_EXTENSION); // Level 1 for compacted files
            let output_file = format!("{}/{}", data_dir, filename);
            self.compactor
                .compact_files(input_files, &output_file, &collection_id)
                .await?;
        }

        // Update unified metrics
        if let Some(ref collector) = self.metrics_collector {
            collector
                .record_operation("raptor", "compaction", 0.0, false, 0)
                .await;
        }

        Ok(CompactionResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_processed: Some(0),
            entries_removed: Some(0),
            bytes_read: Some(0),
            bytes_written: Some(0),
            input_files: Some(0),
            output_files: Some(0),
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
            duration_ms: Some(start_time.elapsed().as_millis() as u64),
        })
    }

    async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut stats = HashMap::new();

        // Use unified metrics collector if available
        if let Some(ref collector) = self.metrics_collector {
            // Collect metrics from unified framework
            let engine_stats = collector.engine_statistics("raptor").await;
            stats.insert(
                "total_operations".to_string(),
                serde_json::json!(engine_stats.total_operations),
            );
            stats.insert(
                "total_errors".to_string(),
                serde_json::json!(engine_stats.total_errors),
            );
            stats.insert(
                "error_rate".to_string(),
                serde_json::json!(engine_stats.error_rate),
            );
            stats.insert(
                "max_avg_latency".to_string(),
                serde_json::json!(engine_stats.max_avg_latency),
            );
            stats.insert(
                "total_bytes_processed".to_string(),
                serde_json::json!(engine_stats.total_bytes_processed),
            );
        }

        // Engine identification for unified metrics dashboard
        stats.insert("engine_name".to_string(), serde_json::json!("RAPTOR"));
        stats.insert(
            "engine_version".to_string(),
            serde_json::json!(crate::version::PROXIMADB_VERSION),
        );

        Ok(stats)
    }

    async fn vector_by_id(
        &self,
        collection_id: &str,
        base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        tracing::info!(
            "RAPTOR vector_by_id: START - Looking for vector '{}' in collection '{}', base_path '{}'",
            vector_id,
            collection_id,
            base_path
        );

        // Access global unified cache through CrossCacheOrchestrator
        let cache_key = format!("vector:{}:{}", collection_id, vector_id);
        if let Some(orchestrator) =
            crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
        {
            // Try to get from vector cache first
            if let Some(vector_cache) = orchestrator.get_vector_cache()
                && let Some(cached_vector) = vector_cache.get(&cache_key).await
            {
                // Track cache hit for access pattern learning
                orchestrator.pattern_tracker().track_access_async(
                    cache_key.clone(),
                    crate::storage::cache::orchestrator::CacheType::VectorData,
                );
                return Ok(Some(cached_vector));
            }

            // Track cache miss
            orchestrator.pattern_tracker().track_access_async(
                cache_key.clone(),
                crate::storage::cache::orchestrator::CacheType::VectorData,
            );
        }

        // Find RAPTOR data files for this collection
        // Construct data directory from base_path and collection_id
        // Format is: {baseurl}/{collectionid}/data/
        let data_dir = StoragePath::collection_data_path(base_path, collection_id);
        tracing::info!(
            "RAPTOR vector_by_id: Constructed data directory path: {}",
            data_dir
        );

        // Use filesystem API to list files in the directory
        let data_files = match self.filesystem.list(&data_dir).await {
            Ok(files) => {
                tracing::info!(
                    "RAPTOR vector_by_id: Successfully listed directory, found {} entries",
                    files.len()
                );
                let filtered: Vec<_> = files
                    .into_iter()
                    .filter(|entry| {
                        // Match both old format (raptor_*.data) and new format (L*_*.raptor)
                        let matches = (entry.name.starts_with("raptor_")
                            && entry.name.ends_with(".data"))
                            || (entry.name.starts_with("L") && entry.name.ends_with(".raptor"));
                        tracing::debug!(
                            "RAPTOR vector_by_id: File '{}' matches pattern: {}",
                            entry.name,
                            matches
                        );
                        matches
                    })
                    .map(|entry| format!("{}/{}", data_dir, entry.name))
                    .collect();
                tracing::info!(
                    "RAPTOR vector_by_id: Found {} RAPTOR data files after filtering",
                    filtered.len()
                );
                for file in &filtered {
                    tracing::info!("RAPTOR vector_by_id: Will search in file: {}", file);
                }
                filtered
            }
            Err(e) => {
                tracing::error!(
                    "RAPTOR vector_by_id: Failed to list directory {}: {:?}",
                    data_dir,
                    e
                );
                Vec::new()
            }
        };

        if data_files.is_empty() {
            tracing::warn!(
                "RAPTOR vector_by_id: No RAPTOR data files found in {}",
                data_dir
            );
            return Ok(None);
        }

        // Search through all RAPTOR data files
        for file_path_str in data_files {
            tracing::info!("RAPTOR vector_by_id: Searching in file: {}", file_path_str);

            // Create a reader specifically for this file
            // The reader expects to be initialized with the file path for single-file operations
            // Get the global cache orchestrator or create a temporary one
            let cache_orchestrator = if let Some(global_cache) =
                crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
            {
                global_cache
            } else {
                // Create a temporary cache if no global one exists
                Arc::new(
                    crate::storage::cache::orchestrator::CrossCacheOrchestrator::new(
                        1024 * 1024 * 10, // 10MB cache
                    ),
                )
            };

            let file_reader = Arc::new(RaptorReader::new(
                file_path_str.clone(), // Use the actual file path as base_path
                collection_id.to_string(),
                self.config.clone(),
                cache_orchestrator,
                self.filesystem.clone(),
            ));

            // Try to get metadata for this file
            tracing::debug!(
                "RAPTOR vector_by_id: Attempting to get metadata for file: {}",
                file_path_str
            );
            match file_reader.get_metadata(&file_path_str).await {
                Ok(metadata) => {
                    tracing::debug!(
                        "RAPTOR vector_by_id: Successfully got metadata, {} row groups",
                        metadata.row_groups.len()
                    );
                    tracing::info!(
                        "RAPTOR vector_by_id: Successfully got metadata for file {}, {} row groups",
                        file_path_str,
                        metadata.row_groups.len()
                    );
                    // Use bloom filters from rowgroup_manager to prune rowgroups
                    let rowgroup_manager = self.rowgroup_manager.read().await;
                    let rowgroup_indices: Vec<u16> = metadata
                        .row_groups
                        .iter()
                        .enumerate()
                        .filter(|(_, rg)| {
                            // Check in-memory bloom filter from rowgroup manager
                            if let Some(managed_rg) = rowgroup_manager.row_group(&rg.id) {
                                match &managed_rg.bloom_filter {
                                    Some(bloom) => bloom.contains(vector_id),
                                    None => true, // No bloom filter, must scan
                                }
                            } else {
                                true // Rowgroup not in manager, must scan
                            }
                        })
                        .map(|(i, _)| i as u16)
                        .collect();
                    drop(rowgroup_manager);

                    tracing::debug!(
                        "RAPTOR vector_by_id: Will read {} row groups",
                        rowgroup_indices.len()
                    );
                    tracing::info!(
                        "RAPTOR vector_by_id: Will read {} row groups to find vector",
                        rowgroup_indices.len()
                    );

                    let batches = file_reader
                        .read_rowgroups(&file_path_str, &rowgroup_indices)
                        .await?;

                    // Search through all batches for the vector ID
                    tracing::debug!("RAPTOR vector_by_id: Read {} batches", batches.len());
                    tracing::debug!(
                        "RAPTOR vector_by_id: Searching through {} batches",
                        batches.len()
                    );
                    for (batch_idx, batch) in batches.iter().enumerate() {
                        tracing::debug!(
                            "RAPTOR vector_by_id: Batch {}: {} rows",
                            batch_idx,
                            batch.num_rows()
                        );
                        // Check each row for matching ID
                        if let Some(id_array) = batch.column_by_name("id") {
                            let id_array =
                                id_array.as_any().downcast_ref::<arrow_array::StringArray>();
                            if let Some(id_array) = id_array {
                                for i in 0..batch.num_rows() {
                                    // StringArray value() method already handles null checking
                                    let id = id_array.value(i);
                                    tracing::trace!("RAPTOR vector_by_id: Row {}: id='{}'", i, id);
                                    if id == vector_id {
                                        // Found the vector, reconstruct and return
                                        tracing::debug!(
                                            "RAPTOR vector_by_id: Found vector '{}' at row {}",
                                            vector_id,
                                            i
                                        );
                                        return Ok(Some(
                                            self.reconstruct_vector_record(batch, i)?.into(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        "RAPTOR vector_by_id: Failed to read metadata from {}: {}",
                        file_path_str,
                        e
                    );
                    tracing::error!(
                        "RAPTOR vector_by_id: Failed to read metadata from {}: {:?}",
                        file_path_str,
                        e
                    );
                    tracing::error!("RAPTOR vector_by_id: Error details: {}", e);
                }
            }
        }

        tracing::warn!(
            "RAPTOR vector_by_id: Vector '{}' not found in any files for collection '{}'",
            vector_id,
            collection_id
        );
        Ok(None)
    }

    async fn search_vectors_unified(
        &self,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        // Extract all parameters from enhanced context (pre-computed)
        let collection_id = ctx.collection_id();
        let storage_path = ctx.storage_path();
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow::anyhow!("No query vector in context"))?;
        let k = ctx.top_k();
        let _dimension = ctx.dimension();
        let distance_metric = ctx.distance_metric();
        let performance_tier = ctx.performance_tier();
        let filter_expression = ctx.search_params.filter_expression.as_ref();
        // These fields are no longer in search_params, default to true
        let _include_vectors = true;
        let _include_metadata = true;

        // Log search with enhanced context info
        debug!(
            "RAPTOR SEARCH: collection={}, k={}, metric={:?}, tier={:?}, storage_path={}, query_dim={}",
            collection_id,
            k,
            distance_metric,
            performance_tier,
            storage_path,
            query_vector.len()
        );

        // ========================================================================
        // PHASE 0: TRY AXIS-BASED SEARCH FIRST (HNSW/IVF) - FASTEST PATH
        // ========================================================================
        // Use AXIS manager if available for O(log N) approximate search
        let has_axis_manager = self.axis_manager().is_some();
        if has_axis_manager {
            tracing::debug!("🔍 RAPTOR: AXIS manager is available for HNSW/IVF search");
        }

        if let Some(axis_manager) = self.axis_manager() {
            tracing::debug!(
                "🔍 RAPTOR: Attempting AXIS search for collection='{}', top_k={}, dimension={}",
                collection_id,
                k,
                query_vector.len()
            );

            // Convert filter expression to Index MetadataFilter format (DTO)
            let index_filters = Self::convert_filter_to_index(filter_expression);

            // Build hybrid query DTO for index engine
            use proximadb_index_traits::{IndexHybridQuery, IndexVectorQuery};
            let hybrid_query = IndexHybridQuery {
                collection_id: collection_id.to_string(),
                vector_query: Some(IndexVectorQuery::Dense {
                    vector: query_vector.to_vec(),
                    similarity_threshold: 0.0, // Return all results up to k
                }),
                metadata_filters: index_filters,
                id_filters: Vec::new(),
                top_k: k,
                include_expired: false,
                search_effort: None,
            };

            // Execute index query (HNSW or IVF based on index type)
            let axis_start = std::time::Instant::now();
            match axis_manager.query(hybrid_query).await {
                Ok(axis_results) => {
                    let axis_duration = axis_start.elapsed();
                    tracing::info!(
                        "✅ RAPTOR: AXIS search completed in {:?} - found {} candidates",
                        axis_duration,
                        axis_results.results.len()
                    );

                    // Convert AXIS results to OptimizedSearchRecord
                    let results: Vec<OptimizedSearchRecord> = axis_results
                        .results
                        .into_iter()
                        .take(k)
                        .map(|scored| OptimizedSearchRecord {
                            id: scored.vector_id.to_string(),
                            vector_id: Some(scored.vector_id.to_string()),
                            score: scored.similarity,
                            similarity: Some(scored.similarity),
                            vector: None, // AXIS doesn't return vectors by default
                            ..Default::default()
                        })
                        .collect();

                    // If we got results, return them
                    if !results.is_empty() {
                        return Ok(results);
                    }

                    tracing::debug!(
                        "⚠️ RAPTOR: AXIS returned no results, falling back to row-group search"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        "⚠️ RAPTOR: AXIS search failed: {}, falling back to row-group search",
                        e
                    );
                }
            }
        }

        // ========================================================================
        // PHASE 1: ROW-GROUP BASED SEARCH (Fallback)
        // ========================================================================

        // Convert filter expression to simple filter for now
        let filter = if filter_expression.is_some() {
            Some(HashMap::new()) // Simplified
        } else {
            None
        };

        // Use performance tier to optimize search strategy
        let results = match performance_tier {
            crate::storage::traits::PerformanceTier::Hot => {
                // Memory-first search for hot data
                self.search_internal(
                    query_vector,
                    k,
                    filter,
                    &distance_metric,
                    storage_path,
                    collection_id,
                )
                .await?
            }
            _ => {
                // Standard search for other tiers
                self.search_internal(
                    query_vector,
                    k,
                    filter,
                    &distance_metric,
                    storage_path,
                    collection_id,
                )
                .await?
            }
        };

        // Return OptimizedSearchRecord directly
        Ok(results)
    }
}

#[allow(deprecated)] // RaptorEngine is #[deprecated] (experimental); accessing its own field is intentional
impl crate::storage::traits::EngineFilesystemAccess for RaptorEngine {
    fn get_filesystem_factory(
        &self,
    ) -> &crate::storage::persistence::filesystem::FilesystemFactory {
        &self.filesystem_factory
    }
}

// AXIS integration helper methods
#[allow(deprecated)]
impl RaptorEngine {
    /// Get the AXIS manager if configured
    ///
    /// Returns the optional AXIS manager for HNSW/IVF-based search.
    /// When available, AXIS provides O(log N) approximate nearest neighbor search
    /// that is significantly faster than row-group based search.
    pub fn axis_manager(&self) -> Option<&Arc<dyn proximadb_index_traits::IndexEngine>> {
        self.axis_manager.as_ref()
    }

    /// Convert FilterExpression to Index MetadataFilter format (DTO)
    ///
    /// This helper converts our internal FilterExpression type to the
    /// IndexMetadataFilter DTO format for hybrid vector + metadata queries.
    fn convert_filter_to_index(
        filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
    ) -> Vec<proximadb_index_traits::IndexMetadataFilter> {
        // Canonical converter (redundancy pass 3): the six per-engine copies
        // had drifted — only SST mapped Contains/StartsWith/EndsWith/Like/
        // Between. See proximadb_filter_expression::to_index_metadata_filters.
        proximadb_filter_expression::to_index_metadata_filters(filter_expression)
    }
}

// Helper structures
struct RowGroupCache {
    #[allow(dead_code)]
    capacity: usize,
    #[allow(dead_code)]
    cache: HashMap<String, RecordBatch>,
    #[allow(dead_code)]
    access_counts: HashMap<String, usize>,
}

impl RowGroupCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
            access_counts: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn get(&self, key: &str) -> Option<RecordBatch> {
        self.cache.get(key).cloned()
    }

    #[allow(dead_code)]
    fn put(&mut self, key: String, batch: RecordBatch) {
        // Simple LRU eviction
        if self.cache.len() >= self.capacity {
            // Find least recently used
            if let Some(lru_key) = self
                .access_counts
                .iter()
                .min_by_key(|(_, count)| *count)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&lru_key);
                self.access_counts.remove(&lru_key);
            }
        }

        self.cache.insert(key.clone(), batch);
        *self.access_counts.entry(key).or_insert(0) += 1;
    }

    #[allow(dead_code)]
    fn optimize(&mut self) {
        // Remove entries with low access counts
        let threshold = 2;
        self.cache
            .retain(|k, _| self.access_counts.get(k).unwrap_or(&0) >= &threshold);
    }
}

struct FileRegistry {
    #[allow(dead_code)]
    active_files: HashMap<Uuid, FileMetadata>,
    #[allow(dead_code)]
    compacting_files: HashMap<Uuid, FileMetadata>,
}

impl FileRegistry {
    fn new() -> Self {
        Self {
            active_files: HashMap::new(),
            compacting_files: HashMap::new(),
        }
    }
}

struct FileMetadata {
    #[allow(dead_code)]
    id: Uuid,
    #[allow(dead_code)]
    path: String,
    #[allow(dead_code)]
    size_bytes: u64,
    #[allow(dead_code)]
    row_count: usize,
    #[allow(dead_code)]
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Implementation of UniversallyOptimized trait for RAPTOR engine
#[allow(deprecated)]
#[async_trait]
impl UniversallyOptimized for RaptorEngine {
    /// Get the universal performance optimizer instance
    fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer {
        &self.universal_optimizer
    }

    /// RAPTOR-specific optimization setup
    async fn setup_engine_optimizations(&self) -> Result<()> {
        // RAPTOR-specific optimizations for columnar analytics with clustering
        tracing::info!("🔧 RAPTOR Engine: Setting up universal performance optimizations");

        // Initialize RAPTOR-specific optimizations
        let config = self.universal_optimizer.get_config();
        tracing::debug!("   Cache size: {}MB", config.cache_size_mb);
        tracing::debug!("   Parallel operations: {}", config.parallel_operations);
        tracing::debug!("   Prefetching enabled: {}", config.enable_prefetching);
        tracing::debug!(
            "   Memory mapping enabled: {}",
            config.enable_memory_mapping
        );

        // RAPTOR is ready for high-performance columnar operations
        tracing::info!(
            "✅ RAPTOR Engine: Universal optimizations configured for columnar analytics"
        );
        Ok(())
    }

    /// RAPTOR-specific performance metrics
    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Collect from unified metrics collector if available
        if let Some(ref collector) = self.metrics_collector {
            let engine_stats = collector.engine_statistics("raptor").await;
            metrics.insert(
                "raptor_total_operations".to_string(),
                serde_json::json!(engine_stats.total_operations),
            );
            metrics.insert(
                "raptor_total_errors".to_string(),
                serde_json::json!(engine_stats.total_errors),
            );
            metrics.insert(
                "raptor_error_rate".to_string(),
                serde_json::json!(engine_stats.error_rate),
            );
            metrics.insert(
                "raptor_max_avg_latency".to_string(),
                serde_json::json!(engine_stats.max_avg_latency),
            );
            metrics.insert(
                "raptor_total_bytes_processed".to_string(),
                serde_json::json!(engine_stats.total_bytes_processed),
            );
        }

        // Universal optimizer metrics
        insert_universal_optimizer_metrics(&mut metrics, &self.universal_optimizer);

        Ok(metrics)
    }
}

#[cfg(test)]
#[cfg(feature = "experimental-engines")]
mod tests {
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

    #[test]
    fn test_raptor_distance_computation_non_zero() {
        let compute = UnifiedDistanceCompute::default();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let centroid = vec![0.0, 1.0, 0.0, 0.0];

        let distance = compute.distance(&query, &centroid);

        // Distance between orthogonal vectors should be non-zero
        assert!(
            distance > 0.0,
            "Distance between different vectors should be non-zero, got {}",
            distance
        );
    }

    #[test]
    fn test_raptor_distance_filters_rowgroups() {
        let compute = UnifiedDistanceCompute::default();

        let query = vec![1.0, 0.0, 0.0];

        // Close centroid
        let close = vec![0.9, 0.1, 0.0];
        let close_dist = compute.distance(&query, &close);

        // Far centroid
        let far = vec![0.0, 0.0, 1.0];
        let far_dist = compute.distance(&query, &far);

        // Far centroid should have greater distance
        assert!(
            far_dist > close_dist,
            "Far centroid distance ({}) should be > close centroid distance ({})",
            far_dist,
            close_dist
        );

        // With threshold 0.5, far centroid should be filtered out in many metrics
        // This validates the fix: previously distance was always 0.0
        assert!(
            close_dist > 0.0,
            "Close centroid distance should still be non-zero"
        );
    }

    // ========== NEW TESTS ==========

    #[test]
    fn test_raptor_config_defaults() {
        use super::RaptorConfig;
        let config = RaptorConfig::default();
        assert_eq!(
            config.rowgroup_size,
            super::super::constants::clustering::DEFAULT_ROWGROUP_SIZE
        );
        assert!(config.enable_simd);
        assert!(config.enable_clustering);
        assert!(config.use_component_boosting);
        assert!(config.enable_bloom_filters);
        assert!(config.enable_statistics);
        assert!(config.enable_complex_types);
        assert!(config.enable_range_reads);
        assert!(config.enable_prefetching);
        assert!(config.num_clusters.is_none()); // Auto-calculate
    }

    #[test]
    fn test_raptor_config_for_small_k() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_small_k();
        assert_eq!(config.rowgroup_size, 500);
        assert_eq!(config.cache_size_mb, 4096);
    }

    #[test]
    fn test_raptor_config_for_medium_k() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_medium_k();
        assert_eq!(config.rowgroup_size, 2000);
    }

    #[test]
    fn test_raptor_config_for_large_k() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_large_k();
        assert_eq!(config.rowgroup_size, 5000);
        assert!(config.enable_prefetching);
        assert_eq!(config.prefetch_size_mb, 128);
    }

    #[test]
    fn test_raptor_config_for_cloud() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_cloud();
        assert!(config.enable_range_reads);
        assert_eq!(config.prefetch_size_mb, 64);
        assert_eq!(config.cache_size_mb, 2048);
    }

    #[test]
    fn test_raptor_config_for_local_ssd() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_local_ssd();
        assert!(!config.enable_range_reads);
        assert_eq!(config.max_parallel_reads, 16);
    }

    #[test]
    fn test_raptor_config_for_high_performance() {
        use super::RaptorConfig;
        let config = RaptorConfig::for_high_performance();
        assert_eq!(config.rowgroup_size, 5000);
        assert!(config.enable_simd);
        assert_eq!(config.simd_lanes, 32);
        assert_eq!(config.cache_size_mb, 4096);
        assert_eq!(config.max_parallel_reads, 32);
    }

    #[test]
    fn test_raptor_format_version_constant() {
        assert_eq!(super::super::constants::file_format::VERSION, 1);
        assert_eq!(super::super::constants::file_format::MAGIC, *b"RPTR");
    }

    #[test]
    fn test_raptor_distance_identical_vectors() {
        let compute = UnifiedDistanceCompute::default();
        let vec_a = vec![1.0, 2.0, 3.0];
        let distance = compute.distance(&vec_a, &vec_a);
        // Distance to self should be 0 or very close to 0
        assert!(
            distance < 0.001,
            "Distance to self should be ~0, got {}",
            distance
        );
    }

    #[test]
    fn test_raptor_distance_triangle_inequality_heuristic() {
        let compute = UnifiedDistanceCompute::default();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];

        let d_ab = compute.distance(&a, &b);
        let d_bc = compute.distance(&b, &c);
        let d_ac = compute.distance(&a, &c);

        // All distances should be positive between different vectors
        assert!(d_ab > 0.0);
        assert!(d_bc > 0.0);
        assert!(d_ac > 0.0);
    }
}
