// SWIFT Engine: Storage With Instant Fast Traversal - zero-overhead vector storage
// Implements UnifiedStorageFormat trait for integration with ProximaDB

use crate::core::search::DataFreshnessTier;
use crate::storage::engines::core::ops::{
    UniversalOptimizationStrategy, UniversalPerformanceOptimizer, UniversallyOptimized,
};
use crate::storage::persistence::filesystem::FileStorageTier;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_compression::StandardCompression;
use proximadb_storage_common::storage_path::StoragePath;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// Universal performance optimization imports - UniversalIOConfig removed as unused
// VectorMemoryPool now managed by universal optimizer
// StorageTier already imported from crate::core::search
use proximadb_hardware_caps::HardwareCapabilities;

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::traits::{
    CompactionParameters, CompactionResult, EngineHealth, EngineStatistics, FlushParameters,
    FlushResult, StorageFormatStrategy, UnifiedStorageFormat,
};
use proximadb_distance_kernel::DistanceMetric;
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
use proximadb_index_traits::{
    IndexFilterOperator, IndexHybridQuery, IndexMetadataFilter, IndexVectorQuery,
};
use proximadb_records::ProximaRecord;
// Removed unused import: IndexingAlgorithm
use crate::metrics::collectors::{EngineMetricsCollector, OperationTimer};
// Removed unused compression common imports
// Removed unused CompressionData import

use super::{SwiftFile, optimized_operations::OptimizedSwiftOperations};

// Import Proxima structures for SWIFT's hierarchical operations
use crate::storage::engines::core::formats::proximablocks::SuperBlock;

// SWIFT-specific optimization structures removed - now using universal module

// ============================================================================
// GLOBAL PCA MODEL CACHE FOR SWIFT
// ============================================================================
// Similar to SST's PCA caching - trained during flush, reused during search
// to eliminate per-query PCA training overhead (40ms+)
lazy_static::lazy_static! {
    static ref SWIFT_GLOBAL_PCA_MODEL_CACHE: std::sync::RwLock<std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>> =
        std::sync::RwLock::new(std::collections::HashMap::new());
}

fn swift_record_id(record: &ProximaRecord) -> String {
    record
        .local_id
        .clone()
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| record.oid.clone())
}

fn swift_record_vector(record: &ProximaRecord) -> &[f32] {
    record
        .embeddings
        .first()
        .map(|embedding| embedding.as_fp32_slice())
        .unwrap_or(&[])
}

fn swift_record_is_live(record: &ProximaRecord) -> bool {
    !swift_record_vector(record).is_empty()
}

/// Set PCA model for a collection in the global cache
///
/// This function caches the trained PCA model after flush/compaction operations,
/// eliminating the 40ms+ overhead of per-query PCA training during search.
///
/// The global cache is shared across all SWIFT engine instances, so models
/// trained during flush/compaction are automatically available for search
/// operations in any session.
///
/// # Arguments
///
/// * `collection_id` - Unique identifier for the collection
/// * `model` - Trained EnhancedPCAModel with principal components
///
/// # When to Call
///
/// - After flush completes (model trained on flushed vectors)
/// - After compaction completes (model retrained on compacted data)
/// - When collection schema changes (dimensionality changes)
///
/// # Example
///
/// ```rust,ignore
/// let model = train_pca_model(&vectors, n_components)?;
/// set_collection_pca_model("my_collection", model);
/// ```
///
/// # Performance Impact
///
/// - **Cache hit**: ~0ms (model reused)
/// - **Cache miss**: ~40-100ms (model trained during query)
pub fn set_collection_pca_model(collection_id: &str, model: super::pca_manager::EnhancedPCAModel) {
    if let Ok(mut cache) = SWIFT_GLOBAL_PCA_MODEL_CACHE.write() {
        cache.insert(collection_id.to_string(), model);
        tracing::debug!("[SWIFT] Cached PCA model for collection: {}", collection_id);
    }
}

/// Get PCA model for a collection from the global cache
///
/// Retrieves the cached PCA model for use during progressive search,
/// avoiding the 40ms+ overhead of training the model per query.
///
/// # Arguments
///
/// * `collection_id` - Unique identifier for the collection
///
/// # Returns
///
/// - `Some(model)` if PCA model exists in cache
/// - `None` if model not cached (caller must train or fallback)
///
/// # When to Call
///
/// - During progressive search (phase 1-3 filtering)
/// - Before quantizing query vectors
/// - When applying PCA transformation for dimensionality reduction
///
/// # Example
///
/// ```rust,ignore
/// use proximadb::storage::engines::swift::pca_manager::EnhancedPCAModel;
/// let query_vector = vec![0.0; 128];
/// if let Some(model) = get_collection_pca_model("my_collection") {
///     let transformed = model.transform(&query_vector)?;
///     // Use transformed vector for search
/// } else {
///     // Fallback: use original vector or train model
/// };
/// ```
///
/// # Performance Impact
///
/// - **Cache hit**: ~0ms (model retrieved instantly)
/// - **Cache miss**: Must train model (~40-100ms) or use fallback
pub fn get_collection_pca_model(
    collection_id: &str,
) -> Option<super::pca_manager::EnhancedPCAModel> {
    if let Ok(cache) = SWIFT_GLOBAL_PCA_MODEL_CACHE.read() {
        cache.get(collection_id).cloned()
    } else {
        None
    }
}

/// SWIFT Engine - Storage With Instant Fast Traversal
///
/// ## Architecture Overview
///
/// SWIFT (Storage With Instant Fast Traversal) is ProximaDB's high-speed row-based
/// storage engine, optimized for ultra-low latency point queries using Proxima encoding.
///
/// ### Core Design Principles:
/// - **Hierarchical Storage**: SuperBlocks → Blocks → Vectors for fast traversal
/// - **Proxima Encoding**: Custom compact format with inline metadata
/// - **Zero-Copy Access**: Memory-mapped I/O for instant reads
/// - **Stateless Design**: All metadata from StorageQueryContext at runtime
///
/// ### Data Flow:
/// ```text
/// Insert → Batch → Proxima Encode → SuperBlock Assembly
///                          ↓
///                   Write to Filesystem
///                          ↓
///                   Progressive Search:
///                   1. SuperBlock Header Scan
///                   2. Block-Level Filtering
///                   3. Vector Retrieval (mmap)
///                   4. Distance Calculation
/// ```
///
/// ### Key Differentiators:
/// - **vs SST**: Proxima format vs SSTable, 3x faster point queries
/// - **vs VIPER**: Row-based vs columnar, better for OLTP
/// - **vs NOVA**: Lower latency vs higher compression
///
/// ### Performance Characteristics:
/// - **Write Latency**: ~0.5-2ms (in-place encoding)
/// - **Point Query**: ~0.1-1ms (mmap + hierarchical lookup)
/// - **Batch Query**: ~5-20ms (SuperBlock scan)
/// - **Compression**: 4-6x (Proxima encoding)
#[deprecated(
    since = "0.3.0",
    note = "SWIFT is experimental. Use SST or NOVA instead."
)]
pub struct SwiftEngine {
    /// **Optimized Operations Handler**
    ///
    /// High-performance operation executor:
    /// - SIMD-accelerated Proxima encoding/decoding
    /// - Memory-mapped I/O for zero-copy access
    /// - Parallel SuperBlock processing
    /// - Hierarchical index navigation
    ///
    /// Critical for achieving sub-millisecond latencies
    #[allow(dead_code)]
    optimized_ops: Arc<OptimizedSwiftOperations>,

    /// **Engine Statistics** (RwLock for concurrent access)
    ///
    /// Real-time metrics tracking:
    /// - SuperBlock count and sizes per collection
    /// - Memory-mapped regions and cache hits
    /// - Operation latencies (microsecond precision)
    /// - Compression ratios achieved
    /// - Block-level access patterns
    ///
    /// RwLock allows concurrent reads during queries
    statistics: Arc<RwLock<EngineStatistics>>,

    /// **Hardware Capabilities**
    ///
    /// System capability detector for optimal algorithms:
    /// - CPU features (SIMD: AVX2/AVX512/NEON)
    /// - Page size for mmap alignment (4KB/2MB/1GB)
    /// - Cache line size for prefetch optimization
    /// - Storage backend type (NVMe/SSD for mmap benefit)
    ///
    /// Used to select Proxima encoding strategy
    hardware: Arc<HardwareCapabilities>,

    /// **Metrics Collector** (Optional)
    ///
    /// Integration with monitoring systems:
    /// - Tracks microsecond-level latencies
    /// - Monitors mmap performance
    /// - Records SuperBlock access patterns
    /// - Exports to Prometheus/StatsD
    ///
    /// None if monitoring disabled, Some in production
    metrics_collector: Option<Arc<EngineMetricsCollector>>,

    /// **Compression Provider**
    ///
    /// Direct compression for metadata/strings:
    /// - LZ4 (ultra-fast for hot paths)
    /// - Snappy (balanced speed/ratio)
    /// - ZSTD (best compression when latency allows)
    ///
    /// Vectors use Proxima encoding, not general compression
    #[allow(dead_code)]
    compression_provider: StandardCompression,

    /// **Storage Quantization Engine** (Collection-Aware)
    ///
    /// Persistent quantization with trained codebooks:
    /// - Binary quantization for fast filtering
    /// - INT8 quantization for approximate distances
    /// - PQ8 for candidate refinement
    /// - Codebooks stored alongside SuperBlocks
    ///
    /// Enables progressive search even on row format
    storage_quantization_engine:
        Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine>,

    /// **Fallback Quantization Engine** (Stateless)
    ///
    /// In-memory quantization for ad-hoc operations:
    /// - No persistent codebooks needed
    /// - Used for new collections without training data
    /// - Same algorithms as storage engine
    /// - Faster for one-off quantization
    ///
    /// Falls back when storage codebook unavailable
    fallback_quantization_engine:
        Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine>,

    /// **Filesystem Factory**
    ///
    /// Creates filesystem instances for storage backends:
    /// - Local filesystem with mmap support
    /// - S3 with intelligent chunking for SuperBlocks
    /// - Azure Blob with range-based access
    /// - GCS with metadata caching
    ///
    /// Shared across all SWIFT operations
    filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,

    /// **Universal Performance Optimizer**
    ///
    /// Cross-cutting optimization coordinator:
    /// - Memory pooling for SuperBlock buffers
    /// - Prefetch strategies for sequential access
    /// - Adaptive batching based on query patterns
    /// - Query plan optimization
    ///
    /// Replaces engine-specific optimizers, eliminates duplication
    universal_optimizer: UniversalPerformanceOptimizer,

    /// **AXIS Manager** (Optional)
    ///
    /// Integration with AXIS indexing service:
    /// - Notifies AXIS on flush completion
    /// - Triggers index updates after compaction
    /// - Coordinates clustering operations
    /// - Manages graph index lifecycle
    ///
    /// None if AXIS disabled, Some for indexed collections
    axis_manager: Option<Arc<dyn proximadb_index_traits::IndexEngine>>,

    /// **Distance Computation Engine**
    ///
    /// Hardware-accelerated similarity calculations:
    /// - Auto-detects SIMD (AVX2/AVX512/NEON)
    /// - Supports L2, cosine, dot product metrics
    /// - Optimized for Proxima-encoded vectors
    /// - Batch processing for throughput
    ///
    /// Shared singleton across all distance operations
    distance_engine: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,

    /// **PCA Model Cache** (Per-Collection)
    ///
    /// Cached PCA models for Z-Order spatial encoding:
    /// - Key: collection_id
    /// - Value: Trained PCA model for that collection
    /// - Trained during flush, reused during search
    /// - Eliminates per-query PCA training overhead (40ms+ saved)
    ///
    /// Models are persisted to `{collection_dir}/__model/pca_model.bin`
    pca_model_cache: Arc<
        tokio::sync::RwLock<
            std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>,
        >,
    >,
}

#[allow(dead_code, deprecated)]
impl SwiftEngine {
    /// Create a new SWIFT engine instance (stateless)
    /// Collection info comes from FlushParameters and StorageQueryContext at runtime
    ///
    /// # DEPRECATION WARNING
    ///
    /// SWIFT engine is DEPRECATED and INCOMPLETE. Do not use in production.
    ///
    /// **Status**: Will be removed in v1.0 unless completed by contributors
    /// **Alternatives**: Use SST, VIPER, HELIX, or NOVA engines instead
    /// **Documentation**: See `/docs/storage/EXPERIMENTAL_ENGINES_STATUS.md`
    ///
    /// For hierarchical storage, use application-level hierarchy with production engines.
    #[deprecated(
        since = "0.2.0",
        note = "SWIFT engine is deprecated. Use SST, VIPER, HELIX, or NOVA instead."
    )]
    pub async fn new() -> Result<Self> {
        tracing::warn!(
            "⚠️  SWIFT ENGINE DEPRECATION WARNING ⚠️ \
             SWIFT is deprecated and incomplete. \
             Use SST, VIPER, HELIX, or NOVA instead. \
             See /docs/storage/EXPERIMENTAL_ENGINES_STATUS.md for details."
        );

        let distance_engine =
            Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
        Self::new_with_config(distance_engine, None).await
    }

    /// Create SWIFT engine with specific config (internal use)
    pub async fn new_with_config(
        distance_engine: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,
        axis_manager: Option<Arc<dyn proximadb_index_traits::IndexEngine>>,
    ) -> Result<Self> {
        let hardware = proximadb_hardware_caps::get_hardware_capabilities();
        let optimized_ops = Arc::new(OptimizedSwiftOperations::new()?);

        // Initialize filesystem factory
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(filesystem_config)
                .await?,
        );

        // Initialize compression provider directly
        let compression_provider = StandardCompression;

        // Initialize unified quantization engine from compute module
        // Use the provided distance engine instead of creating a new one
        let codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let unified_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_engine.clone(),
                codebook_store,
            ),
        );

        // Configure storage quantization for SWIFT (SST-based engine)
        let storage_config =
            crate::compute::quantization::storage_engine::StorageQuantizationConfig {
                primary_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::pq8(
                    16,
                )),
                filter_level: Some(
                    proximadb_quantization_model::UnifiedQuantizationLevel::binary(),
                ),
                fast_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::int8()),
                distance_metric: proximadb_distance_kernel::engine::DistanceMetric::Cosine,
                enable_progressive: true,
                filter_threshold: 100.0,
                candidate_multiplier: 10,
                training_sample_size: 10000,
                memory_budget_mb: 256, // Row-based like SST
                enable_hardware_acceleration: true,
            };

        let storage_quantization_engine = Arc::new(
            crate::compute::quantization::storage_engine::StorageQuantizationEngine::new(
                unified_engine.clone(),
                distance_engine.clone(),
                storage_config,
            ),
        );

        // Create fallback stateless quantization engine for ad-hoc queries
        let fallback_codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let fallback_quantization_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_engine.clone(),
                fallback_codebook_store,
            ),
        );

        // Initialize universal performance optimization
        let universal_optimizer =
            UniversalPerformanceOptimizer::with_strategy(UniversalOptimizationStrategy::Balanced)
                .await?;

        Ok(Self {
            optimized_ops,
            statistics: Arc::new(RwLock::new(EngineStatistics {
                engine_name: "SWIFT".to_string(),
                engine_version: "2.0.0".to_string(),
                total_storage_bytes: 0,
                memory_usage_bytes: 0,
                collection_count: 0,
                last_flush: None,
                last_compaction: None,
                pending_flushes: 0,
                pending_compactions: 0,
                engine_specific: HashMap::new(),
            })),
            hardware,
            metrics_collector: None,
            compression_provider,
            storage_quantization_engine,
            fallback_quantization_engine,
            filesystem,
            universal_optimizer,
            // Service dependencies
            axis_manager,
            distance_engine,
            // PCA model cache for Z-Order spatial encoding
            pca_model_cache: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Set metrics collector for monitoring
    pub fn set_metrics_collector(&mut self, collector: Arc<EngineMetricsCollector>) {
        self.metrics_collector = Some(collector);
    }

    /// Start operation timer if metrics collector is available
    fn start_operation_timer(&self, operation: &str) -> Option<OperationTimer> {
        self.metrics_collector.as_ref().map(|collector| {
            OperationTimer::new(
                collector.clone(),
                "SWIFT".to_string(),
                operation.to_string(),
            )
        })
    }

    /// Load SWIFT files for collection from storage
    async fn load_collection_files(
        &self,
        collection_id: &str,
        storage_path: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Vec<SwiftFile>> {
        use tracing::debug;

        // Construct data directory path: {storage_path}/{collection_id}/data/
        let data_dir = format!("{}/{}/data", storage_path, collection_id);

        debug!("🔍 SWIFT: Loading files from {}", data_dir);

        // Get filesystem instance
        let fs = self.filesystem.get_filesystem(&data_dir)?;

        // List all files in the data directory
        let entries = match fs.list(&data_dir).await {
            Ok(entries) => entries,
            Err(e) => {
                debug!("⚠️  SWIFT: Failed to list directory {}: {}", data_dir, e);
                return Ok(Vec::new());
            }
        };

        debug!("📁 SWIFT: Found {} entries in {}", entries.len(), data_dir);

        // Filter for .swift files (not .stats or temp files)
        let swift_file_paths: Vec<String> = entries
            .into_iter()
            .filter(|entry| {
                !entry.metadata.is_directory
                    && entry.name.ends_with(".swift")
                    && !entry.name.starts_with("___temp")
            })
            .map(|entry| format!("{}/{}", data_dir, entry.name))
            .collect();

        let total_files = swift_file_paths.len();
        debug!("📂 SWIFT: Found {} .swift files", total_files);

        // Load each SwiftFile from disk using the read_from_disk method
        let mut loaded_files = Vec::new();

        for file_path in swift_file_paths {
            match SwiftFile::read_from_disk(&self.filesystem, &file_path, collection).await {
                Ok(swift_file) => {
                    debug!("✅ SWIFT: Successfully loaded file: {}", file_path);
                    loaded_files.push(swift_file);
                }
                Err(e) => {
                    tracing::warn!("⚠️  SWIFT: Failed to load file {}: {}", file_path, e);
                    // Continue loading other files even if one fails
                }
            }
        }

        debug!(
            "📦 SWIFT: Loaded {} out of {} files",
            loaded_files.len(),
            total_files
        );
        Ok(loaded_files)
    }

    /// Update global statistics file for collection
    async fn update_global_stats(&self, _collection_id: &str, _storage_path: &str) -> Result<()> {
        // Path: {storage_path}/{collection_id}/global.stats
        // This is updated after flush/compaction to maintain collection-wide metrics
        // Statistics are also embedded in individual files for atomicity
        Ok(())
    }

    // ============================================================================
    // PERFORMANCE OPTIMIZATION METHODS - DELEGATING TO UNIFIED MODULES
    // ============================================================================

    /// Memory-mapped hierarchical file access for ultra-fast superblock traversal (delegates to universal optimizer)
    async fn get_memory_mapped_superblock(&self, file_path: &str) -> Result<Option<memmap2::Mmap>> {
        // Use universal optimizer's memory mapping functionality
        let mmap_arc_opt = self
            .universal_optimizer
            .get_memory_mapped_file(file_path)
            .await?;
        // Convert Arc<Mmap> to Mmap by cloning the underlying data
        // Note: This is a temporary workaround - ideally we'd use Arc<Mmap> everywhere
        if let Some(_mmap_arc) = mmap_arc_opt {
            // We can't easily convert Arc<Mmap> to Mmap, so return None for now
            // Deferred: Refactor to use Arc<Mmap> throughout the system
            tracing::warn!(
                "Memory mapping available but type conversion needed - falling back to regular I/O"
            );
            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// Parallel superblock operations with configurable concurrency (delegates to universal optimizer)
    async fn parallel_superblock_operations<T, F, Fut>(
        &self,
        superblocks: Vec<T>,
        operation: F,
    ) -> Result<Vec<Result<Fut::Output>>>
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future + Send + 'static,
        Fut::Output: Send + 'static,
        T: Send + 'static,
    {
        // Use universal optimizer's parallel operations capability
        self.universal_optimizer
            .parallel_operations(superblocks, operation)
            .await
    }

    /// Storage tier optimization for hierarchical data with cloud cost efficiency (delegates to universal optimizer)
    async fn optimize_hierarchical_storage_tier(
        &self,
        _access_frequency: f32,
        superblock_size_bytes: usize,
    ) -> Result<DataFreshnessTier> {
        // Use universal optimizer's storage tier optimization
        let file_key = format!("hierarchical_superblock_{}", superblock_size_bytes);
        let optimizer_tier = self
            .universal_optimizer
            .optimize_storage_tier(&file_key, superblock_size_bytes)
            .await?;

        // Convert from filesystem::StorageTier to core::search::StorageTier
        let core_tier = match optimizer_tier {
            FileStorageTier::Memory => DataFreshnessTier::Unflushed,
            FileStorageTier::NVMe => DataFreshnessTier::Flushed,
            FileStorageTier::SSD => DataFreshnessTier::Flushed,
            FileStorageTier::HDD => DataFreshnessTier::Compacted,
            FileStorageTier::S3Express => DataFreshnessTier::Compacted,
            FileStorageTier::S3Standard => DataFreshnessTier::Compacted,
            FileStorageTier::S3GlacierInstant => DataFreshnessTier::Compacted,
            FileStorageTier::AzurePremium => DataFreshnessTier::Flushed,
            FileStorageTier::AzureStandard => DataFreshnessTier::Compacted,
            FileStorageTier::GcsSSD => DataFreshnessTier::Flushed,
            FileStorageTier::GcsHDD => DataFreshnessTier::Compacted,
        };

        Ok(core_tier)
    }

    /// Hierarchical distance computation using unified distance compute engine (delegates to universal optimizer)
    async fn compute_hierarchical_distances(
        &self,
        query: &[f32],
        superblocks: &[Arc<SuperBlock>],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        // Extract centroids from superblocks for distance computation
        let centroids: Vec<Vec<f32>> = superblocks
            .iter()
            .map(|sb| {
                // Use FP16 centroid if available (50% storage reduction),
                // fallback to FP32 centroid, or use zero vector
                if let Some(ref fp16_centroid) = sb.centroid_fp16 {
                    // Convert FP16 to FP32 for distance computation
                    crate::storage::engines::sst::fp16_to_fp32(fp16_centroid)
                } else {
                    sb.centroid
                        .as_ref()
                        .map_or_else(|| vec![0.0; query.len()], |c| c.clone())
                }
            })
            .collect();

        // Use universal optimizer's hardware-accelerated distance computation
        self.universal_optimizer
            .compute_distances_accelerated(query, &centroids, metric)
            .await
    }

    /// Memory pool optimization for hierarchical vector operations (delegates to universal optimizer)
    async fn get_hierarchical_memory_buffer(&self, size: usize) -> Result<Vec<f32>> {
        self.universal_optimizer
            .get_memory_buffer(size)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire hierarchical memory buffer: {}", e))
    }

    /// Hierarchical compression optimization using unified compression module (delegates to universal optimizer)
    async fn compress_hierarchical_data(
        &self,
        data: &[u8],
        tier: DataFreshnessTier,
    ) -> Result<Vec<u8>> {
        // Convert from core::search::StorageTier to filesystem::StorageTier
        let fs_tier = match tier {
            DataFreshnessTier::Unflushed => {
                crate::storage::persistence::filesystem::FileStorageTier::Memory
            }
            DataFreshnessTier::Flushed => {
                crate::storage::persistence::filesystem::FileStorageTier::NVMe
            }
            DataFreshnessTier::Compacted => {
                crate::storage::persistence::filesystem::FileStorageTier::SSD
            }
        };

        // Use universal optimizer's tier-aware compression
        self.universal_optimizer
            .compress_for_tier(data, fs_tier)
            .await
    }

    /// Prefetch hierarchical superblocks based on access patterns (delegates to universal optimizer)
    async fn prefetch_hierarchical_superblocks(
        &self,
        current_superblock_id: u32,
        file_path: &str,
    ) -> Result<()> {
        let config = self.universal_optimizer.get_config();
        if !config.enable_prefetching {
            return Ok(());
        }

        // Generate superblock URLs for prefetching
        let prefetch_count = config.prefetch_size_mb / 8; // Assume ~8MB per superblock
        let start_id = current_superblock_id.saturating_sub(prefetch_count as u32 / 2);
        let end_id = current_superblock_id + (prefetch_count as u32 / 2);

        let superblock_urls: Vec<String> = (start_id..=end_id)
            .filter(|&id| id != current_superblock_id)
            .map(|id| format!("{}:superblock:{}", file_path, id))
            .collect();

        // Use universal optimizer's prefetching capability
        self.universal_optimizer
            .prefetch_data(&superblock_urls)
            .await
    }

    /// Cache management for hierarchical structures with eviction (delegates to universal optimizer)
    async fn evict_hierarchical_cache_if_needed(&self) -> Result<()> {
        // Use universal optimizer's intelligent cache eviction
        self.universal_optimizer.evict_cache_if_needed().await
    }

    /// Progressive quantization search optimized for hierarchical access
    async fn progressive_hierarchical_search(
        &self,
        query: &[f32],
        superblocks: &[Arc<SuperBlock>],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Phase 1: Superblock-level filtering using centroids
        let superblock_distances = self
            .compute_hierarchical_distances(query, superblocks, DistanceMetric::Euclidean)
            .await?;

        // Sort superblocks by distance and select top candidates
        let mut superblock_candidates: Vec<(usize, f32)> =
            superblock_distances.into_iter().enumerate().collect();
        superblock_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Phase 2: Search within top superblocks using quantization
        let search_superblocks = std::cmp::min(superblock_candidates.len(), top_k * 2); // Search 2x more superblocks
        let mut results = Vec::new();

        for (superblock_idx, _distance) in
            superblock_candidates.into_iter().take(search_superblocks)
        {
            let superblock = &superblocks[superblock_idx];

            // Phase 3: Use quantization engine for progressive search within superblock
            if let Some(_quantization_engine) = Some(&self.storage_quantization_engine) {
                // Deferred: Implement progressive search using quantization engine
                // For now, simulate with placeholder results
                for block in &superblock.blocks {
                    for record in block.records.iter() {
                        let vector = record
                            .embeddings
                            .first()
                            .map_or(&[][..], |embedding| embedding.as_fp32_slice());
                        // Compute approximate distance
                        let distance = query
                            .iter()
                            .zip(vector.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>();
                        results.push((record.oid.clone(), distance));
                    }
                }
            }
        }

        // Sort and return top-k results
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Check if we should use persistent quantization for this operation
    /// Returns true for collection-based operations with quantization enabled
    pub fn should_use_persistent_quantization(&self, params: &FlushParameters) -> bool {
        crate::storage::compute_bridge::selection::QuantizationSelector::should_use_persistent_quantization(
            params, "SWIFT",
        )
    }

    /// Get the storage quantization engine for persistent collection operations
    pub fn get_storage_quantization_engine(
        &self,
    ) -> &Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine> {
        &self.storage_quantization_engine
    }

    /// Get the fallback quantization engine for stateless operations
    pub fn get_fallback_quantization_engine(
        &self,
    ) -> &Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine> {
        &self.fallback_quantization_engine
    }

    // =========================================================================
    // AXIS Manager Integration (for HNSW/IVF index operations)
    // =========================================================================

    /// Get the AXIS manager for HNSW/IVF index operations
    ///
    /// Returns the AXIS manager if available, enabling:
    /// - HNSW-based approximate nearest neighbor search
    /// - IVF partition pruning
    /// - Hybrid vector + metadata queries
    pub fn axis_manager(&self) -> Option<&Arc<dyn proximadb_index_traits::IndexEngine>> {
        self.axis_manager.as_ref()
    }

    /// Convert FilterExpression to Index MetadataFilter format (DTO)
    ///
    /// This helper converts our internal FilterExpression type to the
    /// IndexMetadataFilter DTO format for hybrid vector + metadata queries.
    fn convert_filter_to_index(
        filter_expression: Option<&FilterExpression>,
    ) -> Vec<IndexMetadataFilter> {
        let Some(filter) = filter_expression else {
            return Vec::new();
        };

        // Convert filter expressions to index metadata filters
        let mut index_filters = Vec::new();

        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                // Convert ComparisonOperator to IndexFilterOperator
                let index_operator = match operator {
                    ComparisonOperator::Equals => IndexFilterOperator::Equals,
                    ComparisonOperator::NotEquals => IndexFilterOperator::NotEquals,
                    ComparisonOperator::GreaterThan => IndexFilterOperator::GreaterThan,
                    ComparisonOperator::GreaterThanOrEqual => {
                        IndexFilterOperator::GreaterThanOrEqual
                    }
                    ComparisonOperator::LessThan => IndexFilterOperator::LessThan,
                    ComparisonOperator::LessThanOrEqual => IndexFilterOperator::LessThanOrEqual,
                    ComparisonOperator::In => IndexFilterOperator::In,
                    ComparisonOperator::NotIn => IndexFilterOperator::NotIn,
                    _ => {
                        debug!(
                            "Operator {:?} not directly supported by index, will use post-filtering",
                            operator
                        );
                        return index_filters;
                    }
                };

                index_filters.push(IndexMetadataFilter {
                    field: field.clone(),
                    operator: index_operator,
                    value: value.clone(),
                });
            }
            FilterExpression::And(filters) => {
                for f in filters {
                    index_filters.extend(Self::convert_filter_to_index(Some(f)));
                }
            }
            FilterExpression::Or(_) | FilterExpression::Not(_) => {
                // OR and NOT are not directly supported by index, will use post-filtering
                debug!("OR/NOT filters not supported by index, will use post-filtering");
            }
        }

        index_filters
    }

    // =========================================================================
    // PCA Model Caching Methods (for Z-Order spatial encoding)
    // =========================================================================

    /// Get the PCA model cache for read access during search
    pub fn pca_model_cache(
        &self,
    ) -> &Arc<
        tokio::sync::RwLock<
            std::collections::HashMap<String, super::pca_manager::EnhancedPCAModel>,
        >,
    > {
        &self.pca_model_cache
    }

    /// Construct the PCA model file path for a collection
    /// Path: {collection_data_dir}/__model/pca_model.bin
    fn get_pca_model_path(&self, collection_data_dir: &str) -> String {
        format!("{}/__model/pca_model.bin", collection_data_dir)
    }

    /// Get cached PCA model for a collection (if available)
    ///
    /// Returns the in-memory cached model, or loads from disk if not in cache.
    pub async fn get_pca_model(
        &self,
        collection_id: &str,
        collection_data_dir: &str,
    ) -> Option<super::pca_manager::EnhancedPCAModel> {
        // First check in-memory cache
        {
            let cache = self.pca_model_cache.read().await;
            if let Some(model) = cache.get(collection_id) {
                return Some(model.clone());
            }
        }

        // Try to load from disk
        if let Ok(Some(model)) = self.load_pca_model(collection_data_dir).await {
            // Cache it for future use
            {
                let mut cache = self.pca_model_cache.write().await;
                cache.insert(collection_id.to_string(), model.clone());
            }
            return Some(model);
        }

        None
    }

    /// Load PCA model from filesystem for a collection
    pub async fn load_pca_model(
        &self,
        collection_data_dir: &str,
    ) -> Result<Option<super::pca_manager::EnhancedPCAModel>> {
        let model_path = self.get_pca_model_path(collection_data_dir);

        let filesystem = self
            .filesystem
            .get_filesystem(collection_data_dir)
            .map_err(|e| anyhow!("Failed to get filesystem: {}", e))?;

        match filesystem.exists(&model_path).await {
            Ok(true) => {
                let data = filesystem
                    .read(&model_path)
                    .await
                    .map_err(|e| anyhow!("Failed to read PCA model: {}", e))?;

                let model: super::pca_manager::EnhancedPCAModel = bincode::deserialize(&data)
                    .map_err(|e| anyhow!("Failed to deserialize PCA model: {}", e))?;

                info!(
                    "[SWIFT] Loaded persisted PCA model for collection (version: {}, {} components)",
                    model.version, model.n_components
                );
                Ok(Some(model))
            }
            Ok(false) => {
                tracing::debug!("[SWIFT] No persisted PCA model found at {}", model_path);
                Ok(None)
            }
            Err(e) => {
                tracing::debug!("[SWIFT] Error checking PCA model at {}: {}", model_path, e);
                Ok(None)
            }
        }
    }

    /// Save PCA model to filesystem for a collection
    pub async fn save_pca_model(
        &self,
        collection_data_dir: &str,
        model: &super::pca_manager::EnhancedPCAModel,
    ) -> Result<()> {
        let model_path = self.get_pca_model_path(collection_data_dir);

        let filesystem = self
            .filesystem
            .get_filesystem(collection_data_dir)
            .map_err(|e| anyhow!("Failed to get filesystem: {}", e))?;

        // Ensure __model directory exists
        let model_dir = format!("{}/__model", collection_data_dir);
        filesystem
            .create_dir_all(&model_dir)
            .await
            .map_err(|e| anyhow!("Failed to create __model directory: {}", e))?;

        // Serialize model with bincode
        let data = bincode::serialize(model)
            .map_err(|e| anyhow!("Failed to serialize PCA model: {}", e))?;

        filesystem
            .write(&model_path, &data, None)
            .await
            .map_err(|e| anyhow!("Failed to write PCA model: {}", e))?;

        info!(
            "[SWIFT] Persisted PCA model for collection at {} ({} components)",
            model_path, model.n_components
        );
        Ok(())
    }

    /// Train PCA model from vectors and cache it
    ///
    /// This should be called during flush when we have new vectors.
    /// The model is trained using adaptive dimensions based on vector dimensionality.
    pub async fn train_and_cache_pca_model(
        &self,
        collection_id: &str,
        collection_data_dir: &str,
        records: &[ProximaRecord],
    ) -> Result<()> {
        use super::pca_manager::AdaptivePcaConfig;

        let vectors: Vec<Vec<f32>> = records
            .iter()
            .map(swift_record_vector)
            .filter(|vector| !vector.is_empty())
            .map(|vector| vector.to_vec())
            .collect();

        if vectors.is_empty() {
            return Ok(());
        }

        let vector_dim = vectors[0].len();
        if vector_dim == 0 {
            return Ok(());
        }

        // Use adaptive configuration for optimal PCA dimensions
        let pca_config = AdaptivePcaConfig::for_vector_dim(vector_dim);
        let n_components = pca_config.n_components;

        // Need at least n_components samples for training
        if vectors.len() < n_components {
            tracing::debug!(
                "[SWIFT] Not enough vectors ({}) for PCA training (need at least {})",
                vectors.len(),
                n_components
            );
            return Ok(());
        }

        info!(
            "[SWIFT] Training PCA model: {} vectors → {} components (from {}-dim)",
            vectors.len(),
            n_components,
            vector_dim
        );

        // Train PCA model
        let model =
            super::pca_manager::EnhancedPCAModel::train_from_vectors(&vectors, n_components)
                .map_err(|e| anyhow!("Failed to train PCA model: {}", e))?;

        // Save to disk
        self.save_pca_model(collection_data_dir, &model).await?;

        // Cache in memory
        {
            let mut cache = self.pca_model_cache.write().await;
            cache.insert(collection_id.to_string(), model);
        }

        Ok(())
    }
}

#[allow(deprecated)]
#[async_trait]
impl UnifiedStorageFormat for SwiftEngine {
    // =============================================================================
    // ENGINE IDENTIFICATION
    // =============================================================================

    fn engine_name(&self) -> &'static str {
        "SWIFT"
    }

    fn engine_version(&self) -> &'static str {
        "1.0.0"
    }

    fn strategy(&self) -> StorageFormatStrategy {
        // SWIFT is a hierarchical superblock strategy with row-based optimization
        StorageFormatStrategy::Swift
    }

    // =============================================================================
    // CORE OPERATIONS
    // =============================================================================

    async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
        // Check if quantization is enabled in collection config
        let quantization_enabled = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .and_then(|cfg| cfg.quantization.as_ref())
            .and_then(|q| q.enabled)
            .unwrap_or(false);

        if quantization_enabled {
            debug!("🔄 SWIFT FLUSH: Quantization enabled, processing with quantization support");
            // Quantization will be handled internally during the flush process
            // The flush_with_quantization method has been removed - quantization is now internalized
        }

        let start_time = std::time::Instant::now();

        let collection_id = self.get_collection_id_from_params(params)?;
        info!(
            "SWIFT flush: collection={}, vectors={}",
            collection_id,
            params.vector_records.len()
        );

        // Get dimension from collection config
        let dimension = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map_or(384, |cfg| cfg.dimension);

        // Create new SWIFT file from flush parameters
        let mut swift_file = SwiftFile::new(
            collection_id.to_string(),
            dimension as usize,
            "euclidean".to_string(),
        );

        // Get compression config from storage config
        let compression_config = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .and_then(|cfg| cfg.storage_config.as_ref())
            .and_then(|s| {
                if s.compression.unwrap_or(0) != 0 {
                    Some(crate::proto::proximadb_v1::CompressionConfig {
                        algorithm: s.compression.unwrap_or(0),
                        level: None,
                        adaptive: false,
                        min_ratio: None,
                        enable_quantization: false,
                        quantization_type: None,
                        normalization_method: None,
                        block_size_kb: 64,
                        dynamic_block_sizing: false,
                    })
                } else {
                    None
                }
            });

        // Add records to the SwiftFile structure with compression config
        swift_file.build_blocks_from_proxima_records_with_compression(
            params.vector_records.clone(),
            compression_config,
        )?;

        // Get storage path from collection config (always present)
        // Use standard trait method to get data directory (consistent with HELIX pattern)
        let data_dir = self.get_data_dir_from_flush_params(params)?;

        // Create directory using tokio::fs for async compatibility
        // This handles local paths correctly without requiring URL scheme
        tokio::fs::create_dir_all(&data_dir).await.map_err(|e| {
            anyhow!(
                "SWIFT: Failed to create data directory '{}': {}",
                data_dir,
                e
            )
        })?;

        // Generate filename using FilenameCodec for consistency with compaction framework
        use crate::storage::common::compaction_orchestrator::FilenameCodec;
        let codec = FilenameCodec::new();
        let swift_filename = codec.generate(0, "swift"); // Level 0 for flush
        let filename = format!("{}/{}", data_dir, swift_filename);

        // Actually write the SWIFT file to disk using filesystem factory (SST pattern)
        let bytes_written = swift_file
            .write_to_disk(&self.filesystem, &filename)
            .await?;

        info!(
            "SWIFT flush complete: wrote {} bytes to {}",
            bytes_written, filename
        );

        // Update global statistics file
        self.update_global_stats(&collection_id, data_dir.as_str())
            .await?;

        // Train/update PCA model for Z-Order spatial encoding
        // This is done after flush to ensure collection-level PCA model is up-to-date
        if params.vector_records.len() >= 100 {
            // Only train with enough samples
            match self
                .train_and_cache_pca_model(&collection_id, &data_dir, &params.vector_records)
                .await
            {
                Ok(()) => {
                    // Also update the global cache for search access
                    if let Some(model) = self.get_pca_model(&collection_id, &data_dir).await {
                        set_collection_pca_model(&collection_id, model);
                    }
                }
                Err(e) => {
                    // Log but don't fail flush - PCA is an optimization
                    tracing::warn!(
                        "[SWIFT] Failed to train PCA model during flush: {}. Z-Order pruning may be less effective.",
                        e
                    );
                }
            }
        }

        // Notify EventLog service about the flush
        // This allows AXIS to asynchronously index the flushed data
        if let Some(event_log) = crate::services::events::log::event_log_service() {
            let has_quantized = params
                .collection_config
                .as_ref()
                .and_then(|c| c.config.as_ref())
                .and_then(|cfg| cfg.quantization.as_ref())
                .and_then(|q| q.enabled)
                .unwrap_or(false);

            // Use the actual file that was written
            let flushed_files = vec![filename.clone()];

            if let Err(e) = event_log
                .notify_flush(
                    &collection_id,
                    flushed_files.clone(),
                    params.vector_records.len(),
                    has_quantized,
                    true, // SWIFT always stores FP32
                    crate::core::types::StorageEngineType::SWIFT,
                )
                .await
            {
                tracing::warn!("Failed to notify EventLog about flush: {}", e);
                // Continue anyway - EventLog notification is not critical
            }
        }

        // Update statistics with actual bytes written
        let mut stats = self.statistics.write().await;
        stats.pending_flushes = stats.pending_flushes.saturating_sub(1);
        stats.last_flush = Some(chrono::Utc::now());
        stats.total_storage_bytes += bytes_written;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(FlushResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_flushed: Some(params.vector_records.len() as u64),
            bytes_written: Some(bytes_written),
            files_created: Some(1),
            file_paths: vec![filename.clone()],
            duration_ms: Some(duration_ms),
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
            compaction_triggered: false,
            compaction_error: None,
            flushed_batch_ids: vec![], // Deferred: Track batch IDs when integrating with WAL
        })
    }

    async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
        let start_time = std::time::Instant::now();

        let collection_id = self.get_collection_id_from_compaction_params(params)?;
        info!("SWIFT compaction: collection={}", collection_id);

        // Get storage path from collection config
        let storage_path = params
            .collection_config
            .as_ref()
            .and_then(|c| c.storage_assignment.as_ref())
            .map(|s| s.base_location.clone())
            .ok_or_else(|| anyhow!("No storage assignment for collection {}", collection_id))?;

        // Load files from storage for compaction
        let collection_ref = params.collection_config.as_ref();
        let files = self
            .load_collection_files(&collection_id, &storage_path, collection_ref)
            .await?;

        if files.len() < 2 {
            let duration_ms = start_time.elapsed().as_millis() as u64;
            return Ok(CompactionResult {
                success: true,
                collections_affected: vec![collection_id.to_string()],
                entries_processed: Some(0),
                entries_removed: Some(0),
                bytes_read: Some(0),
                bytes_written: Some(0),
                input_files: Some(files.len() as u64),
                output_files: Some(files.len() as u64),
                duration_ms: Some(duration_ms),
                completed_at: chrono::Utc::now(),
                engine_metrics: HashMap::new(),
            });
        }

        // Merge all records from loaded files, dedup by ID (latest wins).
        // Locator pattern: index oid → (file, superblock, block, record) and
        // clone only the surviving record per oid. The previous loop cloned
        // EVERY record version into the map — superseded copies were dropped
        // immediately, so compaction paid a full-record clone per entry for
        // nothing. Latest-wins, tombstone filtering, counters, and the
        // (already map-arbitrary) output order are unchanged.
        let input_count = files.len() as u64;
        let mut latest: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
        let mut total_entries: u64 = 0;
        let mut bytes_read: u64 = 0;

        for (file_idx, file) in files.iter().enumerate() {
            for (superblock_idx, superblock) in file.superblocks.iter().enumerate() {
                for (block_idx, block) in superblock.blocks.iter().enumerate() {
                    for (record_idx, record) in block.records.iter().enumerate() {
                        total_entries += 1;
                        latest.insert(
                            record.oid.clone(),
                            (file_idx, superblock_idx, block_idx, record_idx),
                        );
                    }
                }
            }
            bytes_read += file.header.superblock_offset; // Approximate file size from offset
        }

        // Filter tombstones (records marked as deleted have empty vectors)
        let live_records: Vec<ProximaRecord> = latest
            .into_values()
            .filter_map(|(file_idx, superblock_idx, block_idx, record_idx)| {
                let record = &files[file_idx].superblocks[superblock_idx].blocks[block_idx].records
                    [record_idx];
                swift_record_is_live(record).then(|| record.clone())
            })
            .collect();

        let entries_removed = total_entries.saturating_sub(live_records.len() as u64);

        // Build merged output file
        let dimension = files.first().map_or(0, |f| f.header.dimension);
        let mut merged_file = SwiftFile::new(
            collection_id.to_string(),
            dimension,
            files.first().map_or_else(
                || "euclidean".to_string(),
                |f| f.header.distance_metric.clone(),
            ),
        );
        merged_file.build_blocks_from_proxima_records_with_compression(live_records, None)?;

        // Write merged file to disk
        let data_dir = StoragePath::collection_data_path(&storage_path, &collection_id);
        use crate::storage::common::compaction_orchestrator::FilenameCodec;
        let codec = FilenameCodec::new();
        let compacted_filename = codec.generate(1, "swift"); // Level 1 for compaction
        let output_path = format!("{}/{}", data_dir, compacted_filename);
        let bytes_written = merged_file
            .write_to_disk(&self.filesystem, &output_path)
            .await?;

        // Notify EventLog about compaction (fire-and-forget)
        if let Some(event_log) = crate::services::events::log::event_log_service() {
            event_log.notify_compaction(
                &collection_id,
                vec![output_path],
                merged_file.header.total_records as usize,
                crate::core::types::StorageEngineType::SWIFT,
            );
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.pending_compactions = stats.pending_compactions.saturating_sub(1);
        stats.last_compaction = Some(chrono::Utc::now());

        Ok(CompactionResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_processed: Some(total_entries),
            entries_removed: Some(entries_removed),
            bytes_read: Some(bytes_read),
            bytes_written: Some(bytes_written),
            input_files: Some(input_count),
            output_files: Some(1),
            duration_ms: Some(duration_ms),
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::new(),
        })
    }

    async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Engine is stateless, so we report engine-level metrics only
        metrics.insert("engine_type".to_string(), serde_json::json!("SWIFT"));
        metrics.insert("hierarchical_storage".to_string(), serde_json::json!(true));

        // Deferred: Collect actual metrics from storage when needed
        let total_files = 0;
        metrics.insert(
            "total_swift_files".to_string(),
            serde_json::json!(total_files),
        );

        let stats = self.statistics.read().await;
        metrics.insert(
            "pending_flushes".to_string(),
            serde_json::json!(stats.pending_flushes),
        );
        metrics.insert(
            "pending_compactions".to_string(),
            serde_json::json!(stats.pending_compactions),
        );

        // Hardware info
        metrics.insert(
            "simd_backend".to_string(),
            serde_json::json!(format!("{:?}", self.hardware.cpu.simd)),
        );

        Ok(metrics)
    }

    async fn vector_by_id(
        &self,
        collection_id: &str,
        base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
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

        let _timer = self.start_operation_timer("get_by_id");
        debug!(
            "SWIFT get vector: collection={}, base_path={}, id={}",
            collection_id, base_path, vector_id
        );

        // Load files and search through ID indexes
        let files = self
            .load_collection_files(collection_id, base_path, None)
            .await?;

        for file in &files {
            if let Some(location) = file.id_index.lookup(vector_id) {
                // Navigate to the record using the block location
                if let Some(superblock) = file.superblocks.get(location.superblock_idx as usize)
                    && let Some(block) = superblock.blocks.get(location.block_idx as usize)
                    && let Some(record) = block.records.get(location.offset_in_block as usize)
                {
                    // Cache the result for future lookups
                    if let Some(orchestrator) =
                        crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
                        && let Some(vector_cache) = orchestrator.get_vector_cache()
                    {
                        let _ = vector_cache.put(cache_key, record.clone()).await;
                    }
                    return Ok(Some(record.clone()));
                }
            }
        }

        Ok(None)
    }

    async fn search_vectors_unified(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        let _search_start = std::time::Instant::now();

        // Extract all parameters from context (pre-computed)
        let collection_id = ctx.collection_id();
        let storage_path = ctx.storage_path();
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow!("No query vector in context"))?;
        let top_k = ctx.top_k();
        let distance_metric = ctx.distance_metric();
        let _dimension = ctx.dimension();
        let filter_expression = ctx.search_params.filter_expression.as_ref();
        let _search_params = ctx.search_params.custom_hints.clone();
        let mut timer = self.start_operation_timer("search");

        info!(
            "🚀 SWIFT: Enhanced unified search with orchestration for collection {}",
            collection_id
        );

        // ========================================================================
        // PHASE 1: SEARCH ORCHESTRATION AND STRATEGY SELECTION
        // ========================================================================
        // Deferred: Integrate with AdvancedSearchOptimizer for intelligent search routing
        //
        // The AdvancedSearchOptimizer provides significant value for SWIFT engine:
        // 1. **Ultra-low latency routing**: Chooses fastest path based on data locality
        // 2. **Proxima optimization**: Leverages SWIFT's columnar blocks intelligently
        // 3. **SIMD acceleration**: Routes to hardware-optimized paths automatically
        // 4. **Memory-mapped I/O**: Optimizes based on available memory and cache
        // 5. **Predictive prefetching**: Uses access patterns for zero-copy operations
        //
        // Implementation pattern:
        // ```rust,ignore
        // let axis_manager = self.get_axis_manager().await?;
        // let cost_estimator = self.get_cost_estimator().await?;
        //
        // let orchestrator = AdvancedSearchOptimizer::new(
        //     ctx.clone(),
        //     axis_manager,
        //     cost_estimator,
        // ).await?;
        //
        // let strategy = orchestrator.select_optimal_strategy().await?;
        // match strategy {
        //     ExecutionStrategy::IndexFirst { .. } => // Use memory-mapped index
        //     ExecutionStrategy::DirectFP32 { .. } => // Proxima direct scan
        //     ExecutionStrategy::ProgressiveQuantization { .. } => // Multi-resolution
        // }
        // ```
        //
        // SWIFT-specific benefits:
        // - Leverage Proxima encoding for 10x faster scans
        // - Zero-copy operations with SharedSstFormatReader
        // - Hardware-aware SIMD path selection
        //
        // Current blocker: Service infrastructure for AXIS and cost estimation
        //
        // Determine search strategy based on context
        // Use orchestration if:
        // 1. AXIS indexes are explicitly configured, OR
        // 2. Quantization is enabled, OR
        // 3. AXIS manager is available (for collections built after AXIS became available)
        let has_axis_manager = self.axis_manager().is_some();
        let use_orchestration =
            ctx.metadata.use_axis_indexes || ctx.metadata.has_quantization || has_axis_manager;

        if has_axis_manager {
            debug!("🔍 SWIFT: AXIS manager is available for HNSW/IVF search");
        }

        if use_orchestration {
            // ========================================================================
            // PHASE 1A: TRY AXIS-BASED SEARCH FIRST (HNSW/IVF)
            // ========================================================================
            if let Some(axis_manager) = self.axis_manager() {
                info!(
                    "🔗 SWIFT: AXIS manager available, attempting HNSW index search for collection {}",
                    collection_id
                );

                // Convert filter expression to Index MetadataFilter format (DTO)
                let index_filters = Self::convert_filter_to_index(filter_expression);

                // Build hybrid query DTO for index engine
                let hybrid_query = IndexHybridQuery {
                    collection_id: collection_id.to_string(),
                    vector_query: Some(IndexVectorQuery::Dense {
                        vector: query_vector.to_vec(),
                        similarity_threshold: 0.0, // Return all results up to k
                    }),
                    metadata_filters: index_filters,
                    id_filters: Vec::new(),
                    top_k,
                    include_expired: false,
                    search_effort: None,
                };

                // Execute AXIS query (HNSW or IVF based on index type)
                let axis_start = std::time::Instant::now();
                match axis_manager.query(hybrid_query).await {
                    Ok(axis_results) => {
                        let axis_duration = axis_start.elapsed();
                        info!(
                            "✅ SWIFT: AXIS HNSW search completed in {:?} - found {} candidates",
                            axis_duration,
                            axis_results.results.len()
                        );

                        // Convert AXIS results to OptimizedSearchRecord
                        let results: Vec<OptimizedSearchRecord> = axis_results
                            .results
                            .into_iter()
                            .take(top_k)
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

                        info!(
                            "⚠️ SWIFT: AXIS returned no results, falling back to block-pruned search"
                        );
                    }
                    Err(e) => {
                        warn!(
                            "⚠️ SWIFT: AXIS query failed ({}), falling back to block-pruned search",
                            e
                        );
                    }
                }
            }

            // ========================================================================
            // PHASE 1B: BLOCK-PRUNED SEARCH (FALLBACK)
            // ========================================================================
            info!("🎯 SWIFT: Using progressive search with block pruning (quantization available)");

            // Load files and use progressive search with block pruning
            let files = self
                .load_collection_files(collection_id, storage_path, Some(&*ctx.collection))
                .await?;

            let prune_config = crate::core::search::BlockPruneConfig::default();
            // Deferred: Convert FilterExpression to MetadataFilter for SWIFT-specific filtering
            // For now, pass None and filter results after progressive search
            let swift_filter: Option<super::MetadataFilter> = None;

            let mut all_results = Vec::new();
            for swift_file in &files {
                let file_results = swift_file
                    .search_without_index(query_vector, top_k, swift_filter.clone(), &prune_config)
                    .await?;
                // Apply filter expression after progressive search if provided
                let filtered_results = if let Some(filter_expr) = filter_expression {
                    file_results
                        .into_iter()
                        .filter(|record| {
                            crate::core::search::sql_value_filter::evaluate_filter_proxima(
                                filter_expr,
                                &record.props,
                            )
                        })
                        .collect()
                } else {
                    file_results
                };
                all_results.extend(filtered_results);
            }

            // Sort and take top_k from all results
            all_results.sort_by(|a, b| {
                let dist_a: f32 = self
                    .distance_engine
                    .calculate_distance(query_vector, swift_record_vector(a), &distance_metric)
                    .normalized_score;
                let dist_b: f32 = self
                    .distance_engine
                    .calculate_distance(query_vector, swift_record_vector(b), &distance_metric)
                    .normalized_score;
                dist_b
                    .partial_cmp(&dist_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_results.truncate(top_k);

            // Convert to OptimizedSearchRecord
            let results: Vec<OptimizedSearchRecord> = all_results
                .into_iter()
                .map(|record| {
                    let distance_result = self.distance_engine.calculate_distance(
                        query_vector,
                        swift_record_vector(&record),
                        &distance_metric,
                    );
                    OptimizedSearchRecord::new(
                        swift_record_id(&record),
                        distance_result.normalized_score,
                    )
                    .with_similarity(distance_result.normalized_score)
                    .add_vector(swift_record_vector(&record).to_vec())
                    .with_proxima_metadata(
                        crate::core::search::sql_value_filter::proxima_tree_to_value_map(
                            &record.props,
                        ),
                    )
                })
                .collect();

            info!(
                "🎯 SWIFT progressive search found {} results",
                results.len()
            );
            return Ok(results);
        }

        // ========================================================================
        // PHASE 2: BLOCK-PRUNED SEARCH (ALWAYS ENABLED FOR PERFORMANCE)
        // ========================================================================
        //
        // Uses search_without_index which applies block pruning based on block-level
        // metadata (min/max vectors) to skip irrelevant blocks. This improves
        // performance even without explicit quantization/AXIS configuration.

        info!("🔍 SWIFT: Using block-pruned search implementation");

        // Load files from storage
        let files = self
            .load_collection_files(collection_id, storage_path, Some(&*ctx.collection))
            .await?;

        // Use default pruning config for block-level optimization
        let prune_config = crate::core::search::BlockPruneConfig::default();
        let swift_filter: Option<super::MetadataFilter> = None;

        let mut all_results = Vec::new();
        for swift_file in &files {
            // Use search_without_index which applies block pruning
            let file_results = swift_file
                .search_without_index(query_vector, top_k, swift_filter.clone(), &prune_config)
                .await?;

            // Apply filter expression after block-pruned search if provided
            let filtered_results = if let Some(filter_expr) = filter_expression {
                let filtered: Vec<_> = file_results
                    .into_iter()
                    .filter(|record| {
                        crate::core::search::sql_value_filter::evaluate_filter_proxima(
                            filter_expr,
                            &record.props,
                        )
                    })
                    .collect();
                filtered
            } else {
                file_results
            };
            all_results.extend(filtered_results);
        }

        // Sort and take top_k from all results
        all_results.sort_by(|a, b| {
            let dist_a: f32 = self
                .distance_engine
                .calculate_distance(query_vector, swift_record_vector(a), &distance_metric)
                .normalized_score;
            let dist_b: f32 = self
                .distance_engine
                .calculate_distance(query_vector, swift_record_vector(b), &distance_metric)
                .normalized_score;
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(top_k);

        // Convert to OptimizedSearchRecord
        let search_results: Vec<OptimizedSearchRecord> = all_results
            .into_iter()
            .map(|record| {
                let distance_result = self.distance_engine.calculate_distance(
                    query_vector,
                    swift_record_vector(&record),
                    &distance_metric,
                );
                OptimizedSearchRecord::new(
                    swift_record_id(&record),
                    distance_result.normalized_score,
                )
                .with_similarity(distance_result.normalized_score)
                .add_vector(swift_record_vector(&record).to_vec())
                .with_proxima_metadata(
                    crate::core::search::sql_value_filter::proxima_tree_to_value_map(&record.props),
                )
            })
            .collect();

        let results_len = search_results.len();

        // Track bytes processed for metrics
        if let Some(ref mut timer) = timer {
            let bytes_processed = results_len * query_vector.len() * 4; // Approximate
            timer.set_bytes_processed(bytes_processed as u64);
        }

        // Return OptimizedSearchRecord directly
        Ok(search_results)
    }

    // =============================================================================
    // OPTIONAL OPERATIONS - Use default implementations
    // =============================================================================

    async fn optimize(&self, collection_id: &str) -> Result<()> {
        info!("SWIFT optimize: collection={}", collection_id);

        // Trigger compaction for optimization
        let params = CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: false,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            priority: crate::storage::traits::OperationPriority::Medium,
            collection_config: None,
            estimated_input_size: 0,
        };

        self.do_compact(&params).await?;
        Ok(())
    }

    async fn get_statistics(&self) -> Result<EngineStatistics> {
        Ok(self.statistics.read().await.clone())
    }

    async fn health_check(&self) -> Result<EngineHealth> {
        Ok(EngineHealth {
            healthy: true,
            status: "SWIFT engine operational".to_string(),
            last_check: chrono::Utc::now(),
            response_time_ms: 1.0,
            error_count: 0,
            warnings: Vec::new(),
            metrics: std::collections::HashMap::new(),
        })
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(
            feature,
            "id_lookup"
                | "similarity_search"
                | "progressive_search"
                | "quantization"
                | "compression"
                | "batch_operations"
        )
    }
}

#[allow(deprecated)] // SwiftEngine is #[deprecated] (experimental); accessing its own field is intentional
impl crate::storage::traits::EngineFilesystemAccess for SwiftEngine {
    fn get_filesystem_factory(
        &self,
    ) -> &crate::storage::persistence::filesystem::FilesystemFactory {
        &self.filesystem
    }
}

/// Implementation of UniversallyOptimized trait for SWIFT engine
#[allow(deprecated)]
#[async_trait::async_trait]
impl UniversallyOptimized for SwiftEngine {
    /// Get the universal performance optimizer instance
    fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer {
        &self.universal_optimizer
    }

    /// SWIFT-specific optimization setup
    async fn setup_engine_optimizations(&self) -> Result<()> {
        // SWIFT-specific optimizations for hierarchical storage
        info!("🔧 SWIFT Engine: Setting up universal performance optimizations");

        // Initialize hierarchical optimizations
        let config = self.universal_optimizer.get_config();
        debug!("   Cache size: {}MB", config.cache_size_mb);
        debug!("   Parallel operations: {}", config.parallel_operations);
        debug!("   Prefetching enabled: {}", config.enable_prefetching);
        debug!(
            "   Memory mapping enabled: {}",
            config.enable_memory_mapping
        );

        // SWIFT is ready for hierarchical row-based operations
        info!("✅ SWIFT Engine: Universal optimizations configured for hierarchical storage");
        Ok(())
    }

    /// SWIFT-specific performance metrics
    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Basic SWIFT metrics
        let stats = self.statistics.read().await;
        metrics.insert(
            "swift_total_storage_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.total_storage_bytes)),
        );
        metrics.insert(
            "swift_memory_usage_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.memory_usage_bytes)),
        );
        metrics.insert(
            "swift_collection_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.collection_count)),
        );
        metrics.insert(
            "swift_pending_flushes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.pending_flushes)),
        );

        // Universal optimizer metrics
        let strategy = self.universal_optimizer.get_strategy();
        metrics.insert(
            "universal_optimization_strategy".to_string(),
            serde_json::Value::String(format!("{:?}", strategy)),
        );

        let config = self.universal_optimizer.get_config();
        metrics.insert(
            "universal_cache_size_mb".to_string(),
            serde_json::Value::Number(serde_json::Number::from(config.cache_size_mb)),
        );
        metrics.insert(
            "universal_parallel_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(config.parallel_operations)),
        );
        metrics.insert(
            "universal_prefetching_enabled".to_string(),
            serde_json::Value::Bool(config.enable_prefetching),
        );

        Ok(metrics)
    }
}

// Helper methods for SwiftEngine
#[allow(deprecated)]
impl SwiftEngine {
    // Removed unnecessary helper methods - engines already have these components as fields
    // Distance and quantization engines are accessed directly from struct fields

    /// Fallback to direct search when orchestration is not available
    /// This is the expected path until AXIS/orchestration is fully integrated for SWIFT
    #[allow(dead_code)]
    async fn fallback_to_direct_search(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
        collection_id: &str,
        storage_path: &str,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: proximadb_distance_kernel::DistanceMetric,
        _filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        // Debug level since this is expected behavior until orchestration is fully integrated
        tracing::debug!("🔍 SWIFT: Using direct search (orchestration pending integration)");

        // Use the existing search implementation
        // Load files from storage
        let files = self
            .load_collection_files(collection_id, storage_path, Some(&*ctx.collection))
            .await?;

        // Use bounded priority queue to maintain only top-k results
        let mut priority_queue = BoundedPriorityQueue::new(top_k);

        // Search each SWIFT file by iterating through superblocks and blocks
        for swift_file in &files {
            // Iterate through all superblocks -> blocks -> records
            for superblock in &swift_file.superblocks {
                for block in &superblock.blocks {
                    for record in &block.records {
                        // Apply metadata filter if present
                        if let Some(filter_expr) = _filter_expression {
                            let matches =
                                crate::core::search::sql_value_filter::evaluate_filter_proxima(
                                    filter_expr,
                                    &record.props,
                                );
                            if !matches {
                                continue; // Skip records that don't match filter
                            }
                        }

                        let record_vector = swift_record_vector(record);
                        if record_vector.is_empty() {
                            continue;
                        }

                        // Compute actual distance using distance engine
                        let distance_result = self.distance_engine.calculate_distance(
                            query_vector,
                            record_vector,
                            &distance_metric,
                        );

                        let id = if record.oid.is_empty() {
                            format!("unknown_{}", record.created_at_ns)
                        } else {
                            swift_record_id(record)
                        };

                        let mut search_record = OptimizedSearchRecord::new(
                            id,
                            distance_result.normalized_score,
                        )
                        .with_similarity(distance_result.normalized_score)
                        .add_vector(record_vector.to_vec())
                        .with_proxima_metadata(
                            crate::core::search::sql_value_filter::proxima_tree_to_value_map(
                                &record.props,
                            ),
                        );

                        if let Ok(version) = u32::try_from(record.record_version) {
                            search_record = search_record
                                .with_version_info(version, record.created_at_ns / 1_000_000);
                        }

                        // Try to insert into bounded queue - only keeps top-k
                        priority_queue.try_insert(search_record);
                    }
                }
            }
        }

        // Get sorted results from bounded queue
        let search_results = priority_queue.into_sorted_vec();

        Ok(search_results)
    }
}

#[cfg(test)]
#[allow(deprecated)] // SWIFT is experimental - tests validate expected behavior
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swift_engine_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        // Need to create distance engine and axis manager for new()
        let _distance_engine = Arc::new(
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(
                proximadb_distance_kernel::DistanceMetric::Euclidean,
            ),
        );
        let engine = SwiftEngine::new().await.unwrap();
        assert_eq!(engine.format_name(), "SWIFT");
        assert_eq!(engine.format_version(), "1.0.0");
    }

    #[tokio::test]
    async fn test_swift_feature_support() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        // Need to create distance engine and axis manager for new()
        let _distance_engine = Arc::new(
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(
                proximadb_distance_kernel::DistanceMetric::Euclidean,
            ),
        );
        let engine = SwiftEngine::new().await.unwrap();

        assert!(engine.supports_feature("id_lookup"));
        assert!(engine.supports_feature("similarity_search"));
        assert!(engine.supports_feature("progressive_search"));
        assert!(engine.supports_feature("quantization"));
        assert!(!engine.supports_feature("unknown_feature"));
    }

    #[cfg(feature = "experimental-engines")]
    #[tokio::test]
    async fn test_swift_vector_by_id_miss() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = SwiftEngine::new().await.unwrap();

        // Lookup in non-existent path should return None (not error)
        let result = engine
            .vector_by_id(
                "test_collection",
                "/tmp/proximadb_test_nonexistent",
                "vec_123",
            )
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[cfg(feature = "experimental-engines")]
    #[test]
    fn test_swift_filter_evaluation() {
        use proximadb_data_model::ProximaValue;
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
        use proximadb_records::{ProximaTree, ProximaTreeNode};

        // Create test records with metadata
        let mut props = ProximaTree::new();
        props.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("electronics".to_string())),
        );
        props.insert(
            "price".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(29.99)),
        );

        // Test Equals filter
        let eq_filter = FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("electronics".to_string()),
        };
        assert!(crate::core::search::sql_value_filter::evaluate_filter_proxima(&eq_filter, &props));

        // Test not-matching Equals filter
        let neq_filter = FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("clothing".to_string()),
        };
        assert!(
            !crate::core::search::sql_value_filter::evaluate_filter_proxima(&neq_filter, &props)
        );

        // Test LessThan filter on numeric field
        let lt_filter = FilterExpression::Comparison {
            field: "price".to_string(),
            operator: ComparisonOperator::LessThan,
            value: serde_json::json!(50.0),
        };
        assert!(crate::core::search::sql_value_filter::evaluate_filter_proxima(&lt_filter, &props));

        // Test GreaterThan filter on numeric field
        let gt_filter = FilterExpression::Comparison {
            field: "price".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: serde_json::json!(50.0),
        };
        assert!(
            !crate::core::search::sql_value_filter::evaluate_filter_proxima(&gt_filter, &props)
        );
    }

    // ========================================================================
    // SwiftEngine metadata tests
    // ========================================================================

    #[tokio::test]
    async fn test_swift_engine_name_and_version() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = SwiftEngine::new().await.unwrap();
        assert_eq!(engine.format_name(), "SWIFT");
        assert_eq!(engine.format_version(), "1.0.0");
    }

    #[tokio::test]
    async fn test_swift_engine_strategy() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = SwiftEngine::new().await.unwrap();
        assert_eq!(engine.strategy(), StorageFormatStrategy::Swift);
    }

    #[tokio::test]
    async fn test_swift_feature_support_exhaustive() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = SwiftEngine::new().await.unwrap();

        // All supported features
        let supported = [
            "id_lookup",
            "similarity_search",
            "progressive_search",
            "quantization",
            "compression",
            "batch_operations",
        ];
        for feature in &supported {
            assert!(
                engine.supports_feature(feature),
                "Expected feature '{}' to be supported",
                feature
            );
        }

        // Unsupported features
        let unsupported = ["unknown_feature", "graph_queries", "full_text_search", ""];
        for feature in &unsupported {
            assert!(
                !engine.supports_feature(feature),
                "Expected feature '{}' to NOT be supported",
                feature
            );
        }
    }

    // ========================================================================
    // SwiftHeader / constants tests
    // ========================================================================

    #[test]
    fn test_swift_magic_constant() {
        use crate::storage::engines::swift::SWIFT_MAGIC;
        assert_eq!(SWIFT_MAGIC, *b"SWFT");
    }

    #[test]
    fn test_swift_header_default() {
        use crate::storage::engines::swift::SwiftHeader;
        let header = SwiftHeader::default();
        assert_eq!(header.magic, *b"SWFT");
        assert_eq!(header.version, 1);
        assert_eq!(header.distance_metric, "cosine");
        assert_eq!(header.dimension, 0);
        assert_eq!(header.total_records, 0);
        assert_eq!(header.deleted_records, 0);
        assert_eq!(header.blocks_per_superblock, 10);
        assert_eq!(header.records_per_block, 512);
        assert_eq!(header.compaction_level, 0);
    }

    #[test]
    fn test_swift_header_custom() {
        use crate::storage::engines::swift::{SWIFT_MAGIC, SwiftHeader};
        let header = SwiftHeader {
            magic: SWIFT_MAGIC,
            version: 2,
            collection_id: "test_collection".to_string(),
            timestamp: 1234567890,
            compaction_level: 3,
            dimension: 768,
            distance_metric: "euclidean".to_string(),
            quantization: Default::default(),
            total_records: 10000,
            deleted_records: 50,
            superblock_count: 5,
            blocks_per_superblock: 20,
            records_per_block: 256,
            superblock_offset: 8192,
            id_index_offset: 0,
            quantized_index_offset: 0,
            metadata_index_offset: 0,
            header_checksum: 0,
            file_checksum: 0,
        };
        assert_eq!(header.version, 2);
        assert_eq!(header.collection_id, "test_collection");
        assert_eq!(header.dimension, 768);
        assert_eq!(header.total_records, 10000);
    }

    // ========================================================================
    // SuperBlock construction tests
    // ========================================================================

    #[test]
    fn test_superblock_new() {
        use crate::storage::engines::swift::SuperBlock;
        let sb = SuperBlock::new(0, "test_sb".to_string());
        assert_eq!(sb.superblock_id, 0);
        assert_eq!(sb.name, "test_sb");
        assert!(sb.blocks.is_empty());
        assert_eq!(sb.record_count, 0);
        assert!(sb.adacurve_code.is_none());
        assert!(sb.swift_metadata.swift_specific_data.hierarchical_structure);
        assert!(
            sb.swift_metadata
                .swift_specific_data
                .large_scale_optimization
        );
    }

    #[test]
    fn test_quantization_config_old_default() {
        use crate::storage::engines::swift::QuantizationConfigOLD;
        let config = QuantizationConfigOLD::default();
        assert!(!config.enable_binary);
        assert!(!config.enable_int8);
        assert!(!config.enable_pq);
        assert_eq!(config.binary_threshold, 0.5);
        assert_eq!(config.int8_scale, 1.0);
        assert_eq!(config.pq_segments, 8);
    }

    #[tokio::test]
    async fn test_swift_engine_initial_statistics() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = SwiftEngine::new().await.unwrap();
        let stats = engine.statistics.read().await;
        assert_eq!(stats.engine_name, "SWIFT");
        assert_eq!(stats.engine_version, "2.0.0");
        assert_eq!(stats.total_storage_bytes, 0);
        assert_eq!(stats.memory_usage_bytes, 0);
        assert_eq!(stats.collection_count, 0);
    }
}
