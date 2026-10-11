// NOVA Engine: Next-gen Optimized Vector Analytics with columnar quantization
// Implements UnifiedStorageFormat trait for integration with ProximaDB

use crate::core::search::{DataFreshnessTier, SearchMode};
use crate::proto::proximadb_v1::VectorRecord;
// Import column constants from columnar module
use crate::storage::engines::core::formats::columnar::constants::QUANTIZATION_COLUMNS;
use crate::storage::engines::core::formats::columnar::{
    FIELD_COLLECTION_ID, FIELD_EXPIRES_AT, FIELD_EXTRA_META, FIELD_ID, FIELD_IS_DELETED,
    FIELD_ROW_GROUP_OFFSET, FIELD_ROW_INDEX, FIELD_SCHEMA_VERSION, FIELD_SOURCE, FIELD_TIMESTAMP,
    FIELD_UPDATED_AT, FIELD_VECTOR_FP32, FIELD_VERSION,
};
use crate::storage::engines::core::ops::{
    UniversalOptimizationStrategy, UniversalPerformanceOptimizer, UniversallyOptimized,
};
use crate::storage::engines::nova::NovaFile;
use crate::storage::traits::{
    CompactionParameters, CompactionResult, EngineHealth, EngineStatistics, FlushParameters,
    FlushResult, OperationPriority, UnifiedStorageFormat,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_compression::StandardCompression;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
// Health status handled internally
use super::operations::{NovaCompactionOperations, NovaFlushOperations, NovaSearchOperations};
use super::optimized_operations::OptimizedNovaOperations;
use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::metrics::collectors::{EngineMetricsCollector, OperationTimer};
use proximadb_distance_kernel::DistanceMetric;
// Arrow schema handled by parquet reader

// Performance optimization handled internally
// NOVA-specific optimization structures removed - now using universal module

use proximadb_hardware_caps::HardwareCapabilities;

/// NOVA Engine - Next-generation Optimized Vector Analytics
///
/// ## Architecture Overview
///
/// NOVA (Next-gen Optimized Vector Analytics) is ProximaDB's progressive columnar
/// storage engine, designed for mixed workloads requiring both analytical batch
/// processing and selective point queries.
///
/// ### Core Design Principles:
/// - **Progressive Quantization**: Multi-level refinement (Binary → INT8 → PQ8 → FP32)
/// - **Operations-Based Architecture**: Modular design with separate flush/compaction/search modules
/// - **Hierarchical Statistics**: Progressive filtering using stored min/max/bloom filters
/// - **Adaptive I/O**: Intelligent buffering and coalescing for cloud storage
///
/// ### Data Flow:
/// ```text
/// Insert → Batch → Quantize (3 levels) → Compress → Parquet Row Groups
///                          ↓
///                   Statistics Collection
///                          ↓
///                   Progressive Search Pipeline:
///                   1. Bloom Filter Check
///                   2. Binary Quantization Scan
///                   3. INT8 Refinement
///                   4. PQ8 Final Filter
///                   5. FP32 Distance (top-k only)
/// ```
///
/// ### Key Differentiators:
/// - **vs SST**: Pure columnar (Parquet) vs hybrid columnar (ProximaBlocks), 10x better compression
/// - **vs VIPER**: Progressive search vs full scan, 5x faster selective queries
/// - **vs SWIFT**: Higher compression vs lower latency
///
/// ### Performance Characteristics:
/// - **Write Latency**: ~10-20ms (batched quantization + compression)
/// - **Selective Query**: ~5-15ms (progressive filtering eliminates 95%+ candidates)
/// - **Batch Scan**: ~50-200ms (full columnar scan with SIMD)
/// - **Compression**: 8-12x (multiple quantization levels + ZSTD)
pub struct NovaEngine {
    /// **Filesystem Factory**
    ///
    /// Creates filesystem instances for different storage backends:
    /// - Local filesystem (file://)
    /// - S3 (s3://) with intelligent chunking
    /// - Azure Blob (azure://) with range optimization
    /// - GCS (gs://) with footer caching
    ///
    /// Shared across all NOVA operations for consistent access patterns
    filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,

    /// **Optimized Operations Handler**
    ///
    /// High-performance operation executor:
    /// - SIMD-accelerated batch processing
    /// - Zero-copy I/O when possible
    /// - Parallel row group processing
    /// - Memory-mapped file access for large scans
    ///
    /// Used for hot-path operations requiring maximum throughput
    _optimized_ops: Arc<OptimizedNovaOperations>,

    /// **Flush Operations Module**
    ///
    /// Handles write path from memory to Parquet:
    /// - Batches vectors into optimal row groups (128K default)
    /// - Applies 3-level quantization (binary, INT8, PQ8)
    /// - Compresses each column independently (ZSTD)
    /// - Computes statistics (min/max/null count)
    /// - Writes bloom filters for string columns
    ///
    /// Modular design allows independent testing and optimization
    flush_ops: Arc<NovaFlushOperations>,

    /// **Compaction Operations Module**
    ///
    /// Background storage optimization:
    /// - Merges small files into larger ones (reduce metadata overhead)
    /// - Recomputes statistics after tombstone cleanup
    /// - Re-sorts data by clustering key if specified
    /// - Updates collection-level metadata
    ///
    /// Runs asynchronously, triggered by file count or size thresholds
    compaction_ops: Arc<NovaCompactionOperations>,

    /// **Search Operations Module**
    ///
    /// Progressive query execution engine:
    /// - **Stage 1**: Bloom filter check (eliminates 90%+ misses)
    /// - **Stage 2**: Binary quantization scan (1 bit/dim, ultra-fast)
    /// - **Stage 3**: INT8 refinement (8x smaller than FP32)
    /// - **Stage 4**: PQ8 final filtering (top-k candidates)
    /// - **Stage 5**: FP32 exact distance (verify top-k only)
    ///
    /// Each stage reduces candidate set by 50-90%, minimizing I/O
    search_ops: Arc<NovaSearchOperations>,

    /// **Engine Statistics** (RwLock for concurrent access)
    ///
    /// Real-time metrics tracking:
    /// - Storage size per collection (compressed + uncompressed)
    /// - Memory usage (buffers, caches, working sets)
    /// - Operation counts (flush, compaction, search)
    /// - Latency percentiles (p50, p95, p99)
    /// - Last operation timestamps
    ///
    /// RwLock allows many concurrent readers, exclusive writer
    statistics: Arc<RwLock<EngineStatistics>>,

    /// **Hardware Capabilities**
    ///
    /// System capability detector:
    /// - CPU features (SIMD: AVX2/AVX512/NEON/SSE)
    /// - Memory available and speed (DDR4/DDR5)
    /// - Storage type (NVMe/SSD/HDD)
    /// - Network capabilities for cloud storage
    ///
    /// Used to select optimal algorithms at runtime
    hardware: Arc<HardwareCapabilities>,

    /// **Metrics Collector** (Optional)
    ///
    /// Integration with monitoring systems:
    /// - Exports to Prometheus/StatsD
    /// - Aggregates metrics across operations
    /// - Provides operation timing decorators
    /// - Tracks custom engine-specific metrics
    ///
    /// None if monitoring disabled, Some in production
    metrics_collector: Option<Arc<EngineMetricsCollector>>,

    /// **Compression Provider**
    ///
    /// Direct compression interface (no adapter overhead):
    /// - ZSTD (best compression for vectors)
    /// - Snappy (fast for metadata)
    /// - LZ4 (fastest for hot paths)
    /// - Adaptive selection based on data characteristics
    ///
    /// Stateless provider, thread-safe for concurrent use
    _compression_provider: StandardCompression,

    /// **Storage Quantization Engine** (Collection-Aware)
    ///
    /// Persistent quantization with trained codebooks:
    /// - PQ8 codebooks stored per collection in filesystem
    /// - Binary quantization (1 bit per dimension)
    /// - INT8 quantization with learned scaling factors
    /// - Training uses k-means++ on first 10K vectors
    /// - Codebooks reused forever after first flush
    ///
    /// Critical for consistent progressive search results
    storage_quantization_engine:
        Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine>,

    /// **Fallback Quantization Engine** (Stateless)
    ///
    /// In-memory quantization for ad-hoc operations:
    /// - No persistent codebooks needed
    /// - Used when collection codebook unavailable
    /// - Same algorithms as storage engine
    /// - Faster for one-off quantization tasks
    ///
    /// Falls back when storage engine doesn't have trained codebooks
    fallback_quantization_engine:
        Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine>,

    /// **Distance Computation Engine**
    ///
    /// Hardware-accelerated similarity calculations:
    /// - Auto-detects SIMD (AVX2/AVX512/NEON)
    /// - Supports L2, cosine, dot product metrics
    /// - Batch processing for throughput (1M+ vectors/sec)
    /// - Progressive refinement (coarse → fine distances)
    ///
    /// Shared singleton across all distance operations
    _distance_engine: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,

    /// **Universal Performance Optimizer**
    ///
    /// Cross-cutting optimization coordinator:
    /// - Vector memory pooling (reduces allocations)
    /// - I/O coalescing for cloud storage
    /// - Adaptive batching based on system load
    /// - Query plan optimization
    ///
    /// Replaces engine-specific optimizers, eliminates code duplication
    universal_optimizer: UniversalPerformanceOptimizer,

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
#[allow(dead_code)]
impl NovaEngine {
    /// Create new NOVA engine instance
    pub async fn new() -> Result<Self> {
        let hardware = proximadb_hardware_caps::get_hardware_capabilities();
        let optimized_ops = Arc::new(OptimizedNovaOperations::new()?);

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
        let distance_compute =
            Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
        let codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let unified_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_compute.clone(),
                codebook_store,
            ),
        );

        // Configure storage quantization for NOVA (columnar engine)
        let storage_config =
            crate::compute::quantization::storage_engine::StorageQuantizationConfig {
                primary_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::Pq8),
                filter_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::Binary),
                fast_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::Int8),
                distance_metric: DistanceMetric::Cosine,
                enable_progressive: true,
                filter_threshold: 100.0,
                candidate_multiplier: 10,
                training_sample_size: 10000,
                memory_budget_mb: 512, // Columnar uses more memory
                enable_hardware_acceleration: true,
            };

        let storage_quantization_engine = Arc::new(
            crate::compute::quantization::storage_engine::StorageQuantizationEngine::new(
                unified_engine.clone(),
                distance_compute.clone(),
                storage_config,
            ),
        );

        // Create fallback stateless quantization engine for ad-hoc queries
        let fallback_codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );
        let fallback_quantization_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_compute.clone(),
                fallback_codebook_store,
            ),
        );

        // Initialize universal performance optimization
        let universal_optimizer =
            UniversalPerformanceOptimizer::with_strategy(UniversalOptimizationStrategy::Balanced)
                .await?;

        // Initialize modularized operation handlers
        let flush_ops = Arc::new(NovaFlushOperations::new(filesystem.clone()));
        let compaction_ops = Arc::new(NovaCompactionOperations::new(filesystem.clone()));
        let search_ops = Arc::new(NovaSearchOperations::new(
            filesystem.clone(),
            DistanceMetric::Cosine,
        ));

        // NOVA benefits from UnifiedCachingFilesystem for caching hierarchical stats
        // We'll create collection-specific instances during operations since we need
        // the actual storage path to get the right filesystem from the factory

        Ok(Self {
            filesystem,
            _optimized_ops: optimized_ops,
            flush_ops,
            compaction_ops,
            search_ops,
            statistics: Arc::new(RwLock::new(EngineStatistics {
                engine_name: "NOVA".to_string(),
                engine_version: "1.0.0".to_string(), // Release 1 version
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
            _compression_provider: compression_provider,
            storage_quantization_engine,
            fallback_quantization_engine,
            _distance_engine: distance_compute,
            universal_optimizer,
            axis_manager: None, // AXIS manager will be set externally if available
        })
    }
    /// Set metrics collector for monitoring
    pub fn set_metrics_collector(&mut self, collector: Arc<EngineMetricsCollector>) {
        self.metrics_collector = Some(collector);
    }

    /// Start operation timer if metrics collector is available
    fn start_operation_timer(&self, operation: &str) -> Option<OperationTimer> {
        self.metrics_collector.as_ref().map(|collector| {
            OperationTimer::new(collector.clone(), "NOVA".to_string(), operation.to_string())
        })
    }

    /// Load NOVA files for collection from storage
    /// UnifiedCachingFilesystem provides transparent cloud storage support:
    /// - Cloud files (S3/GCS/Azure) are automatically downloaded to local disk cache on first access
    /// - Subsequent reads use the local cached copy (path: /tmp/proximadb/cache/{collection}/nova/)
    /// - Parquet metadata and footers are cached separately for fast schema access
    /// - Hot files remain in cache based on LRU policy and access patterns
    async fn load_collection_files(
        &self,
        collection_id: &str,
        storage_path: &str,
    ) -> Result<Vec<super::NovaFile>> {
        // Get UnifiedCachingFilesystem for NOVA
        // This creates a collection-specific cache instance that:
        // - Downloads cloud files to local disk cache on first read
        // - Caches Parquet metadata/footers for fast schema access
        // - Tracks access patterns for intelligent prefetching
        let unified_fs = self
            .filesystem
            .get_unified_caching_filesystem(
                storage_path,
                collection_id.to_string(),
                crate::storage::engines::ENGINE_NOVA.to_string(),
            )
            .map_err(|e| anyhow!("Failed to create unified filesystem: {}", e))?;

        // List all NOVA files in the collection directory
        let files = unified_fs.list(storage_path).await?;
        let mut nova_files = Vec::new();

        // Filter for NOVA Parquet files (using NOVA_FILE_EXT constant)
        for file_path in files {
            if file_path
                .name
                .ends_with(crate::storage::engines::constants::NOVA_FILE_EXT)
            {
                // Create a reader for this file based on query type
                let reader = super::columnar_strategy_reader::UnifiedNOVAReader::for_search(
                    self.filesystem.clone(),
                    collection_id.to_string(),
                    128, // Deferred: Pass actual dimension from StorageQueryContext when available
                )?;

                // Read vectors using the cached filesystem (metadata will be cached)
                let vectors = reader.read_progressive(&file_path.name).await?;

                // Compute dimension and zone maps from loaded vectors
                let dimension = if !vectors.is_empty() {
                    vectors[0].vector.len()
                } else {
                    0
                };

                // Compute zone maps for pruning optimization
                let zone_maps = if !vectors.is_empty() && dimension > 0 {
                    match self.compute_basic_zone_maps(&vectors, dimension) {
                        Ok(zm) => {
                            tracing::debug!(
                                "[NOVA] Zone maps computed for file {}: {} dimensions, {} vectors",
                                file_path.name,
                                dimension,
                                vectors.len()
                            );
                            Some(zm)
                        }
                        Err(e) => {
                            tracing::warn!("[NOVA] Failed to compute zone maps: {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                // Create NovaFile structure with zone maps for pruning
                let nova_file = super::NovaFile {
                    quantized_columns: super::quantized_columns::QuantizedColumnMetadata::default(),
                    schema: Arc::new(arrow_schema::Schema::empty()),
                    metadata:
                        crate::storage::engines::core::formats::columnar::ColumnarFileMetadata {
                            collection_id: collection_id.to_string(),
                            num_vectors: vectors.len() as u64,
                            dimension,
                            distance_metric: proximadb_distance_kernel::DistanceMetric::Euclidean,
                            quantization: Default::default(),
                            column_stats: Default::default(),
                            version: 1,
                            timestamp: chrono::Utc::now(),
                            modified_at: chrono::Utc::now(),
                        },
                    row_groups: Vec::new(),
                    enhanced_stats: Vec::new(),
                    superblocks: Vec::new(),
                    advanced_zone_maps: zone_maps,
                };

                nova_files.push(nova_file);
            }
        }

        // If no files found, return empty vec (normal for new collections)
        if nova_files.is_empty() {
            debug!(
                "No NOVA files found for collection {} in {}",
                collection_id, storage_path
            );
        } else {
            info!(
                "Loaded {} NOVA files for collection {} from {} (cached)",
                nova_files.len(),
                collection_id,
                storage_path
            );
        }

        Ok(nova_files)
    }

    /// Update global statistics file for collection
    async fn update_global_stats(&self, _collection_id: &str, _storage_path: &str) -> Result<()> {
        // Path: {storage_path}/{collection_id}/global.stats
        // This is updated after flush/compaction to maintain collection-wide metrics
        // File-level statistics are embedded in Parquet metadata properties
        Ok(())
    }

    /// Compute enhanced row group statistics (optimized NOVA design)
    fn compute_enhanced_row_group_stats(
        &self,
        records: &[VectorRecord],
        dimension: usize,
    ) -> Result<Vec<super::hierarchical_stats::EnhancedRowGroupStats>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        // Group vectors into row groups (default: 10K vectors per group)
        let row_group_size = 10000;
        let mut stats = Vec::new();

        for (group_idx, chunk) in records.chunks(row_group_size).enumerate() {
            let mut min_vals = vec![f32::INFINITY; dimension];
            let mut max_vals = vec![f32::NEG_INFINITY; dimension];
            let mut sum_vals = vec![0.0f32; dimension];
            let mut null_counts = vec![0u64; dimension];

            // Compute per-dimension statistics
            for record in chunk {
                if record.vector.len() != dimension {
                    continue; // Skip malformed vectors
                }

                for (dim_idx, &value) in record.vector.iter().enumerate() {
                    if value.is_finite() {
                        min_vals[dim_idx] = min_vals[dim_idx].min(value);
                        max_vals[dim_idx] = max_vals[dim_idx].max(value);
                        sum_vals[dim_idx] += value;
                    } else {
                        null_counts[dim_idx] += 1;
                    }
                }
            }

            // Compute centroid for pruning
            let centroid: Vec<f32> = sum_vals
                .iter()
                .map(|&sum| sum / chunk.len() as f32)
                .collect();

            // Create enhanced statistics using the optimized design
            let enhanced_stat = super::EnhancedRowGroupStats::create_basic(
                group_idx as u32,
                chunk.len() as u64,
                dimension,
                min_vals,
                max_vals,
                centroid,
                null_counts,
                1.0 / records.len() as f32, // Basic selectivity estimate
                0.7,                        // Placeholder - actual ratio computed during write
                0,                          // Will be updated based on query patterns
            );

            stats.push(enhanced_stat);
        }

        Ok(stats)
    }

    /// Compute basic zone maps for dimension-level pruning (simplified design)
    fn compute_basic_zone_maps(
        &self,
        records: &[VectorRecord],
        dimension: usize,
    ) -> Result<super::hierarchical_stats::BasicZoneMaps> {
        if records.is_empty() {
            return Ok(super::hierarchical_stats::BasicZoneMaps {
                dimension_ranges: Vec::new(),
                total_vectors: 0,
                creation_time: chrono::Utc::now(),
            });
        }

        let mut dimension_ranges = Vec::with_capacity(dimension);

        // Compute min/max range for each dimension across all vectors
        for dim_idx in 0..dimension {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut valid_count = 0;

            for record in records {
                if dim_idx < record.vector.len() {
                    let value = record.vector[dim_idx];
                    if value.is_finite() {
                        min_val = min_val.min(value);
                        max_val = max_val.max(value);
                        valid_count += 1;
                    }
                }
            }

            dimension_ranges.push(super::hierarchical_stats::DimensionRange {
                dimension_index: dim_idx,
                min_value: if valid_count > 0 { min_val } else { 0.0 },
                max_value: if valid_count > 0 { max_val } else { 0.0 },
                selectivity: valid_count as f32 / records.len() as f32,
            });
        }

        Ok(super::hierarchical_stats::BasicZoneMaps {
            dimension_ranges,
            total_vectors: records.len() as u64,
            creation_time: chrono::Utc::now(),
        })
    }

    // ============================================================================
    // PERFORMANCE OPTIMIZATION METHODS - DELEGATING TO UNIFIED MODULES
    // ============================================================================

    /// Fast read optimization using memory-mapped Parquet files (delegates to universal optimizer)
    async fn mmap_parquet_file(&self, file_path: &str) -> Result<Vec<u8>> {
        // Use universal optimizer's memory mapping functionality
        if let Some(mmap) = self
            .universal_optimizer
            .get_memory_mapped_file(file_path)
            .await?
        {
            Ok(mmap.to_vec())
        } else {
            // Fallback to regular file reading for cloud storage
            self.universal_optimizer
                .read_data_optimized(file_path)
                .await
        }
    }

    /// Columnar I/O optimization with parallel column reads (delegates to universal optimizer)
    async fn parallel_column_read(
        &self,
        file_path: &str,
        column_indices: &[usize],
    ) -> Result<Vec<Vec<u8>>> {
        // Use universal optimizer for parallel operations
        let optimizer = self.universal_optimizer.clone();
        let file_path_owned = file_path.to_string();
        let read_operations: Vec<_> = column_indices
            .iter()
            .map(|&column_idx| {
                let file_path = file_path_owned.clone();
                let optimizer_clone = optimizer.clone();
                async move {
                    // Simulate column-specific read (in production, use actual column reader)
                    optimizer_clone
                        .read_data_optimized(&format!("{}:col:{}", file_path, column_idx))
                        .await
                }
            })
            .collect();

        let results = self
            .universal_optimizer
            .parallel_operations(read_operations, |operation| operation)
            .await?;

        // Unwrap the nested Results
        let mut unwrapped_results = Vec::new();
        for res in results {
            match res {
                Ok(Ok(data)) => unwrapped_results.push(data),
                Ok(Err(e)) => return Err(anyhow::anyhow!("Column read failed: {}", e)),
                Err(e) => return Err(anyhow::anyhow!("Column read failed: {}", e)),
            }
        }

        Ok(unwrapped_results)
    }

    /// Storage tier optimization for Parquet files based on access patterns (delegates to universal optimizer)
    async fn optimize_parquet_storage_tier(
        &self,
        file_path: &str,
        _row_group_stats: &super::hierarchical_stats::EnhancedRowGroupStats,
    ) -> Result<DataFreshnessTier> {
        // Use common utility for consistent vector size estimation
        // Default configuration since NovaEngine doesn't have config field
        let dimension = 1536; // Default dimension
        // Estimate storage size based on dimension and number of row groups
        // Use a reasonable estimate of vectors per row group (e.g., 10000)
        let estimated_vectors = 10000; // Default estimate for row group size
        let estimated_size = crate::storage::engines::core::ops::estimate_vector_storage_size(
            dimension,
            None, // No quantization config available
            estimated_vectors,
        );

        // Use universal optimizer's storage tier optimization
        let infrastructure_tier = self
            .universal_optimizer
            .optimize_storage_tier(file_path, estimated_size as usize)
            .await?;

        // Convert from filesystem::StorageTier to multi_tier_deduplication::StorageTier
        let tier = match infrastructure_tier {
            crate::storage::persistence::filesystem::FileStorageTier::Memory => {
                DataFreshnessTier::Unflushed
            }
            crate::storage::persistence::filesystem::FileStorageTier::NVMe => {
                DataFreshnessTier::Flushed
            }
            crate::storage::persistence::filesystem::FileStorageTier::SSD => {
                DataFreshnessTier::Flushed
            }
            _ => DataFreshnessTier::Compacted,
        };

        Ok(tier)
    }

    /// Compression optimization using unified compression module (delegates to universal optimizer)
    async fn compress_parquet_optimized(
        &self,
        data: &[u8],
        tier: DataFreshnessTier,
    ) -> Result<Vec<u8>> {
        // Convert from multi_tier_deduplication::StorageTier to filesystem::StorageTier
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

    /// Distance computation using unified distance compute engine (delegates to universal optimizer)
    async fn compute_distances_unified(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        // Use universal optimizer's hardware-accelerated distance computation
        self.universal_optimizer
            .compute_distances_accelerated(query, candidates, metric)
            .await
    }

    /// Row group prefetching optimization (delegates to universal optimizer)
    async fn prefetch_row_groups(
        &self,
        file_path: &str,
        row_group_indices: &[usize],
    ) -> Result<()> {
        let config = self.universal_optimizer.get_config();
        if !config.enable_prefetching {
            return Ok(());
        }

        // Generate row group URLs for prefetching
        let prefetch_count = (config.prefetch_size_mb / 10).min(row_group_indices.len()); // Assume ~10MB per row group
        let row_group_urls: Vec<String> = row_group_indices
            .iter()
            .take(prefetch_count)
            .map(|&idx| format!("{}:rg:{}", file_path, idx))
            .collect();

        // Use universal optimizer's prefetching capability
        self.universal_optimizer
            .prefetch_data(&row_group_urls)
            .await
    }

    /// Memory pool optimization for columnar operations (delegates to universal optimizer)
    async fn get_columnar_buffer(&self, size: usize) -> Result<Vec<f32>> {
        self.universal_optimizer
            .get_memory_buffer(size)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire columnar buffer: {}", e))
    }

    /// Write NOVA file to disk using StreamingParquetWriter with sidecar metadata
    async fn write_nova_file_to_disk(
        &self,
        nova_file: &NovaFile,
        file_path: &str,
        params: &FlushParameters,
        _collection_id: &str,
    ) -> Result<u64> {
        use super::nova_meta_collector::{NovaCollectorConfig, NovaMetadataCollector};
        use crate::storage::engines::core::formats::columnar::{
            hybrid_writer::{HybridParquetWriter, HybridWriterConfig},
            parquet_write_engine::ParquetWriterConfig,
        };

        // Get filterable columns from collection config (use proto type directly)
        let filterable_columns = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map_or_else(
                || {
                    vec![crate::proto::proximadb_v1::FilterableColumnSpec {
                        name: FIELD_ID.to_string(),
                        data_type: crate::proto::proximadb_v1::FilterableDataType::FilterableString
                            as i32,
                        indexed: true,
                        supports_range: false,
                        estimated_cardinality: Some(1000000),
                    }]
                },
                |cfg| cfg.filterable_columns.clone(),
            );

        // Configure writer with NOVA-specific settings
        // Include both ID and filterable columns in bloom filters
        let mut bloom_columns = vec![FIELD_ID.to_string()];
        bloom_columns.extend(filterable_columns.iter().map(|c| c.name.clone()));

        let writer_config = ParquetWriterConfig {
            compression: parquet::basic::Compression::ZSTD(Default::default()),
            row_group_size: 50_000, // 50K vectors per row group
            write_batch_size: 10_000,
            enable_bloom_filters: true,
            bloom_filter_fpp: 0.01,    // 1% false positive rate
            bloom_filter_ndv: 1000000, // Expect up to 1M unique IDs
            enable_statistics: true,
            enable_page_index: true,
            enable_dictionary: true,
            quantization: crate::proto::proximadb_v1::QuantizationConfig::default(),
            id_less_storage: false, // Keep IDs for compatibility
            page_size: 8192,
            sort_columns: vec![], // No sorting for now
            filterable_metadata_columns: Some(
                filterable_columns.iter().map(|c| c.name.clone()).collect(),
            ),
            compression_level: None,
            max_records_per_file: None,
            target_file_size_bytes: None,
            enable_async_io: false,
        };

        // Create NOVA metadata collector for sidecar generation
        let nova_collector = NovaMetadataCollector::new(NovaCollectorConfig {
            row_groups_per_superblock: 10, // 10 row groups per SuperBlock
            compute_vector_stats: true,
            sample_rate: 0.1, // Sample 10% for expensive statistics
        });

        // Configure HybridParquetWriter for adaptive optimization
        let hybrid_config = HybridWriterConfig {
            base_config: writer_config,
            ..Default::default()
        };

        // Use HybridParquetWriter with integrated disk cache and metadata collection
        // This handles:
        // 1. Writing to temp file
        // 2. Collecting metadata during write
        // 3. Finalizing the writer
        // 4. Uploading to cloud/local storage
        // 5. Populating disk cache for future reads
        // 6. Returning metadata collector for sidecar generation
        let (stats, collector) = HybridParquetWriter::write_with_cache(
            &params.vector_records,
            nova_file.metadata.dimension,
            hybrid_config,
            file_path,
            &self.filesystem,
            Some(filterable_columns),
            Some(Box::new(nova_collector)),
        )
        .await?;

        let bytes_written = stats.file_size;

        // Write sidecar metadata file if collector has data
        if let Some(collector) = collector {
            let sidecar_path = format!("{}.{}", file_path, collector.sidecar_extension());
            let sidecar_data = collector.serialize_metadata()?;

            // Write sidecar using filesystem (this also gets cached)
            let fs = self
                .filesystem
                .get_filesystem(&self.determine_fs_url(file_path))?;
            fs.write(&sidecar_path, &sidecar_data, None).await?;

            info!(
                "NOVA: Wrote sidecar metadata ({} bytes) to {} with disk cache",
                sidecar_data.len(),
                sidecar_path
            );
        }

        debug!(
            "NOVA: Wrote {} records to {} with disk cache ({}MB)",
            stats.total_records,
            file_path,
            bytes_written / 1024 / 1024
        );

        info!(
            "NOVA: Successfully wrote {} bytes to {} with {} row groups",
            bytes_written, file_path, stats.total_row_groups
        );
        Ok(bytes_written)
    }

    /// Helper method to get file size in GB
    async fn get_file_size_gb(&self, file_path: &str) -> Result<f32> {
        let metadata = tokio::fs::metadata(file_path).await?;
        Ok(metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0))
    }

    /// Convert VectorRecords to Arrow RecordBatch
    fn vectors_to_record_batch(
        &self,
        records: &[VectorRecord],
        schema: &Arc<arrow_schema::Schema>,
    ) -> Result<arrow_array::RecordBatch> {
        use arrow_array::builder::*;
        use arrow_array::builder::{FixedSizeBinaryBuilder, FixedSizeListBuilder, Int8Builder};
        use std::sync::Arc;

        // Build arrays for each field
        let mut id_builder = StringBuilder::new();

        // Get dimension from schema for the vector field
        let dimension =
            if let arrow_schema::DataType::FixedSizeList(_, dim) = schema.fields()[1].data_type() {
                *dim as usize
            } else {
                // Fallback: use first record's vector dimension
                records.first().map_or(0, |r| r.vector.len())
            };

        // Build vector column as FixedSizeList
        let values_builder = Float32Builder::new();
        let mut vector_builder = FixedSizeListBuilder::new(values_builder, dimension as i32);

        let mut timestamp_builder = Int64Builder::new();
        let mut version_builder = UInt32Builder::new();

        // Check if quantization fields are present in schema (they would be after the 4 core fields)
        let mut quantization_field_count = 0;
        for field in schema.fields().iter().skip(4) {
            if field.name().starts_with("vector_")
                || field.name() == "int8_scale"
                || field.name() == "int8_zero_point"
            {
                quantization_field_count += 1;
            } else {
                break; // Stop when we hit the first non-quantization field
            }
        }

        // Build metadata columns dynamically based on schema (skip core + quantization fields)
        let mut metadata_builders: Vec<Box<dyn arrow_array::builder::ArrayBuilder>> = Vec::new();
        for field_idx in (4 + quantization_field_count)..schema.fields().len() {
            let field = &schema.fields()[field_idx];
            let builder: Box<dyn arrow_array::builder::ArrayBuilder> = match field.data_type() {
                arrow_schema::DataType::Utf8 => Box::new(StringBuilder::new()),
                arrow_schema::DataType::Int64 => Box::new(Int64Builder::new()),
                arrow_schema::DataType::Float64 => Box::new(Float64Builder::new()),
                arrow_schema::DataType::Boolean => Box::new(BooleanBuilder::new()),
                _ => Box::new(StringBuilder::new()), // Default to string
            };
            metadata_builders.push(builder);
        }

        for record in records {
            // ID column
            id_builder.append_value(&record.id);

            // Vector column as FixedSizeList
            let values_builder = vector_builder.values();
            for val in &record.vector {
                values_builder.append_value(*val);
            }
            vector_builder.append(true);

            // Timestamp column
            timestamp_builder.append_value(record.timestamp.unwrap_or(0));

            // Version column
            version_builder.append_option(record.version);

            // Metadata columns
            for (field_idx, builder) in metadata_builders.iter_mut().enumerate() {
                let field = &schema.fields()[field_idx + 4 + quantization_field_count];
                let field_name = field.name();

                // Get metadata value for this field
                let metadata_value = record.metadata.get(field_name);

                // Append value based on field type
                match field.data_type() {
                    arrow_schema::DataType::Utf8 => {
                        let string_builder = builder
                            .as_any_mut()
                            .downcast_mut::<StringBuilder>()
                            .ok_or_else(|| {
                            anyhow!("Failed to downcast metadata builder to StringBuilder")
                        })?;
                        if let Some(value) = metadata_value {
                            // TD-PROTO-2: JSONB decodes to compact JSON text
                            // instead of being dropped by the `_` wildcard.
                            // The decoded String needs an owned binding before
                            // it can be borrowed as &str below.
                            let jsonb_text = value.value.as_ref().and_then(|v| match v {
                                crate::proto::proximadb_v1::sql_value::Value::JsonbValue(b) => {
                                    // The string-only seam (round 6: was the
                                    // one bypass among string-only consumers).
                                    Some(
                                        proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(
                                            b,
                                        ),
                                    )
                                }
                                _ => None,
                            });
                            let borrowed = jsonb_text.as_deref().or_else(|| {
                                value.value.as_ref().and_then(|v| match v {
                                    crate::proto::proximadb_v1::sql_value::Value::StringValue(
                                        s,
                                    ) => Some(s.as_str()),
                                    _ => None,
                                })
                            });
                            if let Some(s) = borrowed {
                                string_builder.append_value(s);
                            } else {
                                string_builder.append_null();
                            }
                        } else {
                            string_builder.append_null();
                        }
                    }
                    arrow_schema::DataType::Int64 => {
                        let int_builder = builder
                            .as_any_mut()
                            .downcast_mut::<Int64Builder>()
                            .ok_or_else(|| {
                                anyhow!("Failed to downcast metadata builder to Int64Builder")
                            })?;
                        if let Some(value) = metadata_value {
                            if let Some(i) = value.value.as_ref().and_then(|v| match v {
                                crate::proto::proximadb_v1::sql_value::Value::Int64Value(i) => {
                                    Some(*i)
                                }
                                _ => None,
                            }) {
                                int_builder.append_value(i);
                            } else {
                                int_builder.append_null();
                            }
                        } else {
                            int_builder.append_null();
                        }
                    }
                    arrow_schema::DataType::Float64 => {
                        let float_builder = builder
                            .as_any_mut()
                            .downcast_mut::<Float64Builder>()
                            .ok_or_else(|| {
                            anyhow!("Failed to downcast metadata builder to Float64Builder")
                        })?;
                        if let Some(value) = metadata_value {
                            if let Some(f) = value.value.as_ref().and_then(|v| match v {
                                crate::proto::proximadb_v1::sql_value::Value::NumberValue(f) => {
                                    Some(*f)
                                }
                                _ => None,
                            }) {
                                float_builder.append_value(f);
                            } else {
                                float_builder.append_null();
                            }
                        } else {
                            float_builder.append_null();
                        }
                    }
                    arrow_schema::DataType::Boolean => {
                        let bool_builder = builder
                            .as_any_mut()
                            .downcast_mut::<BooleanBuilder>()
                            .ok_or_else(|| {
                                anyhow!("Failed to downcast metadata builder to BooleanBuilder")
                            })?;
                        if let Some(value) = metadata_value {
                            if let Some(b) = value.value.as_ref().and_then(|v| match v {
                                crate::proto::proximadb_v1::sql_value::Value::BoolValue(b) => {
                                    Some(*b)
                                }
                                _ => None,
                            }) {
                                bool_builder.append_value(b);
                            } else {
                                bool_builder.append_null();
                            }
                        } else {
                            bool_builder.append_null();
                        }
                    }
                    _ => {
                        // Default to string representation
                        let string_builder = builder
                            .as_any_mut()
                            .downcast_mut::<StringBuilder>()
                            .ok_or_else(|| {
                            anyhow!("Failed to downcast fallback metadata builder to StringBuilder")
                        })?;
                        string_builder.append_null();
                    }
                }
            }
        }

        // Create arrays
        let mut arrays: Vec<Arc<dyn arrow_array::Array>> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(vector_builder.finish()),
            Arc::new(timestamp_builder.finish()),
            Arc::new(version_builder.finish()),
        ];

        // Add null arrays for quantization fields if present
        for field_idx in 4..(4 + quantization_field_count) {
            let field = &schema.fields()[field_idx];
            let null_array = match field.data_type() {
                arrow_schema::DataType::FixedSizeBinary(len) => {
                    let mut builder = FixedSizeBinaryBuilder::new(*len);
                    for _ in 0..records.len() {
                        builder.append_null();
                    }
                    Arc::new(builder.finish()) as Arc<dyn arrow_array::Array>
                }
                arrow_schema::DataType::Float32 => {
                    let mut builder = Float32Builder::new();
                    for _ in 0..records.len() {
                        builder.append_null();
                    }
                    Arc::new(builder.finish()) as Arc<dyn arrow_array::Array>
                }
                arrow_schema::DataType::Int8 => {
                    let mut builder = Int8Builder::new();
                    for _ in 0..records.len() {
                        builder.append_null();
                    }
                    Arc::new(builder.finish()) as Arc<dyn arrow_array::Array>
                }
                _ => {
                    // Default to null string array
                    let mut builder = StringBuilder::new();
                    for _ in 0..records.len() {
                        builder.append_null();
                    }
                    Arc::new(builder.finish()) as Arc<dyn arrow_array::Array>
                }
            };
            arrays.push(null_array);
        }

        // Add metadata arrays
        for mut builder in metadata_builders {
            arrays.push(builder.finish());
        }

        // Create record batch
        arrow_array::RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| anyhow!("Failed to create record batch: {}", e))
    }

    /// Determine filesystem URL from path
    fn determine_fs_url(&self, path: &str) -> String {
        if path.starts_with("s3://")
            || path.starts_with("gs://")
            || path.starts_with("azure://")
            || path.starts_with("wasbs://")
        {
            path.to_string()
        } else {
            "file://".to_string()
        }
    }

    /// Check if we should use persistent quantization for this operation
    /// Returns true for collection-based operations with quantization enabled
    pub fn should_use_persistent_quantization(&self, params: &FlushParameters) -> bool {
        crate::compute::quantization::QuantizationSelector::should_use_persistent_quantization(
            params, "NOVA",
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

    /// Wrapper function for tests and benchmarks
    /// Converts legacy storage_url parameter to proper StorageQueryContext
    ///
    /// # Parameters
    /// - `collection_id`: Collection identifier
    /// - `storage_url`: Storage URL (can be full path with /data or base path)
    /// - `query_vector`: Query vector
    /// - `k`: Number of results to return
    pub async fn search_vectors(
        &self,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<crate::proto::proximadb_v1::SearchResult>> {
        info!(
            "🔍 NOVA Engine: search_vectors called - collection={}, storage_url={}, k={}",
            collection_id, storage_url, k
        );

        use crate::core::search::SearchParams;
        use crate::storage::traits::{StorageQueryContext, StorageQueryMetadata};

        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector.to_vec()),
            top_k: Some(k as u16),
            ..SearchParams::default()
        });

        // Extract base_location from storage_url (tests/benchmarks often pass full path)
        // Production behavior: metadata.storage_path should be base_location
        let base_location = if storage_url.contains(&format!("/{}/data", collection_id)) {
            storage_url.replace(&format!("/{}/data", collection_id), "")
        } else {
            storage_url.to_string()
        };

        // Create minimal collection config for testing
        let collection = crate::proto::proximadb_v1::Collection {
            id: collection_id.to_string(),
            config: Some(crate::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                dimension: query_vector.len() as u32,
                distance_metric: Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                storage_engine: Some(crate::proto::proximadb_v1::StorageEngine::Nova as i32),
                ..Default::default()
            }),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                base_location: base_location.clone(),
                primary_path: storage_url.to_string(),
                backup_paths: vec![],
                engine: crate::proto::proximadb_v1::StorageEngine::Nova as i32,
                engine_config: Default::default(),
                assigned_at: 0,
                ..Default::default()
            }),
            ..Default::default()
        };

        let collection = Arc::new(collection);

        let ctx = StorageQueryContext {
            search_params,
            collection,
            metadata: StorageQueryMetadata {
                collection_id: collection_id.to_string(),
                use_axis_indexes: false,
                has_quantization: false,
                storage_path: base_location, // Use base_location, not full path
                ..Default::default()
            },
            user_context: None,
            tenant_context: None,
        };

        let internal_results = self.search_vectors_unified(&ctx).await?;
        debug!(
            "search_vectors_unified returned {} results",
            internal_results.len()
        );

        // Convert OptimizedSearchRecord to SearchVectorRecord and wrap in SearchResult
        let search_records: Vec<crate::proto::proximadb_v1::SearchVectorRecord> = internal_results
            .into_iter()
            .map(|r| {
                let vector = r
                    .vector
                    .as_ref()
                    .map(|arc| (**arc).clone())
                    .unwrap_or_default();
                crate::proto::proximadb_v1::SearchVectorRecord {
                    id: r.id,
                    score: r.similarity.unwrap_or(r.score) as f64,
                    vector,
                    metadata: crate::core::search::results::proxima_map_to_sql(r.metadata.clone()),
                    version: None,
                    similarity: r.similarity,
                    timestamp: None,
                    source: r.source.map(|sc| match sc.data {
                        Some(crate::proto::proximadb_v1::source_content::Data::TextContent(
                            text,
                        )) => text,
                        Some(
                            crate::proto::proximadb_v1::source_content::Data::ExternalReference(
                                url,
                            ),
                        ) => url,
                        Some(crate::proto::proximadb_v1::source_content::Data::BinaryContent(
                            _,
                        )) => "[Binary Content]".to_string(),
                        None => "[Empty Content]".to_string(),
                    }),
                    expanded_context: r
                        .expanded_context
                        .iter()
                        .map(|sc| match &sc.data {
                            Some(
                                crate::proto::proximadb_v1::source_content::Data::TextContent(text),
                            ) => text.clone(),
                            Some(
                                crate::proto::proximadb_v1::source_content::Data::ExternalReference(
                                    url,
                                ),
                            ) => url.clone(),
                            Some(
                                crate::proto::proximadb_v1::source_content::Data::BinaryContent(_),
                            ) => "[Binary Content]".to_string(),
                            None => "[Empty Content]".to_string(),
                        })
                        .collect(),
                    semantic_similarity: None,
                    quantization_info: None,
                    engine_stats: std::collections::HashMap::new(),
                    index_path: None,
                }
            })
            .collect();

        // Wrap in SearchResult
        Ok(vec![crate::proto::proximadb_v1::SearchResult {
            results: search_records.clone(),
            total_found: search_records.len() as i64,
            collection_id: Some(collection_id.to_string()),
        }])
    }
}

#[async_trait]
impl UnifiedStorageFormat for NovaEngine {
    // =============================================================================
    // ENGINE IDENTIFICATION
    fn engine_name(&self) -> &'static str {
        "NOVA"
    }

    fn engine_version(&self) -> &'static str {
        "1.0.0" // Release 1 version
    }

    fn strategy(&self) -> crate::storage::traits::StorageFormatStrategy {
        crate::storage::traits::StorageFormatStrategy::Nova
    }

    /// Read ALL records of a collection from persisted Parquet files (Phase 8
    /// F1). Discovers `.parquet` files at the service-resolved `storage_url` and
    /// reads them via the columnar `UnifiedParquetReader` (returns canonical
    /// `ProximaRecord`s). WAL/memtable records are merged in by the service.
    async fn read_all_records(
        &self,
        collection_id: &str,
        storage_url: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let Some(storage_url) = storage_url else {
            return Ok(Vec::new());
        };
        let fs = self.filesystem.get_filesystem(storage_url)?;
        let mut files = Vec::new();
        if let Ok(entries) = fs.list(storage_url).await {
            for entry in &entries {
                if !entry.metadata.is_directory && entry.url.ends_with(".parquet") {
                    files.push(entry.url.clone());
                }
            }
        }
        if files.is_empty() {
            return Ok(Vec::new());
        }
        // UnifiedParquetReader wants a concrete UnifiedCachingFilesystem; build a
        // minimal one over the base fs (mirrors sst_reader.rs direct mode).
        let base_fs = self.filesystem.get_filesystem(storage_url)?;
        let caching = std::sync::Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                base_fs,
                collection_id.to_string(),
                "nova".to_string(),
            ),
        );
        let reader = crate::storage::engines::core::formats::columnar::UnifiedParquetReader::new(
            files,
            0,
            self.filesystem.clone(),
            caching,
            collection_id.to_string(),
            "nova".to_string(),
        )?;
        reader.read_all_records(0, None).await
    }

    // CORE OPERATIONS
    async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
        // Delegate to modularized flush operations
        let result = self.flush_ops.flush(params).await?;

        // Index the just-flushed vectors into AXIS directly (ADR-078). This lives
        // in each engine's `do_flush` (not the trait default) so the traits layer
        // stays a pure consumer with no root-service dependency (root-crate
        // decomposition gap 4). Best-effort: errors are logged inside the hook,
        // never propagated — the flush itself already succeeded.
        if result.success {
            crate::storage::common::axis_flush_hook::index_flushed_into_axis(
                self.axis_manager().cloned(),
                params,
                result.file_paths.clone(),
            )
            .await;
        }

        Ok(result)
    }

    /// NOVA's LSM bulk-load override (Phase 2F-b).
    ///
    /// Bypasses WAL + memtable entirely: the queue drainer hands us a
    /// pre-sorted batch, and we write a single Parquet file directly
    /// through `flush_ops.flush()` — the SAME writer the normal flush
    /// path uses for memtable-to-disk promotion. Because the writer is
    /// already record-shape-aware (`Vec<ProximaRecord>` in,
    /// VectorRecord-on-the-fly conversion + Parquet out), no engine
    /// refactor was needed — the trait method just builds a synthetic
    /// `FlushParameters` and delegates.
    ///
    /// Returns `used_engine_specific_path = true` so the caller's
    /// `BulkLoader` knows it doesn't need the per-record WAL fallback.
    async fn ingest_sorted_segment(
        &self,
        collection_id: &str,
        base_path: &str,
        records: Vec<proximadb_records::ProximaRecord>,
    ) -> Result<crate::storage::traits::SegmentIngestResult> {
        use crate::proto::proximadb_v1::{Collection, StorageAssignment};

        let count = records.len();
        if count == 0 {
            return Ok(crate::storage::traits::SegmentIngestResult {
                collection_id: collection_id.to_string(),
                record_count: 0,
                synthetic_segment_id: "empty".to_string(),
                used_engine_specific_path: true,
            });
        }

        // Build a minimal Collection so flush_ops knows where to write.
        // `config = None` lets dimension fall back to inspecting
        // records[0].embeddings[0].dim inside the flush path (per the
        // existing fallback chain). `storage_assignment.base_location`
        // is the only field flush_ops::flush actually reads.
        let collection_config = Some(Collection {
            id: collection_id.to_string(),
            config: None,
            stats: None,
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(StorageAssignment {
                primary_path: base_path.to_string(),
                backup_paths: vec![],
                engine: 0, // engine enum slot — flush_ops doesn't read this
                engine_config: std::collections::HashMap::new(),
                base_location: base_path.to_string(),
                assigned_at: 0,
                ..Default::default()
            }),
        });

        let params = FlushParameters {
            collection_id: Some(
                collection_id
                    .parse::<u64>()
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "NOVA bulk ingest requires a numeric catalog object id, got {collection_id:?}: {error}"
                        )
                    })?
                    .to_string(),
            ),
            force: true,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            vector_records: records,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config,
            estimated_size: 0,
        };

        let flush_result = self.flush_ops.flush(&params).await?;
        let synthetic_segment_id = flush_result
            .file_paths
            .first()
            .cloned()
            .unwrap_or_else(|| format!("nova-bulkload-{collection_id}-{count}"));

        Ok(crate::storage::traits::SegmentIngestResult {
            collection_id: collection_id.to_string(),
            record_count: count,
            synthetic_segment_id,
            used_engine_specific_path: true,
        })
    }

    async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
        // Delegate to modularized compaction operations
        self.compaction_ops.compact(params).await
    }

    async fn collection_stats(
        &self,
        _collection_id: &str,
    ) -> Result<crate::storage::traits::CollectionStats> {
        let stats = self.statistics.read().await;
        let total_bytes = stats.total_storage_bytes;
        let collection_count = stats.collection_count as u64;

        // Estimate per-collection row count from total storage and collection count
        let per_collection_bytes = total_bytes
            .checked_div(collection_count)
            .unwrap_or(total_bytes);

        // NOVA uses columnar Parquet: avg ~256 bytes per vector after compression
        let avg_record_bytes: u64 = 256;
        let estimated_row_count = if avg_record_bytes > 0 && per_collection_bytes > 0 {
            per_collection_bytes / avg_record_bytes
        } else {
            0
        };

        Ok(crate::storage::traits::CollectionStats {
            row_count: estimated_row_count,
            avg_vector_bytes: avg_record_bytes,
            engine_strategy: crate::storage::traits::StorageFormatStrategy::Nova,
            has_metadata_index: true, // NOVA has zone maps and bloom filters
            has_hnsw_index: false,
            total_bytes: per_collection_bytes,
            dimension: None,
            index_type: Some("zone_map".to_string()),
        })
    }

    async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Engine is stateless, so we report engine-level metrics only
        metrics.insert("engine_type".to_string(), serde_json::json!("NOVA"));
        metrics.insert("columnar_engine".to_string(), serde_json::json!(true));

        // Deferred: Collect actual metrics from storage when needed
        let total_files = 0;
        let total_row_groups = 0;
        metrics.insert(
            "total_parquet_files".to_string(),
            serde_json::json!(total_files),
        );
        metrics.insert(
            "total_row_groups".to_string(),
            serde_json::json!(total_row_groups),
        );
        let stats = self.statistics.read().await;
        // Use existing fields instead of non-existent ones
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
            serde_json::json!(format!("{:?}", self.hardware.cpu)),
        );
        metrics.insert("columnar_optimization".to_string(), serde_json::json!(true));
        Ok(metrics)
    }

    async fn vector_by_id(
        &self,
        collection_id: &str,
        base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        debug!(
            "NOVA get vector: collection={}, base_path={}, id={}",
            collection_id, base_path, vector_id
        );

        // Point read over this collection's flushed Parquet files. This was
        // previously a `Ok(None)` placeholder, then (post-#1743) a naive
        // first-match scan via `UnifiedParquetReader::read_all_records` —
        // that reader excludes `version`/`expires_at` from the `ProximaRecord`s
        // it builds (see columnar_reader.rs), so it can't be used for MVCC
        // resolution or dead-record filtering. This reads the raw Arrow
        // columns directly instead, mirroring the pattern VIPER's own
        // point-read and the shared compaction dedup
        // (columnar_compaction.rs::deduplicate_arrow_batches) already use —
        // NOT VIPER's dead-record check itself, which has its own bugs (see
        // TD-DSEFF-2): it compares the millisecond `expires_at` column
        // against `timestamp_micros()` (1000x unit mismatch) and skips
        // candidates mid-scan rather than resolving MVCC first, so a
        // tombstone can lose to a stale live version instead of shadowing it.
        let data_dir = format!("{}/{}/data", base_path, collection_id);
        // Same plumbing the search path uses: base filesystem from the
        // factory + the collection-scoped UnifiedCachingFilesystem wrapper.
        let fs = self.filesystem.get_filesystem(&data_dir)?;
        let unified_fs = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                fs,
                collection_id.to_string(),
                crate::storage::engines::ENGINE_NOVA.to_string(),
            ),
        );

        use proximadb_storage_filesystem_types::FileSystem as _;
        // A missing data directory means the collection has not flushed yet.
        // Any other filesystem error is material: swallowing it could make an
        // older file look like the MVCC winner.
        if !unified_fs.exists(&data_dir).await? {
            return Ok(None);
        }
        let entries = unified_fs.list(&data_dir).await?;
        let files: Vec<String> = entries
            .into_iter()
            .filter(|e| !e.metadata.is_directory && e.name.ends_with(".parquet"))
            .map(|e| format!("{}/{}", data_dir, e.name))
            .collect();

        use arrow_array::{
            Array, BooleanArray, Float32Array, Float64Array, Int64Array, ListArray, MapArray,
            StringArray, StructArray, UInt32Array,
        };
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        // Resolve MVCC winner across ALL candidate rows in ALL files before
        // any liveness check — a delete is itself a newer version and must
        // be able to shadow an older live one (CLAUDE.md invariant 16d).
        let mut best_match: Option<(VectorRecord, i64, i64)> = None; // (record, version, timestamp)

        for file_path in &files {
            let file_fs = self.filesystem.get_filesystem(file_path)?;
            let data = file_fs.read(file_path).await.map_err(|e| {
                anyhow!(
                    "NOVA point read for '{}' could not read candidate file '{}': {}",
                    vector_id,
                    file_path,
                    e
                )
            })?;
            let reader_builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(data))
                .map_err(|e| {
                    anyhow!(
                        "NOVA point read for '{}' could not decode candidate file '{}': {}",
                        vector_id,
                        file_path,
                        e
                    )
                })?;
            let mut batch_reader = reader_builder.build()?;
            for batch in &mut batch_reader {
                let batch = batch?;
                let id_array = batch
                    .column_by_name(FIELD_ID)
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| anyhow!("Missing or invalid 'id' column"))?;
                for row_idx in 0..batch.num_rows() {
                    if id_array.is_null(row_idx) || id_array.value(row_idx) != vector_id {
                        continue;
                    }
                    let timestamp = batch
                        .column_by_name(FIELD_TIMESTAMP)
                        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                        .filter(|a| !a.is_null(row_idx))
                        .map_or(0, |a| a.value(row_idx));
                    // NOVA's schema (schema_builder.rs) declares FIELD_VERSION
                    // as UInt32, NOT Int64 — unlike VIPER's own factory schema,
                    // which declares it Int8 (yet another VIPER unit/type
                    // inconsistency, out of scope here per TD-DSEFF-2).
                    let version: i64 = batch
                        .column_by_name(FIELD_VERSION)
                        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
                        .filter(|a| !a.is_null(row_idx))
                        .map_or(0, |a| a.value(row_idx) as i64);
                    let expires_at = batch
                        .column_by_name(FIELD_EXPIRES_AT)
                        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                        .and_then(|a| {
                            if a.is_null(row_idx) {
                                None
                            } else {
                                Some(a.value(row_idx))
                            }
                        });

                    let vector_values = batch
                        .column_by_name(FIELD_VECTOR_FP32)
                        .and_then(|col| {
                            crate::storage::engines::viper::engine::list_row_values(col, row_idx)
                        })
                        .ok_or_else(|| anyhow!("Missing or invalid 'vector_fp32' column"))?;
                    let vector_float_array = vector_values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| anyhow!("Invalid vector values type"))?;
                    let vector: Vec<f32> = (0..vector_float_array.len())
                        .map(|i| vector_float_array.value(i))
                        .collect();

                    let updated_at = batch
                        .column_by_name(FIELD_UPDATED_AT)
                        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                        .filter(|a| !a.is_null(row_idx))
                        .map_or(0, |a| a.value(row_idx));

                    let source = batch
                        .column_by_name(FIELD_SOURCE)
                        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                        .filter(|a| !a.is_null(row_idx))
                        .map(|a| a.value(row_idx).to_string());

                    let mut metadata_map: HashMap<String, crate::proto::proximadb_v1::SqlValue> =
                        HashMap::new();
                    if let Some(extra_meta_col) = batch.column_by_name(FIELD_EXTRA_META) {
                        let (struct_array, range) = if let Some(map_array) =
                            extra_meta_col.as_any().downcast_ref::<MapArray>()
                            && !map_array.is_null(row_idx)
                        {
                            let start = map_array.value_offsets()[row_idx] as usize;
                            let end = map_array.value_offsets()[row_idx + 1] as usize;
                            (Some(map_array.entries().clone()), start..end)
                        } else if let Some(list_array) =
                            extra_meta_col.as_any().downcast_ref::<ListArray>()
                            && !list_array.is_null(row_idx)
                        {
                            let values = list_array.value(row_idx);
                            let len = values.len();
                            (
                                values.as_any().downcast_ref::<StructArray>().cloned(),
                                0..len,
                            )
                        } else {
                            (None, 0..0)
                        };

                        if let Some(struct_array) = struct_array
                            && let (Some(key_array), Some(value_array)) = (
                                struct_array
                                    .column(0)
                                    .as_any()
                                    .downcast_ref::<StringArray>(),
                                struct_array
                                    .column(1)
                                    .as_any()
                                    .downcast_ref::<StringArray>(),
                            )
                        {
                            for kv_idx in range {
                                if !struct_array.is_null(kv_idx)
                                    && !key_array.is_null(kv_idx)
                                    && !value_array.is_null(kv_idx)
                                {
                                    metadata_map.insert(
                                        key_array.value(kv_idx).to_string(),
                                        crate::proto::proximadb_v1::SqlValue {
                                            value: Some(
                                                crate::proto::proximadb_v1::sql_value::Value::StringValue(
                                                    value_array.value(kv_idx).to_string(),
                                                ),
                                            ),
                                        },
                                    );
                                }
                            }
                        }
                    }
                    for field in batch.schema().fields() {
                        let field_name = field.name();
                        let is_system_field = matches!(
                            field_name.as_str(),
                            FIELD_ID
                                | FIELD_COLLECTION_ID
                                | FIELD_ROW_GROUP_OFFSET
                                | FIELD_ROW_INDEX
                                | FIELD_VECTOR_FP32
                                | FIELD_TIMESTAMP
                                | "created_at"
                                | FIELD_UPDATED_AT
                                | FIELD_VERSION
                                | FIELD_EXPIRES_AT
                                | FIELD_SOURCE
                                | FIELD_EXTRA_META
                                | FIELD_IS_DELETED
                                | FIELD_SCHEMA_VERSION
                        ) || QUANTIZATION_COLUMNS
                            .contains(&field_name.as_str());
                        if !is_system_field
                            && let Some(column) = batch.column_by_name(field_name)
                            && !column.is_null(row_idx)
                        {
                            use crate::proto::proximadb_v1::sql_value::Value;
                            let value = match field.data_type() {
                                arrow_schema::DataType::Utf8 => column
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .map(|a| Value::StringValue(a.value(row_idx).to_string())),
                                arrow_schema::DataType::Int64 => column
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .map(|a| Value::Int64Value(a.value(row_idx))),
                                arrow_schema::DataType::Float64 => column
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .map(|a| Value::NumberValue(a.value(row_idx))),
                                arrow_schema::DataType::Boolean => column
                                    .as_any()
                                    .downcast_ref::<BooleanArray>()
                                    .map(|a| Value::BoolValue(a.value(row_idx))),
                                _ => None,
                            };
                            if let Some(value) = value {
                                metadata_map.insert(
                                    field_name.to_string(),
                                    crate::proto::proximadb_v1::SqlValue { value: Some(value) },
                                );
                            }
                        }
                    }

                    let record = VectorRecord {
                        id: vector_id.to_string(),
                        vector,
                        metadata: metadata_map,
                        timestamp: Some(timestamp),
                        updated_at: Some(updated_at),
                        expires_at,
                        version: Some(version as u32),
                        source,
                    };

                    let is_better = match &best_match {
                        Some((_, best_version, best_timestamp)) => {
                            version > *best_version
                                || (version == *best_version && timestamp > *best_timestamp)
                        }
                        None => true,
                    };
                    if is_better {
                        best_match = Some((record, version, timestamp));
                    }
                }
            }
        }

        let Some((record, ..)) = best_match else {
            return Ok(None);
        };
        // `From<VectorRecord> for ProximaRecord` (conversions.rs) already
        // correctly converts the millisecond `expires_at` to `valid_to_ns`
        // (nanoseconds) via `ms_to_ns`, preserving the `Some(0)` tombstone
        // sentinel through the conversion — reuse it rather than re-deriving
        // the unit conversion here.
        let record = proximadb_records::ProximaRecord::from(record);
        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        if record.is_dead(now_ns) {
            return Ok(None);
        }

        // Do not use the global vector cache here: NOVA's flush path has no
        // version-aware invalidation, so an ID-only key can hide a newer MVCC
        // version or tombstone indefinitely.
        Ok(Some(record))
    }

    async fn search_vectors_unified(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        let collection_id = ctx.collection_id();
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow!("Query vector required for search"))?;
        let k = ctx.top_k();
        let filter_expression = ctx.search_params.filter_expression.as_ref();

        // ========================================================================
        // PHASE 0: TRY AXIS-BASED SEARCH FIRST (HNSW/IVF) - FASTEST PATH
        // ========================================================================
        // Use AXIS manager if available for O(log N) approximate search
        let exact_mode = matches!(ctx.search_params.search_mode, SearchMode::Exact);
        let has_axis_manager = self.axis_manager().is_some();
        if has_axis_manager {
            tracing::debug!("🔍 NOVA: AXIS manager is available for HNSW/IVF search");
        }

        if exact_mode {
            tracing::debug!("🎯 NOVA: SearchMode::Exact requested; skipping approximate AXIS path");
        } else if let Some(axis_manager) = self.axis_manager() {
            tracing::debug!(
                "🔍 NOVA: Attempting AXIS search for collection='{}', top_k={}, dimension={}",
                collection_id,
                k,
                query_vector.len()
            );

            // Convert filter expression to index format
            let index_filters = Self::convert_filter_to_index(filter_expression);

            // Build hybrid query for AXIS
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

            // Execute AXIS query (HNSW or IVF based on index type)
            let axis_start = std::time::Instant::now();
            match axis_manager.query(hybrid_query).await {
                Ok(axis_results) => {
                    let axis_duration = axis_start.elapsed();
                    tracing::info!(
                        "✅ NOVA: AXIS search completed in {:?} - found {} candidates",
                        axis_duration,
                        axis_results.results.len()
                    );

                    // Convert AXIS results to OptimizedSearchRecord
                    let results: Vec<OptimizedSearchRecord> = axis_results
                        .results
                        .into_iter()
                        .take(k)
                        .map(|scored| OptimizedSearchRecord {
                            id: scored.vector_id.clone(),
                            vector_id: Some(scored.vector_id),
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
                        "⚠️ NOVA: AXIS returned no results, falling back to progressive columnar search"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        "⚠️ NOVA: AXIS search failed: {}, falling back to progressive columnar search",
                        e
                    );
                }
            }
        }

        // ========================================================================
        // PHASE 1: PROGRESSIVE COLUMNAR SEARCH (Fallback)
        // ========================================================================
        // Delegate to modularized search operations for progressive columnar search
        self.search_ops.search_vectors_unified(ctx).await
    }

    // Old search implementation removed - using modularized search_operations.rs

    // OPTIONAL OPERATIONS
    async fn optimize(&self, collection_id: &str) -> Result<()> {
        info!("NOVA optimize: collection={}", collection_id);
        // Trigger compaction for optimization
        let params = CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: false,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: Some(60000),
            priority: OperationPriority::Low,
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
            status: "Healthy".to_string(),
            last_check: chrono::Utc::now(),
            response_time_ms: 1.0,
            warnings: vec![],
            metrics: HashMap::new(),
            error_count: 0,
        })
    }

    fn supports_feature(&self, feature: &str) -> bool {
        matches!(
            feature,
            "id_lookup"
                | "similarity_search"
                | "columnar_search"
                | "quantization"
                | "compression"
                | "batch_operations"
                | "predicate_pushdown"
                | "projection"
        )
    }

    /// Engine-level RLS predicate (Tier 1 multi-tenant safety hardening).
    ///
    /// Mirrors the SST implementation at `src/storage/engines/sst/trait_impl.rs:303`.
    /// Extracts tenant_id from the query context so NOVA's scan iterator can
    /// apply row-level isolation without referencing application-layer middleware.
    fn rls_record_filter(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Option<crate::storage::traits::RlsRecordPredicate> {
        let tenant_id = ctx
            .tenant_context
            .as_ref()
            .map(|tc| tc.tenant_id.as_str())
            .or_else(|| {
                ctx.user_context
                    .as_ref()
                    .and_then(|uc| uc.tenant_id.as_deref())
            });

        let principal = ctx.user_context.as_ref().map(|uc| uc.user_id.as_str());

        if tenant_id.is_none() && principal.is_none() {
            return None;
        }

        Some(crate::storage::traits::RlsRecordPredicate {
            required_tenant_id: tenant_id.map(str::to_string),
            required_principal: principal.map(str::to_string),
        })
    }
}

impl crate::storage::traits::EngineFilesystemAccess for NovaEngine {
    fn get_filesystem_factory(
        &self,
    ) -> &crate::storage::persistence::filesystem::FilesystemFactory {
        &self.filesystem
    }
}

impl NovaFile {
    /// Load record at specific location
    pub fn load_record_at_location(
        &self,
        location: &crate::storage::engines::core::formats::columnar::id_index::ParquetLocation,
    ) -> Result<VectorRecord> {
        // In production, would load from Parquet row group
        Ok(VectorRecord {
            id: format!("vec_rg{}_row{}", location.row_group_id, location.row_offset),
            vector: vec![0.0; self.metadata.dimension],
            metadata: std::collections::HashMap::new(),
            timestamp: Some(0),
            updated_at: None,
            expires_at: None,
            version: None,
            source: None,
        })
    }
}

// Deferred: Fix columnar search config implementation when module is available
/*
impl crate::storage::engines::core::formats::columnar::columnar_search::ColumnarSearchConfig {
    /// Create from search parameters
    pub fn from_params(params: Option<&serde_json::Value>) -> Self {
        // Parse parameters or use defaults
        Self::default()
    }
}
*/

/// Implementation of UniversallyOptimized trait for NOVA engine
#[async_trait]
impl UniversallyOptimized for NovaEngine {
    /// Get the universal performance optimizer instance
    fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer {
        &self.universal_optimizer
    }

    /// NOVA-specific optimization setup
    async fn setup_engine_optimizations(&self) -> Result<()> {
        // NOVA-specific optimizations for columnar storage
        info!("🔧 NOVA Engine: Setting up universal performance optimizations");

        // Initialize columnar-specific optimizations
        let config = self.universal_optimizer.get_config();
        debug!("   Cache size: {}MB", config.cache_size_mb);
        debug!("   Parallel operations: {}", config.parallel_operations);
        debug!("   Prefetching enabled: {}", config.enable_prefetching);
        debug!(
            "   Memory mapping enabled: {}",
            config.enable_memory_mapping
        );

        // NOVA is ready for columnar analytics operations
        info!("✅ NOVA Engine: Universal optimizations configured for columnar analytics");
        Ok(())
    }

    /// NOVA-specific performance metrics
    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Basic NOVA metrics
        let stats = self.statistics.read().await;
        metrics.insert(
            "nova_total_storage_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.total_storage_bytes)),
        );
        metrics.insert(
            "nova_memory_usage_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.memory_usage_bytes)),
        );
        metrics.insert(
            "nova_collection_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(stats.collection_count)),
        );
        metrics.insert(
            "nova_pending_flushes".to_string(),
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

// Helper methods for NovaEngine
impl NovaEngine {
    /// Get the AXIS manager if configured
    ///
    /// Returns the optional AXIS manager for HNSW/IVF-based search.
    /// When available, AXIS provides O(log N) approximate nearest neighbor search
    /// that is significantly faster than progressive columnar search.
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
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};
        use proximadb_index_traits::{IndexFilterOperator, IndexMetadataFilter};

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
                        tracing::debug!(
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
                tracing::debug!("OR/NOT filters not supported by index, will use post-filtering");
            }
        }

        index_filters
    }

    // Removed unnecessary helper methods - engines receive all params directly
    // No need for CollectionService, distance/quantization engines are already in the struct
}

// Additional helper methods for NovaEngine
impl NovaEngine {
    /// Fallback to direct search when orchestration fails
    #[allow(dead_code)]
    async fn fallback_to_direct_search(
        &self,
        _ctx: &crate::storage::traits::StorageQueryContext,
        collection_id: &str,
        storage_path: &str,
        _query_vector: &[f32],
        top_k: usize,
        distance_metric: proximadb_distance_kernel::DistanceMetric,
        _filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        tracing::warn!("🔄 NOVA: Falling back to direct search implementation");

        // Use the existing search implementation
        // Load files from storage
        let files = self
            .load_collection_files(collection_id, storage_path)
            .await?;
        let mut all_results = Vec::new();

        // Search each NOVA file using columnar optimization
        for _nova_file in &files {
            // Placeholder - would implement actual columnar search
            let results: Vec<(crate::proto::proximadb_v1::VectorRecord, f32)> = Vec::new();

            // Convert to search results
            for (record, score) in results {
                all_results.push((record, score));
            }
        }

        // Use bounded priority queue for efficient top-k selection
        let mut priority_queue = BoundedPriorityQueue::new(top_k);

        // Insert all results into bounded queue
        for (record, distance) in all_results {
            // Use SimilarityResult for proper normalization
            let similarity_result =
                proximadb_distance_kernel::SimilarityResult::new(distance, distance_metric);

            let search_record = OptimizedSearchRecord {
                id: record.id.clone(),
                vector_id: Some(record.id.clone()),
                score: similarity_result.normalized_score,
                similarity: Some(similarity_result.normalized_score),
                vector: Some(Arc::new(record.vector.clone())),
                metadata: crate::core::search::results::sql_map_to_proxima(record.metadata.clone()),
                ..Default::default()
            };

            priority_queue.try_insert(search_record);
        }

        // Get sorted results from bounded queue
        let final_results = priority_queue.into_sorted_vec();

        // Return the results from bounded priority queue
        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nova_engine_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = NovaEngine::new().await.unwrap();
        assert_eq!(engine.format_name(), "NOVA");
        assert_eq!(engine.format_version(), "1.0.0");
    }

    #[tokio::test]
    async fn test_nova_feature_support() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let engine = NovaEngine::new().await.unwrap();
        assert!(engine.supports_feature("id_lookup"));
        assert!(engine.supports_feature("columnar_search"));
        assert!(engine.supports_feature("predicate_pushdown"));
        assert!(engine.supports_feature("projection"));
        assert!(!engine.supports_feature("unknown_feature"));
    }
}
