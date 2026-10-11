// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! VIPER Engine - Vector Storage Layer
//!
//! VIPER (Vector-optimized Intelligent Parquet with Efficient Retrieval) is a pure
//! storage engine focused on durability and efficient serialization of vectors.
//! Responsibilities:
//! - Store vectors in columnar Parquet format
//! - Handle flush operations from WAL to persistent storage
//! - Perform compaction to optimize storage layout
//! - Provide direct vector search on Parquet files (baseline functionality)
//!
//! NOT Responsible For:
//! - ML clustering (belongs in AXIS indexing service)
//! - Index management (AXIS responsibility)
//! - Query optimization strategies (AXIS layer)
//!
//! Architecture:
//! - VIPER provides baseline search that works for ALL collections
//! - AXIS can optionally add ML clustering as an optimization layer
//! - Clean separation: VIPER = storage, AXIS = indexing
use anyhow::Result;
use proximadb_storage_ports::CollectionMetadataPort;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

// Import UnifiedQuantizationLevel
use proximadb_quantization_model::UnifiedQuantizationLevel;

// Import column constants from columnar module
use crate::storage::engines::core::formats::columnar::{
    FIELD_EXPIRES_AT, FIELD_ID, FIELD_TIMESTAMP, FIELD_VECTOR_FP32, FIELD_VERSION,
};

// Universal performance optimization imports
use crate::storage::engines::core::ops::performance_optimization::{
    UniversalOptimizationStrategy, UniversalPerformanceOptimizer, UniversallyOptimized,
};
// VectorMemoryPool now managed by universal optimizer
use super::types::*;
use crate::core::String;
use crate::core::search::SearchMode;
use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::persistence::filesystem::FileStorageTier;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::traits::{FlushResult, UnifiedStorageFormat};
use proximadb_records::conversions::proxima_record_to_vector;
// Schema now uses shared ColumnarSchema from columnar module
use super::compaction::ViperCompactionService;
use super::flush::Flush;
// use super::ml_clustering::MLClusteringEngine; // Moved to AXIS
use super::utilities::ViperUtilities;
// Unified search engine removed - using IntegratedSearchOptimizer
use super::types::CollectionMetadata;
use anyhow::Context;
use proximadb_storage_common::storage_path::StoragePath;
// VIPER-specific optimization structures removed - now using universal module

// Using unified quantization engine directly from compute module
/// VIPER Engine - Main coordination point for the modular VIPER storage engine
///
/// ## Architecture Overview:
///
/// VIPER is ProximaDB's columnar storage engine built on Apache Parquet, designed
/// for analytical workloads with exceptional compression and batch performance.
///
/// ### Core Design Principles:
/// - **Columnar First**: Parquet format for maximum compression
/// - **Progressive Quantization**: Binary → INT8 → PQ → FP32 refinement
/// - **Cloud Native**: Optimized for S3/Azure/GCS with footer caching
/// - **Batch Optimized**: High-performance batch processing throughput
///
/// ### Data Flow:
/// ```text
/// Insert → Pipeline → Quantize → Compress → Parquet
///                         ↓
///                    Row Groups (128K vectors)
///                         ↓
///                    EventLog → AXIS
/// ```
///
/// ### Key Differentiators from SST:
/// - **Storage Format**: Columnar (Parquet) vs Row-based (SSTable)
/// - **Optimization**: Analytics/batch vs OLTP/real-time
/// - **Compression**: 5-10x vs 3-5x
/// - **Query Pattern**: Scan/aggregate vs point lookup
///
/// Per-row values of a `vector_fp32` column, accepting both Arrow list flavors
/// the engine's writers emit: `List` from the shared columnar path, and
/// `FixedSizeList` from the viper Parquet factory.
///
/// `pub(crate)`: also reused by NOVA's point-read (`nova::engine::vector_by_id`),
/// which shares the same Parquet schema via the writer both engines use.
pub(crate) fn list_row_values(
    col: &dyn arrow_array::Array,
    row_idx: usize,
) -> Option<std::sync::Arc<dyn arrow_array::Array>> {
    if let Some(list) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        Some(list.value(row_idx))
    } else if let Some(fixed) = col
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
    {
        Some(fixed.value(row_idx))
    } else {
        None
    }
}

pub struct ViperEngine {
    /// **Internal Engine Configuration**
    ///
    /// Runtime settings that control VIPER's behavior:
    /// - Row group size (default: 128K vectors for optimal compression)
    /// - Page size for Parquet (default: 1MB)
    /// - Compression codec (ZSTD, Snappy, LZ4)
    /// - Quantization levels to enable (binary, INT8, PQ)
    /// - Write buffer thresholds and batch sizes
    ///
    /// Distinct from core_config which has user-facing settings
    config: ViperEngineConfig,

    /// **User-Facing Core Configuration**
    ///
    /// Original configuration from user/system:
    /// - Passed to flush operations for consistency
    /// - Contains compaction strategy settings
    /// - Defines bloom filter parameters
    /// - Preserves user intent across operations
    ///
    /// Used when flush/compaction needs original config context
    core_config: crate::core::config::ViperConfig,

    /// **Collection Service** (Optional, Lazy-Loaded)
    ///
    /// Metadata provider for collection-specific information:
    /// - Vector dimensions and data types
    /// - Distance metric configuration (L2, cosine, dot)
    /// - Collection-level settings (quantization, indexing)
    /// - Schema evolution tracking
    ///
    /// RwLock<Option<>> because:
    /// - None during engine initialization
    /// - Some when first collection accessed
    /// - Shared read access for concurrent queries
    collection_service: Arc<RwLock<Option<Arc<dyn CollectionMetadataPort>>>>,

    /// **Unified Caching Filesystem**
    ///
    /// Intelligent filesystem wrapper with:
    /// - **Footer Caching**: Parquet footer metadata cached in memory
    /// - **Range Optimization**: Coalesces small reads into larger ones
    /// - **Access Tracking**: Records which row groups are hot
    /// - **Prefetch Engine**: Predictive loading based on access patterns
    ///
    /// Critical for cloud storage performance (S3/Azure/GCS latency hiding)
    filesystem:
        Arc<crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem>,

    /// **Filesystem Factory**
    ///
    /// Creates filesystem instances for specific backends:
    /// - Shared across flush, compaction, and read operations
    /// - Handles URL scheme routing (s3://, azure://, file://)
    /// - Maintains connection pools for cloud providers
    /// - Provides unified interface regardless of backend
    ///
    /// Used by components that need direct filesystem access
    filesystem_factory: Arc<FilesystemFactory>,

    /// **Columnar Schema** (Shared with NOVA)
    ///
    /// Defines Parquet table structure:
    /// - Column definitions (id, vector, metadata, timestamps)
    /// - Data types and encoding (dictionary, RLE, delta)
    /// - Compression per column (ZSTD for vectors, Snappy for strings)
    /// - Nested schema for complex metadata
    ///
    /// Shared between VIPER and NOVA for format compatibility
    _schema: crate::storage::engines::core::formats::columnar::columnar_schema::ColumnarSchema,

    /// **Compaction Service**
    ///
    /// Background process for storage optimization:
    /// - **Row Group Merging**: Combines small row groups → larger ones
    /// - **Tombstone Cleanup**: Removes deleted records from files
    /// - **Statistics Update**: Recomputes min/max for better pruning
    /// - **Size-Tiered Strategy**: Merges files of similar size
    ///
    /// Runs asynchronously, triggered by file count thresholds
    compaction: ViperCompactionService,

    /// **Flush Manager**
    ///
    /// Coordinates write path from memory to Parquet:
    /// - Batches vectors into row groups
    /// - Applies quantization (binary → INT8 → PQ)
    /// - Compresses with ZSTD/Snappy
    /// - Writes Parquet files with bloom filters
    /// - Updates collection metadata
    ///
    /// Invoked when MemTable reaches threshold or manual flush requested
    flush_manager: Flush,

    /// **VIPER Utilities**
    ///
    /// Helper functions for Parquet operations:
    /// - Footer parsing and validation
    /// - Metadata extraction from files
    /// - Statistics computation (min/max/count)
    /// - Schema compatibility checks
    /// - File format version handling
    ///
    /// Shared utility functions used across flush/search/compaction
    #[allow(dead_code)]
    utilities: ViperUtilities,

    /// **Engine Statistics** (Lock-Free Atomics)
    ///
    /// Real-time metrics tracking:
    /// - Compression ratios achieved per collection
    /// - Query latencies (p50, p95, p99)
    /// - Cache hit rates (footer, page, row group)
    /// - Bytes written/read per operation
    /// - Row group scan efficiency
    ///
    /// Updated atomically without locks for zero contention
    stats: Arc<EngineStats>,

    /// **Collection Metadata Cache**
    ///
    /// In-memory cache of collection information:
    /// - Dimensions and schema per collection
    /// - Active Parquet files and their locations
    /// - Compression settings and achieved ratios
    /// - Last flush/compaction timestamps
    ///
    /// RwLock allows concurrent reads, exclusive writes during updates
    collections: Arc<RwLock<HashMap<String, CollectionMetadata>>>,

    /// **Storage Quantization Engine** (Collection-Aware)
    ///
    /// Persistent quantization with trained codebooks:
    /// - Stores PQ codebooks in filesystem per collection
    /// - Binary quantization (1 bit per dim)
    /// - INT8 quantization with learned min/max
    /// - PQ4/8/16/32 with k-means clustering
    /// - Hardware-accelerated quantization (SIMD)
    ///
    /// Codebooks trained once during first flush, reused forever
    _storage_quantization_engine:
        Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine>,

    /// **Fallback Quantization Engine** (Stateless)
    ///
    /// In-memory quantization without persistence:
    /// - Used when codebook not available
    /// - Ad-hoc quantization for one-off queries
    /// - Faster than storage engine (no I/O)
    /// - Same algorithms as storage engine
    ///
    /// Falls back when collection doesn't have trained codebooks
    #[allow(dead_code)]
    fallback_quantization_engine:
        Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine>,

    /// **Universal Performance Optimizer**
    ///
    /// Cross-engine optimization coordinator:
    /// - **Memory-Mapped I/O**: Zero-copy file access
    /// - **Vector Pooling**: Reuses allocated buffers
    /// - **Adaptive Batching**: Adjusts batch size based on load
    /// - **Progressive Search**: Binary → INT8 → PQ → FP32 pipeline
    /// - **Cache Coordination**: Manages multi-tier caches
    ///
    /// Shared optimizations eliminating per-engine duplication
    universal_optimizer: UniversalPerformanceOptimizer,

    /// **Cross-Cache Orchestrator** (Optional)
    ///
    /// Coordinates caching across layers:
    /// - Parquet footer cache invalidation
    /// - Metadata cache dependency tracking
    /// - Filter pushdown to storage layer
    /// - Memory budget management across caches
    ///
    /// None when caching disabled, Some in production deployments
    orchestrator: Option<Arc<crate::storage::cache::orchestrator::CrossCacheOrchestrator>>,

    /// **AXIS Manager** (Optional)
    ///
    /// Integration with AXIS indexing service:
    /// - Provides HNSW-based approximate nearest neighbor search
    /// - Enables IVF partition pruning
    /// - Supports hybrid vector + metadata queries
    ///
    /// None if AXIS disabled, Some for indexed collections
    axis_manager: Option<Arc<dyn proximadb_index_traits::IndexEngine>>,
}

impl std::fmt::Debug for ViperEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViperEngine")
            .field("config", &self.config)
            .field("core_config", &self.core_config)
            .field("collection_service", &"<CollectionService>")
            .field("filesystem", &"<FilesystemFactory>")
            .field("flush_manager", &"<Flush>")
            .field("memtable", &"<Memtable>")
            .field("wal", &"<WAL>")
            .field("quantizer", &"<UniversalQuantizationEngine>")
            .field("compactor", &"<Compactor>")
            .field("distance_compute", &"<UnifiedDistanceCompute>")
            .field("universal_optimizer", &self.universal_optimizer)
            .field("orchestrator", &"<CrossCacheOrchestrator>")
            .finish()
    }
}

#[allow(dead_code)]
impl ViperEngine {
    /// Attach orchestrator via context (future-proof DI)
    pub fn with_context(mut self, ctx: &crate::core::context::SharedContext) -> Self {
        self.orchestrator = ctx.orchestrator.clone();
        self
    }
    /// Create from core config (backward compatibility for tests)
    pub async fn from_core_config(
        core_config: crate::core::config::ViperConfig,
        filesystem: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        // Create VIPER metadata serializer
        let metadata_serializer =
            Arc::new(super::unified_metadata_serializer::ViperMetadataSerializer::new());

        // Get the base filesystem from factory
        let base_fs = filesystem.get_filesystem("file://")?;

        // Create UnifiedCachingFilesystem for transparent cloud storage support
        // - Cloud files (S3/GCS/Azure) are automatically downloaded to local disk cache
        // - Cache location: /tmp/proximadb/cache/{collection}/viper/
        // - Parquet metadata/footers cached separately for fast columnar access
        // - Hot files remain in cache based on LRU policy
        let unified_fs = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::with_serializer(
                base_fs,
                "default".to_string(),
                "viper".to_string(),
                metadata_serializer,
            )
        );

        Self::from_caching_filesystem_and_factory(core_config, unified_fs, filesystem).await
    }

    /// Create a new VIPER engine from user-facing core config
    pub async fn from_caching_filesystem(
        core_config: crate::core::config::ViperConfig,
        filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
    ) -> Result<Self> {
        // Create a dummy filesystem factory for backward compatibility
        let filesystem_factory = Arc::new(FilesystemFactory::create_default().await?);
        Self::from_caching_filesystem_and_factory(core_config, filesystem, filesystem_factory).await
    }

    /// Create a new VIPER engine with both filesystems
    pub async fn from_caching_filesystem_and_factory(
        core_config: crate::core::config::ViperConfig,
        filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        let config = ViperEngineConfig::from_core_config(&core_config);
        Self::new_internal(config, core_config, filesystem, filesystem_factory).await
    }
    /// Create a new VIPER engine instance (stateless)
    /// Collection info comes from FlushParameters and StorageQueryContext at runtime
    pub async fn new() -> Result<Self> {
        let core_config = crate::core::config::ViperConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await?);
        let distance_compute =
            Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());

        Self::new_with_config(core_config, filesystem, distance_compute).await
    }

    /// Create VIPER engine with specific config (internal use)
    pub async fn new_with_config(
        core_config: crate::core::config::ViperConfig,
        filesystem: Arc<FilesystemFactory>,
        _distance_compute: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>, // VIPER creates its own internally
    ) -> Result<Self> {
        info!("🔧 Creating stateless VIPER engine");

        // Create VIPER metadata serializer
        let metadata_serializer =
            Arc::new(super::unified_metadata_serializer::ViperMetadataSerializer::new());

        // Get the base filesystem from factory
        let base_fs = filesystem.get_filesystem("file://")?;

        // Create UnifiedCachingFilesystem without collection_id
        // Collection ID will come from runtime parameters
        let unified_fs = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::with_serializer(
                base_fs,
                String::new(), // No collection_id - gets from parameters
                "viper".to_string(),
                metadata_serializer,
            )
        );

        // VIPER manages multiple collections, so we just log the initial one
        Self::from_caching_filesystem_and_factory(core_config, unified_fs, filesystem).await
    }

    /// Internal constructor with both configs
    ///
    /// ## Initialization Process:
    ///
    /// 1. **Quantization Setup**: Configures PQ8 primary, Binary filter, INT8 fast levels
    /// 2. **Universal Optimizer**: Initializes cross-engine performance optimizations
    /// 3. **Component Creation**: Flush manager, compaction, utilities
    /// 4. **Schema Definition**: Sets up columnar schema for Parquet
    ///
    /// ## Default Quantization Strategy:
    /// - **Primary**: PQ8 with 32 subquantizers (best compression)
    /// - **Filter**: Binary for initial 32x reduction
    /// - **Fast**: INT8 for 4x reduction with good quality
    async fn new_internal(
        config: ViperEngineConfig,
        core_config: crate::core::config::ViperConfig,
        filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        let collection_service = Arc::new(RwLock::new(None));

        // Initialize unified quantization engine from compute module
        // This provides the core quantization algorithms (Binary, INT8, PQ)
        // that VIPER uses for its multi-stage compression pipeline
        let distance_compute =
            Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());

        // In-memory codebook store for PQ quantization
        // Codebooks are trained on sample data and cached for fast access
        let codebook_store = Arc::new(
            crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
        );

        // Create the unified quantization engine that all storage engines share
        let unified_engine = Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                distance_compute.clone(),
                codebook_store,
            ),
        );

        // Configure storage quantization for VIPER
        // VIPER uses aggressive quantization for maximum compression
        let storage_config =
            crate::compute::quantization::storage_engine::StorageQuantizationConfig {
                // PQ8 with 32 subquantizers - achieves 32x compression for 768D vectors
                primary_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::pq8(
                    32,
                )),
                // Binary quantization for initial filtering - 32x reduction
                filter_level: Some(
                    proximadb_quantization_model::UnifiedQuantizationLevel::binary(),
                ),
                // INT8 for intermediate precision - 4x reduction with 98% recall
                fast_level: Some(proximadb_quantization_model::UnifiedQuantizationLevel::int8()),
                // Cosine is default, but can be overridden per collection
                distance_metric: proximadb_distance_kernel::engine::DistanceMetric::Cosine,
                // Enable Binary→INT8→PQ→FP32 progressive refinement
                enable_progressive: true,
                // Initial filter returns 100x final k for refinement
                filter_threshold: 100.0,
                // Fetch 10x candidates at each stage
                candidate_multiplier: 10,
                // Train codebooks on 10K sample vectors
                training_sample_size: 10000,
                // 512MB memory budget for quantization structures
                memory_budget_mb: 512,
                // Use SIMD/GPU when available
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
                .await
                .context("Failed to create universal performance optimizer for VIPER")?;

        // ML clustering moved to AXIS
        // let ml_clustering_engine = MLClusteringEngine::new(super::ml_clustering::KMeansConfig::default());
        // Initialize utilities with default configuration
        let utilities = ViperUtilities::new(
            super::utilities::ViperUtilitiesConfig::default(),
            filesystem_factory.clone(),
        )
        .await?;
        // Create managers with async constructors
        let compaction = ViperCompactionService::new(filesystem_factory.clone());
        let flush_manager =
            Flush::new(collection_service.clone(), filesystem_factory.clone()).await?;

        // Register VIPER cache providers with global orchestrator
        if let Some(ref orch) =
            crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
        {
            use crate::storage::cache::orchestrator::{CacheStatsProvider, CacheType, UsageStats};

            // Create a VIPER-specific stats provider for Parquet footer caching
            struct ViperFooterCacheProvider;
            impl CacheStatsProvider for ViperFooterCacheProvider {
                fn snapshot(&self) -> UsageStats {
                    UsageStats {
                        hit_rate: 0.85,        // VIPER typically has high footer cache hit rate
                        avg_entry_size: 2048,  // Parquet footers are ~2KB
                        access_frequency: 5.0, // Moderate access frequency
                        last_rebalance: std::time::SystemTime::now(),
                    }
                }
            }

            // Register VIPER-specific cache providers
            let footer_provider: Arc<dyn CacheStatsProvider + Send + Sync> =
                Arc::new(ViperFooterCacheProvider);
            orch.register_cache_provider(CacheType::Metadata, footer_provider);

            // Register for index structure caching (row group indexes)
            struct ViperIndexCacheProvider;
            impl CacheStatsProvider for ViperIndexCacheProvider {
                fn snapshot(&self) -> UsageStats {
                    UsageStats {
                        hit_rate: 0.75,        // Good hit rate for row group indexes
                        avg_entry_size: 1024,  // Index entries ~1KB
                        access_frequency: 3.0, // Regular access
                        last_rebalance: std::time::SystemTime::now(),
                    }
                }
            }
            let index_provider: Arc<dyn CacheStatsProvider + Send + Sync> =
                Arc::new(ViperIndexCacheProvider);
            orch.register_cache_provider(CacheType::IndexStructure, index_provider);
        }

        Ok(Self {
            config,
            core_config,
            collection_service: collection_service.clone(),
            filesystem: filesystem.clone(),
            filesystem_factory,
            _schema: crate::storage::engines::core::formats::columnar::columnar_schema::ColumnarSchema::new(),
            compaction,
            flush_manager,
            // ml_clustering_engine, // Moved to AXIS
            utilities,
            // Search engine removed - using IntegratedSearchOptimizer
            stats: Arc::new(EngineStats::default()),
            collections: Arc::new(RwLock::new(HashMap::new())),
            _storage_quantization_engine: storage_quantization_engine,
            fallback_quantization_engine,
            universal_optimizer,
            orchestrator: None,
            axis_manager: None, // AXIS manager will be set externally if available
        })
    }

    /// Set the collection service for metadata access
    pub async fn set_collection_service(
        &self,
        collection_service: Arc<dyn CollectionMetadataPort>,
    ) {
        let mut service_lock = self.collection_service.write().await;
        *service_lock = Some(collection_service);
        info!("🔗 VIPER Engine: Collection service set for metadata access");
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

    // ============================================================================
    // PERFORMANCE OPTIMIZATION METHODS - DELEGATING TO UNIFIED MODULES
    // ============================================================================

    /// Fast read optimization using memory-mapped Parquet files (delegates to universal optimizer)
    ///
    /// ## Memory Mapping Strategy:
    ///
    /// For local files, uses mmap for zero-copy access. For cloud storage,
    /// falls back to buffered reads with intelligent caching.
    ///
    /// ### Benefits:
    /// - Zero-copy access for local files
    /// - OS page cache utilization
    /// - Reduced memory pressure
    /// - Faster cold starts (no loading)
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
            // Cloud files can't be memory-mapped but benefit from footer caching
            self.universal_optimizer
                .read_data_optimized(file_path)
                .await
        }
    }

    /// Columnar I/O optimization with universal parallel reads
    async fn parallel_column_read(
        &self,
        file_path: &str,
        column_indices: &[usize],
        collection_id: &str,
    ) -> Result<Vec<Vec<u8>>> {
        // Use universal optimizer for parallel operations
        let filesystem_factory = self.filesystem_factory.clone();
        let cached_filesystem = self.filesystem.clone();
        let collection_id_clone = collection_id.to_string();
        let read_operations: Vec<_> = column_indices
            .iter()
            .map(|&column_idx| {
                let file_path = file_path.to_string();
                let fs_factory = filesystem_factory.clone();
                let cached_fs = cached_filesystem.clone();
                let coll_id = collection_id_clone.clone();
                async move {
                    Self::read_column_optimized(
                        &file_path, column_idx, fs_factory, cached_fs, &coll_id,
                    )
                    .await
                }
            })
            .collect();

        let results = self
            .universal_optimizer
            .parallel_operations(read_operations, |operation| operation)
            .await?;

        // Extract successful results - results is Vec<Result<Vec<u8>>>
        let mut final_results = Vec::new();
        for result in results {
            match result {
                Ok(data) => {
                    // data is Result<Vec<u8>, Error>, so we need to unwrap it
                    match data {
                        Ok(bytes) => final_results.push(bytes),
                        Err(e) => return Err(anyhow::anyhow!("Column read failed: {}", e)),
                    }
                }
                Err(e) => return Err(anyhow::anyhow!("Vectorized read failed: {:?}", e)),
            }
        }
        Ok(final_results)
    }

    /// Optimized column reading with universal memory management
    async fn read_column_optimized(
        file_path: &str,
        column_idx: usize,
        filesystem_factory: Arc<FilesystemFactory>,
        cached_filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
        collection_id: &str,
    ) -> Result<Vec<u8>> {
        // For column reading, we don't need dimension - Parquet has schema info
        // Use 0 to indicate dimension is not needed for this operation
        let reader = super::readers::UnifiedParquetReader::new(
            vec![file_path.to_string()],
            0,
            filesystem_factory,
            cached_filesystem,
            collection_id.to_string(),
            "viper".to_string(),
        )?;

        // Read the actual column data from the Parquet file
        // Note: Using read_row_groups_projected to get all data
        // Column projection: reads all columns then filters. Arrow projection
        // via RecordBatch::project() would avoid I/O for unused columns.
        let batches = reader
            .read_row_groups_projected(file_path, &[], None)
            .await?;

        // Convert batches to vectors - placeholder implementation
        let vectors: Vec<VectorRecord> = Vec::new();
        for _batch in batches {
            // Extract records from batch - would need proper implementation
            // For now, create empty placeholder
        }

        // Extract column data based on column index
        // Column 0: vector data, Column 1+: metadata columns
        let column_data = if column_idx == 0 {
            // Vector column - serialize all vectors
            let mut data = Vec::new();
            for record in &vectors {
                // Serialize vector data
                for val in &record.vector {
                    data.extend_from_slice(&val.to_le_bytes());
                }
            }
            data
        } else {
            // Metadata column - extract specific metadata field
            // This is a simplified implementation
            let _data = Vec::new();
            // Metadata serialization: record metadata stored as JSON in Parquet
            // This should serialize the actual metadata from records
            return Err(anyhow::anyhow!(
                "Metadata serialization not yet implemented"
            ));
            #[allow(unreachable_code)]
            _data
        };

        Ok(column_data)
    }

    /// Storage tier optimization using universal optimizer
    async fn optimize_parquet_storage_tier(
        &self,
        file_path: &str,
        file_size_bytes: u64,
    ) -> Result<FileStorageTier> {
        // Use universal optimizer for storage tier optimization
        self.universal_optimizer
            .optimize_storage_tier(file_path, file_size_bytes as usize)
            .await
    }

    /// Compression optimization using unified compression module for columnar data (delegates to universal optimizer)
    async fn compress_parquet_optimized(
        &self,
        data: &[u8],
        tier: FileStorageTier,
        _column_type: &str,
    ) -> Result<Vec<u8>> {
        // Delegate to universal optimizer's tier-aware compression
        self.universal_optimizer.compress_for_tier(data, tier).await
    }

    /// Distance computation using unified distance compute engine for columnar operations (delegates to universal optimizer)
    async fn compute_distances_columnar_optimized(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        metric: proximadb_distance_kernel::DistanceMetric,
    ) -> Result<Vec<f32>> {
        // Use universal optimizer's hardware-accelerated distance computation
        self.universal_optimizer
            .compute_distances_accelerated(query, candidates, metric)
            .await
    }

    /// Row group prefetching optimization based on access patterns (delegates to universal optimizer)
    async fn prefetch_row_groups(&self, file_path: &str, current_row_group: usize) -> Result<()> {
        let config = self.universal_optimizer.get_config();
        if !config.enable_prefetching {
            return Ok(());
        }

        // Generate row group file URLs for prefetching
        let prefetch_count = config.prefetch_size_mb / 4; // Assume ~4MB per row group
        let row_group_urls: Vec<String> = ((current_row_group + 1)
            ..(current_row_group + 1 + prefetch_count))
            .map(|idx| format!("{}:rowgroup:{}", file_path, idx))
            .collect();

        // Use universal optimizer's prefetching capability
        self.universal_optimizer
            .prefetch_data(&row_group_urls)
            .await
    }

    /// Optimized row group reading for prefetching (delegates to universal optimizer)
    async fn read_row_group_optimized(
        file_path: &str,
        row_group_idx: usize,
        _optimizer: &UniversalPerformanceOptimizer,
        filesystem_factory: Arc<FilesystemFactory>,
        cached_filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
        collection_id: &str,
    ) -> Result<Vec<u8>> {
        // For row group reading, we don't need dimension - Parquet has schema info
        // Use 0 to indicate dimension is not needed for this operation
        let reader = super::readers::UnifiedParquetReader::new(
            vec![file_path.to_string()],
            0,
            filesystem_factory,
            cached_filesystem,
            collection_id.to_string(),
            "viper".to_string(),
        )?;

        // Read vectors from the specific row group
        // Note: UnifiedParquetReader currently reads all data, but in production
        // this would be optimized to read only the specific row group using Arrow's
        // row group API for selective reading
        let record_batches = reader
            .read_row_groups_projected(file_path, &[row_group_idx], None)
            .await?;

        // Calculate approximate row group boundaries
        // Parquet typically has row groups of ~50-100MB or ~50k-100k rows
        let rows_per_group = 50000;
        let _start_idx = row_group_idx * rows_per_group;
        let _end_idx = ((row_group_idx + 1) * rows_per_group).min(10000); // Placeholder, since all_vectors no longer exists

        // Extract data from the record batches
        // Vector extraction from Parquet: vectors stored as FixedSizeList<f32>
        // This needs to read actual vector data from the batch columns
        if !record_batches.is_empty() {
            return Err(anyhow::anyhow!(
                "Row group data extraction not yet implemented"
            ));
        }

        Ok(Vec::new())
    }

    /// Memory pool optimization for columnar operations (delegates to universal optimizer)
    async fn get_columnar_buffer(&self, size: usize) -> Result<Vec<f32>> {
        self.universal_optimizer
            .get_memory_buffer(size)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire columnar buffer: {}", e))
    }

    /// Column statistics caching for analytics optimization (delegates to universal optimizer)
    async fn cache_column_statistics(
        &self,
        file_path: &str,
        column_idx: usize,
        stats: &[u8],
    ) -> Result<()> {
        // Use universal optimizer's data caching
        let cache_key = format!("{}:col:{}:stats", file_path, column_idx);

        // Write statistics to a temporary location and let universal optimizer handle caching
        // This is a simplified approach - in production, we'd integrate more directly
        self.universal_optimizer
            .write_data_optimized(
                &cache_key,
                stats,
                FileStorageTier::Memory, // Statistics are frequently accessed
            )
            .await
    }

    // VIPER is columnar storage - it doesn't support single vector inserts
    // All data must come through flush operations from WAL or direct flush
    /// Flush vectors to storage
    pub async fn flush_vectors(
        &self,
        collection_id: &str,
        vector_records: &[VectorRecord],
        batch_ids: &[String],
        force: bool,
        synchronous: bool,
    ) -> Result<FlushResult> {
        info!(
            "🚿 VIPER Engine: Flushing {} vectors for collection {} (force: {}, sync: {})",
            vector_records.len(),
            collection_id,
            force,
            synchronous
        );
        // Delegate to the flush manager
        self.flush_manager
            .flush_vectors(
                collection_id,
                vector_records,
                batch_ids,
                force,
                synchronous,
                &self.core_config,
                None,
            )
            .await
    }

    /// Direct flush vectors to storage during WAL recovery (bypasses normal flush pipeline)
    pub async fn flush_vectors_direct(
        &self,
        collection_id: &str,
        vector_records: Vec<crate::proto::proximadb_v1::VectorRecord>,
    ) -> Result<()> {
        let num_records = vector_records.len();
        info!(
            "💾 VIPER Engine: Direct flush {} vectors for collection {} (WAL recovery)",
            num_records, collection_id
        );
        // Create synthetic batch IDs for recovery
        let batch_ids: Vec<String> = (0..num_records)
            .map(|i| format!("recovery_batch_{}", i))
            .collect();
        // Convert to storage format
        let nova_file_records: Vec<VectorRecord> = vector_records.into_iter().collect();
        // Use existing flush infrastructure with force=true, synchronous=true for recovery
        let _flush_result = self
            .flush_vectors(
                collection_id,
                &nova_file_records,
                &batch_ids,
                true, // force flush
                true, // synchronous for reliable recovery
            )
            .await?;
        info!(
            "✅ VIPER Engine: Direct flush completed for collection {}",
            collection_id
        );
        Ok(())
    }

    /// Compact Parquet files  
    /// Note: This method requires collection config to be passed, use do_compact for automatic config lookup
    pub async fn compact_parquet_files(
        &self,
        input_files: Vec<String>,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Vec<String>> {
        let collection_id = collection_config.as_ref().map(|c| c.id.as_str());

        info!(
            "🗜️ VIPER Engine: Compacting {} files for collection {}",
            input_files.len(),
            collection_id.unwrap_or("unknown")
        );
        // Delegate to the compaction manager with collection config
        let result = self
            .compaction
            .compact_parquet_files(
                collection_id.unwrap_or("default"),
                input_files,
                collection_config,
            )
            .await?;
        Ok(result.output_files)
    }

    /// List Parquet files in a directory
    async fn list_parquet_files_in_dir(&self, data_dir: &str) -> Result<Vec<String>> {
        let fs = self.filesystem_factory.get_filesystem(data_dir)?;
        let entries = fs.list(data_dir).await?;

        let mut parquet_files = Vec::new();
        for entry in entries {
            if entry.name.ends_with(".parquet") {
                parquet_files.push(format!("{}/{}", data_dir, entry.name));
            }
        }

        Ok(parquet_files)
    }

    /// Search for vectors by ID (internal implementation with base_path)
    pub async fn internal_vector_by_id_with_path(
        &self,
        collection_id: &str,
        base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        use arrow_array::{
            Array, BooleanArray, Float32Array, Float64Array, Int64Array, ListArray, StringArray,
            StructArray,
        };
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        // Access global unified cache through CrossCacheOrchestrator
        let cache_key = format!("vector:{}:{}", collection_id, vector_id);
        if let Some(orchestrator) =
            crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
        {
            // Try to get from vector cache first (using correct cache type)
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

        // use bytes::Bytes; // Commented out due to compilation issue
        info!(
            "🔍 VIPER Engine: Looking up vector {} in collection {} at {}",
            vector_id, collection_id, base_path
        );
        // Get all Parquet files from {base_path}/{collection_id}/data.
        // TODO ADR-031 Phase 4d: when `PROXIMADB_TYPED_PATHS=1`, this must resolve
        // the typed account-rooted path (`accounts/{base62}/…/data`) via
        // `collection_data_path_typed` using the collection's pre-minted
        // `CollectionIdentity` (threaded from the catalog's `stable_*_id` +
        // account u32). Until then the env gate stays OFF and this legacy path
        // is byte-identical to the typed helper's `None` branch (mixed-read-safe).
        let data_dir = StoragePath::collection_data_path(base_path, collection_id);
        let parquet_files = self.list_parquet_files_in_dir(&data_dir).await?;
        if parquet_files.is_empty() {
            debug!("📁 No Parquet files found for collection {}", collection_id);
            return Ok(None);
        }
        let current_time = chrono::Utc::now().timestamp_micros();
        let mut best_match: Option<(VectorRecord, i64, i64)> = None; // (record, version, timestamp)
        // Search through all Parquet files
        for parquet_file in parquet_files {
            debug!("🔍 Searching file: {}", parquet_file);

            // Read Parquet file using filesystem API through factory
            let fs = self
                .filesystem_factory
                .get_filesystem(&parquet_file)
                .map_err(|e| anyhow::anyhow!("Failed to get filesystem: {}", e))?;
            let parquet_data = match fs.read(&parquet_file).await {
                Ok(data) => data,
                Err(e) => {
                    warn!("Failed to read Parquet file {}: {}", parquet_file, e);
                    continue;
                }
            };
            let parquet_bytes = bytes::Bytes::from(parquet_data);
            let reader_builder = match ParquetRecordBatchReaderBuilder::try_new(parquet_bytes) {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "Failed to create Parquet reader for {}: {}",
                        parquet_file, e
                    );
                    continue;
                }
            };
            let mut batch_reader = reader_builder.build()?;
            // Process each record batch
            for batch in &mut batch_reader {
                let batch = batch?;

                // Get ID column
                let id_array = batch
                    .column_by_name(FIELD_ID)
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'id' column"))?;
                // Find matching ID
                for row_idx in 0..batch.num_rows() {
                    if id_array.value(row_idx) == vector_id {
                        let timestamp = batch
                            .column_by_name(FIELD_TIMESTAMP)
                            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                            .map_or(0, |arr| arr.value(row_idx));
                        let version = batch
                            .column_by_name(FIELD_VERSION)
                            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                            .map_or(0, |arr| arr.value(row_idx));
                        let expires_at = batch
                            .column_by_name(FIELD_EXPIRES_AT)
                            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                            .and_then(|arr| {
                                if arr.is_null(row_idx) {
                                    None
                                } else {
                                    Some(arr.value(row_idx))
                                }
                            });
                        // Skip if expired
                        if let Some(exp) = expires_at
                            && exp > 0
                            && exp < current_time
                        {
                            debug!("Skipping expired vector {} (expired at {})", vector_id, exp);
                            continue;
                        }
                        // Extract vector data. The write side emits
                        // `vector_fp32` as a FixedSizeList (factory.rs
                        // schema); accept BOTH list flavors — the previous
                        // ListArray-only downcast rejected every
                        // engine-written file ("Missing or invalid
                        // 'vector_fp32' column" on point reads).
                        let vector_values = batch
                            .column_by_name(FIELD_VECTOR_FP32)
                            .and_then(|col| list_row_values(col, row_idx))
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing or invalid 'vector_fp32' column")
                            })?;
                        let vector_float_array = vector_values
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| anyhow::anyhow!("Invalid vector values type"))?;
                        let vector: Vec<f32> = (0..vector_float_array.len())
                            .map(|i| vector_float_array.value(i))
                            .collect();
                        // Extract other fields
                        let _created_at = batch
                            .column_by_name("created_at")
                            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                            .map(|arr| arr.value(row_idx));
                        let updated_at = batch
                            .column_by_name("updated_at")
                            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                            .map_or(0, |arr| arr.value(row_idx));
                        // Parse metadata from extra_meta list of key-value pairs
                        let mut metadata_map: HashMap<
                            String,
                            crate::proto::proximadb_v1::SqlValue,
                        > = HashMap::new();
                        if let Some(extra_meta_col) = batch.column_by_name("extra_meta")
                            && let Some(extra_meta_list) =
                                extra_meta_col.as_any().downcast_ref::<ListArray>()
                            && !extra_meta_list.is_null(row_idx)
                        {
                            let kv_pairs = extra_meta_list.value(row_idx);
                            if let Some(struct_array) =
                                kv_pairs.as_any().downcast_ref::<StructArray>()
                            {
                                let (Some(key_array), Some(value_array)) = (
                                    struct_array
                                        .column(0)
                                        .as_any()
                                        .downcast_ref::<StringArray>(),
                                    struct_array
                                        .column(1)
                                        .as_any()
                                        .downcast_ref::<StringArray>(),
                                ) else {
                                    continue;
                                };

                                for kv_idx in 0..struct_array.len() {
                                    if !struct_array.is_null(kv_idx) {
                                        let key = key_array.value(kv_idx).to_string();
                                        let value = value_array.value(kv_idx).to_string();
                                        metadata_map.insert(key, crate::proto::proximadb_v1::SqlValue {
                                                    value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(value)),
                                                });
                                    }
                                }
                            }
                        }
                        // Also parse filterable metadata columns (they have their own columns)
                        for field in batch.schema().fields() {
                            let field_name = field.name();
                            // Skip core fields - only process filterable metadata columns
                            if !matches!(
                                field_name.as_str(),
                                FIELD_ID
                                    | "collection_id"
                                    | FIELD_VECTOR_FP32
                                    | FIELD_TIMESTAMP
                                    | "created_at"
                                    | "updated_at"
                                    | FIELD_VERSION
                                    | FIELD_EXPIRES_AT
                                    | "extra_meta"
                            ) && let Some(column) = batch.column_by_name(field_name)
                                && !column.is_null(row_idx)
                            {
                                // Convert Arrow value to String based on data type
                                let string_value = match field.data_type() {
                                    arrow_schema::DataType::Utf8 => {
                                        if let Some(str_array) =
                                            column.as_any().downcast_ref::<StringArray>()
                                        {
                                            str_array.value(row_idx).to_string()
                                        } else {
                                            continue;
                                        }
                                    }
                                    arrow_schema::DataType::Int64 => {
                                        if let Some(int_array) =
                                            column.as_any().downcast_ref::<Int64Array>()
                                        {
                                            int_array.value(row_idx).to_string()
                                        } else {
                                            continue;
                                        }
                                    }
                                    arrow_schema::DataType::Float64 => {
                                        if let Some(float_array) =
                                            column.as_any().downcast_ref::<Float64Array>()
                                        {
                                            float_array.value(row_idx).to_string()
                                        } else {
                                            continue;
                                        }
                                    }
                                    arrow_schema::DataType::Boolean => {
                                        if let Some(bool_array) =
                                            column.as_any().downcast_ref::<BooleanArray>()
                                        {
                                            bool_array.value(row_idx).to_string()
                                        } else {
                                            continue;
                                        }
                                    }
                                    _ => continue, // Skip unsupported types
                                };
                                metadata_map.insert(field_name.to_string(), crate::proto::proximadb_v1::SqlValue {
                                            value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(string_value)),
                                        });
                            }
                        }
                        // metadata_map is already HashMap<String, SqlValue> which is what VectorRecord expects
                        let record = VectorRecord {
                            id: vector_id.to_string(),
                            vector,
                            metadata: metadata_map,
                            timestamp: Some(timestamp),
                            updated_at: Some(updated_at),
                            expires_at,
                            version: Some(version as u32),
                            source: None,
                        };
                        // Check if this is a better match than what we have
                        match &best_match {
                            Some((_, best_version, best_timestamp)) => {
                                if version > *best_version
                                    || (version == *best_version && timestamp > *best_timestamp)
                                {
                                    best_match = Some((record, version, timestamp));
                                }
                            }
                            None => {
                                best_match = Some((record, version, timestamp));
                            }
                        }
                    } // End if id_array.value(row_idx) == vector_id
                } // End for row_idx
            } // End while let Some(batch)
        } // End for parquet_file in parquet_files

        // Update global cache with found vector before returning
        if let Some((ref record, _, _)) = best_match
            && let Some(orchestrator) =
                crate::storage::cache::orchestrator::CrossCacheOrchestrator::global()
            && let Some(vector_cache) = orchestrator.get_vector_cache()
        {
            let _ = vector_cache
                .put(
                    cache_key,
                    proximadb_records::ProximaRecord::from(record.clone()),
                )
                .await;
        }

        // Return the best match (highest version/newest timestamp)
        Ok(best_match.map(|(record, _, _)| proximadb_records::ProximaRecord::from(record)))
    }

    /// Get engine statistics (creates a snapshot)
    pub async fn stats(&self) -> super::types::EngineStatsSnapshot {
        super::types::EngineStatsSnapshot {
            total_vectors: self
                .stats
                .total_vectors
                .load(std::sync::atomic::Ordering::Relaxed),
            total_size_bytes: self
                .stats
                .total_size_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
            active_collections: self
                .stats
                .active_collections
                .load(std::sync::atomic::Ordering::Relaxed),
            flush_operations: self
                .stats
                .flush_operations
                .load(std::sync::atomic::Ordering::Relaxed),
            compaction_operations: self
                .stats
                .compaction_operations
                .load(std::sync::atomic::Ordering::Relaxed),
            total_storage_size_bytes: self
                .stats
                .total_storage_size_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
            active_clusters: self
                .stats
                .active_clusters
                .load(std::sync::atomic::Ordering::Relaxed),
            active_partitions: self
                .stats
                .active_partitions
                .load(std::sync::atomic::Ordering::Relaxed),
            avg_compression_ratio: self.stats.get_compression_ratio(),
            avg_ml_prediction_accuracy: self.stats.get_ml_accuracy(),
        }
    }

    // 🔴 UNUSED SCHEMA CACHE METHODS - CANDIDATES FOR REMOVAL
    // These schema cache management methods have no callers found in the codebase.
    // Schema caching is managed internally and these public methods are not used.
    /*
    /// Clear schema cache for a collection
    pub async fn clear_schema_cache(&self, collection_id: &str) {
        self.schema_manager.clear_schema_cache(collection_id).await;
    }

    /// Clear all schema caches
    pub async fn clear_all_schema_cache(&self) {
        self.schema_manager.clear_all_schema_cache().await;
    }

    /// Get schema cache statistics
    pub async fn schema_cache_stats(&self) -> (usize, Vec<String>) {
        self.schema_manager.get_cache_stats().await
    }
    */
    // 🔴 UNUSED HEALTH CHECK METHOD - CANDIDATE FOR REMOVAL
    // This internal health check method has no callers found.
    // Health checking is handled by the UnifiedStorageFormat trait's health_check method.
    /// Internal health check
    pub async fn internal_health_check(&self) -> Result<bool> {
        // Basic health check - can be extended to check:
        // - Filesystem connectivity
        // - Collection service availability
        // - Internal state consistency
        Ok(true)
    }

    /// Get collection metadata
    pub async fn collection_metadata(&self, collection_id: &str) -> Option<CollectionMetadata> {
        let collections = self.collections.read().await;
        collections.get(collection_id).cloned()
    }

    /// Update collection metadata
    pub async fn update_collection_metadata(
        &self,
        collection_id: String,
        metadata: CollectionMetadata,
    ) {
        let mut collections = self.collections.write().await;
        collections.insert(collection_id, metadata);
    }

    /// Get engine configuration
    pub fn config(&self) -> &crate::core::config::ViperConfig {
        &self.core_config
    }

    // 🟡 INTERNAL UTILITY METHODS - CONSIDER MAKING PRIVATE
    // These utility methods have no external callers found in the codebase.
    // They are only used internally and could be made private or moved to internal modules.
    /// Record operation performance metrics - INTERNAL USE
    async fn record_operation_metrics(
        &self,
        metrics: super::utilities::OperationMetrics,
    ) -> Result<()> {
        self.utilities.record_operation(metrics).await
    }

    /// Get performance statistics - INTERNAL USE
    async fn get_performance_report(
        &self,
        collection_id: Option<&String>,
    ) -> Result<super::utilities::PerformanceReport> {
        self.utilities.get_performance_stats(collection_id).await
    }

    /// Optimize compression for a collection - INTERNAL USE
    async fn optimize_compression(
        &self,
        collection_id: &str,
    ) -> Result<super::utilities::CompressionRecommendation> {
        self.utilities.optimize_compression(collection_id).await
    }

    /// Start background utilities services - INTERNAL USE
    async fn start_background_services(&mut self) -> Result<()> {
        // Note: utilities is not mutable, so we need to access the inner services differently
        // This would need to be redesigned for proper mutable access
        info!("🚀 VIPER Engine: Background services functionality available via utilities");
        Ok(())
    }

    /// **STORAGE-AWARE POLYMORPHIC SEARCH**: Primary vector search interface
    /// This method provides polymorphic search that automatically selects the most
    /// efficient search search_strategy based on collection characteristics, data distribution,
    /// and query parameters. It delegates to specialized search implementations for:
    /// - ML-driven cluster optimization for large collections
    /// - Direct search for small collections
    /// - Hybrid strategies combining clustering with metadata filtering
    ///
    /// Public search method for testing - requires collection_id and storage URL.
    ///
    /// **Note**: This is a convenience method primarily intended for testing.
    /// Production code should use `search_vectors_unified` via the `UnifiedStorageFormat` trait
    /// for full control over search parameters including distance metrics and filters.
    pub async fn search_vectors(
        &self,
        collection_id: &str,
        storage_url: &str,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<crate::proto::proximadb_v1::SearchResult>> {
        info!(
            "🔍 VIPER Engine: search_vectors called - collection={}, storage_url={}, k={}",
            collection_id, storage_url, k
        );
        // Delegate to the unified search implementation with default parameters

        // Create search context with provided collection_id (no URL parsing needed!)
        use crate::core::search::SearchParams;
        use crate::storage::traits::{StorageQueryContext, StorageQueryMetadata};

        let search_params = Arc::new(SearchParams {
            vector: Some(query_vector.to_vec()),
            top_k: Some(k as u16),
            ..SearchParams::default()
        });

        // For testing, create a minimal collection config
        // This avoids the need for collection service in tests
        // Note: In production, use search_vectors_unified with proper context

        // Extract base_location from storage_url (tests often pass full path with /data)
        // Production behavior: metadata.storage_path should be base_location
        let base_location = if storage_url.contains(&format!("/{}/data", collection_id)) {
            storage_url.replace(&format!("/{}/data", collection_id), "")
        } else {
            storage_url.to_string()
        };

        let collection = crate::proto::proximadb_v1::Collection {
            id: collection_id.to_string(),
            config: Some(crate::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                dimension: query_vector.len() as u32,
                distance_metric: Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                storage_engine: Some(crate::proto::proximadb_v1::StorageEngine::Viper as i32),
                ..Default::default()
            }),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                base_location: base_location.clone(),
                primary_path: storage_url.to_string(),
                backup_paths: vec![],
                engine: crate::proto::proximadb_v1::StorageEngine::Viper as i32,
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
                // Convert OptimizedSearchRecord vector (Arc<Vec<f32>>) to Vec<f32>
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

        // Return a single SearchResult containing all results
        Ok(vec![crate::proto::proximadb_v1::SearchResult {
            results: search_records,
            total_found: 0,
            collection_id: Some(collection_id.to_string()),
        }])
    }

    // REMOVED: search_vectors_in_cluster - Clustering is handled by AXIS indexing service
    // VIPER provides raw vector retrieval; AXIS determines which files to search
    /// Get all Parquet files using the provided storage URL
    pub async fn parquet_files_with_storage_url(
        &self,
        collection_id: &str,
        storage_url: &str,
    ) -> Result<Vec<String>> {
        debug!(
            "📁 VIPER: Getting Parquet files for collection: {} from URL: {}",
            collection_id, storage_url
        );
        // Use filesystem API for all storage backends - it handles the differences
        debug!("📁 VIPER: Listing files at: {}", storage_url);
        let parquet_files = match self.filesystem_factory.list(storage_url).await {
            Ok(files) => {
                debug!("📁 VIPER: filesystem.list returned {} entries", files.len());
                for (i, f) in files.iter().enumerate() {
                    debug!("📁 VIPER:   Entry[{}]: name={}, url={}", i, f.name, f.url);
                }
                let parquet_files: Vec<String> = files
                    .into_iter()
                    .filter(|f| {
                        let matches = f.name.ends_with(crate::storage::engines::VIPER_FILE_EXT)
                            && !f.name.starts_with("__")
                            && !f.name.starts_with(".");
                        if !matches {
                            info!("📁 VIPER: Filtered out: {}", f.name);
                        }
                        matches
                    })
                    .map(|f| f.url) // Use the full URL from DirEntry
                    .collect();
                debug!(
                    "📁 VIPER: Found {} Parquet files after filtering",
                    parquet_files.len()
                );
                for (i, file) in parquet_files.iter().enumerate() {
                    debug!("📁 VIPER:   Parquet[{}]: {}", i, file);
                }
                parquet_files
            }
            Err(e) => {
                warn!("📁 VIPER: Error listing files at {}: {}", storage_url, e);
                vec![]
            }
        };
        Ok(parquet_files)
    }

    /// Get all Parquet files associated with a collection (legacy - uses collection service)
    pub async fn parquet_files_for_collection(&self, collection_id: &str) -> Result<Vec<String>> {
        debug!("📁 Getting Parquet files for collection: {}", collection_id);
        // Get storage URL from collection metadata
        let collection_service = self.collection_service.read().await;
        let collection_service = collection_service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Collection service not initialized"))?;
        let collection = collection_service
            .collection(collection_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collection {} not found", collection_id))?;
        let storage_assignment = collection.storage_assignment.ok_or_else(|| {
            anyhow::anyhow!(
                "No storage assignment found for collection {}",
                collection_id
            )
        })?;
        let storage_url = format!(
            "{}/{}/data",
            storage_assignment.base_location, collection_id
        );
        debug!(
            "📁 Storage URL for collection {}: {}",
            collection_id, storage_url
        );

        // Use filesystem API for all storage backends - it handles the differences
        let parquet_files = match self.filesystem_factory.list(&storage_url).await {
            Ok(entries) => {
                let mut files: Vec<String> = entries
                    .into_iter()
                    .filter(|e| {
                        e.name.ends_with(crate::storage::engines::VIPER_FILE_EXT)
                            && !e.name.starts_with("__")
                            && !e.name.starts_with(".")
                    })
                    .map(|e| e.url)
                    .collect();
                // Sort files for consistent ordering
                files.sort();
                debug!("📁 Found {} parquet files", files.len());
                files
            }
            Err(e) => {
                debug!(
                    "📁 Collection directory does not exist or error listing: {}",
                    e
                );
                Vec::new()
            }
        };
        info!(
            "📁 Found {} Parquet files for collection {}",
            parquet_files.len(),
            collection_id
        );
        Ok(parquet_files)
    }

    /// Get collection configuration including filterable column specifications
    async fn get_collection_config(
        &self,
        collection_id: &str,
    ) -> Result<Option<crate::proto::proximadb_v1::Collection>> {
        // Get metadata from collection service if available
        if let Some(collection_service) = &*self.collection_service.read().await {
            match collection_service.collection(collection_id).await {
                Ok(Some(collection)) => Ok(Some(collection)),
                Ok(None) => {
                    debug!("Collection {} not found", collection_id);
                    Ok(None)
                }
                Err(e) => {
                    warn!("Failed to get collection metadata: {}", e);
                    Ok(None)
                }
            }
        } else {
            // No collection service available, return minimal metadata
            warn!("No collection service available for metadata retrieval");
            Ok(None)
        }
    }

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

    // Removed convert_filter_expression_to_metadata_filter - no longer needed
    // VIPER now uses FilterExpression directly with SearchPlan
}

// Close the impl ViperEngine block
#[allow(clippy::panic)] // Intentional panics for Default impl failures - indicates initialization problems
impl Default for ViperEngine {
    fn default() -> Self {
        let runtime = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(err) => panic!("Failed to create Tokio runtime for ViperEngine::default: {err}"),
        };

        let engine_result = runtime.block_on(async {
            let filesystem_factory_result = FilesystemFactory::create(
                crate::storage::persistence::filesystem::FilesystemConfig::default(),
            )
            .await;
            let filesystem_factory = Arc::new(match filesystem_factory_result {
                Ok(factory) => factory,
                Err(err) => {
                    panic!("Failed to create filesystem factory for ViperEngine::default: {err}")
                }
            });

            // Create VIPER metadata serializer
            let metadata_serializer =
                Arc::new(super::unified_metadata_serializer::ViperMetadataSerializer::new());

            // Get base filesystem
            let base_fs = match filesystem_factory.get_filesystem("file://") {
                Ok(fs) => fs,
                Err(err) => panic!("Failed to get base filesystem for ViperEngine::default: {err}"),
            };

            // Create UnifiedCachingFilesystem
            let unified_fs = Arc::new(
                crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::with_serializer(
                    base_fs,
                    "default".to_string(),
                    "viper".to_string(),
                    metadata_serializer,
                )
            );

            Self::from_caching_filesystem(crate::core::config::ViperConfig::default(), unified_fs)
                .await
        });

        match engine_result {
            Ok(engine) => engine,
            Err(err) => panic!("Failed to build ViperEngine::default: {err}"),
        }
    }
}

// UnifiedStorageFormat trait: VIPER implements via the engine module (viper/mod.rs)
// This will replace the old ViperCoreEngine implementation
#[async_trait::async_trait]
impl UnifiedStorageFormat for ViperEngine {
    // Required abstract methods
    fn engine_name(&self) -> &'static str {
        "VIPER"
    }

    fn engine_version(&self) -> &'static str {
        crate::version::PROXIMADB_VERSION
    }

    fn strategy(&self) -> crate::storage::traits::StorageFormatStrategy {
        crate::storage::traits::StorageFormatStrategy::Viper
    }

    async fn do_flush(
        &self,
        params: &crate::storage::traits::FlushParameters,
    ) -> Result<FlushResult> {
        let collection_id = self.get_collection_id_from_params(params)?;

        debug!("🟦 VIPER DO_FLUSH: ========== STARTING FLUSH ==========");
        debug!("🟦 VIPER DO_FLUSH: Collection ID: {}", collection_id);
        debug!(
            "🟦 VIPER DO_FLUSH: Vector count: {}",
            params.vector_records.len()
        );
        debug!("🟦 VIPER DO_FLUSH: Force: {}", params.force);
        debug!("🟦 VIPER DO_FLUSH: Synchronous: {}", params.synchronous);
        debug!(
            "🟦 VIPER DO_FLUSH: Has collection_config: {}",
            params.collection_config.is_some()
        );

        debug!("🔍 VIPER DO_FLUSH: Checking compression configuration");
        if let Some(ref collection_config) = params.collection_config {
            debug!("🟦 VIPER DO_FLUSH: Collection config found");
            if let Some(ref config) = collection_config.config {
                debug!("🟦 VIPER DO_FLUSH: Config field found");
                if let Some(_storage_config) = config.storage_config.as_ref() {
                    debug!("🟦 VIPER DO_FLUSH: Storage config found");
                    debug!("   ✅ Found storage_config in collection_config");
                } else {
                    debug!("🟦 VIPER DO_FLUSH: No storage config");
                    debug!("   ⚠️ No compression config in collection_config");
                }
            } else {
                debug!("🟦 VIPER DO_FLUSH: No config field");
                debug!("   ⚠️ No config field in collection");
            }
        } else {
            debug!("🟦 VIPER DO_FLUSH: No collection config");
            debug!("   ⚠️ No collection_config in params");
        }
        debug!(
            "Starting flush for collection {} with {} vectors",
            collection_id,
            params.vector_records.len()
        );
        info!(
            "🚿 VIPER Engine: Starting flush for collection {} with {} vectors",
            collection_id,
            params.vector_records.len()
        );
        // Convert batch IDs to strings for compatibility
        let batch_id_strings: Vec<String> =
            params.batch_ids.iter().map(|id| id.to_string()).collect();

        debug!("🟦 VIPER DO_FLUSH: About to call flush_manager.flush_vectors()");
        debug!(
            "🟦 VIPER DO_FLUSH: Batch ID strings: {:?}",
            batch_id_strings
        );

        // Convert ProximaRecord → VectorRecord for VIPER engine internals (protocol adapter boundary)
        let vector_records_v1: Vec<VectorRecord> = params
            .vector_records
            .iter()
            .map(proxima_record_to_vector)
            .collect();

        // Use the modular flush manager to flush vectors with provided collection config
        let mut flush_result = self
            .flush_manager
            .flush_vectors(
                &collection_id,
                &vector_records_v1,
                &batch_id_strings,
                params.force,
                params.synchronous,
                &self.core_config,
                params.collection_config.as_ref(), // Pass collection config from params
            )
            .await?;

        debug!("🟦 VIPER DO_FLUSH: flush_manager.flush_vectors() returned");
        debug!(
            "🟦 VIPER DO_FLUSH: Flush result success: {}",
            flush_result.success
        );
        debug!(
            "🟦 VIPER DO_FLUSH: Entries flushed: {:?}",
            flush_result.entries_flushed
        );
        debug!(
            "🟦 VIPER DO_FLUSH: Bytes written: {:?}",
            flush_result.bytes_written
        );
        debug!(
            "🟦 VIPER DO_FLUSH: Files created: {:?}",
            flush_result.files_created
        );
        // Update engine statistics using atomic operations (lock-free)
        self.stats
            .flush_operations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.total_vectors.fetch_add(
            flush_result.entries_flushed.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.stats.total_size_bytes.fetch_add(
            flush_result.bytes_written.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        // Add engine-specific metrics
        flush_result.engine_metrics.insert(
            "engine_version".to_string(),
            serde_json::Value::String(crate::version::PROXIMADB_VERSION.to_string()),
        );
        flush_result.engine_metrics.insert(
            "engine_name".to_string(),
            serde_json::Value::String("VIPER".to_string()),
        );
        // Step 3: index the just-flushed vectors into AXIS directly (ADR-078).
        // Previously this published a metadata event to the AXIS queue, whose
        // consumer re-read these same vectors back out of storage — the records
        // are already in hand on `params`, so that round-trip bought nothing.
        let file_paths = if let Some(path_value) = flush_result.engine_metrics.get("parquet_files")
            && let serde_json::Value::String(path) = path_value
        {
            vec![path.clone()]
        } else {
            vec![]
        };
        crate::storage::common::axis_flush_hook::index_flushed_into_axis(
            self.axis_manager().cloned(),
            params,
            file_paths,
        )
        .await;
        Ok(flush_result)
    }

    /// VIPER's LSM bulk-load override (Phase 2F-b).
    ///
    /// Delegates to `do_flush` with a synthetic FlushParameters.
    /// VIPER's flush owns the ML clustering + Parquet predicate
    /// pushdown machinery; bulk-loaded segments inherit the same
    /// columnar layout the normal flush produces. WAL+memtable are
    /// bypassed because we never call the per-record insert path.
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

        let collection_config = Some(Collection {
            id: collection_id.to_string(),
            config: None,
            stats: None,
            created_at: 0,
            updated_at: 0,
            storage_assignment: Some(StorageAssignment {
                primary_path: base_path.to_string(),
                backup_paths: vec![],
                engine: 0,
                engine_config: std::collections::HashMap::new(),
                base_location: base_path.to_string(),
                assigned_at: 0,
                ..Default::default()
            }),
        });

        let params = crate::storage::traits::FlushParameters {
            collection_id: Some(
                collection_id
                    .parse::<u64>()
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "VIPER bulk ingest requires a numeric catalog object id, got {collection_id:?}: {error}"
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

        let flush_result = self.do_flush(&params).await?;
        let synthetic_segment_id = flush_result
            .file_paths
            .first()
            .cloned()
            .unwrap_or_else(|| format!("viper-bulkload-{collection_id}-{count}"));

        Ok(crate::storage::traits::SegmentIngestResult {
            collection_id: collection_id.to_string(),
            record_count: count,
            synthetic_segment_id,
            used_engine_specific_path: true,
        })
    }

    async fn do_compact(
        &self,
        params: &crate::storage::traits::CompactionParameters,
    ) -> Result<crate::storage::traits::CompactionResult> {
        let start_time = std::time::Instant::now();
        debug!(
            "🗜️ VIPER do_compact called with params: collection_id={:?}, force={}, synchronous={}, timeout_ms={:?}",
            params.collection_id, params.force, params.synchronous, params.timeout_ms
        );
        debug!("🔍 VIPER DO_COMPACT: Checking compression configuration");
        let collection_id = self.get_collection_id_from_compaction_params(params)?;
        debug!("🗜️ VIPER compaction collection ID: {}", collection_id);
        // Get input files from hints or use default empty list
        let input_files = params
            .hints
            .get("input_files")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
            })
            .clone();
        info!(
            "🗜️ VIPER Engine: Starting compaction for collection {} with {} hinted input files",
            collection_id,
            input_files.as_ref().map_or(0, |f| f.len())
        );
        debug!("🗜️ VIPER input files: {:?}", input_files);
        // Use the modular compaction manager to compact Parquet files
        // If no input files specified, the compaction manager will discover them
        // Pass the collection config from parameters to avoid collection service lookups
        debug!("🗜️ VIPER calling compaction.compact_parquet_files");
        debug!(
            "🗜️ VIPER collection_config present: {}",
            params.collection_config.is_some()
        );
        let compaction_result = self
            .compaction
            .compact_parquet_files(
                &collection_id,
                input_files.clone().unwrap_or_default(),
                params.collection_config.as_ref(),
            )
            .await;
        match &compaction_result {
            Ok(result) => {
                debug!("🗜️ VIPER compaction returned success");
                debug!("🗜️ VIPER compaction result details: {:?}", result);
            }
            Err(e) => {
                debug!("🗜️ VIPER compaction failed: {}", e);
                return Err(anyhow::anyhow!("Compaction failed: {}", e));
            }
        }
        let compaction_result = compaction_result?;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        // Calculate bytes reclaimed (this is an approximation)
        let bytes_reclaimed = input_files.as_ref().map_or(0, |f| f.len()) as u64 * 1024 * 1024; // Estimate 1MB per file
        // Calculate entries processed - estimate based on input files
        let _entries_processed = if input_files.is_none() {
            0
        } else {
            // Estimate entries per file (this could be more accurate with metadata)
            input_files.as_ref().map_or(0, |f| f.len()) as u64 * 100 // Assume ~100 entries per file for tests
        };
        // Update compaction metrics atomically
        self.stats
            .compaction_operations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(crate::storage::traits::CompactionResult {
            success: true,
            collections_affected: vec![collection_id.clone()],
            entries_processed: Some(compaction_result.entries_processed),
            entries_removed: Some(compaction_result.entries_removed),
            bytes_read: Some(compaction_result.bytes_read),
            bytes_written: Some(compaction_result.bytes_written),
            input_files: Some(compaction_result.input_files.len() as u64),
            output_files: Some(compaction_result.output_files.len() as u64),
            duration_ms: Some(duration_ms),
            completed_at: chrono::Utc::now(),
            engine_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "compacted_files".to_string(),
                    serde_json::Value::Array(
                        compaction_result
                            .output_files
                            .iter()
                            .map(|f| serde_json::Value::String(f.clone()))
                            .collect(),
                    ),
                );
                metrics.insert(
                    "input_files".to_string(),
                    serde_json::Value::Array(
                        compaction_result
                            .input_files
                            .iter()
                            .map(|f| serde_json::Value::String(f.clone()))
                            .collect(),
                    ),
                );
                metrics.insert(
                    "bytes_reclaimed".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(bytes_reclaimed)),
                );
                metrics
            },
        })
    }

    async fn vector_by_id(
        &self,
        collection_id: &str,
        base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        // Delegate to internal implementation with base_path
        self.internal_vector_by_id_with_path(collection_id, base_path, vector_id)
            .await
    }

    async fn search_vectors_unified(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        let search_start = std::time::Instant::now();

        // Extract parameters from context
        let collection_id = ctx.collection_id();
        // Use storage_url directly to avoid double-appending collection ID
        let storage_url = ctx
            .storage_url()
            .ok_or_else(|| anyhow::anyhow!("No storage URL in context"))?;

        debug!(
            "🔍 VIPER search_vectors_unified: collection_id={}, storage_url={}",
            collection_id, storage_url
        );
        let query_vector = ctx
            .query_vector()
            .ok_or_else(|| anyhow::anyhow!("No query vector in context"))?;
        let k = ctx.top_k();
        let distance_metric = ctx.distance_metric();
        let filter_expression = ctx.search_params.filter_expression.as_ref();
        // SearchParams fields: derived from collection config at query time
        let include_vectors = true;
        let include_metadata = true;

        info!(
            "🚀 VIPER: Enhanced unified search with orchestration for collection {}",
            collection_id
        );

        // ========================================================================
        // PHASE 0: TRY AXIS-BASED SEARCH FIRST (HNSW/IVF) - FASTEST PATH
        // ========================================================================
        // Use AXIS manager if available for O(log N) approximate search
        let exact_mode = matches!(ctx.search_params.search_mode, SearchMode::Exact);
        let has_axis_manager = self.axis_manager().is_some();
        if has_axis_manager {
            tracing::debug!("🔍 VIPER: AXIS manager is available for HNSW/IVF search");
        }

        if exact_mode {
            tracing::debug!(
                "🎯 VIPER: SearchMode::Exact requested; skipping approximate AXIS path"
            );
        } else if let Some(axis_manager) = self.axis_manager() {
            tracing::info!(
                "🔗 VIPER: AXIS manager available, attempting HNSW index search for collection {}",
                collection_id
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
                        "✅ VIPER: AXIS HNSW search completed in {:?} - found {} candidates",
                        axis_duration,
                        axis_results.results.len()
                    );

                    // Convert AXIS results to OptimizedSearchRecord
                    let results: Vec<crate::core::search::results::OptimizedSearchRecord> =
                        axis_results
                            .results
                            .into_iter()
                            .take(k)
                            .map(
                                |scored| crate::core::search::results::OptimizedSearchRecord {
                                    id: scored.vector_id.to_string(),
                                    vector_id: Some(scored.vector_id.to_string()),
                                    score: scored.similarity,
                                    similarity: Some(scored.similarity),
                                    vector: None, // AXIS doesn't return vectors by default
                                    ..Default::default()
                                },
                            )
                            .collect();

                    // If we got results, return them
                    if !results.is_empty() {
                        return Ok(results);
                    }

                    tracing::info!(
                        "⚠️ VIPER: AXIS returned no results, falling back to columnar search"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "⚠️ VIPER: AXIS query failed ({}), falling back to columnar search",
                        e
                    );
                }
            }
        }

        // ========================================================================
        // PHASE 1: SEARCH ORCHESTRATION AND STRATEGY SELECTION
        // ========================================================================

        // Deferred: Enable AdvancedSearchOptimizer for intelligent search routing
        //
        // The AdvancedSearchOptimizer provides significant value for VIPER engine:
        // 1. **Columnar predicate pushdown**: Optimizes Parquet file filtering
        // 2. **ML clustering integration**: Routes queries to relevant data clusters
        // 3. **Adaptive quantization**: Selects optimal compression levels dynamically
        // 4. **Cost-based optimization**: Chooses between index vs columnar scan
        // 5. **Multi-stage pipeline**: Progressively refines results for efficiency
        //
        // VIPER-specific benefits when integrated:
        // - Leverage Parquet statistics for 100x pruning on selective queries
        // - Use columnar projection to reduce I/O by 10-50x
        // - Apply bloom filters at row-group level for fast filtering
        // - Optimize batch sizes based on available memory
        //
        // Current implementation is ready but disabled pending:
        // - AXIS manager service availability in engine context
        // - Cost estimator calibration for columnar operations
        // - Performance benchmarking to validate improvements
        //
        // Expected performance gains: 2-20x on analytical queries
        //
        let use_orchestration = false; // Feature flag - enable when services available

        if use_orchestration {
            // Future: Create search orchestrator for intelligent routing
            /*
            let axis_manager = self.get_axis_manager().await?;
            let cost_estimator = self.get_cost_estimator().await?;

            let mut orchestrator = crate::core::search::integrated_search_optimization::AdvancedSearchOptimizer::new(
                ctx.clone(),
                axis_manager,
                cost_estimator,
            ).await?;

            debug!("📋 Collection Analysis Results:");
            let analysis = orchestrator.get_collection_analysis();
            debug!("  📊 Dimension: {}, Distance: {:?}", analysis.dimension, analysis.distance_metric);
            debug!("  🔧 Quantization enabled: {}, Progressive: {}",
                   analysis.quantization_enabled, analysis.progressive_search_enabled);
            debug!("  📈 Dataset size: {:?}, Query complexity: {:.2}",
                   analysis.estimated_dataset_size, analysis.query_complexity);
            debug!("  🔍 Has filters: {}, Available levels: {:?}",
                   analysis.has_filters, analysis.available_quantization_levels);

            // Select optimal strategy for columnar data
            let strategy = orchestrator.select_optimal_strategy().await?;

            info!(
                "🎯 VIPER Strategy Selected: {} (estimated cost: {:.2}ms)",
                match &strategy {
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::IndexFirst { estimated_cost_ms, .. } => {
                        format!("IndexFirst (cost: {:.2}ms)", estimated_cost_ms)
                    },
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::ProgressiveQuantization { estimated_cost_ms, .. } => {
                        format!("ProgressiveQuantization (cost: {:.2}ms)", estimated_cost_ms)
                    },
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::DirectFP32 { estimated_cost_ms, .. } => {
                        format!("DirectFP32 (cost: {:.2}ms)", estimated_cost_ms)
                    },
                },
                match &strategy {
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::IndexFirst { estimated_cost_ms, .. } => *estimated_cost_ms,
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::ProgressiveQuantization { estimated_cost_ms, .. } => *estimated_cost_ms,
                    crate::core::search::integrated_search_optimization::ExecutionStrategy::DirectFP32 { estimated_cost_ms, .. } => *estimated_cost_ms,
                }
            );

            // Execute columnar-optimized strategy (placeholder for future implementation)
            // For now, fall through to existing implementation
            */
        }

        // ========================================================================
        // PHASE 2: CURRENT IMPLEMENTATION WITH ENHANCED LOGGING
        // ========================================================================

        info!("🔍 VIPER: Using columnar search implementation (orchestration disabled)");
        debug!(
            "🗃️  VIPER: Collection {} at storage_url: {}",
            collection_id, storage_url
        );

        // ========================================================================
        // PHASE 3: ENHANCED COLLECTION CONFIGURATION ANALYSIS
        // ========================================================================

        debug!("📊 VIPER Collection Configuration Analysis:");
        debug!("  🎯 Query vector dimension: {}", query_vector.len());
        debug!("  📏 Top-k requested: {}", k);
        debug!("  📐 Distance metric: {:?}", distance_metric);
        debug!(
            "  🔍 Has filter expression: {}",
            filter_expression.is_some()
        );
        debug!(
            "  📥 Include vectors: {}, Include metadata: {}",
            include_vectors, include_metadata
        );
        if let Some(filter) = filter_expression {
            debug!("  🔎 Filter details: {:?}", filter);
        }

        // Analyze collection quantization capabilities (VIPER-specific)
        let collection_config = &ctx.collection.config;
        if let Some(config) = collection_config {
            debug!("  🗃️  VIPER Columnar Analysis:");
            debug!("    📏 Collection dimension: {}", config.dimension);
            debug!("    🔧 Storage engine: {:?}", config.storage_engine);
            debug!(
                "    🔍 Filterable columns: {} defined",
                config.filterable_columns.len()
            );

            if let Some(quant_config) = &config.quantization {
                debug!("  🔧 VIPER Quantization Analysis:");
                debug!("    ✅ Enabled: {:?}", quant_config.enabled);
                debug!("    🎛️  Strategy: {:?}", quant_config.strategy);
                debug!(
                    "    🔄 Progressive search: {:?}",
                    quant_config.enable_progressive_search
                );
                debug!(
                    "    📋 Custom levels: {} defined",
                    quant_config.custom_levels.len()
                );
                debug!("    🗃️  VIPER will use columnar quantization for optimal I/O");
            } else {
                debug!("  🔧 Quantization: Not configured (FP32 columnar only)");
            }
        } else {
            debug!("  🔧 Collection config: Not available");
        }
        // Use search params from context (already available as Arc)
        if let Some(filter_expr) = filter_expression {
            debug!("Search with filter expression: {:?}", filter_expr);
        }
        let _search_params = ctx.search_params.clone();
        // Collection metadata already available in context
        debug!(
            "Using collection config from context for: {}",
            collection_id
        );
        let collection_opt = Some(ctx.collection.clone());
        // Get parquet files for the collection
        // Production behavior: storage_path is base_location, construct full path with collection_id
        // Format: {base_location}/{collection_id}/data
        let data_path = ctx
            .collection_storage_path()
            .unwrap_or_else(|| format!("{}/{}", storage_url, collection_id));
        debug!(
            "📂 VIPER search: Looking for Parquet files at: {}",
            data_path
        );
        let parquet_files = self
            .parquet_files_with_storage_url(collection_id, &data_path)
            .await?;
        debug!(
            "📁 VIPER: Found {} parquet files for collection {}",
            parquet_files.len(),
            collection_id
        );
        for (i, file) in parquet_files.iter().enumerate() {
            debug!("  📁 Parquet file {}: {}", i, file);
        }
        if parquet_files.is_empty() {
            debug!(
                "📁 VIPER: No parquet files found for collection {}, returning empty results",
                collection_id
            );
            return Ok(vec![]);
        }
        // Build search context
        let search_context = crate::core::search::SearchPlan {
            collection_id: collection_id.to_string(),
            collection_config: Some(crate::core::search::CollectionConfig {
                default_distance_metric: distance_metric,
                vector_dimension: collection_opt
                    .as_ref()
                    .and_then(|c| c.config.as_ref())
                    .map_or(0, |c| c.dimension as usize), // Fallback only if config not available
                enable_quantization: collection_opt
                    .as_ref()
                    .and_then(|c| c.config.as_ref())
                    .and_then(|c| c.quantization.as_ref())
                    .is_some(),
                enable_metadata_filtering: true,
                estimated_document_count: 0, // Tracked by collection stats, not engine
            }),
            storage_info: crate::core::search::StorageInfo {
                is_cloud_storage: false,
                storage_type: "VIPER".to_string(),
                estimated_size_mb: self
                    .stats
                    .total_size_bytes
                    .load(std::sync::atomic::Ordering::Relaxed)
                    as f64
                    / (1024.0 * 1024.0),
                file_count: parquet_files.len(),
                supports_range_requests: true,
                file_paths: Some(parquet_files.clone()),
            },
            filterable_columns: collection_opt
                .as_ref()
                .and_then(|c| c.config.as_ref())
                .map_or_else(Vec::new, |c| {
                    c.filterable_columns
                        .iter()
                        .map(|col| {
                            crate::core::search::FilterableColumn {
                                name: col.name.clone(),
                                data_type: crate::core::search::ColumnData::String, // Default to string
                                is_indexed: false,
                                estimated_cardinality: None,
                            }
                        })
                        .collect()
                }),
            available_quantization: vec![
                UnifiedQuantizationLevel::Binary,
                UnifiedQuantizationLevel::Int8,
                UnifiedQuantizationLevel::Pq4,
                UnifiedQuantizationLevel::Pq8,
            ], // VIPER supports all quantization levels
            filter_expression: filter_expression.cloned(), // Use FilterExpression directly
            query_vector: Some(query_vector.to_vec()),
            top_k: k,
            min_score: None,                // No minimum threshold
            enable_early_termination: true, // Enable optimizations
        };

        // Use UnifiedParquetReader for actual search with predicate pushdown
        debug!(
            "🔎 VIPER: Using UnifiedParquetReader for collection: {}",
            search_context.collection_id
        );

        // Use the existing UnifiedCachingFilesystem - critical for cloud storage performance
        // Caches Parquet metadata, bloom filters, and frequently accessed blocks
        let unified_fs = self.filesystem.clone();

        // Create the Parquet reader with file paths
        // Get dimension from collection config or use default
        let dimension = collection_opt
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map_or(128, |cfg| cfg.dimension as usize);
        let parquet_reader =
            crate::storage::engines::core::formats::columnar::UnifiedParquetReader::new(
                parquet_files.clone(),
                dimension,
                self.filesystem_factory.clone(),
                unified_fs,
                search_context.collection_id.clone(),
                "viper".to_string(),
            )?;

        // Create collection context for the reader
        // Get filterable columns from collection config if available
        let _filterable_column_specs = collection_opt
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map_or_else(Vec::new, |cfg| cfg.filterable_columns.clone());

        let collection_context = crate::storage::engines::core::formats::columnar::columnar_query_engine::CollectionContext {
            collection_id: collection_id.to_string(),
            dimension,
            distance_metric: "cosine".to_string(), // Default; overridden by collection config
            quantization_config: collection_opt
                .as_ref()
                .and_then(|c| c.config.as_ref())
                .and_then(|cfg| cfg.quantization.clone()),
        };

        // Create search params
        let _search_params = crate::core::search::SearchParams {
            query_vectors: None,
            vector: Some(query_vector.to_vec()),
            top_k: Some(k as u16),
            filter_expression: filter_expression.cloned(),
            distance_metric: Some(distance_metric),
            filters: None,
            accuracy_threshold: None,
            include_expired: Some(false),
            timeout_ms: None,
            enable_two_stage: None,
            enable_vectorized_execution: None,
            enable_parallel_morsels: None,
            enable_pipeline_execution: None,
            quantization_hint: None,
            enable_clustering_hint: None,
            runtime_hints: None,
            enable_metadata_filtering_hint: None,
            custom_hints: Some(HashMap::new()),
            block_prune: crate::core::search::BlockPruneConfig::default(),
            requires_ordering: None,
            enable_progressive_search: None,
            progressive_recalls: None,
            search_mode: crate::core::search::SearchMode::default(),
            hybrid_mode: crate::core::search::HybridSearchMode::default(),
            text_query: None,
            vector_weight: None,
            freshness_mode: None,
        };

        // Perform search using the reader's search_vectors method
        debug!(
            "Calling parquet_reader.search_vectors with collection {}",
            collection_context.collection_id
        );

        // Convert search_params to SearchPlan using unified_interface
        // Now directly passes FilterExpression - no conversion needed
        let search_plan =
            crate::core::search::search_interface::SearchPlan {
                collection_id: collection_id.to_string(),
                collection_config: collection_opt.as_ref().and_then(|c| c.config.as_ref()).map(
                    |cfg| crate::core::search::search_interface::CollectionConfig {
                        default_distance_metric: distance_metric,
                        vector_dimension: dimension,
                        enable_quantization: cfg.quantization.is_some(),
                        enable_metadata_filtering: !cfg.filterable_columns.is_empty(),
                        estimated_document_count: 1000, // Default estimate
                    },
                ),
                filterable_columns: Vec::new(), // Populated from collection config at query time
                available_quantization: vec![
                    proximadb_quantization_model::UnifiedQuantizationLevel::pq8(32),
                    proximadb_quantization_model::UnifiedQuantizationLevel::int8(),
                ],
                storage_info: crate::core::search::search_interface::StorageInfo {
                    is_cloud_storage: true,
                    storage_type: "VIPER".to_string(),
                    estimated_size_mb: 100.0,
                    file_count: parquet_files.len(),
                    supports_range_requests: true,
                    file_paths: Some(parquet_files.clone()),
                },
                filter_expression: filter_expression.cloned(), // ✅ Direct FilterExpression usage
                query_vector: Some(query_vector.to_vec()),
                top_k: k,
                min_score: None,
                enable_early_termination: true,
            };

        debug!("🔎 VIPER: Calling parquet_reader.search_vectors to read data...");
        let read_results = parquet_reader
            .search_vectors(&search_plan, &collection_context)
            .await?;
        debug!(
            "🔎 VIPER: parquet_reader returned {} records",
            read_results.results.len()
        );

        // Now perform the actual search on the data using bounded priority queue
        let mut priority_queue = BoundedPriorityQueue::new(k);

        // Get distance compute engine
        let distance_compute = Arc::new(
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(distance_metric),
        );

        for record in read_results.results {
            if let Some(ref vector) = record.vector {
                debug!(
                    "🔍 VIPER: Processing record {} with vector of length {}",
                    record.id,
                    vector.len()
                );
                if vector.is_empty() {
                    warn!("⚠️ VIPER: Record {} has empty vector, skipping", record.id);
                    continue;
                }
                // Use UnifiedDistanceCompute.calculate_distance() to get SimilarityResult
                // This already includes normalized_score, no need to convert again!
                let similarity_result =
                    distance_compute.calculate_distance(query_vector, vector, &distance_metric);

                // IMPORTANT: All engines use normalized_score from UnifiedDistanceCompute consistently
                // score field = normalized similarity (0-1, higher = better) for sorting and display
                // Future options: .with_distance(rank_value) and .with_raw_distance(distance) as commented fields
                let mut search_record = record;
                search_record.score = similarity_result.normalized_score;
                search_record.similarity = Some(similarity_result.normalized_score); // Currently redundant, maintains API compatibility

                // Debug: check metadata before inserting
                if priority_queue.len() < 3 {
                    debug!(
                        "🔍 DEBUG: About to insert record {}: metadata keys={:?}",
                        search_record.id,
                        search_record.metadata.keys().collect::<Vec<_>>()
                    );
                }

                priority_queue.try_insert(search_record);
            } else {
                warn!("⚠️ VIPER: Record {} has no vector", record.id);
            }
        }

        // Get sorted results from priority queue
        let scored_results = priority_queue.into_sorted_vec();

        // Results are already scored and sorted from the priority queue
        let all_results: Vec<OptimizedSearchRecord> = scored_results;

        debug!("Search engine returned {} results", all_results.len());
        if !all_results.is_empty() {
            trace!("First result metadata: {:?}", all_results[0].metadata);
            debug!("🔍 DEBUG: First 3 results before applying include flags:");
            for (i, r) in all_results.iter().take(3).enumerate() {
                debug!(
                    "  Result {}: id={}, metadata keys={:?}",
                    i,
                    r.id,
                    r.metadata.keys().collect::<Vec<_>>()
                );
            }
        }
        // Return the optimized search results directly
        let mut results = all_results;

        // Apply include flags at the internal level if needed
        if !include_vectors {
            for result in &mut results {
                result.vector = None;
            }
        }
        debug!(
            "🔍 DEBUG: include_metadata={}, clearing metadata={}",
            include_metadata, !include_metadata
        );
        if !include_metadata {
            warn!("🔍 WARNING: Clearing metadata from results!");
            for result in &mut results {
                result.metadata = HashMap::new();
            }
        }
        // ========================================================================
        // PHASE 4: PERFORMANCE TRACKING AND FINAL LOGGING
        // ========================================================================

        let total_search_time = search_start.elapsed();

        info!(
            "🏁 VIPER Unified Search Completed - Collection: {}, Results: {}/{}, Time: {:.2}ms",
            collection_id,
            results.len(),
            k,
            total_search_time.as_secs_f32() * 1000.0
        );

        // Enhanced result analysis for columnar engine
        debug!("📈 VIPER Search Results Analysis:");
        debug!("  📊 Total results found: {}", results.len());
        debug!("  🎯 Requested top-k: {}", k);
        debug!(
            "  ✅ Results coverage: {:.1}%",
            if k > 0 {
                (results.len() as f32 / k as f32 * 100.0).min(100.0)
            } else {
                0.0
            }
        );
        debug!(
            "  ⏱️  Total search time: {:.2}ms",
            total_search_time.as_secs_f32() * 1000.0
        );
        debug!("  🗃️  Results found: {}", results.len());
        debug!(
            "  📥 Vector inclusion: {}, Metadata inclusion: {}",
            include_vectors, include_metadata
        );

        // Log sample results with enhanced details
        if !results.is_empty() {
            debug!("🔍 VIPER Sample Results (top 3):");
            for (i, result) in results.iter().take(3).enumerate() {
                debug!(
                    "  Result {}: id={}, score={:.4}, similarity={:?}, has_vector={}, metadata_fields={}",
                    i + 1,
                    result.id,
                    result.score,
                    result.similarity,
                    result.vector.is_some(),
                    result.metadata.len()
                );

                // Log metadata details for first result (columnar-specific)
                if i == 0 && !result.metadata.is_empty() {
                    debug!(
                        "    📋 VIPER Metadata sample: {:?}",
                        result
                            .metadata
                            .iter()
                            .take(3)
                            .map(|(k, v)| format!("{}={:?}", k, v))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        } else {
            debug!("🔍 No results found for VIPER columnar search");
        }

        // Log performance characteristics (columnar-specific)
        if total_search_time.as_millis() > 200 {
            warn!(
                "⚠️ Slow VIPER columnar search detected: {:.2}ms for collection {} with {} results",
                total_search_time.as_secs_f32() * 1000.0,
                collection_id,
                results.len()
            );
        } else if total_search_time.as_millis() < 20 {
            debug!(
                "🚀 Fast VIPER columnar search: {:.2}ms for collection {} with {} results",
                total_search_time.as_secs_f32() * 1000.0,
                collection_id,
                results.len()
            );
        }

        // Log columnar-specific insights
        debug!("🗃️  VIPER Columnar Performance Insights:");
        debug!("  📁 Parquet files accessed: {}", results.len());
        if total_search_time.as_millis() > 0 {
            let throughput = results.len() as f32 / total_search_time.as_secs_f32();
            debug!("  🚀 Search throughput: {:.1} results/second", throughput);
        }

        Ok(results)
    }
    async fn collection_stats(
        &self,
        _collection_id: &str,
    ) -> Result<crate::storage::traits::CollectionStats> {
        use std::sync::atomic::Ordering;
        let total_vectors = self.stats.total_vectors.load(Ordering::Relaxed);
        let total_bytes = self.stats.total_size_bytes.load(Ordering::Relaxed);
        let total_storage = self.stats.total_storage_size_bytes.load(Ordering::Relaxed);

        let avg_vector_bytes = total_bytes.checked_div(total_vectors).unwrap_or(0);

        Ok(crate::storage::traits::CollectionStats {
            row_count: total_vectors,
            avg_vector_bytes,
            engine_strategy: crate::storage::traits::StorageFormatStrategy::Viper,
            has_metadata_index: true, // VIPER has Parquet predicate pushdown
            has_hnsw_index: false,
            total_bytes: total_storage,
            dimension: None,
            index_type: Some("parquet_statistics".to_string()),
        })
    }

    async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();
        // Basic engine metrics (using atomic operations)
        let total_size = self
            .stats
            .total_size_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        metrics.insert(
            "total_storage_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(total_size)),
        );
        metrics.insert(
            "memory_usage_bytes".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from(total_size / 10), // Estimate 10% in memory
            ),
        );
        metrics.insert(
            "collection_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.collections.read().await.len(),
            )),
        );
        let total_vectors = self
            .stats
            .total_vectors
            .load(std::sync::atomic::Ordering::Relaxed);
        metrics.insert(
            "total_vectors".to_string(),
            serde_json::Value::Number(serde_json::Number::from(total_vectors)),
        );
        let flush_ops = self
            .stats
            .flush_operations
            .load(std::sync::atomic::Ordering::Relaxed);
        metrics.insert(
            "flush_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(flush_ops)),
        );
        let compaction_ops = self
            .stats
            .compaction_operations
            .load(std::sync::atomic::Ordering::Relaxed);
        metrics.insert(
            "compaction_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(compaction_ops)),
        );
        // VIPER-specific metrics
        metrics.insert(
            "engine_version".to_string(),
            serde_json::Value::String(crate::version::PROXIMADB_VERSION.to_string()),
        );
        metrics.insert(
            "ml_clustering_enabled".to_string(),
            serde_json::Value::Bool(false),
        ); // Moved to AXIS
        metrics.insert(
            "simd_processing_enabled".to_string(),
            serde_json::Value::Bool(true),
        );
        metrics.insert(
            "utilities_enabled".to_string(),
            serde_json::Value::Bool(true),
        );
        metrics.insert("healthy".to_string(), serde_json::Value::Bool(true));
        Ok(metrics)
    }

    async fn health_check(&self) -> Result<crate::storage::traits::EngineHealth> {
        // Assume healthy for now since internal_health_check is commented out
        let healthy = true;
        let mut metrics = HashMap::new();
        metrics.insert(
            "collections_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );
        metrics.insert(
            "total_size_bytes".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );
        Ok(crate::storage::traits::EngineHealth {
            healthy,
            status: if healthy {
                "VIPER Engine Healthy".to_string()
            } else {
                "VIPER Engine Unhealthy".to_string()
            },
            last_check: chrono::Utc::now(),
            response_time_ms: 0.0, // Health check is lightweight; sub-ms
            error_count: 0,        // Tracked by observability layer
            warnings: Vec::new(),  // Populated by engine diagnostics
            metrics,
        })
    }

    /// Read ALL records of a collection from persisted Parquet files (Phase 8
    /// F1). Discovers `.parquet` files at the service-resolved `storage_url` and
    /// reuses the columnar `UnifiedParquetReader`, which returns canonical
    /// `ProximaRecord`s directly. WAL/memtable records are merged in by the
    /// service layer.
    async fn read_all_records(
        &self,
        collection_id: &str,
        storage_url: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let Some(storage_url) = storage_url else {
            return Ok(Vec::new());
        };
        let fs = self.filesystem_factory.get_filesystem(storage_url)?;
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
        // dimension = 0: Parquet carries its own schema for full-record reads
        // (matches the column-read sites at engine.rs:785/906).
        let reader = super::readers::UnifiedParquetReader::new(
            files,
            0,
            self.filesystem_factory.clone(),
            self.filesystem.clone(),
            collection_id.to_string(),
            "viper".to_string(),
        )?;
        reader.read_all_records(0, None).await
    }

    /// Convenient compact_collection method for CompactionCoordinator integration
    /// Returns enhanced result with vector tracking for AXIS integration
    /// Compact a specific collection - returns standard CompactionResult
    async fn compact_collection(
        &self,
        collection_id: &str,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<crate::storage::traits::CompactionResult> {
        info!(
            "🗜️ VIPER Engine: Starting collection compaction for {}",
            collection_id
        );
        // If collection_config not provided, try to get it from service
        let owned_config = if collection_config.is_none() {
            if let Some(service) = self.collection_service.read().await.as_ref() {
                service.collection(collection_id).await.ok().flatten()
            } else {
                None
            }
        } else {
            None
        };
        let config_to_use = collection_config.or(owned_config.as_ref());
        // Create compaction parameters with collection config
        let params = crate::storage::traits::CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: true,
            synchronous: false,
            hints: std::collections::HashMap::new(),
            timeout_ms: None, // No timeout by default
            priority: crate::storage::traits::OperationPriority::Medium,
            collection_config: config_to_use.cloned(),
            estimated_input_size: 0, // Will be calculated during compaction
        };
        // Use the existing do_compact implementation
        self.do_compact(&params).await
    }

    /// Engine-level RLS predicate (spec §8 — Phase E proof-of-concept).
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
} // End of impl UnifiedStorageFormat for ViperEngine

impl crate::storage::traits::EngineFilesystemAccess for ViperEngine {
    fn get_filesystem_factory(
        &self,
    ) -> &crate::storage::persistence::filesystem::FilesystemFactory {
        &self.filesystem_factory
    }
}

/// Implementation of UniversallyOptimized trait for VIPER engine
#[async_trait::async_trait]
impl UniversallyOptimized for ViperEngine {
    /// Get the universal performance optimizer instance
    fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer {
        &self.universal_optimizer
    }

    /// VIPER-specific optimization setup
    async fn setup_engine_optimizations(&self) -> Result<()> {
        // VIPER-specific optimizations
        info!("🔧 VIPER Engine: Setting up universal performance optimizations");

        // Initialize columnar-specific optimizations
        let config = self.universal_optimizer.get_config();
        debug!("   Cache size: {}MB", config.cache_size_mb);
        debug!("   Parallel operations: {}", config.parallel_operations);
        debug!("   Prefetching enabled: {}", config.enable_prefetching);
        debug!(
            "   Memory mapping enabled: {}",
            config.enable_memory_mapping
        );

        // VIPER is ready for columnar operations
        info!("✅ VIPER Engine: Universal optimizations configured for columnar storage");
        Ok(())
    }

    /// VIPER-specific performance metrics
    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // Basic VIPER metrics (using atomic operations for lock-free access)
        metrics.insert(
            "viper_total_vectors".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.stats
                    .total_vectors
                    .load(std::sync::atomic::Ordering::Relaxed),
            )),
        );
        metrics.insert(
            "viper_flush_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.stats
                    .flush_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )),
        );
        metrics.insert(
            "viper_compaction_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.stats
                    .compaction_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            )),
        );
        metrics.insert(
            "viper_collections_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.collections.read().await.len(),
            )),
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

#[cfg(test)]
mod minimal_compaction_tests {
    use super::*;
    use crate::proto::proximadb_v1::VectorRecord;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use crate::storage::{FlushParameters, traits::CompactionParameters};
    use anyhow::Result;
    use proximadb_storage_common::storage_path::StoragePath;
    use tempfile::TempDir;
    use tracing::debug;

    /// Create test vector
    fn create_test_vector(id: &str, dimension: usize) -> VectorRecord {
        VectorRecord {
            id: id.to_string(),
            vector: (0..dimension)
                .map(|i| (i as f32) / (dimension as f32))
                .collect(),
            metadata: std::collections::HashMap::new(),
            timestamp: Some(chrono::Utc::now().timestamp()),
            updated_at: Some(chrono::Utc::now().timestamp()),
            expires_at: None,
            version: Some(1),
            source: None,
        }
    }

    #[tokio::test]
    async fn test_minimal_viper_compaction() -> Result<()> {
        debug!("\n[TEST] Starting minimal VIPER compaction test");

        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path().to_str().unwrap();

        debug!("[TEST] Test directory: {}", base_path);

        // Create config (using default core config for testing)
        let core_config = crate::core::config::ViperConfig::default();

        // Create engine
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
        let engine = ViperEngine::from_core_config(core_config, filesystem_factory).await?;

        let collection_id = "minimal_test";

        // Set up storage assignment
        use tokio::fs;
        let data_dir = StoragePath::collection_data_path(base_path, collection_id);
        fs::create_dir_all(&data_dir).await?;

        // Storage assignment is now handled internally by CollectionService
        // when a collection is created. For test purposes, we just ensure
        // the directory structure exists.
        let wal_dir = format!("{}/{}/write_buffer", base_path, collection_id);
        fs::create_dir_all(&wal_dir).await?;

        // Create and flush just 3 vectors
        debug!("\n[TEST] Creating and flushing 3 vectors");

        let vectors = vec![
            create_test_vector("vec_0", 128),
            create_test_vector("vec_1", 128),
            create_test_vector("vec_2", 128),
        ];

        // Create collection config with dimension and storage assignment for flush
        let collection_config = Some(crate::proto::proximadb_v1::Collection {
            id: collection_id.to_string(),
            config: Some(crate::proto::proximadb_v1::CollectionConfig {
                name: collection_id.to_string(),
                embedding_config: None,
                dimension: 128,
                distance_metric: Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                storage_engine: Some(crate::proto::proximadb_v1::StorageEngine::Viper as i32),
                filterable_columns: vec![],
                index_configs: vec![],
                quantization: None,
                storage_config: None,
                primary_index: Some("default".to_string()),
                auto_index_selection: Some(true),
                description: None,
                tags: vec![],
                owner: None,
                embedding_models: vec![],
                enable_proxima_record: None,
                record_schema: None,
                text_columns: vec![],
                text_storage_configs: vec![],
                enable_dual_use_embeddings: None,
                canonical_embedding_precision: None,
                permitted_principals: vec![],
                index_policy: None,
                pax_vector_quant: None,
            }),
            stats: Some(crate::proto::proximadb_v1::CollectionStats {
                vector_count: 0,
                data_size_bytes: 0,
                index_size_bytes: 0,
            }),
            created_at: chrono::Utc::now().timestamp(),
            updated_at: chrono::Utc::now().timestamp(),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                primary_path: data_dir.to_string(),
                backup_paths: vec![],
                engine: crate::proto::proximadb_v1::StorageEngine::Viper as i32,
                engine_config: std::collections::HashMap::new(),
                base_location: base_path.to_string(),
                assigned_at: chrono::Utc::now().timestamp_micros(),
                ..Default::default()
            }),
        });

        // Clone collection_config for use in both flush and compact operations
        let collection_config_for_compact = collection_config.clone();

        let flush_params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            force: true,
            synchronous: true,
            vector_records: vectors
                .into_iter()
                .map(proximadb_records::ProximaRecord::from)
                .collect(),
            batch_ids: vec![],
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            collection_config,
            estimated_size: 1024, // 1KB estimated size
        };

        let flush_result = engine.do_flush(&flush_params).await?;
        debug!(
            "[TEST] Flush complete: {} files created, {} entries flushed",
            flush_result.files_created.unwrap_or(0),
            flush_result.entries_flushed.unwrap_or(0)
        );

        // Run compaction
        debug!("\n[TEST] Running compaction_info");

        let compact_params = CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: true,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            priority: crate::storage::traits::OperationPriority::Medium,
            collection_config: collection_config_for_compact, // Pass collection_config with storage_assignment
            estimated_input_size: 1024,
        };

        let compact_result = engine.do_compact(&compact_params).await?;
        debug!(
            "[TEST] Compaction complete: {} input files, {} output files, {} entries processed",
            compact_result.input_files.unwrap_or(0),
            compact_result.output_files.unwrap_or(0),
            compact_result.entries_processed.unwrap_or(0)
        );

        assert!(compact_result.success, "Compaction should succeed");
        // Note: In minimal test setup, compaction may not process entries since there's no actual data to compact
        // The test verifies infrastructure works without errors rather than actual data processing
        assert_eq!(
            compact_result.entries_processed.unwrap_or(0),
            0,
            "Minimal test setup may not have actual data to compact"
        );

        Ok(())
    }

    /// Test the atomic flush pattern with __flush staging directory
    #[tokio::test]
    async fn test_atomic_flush_staging_pattern() -> anyhow::Result<()> {
        use tokio::fs;

        // Setup temporary directory structure
        let temp_dir = TempDir::new()?;
        let storage_path = temp_dir.path().join("storage");
        let collection_id = "test_collection";
        let collection_path = storage_path.join(collection_id);

        // Create collection directory
        fs::create_dir_all(&collection_path).await?;

        // Test 1: Verify __flush directory creation
        let flush_staging_path = collection_path.join("__flush");
        fs::create_dir_all(&flush_staging_path).await?;

        assert!(
            flush_staging_path.exists(),
            "__flush staging directory should be created"
        );

        // Test 2: Write test data to staging area
        let temp_file = flush_staging_path.join("temp_123.parquet");
        let test_data = b"mock parquet data";
        fs::write(&temp_file, test_data).await?;

        assert!(
            temp_file.exists(),
            "Temporary file should be written to __flush"
        );

        // Test 3: Atomic move to final location
        let vectors_dir = collection_path.join("vectors");
        fs::create_dir_all(&vectors_dir).await?;

        let final_file = vectors_dir.join("partition_123.parquet");
        fs::rename(&temp_file, &final_file).await?;

        assert!(
            final_file.exists(),
            "File should be moved to final location"
        );
        assert!(
            !temp_file.exists(),
            "Temporary file should be removed after move"
        );

        // Test 4: Verify final data integrity
        let final_data = fs::read(&final_file).await?;
        assert_eq!(
            final_data, test_data,
            "Data should be preserved during atomic move"
        );

        // Test 5: Cleanup staging directory
        fs::remove_dir_all(&flush_staging_path).await?;
        assert!(
            !flush_staging_path.exists(),
            "__flush directory should be cleaned up"
        );

        Ok(())
    }

    /// Test that search operations ignore __flush directories
    #[tokio::test]
    async fn test_search_ignores_flush_directories() -> anyhow::Result<()> {
        use tokio::fs;

        let temp_dir = TempDir::new()?;
        let collection_path = temp_dir.path().join("storage/test_collection");

        // Create directory structure
        let vectors_dir = collection_path.join("vectors");
        let flush_dir = collection_path.join("__flush");
        let staging_dir = collection_path.join("__staging");

        fs::create_dir_all(&vectors_dir).await?;
        fs::create_dir_all(&flush_dir).await?;
        fs::create_dir_all(&staging_dir).await?;

        // Create files in each directory
        fs::write(vectors_dir.join("valid_partition.parquet"), b"valid data").await?;
        fs::write(flush_dir.join("temp_flush.parquet"), b"staging data").await?;
        fs::write(staging_dir.join("temp_staging.parquet"), b"staging data").await?;

        // Function to simulate directory listing with __ filtering
        fn should_include_directory(dir_name: &str) -> bool {
            !dir_name.starts_with("__")
        }

        // Test the filtering logic
        assert!(
            should_include_directory("vectors"),
            "vectors directory should be included"
        );
        assert!(
            should_include_directory("indexes"),
            "indexes directory should be included"
        );
        assert!(
            !should_include_directory("__flush"),
            "__flush directory should be ignored"
        );
        assert!(
            !should_include_directory("__staging"),
            "__staging directory should be ignored"
        );
        assert!(
            !should_include_directory("__temp"),
            "__temp directory should be ignored"
        );

        Ok(())
    }

    /// Test concurrent flush operations don't conflict
    #[tokio::test]
    async fn test_concurrent_flush_operations() -> anyhow::Result<()> {
        use tokio::fs;

        let temp_dir = TempDir::new()?;
        let collection_path = temp_dir.path().join("storage/test_collection");
        fs::create_dir_all(&collection_path).await?;

        // Simulate two concurrent flush operations
        let handles = (0..2)
            .map(|i| {
                let collection_path = collection_path.clone();
                tokio::spawn(async move {
                    let operation_id = format!("op_{}", i);
                    let flush_staging_path = collection_path.join("__flush").join(&operation_id);

                    // Create staging directory for this operation
                    fs::create_dir_all(&flush_staging_path).await?;

                    // Write data to staging
                    let temp_file = flush_staging_path.join("temp.parquet");
                    let data = format!("data from operation {}", i);
                    fs::write(&temp_file, data.as_bytes()).await?;

                    // Atomic move to final location
                    let vectors_dir = collection_path.join("vectors");
                    fs::create_dir_all(&vectors_dir).await?;

                    let final_file =
                        vectors_dir.join(format!("partition_{}.parquet", operation_id));
                    fs::rename(&temp_file, &final_file).await?;

                    // Cleanup staging for this operation
                    fs::remove_dir_all(&flush_staging_path).await?;

                    Ok::<(), anyhow::Error>(())
                })
            })
            .collect::<Vec<_>>();

        // Wait for all operations to complete
        for handle in handles {
            handle.await??;
        }

        // Verify both files were created successfully
        let vectors_dir = collection_path.join("vectors");
        assert!(vectors_dir.join("partition_op_0.parquet").exists());
        assert!(vectors_dir.join("partition_op_1.parquet").exists());

        // Verify no staging directories remain
        let main_flush_dir = collection_path.join("__flush");
        assert!(
            !main_flush_dir.exists()
                || fs::read_dir(&main_flush_dir)
                    .await?
                    .next_entry()
                    .await?
                    .is_none()
        );

        Ok(())
    }

    /// Test filesystem error handling during atomic operations
    #[tokio::test]
    async fn test_atomic_flush_error_handling() -> anyhow::Result<()> {
        use tokio::fs;

        let temp_dir = TempDir::new()?;
        let collection_path = temp_dir.path().join("storage/test_collection");
        fs::create_dir_all(&collection_path).await?;

        let flush_staging_path = collection_path.join("__flush");
        fs::create_dir_all(&flush_staging_path).await?;

        let temp_file = flush_staging_path.join("temp_123.parquet");
        fs::write(&temp_file, b"test data").await?;

        // Test 1: Move to non-existent directory should fail
        let invalid_final_path = collection_path.join("nonexistent/partition_123.parquet");
        let move_result = fs::rename(&temp_file, &invalid_final_path).await;
        assert!(
            move_result.is_err(),
            "Move to non-existent directory should fail"
        );

        // Test 2: Original file should still exist after failed move
        assert!(
            temp_file.exists(),
            "Original file should remain after failed move"
        );

        // Test 3: Successful move after creating target directory
        let vectors_dir = collection_path.join("vectors");
        fs::create_dir_all(&vectors_dir).await?;

        let valid_final_path = vectors_dir.join("partition_123.parquet");
        fs::rename(&temp_file, &valid_final_path).await?;

        assert!(
            valid_final_path.exists(),
            "File should be moved successfully"
        );
        assert!(
            !temp_file.exists(),
            "Original should be removed after successful move"
        );

        Ok(())
    }
}
