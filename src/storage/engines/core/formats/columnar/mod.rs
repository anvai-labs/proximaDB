//! # Shared Columnar Storage Infrastructure - Apache Parquet-based Vector Storage
//!
//! This module provides the columnar storage foundation for NOVA and VIPER engines,
//! implementing high-performance Apache Parquet-based storage with extensive optimizations
//! for vector search workloads. It eliminates code duplication while providing each engine
//! the flexibility to implement their specific optimizations.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Columnar Storage Module                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌────────────────────────────┐  ┌──────────────────────────┐ │
//! │  │    Parquet I/O Layer      │  │   Query Engine          │ │
//! │  │  - Range reads            │  │  - Predicate pushdown   │ │
//! │  │  - Footer cache           │  │  - Progressive search   │ │
//! │  │  - Cloud optimization     │  │  - Statistics pruning   │ │
//! │  └────────────────────────────┘  └──────────────────────────┘ │
//! │  ┌────────────────────────────┐  ┌──────────────────────────┐ │
//! │  │      ID Index             │  │   Batch Operations      │ │
//! │  │  - Bloom filters          │  │  - Memory pools         │ │
//! │  │  - Dictionary encoding    │  │  - Parallel processing  │ │
//! │  │  - O(log n) lookups       │  │  - Compression          │ │
//! │  └────────────────────────────┘  └──────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Common Capabilities Provided
//!
//! ### 1. Parquet File Operations
//! - **UnifiedParquetReader**: Cloud-optimized reader with range requests and caching
//! - **StreamingParquetWriter**: Memory-efficient writer with configurable batch sizes
//! - **ParquetFooterCache**: Cached footer metadata (70-90% cloud API reduction)
//! - **Bloom Filter Integration**: Built-in bloom filters for O(1) ID existence checks
//! - **Row Group Management**: Efficient row group selection and pruning
//!
//! ### 2. Columnar Optimization
//! - **ColumnarOptimizer**: Query planning and execution optimization
//! - **Progressive Search**: Binary → INT8 → PQ → FP32 refinement pipeline
//! - **Streaming Iterator**: Memory-efficient iteration over large files
//! - **Statistics-Based Pruning**: Use Parquet statistics for early filtering
//! - **Projection Pushdown**: Read only required columns
//!
//! ### 3. ID Index Management  
//! - **ColumnarIdIndex**: Fast O(log n) ID lookups using row group metadata
//! - **Dictionary Encoding**: Efficient storage of string IDs
//! - **Row Offset Tracking**: Optional offset optimization for ID-less storage
//! - **Bloom Filter Cache**: Cached bloom filters per row group
//! - **Index Statistics**: Track hit rates and performance metrics

// Field name constants have been moved to the constants module to avoid duplication
// Re-exported below for backward compatibility

// ### 4. Schema & Metadata Management
// - **ColumnarSchemaBuilder**: unified schema creation and validation (storage-common)
// - **NativeMetadataHandler**: Efficient metadata filtering and projection
// - **FilterableColumnSpec**: Define filterable columns with proper types
// - **Schema Evolution**: Support for schema versioning and migration
//
// ### 5. Batch Operations
// - **ColumnarBatchOperations**: Optimized batch read/write operations
// - **Memory Pool Integration**: Reuse buffers across operations
// - **Parallel Processing**: Multi-threaded batch processing
// - **Compression Support**: Per-column compression configuration
//
// ### 6. Performance Features
// - **Footer Cache**: Reduce cloud API calls by 70-90%
// - **Streaming Row Groups**: 80% memory reduction for large files
// - **Dictionary Encoding**: 60% storage reduction for IDs
// - **Bloom Filters**: 95% reduction in metadata scanning
// - **Progressive Quantization**: 90% faster similarity search
// ## Key Optimizations Implemented
// 1. **Parquet Bloom Filters**: Built-in bloom filters for efficient ID lookups (benefits both engines)
// 2. **Streaming Row Groups**: Memory-efficient streaming access to large Parquet files
// 3. **ID-Aware Storage**: Keep customer ID column with optional row offset optimizations
// 4. **Progressive Search**: Binary → INT8 → PQ → FP32 quantization pipeline
// 5. **Unified Optimization**: Shared caching, statistics, and query planning
// ## Benefits for Both VIPER and NOVA
// - 95% reduction in metadata scanning overhead (bloom filters)
// - 80% memory reduction during large file processing (streaming)
// - Dictionary encoding for efficient ID storage with fast lookups
// - 90% faster similarity search with progressive quantization
// - Zero code duplication between engines for core columnar operations

pub mod columnar_io;
pub mod constants; // Column name constants
pub mod id_index;
pub use proximadb_storage_common::text_search::metadata_filter_strategy; // extracted TD-DECOMP-73
pub mod optimization; // NEW: Consolidated Parquet and Arrow IPC operations

// Modular components with semantic names (replacing old monolithic files)
pub use proximadb_engine_core::parquet_write_engine; // extracted TD-DECOMP-78
pub mod columnar_query_engine;

// Quantization now handled by unified compute module
pub mod batch_operations;
pub use proximadb_engine_core::proximablocks::columnar_config_types::{
    ColumnStatistics, ColumnarColumnStatistics, ColumnarConfig, ColumnarFileMetadata,
    OptimizationThresholds, QuantizationConfig,
};
pub use proximadb_storage_common::columnar_schema_full as schema; // extracted TD-DECOMP-74

pub mod config_builder;
pub use proximadb_engine_core::proximablocks::columnar_footer_cache as footer_cache; // extracted TD-DECOMP-76
pub mod hybrid_writer;
pub mod metadata_collector;
pub mod native_metadata;
pub use proximadb_engine_core::proximablocks::nova_metadata; // extracted TD-DECOMP-76
pub mod parquet_io_layer; // Low-level I/O operations (formerly shared_parquet_reader)
pub use metadata_collector::MetadataCollector;
pub use proximadb_engine_core::proximablocks::columnar_utilities as utilities; // extracted TD-DECOMP-76
pub use proximadb_engine_core::proximablocks::parquet_metadata; // extracted TD-DECOMP-76
pub mod columnar_compaction; // Unified Parquet compaction using StreamingParquetWriter
// quantization_config_conversion moved to common/quantization_adapter.rs

// New unified columnar infrastructure
pub use proximadb_engine_core::proximablocks::columnar_common as common; // extracted TD-DECOMP-80
pub use proximadb_engine_core::proximablocks::columnar_serialization as serialization; // extracted TD-DECOMP-80
// NOTE: Distance computation has been moved to proximadb_distance_kernel::quantized

// TEXT column storage, filtering, and full-text search (Phase 3)
pub use proximadb_storage_common::text_search::fulltext_index; // extracted TD-DECOMP-70
pub use proximadb_storage_common::text_search::metadata_filter_types::{
    ColumnarMetadataFilter, FilterCondition, FilterLogic, MetadataFilter,
};
pub use proximadb_storage_common::text_search::text_filter; // extracted TD-DECOMP-73

pub use proximadb_storage_common::text_search::text_storage; // extracted TD-DECOMP-70

// Examples demonstrating optimization benefits (moved to tests)
#[cfg(test)]
mod examples_test;

// Comprehensive tests for ID-aware columnar storage
#[cfg(test)]
mod simple_branched_test;
#[cfg(test)]
mod tests;

// Re-export common compression mapping function
pub use common::map_core_to_parquet_compression;

// Re-export column name constants
pub use constants::{
    DEFAULT_PAGE_SIZE, DEFAULT_ROW_GROUP_SIZE, DEFAULT_WRITE_BATCH_SIZE, FIELD_COLLECTION_ID,
    FIELD_EXPIRES_AT, FIELD_EXTRA_META, FIELD_ID, FIELD_IS_DELETED, FIELD_Q_BINARY, FIELD_Q_INT8,
    FIELD_Q_PQ4, FIELD_Q_PQ8, FIELD_Q_PQ16, FIELD_Q_PQ32, FIELD_QP_BINARY_THRESHOLD,
    FIELD_QP_INT8_MAX, FIELD_QP_INT8_MIN, FIELD_QP_INT8_SCALE, FIELD_QP_PQ_CENTROIDS,
    FIELD_QP_PQ_SUBQUANTIZERS, FIELD_ROW_GROUP_OFFSET, FIELD_ROW_INDEX, FIELD_SCHEMA_VERSION,
    FIELD_SOURCE, FIELD_TIMESTAMP, FIELD_UPDATED_AT, FIELD_VECTOR_FP32, FIELD_VERSION,
    PARQUET_EXTENSION, VIPER_FILE_EXTENSION,
};

// Re-exports for convenience
pub use columnar_compaction::{
    ColumnarCompactionResult, UnifiedColumnarCompaction, VersionContinuityMode,
};
pub use id_index::{ColumnarIdIndex, IndexStats, ParquetLocation};
pub use optimization::{ColumnarOptimizer, ProgressiveSearchConfig, StreamingRowGroupIterator};
// Re-export from columnar_query_engine module
pub use columnar_query_engine::{
    BranchedFilterExecutor,

    CacheStrategy,
    CollectionContext,
    FilterPath,
    FilterValue,
    PagePruningInfo,
    PageRange,
    // Core query types with semantic names
    ParquetQueryEngine,
    ParquetReader,
    QuantizationMethod,
    QueryConfig,
    QueryStatistics,
    ReaderConfig,
    ReadingStrategy,
    ReadingStrategySelector,
    RowGroupAccessPattern,
    SchemaMapping,
    SearchType,
    SeekRange,
    Stage2Strategy,
    // Unified reader types (for backward compatibility)
    UnifiedParquetReader,
    VectorPosition,
};

// Re-export from parquet_write_engine module
pub use parquet_write_engine::{
    BatchParquetWriter, IdLessLookup, ParquetWriter, ParquetWriterConfig, StreamingParquetWriter,
    StreamingParquetWriterStats,
};
// Quantization now handled by unified compute module
pub use self::metadata_filter_strategy::{
    FilterPerformanceMetrics, MetadataFilterAnalyzer, MetadataFilterStrategy,
};
pub use batch_operations::ColumnarBatchOperations;
pub use footer_cache::{FooterCacheConfig, FooterCacheStats, ParquetFooterCache, WarmingStrategy};
pub use utilities::ColumnarUtilities;

pub use config_builder::{
    FooterCacheBuilder, HybridWriterBuilder, ParquetConfigBuilder, ParquetPresets,
};
pub use hybrid_writer::{
    HybridParquetWriter, HybridWriterConfig, HybridWriterStatistics, InsertionPattern, PatternType,
    WriterMode,
};
pub use native_metadata::{
    MetadataFieldType, NativeMetadataHandler, NativeMetadataQueryOptimizer, NativeMetadataStats,
    NativePredicate, OptimizedFilter, PredicateOperator,
};

// NEW: Export shared Parquet reader components
pub use footer_cache as FooterCache;
pub use parquet_io_layer::{
    ColumnMmapStrategy, ParquetFooterCache as SharedFooterCache, ParquetMmapStrategy,
    ReaderStatsSummary as ParquetReaderStats, RowGroupMetadata,
    SharedParquetFormatReader as ParquetIOLayer,
};
// Alias for backward compatibility - module-level
pub use columnar_query_engine as ParquetQueryEngineModule;

// NEW: Export zero-copy metadata serialization components
pub use parquet_metadata::{
    ParquetColumnHeader, ParquetFooterHeader, ParquetMetadata, ParquetMetadataSerializer,
    ParquetRowGroupHeader,
};

// New unified infrastructure exports
pub use schema::{
    ColumnarFilterableSpec, ColumnarSchemaBuilder, ColumnarSchemaConfig, CompressionMetadata,
    FilterableData, create_schema_from_collection, validate_schema_compatibility,
};
pub use serialization::{
    ColumnarSerializationConfig, ColumnarSerializer, FormatPreference, SerializationResult,
};
// NOTE: SelectedFormat and QuantizedVectorData have been moved to proximadb_distance_kernel::quantized
// NOTE: Distance computation has been moved to proximadb_distance_kernel::quantized
// Use: proximadb_distance_kernel::{QuantizedDistanceCalculator, QuantizedDistanceConfig, ...}
pub use common::{
    CommonColumnarConfig, CommonColumnarOperations, DistanceComputationConfig, OptimalBatchSizes,
    PerformanceMonitor, RowGroupSizeOptimization, SchemaGenerationConfig,
    SerializationOptimizationConfig, ViperOptimizations,
};

// TEXT column storage and filtering re-exports
pub use text_filter::{
    TextColumnFilterEvaluator, TextComparisonOp, TextFilterBuilder, TextFilterError,
    TextFilterStats,
};
pub use text_storage::{
    // Constants
    CHUNKED_THRESHOLD,
    ChunkPosition,
    // RAG Chunking
    ChunkingConfig,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE,
    INLINE_THRESHOLD,
    MAX_BOUNDARY_SEARCH,
    MIN_CHUNK_SIZE,
    // Storage types
    SidecarCompression,
    SidecarRef,
    StorageType,
    TextChunk,
    TextChunker,
    TextColumnReader,
    TextColumnWriter,
    TextStorageConfig,
    TextStorageError,
    TextStorageStats,
    // Functions
    determine_storage_strategy,
};

// Full-text search index re-exports
pub use fulltext_index::{
    // Core index types
    BM25Config,
    BM25Scorer,
    ChunkIndexing,
    ChunkSearchResult,
    DocumentMetadata,
    FullTextIndex,
    FullTextIndexBuilder,
    FullTextIndexError,
    FulltextSearchResult as FullTextSearchResult,
    // Posting types
    Posting,
    PostingList,
    // Search types
    SearchOptions,
    // Statistics
    TextStatistics,
    // Tokenization
    Token,
    Tokenizer,
    TokenizerConfig,
    TokenizerType,
};

use anyhow::Result;
use arrow_schema::Schema;
use parquet::file::metadata::RowGroupMetaData;
use std::sync::Arc;

use proximadb_records::ProximaRecord;

/// Search mode for columnar engines
#[derive(Debug, Clone)]
pub enum ColumnarSearchMode {
    /// AXIS returns IDs, we lookup full vectors
    IndexDriven { ids: Vec<String> },

    /// Full similarity search without AXIS
    IndexFree {
        query: Vec<f32>,
        top_k: usize,
        filter: Option<ColumnarMetadataFilter>,
    },

    /// Hybrid mode - use AXIS for initial candidates, refine with local search
    Hybrid {
        axis_ids: Vec<String>,
        query: Vec<f32>,
        rerank_factor: f32,
    },
}

/// Backwards-compat alias for [`ColumnarRowGroupStats`].
pub type RowGroupStats = ColumnarRowGroupStats;

/// Row group statistics for optimization
#[derive(Debug, Clone)]
pub struct ColumnarRowGroupStats {
    pub row_group_id: usize,
    pub num_rows: u64,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub id_range: Option<(String, String)>,
    pub has_quantized_columns: bool,
    pub bloom_filter_size: Option<usize>,
}

/// Backwards-compat alias for [`ColumnarSearchCandidate`].
pub type SearchCandidate = ColumnarSearchCandidate;

/// Search candidate for progressive refinement
#[derive(Debug, Clone)]
pub struct ColumnarSearchCandidate {
    pub row_group_id: usize,
    pub row_offset: u32,
    pub similarity: f32,
    pub vector_id: Option<String>,
}

/// Common columnar operations trait
#[allow(async_fn_in_trait)]
pub trait ColumnarOperations {
    /// Search vectors based on mode
    async fn search(&self, mode: ColumnarSearchMode) -> Result<Vec<ProximaRecord>>;

    /// Get vectors by IDs (optimized batch lookup)
    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<ProximaRecord>>;

    /// Progressive similarity search
    async fn progressive_search(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<ColumnarMetadataFilter>,
    ) -> Result<Vec<ProximaRecord>>;

    /// Get row group statistics
    fn row_group_stats(&self) -> Vec<ColumnarRowGroupStats>;

    /// Optimize row group layout
    async fn optimize_layout(&self, collection_id: &str) -> Result<()>;
}

/// Columnar optimizations
#[derive(Debug, Clone)]
pub struct ColumnarOptimizations {
    /// Columnar projection - only load needed columns
    pub projection: Vec<String>,

    /// Predicate pushdown - filter at storage level
    pub predicates: Vec<FilterCondition>,

    /// Row group pruning - skip irrelevant groups
    pub pruned_groups: Vec<usize>,

    /// Quantization level for search
    pub quantization_level: QuantizationLevel,
}

// DEPRECATED: Replaced with proto-generated QuantizationLevel
pub use crate::proto::proximadb_v1::QuantizationLevel;

/// Create optimized Parquet schema for vectors with mandatory ID column
pub fn create_columnar_schema(
    dimension: usize,
    config: &QuantizationConfig,
    filterable_columns: &[String],
) -> Arc<Schema> {
    use arrow_schema::{DataType, Field};

    let mut fields = vec![
        // Core fields - ID is ALWAYS required for customer APIs
        Field::new(FIELD_ID, DataType::Utf8, false), // NOT NULL - critical for get_by_id, delete_by_id APIs
        Field::new(
            FIELD_VECTOR_FP32,
            DataType::FixedSizeBinary(dimension as i32 * 4),
            false,
        ),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("version", DataType::Int64, true),
        // Row group offset for internal optimizations (optional)
        Field::new("row_group_offset", DataType::UInt32, true),
        Field::new("row_index", DataType::UInt32, true),
    ];

    // Add quantized columns if enabled
    if config.enable_binary.unwrap_or(false) {
        fields.push(Field::new(
            "vector_binary",
            DataType::FixedSizeBinary(dimension.div_ceil(8) as i32),
            true,
        ));
    }

    if config.enable_int8.unwrap_or(false) {
        fields.push(Field::new(
            "vector_int8",
            DataType::FixedSizeBinary(dimension as i32),
            true,
        ));
        fields.push(Field::new("int8_scale", DataType::Float32, true));
        fields.push(Field::new("int8_zero_point", DataType::Int8, true));
    }

    if config.enable_pq.unwrap_or(false) {
        fields.push(Field::new(
            "vector_pq",
            DataType::FixedSizeBinary(config.pq_segments.unwrap_or(8) as i32),
            true,
        ));
        fields.push(Field::new("pq_codebook", DataType::Binary, true));
    }

    // Add filterable metadata columns
    for column in filterable_columns {
        // Infer type from first value (in production, use schema registry)
        fields.push(Field::new(column, DataType::Utf8, true));
    }

    // Add extra metadata field for non-filterable metadata
    fields.push(Field::new(
        FIELD_EXTRA_META,
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false, // not sorted
        ),
        true, // nullable
    ));

    Arc::new(Schema::new(fields))
}

/// Estimate memory usage for a row group
pub fn estimate_row_group_memory(row_group: &RowGroupMetaData, schema: &Schema) -> usize {
    let mut total = 0;

    for (idx, column) in row_group.columns().iter().enumerate() {
        if idx < schema.fields().len() {
            let uncompressed_size = column.uncompressed_size() as usize;
            // Add overhead for Arrow arrays
            total += uncompressed_size + (uncompressed_size / 10); // 10% overhead estimate
        }
    }

    total
}

/// Factory for creating optimized columnar components
pub struct ColumnarFactory;

impl ColumnarFactory {
    /// Create optimized Parquet reader for VIPER/NOVA engines
    /// Note: enable_id_less is optimization only, ID column is always kept
    pub async fn create_optimized_reader(
        _filesystem: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
        config: ColumnarConfig,
        _enable_id_less_optimization: bool,
    ) -> Result<UnifiedParquetReader> {
        // Note: UnifiedParquetReader now takes file_paths and dimension
        // This factory method needs to be updated based on actual usage
        // Convert ColumnarConfig to ReaderConfig
        let reader_config = ReaderConfig {
            enable_pushdown_predicates: config.enable_predicate_pushdown,
            enable_row_group_pruning: config.enable_row_group_pruning,
            enable_page_index: true,
            batch_size: 1024,
            cache_metadata: true,
            parallel_row_groups: false, // Add missing field
            cache_context: None,        // Add missing cache_context field
        };

        // Create reader with empty paths and 0 dimension - caller will set actual values
        // Create a default UnifiedCachingFilesystem for testing/default usage
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default().await?,
        );
        let base_fs = filesystem_factory.get_filesystem("file://")?;
        let cached_filesystem = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                base_fs,
                "default_collection".to_string(),
                "columnar".to_string(),
            ),
        );
        let reader = UnifiedParquetReader::new(
            vec![],
            0,
            filesystem_factory,
            cached_filesystem,
            "default_collection".to_string(),
            "columnar".to_string(),
        )?;
        Ok(reader.with_config(reader_config))
    }

    /// Create streaming Parquet writer with all optimizations
    /// Note: id_less_storage should typically be false to keep customer ID column
    pub async fn create_streaming_writer<P: AsRef<std::path::Path>>(
        file_path: P,
        dimension: usize,
        enable_bloom_filters: bool,
        enable_id_less_optimization: bool,
        quantization: QuantizationConfig,
    ) -> Result<StreamingParquetWriter> {
        let config = ParquetWriterConfig {
            enable_bloom_filters,
            id_less_storage: enable_id_less_optimization, // Should be false for customer APIs
            quantization,                                 // Use proto config directly
            ..Default::default()
        };

        // Root-side convenience: constructs the default filesystem + quantization
        // engine and injects them through the ports (TD-DECOMP-78 moved the
        // default-construction responsibility to the composition root).
        let filesystem_factory: std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> =
            std::sync::Arc::new(
                crate::storage::persistence::filesystem::FilesystemFactory::create_default()
                    .await?,
            );
        let quantization_engine: Option<
            std::sync::Arc<dyn proximadb_storage_ports::QuantizationEnginePort>,
        > = Some(std::sync::Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                std::sync::Arc::new(
                    proximadb_distance_kernel::engine::UnifiedDistanceCompute::default(),
                ),
                std::sync::Arc::new(
                    crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
                ),
            ),
        ));
        StreamingParquetWriter::new(
            file_path,
            dimension,
            config,
            None,
            filesystem_factory,
            quantization_engine,
        )
        .await
    }

    /// Create columnar optimizer with hardware-specific settings
    #[allow(clippy::todo)] // Deferred: Refactor to async and provide proper arguments (5 args required)
    pub fn create_optimizer(
        _hardware: Arc<proximadb_hardware_caps::HardwareCapabilities>,
        _config: ColumnarConfig,
    ) -> ColumnarOptimizer {
        let _distance_compute = Arc::new(
            proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(
                proximadb_distance_kernel::engine::DistanceMetric::Cosine,
            ),
        );
        // Deferred: Fix this to be async and provide proper arguments
        todo!("ColumnarOptimizer::new requires async and 5 arguments - needs refactoring")
    }
}

/// Optimization recommendations based on dataset characteristics
pub struct OptimizationRecommendations {
    pub use_bloom_filters: bool,
    pub use_id_less_storage: bool,
    pub enable_progressive_search: bool,
    pub row_group_size: usize,
    pub quantization_strategy: QuantizationStrategy,
}

#[derive(Debug, Clone)]
pub enum QuantizationStrategy {
    None,
    BinaryOnly,
    Int8Only,
    ProductQuantization,
    Progressive,
}

impl OptimizationRecommendations {
    /// Generate recommendations based on dataset characteristics
    pub fn for_dataset(
        num_vectors: u64,
        dimension: usize,
        _query_pattern: QueryPattern,
        storage_budget: StorageBudget,
    ) -> Self {
        let use_bloom_filters = num_vectors > 100_000;
        let use_id_less_storage = num_vectors > 1_000_000;
        let enable_progressive_search = dimension >= 256;

        let row_group_size = match num_vectors {
            0..=10_000 => 1_000,
            10_001..=100_000 => 5_000,
            100_001..=1_000_000 => 10_000,
            _ => 50_000,
        };

        let quantization_strategy = match (dimension, storage_budget) {
            (_, StorageBudget::Minimal) => QuantizationStrategy::Progressive,
            (d, StorageBudget::Balanced) if d >= 512 => QuantizationStrategy::ProductQuantization,
            (d, StorageBudget::Balanced) if d >= 128 => QuantizationStrategy::Int8Only,
            (d, StorageBudget::Performance) if d >= 256 => QuantizationStrategy::BinaryOnly,
            _ => QuantizationStrategy::None,
        };

        Self {
            use_bloom_filters,
            use_id_less_storage,
            enable_progressive_search,
            row_group_size,
            quantization_strategy,
        }
    }
}

#[derive(Debug, Clone)]
pub enum QueryPattern {
    IdLookupHeavy,
    SimilaritySearchHeavy,
    Mixed,
}

#[derive(Debug, Clone)]
pub enum StorageBudget {
    Performance, // Prioritize speed
    Balanced,    // Balance speed and storage
    Minimal,     // Minimize storage cost
}

// Tests moved to separate tests module file
#[cfg(test)]
mod inline_tests {
    use super::*;

    #[test]
    fn test_create_columnar_schema() {
        // Test with quantization enabled
        let mut config = QuantizationConfig::default();
        config.enable_binary = Some(true);
        config.enable_int8 = Some(true);
        config.enable_pq = Some(true);

        let filterable = vec!["category".to_string(), "price".to_string()];

        let schema = create_columnar_schema(768, &config, &filterable);

        // Check core fields
        assert!(schema.field_with_name(FIELD_ID).is_ok());
        assert!(schema.field_with_name(FIELD_VECTOR_FP32).is_ok());
        assert!(schema.field_with_name("timestamp").is_ok());

        // Check quantized fields (only present when enabled)
        assert!(schema.field_with_name("vector_binary").is_ok());
        assert!(schema.field_with_name("vector_int8").is_ok());
        assert!(schema.field_with_name("vector_pq").is_ok());

        // Check metadata fields
        assert!(schema.field_with_name("category").is_ok());
        assert!(schema.field_with_name("price").is_ok());
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();

        // Proto defaults are all false/0
        assert!(!config.enable_binary.unwrap_or(false));
        assert!(!config.enable_int8.unwrap_or(false));
        assert!(!config.enable_pq.unwrap_or(false));
        assert_eq!(config.pq_segments.unwrap_or(0), 0);
        assert_eq!(config.pq_bits.unwrap_or(0), 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        // Test small dataset recommendations
        let small_recs = OptimizationRecommendations::for_dataset(
            1_000,
            128,
            QueryPattern::Mixed,
            StorageBudget::Performance,
        );

        assert!(!small_recs.use_bloom_filters); // Small dataset
        assert!(!small_recs.use_id_less_storage); // Small dataset
        assert!(!small_recs.enable_progressive_search); // Low dimension
        assert_eq!(small_recs.row_group_size, 1_000);

        // Test large dataset recommendations
        let large_recs = OptimizationRecommendations::for_dataset(
            10_000_000,
            768,
            QueryPattern::SimilaritySearchHeavy,
            StorageBudget::Minimal,
        );

        assert!(large_recs.use_bloom_filters); // Large dataset
        assert!(large_recs.use_id_less_storage); // Large dataset
        assert!(large_recs.enable_progressive_search); // High dimension
        assert_eq!(large_recs.row_group_size, 50_000);

        match large_recs.quantization_strategy {
            QuantizationStrategy::Progressive => (), // Expected for minimal storage
            _ => panic!("Expected progressive quantization for minimal storage"),
        }
    }

    #[test]
    fn test_quantization_strategy_selection() {
        // High dimension + minimal storage = Progressive
        let recs = OptimizationRecommendations::for_dataset(
            1_000_000,
            1024,
            QueryPattern::Mixed,
            StorageBudget::Minimal,
        );
        matches!(
            recs.quantization_strategy,
            QuantizationStrategy::Progressive
        );

        // Medium dimension + balanced = INT8
        let recs = OptimizationRecommendations::for_dataset(
            1_000_000,
            256,
            QueryPattern::Mixed,
            StorageBudget::Balanced,
        );
        matches!(recs.quantization_strategy, QuantizationStrategy::Int8Only);

        // High dimension + performance = Binary
        let recs = OptimizationRecommendations::for_dataset(
            1_000_000,
            512,
            QueryPattern::SimilaritySearchHeavy,
            StorageBudget::Performance,
        );
        matches!(recs.quantization_strategy, QuantizationStrategy::BinaryOnly);
    }

    #[test]
    fn test_columnar_config_defaults() {
        let config = ColumnarConfig::default();

        assert!(config.enable_predicate_pushdown);
        assert!(config.enable_projection);
        assert!(config.enable_row_group_pruning);
        assert_eq!(config.max_cache_size_bytes, 512 * 1024 * 1024);

        // Test quantization defaults - proto defaults are all false
        assert!(!config.quantization.enable_binary.unwrap_or(false));
        assert!(!config.quantization.enable_int8.unwrap_or(false));
        assert!(!config.quantization.enable_pq.unwrap_or(false));

        // Test optimization thresholds
        assert_eq!(
            config.optimization_thresholds.row_group_pruning_threshold,
            1000
        );
        assert_eq!(config.optimization_thresholds.simd_threshold, 10000);
    }

    #[test]
    fn test_row_group_size_scaling() {
        // Test row group size recommendations scale with dataset size
        let small = OptimizationRecommendations::for_dataset(
            5_000,
            128,
            QueryPattern::Mixed,
            StorageBudget::Balanced,
        );
        let medium = OptimizationRecommendations::for_dataset(
            50_000,
            128,
            QueryPattern::Mixed,
            StorageBudget::Balanced,
        );
        let large = OptimizationRecommendations::for_dataset(
            5_000_000,
            128,
            QueryPattern::Mixed,
            StorageBudget::Balanced,
        );

        assert!(small.row_group_size < medium.row_group_size);
        assert!(medium.row_group_size < large.row_group_size);

        assert_eq!(small.row_group_size, 1_000);
        assert_eq!(medium.row_group_size, 5_000);
        assert_eq!(large.row_group_size, 50_000);
    }
}

#[cfg(test)]
pub(crate) mod writer_port_helpers {
    //! Test construction of the streaming-writer ports (TD-DECOMP-78): the
    //! default factory + engine construction lives root-side now.
    use std::sync::Arc;

    pub(crate) async fn default_ports() -> (
        Arc<dyn proximadb_storage_ports::FilesystemPort>,
        Option<Arc<dyn proximadb_storage_ports::QuantizationEnginePort>>,
    ) {
        let factory: Arc<dyn proximadb_storage_ports::FilesystemPort> = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default()
                .await
                .expect("default filesystem"),
        );
        let quant = crate::storage::engines::core::formats::columnar::hybrid_writer::root_quantization_engine();
        (factory, quant)
    }

    /// Test helper: build a StreamingParquetWriter with root-side default ports.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn make_streaming_writer<P: AsRef<std::path::Path>>(
        file_path: P,
        dimension: usize,
        config: crate::storage::engines::core::formats::columnar::parquet_write_engine::writer_config::ParquetWriterConfig,
    ) -> anyhow::Result<
        crate::storage::engines::core::formats::columnar::parquet_write_engine::streaming_writer::StreamingParquetWriter,
    >{
        let (factory, quant) = default_ports().await;
        crate::storage::engines::core::formats::columnar::parquet_write_engine::streaming_writer::StreamingParquetWriter::new(
            file_path, dimension, config, None, factory, quant,
        )
        .await
    }
}
