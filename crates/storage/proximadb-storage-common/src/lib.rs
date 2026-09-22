//! Shared storage primitives for ProximaDB storage engines and modality storage.
//!
//! Keep this crate narrow. Storage helpers belong here only when they are reused
//! by multiple storage engines or modality storage implementations.

pub mod auto_scheduler;
pub mod bitmap;
pub mod cache_config;
pub mod coarse_directory;
pub mod collection_path;
pub mod column_projector;
pub mod columnar_constants;
pub mod columnar_schema_full;
pub mod compaction_override;
pub mod deletion_vector;
pub mod delta_simd;
pub mod engine_constants;
pub mod engine_profile;
pub mod engine_type;
pub mod flush_integration;
pub mod format_conversion;
pub mod format_splits;
pub mod format_traits;
pub mod fused_quantization;
pub mod glob;
pub mod hierarchical_stats;
pub mod hilbert_curve;
pub mod id_index;
pub mod iops_budget;
pub mod metadata_collector;
pub mod mmap_file;
pub mod native_metadata;
pub mod proto_metadata_helper;
pub mod write_strategy;
/// Re-export shim — the canonical `ObjectStoreBridge` contract now lives in
/// the foundation `proximadb-catalog-schema` crate (extracted from the control
/// catalog), so this storage crate depends downward on foundation only — no
/// grandfathered storage -> control edge. All historical
/// `proximadb_storage_common::object_store_bridge::*` paths keep working.
pub mod object_store_bridge {
    pub use proximadb_catalog_schema::object_store_bridge::*;
}
pub mod morsel_scheduler;
pub mod observability_cardinality;
pub mod observability_partitioning;
pub mod observability_rollups;
pub mod oid_position_resolver;
pub mod oid_resolver_cache;
pub mod pax_block;
pub mod pax_striped_plan;
pub mod proxima_arrow;
pub mod proxima_parquet;
pub mod proxima_schema;
pub mod query_metrics;
pub mod ranged_segment;
pub mod segment_layout;
pub mod segment_reader;
pub mod smart_io_metrics;
pub mod spatial_encoding;
pub mod spill_regions;
pub mod storage_error;
pub mod storage_path;
pub mod storage_profile;
pub mod swift_id_index;
pub mod tenant_performance;
pub mod text_search;
pub mod tiering_policy;
pub mod tiering_retention;
pub mod transaction_isolation;
pub mod tst_compression;
pub mod two_phase_commit;
pub mod unified_cache_config;
pub mod wal_entry;
pub mod writer_statistics;
pub mod zero_copy_traits;

pub use auto_scheduler::{
    AutoSchedulerConfig, CompactionPolicy, OperationPriority, OperationStatus, OperationType,
    ScheduledOperation,
};
pub use bitmap::{BitmapError, BitmapIteratorAll, RoaringBitmap};
pub use cache_config::{
    AlertThresholds, CacheConfig, CoordinationConfig, EvictionPolicy, FilterCacheConfig,
    GlobalCacheConfig, IndexCacheConfig, MetadataStoreConfig, MonitoringConfig, QueryCacheConfig,
    VectorCacheConfig,
};
pub use collection_path::slug_for;
pub use column_projector::{ColumnProjection, ProjectionBuilder};
pub use columnar_constants::{
    DEFAULT_PAGE_SIZE, DEFAULT_ROW_GROUP_SIZE, DEFAULT_WRITE_BATCH_SIZE, FIELD_COLLECTION_ID,
    FIELD_EXPIRES_AT, FIELD_EXTRA_META, FIELD_ID, FIELD_IS_DELETED, FIELD_Q_BINARY, FIELD_Q_INT8,
    FIELD_Q_PQ4, FIELD_Q_PQ8, FIELD_Q_PQ16, FIELD_Q_PQ32, FIELD_QP_BINARY_THRESHOLD,
    FIELD_QP_INT8_MAX, FIELD_QP_INT8_MIN, FIELD_QP_INT8_SCALE, FIELD_QP_PQ_CENTROIDS,
    FIELD_QP_PQ_SUBQUANTIZERS, FIELD_ROW_GROUP_OFFSET, FIELD_ROW_INDEX, FIELD_SCHEMA_VERSION,
    FIELD_SOURCE, FIELD_TIMESTAMP, FIELD_UPDATED_AT, FIELD_VECTOR_FP32, FIELD_VERSION,
    PARQUET_EXTENSION, QUANTIZATION_COLUMNS, QUANTIZATION_PARAMETER_COLUMNS,
    QUANTIZED_VECTOR_COLUMNS, REQUIRED_COLUMNS, TEMPORAL_COLUMNS, VIPER_FILE_EXTENSION,
};
pub use compaction_override::{
    COMPACTION_L0_THRESHOLD_TAG_PREFIX, COMPACTION_TAG_PREFIX, GlobalCompactionGate,
    L0_COMPACTION_ENABLED_ENV, global_compaction_gate, resolve_compaction_armed,
    resolve_l0_threshold,
};
pub use delta_simd::{delta_decode_f32, delta_decode_i32_prefix_sum, delta_decode_i64_prefix_sum};
pub use engine_constants::{
    BLOOM_FILTER_EXT, DEFAULT_BLOCK_METADATA_OVERHEAD_BYTES, DEFAULT_TARGET_BLOCK_SIZE_BYTES,
    ENGINE_COLUMNAR, ENGINE_HELIX, ENGINE_NOVA, ENGINE_RAPTOR, ENGINE_SST, ENGINE_SWIFT,
    ENGINE_VIPER, HELIX_FILE_EXT, HELIX_MAGIC, INDEX_EXT, MAX_TARGET_BLOCK_SIZE_BYTES,
    METADATA_EXT, MIN_TARGET_BLOCK_SIZE_BYTES, NOVA_FILE_EXT, NOVA_MAGIC, PRISM_FILE_EXT,
    RAPTOR_FILE_EXT, RAPTOR_MAGIC, SST_FILE_EXT, SST_MAGIC, STATS_EXT, SWIFT_FILE_EXT, SWIFT_MAGIC,
    VIPER_FILE_EXT, VIPER_MAGIC,
};
pub use engine_profile::EngineProfile;
pub use engine_type::StorageEngineType;
pub use flush_integration::{FlushConfig, FlushIntegration, FlushStats};
pub use format_conversion::{
    CompressionFormat, ConversionError, ConversionResult, ConversionStatistics, FormatConverter,
    QuantizedFormat, StorageFormat,
};
pub use format_splits::{
    ColumnBounds, FileSplit, SpatialBounds, SplitLocality, SplitStatistics, SplitType,
};
pub use format_traits::{
    ColumnStats, CompactionContext, CompactionResult, ComparisonOp, DefaultFormatDetector,
    FileEntry, FileStats, FilterExpression as FormatFilterExpression, FormatDetector,
    FormatStatistics, FormatType, InternalFormat, MergeAction, OpenTableFormat, OptimizeContext,
    OptimizeResult, ReadContext, RecordBatchStream, Snapshot, StorageFormat as StorageFormatTrait,
    VectorBatch, VectorBatchStream, VectorReadContext, VectorWriteContext, WriteContext, WriteMode,
    WriteResult,
};
pub use fused_quantization::{
    QuantizationParams, fused_decode_binary_to_f32, fused_decode_int4_to_f32,
    fused_decode_int8_to_f32, progressive_decode_binary_int8_f32,
};
pub use glob::{GlobError, GlobMatcher, GlobPattern, glob_match};
pub use hierarchical_stats::{
    EnhancedRowGroupStats, QuantizationStats as HierarchicalQuantizationStats, SelectivityHints,
    SuperBlock, ZoneMap,
};
pub use hilbert_curve::{HilbertCurve, HilbertStats, HilbertUtils};
pub use id_index::{
    BloomFilter, ColumnarIdIndex, IndexStats, PageIdIndex, ParquetLocation, RowGroupIdIndex,
};
pub use metadata_collector::{MetadataCollectionConfig, MetadataCollector, NoOpCollector};
pub use mmap_file::{Advice, MmapFile, MmapMutFile, MmapParquetReader, MmapSstReader, MmapView};
pub use native_metadata::{
    FieldStatistics, MetadataFieldType, NativeMetadataHandler, NativeMetadataQueryOptimizer,
    NativeMetadataStats, NativePredicate, OptimizedFilter, PredicateOperator,
};
pub use observability_cardinality::{
    CardinalityConfig, CardinalityLimiter, CheckResult, LabelStats, LimitAction,
};
pub use observability_partitioning::{
    Partition, PartitionConfig, PartitionGranularity, PartitionRange, TimePartitioner,
};
pub use observability_rollups::{
    AggregationFunction, RollupConfig, RollupInterval, RollupManager, RollupView,
};
pub use proxima_schema::{
    AutoGenerateType, AvroStyleField, AvroStyleSchema, AvroStyleType, DefaultValue, ProximaColumn,
    ProximaSchema, ProximaSchemaAvroExt, TimeUnit,
};
pub use query_metrics::{QueryStatistics, StatisticsCollector};
pub use smart_io_metrics::{FileIoMetrics, IoMetrics, IoMetricsSnapshot};
pub use spatial_encoding::{CodeType, SpatialCode, U512};
pub use storage_error::{ErrorContext, StorageError, StorageErrorKind};
pub use storage_path::StoragePath;
pub use storage_profile::{
    CHURN_WORKING_SET_MB_ENV, STORAGE_PROFILE_ENV, STORAGE_PROFILE_TAG_PREFIX, StorageProfile,
    churn_working_set_ceiling_bytes, resolve_storage_profile,
};
pub use swift_id_index::{
    BPlusNode, BlockLocation, IdIndex, IndexStats as SwiftIndexStats, RecordLocation,
    TwoLevelIdIndex,
};
pub use tenant_performance::{
    PerformanceMonitoringConfig, SLACheckResult, SLAViolationType, TenantMetrics,
    TenantPerformanceMonitor, TenantSLA,
};
pub use tiering_policy::{
    PerformanceTier, PolicyAction, PolicyCondition, TieringMetadata, TieringPolicy, TieringRule,
};
pub use tiering_retention::{
    RetentionAction, RetentionCondition, RetentionMetadata, RetentionOperationStatus,
    RetentionPolicy, RetentionRule,
};
pub use transaction_isolation::{IsolationLevel, IsolationManager, ReadSnapshot, WriteSet};
pub use tst_compression::{TimeSeriesCompressor, TstCompressionConfig};
pub use two_phase_commit::{
    CommitResult, ParticipantState, ParticipantType, PrepareResult, TransactionState,
    TwoPhaseCommitConfig, TwoPhaseCommitProtocol, TwoPhaseCommitStats, TwoPhaseParticipant,
    TwoPhaseTransaction,
};
pub use unified_cache_config::{
    CacheBehaviorConfig, ConfigError as UnifiedCacheConfigError, DiskCacheConfig,
    IOOptimizationConfig, InvalidationStrategy, MemoryConfig, PerformanceConfig, StorageTier,
    UnifiedCacheConfig, UnifiedEvictionPolicy, WorkloadType,
};
pub use wal_entry::{
    CanonicalOperation,
    CanonicalWalEntry,
    CdcLogicalView,
    CdcOperation,
    CdcRecordEvent,
    EdgeRef,
    OpenTableFormat as CanonicalOpenTableFormat,
    ProjectionDirective,
    ProjectionFreshness,
    ProjectionRebuilder,
    RecoveryResult,
    SnapshotManifest,
    // T3.1 Slices 3 + 4 — branch / merge primitives for ADR-012 orchestration.
    filter_wal_by_branch_lsn,
    find_merge_base_lsn,
    latest_checkpoint,
    recover_from_canonical_wal,
};
pub use writer_statistics::{AggregatedBatchStats, BatchWriteStats, StreamingParquetWriterStats};
pub use zero_copy_traits::{
    AccessFrequency, CollectionContext, EngineMetadata, MetadataSerializer,
    QueryContext as ZeroCopyQueryContext, QueryType,
};
