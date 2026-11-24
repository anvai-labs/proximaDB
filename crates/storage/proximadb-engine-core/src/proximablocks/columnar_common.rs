//! Common Columnar Logic for the NOVA Engine
//!
//! This module extracts columnar functionality shared across engines,
//! eliminating code duplication while providing specialized optimizations
//! for each engine through composition and configuration.

use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_schema::Schema;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::proximablocks::columnar_config_types::{
    ColumnarConfig, ColumnarFileMetadata, QuantizationConfig,
};
use crate::proximablocks::columnar_serialization::{
    ColumnarSerializationConfig, ColumnarSerializer, FormatPreference, MemoryOptimizationConfig,
    SIMDConfig, SerializationCompressionConfig, SerializationResult,
};
use proximadb_compression::CompressionAlgorithm;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::columnar_schema_full::{
    self as schema_full, ColumnarFilterableSpec, ColumnarSchemaBuilder, ColumnarSchemaConfig,
    CompressionMetadata, SchemaOptimization, WriterPropertiesConfig,
};
use proximadb_storage_ports::{FilesystemPort, StorageQuantizationEnginePort};

/// Common columnar-engine configuration (NOVA and friends)
#[derive(Debug, Clone, Default)]
pub struct CommonColumnarConfig {
    /// Base columnar configuration
    pub base_config: ColumnarConfig,

    /// Schema generation configuration
    pub schema_config: SchemaGenerationConfig,

    /// Serialization and compression settings
    pub serialization_config: SerializationOptimizationConfig,

    /// Distance computation settings
    // distance_config removed - engines use compute module directly

    /// Engine-specific optimizations
    pub engine_optimizations: EngineOptimizations,

    /// Performance monitoring settings
    pub monitoring_config: ColumnarMonitoringConfig,
}

/// Schema generation configuration
#[derive(Debug, Clone)]
pub struct SchemaGenerationConfig {
    /// Auto-detect quantization from collection metadata
    pub auto_detect_quantization: bool,

    /// Default compression strategy
    pub default_compression_strategy: CompressionStrategy,

    /// Schema caching TTL in seconds
    pub schema_cache_ttl_seconds: u64,

    /// Maximum cached schemas per collection
    pub max_cached_schemas: usize,
}

/// Compression strategy configuration
#[derive(Debug, Clone)]
pub struct CompressionStrategy {
    /// Compression algorithm selection
    pub algorithm_selection: CompressionAlgorithmSelection,

    /// Compression levels per data type
    pub compression_levels: CompressionLevels,

    /// Enable adaptive compression based on data characteristics
    pub enable_adaptive_compression: bool,
}

/// Algorithm selection strategy
#[derive(Debug, Clone)]
pub enum CompressionAlgorithmSelection {
    /// Fixed algorithm for all data
    Fixed(proximadb_compression::CompressionAlgorithm),
    /// Per-column type optimization
    PerColumnType,
    /// Adaptive based on data characteristics
    Adaptive,
    /// Engine-specific optimization
    EngineOptimized,
}

/// Compression levels for different data types
#[derive(Debug, Clone)]
pub struct CompressionLevels {
    pub fp32_vectors: Option<i32>,
    pub quantized_vectors: Option<i32>,
    pub metadata_columns: Option<i32>,
    pub id_columns: Option<i32>,
}

/// Serialization optimization configuration
#[derive(Debug, Clone, Default)]
pub struct SerializationOptimizationConfig {
    /// Memory pool configuration
    pub memory_pools: ColumnarMemoryPoolConfig,

    /// SIMD optimization settings
    pub simd_settings: SIMDOptimizationSettings,

    /// Batch processing configuration
    pub batch_processing: BatchProcessingConfig,

    /// Zero-copy optimization settings
    pub zero_copy_optimization: ZeroCopyConfig,
}

/// Backwards-compat alias for [`ColumnarMemoryPoolConfig`].
pub type MemoryPoolConfig = ColumnarMemoryPoolConfig;

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct ColumnarMemoryPoolConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,

    /// Pool size limits
    pub pool_size_limits: PoolSizeLimits,

    /// Pool cleanup interval in seconds
    pub cleanup_interval_seconds: u64,

    /// Enable pool statistics collection
    pub enable_statistics: bool,
}

/// Pool size limits for different data types
#[derive(Debug, Clone)]
pub struct PoolSizeLimits {
    pub fp32_pool_max_vectors: usize,
    pub int8_pool_max_vectors: usize,
    pub binary_pool_max_vectors: usize,
    pub pq_pool_max_vectors: usize,
    pub max_vector_size_bytes: usize,
}

/// SIMD optimization settings
#[derive(Debug, Clone)]
pub struct SIMDOptimizationSettings {
    /// Enable SIMD acceleration
    pub enable_simd: bool,

    /// Minimum batch size for SIMD
    pub min_batch_size: usize,

    /// Target instruction set
    pub target_instruction_set: String,

    /// Enable auto-vectorization
    pub enable_auto_vectorization: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Optimal batch size for different operations
    pub optimal_batch_sizes: OptimalBatchSizes,

    /// Enable adaptive batch sizing
    pub enable_adaptive_batching: bool,

    /// Memory budget for batch processing
    pub memory_budget_mb: usize,
}

/// Optimal batch sizes for different operations
#[derive(Debug, Clone)]
pub struct OptimalBatchSizes {
    pub serialization_batch_size: usize,
    pub distance_computation_batch_size: usize,
    pub compression_batch_size: usize,
    pub decompression_batch_size: usize,
}

/// Zero-copy optimization configuration
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Enable zero-copy for fixed-size data
    pub enable_zero_copy_fixed_size: bool,

    /// Enable memory mapping for large files
    pub enable_memory_mapping: bool,

    /// Memory mapping threshold in MB
    pub mmap_threshold_mb: usize,

    /// Page alignment requirements
    pub page_alignment_bytes: usize,
}

/// Distance computation configuration
#[derive(Debug, Clone)]
pub struct DistanceComputationConfig {
    /// Default distance metric
    pub default_distance_metric: proximadb_distance_kernel::DistanceMetric,

    /// Progressive search configuration
    pub progressive_search: ColumnarCommonProgressiveSearchConfig,

    /// Distance caching settings
    pub distance_caching: DistanceCachingConfig,

    /// Hardware acceleration preferences
    pub hardware_acceleration: HardwareAccelerationConfig,
}

/// Backwards-compat alias for [`ColumnarCommonProgressiveSearchConfig`].
pub type ProgressiveSearchConfig = ColumnarCommonProgressiveSearchConfig;

/// Progressive search configuration
#[derive(Debug, Clone)]
pub struct ColumnarCommonProgressiveSearchConfig {
    /// Enable progressive search
    pub enable_progressive: bool,

    /// Quality thresholds for each stage
    pub quality_thresholds: QualityThresholds,

    /// Early termination settings
    pub early_termination: EarlyTerminationConfig,
}

/// Quality thresholds for progressive search stages
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub binary_threshold: f32,
    pub int8_threshold: f32,
    pub pq_threshold: f32,
    pub fp32_threshold: f32,
}

/// Early termination configuration
#[derive(Debug, Clone)]
pub struct EarlyTerminationConfig {
    /// Enable early termination based on quality
    pub enable_quality_based: bool,

    /// Enable early termination based on result count
    pub enable_count_based: bool,

    /// Confidence threshold for early termination
    pub confidence_threshold: f32,
}

/// Distance caching configuration
#[derive(Debug, Clone)]
pub struct DistanceCachingConfig {
    /// Enable PQ distance table caching
    pub enable_pq_caching: bool,

    /// Cache size in MB
    pub cache_size_mb: usize,

    /// Cache eviction policy
    pub eviction_policy: String,

    /// Precompute tables on collection load
    pub precompute_on_load: bool,
}

/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct HardwareAccelerationConfig {
    /// Prefer GPU for large computations
    pub prefer_gpu: bool,

    /// GPU threshold (number of vectors)
    pub gpu_threshold: usize,

    /// Enable CPU SIMD acceleration
    pub enable_cpu_simd: bool,

    /// Enable custom instruction optimization
    pub enable_custom_instructions: bool,
}

/// Engine-specific optimizations
#[derive(Debug, Clone, Default)]
pub struct EngineOptimizations {
    /// NOVA-specific optimizations
    pub nova_optimizations: NovaOptimizations,

    /// Shared optimizations
    pub shared_optimizations: SharedOptimizations,
}

/// NOVA-specific optimizations
#[derive(Debug, Clone)]
pub struct NovaOptimizations {
    /// Enable hierarchical statistics
    pub enable_hierarchical_stats: bool,

    /// Zone map configuration
    pub zone_map_config: ZoneMapOptimization,

    /// Streaming processing settings
    pub streaming_processing: StreamingProcessingConfig,

    /// Advanced caching strategies
    pub advanced_caching: AdvancedCachingConfig,
}

/// Shared optimizations between engines
#[derive(Debug, Clone)]
pub struct SharedOptimizations {
    /// Enable bloom filters for ID lookups
    pub enable_bloom_filters: bool,

    /// Dictionary encoding for low-cardinality columns
    pub enable_dictionary_encoding: bool,

    /// Run-length encoding for repetitive data
    pub enable_rle_encoding: bool,

    /// Column statistics collection
    pub enable_column_statistics: bool,
}

// Detailed configuration structs
#[derive(Debug, Clone)]
pub struct RowGroupSizeOptimization {
    pub min_size: usize,
    pub max_size: usize,
    pub target_compression_ratio: f32,
    pub adaptive_sizing: bool,
}

#[derive(Debug, Clone)]
pub struct ZoneMapOptimization {
    pub enable_zone_maps: bool,
    pub zone_size: usize,
    pub enable_nested_zones: bool,
    pub max_zone_depth: usize,
}

#[derive(Debug, Clone)]
pub struct StreamingProcessingConfig {
    pub enable_streaming: bool,
    pub stream_buffer_size: usize,
    pub max_concurrent_streams: usize,
    pub stream_timeout_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct AdvancedCachingConfig {
    pub enable_adaptive_caching: bool,
    pub cache_size_mb: usize,
    pub cache_levels: usize,
    pub prefetch_strategy: String,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct ColumnarMonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,

    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,

    /// Enable detailed tracing
    pub enable_detailed_tracing: bool,

    /// Resource usage monitoring
    pub resource_monitoring: ResourceColumnarMonitoringConfig,
}

/// Resource monitoring configuration
#[derive(Debug, Clone)]
pub struct ResourceColumnarMonitoringConfig {
    /// Monitor memory usage
    pub monitor_memory: bool,

    /// Monitor CPU usage
    pub monitor_cpu: bool,

    /// Monitor I/O operations
    pub monitor_io: bool,

    /// Enable alerting on resource thresholds
    pub enable_alerting: bool,
}

/// Common columnar operations implementation
pub struct CommonColumnarOperations {
    /// Configuration
    config: CommonColumnarConfig,

    /// Schema builder for dynamic schema generation
    schema_builder: Arc<ColumnarSchemaBuilder>,

    /// Serializer for transparent format conversion
    serializer: Arc<ColumnarSerializer>,

    // Distance computation removed - engines should use compute module directly
    /// Filesystem factory for I/O operations
    #[allow(dead_code)]
    filesystem_factory: Arc<dyn FilesystemPort>,

    /// Metadata cache
    metadata_cache: Arc<RwLock<MetadataCache>>,

    /// Performance monitor
    performance_monitor: Arc<ColumnarPerformanceMonitor>,
}

/// Metadata cache for columnar files
#[derive(Debug)]
struct MetadataCache {
    /// Cached file metadata
    file_metadata: HashMap<String, CachedFileMetadata>,

    /// Cache statistics
    hits: usize,
    misses: usize,

    /// Memory usage tracking
    memory_usage_bytes: usize,
}

/// Cached file metadata with expiration
#[derive(Debug, Clone)]
struct CachedFileMetadata {
    metadata: ColumnarFileMetadata,
    schema: Arc<Schema>,
    compression_metadata: CompressionMetadata,
    timestamp: std::time::Instant,
    last_accessed: std::time::Instant,
    access_count: usize,
}

/// Backwards-compat alias for [`ColumnarPerformanceMonitor`].
pub type PerformanceMonitor = ColumnarPerformanceMonitor;

/// Performance monitoring and metrics collection
#[derive(Debug)]
pub struct ColumnarPerformanceMonitor {
    /// Operation metrics
    operation_metrics: Arc<RwLock<ColumnarCommonOperationMetrics>>,

    /// Resource usage metrics
    resource_metrics: Arc<RwLock<ResourceMetrics>>,

    /// Configuration
    config: ColumnarMonitoringConfig,
}

/// Backwards-compat alias for [`ColumnarCommonOperationMetrics`].
#[allow(dead_code)]
pub type OperationMetrics = ColumnarCommonOperationMetrics;

/// Operation performance metrics
#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct ColumnarCommonOperationMetrics {
    /// Serialization metrics
    serialization_ops: usize,
    serialization_total_time_ms: f64,
    serialization_bytes_processed: usize,

    /// Distance computation metrics
    #[allow(dead_code)]
    distance_ops: usize,
    #[allow(dead_code)]
    distance_total_time_ms: f64,
    #[allow(dead_code)]
    distance_vectors_processed: usize,

    /// Schema generation metrics
    schema_generations: usize,
    #[allow(dead_code)]
    schema_cache_hits: usize,
    #[allow(dead_code)]
    schema_cache_misses: usize,

    /// I/O metrics
    #[allow(dead_code)]
    read_ops: usize,
    #[allow(dead_code)]
    write_ops: usize,
    bytes_read: usize,
    bytes_written: usize,
}

/// Resource usage metrics
#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct ResourceMetrics {
    /// Memory usage
    #[allow(dead_code)]
    memory_usage_bytes: usize,
    #[allow(dead_code)]
    peak_memory_usage_bytes: usize,

    /// CPU usage
    #[allow(dead_code)]
    cpu_usage_percent: f32,

    /// I/O metrics
    #[allow(dead_code)]
    disk_read_mb_s: f32,
    #[allow(dead_code)]
    disk_write_mb_s: f32,

    /// Cache metrics
    #[allow(dead_code)]
    cache_hit_ratio: f32,
    #[allow(dead_code)]
    cache_memory_usage_bytes: usize,
}

impl CommonColumnarOperations {
    /// Create new common operations instance
    pub async fn new(
        config: CommonColumnarConfig,
        filesystem_factory: Arc<dyn FilesystemPort>,
    ) -> Result<Self> {
        Self::with_quantization_engine(config, filesystem_factory, None).await
    }

    /// Composition-root variant (TD-DECOMP-80): inject the quantization engine
    /// (port) the serializer should use for transparent quantization. Passing
    /// `None` mirrors the pre-extraction behavior of a `None` engine only when
    /// no quantization is configured; root callers that configure quantization
    /// supply the concrete engine (impl'd in the modality kernel crate).
    pub async fn with_quantization_engine(
        config: CommonColumnarConfig,
        filesystem_factory: Arc<dyn FilesystemPort>,
        quantization_engine: Option<Arc<dyn StorageQuantizationEnginePort>>,
    ) -> Result<Self> {
        info!(
            "Initializing common columnar operations with config: {:?}",
            config.base_config
        );

        // Initialize schema builder
        let schema_builder = Arc::new(ColumnarSchemaBuilder::new());

        // Initialize serializer
        let serialization_config = ColumnarSerializationConfig {
            dimension: 768, // Deferred: Make configurable
            quantization: config.base_config.quantization.clone().into(),
            compression: config.serialization_config.to_serialization_compression(),
            memory_optimization: config.serialization_config.to_memory_optimization(),
            simd_config: config.serialization_config.to_simd_config(),
        };
        let serializer = match quantization_engine {
            Some(engine) => Arc::new(
                ColumnarSerializer::new(serialization_config)?.with_quantization_engine(engine),
            ),
            None => Arc::new(ColumnarSerializer::new(serialization_config)?),
        };

        // Distance computation removed - use compute module directly in engines

        // Initialize metadata cache
        let metadata_cache = Arc::new(RwLock::new(MetadataCache {
            file_metadata: HashMap::new(),
            hits: 0,
            misses: 0,
            memory_usage_bytes: 0,
        }));

        // Initialize performance monitor
        let performance_monitor = Arc::new(ColumnarPerformanceMonitor::new(
            config.monitoring_config.clone(),
        ));

        info!("Common columnar operations initialized successfully");

        Ok(Self {
            config,
            schema_builder,
            serializer,
            // distance_compute removed - use compute module directly
            filesystem_factory,
            metadata_cache,
            performance_monitor,
        })
    }

    /// Generate optimized schema for collection
    pub async fn generate_schema(
        &self,
        collection_id: &str,
        dimension: usize,
        quantization: Option<&QuantizationConfig>,
        filterable_columns: &[proximadb_proto::proximadb_v1::FilterableColumnSpec],
    ) -> Result<(Arc<Schema>, CompressionMetadata)> {
        let start_time = std::time::Instant::now();

        debug!(
            "Generating schema for collection: {} (dim: {})",
            collection_id, dimension
        );

        let schema_config = ColumnarSchemaConfig {
            dimension,
            quantization: quantization.cloned(),
            filterable_columns: ColumnarFilterableSpec::from_proto_vec(filterable_columns),
            optimization: self.config.schema_config.to_schema_optimization(),
            compression_strategy: self
                .config
                .schema_config
                .default_compression_strategy
                .to_columnar_compression(),
        };

        let (schema, compression_metadata) = self
            .schema_builder
            .build_schema(collection_id, &schema_config)
            .await?;

        let generation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update metrics
        self.performance_monitor
            .record_schema_generation(generation_time)
            .await;

        info!(
            "Schema generated for collection {} in {:.2}ms ({} fields)",
            collection_id,
            generation_time,
            schema.fields().len()
        );

        Ok((schema, compression_metadata))
    }

    /// Serialize vector records with transparent quantization
    pub async fn serialize_records(
        &self,
        records: &[ProximaRecord],
        schema: &Schema,
    ) -> Result<SerializationResult> {
        let start_time = std::time::Instant::now();

        debug!(
            "Serializing {} records with transparent quantization",
            records.len()
        );

        let result = self.serializer.serialize_vectors(records, schema).await?;

        let serialization_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let bytes_processed = records.first().map_or(0, |r| {
            records.len() * r.embeddings.first().map_or(0, |e| e.values.len()) * 4
        });

        // Update metrics
        self.performance_monitor
            .record_serialization(serialization_time, bytes_processed)
            .await;

        info!(
            "Serialized {} records in {:.2}ms (compression ratio: {:.2}x)",
            records.len(),
            serialization_time,
            result.metadata.compression_stats.compression_ratio
        );

        Ok(result)
    }

    /// Deserialize records with format preference
    pub async fn deserialize_records(
        &self,
        arrays: &HashMap<String, ArrayRef>,
        schema: &Schema,
        format_preference: FormatPreference,
    ) -> Result<Vec<ProximaRecord>> {
        let start_time = std::time::Instant::now();

        debug!(
            "Deserializing records with format preference: {:?}",
            format_preference
        );

        let records = self
            .serializer
            .deserialize_vectors(arrays, schema, format_preference)
            .await?;

        let deserialization_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update metrics
        self.performance_monitor
            .record_deserialization(deserialization_time, records.len())
            .await;

        info!(
            "Deserialized {} records in {:.2}ms",
            records.len(),
            deserialization_time
        );

        Ok(records)
    }

    // Distance computation methods removed - use compute module directly
    // Columnar engines (NOVA) should use:
    // - proximadb_distance_kernel::engine::UnifiedDistanceCompute
    // - crate::compute::quantization::storage_engine::StorageQuantizationEngine

    // Batch distance computation removed - use compute module directly

    // Progressive distance computation removed - use compute module directly

    /// Load file metadata with caching
    pub async fn load_file_metadata<P: AsRef<Path>>(
        &self,
        file_path: P,
    ) -> Result<(ColumnarFileMetadata, Arc<Schema>, CompressionMetadata)> {
        let path_str = file_path.as_ref().to_string_lossy().to_string();

        // Check cache first
        {
            let mut cache = self.metadata_cache.write().await;
            if let Some(cached) = cache.file_metadata.get_mut(&path_str)
                && !self.is_cache_expired(cached)
            {
                cached.last_accessed = std::time::Instant::now();
                cached.access_count += 1;

                let metadata = cached.metadata.clone();
                let schema = cached.schema.clone();
                let compression_metadata = cached.compression_metadata.clone();

                cache.hits += 1;

                debug!("File metadata cache hit for: {}", path_str);
                return Ok((metadata, schema, compression_metadata));
            }
        }

        // Load metadata from file
        let start_time = std::time::Instant::now();
        let (metadata, schema, compression_metadata) =
            self.load_file_metadata_from_disk(&file_path).await?;
        let load_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Cache the result
        {
            let mut cache = self.metadata_cache.write().await;
            let cached = CachedFileMetadata {
                metadata: metadata.clone(),
                schema: schema.clone(),
                compression_metadata: compression_metadata.clone(),
                timestamp: std::time::Instant::now(),
                last_accessed: std::time::Instant::now(),
                access_count: 1,
            };

            cache.file_metadata.insert(path_str.clone(), cached);
            cache.misses += 1;
            cache.memory_usage_bytes += self.estimate_metadata_size(&metadata);

            // Evict old entries if cache is too large
            self.evict_metadata_cache_if_needed(&mut cache).await;
        }

        debug!(
            "File metadata loaded from disk in {:.2}ms: {}",
            load_time, path_str
        );

        Ok((metadata, schema, compression_metadata))
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(
        &self,
    ) -> Result<(ColumnarCommonOperationMetrics, ResourceMetrics)> {
        let operation_metrics = {
            let guard = self.performance_monitor.operation_metrics.read().await;
            (*guard).clone()
        };
        let resource_metrics = {
            let guard = self.performance_monitor.resource_metrics.read().await;
            (*guard).clone()
        };

        Ok((operation_metrics, resource_metrics))
    }

    /// Clear caches and reset metrics
    pub async fn reset_caches_and_metrics(&self) -> Result<()> {
        // Clear metadata cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.file_metadata.clear();
            cache.hits = 0;
            cache.misses = 0;
            cache.memory_usage_bytes = 0;
        }

        // Clear schema cache
        // Note: ColumnarSchemaBuilder doesn't expose clear method, would need to add

        // Reset performance metrics
        self.performance_monitor.reset_metrics().await;

        info!("Caches and metrics reset successfully");

        Ok(())
    }

    // Helper methods
    async fn load_file_metadata_from_disk<P: AsRef<Path>>(
        &self,
        _file_path: P,
    ) -> Result<(ColumnarFileMetadata, Arc<Schema>, CompressionMetadata)> {
        // This would implement actual file metadata loading
        // For now, return placeholder data
        info!("File metadata loading from disk not fully implemented");

        use proximadb_distance_kernel::DistanceMetric;

        let metadata = ColumnarFileMetadata {
            collection_id: "placeholder".to_string(),
            num_vectors: 0,
            dimension: 768,
            distance_metric: DistanceMetric::Cosine,
            quantization: QuantizationConfig::default(),
            column_stats: HashMap::new(),
            version: 1,
            timestamp: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
        };

        let schema = Arc::new(arrow_schema::Schema::empty());
        let compression_metadata = CompressionMetadata {
            column_compression: HashMap::new(),
            compression_ratios: HashMap::new(),
            writer_properties: WriterPropertiesConfig::default(),
        };

        Ok((metadata, schema, compression_metadata))
    }

    fn is_cache_expired(&self, cached: &CachedFileMetadata) -> bool {
        let ttl =
            std::time::Duration::from_secs(self.config.schema_config.schema_cache_ttl_seconds);
        cached.timestamp.elapsed() > ttl
    }

    fn estimate_metadata_size(&self, _metadata: &ColumnarFileMetadata) -> usize {
        // Rough estimate of metadata memory usage
        1024 // 1KB per metadata entry
    }

    async fn evict_metadata_cache_if_needed(&self, cache: &mut MetadataCache) {
        let max_entries = self.config.schema_config.max_cached_schemas;

        if cache.file_metadata.len() > max_entries {
            // Simple LRU eviction - remove oldest accessed entries
            let mut entries: Vec<_> = cache
                .file_metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.last_accessed))
                .collect();
            entries.sort_by_key(|(_, last_accessed)| *last_accessed);

            let to_remove = entries.len() - max_entries;
            let paths_to_remove: Vec<_> = entries
                .iter()
                .take(to_remove)
                .map(|(path, _)| path.clone())
                .collect();
            for path in paths_to_remove {
                cache.file_metadata.remove(&path);
            }

            debug!("Evicted {} metadata cache entries", to_remove);
        }
    }
}

impl ColumnarPerformanceMonitor {
    fn new(config: ColumnarMonitoringConfig) -> Self {
        Self {
            operation_metrics: Arc::new(RwLock::new(ColumnarCommonOperationMetrics::default())),
            resource_metrics: Arc::new(RwLock::new(ResourceMetrics::default())),
            config,
        }
    }

    async fn record_schema_generation(&self, _duration_ms: f64) {
        if self.config.enable_metrics {
            let mut metrics = self.operation_metrics.write().await;
            metrics.schema_generations += 1;
        }
    }

    async fn record_serialization(&self, duration_ms: f64, bytes_processed: usize) {
        if self.config.enable_metrics {
            let mut metrics = self.operation_metrics.write().await;
            metrics.serialization_ops += 1;
            metrics.serialization_total_time_ms += duration_ms;
            metrics.serialization_bytes_processed += bytes_processed;
        }
    }

    async fn record_deserialization(&self, _duration_ms: f64, _record_count: usize) {
        if self.config.enable_metrics {
            let _metrics = self.operation_metrics.write().await;
            // Deserialization metrics would be tracked separately if needed
        }
    }

    #[allow(dead_code)]
    async fn record_distance_computation(&self, _duration_ms: f64, _vector_count: usize) {
        if self.config.enable_metrics {
            let mut metrics = self.operation_metrics.write().await;
            metrics.distance_ops += 1;
        }
    }

    async fn reset_metrics(&self) {
        let mut op_metrics = self.operation_metrics.write().await;
        *op_metrics = ColumnarCommonOperationMetrics::default();

        let mut res_metrics = self.resource_metrics.write().await;
        *res_metrics = ResourceMetrics::default();
    }
}

// Trait implementations for configuration conversions
// Note: Option<T> already has From<T> implementation in standard library

// Extension trait implementations would go here to convert between different config types
impl SerializationOptimizationConfig {
    fn to_serialization_compression(&self) -> SerializationCompressionConfig {
        SerializationCompressionConfig::default()
    }

    fn to_memory_optimization(&self) -> MemoryOptimizationConfig {
        MemoryOptimizationConfig::default()
    }

    fn to_simd_config(&self) -> SIMDConfig {
        SIMDConfig::default()
    }
}

impl DistanceComputationConfig {
    #[allow(dead_code)]
    fn to_simd_optimization(&self) -> proximadb_distance_kernel::quantized::SIMDOptimization {
        proximadb_distance_kernel::quantized::SIMDOptimization::default()
    }

    #[allow(dead_code)]
    fn to_cache_config(&self) -> proximadb_distance_kernel::quantized::DistanceCacheConfig {
        proximadb_distance_kernel::quantized::DistanceCacheConfig::default()
    }

    #[allow(dead_code)]
    fn to_approximation_config(&self) -> proximadb_distance_kernel::quantized::ApproximationConfig {
        proximadb_distance_kernel::quantized::ApproximationConfig::default()
    }

    #[allow(dead_code)]
    fn to_hardware_preferences(&self) -> proximadb_distance_kernel::quantized::HardwarePreferences {
        proximadb_distance_kernel::quantized::HardwarePreferences::default()
    }
}

impl SchemaGenerationConfig {
    fn to_schema_optimization(&self) -> SchemaOptimization {
        SchemaOptimization::default()
    }
}

impl CompressionStrategy {
    fn to_columnar_compression(&self) -> schema_full::CompressionStrategy {
        schema_full::CompressionStrategy::default()
    }
}

// Default implementations

// Implement Default for all the nested config structs...
impl Default for SchemaGenerationConfig {
    fn default() -> Self {
        Self {
            auto_detect_quantization: true,
            default_compression_strategy: CompressionStrategy::default(),
            schema_cache_ttl_seconds: 3600,
            max_cached_schemas: 1000,
        }
    }
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self {
            algorithm_selection: CompressionAlgorithmSelection::PerColumnType,
            compression_levels: CompressionLevels::default(),
            enable_adaptive_compression: true,
        }
    }
}

impl Default for CompressionLevels {
    fn default() -> Self {
        Self {
            fp32_vectors: Some(6),      // ZSTD level 6
            quantized_vectors: Some(3), // LZ4 level 3
            metadata_columns: Some(9),  // High compression for metadata
            id_columns: Some(6),
        }
    }
}

impl Default for ColumnarMemoryPoolConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_size_limits: PoolSizeLimits::default(),
            cleanup_interval_seconds: 300,
            enable_statistics: true,
        }
    }
}

impl Default for PoolSizeLimits {
    fn default() -> Self {
        Self {
            fp32_pool_max_vectors: 1000,
            int8_pool_max_vectors: 2000,
            binary_pool_max_vectors: 5000,
            pq_pool_max_vectors: 3000,
            max_vector_size_bytes: 64 * 1024, // 64KB
        }
    }
}

impl Default for SIMDOptimizationSettings {
    fn default() -> Self {
        Self {
            enable_simd: true,
            min_batch_size: 64,
            target_instruction_set: "auto".to_string(),
            enable_auto_vectorization: true,
        }
    }
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            optimal_batch_sizes: OptimalBatchSizes::default(),
            enable_adaptive_batching: true,
            memory_budget_mb: 256,
        }
    }
}

impl Default for OptimalBatchSizes {
    fn default() -> Self {
        Self {
            serialization_batch_size: 1000,
            distance_computation_batch_size: 500,
            compression_batch_size: 2000,
            decompression_batch_size: 1500,
        }
    }
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            enable_zero_copy_fixed_size: true,
            enable_memory_mapping: true,
            mmap_threshold_mb: 100,
            page_alignment_bytes: 4096,
        }
    }
}

impl Default for DistanceComputationConfig {
    fn default() -> Self {
        Self {
            default_distance_metric: proximadb_distance_kernel::DistanceMetric::Cosine,
            progressive_search: ColumnarCommonProgressiveSearchConfig::default(),
            distance_caching: DistanceCachingConfig::default(),
            hardware_acceleration: HardwareAccelerationConfig::default(),
        }
    }
}

impl Default for ColumnarCommonProgressiveSearchConfig {
    fn default() -> Self {
        Self {
            enable_progressive: true,
            quality_thresholds: QualityThresholds::default(),
            early_termination: EarlyTerminationConfig::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            binary_threshold: 0.7,
            int8_threshold: 0.9,
            pq_threshold: 0.85,
            fp32_threshold: 1.0,
        }
    }
}

impl Default for EarlyTerminationConfig {
    fn default() -> Self {
        Self {
            enable_quality_based: true,
            enable_count_based: true,
            confidence_threshold: 0.95,
        }
    }
}

impl Default for DistanceCachingConfig {
    fn default() -> Self {
        Self {
            enable_pq_caching: true,
            cache_size_mb: 256,
            eviction_policy: "LRU".to_string(),
            precompute_on_load: true,
        }
    }
}

impl Default for HardwareAccelerationConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            gpu_threshold: 10000,
            enable_cpu_simd: true,
            enable_custom_instructions: true,
        }
    }
}

impl Default for NovaOptimizations {
    fn default() -> Self {
        Self {
            enable_hierarchical_stats: true,
            zone_map_config: ZoneMapOptimization::default(),
            streaming_processing: StreamingProcessingConfig::default(),
            advanced_caching: AdvancedCachingConfig::default(),
        }
    }
}

impl Default for SharedOptimizations {
    fn default() -> Self {
        Self {
            enable_bloom_filters: true,
            enable_dictionary_encoding: true,
            enable_rle_encoding: true,
            enable_column_statistics: true,
        }
    }
}

impl Default for RowGroupSizeOptimization {
    fn default() -> Self {
        Self {
            min_size: 10000,
            max_size: 100000,
            target_compression_ratio: 3.0,
            adaptive_sizing: true,
        }
    }
}

impl Default for ZoneMapOptimization {
    fn default() -> Self {
        Self {
            enable_zone_maps: true,
            zone_size: 10000,
            enable_nested_zones: true,
            max_zone_depth: 3,
        }
    }
}

impl Default for StreamingProcessingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            stream_buffer_size: 1024 * 1024, // 1MB
            max_concurrent_streams: 8,
            stream_timeout_seconds: 300,
        }
    }
}

impl Default for AdvancedCachingConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_caching: true,
            cache_size_mb: 512,
            cache_levels: 3,
            prefetch_strategy: "adaptive".to_string(),
        }
    }
}

impl Default for ColumnarMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_seconds: 60,
            enable_detailed_tracing: false,
            resource_monitoring: ResourceColumnarMonitoringConfig::default(),
        }
    }
}

impl Default for ResourceColumnarMonitoringConfig {
    fn default() -> Self {
        Self {
            monitor_memory: true,
            monitor_cpu: true,
            monitor_io: true,
            enable_alerting: false,
        }
    }
}

/// Map core compression algorithm to Parquet compression
/// This is shared across columnar engines for consistent compression handling
pub fn map_core_to_parquet_compression(
    algorithm: CompressionAlgorithm,
    level: Option<i32>,
) -> Result<parquet::basic::Compression> {
    use parquet::basic::Compression;

    let compression = match algorithm {
        CompressionAlgorithm::None => Compression::UNCOMPRESSED,
        CompressionAlgorithm::Zstd => {
            if let Some(lvl) = level {
                Compression::ZSTD(parquet::basic::ZstdLevel::try_new(lvl)?)
            } else {
                Compression::ZSTD(parquet::basic::ZstdLevel::default())
            }
        }
        CompressionAlgorithm::Lz4 => Compression::LZ4,
        CompressionAlgorithm::Snappy => Compression::SNAPPY,
        CompressionAlgorithm::Gzip => {
            if let Some(lvl) = level {
                Compression::GZIP(parquet::basic::GzipLevel::try_new(lvl as u32)?)
            } else {
                Compression::GZIP(parquet::basic::GzipLevel::default())
            }
        }
        CompressionAlgorithm::Brotli => {
            if let Some(lvl) = level {
                Compression::BROTLI(parquet::basic::BrotliLevel::try_new(lvl as u32)?)
            } else {
                Compression::BROTLI(parquet::basic::BrotliLevel::default())
            }
        }
        CompressionAlgorithm::Lzo => Compression::LZO,
        // Map unsupported algorithms to fallbacks
        CompressionAlgorithm::Deflate | CompressionAlgorithm::Zlib => {
            // Deflate/Zlib are similar to GZIP
            if let Some(lvl) = level {
                Compression::GZIP(parquet::basic::GzipLevel::try_new(lvl as u32)?)
            } else {
                Compression::GZIP(parquet::basic::GzipLevel::default())
            }
        }
        CompressionAlgorithm::Lz4hc => Compression::LZ4, // Use regular LZ4
        CompressionAlgorithm::Xz | CompressionAlgorithm::Lzma => {
            // XZ and LZMA provide high compression, map to ZSTD with high level
            let high_level = level.unwrap_or(9).max(9);
            Compression::ZSTD(parquet::basic::ZstdLevel::try_new(high_level)?)
        }
        CompressionAlgorithm::Bzip2 => {
            // BZip2 provides good compression, map to Brotli
            if let Some(lvl) = level {
                Compression::BROTLI(parquet::basic::BrotliLevel::try_new(lvl as u32)?)
            } else {
                Compression::BROTLI(parquet::basic::BrotliLevel::default())
            }
        }
        CompressionAlgorithm::Mixed => {
            // Mixed compression - Use ZSTD level 3 as default for Parquet
            // Per-column optimization will be handled at the writer level
            info!("Using Mixed compression with ZSTD level 3 as base");
            Compression::ZSTD(parquet::basic::ZstdLevel::try_new(3)?)
        }
    };

    Ok(compression)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = CommonColumnarConfig::default();

        // Test basic configuration
        assert!(config.schema_config.auto_detect_quantization);
        assert_eq!(config.schema_config.schema_cache_ttl_seconds, 3600);

        // Test serialization config
        assert!(config.serialization_config.memory_pools.enable_pooling);
        assert!(config.serialization_config.simd_settings.enable_simd);

        // Distance config removed - engines use compute module directly

        // Test engine optimizations
        assert!(
            config
                .engine_optimizations
                .nova_optimizations
                .enable_hierarchical_stats
        );
        assert!(
            config
                .engine_optimizations
                .shared_optimizations
                .enable_bloom_filters
        );
    }

    #[test]
    fn test_compression_levels() {
        let levels = CompressionLevels::default();

        assert_eq!(levels.fp32_vectors, Some(6));
        assert_eq!(levels.quantized_vectors, Some(3));
        assert_eq!(levels.metadata_columns, Some(9));
        assert_eq!(levels.id_columns, Some(6));
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds::default();

        assert_eq!(thresholds.binary_threshold, 0.7);
        assert_eq!(thresholds.int8_threshold, 0.9);
        assert_eq!(thresholds.pq_threshold, 0.85);
        assert_eq!(thresholds.fp32_threshold, 1.0);
    }

    #[test]
    fn test_pool_size_limits() {
        let limits = PoolSizeLimits::default();

        assert_eq!(limits.fp32_pool_max_vectors, 1000);
        assert_eq!(limits.int8_pool_max_vectors, 2000);
        assert_eq!(limits.binary_pool_max_vectors, 5000);
        assert_eq!(limits.pq_pool_max_vectors, 3000);
        assert_eq!(limits.max_vector_size_bytes, 64 * 1024);
    }
}
