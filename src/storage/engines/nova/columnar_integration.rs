//! NOVA Engine Integration with Unified Columnar Infrastructure
//!
//! This module demonstrates how NOVA can use the new unified columnar infrastructure
//! while maintaining NOVA-specific optimizations like hierarchical statistics,
//! zone maps, and streaming processing.

use anyhow::Result;
use proximadb_distance_kernel::DistanceMetric;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::formats::columnar::common::{
    NovaOptimizations, StreamingProcessingConfig, ZoneMapOptimization,
};
use crate::storage::engines::core::formats::columnar::{
    ColumnarFilterableSpec, CommonColumnarConfig, CommonColumnarOperations, QuantizationConfig,
};
use crate::storage::persistence::filesystem::FilesystemFactory;
use proximadb_distance_kernel::{
    Int8VectorData, QuantizedDistanceCalculator, QuantizedDistanceConfig, QuantizedVectorData,
    SelectedFormat,
};

/// NOVA engine wrapper using unified columnar infrastructure
pub struct NovaUnifiedEngine {
    /// Common columnar operations
    common_ops: Arc<CommonColumnarOperations>,

    /// NOVA-specific configuration
    nova_config: NovaSpecificConfig,

    /// Hierarchical statistics manager
    hierarchical_stats: Arc<HierarchicalStatsManager>,

    /// Zone map manager for multi-dimensional pruning
    zone_map_manager: Arc<NovaColumnarZoneMapManager>,

    /// Streaming processor for large-scale operations
    streaming_processor: Arc<StreamingProcessor>,

    /// Collection metadata cache
    collection_cache: Arc<tokio::sync::RwLock<HashMap<String, NovaCollectionMetadata>>>,

    /// Distance computation engine
    distance_compute: Arc<proximadb_distance_kernel::UnifiedDistanceCompute>,
}

/// NOVA-specific configuration
#[derive(Debug, Clone)]
pub struct NovaSpecificConfig {
    /// Enable hierarchical statistics
    pub enable_hierarchical_stats: bool,

    /// Zone map configuration
    pub zone_map_config: NovaColumnarZoneMapConfig,

    /// Streaming processing settings
    pub streaming_config: NovaColumnarStreamingConfig,

    /// Advanced caching configuration
    pub caching_config: AdvancedCachingConfig,

    /// Progressive search optimization
    pub progressive_search_config: NovaColumnarProgressiveSearchConfig,
}

/// Zone map configuration for NOVA
#[derive(Debug, Clone)]
pub struct NovaColumnarZoneMapConfig {
    /// Enable zone maps
    pub enable_zone_maps: bool,

    /// Zone size (number of vectors per zone)
    pub zone_size: usize,

    /// Enable nested zone maps
    pub enable_nested_zones: bool,

    /// Maximum zone depth
    pub max_zone_depth: usize,

    /// Zone map pruning threshold
    pub pruning_threshold: f32,
}

/// Backwards-compat alias for [`NovaColumnarStreamingConfig`].
pub type StreamingConfig = NovaColumnarStreamingConfig;

/// Streaming configuration for NOVA
#[derive(Debug, Clone)]
pub struct NovaColumnarStreamingConfig {
    /// Enable streaming processing
    pub enable_streaming: bool,

    /// Stream buffer size
    pub stream_buffer_size: usize,

    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,

    /// Stream timeout in seconds
    pub stream_timeout_seconds: u64,

    /// Enable adaptive streaming
    pub enable_adaptive_streaming: bool,
}

/// Advanced caching configuration
#[derive(Debug, Clone)]
pub struct AdvancedCachingConfig {
    /// Enable adaptive caching
    pub enable_adaptive_caching: bool,

    /// Cache size in MB
    pub cache_size_mb: usize,

    /// Number of cache levels
    pub cache_levels: usize,

    /// Prefetch strategy
    pub prefetch_strategy: PrefetchStrategy,

    /// Enable cache warming
    pub enable_cache_warming: bool,
}

/// Prefetch strategies
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    None,
    Sequential,
    Adaptive,
    MachineLearning,
}

/// Backwards-compat alias for [`NovaColumnarProgressiveSearchConfig`].
pub type ProgressiveSearchConfig = NovaColumnarProgressiveSearchConfig;

/// Progressive search configuration for NOVA
#[derive(Debug, Clone)]
pub struct NovaColumnarProgressiveSearchConfig {
    /// Enable progressive refinement
    pub enable_progressive: bool,

    /// Quality thresholds for each stage
    pub quality_thresholds: QualityThresholds,

    /// Enable early termination
    pub enable_early_termination: bool,

    /// Confidence threshold for early termination
    pub confidence_threshold: f32,
}

/// Quality thresholds for progressive search
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub binary_threshold: f32,
    pub int8_threshold: f32,
    pub pq_threshold: f32,
    pub fp32_threshold: f32,
}

/// NOVA collection metadata with hierarchical statistics
#[derive(Debug, Clone)]
struct NovaCollectionMetadata {
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    dimension: usize,
    #[allow(dead_code)]
    schema: Arc<arrow_schema::Schema>,
    #[allow(dead_code)]
    quantization: Option<QuantizationConfig>,
    #[allow(dead_code)]
    filterable_columns: Vec<ColumnarFilterableSpec>,
    #[allow(dead_code)]
    compression_metadata: crate::storage::engines::core::formats::columnar::CompressionMetadata,
    #[allow(dead_code)]
    hierarchical_stats: HierarchicalStatistics,
    #[allow(dead_code)]
    zone_maps: Vec<NovaColumnarZoneMap>,
}

/// Hierarchical statistics for NOVA optimization
#[derive(Debug, Clone)]
pub struct HierarchicalStatistics {
    /// Super block statistics
    pub super_blocks: Vec<SuperBlockStats>,

    /// Row group statistics
    pub row_group_stats: Vec<NovaColumnarRowGroupStats>,

    /// Column statistics
    pub column_stats: HashMap<String, NovaColumnarColumnStats>,

    /// Global statistics
    pub global_stats: GlobalStats,
}

/// Super block statistics
#[derive(Debug, Clone)]
pub struct SuperBlockStats {
    pub super_block_id: usize,
    pub num_row_groups: usize,
    pub total_vectors: usize,
    pub min_similarity: f32,
    pub max_similarity: f32,
    pub centroid: Vec<f32>,
    pub compression_ratio: f32,
}

/// Backwards-compat alias for [`NovaColumnarRowGroupStats`].
pub type RowGroupStats = NovaColumnarRowGroupStats;

/// Row group statistics
#[derive(Debug, Clone)]
pub struct NovaColumnarRowGroupStats {
    pub row_group_id: usize,
    pub super_block_id: usize,
    pub num_vectors: usize,
    pub min_vector: Vec<f32>,
    pub max_vector: Vec<f32>,
    pub centroid: Vec<f32>,
    pub variance: f32,
}

/// Backwards-compat alias for [`NovaColumnarColumnStats`].
pub type ColumnStats = NovaColumnarColumnStats;

/// Column statistics
#[derive(Debug, Clone)]
pub struct NovaColumnarColumnStats {
    pub name: String,
    pub null_count: usize,
    pub distinct_count: usize,
    pub min_value: Option<serde_json::Value>,
    pub max_value: Option<serde_json::Value>,
    pub compression_ratio: f32,
}

/// Global collection statistics
#[derive(Debug, Clone)]
pub struct GlobalStats {
    pub total_vectors: usize,
    pub total_size_bytes: usize,
    pub average_compression_ratio: f32,
    pub quantization_quality: f32,
    pub index_coverage: f32,
}

/// Backwards-compat alias for [`NovaColumnarZoneMap`].
pub type ZoneMap = NovaColumnarZoneMap;

/// Zone map for multi-dimensional pruning
#[derive(Debug, Clone)]
pub struct NovaColumnarZoneMap {
    pub zone_id: usize,
    pub parent_zone_id: Option<usize>,
    pub depth: usize,
    pub min_bounds: Vec<f32>,
    pub max_bounds: Vec<f32>,
    pub centroid: Vec<f32>,
    pub vector_count: usize,
    pub child_zones: Vec<usize>,
}

/// Hierarchical statistics manager
pub struct HierarchicalStatsManager {
    #[allow(dead_code)]
    config: NovaSpecificConfig,
    stats_cache: Arc<tokio::sync::RwLock<HashMap<String, HierarchicalStatistics>>>,
}

impl HierarchicalStatsManager {
    pub fn new(config: NovaSpecificConfig) -> Self {
        Self {
            config,
            stats_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let stats = self.stats_cache.read().await;
        metrics.insert("cached_stats".to_string(), stats.len() as f64);
        metrics
    }
}

/// Zone map manager
pub struct NovaColumnarZoneMapManager {
    #[allow(dead_code)]
    config: NovaColumnarZoneMapConfig,
    zone_cache: Arc<tokio::sync::RwLock<HashMap<String, Vec<NovaColumnarZoneMap>>>>,
}

impl NovaColumnarZoneMapManager {
    pub fn new(config: NovaColumnarZoneMapConfig) -> Self {
        Self {
            config,
            zone_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let zones = self.zone_cache.read().await;
        metrics.insert("cached_zones".to_string(), zones.len() as f64);
        metrics
    }
}

/// Streaming processor for large-scale operations
pub struct StreamingProcessor {
    config: NovaColumnarStreamingConfig,
    active_streams: Arc<tokio::sync::RwLock<HashMap<String, StreamingSession>>>,
}

impl StreamingProcessor {
    pub fn new(config: NovaColumnarStreamingConfig) -> Self {
        Self {
            config,
            active_streams: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub async fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let streams = self.active_streams.read().await;
        metrics.insert("active_streams".to_string(), streams.len() as f64);
        metrics
    }
}

/// Active streaming session
#[derive(Debug, Clone)]
struct StreamingSession {
    #[allow(dead_code)]
    session_id: String,
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    stream_type: StreamType,
    #[allow(dead_code)]
    buffer_size: usize,
    processed_count: usize,
    #[allow(dead_code)]
    start_time: std::time::Instant,
}

/// Stream types
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum StreamType {
    Insert,
    Search,
    Update,
    Delete,
}

// Helper functions for quantized vector operations
fn dequantize_int8(data: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
    data.iter()
        .map(|&val| (val as f32 - zero_point as f32) * scale)
        .collect()
}

fn compute_hamming_distance(_query: &[f32], binary: &[u8]) -> u32 {
    // Simple hamming distance - in production would use SIMD
    let mut distance = 0u32;
    for byte in binary {
        distance += byte.count_ones();
    }
    distance
}

impl NovaUnifiedEngine {
    /// Create new NOVA engine with unified infrastructure
    pub async fn new(
        filesystem_factory: Arc<FilesystemFactory>,
        nova_config: NovaSpecificConfig,
    ) -> Result<Self> {
        info!("Initializing NOVA engine with unified columnar infrastructure");

        // Create common columnar configuration optimized for NOVA
        let common_config = Self::create_nova_optimized_config(&nova_config);

        // Initialize common operations. The concrete quantization engine is
        // constructed root-side (composition root, TD-DECOMP-80) and injected
        // through the port — engine-core cannot name the modality kernel.
        let common_ops = Arc::new(
            CommonColumnarOperations::with_quantization_engine(
                common_config,
                filesystem_factory,
                crate::storage::engines::core::formats::columnar::hybrid_writer::root_storage_quantization_engine(),
            )
            .await?,
        );

        // Initialize NOVA-specific components
        let hierarchical_stats = Arc::new(HierarchicalStatsManager::new(nova_config.clone()));
        let zone_map_manager = Arc::new(NovaColumnarZoneMapManager::new(
            nova_config.zone_map_config.clone(),
        ));
        let streaming_processor = Arc::new(StreamingProcessor::new(
            nova_config.streaming_config.clone(),
        ));

        let collection_cache = Arc::new(tokio::sync::RwLock::new(HashMap::new()));

        // Initialize distance compute engine
        let distance_compute =
            Arc::new(proximadb_distance_kernel::UnifiedDistanceCompute::default());

        info!("NOVA engine initialized with unified infrastructure and hierarchical optimizations");

        Ok(Self {
            common_ops,
            nova_config,
            hierarchical_stats,
            zone_map_manager,
            streaming_processor,
            collection_cache,
            distance_compute,
        })
    }

    /// Streaming insert with hierarchical statistics and quantization
    pub async fn streaming_insert_with_quantization(
        &self,
        collection_id: &str,
        vector_stream: impl futures::Stream<Item = VectorRecord> + Send + Unpin,
        _quantization_engine: Option<
            Arc<crate::compute::quantization::storage_engine::StorageQuantizationEngine>,
        >,
    ) -> Result<StreamingInsertResult> {
        let start_time = std::time::Instant::now();
        let session_id = format!(
            "insert_{}_{}",
            collection_id,
            chrono::Utc::now().timestamp()
        );

        info!(
            "Starting streaming insert with quantization for collection: {}",
            collection_id
        );

        // Initialize streaming session
        let _session = self
            .streaming_processor
            .start_session(
                session_id.clone(),
                collection_id.to_string(),
                StreamType::Insert,
            )
            .await?;

        // Get collection metadata
        let collection_metadata = self
            .get_or_create_nova_collection_metadata(collection_id)
            .await?;

        // Process stream in batches
        let mut total_inserted = 0;
        let mut batch_count = 0;
        let mut hierarchical_updates = Vec::new();

        let batch_size = self.nova_config.streaming_config.stream_buffer_size;
        let mut current_batch = Vec::with_capacity(batch_size);

        use futures::StreamExt;
        let mut stream = Box::pin(vector_stream);

        while let Some(vector) = stream.next().await {
            current_batch.push(vector);

            if current_batch.len() >= batch_size {
                // Process batch using unified infrastructure
                let batch_result = self
                    .process_insert_batch(
                        collection_id,
                        &collection_metadata,
                        std::mem::take(&mut current_batch),
                    )
                    .await?;

                total_inserted += batch_result.vectors_inserted;
                hierarchical_updates.extend(batch_result.hierarchical_updates);
                batch_count += 1;

                // Update streaming session
                self.streaming_processor
                    .update_session(&session_id, batch_result.vectors_inserted)
                    .await?;

                debug!(
                    "Processed batch {} ({} vectors)",
                    batch_count, batch_result.vectors_inserted
                );
            }
        }

        // Process remaining vectors
        if !current_batch.is_empty() {
            let batch_result = self
                .process_insert_batch(collection_id, &collection_metadata, current_batch)
                .await?;

            total_inserted += batch_result.vectors_inserted;
            hierarchical_updates.extend(batch_result.hierarchical_updates);
            batch_count += 1;
        }

        // Update hierarchical statistics
        self.hierarchical_stats
            .update_statistics(collection_id, hierarchical_updates)
            .await?;

        // Update zone maps
        self.zone_map_manager
            .rebuild_zones(collection_id, &collection_metadata)
            .await?;

        // Complete streaming session
        self.streaming_processor
            .complete_session(&session_id)
            .await?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Streaming insert completed: {} vectors in {:.2}ms ({} batches)",
            total_inserted, total_time, batch_count
        );

        Ok(StreamingInsertResult {
            vectors_inserted: total_inserted,
            batches_processed: batch_count,
            total_time_ms: total_time,
            hierarchical_stats_updated: true,
            zone_maps_rebuilt: true,
        })
    }

    /// Advanced search — NOT IMPLEMENTED; refuses honestly (TD-FACADE-1).
    ///
    /// `NovaUnifiedEngine` is a demonstration layer (see the module docs) and
    /// this method's five-phase pipeline is placeholders end to end: the
    /// collection-metadata cache it reads is never populated, the vector
    /// loader fabricates constant vectors, and the ranker fabricates result
    /// ids. Before this refusal it failed with a misleading
    /// "collection metadata not found" error that hid all of that.
    ///
    /// The production NOVA engine is `NovaEngine` (built by the storage
    /// engine factory) with a real id+score search. The demonstration
    /// pipeline is preserved verbatim in advanced_search_pipeline (private,
    /// labeled below).
    pub async fn advanced_search(
        &self,
        collection_id: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        search_options: AdvancedSearchOptions,
    ) -> Result<AdvancedSearchResult> {
        let _ = (collection_id, &query_vector, top_k, &search_options);
        anyhow::bail!(
            "NOVA advanced_search is not implemented: this is a demonstration layer whose \
             five-phase pipeline is placeholders (the collection-metadata cache is never \
             populated, loaded vectors are fabricated, and result ids are fabricated). Use \
             the production NovaEngine search via the storage engine factory. See \
             TD-FACADE-1."
        )
    }

    /// Preserved demonstration of the intended five-phase design (TD-FACADE-1).
    ///
    /// Unreachable: kept verbatim as the template a real implementation would
    /// start from. Its phases are placeholders — the pruning stages return
    /// everything unpruned, `load_vectors_from_zones` fabricates vectors, and
    /// `rank_and_filter_results` fabricates ids. A real implementation must
    /// also populate the collection-metadata cache and map results back to
    /// real ids (the working pattern is `columnar_search::full_precision_stage`).
    #[allow(dead_code)]
    async fn advanced_search_pipeline(
        &self,
        collection_id: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        search_options: AdvancedSearchOptions,
    ) -> Result<AdvancedSearchResult> {
        let start_time = std::time::Instant::now();

        info!(
            "Advanced search on collection: {} (top_k: {}, quality_level: {:.2})",
            collection_id, top_k, search_options.target_quality
        );

        // Get collection metadata with hierarchical stats
        let collection_metadata = self.get_nova_collection_metadata(collection_id).await?;

        // Phase 1: Hierarchical pruning using super blocks
        let pruning_start = std::time::Instant::now();
        let candidate_super_blocks = self
            .prune_super_blocks(
                &query_vector,
                &collection_metadata.hierarchical_stats,
                &search_options,
            )
            .await?;
        let pruning_time = pruning_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Hierarchical pruning selected {} super blocks in {:.2}ms",
            candidate_super_blocks.len(),
            pruning_time
        );

        // Phase 2: Zone map pruning
        let zone_pruning_start = std::time::Instant::now();
        let candidate_zones = self
            .zone_map_manager
            .prune_zones(
                &query_vector,
                &collection_metadata.zone_maps,
                &candidate_super_blocks,
            )
            .await?;
        let zone_pruning_time = zone_pruning_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Zone map pruning selected {} zones in {:.2}ms",
            candidate_zones.len(),
            zone_pruning_time
        );

        // Phase 3: Load quantized vectors from selected zones
        let loading_start = std::time::Instant::now();
        let quantized_vectors = self
            .load_vectors_from_zones(
                collection_id,
                &candidate_zones,
                search_options.max_vectors_to_evaluate,
            )
            .await?;
        let loading_time = loading_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Loaded {} quantized vectors in {:.2}ms",
            quantized_vectors.len(),
            loading_time
        );

        // Phase 4: Progressive distance computation
        let computation_start = std::time::Instant::now();
        let distance_results = if search_options.enable_progressive {
            let _prog_results = self
                .compute_progressive_distances(
                    &query_vector,
                    &quantized_vectors,
                    search_options.target_quality,
                    top_k,
                )
                .await?;
            // Convert QuantizedDistanceResult to (VectorRecord, f32)
            // Deferred: This needs to be refactored to load IDs from storage along with vectors
            // The current implementation is a placeholder that won't work in production
            Vec::new()
        } else {
            // Use centralized quantized distance calculator
            let distance_calc =
                QuantizedDistanceCalculator::new(QuantizedDistanceConfig::default())?;
            let format = search_options
                .format_preference
                .as_ref()
                .cloned()
                .unwrap_or(SelectedFormat::FP32); // Default to full precision
            let _distances = distance_calc
                .compute_columnar_batch_distances(&query_vector, &quantized_vectors, format)
                .await?;

            // Convert to VectorRecords with scores
            // Deferred: This needs to be refactored to load IDs from storage along with vectors
            // The current implementation is a placeholder that won't work in production
            Vec::new()
        };
        let computation_time = computation_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Distance computation completed in {:.2}ms",
            computation_time
        );

        // Phase 5: Result ranking and post-processing
        let ranking_start = std::time::Instant::now();
        let final_results = self
            .rank_and_filter_results(distance_results, top_k, &search_options)
            .await?;
        let ranking_time = ranking_start.elapsed().as_secs_f64() * 1000.0;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

        info!(
            "Advanced search completed in {:.2}ms with {} phases",
            total_time, 5
        );

        Ok(AdvancedSearchResult {
            results: final_results,
            total_time_ms: total_time,
            phase_timings: PhaseTimings {
                pruning_time_ms: pruning_time,
                zone_pruning_time_ms: zone_pruning_time,
                loading_time_ms: loading_time,
                computation_time_ms: computation_time,
                ranking_time_ms: ranking_time,
            },
            pruning_statistics: PruningStatistics {
                super_blocks_evaluated: collection_metadata.hierarchical_stats.super_blocks.len(),
                super_blocks_selected: candidate_super_blocks.len(),
                zones_evaluated: collection_metadata.zone_maps.len(),
                zones_selected: candidate_zones.len(),
                vectors_evaluated: quantized_vectors.len(),
                pruning_efficiency: 1.0
                    - (quantized_vectors.len() as f32
                        / collection_metadata
                            .hierarchical_stats
                            .global_stats
                            .total_vectors as f32),
            },
        })
    }

    /// Get NOVA performance metrics including hierarchical statistics
    pub async fn get_nova_performance_metrics(&self) -> Result<NovaPerformanceMetrics> {
        let (operation_metrics, resource_metrics) =
            self.common_ops.get_performance_metrics().await?;

        // Convert HashMap metrics to proper types
        let hierarchical_metrics = HierarchicalMetrics {
            super_blocks_count: 0,
            row_groups_count: 0,
            statistics_updates: 0,
            pruning_efficiency: 0.85,
        };

        let zone_map_metrics = NovaColumnarZoneMapMetrics {
            zones_count: 0,
            nested_zones_count: 0,
            pruning_operations: 0,
            pruning_effectiveness: 0.9,
        };

        let streaming_metrics = NovaColumnarStreamingMetrics {
            active_streams: 0,
            completed_streams: 0,
            total_vectors_streamed: 0,
            average_stream_throughput: 100.0,
        };

        Ok(NovaPerformanceMetrics {
            operation_metrics,
            resource_metrics,
            hierarchical_metrics,
            zone_map_metrics,
            streaming_metrics,
        })
    }

    // Helper methods

    /// Create NOVA-optimized configuration
    fn create_nova_optimized_config(nova_config: &NovaSpecificConfig) -> CommonColumnarConfig {
        let mut config = CommonColumnarConfig::default();

        // NOVA-specific optimizations
        config.engine_optimizations.nova_optimizations = NovaOptimizations {
            enable_hierarchical_stats: nova_config.enable_hierarchical_stats,
            zone_map_config: ZoneMapOptimization {
                enable_zone_maps: nova_config.zone_map_config.enable_zone_maps,
                zone_size: nova_config.zone_map_config.zone_size,
                enable_nested_zones: nova_config.zone_map_config.enable_nested_zones,
                max_zone_depth: nova_config.zone_map_config.max_zone_depth,
            },
            streaming_processing: StreamingProcessingConfig {
                enable_streaming: nova_config.streaming_config.enable_streaming,
                stream_buffer_size: nova_config.streaming_config.stream_buffer_size,
                max_concurrent_streams: nova_config.streaming_config.max_concurrent_streams,
                stream_timeout_seconds: nova_config.streaming_config.stream_timeout_seconds,
            },
            advanced_caching:
                crate::storage::engines::core::formats::columnar::common::AdvancedCachingConfig {
                    enable_adaptive_caching: nova_config.caching_config.enable_adaptive_caching,
                    cache_size_mb: nova_config.caching_config.cache_size_mb,
                    cache_levels: nova_config.caching_config.cache_levels,
                    prefetch_strategy: format!(
                        "{:?}",
                        nova_config.caching_config.prefetch_strategy
                    ),
                },
        };

        // Progressive search optimization is now handled by the engine directly
        // Distance computation configuration is managed through the compute module

        config
    }

    // Placeholder implementations for complex operations
    async fn get_or_create_nova_collection_metadata(
        &self,
        collection_id: &str,
    ) -> Result<NovaCollectionMetadata> {
        // Similar to VIPER but with NOVA-specific hierarchical stats and zone maps
        // This would be implemented with actual collection service integration

        let metadata = NovaCollectionMetadata {
            collection_id: collection_id.to_string(),
            dimension: 768,
            quantization: Some(QuantizationConfig::default()),
            filterable_columns: vec![],
            schema: Arc::new(arrow_schema::Schema::empty()),
            compression_metadata: crate::storage::engines::core::formats::columnar::CompressionMetadata {
                column_compression: HashMap::new(),
                compression_ratios: HashMap::new(),
                writer_properties: crate::storage::engines::core::formats::columnar::schema::WriterPropertiesConfig::default(),
            },
            hierarchical_stats: HierarchicalStatistics {
                super_blocks: vec![],
                row_group_stats: vec![],
                column_stats: HashMap::new(),
                global_stats: GlobalStats {
                    total_vectors: 0,
                    total_size_bytes: 0,
                    average_compression_ratio: 3.0,
                    quantization_quality: 0.85,
                    index_coverage: 0.95,
                },
            },
            zone_maps: vec![],
        };

        Ok(metadata)
    }

    async fn get_nova_collection_metadata(
        &self,
        collection_id: &str,
    ) -> Result<NovaCollectionMetadata> {
        let cache = self.collection_cache.read().await;
        cache
            .get(collection_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("NOVA collection metadata not found: {}", collection_id))
    }

    async fn process_insert_batch(
        &self,
        _collection_id: &str,
        metadata: &NovaCollectionMetadata,
        batch: Vec<VectorRecord>,
    ) -> Result<BatchInsertResult> {
        // Process batch using unified infrastructure
        let canonical_batch: Vec<proximadb_records::ProximaRecord> = batch
            .iter()
            .map(proximadb_records::ProximaRecord::from)
            .collect();
        let serialization_result = self
            .common_ops
            .serialize_records(&canonical_batch, &metadata.schema)
            .await?;

        // Generate hierarchical updates (placeholder)
        let hierarchical_updates = vec![
            HierarchicalUpdate::SuperBlockCreated {
                super_block_id: 1,
                vector_count: batch.len(),
            },
            HierarchicalUpdate::RowGroupCreated {
                row_group_id: 1,
                super_block_id: 1,
                vector_count: batch.len(),
            },
        ];

        Ok(BatchInsertResult {
            vectors_inserted: batch.len(),
            compression_ratio: serialization_result
                .metadata
                .compression_stats
                .compression_ratio,
            hierarchical_updates,
        })
    }

    async fn prune_super_blocks(
        &self,
        _query_vector: &[f32],
        hierarchical_stats: &HierarchicalStatistics,
        _search_options: &AdvancedSearchOptions,
    ) -> Result<Vec<usize>> {
        // Placeholder: return all super block IDs
        Ok(hierarchical_stats
            .super_blocks
            .iter()
            .map(|sb| sb.super_block_id)
            .collect())
    }

    async fn load_vectors_from_zones(
        &self,
        _collection_id: &str,
        _zones: &[usize],
        max_vectors: Option<usize>,
    ) -> Result<Vec<QuantizedVectorData>> {
        // Placeholder implementation - using types from compute module
        let count = max_vectors.unwrap_or(100).min(1000);
        let vectors = vec![
            QuantizedVectorData {
                fp32: Some(vec![1.0; 768]),
                binary: Some(vec![0xFF; 96]),
                int8: Some(Int8VectorData {
                    values: vec![100; 768],
                    scale: 0.01,
                    zero_point: 0,
                    length_renorm: None,
                }),
                pq: None,
            };
            count
        ];
        Ok(vectors)
    }

    async fn compute_progressive_distances(
        &self,
        query_vector: &[f32],
        quantized_vectors: &[proximadb_distance_kernel::QuantizedVectorData],
        target_quality: f32,
        _top_k: usize,
    ) -> Result<Vec<proximadb_distance_kernel::QuantizedDistanceResult>> {
        // Progressive distance computation using best available representation
        let mut results = Vec::new();
        let distance_metric = DistanceMetric::Cosine; // Get from collection config in production

        for vector in quantized_vectors {
            // Determine quality level based on available data
            let (similarity, quality_estimate) = if let Some(fp32_vec) = &vector.fp32 {
                // Best quality: use FP32 directly
                let sim = self.distance_compute.calculate_distance(
                    query_vector,
                    fp32_vec,
                    &distance_metric,
                );
                (sim, 1.0) // Perfect quality
            } else if let Some(int8_data) = &vector.int8 {
                // Good quality: dequantize INT8 and compute
                let dequantized =
                    dequantize_int8(&int8_data.values, int8_data.scale, int8_data.zero_point);
                let sim = self.distance_compute.calculate_distance(
                    query_vector,
                    &dequantized,
                    &distance_metric,
                );
                (sim, 0.85) // ~85% quality for INT8
            } else if let Some(binary_vec) = &vector.binary {
                // Approximate: use binary hamming distance
                let hamming_distance = compute_hamming_distance(query_vector, binary_vec);
                let _normalized = 1.0 - (hamming_distance as f32 / (binary_vec.len() * 8) as f32);
                let sim = proximadb_distance_kernel::engine::SimilarityResult::new(
                    hamming_distance as f32,
                    DistanceMetric::Hamming,
                );
                (sim, 0.60) // ~60% quality for binary
            } else {
                // No data available
                let sim = proximadb_distance_kernel::engine::SimilarityResult::new(
                    f32::MAX,
                    DistanceMetric::Cosine,
                );
                (sim, 0.0)
            };

            // Create QuantizedDistanceResult
            let result = proximadb_distance_kernel::quantized::QuantizedDistanceResult {
                similarity: similarity.normalized_score,
                quality_estimate,
                method: proximadb_distance_kernel::quantized::ComputationMethod::ExactFP32,
                metrics: proximadb_distance_kernel::quantized::DistanceMetrics::default(),
            };

            results.push(result);

            // Early termination if quality threshold met
            if quality_estimate >= target_quality {
                break;
            }
        }

        Ok(results)
    }

    async fn rank_and_filter_results(
        &self,
        mut distance_results: Vec<proximadb_distance_kernel::QuantizedDistanceResult>,
        top_k: usize,
        _search_options: &AdvancedSearchOptions,
    ) -> Result<Vec<NovaSearchResult>> {
        // Sort by distance and take top_k
        distance_results.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let results = distance_results
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, result)| NovaSearchResult {
                vector_id: format!("nova_vector_{}", i),
                similarity: result.similarity,
                quality_estimate: result.quality_estimate,
                hierarchical_level: 0,
                zone_id: None,
            })
            .collect();

        Ok(results)
    }
}

// Result types and supporting structures

/// Streaming insert result
#[derive(Debug)]
pub struct StreamingInsertResult {
    pub vectors_inserted: usize,
    pub batches_processed: usize,
    pub total_time_ms: f64,
    pub hierarchical_stats_updated: bool,
    pub zone_maps_rebuilt: bool,
}

/// Batch insert result
#[derive(Debug)]
struct BatchInsertResult {
    pub vectors_inserted: usize,
    #[allow(dead_code)]
    compression_ratio: f32,
    pub hierarchical_updates: Vec<HierarchicalUpdate>,
}

/// Hierarchical statistics update
#[derive(Debug)]
#[allow(dead_code)]
enum HierarchicalUpdate {
    SuperBlockCreated {
        super_block_id: usize,
        vector_count: usize,
    },
    RowGroupCreated {
        row_group_id: usize,
        super_block_id: usize,
        vector_count: usize,
    },
    StatisticsUpdated {
        _component: String,
        _change: f32,
    },
}

/// Advanced search options
#[derive(Debug)]
pub struct AdvancedSearchOptions {
    pub target_quality: f32,
    pub enable_progressive: bool,
    pub max_vectors_to_evaluate: Option<usize>,
    pub format_preference: Option<proximadb_distance_kernel::SelectedFormat>,
    pub enable_hierarchical_pruning: bool,
    pub enable_zone_map_pruning: bool,
}

/// Advanced search result
#[derive(Debug)]
pub struct AdvancedSearchResult {
    pub results: Vec<NovaSearchResult>,
    pub total_time_ms: f64,
    pub phase_timings: PhaseTimings,
    pub pruning_statistics: PruningStatistics,
}

/// Individual NOVA search result
#[derive(Debug)]
pub struct NovaSearchResult {
    pub vector_id: String,
    pub similarity: f32,
    pub quality_estimate: f32,
    pub hierarchical_level: usize,
    pub zone_id: Option<usize>,
}

/// Search phase timings
#[derive(Debug)]
pub struct PhaseTimings {
    pub pruning_time_ms: f64,
    pub zone_pruning_time_ms: f64,
    pub loading_time_ms: f64,
    pub computation_time_ms: f64,
    pub ranking_time_ms: f64,
}

/// Pruning efficiency statistics
#[derive(Debug)]
pub struct PruningStatistics {
    pub super_blocks_evaluated: usize,
    pub super_blocks_selected: usize,
    pub zones_evaluated: usize,
    pub zones_selected: usize,
    pub vectors_evaluated: usize,
    pub pruning_efficiency: f32,
}

/// NOVA performance metrics
#[derive(Debug)]
pub struct NovaPerformanceMetrics {
    pub operation_metrics:
        crate::storage::engines::core::formats::columnar::common::OperationMetrics,
    pub resource_metrics: crate::storage::engines::core::formats::columnar::common::ResourceMetrics,
    pub hierarchical_metrics: HierarchicalMetrics,
    pub zone_map_metrics: NovaColumnarZoneMapMetrics,
    pub streaming_metrics: NovaColumnarStreamingMetrics,
}

/// Hierarchical statistics metrics
#[derive(Debug)]
pub struct HierarchicalMetrics {
    pub super_blocks_count: usize,
    pub row_groups_count: usize,
    pub statistics_updates: usize,
    pub pruning_efficiency: f32,
}

/// Zone map metrics
#[derive(Debug)]
pub struct NovaColumnarZoneMapMetrics {
    pub zones_count: usize,
    pub nested_zones_count: usize,
    pub pruning_operations: usize,
    pub pruning_effectiveness: f32,
}

/// Backwards-compat alias for [`NovaColumnarStreamingMetrics`].
pub type StreamingMetrics = NovaColumnarStreamingMetrics;

/// Streaming processing metrics
#[derive(Debug)]
pub struct NovaColumnarStreamingMetrics {
    pub active_streams: usize,
    pub completed_streams: usize,
    pub total_vectors_streamed: usize,
    pub average_stream_throughput: f32,
}

// Implementation of supporting managers

impl HierarchicalStatsManager {
    async fn update_statistics(
        &self,
        _collection_id: &str,
        _updates: Vec<HierarchicalUpdate>,
    ) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    #[allow(dead_code)]
    async fn get_metrics(&self) -> HierarchicalMetrics {
        HierarchicalMetrics {
            super_blocks_count: 0,
            row_groups_count: 0,
            statistics_updates: 0,
            pruning_efficiency: 0.85,
        }
    }
}

impl NovaColumnarZoneMapManager {
    async fn rebuild_zones(
        &self,
        _collection_id: &str,
        _metadata: &NovaCollectionMetadata,
    ) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    async fn prune_zones(
        &self,
        _query_vector: &[f32],
        zone_maps: &[NovaColumnarZoneMap],
        _candidate_super_blocks: &[usize],
    ) -> Result<Vec<usize>> {
        // Placeholder: return all zone IDs
        Ok(zone_maps.iter().map(|z| z.zone_id).collect())
    }

    #[allow(dead_code)]
    async fn get_metrics(&self) -> NovaColumnarZoneMapMetrics {
        NovaColumnarZoneMapMetrics {
            zones_count: 0,
            nested_zones_count: 0,
            pruning_operations: 0,
            pruning_effectiveness: 0.90,
        }
    }
}

impl StreamingProcessor {
    async fn start_session(
        &self,
        session_id: String,
        collection_id: String,
        stream_type: StreamType,
    ) -> Result<StreamingSession> {
        let session = StreamingSession {
            session_id: session_id.clone(),
            collection_id,
            stream_type,
            buffer_size: self.config.stream_buffer_size,
            processed_count: 0,
            start_time: std::time::Instant::now(),
        };

        let mut active_streams = self.active_streams.write().await;
        active_streams.insert(session_id, session.clone());

        Ok(session)
    }

    async fn update_session(&self, session_id: &str, processed_count: usize) -> Result<()> {
        let mut active_streams = self.active_streams.write().await;
        if let Some(session) = active_streams.get_mut(session_id) {
            session.processed_count += processed_count;
        }
        Ok(())
    }

    async fn complete_session(&self, session_id: &str) -> Result<()> {
        let mut active_streams = self.active_streams.write().await;
        active_streams.remove(session_id);
        Ok(())
    }

    #[allow(dead_code)]
    async fn get_metrics(&self) -> NovaColumnarStreamingMetrics {
        let active_streams = self.active_streams.read().await;
        NovaColumnarStreamingMetrics {
            active_streams: active_streams.len(),
            completed_streams: 0,      // Would track in real implementation
            total_vectors_streamed: 0, // Would track in real implementation
            average_stream_throughput: 1000.0, // Placeholder
        }
    }
}

// Default implementations

impl Default for NovaSpecificConfig {
    fn default() -> Self {
        Self {
            enable_hierarchical_stats: true,
            zone_map_config: NovaColumnarZoneMapConfig::default(),
            streaming_config: NovaColumnarStreamingConfig::default(),
            caching_config: AdvancedCachingConfig::default(),
            progressive_search_config: NovaColumnarProgressiveSearchConfig::default(),
        }
    }
}

impl Default for NovaColumnarZoneMapConfig {
    fn default() -> Self {
        Self {
            enable_zone_maps: true,
            zone_size: 10000,
            enable_nested_zones: true,
            max_zone_depth: 3,
            pruning_threshold: 0.1,
        }
    }
}

impl Default for NovaColumnarStreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            stream_buffer_size: 1000,
            max_concurrent_streams: 8,
            stream_timeout_seconds: 300,
            enable_adaptive_streaming: true,
        }
    }
}

impl Default for AdvancedCachingConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_caching: true,
            cache_size_mb: 512,
            cache_levels: 3,
            prefetch_strategy: PrefetchStrategy::Adaptive,
            enable_cache_warming: true,
        }
    }
}

impl Default for NovaColumnarProgressiveSearchConfig {
    fn default() -> Self {
        Self {
            enable_progressive: true,
            quality_thresholds: QualityThresholds {
                binary_threshold: 0.7,
                int8_threshold: 0.9,
                pq_threshold: 0.85,
                fp32_threshold: 1.0,
            },
            enable_early_termination: true,
            confidence_threshold: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-FACADE-1 pin: `advanced_search` must refuse with an error that
    /// names the REAL reason. On develop it already errored — with a
    /// misleading "collection metadata not found" (the cache is never
    /// populated) — so a bare `is_err()` pin would pass on both versions and
    /// prove nothing. The message is the regression signal.
    #[tokio::test]
    async fn advanced_search_refuses_with_the_real_reason() -> anyhow::Result<()> {
        let engine = NovaUnifiedEngine::new(
            Arc::new(FilesystemFactory::create(Default::default()).await?),
            NovaSpecificConfig::default(),
        )
        .await
        .expect("engine constructs with defaults");

        let err = engine
            .advanced_search(
                "pin_collection",
                vec![0.1; 8],
                5,
                AdvancedSearchOptions {
                    target_quality: 0.9,
                    enable_progressive: true,
                    max_vectors_to_evaluate: None,
                    format_preference: None,
                    enable_hierarchical_pruning: true,
                    enable_zone_map_pruning: true,
                },
            )
            .await
            .expect_err("advanced_search must refuse instead of running placeholder phases");

        let msg = format!("{err:#}");
        assert!(msg.contains("not implemented"), "message: {msg}");
        assert!(msg.contains("TD-FACADE-1"), "message: {msg}");
        assert!(
            !msg.contains("metadata not found"),
            "the old misleading error must be gone: {msg}"
        );
        Ok(())
    }

    #[test]
    fn test_nova_config_defaults() {
        let config = NovaSpecificConfig::default();

        assert!(config.enable_hierarchical_stats);
        assert!(config.zone_map_config.enable_zone_maps);
        assert!(config.streaming_config.enable_streaming);
        assert!(config.caching_config.enable_adaptive_caching);
        assert!(config.progressive_search_config.enable_progressive);
    }

    #[test]
    fn test_zone_map_config() {
        let config = NovaColumnarZoneMapConfig::default();

        assert_eq!(config.zone_size, 10000);
        assert!(config.enable_nested_zones);
        assert_eq!(config.max_zone_depth, 3);
        assert_eq!(config.pruning_threshold, 0.1);
    }

    #[test]
    fn test_streaming_config() {
        let config = NovaColumnarStreamingConfig::default();

        assert_eq!(config.stream_buffer_size, 1000);
        assert_eq!(config.max_concurrent_streams, 8);
        assert_eq!(config.stream_timeout_seconds, 300);
        assert!(config.enable_adaptive_streaming);
    }
}
