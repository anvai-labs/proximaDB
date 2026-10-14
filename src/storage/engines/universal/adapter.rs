//! Universal Distance Adapter - Core Implementation
//!
//! This module provides the main implementation of the universal distance adapter
//! that integrates PQ and INT8 optimized distance computations across all storage engines.

use anyhow::Result;
use proximadb_kernel::uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

use crate::proto::proximadb_v1::VectorRecord;
use proximadb_distance_kernel::{
    DistanceMetric, Int8VectorData, PQVectorData, QuantizedDistanceResult, QuantizedVectorData,
    SelectedFormat, SimilarityResult, UnifiedDistanceCompute,
};
use proximadb_hardware_caps::HardwareCapabilities;

use super::{
    config::ProgressiveRefinementConfig as ConfigProgressiveRefinementConfig,
    config::UniversalAdapterConfig,
    conversion::{FormatConverter, StorageFormat},
    distance_cache::DistanceTableCache,
    progressive_refinement::{
        ProgressiveRefinementConfig, ProgressiveRefinementPipeline, QualityMetrics, RefinementStage,
    },
    quantized_calculator::UniversalQuantizedCalculator,
    storage_integration::{EngineType, StorageEngineAdapter},
};

// Hardware acceleration types
#[derive(Debug, Clone)]
pub struct HardwareAccelerationManager {
    #[allow(dead_code)]
    capabilities: HardwareCapabilities,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationStrategy {
    BalancedMode,
    SpeedOptimized,
    QualityOptimized,
}

impl HardwareAccelerationManager {
    pub fn new(capabilities: HardwareCapabilities) -> Self {
        Self { capabilities }
    }

    pub async fn get_statistics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }

    pub fn select_strategy(&self, _format: &StorageFormat) -> OptimizationStrategy {
        OptimizationStrategy::BalancedMode
    }
}

/// Main universal distance adapter that provides unified interface for all storage engines
#[derive(Debug)]
pub struct UniversalDistanceAdapter {
    /// Configuration for the adapter
    config: UniversalAdapterConfig,

    /// Unified distance computation engine
    #[allow(dead_code)]
    distance_engine: Arc<UnifiedDistanceCompute>,

    /// Quantized distance calculator
    quantized_calculator: Arc<UniversalQuantizedCalculator>,

    /// Progressive refinement pipeline
    refinement_pipeline: Arc<ProgressiveRefinementPipeline>,

    /// Hardware acceleration manager
    hardware_manager: Arc<HardwareAccelerationManager>,

    /// Format converter for storage format conversions
    format_converter: Arc<FormatConverter>,

    /// Storage engine adapters
    engine_adapters: Arc<RwLock<HashMap<EngineType, Arc<dyn StorageEngineAdapter>>>>,

    /// Distance table cache for PQ operations
    distance_cache: Arc<DistanceTableCache>,

    /// Hardware capabilities
    hardware_capabilities: HardwareCapabilities,
}

/// Request for distance computation
#[derive(Debug, Clone)]
pub struct DistanceComputationRequest {
    /// Query vector
    pub query_vector: Vec<f32>,

    /// Candidate vectors to compare against
    pub candidates: Vec<CandidateVector>,

    /// Distance metric to use
    pub distance_metric: DistanceMetric,

    /// Storage format of the candidate vectors
    pub storage_format: StorageFormat,

    /// Progressive refinement configuration
    pub refinement_config: Option<ConfigProgressiveRefinementConfig>,

    /// Maximum number of results to return
    pub max_results: usize,

    /// Enable hardware acceleration
    pub enable_acceleration: bool,

    /// Target quality threshold (0.0-1.0)
    pub quality_threshold: Option<f32>,

    /// Collection ID for context
    pub collection_id: Uuid,

    /// Engine type for optimization
    pub engine_type: EngineType,
}

/// Candidate vector with metadata
#[derive(Debug, Clone)]
pub struct CandidateVector {
    /// Vector ID
    pub id: Uuid,

    /// Vector data in storage format
    pub data: Vec<u8>,

    /// Original vector (if available for final refinement)
    pub original_vector: Option<Vec<f32>>,

    /// Metadata for the vector
    pub metadata: Option<HashMap<String, String>>,

    /// Quality score if available from previous stages
    pub quality_score: Option<f32>,
}

/// Result of distance computation
#[derive(Debug, Clone)]
pub struct DistanceComputationResult {
    /// Computed similarity results
    pub results: Vec<SimilarityResult>,

    /// Vector IDs corresponding to results
    pub vector_ids: Vec<Uuid>,

    /// Quality metrics for the computation
    pub quality_metrics: QualityMetrics,

    /// Performance metrics
    pub performance_metrics: UniversalAdapterPerformanceMetrics,

    /// Stages used in progressive refinement
    pub refinement_stages: Vec<RefinementStage>,

    /// Final stage that produced the results
    pub final_stage: RefinementStage,

    /// Cache hit information
    pub cache_hits: usize,

    /// Total computation time in microseconds
    pub computation_time_us: u64,
}

/// Backwards-compat alias for [`UniversalAdapterPerformanceMetrics`].
pub type PerformanceMetrics = UniversalAdapterPerformanceMetrics;

/// Performance metrics for distance computation
#[derive(Debug, Clone)]
pub struct UniversalAdapterPerformanceMetrics {
    /// Total computation time in microseconds
    pub total_time_us: u64,

    /// Time spent in each refinement stage
    pub stage_times_us: HashMap<RefinementStage, u64>,

    /// Number of distance calculations performed
    pub distance_calculations: usize,

    /// Number of vectors processed at each stage
    pub vectors_per_stage: HashMap<RefinementStage, usize>,

    /// Hardware acceleration usage
    pub acceleration_used: Option<OptimizationStrategy>,

    /// Memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Cache performance
    pub cache_hit_rate: f32,
}

/// Error types for the universal adapter
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Hardware acceleration error: {0}")]
    HardwareAcceleration(String),

    #[error("Format conversion error: {0}")]
    FormatConversion(String),

    #[error("Distance computation error: {0}")]
    DistanceComputation(String),

    #[error("Progressive refinement error: {0}")]
    ProgressiveRefinement(String),

    #[error("Storage engine integration error: {0}")]
    StorageEngineIntegration(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for adapter operations
pub type AdapterResult<T> = Result<T, AdapterError>;

impl UniversalDistanceAdapter {
    /// Convert from config module's ProgressiveRefinementConfig to progressive_refinement module's
    fn convert_refinement_config(
        &self,
        config: &ConfigProgressiveRefinementConfig,
    ) -> ProgressiveRefinementConfig {
        use super::progressive_refinement::RefinementStrategy;

        ProgressiveRefinementConfig {
            search_strategy: RefinementStrategy::Sequential, // Default strategy
            candidates_per_stage: config.candidates_per_stage.clone(),
            quality_thresholds: config.quality_thresholds.clone(),
            enable_parallel_processing: config.enable_parallel_processing,
            max_memory_usage_mb: config.max_memory_usage_mb,
            enable_stage_skipping: config.enable_stage_skipping,
            min_improvement_threshold: config.min_improvement_threshold,
        }
    }

    /// Create a new universal distance adapter
    pub async fn new() -> AdapterResult<Self> {
        Self::with_config(UniversalAdapterConfig::default()).await
    }

    /// Create a new universal distance adapter with custom configuration
    pub async fn with_config(config: UniversalAdapterConfig) -> AdapterResult<Self> {
        info!(
            "Initializing Universal Distance Adapter v{}",
            super::UNIVERSAL_ADAPTER_VERSION
        );

        // Initialize hardware capabilities
        let hardware_capabilities = HardwareCapabilities::detect_with_config(
            crate::core::config::HardwareConfig::default(),
        )
        .map_err(|e| {
            AdapterError::HardwareAcceleration(format!("Failed to detect hardware: {}", e))
        })?;

        // Initialize hardware acceleration manager
        let hardware_manager = Arc::new(HardwareAccelerationManager::new(
            hardware_capabilities.clone(),
        ));

        // Initialize unified distance compute engine
        let distance_engine = Arc::new(UnifiedDistanceCompute::default());

        // Initialize quantized distance calculator
        let quantized_calculator = Arc::new(
            UniversalQuantizedCalculator::new(&config, &hardware_capabilities)
                .await
                .map_err(|e| AdapterError::Configuration(e.to_string()))?,
        );

        // Initialize progressive refinement pipeline
        let refinement_pipeline = Arc::new(
            ProgressiveRefinementPipeline::new(
                &config.refinement_stages,
                quantized_calculator.clone(),
                distance_engine.clone(),
            )
            .await?,
        );

        // Initialize format converter
        let format_converter = Arc::new(
            FormatConverter::new()
                .await
                .map_err(|e| AdapterError::FormatConversion(e.to_string()))?,
        );

        // Initialize distance table cache
        let cache_orchestrator =
            Arc::new(crate::storage::cache::orchestrator::CrossCacheOrchestrator::new(1000));
        let distance_cache = Arc::new(
            DistanceTableCache::new(&config.cache_config, cache_orchestrator)
                .await
                .map_err(|e| AdapterError::Cache(e.to_string()))?,
        );

        // Initialize storage engine adapters
        let engine_adapters = Arc::new(RwLock::new(HashMap::new()));

        let adapter = Self {
            config,
            distance_engine,
            quantized_calculator,
            refinement_pipeline,
            hardware_manager,
            format_converter,
            engine_adapters,
            distance_cache,
            hardware_capabilities: hardware_capabilities.clone(),
        };

        // Initialize storage engine adapters
        adapter.initialize_engine_adapters().await?;

        info!("Universal Distance Adapter initialized successfully");
        debug!("Hardware capabilities: {:?}", adapter.hardware_capabilities);

        Ok(adapter)
    }

    /// Compute distance with automatic format detection and progressive refinement
    pub async fn compute_progressive_distance(
        &self,
        request: DistanceComputationRequest,
    ) -> AdapterResult<DistanceComputationResult> {
        let start_time = std::time::Instant::now();

        trace!(
            "Starting progressive distance computation for {} candidates",
            request.candidates.len()
        );

        // Validate request
        self.validate_request(&request)?;

        // Get storage engine adapter
        let _engine_adapter = self.get_engine_adapter(&request.engine_type).await?;

        // Convert storage format if needed
        let converted_candidates = self
            .convert_candidates_if_needed(&request.candidates, &request.storage_format)
            .await?;

        // Execute progressive refinement pipeline
        let refinement_config = if let Some(config) = request.refinement_config {
            self.convert_refinement_config(&config)
        } else {
            self.convert_refinement_config(&self.config.progressive_refinement)
        };

        let refinement_result = self
            .refinement_pipeline
            .execute_progressive_search(
                &request.query_vector,
                &converted_candidates,
                &request.distance_metric,
                &refinement_config,
                request.max_results,
            )
            .await
            .map_err(|e| {
                AdapterError::ProgressiveRefinement(format!("Refinement failed: {}", e))
            })?;

        // Prepare final result
        let computation_time_us = start_time.elapsed().as_micros() as u64;

        let result = DistanceComputationResult {
            results: refinement_result.similarity_results,
            vector_ids: refinement_result.vector_ids,
            quality_metrics: refinement_result.quality_metrics,
            performance_metrics: UniversalAdapterPerformanceMetrics {
                total_time_us: computation_time_us,
                stage_times_us: refinement_result.stage_times,
                distance_calculations: refinement_result.total_distance_calculations,
                vectors_per_stage: refinement_result.vectors_per_stage,
                acceleration_used: refinement_result.acceleration_used,
                memory_usage_bytes: refinement_result.memory_usage_bytes,
                cache_hit_rate: refinement_result.cache_hit_rate,
            },
            refinement_stages: refinement_result.stages_used,
            final_stage: refinement_result.final_stage,
            cache_hits: refinement_result.cache_hits,
            computation_time_us,
        };

        debug!(
            "Progressive distance computation completed in {}μs",
            computation_time_us
        );

        Ok(result)
    }

    /// Compute distance using specific quantization format
    pub async fn compute_quantized_distance(
        &self,
        query_vector: &[f32],
        candidates: &[CandidateVector],
        distance_metric: &DistanceMetric,
        quantization_format: &SelectedFormat,
    ) -> AdapterResult<Vec<QuantizedDistanceResult>> {
        trace!(
            "Computing quantized distance for {} candidates",
            candidates.len()
        );

        // Convert candidates to quantized format
        let quantized_candidates = self
            .convert_to_quantized_format(candidates, quantization_format)
            .await?;

        // Compute distances using quantized calculator
        let results = self
            .quantized_calculator
            .compute_distances(
                query_vector,
                &quantized_candidates,
                distance_metric,
                quantization_format,
            )
            .await
            .map_err(|e| {
                AdapterError::DistanceComputation(format!(
                    "Quantized distance computation failed: {}",
                    e
                ))
            })?;

        Ok(results)
    }

    /// Get supported storage formats for an engine
    pub async fn get_supported_formats(
        &self,
        engine_type: &EngineType,
    ) -> AdapterResult<Vec<StorageFormat>> {
        let adapter = self.get_engine_adapter(engine_type).await?;
        Ok(adapter.supported_formats())
    }

    /// Get optimal storage format for given parameters
    pub async fn get_optimal_format(
        &self,
        engine_type: &EngineType,
        vector_dimension: usize,
        dataset_size: usize,
        target_recall: f32,
    ) -> AdapterResult<StorageFormat> {
        let adapter = self.get_engine_adapter(engine_type).await?;
        adapter
            .optimal_format(vector_dimension, dataset_size, target_recall)
            .await
    }

    /// Warm up caches for a collection
    pub async fn warm_cache(
        &self,
        collection_id: Uuid,
        engine_type: &EngineType,
        sample_vectors: &[VectorRecord],
    ) -> AdapterResult<()> {
        info!(
            "Warming cache for collection {} with {} sample vectors",
            collection_id,
            sample_vectors.len()
        );

        let adapter = self.get_engine_adapter(engine_type).await?;
        adapter
            .warm_cache(collection_id, sample_vectors)
            .await
            .map_err(|e| AdapterError::Cache(format!("Cache warming failed: {}", e)))?;

        Ok(())
    }

    /// Get adapter statistics
    pub async fn get_statistics(&self) -> AdapterResult<AdapterStatistics> {
        let cache_stats = self.distance_cache.get_statistics().await;
        let hardware_stats = self.hardware_manager.get_statistics().await;

        Ok(AdapterStatistics {
            cache_hit_rate: cache_stats.hit_rate_percent,
            cache_size_mb: cache_stats.size_mb,
            total_computations: cache_stats.total_requests,
            hardware_acceleration_usage: hardware_stats
                .get("acceleration_usage_rate")
                .copied()
                .unwrap_or(0.0) as f32,
            supported_engines: self.get_supported_engines().await,
            average_computation_time_us: hardware_stats
                .get("average_operation_time_us")
                .copied()
                .unwrap_or(0.0) as u64,
        })
    }

    // Private helper methods

    async fn initialize_engine_adapters(&self) -> AdapterResult<()> {
        let mut adapters = self.engine_adapters.write().await;

        // Initialize each storage engine adapter
        for engine_config in &self.config.storage_engines {
            let adapter: Arc<dyn StorageEngineAdapter> = match engine_config.engine_type {
                EngineType::PRISM => {
                    Arc::new(super::storage_integration::PRISMAdapter::new(engine_config).await?)
                }
                EngineType::NOVA => {
                    Arc::new(super::storage_integration::NOVAAdapter::new(engine_config).await?)
                }
                EngineType::SWIFT => {
                    Arc::new(super::storage_integration::SWIFTAdapter::new(engine_config).await?)
                }
                EngineType::VIPER => {
                    Arc::new(super::storage_integration::VIPERAdapter::new(engine_config).await?)
                }
                EngineType::SST => {
                    Arc::new(super::storage_integration::SSTAdapter::new(engine_config).await?)
                }
            };

            adapters.insert(engine_config.engine_type, adapter);
        }

        Ok(())
    }

    async fn get_engine_adapter(
        &self,
        engine_type: &EngineType,
    ) -> AdapterResult<Arc<dyn StorageEngineAdapter>> {
        let adapters = self.engine_adapters.read().await;
        adapters.get(engine_type).cloned().ok_or_else(|| {
            AdapterError::StorageEngineIntegration(format!(
                "No adapter found for engine type: {:?}",
                engine_type
            ))
        })
    }

    fn validate_request(&self, request: &DistanceComputationRequest) -> AdapterResult<()> {
        if request.query_vector.is_empty() {
            return Err(AdapterError::Configuration(
                "Query vector cannot be empty".to_string(),
            ));
        }

        if request.candidates.is_empty() {
            return Err(AdapterError::Configuration(
                "Candidates cannot be empty".to_string(),
            ));
        }

        if request.max_results == 0 {
            return Err(AdapterError::Configuration(
                "Max results must be greater than 0".to_string(),
            ));
        }

        if let Some(threshold) = request.quality_threshold
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(AdapterError::Configuration(
                "Quality threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    async fn convert_candidates_if_needed(
        &self,
        candidates: &[CandidateVector],
        _storage_format: &StorageFormat,
    ) -> AdapterResult<Vec<CandidateVector>> {
        // For now, return candidates as-is
        // Deferred: Implement format conversion logic
        Ok(candidates.to_vec())
    }

    async fn convert_to_quantized_format(
        &self,
        candidates: &[CandidateVector],
        quantization_format: &SelectedFormat,
    ) -> AdapterResult<Vec<QuantizedVectorData>> {
        let mut quantized_candidates = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let quantized_data = match quantization_format {
                SelectedFormat::INT8 => {
                    // Convert to INT8 format
                    let int8_data = self
                        .format_converter
                        .to_int8(&candidate.data)
                        .await
                        .map_err(|e| {
                            AdapterError::FormatConversion(format!("INT8 conversion failed: {}", e))
                        })?;
                    QuantizedVectorData {
                        fp32: None,
                        binary: None,
                        int8: Some(Int8VectorData {
                            values: int8_data,
                            scale: 1.0,    // Default scale
                            zero_point: 0, // Default zero point
                            length_renorm: None,
                        }),
                        pq: None,
                    }
                }
                SelectedFormat::PQ => {
                    // Convert to PQ format - assuming default segments/bits
                    let pq_data = self
                        .format_converter
                        .to_pq(&candidate.data, 16, 8)
                        .await
                        .map_err(|e| {
                            AdapterError::FormatConversion(format!("PQ conversion failed: {}", e))
                        })?;
                    QuantizedVectorData {
                        fp32: None,
                        binary: None,
                        int8: None,
                        pq: Some(PQVectorData {
                            codes: pq_data,
                            codebook: vec![], // Would need actual codebook
                            codebook_hash: 0, // Placeholder
                            length_renorm: None,
                        }),
                    }
                }
                SelectedFormat::Binary => {
                    // Convert to binary format
                    let binary_data = self
                        .format_converter
                        .to_binary(&candidate.data)
                        .await
                        .map_err(|e| {
                            AdapterError::FormatConversion(format!(
                                "Binary conversion failed: {}",
                                e
                            ))
                        })?;
                    QuantizedVectorData {
                        fp32: None,
                        binary: Some(binary_data),
                        int8: None,
                        pq: None,
                    }
                }
                SelectedFormat::FP32 => {
                    // Keep as FP32
                    // For FP32, convert bytes to Vec<f32>
                    let fp32_data = if let Some(ref original) = candidate.original_vector {
                        original.clone()
                    } else {
                        // Convert from bytes to f32
                        candidate
                            .data
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|chunk| f32::from_le_bytes(*chunk))
                            .collect()
                    };
                    QuantizedVectorData {
                        fp32: Some(fp32_data),
                        binary: None,
                        int8: None,
                        pq: None,
                    }
                }
            };

            quantized_candidates.push(quantized_data);
        }

        Ok(quantized_candidates)
    }

    async fn get_supported_engines(&self) -> Vec<EngineType> {
        let adapters = self.engine_adapters.read().await;
        adapters.keys().cloned().collect()
    }
}

/// Statistics for the universal adapter
#[derive(Debug, Clone)]
pub struct AdapterStatistics {
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,

    /// Current cache size in MB
    pub cache_size_mb: usize,

    /// Total number of computations performed
    pub total_computations: u64,

    /// Hardware acceleration usage rate (0.0-1.0)
    pub hardware_acceleration_usage: f32,

    /// List of supported storage engines
    pub supported_engines: Vec<EngineType>,

    /// Average computation time in microseconds
    pub average_computation_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = UniversalDistanceAdapter::new().await;
        assert!(adapter.is_ok());
    }

    // Additional tests will be implemented in the tests module
}
