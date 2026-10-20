//! Unified Search Interface - Eliminates Adapter Fragmentation
//!
//! This module provides a single, unified interface for all search engines,
//! eliminating the need for multiple adapter layers and result type conversions.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::compute::distance_computation::DistanceMetric;
use crate::compute::distance_computation::engine::UnifiedDistanceCompute;
use crate::compute::quantization::quantization_engine::{
    UnifiedQuantizationEngine, UnifiedQuantizationLevel,
};
use crate::core::search::{OptimizedSearchRecord, SearchParams, SearchResultSet};
use proximadb_runtime::CollectionPort;

/// SearchPlan - High-level search execution plan with optimization metadata
///
/// This structure represents the planning and optimization layer for searches,
/// containing all metadata needed to make intelligent routing and optimization decisions.
///
/// # Purpose
/// - Query planning and optimization
/// - Resource allocation decisions  
/// - Quantization strategy selection
/// - Filter pushdown optimization
///
/// # Usage
/// Created by search coordinators and optimizers before executing the actual search.
#[derive(Debug, Clone)]
pub struct SearchPlan {
    /// Collection metadata for optimization
    pub collection_id: String,
    /// Collection configuration for optimization hints
    pub collection_config: Option<SearchCollectionConfig>,
    /// Filterable metadata columns from collection config
    pub filterable_columns: Vec<SearchFilterableColumn>,
    /// Available quantization methods for this collection
    pub available_quantization: Vec<UnifiedQuantizationLevel>,
    /// Storage characteristics
    pub storage_info: StorageInfo,
    /// Unified filter expression to apply during search (for row group pruning and filtering)
    /// Uses centralized sql_value_filter::evaluate_filter for type-safe SqlValue metadata filtering
    pub filter_expression: Option<crate::core::search::FilterExpression>,
    /// Query vector for similarity search (needed for quantized pre-filtering)
    pub query_vector: Option<Vec<f32>>,

    /// Number of top results to return
    pub top_k: usize,

    /// Minimum score threshold for results (dynamically updated during search)
    /// - None: No minimum threshold (useful for full fidelity searches)
    /// - Some(score): Only return results with score >= this value
    ///
    /// Engines can update this as they find better results
    pub min_score: Option<f32>,

    /// Whether to enable early termination optimizations
    /// - true: Allow early termination when sufficient high-quality results found
    /// - false: Full scan for maximum recall (full fidelity mode)
    pub enable_early_termination: bool,
}

/// Backwards-compat alias for [`SearchCollectionConfig`].
pub type CollectionConfig = SearchCollectionConfig;

/// Collection configuration for search optimization
#[derive(Debug, Clone)]
pub struct SearchCollectionConfig {
    /// Default distance metric
    pub default_distance_metric: DistanceMetric,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Enable quantization for this collection
    pub enable_quantization: bool,
    /// Enable metadata filtering
    pub enable_metadata_filtering: bool,
    /// Estimated document count
    pub estimated_document_count: usize,
}

/// Backwards-compat alias for [`SearchFilterableColumn`].
pub type FilterableColumn = SearchFilterableColumn;

/// Filterable metadata column configuration
#[derive(Debug, Clone)]
pub struct SearchFilterableColumn {
    /// Column name
    pub name: String,
    /// Column data type
    pub data_type: ColumnData,
    /// Whether this column is indexed
    pub is_indexed: bool,
    /// Estimated cardinality for optimization
    pub estimated_cardinality: Option<usize>,
}

/// Column data types for type-safe filtering
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// UTF-8 string column
    String,
    /// 64-bit integer column
    Integer,
    /// 64-bit float column
    Float,
    /// Boolean column
    Boolean,
    /// Date/time column
    DateTime,
    /// JSON column
    Json,
}

/// Storage characteristics for optimization
#[derive(Debug, Clone)]
pub struct StorageInfo {
    /// Is this cloud storage?
    pub is_cloud_storage: bool,
    /// Storage type (S3, GCS, Azure, Local)
    pub storage_type: String,
    /// Estimated total file size
    pub estimated_size_mb: f64,
    /// Number of files
    pub file_count: usize,
    /// Support for range requests
    pub supports_range_requests: bool,
    /// Actual file paths (optional, used when available to avoid filesystem queries)
    pub file_paths: Option<Vec<String>>,
}

/// Unified search interface - implemented by all engines
#[async_trait]
pub trait UnifiedSearchEngine: Send + Sync {
    /// Engine identifier for debugging and metrics
    fn engine_id(&self) -> &str;

    /// Search vectors with unified parameters and semantic results
    async fn search_unified(
        &self,
        context: &SearchPlan,
        params: &SearchParams,
        distance_compute: &UnifiedDistanceCompute,
        quantization_engine: Option<&UnifiedQuantizationEngine>,
    ) -> Result<SearchResultSet>;

    /// Check if this engine can handle the given search context
    async fn can_handle(&self, context: &SearchPlan, params: &SearchParams) -> bool;

    /// Get optimization recommendations for this engine
    async fn optimization_hints(&self, context: &SearchPlan) -> Vec<OptimizationHint>;

    /// Estimate search cost for query planning
    async fn estimate_cost(&self, context: &SearchPlan, params: &SearchParams) -> f64;
}

/// Optimization hints for search execution
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Use quantized search for large datasets
    UseQuantization {
        /// Quantization level to use
        method: UnifiedQuantizationLevel,
        /// Expected speedup factor
        expected_speedup: f32,
    },
    /// Use metadata filtering for selective queries
    UseMetadataFiltering {
        /// Estimated selectivity (0.0 to 1.0)
        selectivity_estimate: f32,
    },
    /// Use column projection for bandwidth optimization
    UseColumnProjection {
        /// Columns to project
        columns: Vec<String>,
    },
    /// Use range requests for cloud storage
    UseRangeRequests {
        /// Chunk size for range requests in megabytes
        chunk_size_mb: f32,
    },
    /// Use caching for frequently accessed data
    UseCaching {
        /// Cache key for this search pattern
        cache_key: String,
    },
}

/// Unified search orchestrator - replaces VectorOperationsService search logic
pub struct IntegratedSearchOptimizer {
    engines: Vec<Arc<dyn UnifiedSearchEngine>>,
    distance_compute: Arc<UnifiedDistanceCompute>,
    quantization_engine: Arc<UnifiedQuantizationEngine>,
    /// Collection port for metadata lookup (Task #76 migration: was
    /// `Arc<CollectionService>`, now port-typed view).
    collection_port: Arc<dyn CollectionPort>,
}

impl IntegratedSearchOptimizer {
    /// Create new orchestrator with registered engines
    pub fn new(
        distance_compute: Arc<UnifiedDistanceCompute>,
        quantization_engine: Arc<UnifiedQuantizationEngine>,
        collection_port: Arc<dyn CollectionPort>,
    ) -> Self {
        Self {
            engines: Vec::new(),
            distance_compute,
            quantization_engine,
            collection_port,
        }
    }

    /// Register a search engine
    pub fn register_engine(&mut self, engine: Arc<dyn UnifiedSearchEngine>) {
        self.engines.push(engine);
    }

    /// Execute unified search with automatic engine selection and optimization
    pub async fn search(
        &self,
        collection_id: &str,
        params: SearchParams,
    ) -> Result<SearchResultSet> {
        // 1. Build unified search context from collection configuration
        let context = self.build_search_context(collection_id, &params).await?;

        // 2. Select optimal engines based on context and params
        let selected_engines = self.select_engines(&context, &params).await?;

        // 3. Execute search across engines with unified distance computation
        let mut all_results: Vec<OptimizedSearchRecord> = Vec::new();
        let mut total_processing_time = 0u64;

        for engine in selected_engines {
            let engine_results = engine
                .search_unified(
                    &context,
                    &params,
                    &self.distance_compute,
                    Some(&self.quantization_engine),
                )
                .await?;

            // Engine results already contain OptimizedSearchRecord
            all_results.extend(engine_results.results.iter().cloned());
            total_processing_time += engine_results.processing_time_us;
        }

        // 4. Apply unified ranking using semantic distance information
        self.apply_unified_ranking(&mut all_results, &params)
            .await?;

        // 5. Return unified result set
        let total_count = all_results.len() as u64;
        Ok(SearchResultSet {
            results: all_results.into(),
            total_count,
            query_id: params
                .custom_hints
                .as_ref()
                .and_then(|h| h.get("query_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            processing_time_us: total_processing_time,
            algorithm: "IntegratedSearchOptimizer".to_string(),
            metadata: HashMap::new(),
        })
    }

    /// Build search context from collection configuration
    async fn build_search_context(
        &self,
        collection_id: &str,
        _params: &SearchParams,
    ) -> Result<SearchPlan> {
        // Get collection from port
        let collection = self
            .collection_port
            .get_collection_internal(collection_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Collection not found: {}", collection_id))?;

        // Build filterable columns from collection config
        let filterable_columns: Vec<SearchFilterableColumn> =
            collection.config.as_ref().map_or_else(Vec::new, |config| {
                config
                    .filterable_columns
                    .iter()
                    .map(|col| SearchFilterableColumn {
                        name: col.name.clone(),
                        data_type: match col.data_type {
                            1 => ColumnData::String,
                            2 => ColumnData::Integer,
                            3 => ColumnData::Float,
                            4 => ColumnData::Boolean,
                            5 => ColumnData::DateTime,
                            _ => ColumnData::Json,
                        },
                        is_indexed: col.indexed,
                        estimated_cardinality: col.estimated_cardinality.map(|c| c as usize),
                    })
                    .collect()
            });

        // Analyze storage characteristics
        let storage_info = self.analyze_storage_info(collection_id).await?;

        // Build collection config
        let collection_config = collection
            .config
            .as_ref()
            .map(|config| SearchCollectionConfig {
                default_distance_metric: config
                    .distance_metric
                    .and_then(|m| DistanceMetric::try_from(m).ok())
                    .unwrap_or(DistanceMetric::Cosine),
                vector_dimension: config.dimension as usize,
                enable_quantization: config.quantization.is_some(),
                enable_metadata_filtering: !filterable_columns.is_empty(),
                estimated_document_count: storage_info.file_count * 1000, // Rough estimate
            });

        Ok(SearchPlan {
            collection_id: collection_id.to_string(),
            collection_config,
            filterable_columns,
            available_quantization: vec![
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::pq8(
                    32,
                ),
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::pq4(
                    32,
                ),
                crate::compute::quantization::quantization_engine::UnifiedQuantizationLevel::int8(),
            ],
            storage_info,
            filter_expression: None, // Deferred: Extract from search request
            query_vector: None,      // Deferred: Extract from search request
            top_k: 100,              // Default top-k
            min_score: None,         // No minimum score by default
            enable_early_termination: true, // Enable optimizations by default
        })
    }

    /// Select optimal engines for search execution
    async fn select_engines(
        &self,
        context: &SearchPlan,
        params: &SearchParams,
    ) -> Result<Vec<Arc<dyn UnifiedSearchEngine>>> {
        let mut selected = Vec::new();

        for engine in &self.engines {
            if engine.can_handle(context, params).await {
                selected.push(engine.clone());
            }
        }

        // Sort by estimated cost for optimal execution order
        let mut costs = Vec::new();
        for engine in &selected {
            let cost = engine.estimate_cost(context, params).await;
            costs.push(cost);
        }

        let mut indices: Vec<usize> = (0..selected.len()).collect();
        indices.sort_by(|&i, &j| {
            costs[i]
                .partial_cmp(&costs[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_selected = indices.into_iter().map(|i| selected[i].clone()).collect();

        Ok(sorted_selected)
    }

    /// Apply unified ranking using semantic distance information
    async fn apply_unified_ranking(
        &self,
        results: &mut Vec<OptimizedSearchRecord>,
        params: &SearchParams,
    ) -> Result<()> {
        // Sort by score (higher = better, so reverse order)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to requested k
        if let Some(k) = params.top_k {
            results.truncate(k as usize);
        }

        // Note: Rank is not a field in SearchResult anymore
        // Rank is implicit from the position in the results vector

        Ok(())
    }

    /// Analyze storage characteristics for optimization
    async fn analyze_storage_info(&self, _collection_id: &str) -> Result<StorageInfo> {
        // This would integrate with actual storage analysis
        // For now, provide reasonable defaults
        Ok(StorageInfo {
            is_cloud_storage: true,
            storage_type: "S3".to_string(),
            estimated_size_mb: 1000.0,
            file_count: 10,
            supports_range_requests: true,
            file_paths: None,
        })
    }
}

// Removed manual From<i32> implementation - prost::Enumeration provides TryFrom<i32>
