//! Universal search components shared across all storage engines
//!
//! This module provides the common search pipeline infrastructure that all engines
//! can leverage while maintaining their specific optimizations.

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use std::sync::Arc;

use crate::core::search::OptimizedSearchRecord;
use crate::proto::proximadb_v1::VectorRecord;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use proximadb_filter_expression::FilterExpression;
use proximadb_storage_ports::QuantizationEnginePort;

/// Backwards-compat alias for [`UniversalSearchConfig`].
pub type SearchConfig = UniversalSearchConfig;

/// Configuration for the universal search pipeline
#[derive(Debug, Clone)]
pub struct UniversalSearchConfig {
    /// Number of results to return
    pub top_k: usize,

    /// Include vectors in results
    pub include_vectors: bool,

    /// Include metadata in results
    pub include_metadata: bool,

    /// Enable final reranking stage
    pub enable_reranking: bool,

    /// Maximum files to search in parallel
    pub max_parallel_files: usize,

    /// Enable progressive quantization
    pub enable_progressive_search: bool,

    /// Distance metric for similarity calculation
    pub distance_metric: proximadb_distance_kernel::DistanceMetric,
}

impl Default for UniversalSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            include_vectors: false,
            include_metadata: true,
            enable_reranking: true,
            max_parallel_files: 4,
            enable_progressive_search: true,
            distance_metric: proximadb_distance_kernel::DistanceMetric::Cosine,
        }
    }
}

/// Configuration for progressive quantization search
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Enable binary filtering stage
    pub enable_binary: bool,

    /// Threshold for binary filtering (0.0 - 1.0)
    pub binary_threshold: f32,

    /// Enable INT8 approximation stage
    pub enable_int8: bool,

    /// Number of candidates to keep after INT8
    pub int8_top_k: usize,

    /// Enable product quantization stage
    pub enable_pq: bool,

    /// Number of candidates to keep after PQ
    pub pq_top_k: usize,

    /// Final number of results
    pub final_top_k: usize,

    /// Distance metric for the full-precision rerank stage. Defaults to Cosine
    /// (the product default); callers with the collection's metric should set it
    /// so the rerank matches the collection (P1.2 — was hard-coded Cosine via
    /// `DistanceMetric::default()` = Unspecified).
    pub distance_metric: proximadb_distance_kernel::DistanceMetric,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            enable_binary: true,
            binary_threshold: 0.7,
            enable_int8: true,
            int8_top_k: 100,
            enable_pq: true,
            pq_top_k: 50,
            final_top_k: 10,
            distance_metric: proximadb_distance_kernel::DistanceMetric::Cosine,
        }
    }
}

/// Trait for searchable files (implemented by each engine)
pub trait SearchableFile: Send + Sync {
    /// Get unique identifier for the file
    fn id(&self) -> &str;

    /// Get file size in bytes
    fn size(&self) -> u64;

    /// Check if file might contain relevant data (bloom filter, metadata, etc.)
    fn might_contain(&self, filter: &Option<FilterExpression>) -> bool;
}

/// Trait for searchable blocks within files
pub trait SearchableBlock: Send + Sync {
    /// Get block identifier
    fn id(&self) -> &str;

    /// Get records in this block
    fn records(&self) -> &[VectorRecord];

    /// Check if block is relevant for the search
    fn is_relevant(&self, filter: &Option<FilterExpression>) -> bool;
}

/// Trait for engine-specific file searching
#[async_trait]
pub trait FileSearcher<F: SearchableFile, B: SearchableBlock>: Send + Sync {
    /// Search a single file
    async fn search_file(
        &self,
        file: &F,
        query_vector: &[f32],
        config: &UniversalSearchConfig,
    ) -> Result<Vec<OptimizedSearchRecord>>;

    /// Get searchable blocks from a file
    async fn get_blocks(&self, file: &F) -> Result<Vec<B>>;

    /// Search a single block
    async fn search_block(
        &self,
        block: &B,
        query_vector: &[f32],
        config: &UniversalSearchConfig,
    ) -> Result<Vec<OptimizedSearchRecord>>;
}

/// Universal search pipeline used by all engines
pub struct UniversalSearchPipeline {
    distance_compute: Arc<UnifiedDistanceCompute>,
    quantization_engine: Arc<dyn QuantizationEnginePort>,
    #[allow(dead_code)]
    filter_processor: Arc<FilterProcessor>,
    result_manager: Arc<ResultManager>,
}

impl UniversalSearchPipeline {
    pub fn new(
        distance_compute: Arc<UnifiedDistanceCompute>,
        quantization_engine: Arc<dyn QuantizationEnginePort>,
    ) -> Self {
        let dc = distance_compute.clone();
        Self {
            distance_compute,
            quantization_engine,
            filter_processor: Arc::new(FilterProcessor::new()),
            result_manager: Arc::new(ResultManager::new(dc)),
        }
    }

    /// Common search pipeline for all engines
    pub async fn search_pipeline<F, B, S>(
        &self,
        files: Vec<F>,
        query_vector: &[f32],
        config: UniversalSearchConfig,
        file_searcher: Arc<S>,
    ) -> Result<Vec<OptimizedSearchRecord>>
    where
        F: SearchableFile + Send + 'static,
        B: SearchableBlock + Send + 'static,
        S: FileSearcher<F, B> + 'static,
    {
        // 1. File-level filtering (filter is embedded in config)
        let filtered_files = self.filter_files(files, &None)?;

        if filtered_files.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Parallel file search with controlled concurrency
        let file_results = self
            .search_files_parallel(filtered_files, query_vector, &config, file_searcher)
            .await?;

        // 3. Merge and rank results
        let mut merged = self.result_manager.merge_results(file_results)?;

        // 4. Apply final ranking
        merged = self.result_manager.rank_by_distance(merged, &None)?;

        // 5. Select top-k
        merged = self.result_manager.select_top_k(merged, config.top_k)?;

        // 6. Final reranking if enabled
        if config.enable_reranking && !merged.is_empty() {
            merged = self.rerank_results(merged, query_vector, &config).await?;
        }

        // 7. Include/exclude fields as requested
        self.result_manager.apply_field_config(merged, &config)
    }

    /// Filter files based on bloom filters and metadata
    fn filter_files<F: SearchableFile>(
        &self,
        files: Vec<F>,
        filter: &Option<FilterExpression>,
    ) -> Result<Vec<F>> {
        if filter.is_none() {
            return Ok(files);
        }

        Ok(files
            .into_iter()
            .filter(|f| f.might_contain(filter))
            .collect())
    }

    /// Search files in parallel with controlled concurrency
    async fn search_files_parallel<F, B, S>(
        &self,
        files: Vec<F>,
        query_vector: &[f32],
        config: &UniversalSearchConfig,
        file_searcher: Arc<S>,
    ) -> Result<Vec<Vec<OptimizedSearchRecord>>>
    where
        F: SearchableFile + Send + 'static,
        B: SearchableBlock + Send + 'static,
        S: FileSearcher<F, B> + 'static,
    {
        let query_vector = query_vector.to_vec();
        let config = config.clone();
        let max_parallel = config.max_parallel_files;

        // Create futures for parallel search
        let search_futures = files.into_iter().map(move |file| {
            let searcher = file_searcher.clone();
            let query = query_vector.clone();
            let cfg = config.clone();

            async move { searcher.search_file(&file, &query, &cfg).await }
        });

        // Execute with controlled parallelism
        let results: Vec<Result<Vec<OptimizedSearchRecord>>> = stream::iter(search_futures)
            .buffer_unordered(max_parallel)
            .collect()
            .await;

        // Collect successful results
        results.into_iter().collect()
    }

    /// Rerank top results with full precision
    async fn rerank_results(
        &self,
        candidates: Vec<OptimizedSearchRecord>,
        query_vector: &[f32],
        config: &UniversalSearchConfig,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        if !config.include_vectors {
            // Can't rerank without vectors
            return Ok(candidates);
        }

        // Recompute distances with full precision using batch method
        let mut reranked = Vec::with_capacity(candidates.len());

        // Collect vectors for batch processing
        let mut vectors_to_rerank = Vec::new();
        let mut indices_with_vectors = Vec::new();

        for (idx, result) in candidates.iter().enumerate() {
            if let Some(ref vector) = result.vector {
                vectors_to_rerank.push(vector.as_slice());
                indices_with_vectors.push(idx);
            }
        }

        // Batch compute distances if we have vectors
        let distances = if !vectors_to_rerank.is_empty() {
            self.distance_compute.as_ref().batch_distance_pooled_simd(
                query_vector,
                &vectors_to_rerank,
                &config.distance_metric,
            )
        } else {
            Vec::new()
        };

        // Apply distances back to results
        let mut distance_idx = 0;
        for (idx, mut result) in candidates.into_iter().enumerate() {
            if indices_with_vectors.contains(&idx) {
                result.similarity = Some(distances[distance_idx].distance);
                distance_idx += 1;
            }
            reranked.push(result);
        }

        // Sort by distance
        reranked.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(reranked)
    }

    /// Progressive quantization search implementation
    pub async fn progressive_search(
        &self,
        records: Vec<VectorRecord>,
        query_vector: &[f32],
        config: &ProgressiveConfig,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let mut candidates = records;

        // Stage 1: Binary filtering
        if config.enable_binary && !candidates.is_empty() {
            candidates = self
                .binary_filter(candidates, query_vector, config.binary_threshold)
                .await?;
        }

        // Stage 2: INT8 approximation
        if config.enable_int8 && !candidates.is_empty() {
            candidates = self
                .int8_rank(candidates, query_vector, config.int8_top_k)
                .await?;
        }

        // Stage 3: PQ ranking
        if config.enable_pq && !candidates.is_empty() {
            candidates = self
                .pq_rank(candidates, query_vector, config.pq_top_k)
                .await?;
        }

        // Stage 4: Full precision ranking
        self.full_precision_rank(
            candidates,
            query_vector,
            config.final_top_k,
            &config.distance_metric,
        )
        .await
    }

    /// Binary filtering stage
    async fn binary_filter(
        &self,
        records: Vec<VectorRecord>,
        query_vector: &[f32],
        _threshold: f32,
    ) -> Result<Vec<VectorRecord>> {
        // Use the quantization engine for binary filtering via the port's
        // level-specific method. The generic level-based quantize() is intentionally
        // NOT on QuantizationEnginePort (it would drag the modality
        // UnifiedQuantizationLevel type across the storage seam). The result is
        // currently unused — binary filtering is a passthrough (see the loop below).
        let _binary_query = self.quantization_engine.quantize_to_binary(query_vector)?;

        let mut filtered = Vec::new();
        for record in records {
            // quantized_vector removed - internalized in storage
            // For now, include all records without binary filtering
            filtered.push(record);
        }

        Ok(filtered)
    }

    /// INT8 approximation stage
    async fn int8_rank(
        &self,
        records: Vec<VectorRecord>,
        _query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<VectorRecord>> {
        // This would use INT8 quantization from the quantization engine
        // For now, return top_k records
        let mut candidates = records;
        candidates.truncate(top_k.min(candidates.len()));
        Ok(candidates)
    }

    /// Product quantization ranking stage
    async fn pq_rank(
        &self,
        records: Vec<VectorRecord>,
        _query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<VectorRecord>> {
        // This would use PQ from the quantization engine
        // For now, return top_k records
        let mut candidates = records;
        candidates.truncate(top_k.min(candidates.len()));
        Ok(candidates)
    }

    /// Full precision final ranking
    async fn full_precision_rank(
        &self,
        records: Vec<VectorRecord>,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: &proximadb_distance_kernel::DistanceMetric,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let mut results = Vec::with_capacity(records.len());

        for record in records {
            let similarity_result = self.distance_compute.as_ref().calculate_distance(
                query_vector,
                &record.vector,
                distance_metric,
            );

            // Convert metadata directly to the typed map — no serde_json
            // round-trip (the JSON detour decoded JSONB per candidate only to
            // drop object/array payloads on the floor at the back-conversion).
            // Non-JSON legacy shapes keep their pre-existing placeholders and
            // unset-oneof keys stay absent. One deliberate delta vs the old
            // detour: Int64Value passes through as Int64Value (the detour
            // coerced it to NumberValue(f64), losing precision above 2^53) —
            // closer to the stored truth. NumberValue passes through
            // UNCHANGED (round 6 corrected an earlier claim here); non-finite
            // floats render as null only later, at the serde/JSON boundary —
            // pre-existing serde behavior, not this rewrite's.
            // NOTE: this pipeline currently has no live constructor, so the
            // delta is latent until it is wired.
            let mut typed_metadata_map = std::collections::HashMap::new();
            for (key, value) in record.metadata {
                use crate::proto::proximadb_v1::{self as proximadb_v1};
                let sql_value = match value.value {
                    Some(proximadb_v1::sql_value::Value::StringValue(s)) => {
                        proximadb_v1::SqlValue {
                            value: Some(proximadb_v1::sql_value::Value::StringValue(s)),
                        }
                    }
                    Some(proximadb_v1::sql_value::Value::NumberValue(f)) => {
                        proximadb_v1::SqlValue {
                            value: Some(proximadb_v1::sql_value::Value::NumberValue(f)),
                        }
                    }
                    Some(proximadb_v1::sql_value::Value::BoolValue(b)) => proximadb_v1::SqlValue {
                        value: Some(proximadb_v1::sql_value::Value::BoolValue(b)),
                    },
                    Some(proximadb_v1::sql_value::Value::Int64Value(i)) => proximadb_v1::SqlValue {
                        value: Some(proximadb_v1::sql_value::Value::Int64Value(i)),
                    },
                    // TD-PROTO-2: pass the declared variant through untouched.
                    Some(proximadb_v1::sql_value::Value::JsonbValue(bytes)) => {
                        proximadb_v1::SqlValue {
                            value: Some(proximadb_v1::sql_value::Value::JsonbValue(bytes)),
                        }
                    }
                    Some(proximadb_v1::sql_value::Value::BytesValue(_)) => proximadb_v1::SqlValue {
                        value: Some(proximadb_v1::sql_value::Value::StringValue(
                            "[binary data]".to_string(),
                        )),
                    },
                    Some(proximadb_v1::sql_value::Value::ArrayValue(_)) => proximadb_v1::SqlValue {
                        value: Some(proximadb_v1::sql_value::Value::StringValue(
                            "[array]".to_string(),
                        )),
                    },
                    Some(proximadb_v1::sql_value::Value::ObjectValue(_)) => {
                        proximadb_v1::SqlValue {
                            value: Some(proximadb_v1::sql_value::Value::StringValue(
                                "[object]".to_string(),
                            )),
                        }
                    }
                    // NullValue and unset oneof: the legacy path kept the key
                    // with an unset oneof for NullValue, and dropped keys
                    // whose oneof was entirely unset.
                    Some(proximadb_v1::sql_value::Value::NullValue(_)) => {
                        proximadb_v1::SqlValue { value: None }
                    }
                    None => continue,
                };
                typed_metadata_map.insert(key, sql_value);
            }
            results.push(
                OptimizedSearchRecord::new(record.id.clone(), similarity_result.normalized_score)
                    .with_similarity(similarity_result.normalized_score)
                    .add_vector(record.vector)
                    .with_metadata(typed_metadata_map)
                    .with_version_info(record.version.unwrap_or(0), record.timestamp.unwrap_or(0)),
            );
        }

        // Sort by score (higher score = better result)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k.min(results.len()));
        Ok(results)
    }
}

/// Filter processor for handling filter expressions
pub struct FilterProcessor {
    // Could add metadata index here for optimization
}

impl FilterProcessor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_filter(&self, _filter: &FilterExpression) -> Result<FilterPlan> {
        // Convert filter expression to execution plan
        Ok(FilterPlan {
            // Implementation details
        })
    }

    pub fn apply_to_metadata(
        &self,
        metadata: &Option<serde_json::Value>,
        _filter: &FilterExpression,
    ) -> bool {
        // Apply filter to metadata
        // This is a simplified implementation
        if metadata.is_none() {
            return false;
        }

        // Actual filter logic would go here
        true
    }

    pub fn optimize_filter(&self, filter: FilterExpression) -> FilterExpression {
        // Optimize filter expression (e.g., push down predicates)
        filter
    }
}

impl Default for FilterProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution plan for filters
pub struct FilterPlan {
    // Implementation details
}

/// Result manager for handling search results
pub struct ResultManager {
    #[allow(dead_code)]
    distance_compute: Arc<UnifiedDistanceCompute>,
}

impl ResultManager {
    pub fn new(distance_compute: Arc<UnifiedDistanceCompute>) -> Self {
        Self { distance_compute }
    }

    /// Merge multiple result sets
    pub fn merge_results(
        &self,
        results: Vec<Vec<OptimizedSearchRecord>>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let mut merged = Vec::new();
        for result_set in results {
            merged.extend(result_set);
        }
        Ok(merged)
    }

    /// Rank results by distance
    pub fn rank_by_distance(
        &self,
        mut results: Vec<OptimizedSearchRecord>,
        _distance_metric: &Option<()>,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        results.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Select top-k results
    pub fn select_top_k(
        &self,
        mut results: Vec<OptimizedSearchRecord>,
        k: usize,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        results.truncate(k.min(results.len()));
        Ok(results)
    }

    /// Apply field configuration to results
    pub fn apply_field_config(
        &self,
        mut results: Vec<OptimizedSearchRecord>,
        config: &UniversalSearchConfig,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        for result in &mut results {
            if !config.include_vectors {
                result.vector = None;
            }
            if !config.include_metadata {
                result.metadata.clear();
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_config_default() {
        let config = UniversalSearchConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.max_parallel_files, 4);
        assert!(config.enable_progressive_search);
    }

    #[test]
    fn test_progressive_config_default() {
        let config = ProgressiveConfig::default();
        assert!(config.enable_binary);
        assert_eq!(config.final_top_k, 10);
    }
}
