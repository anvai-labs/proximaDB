//! Unified Columnar Optimization Module
//!
//! Provides shared optimizations for both VIPER and NOVA engines:
//! - Parquet bloom filter optimization
//! - Streaming row group access
//! - Progressive search with quantization
//! - Cost-based query optimization

use crate::storage::engines::core::formats::columnar::{
    ColumnarConfig, MetadataFilter, RowGroupStats, SearchCandidate,
};
use crate::storage::persistence::filesystem::FilesystemFactory;
use anyhow::Result;
use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::bloom_filter::Sbbf as BloomFilter;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use proximadb_distance_kernel::{DistanceMetric, engine::UnifiedDistanceCompute};
use proximadb_records::{EmbeddingCell, ProximaRecord};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};
/// Unified columnar optimization engine
pub struct ColumnarOptimizer {
    /// Distance computation engine
    distance_compute: Arc<UnifiedDistanceCompute>,

    /// Configuration
    config: ColumnarConfig,

    /// Filesystem factory (temporarily replacing zero_copy_fs)
    zero_copy_fs: Arc<FilesystemFactory>,

    /// Filesystem factory for writes (selects based on URL scheme)
    #[allow(dead_code)]
    filesystem_factory: Arc<FilesystemFactory>,

    /// Cached bloom filters per file
    bloom_filter_cache: parking_lot::RwLock<HashMap<String, Arc<FileBloomFilters>>>,
    /// Row group statistics cache
    stats_cache: parking_lot::RwLock<HashMap<String, Arc<Vec<RowGroupStats>>>>,
}
/// Bloom filters for a Parquet file
#[derive(Debug)]
pub struct FileBloomFilters {
    pub file_path: String,
    pub filters: HashMap<String, RowGroupBloomFilters>,
    pub total_size_bytes: usize,
    pub false_positive_rate: f64,
}
/// Bloom filters for a single row group
#[derive(Debug)]
pub struct RowGroupBloomFilters {
    #[allow(dead_code)]
    pub row_group_id: usize,
    pub column_filters: HashMap<String, BloomFilterInfo>,
}
/// Bloom filter information
#[derive(Debug)]
pub struct BloomFilterInfo {
    pub field: String,
    pub size_bytes: usize,
    pub hash_functions: u32,
    pub num_items: u64,
}
/// Streaming row group iterator
pub struct StreamingRowGroupIterator {
    file_path: String,
    #[allow(dead_code)]
    metadata: Arc<ParquetMetaData>,
    selected_row_groups: Vec<usize>,
    current_index: usize,
    column_projection: Option<Vec<String>>,
    #[allow(dead_code)]
    batch_size: usize,
}
/// Backwards-compat alias for [`ColumnarOptProgressiveSearchConfig`].
pub type ProgressiveSearchConfig = ColumnarOptProgressiveSearchConfig;

/// Progressive search configuration
#[derive(Debug, Clone)]
pub struct ColumnarOptProgressiveSearchConfig {
    /// Enable binary quantization filtering
    pub use_binary_filter: bool,
    /// Binary filter threshold (0.0-1.0)
    pub binary_threshold: f32,
    /// Enable INT8 quantization search
    pub use_int8_search: bool,
    /// INT8 search candidates multiplier
    pub int8_multiplier: f32,
    /// Enable PQ search
    pub use_pq_search: bool,
    /// PQ search candidates multiplier
    pub pq_multiplier: f32,
    /// Final FP32 reranking
    pub final_rerank: bool,
}

impl Default for ColumnarOptProgressiveSearchConfig {
    fn default() -> Self {
        Self {
            use_binary_filter: true,
            binary_threshold: 0.8,
            use_int8_search: true,
            int8_multiplier: 5.0,
            use_pq_search: true,
            pq_multiplier: 2.0,
            final_rerank: true,
        }
    }
}

/// Cost-based optimization statistics
#[derive(Debug)]
pub struct OptimizationStats {
    pub total_row_groups: usize,
    pub pruned_row_groups: usize,
    pub bloom_filter_hits: usize,
    pub bloom_filter_false_positives: usize,
    pub bytes_read: u64,
    pub bytes_skipped: u64,
    pub search_time_ms: u64,
    pub optimization_overhead_ms: u64,
}

impl ColumnarOptimizer {
    /// Create new columnar optimizer
    pub async fn new(
        distance_compute: Arc<UnifiedDistanceCompute>,
        config: ColumnarConfig,
        filesystem_factory: Arc<FilesystemFactory>,
        _collection_id: String,
        _engine_type: String, // "viper" or "nova"
    ) -> Result<Self> {
        // Create zero-copy filesystem with caching for efficient reads
        // Get a filesystem instance for the collection
        // For now, skip zero-copy filesystem as it requires Arc<dyn FileSystem>
        // and FilesystemFactory only provides &dyn FileSystem
        // This is a known limitation that needs refactoring
        let zero_copy_fs = filesystem_factory.clone();

        Ok(Self {
            distance_compute,
            config,
            zero_copy_fs,
            filesystem_factory,
            bloom_filter_cache: parking_lot::RwLock::new(HashMap::new()),
            stats_cache: parking_lot::RwLock::new(HashMap::new()),
        })
    }
    /// Load and cache bloom filters for a Parquet file
    pub async fn load_bloom_filters(&self, file_path: &str) -> Result<Arc<FileBloomFilters>> {
        // Check cache first
        {
            let cache = self.bloom_filter_cache.read();
            if let Some(filters) = cache.get(file_path) {
                return Ok(filters.clone());
            }
        }
        info!("Loading bloom filters for: {}", file_path);
        // Use filesystem factory to read file
        let data = self.zero_copy_fs.read(file_path).await?;
        let bytes = bytes::Bytes::from(data);
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let metadata = reader_builder.metadata();
        let mut file_filters = HashMap::new();
        let mut total_size = 0;
        // Process each row group
        for (rg_idx, row_group) in metadata.row_groups().iter().enumerate() {
            let mut column_filters = HashMap::new();

            // Check each column for bloom filters
            for (col_idx, column) in row_group.columns().iter().enumerate() {
                if self.extract_bloom_filter(column)?.is_some() {
                    let filter_info = BloomFilterInfo {
                        field: format!("col_{col_idx}"),
                        size_bytes: 1024,  // Default size estimate for bloom filter
                        hash_functions: 3, // Default value
                        num_items: 1000,   // Default value
                    };

                    total_size += filter_info.size_bytes;
                    column_filters.insert(filter_info.field.clone(), filter_info);
                }
            }

            if !column_filters.is_empty() {
                file_filters.insert(
                    format!("rg_{rg_idx}"),
                    RowGroupBloomFilters {
                        row_group_id: rg_idx,
                        column_filters,
                    },
                );
            }
        }

        let filters = Arc::new(FileBloomFilters {
            file_path: file_path.to_string(),
            filters: file_filters,
            total_size_bytes: total_size,
            false_positive_rate: 0.01, // Default
        });
        // Cache the filters
        {
            let mut cache = self.bloom_filter_cache.write();
            cache.insert(file_path.to_string(), filters.clone());
        }

        debug!(
            "Loaded {} bloom filters, total size: {} bytes",
            filters.filters.len(),
            total_size
        );
        Ok(filters)
    }

    /// Extract bloom filter from column metadata
    fn extract_bloom_filter(
        &self,
        _column: &parquet::file::metadata::ColumnChunkMetaData,
    ) -> Result<Option<BloomFilter>> {
        // In production, this would extract actual Parquet bloom filters
        // For now, return None as placeholder
        Ok(None)
    }

    /// Create streaming iterator for row groups
    pub async fn create_streaming_iterator(
        &self,
        file_path: &str,
        row_group_filter: Option<&MetadataFilter>,
        column_projection: Option<Vec<String>>,
    ) -> Result<StreamingRowGroupIterator> {
        info!("Creating streaming iterator for: {}", file_path);

        // Use filesystem factory to read file
        let data = self.zero_copy_fs.read(file_path).await?;
        let bytes = bytes::Bytes::from(data);
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let metadata = reader_builder.metadata().clone();
        // Select relevant row groups
        let selected_row_groups = self.select_row_groups(&metadata, row_group_filter).await?;
        debug!(
            "Selected {} row groups out of {}",
            selected_row_groups.len(),
            metadata.num_row_groups()
        );
        Ok(StreamingRowGroupIterator {
            file_path: file_path.to_string(),
            metadata,
            selected_row_groups,
            current_index: 0,
            column_projection,
            batch_size: self.config.optimization_thresholds.simd_threshold,
        })
    }

    /// Select row groups based on filter
    async fn select_row_groups(
        &self,
        metadata: &ParquetMetaData,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<usize>> {
        let mut selected = Vec::new();
        for (idx, row_group) in metadata.row_groups().iter().enumerate() {
            if self.should_include_row_group(row_group, filter).await? {
                selected.push(idx);
            }
        }
        Ok(selected)
    }
    /// Check if row group should be included
    async fn should_include_row_group(
        &self,
        row_group: &RowGroupMetaData,
        filter: Option<&MetadataFilter>,
    ) -> Result<bool> {
        let Some(filter) = filter else {
            return Ok(true);
        };
        // Check each filter condition against row group statistics
        for condition in &filter.conditions {
            match condition {
                crate::storage::engines::core::formats::columnar::FilterCondition::Equals(
                    column,
                    value,
                ) => {
                    if !self
                        .check_equals_condition(row_group, column, value)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                crate::storage::engines::core::formats::columnar::FilterCondition::Range(
                    column,
                    min,
                    max,
                ) => {
                    if !self
                        .check_range_condition(row_group, column, min, max)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                _ => {
                    // For other conditions, include the row group for safety
                    continue;
                }
            }
        }
        Ok(true)
    }

    /// Check equals condition against row group
    async fn check_equals_condition(
        &self,
        _row_group: &RowGroupMetaData,
        _column: &str,
        _value: &serde_json::Value,
    ) -> Result<bool> {
        // In production, would check column statistics
        Ok(true)
    }

    /// Check range condition against row group
    async fn check_range_condition(
        &self,
        _row_group: &RowGroupMetaData,
        _column: &str,
        _min: &serde_json::Value,
        _max: &serde_json::Value,
    ) -> Result<bool> {
        // In production, would check min/max statistics
        Ok(true)
    }
    /// Perform progressive similarity search
    pub async fn progressive_search(
        &self,
        file_paths: &[String],
        query_vector: &[f32],
        top_k: usize,
        distance_metric: &DistanceMetric,
        filter: Option<&MetadataFilter>,
        config: &ColumnarOptProgressiveSearchConfig,
    ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
        info!(
            "Progressive search across {} files, top_k={}",
            file_paths.len(),
            top_k
        );
        let mut all_candidates = Vec::new();
        let mut stats = OptimizationStats {
            total_row_groups: 0,
            pruned_row_groups: 0,
            bloom_filter_hits: 0,
            bloom_filter_false_positives: 0,
            bytes_read: 0,
            bytes_skipped: 0,
            search_time_ms: 0,
            optimization_overhead_ms: 0,
        };
        let start_time = std::time::Instant::now();
        for file_path in file_paths {
            let file_candidates = self
                .search_file_progressive(
                    file_path,
                    query_vector,
                    top_k,
                    distance_metric,
                    filter,
                    config,
                    &mut stats,
                )
                .await?;
            all_candidates.extend(file_candidates);
        }

        stats.search_time_ms = start_time.elapsed().as_millis() as u64;
        // Final ranking and selection
        all_candidates.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(top_k);
        info!("Progressive search complete: {:?}", stats);
        // Convert candidates to SearchResults
        let mut results = Vec::new();
        for candidate in all_candidates {
            if let Some(vector) = self.load_vector_at_location(&candidate).await? {
                results.push(crate::core::search::results::OptimizedSearchRecord {
                    id: candidate.vector_id.unwrap_or_else(|| {
                        format!("rg{}_row{}", candidate.row_group_id, candidate.row_offset)
                    }),
                    score: candidate.similarity,
                    similarity: Some(1.0 - candidate.similarity),
                    vector: vector
                        .embeddings
                        .into_iter()
                        .next()
                        .map(|e| Arc::new(e.values.to_fp32_owned())),
                    metadata: HashMap::new(), // Use SqlValue metadata
                    ..Default::default()
                });
            }
        }
        Ok(results)
    }
    /// Search single file with progressive strategy
    async fn search_file_progressive(
        &self,
        file_path: &str,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: &DistanceMetric,
        filter: Option<&MetadataFilter>,
        config: &ColumnarOptProgressiveSearchConfig,
        _stats: &mut OptimizationStats,
    ) -> Result<Vec<SearchCandidate>> {
        debug!("Progressive search in file: {}", file_path);
        // Create streaming iterator
        let mut iterator = self
            .create_streaming_iterator(
                file_path, filter, None, // Load all columns for now
            )
            .await?;
        let mut candidates = Vec::new();
        // Stage 1: Binary filtering (if enabled)
        if config.use_binary_filter {
            candidates = self
                .binary_filter_stage(&mut iterator, query_vector, config)
                .await?;
            debug!("Binary filter stage: {} candidates", candidates.len());
        }

        // Stage 2: INT8 quantized search (if enabled)
        if config.use_int8_search && !candidates.is_empty() {
            candidates = self
                .int8_search_stage(&mut iterator, query_vector, &candidates, config)
                .await?;
            debug!("INT8 search stage: {} candidates", candidates.len());
        }

        // Stage 3: PQ search (if enabled)
        if config.use_pq_search && !candidates.is_empty() {
            candidates = self
                .pq_search_stage(&mut iterator, query_vector, &candidates, config)
                .await?;
            debug!("PQ search stage: {} candidates", candidates.len());
        }

        // Stage 4: Final FP32 reranking (if enabled)
        if config.final_rerank && !candidates.is_empty() {
            candidates = self
                .fp32_rerank_stage(&mut iterator, query_vector, &candidates, distance_metric)
                .await?;
            debug!("FP32 rerank stage: {} candidates", candidates.len());
        }
        // Sort and limit
        candidates.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Binary filtering stage
    async fn binary_filter_stage(
        &self,
        iterator: &mut StreamingRowGroupIterator,
        _query_vector: &[f32],
        config: &ColumnarOptProgressiveSearchConfig,
    ) -> Result<Vec<SearchCandidate>> {
        let mut candidates = Vec::new();
        while let Some(batch) = iterator.next().await? {
            // Find binary vector column
            if let Some(_binary_col) = batch.column_by_name("vector_binary") {
                let binary_candidates = self.process_binary_batch(
                    &batch,
                    iterator.current_row_group(),
                    config.binary_threshold,
                )?;

                candidates.extend(binary_candidates);
            }
        }
        Ok(candidates)
    }

    /// Process binary batch
    fn process_binary_batch(
        &self,
        batch: &RecordBatch,
        row_group_id: usize,
        threshold: f32,
    ) -> Result<Vec<SearchCandidate>> {
        let mut candidates = Vec::new();
        // This is a simplified implementation
        // In production, would use efficient binary operations
        for row_idx in 0..batch.num_rows() {
            // Simulate binary similarity check
            let similarity = 0.8; // Placeholder
            if similarity >= threshold {
                candidates.push(SearchCandidate {
                    row_group_id,
                    row_offset: row_idx as u32,
                    similarity: 1.0 - similarity,
                    vector_id: None, // Will be filled later if needed
                });
            }
        }
        Ok(candidates)
    }

    /// INT8 quantized search stage
    async fn int8_search_stage(
        &self,
        _iterator: &mut StreamingRowGroupIterator,
        _query_vector: &[f32],
        candidates: &[SearchCandidate],
        _config: &ColumnarOptProgressiveSearchConfig,
    ) -> Result<Vec<SearchCandidate>> {
        // Refine candidates using INT8 quantized vectors
        // For now, just return the input candidates
        Ok(candidates.to_vec())
    }
    /// Product Quantization search stage
    async fn pq_search_stage(
        &self,
        _iterator: &mut StreamingRowGroupIterator,
        _query_vector: &[f32],
        candidates: &[SearchCandidate],
        _config: &ColumnarOptProgressiveSearchConfig,
    ) -> Result<Vec<SearchCandidate>> {
        // Refine candidates using PQ vectors
        Ok(candidates.to_vec())
    }
    /// Final FP32 reranking stage
    async fn fp32_rerank_stage(
        &self,
        _iterator: &mut StreamingRowGroupIterator,
        query_vector: &[f32],
        candidates: &[SearchCandidate],
        distance_metric: &DistanceMetric,
    ) -> Result<Vec<SearchCandidate>> {
        let mut reranked = Vec::new();
        for candidate in candidates {
            // Load full FP32 vector and compute exact distance
            if let Some(vector) = self.load_vector_at_candidate(candidate).await? {
                let distance_result = self.distance_compute.as_ref().calculate_distance(
                    query_vector,
                    &vector,
                    distance_metric,
                );
                let mut updated_candidate = candidate.clone();
                updated_candidate.similarity = distance_result.normalized_score;
                reranked.push(updated_candidate);
            }
        }
        Ok(reranked)
    }
    /// Load vector at specific candidate location
    async fn load_vector_at_candidate(
        &self,
        _candidate: &SearchCandidate,
    ) -> Result<Option<Vec<f32>>> {
        // This is a placeholder implementation
        // In production, would load the actual vector from the row group
        Ok(Some(vec![0.0; 768])) // Placeholder vector
    }

    /// Load full ProximaRecord at candidate location
    async fn load_vector_at_location(
        &self,
        candidate: &SearchCandidate,
    ) -> Result<Option<ProximaRecord>> {
        // In production, would load the full record from Parquet
        let oid = candidate
            .vector_id
            .clone()
            .unwrap_or_else(|| format!("unknown_{}", candidate.row_offset));
        let values = vec![0.0_f32; 768]; // Placeholder
        let dim = values.len() as u32;
        let mut record = ProximaRecord {
            oid,
            ..Default::default()
        };
        record
            .embeddings
            .push(EmbeddingCell::new_fp32("default", "vector", dim, values));
        Ok(Some(record))
    }

    /// Optimize row group layout for better performance
    pub async fn optimize_layout(&self, file_path: &str) -> Result<()> {
        info!("Optimizing row group layout for: {}", file_path);
        // This would analyze access patterns and reorganize data
        // For now, just return success
        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        let bloom_cache = self.bloom_filter_cache.read();
        stats.insert(
            "bloom_filter_cache_size".to_string(),
            serde_json::Value::Number(bloom_cache.len().into()),
        );
        let stats_cache = self.stats_cache.read();
        stats.insert(
            "stats_cache_size".to_string(),
            serde_json::Value::Number(stats_cache.len().into()),
        );
        stats
    }

    /// Clear optimization caches
    pub fn clear_caches(&self) {
        let mut bloom_cache = self.bloom_filter_cache.write();
        let mut stats_cache = self.stats_cache.write();
        bloom_cache.clear();
        stats_cache.clear();
        info!("Cleared columnar optimization caches");
    }
}

impl StreamingRowGroupIterator {
    /// Get next batch of records
    pub async fn next(&mut self) -> Result<Option<RecordBatch>> {
        if self.current_index >= self.selected_row_groups.len() {
            return Ok(None);
        }

        let row_group_idx = self.selected_row_groups[self.current_index];
        self.current_index += 1;
        // Deferred: StreamingRowGroupIterator needs refactoring to use zero-copy filesystem
        // For now, keeping direct file access but this breaks cloud compatibility
        let file = std::fs::File::open(&self.file_path)?;
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        // Build reader with column projection and row group selection
        let mut reader = {
            let schema = reader_builder.schema();
            let parquet_schema = reader_builder.parquet_schema();

            // Determine if we need column projection
            let needs_projection = if let Some(ref columns) = self.column_projection {
                let mut projection_indices = Vec::new();
                for name in columns {
                    if let Ok(field) = schema.field_with_name(name)
                        && let Some(index) = schema
                            .fields()
                            .iter()
                            .position(|f| f.name() == field.name())
                    {
                        projection_indices.push(index);
                    }
                }
                if !projection_indices.is_empty() {
                    Some(parquet::arrow::ProjectionMask::leaves(
                        parquet_schema,
                        projection_indices,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            // Apply projection and select specific row group
            if let Some(projection) = needs_projection {
                reader_builder
                    .with_projection(projection)
                    .with_row_groups(vec![row_group_idx])
                    .build()?
            } else {
                reader_builder
                    .with_row_groups(vec![row_group_idx])
                    .build()?
            }
        };
        // Read first batch (could be extended to read all batches)
        if let Some(batch) = reader.next() {
            Ok(Some(batch?))
        } else {
            Ok(None)
        }
    }

    /// Get current row group index
    pub fn current_row_group(&self) -> usize {
        if self.current_index > 0 {
            self.selected_row_groups[self.current_index - 1]
        } else {
            0
        }
    }

    /// Check if iterator has more data
    pub fn has_next(&self) -> bool {
        self.current_index < self.selected_row_groups.len()
    }

    /// Get total number of selected row groups
    pub fn total_row_groups(&self) -> usize {
        self.selected_row_groups.len()
    }
}

/// Proxy for bloom filter operations
pub struct BloomFilterProxy {
    size_bytes: usize,
    hash_functions: u32,
    false_positive_rate: f64,
    num_items: u64,
}

impl BloomFilterProxy {
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn hash_functions(&self) -> u32 {
        self.hash_functions
    }

    pub fn false_positive_rate(&self) -> f64 {
        self.false_positive_rate
    }

    pub fn num_items(&self) -> u64 {
        self.num_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    async fn test_columnar_optimizer_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let distance_compute = Arc::new(UnifiedDistanceCompute::new(
            crate::proto::proximadb_v1::DistanceMetric::Cosine,
        ));
        let config = ColumnarConfig::default();
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(Default::default())
                .await
                .unwrap(),
        );
        let optimizer = ColumnarOptimizer::new(
            distance_compute,
            config,
            filesystem_factory,
            "test_base_path".to_string(),
            "test_collection".to_string(),
        )
        .await
        .unwrap();
        let stats = optimizer.get_optimization_stats();
        assert!(stats.contains_key("bloom_filter_cache_size"));
        assert!(stats.contains_key("stats_cache_size"));
    }

    #[test]
    fn test_progressive_search_config() {
        let config = ColumnarOptProgressiveSearchConfig::default();
        assert!(config.use_binary_filter);
        assert!(config.use_int8_search);
        assert!(config.use_pq_search);
        assert!(config.final_rerank);
        assert_eq!(config.binary_threshold, 0.8);
    }
}
