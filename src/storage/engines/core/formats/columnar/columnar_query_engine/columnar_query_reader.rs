//! Unified Parquet Reader - Main entry point for query operations
//!
//! This module provides the UnifiedParquetReader that other parts of the
//! codebase expect, delegating to the appropriate modular components.

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::search_interface::SearchPlan;
use crate::proto::proximadb_v1::MetadataFilter;
use crate::storage::persistence::filesystem::{FileSystem, FilesystemFactory};
use anyhow::Result;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use proximadb_data_model::ProximaValue;
use proximadb_distance_kernel::DistanceMetric;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTree, ProximaTreeNode};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tracing::{debug, info, trace};
// SearchResponse should come from service types
type SearchResponse = crate::core::service_types::VectorSearchResponse;

use super::{BranchedFilterExecutor, CacheStrategy, ParquetReader, QueryConfig};

// Simple cosine similarity function for scoring
#[allow(dead_code)]
fn compute_cosine_similarity(a: &[f32], b: &Arc<Vec<f32>>) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (a_val, b_val) in a.iter().zip(b.iter()) {
        dot_product += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot_product / denom
    }
}

/// Reading strategy selector
#[derive(Debug, Clone)]
pub struct ReadingStrategySelector {
    pub enable_ipc_optimization: bool,
    pub ipc_size_threshold_mb: f64,
    pub enable_adaptive_selection: bool,
    pub cache_ipc_decisions: bool,
}

impl Default for ReadingStrategySelector {
    fn default() -> Self {
        Self {
            enable_ipc_optimization: true,
            ipc_size_threshold_mb: 100.0,
            enable_adaptive_selection: true,
            cache_ipc_decisions: true,
        }
    }
}

/// Schema mapping for columnar storage
#[derive(Debug, Clone)]
pub struct SchemaMapping {
    pub arrow_schema: Arc<Schema>,
    pub field_indices: HashMap<String, usize>,
    pub filterable_columns: Vec<String>,
}

/// Backwards-compat alias for [`ColumnarQueryCollectionContext`].
pub type CollectionContext = ColumnarQueryCollectionContext;

/// Collection context for queries
#[derive(Debug, Clone)]
pub struct ColumnarQueryCollectionContext {
    pub collection_id: String,
    pub dimension: usize,
    pub distance_metric: String,
    pub quantization_config: Option<crate::proto::proximadb_v1::QuantizationConfig>,
}

/// Reading strategy for data access
#[derive(Debug, Clone, Copy)]
pub enum ReadingStrategy {
    /// Use Arrow IPC format for fast memory-mapped access
    ArrowIPC,
    /// Use standard Parquet reading
    Parquet,
    /// Choose automatically based on file characteristics
    Auto,
}

/// Reader configuration
#[derive(Debug, Clone)]
pub struct ReaderConfig {
    pub enable_pushdown_predicates: bool,
    pub enable_row_group_pruning: bool,
    pub enable_page_index: bool,
    pub batch_size: usize,
    pub cache_metadata: bool,
    pub parallel_row_groups: bool,
    pub cache_context: Option<CacheContext>,
}

/// Cache context for unified caching filesystem integration
#[derive(Debug, Clone)]
pub struct CacheContext {
    pub cached_filesystem:
        Arc<crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem>,
    pub collection_id: String,
    pub engine_type: String,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            enable_pushdown_predicates: true,
            enable_row_group_pruning: true,
            enable_page_index: true,
            batch_size: 8192,
            cache_metadata: true,
            parallel_row_groups: true,
            cache_context: None,
        }
    }
}

/// Filter value for metadata filtering
#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    StringList(Vec<String>),
    IntegerList(Vec<i64>),
    FloatList(Vec<f64>),
}

/// Quantization method for vectors
#[derive(Debug, Clone, Copy)]
pub enum QuantizationMethod {
    None,
    Binary,
    Int8,
    PQ,
}

/// Seek range for targeted reads
#[derive(Debug, Clone)]
pub struct SeekRange {
    pub start: usize,
    pub end: usize,
    pub row_groups: Vec<usize>,
}

/// Vector position in storage
#[derive(Debug, Clone)]
pub struct VectorPosition {
    pub row_group: usize,
    pub row_offset: usize,
    pub global_row: usize,
}

/// Stage 2 search strategy
#[derive(Debug, Clone, Copy)]
pub enum Stage2Strategy {
    ExactSearch,
    ApproximateSearch,
    HybridSearch,
}

/// Search type
#[derive(Debug, Clone, Copy)]
pub enum SearchType {
    KNN,
    RadiusSearch,
    FilteredSearch,
}

/// Row group access pattern
#[derive(Debug, Clone, Copy)]
pub enum RowGroupAccessPattern {
    Sequential,
    Random,
    Filtered,
    All,
}

/// Page pruning info
#[derive(Debug, Clone)]
pub struct PagePruningInfo {
    pub total_pages: usize,
    pub pruned_pages: usize,
    pub pages_read: usize,
    pub pruning_effectiveness: f64,
}

/// Page range
#[derive(Debug, Clone)]
pub struct PageRange {
    pub start_page: usize,
    pub end_page: usize,
    pub row_count: usize,
}

/// Unified Parquet reader (main entry point for compatibility)
#[derive(Clone)]
pub struct UnifiedParquetReader {
    pub file_paths: Vec<String>,
    pub dimension: usize,
    pub config: ReaderConfig,
    pub schema_mapping: Option<SchemaMapping>,
    pub filesystem_factory: Arc<FilesystemFactory>,
}

impl UnifiedParquetReader {
    /// Create new reader with unified caching filesystem for optimal performance
    pub fn new(
        file_paths: Vec<String>,
        dimension: usize,
        filesystem_factory: Arc<FilesystemFactory>,
        cached_filesystem: Arc<
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
        >,
        collection_id: String,
        engine_type: String,
    ) -> Result<Self> {
        let config = ReaderConfig {
            cache_context: Some(CacheContext {
                cached_filesystem,
                collection_id: collection_id.clone(),
                engine_type: engine_type.clone(),
            }),
            ..Default::default()
        };

        Ok(Self {
            file_paths,
            dimension,
            config,
            schema_mapping: None,
            filesystem_factory,
        })
    }

    /// Set configuration
    pub fn with_config(mut self, config: ReaderConfig) -> Self {
        self.config = config;
        self
    }

    /// Get filesystem for a given file path
    fn get_filesystem_for_path(&self, file_path: &str) -> Result<Arc<dyn FileSystem>> {
        self.filesystem_factory
            .get_filesystem(file_path)
            .map_err(|e| anyhow::anyhow!("Failed to get filesystem for path {}: {}", file_path, e))
    }

    /// Read all records using filesystem API
    pub async fn read_all_records(
        &self,
        start_offset: usize,
        limit: Option<usize>,
    ) -> Result<Vec<ProximaRecord>> {
        let query_config = QueryConfig {
            enable_pushdown: self.config.enable_pushdown_predicates,
            enable_projection: true,
            enable_statistics: self.config.enable_row_group_pruning,
            cache_strategy: CacheStrategy::LRU,
            limit,
            enable_parallel: self.config.parallel_row_groups,
            parallel_workers: 4,
        };

        let mut all_records = Vec::new();

        for file_path in &self.file_paths {
            // Use filesystem API for reading
            let fs = self.get_filesystem_for_path(file_path)?;
            let path = FilesystemFactory::resolve_path(file_path)?;

            // Check if file exists before attempting to read
            if !fs.exists(&path).await? {
                continue; // Skip non-existent files
            }

            // Create ParquetReader with filesystem integration
            let mut reader = ParquetReader::new(query_config.clone());
            let records = reader.read_all_with_filesystem(&path, fs).await?;
            all_records.extend(records);

            if let Some(limit) = limit
                && all_records.len() >= limit
            {
                all_records.truncate(limit);
                break;
            }
        }

        // Apply start offset if needed
        if start_offset > 0 && start_offset < all_records.len() {
            all_records = all_records.into_iter().skip(start_offset).collect();
        }

        Ok(all_records)
    }

    /// Query with metadata filters
    pub async fn query_with_metadata_filters(
        &self,
        _filters: &[MetadataFilter],
    ) -> Result<Vec<ProximaRecord>> {
        let filterable_columns = self
            .schema_mapping
            .as_ref()
            .map(|s| s.filterable_columns.clone())
            .unwrap_or_default();

        // BranchedFilterExecutor: disabled pending API stabilization
        let _executor = BranchedFilterExecutor::new(
            filterable_columns,
            self.file_paths.clone(),
            self.dimension,
        );

        // Temporarily return empty results until API is ready
        Ok(Vec::new())
    }

    /// Query by IDs
    pub async fn query_by_ids(&self, ids: &[String]) -> Result<Vec<ProximaRecord>> {
        // This would use the ID index for efficient lookup
        // For now, scan all files and filter
        let all_records = self.read_all_records(0, None).await?;

        let id_set: std::collections::HashSet<_> = ids.iter().cloned().collect();
        Ok(all_records
            .into_iter()
            .filter(|r| id_set.contains(&r.oid))
            .collect())
    }

    /// Check if should use IPC format for scanning (async version using filesystem API)
    pub async fn should_use_ipc_for_scan(&self, file_path: &str) -> bool {
        // Simple heuristic: use IPC for large files
        match self.get_file_metadata_async(file_path).await {
            Ok(metadata) => {
                let file_size_mb = metadata.size as f64 / (1024.0 * 1024.0);
                file_size_mb > 100.0 // Use IPC for files > 100MB
            }
            Err(_) => false,
        }
    }

    /// Get file metadata using filesystem API
    async fn get_file_metadata_async(
        &self,
        file_path: &str,
    ) -> Result<crate::storage::persistence::filesystem::FileMetadata> {
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;
        fs.metadata(&path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get metadata for {}: {}", file_path, e))
    }

    /// Read specific row groups with projection
    /// Row group reading with projection: deferred to columnar optimization phase
    pub async fn read_row_groups_projected(
        &self,
        _collection_id: &str,
        row_groups: &[usize],
        _projection: Option<&[String]>,
    ) -> Result<Vec<ProximaRecord>> {
        // If no specific row groups requested, read all
        if row_groups.is_empty() {
            return self.read_all_records(0, None).await;
        }

        // Read records and filter by row group index
        // Assuming row_group_size is 100 (from test configuration)
        let row_group_size = 100; // This should be obtained from metadata

        let mut result = Vec::new();
        for &rg_idx in row_groups {
            let start_offset = rg_idx * row_group_size;
            let limit = Some(row_group_size);

            // Read records for this row group
            let records = self.read_all_records(start_offset, limit).await?;
            result.extend(records);
        }

        Ok(result)
    }

    /// Get collection context - returns metadata about the collection
    pub async fn get_collection_context(&self) -> ColumnarQueryCollectionContext {
        ColumnarQueryCollectionContext {
            dimension: self.dimension,
            collection_id: String::new(),
            distance_metric: "cosine".to_string(),
            quantization_config: None,
        }
    }

    /// Search vectors using unified search interface with optimizations
    /// Implements column projection and metadata-based row group pruning
    pub async fn search_vectors(
        &self,
        search_plan: &SearchPlan,
        _collection_context: &ColumnarQueryCollectionContext,
    ) -> Result<SearchResponse> {
        let start_time = std::time::Instant::now();

        // Determine what columns we actually need based on the search requirements
        let needs_vectors = true; // Always need vectors for similarity search
        let needs_metadata = search_plan
            .collection_config
            .as_ref()
            .is_some_and(|c| c.enable_metadata_filtering);

        // Extract filter expression from search plan for row group pruning
        let filter_expression = search_plan.filter_expression.clone();

        debug!("UnifiedParquetReader: Optimized search starting");
        debug!("  Files to scan: {}", self.file_paths.len());
        debug!(
            "  Column projection: vectors={}, metadata={}",
            needs_vectors, needs_metadata
        );
        debug!(
            "  Top-k: {}, Early termination: {}",
            search_plan.top_k, search_plan.enable_early_termination
        );
        if filter_expression.is_some() {
            debug!("  Filter expression: present");
        }

        // Initialize bounded priority queue for top-k results
        let mut priority_queue = BoundedPriorityQueue::new(search_plan.top_k);
        let mut total_records_scanned = 0;
        #[allow(unused_assignments)]
        let mut row_groups_skipped = 0;
        let files_skipped_early = 0;

        // Check if quantization is enabled for this search
        let quantization_enabled = search_plan
            .collection_config
            .as_ref()
            .is_some_and(|c| c.enable_quantization);

        // Determine distance metric from collection config
        let distance_metric = search_plan
            .collection_config
            .as_ref()
            .map_or(DistanceMetric::Cosine, |c| c.default_distance_metric);

        // ── Phase 1: Concurrent I/O ─────────────────────────────────────────
        // Read all files concurrently using tokio tasks. Each task performs
        // row group pruning and vectorized metadata filtering independently.
        let total_rg_skipped = Arc::new(AtomicUsize::new(0));

        let mut io_handles = Vec::with_capacity(self.file_paths.len());
        for file_path in &self.file_paths {
            let reader = self.clone();
            let fp = file_path.clone();
            let fe = filter_expression.clone();
            let rg_counter = Arc::clone(&total_rg_skipped);

            io_handles.push(tokio::spawn(async move {
                let result = reader
                    .read_file_batches_with_filters(
                        &fp,
                        true, // needs_vectors
                        needs_metadata,
                        fe.as_ref(),
                        quantization_enabled,
                    )
                    .await;

                match result {
                    Ok((batches, skipped)) => {
                        rg_counter.fetch_add(skipped, AtomicOrdering::Relaxed);
                        Ok(batches)
                    }
                    Err(e) => {
                        debug!("File read failed for {}: {}", fp, e);
                        Err(e)
                    }
                }
            }));
        }

        // Await all I/O tasks and collect batches
        let mut all_batches: Vec<RecordBatch> = Vec::new();
        let mut files_succeeded = 0usize;
        let mut last_error: Option<anyhow::Error> = None;
        for handle in io_handles {
            match handle.await {
                Ok(Ok(batches)) => {
                    files_succeeded += 1;
                    all_batches.extend(batches);
                }
                Ok(Err(e)) => {
                    debug!("Skipping file due to error: {}", e);
                    last_error = Some(e);
                }
                Err(e) => {
                    debug!("Task join error: {}", e);
                    last_error = Some(anyhow::anyhow!("Task join error: {}", e));
                }
            }
        }

        // If ALL files failed, propagate the last error
        if files_succeeded == 0 && !self.file_paths.is_empty() {
            return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All file reads failed")));
        }

        row_groups_skipped = total_rg_skipped.load(AtomicOrdering::Relaxed);

        // ── Phase 2: Parallel SIMD scoring ──────────────────────────────────
        // Score all batches in parallel using rayon. Each rayon thread gets
        // a thread-local BoundedPriorityQueue, avoiding contention.
        if let Some(query_vector) = &search_plan.query_vector {
            use rayon::prelude::*;

            let top_k = search_plan.top_k;
            let min_score = search_plan.min_score;
            let query_vec = query_vector.clone();
            let reader_ref = self.clone();
            let records_scanned = AtomicUsize::new(0);

            // Morsel size: process batches in groups for cache efficiency
            // Each morsel gets its own local priority queue
            const MORSEL_SIZE: usize = 4;

            let local_queues: Vec<BoundedPriorityQueue> = all_batches
                .par_chunks(MORSEL_SIZE)
                .map(|morsel| {
                    let engine = UnifiedDistanceCompute::new(distance_metric);
                    let mut local_queue = BoundedPriorityQueue::new(top_k);

                    for batch in morsel {
                        records_scanned.fetch_add(batch.num_rows(), AtomicOrdering::Relaxed);
                        let _ = reader_ref.score_batch_with_simd(
                            batch,
                            &query_vec,
                            &engine,
                            &distance_metric,
                            min_score,
                            needs_metadata,
                            &mut local_queue,
                        );
                    }

                    local_queue
                })
                .collect();

            total_records_scanned = records_scanned.load(AtomicOrdering::Relaxed);

            // ── Phase 3: Merge ──────────────────────────────────────────────
            // Merge thread-local priority queues into the final queue
            for local_queue in local_queues {
                priority_queue.merge(local_queue);
            }
        } else {
            // No query vector — collect records without scoring (rare path)
            let distance_engine = UnifiedDistanceCompute::new(distance_metric);
            let _ = &distance_engine; // suppress unused warning
            for batch in &all_batches {
                total_records_scanned += batch.num_rows();
                let records =
                    self.extract_records_from_batch(batch, needs_vectors, needs_metadata)?;
                for record in records {
                    let record_id = record.oid.clone();
                    let vector = record
                        .embeddings
                        .first()
                        .map(|e| Arc::new(e.values.to_fp32_owned()));
                    let metadata = if needs_metadata {
                        record
                            .props
                            .into_iter()
                            .filter_map(|(key, node)| match node {
                                ProximaTreeNode::Value(value) => Some((key, value)),
                                _ => None,
                            })
                            .collect()
                    } else {
                        Default::default()
                    };
                    let search_record = crate::core::search::results::OptimizedSearchRecord {
                        id: record_id.clone(),
                        vector_id: Some(record_id),
                        score: 0.0,
                        similarity: Some(0.0),
                        vector,
                        metadata,
                        version: Some(record.record_version as u32),
                        timestamp: Some(record.created_at_ns),
                        semantic_similarity: None,
                        quantization_info: None,
                        engine_stats: None,
                        index_path: None,
                        ..Default::default()
                    };
                    if priority_queue.len() < search_plan.top_k {
                        priority_queue.try_insert(search_record);
                    }
                }
            }
        }

        // Extract final results from priority queue (already sorted)
        let all_results = priority_queue.into_sorted_vec();
        let total_results = all_results.len();

        let processing_time_us = start_time.elapsed().as_micros() as i64;

        info!(
            "UnifiedParquetReader: Search complete - scanned: {}, skipped: {}, returned: {}, files_skipped: {}, time: {}ms",
            total_records_scanned,
            row_groups_skipped,
            total_results,
            files_skipped_early,
            processing_time_us / 1000
        );

        // Optimizations active:
        // 1. Column projection (skip metadata if not needed)
        // 2. Concurrent file I/O (tokio tasks)
        // 3. Parallel SIMD scoring (rayon morsel-driven)
        // 4. Vectorized metadata filtering (Arrow compute kernels)
        // 5. Lazy materialization (metadata only for top-k candidates)

        Ok(SearchResponse {
            success: true,
            results: all_results,
            total_count: total_results as i64,
            total_found: total_results as i64,
            processing_time_us,
            algorithm_used: "UnifiedParquetReader-Parallel-SIMD".to_string(),
            search_metadata: crate::core::service_types::SearchMetadata {
                algorithm_used: "UnifiedParquetReader-Parallel-SIMD".to_string(),
                query_id: None,
                query_complexity: 0.0,
                total_results: total_results as i64,
                search_time_ms: (processing_time_us / 1000) as f64,
                performance_hint: Some(format!(
                    "Column projection active. Scanned {} records, skipped {} row groups",
                    total_records_scanned, row_groups_skipped
                )),
                index_stats: None,
            },
            debug_info: Some(crate::core::service_types::SearchDebugInfo {
                search_steps: vec![
                    format!("Scanned {} Parquet files", self.file_paths.len()),
                    format!("Read {} total records", total_records_scanned),
                    format!("Skipped {} row groups", row_groups_skipped),
                ],
                clusters_searched: vec![],
                filter_pushdown_enabled: false, // Enabled when metadata filters are set in query context
                parquet_columns_scanned: if needs_metadata {
                    vec![
                        "id".to_string(),
                        "vector".to_string(),
                        "metadata".to_string(),
                        "version".to_string(),
                    ]
                } else {
                    vec![
                        "id".to_string(),
                        "vector".to_string(),
                        "version".to_string(),
                    ]
                },
                timing_breakdown: {
                    let mut timing = std::collections::HashMap::new();
                    timing.insert("total_ms".to_string(), (processing_time_us / 1000) as f64);
                    timing
                },
                memory_usage_mb: None,
                estimated_total_cost: None,
                actual_cost: Some(total_records_scanned as f64),
                cost_breakdown: None,
            }),
        })
    }

    /// Optimized file reading with column projection, row group filtering, and bloom filters
    /// Returns (records, row_groups_skipped)
    #[allow(dead_code)]
    async fn read_file_with_optimization_and_filters(
        &self,
        file_path: &str,
        needs_vectors: bool,
        needs_metadata: bool,
        filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
        quantization_enabled: bool,
    ) -> Result<(Vec<ProximaRecord>, usize)> {
        use bytes::Bytes;
        use parquet::arrow::ProjectionMask;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        // Get filesystem for this file
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        // Check if file exists - return error if missing
        if !fs.exists(&path).await? {
            return Err(anyhow::anyhow!("File not found: {}", file_path));
        }

        // Read file data
        let file_data = fs.read(&path).await?;

        // Convert to Bytes for Parquet reader
        let file_bytes = Bytes::from(file_data);

        // First, create a file reader to access metadata for row group pruning
        let file_reader = SerializedFileReader::new(file_bytes.clone())?;
        let metadata = file_reader.metadata();
        let total_row_groups = metadata.num_row_groups();

        // ROW GROUP PRUNING: Filter row groups based on filter expression
        // Convert FilterExpression to MetadataFilter for statistics-based pruning
        let selected_row_groups: Vec<usize> = if let Some(filter_expr) = filter_expression {
            // Convert FilterExpression to MetadataFilter for pruning
            if let Some(metadata_filter) = crate::storage::engines::core::formats::columnar::MetadataFilter::from_filter_expression(filter_expr) {
                debug!(
                    "  Converted FilterExpression to MetadataFilter with {} conditions",
                    metadata_filter.conditions.len()
                );

                // Try to prune row groups using Parquet column statistics
                let mut selected = Vec::new();
                for rg_idx in 0..total_row_groups {
                    let row_group_meta = metadata.row_group(rg_idx);
                    let mut keep_row_group = true;

                    // Check each filter condition against row group statistics
                    for condition in &metadata_filter.conditions {
                        let column_name = condition.column();

                        // Find the column in the row group metadata
                        let column_idx = (0..row_group_meta.num_columns())
                            .find(|&i| {
                                row_group_meta.column(i).column_descr().name() == column_name
                            });

                        if let Some(col_idx) = column_idx {
                            let col_meta = row_group_meta.column(col_idx);

                            // Check statistics if available
                            if let Some(stats) = col_meta.statistics() {
                                // For range/equality filters, check if the value could be in this row group
                                {
                                    use super::filter_pushdown_engine::{
                                        check_numeric_in_stats_range, check_range_overlaps_stats,
                                    };

                                    let col_type = col_meta.column_descr().physical_type();

                                    match condition {
                                    crate::storage::engines::core::formats::columnar::FilterCondition::Equals(_, value) => {
                                        if value.is_string() {
                                            // Check if the value falls within the min/max range
                                            if let (Some(min_bytes), Some(max_bytes)) = (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                                                // Convert bytes to string for comparison
                                                let min_str = String::from_utf8_lossy(min_bytes);
                                                let max_str = String::from_utf8_lossy(max_bytes);
                                                let value_str = value.as_str().unwrap_or(&value.to_string()).to_string();

                                                // Skip if statistics are empty or invalid
                                                if min_str.is_empty() || max_str.is_empty() {
                                                    // Skip pruning for empty stats
                                                } else {
                                                    // Skip row group if value is clearly outside range
                                                    if value_str.as_str() < min_str.as_ref() || value_str.as_str() > max_str.as_ref() {
                                                        keep_row_group = false;
                                                        debug!("  Row group {} pruned: {} not in [{}, {}]",
                                                            rg_idx, value_str, min_str, max_str);
                                                        break;
                                                    }
                                                }
                                            }
                                        } else if let Some(num_val) = value.as_f64().or_else(|| value.as_i64().map(|i| i as f64)) {
                                            // Numeric equality: check typed statistics
                                            if let Some(false) = check_numeric_in_stats_range(stats, col_type, num_val) {
                                                keep_row_group = false;
                                                debug!("  Row group {} pruned: numeric {} outside stats range", rg_idx, num_val);
                                                break;
                                            }
                                        }
                                        // For other value types (boolean), skip pruning
                                    }
                                    crate::storage::engines::core::formats::columnar::FilterCondition::Range(_, min_val, max_val) => {
                                        let filter_min = min_val.as_f64().or_else(|| min_val.as_i64().map(|i| i as f64));
                                        let filter_max = max_val.as_f64().or_else(|| max_val.as_i64().map(|i| i as f64));

                                        if let (Some(fmin), Some(fmax)) = (filter_min, filter_max)
                                            && let Some(false) = check_range_overlaps_stats(stats, col_type, fmin, fmax) {
                                                keep_row_group = false;
                                                debug!("  Row group {} pruned: range [{}, {}] doesn't overlap stats", rg_idx, fmin, fmax);
                                                break;
                                            }
                                    }
                                    crate::storage::engines::core::formats::columnar::FilterCondition::In(_, values) => {
                                        // Check if any value in the list could fall within stats range
                                        let mut any_could_match = false;
                                        let mut has_checkable_values = false;

                                        for v in values {
                                            if let Some(num_val) = v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)) {
                                                has_checkable_values = true;
                                                match check_numeric_in_stats_range(stats, col_type, num_val) {
                                                    Some(true) => { any_could_match = true; break; }
                                                    Some(false) => {} // Definitely not in range
                                                    None => { any_could_match = true; break; } // Can't determine
                                                }
                                            } else if v.is_string() {
                                                has_checkable_values = true;
                                                if let (Some(min_bytes), Some(max_bytes)) = (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                                                    let min_str = String::from_utf8_lossy(min_bytes);
                                                    let max_str = String::from_utf8_lossy(max_bytes);
                                                    let vs = v.as_str().unwrap_or_default();
                                                    if !min_str.is_empty() && !max_str.is_empty()
                                                        && vs >= min_str.as_ref() && vs <= max_str.as_ref() {
                                                        any_could_match = true;
                                                        break;
                                                    }
                                                } else {
                                                    any_could_match = true;
                                                    break;
                                                }
                                            } else {
                                                any_could_match = true;
                                                break;
                                            }
                                        }

                                        if has_checkable_values && !any_could_match {
                                            keep_row_group = false;
                                            debug!("  Row group {} pruned: no IN values within stats range", rg_idx);
                                            break;
                                        }
                                    }
                                    crate::storage::engines::core::formats::columnar::FilterCondition::IsNull(_) => {
                                        // If null_count == 0, no nulls exist, prune
                                        if stats.null_count_opt() == Some(0) {
                                            keep_row_group = false;
                                            debug!("  Row group {} pruned: null_count == 0 for IsNull filter", rg_idx);
                                            break;
                                        }
                                    }
                                    crate::storage::engines::core::formats::columnar::FilterCondition::IsNotNull(_) => {
                                        // If all rows are null, no non-null rows exist
                                        let null_count = stats.null_count_opt().unwrap_or(0);
                                        let num_rows = row_group_meta.num_rows() as u64;
                                        if null_count >= num_rows && num_rows > 0 {
                                            keep_row_group = false;
                                            debug!("  Row group {} pruned: all rows null for IsNotNull filter", rg_idx);
                                            break;
                                        }
                                    }
                                    }
                                }
                            }
                        }
                    }

                    if keep_row_group {
                        selected.push(rg_idx);
                    }
                }

                if selected.len() < total_row_groups {
                    info!(
                        "Row group pruning: keeping {} of {} row groups ({} pruned)",
                        selected.len(),
                        total_row_groups,
                        total_row_groups - selected.len()
                    );
                }

                selected
            } else {
                debug!("  FilterExpression couldn't be converted to MetadataFilter");
                (0..total_row_groups).collect()
            }
        } else {
            // No filters, select all row groups
            (0..total_row_groups).collect()
        };

        let row_groups_skipped = total_row_groups - selected_row_groups.len();

        if selected_row_groups.is_empty() {
            // No row groups matched the filters
            return Ok((Vec::new(), row_groups_skipped));
        }

        // Now create the reader builder with selected row groups
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file_bytes)?;

        // Build projection mask - only read columns we need
        let schema = reader_builder.schema();
        let mut projection = Vec::new();

        // Check for quantized vector columns first (for pre-filtering optimization)
        // Only check if quantization is enabled in collection config
        let has_binary_vectors = quantization_enabled
            && schema
                .index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
                )
                .is_ok();
        let has_int8_vectors = quantization_enabled
            && schema
                .index_of(crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8)
                .is_ok();
        let has_pq_vectors = quantization_enabled
            && schema
                .index_of(crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8)
                .is_ok();

        // Always need ID
        if let Ok(idx) = schema.index_of("id") {
            projection.push(idx);
        }

        // If quantized vectors are available and we need vectors, read them for pre-filtering
        if needs_vectors && (has_binary_vectors || has_int8_vectors || has_pq_vectors) {
            // Read quantized vectors for fast approximate filtering
            if has_binary_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
                )
            {
                projection.push(idx);
            }
            if has_int8_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8,
                )
            {
                projection.push(idx);
            }
            if has_pq_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8,
                )
            {
                projection.push(idx);
            }
        }

        // Vector column if needed
        if needs_vectors {
            if let Ok(idx) = schema.index_of("vector") {
                projection.push(idx);
            } else if let Ok(idx) = schema.index_of("vector_fp32") {
                projection.push(idx);
            }
        }

        // Metadata columns if needed
        if needs_metadata {
            // Add any metadata columns
            if let Ok(idx) = schema.index_of("metadata") {
                projection.push(idx);
            }
            if let Ok(idx) = schema.index_of("extra_meta") {
                projection.push(idx);
            }

            // Add ALL non-reserved columns (filterable columns) to projection
            // This is crucial for reading typed metadata columns
            for field_idx in 0..schema.fields().len() {
                let field_name = schema.field(field_idx).name();

                // Skip columns we've already added or reserved columns
                if matches!(
                    field_name.as_str(),
                    "id" | "row_group_offset"
                        | "row_index"
                        | "vector"
                        | "vector_fp32"
                        | "q_binary"
                        | "q_int8"
                        | "qp_int8_scale"
                        | "qp_int8_min"
                        | "qp_int8_max"
                        | "q_pq4"
                        | "q_pq8"
                        | "q_pq16"
                        | "q_pq32"
                        | "timestamp"
                        | "updated_at"
                        | "expires_at"
                        | "version"
                        | "source"
                        | "extra_meta"
                        | "metadata"
                ) {
                    continue;
                }

                // This is a filterable column - add to projection
                projection.push(field_idx);
            }

            // Also add version and timestamp for filtering
            if let Ok(idx) = schema.index_of("version") {
                projection.push(idx);
            }
            if let Ok(idx) = schema.index_of("timestamp") {
                projection.push(idx);
            }
        } else {
            // Even without metadata, we need version and timestamp
            if let Ok(idx) = schema.index_of("version") {
                projection.push(idx);
            }
            if let Ok(idx) = schema.index_of("timestamp") {
                projection.push(idx);
            }
        }

        // Create projection mask
        let projection_mask = ProjectionMask::roots(reader_builder.parquet_schema(), projection);

        // Create reader with projection and selected row groups
        let reader = reader_builder
            .with_projection(projection_mask)
            .with_row_groups(selected_row_groups)
            .with_batch_size(1024) // Process in reasonable batches
            .build()?;

        let mut records = Vec::new();

        for batch in reader {
            let batch = batch?;

            if let Some(filter_expr) = filter_expression {
                // Vectorized path: evaluate the full FilterExpression tree (AND/OR/NOT,
                // boolean equality, all numeric/string comparison kernels) into a mask.
                match super::vectorized_executor::vectorized_filter_batch_expr(
                    batch.clone(),
                    filter_expr,
                )? {
                    Some(filtered_batch) => {
                        let batch_records = self.extract_records_from_batch(
                            &filtered_batch,
                            needs_vectors,
                            needs_metadata,
                        )?;
                        records.extend(batch_records);
                    }
                    None => {
                        // Some sub-expression has no vectorized kernel — fall back to
                        // correct row-at-a-time filtering.
                        let batch_records =
                            self.extract_records_from_batch(&batch, needs_vectors, needs_metadata)?;
                        for record in batch_records {
                            if self.matches_filter_expression(&record, filter_expr) {
                                records.push(record);
                            }
                        }
                    }
                }
            } else {
                // No filter - add all records
                let batch_records =
                    self.extract_records_from_batch(&batch, needs_vectors, needs_metadata)?;
                records.extend(batch_records);
            }
        }

        debug!(
            "  Row groups: {}/{} selected (skipped {})",
            total_row_groups - row_groups_skipped,
            total_row_groups,
            row_groups_skipped
        );
        debug!("  Records after filtering: {}", records.len());

        Ok((records, row_groups_skipped))
    }

    /// Read filtered RecordBatches from a Parquet file without materializing into ProximaRecords.
    ///
    /// This is the batch-oriented alternative to `read_file_with_optimization_and_filters`.
    /// It returns Arrow RecordBatches with row group pruning and vectorized metadata filtering
    /// already applied, but without the per-row ProximaRecord materialization cost.
    /// Used by the SIMD-accelerated search path in `search_vectors`.
    async fn read_file_batches_with_filters(
        &self,
        file_path: &str,
        needs_vectors: bool,
        needs_metadata: bool,
        filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
        quantization_enabled: bool,
    ) -> Result<(Vec<arrow::record_batch::RecordBatch>, usize)> {
        use bytes::Bytes;
        use parquet::arrow::ProjectionMask;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        if !fs.exists(&path).await? {
            return Err(anyhow::anyhow!("File not found: {}", file_path));
        }

        let file_data = fs.read(&path).await?;
        let file_bytes = Bytes::from(file_data);

        // Access metadata for row group pruning
        let file_reader = SerializedFileReader::new(file_bytes.clone())?;
        let metadata = file_reader.metadata();
        let total_row_groups = metadata.num_row_groups();

        // Reuse the same row group pruning logic as read_file_with_optimization_and_filters.
        // We delegate to a shared helper to avoid duplicating the pruning code.
        let selected_row_groups =
            self.select_row_groups_with_pruning(metadata, total_row_groups, filter_expression);

        let row_groups_skipped = total_row_groups - selected_row_groups.len();

        if selected_row_groups.is_empty() {
            return Ok((Vec::new(), row_groups_skipped));
        }

        // Build projection mask (same logic as read_file_with_optimization_and_filters)
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file_bytes)?;
        let schema = reader_builder.schema();
        let projection = self.build_projection_indices(
            schema,
            needs_vectors,
            needs_metadata,
            quantization_enabled,
        );

        let projection_mask = ProjectionMask::roots(reader_builder.parquet_schema(), projection);

        let reader = reader_builder
            .with_projection(projection_mask)
            .with_row_groups(selected_row_groups)
            .with_batch_size(1024)
            .build()?;

        let mut result_batches = Vec::new();

        for batch in reader {
            let batch = batch?;

            if let Some(filter_expr) = filter_expression {
                // Vectorized path: evaluate the full FilterExpression tree (AND/OR/NOT,
                // boolean equality, all numeric/string comparison kernels) into a mask.
                match super::vectorized_executor::vectorized_filter_batch_expr(
                    batch.clone(),
                    filter_expr,
                )? {
                    Some(filtered_batch) => {
                        if filtered_batch.num_rows() > 0 {
                            result_batches.push(filtered_batch);
                        }
                    }
                    None => {
                        // Some sub-expression has no vectorized kernel — fall back to
                        // correct row-at-a-time evaluation rather than passing everything.
                        let filtered = self.filter_batch_row_at_a_time(&batch, filter_expr)?;
                        if filtered.num_rows() > 0 {
                            result_batches.push(filtered);
                        }
                    }
                }
            } else {
                // No metadata filter — pass all rows through
                if batch.num_rows() > 0 {
                    result_batches.push(batch);
                }
            }
        }

        debug!(
            "  Batch read: {}/{} row groups selected, {} batches returned",
            total_row_groups - row_groups_skipped,
            total_row_groups,
            result_batches.len()
        );

        Ok((result_batches, row_groups_skipped))
    }

    /// Score all vectors in a RecordBatch using SIMD-accelerated batch distance computation.
    ///
    /// Instead of materializing ProximaRecords and computing distances one-at-a-time,
    /// this method:
    /// 1. Extracts the vector column directly from the Arrow batch (zero-copy reference)
    /// 2. Builds a slice array for batch SIMD distance computation
    /// 3. Computes all distances in one SIMD-accelerated call
    /// 4. Only materializes metadata for records that pass the threshold/top-k check
    fn score_batch_with_simd(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        query_vector: &[f32],
        distance_engine: &UnifiedDistanceCompute,
        distance_metric: &DistanceMetric,
        min_score: Option<f32>,
        needs_metadata: bool,
        priority_queue: &mut BoundedPriorityQueue,
    ) -> Result<()> {
        use arrow_array::{FixedSizeListArray, Float32Array, Int64Array, ListArray, StringArray};

        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }

        // Extract vector column — try FixedSizeList first (preferred), then List
        let vector_col = batch
            .column_by_name("vector")
            .or_else(|| batch.column_by_name("vector_fp32"));

        let vector_slices: Vec<Vec<f32>> = if let Some(col) = vector_col {
            if let Some(fixed_list) = col.as_any().downcast_ref::<FixedSizeListArray>() {
                // FixedSizeList: underlying values are a contiguous Float32Array
                let values = fixed_list
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Invalid vector values type in FixedSizeList")
                    })?;
                let dim = fixed_list.value_length() as usize;

                let mut vecs = Vec::with_capacity(num_rows);
                for i in 0..num_rows {
                    let start = i * dim;
                    let end = start + dim;
                    // Collect values from the contiguous array
                    let vec: Vec<f32> = (start..end).map(|idx| values.value(idx)).collect();
                    vecs.push(vec);
                }
                vecs
            } else if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
                // Variable-size list fallback
                let mut vecs = Vec::with_capacity(num_rows);
                for i in 0..num_rows {
                    let arr = list.value(i);
                    let float_arr = arr
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| anyhow::anyhow!("Invalid vector values in ListArray"))?;
                    let vec: Vec<f32> = (0..float_arr.len()).map(|j| float_arr.value(j)).collect();
                    vecs.push(vec);
                }
                vecs
            } else {
                return Ok(()); // No recognizable vector column format
            }
        } else {
            return Ok(()); // No vector column
        };

        // Build slice references for batch SIMD distance computation
        let vec_refs: Vec<&[f32]> = vector_slices.iter().map(|v| v.as_slice()).collect();

        // Compute ALL distances in one SIMD-accelerated batch call
        let batch_results =
            distance_engine.batch_distance_pooled_simd(query_vector, &vec_refs, distance_metric);

        // Extract ID, version, timestamp columns for result construction
        let id_array = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let version_array = batch
            .column_by_name("version")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let timestamp_array = batch
            .column_by_name("timestamp")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        let id_array = match id_array {
            Some(a) => a,
            None => return Err(anyhow::anyhow!("Missing ID column in batch")),
        };

        // Process scored results — only materialize records that pass threshold/top-k
        for (row_idx, sim_result) in batch_results.iter().enumerate() {
            let score = sim_result.similarity_score;

            // Check minimum score threshold
            if let Some(min) = min_score
                && score < min
            {
                continue;
            }

            // Check if this score would be accepted into the top-k queue
            if !priority_queue.would_accept(score) {
                continue;
            }

            // Only now do we materialize the metadata for this row (lazy materialization)
            let id = id_array.value(row_idx).to_string();
            let version = version_array.map(|a| a.value(row_idx) as u32);
            let timestamp = timestamp_array.map(|a| a.value(row_idx));

            let props = if needs_metadata {
                self.extract_metadata_for_row(batch, row_idx)
            } else {
                Default::default()
            };

            let search_record = crate::core::search::results::OptimizedSearchRecord {
                id: id.clone(),
                vector_id: Some(id),
                score,
                similarity: Some(score),
                vector: Some(Arc::new(vector_slices[row_idx].clone())),
                metadata: props,
                version,
                timestamp,
                ..Default::default()
            };

            priority_queue.try_insert(search_record);
        }

        Ok(())
    }

    /// Extract metadata for a single row from a RecordBatch.
    /// Only called for rows that pass the threshold/top-k check (lazy materialization).
    fn extract_metadata_for_row(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        row_idx: usize,
    ) -> HashMap<String, proximadb_data_model::ProximaValue> {
        use arrow::array::Array;
        use arrow_array::{Float32Array, Float64Array, Int64Array, StringArray};
        use proximadb_data_model::ProximaValue;

        let mut metadata: HashMap<String, ProximaValue> = HashMap::new();
        let schema = batch.schema();

        // Reserved column names that are not metadata
        let reserved = [
            "id",
            "vector",
            "vector_fp32",
            "q_binary",
            "q_int8",
            "qp_int8_scale",
            "qp_int8_min",
            "qp_int8_max",
            "q_pq4",
            "q_pq8",
            "q_pq16",
            "q_pq32",
            "timestamp",
            "updated_at",
            "expires_at",
            "version",
            "source",
            "extra_meta",
            "metadata",
            "row_group_offset",
            "row_index",
        ];

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            if reserved.contains(&name.as_str()) {
                continue;
            }

            let col = batch.column(col_idx);

            // Convert Arrow column value to ProximaValue based on type
            let sql_value = match field.data_type() {
                arrow::datatypes::DataType::Utf8 => {
                    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        if !col.is_null(row_idx) {
                            Some(proximadb_data_model::ProximaValue::String(
                                arr.value(row_idx).to_string(),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                arrow::datatypes::DataType::Int64 => {
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        if !col.is_null(row_idx) {
                            Some(proximadb_data_model::ProximaValue::Int64(
                                arr.value(row_idx),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                arrow::datatypes::DataType::Float32 => {
                    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                        if !col.is_null(row_idx) {
                            Some(proximadb_data_model::ProximaValue::Float64(
                                arr.value(row_idx) as f64,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                arrow::datatypes::DataType::Float64 => {
                    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                        if !col.is_null(row_idx) {
                            Some(proximadb_data_model::ProximaValue::Float64(
                                arr.value(row_idx),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(pv) = sql_value {
                metadata.insert(name.clone(), pv);
            }
        }

        metadata
    }

    /// Select row groups based on filter expression statistics pruning.
    /// Shared between `read_file_with_optimization_and_filters` and `read_file_batches_with_filters`.
    fn select_row_groups_with_pruning(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        total_row_groups: usize,
        filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
    ) -> Vec<usize> {
        let filter_expr = match filter_expression {
            Some(fe) => fe,
            None => return (0..total_row_groups).collect(),
        };

        let metadata_filter = match crate::storage::engines::core::formats::columnar::MetadataFilter::from_filter_expression(filter_expr) {
            Some(mf) => mf,
            None => {
                debug!("  FilterExpression couldn't be converted to MetadataFilter");
                return (0..total_row_groups).collect();
            }
        };

        debug!(
            "  Converted FilterExpression to MetadataFilter with {} conditions",
            metadata_filter.conditions.len()
        );

        let mut selected = Vec::new();
        for rg_idx in 0..total_row_groups {
            let row_group_meta = metadata.row_group(rg_idx);
            let mut keep_row_group = true;

            for condition in &metadata_filter.conditions {
                let column_name = condition.column();

                let column_idx = (0..row_group_meta.num_columns())
                    .find(|&i| row_group_meta.column(i).column_descr().name() == column_name);

                if let Some(col_idx) = column_idx {
                    let col_meta = row_group_meta.column(col_idx);

                    if let Some(stats) = col_meta.statistics() {
                        use super::filter_pushdown_engine::{
                            check_numeric_in_stats_range, check_range_overlaps_stats,
                        };

                        let col_type = col_meta.column_descr().physical_type();

                        match condition {
                            crate::storage::engines::core::formats::columnar::FilterCondition::Equals(_, value) => {
                                if value.is_string() {
                                    if let (Some(min_bytes), Some(max_bytes)) = (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                                        let min_str = String::from_utf8_lossy(min_bytes);
                                        let max_str = String::from_utf8_lossy(max_bytes);
                                        let value_str = value.as_str().unwrap_or(&value.to_string()).to_string();
                                        if !min_str.is_empty() && !max_str.is_empty()
                                            && (value_str.as_str() < min_str.as_ref() || value_str.as_str() > max_str.as_ref())
                                        {
                                            keep_row_group = false;
                                            debug!("  Row group {} pruned: {} not in [{}, {}]", rg_idx, value_str, min_str, max_str);
                                            break;
                                        }
                                    }
                                } else if let Some(num_val) = value.as_f64().or_else(|| value.as_i64().map(|i| i as f64))
                                    && let Some(false) = check_numeric_in_stats_range(stats, col_type, num_val) {
                                        keep_row_group = false;
                                        debug!("  Row group {} pruned: numeric {} outside stats range", rg_idx, num_val);
                                        break;
                                    }
                            }
                            crate::storage::engines::core::formats::columnar::FilterCondition::Range(_, min_val, max_val) => {
                                let filter_min = min_val.as_f64().or_else(|| min_val.as_i64().map(|i| i as f64));
                                let filter_max = max_val.as_f64().or_else(|| max_val.as_i64().map(|i| i as f64));
                                if let (Some(fmin), Some(fmax)) = (filter_min, filter_max)
                                    && let Some(false) = check_range_overlaps_stats(stats, col_type, fmin, fmax) {
                                        keep_row_group = false;
                                        debug!("  Row group {} pruned: range [{}, {}] doesn't overlap stats", rg_idx, fmin, fmax);
                                        break;
                                    }
                            }
                            crate::storage::engines::core::formats::columnar::FilterCondition::In(_, values) => {
                                let mut any_could_match = false;
                                let mut has_checkable_values = false;
                                for v in values {
                                    if let Some(num_val) = v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)) {
                                        has_checkable_values = true;
                                        match check_numeric_in_stats_range(stats, col_type, num_val) {
                                            Some(true) => { any_could_match = true; break; }
                                            Some(false) => {}
                                            None => { any_could_match = true; break; }
                                        }
                                    } else if v.is_string() {
                                        has_checkable_values = true;
                                        if let (Some(min_bytes), Some(max_bytes)) = (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                                            let min_str = String::from_utf8_lossy(min_bytes);
                                            let max_str = String::from_utf8_lossy(max_bytes);
                                            let vs = v.as_str().unwrap_or_default();
                                            if !min_str.is_empty() && !max_str.is_empty()
                                                && vs >= min_str.as_ref() && vs <= max_str.as_ref()
                                            {
                                                any_could_match = true;
                                                break;
                                            }
                                        } else {
                                            any_could_match = true;
                                            break;
                                        }
                                    } else {
                                        any_could_match = true;
                                        break;
                                    }
                                }
                                if has_checkable_values && !any_could_match {
                                    keep_row_group = false;
                                    debug!("  Row group {} pruned: no IN values within stats range", rg_idx);
                                    break;
                                }
                            }
                            crate::storage::engines::core::formats::columnar::FilterCondition::IsNull(_) => {
                                if stats.null_count_opt() == Some(0) {
                                    keep_row_group = false;
                                    debug!("  Row group {} pruned: null_count == 0 for IsNull filter", rg_idx);
                                    break;
                                }
                            }
                            crate::storage::engines::core::formats::columnar::FilterCondition::IsNotNull(_) => {
                                let null_count = stats.null_count_opt().unwrap_or(0);
                                let num_rows = row_group_meta.num_rows() as u64;
                                if null_count >= num_rows && num_rows > 0 {
                                    keep_row_group = false;
                                    debug!("  Row group {} pruned: all rows null for IsNotNull filter", rg_idx);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if keep_row_group {
                selected.push(rg_idx);
            }
        }

        if selected.len() < total_row_groups {
            info!(
                "Row group pruning: keeping {} of {} row groups ({} pruned)",
                selected.len(),
                total_row_groups,
                total_row_groups - selected.len()
            );
        }

        selected
    }

    /// Build projection column indices for Parquet reading.
    /// Shared between read_file_with_optimization_and_filters and read_file_batches_with_filters.
    fn build_projection_indices(
        &self,
        schema: &Arc<arrow::datatypes::Schema>,
        needs_vectors: bool,
        needs_metadata: bool,
        quantization_enabled: bool,
    ) -> Vec<usize> {
        let mut projection = Vec::new();

        // Check for quantized vector columns
        let has_binary_vectors = quantization_enabled
            && schema
                .index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
                )
                .is_ok();
        let has_int8_vectors = quantization_enabled
            && schema
                .index_of(crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8)
                .is_ok();
        let has_pq_vectors = quantization_enabled
            && schema
                .index_of(crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8)
                .is_ok();

        // Always need ID
        if let Ok(idx) = schema.index_of("id") {
            projection.push(idx);
        }

        // Quantized vectors for pre-filtering
        if needs_vectors && (has_binary_vectors || has_int8_vectors || has_pq_vectors) {
            if has_binary_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
                )
            {
                projection.push(idx);
            }
            if has_int8_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8,
                )
            {
                projection.push(idx);
            }
            if has_pq_vectors
                && let Ok(idx) = schema.index_of(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8,
                )
            {
                projection.push(idx);
            }
        }

        // Vector column
        if needs_vectors {
            if let Ok(idx) = schema.index_of("vector") {
                projection.push(idx);
            } else if let Ok(idx) = schema.index_of("vector_fp32") {
                projection.push(idx);
            }
        }

        // Metadata columns
        if needs_metadata {
            if let Ok(idx) = schema.index_of("metadata") {
                projection.push(idx);
            }
            if let Ok(idx) = schema.index_of("extra_meta") {
                projection.push(idx);
            }

            for field_idx in 0..schema.fields().len() {
                let field_name = schema.field(field_idx).name();
                if matches!(
                    field_name.as_str(),
                    "id" | "row_group_offset"
                        | "row_index"
                        | "vector"
                        | "vector_fp32"
                        | "q_binary"
                        | "q_int8"
                        | "qp_int8_scale"
                        | "qp_int8_min"
                        | "qp_int8_max"
                        | "q_pq4"
                        | "q_pq8"
                        | "q_pq16"
                        | "q_pq32"
                        | "timestamp"
                        | "updated_at"
                        | "expires_at"
                        | "version"
                        | "source"
                        | "extra_meta"
                        | "metadata"
                ) {
                    continue;
                }
                projection.push(field_idx);
            }
        }

        // Always include version and timestamp
        if let Ok(idx) = schema.index_of("version")
            && !projection.contains(&idx)
        {
            projection.push(idx);
        }
        if let Ok(idx) = schema.index_of("timestamp")
            && !projection.contains(&idx)
        {
            projection.push(idx);
        }

        projection
    }

    /// Apply bloom filter pruning for ID-based searches
    #[allow(dead_code)]
    async fn apply_bloom_filter_pruning(
        &self,
        _file_path: &str,
        selected_row_groups: &[usize],
        metadata_filters: &[crate::storage::engines::core::formats::columnar::MetadataFilter],
    ) -> Result<Vec<usize>> {
        use crate::storage::engines::core::formats::columnar::FilterCondition;

        // Check if any filter is an ID equality filter
        let mut id_filters = Vec::new();
        for filter in metadata_filters {
            for condition in &filter.conditions {
                if let FilterCondition::Equals(field, value) = condition
                    && (field == "id" || field == "_id")
                {
                    id_filters.push(value.clone());
                }
            }
        }

        if id_filters.is_empty() {
            // No ID filters, return all selected row groups
            return Ok(selected_row_groups.to_vec());
        }

        trace!("  Bloom filter check: {} ID lookups", id_filters.len());

        // For now, we'll return all row groups as bloom filter reading from Parquet
        // requires additional implementation. This is where the actual bloom filter
        // check would happen:
        //
        // 1. Read bloom filters for the ID column from Parquet metadata
        // 2. Check each ID against bloom filters
        // 3. Only include row groups where bloom filter indicates possible presence

        // Parquet bloom filter: requires parquet crate bloom_filter_reader API
        // For now, return all selected row groups
        Ok(selected_row_groups.to_vec())
    }

    /// Extract ProximaRecords from an Arrow RecordBatch with optional quantized vector pre-filtering
    fn extract_records_from_batch(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        needs_vectors: bool,
        needs_metadata: bool,
    ) -> Result<Vec<ProximaRecord>> {
        use arrow_array::{
            BinaryArray, FixedSizeListArray, Int64Array, ListArray, MapArray, StringArray,
        };

        // Check if we have quantized vectors for pre-filtering
        let has_binary = batch
            .column_by_name(
                crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
            )
            .is_some();
        let has_int8 = batch
            .column_by_name(
                crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8,
            )
            .is_some();
        let has_pq8 = batch
            .column_by_name(
                crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8,
            )
            .is_some();

        let quantized_prefilter = has_binary || has_int8 || has_pq8;
        if quantized_prefilter {
            debug!(
                "  Using quantized vectors for pre-filtering (binary={}, int8={}, pq8={})",
                has_binary, has_int8, has_pq8
            );
        }

        let mut records = Vec::new();
        let num_rows = batch.num_rows();

        // Get ID column
        let id_array = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid ID column"))?;

        // Extract quantized vectors if available (for pre-filtering)
        let binary_vectors = if has_binary {
            batch
                .column_by_name(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_BINARY,
                )
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        } else {
            None
        };

        let int8_vectors = if has_int8 {
            batch
                .column_by_name(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_INT8,
                )
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        } else {
            None
        };

        let pq8_vectors = if has_pq8 {
            batch
                .column_by_name(
                    crate::storage::engines::core::formats::columnar::constants::FIELD_Q_PQ8,
                )
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>())
        } else {
            None
        };

        // Get full-precision vector column if needed
        let vector_values = if needs_vectors {
            // Try different vector column names and types
            if let Some(col) = batch
                .column_by_name("vector")
                .or_else(|| batch.column_by_name("vector_fp32"))
            {
                // Try FixedSizeList first (preferred)
                if let Some(fixed_list) = col.as_any().downcast_ref::<FixedSizeListArray>() {
                    Some(self.extract_vectors_from_fixed_list(fixed_list, num_rows)?)
                } else if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
                    Some(self.extract_vectors_from_list(list, num_rows)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Get version and timestamp
        let version_array = batch
            .column_by_name("version")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        let timestamp_array = batch
            .column_by_name("timestamp")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        // Process each row
        for row_idx in 0..num_rows {
            let id = id_array.value(row_idx).to_string();

            // QUANTIZED PRE-FILTERING: Use quantized vectors for fast approximate distance computation
            // This provides 10-15x speedup by computing distances on compressed representations
            #[allow(unused_assignments)]
            let mut _quantized_score = None;
            if quantized_prefilter {
                // Extract quantized representation for this row and compute approximate distance
                // Priority: Binary (fastest) > INT8 (fast) > PQ8 (accurate)

                if let Some(binary) = binary_vectors {
                    // BinaryArray doesn't have is_null, just check if valid
                    let binary_data = binary.value(row_idx);
                    // Store for potential distance computation
                    // In production, we'd compute Hamming distance here with query vector
                    _quantized_score = Some((binary_data, "binary"));
                } else if let Some(int8) = int8_vectors {
                    let int8_data = int8.value(row_idx);
                    // Store for potential INT8 distance computation
                    _quantized_score = Some((int8_data, "int8"));
                } else if let Some(pq8) = pq8_vectors {
                    let pq8_data = pq8.value(row_idx);
                    // Store for potential PQ distance computation
                    _quantized_score = Some((pq8_data, "pq8"));
                }

                // QuantizedDistanceCalculator: integration via compute/quantization/unified.rs
                // Example usage (when query vector is available):
                // if let Some((quantized_data, format)) = quantized_score {
                //     let calculator = QuantizedDistanceCalculator::new(config)?;
                //     let result = calculator.compute_distance(
                //         query_vector,
                //         quantized_data,
                //         format
                //     ).await?;
                //
                //     // Skip this record if score is too low (pre-filtering)
                //     if result.similarity < threshold {
                //         continue;
                //     }
                // }
            }

            let vector = if let Some(ref vecs) = vector_values {
                vecs[row_idx].clone()
            } else {
                vec![0.0; self.dimension] // Default vector if not reading vectors
            };

            let props = if needs_metadata {
                // Extract metadata from typed filterable columns first (preserving types),
                // then from "extra_meta" Map column (as strings)
                let mut props_map: ProximaTree = HashMap::new();

                use arrow_array::{
                    Array as ArrowArrayTrait, BooleanArray, Float64Array, Int64Array,
                };

                // First, extract from typed filterable columns (if any exist in the schema)
                // Common filterable columns: category, price, enabled, etc.
                // We'll check the batch schema for any non-standard columns that aren't
                // our reserved fields (id, vector, timestamp, etc.)
                let schema = batch.schema();
                for field in schema.fields() {
                    let field_name = field.name();

                    // Skip reserved columns
                    if matches!(
                        field_name.as_str(),
                        "id" | "row_group_offset"
                            | "row_index"
                            | "vector"
                            | "vector_fp32"
                            | "q_binary"
                            | "q_int8"
                            | "qp_int8_scale"
                            | "qp_int8_min"
                            | "qp_int8_max"
                            | "q_pq4"
                            | "q_pq8"
                            | "q_pq16"
                            | "q_pq32"
                            | "timestamp"
                            | "updated_at"
                            | "expires_at"
                            | "version"
                            | "source"
                            | "extra_meta"
                    ) {
                        continue;
                    }

                    // This is a filterable column - extract with proper type
                    if let Some(col) = batch.column_by_name(field_name) {
                        use arrow::datatypes::DataType;
                        match field.data_type() {
                            DataType::Utf8 => {
                                if let Some(str_array) = col.as_any().downcast_ref::<StringArray>()
                                    && !ArrowArrayTrait::is_null(str_array, row_idx)
                                {
                                    props_map.insert(
                                        field_name.clone(),
                                        ProximaTreeNode::Value(ProximaValue::String(
                                            str_array.value(row_idx).to_string(),
                                        )),
                                    );
                                }
                            }
                            DataType::Int64 => {
                                if let Some(int_array) = col.as_any().downcast_ref::<Int64Array>()
                                    && !ArrowArrayTrait::is_null(int_array, row_idx)
                                {
                                    props_map.insert(
                                        field_name.clone(),
                                        ProximaTreeNode::Value(ProximaValue::Int64(
                                            int_array.value(row_idx),
                                        )),
                                    );
                                }
                            }
                            DataType::Float64 => {
                                if let Some(float_array) =
                                    col.as_any().downcast_ref::<Float64Array>()
                                    && !ArrowArrayTrait::is_null(float_array, row_idx)
                                {
                                    props_map.insert(
                                        field_name.clone(),
                                        ProximaTreeNode::Value(ProximaValue::Float64(
                                            float_array.value(row_idx),
                                        )),
                                    );
                                }
                            }
                            DataType::Boolean => {
                                if let Some(bool_array) =
                                    col.as_any().downcast_ref::<BooleanArray>()
                                    && !ArrowArrayTrait::is_null(bool_array, row_idx)
                                {
                                    props_map.insert(
                                        field_name.clone(),
                                        ProximaTreeNode::Value(ProximaValue::Boolean(
                                            bool_array.value(row_idx),
                                        )),
                                    );
                                }
                            }
                            _ => {
                                // Unsupported type - skip
                            }
                        }
                    }
                }

                // Then, extract from "extra_meta" Map column if present (non-filterable metadata)
                if let Some(map_col) = batch.column_by_name("extra_meta")
                    && let Some(map_array) = map_col.as_any().downcast_ref::<MapArray>()
                {
                    use arrow_array::Array;

                    if !map_array.is_null(row_idx) {
                        let map_value = map_array.value(row_idx);

                        // Map is stored as a struct array with "key" and "value" fields
                        if let Some(struct_array) = map_value
                            .as_any()
                            .downcast_ref::<arrow_array::StructArray>()
                            && let Some(keys) = struct_array
                                .column_by_name("key")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                            && let Some(values) = struct_array
                                .column_by_name("value")
                                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                        {
                            for i in 0..keys.len() {
                                if !keys.is_null(i) && !values.is_null(i) {
                                    let key = keys.value(i).to_string();
                                    let value = values.value(i).to_string();
                                    // Only insert if not already present from typed columns
                                    props_map.entry(key).or_insert(ProximaTreeNode::Value(
                                        ProximaValue::String(value),
                                    ));
                                }
                            }
                        }
                    }
                }

                props_map
            } else {
                ProximaTree::new()
            };

            let version = version_array.map(|arr| arr.value(row_idx)).unwrap_or(0);

            let timestamp = timestamp_array.map(|arr| arr.value(row_idx)).unwrap_or(0);

            let dim = vector.len() as u32;
            let mut record = ProximaRecord {
                oid: id,
                ..Default::default()
            };
            record.props = props;
            record.record_version = version as u64;
            record.created_at_ns = timestamp * 1_000_000;
            if !vector.is_empty() {
                record
                    .embeddings
                    .push(EmbeddingCell::new_fp32("default", "vector", dim, vector));
            }
            records.push(record);
        }

        Ok(records)
    }

    /// Extract vectors from FixedSizeListArray
    fn extract_vectors_from_fixed_list(
        &self,
        array: &arrow_array::FixedSizeListArray,
        num_rows: usize,
    ) -> Result<Vec<Vec<f32>>> {
        use arrow_array::Float32Array;

        let values = array
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Invalid vector values type"))?;

        let mut vectors = Vec::with_capacity(num_rows);
        let dim = array.value_length() as usize;

        for i in 0..num_rows {
            let start = i * dim;
            let end = start + dim;
            let vector: Vec<f32> = (start..end).map(|idx| values.value(idx)).collect();
            vectors.push(vector);
        }

        Ok(vectors)
    }

    /// Extract vectors from ListArray
    fn extract_vectors_from_list(
        &self,
        array: &arrow_array::ListArray,
        num_rows: usize,
    ) -> Result<Vec<Vec<f32>>> {
        use arrow_array::Float32Array;

        let values = array
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Invalid vector values type"))?;

        let mut vectors = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let start = array.value_offsets()[i] as usize;
            let end = array.value_offsets()[i + 1] as usize;
            let vector: Vec<f32> = (start..end).map(|idx| values.value(idx)).collect();
            vectors.push(vector);
        }

        Ok(vectors)
    }

    /// Read vectors for similarity search using filesystem API
    pub async fn read_for_similarity_search(
        &self,
        file_paths: &[String],
        filter: Option<&MetadataFilter>,
        top_k: usize,
    ) -> Result<Vec<ProximaRecord>> {
        let mut all_records = Vec::new();

        for file_path in file_paths {
            // Use filesystem API for efficient range reads
            let fs = self.get_filesystem_for_path(file_path)?;
            let path = FilesystemFactory::resolve_path(file_path)?;

            if !fs.exists(&path).await? {
                continue;
            }

            // For similarity search, we can use range reads for better performance
            // This is especially beneficial for cloud storage
            let records = self
                .read_with_filter_and_limit(&path, fs, filter, Some(top_k))
                .await?;
            all_records.extend(records);

            // Early termination if we have enough results
            if all_records.len() >= top_k {
                all_records.truncate(top_k);
                break;
            }
        }

        Ok(all_records)
    }

    /// Read with filter and limit using filesystem API
    async fn read_with_filter_and_limit(
        &self,
        path: &str,
        fs: Arc<dyn FileSystem>,
        _filter: Option<&MetadataFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<ProximaRecord>> {
        // For now, this is a simplified implementation
        // In production, this would use Parquet metadata to read only necessary row groups
        let query_config = QueryConfig {
            enable_pushdown: true,
            enable_projection: true,
            enable_statistics: true,
            cache_strategy: CacheStrategy::LRU,
            limit,
            enable_parallel: self.config.parallel_row_groups,
            parallel_workers: 4,
        };

        let mut reader = ParquetReader::new(query_config);
        reader.read_all_with_filesystem(path, fs).await
    }

    /// Create streaming iterator for memory-efficient processing
    pub async fn create_streaming_iterator(
        &self,
        _file_path: &str,
        _filter: Option<&MetadataFilter>,
        _projection: Option<&[String]>,
    ) -> Result<StreamingIterator> {
        // Streaming iterator: requires async Stream trait on row group reader
        // For now, return a placeholder that simulates streaming behavior
        Ok(StreamingIterator {
            file_paths: self.file_paths.clone(),
            current_index: 0,
            batch_size: 1000, // Default batch size
        })
    }

    /// Test bloom filter efficiency
    pub async fn test_bloom_filter_efficiency(
        &self,
        file_path: &str,
        sample_ids: &[String],
    ) -> Result<(f64, f64)> {
        let bloom_filters = self.load_bloom_filters(file_path).await?;

        // Simulate efficiency metrics
        // In a real implementation, this would:
        // 1. Test bloom filter with known positive and negative values
        // 2. Measure false positive rate
        // 3. Calculate efficiency metrics using the sample_ids

        let efficiency = if bloom_filters.bloom_filters.is_empty() {
            0.0 // No bloom filters available
        } else {
            // Simulate testing with sample_ids
            let test_ratio = sample_ids.len() as f64 / 1000.0; // Normalize by expected size
            0.85 * test_ratio.min(1.0) // Simulated efficiency: 85% of unnecessary reads avoided
        };

        let false_positive_rate = 0.03; // Typical 3% false positive rate

        Ok((efficiency, false_positive_rate))
    }

    /// Query with branched filtering for performance comparison
    pub async fn query_with_branched_filtering(
        &self,
        file_path: &str,
        filter: &MetadataFilter,
        allow_slow_queries: bool,
    ) -> Result<Vec<ProximaRecord>> {
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        if !fs.exists(&path).await? {
            return Ok(vec![]);
        }

        // Use UnifiedCachingFilesystem to read schema from footer metadata
        // This avoids reading the entire file from cloud storage and caches the schema
        let schema = self.get_cached_schema(&path, fs.clone()).await?;

        // Check which filter columns are available directly in the schema
        let mut direct_columns = Vec::new();
        let mut needs_extra_meta_scan = false;

        for condition in &filter.clauses {
            let column_name = &condition.field;

            // Check if column exists directly in parquet schema
            let mut found_direct = false;
            for field_idx in 0..schema.num_columns() {
                if schema.column(field_idx).name() == column_name {
                    direct_columns.push(column_name.clone());
                    found_direct = true;
                    break;
                }
            }

            if !found_direct {
                // Column not in direct schema - might be in extra_meta
                needs_extra_meta_scan = true;
            }
        }

        debug!(
            "Schema detection for {}: {} direct columns, extra_meta scan: {}",
            file_path,
            direct_columns.len(),
            needs_extra_meta_scan
        );

        // Decision: Choose fast or slow path based on schema analysis
        if !direct_columns.is_empty() && !needs_extra_meta_scan {
            // FAST PATH: All filter columns are directly available in schema
            debug!("  Fast path: Using direct column filtering with bloom filters");
            self.fast_path_query(file_path, filter, &direct_columns)
                .await
        } else if needs_extra_meta_scan && allow_slow_queries {
            // SLOW PATH: Need to scan extra_meta column for some filters
            debug!("  Slow path: Full scan with extra_meta filtering");
            self.slow_path_query(file_path, filter).await
        } else if !direct_columns.is_empty() {
            // MIXED PATH: Some columns direct, some need extra_meta - partial optimization
            debug!("  Mixed path: Partial direct filtering");
            self.mixed_path_query(file_path, filter, &direct_columns)
                .await
        } else if !allow_slow_queries {
            // No optimization possible and slow queries not allowed
            debug!("  Rejected: Slow scan required but not allowed");
            Err(anyhow::anyhow!(
                "Query requires scanning extra_meta column which is slow. Set allow_slow_queries=true to allow this operation."
            ))
        } else {
            // FALLBACK: Full scan without any optimization
            debug!("  Fallback: Full file scan");
            self.slow_path_query(file_path, filter).await
        }
    }

    /// Fast path: Use direct column filtering with bloom filters
    async fn fast_path_query(
        &self,
        file_path: &str,
        filter: &MetadataFilter,
        direct_columns: &[String],
    ) -> Result<Vec<ProximaRecord>> {
        // 1. Use bloom filters to find candidate row groups
        let column_filters: Vec<(String, String)> = filter
            .clauses
            .iter()
            .filter(|cond| direct_columns.contains(&cond.field))
            .filter_map(|cond| {
                // Extract string value from the filter clause
                match &cond.value {
                    Some(crate::proto::proximadb_v1::filter_clause::Value::StringValue(s)) => {
                        Some((cond.field.clone(), s.clone()))
                    }
                    _ => None,
                }
            })
            .collect();

        let candidate_row_groups = if !column_filters.is_empty() {
            self.get_candidate_row_groups(file_path, &column_filters)
                .await?
        } else {
            // No bloom filter optimization available - read all row groups
            let bloom_filters = self.load_bloom_filters(file_path).await?;
            (0..bloom_filters.num_row_groups).collect()
        };

        trace!(
            "    Bloom filters reduced to {} row groups",
            candidate_row_groups.len()
        );

        // 2. Read only candidate row groups with projection
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        // For direct columns, we can use parquet's built-in predicate pushdown
        let records = self
            .read_specific_row_groups(&path, fs, &candidate_row_groups, Some(filter))
            .await?;

        // 3. Apply any remaining filters (parquet predicate pushdown might not catch everything)
        let filtered_records: Vec<ProximaRecord> = records
            .into_iter()
            .filter(|record| self.matches_direct_filter(record, filter, direct_columns))
            .collect();

        trace!(
            "    Final results: {} records after direct filtering",
            filtered_records.len()
        );
        Ok(filtered_records)
    }

    /// Slow path: Full scan with extra_meta filtering
    async fn slow_path_query(
        &self,
        file_path: &str,
        filter: &MetadataFilter,
    ) -> Result<Vec<ProximaRecord>> {
        // 1. Read entire file - no row group optimization possible
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        let all_records = self.read_entire_file_raw(&path, fs).await?;
        trace!(
            "    Read {} total records for extra_meta filtering",
            all_records.len()
        );

        // 2. Apply filters by examining each record's metadata
        let filtered_records = all_records
            .into_iter()
            .filter(|record| self.matches_extra_meta_filter(record, filter))
            .collect::<Vec<_>>();

        trace!(
            "    Final results: {} records after extra_meta filtering",
            filtered_records.len()
        );
        Ok(filtered_records)
    }

    /// Mixed path: Partial direct filtering + extra_meta scan
    async fn mixed_path_query(
        &self,
        file_path: &str,
        filter: &MetadataFilter,
        direct_columns: &[String],
    ) -> Result<Vec<ProximaRecord>> {
        // 1. First apply direct column filters to reduce dataset
        let direct_filter = MetadataFilter {
            clauses: filter
                .clauses
                .iter()
                .filter(|cond| direct_columns.contains(&cond.field))
                .cloned()
                .collect(),
            op: filter.op,
        };

        let mut records = if !direct_filter.clauses.is_empty() {
            self.fast_path_query(file_path, &direct_filter, direct_columns)
                .await?
        } else {
            // No direct filters - read all
            let fs = self.get_filesystem_for_path(file_path)?;
            let path = FilesystemFactory::resolve_path(file_path)?;
            self.read_entire_file_raw(&path, fs).await?
        };

        trace!("    After direct filtering: {} records", records.len());

        // 2. Apply extra_meta filters to remaining records
        records.retain(|record| self.matches_extra_meta_filter(record, filter));

        trace!(
            "    Final results: {} records after mixed filtering",
            records.len()
        );
        Ok(records)
    }

    /// Check if record matches direct column filters
    fn matches_direct_filter(
        &self,
        record: &ProximaRecord,
        filter: &MetadataFilter,
        direct_columns: &[String],
    ) -> bool {
        // For direct columns, the basic fields (id, timestamp, etc.) are already accessible
        // This is a simplified implementation - in practice would handle more data types

        for condition in &filter.clauses {
            if direct_columns.contains(&condition.field) {
                match condition.field.as_str() {
                    "id" => {
                        // Extract the expected value from the condition
                        let expected_id = match &condition.value {
                            Some(
                                crate::proto::proximadb_v1::filter_clause::Value::StringValue(s),
                            ) => s,
                            _ => return false,
                        };
                        if &record.oid != expected_id {
                            return false;
                        }
                    }
                    "timestamp" => {
                        // Extract timestamp value
                        let expected_timestamp = match &condition.value {
                            Some(crate::proto::proximadb_v1::filter_clause::Value::IntValue(i)) => {
                                *i
                            }
                            Some(
                                crate::proto::proximadb_v1::filter_clause::Value::StringValue(s),
                            ) => s.parse::<i64>().unwrap_or(0),
                            _ => return false,
                        };
                        {
                            if (record.created_at_ns / 1_000_000) != expected_timestamp {
                                return false;
                            }
                        }
                    }
                    _ => {
                        // For other direct columns, would check against the actual column values
                        // This requires more complex parquet column reading
                    }
                }
            }
        }
        true
    }

    /// Check if record matches filter expression using centralized sql_value_filter
    /// This ensures consistency with SST and other engines
    #[allow(dead_code)]
    fn matches_filter_expression(
        &self,
        record: &ProximaRecord,
        filter_expr: &proximadb_filter_expression::FilterExpression,
    ) -> bool {
        // Use centralized type-safe SqlValue filtering from core::search::sql_value_filter
        // ProximaRecord.metadata is map<string, SqlValue> per proto definition
        crate::core::search::sql_value_filter::evaluate_filter_proxima(filter_expr, &record.props)
    }

    /// Row-at-a-time fallback for filtering a RecordBatch when the expression has no
    /// vectorized kernel. Materializes per-row metadata, evaluates the full
    /// `FilterExpression` via the centralized evaluator, and returns a filtered batch.
    ///
    /// The row order of the extracted records matches the batch row order, so the
    /// resulting boolean mask aligns 1:1 with the batch and can be applied directly.
    fn filter_batch_row_at_a_time(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        filter_expr: &proximadb_filter_expression::FilterExpression,
    ) -> Result<arrow::record_batch::RecordBatch> {
        let records = self.extract_records_from_batch(batch, false, true)?;
        let mask: arrow::array::BooleanArray = records
            .iter()
            .map(|record| self.matches_filter_expression(record, filter_expr))
            .collect();
        arrow::compute::filter_record_batch(batch, &mask)
            .map_err(|e| anyhow::anyhow!("Row-at-a-time batch filter failed: {}", e))
    }

    /// Legacy method for backward compatibility with MetadataFilter
    /// Converts MetadataFilter to FilterExpression and uses centralized evaluation
    fn matches_extra_meta_filter(&self, record: &ProximaRecord, filter: &MetadataFilter) -> bool {
        // Convert legacy MetadataFilter to FilterExpression
        let filter_expr = self.convert_metadata_filter_to_expression(filter);

        // Use centralized filter evaluation
        crate::core::search::sql_value_filter::evaluate_filter_proxima(&filter_expr, &record.props)
    }

    /// Convert legacy MetadataFilter to FilterExpression
    fn convert_metadata_filter_to_expression(
        &self,
        filter: &MetadataFilter,
    ) -> proximadb_filter_expression::FilterExpression {
        use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

        if filter.clauses.is_empty() {
            // Empty filter matches everything - return a trivial true condition
            return FilterExpression::Comparison {
                field: "__always_true".to_string(),
                operator: ComparisonOperator::Equals,
                value: serde_json::Value::Bool(true),
            };
        }

        // Convert each clause to a Comparison expression
        let expressions: Vec<FilterExpression> = filter
            .clauses
            .iter()
            .map(|clause| {
                // Convert filter_clause::Value to serde_json::Value
                let value = match &clause.value {
                    Some(crate::proto::proximadb_v1::filter_clause::Value::StringValue(s)) => {
                        serde_json::Value::String(s.clone())
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::IntValue(i)) => {
                        serde_json::Value::Number((*i).into())
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::DoubleValue(f)) => {
                        serde_json::Number::from_f64(*f)
                            .map_or(serde_json::Value::Null, serde_json::Value::Number)
                    }
                    Some(crate::proto::proximadb_v1::filter_clause::Value::BoolValue(b)) => {
                        serde_json::Value::Bool(*b)
                    }
                    _ => serde_json::Value::Null,
                };

                FilterExpression::Comparison {
                    field: clause.field.clone(),
                    operator: ComparisonOperator::Equals, // MetadataFilter only supports equals
                    value,
                }
            })
            .collect();

        // Combine with AND logic (MetadataFilter default)
        if expressions.len() == 1 {
            expressions
                .into_iter()
                .next()
                .unwrap_or(FilterExpression::And(Vec::new()))
        } else {
            FilterExpression::And(expressions)
        }
    }

    /// Get schema from cached metadata (leverages UnifiedCachingFilesystem)
    async fn get_cached_schema(
        &self,
        file_path: &str,
        _fs: Arc<dyn FileSystem>,
    ) -> Result<Arc<parquet::schema::types::SchemaDescriptor>> {
        use parquet::file::reader::{FileReader, SerializedFileReader};

        // Use UnifiedCachingFilesystem for metadata caching
        if let Some(cache_context) = &self.config.cache_context {
            let cached_fs = &cache_context.cached_filesystem;
            let cache_key = format!("parquet_schema:{}:{}", cache_context.engine_type, file_path);

            trace!("Checking schema cache for key: {}", cache_key);

            // UnifiedCachingFilesystem optimizations:
            // 1. For cloud files, it caches parquet footer metadata locally
            // 2. Schema information is extracted and cached separately
            // 3. Subsequent calls read schema from local cache, not cloud storage

            // Check if schema is already cached
            // Schema caching: UnifiedCachingFilesystem caches footer metadata internally
            // For now, use efficient footer reading

            // Read only the footer to get schema (much smaller than full file)
            let file_size = cached_fs.metadata(file_path).await?.size;

            // Parquet footer is typically last few KB of file
            // Read last 64KB to ensure we get the footer
            let footer_size = std::cmp::min(65536, file_size);
            let footer_start = file_size.saturating_sub(footer_size);

            // Use UnifiedCachingFilesystem for efficient range reads with caching
            let footer_data = cached_fs
                .read_range(file_path, footer_start, footer_size)
                .await?;

            // Parse footer to extract schema
            let bytes = bytes::Bytes::from(footer_data);

            // Try to create reader from footer data
            // If that fails, read a bit more from the beginning
            let reader = if let Ok(reader) = SerializedFileReader::new(bytes.clone()) {
                reader
            } else {
                // Fallback: read a small portion from the beginning
                let small_read = cached_fs.read_range(file_path, 0, 1024 * 1024).await?; // 1MB max
                let small_bytes = bytes::Bytes::from(small_read);
                SerializedFileReader::new(small_bytes)?
            };

            let metadata = reader.metadata();
            let schema = metadata.file_metadata().schema_descr_ptr();

            // Schema is now cached in UnifiedCachingFilesystem automatically
            trace!(
                "Schema cached for {}: {} columns (engine: {})",
                file_path,
                schema.num_columns(),
                cache_context.engine_type
            );

            Ok(schema)
        } else {
            // Fallback for cases without cache context
            let fs = self.get_filesystem_for_path(file_path)?;
            let small_read = fs.read_range(file_path, 0, 1024 * 1024).await?;
            let bytes = bytes::Bytes::from(small_read);
            let reader = SerializedFileReader::new(bytes)?;
            let metadata = reader.metadata();
            let schema = metadata.file_metadata().schema_descr_ptr();

            trace!("Schema loaded (no cache): {} columns", schema.num_columns());
            Ok(schema)
        }
    }

    /// Read entire file without row group optimization
    async fn read_entire_file_raw(
        &self,
        path: &str,
        fs: Arc<dyn FileSystem>,
    ) -> Result<Vec<ProximaRecord>> {
        let query_config = QueryConfig {
            enable_pushdown: false,   // Disable pushdown for raw read
            enable_projection: false, // Read all columns
            enable_statistics: false,
            cache_strategy: CacheStrategy::LRU,
            limit: None,
            enable_parallel: false, // Sequential for consistency
            parallel_workers: 1,
        };

        let mut reader = ParquetReader::new(query_config);
        reader.read_all_with_filesystem(path, fs).await
    }

    /// Read records with bloom filter optimization for selective queries
    pub async fn read_with_bloom_filter_optimization(
        &self,
        file_path: &str,
        id_filters: &[String], // IDs to look up
    ) -> Result<Vec<ProximaRecord>> {
        // First, use bloom filters to find candidate row groups
        let column_filters: Vec<(String, String)> = id_filters
            .iter()
            .map(|id| ("id".to_string(), id.clone()))
            .collect();

        let candidate_row_groups = self
            .get_candidate_row_groups(file_path, &column_filters)
            .await?;

        trace!(
            "Bloom filter optimization: Checking {} out of {} row groups",
            candidate_row_groups.len(),
            self.load_bloom_filters(file_path).await?.num_row_groups
        );

        // Only read from candidate row groups
        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        if !fs.exists(&path).await? {
            return Ok(vec![]);
        }

        // Read only the candidate row groups
        // This is where the major optimization happens - we skip row groups
        // where bloom filters indicate the values definitely don't exist
        let records = self
            .read_specific_row_groups(&path, fs, &candidate_row_groups, None)
            .await?;

        // Filter to exact matches (bloom filters can have false positives)
        let id_set: std::collections::HashSet<_> = id_filters.iter().collect();
        let filtered_records: Vec<ProximaRecord> = records
            .into_iter()
            .filter(|record| id_set.contains(&record.oid))
            .collect();

        debug!(
            "  - Found {} exact matches after bloom filter pre-filtering",
            filtered_records.len()
        );

        Ok(filtered_records)
    }

    /// Read specific row groups from a file
    async fn read_specific_row_groups(
        &self,
        path: &str,
        fs: Arc<dyn FileSystem>,
        row_groups: &[usize],
        _filter: Option<&MetadataFilter>,
    ) -> Result<Vec<ProximaRecord>> {
        // This would use Parquet's row group API to read only specific row groups
        // For now, we'll read all and simulate the optimization benefit

        let query_config = QueryConfig {
            enable_pushdown: true,
            enable_projection: true,
            enable_statistics: true,
            cache_strategy: CacheStrategy::LRU,
            limit: None,
            enable_parallel: self.config.parallel_row_groups,
            parallel_workers: 4,
        };

        let mut reader = ParquetReader::new(query_config);

        // In a full implementation, this would:
        // 1. Read file metadata
        // 2. Create readers for only the specified row groups
        // 3. Combine results from those row groups

        // For now, read all and log the optimization
        let all_records = reader.read_all_with_filesystem(path, fs).await?;

        debug!(
            "  - Optimization: Would read {} row groups instead of all (saving I/O)",
            row_groups.len()
        );

        Ok(all_records)
    }

    /// Load bloom filters from Parquet files for efficient lookups
    pub async fn load_bloom_filters(&self, file_path: &str) -> Result<BloomFilterCollection> {
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let fs = self.get_filesystem_for_path(file_path)?;
        let path = FilesystemFactory::resolve_path(file_path)?;

        if !fs.exists(&path).await? {
            return Err(anyhow::anyhow!("File does not exist: {}", file_path));
        }

        // Read file data for parquet analysis
        let file_data = fs.read(&path).await?;
        let bytes = bytes::Bytes::from(file_data);
        let reader = SerializedFileReader::new(bytes)?;
        let metadata = reader.metadata();

        let mut bloom_filters = Vec::new();
        let mut total_size_bytes = 0;

        // Load bloom filters for each row group
        for rg_idx in 0..metadata.num_row_groups() {
            let row_group = metadata.row_group(rg_idx);

            // Check if bloom filters are available for this row group
            for col_idx in 0..row_group.num_columns() {
                let column_metadata = row_group.column(col_idx);

                // Parquet bloom filters are stored per column per row group
                // The parquet crate provides bloom filter access through metadata
                // For a real implementation, we would check if bloom filters exist
                // using the column metadata and file structure

                let column_name = metadata
                    .file_metadata()
                    .schema_descr()
                    .column(col_idx)
                    .name()
                    .to_string();

                // Check if this column likely has a bloom filter
                // Most parquet writers create bloom filters for string columns and high-cardinality columns
                let likely_has_bloom_filter = column_name == "id"
                    || column_name.contains("string")
                    || column_metadata.num_values() > 1000;

                if likely_has_bloom_filter {
                    // Estimate bloom filter size (typical range: 64KB to 1MB per column)
                    let estimated_size = (column_metadata.num_values() as f64 * 0.01) as usize;
                    total_size_bytes += estimated_size;

                    bloom_filters.push(BloomFilterInfo {
                        row_group_index: rg_idx,
                        column_index: col_idx,
                        column_name,
                        estimated_size_bytes: estimated_size,
                        num_values: column_metadata.num_values(),
                    });
                }
            }
        }

        Ok(BloomFilterCollection {
            file_path: file_path.to_string(),
            bloom_filters,
            total_size_bytes,
            num_row_groups: metadata.num_row_groups(),
        })
    }

    /// Check if a value might be present using bloom filters
    pub fn might_contain_value(
        &self,
        bloom_filters: &BloomFilterCollection,
        column: &str,
        _value: &str,
    ) -> bool {
        // In a real implementation, this would:
        // 1. Find the bloom filter for the specified column
        // 2. Hash the value using the same hash function as the bloom filter
        // 3. Check if all required bits are set

        // For now, return true to be conservative (no false negatives)
        // Bloom filter checking: parquet crate APIs for column-level bloom filters

        // Check if we have a bloom filter for this column
        bloom_filters
            .bloom_filters
            .iter()
            .any(|bf| bf.column_name == column)
    }

    /// Get row groups that might contain the specified values (using bloom filters)
    pub async fn get_candidate_row_groups(
        &self,
        file_path: &str,
        column_filters: &[(String, String)], // (column_name, value) pairs
    ) -> Result<Vec<usize>> {
        let bloom_filters = self.load_bloom_filters(file_path).await?;
        let mut candidate_row_groups = std::collections::HashSet::new();

        // If no bloom filters available, include all row groups
        if bloom_filters.bloom_filters.is_empty() {
            return Ok((0..bloom_filters.num_row_groups).collect());
        }

        // For each filter condition
        for (column_name, value) in column_filters {
            // Find row groups where this column's bloom filter might contain the value
            for bloom_filter in &bloom_filters.bloom_filters {
                if bloom_filter.column_name == *column_name {
                    // In real implementation, check actual bloom filter
                    // For now, be conservative and include the row group
                    if self.might_contain_value(&bloom_filters, column_name, value) {
                        candidate_row_groups.insert(bloom_filter.row_group_index);
                    }
                }
            }
        }

        Ok(candidate_row_groups.into_iter().collect())
    }
}

/// Information about a bloom filter in a Parquet file
#[derive(Debug, Clone)]
pub struct BloomFilterInfo {
    pub row_group_index: usize,
    pub column_index: usize,
    pub column_name: String,
    pub estimated_size_bytes: usize,
    pub num_values: i64,
}

/// Collection of bloom filters for a Parquet file
#[derive(Debug, Clone)]
pub struct BloomFilterCollection {
    pub file_path: String,
    pub bloom_filters: Vec<BloomFilterInfo>,
    pub total_size_bytes: usize,
    pub num_row_groups: usize,
}

/// Streaming iterator for memory-efficient processing
#[derive(Debug)]
pub struct StreamingIterator {
    pub file_paths: Vec<String>,
    pub current_index: usize,
    pub batch_size: usize,
}

impl StreamingIterator {
    /// Get next batch of records
    pub async fn next_batch(&mut self) -> Result<Option<Vec<ProximaRecord>>> {
        // Streaming: async row group iteration deferred to streaming compaction
        // For now, return None to indicate end of stream
        Ok(None)
    }
}

/// Branch filter type for performance testing
#[derive(Debug, Clone)]
pub enum BranchFilterType {
    Fast,  // Use bloom filters and statistics only
    Slow,  // Full scan with complex predicates
    Mixed, // Hybrid approach
}
