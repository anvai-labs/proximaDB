// NOVA Columnar Search - Progressive columnar search with Parquet optimization
// Implements UnifiedStorageFormat's search_vectors_unified interface
// NOVA-specific optimizations for analytics workloads

use anyhow::{Result, anyhow};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use tracing::{debug, info};

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::engines::core::formats::columnar::{
    MetadataFilter, columnar_query_engine::columnar_query_reader::UnifiedParquetReader,
};
use proximadb_distance_kernel::{DistanceMetric, engine::UnifiedDistanceCompute};

use super::{
    NovaFile,
    progressive_search::{ProgressiveColumnarSearch, ProgressiveSearchConfig},
    streaming_processor::{StreamingConfig, StreamingRowGroupProcessor},
};

/// Configuration for columnar search in NOVA
#[derive(Debug, Clone)]
pub struct ColumnarSearchConfig {
    /// Enable predicate pushdown to Parquet
    pub enable_predicate_pushdown: bool,

    /// Enable row group pruning based on statistics
    pub enable_row_group_pruning: bool,

    /// Enable column projection
    pub enable_projection: bool,

    /// Enable progressive search stages
    pub enable_progressive_search: bool,

    /// Enable streaming for memory efficiency
    pub enable_streaming: bool,

    /// Maximum candidates to process
    pub max_candidates: usize,

    /// Search mode
    pub search_mode: SearchMode,

    /// Memory budget in bytes
    pub memory_budget: Option<usize>,

    /// Latency budget in milliseconds
    pub latency_budget_ms: Option<u64>,
}

impl Default for ColumnarSearchConfig {
    fn default() -> Self {
        Self {
            enable_predicate_pushdown: true,
            enable_row_group_pruning: true,
            enable_projection: true,
            enable_progressive_search: true,
            enable_streaming: true,
            max_candidates: 10000,
            search_mode: SearchMode::Progressive,
            memory_budget: Some(1024 * 1024 * 1024), // 1GB default
            latency_budget_ms: Some(1000),           // 1 second default
        }
    }
}

/// Search modes for NOVA
#[derive(Debug, Clone, PartialEq)]
pub enum SearchMode {
    /// Full precision search (no quantization)
    FullPrecision,
    /// Progressive search with quantization stages
    Progressive,
    /// Streaming search for large datasets
    Streaming,
    /// Hybrid mode (adaptive based on query)
    Hybrid,
}

/// Search candidate for columnar processing
///
/// Represents a potential match found during columnar search, containing
/// location information and similarity score for ranking.
#[derive(Debug, Clone)]
struct SearchCandidate {
    /// Row group identifier in the Parquet file
    row_group_id: usize,
    /// Row offset within the row group
    row_offset: u32,
    /// Similarity score (higher is better)
    similarity: f32,
    /// Optional vector record identifier
    vector_id: Option<String>,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity && self.row_group_id == other.row_group_id
    }
}

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap (best candidates first)
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Main columnar search implementation for NOVA
pub struct NovaColumnarSearch {
    config: ColumnarSearchConfig,
    parquet_reader: Arc<UnifiedParquetReader>,
    distance_compute: Arc<UnifiedDistanceCompute>,
    progressive_search: Option<Arc<ProgressiveColumnarSearch>>,
    streaming_processor: Option<Arc<StreamingRowGroupProcessor>>,
}

impl NovaColumnarSearch {
    /// Create new columnar search engine
    pub async fn new(
        config: ColumnarSearchConfig,
        parquet_reader: Arc<UnifiedParquetReader>,
        distance_compute: Arc<UnifiedDistanceCompute>,
    ) -> Result<Self> {
        // Initialize progressive search if enabled
        let progressive_search = if config.enable_progressive_search {
            let prog_config = ProgressiveSearchConfig::from(&config);
            // Create quantization engine for progressive search
            let quant_distance_compute = Arc::new(UnifiedDistanceCompute::default());
            let quant_config =
                crate::compute::quantization::storage_engine::StorageQuantizationConfig::default();
            // Create codebook store for UnifiedQuantizationEngine
            let codebook_store = Arc::new(
                crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
            );
            let unified_quant_engine = Arc::new(
                crate::compute::quantization::UnifiedQuantizationEngine::new(
                    quant_distance_compute.clone(),
                    codebook_store,
                ),
            );
            let _ = Arc::new(
                crate::compute::quantization::storage_engine::StorageQuantizationEngine::new(
                    unified_quant_engine.clone(),
                    quant_distance_compute.clone(),
                    quant_config,
                ),
            );
            Some(Arc::new(ProgressiveColumnarSearch::new(
                prog_config,
                DistanceMetric::Euclidean, // Default, will be overridden per search
                quant_distance_compute,
                unified_quant_engine,
            )))
        } else {
            None
        };

        // Initialize streaming processor if enabled
        let streaming_processor = if config.enable_streaming {
            let stream_config = StreamingConfig::from(&config);
            Some(Arc::new(StreamingRowGroupProcessor::new(stream_config)))
        } else {
            None
        };

        Ok(Self {
            config,
            parquet_reader,
            distance_compute,
            progressive_search,
            streaming_processor,
        })
    }

    /// Main entry point for NOVA's unified search
    pub async fn search_nova(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        filter: Option<&MetadataFilter>,
        _search_params: Option<&serde_json::Value>,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        info!(
            "NOVA columnar search: dimension={}, top_k={}, mode={:?}",
            query_vector.len(),
            top_k,
            self.config.search_mode
        );

        // Select search strategy based on mode
        match self.config.search_mode {
            SearchMode::Progressive => {
                self.search_progressive(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
            SearchMode::Streaming => {
                self.search_streaming(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
            SearchMode::FullPrecision => {
                self.search_full_precision(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
            SearchMode::Hybrid => {
                self.search_hybrid(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
        }
    }

    /// Progressive search with quantization stages
    async fn search_progressive(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        let _progressive_search = self
            .progressive_search
            .as_ref()
            .ok_or_else(|| anyhow!("Progressive search not initialized"))?;

        debug!("Starting progressive search");

        // Stage 1: Binary filtering
        let binary_candidates = self
            .binary_filter_stage(
                nova_file,
                query_vector,
                top_k * 100, // Expand for binary stage
                distance_metric,
                filter,
            )
            .await?;

        // Stage 2: INT8 filtering
        let int8_candidates = self
            .int8_filter_stage(
                nova_file,
                query_vector,
                &binary_candidates,
                top_k * 20, // Narrow down
                distance_metric,
            )
            .await?;

        // Stage 3: PQ filtering
        let pq_candidates = self
            .pq_filter_stage(
                nova_file,
                query_vector,
                &int8_candidates,
                top_k * 5, // Further narrow
                distance_metric,
            )
            .await?;

        // Stage 4: Full precision reranking
        let final_results = self
            .full_precision_stage(
                nova_file,
                query_vector,
                &pq_candidates,
                top_k,
                distance_metric,
            )
            .await?;

        Ok(final_results)
    }

    /// Streaming search for memory efficiency
    async fn search_streaming(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        let _streaming_processor = self
            .streaming_processor
            .as_ref()
            .ok_or_else(|| anyhow!("Streaming processor not initialized"))?;

        info!(
            "Starting streaming search across {} row groups",
            nova_file.row_groups.len()
        );

        // Create streaming context
        let (tx, mut rx) = mpsc::channel(100);
        let semaphore = Arc::new(Semaphore::new(4)); // Limit concurrent row groups

        // Process row groups in parallel with backpressure
        let mut handles = Vec::new();
        for (rg_idx, row_group) in nova_file.row_groups.iter().enumerate() {
            if !self.should_process_row_group(row_group, filter)? {
                continue;
            }

            let sem = semaphore.clone();
            let tx = tx.clone();
            let query = query_vector.to_vec();
            let metric = distance_metric;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.ok();

                // Process row group with streaming
                let candidates =
                    process_row_group_streaming("", rg_idx, &query, metric, top_k * 2).await;

                if let Ok(candidates) = candidates {
                    let _ = tx.send(candidates).await;
                }
            });

            handles.push(handle);
        }

        // Drop original sender to close channel when done
        drop(tx);

        // Collect results with priority queue for top-k
        let mut heap = BinaryHeap::new();
        while let Some(candidates) = rx.recv().await {
            for (record, distance) in candidates {
                // Create SimilarityResult using constructor
                let similarity_result = proximadb_distance_kernel::engine::SimilarityResult::new(
                    distance,
                    distance_metric,
                );

                heap.push(SearchCandidate {
                    row_group_id: 0,
                    row_offset: 0,
                    similarity: similarity_result.rank_value, // Use rank_value for proper ordering
                    vector_id: Some(record.id.clone()),
                });

                // Keep only top candidates in memory
                if heap.len() > top_k * 2 {
                    heap.pop();
                }
            }
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Extract final results
        let mut results = Vec::new();
        while let Some(candidate) = heap.pop() {
            if results.len() >= top_k {
                break;
            }

            // Load full record for final results
            if let Some(record) = self
                .load_record_by_id(nova_file, &candidate.vector_id)
                .await?
            {
                results.push((record, 1.0 - candidate.similarity));
            }
        }

        results.reverse(); // Best first
        Ok(results)
    }

    /// Full precision search without quantization
    async fn search_full_precision(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        debug!(
            "Full precision search across {} row groups",
            nova_file.row_groups.len()
        );

        let mut all_candidates = Vec::new();

        // Process each row group
        for (rg_idx, row_group) in nova_file.row_groups.iter().enumerate() {
            if !self.should_process_row_group(row_group, filter)? {
                continue;
            }

            // Load row group with projection
            let _projection = if self.config.enable_projection {
                Some(vec!["id".to_string(), "vector".to_string()])
            } else {
                None
            };

            let batch = self
                .parquet_reader
                .read_row_groups_projected(
                    &nova_file.metadata.collection_id,
                    &[rg_idx],
                    _projection.as_deref(),
                )
                .await?;

            // Compute distances for all vectors in batch
            // batch is Vec<VectorRecord>, process each record
            for record in batch {
                // Create distance compute instance and compute distance
                let distance_compute =
                    proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(distance_metric);
                let distance_result = distance_compute.calculate_distance(
                    query_vector,
                    record
                        .embeddings
                        .first()
                        .map_or(&[][..], |embedding| embedding.as_fp32_slice()),
                    &distance_metric,
                );
                let distance = distance_result.distance;

                all_candidates.push(SearchCandidate {
                    row_group_id: rg_idx,
                    row_offset: 0, // Will be calculated when proper indexing is implemented
                    similarity: 1.0 - distance,
                    vector_id: Some(record.oid.clone()),
                });
            }
        }

        // Use bounded priority queue for efficient top-k selection
        let mut priority_queue = BoundedPriorityQueue::new(top_k);

        for candidate in all_candidates {
            let search_record = crate::core::search::results::OptimizedSearchRecord {
                id: candidate.vector_id.clone().unwrap_or_default(),
                vector_id: candidate.vector_id.clone(),
                score: candidate.similarity,
                similarity: Some(1.0 - candidate.similarity), // Convert similarity back to distance
                vector: None,                                 // Will be loaded later
                metadata: Default::default(),
                debug_info: None,
                version: None,
                timestamp: None,
                updated_at: None,
                expires_at: None,
                source: None,
                expanded_context: vec![],
                semantic_similarity: None,
                quantization_info: None,
                engine_stats: None,
                index_path: None,
                ..Default::default()
            };
            priority_queue.try_insert(search_record);
        }

        // Get top candidates from bounded queue
        let top_candidates = priority_queue.into_sorted_vec();

        // Load full records for top candidates
        let mut results = Vec::new();
        for candidate in &top_candidates {
            if let Some(record) = self
                .load_record_by_id(nova_file, &candidate.vector_id)
                .await?
            {
                results.push((record, candidate.score));
            }
        }

        Ok(results)
    }

    /// Hybrid search with adaptive strategy selection
    async fn search_hybrid(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        top_k: usize,
        distance_metric: DistanceMetric,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        // Analyze query characteristics
        let query_complexity = self.analyze_query_complexity(query_vector, filter);
        let dataset_size = nova_file.metadata.num_vectors;

        // Select strategy based on analysis
        let strategy = if dataset_size > 1_000_000 && query_complexity > 0.5 {
            SearchMode::Streaming // Large dataset, complex query
        } else if dataset_size > 100_000 {
            SearchMode::Progressive // Medium dataset
        } else {
            SearchMode::FullPrecision // Small dataset
        };

        info!("Hybrid search selected strategy: {:?}", strategy);

        // Execute selected strategy
        match strategy {
            SearchMode::Progressive => {
                self.search_progressive(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
            SearchMode::Streaming => {
                self.search_streaming(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
            _ => {
                self.search_full_precision(nova_file, query_vector, top_k, distance_metric, filter)
                    .await
            }
        }
    }

    // Helper methods for progressive stages

    async fn binary_filter_stage(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        max_candidates: usize,
        distance_metric: DistanceMetric,
        _filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchCandidate>> {
        debug!("Binary filter stage: max_candidates={}", max_candidates);

        let mut candidates = BinaryHeap::new();

        // Check if binary column exists - if not, fall back to full vector scan
        let use_binary_filter = nova_file.quantized_columns.binary_column.is_some();

        if !use_binary_filter {
            debug!(
                "Binary column not available, falling back to full vector scan for initial candidates"
            );
        }

        // Track pruning statistics
        let mut row_groups_pruned = 0;
        let total_row_groups = nova_file.row_groups.len();

        // Process each row group
        for (rg_idx, _row_group) in nova_file.row_groups.iter().enumerate() {
            // Use enhanced stats for zone map pruning if available
            if let Some(_enhanced_stats) = nova_file.enhanced_stats.get(rg_idx) {
                // Use zone map intersection check for pruning
                // Dynamic threshold: use k-th best distance if we have enough candidates,
                // otherwise use a generous initial threshold based on metric
                let metric_name = match distance_metric {
                    DistanceMetric::Euclidean => "euclidean",
                    DistanceMetric::Cosine => "cosine",
                    DistanceMetric::DotProduct => "dot_product",
                    _ => "euclidean", // Default fallback
                }
                .to_string();

                // Calculate dynamic threshold with expansion factor for safety margin
                let current_threshold = if candidates.len() >= max_candidates {
                    // Use the worst (k-th) candidate's distance with 50% expansion
                    // This ensures we don't prune too aggressively
                    let kth_best = candidates
                        .peek()
                        .map_or(f32::MAX, |c: &SearchCandidate| 1.0 - c.similarity);
                    kth_best * 1.5
                } else {
                    // Initial generous threshold based on distance metric
                    match distance_metric {
                        DistanceMetric::Cosine => 2.0,      // Cosine distance range: [0, 2]
                        DistanceMetric::DotProduct => 10.0, // Generous for dot product
                        _ => {
                            // Euclidean: sqrt(dimensions) is a reasonable upper bound for normalized vectors
                            let dim = query_vector.len() as f32;
                            dim.sqrt() * 2.0
                        }
                    }
                };

                if !_enhanced_stats.vector_zone_map.intersects_query(
                    query_vector,
                    metric_name.clone(),
                    current_threshold,
                ) {
                    debug!(
                        "Skipping row group {} via zone map pruning (threshold={:.4})",
                        rg_idx, current_threshold
                    );
                    row_groups_pruned += 1;
                    continue;
                }
            }

            // Load vectors (binary or full depending on availability)
            let column_to_load = if use_binary_filter {
                "vector_binary"
            } else {
                "vector" // Fall back to full vectors
            };

            let batch = self
                .parquet_reader
                .read_row_groups_projected(
                    &nova_file.metadata.collection_id,
                    &[rg_idx],
                    Some(&[column_to_load.to_string()]),
                )
                .await?;

            // Process records
            for (row_offset, record) in batch.into_iter().enumerate() {
                let similarity = if use_binary_filter {
                    // Binary filtering: compute Hamming distance
                    // For now, use a placeholder until quantized columns are stored
                    let hamming_distance = 0.0;
                    1.0 - (hamming_distance / 256.0)
                } else {
                    // Full vector mode: compute actual distance
                    let vector = record
                        .embeddings
                        .first()
                        .map_or(&[][..], |embedding| embedding.as_fp32_slice());
                    if !vector.is_empty() {
                        let distance = self.distance_compute.calculate_distance(
                            query_vector,
                            vector,
                            &distance_metric,
                        );
                        distance.normalized_score
                    } else {
                        0.0
                    }
                };

                candidates.push(SearchCandidate {
                    row_group_id: rg_idx,
                    row_offset: row_offset as u32,
                    similarity,
                    vector_id: Some(record.oid.clone()),
                });

                if candidates.len() > max_candidates {
                    candidates.pop();
                }
            }
        }

        let result = candidates.into_sorted_vec();
        debug!(
            "Binary filter stage completed: {} candidates (binary_filter={})",
            result.len(),
            use_binary_filter
        );
        if row_groups_pruned > 0 {
            info!(
                "Zone map pruning: skipped {} of {} row groups ({:.1}% pruned)",
                row_groups_pruned,
                total_row_groups,
                (row_groups_pruned as f64 / total_row_groups as f64) * 100.0
            );
        }
        Ok(result)
    }

    async fn int8_filter_stage(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        candidates: &[SearchCandidate],
        max_candidates: usize,
        _distance_metric: DistanceMetric,
    ) -> Result<Vec<SearchCandidate>> {
        debug!(
            "INT8 filter stage: input={}, max={}",
            candidates.len(),
            max_candidates
        );

        if nova_file.quantized_columns.int8_column.is_none() {
            return Ok(candidates.to_vec());
        }

        // Quantize query to INT8
        let _query_int8 = quantize_to_int8(query_vector);

        let mut refined_candidates = BinaryHeap::new();

        // Group candidates by row group for batch processing
        let mut grouped: HashMap<usize, Vec<&SearchCandidate>> = HashMap::new();
        for candidate in candidates {
            grouped
                .entry(candidate.row_group_id)
                .or_default()
                .push(candidate);
        }

        // Process each row group
        for (rg_idx, group_candidates) in grouped {
            // Load INT8 column
            let batch = self
                .parquet_reader
                .read_row_groups_projected(
                    &nova_file.metadata.collection_id,
                    &[rg_idx],
                    Some(&["vector_int8".to_string()]),
                )
                .await?;

            // Compute INT8 distances for candidates
            for _record in batch {
                // Check if record has int8 quantized vector
                // VectorRecord doesn't have quantized field in proto
                // Deferred: Implement proper quantized vector access
                // For now, skip int8 processing
                {
                    for candidate in &group_candidates {
                        // Skip int8 distance computation for now
                        let int8_distance = 0.0;

                        refined_candidates.push(SearchCandidate {
                            row_group_id: candidate.row_group_id,
                            row_offset: candidate.row_offset,
                            similarity: 1.0 - int8_distance,
                            vector_id: candidate.vector_id.clone(),
                        });

                        if refined_candidates.len() > max_candidates {
                            refined_candidates.pop();
                        }
                    }
                    // Removed extra brace - not needed since we removed if let
                }
            }
        }

        Ok(refined_candidates.into_sorted_vec())
    }

    async fn pq_filter_stage(
        &self,
        nova_file: &NovaFile,
        _query_vector: &[f32],
        candidates: &[SearchCandidate],
        max_candidates: usize,
        _distance_metric: DistanceMetric,
    ) -> Result<Vec<SearchCandidate>> {
        debug!(
            "PQ filter stage: input={}, max={}",
            candidates.len(),
            max_candidates
        );

        if nova_file.quantized_columns.pq_column.is_none() {
            return Ok(candidates.to_vec());
        }

        // Prepare PQ distance table for query
        // Deferred: compute_pq_distance_table function not found - commented out
        // let _pq_table = compute_pq_distance_table(query_vector, 32, 256);

        let mut refined_candidates = BinaryHeap::new();

        // Group by row group
        let mut grouped: HashMap<usize, Vec<&SearchCandidate>> = HashMap::new();
        for candidate in candidates {
            grouped
                .entry(candidate.row_group_id)
                .or_default()
                .push(candidate);
        }

        // Process each row group
        for (rg_idx, group_candidates) in grouped {
            // Load PQ column
            let batch = self
                .parquet_reader
                .read_row_groups_projected(
                    &nova_file.metadata.collection_id,
                    &[rg_idx],
                    Some(&["vector_pq".to_string()]),
                )
                .await?;

            // Compute PQ distances
            for _record in batch {
                // Check if record has PQ quantized vector
                // VectorRecord doesn't have quantized field in proto
                // Deferred: Implement proper quantized vector access
                // For now, skip PQ processing
                {
                    for candidate in &group_candidates {
                        // Skip PQ distance computation for now
                        let pq_distance = 0.0;

                        refined_candidates.push(SearchCandidate {
                            row_group_id: candidate.row_group_id,
                            row_offset: candidate.row_offset,
                            similarity: 1.0 - pq_distance,
                            vector_id: candidate.vector_id.clone(),
                        });

                        if refined_candidates.len() > max_candidates {
                            refined_candidates.pop();
                        }
                    }
                    // Removed extra brace - not needed since we removed if let
                }
            }
        }

        Ok(refined_candidates.into_sorted_vec())
    }

    async fn full_precision_stage(
        &self,
        nova_file: &NovaFile,
        query_vector: &[f32],
        candidates: &[SearchCandidate],
        top_k: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<(VectorRecord, f32)>> {
        debug!(
            "Full precision stage: input={}, top_k={}",
            candidates.len(),
            top_k
        );

        let mut final_results = Vec::new();

        // Group by row group
        let mut grouped: HashMap<usize, Vec<&SearchCandidate>> = HashMap::new();
        for candidate in candidates {
            grouped
                .entry(candidate.row_group_id)
                .or_default()
                .push(candidate);
        }

        // Process each row group
        for (rg_idx, group_candidates) in grouped {
            // Load full vectors
            let batch = self
                .parquet_reader
                .read_row_groups_projected(
                    &nova_file.metadata.collection_id,
                    &[rg_idx],
                    None, // Load all columns for final stage
                )
                .await?;

            // Compute exact distances using batch processing
            // Collect all vectors from candidates for batch processing
            let mut batch_records = Vec::new();

            for candidate in &group_candidates {
                if let Some(record) =
                    // Find record by ID or index
                    batch
                        .iter()
                        .find(|r| &r.oid == candidate.vector_id.as_ref().unwrap_or(&String::new()))
                        .cloned()
                {
                    batch_records.push(record);
                }
            }

            // Batch compute distances if we have vectors
            if !batch_records.is_empty() {
                // Collect vector references after all records are collected
                let batch_vectors: Vec<&[f32]> = batch_records
                    .iter()
                    .map(|r| {
                        r.embeddings
                            .first()
                            .map_or(&[][..], |embedding| embedding.as_fp32_slice())
                    })
                    .collect();

                let distance_compute =
                    proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(distance_metric);
                let distances = distance_compute.batch_distance_pooled_simd(
                    query_vector,
                    &batch_vectors,
                    &distance_metric,
                );

                // Combine records with distances
                for (record, distance_result) in batch_records.into_iter().zip(distances.iter()) {
                    final_results.push((
                        crate::proto::proximadb_v1::VectorRecord::from(&record),
                        distance_result.distance,
                    ));
                }
            }
        }

        // Sort and take top-k
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(top_k);

        Ok(final_results)
    }

    // Helper methods

    fn should_process_row_group(
        &self,
        _row_group: &parquet::file::metadata::RowGroupMetaData,
        filter: Option<&MetadataFilter>,
    ) -> Result<bool> {
        if !self.config.enable_row_group_pruning {
            return Ok(true);
        }

        // Check row group statistics against filter
        if let Some(_filter) = filter {
            // Implement row group pruning logic based on statistics
            // For now, return true (process all)
            Ok(true)
        } else {
            Ok(true)
        }
    }

    #[allow(dead_code)]
    fn compute_batch_distances(
        &self,
        batch: &RecordBatch,
        query_vector: &[f32],
        distance_metric: DistanceMetric,
        row_group_id: usize,
    ) -> Result<Vec<(SearchCandidate, f32)>> {
        let mut candidates = Vec::new();

        // Get vector column
        let vector_col = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow!("Vector column not found"))?;

        // Get ID column if available
        let id_col = batch
            .column_by_name("id")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>());

        // Process each row
        for row_idx in 0..batch.num_rows() {
            // Extract vector (simplified - would need proper conversion)
            let vector = extract_vector_from_column(vector_col, row_idx)?;

            // Compute distance and get proper SimilarityResult
            let similarity_result =
                self.distance_compute
                    .calculate_distance(query_vector, &vector, &distance_metric);

            let vector_id = id_col.map(|arr| arr.value(row_idx).to_string());

            candidates.push((
                SearchCandidate {
                    row_group_id,
                    row_offset: row_idx as u32,
                    similarity: similarity_result.rank_value, // Use rank_value for proper ordering
                    vector_id,
                },
                similarity_result.raw_value, // Keep raw distance for processing
            ));
        }

        Ok(candidates)
    }

    async fn load_record_by_id(
        &self,
        nova_file: &NovaFile,
        vector_id: &Option<String>,
    ) -> Result<Option<VectorRecord>> {
        if let Some(id) = vector_id {
            // Use ID index or scan to find record
            // Simplified implementation
            Ok(Some(VectorRecord {
                id: id.clone(),
                vector: vec![0.0; nova_file.metadata.dimension],
                metadata: std::collections::HashMap::new(),
                timestamp: Some(0),
                updated_at: None,
                expires_at: None,
                version: None,
                source: None,
            }))
        } else {
            Ok(None)
        }
    }

    #[allow(dead_code)]
    fn extract_record_from_batch(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> Result<Option<VectorRecord>> {
        if row_idx >= batch.num_rows() {
            return Ok(None);
        }

        // Extract ID
        let id = batch
            .column_by_name("id")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .map(|arr| arr.value(row_idx).to_string());

        // Extract vector
        let vector = batch
            .column_by_name("vector")
            .map(|col| extract_vector_from_column(col, row_idx))
            .transpose()?
            .clone();

        // Extract other fields as needed
        Ok(Some(VectorRecord {
            id: id.unwrap_or_else(|| format!("unknown_{}", row_idx)),
            vector: vector.unwrap_or_default(),
            metadata: std::collections::HashMap::new(),
            timestamp: Some(0),
            updated_at: None,
            expires_at: None,
            version: None,
            source: None, // No source information available from Arrow batch
        }))
    }

    fn analyze_query_complexity(
        &self,
        query_vector: &[f32],
        filter: Option<&MetadataFilter>,
    ) -> f32 {
        let mut complexity = 0.0;

        // Check vector sparsity
        let non_zero = query_vector.iter().filter(|&&v| v != 0.0).count();
        complexity += (non_zero as f32) / (query_vector.len() as f32);

        // Check filter complexity
        if let Some(filter) = filter {
            complexity += filter.conditions.len() as f32 * 0.1;
        }

        complexity.min(1.0)
    }
}

// Helper functions for quantization stages

fn quantize_to_int8(vector: &[f32]) -> Vec<i8> {
    // Find min/max for scaling
    let min = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let scale = 255.0 / (max - min);

    vector
        .iter()
        .map(|&v| ((v - min) * scale - 128.0) as i8)
        .collect()
}

/// Extract a vector from an Arrow column at the specified row index
/// Handles Float32Array and other vector representations
#[allow(dead_code)]
fn extract_vector_from_column(column: &ArrayRef, row_idx: usize) -> Result<Vec<f32>> {
    // Try Float32Array first
    if let Some(float_array) = column.as_any().downcast_ref::<Float32Array>()
        && row_idx < float_array.len()
        && float_array.value(row_idx).is_finite()
    {
        // For now, return a placeholder
        // In production, would properly extract the vector
        return Ok(vec![float_array.value(row_idx); 768]);
    }

    // Try other formats (FixedSizeBinary, etc.)
    // Placeholder implementation
    Ok(vec![0.0; 768])
}

async fn process_row_group_streaming(
    _file_path: &str,
    _row_group_idx: usize,
    _query_vector: &[f32],
    _distance_metric: DistanceMetric,
    _max_candidates: usize,
) -> Result<Vec<(VectorRecord, f32)>> {
    // Simplified streaming processing
    // In production, would stream through row group with memory bounds
    Ok(Vec::new())
}

// Extension methods for config conversion

impl From<&ColumnarSearchConfig> for ProgressiveSearchConfig {
    fn from(config: &ColumnarSearchConfig) -> Self {
        ProgressiveSearchConfig {
            binary_config: Default::default(),
            int8_config: Default::default(),
            pq_config: Default::default(),
            full_precision_config: Default::default(),
            streaming_config: StreamingConfig::from(config),
            cost_based_ordering: true,
            adaptive_thresholds: true,
            enable_superblock_pruning: true,
            quality_target: 0.95,
            latency_budget_ms: config.latency_budget_ms,
            memory_budget_bytes: config.memory_budget,
        }
    }
}

impl From<&ColumnarSearchConfig> for StreamingConfig {
    fn from(config: &ColumnarSearchConfig) -> Self {
        StreamingConfig {
            max_memory_bytes: config.memory_budget.unwrap_or(1024 * 1024 * 1024),
            prefetch_queue_size: 2,
            max_concurrent_processors: 4,
            processing_timeout: std::time::Duration::from_secs(30),
            batch_size: 1000,
            enable_backpressure: true,
            backpressure_threshold: 0.8,
        }
    }
}

impl ColumnarSearchConfig {
    /// Create config from search parameters
    pub fn from_params(params: Option<&serde_json::Value>) -> Self {
        if let Some(params) = params {
            // Parse parameters from JSON
            let mut config = Self::default();

            if let Some(mode) = params.get("search_mode").and_then(|v| v.as_str()) {
                config.search_mode = match mode {
                    "progressive" => SearchMode::Progressive,
                    "streaming" => SearchMode::Streaming,
                    "full_precision" => SearchMode::FullPrecision,
                    "hybrid" => SearchMode::Hybrid,
                    _ => SearchMode::Progressive,
                };
            }

            if let Some(max) = params.get("max_candidates").and_then(|v| v.as_u64()) {
                config.max_candidates = max as usize;
            }

            if let Some(budget) = params.get("memory_budget_mb").and_then(|v| v.as_u64()) {
                config.memory_budget = Some((budget * 1024 * 1024) as usize);
            }

            config
        } else {
            Self::default()
        }
    }
}

/// Build projection mask based on config and filter
#[allow(dead_code)]
fn build_projection_mask(
    _config: &ColumnarSearchConfig,
    filter: &Option<MetadataFilter>,
) -> Vec<String> {
    let mut projection = vec!["id".to_string(), "vector".to_string()];

    // Add quantized columns if enabled
    if _config.enable_projection {
        projection.push("vector_binary".to_string());
        projection.push("vector_int8".to_string());
        projection.push("vector_pq".to_string());
    }

    // Add filter columns
    if let Some(filter) = filter {
        for condition in &filter.conditions {
            let column = condition.column();
            if !projection.contains(&column.to_string()) {
                projection.push(column.to_string());
            }
        }
    }

    projection
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(SearchCandidate {
            row_group_id: 0,
            row_offset: 0,
            similarity: 10.0,
            vector_id: None,
        });

        heap.push(SearchCandidate {
            row_group_id: 0,
            row_offset: 1,
            similarity: 5.0,
            vector_id: None,
        });

        heap.push(SearchCandidate {
            row_group_id: 0,
            row_offset: 2,
            similarity: 15.0,
            vector_id: None,
        });

        // Should pop in order: 5.0, 10.0, 15.0 (lowest similarity first for min-heap)
        assert_eq!(heap.pop().unwrap().similarity, 5.0);
        assert_eq!(heap.pop().unwrap().similarity, 10.0);
        assert_eq!(heap.pop().unwrap().similarity, 15.0);
    }

    #[test]
    fn test_projection_mask() {
        let config = ColumnarSearchConfig::default();

        // Create a filter for testing
        let filter = Some(MetadataFilter {
            conditions: vec![],
            logic: crate::storage::engines::core::formats::columnar::FilterLogic::And,
        });

        let projection = build_projection_mask(&config, &filter);
        assert!(projection.contains(&"id".to_string()));
        assert!(projection.contains(&"vector".to_string()));
        assert!(projection.contains(&"vector_binary".to_string()));
    }
}
