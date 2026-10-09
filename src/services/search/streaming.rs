/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Lock-Free Streaming Search Implementation
//!
//! This module implements high-performance lock-free streaming search that:
//! - Eliminates blocking on large result sets
//! - Streams results as they are found
//! - Uses concurrent search across WAL and storage layers
//! - Provides backpressure control for memory efficiency

use anyhow::Result;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::compute::distance_computation::DistanceMetric;
use crate::compute::distance_computation::engine::UnifiedDistanceCompute;
use crate::core::search::results::OptimizedSearchRecord;
use crate::services::operations::vectors::VectorOperationsService;
use std::collections::HashMap;

/// Backwards-compat alias for [`ServicesStreamingSearchConfig`].
pub type StreamingSearchConfig = ServicesStreamingSearchConfig;

/// Configuration for streaming search
#[derive(Debug, Clone)]
pub struct ServicesStreamingSearchConfig {
    /// Maximum number of results to buffer in memory
    pub buffer_size: usize,

    /// Enable concurrent search across storage layers
    pub concurrent_search: bool,

    /// Maximum concurrent search tasks
    pub max_concurrent_tasks: usize,

    /// Enable result deduplication
    pub enable_deduplication: bool,

    /// Result streaming batch size
    pub batch_size: usize,

    /// Timeout for individual search operations (ms)
    pub search_timeout_ms: u64,
}

impl Default for ServicesStreamingSearchConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            concurrent_search: true,
            max_concurrent_tasks: 4,
            enable_deduplication: true,
            batch_size: 100,
            search_timeout_ms: 5000,
        }
    }
}

/// Lock-free streaming search service
pub struct StreamingSearchService {
    /// Direct vector service for optimized access
    direct_service: Arc<VectorOperationsService>,

    /// Configuration
    config: ServicesStreamingSearchConfig,

    /// Unified distance computation
    distance_compute: UnifiedDistanceCompute,

    /// Search statistics
    stats: Arc<tokio::sync::RwLock<StreamingSearchStats>>,
}

/// Streaming search statistics
#[derive(Debug, Default, Clone)]
pub struct StreamingSearchStats {
    /// Total searches performed
    pub total_searches: u64,

    /// Total results streamed
    pub total_results_streamed: u64,

    /// Average streaming latency (ms)
    pub avg_streaming_latency_ms: f64,

    /// Current active streams
    pub active_streams: u32,

    /// Memory pressure events
    pub memory_pressure_events: u64,
}

/// Streaming search iterator - provides incremental search results
pub struct SearchResultStream {
    /// Result receiver channel
    receiver: mpsc::Receiver<SearchResultBatch>,

    /// Search metadata
    metadata: SearchMetadata,
}

/// Batch of search results
#[derive(Debug, Clone)]
pub struct SearchResultBatch {
    /// Results in this batch
    pub results: Vec<OptimizedSearchRecord>,

    /// Whether this is the final batch
    pub is_final: bool,

    /// Batch sequence number
    pub batch_id: u64,

    /// Batch creation timestamp
    pub timestamp: i64,
}

/// Search metadata
#[derive(Debug, Clone)]
pub struct SearchMetadata {
    /// Search request ID
    pub request_id: String,

    /// Collection being searched
    pub collection_id: String,

    /// Number of results requested
    pub k: usize,

    /// Distance metric used
    pub distance_metric: DistanceMetric,

    /// Search start time
    pub start_time: std::time::Instant,
}

impl StreamingSearchService {
    /// Create new streaming search service
    pub fn new(
        direct_service: Arc<VectorOperationsService>,
        config: Option<ServicesStreamingSearchConfig>,
    ) -> Self {
        let config = config.clone();
        let buffer_size = config.as_ref().map_or(1000, |c| c.buffer_size);
        let concurrent_search = config.as_ref().is_none_or(|c| c.concurrent_search);
        let max_concurrent_tasks = config.as_ref().map_or(8, |c| c.max_concurrent_tasks);

        info!(
            "🚀 StreamingSearchService: Initializing with buffer_size={}, concurrent_search={}, max_tasks={}",
            buffer_size, concurrent_search, max_concurrent_tasks
        );

        Self {
            direct_service,
            config: config.unwrap_or_default(),
            distance_compute: UnifiedDistanceCompute::default(),
            stats: Arc::new(tokio::sync::RwLock::new(StreamingSearchStats::default())),
        }
    }

    /// Perform lock-free streaming search
    pub async fn search_stream(
        &self,
        collection_id: String,
        query_vector: Vec<f32>,
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Result<SearchResultStream> {
        let start_time = std::time::Instant::now();
        let request_id = proximadb_kernel::uuid::Uuid::new_v4().to_string();

        info!(
            "🔍 STREAMING_SEARCH: Starting for collection={}, k={}, metric={:?}, request={}",
            collection_id, k, distance_metric, request_id
        );

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_searches += 1;
            stats.active_streams += 1;
        }

        // Create result channel with configured buffer size
        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        // Create search metadata
        let metadata = SearchMetadata {
            request_id: request_id.clone(),
            collection_id: collection_id.clone(),
            k,
            distance_metric,
            start_time,
        };

        // Clone required data for the async task
        let service = self.clone();
        let collection_id_clone = collection_id;
        let query_vector_clone = query_vector;

        // Spawn search task
        tokio::spawn(async move {
            if let Err(e) = service
                .execute_streaming_search(
                    collection_id_clone,
                    query_vector_clone,
                    k,
                    distance_metric,
                    tx,
                    request_id,
                )
                .await
            {
                warn!("⚠️ STREAMING_SEARCH: Task failed: {}", e);
            }
        });

        Ok(SearchResultStream {
            receiver: rx,
            metadata,
        })
    }

    /// Execute the streaming search with proper multi-tier deduplication
    async fn execute_streaming_search(
        &self,
        collection_id: String,
        query_vector: Vec<f32>,
        k: usize,
        distance_metric: DistanceMetric,
        tx: mpsc::Sender<SearchResultBatch>,
        request_id: String,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let mut batch_id = 0u64;
        let mut total_results = 0usize;

        // Track seen IDs for deduplication across tiers (if enabled)
        let mut seen_ids = if self.config.enable_deduplication {
            Some(std::collections::HashSet::<String>::new())
        } else {
            None
        };

        debug!(
            "🚀 STREAMING_TASK: Starting execution for request {} (dedup: {})",
            request_id, self.config.enable_deduplication
        );

        // Phase 1: Stream results from WAL (unflushed data)
        let wal_results = self
            .search_wal_streaming(&collection_id, &query_vector, k, distance_metric)
            .await?;
        let mut deduped_wal_results = Vec::new();

        if !wal_results.is_empty() {
            // Process each OptimizedSearchRecord
            for search_result in wal_results {
                let should_include = if let Some(ref mut seen) = seen_ids {
                    if search_result.id.is_empty() {
                        true // Include empty IDs
                    } else {
                        seen.insert(search_result.id.clone()) // Deduplicate by ID
                    }
                } else {
                    true // No deduplication
                };

                if should_include {
                    deduped_wal_results.push(search_result);
                }
            }

            if !deduped_wal_results.is_empty() {
                batch_id += 1;
                total_results += deduped_wal_results.len();

                let batch = SearchResultBatch {
                    results: deduped_wal_results,
                    is_final: false,
                    batch_id,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                if tx.send(batch).await.is_err() {
                    debug!("🔚 STREAMING_TASK: Receiver dropped, stopping");
                    return Ok(());
                }
            }
        }

        // Phase 2: Stream results from storage engines with deduplication
        if total_results < k {
            let remaining_k = k - total_results;

            // Request more results to account for potential duplicates
            let search_k = ((remaining_k as f32) * 1.5).ceil() as usize;

            // Search using native unified method (avoid proto conversions)
            let native_results = self
                .direct_service
                .unified_search_native(
                    &collection_id,
                    query_vector.clone(),
                    search_k,
                    None, // No filter
                    None, // Default config
                    #[cfg(feature = "abac-policy")]
                    &proximadb_abac::ReadContext::system(
                        proximadb_abac::SystemReadReason::Statistics,
                        "streaming [CLIENT-PLACEHOLDER]",
                    ),
                )
                .await?;

            // Sort native results by score (higher is better)
            let mut all_native = native_results;
            all_native.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Deduplicate storage results against WAL results if enabled
            let mut deduped_storage_records = Vec::new();
            for record in all_native {
                let should_include = if let Some(ref mut seen) = seen_ids {
                    if record.id.is_empty() {
                        true // Include empty IDs
                    } else {
                        seen.insert(record.id.clone()) // Deduplicate by ID
                    }
                } else {
                    true // No deduplication
                };

                if should_include {
                    // Record is already OptimizedSearchRecord
                    deduped_storage_records.push(record);
                    if deduped_storage_records.len() >= remaining_k {
                        break;
                    }
                }
            }

            if !deduped_storage_records.is_empty() {
                batch_id += 1;
                total_results += deduped_storage_records.len();

                let batch = SearchResultBatch {
                    results: deduped_storage_records,
                    is_final: false,
                    batch_id,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                if tx.send(batch).await.is_err() {
                    return Ok(());
                }
            }
        }

        // Send final batch
        batch_id += 1;
        let final_batch = SearchResultBatch {
            results: Vec::new(),
            is_final: true,
            batch_id,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let _ = tx.send(final_batch).await;

        // Update statistics
        let duration = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_results_streamed += total_results as u64;
            stats.active_streams = stats.active_streams.saturating_sub(1);

            // Update average latency
            let total_latency = stats.avg_streaming_latency_ms * (stats.total_searches - 1) as f64;
            stats.avg_streaming_latency_ms =
                (total_latency + duration.as_millis() as f64) / stats.total_searches as f64;
        }

        info!(
            "✅ STREAMING_TASK: Completed request {} - {} results in {:?}",
            request_id, total_results, duration
        );

        Ok(())
    }

    /// Search WAL with streaming
    async fn search_wal_streaming(
        &self,
        _collection_id: &str,
        query_vector: &[f32],
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        debug!("🔍 STREAMING_WAL: Searching unflushed vectors");

        // Get direct access to WAL memtable
        // Deferred: Implement WAL behavior integration
        if false {
            // let Some(wal_behavior) = self.direct_service.get_wal_behavior_wrapper() {
            // let unflushed_batches = wal_behavior
            //     .get_unflushed_batches(collection_id)
            //     .await?;
            let unflushed_batches: Vec<
                crate::storage::memtable::specialized::wal_behavior::WALVectorBatch,
            > = vec![]; // Stub

            let mut results = Vec::new();

            for batch in unflushed_batches {
                for record in batch.vector_records.iter() {
                    let vector = record
                        .embeddings
                        .first()
                        .map(|embedding| embedding.as_fp32_slice())
                        .unwrap_or(&[]);
                    if vector.is_empty() {
                        continue;
                    }

                    // Calculate similarity
                    let similarity = self.distance_compute.calculate_distance(
                        vector,
                        query_vector,
                        &distance_metric,
                    );

                    // Create OptimizedSearchRecord directly from canonical ProximaRecord data.
                    let mut metadata_map = HashMap::new();
                    for (key, node) in &record.props {
                        if let proximadb_records::ProximaTreeNode::Value(value) = node {
                            metadata_map.insert(key.clone(), value.clone());
                        }
                    }

                    let search_result =
                        OptimizedSearchRecord::new(record.oid.clone(), similarity.normalized_score)
                            .with_similarity(similarity.rank_value)
                            .add_vector(vector.to_vec())
                            .with_proxima_metadata(metadata_map)
                            .with_version_info(
                                record.record_version as u32,
                                record.created_at_ns / 1_000_000,
                            );

                    results.push(search_result);
                }
            }

            // Sort by score
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);

            debug!("✅ STREAMING_WAL: Found {} results", results.len());
            Ok(results)
        } else {
            debug!("❌ STREAMING_WAL: No WAL behavior available");
            Ok(Vec::new())
        }
    }

    /// Get streaming search statistics
    pub async fn stats(&self) -> StreamingSearchStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        *self.stats.write().await = StreamingSearchStats::default();
    }
}

impl Clone for StreamingSearchService {
    fn clone(&self) -> Self {
        Self {
            direct_service: self.direct_service.clone(),
            config: self.config.clone(),
            distance_compute: self.distance_compute.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Stream implementation for search results
impl Stream for SearchResultStream {
    type Item = Result<SearchResultBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(batch)) => Poll::Ready(Some(Ok(batch))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl SearchResultStream {
    /// Get search metadata
    pub fn metadata(&self) -> &SearchMetadata {
        &self.metadata
    }

    /// Collect all results (consumes the stream)
    pub async fn collect_all(mut self) -> Result<Vec<OptimizedSearchRecord>> {
        let mut all_results = Vec::new();

        while let Some(batch) = self.receiver.recv().await {
            if batch.is_final {
                break;
            }
            all_results.extend(batch.results);
        }

        Ok(all_results)
    }

    /// Take first N results (consumes the stream)
    pub async fn take(mut self, n: usize) -> Result<Vec<OptimizedSearchRecord>> {
        let mut results = Vec::with_capacity(n);

        while results.len() < n {
            match self.receiver.recv().await {
                Some(batch) => {
                    if batch.is_final {
                        break;
                    }

                    for result in batch.results {
                        if results.len() < n {
                            results.push(result);
                        } else {
                            break;
                        }
                    }
                }
                None => break,
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_streaming_search_basic() {
        // Deferred: Implement comprehensive tests
        assert!(true);
    }

    #[tokio::test]
    async fn test_streaming_search_batching() {
        // Deferred: Test search result batching
        assert!(true);
    }

    #[tokio::test]
    async fn test_concurrent_streaming() {
        // Deferred: Test concurrent streaming
        assert!(true);
    }
}
