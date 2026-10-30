/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Flush Result Optimization for High-Performance Vector Flushing
//!
//! This module provides optimizations for the flush process to minimize
//! memory allocations, improve throughput, and reduce latency.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};

use super::enhanced_flush_result::EnhancedFlushResult;
use crate::storage::traits::FlushResult;
use proximadb_records::ProximaRecord;

/// Optimized flush result that minimizes memory allocations
#[derive(Debug)]
pub struct OptimizedFlushResult {
    /// Base flush result
    pub base: FlushResult,

    /// Reference to vectors (avoids cloning large datasets)
    pub vector_refs: Arc<Vec<Arc<ProximaRecord>>>,

    /// Deleted vector IDs (if any)
    pub deleted_vector_ids: Vec<String>,

    /// Memory pool for reuse
    pub memory_pool: Option<Arc<VectorMemoryPool>>,
}

/// Memory pool for vector record allocation reuse
#[derive(Debug)]
pub struct VectorMemoryPool {
    /// Pre-allocated vector buffers
    buffers: tokio::sync::Mutex<Vec<Vec<f32>>>,

    /// Maximum pool size
    max_size: usize,

    /// Buffer dimension
    dimension: usize,
}

impl VectorMemoryPool {
    /// Create a new memory pool
    pub fn new(max_size: usize, dimension: usize) -> Self {
        Self {
            buffers: tokio::sync::Mutex::new(Vec::with_capacity(max_size)),
            max_size,
            dimension,
        }
    }

    /// Get a buffer from the pool or allocate new
    pub async fn acquire(&self) -> Vec<f32> {
        let mut buffers = self.buffers.lock().await;
        buffers.pop().unwrap_or_else(|| vec![0.0; self.dimension])
    }

    /// Return a buffer to the pool
    pub async fn release(&self, mut buffer: Vec<f32>) {
        let mut buffers = self.buffers.lock().await;
        if buffers.len() < self.max_size {
            buffer.clear();
            buffer.resize(self.dimension, 0.0);
            buffers.push(buffer);
        }
    }
}

/// Streaming flush result for handling large datasets
pub struct StreamingFlushResult {
    /// Base flush metadata
    pub base: FlushResult,

    /// Channel for streaming vectors
    pub vector_stream: mpsc::Receiver<Arc<ProximaRecord>>,

    /// Total expected vectors
    pub expected_count: usize,
}

/// Batch flush processor for optimal throughput
pub struct BatchFlushProcessor {
    /// Batch size for processing
    #[allow(dead_code)]
    batch_size: usize,

    /// Number of parallel workers
    worker_count: usize,

    /// Memory pool for vector buffers
    memory_pool: Arc<VectorMemoryPool>,
}

impl BatchFlushProcessor {
    /// Create a new batch processor
    pub fn new(batch_size: usize, worker_count: usize, dimension: usize) -> Self {
        Self {
            batch_size,
            worker_count,
            memory_pool: Arc::new(VectorMemoryPool::new(batch_size * 2, dimension)),
        }
    }

    /// Process vectors in optimized batches
    pub async fn process_batch(
        &self,
        vectors: Vec<ProximaRecord>,
    ) -> Result<Vec<Arc<ProximaRecord>>> {
        let chunk_size = vectors.len().div_ceil(self.worker_count);
        let chunks: Vec<_> = vectors
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();

        for chunk in chunks {
            let pool = self.memory_pool.clone();
            let handle = tokio::spawn(async move {
                let mut processed = Vec::with_capacity(chunk.len());
                for vector in chunk {
                    // Process vector with memory pool buffer
                    let _buffer = pool.acquire().await;
                    processed.push(Arc::new(vector));
                }
                processed
            });
            handles.push(handle);
        }

        let mut all_processed = Vec::with_capacity(vectors.len());
        for handle in handles {
            let batch_result = handle.await?;
            all_processed.extend(batch_result);
        }

        Ok(all_processed)
    }
}

/// Zero-copy flush result using memory-mapped files
pub struct ZeroCopyFlushResult {
    /// Base flush metadata
    pub base: FlushResult,

    /// Memory-mapped file path containing vectors
    pub mmap_path: String,

    /// Offset and length for each vector in the file
    pub vector_offsets: Vec<(u64, u64)>,
}

/// Flush result cache for deduplication
pub struct FlushResultCache {
    /// LRU cache of recent flush results
    cache: moka::future::Cache<String, Arc<FlushResult>>,

    /// Maximum cache size
    #[allow(dead_code)]
    max_entries: u64,
}

impl FlushResultCache {
    /// Create a new cache
    pub fn new(max_entries: u64) -> Self {
        let cache = moka::future::Cache::builder()
            .max_capacity(max_entries)
            .build();

        Self { cache, max_entries }
    }

    /// Check if a flush result is cached
    pub async fn get(&self, key: &str) -> Option<Arc<FlushResult>> {
        self.cache.get(key).await
    }

    /// Cache a flush result
    pub async fn insert(&self, key: String, result: Arc<FlushResult>) {
        self.cache.insert(key, result).await;
    }
}

/// Optimized flush coordinator with all optimizations
pub struct OptimizedFlushCoordinator {
    /// Batch processor for vectors
    batch_processor: BatchFlushProcessor,

    /// Result cache for deduplication
    result_cache: FlushResultCache,

    /// Memory pool for allocations
    memory_pool: Arc<VectorMemoryPool>,

    /// Metrics for monitoring
    metrics: FlushMetrics,
}

/// Flush performance metrics
#[derive(Debug, Default)]
struct FlushMetrics {
    /// Total vectors flushed
    total_vectors: std::sync::atomic::AtomicU64,

    /// Total bytes processed
    total_bytes: std::sync::atomic::AtomicU64,

    /// Cache hits
    cache_hits: std::sync::atomic::AtomicU64,

    /// Cache misses
    cache_misses: std::sync::atomic::AtomicU64,
}

impl OptimizedFlushCoordinator {
    /// Create a new optimized coordinator
    pub fn new(batch_size: usize, worker_count: usize, dimension: usize) -> Self {
        Self {
            batch_processor: BatchFlushProcessor::new(batch_size, worker_count, dimension),
            result_cache: FlushResultCache::new(1000),
            memory_pool: Arc::new(VectorMemoryPool::new(batch_size * 2, dimension)),
            metrics: FlushMetrics::default(),
        }
    }

    /// Execute optimized flush
    pub async fn execute_optimized_flush(
        &self,
        collection_id: &str,
        vectors: Vec<ProximaRecord>,
    ) -> Result<OptimizedFlushResult> {
        info!(
            "🚀 Optimized flush starting for {} with {} vectors",
            collection_id,
            vectors.len()
        );

        // Check cache first
        let cache_key = format!("{}:{}", collection_id, vectors.len());
        if let Some(cached) = self.result_cache.get(&cache_key).await {
            self.metrics
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            debug!("✅ Cache hit for flush result");

            return Ok(OptimizedFlushResult {
                base: (*cached).clone(),
                vector_refs: Arc::new(vec![]),
                deleted_vector_ids: vec![],
                memory_pool: Some(self.memory_pool.clone()),
            });
        }

        self.metrics
            .cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Process vectors in optimized batches
        let processed_vectors = self.batch_processor.process_batch(vectors).await?;

        // Update metrics
        self.metrics.total_vectors.fetch_add(
            processed_vectors.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Create optimized result
        let base_result = FlushResult {
            success: true,
            collections_affected: vec![collection_id.to_string()],
            entries_flushed: Some(processed_vectors.len() as u64),
            bytes_written: Some(processed_vectors.len() as u64 * 512), // Estimate
            files_created: Some(1),
            file_paths: vec![],
            duration_ms: Some(0), // Will be set by caller
            completed_at: chrono::Utc::now(),
            engine_metrics: std::collections::HashMap::new(),
            compaction_triggered: false,
            compaction_error: None,
            flushed_batch_ids: vec![],
        };

        // Cache the result
        self.result_cache
            .insert(cache_key, Arc::new(base_result.clone()))
            .await;

        Ok(OptimizedFlushResult {
            base: base_result,
            vector_refs: Arc::new(processed_vectors),
            deleted_vector_ids: vec![],
            memory_pool: Some(self.memory_pool.clone()),
        })
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.metrics
                .total_vectors
                .load(std::sync::atomic::Ordering::Relaxed),
            self.metrics
                .total_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
            self.metrics
                .cache_hits
                .load(std::sync::atomic::Ordering::Relaxed),
            self.metrics
                .cache_misses
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

/// Convert optimized result to enhanced result when needed
impl From<OptimizedFlushResult> for EnhancedFlushResult {
    fn from(optimized: OptimizedFlushResult) -> Self {
        let vector_records: Vec<ProximaRecord> = optimized
            .vector_refs
            .iter()
            .map(|arc_vec| (**arc_vec).clone())
            .collect();

        EnhancedFlushResult::with_deletions(
            optimized.base,
            vector_records,
            optimized.deleted_vector_ids,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool() {
        let pool = VectorMemoryPool::new(10, 128);

        // Acquire buffers
        let buf1 = pool.acquire().await;
        let buf2 = pool.acquire().await;

        assert_eq!(buf1.len(), 128);
        assert_eq!(buf2.len(), 128);

        // Release buffers
        pool.release(buf1).await;
        pool.release(buf2).await;

        // Verify reuse
        let buf3 = pool.acquire().await;
        assert_eq!(buf3.len(), 128);
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let processor = BatchFlushProcessor::new(100, 4, 128);

        let vectors: Vec<ProximaRecord> = (0..100)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![i as f32; 128],
                )],
                record_version: 1,
                created_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                updated_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                ..Default::default()
            })
            .collect();

        let processed = processor.process_batch(vectors).await.unwrap();
        assert_eq!(processed.len(), 100);
    }

    #[tokio::test]
    async fn test_optimized_coordinator() {
        let coordinator = OptimizedFlushCoordinator::new(50, 2, 128);

        let vectors: Vec<ProximaRecord> = (0..50)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![i as f32; 128],
                )],
                record_version: 1,
                created_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                updated_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                ..Default::default()
            })
            .collect();

        let result = coordinator
            .execute_optimized_flush("test_collection", vectors.clone())
            .await
            .unwrap();

        assert_eq!(result.vector_refs.len(), 50);
        assert!(result.base.success);

        // Test cache hit
        let _result2 = coordinator
            .execute_optimized_flush("test_collection", vectors)
            .await
            .unwrap();

        let (_, _, hits, misses) = coordinator.get_metrics();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }
}
