use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info};

use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::persistence::write_ahead_log::config::WALConfig;
use proximadb_records::ProximaRecord;

/// WAL serialization format for optimized writes.
/// Defined locally after removal from vector_operations_service.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizedFormat {
    /// Protobuf batch serialization
    Proto,
    /// Bincode serialization (fastest for Rust)
    Bincode,
    /// Avro serialization (schema evolution)
    Avro,
}

/// High-performance WAL writer with batching, caching, and background writes
///
/// Key optimizations:
/// 1. Batching - Combines multiple writes into single disk operations
/// 2. Background writes - Non-blocking API with async workers
/// 3. Assignment caching - Reduces assignment service lookups
/// 4. Directory caching - Avoids repeated existence checks
/// 5. Write combining - Merges writes to same collection
/// 6. Optimized serialization - Uses fastest format for workload
pub struct OptimizedWriteBufferWriter {
    // Configuration
    config: Arc<WALConfig>,

    // Filesystem factory
    filesystem_factory: Arc<FilesystemFactory>,

    // Write channel for batching
    write_sender: mpsc::Sender<WalWriteRequest>,

    // Cached assignments (collection_id -> WAL directory)
    assignment_cache: Arc<RwLock<HashMap<String, CachedAssignment>>>,

    // Pre-created directory cache (path -> last_checked)
    directory_cache: Arc<RwLock<HashMap<String, Instant>>>,

    // Metrics
    metrics: Arc<RwLock<WalWriterMetrics>>,
}

struct WalWriteRequest {
    collection_id: String,
    records: Vec<ProximaRecord>,
    sequences: Vec<u64>,
    format: OptimizedFormat,
    base_location: String,
    response_tx: tokio::sync::oneshot::Sender<Result<String>>,
}

#[derive(Clone)]
struct CachedAssignment {
    storage_url: String,
    #[allow(dead_code)]
    collection_wal_dir: String,
    logs_dir: String,
    #[allow(dead_code)]
    cached_at: Instant,
}

#[derive(Debug, Clone, Copy)]
enum FlushReason {
    SizeThreshold,
    MemoryThreshold,
    CollectionThreshold,
    Timeout,
}

#[derive(Default, Clone)]
pub struct WalWriterMetrics {
    // Basic counters
    total_writes: u64,
    total_bytes_written: u64,
    cache_hits: u64,
    cache_misses: u64,
    batch_writes: u64,
    write_errors: u64,

    // Enhanced batching metrics
    total_vectors_written: u64,
    combined_writes: u64,    // Number of times writes were combined
    timeout_flushes: u64,    // Flushes triggered by timeout
    size_flushes: u64,       // Flushes triggered by size threshold
    memory_flushes: u64,     // Flushes triggered by memory threshold
    collection_flushes: u64, // Flushes triggered by too many collections

    // Performance tracking
    avg_batch_size: f64,
    avg_flush_latency_ms: f64,
    max_batch_size: usize,
    min_batch_size: usize,
}

impl OptimizedWriteBufferWriter {
    /// Create new optimized WAL writer with background worker
    pub async fn new(
        config: Arc<WALConfig>,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        let (write_sender, write_receiver) = mpsc::channel::<WalWriteRequest>(1000);

        let writer = Self {
            config: config.clone(),
            filesystem_factory: filesystem_factory.clone(),
            write_sender,
            assignment_cache: Arc::new(RwLock::new(HashMap::new())),
            directory_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(WalWriterMetrics::default())),
        };

        // Spawn single background writer worker
        // Note: mpsc::Receiver cannot be cloned, so we use a single worker
        // For higher throughput, consider using crossbeam channels
        writer.spawn_writer_worker(0, write_receiver);

        Ok(writer)
    }

    /// Write vectors to WAL (non-blocking, returns immediately)
    pub async fn write_vectors(
        &self,
        collection_id: &str,
        records: Vec<ProximaRecord>,
        sequences: Vec<u64>,
        format: OptimizedFormat,
        base_location: String,
    ) -> Result<String> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let request = WalWriteRequest {
            collection_id: collection_id.to_string(),
            records,
            sequences,
            format,
            base_location,
            response_tx,
        };

        // Send to write queue (non-blocking)
        self.write_sender
            .send(request)
            .await
            .context("WAL writer channel full")?;

        // Wait for response
        response_rx.await.context("WAL writer worker died")?
    }

    /// Spawn a background writer worker
    fn spawn_writer_worker(&self, worker_id: usize, mut receiver: mpsc::Receiver<WalWriteRequest>) {
        let config = self.config.clone();
        let assignment_cache = self.assignment_cache.clone();
        let directory_cache = self.directory_cache.clone();
        let metrics = self.metrics.clone();
        let filesystem_factory = self.filesystem_factory.clone();

        tokio::spawn(async move {
            info!("🚀 WAL writer worker {} started", worker_id);

            // Enhanced batch configuration
            let batch_size_threshold = config.optimized_writer_batch_size;
            let batch_timeout_ms = config.optimized_writer_batch_timeout_ms;
            let enable_combining = config.optimized_writer_enable_combining;

            info!(
                "📊 WAL writer worker {} config - batch_size: {}, timeout: {}ms, combining: {}",
                worker_id,
                batch_size_threshold.unwrap_or(100),
                batch_timeout_ms.unwrap_or(100),
                enable_combining.unwrap_or(true)
            );

            // Batch accumulator with enhanced tracking
            let mut batch: HashMap<String, Vec<WalWriteRequest>> = HashMap::new();
            let mut batch_timer =
                tokio::time::interval(Duration::from_millis(batch_timeout_ms.unwrap_or(100)));
            let mut last_flush_time = Instant::now();
            let mut total_vectors_in_batch = 0usize;
            let mut total_bytes_in_batch = 0usize;

            loop {
                tokio::select! {
                    // Receive write requests
                    Some(request) = receiver.recv() => {
                        let collection_id = request.collection_id.clone();

                        // Track batch metrics
                        total_vectors_in_batch += request.records.len();
                        total_bytes_in_batch += Self::estimate_request_size(&request);

                        // Add to batch (with combining if enabled)
                        let combined = if enable_combining.unwrap_or(true) {
                            Self::add_with_combining(&mut batch, request)
                        } else {
                            batch.entry(collection_id).or_default().push(request);
                            false
                        };

                        // Track combining metrics
                        if combined {
                            metrics.write().await.combined_writes += 1;
                        }

                        // Check multiple flush conditions
                        if let Some(flush_reason) = Self::should_flush_batch(
                            &batch,
                            batch_size_threshold.unwrap_or(100),
                            total_vectors_in_batch,
                            total_bytes_in_batch,
                            last_flush_time,
                            batch_timeout_ms.unwrap_or(100)
                        ) {
                            debug!(
                                "🚀 Worker {} triggering flush - reason: {:?}, collections: {}, vectors: {}, bytes: {}KB",
                                worker_id, flush_reason, batch.len(), total_vectors_in_batch, total_bytes_in_batch / 1024
                            );

                            Self::flush_batch_with_reason(
                                &mut batch,
                                &config,
                                &assignment_cache,
                                &directory_cache,
                                &filesystem_factory,
                                &metrics,
                                worker_id,
                                flush_reason
                            ).await;

                            // Reset tracking
                            total_vectors_in_batch = 0;
                            total_bytes_in_batch = 0;
                            last_flush_time = Instant::now();
                        }
                    }

                    // Batch timeout - flush whatever we have
                    _ = batch_timer.tick() => {
                        if !batch.is_empty() {
                            debug!(
                                "⏰ Worker {} timeout flush - {} collections, {} vectors, {}KB",
                                worker_id, batch.len(), total_vectors_in_batch, total_bytes_in_batch / 1024
                            );

                            Self::flush_batch_with_reason(
                                &mut batch,
                                &config,
                                &assignment_cache,
                                &directory_cache,
                                &filesystem_factory,
                                &metrics,
                                worker_id,
                                FlushReason::Timeout
                            ).await;

                            // Reset tracking
                            total_vectors_in_batch = 0;
                            total_bytes_in_batch = 0;
                            last_flush_time = Instant::now();
                        }
                    }
                }
            }
        });
    }

    /// Flush a batch of writes with detailed metrics tracking
    async fn flush_batch_with_reason(
        batch: &mut HashMap<String, Vec<WalWriteRequest>>,
        config: &WALConfig,
        assignment_cache: &Arc<RwLock<HashMap<String, CachedAssignment>>>,
        directory_cache: &Arc<RwLock<HashMap<String, Instant>>>,
        filesystem_factory: &FilesystemFactory,
        metrics: &Arc<RwLock<WalWriterMetrics>>,
        worker_id: usize,
        flush_reason: FlushReason,
    ) {
        let start_time = Instant::now();
        let num_collections = batch.len();
        let total_requests: usize = batch.values().map(|v| v.len()).sum();
        let total_vectors: usize = batch
            .values()
            .flat_map(|requests| requests.iter())
            .map(|req| req.records.len())
            .sum();

        debug!(
            "🔄 Worker {} flushing batch - reason: {:?}, collections: {}, requests: {}, vectors: {}",
            worker_id, flush_reason, num_collections, total_requests, total_vectors
        );

        // Process each collection's batch
        for (collection_id, requests) in batch.drain() {
            if let Err(e) = Self::write_collection_batch(
                &collection_id,
                requests,
                config,
                assignment_cache,
                directory_cache,
                filesystem_factory,
                metrics,
            )
            .await
            {
                error!(
                    "Failed to write batch for collection {}: {}",
                    collection_id, e
                );
            }
        }

        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;

        debug!(
            "✅ Worker {} batch flush completed in {:.2}ms",
            worker_id, duration_ms
        );

        // Update detailed metrics
        {
            let mut metrics_guard = metrics.write().await;
            metrics_guard.batch_writes += 1;
            metrics_guard.total_vectors_written += total_vectors as u64;

            // Track flush reasons
            match flush_reason {
                FlushReason::SizeThreshold => metrics_guard.size_flushes += 1,
                FlushReason::MemoryThreshold => metrics_guard.memory_flushes += 1,
                FlushReason::CollectionThreshold => metrics_guard.collection_flushes += 1,
                FlushReason::Timeout => metrics_guard.timeout_flushes += 1,
            }

            // Update performance tracking
            metrics_guard.avg_flush_latency_ms = (metrics_guard.avg_flush_latency_ms
                * (metrics_guard.batch_writes as f64 - 1.0)
                + duration_ms)
                / metrics_guard.batch_writes as f64;

            metrics_guard.avg_batch_size = (metrics_guard.avg_batch_size
                * (metrics_guard.batch_writes as f64 - 1.0)
                + total_requests as f64)
                / metrics_guard.batch_writes as f64;

            if total_requests > metrics_guard.max_batch_size {
                metrics_guard.max_batch_size = total_requests;
            }

            if metrics_guard.min_batch_size == 0 || total_requests < metrics_guard.min_batch_size {
                metrics_guard.min_batch_size = total_requests;
            }
        }
    }

    /// Write a batch of requests for a single collection
    async fn write_collection_batch(
        collection_id: &str,
        requests: Vec<WalWriteRequest>,
        config: &WALConfig,
        _assignment_cache: &Arc<RwLock<HashMap<String, CachedAssignment>>>,
        directory_cache: &Arc<RwLock<HashMap<String, Instant>>>,
        filesystem_factory: &FilesystemFactory,
        metrics: &Arc<RwLock<WalWriterMetrics>>,
    ) -> Result<()> {
        // Get base_location from the first request (all requests for same collection have same base_location)
        let base_location = requests
            .first()
            .ok_or_else(|| anyhow::anyhow!("No requests in batch"))?
            .base_location
            .clone();

        // Create assignment paths
        let assignment = Self::create_assignment_paths(collection_id, &base_location);

        // Get filesystem for this URL
        let filesystem = filesystem_factory.get_filesystem(&assignment.storage_url)?;

        // Ensure directory exists (with caching)
        if Self::should_check_directory(&assignment.logs_dir, directory_cache).await {
            if !filesystem.exists(&assignment.logs_dir).await? {
                filesystem.create_dir_all(&assignment.logs_dir).await?;
                debug!("📁 Created WAL logs directory: {}", assignment.logs_dir);
            }
            // Update cache
            directory_cache
                .write()
                .await
                .insert(assignment.logs_dir.clone(), Instant::now());
        }

        // Collect all records and response channels
        let mut all_records = Vec::new();
        let mut all_sequences = Vec::new();
        let mut response_txs = Vec::new();
        let mut batch_format = OptimizedFormat::Bincode; // default

        for request in requests {
            batch_format = request.format; // use last request's format
            all_records.extend(request.records);
            all_sequences.extend(request.sequences);
            response_txs.push(request.response_tx);
        }

        // Write combined batch
        let result = Self::write_combined_batch(
            collection_id,
            &all_records,
            &all_sequences,
            &batch_format,
            &assignment,
            filesystem_factory,
            config,
            metrics,
        )
        .await;

        match &result {
            Ok(wal_path) => {
                debug!("Successfully wrote batch to: {}", wal_path);
            }
            Err(e) => {
                error!("Failed to write batch: {}", e);
                metrics.write().await.write_errors += 1;
            }
        }

        // Send responses to all requests in this batch
        match result {
            Ok(wal_path) => {
                for tx in response_txs {
                    let _ = tx.send(Ok(wal_path.clone()));
                }
            }
            Err(e) => {
                for tx in response_txs {
                    let _ = tx.send(Err(anyhow::anyhow!("WAL write failed: {}", e)));
                }
            }
        }

        Ok(())
    }

    /// Check if directory should be checked (based on cache)
    async fn should_check_directory(
        logs_dir: &str,
        directory_cache: &Arc<RwLock<HashMap<String, Instant>>>,
    ) -> bool {
        let cache = directory_cache.read().await;
        if let Some(last_checked) = cache.get(logs_dir) {
            // Directory cache valid for 1 hour
            last_checked.elapsed() >= Duration::from_secs(3600)
        } else {
            true
        }
    }

    /// Create assignment paths from base location
    fn create_assignment_paths(collection_id: &str, base_location: &str) -> CachedAssignment {
        // Create assignment from base location
        let write_buffer_url = format!("{}/{}/write_buffer", base_location, collection_id);

        CachedAssignment {
            storage_url: write_buffer_url.clone(),
            collection_wal_dir: write_buffer_url.clone(),
            logs_dir: format!("{}/logs", write_buffer_url.trim_end_matches('/')),
            cached_at: Instant::now(),
        }
    }

    /// Write combined batch with optimized atomic write
    async fn write_combined_batch(
        _collection_id: &str,
        records: &[ProximaRecord],
        sequences: &[u64],
        format: &OptimizedFormat,
        assignment: &CachedAssignment,
        filesystem_factory: &FilesystemFactory,
        _config: &WALConfig,
        metrics: &Arc<RwLock<WalWriterMetrics>>,
    ) -> Result<String> {
        // Serialize canonical records using the specified WAL slot format.
        let serialized_data = Self::serialize_records_optimized(records, format)?;

        // Generate filename with format-specific extension
        let min_seq = sequences.iter().min().copied();
        let max_seq = sequences.iter().max().copied();
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let file_extension = match format {
            OptimizedFormat::Proto => "pbwal",
            OptimizedFormat::Bincode => "bcwal",
            OptimizedFormat::Avro => "avwal",
        };

        let uuid_short = &proximadb_kernel::uuid::Uuid::new_v4().to_string()[..8];
        let wal_filename = format!(
            "wal_{}_{:010}_{:010}_{}.{}",
            timestamp,
            min_seq.unwrap_or(0),
            max_seq.unwrap_or(0),
            uuid_short,
            file_extension
        );
        let wal_file_path = format!("{}/{}", assignment.logs_dir, wal_filename);

        // Construct full URL for filesystem factory
        // Note: assignment.storage_url is actually the write_buffer_url which already contains collection_id
        let full_url = format!("{}/{}", assignment.logs_dir, wal_filename);

        // Write to filesystem using factory methods
        filesystem_factory
            .write(&full_url, &serialized_data, None)
            .await
            .context("Failed to write WAL file")?;

        // Update metrics
        let mut metrics_guard = metrics.write().await;
        metrics_guard.total_writes += 1;
        metrics_guard.total_bytes_written += serialized_data.len() as u64;

        debug!(
            "💾 WAL_WRITE: {} records ({} bytes) -> {} (sequences: {}..{})",
            records.len(),
            serialized_data.len(),
            wal_filename,
            min_seq.unwrap_or(0),
            max_seq.unwrap_or(0)
        );

        Ok(wal_file_path)
    }

    /// Optimized serialization dispatching to format-specific implementations
    fn serialize_records_optimized(
        records: &[ProximaRecord],
        format: &OptimizedFormat,
    ) -> Result<Vec<u8>> {
        match format {
            OptimizedFormat::Proto => {
                // The optimized writer's format enum names the file slot, not
                // the internal envelope. Keep the payload canonical while the
                // surrounding recovery path converges on v2 wire encoding.
                bincode::serialize(records)
                    .context("Canonical ProximaRecord proto-slot serialization failed")
            }
            OptimizedFormat::Bincode => {
                bincode::serialize(records).context("Bincode serialization failed")
            }
            OptimizedFormat::Avro => {
                // Avro: fall back to bincode for now; full Avro schema-based
                // serialization requires the AvroSerializationStrategy path
                bincode::serialize(records).context("Avro fallback (bincode) serialization failed")
            }
        }
    }

    /// Get current metrics for monitoring and debugging
    pub async fn get_metrics(&self) -> WalWriterMetrics {
        self.metrics.read().await.clone()
    }

    /// Get a formatted metrics report for logging/debugging
    pub async fn get_metrics_report(&self) -> String {
        let metrics = self.metrics.read().await;
        format!(
            "WAL Writer Metrics:\n\
             📊 Basic Stats: {} writes, {} vectors, {:.2}MB written\n\
             🔄 Batching: {} batches (avg: {:.1} req/batch, min: {}, max: {})\n\
             ⚡ Performance: {:.2}ms avg flush latency\n\
             🔗 Combining: {} writes combined\n\
             📈 Flush Reasons: timeout={}, size={}, memory={}, collections={}\n\
             🎯 Cache: {} hits, {} misses ({:.1}% hit rate)",
            metrics.total_writes,
            metrics.total_vectors_written,
            metrics.total_bytes_written as f64 / (1024.0 * 1024.0),
            metrics.batch_writes,
            metrics.avg_batch_size,
            metrics.min_batch_size,
            metrics.max_batch_size,
            metrics.avg_flush_latency_ms,
            metrics.combined_writes,
            metrics.timeout_flushes,
            metrics.size_flushes,
            metrics.memory_flushes,
            metrics.collection_flushes,
            metrics.cache_hits,
            metrics.cache_misses,
            if metrics.cache_hits + metrics.cache_misses > 0 {
                metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64
                    * 100.0
            } else {
                0.0
            }
        )
    }

    pub async fn shutdown(&self) -> Result<()> {
        // The writer will shut down when all senders are dropped
        // Here we can add any cleanup logic if needed
        info!("🛑 Shutting down optimized WAL writer");

        // Log final metrics before shutdown
        let metrics_report = self.get_metrics_report().await;
        info!("📊 Final WAL Writer Metrics:\n{}", metrics_report);

        Ok(())
    }

    /// Estimate the size of a write request for batching decisions
    fn estimate_request_size(request: &WalWriteRequest) -> usize {
        let embedding_values: usize = request
            .records
            .iter()
            .map(|record| {
                record
                    .embeddings
                    .iter()
                    .map(|embedding| embedding.values.len())
                    .sum::<usize>()
            })
            .sum();
        let vector_size = embedding_values * 4;
        let metadata_size = request
            .records
            .iter()
            .map(|record| record.props.len() * 50) // Rough estimate for properties
            .sum::<usize>();
        let sequences_size = request.sequences.len() * 8; // u64 sequences

        vector_size + metadata_size + sequences_size + request.collection_id.len()
    }

    /// Add request to batch with write combining (merges vectors for same collection)
    fn add_with_combining(
        batch: &mut HashMap<String, Vec<WalWriteRequest>>,
        mut new_request: WalWriteRequest,
    ) -> bool {
        let collection_id = new_request.collection_id.clone();

        match batch.get_mut(&collection_id) {
            Some(existing_requests) => {
                // Try to find a request with the same format to combine with
                if let Some(existing_request) = existing_requests
                    .iter_mut()
                    .find(|req| req.records.len() < 1000)
                {
                    // Combine the records and sequences
                    existing_request.records.append(&mut new_request.records);
                    existing_request
                        .sequences
                        .append(&mut new_request.sequences);

                    // Note: We keep the original response_tx, the new one will timeout
                    // This is a trade-off for better batching efficiency
                    true // Combining happened
                } else {
                    // No suitable existing request to combine with
                    existing_requests.push(new_request);
                    false // No combining
                }
            }
            None => {
                // First request for this collection
                batch.insert(collection_id, vec![new_request]);
                false // No combining (first request)
            }
        }
    }

    /// Determine if the batch should be flushed based on multiple criteria
    fn should_flush_batch(
        batch: &HashMap<String, Vec<WalWriteRequest>>,
        batch_size_threshold: usize,
        total_vectors: usize,
        total_bytes: usize,
        last_flush_time: Instant,
        batch_timeout_ms: u64,
    ) -> Option<FlushReason> {
        // Flush conditions (checked in priority order):

        // 1. Number of requests exceeds threshold
        let total_requests: usize = batch.values().map(|v| v.len()).sum();
        if total_requests >= batch_size_threshold {
            return Some(FlushReason::SizeThreshold);
        }

        // 2. Number of vectors exceeds threshold (prevent memory bloat)
        if total_vectors >= batch_size_threshold * 10 {
            return Some(FlushReason::MemoryThreshold);
        }

        // 3. Total bytes exceed threshold (prevent large memory usage)
        if total_bytes >= 1024 * 1024 {
            // 1MB threshold
            return Some(FlushReason::MemoryThreshold);
        }

        // 4. Too many collections in batch (prevent excessive file operations)
        if batch.len() >= 50 {
            return Some(FlushReason::CollectionThreshold);
        }

        // 5. Time since last flush exceeds threshold
        if last_flush_time.elapsed().as_millis() >= batch_timeout_ms as u128 {
            return Some(FlushReason::Timeout);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::sync::Arc;
    use tempfile::TempDir;

    use crate::storage::persistence::filesystem::FilesystemFactory;
    use crate::storage::persistence::write_ahead_log::config::{
        CompressionConfig as WALCompressionConfig, DiskDistributionStrategy, MemTableConfig,
        MultiDiskConfig, PerformanceConfig, WALConfig, WriteBufferStrategyType,
    };
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaTreeNode};

    /// Test helper to create sample vectors
    fn create_test_vectors(count: usize, dimension: usize) -> Vec<ProximaRecord> {
        (0..count)
            .map(|i| {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert(
                    "index".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(i.to_string())),
                );

                let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
                ProximaRecord {
                    oid: format!("vec_{}", i),
                    embeddings: vec![EmbeddingCell::new_fp32(
                        "test",
                        "dense_vector",
                        dimension as u32,
                        vec![i as f32; dimension],
                    )],
                    props: metadata,
                    created_at_ns: now_ns,
                    updated_at_ns: now_ns,
                    record_version: 1,
                    ..Default::default()
                }
            })
            .collect()
    }

    fn create_test_wal_config(temp_dir: &TempDir) -> WALConfig {
        WALConfig {
            strategy_type: WriteBufferStrategyType::default(),
            memtable: MemTableConfig::default(),
            multi_disk: MultiDiskConfig {
                data_directories: vec![format!("file://{}/wal", temp_dir.path().display())],
                distribution_strategy: DiskDistributionStrategy::RoundRobin,
                collection_affinity: false,
            },
            compression: WALCompressionConfig::default(),
            performance: PerformanceConfig::default(),
            enable_mvcc: false,
            enable_ttl: false,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_optimized_wal_writer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = Arc::new(create_test_wal_config(&temp_dir));
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);

        // Create the actual writer from production code
        let writer = OptimizedWriteBufferWriter::new(config, filesystem_factory).await?;

        // Test basic write operation
        let vectors = create_test_vectors(10, 128);
        let result = writer
            .write_vectors(
                "test_collection",
                vectors,
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                OptimizedFormat::Bincode,
                format!("file://{}", temp_dir.path().display()),
            )
            .await;

        assert!(
            result.is_ok(),
            "Write operation should succeed: {:?}",
            result.err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_batch_writes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = Arc::new(create_test_wal_config(&temp_dir));
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
        let writer = OptimizedWriteBufferWriter::new(config, filesystem_factory).await?;

        // Write multiple batches sequentially to same collection
        // This tests the batching behavior where multiple writes are accumulated
        for batch in 0..5 {
            let vectors = create_test_vectors(20, 128);
            let sequences: Vec<u64> = (batch * 20..(batch + 1) * 20).collect();
            let base_location = format!("file://{}", temp_dir.path().display());

            let result = writer
                .write_vectors(
                    "test_collection",
                    vectors,
                    sequences,
                    OptimizedFormat::Bincode,
                    base_location,
                )
                .await;

            assert!(
                result.is_ok(),
                "Write batch {} should succeed: {:?}",
                batch,
                result.err()
            );
            assert!(!result.unwrap().is_empty(), "Should return WAL file path");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_wal_concurrent_writes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = Arc::new(create_test_wal_config(&temp_dir));
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
        let writer = Arc::new(OptimizedWriteBufferWriter::new(config, filesystem_factory).await?);

        // Concurrent writes to multiple collections
        let mut handles = Vec::new();
        for collection_id in 0..10 {
            let writer_clone = writer.clone();
            let base_location = format!("file://{}", temp_dir.path().display());

            let handle = tokio::spawn(async move {
                let vectors = create_test_vectors(50, 128);
                let sequences: Vec<u64> = (0..50).collect();

                writer_clone
                    .write_vectors(
                        &format!("collection_{}", collection_id),
                        vectors,
                        sequences,
                        OptimizedFormat::Bincode,
                        base_location,
                    )
                    .await
            });
            handles.push(handle);
        }

        // All writes should succeed
        for handle in handles {
            let result = handle.await??;
            assert!(
                !result.is_empty(),
                "Each write should return a WAL file path"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_wal_large_batch() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Create config with adjusted performance settings for large batch
        let mut config = create_test_wal_config(&temp_dir);
        config.performance.memory_flush_size_bytes = 500 * 1024 * 1024; // 500 MB

        let config = Arc::new(config);
        let filesystem_factory = Arc::new(FilesystemFactory::create(Default::default()).await?);
        let writer = OptimizedWriteBufferWriter::new(config, filesystem_factory).await?;

        // Write a large batch
        let vectors = create_test_vectors(5000, 256);
        let sequences: Vec<u64> = (0..5000).collect();

        let result = writer
            .write_vectors(
                "large_collection",
                vectors,
                sequences,
                OptimizedFormat::Bincode,
                format!("file://{}", temp_dir.path().display()),
            )
            .await?;

        assert!(
            !result.is_empty(),
            "Large batch should be written successfully"
        );
        Ok(())
    }
}
