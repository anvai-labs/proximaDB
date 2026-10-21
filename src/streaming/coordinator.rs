/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Stream coordinator for managing multiple streams
//!
//! The StreamCoordinator is the central component for managing streaming sessions.
//! It handles:
//! - Session lifecycle (creation, closing, timeout)
//! - Global rate limiting
//! - Backpressure coordination
//! - Buffer flushing
//! - Metrics collection

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use super::{
    BackpressureLevel, RateLimiter, SessionConfig, SessionState, StreamConfig, StreamError,
    StreamId, StreamMetrics, StreamResult, StreamSession,
};
use crate::storage::traits::UnifiedStorageFormat;
use proximadb_records::ProximaRecord;

/// Result of pushing records to a stream
#[derive(Debug, Clone)]
pub struct PushResult {
    /// Number of records successfully pushed
    pub pushed: usize,

    /// Number of records dropped (buffer full)
    pub dropped: usize,

    /// Current backpressure level
    pub backpressure: BackpressureLevel,

    /// Buffer utilization percentage (0-100)
    pub buffer_percent: u32,
}

impl PushResult {
    /// Get suggested delay based on backpressure level
    pub fn suggested_delay(&self) -> Duration {
        Duration::from_millis(self.backpressure.delay_ms() as u64)
    }

    /// Check if any records were dropped
    pub fn has_drops(&self) -> bool {
        self.dropped > 0
    }

    /// Check if backpressure requires action
    pub fn requires_slowdown(&self) -> bool {
        self.backpressure.requires_action()
    }
}

/// Stream coordinator for managing multiple streaming sessions
pub struct StreamCoordinator {
    /// Active streaming sessions
    sessions: DashMap<StreamId, StreamSession>,

    /// Global rate limiter
    rate_limiter: RateLimiter,

    /// Coordinator configuration
    config: StreamConfig,

    /// Metrics collector
    metrics: Arc<StreamMetrics>,

    /// Shutdown flag
    shutdown: std::sync::atomic::AtomicBool,
}

impl StreamCoordinator {
    /// Create a new stream coordinator with the given configuration
    pub fn new(config: StreamConfig) -> Self {
        let rate_limiter = if config.global_rate_limit > 0 {
            RateLimiter::new(config.global_rate_limit)
        } else {
            RateLimiter::unlimited()
        };

        Self {
            sessions: DashMap::new(),
            rate_limiter,
            config,
            metrics: Arc::new(StreamMetrics::new()),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create a new streaming session for a collection
    ///
    /// # Arguments
    ///
    /// * `collection` - Target collection name
    /// * `config` - Session-specific configuration
    ///
    /// # Returns
    ///
    /// * `Ok(StreamId)` - The ID of the created session
    /// * `Err(StreamError)` - If session creation fails
    pub async fn create_session(
        &self,
        collection: String,
        config: SessionConfig,
    ) -> StreamResult<StreamId> {
        // Check capacity
        if self.sessions.len() >= self.config.max_streams {
            return Err(StreamError::TooManySessions {
                max: self.config.max_streams,
                current: self.sessions.len(),
            });
        }

        // Determine buffer size
        let buffer_size = config
            .buffer_size
            .unwrap_or(self.config.default_buffer_size);

        // Create ack channel
        let (ack_tx, _ack_rx) = mpsc::channel(1000);

        // Create session
        let session = StreamSession::new(collection.clone(), config, buffer_size, ack_tx);
        let id = session.id.clone();

        info!(
            session_id = %id,
            collection = %collection,
            buffer_size = buffer_size,
            "Created streaming session"
        );

        self.sessions.insert(id.clone(), session);
        self.metrics.sessions_created_total.inc();
        self.metrics.inc_active_streams();

        Ok(id)
    }

    /// Push records to a streaming session
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session to push to
    /// * `records` - Vector records to push
    ///
    /// # Returns
    ///
    /// * `Ok(PushResult)` - Result of the push operation
    /// * `Err(StreamError)` - If push fails
    pub async fn push_records(
        &self,
        session_id: &StreamId,
        records: Vec<ProximaRecord>,
    ) -> StreamResult<PushResult> {
        let start = Instant::now();
        let record_count = records.len();

        // Check rate limit
        if !self.rate_limiter.try_acquire(record_count as u64) {
            self.metrics.rate_limit_rejections.inc();
            return Err(StreamError::RateLimited {
                current_rate: record_count as u64,
                max_rate: self.rate_limiter.rate(),
                retry_after_ms: 100,
            });
        }

        // Get session
        let session =
            self.sessions
                .get(session_id)
                .ok_or_else(|| StreamError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        // Check session state
        if !session.state.load().accepts_records() {
            return Err(StreamError::SessionClosed {
                session_id: session_id.to_string(),
            });
        }

        // Update activity
        session.touch();

        // Push records to buffer
        let mut pushed = 0;
        let mut dropped = 0;

        for record in records {
            match session.buffer.try_push(record) {
                Ok(()) => pushed += 1,
                Err(_) => dropped += 1,
            }
        }

        // Get backpressure level
        let backpressure = session.buffer.backpressure_level();
        let buffer_percent = session.buffer.utilization_percent();

        // Update metrics
        session.increment_received(pushed as u64);
        self.metrics
            .record_received(&session.collection, pushed as u64);

        if dropped > 0 {
            self.metrics.record_dropped(dropped as u64);
            warn!(
                session_id = %session_id,
                dropped = dropped,
                "Records dropped due to buffer full"
            );
        }

        if backpressure != BackpressureLevel::None {
            self.metrics.record_backpressure(backpressure);
        }

        // Record latency
        let latency = start.elapsed();
        self.metrics
            .record_ingestion_latency(&session.collection, latency.as_secs_f64());

        // Update buffer utilization metric
        self.metrics
            .set_buffer_utilization(session_id.as_str(), buffer_percent as f64 / 100.0);

        debug!(
            session_id = %session_id,
            pushed = pushed,
            dropped = dropped,
            backpressure = ?backpressure,
            latency_us = latency.as_micros(),
            "Pushed records to stream"
        );

        Ok(PushResult {
            pushed,
            dropped,
            backpressure,
            buffer_percent,
        })
    }

    /// Drain records from a session's buffer
    ///
    /// This is typically called by a flush task to persist buffered records.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session to drain from
    /// * `max` - Maximum number of records to drain
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<VectorRecord>)` - Drained records
    /// * `Err(StreamError)` - If drain fails
    pub fn drain_records(
        &self,
        session_id: &StreamId,
        max: usize,
    ) -> StreamResult<Vec<ProximaRecord>> {
        let session =
            self.sessions
                .get(session_id)
                .ok_or_else(|| StreamError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let records = session.buffer.drain(max);
        let count = records.len();

        if count > 0 {
            session.increment_processed(count as u64);
            self.metrics.record_processed(count as u64);
            self.metrics.batch_size.observe(count as f64);
        }

        Ok(records)
    }

    /// Close a streaming session
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session to close
    pub fn close_session(&self, session_id: &StreamId) {
        if let Some((_, session)) = self.sessions.remove(session_id) {
            session.transition_to(SessionState::Closed);
            self.metrics.dec_active_streams();
            self.metrics.sessions_closed_total.inc();

            info!(
                session_id = %session_id,
                records_received = session.records_received.load(),
                records_processed = session.records_processed.load(),
                age_secs = session.age_secs(),
                "Closed streaming session"
            );
        }
    }

    /// Get information about a session
    pub fn get_session_info(&self, session_id: &StreamId) -> Option<super::SessionStats> {
        self.sessions.get(session_id).map(|s| s.stats())
    }

    /// List all active session IDs
    pub fn list_sessions(&self) -> Vec<StreamId> {
        self.sessions.iter().map(|e| e.key().clone()).collect()
    }

    /// Get the number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Check and close timed out sessions
    pub fn cleanup_timed_out_sessions(&self) {
        let timeout_secs = self.config.session_timeout.as_secs();
        let mut timed_out = Vec::new();

        for entry in &self.sessions {
            if entry.is_timed_out(timeout_secs) {
                timed_out.push(entry.key().clone());
            }
        }

        for session_id in timed_out {
            warn!(session_id = %session_id, "Session timed out");
            self.close_session(&session_id);
            self.metrics.sessions_timed_out_total.inc();
        }
    }

    /// Start the background cleanup task
    pub fn start_cleanup_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let coordinator = self;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                if coordinator
                    .shutdown
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }

                coordinator.cleanup_timed_out_sessions();
            }
        })
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Close all sessions
        let session_ids: Vec<_> = self.sessions.iter().map(|e| e.key().clone()).collect();
        for session_id in session_ids {
            self.close_session(&session_id);
        }

        info!("Stream coordinator shutdown complete");
    }

    /// Get reference to metrics
    pub fn metrics(&self) -> &Arc<StreamMetrics> {
        &self.metrics
    }

    /// Get reference to configuration
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Get the collection name for a session
    pub fn get_session_collection(&self, session_id: &StreamId) -> Option<String> {
        self.sessions.get(session_id).map(|s| s.collection.clone())
    }

    /// Flush buffered records to storage engine
    ///
    /// This method drains records from the session's buffer and persists them
    /// to the provided storage engine using batch_insert.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session to flush
    /// * `storage` - The storage engine to flush to
    /// * `collection_config` - Optional collection configuration for the storage engine
    ///
    /// # Returns
    ///
    /// * `Ok(FlushStats)` - Statistics about the flush operation
    /// * `Err(StreamError)` - If flush fails
    pub async fn flush_to_storage(
        &self,
        session_id: &StreamId,
        storage: &dyn UnifiedStorageFormat,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> StreamResult<FlushStats> {
        let start = Instant::now();

        // Get session info to verify it exists
        let session_info =
            self.sessions
                .get(session_id)
                .ok_or_else(|| StreamError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let collection_id = session_info.collection.clone();
        drop(session_info); // Release lock before draining

        // Drain records from buffer
        let max_flush_batch = self.config.default_buffer_size.min(10_000);
        let records = self.drain_records(session_id, max_flush_batch)?;

        if records.is_empty() {
            return Ok(FlushStats::default());
        }

        let count = records.len();
        let bytes_estimate = records
            .iter()
            .map(|r| {
                r.oid.len()
                    + r.embeddings
                        .first()
                        .map(|e| e.values.len() * 4)
                        .unwrap_or(0)
                    + 100
            })
            .sum::<usize>();

        // Create flush parameters for storage engine
        let flush_params = crate::storage::traits::FlushParameters {
            collection_id: Some(collection_id.clone()),
            force: true,
            synchronous: true,
            vector_records: records,
            trigger_compaction: false,
            collection_config: collection_config.cloned(),
            estimated_size: bytes_estimate,
            ..Default::default()
        };

        // Flush to storage engine
        let flush_result =
            storage
                .flush(flush_params)
                .await
                .map_err(|e| StreamError::StorageError {
                    message: format!("Storage flush failed: {}", e),
                })?;

        let elapsed = start.elapsed();

        // Record metrics
        self.metrics.flush_latency.observe(elapsed.as_secs_f64());

        info!(
            session_id = %session_id,
            collection = %collection_id,
            records_flushed = count,
            bytes_written = ?flush_result.bytes_written,
            duration_ms = elapsed.as_millis(),
            "Flushed stream records to storage"
        );

        Ok(FlushStats {
            records_flushed: count,
            bytes_written: flush_result.bytes_written.unwrap_or(bytes_estimate as u64) as usize,
            flush_duration: elapsed,
            collection_id,
            success: flush_result.success,
            retry_count: 0,
            was_retried: false,
        })
    }

    /// Start background flush task that periodically flushes all active sessions
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage engine to flush to (wrapped in Arc for sharing)
    /// * `interval` - How often to flush (defaults to config.flush_interval)
    /// * `collection_provider` - Optional provider for collection configs
    ///
    /// # Returns
    ///
    /// A JoinHandle for the background task
    pub fn start_flush_task(
        self: Arc<Self>,
        storage: Arc<dyn UnifiedStorageFormat>,
        interval: Option<Duration>,
    ) -> JoinHandle<()> {
        let flush_interval = interval.unwrap_or(self.config.flush_interval);
        let coordinator = self;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(flush_interval);

            loop {
                ticker.tick().await;

                // Check if shutdown is requested
                if coordinator
                    .shutdown
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    info!("Flush task shutting down");
                    break;
                }

                // Get all active session IDs
                let session_ids: Vec<StreamId> = coordinator
                    .sessions
                    .iter()
                    .filter(|entry| entry.state.load().accepts_records())
                    .map(|entry| entry.key().clone())
                    .collect();

                // Flush each session
                for session_id in session_ids {
                    // Get session info for collection lookup
                    let _collection = match coordinator.get_session_collection(&session_id) {
                        Some(c) => c,
                        None => continue,
                    };

                    // Attempt to flush (no collection config in this simple version)
                    match coordinator
                        .flush_to_storage(&session_id, storage.as_ref(), None)
                        .await
                    {
                        Ok(stats) => {
                            if stats.records_flushed > 0 {
                                debug!(
                                    session_id = %session_id,
                                    records = stats.records_flushed,
                                    "Background flush completed"
                                );
                            }
                        }
                        Err(e) => {
                            warn!(
                                session_id = %session_id,
                                error = %e,
                                "Background flush failed"
                            );
                        }
                    }
                }
            }
        })
    }

    /// Flush all sessions immediately (useful for graceful shutdown)
    pub async fn flush_all(
        &self,
        storage: &dyn UnifiedStorageFormat,
    ) -> Vec<(StreamId, StreamResult<FlushStats>)> {
        let session_ids: Vec<StreamId> = self.sessions.iter().map(|e| e.key().clone()).collect();

        let mut results = Vec::with_capacity(session_ids.len());

        for session_id in session_ids {
            let result = self.flush_to_storage(&session_id, storage, None).await;
            results.push((session_id, result));
        }

        results
    }

    /// Flush buffered records to storage engine with retry logic
    ///
    /// This method implements exponential backoff retry for transient failures.
    /// It will retry up to `retry_config.max_retries` times before giving up.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session to flush
    /// * `storage` - The storage engine to flush to
    /// * `collection_config` - Optional collection configuration
    /// * `retry_config` - Configuration for retry behavior
    ///
    /// # Returns
    ///
    /// * `Ok(FlushStats)` - Statistics about the flush operation (includes retry info)
    /// * `Err(StreamError)` - If flush fails after all retries
    pub async fn flush_to_storage_with_retry(
        &self,
        session_id: &StreamId,
        storage: &dyn UnifiedStorageFormat,
        collection_config: Option<&crate::proto::proximadb_v1::Collection>,
        retry_config: FlushRetryConfig,
    ) -> StreamResult<FlushStats> {
        let start = Instant::now();
        let mut delay = retry_config.initial_delay;
        let mut attempts = 0;

        // Get session info and drain records once (don't retry the drain)
        let session_info =
            self.sessions
                .get(session_id)
                .ok_or_else(|| StreamError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let collection_id = session_info.collection.clone();
        drop(session_info);

        // Drain records from buffer
        let max_flush_batch = self.config.default_buffer_size.min(10_000);
        let records = self.drain_records(session_id, max_flush_batch)?;

        if records.is_empty() {
            return Ok(FlushStats::default());
        }

        let count = records.len();
        let bytes_estimate = records
            .iter()
            .map(|r| {
                r.oid.len()
                    + r.embeddings
                        .first()
                        .map(|e| e.values.len() * 4)
                        .unwrap_or(0)
                    + 100
            })
            .sum::<usize>();

        // Create flush parameters
        let flush_params = crate::storage::traits::FlushParameters {
            collection_id: Some(collection_id.clone()),
            force: true,
            synchronous: true,
            vector_records: records,
            trigger_compaction: false,
            collection_config: collection_config.cloned(),
            estimated_size: bytes_estimate,
            ..Default::default()
        };

        // Retry loop
        loop {
            attempts += 1;

            match storage.flush(flush_params.clone()).await {
                Ok(flush_result) => {
                    let elapsed = start.elapsed();

                    // Record metrics
                    self.metrics.flush_latency.observe(elapsed.as_secs_f64());

                    let was_retried = attempts > 1;
                    if was_retried {
                        info!(
                            session_id = %session_id,
                            collection = %collection_id,
                            records_flushed = count,
                            attempts = attempts,
                            duration_ms = elapsed.as_millis(),
                            "Flush succeeded after retries"
                        );
                    } else {
                        debug!(
                            session_id = %session_id,
                            collection = %collection_id,
                            records_flushed = count,
                            duration_ms = elapsed.as_millis(),
                            "Flushed stream records to storage"
                        );
                    }

                    return Ok(FlushStats {
                        records_flushed: count,
                        bytes_written: flush_result.bytes_written.unwrap_or(bytes_estimate as u64)
                            as usize,
                        flush_duration: elapsed,
                        collection_id,
                        success: flush_result.success,
                        retry_count: attempts - 1,
                        was_retried,
                    });
                }
                Err(e) => {
                    let error_msg = format!("Storage flush failed (attempt {}): {}", attempts, e);
                    warn!(
                        session_id = %session_id,
                        attempt = attempts,
                        max_attempts = retry_config.max_retries + 1,
                        error = %e,
                        "Flush attempt failed"
                    );

                    let _last_error = Some(StreamError::StorageError { message: error_msg });

                    // Check if we should retry
                    if attempts > retry_config.max_retries {
                        let elapsed = start.elapsed();
                        return Err(StreamError::StorageError {
                            message: format!(
                                "Storage flush failed after {} attempts (took {:?}): {}",
                                attempts, elapsed, e
                            ),
                        });
                    }

                    // Exponential backoff
                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(
                        Duration::from_secs_f64(
                            delay.as_secs_f64() * retry_config.backoff_multiplier,
                        ),
                        retry_config.max_delay,
                    );
                }
            }
        }
    }

    /// Flush all sessions with retry logic
    pub async fn flush_all_with_retry(
        &self,
        storage: &dyn UnifiedStorageFormat,
        retry_config: FlushRetryConfig,
    ) -> Vec<(StreamId, StreamResult<FlushStats>)> {
        let session_ids: Vec<StreamId> = self.sessions.iter().map(|e| e.key().clone()).collect();

        let mut results = Vec::with_capacity(session_ids.len());

        for session_id in session_ids {
            let result = self
                .flush_to_storage_with_retry(&session_id, storage, None, retry_config.clone())
                .await;
            results.push((session_id, result));
        }

        results
    }

    /// Get total buffered record count across all sessions
    pub fn total_buffered_records(&self) -> usize {
        self.sessions.iter().map(|s| s.buffer.len()).sum()
    }

    /// Get a snapshot of coordinator statistics
    pub fn stats(&self) -> CoordinatorStats {
        CoordinatorStats {
            active_sessions: self.sessions.len(),
            total_buffered_records: self.total_buffered_records(),
            sessions_created: self.metrics.sessions_created_total.get() as u64,
            sessions_closed: self.metrics.sessions_closed_total.get() as u64,
            rate_limit_rejections: self.metrics.rate_limit_rejections.get() as u64,
        }
    }
}

/// Coordinator statistics snapshot
#[derive(Debug, Clone)]
pub struct CoordinatorStats {
    /// Number of currently active sessions
    pub active_sessions: usize,
    /// Total records currently buffered across all sessions
    pub total_buffered_records: usize,
    /// Total sessions created since startup
    pub sessions_created: u64,
    /// Total sessions closed since startup
    pub sessions_closed: u64,
    /// Total rate limit rejections
    pub rate_limit_rejections: u64,
}

/// Statistics from a flush operation
#[derive(Debug, Clone, Default)]
pub struct FlushStats {
    /// Number of records successfully flushed
    pub records_flushed: usize,

    /// Approximate bytes written to storage
    pub bytes_written: usize,

    /// Time taken for the flush operation
    pub flush_duration: Duration,

    /// Collection that was flushed
    pub collection_id: String,

    /// Whether the storage flush was successful
    pub success: bool,

    /// Number of retry attempts (0 = succeeded on first try)
    pub retry_count: u32,

    /// Whether this flush was retried
    pub was_retried: bool,
}

/// Configuration for flush retry behavior
#[derive(Debug, Clone)]
pub struct FlushRetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay between retries (exponential backoff)
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for FlushRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for StreamCoordinator {
    fn default() -> Self {
        Self::new(StreamConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_session() {
        let coordinator = StreamCoordinator::default();

        let session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        assert!(coordinator.get_session_info(&session_id).is_some());
        assert_eq!(coordinator.session_count(), 1);
    }

    #[tokio::test]
    async fn test_push_records() {
        let coordinator = StreamCoordinator::default();

        let session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        let records: Vec<ProximaRecord> = (0..100)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![0.1; 128],
                )],
                record_version: 1,
                ..Default::default()
            })
            .collect();

        let result = coordinator
            .push_records(&session_id, records)
            .await
            .expect("Failed to push records");

        assert_eq!(result.pushed, 100);
        assert_eq!(result.dropped, 0);
        assert_eq!(result.backpressure, BackpressureLevel::None);
    }

    #[tokio::test]
    async fn test_drain_records() {
        let coordinator = StreamCoordinator::default();

        let session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        let records: Vec<ProximaRecord> = (0..100)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![0.1; 128],
                )],
                record_version: 1,
                ..Default::default()
            })
            .collect();

        coordinator
            .push_records(&session_id, records)
            .await
            .expect("Failed to push records");

        let drained = coordinator
            .drain_records(&session_id, 50)
            .expect("Failed to drain records");
        assert_eq!(drained.len(), 50);

        let remaining = coordinator
            .drain_records(&session_id, 100)
            .expect("Failed to drain remaining records");
        assert_eq!(remaining.len(), 50);
    }

    #[tokio::test]
    async fn test_close_session() {
        let coordinator = StreamCoordinator::default();

        let session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        assert_eq!(coordinator.session_count(), 1);

        coordinator.close_session(&session_id);

        assert_eq!(coordinator.session_count(), 0);
        assert!(coordinator.get_session_info(&session_id).is_none());
    }

    #[tokio::test]
    async fn test_max_sessions() {
        let config = StreamConfig {
            max_streams: 2,
            ..Default::default()
        };
        let coordinator = StreamCoordinator::new(config);

        coordinator
            .create_session("col1".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session col1");
        coordinator
            .create_session("col2".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session col2");

        let result = coordinator
            .create_session("col3".to_string(), SessionConfig::default())
            .await;

        assert!(matches!(result, Err(StreamError::TooManySessions { .. })));
    }

    #[tokio::test]
    async fn test_session_not_found() {
        let coordinator = StreamCoordinator::default();

        let fake_id = StreamId::new();
        let result = coordinator.push_records(&fake_id, vec![]).await;

        assert!(matches!(result, Err(StreamError::SessionNotFound { .. })));
    }
}
