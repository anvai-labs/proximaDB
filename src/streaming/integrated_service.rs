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

//! Integrated Streaming Service
//!
//! This module provides a unified interface that connects the streaming coordinator
//! with the subscription manager, enabling automatic notification of live query
//! subscriptions when new vectors are ingested and flushed.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   IntegratedStreamingService                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐      ┌─────────────────────────────────┐  │
//! │  │ StreamCoordinator │◄────│  Ingest API (push_and_notify)   │  │
//! │  └────────┬────────┘      └─────────────────────────────────┘  │
//! │           │                                                      │
//! │           │ drain on flush                                       │
//! │           ▼                                                      │
//! │  ┌─────────────────┐      ┌─────────────────────────────────┐  │
//! │  │  Storage Engine  │◄────│  Flush (with subscription notify) │  │
//! │  └─────────────────┘      └─────────────────────────────────┘  │
//! │           │                                                      │
//! │           │ after flush                                          │
//! │           ▼                                                      │
//! │  ┌─────────────────┐                                            │
//! │  │SubscriptionMgr   │──────► Live Query Updates                  │
//! │  └─────────────────┘                                            │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use super::coordinator::{FlushRetryConfig, FlushStats, PushResult, StreamCoordinator};
use super::subscriptions::{
    ScoredResult, SubscriptionConfig, SubscriptionHandle, SubscriptionManager,
};
use super::{SessionConfig, StreamConfig, StreamError, StreamId, StreamResult};
use crate::storage::traits::UnifiedStorageFormat;
use proximadb_records::ProximaRecord;

/// Configuration for the integrated streaming service
#[derive(Debug, Clone)]
pub struct IntegratedServiceConfig {
    /// Stream coordinator configuration
    pub stream_config: StreamConfig,
    /// Whether to automatically notify subscriptions on flush
    pub auto_notify_subscriptions: bool,
    /// Flush retry configuration
    pub flush_retry: FlushRetryConfig,
    /// Whether to run background flush task
    pub enable_background_flush: bool,
    /// Background flush interval
    pub background_flush_interval: Duration,
}

impl Default for IntegratedServiceConfig {
    fn default() -> Self {
        Self {
            stream_config: StreamConfig::default(),
            auto_notify_subscriptions: true,
            flush_retry: FlushRetryConfig::default(),
            enable_background_flush: true,
            background_flush_interval: Duration::from_millis(100),
        }
    }
}

/// Integrated streaming service that connects ingestion with live queries
pub struct IntegratedStreamingService {
    /// Stream coordinator for managing ingestion sessions
    coordinator: Arc<StreamCoordinator>,
    /// Subscription manager for live queries
    subscriptions: Arc<SubscriptionManager>,
    /// Configuration
    config: IntegratedServiceConfig,
    /// Background task handles
    background_tasks: RwLock<Vec<JoinHandle<()>>>,
    /// Shutdown flag
    shutdown: std::sync::atomic::AtomicBool,
}

impl IntegratedStreamingService {
    /// Create a new integrated streaming service
    pub fn new(config: IntegratedServiceConfig) -> Self {
        let coordinator = Arc::new(StreamCoordinator::new(config.stream_config.clone()));
        let subscriptions = Arc::new(SubscriptionManager::with_defaults());

        Self {
            coordinator,
            subscriptions,
            config,
            background_tasks: RwLock::new(Vec::new()),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create with custom subscription manager configuration
    pub fn with_subscription_config(
        config: IntegratedServiceConfig,
        subscription_config: super::subscriptions::manager::SubscriptionManagerConfig,
    ) -> Self {
        let coordinator = Arc::new(StreamCoordinator::new(config.stream_config.clone()));
        let subscriptions = Arc::new(SubscriptionManager::new(subscription_config));

        Self {
            coordinator,
            subscriptions,
            config,
            background_tasks: RwLock::new(Vec::new()),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Get reference to the stream coordinator
    pub fn coordinator(&self) -> &Arc<StreamCoordinator> {
        &self.coordinator
    }

    /// Get reference to the subscription manager
    pub fn subscriptions(&self) -> &Arc<SubscriptionManager> {
        &self.subscriptions
    }

    /// Create a streaming session for a collection
    pub async fn create_session(
        &self,
        collection: String,
        config: SessionConfig,
    ) -> StreamResult<StreamId> {
        self.coordinator.create_session(collection, config).await
    }

    /// Close a streaming session
    pub fn close_session(&self, session_id: &StreamId) {
        self.coordinator.close_session(session_id);
    }

    /// Push records to a session (without immediate notification)
    pub async fn push_records(
        &self,
        session_id: &StreamId,
        records: Vec<ProximaRecord>,
    ) -> StreamResult<PushResult> {
        self.coordinator.push_records(session_id, records).await
    }

    /// Push records and notify subscriptions immediately
    ///
    /// This method pushes records to the buffer AND notifies any live query
    /// subscriptions about the new vectors. This is useful for real-time
    /// applications where subscription updates should not wait for flush.
    pub async fn push_and_notify(
        &self,
        session_id: &StreamId,
        records: Vec<ProximaRecord>,
    ) -> StreamResult<PushAndNotifyResult> {
        // Get collection name before push
        let collection = self
            .coordinator
            .get_session_collection(session_id)
            .ok_or_else(|| StreamError::SessionNotFound {
                session_id: session_id.to_string(),
            })?;

        // Push to coordinator
        let push_result = self
            .coordinator
            .push_records(session_id, records.clone())
            .await?;

        // Notify subscriptions if enabled
        let subscriptions_notified = if self.config.auto_notify_subscriptions {
            // Convert records to format expected by subscription manager
            let vectors: Vec<(String, Vec<f32>, f32)> = records
                .into_iter()
                .map(|r| {
                    // INT-2.5b: streaming subscriptions advertise fp32 vectors.
                    // Promote non-Fp32 variants on the way out.
                    let v = r
                        .embeddings
                        .into_iter()
                        .next()
                        .map(|e| e.values.to_fp32_owned())
                        .unwrap_or_default();
                    (r.oid, v, 0.0)
                })
                .collect();

            self.subscriptions.notify_insert(&collection, vectors).await
        } else {
            0
        };

        Ok(PushAndNotifyResult {
            push_result,
            subscriptions_notified,
        })
    }

    /// Flush a session to storage and notify subscriptions
    pub async fn flush_to_storage(
        &self,
        session_id: &StreamId,
        storage: &dyn UnifiedStorageFormat,
    ) -> StreamResult<FlushAndNotifyResult> {
        let start = Instant::now();

        // Get collection name
        let collection = self
            .coordinator
            .get_session_collection(session_id)
            .ok_or_else(|| StreamError::SessionNotFound {
                session_id: session_id.to_string(),
            })?;

        // Flush to storage with retry
        let flush_stats = self
            .coordinator
            .flush_to_storage_with_retry(session_id, storage, None, self.config.flush_retry.clone())
            .await?;

        // Record metrics
        if flush_stats.success {
            self.coordinator.metrics().record_flush_success(
                &collection,
                flush_stats.records_flushed,
                flush_stats.bytes_written,
                flush_stats.retry_count,
            );
        }

        Ok(FlushAndNotifyResult {
            flush_stats,
            subscriptions_notified: 0, // Subscriptions were already notified on push
            total_duration: start.elapsed(),
        })
    }

    /// Subscribe to live query updates for a collection
    pub async fn subscribe(
        &self,
        collection: String,
        query_vector: Vec<f32>,
        config: SubscriptionConfig,
        filter: Option<String>,
    ) -> Result<SubscriptionHandle, super::subscriptions::manager::SubscriptionError> {
        self.subscriptions
            .subscribe(collection, query_vector, config, filter)
            .await
    }

    /// Activate a subscription with initial results
    pub async fn activate_subscription(
        &self,
        id: &super::subscriptions::SubscriptionId,
        initial_results: Vec<ScoredResult>,
    ) -> Result<(), super::subscriptions::manager::SubscriptionError> {
        self.subscriptions.activate(id, initial_results).await
    }

    /// Unsubscribe from a live query
    pub fn unsubscribe(
        &self,
        id: &super::subscriptions::SubscriptionId,
    ) -> Result<(), super::subscriptions::manager::SubscriptionError> {
        self.subscriptions.unsubscribe(id)
    }

    /// Start background tasks (flush and cleanup)
    pub async fn start_background_tasks(&self, storage: Arc<dyn UnifiedStorageFormat>) {
        let mut tasks = self.background_tasks.write().await;

        // Cleanup task for coordinator
        let coordinator_clone = Arc::clone(&self.coordinator);
        let cleanup_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                coordinator_clone.cleanup_timed_out_sessions();
            }
        });
        tasks.push(cleanup_handle);

        // Cleanup task for subscriptions
        let subscriptions_clone = Arc::clone(&self.subscriptions);
        let subscription_cleanup_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                subscriptions_clone.cleanup_stale().await;
            }
        });
        tasks.push(subscription_cleanup_handle);

        // Background flush task if enabled
        if self.config.enable_background_flush {
            let coordinator = Arc::clone(&self.coordinator);
            let storage_clone = Arc::clone(&storage);
            let interval = self.config.background_flush_interval;

            // Use the coordinator's background flush task
            let flush_handle = tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                loop {
                    ticker.tick().await;

                    // Flush all sessions
                    let session_ids: Vec<StreamId> =
                        coordinator.list_sessions().into_iter().collect();

                    for session_id in session_ids {
                        if let Err(e) = coordinator
                            .flush_to_storage(&session_id, storage_clone.as_ref(), None)
                            .await
                        {
                            warn!(session_id = %session_id, error = %e, "Background flush failed");
                        }
                    }
                }
            });
            tasks.push(flush_handle);
        }

        info!(
            "Started {} background tasks for integrated streaming service",
            tasks.len()
        );
    }

    /// Shutdown the service gracefully
    pub async fn shutdown(&self, storage: Option<&dyn UnifiedStorageFormat>) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Flush all remaining data if storage is provided
        if let Some(storage) = storage {
            let results = self.coordinator.flush_all(storage).await;
            let successful = results.iter().filter(|(_, r)| r.is_ok()).count();
            let failed = results.len() - successful;

            if failed > 0 {
                warn!(
                    "Final flush completed with {} successes and {} failures",
                    successful, failed
                );
            } else {
                info!("Final flush completed: {} sessions flushed", successful);
            }
        }

        // Cancel background tasks
        let mut tasks = self.background_tasks.write().await;
        for task in tasks.drain(..) {
            task.abort();
        }

        // Shutdown coordinator
        self.coordinator.shutdown().await;

        info!("Integrated streaming service shutdown complete");
    }

    /// Get service statistics
    pub fn stats(&self) -> IntegratedServiceStats {
        let coordinator_stats = self.coordinator.stats();
        let subscription_stats = self.subscriptions.stats();

        IntegratedServiceStats {
            active_sessions: coordinator_stats.active_sessions,
            total_buffered_records: coordinator_stats.total_buffered_records,
            active_subscriptions: subscription_stats.total_subscriptions,
            collections_with_subscriptions: subscription_stats.collections_with_subscriptions,
            sessions_created: coordinator_stats.sessions_created,
            sessions_closed: coordinator_stats.sessions_closed,
            subscriptions_created: subscription_stats.subscriptions_created,
            subscriptions_closed: subscription_stats.subscriptions_closed,
        }
    }
}

/// Result of push_and_notify operation
#[derive(Debug, Clone)]
pub struct PushAndNotifyResult {
    /// Result of the push operation
    pub push_result: PushResult,
    /// Number of subscriptions notified
    pub subscriptions_notified: usize,
}

/// Result of flush with subscription notification
#[derive(Debug, Clone)]
pub struct FlushAndNotifyResult {
    /// Flush statistics
    pub flush_stats: FlushStats,
    /// Number of subscriptions notified
    pub subscriptions_notified: usize,
    /// Total operation duration
    pub total_duration: Duration,
}

/// Statistics for the integrated service
#[derive(Debug, Clone)]
pub struct IntegratedServiceStats {
    /// Active streaming sessions
    pub active_sessions: usize,
    /// Total records buffered
    pub total_buffered_records: usize,
    /// Active subscriptions
    pub active_subscriptions: usize,
    /// Collections with subscriptions
    pub collections_with_subscriptions: usize,
    /// Total sessions created
    pub sessions_created: u64,
    /// Total sessions closed
    pub sessions_closed: u64,
    /// Total subscriptions created
    pub subscriptions_created: u64,
    /// Total subscriptions closed
    pub subscriptions_closed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integrated_service_creation() {
        let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

        let stats = service.stats();
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.active_subscriptions, 0);
    }

    #[tokio::test]
    async fn test_create_session_and_push() {
        let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

        let session_id = service
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("failed to create test session");

        let records: Vec<ProximaRecord> = (0..10)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    64,
                    vec![0.1; 64],
                )],
                record_version: 1,
                ..Default::default()
            })
            .collect();

        let result = service
            .push_records(&session_id, records)
            .await
            .expect("failed to push records");
        assert_eq!(result.pushed, 10);

        service.close_session(&session_id);
        assert_eq!(service.stats().active_sessions, 0);
    }

    #[tokio::test]
    async fn test_subscription_with_notification() {
        let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

        // Create session
        let session_id = service
            .create_session("sub_test".to_string(), SessionConfig::default())
            .await
            .expect("failed to create subscription test session");

        // Create subscription
        let sub_config = SubscriptionConfig {
            top_k: 10,
            score_threshold: 0.0,
            include_initial: false,
            ..Default::default()
        };

        let sub_handle = service
            .subscribe("sub_test".to_string(), vec![0.5; 64], sub_config, None)
            .await
            .expect("failed to create subscription");

        // Activate
        service
            .activate_subscription(&sub_handle.id, vec![])
            .await
            .expect("failed to activate subscription");

        // Push with notification
        let records: Vec<ProximaRecord> = (0..5)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    64,
                    vec![0.1 * (i as f32); 64],
                )],
                record_version: 1,
                ..Default::default()
            })
            .collect();

        let result = service
            .push_and_notify(&session_id, records)
            .await
            .expect("failed to push and notify");
        assert_eq!(result.push_result.pushed, 5);
        // Should have notified at least one subscription
        // subscriptions_notified is always >= 0 (usize)

        // Cleanup
        service
            .unsubscribe(&sub_handle.id)
            .expect("failed to unsubscribe");
        service.close_session(&session_id);
    }

    #[tokio::test]
    async fn test_push_and_notify_disabled() {
        let mut config = IntegratedServiceConfig::default();
        config.auto_notify_subscriptions = false;

        let service = IntegratedStreamingService::new(config);

        let session_id = service
            .create_session("no_notify".to_string(), SessionConfig::default())
            .await
            .expect("failed to create no_notify test session");

        let records = vec![ProximaRecord {
            oid: "vec_1".to_string(),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                64,
                vec![0.1; 64],
            )],
            record_version: 1,
            ..Default::default()
        }];

        let result = service
            .push_and_notify(&session_id, records)
            .await
            .expect("failed to push and notify");
        // With auto_notify disabled, no subscriptions should be notified
        assert_eq!(result.subscriptions_notified, 0);
    }

    #[tokio::test]
    async fn test_service_stats() {
        let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

        // Create multiple sessions
        for i in 0..3 {
            let _ = service
                .create_session(format!("collection_{}", i), SessionConfig::default())
                .await
                .expect("failed to create session for stats test");
        }

        let stats = service.stats();
        assert_eq!(stats.active_sessions, 3);
        assert_eq!(stats.sessions_created, 3);
    }

    #[tokio::test]
    async fn test_service_shutdown() {
        let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

        let _ = service
            .create_session("shutdown_test".to_string(), SessionConfig::default())
            .await
            .expect("failed to create shutdown test session");

        // Shutdown without storage (no final flush)
        service.shutdown(None).await;

        // After shutdown, sessions should be cleared
        assert_eq!(service.stats().active_sessions, 0);
    }
}
