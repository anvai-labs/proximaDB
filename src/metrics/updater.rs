// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Internal-only metrics update interface
//!
//! This module provides the write path for metrics that is only accessible
//! to internal system components. All updates are non-blocking and failure-tolerant.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, warn};

use super::{
    schema::{CollectionMetrics, FilterableColumnStats},
    store::MetricsPersistenceLayer,
};

const METRICS_UPDATE_CHANNEL_CAPACITY: usize = 10_000;

/// Internal interface for updating metrics - not exposed to external users
#[async_trait]
pub trait InternalMetricsUpdater: Send + Sync {
    /// Record a vector operation (insert/update/delete/search)
    async fn record_operation(
        &self,
        collection_id: &str,
        update: OperationMetricsUpdate,
    ) -> Result<()>;

    /// Record search metrics
    async fn record_search(&self, collection_id: &str, update: SearchMetricsUpdate) -> Result<()>;

    /// Update metrics after a flush operation
    async fn record_flush(&self, collection_id: &str, update: FlushMetricsUpdate) -> Result<()>;

    /// Update metrics after a compaction operation
    async fn record_compaction(
        &self,
        collection_id: &str,
        update: CompactionMetricsUpdate,
    ) -> Result<()>;

    /// Update storage metrics
    async fn update_storage_metrics(
        &self,
        collection_id: &str,
        update: StorageMetricsUpdate,
    ) -> Result<()>;

    /// Update data characteristics for optimization
    async fn update_data_characteristics(
        &self,
        collection_id: &str,
        update: DataCharacteristicsUpdate,
    ) -> Result<()>;

    /// Update filterable column statistics
    async fn update_column_stats(
        &self,
        collection_id: &str,
        column_name: &str,
        stats: FilterableColumnStats,
    ) -> Result<()>;
}

/// Types of vector operations
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    Insert,
    Update,
    Delete,
    Search,
}

/// Metrics update from flush operations
#[derive(Debug, Clone)]
pub struct FlushMetricsUpdate {
    pub vectors_flushed: i64,
    pub bytes_written: i64,
    pub duration_ms: i64,
    pub files_created: i32,
    pub engine_type: String, // "NOVA" or "SST"
    pub timestamp: i64,
}

/// Metrics update from compaction operations
#[derive(Debug, Clone)]
pub struct CompactionMetricsUpdate {
    pub files_before: i32,
    pub files_after: i32,
    pub bytes_before: i64,
    pub bytes_after: i64,
    pub duration_ms: i64,
    pub timestamp: i64,
}

/// Metrics update from search operations
#[derive(Debug, Clone)]
pub struct SearchMetricsUpdate {
    pub query_latency_us: f64,
    pub results_count: i32,
    pub vectors_scanned: i64,
    pub cache_hit: bool,
    pub index_used: String,
    pub timestamp: i64,
}

/// Metrics update from general operations (insert/update/delete)
#[derive(Debug, Clone)]
pub struct OperationMetricsUpdate {
    pub operation_type: String,
    pub latency_us: f64,
    pub success: bool,
    pub bytes_processed: usize,
    pub timestamp: i64,
}

/// Storage layer metrics update
#[derive(Debug, Clone)]
pub struct StorageMetricsUpdate {
    pub parquet_file_count: Option<i32>,
    pub sstable_file_count: Option<i32>,
    pub wal_size_bytes: Option<i64>,
    pub memtable_size_bytes: Option<i64>,
    pub bloom_filter_size_bytes: Option<i64>,
    pub cache_size_bytes: Option<i64>,
}

/// Data characteristics update for optimization
#[derive(Debug, Clone)]
pub struct DataCharacteristicsUpdate {
    pub sparsity_ratio: Option<f32>,
    pub avg_vector_magnitude: Option<f32>,
    pub distinct_metadata_keys: Option<i32>,
    pub avg_metadata_size_bytes: Option<i32>,
}

/// Generic metrics update message
#[derive(Debug, Clone)]
pub enum MetricsUpdate {
    Operation {
        collection_id: String,
        update: OperationMetricsUpdate,
    },
    Search {
        collection_id: String,
        update: SearchMetricsUpdate,
    },
    Flush {
        collection_id: String,
        update: FlushMetricsUpdate,
    },
    Compaction {
        collection_id: String,
        update: CompactionMetricsUpdate,
    },
    Storage {
        collection_id: String,
        update: StorageMetricsUpdate,
    },
    DataCharacteristics {
        collection_id: String,
        update: DataCharacteristicsUpdate,
    },
    ColumnStats {
        collection_id: String,
        column_name: String,
        // Boxed: FilterableColumnStats is the largest payload by far
        // (~300 B vs ~100 B for the next-largest variant), which trips
        // clippy::large_enum_variant's 3× ratio; the box keeps the enum
        // at pointer size.
        stats: Box<FilterableColumnStats>,
    },
}

/// Async metrics updater that processes updates in the background
pub struct AsyncMetricsUpdater {
    /// Channel for sending updates
    tx: mpsc::Sender<MetricsUpdate>,

    /// Metrics store (shared with reader)
    metrics_cache: Arc<RwLock<std::collections::HashMap<String, CollectionMetrics>>>,

    /// Error counter for monitoring
    error_count: Arc<AtomicU64>,

    /// Update counter for monitoring
    update_count: Arc<AtomicU64>,
}

impl AsyncMetricsUpdater {
    /// Create a new async metrics updater
    pub fn new(
        metrics_cache: Arc<RwLock<std::collections::HashMap<String, CollectionMetrics>>>,
    ) -> (Self, mpsc::Receiver<MetricsUpdate>) {
        let (tx, rx) = mpsc::channel(METRICS_UPDATE_CHANNEL_CAPACITY);

        let updater = Self {
            tx,
            metrics_cache,
            error_count: Arc::new(AtomicU64::new(0)),
            update_count: Arc::new(AtomicU64::new(0)),
        };

        (updater, rx)
    }

    /// Process updates from the channel
    pub async fn process_updates(
        &self,
        mut rx: mpsc::Receiver<MetricsUpdate>,
        store: Arc<super::store::MetricsPersistenceLayer>,
    ) {
        while let Some(update) = rx.recv().await {
            self.update_count.fetch_add(1, Ordering::Relaxed);

            // Apply update to in-memory cache
            if let Err(e) = self.apply_update(&update).await {
                warn!("Failed to apply metrics update (non-critical): {}", e);
                self.error_count.fetch_add(1, Ordering::Relaxed);
            }

            // Optionally persist to store (async, best-effort)
            if let Err(e) = store.record_update(update).await {
                debug!("Failed to persist metrics update (non-critical): {}", e);
                // Don't increment error counter for persistence failures
            }
        }
    }

    /// Apply an update to the in-memory metrics cache
    async fn apply_update(&self, update: &MetricsUpdate) -> Result<()> {
        let mut cache = self.metrics_cache.write().await;

        match update {
            MetricsUpdate::Operation {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                match update.operation_type.as_str() {
                    "insert" => {
                        metrics.total_inserts += 1;
                        if update.success {
                            // Update insert latency (simple moving average)
                            let weight = 0.1; // Weight for new value
                            metrics.avg_insert_latency_us = metrics.avg_insert_latency_us
                                * (1.0 - weight)
                                + update.latency_us * weight;
                        }
                    }
                    "update" => metrics.total_updates += 1,
                    "delete" => metrics.total_deletes += 1,
                    _ => {} // Handle unknown operation types gracefully
                }

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::Search {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                metrics.total_searches += 1;

                // Update search latency (simple moving average)
                let weight = 0.1;
                metrics.avg_search_latency_us = metrics.avg_search_latency_us * (1.0 - weight)
                    + update.query_latency_us * weight;

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::Flush {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                metrics.total_flushes += 1;
                metrics.vector_count += update.vectors_flushed;
                metrics.data_size_bytes += update.bytes_written;
                metrics.last_flush_timestamp = update.timestamp;
                metrics.last_flush_duration_ms = update.duration_ms;

                if update.engine_type == "NOVA" {
                    metrics.parquet_file_count += update.files_created;
                } else if update.engine_type == "SST" {
                    metrics.sstable_file_count += update.files_created;
                }

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::Compaction {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                metrics.total_compactions += 1;
                metrics.last_compaction_timestamp = update.timestamp;
                metrics.last_compaction_duration_ms = update.duration_ms;

                // Update file counts
                let file_reduction = update.files_before - update.files_after;
                if metrics.parquet_file_count > 0 {
                    metrics.parquet_file_count -= file_reduction;
                }

                // Update size
                let size_reduction = update.bytes_before - update.bytes_after;
                if size_reduction > 0 {
                    metrics.data_size_bytes =
                        metrics.data_size_bytes.saturating_sub(size_reduction);
                }

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::Storage {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                if let Some(count) = update.parquet_file_count {
                    metrics.parquet_file_count = count;
                }
                if let Some(count) = update.sstable_file_count {
                    metrics.sstable_file_count = count;
                }
                if let Some(size) = update.wal_size_bytes {
                    metrics.wal_size_bytes = size;
                }
                if let Some(size) = update.memtable_size_bytes {
                    metrics.memtable_size_bytes = size;
                }
                if let Some(size) = update.bloom_filter_size_bytes {
                    metrics.bloom_filter_size_bytes = size;
                }
                if let Some(size) = update.cache_size_bytes {
                    metrics.cache_size_bytes = size;
                }

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::DataCharacteristics {
                collection_id,
                update,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                if let Some(ratio) = update.sparsity_ratio {
                    metrics.sparsity_ratio = ratio;
                }
                if let Some(mag) = update.avg_vector_magnitude {
                    metrics.avg_vector_magnitude = mag;
                }
                if let Some(keys) = update.distinct_metadata_keys {
                    metrics.distinct_metadata_keys = keys;
                }
                if let Some(size) = update.avg_metadata_size_bytes {
                    metrics.avg_metadata_size_bytes = size;
                }

                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }

            MetricsUpdate::ColumnStats {
                collection_id,
                column_name,
                stats,
            } => {
                let metrics =
                    cache
                        .entry(collection_id.clone())
                        .or_insert_with(|| CollectionMetrics {
                            collection_id: collection_id.clone(),
                            ..Default::default()
                        });

                metrics
                    .filterable_column_stats
                    .insert(column_name.clone(), (**stats).clone());
                metrics.updated_at = chrono::Utc::now().timestamp_millis();
            }
        }

        Ok(())
    }

    /// Get current error count
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Get total update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl InternalMetricsUpdater for AsyncMetricsUpdater {
    async fn record_operation(
        &self,
        collection_id: &str,
        update: OperationMetricsUpdate,
    ) -> Result<()> {
        // Fire and forget: drop when saturated so metrics cannot OOM or block callers.
        let _ = self.tx.try_send(MetricsUpdate::Operation {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn record_search(&self, collection_id: &str, update: SearchMetricsUpdate) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::Search {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn record_flush(&self, collection_id: &str, update: FlushMetricsUpdate) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::Flush {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn record_compaction(
        &self,
        collection_id: &str,
        update: CompactionMetricsUpdate,
    ) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::Compaction {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn update_storage_metrics(
        &self,
        collection_id: &str,
        update: StorageMetricsUpdate,
    ) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::Storage {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn update_data_characteristics(
        &self,
        collection_id: &str,
        update: DataCharacteristicsUpdate,
    ) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::DataCharacteristics {
            collection_id: collection_id.to_string(),
            update,
        });
        Ok(())
    }

    async fn update_column_stats(
        &self,
        collection_id: &str,
        column_name: &str,
        stats: FilterableColumnStats,
    ) -> Result<()> {
        let _ = self.tx.try_send(MetricsUpdate::ColumnStats {
            collection_id: collection_id.to_string(),
            column_name: column_name.to_string(),
            stats: Box::new(stats),
        });
        Ok(())
    }
}

/// Metrics update service implementation for testing and simple use cases
pub struct MetricsUpdateService {
    store: Arc<MetricsPersistenceLayer>,
}

impl MetricsUpdateService {
    pub fn new(store: Arc<MetricsPersistenceLayer>) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &Arc<MetricsPersistenceLayer> {
        &self.store
    }
}

#[async_trait]
impl InternalMetricsUpdater for MetricsUpdateService {
    async fn record_operation(
        &self,
        collection_id: &str,
        update: OperationMetricsUpdate,
    ) -> Result<()> {
        // Get or create collection metrics
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        // Update operation counts and latencies
        match update.operation_type.as_str() {
            "insert" => {
                metrics.total_inserts += 1;
                metrics.avg_insert_latency_us = if metrics.total_inserts > 1 {
                    (metrics.avg_insert_latency_us * ((metrics.total_inserts - 1) as f64)
                        + update.latency_us)
                        / (metrics.total_inserts as f64)
                } else {
                    update.latency_us
                };
            }
            "update" => metrics.total_updates += 1,
            "delete" => metrics.total_deletes += 1,
            _ => {}
        }

        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn record_search(&self, collection_id: &str, update: SearchMetricsUpdate) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        metrics.total_searches += 1;

        // Update search latency average
        metrics.avg_search_latency_us = if metrics.total_searches > 1 {
            (metrics.avg_search_latency_us * ((metrics.total_searches - 1) as f64)
                + update.query_latency_us)
                / (metrics.total_searches as f64)
        } else {
            update.query_latency_us
        };

        // Update cache hit ratio
        let total_cache_hits = (metrics.cache_hit_ratio * (metrics.total_searches - 1) as f32)
            + if update.cache_hit { 1.0 } else { 0.0 };
        metrics.cache_hit_ratio = total_cache_hits / (metrics.total_searches as f32);

        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn record_flush(&self, collection_id: &str, update: FlushMetricsUpdate) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        metrics.total_flushes += 1;
        metrics.last_flush_timestamp = update.timestamp;
        metrics.last_flush_duration_ms = update.duration_ms;

        // Update file counts based on engine type
        match update.engine_type.as_str() {
            "NOVA" => metrics.parquet_file_count += update.files_created,
            "SST" => metrics.sstable_file_count += update.files_created,
            _ => {}
        }

        metrics.data_size_bytes += update.bytes_written;
        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn record_compaction(
        &self,
        collection_id: &str,
        update: CompactionMetricsUpdate,
    ) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        metrics.total_compactions += 1;
        metrics.last_compaction_timestamp = update.timestamp;
        metrics.last_compaction_duration_ms = update.duration_ms;

        // Update file counts (compaction reduces files)
        let file_reduction = update.files_before - update.files_after;
        if metrics.parquet_file_count >= file_reduction {
            metrics.parquet_file_count -= file_reduction;
        }

        // Update size (compaction usually reduces size)
        let size_reduction = update.bytes_before - update.bytes_after;
        if size_reduction > 0 {
            metrics.data_size_bytes = metrics.data_size_bytes.saturating_sub(size_reduction);
        }

        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn update_storage_metrics(
        &self,
        collection_id: &str,
        update: StorageMetricsUpdate,
    ) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        if let Some(count) = update.parquet_file_count {
            metrics.parquet_file_count = count;
        }
        if let Some(count) = update.sstable_file_count {
            metrics.sstable_file_count = count;
        }
        if let Some(size) = update.wal_size_bytes {
            metrics.wal_size_bytes = size;
        }
        if let Some(size) = update.memtable_size_bytes {
            metrics.memtable_size_bytes = size;
        }

        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn update_data_characteristics(
        &self,
        collection_id: &str,
        update: DataCharacteristicsUpdate,
    ) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        if let Some(sparsity) = update.sparsity_ratio {
            metrics.sparsity_ratio = sparsity;
        }
        if let Some(magnitude) = update.avg_vector_magnitude {
            metrics.avg_vector_magnitude = magnitude;
        }
        if let Some(keys) = update.distinct_metadata_keys {
            metrics.distinct_metadata_keys = keys;
        }

        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }

    async fn update_column_stats(
        &self,
        collection_id: &str,
        column_name: &str,
        stats: FilterableColumnStats,
    ) -> Result<()> {
        let mut metrics = self
            .store
            .collection_metrics(collection_id)
            .await?
            .unwrap_or_else(|| CollectionMetrics {
                collection_id: collection_id.to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            });

        metrics
            .filterable_column_stats
            .insert(column_name.to_string(), stats);
        metrics.updated_at = chrono::Utc::now().timestamp_millis();
        self.store.store_collection_metrics(&metrics).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MetricsConfig;
    use crate::metrics::store::MetricsPersistenceLayer;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use anyhow::Result;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::time::{Duration, sleep};
    use tracing::{debug, info};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn generate_unique_test_path() -> String {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!(
            "/tmp/proximadb_metrics_updater_test_{}_{}",
            counter, timestamp
        )
    }

    async fn create_test_updater() -> Result<Arc<MetricsUpdateService>> {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let unique_path = generate_unique_test_path();

        if std::path::Path::new(&unique_path).exists() {
            std::fs::remove_dir_all(&unique_path).ok();
        }

        let config = MetricsConfig {
            enabled: true,
            collection_partitions: 4,
            storage_path: format!("file://{}", unique_path),
            flush_interval_seconds: 30,
            retention_days: 7,
            parallel_scan_threshold: 10,
            sparsity_threshold: 0.3,
            quantization_size_threshold: 1_000_000,
            snapshot_interval_seconds: 60,
            max_memory_mb: 512,
        };

        let _ = tokio::fs::remove_dir_all(&unique_path).await;

        let filesystem_config = Default::default();
        let filesystem_factory = Arc::new(FilesystemFactory::create(filesystem_config).await?);
        let store = MetricsPersistenceLayer::new(filesystem_factory, config).await?;
        Ok(Arc::new(MetricsUpdateService::new(Arc::new(store))))
    }

    #[tokio::test]
    async fn test_flush_metrics_update() {
        debug!("🧪 TEST: Flush metrics update functionality");

        let updater = create_test_updater().await.unwrap();

        let flush_update = FlushMetricsUpdate {
            vectors_flushed: 5000,
            bytes_written: 10 * 1024 * 1024,
            duration_ms: 2500,
            files_created: 3,
            engine_type: "NOVA".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater
            .record_flush("test_collection_flush", flush_update)
            .await;
        assert!(
            result.is_ok(),
            "Failed to record flush metrics: {:?}",
            result
        );

        sleep(Duration::from_millis(100)).await;

        let store = updater.store();
        let metrics = store
            .collection_metrics("test_collection_flush")
            .await
            .unwrap();
        assert!(metrics.is_some(), "Flush metrics should be stored");

        let collection_metrics = metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, "test_collection_flush");
        assert!(collection_metrics.total_flushes > 0);

        info!("✅ Flush metrics update test passed");
    }

    #[tokio::test]
    async fn test_compaction_metrics_update() {
        debug!("🧪 TEST: Compaction metrics update functionality");

        let updater = create_test_updater().await.unwrap();

        let compaction_update = CompactionMetricsUpdate {
            files_before: 15,
            files_after: 5,
            bytes_before: 50 * 1024 * 1024,
            bytes_after: 30 * 1024 * 1024,
            duration_ms: 5000,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater
            .record_compaction("test_collection_compaction_info", compaction_update)
            .await;
        assert!(
            result.is_ok(),
            "Failed to record compaction metrics: {:?}",
            result
        );

        sleep(Duration::from_millis(100)).await;

        let store = updater.store();
        let metrics = store
            .collection_metrics("test_collection_compaction_info")
            .await
            .unwrap();
        assert!(metrics.is_some(), "Compaction metrics should be stored");

        let collection_metrics = metrics.unwrap();
        assert_eq!(
            collection_metrics.collection_id,
            "test_collection_compaction_info"
        );
        assert!(collection_metrics.total_compactions > 0);
        assert!(collection_metrics.last_compaction_duration_ms > 0);

        info!("✅ Compaction metrics update test passed");
    }

    #[tokio::test]
    async fn test_search_metrics_update() {
        debug!("🧪 TEST: Search metrics update functionality");

        let updater = create_test_updater().await.unwrap();

        let search_update = SearchMetricsUpdate {
            query_latency_us: 1500.0,
            results_count: 10,
            vectors_scanned: 50000,
            cache_hit: true,
            index_used: "hnsw_main".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater
            .record_search("test_collection_search", search_update)
            .await;
        assert!(
            result.is_ok(),
            "Failed to record search metrics: {:?}",
            result
        );

        sleep(Duration::from_millis(100)).await;

        let store = updater.store();
        let metrics = store
            .collection_metrics("test_collection_search")
            .await
            .unwrap();
        assert!(metrics.is_some(), "Search metrics should be stored");

        let collection_metrics = metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, "test_collection_search");
        assert!(collection_metrics.total_searches > 0);
        assert!(collection_metrics.avg_search_latency_us > 0.0);

        info!("✅ Search metrics update test passed");
    }

    #[tokio::test]
    async fn test_operation_metrics_update() {
        debug!("🧪 TEST: Operation metrics update functionality");

        let updater = create_test_updater().await.unwrap();

        let operation_update = OperationMetricsUpdate {
            operation_type: "insert".to_string(),
            latency_us: 250.0,
            success: true,
            bytes_processed: 1024,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater
            .record_operation("test_collection_operation", operation_update)
            .await;
        assert!(
            result.is_ok(),
            "Failed to record operation metrics: {:?}",
            result
        );

        sleep(Duration::from_millis(100)).await;

        let store = updater.store();
        let metrics = store
            .collection_metrics("test_collection_operation")
            .await
            .unwrap();
        assert!(metrics.is_some(), "Operation metrics should be stored");

        let collection_metrics = metrics.unwrap();
        assert_eq!(
            collection_metrics.collection_id,
            "test_collection_operation"
        );
        assert!(collection_metrics.total_inserts > 0);

        info!("✅ Operation metrics update test passed");
    }

    #[tokio::test]
    async fn test_concurrent_metrics_updates() {
        debug!("🧪 TEST: Concurrent metrics updates");

        let updater = create_test_updater().await.unwrap();
        let updater = Arc::new(updater);

        let mut handles = vec![];

        for i in 0..20 {
            let updater_clone = updater.clone();
            let handle = tokio::spawn(async move {
                let collection_id = format!("concurrent_metrics_{:03}", i % 5);

                let flush_update = FlushMetricsUpdate {
                    vectors_flushed: 1000 + i,
                    bytes_written: (i + 1) * 1024 * 1024,
                    duration_ms: 1000 + (i * 100),
                    files_created: 1,
                    engine_type: if i % 2 == 0 { "NOVA" } else { "SST" }.to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                updater_clone
                    .record_flush(&collection_id, flush_update)
                    .await
                    .unwrap();

                let search_update = SearchMetricsUpdate {
                    query_latency_us: 1000.0 + (i as f64 * 100.0),
                    results_count: 10,
                    vectors_scanned: 10000 + (i * 1000),
                    cache_hit: i % 3 == 0,
                    index_used: "hnsw_test".to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                updater_clone
                    .record_search(&collection_id, search_update)
                    .await
                    .unwrap();

                collection_id
            });

            handles.push(handle);
        }

        let mut completed_collections = Vec::new();
        for handle in handles {
            let collection_id = handle.await.unwrap();
            completed_collections.push(collection_id);
        }

        sleep(Duration::from_millis(500)).await;

        debug!(
            "📊 Completed concurrent updates for {} operations",
            completed_collections.len()
        );

        let store = updater.store();
        let unique_collections: std::collections::HashSet<_> =
            completed_collections.into_iter().collect();

        for collection_id in unique_collections {
            let metrics = store.collection_metrics(&collection_id).await.unwrap();
            assert!(
                metrics.is_some(),
                "Metrics not found for collection {}",
                collection_id
            );

            let collection_metrics = metrics.unwrap();
            assert!(
                collection_metrics.total_flushes > 0,
                "No flush metrics for {}",
                collection_id
            );
            assert!(
                collection_metrics.total_searches > 0,
                "No search metrics for {}",
                collection_id
            );

            debug!(
                "📋 Collection '{}': {} flushes, {} searches",
                collection_id, collection_metrics.total_flushes, collection_metrics.total_searches
            );
        }

        info!("✅ Concurrent metrics updates test passed");
    }

    #[tokio::test]
    async fn test_metrics_aggregation_and_calculation() {
        debug!("🧪 TEST: Metrics aggregation and calculation");

        let updater = create_test_updater().await.unwrap();

        let collection_id = "aggregation_test_collection";

        let search_latencies = vec![800.0, 1200.0, 1500.0, 2000.0, 3000.0, 1000.0, 1800.0];

        for latency in &search_latencies {
            let search_update = SearchMetricsUpdate {
                query_latency_us: *latency,
                results_count: 10,
                vectors_scanned: 25000,
                cache_hit: true,
                index_used: "hnsw_agg_test".to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };

            updater
                .record_search(collection_id, search_update)
                .await
                .unwrap();
        }

        sleep(Duration::from_millis(200)).await;

        let store = updater.store();
        let metrics = store.collection_metrics(collection_id).await.unwrap();
        assert!(metrics.is_some(), "Aggregated metrics should exist");

        let collection_metrics = metrics.unwrap();
        assert_eq!(
            collection_metrics.total_searches,
            search_latencies.len() as i64
        );

        assert!(collection_metrics.avg_search_latency_us > 0.0);

        let expected_avg = search_latencies.iter().sum::<f64>() / search_latencies.len() as f64;
        let actual_avg = collection_metrics.avg_search_latency_us;
        let diff = (expected_avg - actual_avg).abs();

        debug!(
            "📊 Expected avg: {:.1}us, Actual avg: {:.1}us, Diff: {:.1}us",
            expected_avg, actual_avg, diff
        );

        assert!(diff < 100.0, "Average latency calculation incorrect");

        info!("✅ Metrics aggregation test passed");
    }

    #[tokio::test]
    async fn test_error_handling_in_metrics_updates() {
        debug!("🧪 TEST: Error handling in metrics updates");

        let updater = create_test_updater().await.unwrap();

        let flush_update = FlushMetricsUpdate {
            vectors_flushed: 1000,
            bytes_written: 1024 * 1024,
            duration_ms: 1000,
            files_created: 1,
            engine_type: "NOVA".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater.record_flush("", flush_update).await;
        assert!(
            result.is_ok(),
            "Empty collection ID should be handled: {:?}",
            result
        );

        let large_flush_update = FlushMetricsUpdate {
            vectors_flushed: i64::MAX,
            bytes_written: i64::MAX,
            duration_ms: i64::MAX,
            files_created: i32::MAX,
            engine_type: "STRESS_TEST".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let result = updater
            .record_flush("stress_test_collection", large_flush_update)
            .await;
        assert!(
            result.is_ok(),
            "Large values should be handled: {:?}",
            result
        );

        let invalid_search_update = SearchMetricsUpdate {
            query_latency_us: 1000.0,
            results_count: 10,
            vectors_scanned: 25000,
            cache_hit: false,
            index_used: "error_test_index".to_string(),
            timestamp: -1,
        };

        let result = updater
            .record_search("error_test_collection", invalid_search_update)
            .await;
        assert!(
            result.is_ok(),
            "Invalid timestamp should be handled gracefully: {:?}",
            result
        );

        info!("✅ Error handling test passed");
    }

    #[tokio::test]
    async fn test_metrics_updater_store_integration() {
        debug!("🧪 TEST: MetricsUpdater and PersistentStore integration");

        let updater = create_test_updater().await.unwrap();

        let collection_id = "integration_test_collection";

        let flush_update = FlushMetricsUpdate {
            vectors_flushed: 2500,
            bytes_written: 5 * 1024 * 1024,
            duration_ms: 1200,
            files_created: 2,
            engine_type: "NOVA".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        updater
            .record_flush(collection_id, flush_update)
            .await
            .unwrap();

        let compaction_update = CompactionMetricsUpdate {
            files_before: 8,
            files_after: 3,
            bytes_before: 20 * 1024 * 1024,
            bytes_after: 12 * 1024 * 1024,
            duration_ms: 3000,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        updater
            .record_compaction(collection_id, compaction_update)
            .await
            .unwrap();

        let search_update = SearchMetricsUpdate {
            query_latency_us: 1800.0,
            results_count: 15,
            vectors_scanned: 30000,
            cache_hit: true,
            index_used: "hnsw_integration".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        updater
            .record_search(collection_id, search_update)
            .await
            .unwrap();

        sleep(Duration::from_millis(300)).await;

        let store = updater.store();
        let metrics = store.collection_metrics(collection_id).await.unwrap();
        assert!(metrics.is_some(), "Integrated metrics should exist");

        let collection_metrics = metrics.unwrap();
        assert_eq!(collection_metrics.collection_id, collection_id);

        assert!(collection_metrics.total_flushes > 0);
        assert!(collection_metrics.last_flush_duration_ms > 0);

        assert!(collection_metrics.total_compactions > 0);
        assert!(collection_metrics.last_compaction_duration_ms > 0);

        assert!(collection_metrics.total_searches > 0);
        assert!(collection_metrics.avg_search_latency_us > 0.0);

        debug!(
            "📊 Integrated metrics: {} flushes, {} compactions, {} searches",
            collection_metrics.total_flushes,
            collection_metrics.total_compactions,
            collection_metrics.total_searches
        );

        info!("✅ MetricsUpdater integration test passed");
    }

    #[tokio::test]
    async fn test_metrics_timestamp_handling() {
        debug!("🧪 TEST: Metrics timestamp handling");

        let updater = create_test_updater().await.unwrap();

        let collection_id = "timestamp_test_collection";
        let current_time = chrono::Utc::now().timestamp_millis();

        let flush_update = FlushMetricsUpdate {
            vectors_flushed: 1500,
            bytes_written: 3 * 1024 * 1024,
            duration_ms: 800,
            files_created: 1,
            engine_type: "SST".to_string(),
            timestamp: current_time,
        };

        updater
            .record_flush(collection_id, flush_update)
            .await
            .unwrap();

        sleep(Duration::from_millis(100)).await;

        let store = updater.store();
        let metrics = store.collection_metrics(collection_id).await.unwrap();
        assert!(metrics.is_some());

        let collection_metrics = metrics.unwrap();
        assert_eq!(collection_metrics.last_flush_timestamp, current_time);
        assert!(collection_metrics.updated_at >= current_time);

        debug!(
            "📅 Flush timestamp: {}, Updated at: {}",
            collection_metrics.last_flush_timestamp, collection_metrics.updated_at
        );

        info!("✅ Timestamp handling test passed");
    }
}
