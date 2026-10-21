/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! EventLog metrics and monitoring

use lazy_static::lazy_static;
use prometheus::{
    CounterVec, GaugeVec, HistogramOpts, HistogramVec, Opts, exponential_buckets,
    register_counter_vec, register_gauge_vec, register_histogram_vec,
};
use std::sync::Arc;
use tracing::{debug, error};
use proximadb_metrics::{register_counter_vec_safe, register_gauge_vec_safe, register_histogram_vec_safe};

lazy_static! {
    /// Total events created by type
    pub static ref EVENTLOG_EVENTS_CREATED: CounterVec = register_counter_vec_safe(
        "proximadb_eventlog_events_created_total",
        "Total number of events created in EventLog",
        &["collection_id", "event_type", "storage_engine"]
    );

    /// Total events processed by type
    pub static ref EVENTLOG_EVENTS_PROCESSED: CounterVec = register_counter_vec_safe(
        "proximadb_eventlog_events_processed_total",
        "Total number of events processed from EventLog",
        &["collection_id", "event_type", "storage_engine", "status"]
    );

    /// Current pending events
    pub static ref EVENTLOG_PENDING_EVENTS: GaugeVec = register_gauge_vec_safe(
        "proximadb_eventlog_pending_events",
        "Current number of pending events in EventLog",
        &["collection_id", "event_type"]
    );

    /// Event processing time
    pub static ref EVENTLOG_PROCESSING_TIME: HistogramVec = register_histogram_vec_safe(
        "proximadb_eventlog_processing_time_seconds",
        "Time taken to process EventLog events",
        &["collection_id", "event_type", "operation"],
        exponential_buckets(0.001, 2.0, 15)
            .unwrap_or_else(|_| vec![0.001, 0.01, 0.1, 1.0, 10.0]),
    );

    /// Vector extraction count
    pub static ref EVENTLOG_VECTORS_EXTRACTED: CounterVec = register_counter_vec_safe(
        "proximadb_eventlog_vectors_extracted_total",
        "Total number of vectors extracted from storage files",
        &["collection_id", "storage_engine", "extraction_mode"]
    );

    /// Files pending indexing
    pub static ref EVENTLOG_FILES_PENDING: GaugeVec = register_gauge_vec_safe(
        "proximadb_eventlog_files_pending_indexing",
        "Current number of files pending AXIS indexing",
        &["collection_id", "storage_engine"]
    );

    /// EventLog WAL size
    pub static ref EVENTLOG_WAL_SIZE_BYTES: GaugeVec = register_gauge_vec_safe(
        "proximadb_eventlog_wal_size_bytes",
        "Current size of EventLog WAL in bytes",
        &["wal_type"]
    );

    /// EventLog queue depth
    pub static ref EVENTLOG_QUEUE_DEPTH: GaugeVec = register_gauge_vec_safe(
        "proximadb_eventlog_queue_depth",
        "Current depth of EventLog queue",
        &["priority"]
    );

    /// Consumer batch size
    pub static ref EVENTLOG_CONSUMER_BATCH_SIZE: HistogramVec = register_histogram_vec_safe(
        "proximadb_eventlog_consumer_batch_size",
        "Number of events processed in each consumer batch",
        &["consumer_id"],
        exponential_buckets(1.0, 2.0, 10)
            .unwrap_or_else(|_| vec![1.0, 2.0, 4.0, 8.0, 16.0]),
    );

    /// Compaction blocked files
    pub static ref EVENTLOG_COMPACTION_BLOCKED_FILES: GaugeVec = register_gauge_vec_safe(
        "proximadb_eventlog_compaction_blocked_files",
        "Number of files blocked from compaction due to pending indexing",
        &["collection_id", "storage_engine"]
    );

    /// Self-healing transitions
    pub static ref EVENTLOG_SELF_HEALING_TRANSITIONS: CounterVec = register_counter_vec_safe(
        "proximadb_eventlog_self_healing_transitions_total",
        "Total number of files transitioning from pending to ready",
        &["collection_id", "storage_engine"]
    );
}

/// EventLog metrics collector
pub struct EventLogMetricsCollector {
    /// Collection ID for scoping metrics
    collection_id: Option<String>,
}

impl EventLogMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            collection_id: None,
        }
    }

    /// Create collector for specific collection
    pub fn for_collection(collection_id: impl Into<String>) -> Self {
        Self {
            collection_id: Some(collection_id.into()),
        }
    }

    /// Record event creation
    pub fn record_event_created(
        &self,
        collection_id: &str,
        event_type: &str,
        storage_engine: &str,
    ) {
        EVENTLOG_EVENTS_CREATED
            .with_label_values(&[collection_id, event_type, storage_engine])
            .inc();

        EVENTLOG_PENDING_EVENTS
            .with_label_values(&[collection_id, event_type])
            .inc();

        debug!(
            "📊 EventLog metric: Event created - collection={}, type={}, engine={}",
            collection_id, event_type, storage_engine
        );
    }

    /// Record event processing
    pub fn record_event_processed(
        &self,
        collection_id: &str,
        event_type: &str,
        storage_engine: &str,
        success: bool,
        duration_secs: f64,
    ) {
        let status = if success { "success" } else { "failure" };

        EVENTLOG_EVENTS_PROCESSED
            .with_label_values(&[collection_id, event_type, storage_engine, status])
            .inc();

        if success {
            EVENTLOG_PENDING_EVENTS
                .with_label_values(&[collection_id, event_type])
                .dec();
        }

        EVENTLOG_PROCESSING_TIME
            .with_label_values(&[collection_id, event_type, "total"])
            .observe(duration_secs);

        debug!(
            "📊 EventLog metric: Event processed - collection={}, type={}, status={}, duration={}s",
            collection_id, event_type, status, duration_secs
        );
    }

    /// Record vector extraction
    pub fn record_vectors_extracted(
        &self,
        collection_id: &str,
        storage_engine: &str,
        extraction_mode: &str,
        count: u64,
    ) {
        EVENTLOG_VECTORS_EXTRACTED
            .with_label_values(&[collection_id, storage_engine, extraction_mode])
            .inc_by(count);

        debug!(
            "📊 EventLog metric: Vectors extracted - collection={}, engine={}, mode={}, count={}",
            collection_id, storage_engine, extraction_mode, count
        );
    }

    /// Record file pending status change
    pub fn record_file_pending_change(
        &self,
        collection_id: &str,
        storage_engine: &str,
        delta: i64,
    ) {
        if delta > 0 {
            EVENTLOG_FILES_PENDING
                .with_label_values(&[collection_id, storage_engine])
                .add(delta as f64);
        } else if delta < 0 {
            EVENTLOG_FILES_PENDING
                .with_label_values(&[collection_id, storage_engine])
                .sub((-delta) as f64);
        }

        debug!(
            "📊 EventLog metric: Files pending change - collection={}, engine={}, delta={}",
            collection_id, storage_engine, delta
        );
    }

    /// Record WAL size
    pub fn record_wal_size(&self, wal_type: &str, size_bytes: u64) {
        EVENTLOG_WAL_SIZE_BYTES
            .with_label_values(&[wal_type])
            .set(size_bytes as f64);

        debug!(
            "📊 EventLog metric: WAL size - type={}, size={}",
            wal_type, size_bytes
        );
    }

    /// Record queue depth
    pub fn record_queue_depth(&self, priority: &str, depth: u64) {
        EVENTLOG_QUEUE_DEPTH
            .with_label_values(&[priority])
            .set(depth as f64);

        debug!(
            "📊 EventLog metric: Queue depth - priority={}, depth={}",
            priority, depth
        );
    }

    /// Record consumer batch size
    pub fn record_consumer_batch(&self, consumer_id: &str, batch_size: u64) {
        EVENTLOG_CONSUMER_BATCH_SIZE
            .with_label_values(&[consumer_id])
            .observe(batch_size as f64);

        debug!(
            "📊 EventLog metric: Consumer batch - id={}, size={}",
            consumer_id, batch_size
        );
    }

    /// Record compaction blocked files
    pub fn record_compaction_blocked(
        &self,
        collection_id: &str,
        storage_engine: &str,
        blocked_count: u64,
    ) {
        EVENTLOG_COMPACTION_BLOCKED_FILES
            .with_label_values(&[collection_id, storage_engine])
            .set(blocked_count as f64);

        debug!(
            "📊 EventLog metric: Compaction blocked - collection={}, engine={}, count={}",
            collection_id, storage_engine, blocked_count
        );
    }

    /// Record self-healing transition
    pub fn record_self_healing(&self, collection_id: &str, storage_engine: &str) {
        EVENTLOG_SELF_HEALING_TRANSITIONS
            .with_label_values(&[collection_id, storage_engine])
            .inc();

        debug!(
            "📊 EventLog metric: Self-healing transition - collection={}, engine={}",
            collection_id, storage_engine
        );
    }

    /// Record operation timing
    pub fn record_operation_time(
        &self,
        collection_id: &str,
        event_type: &str,
        operation: &str,
        duration_secs: f64,
    ) {
        EVENTLOG_PROCESSING_TIME
            .with_label_values(&[collection_id, event_type, operation])
            .observe(duration_secs);

        debug!(
            "📊 EventLog metric: Operation time - collection={}, type={}, op={}, duration={}s",
            collection_id, event_type, operation, duration_secs
        );
    }
}

impl Default for EventLogMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metrics instance
pub fn global_metrics() -> EventLogMetricsCollector {
    EventLogMetricsCollector::new()
}

/// Start metrics reporting task
pub async fn start_metrics_reporter(
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut shutdown = shutdown;
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Log summary metrics
                    debug!("📊 EventLog Metrics Summary:");

                    // Get pending events count
                    // In production, we'd query actual metrics
                    debug!("  Pending events: (queried from metrics)");
                    debug!("  Processing rate: (calculated from counters)");
                    debug!("  Queue depth: (from gauge metrics)");
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        debug!("EventLog metrics reporter shutting down");
                        break;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let collector = EventLogMetricsCollector::new();

        // Record various metrics
        collector.record_event_created("test_collection", "flush", "sst");
        collector.record_event_processed("test_collection", "flush", "sst", true, 0.125);
        collector.record_vectors_extracted("test_collection", "sst", "fp32_only", 100);
        collector.record_file_pending_change("test_collection", "sst", 3);
        collector.record_wal_size("current", 1024 * 1024);
        collector.record_queue_depth("normal", 5);
        collector.record_consumer_batch("consumer_1", 10);
        collector.record_compaction_blocked("test_collection", "sst", 2);
        collector.record_self_healing("test_collection", "sst");

        // Metrics should be recorded without panicking
        // In production tests, we'd verify actual metric values
    }

    #[test]
    fn test_collection_scoped_metrics() {
        let collector = EventLogMetricsCollector::for_collection("scoped_collection");

        // Collection-scoped collector should work the same way
        collector.record_event_created("scoped_collection", "compaction_info", "viper");
        collector.record_vectors_extracted("scoped_collection", "viper", "quantized_only", 50);
    }
}
