// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Metrics module — thin re-export shim.
//!
//! The foundation-pure metrics surface (collectors, aggregators, exporters,
//! schema, `*_metrics` types) lives in `proximadb-metrics`; the storage-coupled
//! collectors (`engine`, `graph`), the persistence layer (`store`), the updater,
//! and the consumption/metering/cache/pinning/tier-migration metrics stay here
//! in the root behind a future `MetricsStoragePort`. All `crate::metrics::*`
//! import paths are unchanged.

// Moved (foundation-pure) modules — re-exported from the crate.
pub use proximadb_metrics::{
    Alert, SystemMetrics, advisor_observations_metrics, aggregator, compression, dml_lock_metrics,
    dr_metrics, exporters, fusion, primary_pod_metrics, recall_drift_metrics, route_metrics,
    schema, td064_metrics, td066_metrics, turboquant_metrics, wal_flush_metrics, wal_scan_metrics,
};

// Deferred (storage-coupled / orchestration) modules stay root.
pub mod cache;
pub mod collection_pin_metrics;
pub mod collectors;
pub mod consumption_metrics;
pub mod io_trace_sink_metrics;
pub mod metering_writer;
pub mod operational_metrics;
pub mod query_service;
pub mod store;
#[cfg(test)]
mod tests;
pub mod tier_migration_metrics;
pub mod updater;
pub use proximadb_usage_event::usage_event;

// Top-level type re-exports (preserve `crate::metrics::X` paths).
pub use cache::{CacheMetricsCollector, CacheMetricsSnapshot, CacheOptimizationHints};
pub use collectors::{
    EngineComparison, EngineMetricsCollector, EngineStatistics, UnifiedMetricsCollector,
};
pub use proximadb_metrics::aggregator::{AggregationWindow, MetricsAggregationEngine};
pub use proximadb_metrics::compression::{
    CompressionMetrics, CompressionMetricsTracker, CompressionResult, DecompressionResult,
};
pub use proximadb_metrics::exporters::{
    ExportFormat, JsonExporter, MetricsExporter, PrometheusExporter,
};
pub use proximadb_metrics::schema::{CollectionMetrics, GlobalMetrics, QueryOptimizationHints};
pub use query_service::{MetricsQueryOptions, MetricsQueryService};
pub use store::MetricsPersistenceLayer;
pub use updater::{
    CompactionMetricsUpdate, FlushMetricsUpdate, InternalMetricsUpdater, MetricsUpdate,
};

pub use proximadb_config::MetricsConfig;
pub use proximadb_metrics::exporters::MetricsExportSnapshot;

/// TD-LEASE-2: outcomes of a lost version CAS in the partition-lease store's
/// claim loop. `own_adopted` = a same-pod race settled as renewal (the
/// 2026-11-01 mid-ingest failure class); `dead_winner_retry` = the winner was
/// expired and the claim retried; `foreign_fenced` = a live foreign lease
/// (or an uncommittable claim) fenced this pod.
pub fn record_lease_conflict_resolution(resolution: &str) {
    if let Some(family) = LEASE_CONFLICT_RESOLUTIONS.as_ref() {
        family.with_label_values(&[resolution]).inc();
    }
}

lazy_static::lazy_static! {
    static ref LEASE_CONFLICT_RESOLUTIONS: Option<prometheus::IntCounterVec> = {
        use prometheus::{IntCounterVec, Opts};
        match IntCounterVec::new(
            Opts::new(
                "proximadb_lease_conflict_resolutions_total",
                "Lost version CAS outcomes in the partition-lease claim loop (TD-LEASE-2)",
            ),
            &["resolution"],
        ) {
            Ok(metric) => {
                let _ = prometheus::register(Box::new(metric.clone()));
                Some(metric)
            }
            Err(_) => None,
        }
    };
}

/// TD-LEASE-2: write-gate rejections at the lease-on-write boundary
/// (`record_ops_service`), by reason — the client-visible failure the
/// conflict-resolution counter explains.
pub fn record_write_gate_rejection(reason: &str) {
    if let Some(family) = WRITE_GATE_REJECTIONS.as_ref() {
        family.with_label_values(&[reason]).inc();
    }
}

lazy_static::lazy_static! {
    static ref WRITE_GATE_REJECTIONS: Option<prometheus::IntCounterVec> = {
        use prometheus::{IntCounterVec, Opts};
        match IntCounterVec::new(
            Opts::new(
                "proximadb_write_gate_rejections_total",
                "Write batches rejected by the lease-on-write gate (TD-LEASE-2)",
            ),
            &["reason"],
        ) {
            Ok(metric) => {
                let _ = prometheus::register(Box::new(metric.clone()));
                Some(metric)
            }
            Err(_) => None,
        }
    };
}
