//! ProximaDB metrics collection — extracted from the root `src/metrics`
//! (decomposition metrics Slice 1).
//!
//! Holds the foundation-pure metrics surface: per-subsystem metric types
//! (`schema`, `*_metrics`), the aggregation engine (`aggregator`), collectors
//! (access/document/filesystem/query/storage/system), exporters (json/prometheus),
//! and the query service. The storage-coupled collectors (`engine`, `graph`),
//! the persistence layer (`store`), the updater, and the consumption/metering/
//! cache/pinning/tier-migration metrics stay in the root behind a future
//! `MetricsStoragePort` — they reach these types via the root re-export shim.

pub mod advisor_observations_metrics;
pub mod aggregator;
pub mod collectors;
pub mod compression;
pub mod dml_lock_metrics;
pub mod dr_metrics;
pub mod exporters;
pub mod fusion;
pub mod primary_pod_metrics;
pub mod recall_drift_metrics;
pub mod route_metrics;
pub mod schema;
pub mod td064_metrics;
pub mod td066_metrics;
pub mod turboquant_metrics;
pub mod wal_flush_metrics;
pub mod wal_scan_metrics;

// Re-exports so cross-module references that resolved via the root
// `crate::metrics::{SystemMetrics, Alert, MetricsConfig}` continue to work.
pub use exporters::SystemMetrics;
pub use proximadb_config::MetricsConfig;
pub use schema::Alert;

use prometheus::{
    CounterVec, GaugeVec, HistogramOpts, HistogramVec, IntGauge, IntGaugeVec, Opts,
    register_counter_vec, register_gauge_vec, register_histogram_vec, register_int_gauge,
    register_int_gauge_vec,
};
use tracing::{error, warn};

// ============================================================================
// Safe metric registration helpers (redundancy pass 3)
//
// Previously stamped as private fns across 13 files (9 in this crate, 3 in
// root src/metrics, 1 in the HELIX engine) with THREE divergent semantics on
// registration clash: the canonical two-stage fallback below, a no-rename
// variant (dr_metrics), and disable-on-clash (HELIX). The canonical fallback
// semantics are hoisted here; HELIX's disable-on-clash behavior survives as
// the `_or_none` variants.
// ============================================================================

/// Register a `CounterVec`; on clash fall back to an unregistered local
/// metric, then to a `{name}_fallback` rename, so the caller always gets a
/// usable handle.
pub fn register_counter_vec_safe(name: &str, help: &str, labels: &[&str]) -> CounterVec {
    match register_counter_vec!(name, help, labels) {
        Ok(metric) => metric,
        Err(reg_err) => {
            error!("Failed to register {}: {}", name, reg_err);
            CounterVec::new(Opts::new(name, help), labels).unwrap_or_else(|_| {
                CounterVec::new(Opts::new(format!("{}_fallback", name), help), labels)
                    .unwrap_or_else(|_| unreachable!("valid counter metric descriptor"))
            })
        }
    }
}

/// [`register_counter_vec_safe`] for `GaugeVec`.
pub fn register_gauge_vec_safe(name: &str, help: &str, labels: &[&str]) -> GaugeVec {
    match register_gauge_vec!(name, help, labels) {
        Ok(metric) => metric,
        Err(reg_err) => {
            error!("Failed to register {}: {}", name, reg_err);
            GaugeVec::new(Opts::new(name, help), labels).unwrap_or_else(|_| {
                GaugeVec::new(Opts::new(format!("{}_fallback", name), help), labels)
                    .unwrap_or_else(|_| unreachable!("valid gauge metric descriptor"))
            })
        }
    }
}

/// [`register_counter_vec_safe`] for `IntGaugeVec`.
pub fn register_int_gauge_vec_safe(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    match register_int_gauge_vec!(name, help, labels) {
        Ok(metric) => metric,
        Err(reg_err) => {
            error!("Failed to register {}: {}", name, reg_err);
            IntGaugeVec::new(Opts::new(name, help), labels).unwrap_or_else(|_| {
                IntGaugeVec::new(Opts::new(format!("{}_fallback", name), help), labels)
                    .unwrap_or_else(|_| unreachable!("valid gauge metric descriptor"))
            })
        }
    }
}

/// [`register_counter_vec_safe`] for a singular `IntGauge`.
pub fn register_int_gauge_safe(name: &str, help: &str) -> IntGauge {
    match register_int_gauge!(name, help) {
        Ok(metric) => metric,
        Err(reg_err) => {
            error!("Failed to register {}: {}", name, reg_err);
            IntGauge::new(name, help).unwrap_or_else(|_| {
                IntGauge::new(format!("{}_fallback", name), help)
                    .unwrap_or_else(|_| unreachable!("valid gauge metric descriptor"))
            })
        }
    }
}

/// Register a `HistogramVec`; on clash fall back to an unregistered local
/// metric with the same buckets, then to a `{name}_fallback` rename.
pub fn register_histogram_vec_safe(
    name: &str,
    help: &str,
    labels: &[&str],
    buckets: Vec<f64>,
) -> HistogramVec {
    match register_histogram_vec!(name, help, labels, buckets.clone()) {
        Ok(metric) => metric,
        Err(reg_err) => {
            error!("Failed to register {}: {}", name, reg_err);
            HistogramVec::new(
                HistogramOpts::new(name, help).buckets(buckets.clone()),
                labels,
            )
            .unwrap_or_else(|_| {
                HistogramVec::new(
                    HistogramOpts::new(format!("{}_fallback", name), help).buckets(buckets),
                    labels,
                )
                .unwrap_or_else(|_| unreachable!("valid histogram metric descriptor"))
            })
        }
    }
}

/// Disable-on-clash variant (HELIX semantics): `None` when the name is
/// already registered, so the metric becomes inert rather than renamed.
pub fn register_counter_vec_or_none(name: &str, help: &str, labels: &[&str]) -> Option<CounterVec> {
    match register_counter_vec!(name, help, labels) {
        Ok(metric) => Some(metric),
        Err(error) => {
            warn!(metric = name, error = %error, "counter metric disabled (name clash)");
            None
        }
    }
}

/// [`register_counter_vec_or_none`] for `GaugeVec`.
pub fn register_gauge_vec_or_none(name: &str, help: &str, labels: &[&str]) -> Option<GaugeVec> {
    match register_gauge_vec!(name, help, labels) {
        Ok(metric) => Some(metric),
        Err(error) => {
            warn!(metric = name, error = %error, "gauge metric disabled (name clash)");
            None
        }
    }
}

/// [`register_counter_vec_or_none`] for `HistogramVec`.
pub fn register_histogram_vec_or_none(
    name: &str,
    help: &str,
    labels: &[&str],
    buckets: Vec<f64>,
) -> Option<HistogramVec> {
    match register_histogram_vec!(name, help, labels, buckets) {
        Ok(metric) => Some(metric),
        Err(error) => {
            warn!(metric = name, error = %error, "histogram metric disabled (name clash)");
            None
        }
    }
}

/// Adapter implementing [`proximadb_storage_ports::OrionMetricsPort`] by delegating
/// to the existing Prometheus-counter free functions. Registered at bootstrap.
pub struct OrionMetricsAdapter;

impl proximadb_storage_ports::OrionMetricsPort for OrionMetricsAdapter {
    fn record_recovery_checkpoint_observation(
        &self,
        graph_id: &str,
        checkpoint_lsn: Option<u64>,
        checkpoint_timestamp_ms: Option<u64>,
    ) {
        td066_metrics::record_recovery_checkpoint_observation(
            graph_id,
            checkpoint_lsn,
            checkpoint_timestamp_ms,
        );
    }
}
