//! Reusable embedded-mode support types.
//!
//! This crate intentionally contains only root-independent support code used by
//! embedded runtimes and bindings. It must not depend on root services, storage
//! engines, protocol servers, or modality implementations.

pub mod histograms;
pub mod metrics;
pub mod search_filter;

pub use histograms::{HistogramStats, LatencyHistogram, RollingWindow};
pub use metrics::{EmbeddedMetrics, EmbeddedMetricsCollector, LatencyStats, LatencyTimer};
pub use search_filter::parse_vector_filter;

/// Convert a ProximaValue to a serde_json::Value (shared by root conversions + embedded).
///
/// One line: records' canonical API-facing rendering (base64 binary, dashed
/// uuid, ISO-capable fallthrough). Kept as a named fn so the binding and root
/// share ONE import surface; do not re-add arm-for-arm copies here — they
/// drift from the canonical spelling the moment any arm changes upstream.
pub fn proxima_value_to_json(v: proximadb_data_model::ProximaValue) -> serde_json::Value {
    proximadb_records::conversions::proxima_value_to_json_canonical(&v)
}
