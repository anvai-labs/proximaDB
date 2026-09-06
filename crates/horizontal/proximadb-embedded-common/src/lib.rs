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
pub fn proxima_value_to_json(v: proximadb_data_model::ProximaValue) -> serde_json::Value {
    use proximadb_data_model::ProximaValue;
    match v {
        ProximaValue::String(s) | ProximaValue::Symbol(s) => serde_json::Value::String(s),
        ProximaValue::Float32(f) => serde_json::Number::from_f64(f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        ProximaValue::Float64(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        ProximaValue::Int64(i) => serde_json::Value::Number(serde_json::Number::from(i)),
        ProximaValue::Int32(i) => serde_json::Value::Number(serde_json::Number::from(i)),
        ProximaValue::Boolean(b) => serde_json::Value::Bool(b),
        ProximaValue::Json(v) | ProximaValue::Jsonb(v) => v,
        ProximaValue::Map(m) | ProximaValue::Struct(m) => serde_json::Value::Object(
            m.into_iter()
                .map(|(k, v)| (k, proxima_value_to_json(v)))
                .collect(),
        ),
        ProximaValue::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(proxima_value_to_json).collect())
        }
        ProximaValue::Null => serde_json::Value::Null,
        // Exotics (Binary/Uuid/ULID/temporals/SparseVector) render through
        // records' canonical API-facing spelling (base64 binary, dashed
        // uuid) — the Rust-Debug fallback produced text no consumer could
        // match or parse.
        other => proximadb_records::conversions::proxima_value_to_json_canonical(&other),
    }
}
