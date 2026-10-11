//! # ProximaDB Observability Modality
//!
//! This crate contains observability operations for logs, metrics, and traces
//! in ProximaDB.
//!
//! ## Architecture
//!
//! The observability modality is organized into several key modules:
//!
//! - canonical record mapping for logs, metric samples, and trace spans
//! - **`query`** - Observability query expressions (log queries, metric queries)
//! - future **`logs`**, **`metrics`**, and **`traces`** runtime modules
//!
//! ## Foundation
//!
//! This crate serves as the foundation for observability operations across ProximaDB,
//! providing reusable contracts and implementations for:
//!
//! - Storage engines that need observability data retention
//! - Query executors that need observability operations
//! - SIEM adapters and external observability systems
//!
//! ## Dependencies
//!
//! - `proximadb-kernel` - Core error types and foundational contracts
//! - `proximadb-proto` - Protocol buffer types
//! - `proximadb-query-filter` - Filter expression contracts
//! - `arrow` - Columnar data structures for observability operations

use proximadb_data_model::ProximaValue;
use proximadb_proto::proto::proximadb_v1::{
    LogEntry, MetricSample, SpanStatusCode, SqlValue, TraceData, sql_value,
};
use proximadb_query_filter::{FilterOperator, FilterValue};
use proximadb_records::{ProximaRecord, ProximaTree, ProximaTreeNode};

/// Stable labels used by observability records.
pub const OBSERVABILITY_LABEL: &str = "observability";
pub const LOG_LABEL: &str = "observability:log";
pub const METRIC_LABEL: &str = "observability:metric";
pub const SPAN_LABEL: &str = "observability:span";

/// Canonical key for an observability record.
///
/// The namespace is part of the durable identity so projections can be rebuilt
/// without relying on handler-local routing state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservabilityRecordKey {
    namespace: String,
    kind: &'static str,
    local_id: String,
}

impl ObservabilityRecordKey {
    pub fn log(namespace: impl Into<String>, timestamp_ns: i64, ordinal: usize) -> Self {
        Self {
            namespace: namespace.into(),
            kind: "log",
            local_id: format!("{timestamp_ns}:{ordinal}"),
        }
    }

    pub fn metric(
        namespace: impl Into<String>,
        name: impl Into<String>,
        timestamp_ns: i64,
        labels: &std::collections::HashMap<String, String>,
    ) -> Self {
        let mut label_pairs = labels.iter().collect::<Vec<_>>();
        label_pairs.sort_by_key(|(k, _)| *k);
        let label_fingerprint = label_pairs
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",");

        Self {
            namespace: namespace.into(),
            kind: "metric",
            local_id: format!("{}:{timestamp_ns}:{label_fingerprint}", name.into()),
        }
    }

    pub fn span(namespace: impl Into<String>, trace_id: &str, span_id: &str) -> Self {
        Self {
            namespace: namespace.into(),
            kind: "span",
            local_id: format!("{trace_id}:{span_id}"),
        }
    }

    pub fn canonical_oid(&self) -> String {
        format!(
            "obs://{}/{}/{}",
            escape_key_part(&self.namespace),
            self.kind,
            escape_key_part(&self.local_id)
        )
    }
}

/// Convert a log entry into the canonical ProximaRecord envelope.
pub fn log_entry_to_proxima_record(
    namespace: &str,
    log: &LogEntry,
    ordinal: usize,
) -> ProximaRecord {
    let key = ObservabilityRecordKey::log(namespace, log.timestamp_ns, ordinal);
    let mut props = ProximaTree::new();
    props.insert("kind".to_string(), string_value("log"));
    props.insert("namespace".to_string(), string_value(namespace));
    props.insert("timestamp_ns".to_string(), int_value(log.timestamp_ns));
    props.insert("severity".to_string(), int_value(log.severity as i64));
    props.insert("message".to_string(), string_value(log.message.clone()));
    if let Some(source) = &log.source {
        props.insert("source".to_string(), string_value(source.clone()));
    }
    if let Some(service) = &log.service {
        props.insert("service".to_string(), string_value(service.clone()));
    }
    props.insert(
        "fields".to_string(),
        ProximaTreeNode::Object(sql_map_to_tree(&log.fields)),
    );

    let mut record = base_observability_record(namespace, key.canonical_oid(), log.timestamp_ns);
    record.local_id = Some(key.local_id);
    record.props = props;
    record.labels.insert(LOG_LABEL);
    record
}

/// Convert a metric sample into the canonical ProximaRecord envelope.
pub fn metric_sample_to_proxima_record(namespace: &str, metric: &MetricSample) -> ProximaRecord {
    let key = ObservabilityRecordKey::metric(
        namespace,
        metric.name.clone(),
        metric.timestamp_ns,
        &metric.labels,
    );
    let mut props = ProximaTree::new();
    props.insert("kind".to_string(), string_value("metric"));
    props.insert("namespace".to_string(), string_value(namespace));
    props.insert("name".to_string(), string_value(metric.name.clone()));
    props.insert("timestamp_ns".to_string(), int_value(metric.timestamp_ns));
    props.insert("value".to_string(), float_value(metric.value));
    props.insert(
        "labels".to_string(),
        ProximaTreeNode::Object(string_map_to_tree(&metric.labels)),
    );

    let mut record = base_observability_record(namespace, key.canonical_oid(), metric.timestamp_ns);
    record.local_id = Some(key.local_id);
    record.props = props;
    record.labels.insert(METRIC_LABEL);
    record
}

/// Convert a trace span into the canonical ProximaRecord envelope.
pub fn trace_data_to_proxima_record(namespace: &str, span: &TraceData) -> ProximaRecord {
    let key = ObservabilityRecordKey::span(namespace, &span.trace_id, &span.span_id);
    let mut props = ProximaTree::new();
    props.insert("kind".to_string(), string_value("span"));
    props.insert("namespace".to_string(), string_value(namespace));
    props.insert("trace_id".to_string(), string_value(span.trace_id.clone()));
    props.insert("span_id".to_string(), string_value(span.span_id.clone()));
    if let Some(parent_span_id) = &span.parent_span_id {
        props.insert(
            "parent_span_id".to_string(),
            string_value(parent_span_id.clone()),
        );
    }
    props.insert("name".to_string(), string_value(span.name.clone()));
    props.insert("span_kind".to_string(), int_value(span.kind as i64));
    props.insert("start_time_ns".to_string(), int_value(span.start_time_ns));
    props.insert("end_time_ns".to_string(), int_value(span.end_time_ns));
    props.insert(
        "duration_ns".to_string(),
        int_value(span.end_time_ns.saturating_sub(span.start_time_ns)),
    );
    if let Some(status) = &span.status {
        let mut status_tree = ProximaTree::new();
        status_tree.insert("code".to_string(), int_value(status.code as i64));
        status_tree.insert(
            "is_error".to_string(),
            bool_value(status.code == SpanStatusCode::Error as i32),
        );
        if let Some(message) = &status.message {
            status_tree.insert("message".to_string(), string_value(message.clone()));
        }
        props.insert("status".to_string(), ProximaTreeNode::Object(status_tree));
    }
    props.insert(
        "attributes".to_string(),
        ProximaTreeNode::Object(sql_map_to_tree(&span.attributes)),
    );

    let mut record = base_observability_record(namespace, key.canonical_oid(), span.start_time_ns);
    record.local_id = Some(key.local_id);
    record.valid_to_ns = Some(span.end_time_ns);
    record.props = props;
    record.labels.insert(SPAN_LABEL);
    record
}

fn base_observability_record(namespace: &str, oid: String, timestamp_ns: i64) -> ProximaRecord {
    let mut record = ProximaRecord {
        oid,
        tenant_id: namespace.to_string(),
        created_at_ns: timestamp_ns,
        updated_at_ns: timestamp_ns,
        valid_from_ns: Some(timestamp_ns),
        origin: Some("observability".to_string()),
        method: Some("observability-ingest".to_string()),
        ..Default::default()
    };
    record.labels.insert(OBSERVABILITY_LABEL);
    record
}

fn sql_map_to_tree(fields: &std::collections::HashMap<String, SqlValue>) -> ProximaTree {
    fields
        .iter()
        .map(|(key, field)| {
            (
                key.clone(),
                ProximaTreeNode::Value(sql_value_to_proxima(field)),
            )
        })
        .collect()
}

fn string_map_to_tree(fields: &std::collections::HashMap<String, String>) -> ProximaTree {
    fields
        .iter()
        .map(|(key, field)| (key.clone(), string_value(field.clone())))
        .collect()
}

fn sql_value_to_proxima(sql: &SqlValue) -> ProximaValue {
    match sql.value.as_ref() {
        Some(sql_value::Value::StringValue(value)) => ProximaValue::String(value.clone()),
        Some(sql_value::Value::NumberValue(value)) => ProximaValue::Float64(*value),
        Some(sql_value::Value::BoolValue(value)) => ProximaValue::Boolean(*value),
        Some(sql_value::Value::Int64Value(value)) => ProximaValue::Int64(*value),
        Some(sql_value::Value::BytesValue(value)) => ProximaValue::Binary(value.clone()),
        // Canonical MessagePack JSONB decode (the shared helper) so a tag-9
        // value compares equal across read paths.
        Some(sql_value::Value::JsonbValue(value)) => ProximaValue::from_jsonb_or_binary(value),
        Some(sql_value::Value::NullValue(_)) | None => ProximaValue::Null,
        Some(sql_value::Value::ArrayValue(array)) => {
            ProximaValue::Array(array.values.iter().map(sql_value_to_proxima).collect())
        }
        Some(sql_value::Value::ObjectValue(object)) => ProximaValue::Map(
            object
                .fields
                .iter()
                .map(|(key, value)| (key.clone(), sql_value_to_proxima(value)))
                .collect(),
        ),
    }
}

fn string_value(value: impl Into<String>) -> ProximaTreeNode {
    ProximaTreeNode::Value(ProximaValue::String(value.into()))
}

fn int_value(value: i64) -> ProximaTreeNode {
    ProximaTreeNode::Value(ProximaValue::Int64(value))
}

fn float_value(value: f64) -> ProximaTreeNode {
    ProximaTreeNode::Value(ProximaValue::Float64(value))
}

fn bool_value(value: bool) -> ProximaTreeNode {
    ProximaTreeNode::Value(ProximaValue::Boolean(value))
}

fn escape_key_part(value: &str) -> String {
    value.replace('%', "%25").replace('/', "%2F")
}

/// Log query expression
#[derive(Debug, Clone)]
pub struct LogQueryExpr {
    /// Collection to query
    pub collection: String,
    /// Filters to apply
    pub filters: Vec<LogFilter>,
    /// Time range
    pub time_range: Option<TimeRange>,
    /// Fields to return
    pub projection: Vec<String>,
    /// Maximum results
    pub limit: Option<u32>,
}

/// Metric query expression
#[derive(Debug, Clone)]
pub struct MetricQueryExpr {
    /// Collection to query
    pub collection: String,
    /// Metric name
    pub metric_name: String,
    /// Aggregation
    pub aggregation: MetricAggregation,
    /// Filters to apply
    pub filters: Vec<MetricFilter>,
    /// Time range
    pub time_range: Option<TimeRange>,
    /// Group by fields
    pub group_by: Vec<String>,
}

/// Metric aggregation
#[derive(Debug, Clone)]
pub enum MetricAggregation {
    /// Average
    Avg,
    /// Sum
    Sum,
    /// Count
    Count,
    /// Min
    Min,
    /// Max
    Max,
    /// Percentile
    Percentile(f64),
}

/// Log filter
#[derive(Debug, Clone)]
pub struct LogFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

/// Metric filter
#[derive(Debug, Clone)]
pub struct MetricFilter {
    pub label: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

/// Time range
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: i64,
    pub end: i64,
}

// TODO: Move these from src/observability
// pub mod logs;
// pub mod metrics;
// pub mod traces;
// pub mod event_log;

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_proto::proto::proximadb_v1::{
        Severity, SpanKind, SpanStatus, SpanStatusCode, sql_value,
    };

    #[test]
    fn test_observability_module_imports() {
        // Basic test to verify the module structure is working
        let _log_query = LogQueryExpr {
            collection: "logs".to_string(),
            filters: vec![],
            time_range: None,
            projection: vec![],
            limit: None,
        };
        // More comprehensive tests will be added as modules are extracted
    }

    #[test]
    fn log_maps_to_canonical_record() {
        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "request_id".to_string(),
            SqlValue {
                value: Some(sql_value::Value::StringValue("req-1".to_string())),
            },
        );

        let record = log_entry_to_proxima_record(
            "tenant-a",
            &LogEntry {
                timestamp_ns: 42,
                severity: Severity::Error as i32,
                message: "failed".to_string(),
                fields,
                source: Some("gateway".to_string()),
                service: Some("checkout".to_string()),
            },
            0,
        );

        assert_eq!(record.tenant_id, "tenant-a");
        assert_eq!(record.oid, "obs://tenant-a/log/42:0");
        assert!(record.labels.contains(OBSERVABILITY_LABEL));
        assert!(record.labels.contains(LOG_LABEL));
        assert!(matches!(
            record.props.get("message"),
            Some(ProximaTreeNode::Value(ProximaValue::String(value))) if value == "failed"
        ));
    }

    #[test]
    fn metric_key_is_stable_across_label_order() {
        let mut left = std::collections::HashMap::new();
        left.insert("service".to_string(), "checkout".to_string());
        left.insert("region".to_string(), "us".to_string());

        let mut right = std::collections::HashMap::new();
        right.insert("region".to_string(), "us".to_string());
        right.insert("service".to_string(), "checkout".to_string());

        let left_record = metric_sample_to_proxima_record(
            "tenant-a",
            &MetricSample {
                name: "latency_ms".to_string(),
                timestamp_ns: 100,
                value: 12.5,
                labels: left,
            },
        );
        let right_record = metric_sample_to_proxima_record(
            "tenant-a",
            &MetricSample {
                name: "latency_ms".to_string(),
                timestamp_ns: 100,
                value: 12.5,
                labels: right,
            },
        );

        assert_eq!(left_record.oid, right_record.oid);
        assert!(left_record.labels.contains(METRIC_LABEL));
    }

    #[test]
    fn trace_span_maps_status_and_valid_time() {
        let record = trace_data_to_proxima_record(
            "tenant-a",
            &TraceData {
                trace_id: "trace-1".to_string(),
                span_id: "span-1".to_string(),
                parent_span_id: None,
                name: "GET /orders".to_string(),
                kind: SpanKind::Server as i32,
                start_time_ns: 1_000,
                end_time_ns: 1_250,
                status: Some(SpanStatus {
                    code: SpanStatusCode::Error as i32,
                    message: Some("timeout".to_string()),
                }),
                attributes: std::collections::HashMap::new(),
                events: Vec::new(),
                links: Vec::new(),
            },
        );

        assert_eq!(record.oid, "obs://tenant-a/span/trace-1:span-1");
        assert_eq!(record.valid_from_ns, Some(1_000));
        assert_eq!(record.valid_to_ns, Some(1_250));
        assert!(record.labels.contains(SPAN_LABEL));
    }
}
