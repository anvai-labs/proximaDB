// Trace storage for distributed tracing
//
// Provides:
// - Span storage and indexing
// - Trace assembly from spans
// - Service dependency graph
// - Latency analysis

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use proximadb_observability::trace_data_to_proxima_record;
use proximadb_records::ProximaRecord;
use tokio::sync::RwLock;

use crate::proto::proximadb_v1::{
    SpanKind, SpanStatus, SpanStatusCode, SqlValue, TraceData as ProtoTraceData, sql_value,
};

/// Trace span for distributed tracing.
///
/// Moved to `proximadb_observability_engine::model` so the
/// `ObservabilityStoragePort` contract can name it without an upward edge.
pub use crate::observability::TraceSpan;

/// Trace storage service
pub struct TraceStorage {
    /// Base path for storage
    _base_path: String,
    /// Observability namespace used as the canonical tenant id.
    namespace: String,
    /// Traces indexed by trace ID
    traces: RwLock<HashMap<String, TraceData>>,
    /// Spans indexed by time
    spans_by_time: RwLock<BTreeMap<i64, Vec<String>>>,
    /// Service index (service -> trace IDs)
    service_index: RwLock<HashMap<String, Vec<String>>>,
    /// Total span count
    span_count: AtomicU64,
    /// Total storage bytes (estimated)
    total_bytes: AtomicU64,
}

/// Data for a single trace
struct TraceData {
    /// Trace ID
    trace_id: String,
    /// All spans in this trace
    spans: Vec<TraceSpan>,
    /// Root span ID
    root_span_id: Option<String>,
    /// First timestamp
    start_time_ns: i64,
    /// Last timestamp
    end_time_ns: i64,
    /// Services involved
    services: Vec<String>,
}

impl TraceStorage {
    /// Create a new trace storage
    pub fn new(base_path: &str) -> Result<Self> {
        Self::new_for_namespace(base_path, "default")
    }

    /// Create trace storage for a specific observability namespace.
    pub fn new_for_namespace(base_path: &str, namespace: &str) -> Result<Self> {
        Ok(Self {
            _base_path: base_path.to_string(),
            namespace: namespace.to_string(),
            traces: RwLock::new(HashMap::new()),
            spans_by_time: RwLock::new(BTreeMap::new()),
            service_index: RwLock::new(HashMap::new()),
            span_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
        })
    }

    /// Project a trace span onto the canonical observability record shape.
    ///
    /// TraceStorage currently assembles spans in memory for query ergonomics;
    /// this projection is the shared bridge for WAL replay into canonical
    /// ProximaRecord/PAX storage without a second trace-specific envelope.
    pub fn span_projection_record(&self, span: &TraceSpan) -> ProximaRecord {
        let trace_data = ProtoTraceData {
            trace_id: span.trace_id.clone(),
            span_id: span.span_id.clone(),
            parent_span_id: (!span.parent_span_id.is_empty()).then(|| span.parent_span_id.clone()),
            name: span.name.clone(),
            kind: SpanKind::Unspecified as i32,
            start_time_ns: span.start_time_ns,
            end_time_ns: span.end_time_ns,
            status: Some(SpanStatus {
                code: if span.status == 0 {
                    SpanStatusCode::Ok as i32
                } else {
                    SpanStatusCode::Error as i32
                },
                message: (!span.status_message.is_empty()).then(|| span.status_message.clone()),
            }),
            attributes: span
                .attributes
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        SqlValue {
                            value: Some(sql_value::Value::StringValue(value.clone())),
                        },
                    )
                })
                .collect(),
            events: Vec::new(),
            links: Vec::new(),
        };

        trace_data_to_proxima_record(&self.namespace, &trace_data)
    }

    /// Write a trace span
    pub async fn write(&self, span: &TraceSpan) -> Result<()> {
        let trace_id = &span.trace_id;

        // Update trace data
        {
            let mut traces = self.traces.write().await;
            let trace = traces.entry(trace_id.clone()).or_insert_with(|| TraceData {
                trace_id: trace_id.clone(),
                spans: Vec::new(),
                root_span_id: None,
                start_time_ns: span.start_time_ns,
                end_time_ns: span.end_time_ns,
                services: Vec::new(),
            });

            // Update root span if this is a root span (no parent)
            if span.parent_span_id.is_empty() {
                trace.root_span_id = Some(span.span_id.clone());
            }

            // Update time bounds
            trace.start_time_ns = trace.start_time_ns.min(span.start_time_ns);
            trace.end_time_ns = trace.end_time_ns.max(span.end_time_ns);

            // Add service if not present
            if !trace.services.contains(&span.service_name) {
                trace.services.push(span.service_name.clone());
            }

            trace.spans.push(span.clone());
        }

        // Update time index
        {
            let mut spans_by_time = self.spans_by_time.write().await;
            spans_by_time
                .entry(span.start_time_ns)
                .or_default()
                .push(span.span_id.clone());
        }

        // Update service index
        {
            let mut service_index = self.service_index.write().await;
            let traces = service_index.entry(span.service_name.clone()).or_default();
            if !traces.contains(trace_id) {
                traces.push(trace_id.clone());
            }
        }

        self.span_count.fetch_add(1, Ordering::Relaxed);

        // Track storage bytes
        let span_size = self.estimate_span_size(span);
        self.total_bytes.fetch_add(span_size, Ordering::Relaxed);

        Ok(())
    }

    /// Query spans by trace ID
    pub async fn query_by_trace_id(&self, trace_id: &str) -> Result<Vec<TraceSpan>> {
        let traces = self.traces.read().await;

        if let Some(trace) = traces.get(trace_id) {
            Ok(trace.spans.clone())
        } else {
            Ok(Vec::new())
        }
    }

    /// Query traces by time range
    pub async fn query_by_time(
        &self,
        start_ns: i64,
        end_ns: i64,
        limit: usize,
    ) -> Result<Vec<TraceSummary>> {
        let traces = self.traces.read().await;

        let mut results: Vec<_> = traces
            .values()
            .filter(|t| t.start_time_ns >= start_ns && t.start_time_ns <= end_ns)
            .map(|t| TraceSummary {
                trace_id: t.trace_id.clone(),
                start_time_ns: t.start_time_ns,
                duration_ns: t.end_time_ns - t.start_time_ns,
                span_count: t.spans.len(),
                services: t.services.clone(),
                root_service: t
                    .spans
                    .iter()
                    .find(|s| s.parent_span_id.is_empty())
                    .map(|s| s.service_name.clone())
                    .unwrap_or_default(),
                root_operation: t
                    .spans
                    .iter()
                    .find(|s| s.parent_span_id.is_empty())
                    .map(|s| s.name.clone())
                    .unwrap_or_default(),
            })
            .collect();

        // Sort by start time descending
        results.sort_by_key(|r| std::cmp::Reverse(r.start_time_ns));

        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Query traces by service
    pub async fn query_by_service(
        &self,
        service: &str,
        start_ns: i64,
        end_ns: i64,
        limit: usize,
    ) -> Result<Vec<TraceSummary>> {
        let service_index = self.service_index.read().await;
        let trace_ids = service_index.get(service);

        if trace_ids.is_none() {
            return Ok(Vec::new());
        }
        let Some(trace_ids) = trace_ids else {
            return Ok(Vec::new());
        };

        let traces = self.traces.read().await;
        let mut results = Vec::new();

        for trace_id in trace_ids {
            if let Some(trace) = traces.get(trace_id)
                && trace.start_time_ns >= start_ns
                && trace.start_time_ns <= end_ns
            {
                results.push(TraceSummary {
                    trace_id: trace.trace_id.clone(),
                    start_time_ns: trace.start_time_ns,
                    duration_ns: trace.end_time_ns - trace.start_time_ns,
                    span_count: trace.spans.len(),
                    services: trace.services.clone(),
                    root_service: trace
                        .spans
                        .iter()
                        .find(|s| s.parent_span_id.is_empty())
                        .map(|s| s.service_name.clone())
                        .unwrap_or_default(),
                    root_operation: trace
                        .spans
                        .iter()
                        .find(|s| s.parent_span_id.is_empty())
                        .map(|s| s.name.clone())
                        .unwrap_or_default(),
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Get service dependency graph
    pub async fn service_dependencies(&self) -> Result<Vec<ServiceDependency>> {
        let traces = self.traces.read().await;
        let mut deps: HashMap<(String, String), u64> = HashMap::new();

        for trace in traces.values() {
            // Build parent -> child relationships
            let span_map: HashMap<_, _> =
                trace.spans.iter().map(|s| (s.span_id.clone(), s)).collect();

            for span in &trace.spans {
                if !span.parent_span_id.is_empty()
                    && let Some(parent) = span_map.get(&span.parent_span_id)
                    && parent.service_name != span.service_name
                {
                    let key = (parent.service_name.clone(), span.service_name.clone());
                    *deps.entry(key).or_insert(0) += 1;
                }
            }
        }

        let results: Vec<_> = deps
            .into_iter()
            .map(|((from, to), count)| ServiceDependency {
                source_service: from,
                target_service: to,
                call_count: count,
            })
            .collect();

        Ok(results)
    }

    /// Get total span count
    pub async fn count(&self) -> u64 {
        self.span_count.load(Ordering::Relaxed)
    }

    /// Get the total storage size in bytes
    pub async fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Estimate the size of a trace span in bytes
    fn estimate_span_size(&self, span: &TraceSpan) -> u64 {
        let mut size = 100; // Base overhead

        // Add strings
        size += span.trace_id.len() as u64;
        size += span.span_id.len() as u64;
        size += span.parent_span_id.len() as u64;
        size += span.name.len() as u64;
        size += span.service_name.len() as u64;
        size += span.status_message.len() as u64;

        // Add timestamps
        size += 16; // 2 x i64

        // Add attributes
        for (key, val) in &span.attributes {
            size += key.len() as u64;
            size += val.len() as u64;
        }

        // Add status
        size += 4; // i32

        size
    }

    /// Get trace count
    pub async fn trace_count(&self) -> usize {
        self.traces.read().await.len()
    }

    /// Delete old traces
    pub async fn delete_before(&self, timestamp_ns: i64) -> Result<usize> {
        let mut traces = self.traces.write().await;
        let to_remove: Vec<_> = traces
            .iter()
            .filter(|(_, t)| t.end_time_ns < timestamp_ns)
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for trace_id in to_remove {
            traces.remove(&trace_id);
        }

        Ok(count)
    }
}

/// Summary of a trace — moved to `proximadb_observability_engine::model`.
pub use crate::observability::TraceSummary;

/// Service dependency edge
#[derive(Debug, Clone)]
pub struct ServiceDependency {
    /// Source service
    pub source_service: String,
    /// Target service
    pub target_service: String,
    /// Number of calls
    pub call_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_observability::SPAN_LABEL;

    fn make_span(
        trace_id: &str,
        span_id: &str,
        parent_span_id: &str,
        service: &str,
        name: &str,
        start_ns: i64,
        end_ns: i64,
    ) -> TraceSpan {
        TraceSpan {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            parent_span_id: parent_span_id.to_string(),
            name: name.to_string(),
            service_name: service.to_string(),
            start_time_ns: start_ns,
            end_time_ns: end_ns,
            attributes: HashMap::new(),
            status: 0,
            status_message: String::new(),
        }
    }

    #[test]
    fn projection_record_uses_observability_canonical_shape() {
        let storage = TraceStorage::new_for_namespace("/tmp/test_trace_projection", "tenant-a")
            .expect("Failed to create TraceStorage");
        let mut span = make_span("trace1", "span1", "", "svc-a", "op1", 100, 250);
        span.attributes
            .insert("http.method".to_string(), "GET".to_string());

        let record = storage.span_projection_record(&span);

        assert_eq!(record.oid, "obs://tenant-a/span/trace1:span1");
        assert_eq!(record.tenant_id, "tenant-a");
        assert!(record.labels.iter().any(|label| label == SPAN_LABEL));
        assert_eq!(record.created_at_ns, 100);
        assert_eq!(record.updated_at_ns, 100);
        assert_eq!(record.valid_from_ns, Some(100));
        // End time is data (props.end_time_ns), not a TTL (valid_to_ns).
        assert_eq!(record.valid_to_ns, None);
        assert!(record.props.contains_key("attributes"));
    }

    #[tokio::test]
    async fn test_write_and_query() {
        let storage = TraceStorage::new("/tmp/test").unwrap();

        let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

        storage
            .write(&make_span(
                "trace1",
                "span1",
                "",
                "svc-a",
                "op1",
                now,
                now + 1000,
            ))
            .await
            .unwrap();
        storage
            .write(&make_span(
                "trace1",
                "span2",
                "span1",
                "svc-b",
                "op2",
                now + 100,
                now + 500,
            ))
            .await
            .unwrap();

        let spans = storage.query_by_trace_id("trace1").await.unwrap();
        assert_eq!(spans.len(), 2);
    }

    #[tokio::test]
    async fn test_service_dependencies() {
        let storage = TraceStorage::new("/tmp/test").unwrap();

        let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

        storage
            .write(&make_span(
                "trace1",
                "span1",
                "",
                "frontend",
                "handle",
                now,
                now + 1000,
            ))
            .await
            .unwrap();
        storage
            .write(&make_span(
                "trace1",
                "span2",
                "span1",
                "backend",
                "process",
                now + 100,
                now + 500,
            ))
            .await
            .unwrap();
        storage
            .write(&make_span(
                "trace1",
                "span3",
                "span2",
                "database",
                "query",
                now + 200,
                now + 400,
            ))
            .await
            .unwrap();

        let deps = storage.service_dependencies().await.unwrap();
        assert_eq!(deps.len(), 2);
    }
}
