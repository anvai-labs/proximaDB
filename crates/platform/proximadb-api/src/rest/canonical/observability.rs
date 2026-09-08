//! # Observability REST Handlers
//!
//! REST endpoints for logs, metrics, and traces.  All handlers delegate to
//! `ObservabilityPort` so this module compiles without any dependency on
//! root-crate concrete service types.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Router,
    extract::{Json, Path, State},
    response::Json as JsonResponse,
    routing::post,
};
use proximadb_proto::v1::{
    AggregateMetricsRequest, CreateObservabilityNamespaceRequest, IngestLogsRequest,
    IngestMetricsRequest, LogEntry, LogFilter, MetricAggregation, MetricSample,
    ObservabilityNamespaceConfig, QueryLogsRequest, RetentionConfig, Severity, SortOrder, SqlValue,
};
use proximadb_runtime::ObservabilityPort;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::rest::errors::{RestError, RestResult};

// ── State ─────────────────────────────────────────────────────────────────────

/// Axum state for observability REST endpoints.
#[derive(Clone)]
pub struct ObservabilityRestState {
    pub observability_port: Arc<dyn ObservabilityPort>,
}

// ── Legacy stub types kept for re-export compatibility ────────────────────────

/// Logs handler stub.
pub struct LogsHandler;

impl LogsHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LogsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics handler stub.
pub struct MetricsHandler;

impl MetricsHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetricsHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

/// REST request body for observability-namespace creation.
///
/// `…Body` suffix distinguishes this local REST shape from the proto-
/// generated `proximadb_proto::v1::CreateNamespaceRequest` (the gRPC/wire
/// type). Same convention as `CreateDocumentCollectionRequestBody`.
#[derive(Debug, Deserialize)]
pub struct CreateNamespaceRequestBody {
    pub name: String,
    #[serde(default = "default_hot")]
    pub hot_retention_days: u32,
    #[serde(default = "default_warm")]
    pub warm_retention_days: u32,
    #[serde(default = "default_cold")]
    pub cold_retention_days: u32,
}

fn default_hot() -> u32 {
    1
}
fn default_warm() -> u32 {
    7
}
fn default_cold() -> u32 {
    30
}

#[derive(Debug, Deserialize)]
pub struct LogEntryRequest {
    pub timestamp_ns: Option<i64>,
    pub message: String,
    #[serde(default = "default_severity_str")]
    pub severity: String,
    pub source: Option<String>,
    pub service: Option<String>,
    #[serde(default)]
    pub fields: HashMap<String, serde_json::Value>,
}

fn default_severity_str() -> String {
    "info".to_string()
}

#[derive(Debug, Deserialize)]
pub struct BulkLogRequest {
    pub logs: Vec<LogEntryRequest>,
}

#[derive(Debug, Deserialize)]
pub struct LogQueryRequest {
    pub start_time_ns: Option<i64>,
    pub end_time_ns: Option<i64>,
    pub query: Option<String>,
    #[serde(default)]
    pub severities: Vec<String>,
    #[serde(default)]
    pub services: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    pub cursor: Option<String>,
}

fn default_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct MetricSampleRequest {
    pub name: String,
    pub value: f64,
    pub timestamp_ns: Option<i64>,
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct BulkMetricRequest {
    pub metrics: Vec<MetricSampleRequest>,
}

#[derive(Debug, Deserialize)]
pub struct MetricAggregationRequest {
    pub metric_name: String,
    pub start_time_ns: i64,
    pub end_time_ns: i64,
    #[serde(default = "default_aggregation")]
    pub aggregation: String,
    #[serde(default = "default_step")]
    pub step_seconds: u32,
    #[serde(default)]
    pub group_by: Vec<String>,
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

fn default_aggregation() -> String {
    "avg".to_string()
}
fn default_step() -> u32 {
    60
}

#[derive(Debug, Deserialize)]
pub struct TraceIngestRequest {
    pub spans: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct TraceQueryRequest {
    pub trace_id: Option<String>,
    pub service: Option<String>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct PromQLRequest {
    pub query: String,
    pub start_ns: Option<i64>,
    pub end_ns: Option<i64>,
    pub step_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub ingested: u64,
    pub failed: u64,
    pub success: bool,
}

#[derive(Debug, Serialize)]
pub struct LogEntryResponse {
    pub timestamp_ns: i64,
    pub severity: String,
    pub message: String,
    pub source: Option<String>,
    pub service: Option<String>,
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct LogResponse {
    pub logs: Vec<LogEntryResponse>,
    pub next_cursor: Option<String>,
    pub total_matched: Option<u64>,
    pub query_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct TimeSeriesResponse {
    pub labels: HashMap<String, String>,
    pub points: Vec<DataPointResponse>,
}

#[derive(Debug, Serialize)]
pub struct DataPointResponse {
    pub timestamp_ns: i64,
    pub value: f64,
}

#[derive(Debug, Serialize)]
pub struct MetricAggResponse {
    pub series: Vec<TimeSeriesResponse>,
    pub query_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct TraceResponse {
    pub spans: Vec<serde_json::Value>,
    pub total: u64,
}

#[derive(Debug, Serialize)]
pub struct PromQLResponse {
    pub result_type: String,
    pub result: Vec<serde_json::Value>,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn create_observability_router() -> Router<ObservabilityRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            .route("/namespaces", post(create_namespace))
            .route("/namespaces/{namespace}/logs/bulk", post(ingest_logs))
            .route("/namespaces/{namespace}/logs", post(ingest_log))
            .route("/namespaces/{namespace}/logs/search", post(query_logs))
            .route("/namespaces/{namespace}/metrics/bulk", post(ingest_metrics))
            .route("/namespaces/{namespace}/metrics", post(ingest_metric))
            .route(
                "/namespaces/{namespace}/metrics/aggregate",
                post(aggregate_metrics),
            )
            .route("/namespaces/{namespace}/metrics/promql", post(query_promql))
            .route("/namespaces/{namespace}/traces/bulk", post(ingest_traces))
            .route("/namespaces/{namespace}/traces/search", post(query_traces)),
    )
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn create_namespace(
    State(state): State<ObservabilityRestState>,
    Json(request): Json<CreateNamespaceRequestBody>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    let retention = RetentionConfig {
        hot_retention_hours: (request.hot_retention_days * 24) as u64,
        warm_retention_days: request.warm_retention_days as u64,
        cold_retention_days: request.cold_retention_days as u64,
        archive_retention_days: 365,
    };

    let config = ObservabilityNamespaceConfig {
        name: request.name.clone(),
        retention: Some(retention),
        ingestion: None,
        alert_rules: Vec::new(),
        access: None,
    };

    state
        .observability_port
        .create_namespace(CreateObservabilityNamespaceRequest {
            config: Some(config),
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to create namespace: {}", e)))?;

    Ok(JsonResponse(
        serde_json::json!({ "success": true, "namespace": request.name }),
    ))
}

async fn ingest_log(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<LogEntryRequest>,
) -> RestResult<JsonResponse<IngestResponse>> {
    let entry = convert_log_request(&request)?;

    let result = state
        .observability_port
        .ingest_logs(IngestLogsRequest {
            namespace,
            logs: vec![entry],
            format: None,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to ingest log: {}", e)))?;

    Ok(JsonResponse(IngestResponse {
        ingested: result.ingested,
        failed: result.failed,
        success: result.failed == 0,
    }))
}

async fn ingest_logs(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<BulkLogRequest>,
) -> RestResult<JsonResponse<IngestResponse>> {
    debug!("Ingesting {} logs into {}", request.logs.len(), namespace);

    let entries: Vec<LogEntry> = request
        .logs
        .iter()
        .map(convert_log_request)
        .collect::<Result<Vec<_>, _>>()?;

    let result = state
        .observability_port
        .ingest_logs(IngestLogsRequest {
            namespace,
            logs: entries,
            format: None,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to ingest logs: {}", e)))?;

    Ok(JsonResponse(IngestResponse {
        ingested: result.ingested,
        failed: result.failed,
        success: result.failed == 0,
    }))
}

async fn query_logs(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<LogQueryRequest>,
) -> RestResult<JsonResponse<LogResponse>> {
    debug!("Querying logs in namespace: {}", namespace);

    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let one_hour_ns = 3_600_000_000_000_i64;

    let severities: Vec<i32> = request
        .severities
        .iter()
        .map(|s| parse_severity(s) as i32)
        .collect();

    let filter =
        if severities.is_empty() && request.services.is_empty() && request.sources.is_empty() {
            None
        } else {
            Some(LogFilter {
                severities,
                services: request.services.clone(),
                sources: request.sources.clone(),
                field_filters: Vec::new(),
            })
        };

    let result = state
        .observability_port
        .query_logs(QueryLogsRequest {
            namespace,
            start_time_ns: request.start_time_ns.unwrap_or(now_ns - one_hour_ns),
            end_time_ns: request.end_time_ns.unwrap_or(now_ns),
            query: request.query,
            filter,
            limit: request.limit,
            cursor: request.cursor,
            sort: SortOrder::Desc as i32,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to query logs: {}", e)))?;

    let logs: Vec<LogEntryResponse> = result
        .logs
        .into_iter()
        .map(convert_log_to_response)
        .collect();

    Ok(JsonResponse(LogResponse {
        logs,
        next_cursor: result.next_cursor,
        total_matched: Some(result.total_matched),
        query_time_ms: result.query_time_ms,
    }))
}

async fn ingest_metric(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<MetricSampleRequest>,
) -> RestResult<JsonResponse<IngestResponse>> {
    let sample = convert_metric_request(&request);

    let result = state
        .observability_port
        .ingest_metrics(IngestMetricsRequest {
            namespace,
            samples: vec![sample],
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to ingest metric: {}", e)))?;

    Ok(JsonResponse(IngestResponse {
        ingested: result.ingested,
        failed: result.failed,
        success: result.failed == 0,
    }))
}

async fn ingest_metrics(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<BulkMetricRequest>,
) -> RestResult<JsonResponse<IngestResponse>> {
    debug!(
        "Ingesting {} metrics into {}",
        request.metrics.len(),
        namespace
    );

    let samples: Vec<MetricSample> = request.metrics.iter().map(convert_metric_request).collect();

    let result = state
        .observability_port
        .ingest_metrics(IngestMetricsRequest { namespace, samples })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to ingest metrics: {}", e)))?;

    Ok(JsonResponse(IngestResponse {
        ingested: result.ingested,
        failed: result.failed,
        success: result.failed == 0,
    }))
}

async fn aggregate_metrics(
    State(state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<MetricAggregationRequest>,
) -> RestResult<JsonResponse<MetricAggResponse>> {
    debug!("Aggregating metrics in namespace: {}", namespace);

    let result = state
        .observability_port
        .aggregate_metrics(AggregateMetricsRequest {
            namespace,
            metric_name: request.metric_name,
            start_time_ns: request.start_time_ns,
            end_time_ns: request.end_time_ns,
            aggregation: parse_aggregation(&request.aggregation) as i32,
            step_seconds: request.step_seconds,
            label_filters: request.labels,
            group_by: request.group_by,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to aggregate metrics: {}", e)))?;

    let series: Vec<TimeSeriesResponse> = result
        .series
        .into_iter()
        .map(|s| TimeSeriesResponse {
            labels: s.labels,
            points: s
                .points
                .into_iter()
                .map(|p| DataPointResponse {
                    timestamp_ns: p.timestamp_ns,
                    value: p.value,
                })
                .collect(),
        })
        .collect();

    Ok(JsonResponse(MetricAggResponse {
        series,
        query_time_ms: result.query_time_ms,
    }))
}

async fn query_promql(
    State(_state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<PromQLRequest>,
) -> RestResult<JsonResponse<PromQLResponse>> {
    debug!("PromQL query in namespace {}: {}", namespace, request.query);
    // No PromQL engine yet (arrives with CHRONO). Fail honestly with 501
    // rather than returning an empty vector — an always-empty result is
    // indistinguishable from "no data" and silently misleads pollers/dashboards
    // (the silent-Ok anti-pattern; TD-SPECRAT-1 wave 2).
    Err(RestError::NotImplemented(
        "PromQL queries arrive with the CHRONO engine (TD-SPECRAT-1 follow-up)".to_string(),
    ))
}

async fn ingest_traces(
    State(_state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<TraceIngestRequest>,
) -> RestResult<JsonResponse<IngestResponse>> {
    debug!(
        "Ingesting {} trace spans into {}",
        request.spans.len(),
        namespace
    );
    // The PORT already carries trace ingest (backed by storage and served
    // over gRPC today) — what is missing is this REST adapter's JSON-span →
    // proto TraceData mapping. Previously this handler fabricated success
    // (ingested=total) without persisting anything (the silent-Ok
    // anti-pattern). Fail honestly with 501 until the adapter mapping lands
    // (TD-SPECRAT-1 wave 2).
    Err(RestError::NotImplemented(
        "Trace ingest REST adapter is not mapped to the port yet (TD-SPECRAT-1 follow-up)"
            .to_string(),
    ))
}

async fn query_traces(
    State(_state): State<ObservabilityRestState>,
    Path(namespace): Path<String>,
    Json(request): Json<TraceQueryRequest>,
) -> RestResult<JsonResponse<TraceResponse>> {
    debug!(
        "Querying traces in namespace: {} (trace_id={:?}, service={:?}, range={}..{})",
        namespace, request.trace_id, request.service, request.start_ns, request.end_ns
    );
    // The PORT's query_traces is backed by storage (gRPC serves it); this
    // REST adapter is not mapped yet. Returning an empty result set would
    // read as "no matching spans" instead of "capability absent" — fail
    // honestly with 501 (TD-SPECRAT-1 wave 2).
    Err(RestError::NotImplemented(
        "Trace query REST adapter is not mapped to the port yet (TD-SPECRAT-1 follow-up)"
            .to_string(),
    ))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn convert_log_request(req: &LogEntryRequest) -> RestResult<LogEntry> {
    use proximadb_proto::v1::sql_value::Value as SV;

    let timestamp_ns = req
        .timestamp_ns
        .unwrap_or_else(|| chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));

    let fields: HashMap<String, SqlValue> = req
        .fields
        .iter()
        .map(|(k, v)| {
            let inner = match v {
                serde_json::Value::String(s) => Some(SV::StringValue(s.clone())),
                serde_json::Value::Bool(b) => Some(SV::BoolValue(*b)),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Some(SV::Int64Value(i))
                    } else {
                        n.as_f64().map(SV::NumberValue)
                    }
                }
                _ => Some(SV::StringValue(v.to_string())),
            };
            (k.clone(), SqlValue { value: inner })
        })
        .collect();

    Ok(LogEntry {
        timestamp_ns,
        severity: parse_severity(&req.severity) as i32,
        message: req.message.clone(),
        fields,
        source: req.source.clone(),
        service: req.service.clone(),
    })
}

fn convert_log_to_response(entry: LogEntry) -> LogEntryResponse {
    let fields: HashMap<String, serde_json::Value> = entry
        .fields
        .into_iter()
        .map(|(k, v)| {
            let json_val = proximadb_records::conversions::sql_value_to_json(&v);
            (k, json_val)
        })
        .collect();

    LogEntryResponse {
        timestamp_ns: entry.timestamp_ns,
        severity: severity_to_string(Severity::try_from(entry.severity).unwrap_or(Severity::Info)),
        message: entry.message,
        source: entry.source,
        service: entry.service,
        fields,
    }
}

fn convert_metric_request(req: &MetricSampleRequest) -> MetricSample {
    MetricSample {
        name: req.name.clone(),
        value: req.value,
        timestamp_ns: req
            .timestamp_ns
            .unwrap_or_else(|| chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        labels: req.labels.clone(),
    }
}

fn parse_severity(s: &str) -> Severity {
    match s.to_lowercase().as_str() {
        "trace" | "verbose" => Severity::Trace,
        "debug" => Severity::Debug,
        "info" | "information" => Severity::Info,
        "warn" | "warning" => Severity::Warn,
        "error" | "err" => Severity::Error,
        "fatal" | "critical" => Severity::Fatal,
        _ => Severity::Info,
    }
}

fn severity_to_string(severity: Severity) -> String {
    match severity {
        Severity::Trace => "trace",
        Severity::Debug => "debug",
        Severity::Info => "info",
        Severity::Warn => "warn",
        Severity::Error => "error",
        Severity::Fatal => "fatal",
        Severity::Unspecified => "info",
    }
    .to_string()
}

fn parse_aggregation(s: &str) -> MetricAggregation {
    match s.to_lowercase().as_str() {
        "sum" => MetricAggregation::Sum,
        "min" => MetricAggregation::Min,
        "max" => MetricAggregation::Max,
        "count" => MetricAggregation::Count,
        "rate" => MetricAggregation::Rate,
        "p50" => MetricAggregation::P50,
        "p90" => MetricAggregation::P90,
        "p95" => MetricAggregation::P95,
        "p99" => MetricAggregation::P99,
        _ => MetricAggregation::Avg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use proximadb_proto::v1 as proto;

    struct MockObservabilityPort;

    #[async_trait]
    impl ObservabilityPort for MockObservabilityPort {
        async fn create_namespace(
            &self,
            _request: proto::CreateObservabilityNamespaceRequest,
        ) -> Result<proto::CreateObservabilityNamespaceResponse> {
            Ok(proto::CreateObservabilityNamespaceResponse::default())
        }

        async fn list_namespaces(
            &self,
            _request: proto::ListNamespacesRequest,
        ) -> Result<proto::ListNamespacesResponse> {
            Ok(proto::ListNamespacesResponse::default())
        }

        async fn delete_namespace(
            &self,
            _request: proto::DeleteNamespaceRequest,
        ) -> Result<proto::DeleteNamespaceResponse> {
            Ok(proto::DeleteNamespaceResponse::default())
        }

        async fn ingest_logs(
            &self,
            request: proto::IngestLogsRequest,
        ) -> Result<proto::IngestLogsResponse> {
            Ok(proto::IngestLogsResponse {
                ingested: request.logs.len() as u64,
                failed: 0,
                errors: Vec::new(),
                processing_time_ms: 1,
            })
        }

        async fn query_logs(
            &self,
            _request: proto::QueryLogsRequest,
        ) -> Result<proto::QueryLogsResponse> {
            Ok(proto::QueryLogsResponse {
                logs: vec![proto::LogEntry {
                    timestamp_ns: 7,
                    severity: proto::Severity::Warn as i32,
                    message: "warned".to_string(),
                    fields: HashMap::new(),
                    source: Some("node-a".to_string()),
                    service: Some("api".to_string()),
                }],
                next_cursor: Some("next".to_string()),
                total_matched: 1,
                query_time_ms: 2,
            })
        }

        async fn stream_logs(
            &self,
            _request: proto::QueryLogsRequest,
        ) -> Result<Vec<proto::LogEntry>> {
            Ok(Vec::new())
        }

        async fn ingest_metrics(
            &self,
            request: proto::IngestMetricsRequest,
        ) -> Result<proto::IngestMetricsResponse> {
            Ok(proto::IngestMetricsResponse {
                ingested: request.samples.len() as u64,
                failed: 0,
                processing_time_ms: 1,
            })
        }

        async fn query_metrics(
            &self,
            _request: proto::QueryMetricsRequest,
        ) -> Result<proto::QueryMetricsResponse> {
            Ok(proto::QueryMetricsResponse::default())
        }

        async fn aggregate_metrics(
            &self,
            _request: proto::AggregateMetricsRequest,
        ) -> Result<proto::AggregateMetricsResponse> {
            let mut labels = HashMap::new();
            labels.insert("service".to_string(), "api".to_string());
            Ok(proto::AggregateMetricsResponse {
                series: vec![proto::TimeSeriesResult {
                    labels,
                    points: vec![proto::DataPoint {
                        timestamp_ns: 10,
                        value: 99.0,
                    }],
                }],
                query_time_ms: 3,
            })
        }

        async fn ingest_traces(
            &self,
            request: proto::IngestTracesRequest,
        ) -> Result<proto::IngestTracesResponse> {
            Ok(proto::IngestTracesResponse {
                ingested: request.traces.len() as u64,
                failed: 0,
                processing_time_ms: 1,
            })
        }

        async fn query_traces(
            &self,
            _request: proto::QueryTracesRequest,
        ) -> Result<proto::QueryTracesResponse> {
            Ok(proto::QueryTracesResponse::default())
        }

        async fn get_trace(
            &self,
            _request: proto::GetTraceRequest,
        ) -> Result<proto::GetTraceResponse> {
            Ok(proto::GetTraceResponse::default())
        }

        async fn upsert_alert_rule(
            &self,
            _request: proto::UpsertAlertRuleRequest,
        ) -> Result<proto::UpsertAlertRuleResponse> {
            Ok(proto::UpsertAlertRuleResponse::default())
        }

        async fn delete_alert_rule(
            &self,
            _request: proto::DeleteAlertRuleRequest,
        ) -> Result<proto::DeleteAlertRuleResponse> {
            Ok(proto::DeleteAlertRuleResponse::default())
        }

        async fn list_alerts(
            &self,
            _request: proto::ListAlertsRequest,
        ) -> Result<proto::ListAlertsResponse> {
            Ok(proto::ListAlertsResponse::default())
        }
    }

    fn state() -> State<ObservabilityRestState> {
        State(ObservabilityRestState {
            observability_port: Arc::new(MockObservabilityPort),
        })
    }

    #[test]
    fn test_parse_severity() {
        assert_eq!(parse_severity("debug"), Severity::Debug);
        assert_eq!(parse_severity("INFO"), Severity::Info);
        assert_eq!(parse_severity("warn"), Severity::Warn);
        assert_eq!(parse_severity("ERROR"), Severity::Error);
        assert_eq!(parse_severity("fatal"), Severity::Fatal);
    }

    #[test]
    fn defaults_severity_and_aggregation_helpers_cover_all_wire_names() {
        assert_eq!(default_hot(), 1);
        assert_eq!(default_warm(), 7);
        assert_eq!(default_cold(), 30);
        assert_eq!(default_severity_str(), "info");
        assert_eq!(default_limit(), 100);
        assert_eq!(default_aggregation(), "avg");
        assert_eq!(default_step(), 60);

        assert_eq!(parse_severity("trace"), Severity::Trace);
        assert_eq!(parse_severity("verbose"), Severity::Trace);
        assert_eq!(parse_severity("information"), Severity::Info);
        assert_eq!(parse_severity("warning"), Severity::Warn);
        assert_eq!(parse_severity("err"), Severity::Error);
        assert_eq!(parse_severity("critical"), Severity::Fatal);
        assert_eq!(parse_severity("unknown"), Severity::Info);

        for (severity, expected) in [
            (Severity::Trace, "trace"),
            (Severity::Debug, "debug"),
            (Severity::Info, "info"),
            (Severity::Warn, "warn"),
            (Severity::Error, "error"),
            (Severity::Fatal, "fatal"),
            (Severity::Unspecified, "info"),
        ] {
            assert_eq!(severity_to_string(severity), expected);
        }

        for (name, expected) in [
            ("sum", MetricAggregation::Sum),
            ("min", MetricAggregation::Min),
            ("max", MetricAggregation::Max),
            ("count", MetricAggregation::Count),
            ("rate", MetricAggregation::Rate),
            ("p50", MetricAggregation::P50),
            ("p90", MetricAggregation::P90),
            ("p95", MetricAggregation::P95),
            ("p99", MetricAggregation::P99),
            ("avg", MetricAggregation::Avg),
        ] {
            assert_eq!(parse_aggregation(name), expected);
        }
    }

    #[test]
    fn log_and_metric_conversion_helpers_preserve_request_fields() {
        let mut fields = HashMap::new();
        fields.insert("text".to_string(), serde_json::json!("value"));
        fields.insert("ok".to_string(), serde_json::json!(true));
        fields.insert("count".to_string(), serde_json::json!(3));
        fields.insert("ratio".to_string(), serde_json::json!(1.5));
        fields.insert("nested".to_string(), serde_json::json!({"k": "v"}));

        let mut entry = convert_log_request(&LogEntryRequest {
            timestamp_ns: Some(123),
            message: "hello".to_string(),
            severity: "error".to_string(),
            source: Some("node-a".to_string()),
            service: Some("api".to_string()),
            fields,
        })
        .unwrap();
        assert_eq!(entry.timestamp_ns, 123);
        assert_eq!(entry.severity, Severity::Error as i32);
        assert_eq!(entry.fields.len(), 5);

        let jsonb = serde_json::json!({"trace": {"sampled": true}});
        entry.fields.insert(
            "jsonb".to_string(),
            SqlValue {
                value: Some(proximadb_proto::v1::sql_value::Value::JsonbValue(
                    proximadb_data_model::ProximaValue::to_jsonb_vec(&jsonb)
                        .expect("encode JSONB test value"),
                )),
            },
        );

        let response = convert_log_to_response(entry);
        assert_eq!(response.severity, "error");
        assert_eq!(response.fields["text"], serde_json::json!("value"));
        assert_eq!(response.fields["count"], serde_json::json!(3));
        assert_eq!(response.fields["jsonb"], jsonb);

        let mut labels = HashMap::new();
        labels.insert("service".to_string(), "api".to_string());
        let sample = convert_metric_request(&MetricSampleRequest {
            name: "latency_ms".to_string(),
            value: 12.5,
            timestamp_ns: Some(999),
            labels: labels.clone(),
        });
        assert_eq!(sample.name, "latency_ms");
        assert_eq!(sample.value, 12.5);
        assert_eq!(sample.timestamp_ns, 999);
        assert_eq!(sample.labels, labels);
    }

    #[tokio::test]
    async fn observability_handlers_route_successful_requests_through_port() {
        let JsonResponse(namespace) = create_namespace(
            state(),
            Json(CreateNamespaceRequestBody {
                name: "ops".to_string(),
                hot_retention_days: 2,
                warm_retention_days: 8,
                cold_retention_days: 31,
            }),
        )
        .await
        .unwrap();
        assert_eq!(namespace["namespace"], "ops");

        let JsonResponse(single_log) = ingest_log(
            state(),
            Path("ops".to_string()),
            Json(LogEntryRequest {
                timestamp_ns: Some(1),
                message: "one".to_string(),
                severity: "info".to_string(),
                source: None,
                service: None,
                fields: HashMap::new(),
            }),
        )
        .await
        .unwrap();
        assert!(single_log.success);

        let JsonResponse(bulk_logs) = ingest_logs(
            state(),
            Path("ops".to_string()),
            Json(BulkLogRequest {
                logs: vec![LogEntryRequest {
                    timestamp_ns: Some(2),
                    message: "two".to_string(),
                    severity: "warn".to_string(),
                    source: None,
                    service: None,
                    fields: HashMap::new(),
                }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(bulk_logs.ingested, 1);

        let JsonResponse(logs) = query_logs(
            state(),
            Path("ops".to_string()),
            Json(LogQueryRequest {
                start_time_ns: None,
                end_time_ns: None,
                query: Some("warned".to_string()),
                severities: vec!["warn".to_string()],
                services: vec!["api".to_string()],
                sources: vec!["node-a".to_string()],
                limit: 10,
                cursor: Some("cursor".to_string()),
            }),
        )
        .await
        .unwrap();
        assert_eq!(logs.logs.len(), 1);
        assert_eq!(logs.next_cursor.as_deref(), Some("next"));

        let JsonResponse(single_metric) = ingest_metric(
            state(),
            Path("ops".to_string()),
            Json(MetricSampleRequest {
                name: "latency".to_string(),
                value: 1.0,
                timestamp_ns: Some(1),
                labels: HashMap::new(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(single_metric.ingested, 1);

        let JsonResponse(bulk_metrics) = ingest_metrics(
            state(),
            Path("ops".to_string()),
            Json(BulkMetricRequest {
                metrics: vec![MetricSampleRequest {
                    name: "latency".to_string(),
                    value: 2.0,
                    timestamp_ns: Some(2),
                    labels: HashMap::new(),
                }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(bulk_metrics.ingested, 1);

        let JsonResponse(aggregated) = aggregate_metrics(
            state(),
            Path("ops".to_string()),
            Json(MetricAggregationRequest {
                metric_name: "latency".to_string(),
                start_time_ns: 1,
                end_time_ns: 2,
                aggregation: "p95".to_string(),
                step_seconds: 60,
                group_by: vec!["service".to_string()],
                labels: HashMap::new(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(aggregated.series[0].points[0].value, 99.0);
    }

    #[tokio::test]
    async fn unwired_capabilities_fail_501_not_fabricated_success() {
        // TD-SPECRAT-1 wave 2: promql + traces are not wired to storage.
        // They MUST fail with NotImplemented (501) — an empty/successful
        // shape is indistinguishable from "no data" and misleads callers
        // (the silent-Ok anti-pattern this test pins against regressing).
        let promql = query_promql(
            state(),
            Path("ops".to_string()),
            Json(PromQLRequest {
                query: "up".to_string(),
                start_ns: Some(1),
                end_ns: Some(2),
                step_ms: Some(1000),
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(promql, RestError::NotImplemented(_)));

        let traces = ingest_traces(
            state(),
            Path("ops".to_string()),
            Json(TraceIngestRequest {
                spans: vec![serde_json::json!({"span_id": "s1"})],
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(traces, RestError::NotImplemented(_)));

        let query = query_traces(
            state(),
            Path("ops".to_string()),
            Json(TraceQueryRequest {
                trace_id: Some("trace-1".to_string()),
                service: Some("api".to_string()),
                start_ns: 1,
                end_ns: 2,
                limit: Some(10),
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(query, RestError::NotImplemented(_)));

        let _router = create_observability_router();
        let _logs = LogsHandler;
        let _metrics = MetricsHandler::new();
    }
}
