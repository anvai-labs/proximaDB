//! # Hybrid Search and SQL Handlers
//!
//! REST endpoints for SQL query execution, hybrid vector+keyword search,
//! BM25 full-text indexing, and liveness/readiness health probes.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Extension, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use proximadb_proto::v1::{FusionStrategy, HybridFusionSearchRequest, SqlValue};
use proximadb_runtime::{BM25Document, BM25IndexPort, HybridPort};
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use uuid::Uuid;

use crate::rest::errors::{RestError, RestResult};
use crate::rest::state::{RestAppState, TenantContext};

// ── Hybrid-search state ───────────────────────────────────────────────────────

/// Axum state for hybrid-search REST handlers.
#[derive(Clone)]
pub struct HybridRestState {
    pub hybrid_port: Arc<dyn HybridPort>,
    pub bm25_port: Option<Arc<dyn BM25IndexPort>>,
}

// ── Legacy stub types kept for re-export compatibility ────────────────────────

/// Hybrid search handler stub.
pub struct HybridSearchHandler;

impl HybridSearchHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HybridSearchHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Progressive search handler stub.
pub struct ProgressiveSearchHandler;

impl ProgressiveSearchHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ProgressiveSearchHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hybrid search request/response types ──────────────────────────────────────

/// Request body for `POST /api/v2/hybrid/search`.
#[derive(Debug, Deserialize)]
pub struct HybridSearchRestRequest {
    pub collection: String,
    pub vector: Option<Vec<f32>>,
    pub text_query: Option<String>,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    pub vector_weight: Option<f32>,
    pub rrf_k: Option<u32>,
    pub fusion_strategy: Option<String>,
    pub filters: Option<std::collections::HashMap<String, serde_json::Value>>,
}

fn default_top_k() -> u32 {
    10
}

/// Request body for `POST /api/v2/hybrid/index`.
#[derive(Debug, Deserialize)]
pub struct HybridIndexRestRequest {
    pub collection: String,
    pub documents: Vec<HybridDocumentBody>,
}

/// A text document for BM25 indexing.
#[derive(Debug, Deserialize)]
pub struct HybridDocumentBody {
    pub id: String,
    pub text: String,
}

/// Response for `POST /api/v2/hybrid/index`.
#[derive(Debug, Serialize)]
pub struct HybridIndexRestResponse {
    pub success: bool,
    pub collection: String,
    pub documents_indexed: usize,
    pub total_documents: usize,
}

// ── SQL request/response types ────────────────────────────────────────────────

/// Request body for the `POST /api/v1/sql/execute` endpoint.
#[derive(Debug, Deserialize)]
pub struct SqlQueryRequest {
    pub query: String,
    pub parameters: Option<Vec<SqlValue>>,
    pub collection: Option<String>,
    pub timeout_ms: Option<u64>,
    /// Optional seeding hint, e.g. `"AVERAGE"` / `"PER_SEED"` / `"NONE"`.
    pub seeding: Option<String>,
}

// ── Helper: proto SqlValue → serde_json::Value ────────────────────────────────

pub fn sql_value_to_json(v: &SqlValue) -> serde_json::Value {
    proximadb_records::conversions::sql_value_to_json(v)
}

// ── Handler functions ──────────────────────────────────────────────────────────

/// `POST /api/v2/hybrid/search` — BM25 + vector hybrid search.
///
/// Delegates to `HybridPort::hybrid_search`.
#[allow(deprecated)] // delegates to the deprecated v1 HybridPort::hybrid_search (TD-138 phase 1)
pub async fn hybrid_search(
    State(state): State<HybridRestState>,
    tenant: Option<Extension<TenantContext>>,
    Json(request): Json<HybridSearchRestRequest>,
) -> RestResult<Json<serde_json::Value>> {
    if request.collection.is_empty() {
        return Err(RestError::InvalidArgument(
            "collection is required".to_string(),
        ));
    }
    if request.vector.is_none() && request.text_query.is_none() {
        return Err(RestError::InvalidArgument(
            "at least one of vector or text_query is required".to_string(),
        ));
    }

    // TD-138 phase 1 (deprecate, keep live): surface every use loudly so operators
    // migrate to the v2 fusion seam. The endpoint already replies with HTTP
    // deprecation headers (`with_v1_compatibility_headers`); this is the per-request
    // runtime signal (mirrors TD-143's gRPC v1 deprecation warn).
    tracing::warn!(
        target: "proximadb::deprecation",
        collection = %request.collection,
        "POST /api/v{{1,2}}/hybrid/search (legacy 12-strategy HybridFusionEngine) is DEPRECATED \
         — doc_id-keyed vector+BM25, no graph modality. Use the v2 fusion seam: \
         POST /api/v2/graphs/{{graph_id}}/fusion-search (GraphFusionParams) or gRPC \
         ProximaFusionService.FusionSearch (vector+graph+document, oid-keyed, calibrated). See TD-138."
    );

    let proto_request = build_hybrid_fusion_request(request)?;

    // Tenant identity comes from the tenant middleware (X-Tenant-ID / JWT /
    // default-tenant layer). Threading it through the port lets the BM25
    // metadata-filter gate resolve the allowed id-set under the caller's
    // tenant instead of failing closed to an empty set on tenant-stamped
    // collections (#949).
    let tenant_id = tenant.map(|Extension(t)| t.tenant_id);

    match state
        .hybrid_port
        .hybrid_search(proto_request, tenant_id)
        .await
    {
        Ok(resp) => {
            let results: Vec<serde_json::Value> = resp
                .results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id,
                        "fused_score": r.fused_score,
                        "vector_score": r.vector_score,
                        "bm25_score": r.bm25_score,
                        "bm25_rank": r.bm25_rank,
                        "vector_rank": r.vector_rank,
                        "metadata": r.metadata,
                    })
                })
                .collect();
            Ok(Json(serde_json::json!({
                "results": results,
                "total": resp.results_count,
                "fusion_strategy": resp.fusion_strategy,
            })))
        }
        Err(e) => {
            error!("hybrid_search error: {}", e);
            Err(RestError::Internal(e.to_string()))
        }
    }
}

fn build_hybrid_fusion_request(
    request: HybridSearchRestRequest,
) -> RestResult<HybridFusionSearchRequest> {
    let (fusion_strategy, fusion_params) = rest_fusion_strategy_params(&request)?;
    Ok(HybridFusionSearchRequest {
        collection: request.collection,
        text_query: request.text_query.unwrap_or_default(),
        query_vector: request.vector.unwrap_or_default(),
        fusion_strategy,
        fusion_params,
        top_k: request.top_k,
        filters: request
            .filters
            .unwrap_or_default()
            .into_iter()
            .map(|(key, value)| (key, json_to_prost_value(value)))
            .collect(),
    })
}

fn rest_fusion_strategy_params(
    request: &HybridSearchRestRequest,
) -> RestResult<(i32, Option<proximadb_proto::v1::FusionStrategyParams>)> {
    use proximadb_proto::v1::{FusionStrategyParams, WeightedLinearParams, fusion_strategy_params};

    let name = request
        .fusion_strategy
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(if request.vector_weight.is_some() {
            "weighted_linear"
        } else {
            "rrf"
        })
        .to_ascii_lowercase();

    match name.as_str() {
        "rrf" | "reciprocal_rank" | "reciprocal_rank_fusion" => Ok((
            FusionStrategy::Rrf as i32,
            request.rrf_k.map(|k| FusionStrategyParams {
                params: Some(fusion_strategy_params::Params::RrfK(k)),
            }),
        )),
        "weighted_linear" | "linear" => {
            let vector_weight = request.vector_weight.unwrap_or(0.5);
            if !vector_weight.is_finite() || !(0.0..=1.0).contains(&vector_weight) {
                return Err(RestError::InvalidArgument(
                    "vector_weight must be finite and between 0.0 and 1.0".to_string(),
                ));
            }
            Ok((
                FusionStrategy::WeightedLinear as i32,
                Some(FusionStrategyParams {
                    params: Some(fusion_strategy_params::Params::WeightedLinear(
                        WeightedLinearParams {
                            alpha: 1.0 - vector_weight as f64,
                            bm25_normalize: true,
                            vector_normalize: true,
                        },
                    )),
                }),
            ))
        }
        "borda" | "borda_count" => Ok((FusionStrategy::BordaCount as i32, None)),
        "comb_sum" | "combsum" => Ok((FusionStrategy::CombSum as i32, None)),
        "comb_min" | "combmin" => Ok((FusionStrategy::CombMin as i32, None)),
        "comb_max" | "combmax" => Ok((FusionStrategy::CombMax as i32, None)),
        "rank_biased_precision" | "rbp" => Ok((FusionStrategy::RankBiasedPrecision as i32, None)),
        "condorcet" => Ok((FusionStrategy::Condorcet as i32, None)),
        "dempster_shafer" | "ds" => Ok((FusionStrategy::DempsterShafer as i32, None)),
        "adaptive" => Ok((FusionStrategy::Adaptive as i32, None)),
        other => Err(RestError::InvalidArgument(format!(
            "unsupported fusion_strategy '{}'",
            other
        ))),
    }
}

fn json_to_prost_value(value: serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(value) => Kind::BoolValue(value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Kind::StringValue(value),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.into_iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(fields) => Kind::StructValue(prost_types::Struct {
            fields: fields
                .into_iter()
                .map(|(key, value)| (key, json_to_prost_value(value)))
                .collect(),
        }),
    };

    prost_types::Value { kind: Some(kind) }
}

/// `POST /api/v2/hybrid/index` — index documents for BM25 full-text search.
///
/// Delegates to `BM25IndexPort::index_documents` when the port is wired;
/// returns 501 otherwise.
pub async fn hybrid_index(
    State(state): State<HybridRestState>,
    Json(request): Json<HybridIndexRestRequest>,
) -> RestResult<Json<HybridIndexRestResponse>> {
    if request.collection.is_empty() {
        return Err(RestError::InvalidArgument(
            "collection is required".to_string(),
        ));
    }
    if request.documents.is_empty() {
        return Err(RestError::InvalidArgument(
            "at least one document is required".to_string(),
        ));
    }

    let bm25 = state.bm25_port.as_ref().ok_or_else(|| {
        RestError::NotImplemented("BM25 indexing not wired in this server mode".to_string())
    })?;

    let docs: Vec<BM25Document> = request
        .documents
        .into_iter()
        .map(|d| BM25Document {
            id: d.id,
            text: d.text,
        })
        .collect();

    match bm25.index_documents(request.collection, docs).await {
        Ok(result) => Ok(Json(HybridIndexRestResponse {
            success: true,
            collection: result.collection,
            documents_indexed: result.documents_indexed,
            total_documents: result.total_documents,
        })),
        Err(e) => {
            error!("hybrid_index error: {}", e);
            Err(RestError::Internal(e.to_string()))
        }
    }
}

/// `POST /api/v1/sql/execute` — execute a SQL query.
///
/// Delegates to `ApiHandlersPort::execute_sql_v1`.  An optional `seeding` hint in the
/// request body is prepended as a SQL comment (`-- SEEDING: …`) before dispatch.
///
/// TD-ABAC-9 (ADR-087): consumes the foundation identity Extension inserted by
/// the root tenant middleware, threading the authenticated tenant + subject to
/// the relational read boundary. `None` (Extension absent — e.g. a router
/// mounted without the tenant layer) preserves the legacy unscoped behavior.
pub async fn execute_sql(
    State(state): State<RestAppState>,
    identity: Option<axum::Extension<proximadb_tenant::ResolvedRequestIdentity>>,
    Json(request): Json<SqlQueryRequest>,
) -> RestResult<Json<serde_json::Value>> {
    if request.query.trim().is_empty() {
        return Err(RestError::InvalidArgument(
            "SQL query cannot be empty".to_string(),
        ));
    }

    let start = std::time::Instant::now();
    let request_id = Uuid::new_v4().to_string();

    info!(
        "SQL query {}: {}",
        request_id,
        request.query.chars().take(100).collect::<String>()
    );

    let query = match &request.seeding {
        Some(s) => format!("-- SEEDING: {}\n{}", s.to_ascii_uppercase(), request.query),
        None => request.query.clone(),
    };

    let parameters = request.parameters.map(|values| {
        values
            .iter()
            .map(proximadb_records::conversions::sql_value_to_proxima)
            .collect()
    });
    let port_identity = identity
        .as_ref()
        .map(|id| proximadb_runtime::PortIdentity::from(&id.0))
        .unwrap_or_else(proximadb_runtime::PortIdentity::anonymous);

    match state
        .handlers
        .execute_sql_v1(query, parameters, request.collection, port_identity)
        .await
    {
        Ok(v1_resp) => {
            let elapsed_ms = start.elapsed().as_millis() as u64;

            let rows: Vec<serde_json::Value> = v1_resp
                .rows
                .iter()
                .map(|row| {
                    let mut obj = serde_json::Map::new();
                    for field in &row.fields {
                        let val = field
                            .value
                            .as_ref()
                            .map_or(serde_json::Value::Null, sql_value_to_json);
                        obj.insert(field.key.clone(), val);
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();

            info!("SQL query {} completed in {}ms", request_id, elapsed_ms);

            Ok(Json(serde_json::json!({
                "rows": rows,
                "columns": v1_resp.columns,
                "column_types": v1_resp.column_types,
                "execution_time_ms": elapsed_ms,
                "rows_returned": v1_resp.rows_returned,
                "row_count": v1_resp.rows_returned,
                "rows_scanned": v1_resp.rows_scanned,
                "request_id": request_id
            })))
        }
        Err(e) => {
            error!("SQL query {} failed: {}", request_id, e);
            Err(RestError::Internal(e.to_string()))
        }
    }
}

// ── Health probes ─────────────────────────────────────────────────────────────

/// `GET /health/live` — Kubernetes liveness probe.
///
/// Returns 200 as long as the process is running.
pub async fn liveness_check() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "alive" })),
    )
}

/// `GET /health/ready` — Kubernetes readiness probe.
///
/// Returns 200 when the service is accepting traffic.
pub async fn readiness_check(State(_state): State<RestAppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "ready" })),
    )
}

// ── Router configuration ──────────────────────────────────────────────────────

/// Build the hybrid search router (BM25 + vector search and indexing).
pub fn create_hybrid_search_router() -> Router<HybridRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            .route("/api/v2/hybrid/search", post(hybrid_search))
            .route("/api/v2/hybrid/index", post(hybrid_index)),
    )
}

/// Build the health probe router.
pub fn create_health_router() -> Router<RestAppState> {
    Router::new()
        .route("/health/live", get(liveness_check))
        .route("/health/ready", get(readiness_check))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{ApiCall, RecordingApiPort};
    use proximadb_proto::v1::{SqlArray, SqlObject, SqlRow, SqlRowField};
    use proximadb_runtime::HybridPort;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn sql_value(value: proximadb_proto::v1::sql_value::Value) -> SqlValue {
        SqlValue { value: Some(value) }
    }

    #[test]
    fn default_top_k_and_legacy_handler_stubs_construct() {
        assert_eq!(default_top_k(), 10);
        let _hybrid = HybridSearchHandler;
        let _progressive = ProgressiveSearchHandler::new();
        let _hybrid_router = create_hybrid_search_router();
    }

    #[test]
    fn rest_hybrid_request_preserves_filters_and_weighted_fusion_params() {
        use proximadb_proto::v1::fusion_strategy_params;

        let proto = build_hybrid_fusion_request(HybridSearchRestRequest {
            collection: "docs".to_string(),
            vector: Some(vec![0.1, 0.2]),
            text_query: Some("laptop".to_string()),
            top_k: 7,
            vector_weight: Some(0.8),
            rrf_k: None,
            fusion_strategy: Some("weighted_linear".to_string()),
            filters: Some(HashMap::from([(
                "tenant".to_string(),
                serde_json::json!("acme"),
            )])),
        })
        .unwrap();

        assert_eq!(proto.collection, "docs");
        assert_eq!(proto.text_query, "laptop");
        assert_eq!(proto.query_vector, vec![0.1, 0.2]);
        assert_eq!(proto.top_k, 7);
        assert_eq!(proto.fusion_strategy, FusionStrategy::WeightedLinear as i32);
        assert!(matches!(
            proto.fusion_params.and_then(|params| params.params),
            Some(fusion_strategy_params::Params::WeightedLinear(params))
                if (params.alpha - 0.2).abs() < 1e-6
                    && params.bm25_normalize
                    && params.vector_normalize
        ));
        assert!(matches!(
            proto.filters
                .get("tenant")
                .and_then(|value| value.kind.as_ref()),
            Some(prost_types::value::Kind::StringValue(value)) if value == "acme"
        ));
    }

    #[test]
    fn rest_hybrid_request_preserves_rrf_k() {
        use proximadb_proto::v1::fusion_strategy_params;

        let proto = build_hybrid_fusion_request(HybridSearchRestRequest {
            collection: "docs".to_string(),
            vector: None,
            text_query: Some("laptop".to_string()),
            top_k: 10,
            vector_weight: None,
            rrf_k: Some(17),
            fusion_strategy: Some("rrf".to_string()),
            filters: None,
        })
        .unwrap();

        assert_eq!(proto.fusion_strategy, FusionStrategy::Rrf as i32);
        assert!(matches!(
            proto.fusion_params.and_then(|params| params.params),
            Some(fusion_strategy_params::Params::RrfK(17))
        ));
    }

    #[derive(Default)]
    struct RecordingHybridPort {
        last_request: Mutex<Option<HybridFusionSearchRequest>>,
        last_tenant: Mutex<Option<String>>,
    }

    #[async_trait::async_trait]
    impl HybridPort for RecordingHybridPort {
        async fn hybrid_search(
            &self,
            request: HybridFusionSearchRequest,
            tenant_id: Option<String>,
        ) -> anyhow::Result<proximadb_proto::v1::HybridFusionSearchResponse> {
            *self.last_request.lock().unwrap() = Some(request);
            *self.last_tenant.lock().unwrap() = tenant_id;
            Ok(proximadb_proto::v1::HybridFusionSearchResponse {
                results_count: 1,
                fusion_strategy: FusionStrategy::WeightedLinear as i32,
                ..Default::default()
            })
        }

        async fn list_fusion_strategies(
            &self,
            _request: proximadb_proto::v1::ListFusionStrategiesRequest,
        ) -> anyhow::Result<proximadb_proto::v1::ListFusionStrategiesResponse> {
            Ok(Default::default())
        }
    }

    #[tokio::test]
    async fn rest_hybrid_handler_forwards_filters_and_fusion_to_port() {
        use proximadb_proto::v1::fusion_strategy_params;

        let port = Arc::new(RecordingHybridPort::default());
        let state = HybridRestState {
            hybrid_port: port.clone(),
            bm25_port: None,
        };

        let response = hybrid_search(
            State(state),
            Some(Extension(TenantContext::new("demo1"))),
            Json(HybridSearchRestRequest {
                collection: "docs".to_string(),
                vector: Some(vec![0.1, 0.2]),
                text_query: Some("alpha".to_string()),
                top_k: 5,
                vector_weight: Some(0.75),
                rrf_k: None,
                fusion_strategy: Some("weighted_linear".to_string()),
                filters: Some(HashMap::from([(
                    "region".to_string(),
                    serde_json::json!("us"),
                )])),
            }),
        )
        .await
        .unwrap();

        assert_eq!(response.0["total"], serde_json::json!(1));
        assert_eq!(
            port.last_tenant.lock().unwrap().as_deref(),
            Some("demo1"),
            "handler must forward the middleware tenant to the port (#949)"
        );
        let captured = port
            .last_request
            .lock()
            .unwrap()
            .clone()
            .expect("hybrid handler should call port");
        assert_eq!(captured.collection, "docs");
        assert_eq!(captured.text_query, "alpha");
        assert_eq!(captured.query_vector, vec![0.1, 0.2]);
        assert_eq!(captured.top_k, 5);
        assert_eq!(
            captured.fusion_strategy,
            FusionStrategy::WeightedLinear as i32
        );
        assert!(matches!(
            captured.fusion_params.and_then(|params| params.params),
            Some(fusion_strategy_params::Params::WeightedLinear(params))
                if (params.alpha - 0.25).abs() < 1e-6
        ));
        assert!(matches!(
            captured
                .filters
                .get("region")
                .and_then(|value| value.kind.as_ref()),
            Some(prost_types::value::Kind::StringValue(value)) if value == "us"
        ));
    }

    #[test]
    fn sql_value_to_json_preserves_all_wire_value_shapes() {
        use proximadb_proto::v1::sql_value::Value;

        assert_eq!(
            sql_value_to_json(&sql_value(Value::StringValue("doc".to_string()))),
            serde_json::json!("doc")
        );
        assert_eq!(
            sql_value_to_json(&sql_value(Value::NumberValue(1.5))),
            serde_json::json!(1.5)
        );
        assert_eq!(
            sql_value_to_json(&sql_value(Value::BoolValue(true))),
            serde_json::json!(true)
        );
        assert_eq!(
            sql_value_to_json(&sql_value(Value::Int64Value(42))),
            serde_json::json!(42)
        );
        // Round: canonical rendering — bytes are base64 (was int-array).
        assert_eq!(
            sql_value_to_json(&sql_value(Value::BytesValue(vec![1, 2]))),
            serde_json::json!("AQI=")
        );
        assert_eq!(
            sql_value_to_json(&sql_value(Value::NullValue(0))),
            serde_json::Value::Null
        );
        assert_eq!(
            sql_value_to_json(&sql_value(Value::ArrayValue(SqlArray {
                values: vec![sql_value(Value::StringValue("nested".to_string()))],
            }))),
            serde_json::json!(["nested"])
        );

        let mut fields = HashMap::new();
        fields.insert("k".to_string(), sql_value(Value::Int64Value(7)));
        assert_eq!(
            sql_value_to_json(&sql_value(Value::ObjectValue(SqlObject { fields }))),
            serde_json::json!({"k": 7})
        );
        assert_eq!(
            sql_value_to_json(&SqlValue { value: None }),
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn execute_sql_validates_empty_query_before_port_call() {
        let port = RecordingApiPort::new();

        let err = execute_sql(
            State(RestAppState::new(port.clone())),
            None,
            Json(SqlQueryRequest {
                query: "   ".to_string(),
                parameters: None,
                collection: None,
                timeout_ms: None,
                seeding: None,
            }),
        )
        .await
        .unwrap_err();

        assert!(matches!(err, RestError::InvalidArgument(_)));
        assert!(port.calls().is_empty());
    }

    #[tokio::test]
    async fn execute_sql_routes_to_port_and_shapes_json_rows() {
        use proximadb_proto::v1::sql_value::Value;

        let port = RecordingApiPort::new();
        *port.sql_response.lock().unwrap() = proximadb_proto::v1::ExecuteQueryResponse {
            rows: vec![SqlRow {
                fields: vec![SqlRowField {
                    key: "answer".to_string(),
                    value: Some(sql_value(Value::Int64Value(42))),
                }],
                similarity: None,
            }],
            rows_scanned: 10,
            rows_returned: 1,
            execution_time_ms: 99,
            columns: vec!["answer".to_string()],
            column_types: vec!["INT64".to_string()],
        };

        let Json(body) = execute_sql(
            State(RestAppState::new(port.clone())),
            Some(axum::Extension(proximadb_tenant::ResolvedRequestIdentity {
                tenant: "tenant-a".to_string(),
                subject: Some("alice".to_string()),
                auth_class: proximadb_tenant::AuthClass::Authenticated,
                tenant_stable_id: Some(7),
            })),
            Json(SqlQueryRequest {
                query: "select answer from docs".to_string(),
                parameters: Some(vec![sql_value(Value::StringValue("doc-1".to_string()))]),
                collection: Some("docs".to_string()),
                timeout_ms: Some(1000),
                seeding: Some("average".to_string()),
            }),
        )
        .await
        .unwrap();

        assert_eq!(body["rows"][0]["answer"], 42);
        assert_eq!(body["columns"], serde_json::json!(["answer"]));
        assert_eq!(body["rows_returned"], 1);
        assert_eq!(
            port.calls(),
            vec![ApiCall::Sql {
                query: "-- SEEDING: AVERAGE\nselect answer from docs".to_string(),
                parameter_count: Some(1),
                collection: Some("docs".to_string()),
                // TD-ABAC-9: the foundation identity Extension reaches the port.
                tenant_id: Some("tenant-a".to_string()),
                subject: Some("alice".to_string()),
                tenant_stable_id: Some(7),
                auth_class: proximadb_tenant::AuthClass::Authenticated,
            }]
        );
    }

    #[tokio::test]
    async fn health_routes_return_success_and_routers_construct() {
        let live = liveness_check().await.into_response();
        assert_eq!(live.status(), StatusCode::OK);

        let ready = readiness_check(State(RestAppState::new(RecordingApiPort::new())))
            .await
            .into_response();
        assert_eq!(ready.status(), StatusCode::OK);

        let _health_router = create_health_router();
    }
}
