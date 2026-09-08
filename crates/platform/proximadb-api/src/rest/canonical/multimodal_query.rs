//! # Unified Multi-Model Query Handlers
//!
//! REST endpoints for cross-model queries combining vector, graph, document,
//! and observability data.  All handlers delegate to `UnifiedQueryPort`.
//!
//! ## Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/api/v2/unified/execute`        | POST   | Unified SQL-like query |
//! | `/api/v2/unified/multi-model`    | POST   | Structured multi-model query |
//! | `/api/v2/unified/federated`      | POST   | Federated SQL with extensions |
//! | `/api/v2/unified/distributed`    | POST   | Distributed cross-shard query |
//! | `/api/v2/unified/explain`        | POST   | Explain query plan |
//! | `/api/v2/unified/prepare`        | POST   | Cache a prepared statement |
//! | `/api/v2/unified/execute/{id}`   | POST   | Execute a prepared statement |
//! | `/api/v2/unified/prepared/{id}`  | DELETE | Delete a prepared statement |
//! | `/api/v2/unified/prepared/stats` | POST   | Prepared-statement statistics |
//!
//! The default server wires the root-crate `UnifiedQueryPort` implementation.
//! A 501 remains possible only for alternate constructions whose port reports
//! that an operation is not implemented.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, post},
};
use proximadb_data_model::ProximaValue;
use proximadb_runtime::{InvalidQueryInput, UnifiedQueryPort};
use serde::Deserialize;
use tracing::{error, info};

// ── State ─────────────────────────────────────────────────────────────────────

/// Axum state for unified multi-model query REST handlers.
#[derive(Clone)]
pub struct UnifiedQueryRestState {
    pub unified_query_port: Arc<dyn UnifiedQueryPort>,
}

// ── Request types ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ExecuteQueryRequest {
    query: String,
    #[serde(default)]
    parameters: Option<Vec<serde_json::Value>>,
    collection: Option<String>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ExecuteFederatedRequest {
    query: String,
    #[serde(default)]
    parameters: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct PrepareStatementRequest {
    query: String,
    name: Option<String>,
    #[serde(default)]
    cache_results: bool,
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ExecutePreparedRequest {
    #[serde(default)]
    parameters: Option<Vec<serde_json::Value>>,
    collection: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PreparedStatsRequest {
    #[serde(default)]
    statement_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExplainQueryRequest {
    query: String,
    collection: Option<String>,
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn json_to_proxima_values(params: Option<Vec<serde_json::Value>>) -> Option<Vec<ProximaValue>> {
    params.map(|ps| ps.into_iter().map(json_to_proxima_value).collect())
}

fn json_to_proxima_value(value: serde_json::Value) -> ProximaValue {
    match value {
        serde_json::Value::String(s) => ProximaValue::String(s),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(ProximaValue::Int64)
            .or_else(|| n.as_u64().map(ProximaValue::UInt64))
            .or_else(|| n.as_f64().map(ProximaValue::Float64))
            .unwrap_or(ProximaValue::Null),
        serde_json::Value::Bool(b) => ProximaValue::Boolean(b),
        serde_json::Value::Null => ProximaValue::Null,
        serde_json::Value::Array(values) => {
            ProximaValue::Array(values.into_iter().map(json_to_proxima_value).collect())
        }
        serde_json::Value::Object(fields) => ProximaValue::Map(
            fields
                .into_iter()
                .map(|(key, value)| (key, json_to_proxima_value(value)))
                .collect(),
        ),
    }
}

fn not_implemented(description: &str) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": format!(
                "{description} requires QueryFacadeAdapter/UnifiedHandlers to be extracted \
                 from the root crate (Phase 9.9/9.10). Use the root-crate server endpoint \
                 /api/v1/unified/* in the meantime."
            )
        })),
    )
}

fn extract_json<T>(
    body: Result<Json<T>, JsonRejection>,
) -> Result<T, (StatusCode, Json<serde_json::Value>)> {
    body.map(|Json(value)| value).map_err(|rejection| {
        (
            rejection.status(),
            Json(serde_json::json!({ "error": rejection.body_text() })),
        )
    })
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn execute_query(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<ExecuteQueryRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    if req.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "query cannot be empty" })),
        )
            .into_response();
    }

    info!(
        "Unified query: {}",
        req.query.chars().take(100).collect::<String>()
    );

    let params = json_to_proxima_values(req.parameters);
    match s
        .unified_query_port
        .execute_unified_query(req.query, params, req.collection, req.limit)
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("execute_unified_query").into_response()
            } else {
                error!("Unified query failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn execute_multi_model_query(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    match s.unified_query_port.execute_multi_model_query(req).await {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            // Downcast FIRST: InvalidQueryInput embeds user-controlled
            // component_type text — a client error containing the words
            // 'not implemented' misrouted to 501.
            if e.downcast_ref::<InvalidQueryInput>().is_some() {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            } else if e.to_string().contains("not implemented")
                || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("execute_multi_model_query").into_response()
            } else {
                error!("Multi-model query failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn execute_federated_query(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<ExecuteFederatedRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    if req.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "query cannot be empty" })),
        )
            .into_response();
    }

    let params = json_to_proxima_values(req.parameters);
    match s
        .unified_query_port
        .execute_federated_query(req.query, params)
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("execute_federated_query").into_response()
            } else {
                error!("Federated query failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn execute_distributed_query(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    match s.unified_query_port.execute_distributed_query(req).await {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("execute_distributed_query").into_response()
            } else {
                error!("Distributed query failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn explain_query(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<ExplainQueryRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    match s
        .unified_query_port
        .explain_unified_query(req.query, req.collection)
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("explain_unified_query").into_response()
            } else {
                error!("Explain query failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn prepare_statement(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<PrepareStatementRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    match s
        .unified_query_port
        .prepare_statement(req.name, req.query, req.cache_results, req.ttl_seconds)
        .await
    {
        Ok(statement_id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({ "statement_id": statement_id })),
        )
            .into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("prepare_statement").into_response()
            } else {
                error!("Prepare statement failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn execute_prepared_statement(
    State(s): State<UnifiedQueryRestState>,
    Path(statement_id): Path<String>,
    body: Result<Json<ExecutePreparedRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    let params = json_to_proxima_values(req.parameters);
    match s
        .unified_query_port
        .execute_prepared(statement_id, params, req.collection)
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("execute_prepared").into_response()
            } else {
                error!("Execute prepared statement failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn delete_prepared_statement(
    State(s): State<UnifiedQueryRestState>,
    Path(statement_id): Path<String>,
) -> impl IntoResponse {
    match s.unified_query_port.delete_prepared(statement_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("delete_prepared").into_response()
            } else {
                error!("Delete prepared statement failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

async fn get_prepared_stats(
    State(s): State<UnifiedQueryRestState>,
    body: Result<Json<PreparedStatsRequest>, JsonRejection>,
) -> impl IntoResponse {
    let req = match extract_json(body) {
        Ok(req) => req,
        Err(response) => return response.into_response(),
    };
    match s
        .unified_query_port
        .get_prepared_stats(req.statement_ids)
        .await
    {
        Ok(result) => Json(result).into_response(),
        Err(e) => {
            if e.to_string().contains("not implemented") || e.to_string().contains("UNIMPLEMENTED")
            {
                not_implemented("get_prepared_stats").into_response()
            } else {
                error!("Get prepared stats failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response()
            }
        }
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Build the unified multi-model query router.
///
/// All routes delegate to `UnifiedQueryPort`.  The root-crate implementation
/// returns `501 Not Implemented` until Phase 9.9/9.10 is complete.
pub fn create_multimodal_router() -> Router<UnifiedQueryRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            .route("/execute", post(execute_query))
            .route("/multi-model", post(execute_multi_model_query))
            .route("/federated", post(execute_federated_query))
            .route("/distributed", post(execute_distributed_query))
            .route("/explain", post(explain_query))
            .route("/prepare", post(prepare_statement))
            .route("/execute/{statement_id}", post(execute_prepared_statement))
            .route(
                "/prepared/{statement_id}",
                delete(delete_prepared_statement),
            )
            .route("/prepared/stats", post(get_prepared_stats)),
    )
}

// `create_explain_router` (legacy `/api/v1/sql/explain`) was removed in the
// API standardization hard-rename; SQL explain is served by the canonical
// `/api/v2/query/explain` (create_v2_router).

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use async_trait::async_trait;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, header::CONTENT_TYPE},
    };
    use tower::ServiceExt;

    #[derive(Clone, Copy)]
    enum MockMode {
        Ok,
        NotImplemented,
        InvalidInput,
        Internal,
    }

    struct MockUnifiedQueryPort {
        mode: MockMode,
    }

    impl MockUnifiedQueryPort {
        fn state(mode: MockMode) -> State<UnifiedQueryRestState> {
            State(UnifiedQueryRestState {
                unified_query_port: Arc::new(Self { mode }),
            })
        }

        fn result(&self, op: &str) -> Result<serde_json::Value> {
            match self.mode {
                MockMode::Ok => Ok(serde_json::json!({ "op": op })),
                MockMode::NotImplemented => Err(anyhow!("{op} not implemented")),
                // Kept (the handler lacks the InvalidQueryInput downcast —
                // these endpoints surface client errors as 500 today).
                MockMode::InvalidInput => {
                    Err(InvalidQueryInput(format!("{op} input is invalid")).into())
                }
                MockMode::Internal => Err(anyhow!("{op} failed")),
            }
        }
    }

    #[async_trait]
    impl UnifiedQueryPort for MockUnifiedQueryPort {
        async fn execute_unified_query(
            &self,
            _query: String,
            _parameters: Option<Vec<ProximaValue>>,
            _collection: Option<String>,
            _limit: Option<u32>,
        ) -> Result<serde_json::Value> {
            self.result("execute_unified_query")
        }

        async fn execute_multi_model_query(
            &self,
            _request: serde_json::Value,
        ) -> Result<serde_json::Value> {
            self.result("execute_multi_model_query")
        }

        async fn execute_federated_query(
            &self,
            _query: String,
            _parameters: Option<Vec<ProximaValue>>,
        ) -> Result<serde_json::Value> {
            self.result("execute_federated_query")
        }

        async fn execute_distributed_query(
            &self,
            _request: serde_json::Value,
        ) -> Result<serde_json::Value> {
            self.result("execute_distributed_query")
        }

        async fn explain_unified_query(
            &self,
            _query: String,
            _collection: Option<String>,
        ) -> Result<serde_json::Value> {
            self.result("explain_unified_query")
        }

        async fn prepare_statement(
            &self,
            _name: Option<String>,
            _query: String,
            _cache_results: bool,
            _ttl_seconds: Option<u64>,
        ) -> Result<String> {
            match self.mode {
                MockMode::Ok => Ok("stmt-1".to_string()),
                MockMode::NotImplemented => Err(anyhow!("prepare_statement not implemented")),
                MockMode::InvalidInput => {
                    Err(InvalidQueryInput("prepare_statement input is invalid".to_string()).into())
                }
                MockMode::Internal => Err(anyhow!("prepare_statement failed")),
            }
        }

        async fn execute_prepared(
            &self,
            _statement_id: String,
            _parameters: Option<Vec<ProximaValue>>,
            _collection: Option<String>,
        ) -> Result<serde_json::Value> {
            self.result("execute_prepared")
        }

        async fn delete_prepared(&self, _statement_id: String) -> Result<()> {
            match self.mode {
                MockMode::Ok => Ok(()),
                MockMode::NotImplemented => Err(anyhow!("delete_prepared not implemented")),
                MockMode::InvalidInput => {
                    Err(InvalidQueryInput("delete_prepared input is invalid".to_string()).into())
                }
                MockMode::Internal => Err(anyhow!("delete_prepared failed")),
            }
        }

        async fn get_prepared_stats(
            &self,
            _statement_ids: Vec<String>,
        ) -> Result<serde_json::Value> {
            self.result("get_prepared_stats")
        }
    }

    #[test]
    fn test_json_to_proxima_values_none() {
        assert!(json_to_proxima_values(None).is_none());
    }

    #[test]
    fn test_json_to_proxima_values_primitives() {
        let params = vec![
            serde_json::json!("hello"),
            serde_json::json!(42i64),
            serde_json::json!(true),
        ];
        let result = json_to_proxima_values(Some(params)).unwrap();
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0], ProximaValue::String(ref value) if value == "hello"));
        assert!(matches!(result[1], ProximaValue::Int64(42)));
        assert!(matches!(result[2], ProximaValue::Boolean(true)));
    }

    #[test]
    fn test_json_to_proxima_values_preserves_composites() {
        let params = vec![serde_json::json!({
            "tags": ["a", "b"],
            "score": 9.5,
        })];
        let result = json_to_proxima_values(Some(params)).unwrap();
        assert!(matches!(result[0], ProximaValue::Map(_)));
    }

    #[tokio::test]
    async fn handlers_validate_empty_queries_before_delegating() {
        let execute = execute_query(
            MockUnifiedQueryPort::state(MockMode::Ok),
            Ok(Json(ExecuteQueryRequest {
                query: " ".to_string(),
                parameters: None,
                collection: None,
                limit: None,
            })),
        )
        .await
        .into_response();
        assert_eq!(execute.status(), StatusCode::BAD_REQUEST);

        let federated = execute_federated_query(
            MockUnifiedQueryPort::state(MockMode::Ok),
            Ok(Json(ExecuteFederatedRequest {
                query: "\n".to_string(),
                parameters: None,
            })),
        )
        .await
        .into_response();
        assert_eq!(federated.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn handlers_return_success_for_all_unified_query_port_methods() {
        let state = MockUnifiedQueryPort::state(MockMode::Ok);

        assert_eq!(
            execute_query(
                state.clone(),
                Ok(Json(ExecuteQueryRequest {
                    query: "select * from docs".to_string(),
                    parameters: Some(vec![serde_json::json!("p1")]),
                    collection: Some("docs".to_string()),
                    limit: Some(10),
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            execute_multi_model_query(state.clone(), Ok(Json(serde_json::json!({"vector": {}}))),)
                .await
                .into_response()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            execute_federated_query(
                state.clone(),
                Ok(Json(ExecuteFederatedRequest {
                    query: "select * from docs".to_string(),
                    parameters: Some(vec![serde_json::json!(1)]),
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            execute_distributed_query(state.clone(), Ok(Json(serde_json::json!({"shards": []}))),)
                .await
                .into_response()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            explain_query(
                state.clone(),
                Ok(Json(ExplainQueryRequest {
                    query: "select * from docs".to_string(),
                    collection: Some("docs".to_string()),
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            prepare_statement(
                state.clone(),
                Ok(Json(PrepareStatementRequest {
                    query: "select * from docs".to_string(),
                    name: Some("q1".to_string()),
                    cache_results: true,
                    ttl_seconds: Some(60),
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::CREATED
        );
        assert_eq!(
            execute_prepared_statement(
                state.clone(),
                Path("stmt-1".to_string()),
                Ok(Json(ExecutePreparedRequest {
                    parameters: Some(vec![serde_json::json!(true)]),
                    collection: Some("docs".to_string()),
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_prepared_statement(state.clone(), Path("stmt-1".to_string()))
                .await
                .into_response()
                .status(),
            StatusCode::NO_CONTENT
        );
        assert_eq!(
            get_prepared_stats(
                state,
                Ok(Json(PreparedStatsRequest {
                    statement_ids: vec!["stmt-1".to_string()],
                })),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn handlers_map_not_implemented_and_internal_errors_to_expected_statuses() {
        let not_implemented = execute_multi_model_query(
            MockUnifiedQueryPort::state(MockMode::NotImplemented),
            Ok(Json(serde_json::json!({}))),
        )
        .await
        .into_response();
        assert_eq!(not_implemented.status(), StatusCode::NOT_IMPLEMENTED);

        let invalid_input = execute_multi_model_query(
            MockUnifiedQueryPort::state(MockMode::InvalidInput),
            Ok(Json(serde_json::json!({}))),
        )
        .await
        .into_response();
        assert_eq!(invalid_input.status(), StatusCode::BAD_REQUEST);

        let internal = execute_distributed_query(
            MockUnifiedQueryPort::state(MockMode::Internal),
            Ok(Json(serde_json::json!({}))),
        )
        .await
        .into_response();
        assert_eq!(internal.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn json_rejections_use_bare_error_envelope() {
        let router =
            create_multimodal_router().with_state(MockUnifiedQueryPort::state(MockMode::Ok).0);
        let cases = [
            ("application/json", "{\"query\":", StatusCode::BAD_REQUEST),
            (
                "text/plain",
                "{\"query\":\"select 1\"}",
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ),
            ("application/json", "{}", StatusCode::UNPROCESSABLE_ENTITY),
        ];

        for (content_type, body, expected_status) in cases {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/execute")
                        .header(CONTENT_TYPE, content_type)
                        .body(Body::from(body))
                        .expect("build request"),
                )
                .await
                .expect("execute request");

            assert_eq!(response.status(), expected_status);
            assert_eq!(
                response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok()),
                Some("application/json")
            );
            let bytes = to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("read response body");
            let body: serde_json::Value =
                serde_json::from_slice(&bytes).expect("decode response JSON");
            assert!(body.get("error").and_then(|value| value.as_str()).is_some());
        }
    }

    #[test]
    fn unified_query_routers_construct() {
        let _multimodal = create_multimodal_router();
    }
}
