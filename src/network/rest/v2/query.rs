//! OpenAPI-facing v2 query endpoints for AQL/UQL.
//!
//! These handlers are protocol facades only. They validate the REST contract and
//! lower textual AQL/UQL into the shared `UnifiedQueryPort`. Plain SQL uses the
//! distinct `/api/v2/sql` transport adapter so language contracts do not blur.

use axum::{Extension, Json, extract::State};
use proximadb_data_model::ProximaValue;
use proximadb_records::conversions::json_to_proxima;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use utoipa::ToSchema;

use crate::errors::{ApiError, ApiResult};
use crate::network::rest::canonical::handlers::AppState;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum QueryLanguage {
    Uql,
    Aql,
    Federated,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryRequest {
    pub language: QueryLanguage,
    #[schema(min_length = 1)]
    pub query: String,
    #[serde(default)]
    #[schema(value_type = Option<Vec<Object>>)]
    pub parameters: Option<Vec<serde_json::Value>>,
    pub collection: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ExplainQueryRequest {
    pub language: QueryLanguage,
    #[schema(min_length = 1)]
    pub query: String,
    pub collection: Option<String>,
}

fn validate_query(language: QueryLanguage, query: &str) -> ApiResult<()> {
    if query.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "query cannot be empty".to_string(),
        ));
    }
    debug!(
        "v2 {:?} query: {}",
        language,
        query.chars().take(120).collect::<String>()
    );
    Ok(())
}

/// Lower JSON query parameters to canonical `ProximaValue` via the shared
/// `proximadb_records::conversions::json_to_proxima` (TD-109).
fn json_to_proxima_values(params: Option<Vec<serde_json::Value>>) -> Option<Vec<ProximaValue>> {
    params.map(|values| values.iter().map(json_to_proxima).collect())
}

/// POST /api/v2/query
///
/// The canonical unified query surface: UQL (unified multi-modal), federated SQL
/// extensions, and AQL. This is NOT plain SQL-over-REST and is NOT deprecated:
/// UQL/AQL are non-SQL languages that pgwire and `/api/v2/sql` cannot serve, so
/// this endpoint remains their canonical home. Authenticated programmatic SQL
/// is available through both gRPC `ExecuteQuery` and `/api/v2/sql`; neither is a
/// replacement for this UQL/federated surface.
#[utoipa::path(
    post,
    path = "/api/v2/query",
    tag = "Query",
    operation_id = "executeQuery",
    summary = "Execute AQL or UQL through the shared query facade.",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query result.", body = crate::network::rest::openapi::QueryResponse),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn execute_query(
    State(state): State<AppState>,
    Extension(identity): Extension<proximadb_tenant::ResolvedRequestIdentity>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_query(request.language, &request.query)?;
    let query_port = state
        .unified_query_port
        .as_ref()
        .ok_or_else(|| ApiError::Internal("Unified query port is not configured".to_string()))?;

    // ADR-098 D5 / TD-SPINE-6: the port carries the caller's identity as data
    // (same fail-closed extraction contract as /api/v2/sql — the tenant
    // middleware inserts the foundation identity; a router mounted without it
    // fails closed here instead of constructing an anonymous carrier).
    let context = proximadb_runtime::UnifiedQueryPortContext::for_identity(
        proximadb_runtime::PortIdentity::from(&identity),
    );

    // L4 (round 2): this surface now runs relational reads through the SQL
    // authority, so it carries /api/v2/sql's deadline contract.
    const PORT_QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    let execution = async {
        match request.language {
            QueryLanguage::Uql => {
                query_port
                    .execute_unified_query(
                        request.query,
                        json_to_proxima_values(request.parameters),
                        request.collection,
                        request.limit,
                        context,
                    )
                    .await
            }
            QueryLanguage::Federated => {
                query_port
                    .execute_federated_query(
                        request.query,
                        json_to_proxima_values(request.parameters),
                        context,
                    )
                    .await
            }
            QueryLanguage::Aql => Err(proximadb_runtime::InvalidQueryInput(
                "AQL text execution is not yet exposed on /api/v2/query; submit UQL or federated SQL extensions".to_string(),
            )
            .into()),
        }
    };
    let result = match tokio::time::timeout(PORT_QUERY_TIMEOUT, execution).await {
        Ok(result) => result,
        Err(_) => {
            return Err(ApiError::DeadlineExceeded(format!(
                "Query exceeded the {PORT_QUERY_TIMEOUT:?} deadline"
            )));
        }
    };

    result.map(Json).map_err(|error| {
        // A query addressing a collection that does not resolve for the
        // request (missing, or not resolvable by its tenant) is 404.
        if let Some(missing) = crate::services::operations::vectors::collection_not_found(&error) {
            return ApiError::CollectionNotFound(missing.identifier.clone());
        }
        // MED-4/MED-3 (round-1 review): client-shaped port rejections map to
        // 4xx — a read-only-policy violation (DDL/DML on this query surface)
        // is a 400 naming the policy (unsupported AQL is a 400 too); a missing
        // identity is a 401. L5 (round 2): no write-surface suffix is
        // appended here — the port's policy rejection already names
        // /api/v2/sql, and a READ that merely failed to parse must not be
        // told to use the write surface.
        if error
            .downcast_ref::<proximadb_runtime::InvalidQueryInput>()
            .is_some()
        {
            return ApiError::InvalidArgument(format!("Query rejected: {error}"));
        }
        if error
            .downcast_ref::<proximadb_runtime::MissingQueryIdentity>()
            .is_some()
        {
            return ApiError::Unauthorized(format!("Query rejected: {error}"));
        }
        error!("v2 query execution failed: {}", error);
        ApiError::Internal(format!("Query execution failed: {}", error))
    })
}

/// POST /api/v2/query/explain
#[utoipa::path(
    post,
    path = "/api/v2/query/explain",
    tag = "Query",
    operation_id = "explainQuery",
    summary = "Explain an AQL or UQL query through the shared query facade.",
    request_body = ExplainQueryRequest,
    responses(
        (status = 200, description = "Query plan and lowering details.", body = crate::network::rest::openapi::QueryResponse),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn explain_query(
    State(state): State<AppState>,
    Extension(identity): Extension<proximadb_tenant::ResolvedRequestIdentity>,
    Json(request): Json<ExplainQueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_query(request.language, &request.query)?;
    let query_port = state
        .unified_query_port
        .as_ref()
        .ok_or_else(|| ApiError::Internal("Unified query port is not configured".to_string()))?;

    query_port
        .explain_unified_query(
            request.query,
            request.collection,
            proximadb_runtime::UnifiedQueryPortContext::for_identity(
                proximadb_runtime::PortIdentity::from(&identity),
            ),
        )
        .await
        .map(Json)
        .map_err(|error| {
            error!("v2 query explain failed: {}", error);
            ApiError::Internal(format!("Query explain failed: {}", error))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_uql_request() {
        let request: QueryRequest = serde_json::from_value(serde_json::json!({
            "language": "uql",
            "query": "SEARCH products RETURN id",
            "parameters": ["acme", 42],
            "collection": "products",
            "limit": 10
        }))
        .expect("request should parse");

        assert_eq!(request.language, QueryLanguage::Uql);
        assert_eq!(request.limit, Some(10));
        assert!(json_to_proxima_values(request.parameters).is_some());
    }

    #[test]
    fn parses_aql_request() {
        let request: QueryRequest = serde_json::from_value(serde_json::json!({
            "language": "aql",
            "query": "FIND related entities"
        }))
        .expect("request should parse");

        assert_eq!(request.language, QueryLanguage::Aql);
        assert!(request.collection.is_none());
    }

    #[test]
    fn parses_federated_request() {
        let request: QueryRequest = serde_json::from_value(serde_json::json!({
            "language": "federated",
            "query": "SELECT * FROM VECTOR_SEARCH('items', '[0.1,0.2]', 10)"
        }))
        .expect("request should parse");

        assert_eq!(request.language, QueryLanguage::Federated);
    }
}
