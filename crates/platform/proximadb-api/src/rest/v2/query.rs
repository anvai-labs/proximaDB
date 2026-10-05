//! REST API v2 — query surface. Moved from the root v2 tree under ADR-094.

//! OpenAPI-facing v2 query endpoints for AQL/UQL.
//!
//! These handlers are protocol facades only. They validate the REST contract and
//! lower textual AQL/UQL into the shared `UnifiedQueryPort`. Plain SQL uses the
//! distinct `/api/v2/sql` transport adapter so language contracts do not blur.

use axum::{Json, extract::State};
use proximadb_data_model::ProximaValue;
use proximadb_records::conversions::json_to_proxima;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use utoipa::ToSchema;

use crate::rest::errors::{RestError as ApiError, RestResult as ApiResult};
use crate::rest::state::RestAppState;

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
        (status = 200, description = "Query result.", body = crate::rest::errors::QueryResponse),
        (status = 400, description = "Invalid request.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn execute_query(
    State(state): State<RestAppState>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_query(request.language, &request.query)?;
    let query_port = state
        .unified_query_port
        .as_ref()
        .ok_or_else(|| ApiError::Internal("Unified query port is not configured".to_string()))?;

    let result = match request.language {
        QueryLanguage::Uql => {
            query_port
                .execute_unified_query(
                    request.query,
                    json_to_proxima_values(request.parameters),
                    request.collection,
                    request.limit,
                )
                .await
        }
        QueryLanguage::Federated => {
            query_port
                .execute_federated_query(request.query, json_to_proxima_values(request.parameters))
                .await
        }
        QueryLanguage::Aql => {
            return Err(ApiError::InvalidArgument(
                "AQL text execution is not yet exposed on /api/v2/query; submit UQL or federated SQL extensions".to_string(),
            ));
        }
    };

    result.map(Json).map_err(|error| {
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
        (status = 200, description = "Query plan and lowering details.", body = crate::rest::errors::QueryResponse),
        (status = 400, description = "Invalid request.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn explain_query(
    State(state): State<RestAppState>,
    Json(request): Json<ExplainQueryRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    validate_query(request.language, &request.query)?;
    let query_port = state
        .unified_query_port
        .as_ref()
        .ok_or_else(|| ApiError::Internal("Unified query port is not configured".to_string()))?;

    query_port
        .explain_unified_query(request.query, request.collection)
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
