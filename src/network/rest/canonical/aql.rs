//! AQL REST endpoints (TD-050 RUBICON).

use axum::{
    Router,
    extract::{Json, Path, State},
    response::Json as JsonResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

use crate::errors::{ApiError, ApiResult};
use crate::query::aql::{AqlQuery, AuditTrail, executor::AqlExecutor};

/// State for the AQL router.
#[derive(Clone)]
pub struct AqlApiState {
    pub executor: Arc<AqlExecutor>,
}

impl AqlApiState {
    pub fn new(executor: AqlExecutor) -> Self {
        Self {
            executor: Arc::new(executor),
        }
    }
}

/// Wire the AQL endpoints under a parent router.
pub fn create_router() -> Router<AqlApiState> {
    Router::new()
        .route("/execute", post(execute_aql))
        .route("/audit/{query_id}", get(get_audit_trail))
}

/// Request for AQL execution.
#[derive(Debug, Deserialize)]
pub struct AqlRequest {
    pub query: AqlQuery,
}

/// Response for AQL execution, including the audit trail.
#[derive(Debug, Serialize)]
pub struct AqlResponse {
    pub result: serde_json::Value, // Simplified result for REST
    pub audit_trail: AuditTrail,
}

/// Execute an AQL query and return results with an audit trail.
async fn execute_aql(
    State(state): State<AqlApiState>,
    tenant: Option<axum::Extension<crate::network::middleware::tenant::TenantContext>>,
    Json(request): Json<AqlRequest>,
) -> ApiResult<JsonResponse<AqlResponse>> {
    info!("Executing AQL query: {:?}", request.query);

    // Sources resolve collections within the request tenant (absent ⇒ the
    // default tenant, as for every request surface).
    let request_tenant = Some(
        tenant
            .map(|axum::Extension(t)| t.tenant_id)
            .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string()),
    );
    match state
        .executor
        .execute_for_tenant(request.query, request_tenant)
        .await
    {
        Ok((result, trail)) => {
            // Convert AqlResult rows to JSON for the response
            let json_rows = serde_json::to_value(result.rows)
                .map_err(|e| ApiError::Internal(format!("Failed to serialize rows: {}", e)))?;

            Ok(JsonResponse(AqlResponse {
                result: json_rows,
                audit_trail: trail,
            }))
        }
        // Client-argument errors (e.g. `top_k` above PROXIMADB_SEARCH_MAX_TOP_K)
        // are the caller's to fix: 400, not 500.
        Err(proximadb_kernel::error::ProximaDBError::InvalidInput(msg)) => {
            info!("AQL query rejected: {}", msg);
            Err(ApiError::InvalidArgument(msg))
        }
        Err(e) => {
            error!("AQL execution failed: {}", e);
            Err(ApiError::Internal(e.to_string()))
        }
    }
}

/// Retrieve an audit trail for a specific query.
async fn get_audit_trail(
    State(state): State<AqlApiState>,
    Path(query_id): Path<String>,
) -> ApiResult<JsonResponse<AuditTrail>> {
    info!("Retrieving audit trail for query: {}", query_id);

    match state.executor.get_audit_trail(&query_id).await {
        Ok(trail) => Ok(JsonResponse(trail)),
        Err(e) => {
            error!("Failed to retrieve audit trail for {}: {}", query_id, e);
            Err(ApiError::NotFound(format!(
                "Audit trail for query '{}' not found",
                query_id
            )))
        }
    }
}
