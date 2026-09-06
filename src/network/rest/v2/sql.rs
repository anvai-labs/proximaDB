//! Canonical SQL-over-REST transport adapter.
//!
//! This module owns no SQL parser, planner, catalog, or executor. It validates
//! the single-statement HTTP contract and delegates to the same
//! canonical typed SQL authority also used by authenticated gRPC. Values stay
//! as `ProximaValue` until this module projects them to JSON at the HTTP edge.

use std::time::Duration;

use axum::{Extension, Json, extract::State};
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::errors::{ApiError, ApiResult};
use crate::network::rest::canonical::handlers::AppState;

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 300_000;
const MAX_QUERY_BYTES: usize = 1024 * 1024;

/// One SQL statement executed through the shared SQL authority.
///
/// Parameter binding is intentionally not advertised yet: the relational
/// execution port currently has no typed binding contract. Callers must not be
/// offered a field that an adapter would silently ignore or interpolate.
#[derive(Debug, Deserialize, ToSchema)]
pub struct SqlRequest {
    #[schema(min_length = 1, max_length = 1048576)]
    pub query: String,
    /// Optional collection context used by vector/graph SQL extensions.
    pub collection: Option<String>,
    /// Per-request deadline. Defaults to 30 seconds and is capped at 5 minutes.
    #[schema(minimum = 1, maximum = 300000)]
    pub timeout_ms: Option<u64>,
}

/// Stable JSON envelope for SQL reads and writes.
///
/// For reads, `rows_returned` is the row count. For DDL/DML, the shared SQL
/// authority reports the affected count in `rows_returned` and `rows` is empty.
#[derive(Debug, Serialize, ToSchema)]
pub struct SqlResponse {
    #[schema(value_type = Vec<Object>)]
    pub rows: Vec<serde_json::Value>,
    pub columns: Vec<String>,
    pub column_types: Vec<String>,
    pub rows_returned: u64,
    pub rows_scanned: u64,
    pub execution_time_ms: u64,
    pub request_id: String,
}

fn validate_request(request: &SqlRequest) -> ApiResult<u64> {
    if request.query.trim().is_empty() {
        return Err(ApiError::InvalidArgument(
            "SQL query cannot be empty".to_string(),
        ));
    }
    if request.query.len() > MAX_QUERY_BYTES {
        return Err(ApiError::InvalidArgument(format!(
            "SQL query exceeds the {MAX_QUERY_BYTES}-byte limit"
        )));
    }
    crate::query::sql_frontend::validate_single_statement(&request.query)
        .map_err(|error| ApiError::InvalidArgument(error.to_string()))?;

    let timeout_ms = request.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
    if timeout_ms == 0 || timeout_ms > MAX_TIMEOUT_MS {
        return Err(ApiError::InvalidArgument(format!(
            "timeout_ms must be between 1 and {MAX_TIMEOUT_MS}"
        )));
    }
    Ok(timeout_ms)
}

fn map_execution_error(error: anyhow::Error) -> ApiError {
    if let Some((resource, holder)) = crate::errors::extract_dml_lock_conflict(&error) {
        let message = match holder {
            Some(holder) => format!("resource {resource} is held by {holder}"),
            None => format!("resource {resource} is locked"),
        };
        ApiError::LockConflict(message)
    } else {
        ApiError::Internal(format!("SQL execution failed: {error}"))
    }
}

/// Execute one SQL statement through the canonical shared SQL authority.
///
/// The required foundation identity is inserted by the root tenant middleware
/// after authentication. A router mounted without that middleware fails closed
/// at extraction instead of constructing an anonymous authorization carrier.
#[utoipa::path(
    post,
    path = "/api/v2/sql",
    tag = "SQL",
    operation_id = "executeSql",
    summary = "Execute one authenticated SQL statement.",
    request_body = SqlRequest,
    responses(
        (status = 200, description = "SQL result or write count.", body = SqlResponse),
        (status = 400, description = "Invalid SQL request.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 408, description = "Query deadline exceeded.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 409, description = "Write lock conflict.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 500, description = "SQL execution failure.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn execute_sql(
    State(state): State<AppState>,
    Extension(identity): Extension<proximadb_tenant::ResolvedRequestIdentity>,
    Json(request): Json<SqlRequest>,
) -> ApiResult<Json<SqlResponse>> {
    let timeout_ms = validate_request(&request)?;
    let request_id = Uuid::new_v4().to_string();
    info!(
        %request_id,
        tenant = %identity.tenant,
        subject = ?identity.subject,
        "executing canonical REST SQL request"
    );

    let execution = state.api_handlers.execute_sql(
        request.query,
        None,
        request.collection,
        proximadb_runtime::PortIdentity::from(&identity),
    );
    let response = match tokio::time::timeout(Duration::from_millis(timeout_ms), execution).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            error!(%request_id, %error, "REST SQL execution failed");
            return Err(map_execution_error(error));
        }
        Err(_) => {
            error!(%request_id, timeout_ms, "REST SQL execution timed out");
            return Err(ApiError::DeadlineExceeded(format!(
                "SQL execution exceeded {timeout_ms} ms"
            )));
        }
    };

    let rows_returned = response.rows_affected.unwrap_or(response.rows.len() as u64);
    let rows = response
        .rows
        .iter()
        .map(|row| {
            let mut object = serde_json::Map::new();
            for (index, value) in row.iter().enumerate() {
                let column = response
                    .columns
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("column_{index}"));
                // v2's binary spelling is the per-byte int-array (read AND
                // write agree: /api/v2/records parses only arrays for
                // binary, so any other spelling breaks the read→write round
                // trip inside one API version).
                object.insert(
                    column,
                    proximadb_records::conversions::proxima_to_json(value),
                );
            }
            serde_json::Value::Object(object)
        })
        .collect();

    Ok(Json(SqlResponse {
        rows,
        columns: response.columns,
        column_types: response.column_types,
        rows_returned,
        rows_scanned: response.rows_scanned,
        execution_time_ms: response.execution_time_ms,
        request_id,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_single_statement_and_deadline_bounds() {
        let valid = SqlRequest {
            query: "SELECT 1".to_string(),
            collection: None,
            timeout_ms: None,
        };
        assert_eq!(validate_request(&valid).unwrap(), DEFAULT_TIMEOUT_MS);

        let multiple = SqlRequest {
            query: "SELECT 1; SELECT 2".to_string(),
            collection: None,
            timeout_ms: None,
        };
        assert!(matches!(
            validate_request(&multiple),
            Err(ApiError::InvalidArgument(message))
                if message.contains("Exactly one SQL statement")
        ));

        let zero_timeout = SqlRequest {
            query: "SELECT 1".to_string(),
            collection: None,
            timeout_ms: Some(0),
        };
        assert!(matches!(
            validate_request(&zero_timeout),
            Err(ApiError::InvalidArgument(_))
        ));
    }
}
