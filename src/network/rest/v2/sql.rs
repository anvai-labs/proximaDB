//! Canonical SQL-over-REST transport adapter.
//!
//! This module owns no SQL parser, planner, catalog, or executor. It validates
//! the single-statement HTTP contract and delegates to the same
//! canonical typed SQL authority also used by authenticated gRPC. Values stay
//! as `ProximaValue` until this module projects them to JSON at the HTTP edge.

use std::time::Duration;

use axum::{Extension, Json, extract::State};
use proximadb_data_model::ProximaValue;
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

/// v2 SQL's row rendering policy: `proxima_to_json` (whose Binary spelling
/// is already the per-byte int-array — the one form /api/v2/records' typed
/// write parses, keeping binary SELECT↔write symmetric) with ONE override
/// applied at ALL depths: Uuid DASHED, matching /api/v2/records' read
/// rendering. Temporals are NOT symmetric with the records read path's
/// typed {value, unit} objects — the SQL surface renders bare numbers
/// (pre-existing); aligning them is a deliberate contract change, not a
/// rendering cleanup.
fn v2_row_render(value: &ProximaValue) -> serde_json::Value {
    match value {
        ProximaValue::Uuid(u) => serde_json::Value::String(
            proximadb_kernel::uuid::Uuid::from_bytes(*u).to_hyphenated_string(),
        ),
        ProximaValue::Array(items) => {
            serde_json::Value::Array(items.iter().map(v2_row_render).collect())
        }
        ProximaValue::Map(fields) | ProximaValue::Struct(fields) => serde_json::Value::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), v2_row_render(v)))
                .collect(),
        ),
        other => proximadb_records::conversions::proxima_to_json(other),
    }
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
    } else if let Some(missing) = crate::services::operations::vectors::collection_not_found(&error)
    {
        // A collection that does not resolve for the request (missing, or not
        // resolvable by its tenant) is 404 — one response for both.
        ApiError::CollectionNotFound(missing.identifier.clone())
    } else if let Some(violation) =
        proximadb_catalog::relational::find_relational_constraint_violation(&error)
    {
        // A TYPED user-facing cause (TD-DML-1/TD-DML-3), wherever it sits in
        // the chain: its own message (the same text pgwire sends with the
        // SQLSTATE) and a 4xx keyed on the SQLSTATE class (round 5).
        relational_violation_to_api_error(violation)
    } else {
        // Internal failures: the client gets ONLY the outermost message — the
        // full cause chain can carry lock-store keys, object-store endpoints /
        // keys / signed query strings and layout causes, so it is logged
        // server-side and never rendered (round 5, MED-A).
        error!(error = %format!("{error:#}"), "REST SQL internal failure (full cause chain)");
        ApiError::Internal(format!("SQL execution failed: {error}"))
    }
}

/// Map a typed relational violation to its REST status by SQLSTATE class:
/// 42501 (row-level security / insufficient privilege) → 403; class 23
/// (integrity) and 21 (cardinality) → 409; every other class (22 data
/// exception, 42 syntax/undefined object, 0A not supported) → 400. The body
/// is the violation's own message plus its SQLSTATE — what pgwire reports.
pub(crate) fn relational_violation_to_api_error(
    violation: &proximadb_catalog::relational::RelationalConstraintViolation,
) -> ApiError {
    let sqlstate = violation.sqlstate();
    let message = format!("{} (SQLSTATE {sqlstate})", violation.message);
    match sqlstate {
        "42501" => ApiError::Forbidden(message),
        state if state.starts_with("23") || state.starts_with("21") => ApiError::Conflict(message),
        _ => ApiError::InvalidArgument(message),
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
        (status = 400, description = "Invalid SQL request (incl. data / undefined-object violations).", body = crate::network::rest::openapi::ErrorResponse),
        (status = 403, description = "Insufficient privilege (SQLSTATE 42501, e.g. a row-level-security denial).", body = crate::network::rest::openapi::ErrorResponse),
        (status = 408, description = "Query deadline exceeded.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 409, description = "Write lock conflict, or an integrity / cardinality violation (SQLSTATE class 23 / 21).", body = crate::network::rest::openapi::ErrorResponse),
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
                object.insert(column, v2_row_render(value));
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

    /// Round 5 (MED-A): typed causes map to their own message + 4xx; an
    /// internal failure never renders an inner cause to the client.
    #[test]
    fn execution_errors_map_typed_causes_and_never_render_inner_internal_causes() {
        use proximadb_catalog::relational::{
            RelationalConstraintKind, RelationalConstraintViolation,
        };
        let wrapped = |kind, message: &str| {
            anyhow::Error::new(RelationalConstraintViolation::new(kind, message))
                .context("DML failed")
                .context("outer")
        };
        let render = |error: ApiError| match error {
            ApiError::Forbidden(m) => ("403", m),
            ApiError::Conflict(m) => ("409", m),
            ApiError::InvalidArgument(m) => ("400", m),
            ApiError::Internal(m) => ("500", m),
            other => ("other", other.to_string()),
        };
        let (status, message) = render(map_execution_error(wrapped(
            RelationalConstraintKind::InsufficientPrivilege,
            "new row violates row-level security policy",
        )));
        assert_eq!(status, "403");
        assert_eq!(
            message,
            "new row violates row-level security policy (SQLSTATE 42501)"
        );
        assert_eq!(
            render(map_execution_error(wrapped(
                RelationalConstraintKind::Unique,
                "duplicate key"
            )))
            .0,
            "409"
        );
        assert_eq!(
            render(map_execution_error(wrapped(
                RelationalConstraintKind::InvalidTextRepresentation,
                "bad int"
            )))
            .0,
            "400"
        );
        assert_eq!(
            render(map_execution_error(wrapped(
                RelationalConstraintKind::UndefinedColumn,
                "no column"
            )))
            .0,
            "400"
        );

        // An internal failure whose INNER cause carries a secret-looking
        // detail: only the outermost message reaches the client.
        let internal = anyhow::anyhow!(
            "object store GET https://acct.blob.core.windows.net/bkt/key?sig=SECRET failed"
        )
        .context("lock store tenant=t1 namespace=n1 table=tbl")
        .context("DML failed");
        let (status, message) = render(map_execution_error(internal));
        assert_eq!(status, "500");
        assert_eq!(message, "SQL execution failed: DML failed");
        assert!(
            !message.contains("SECRET") && !message.contains("lock store"),
            "no inner cause may reach the client: {message}"
        );
    }

    #[test]
    fn v2_row_render_is_depth_consistent() {
        // One spelling per type at ALL depths: nested uuids stay dashed,
        // nested binary stays the int-array (the /api/v2/records write
        // form). A top-level-only composition left nested uuids undashed
        // inside one response (round-6/7 review findings).
        let row = ProximaValue::Map(
            [
                (
                    "id".to_string(),
                    ProximaValue::Uuid([
                        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x65,
                        0x54, 0x40, 0x00, 0x00,
                    ]),
                ),
                (
                    "payload".to_string(),
                    ProximaValue::Array(vec![
                        ProximaValue::Binary(vec![0x00, 0x01, 0x02]),
                        ProximaValue::Uuid([
                            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x65,
                            0x54, 0x40, 0x00, 0x00,
                        ]),
                    ]),
                ),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            v2_row_render(&row),
            serde_json::json!({
                "id": "550e8400-e29b-41d4-a716-446554400000",
                "payload": [
                    [0, 1, 2],
                    "550e8400-e29b-41d4-a716-446554400000"
                ]
            })
        );
    }

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
