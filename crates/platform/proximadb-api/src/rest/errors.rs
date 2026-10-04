//! REST API error types — shim over the foundation envelope (ADR-094).
//!
//! `RestError` is now an alias of the unified foundation error enum
//! (`proximadb_api_error::ApiError`); all 9 former variants exist there with
//! identical display strings and HTTP mappings, plus the request-id
//! task-local. `From<anyhow::Error>` comes from the foundation crate (an
//! orphan-rule re-declaration here would not compile).

pub use proximadb_api_error::{current_request_id, ApiError as RestError, REQUEST_ID};

/// whenever the request passed through the request-id middleware; quote it in
/// bug reports.
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

/// Inner body of [`ErrorResponse`].
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct ErrorBody {
    /// Stable machine-readable error code (snake_case).
    #[schema(example = "collection_not_found")]
    pub r#type: String,
    pub message: String,
    /// HTTP status code.
    pub code: i32,
    /// Correlation id (matches the X-Request-ID header).
    pub request_id: Option<String>,
    /// Optional structured context (e.g. migration hints).
    #[schema(value_type = Option<Object>)]
    pub details: Option<serde_json::Value>,
}


/// Result alias for REST handler functions.
pub type RestResult<T> = Result<T, RestError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::IntoResponse;
    use serde_json::Value;

    async fn response_parts(error: RestError) -> (axum::http::StatusCode, Value) {
        let response = error.into_response();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn rest_error_variants_map_to_status_type_message_and_code() {
        let cases = [
            (
                RestError::CollectionNotFound("c1".to_string()),
                axum::http::StatusCode::NOT_FOUND,
                "collection_not_found",
                "Collection not found: c1",
            ),
            (
                RestError::InvalidArgument("bad field".to_string()),
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_argument",
                "Invalid argument: bad field",
            ),
            (
                RestError::Internal("boom".to_string()),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "Internal error: boom",
            ),
            (
                RestError::NotFound("row".to_string()),
                axum::http::StatusCode::NOT_FOUND,
                "not_found",
                "Not found: row",
            ),
            (
                RestError::Conflict("write-write".to_string()),
                axum::http::StatusCode::CONFLICT,
                "conflict",
                "Conflict: write-write",
            ),
        ];

        for (error, expected_status, expected_type, expected_message) in cases {
            let (status, body) = response_parts(error).await;
            assert_eq!(status, expected_status);
            assert_eq!(body["error"]["type"], expected_type);
            assert_eq!(body["error"]["message"], expected_message);
            assert_eq!(body["error"]["code"], expected_status.as_u16());
        }
    }

    #[tokio::test]
    async fn error_envelope_carries_request_id_when_scoped() {
        let (_status, body) = REQUEST_ID
            .scope("req-abc-123".to_string(), async {
                response_parts(RestError::NotFound("x".to_string())).await
            })
            .await;
        assert_eq!(body["error"]["request_id"], "req-abc-123");
    }
}
