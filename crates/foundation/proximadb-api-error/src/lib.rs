//! Unified wire error envelope for ProximaDB API surfaces.
//!
//! Foundation-tier home of the single REST error enum (ADR-094): one type, one
//! `{error:{type,message,code}}` envelope, one request-id task-local, and the
//! gRPC `tonic::Status` mapping (the orphan rules require the conversion to be
//! defined in the enum's own crate). Protocol crates (root REST/gRPC servers,
//! `proximadb-api`) re-export this type; only conversions FROM root-local error
//! types stay in the root crate.
//!
//! Root-specific `From` impls (ProximaDBError, CapabilityCheckError,
//! WalBackpressure chain-walk) remain in the root crate's `src/errors/mod.rs`,
//! targeting this type — orphan rules permit `impl From<LocalType> for
//! ForeignType` when the source is local.
//!
//! Legacy wire quirk preserved deliberately: `UnsupportedCapability` emits the
//! historical `CapabilityError::to_rest_response` shape
//! (`{error:{error_type,message,missing_capability:"capability"}}` — no `code`,
//! no `type` key) because the root construction never populated the structured
//! fields. Unify it only as a reviewed spec change.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

tokio::task_local! {
    /// Per-request correlation id, scoped by the request-id middleware around
    /// each request. Read by `ApiError::into_response` so error envelopes carry
    /// the SAME id the `X-Request-ID` response header advertises, with zero
    /// changes to handler signatures.
    pub static REQUEST_ID: String;
}

/// The request id for the current task scope, if the request-id middleware set
/// one. `None` on paths not wrapped by the middleware (so we never emit a fake
/// id that wouldn't match the `X-Request-ID` header).
pub fn current_request_id() -> Option<String> {
    REQUEST_ID.try_with(|id| id.clone()).ok()
}

/// Unified API error type for consistent error handling across REST and gRPC.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// Collection not found
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    /// Invalid argument provided
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Resource exhausted (rate limiting, memory, etc.)
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Unauthorized access
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Forbidden - insufficient permissions
    #[error("Forbidden: {0}")]
    Forbidden(String),

    /// Operation not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Deadline exceeded
    #[error("Deadline exceeded: {0}")]
    DeadlineExceeded(String),

    /// Already exists
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Generic resource not found (for non-collection resources like prepared statements)
    #[error("Not found: {0}")]
    NotFound(String),

    /// Resource has expired or been removed (HTTP 410 Gone)
    #[error("Gone: {0}")]
    Gone(String),

    /// Vector dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The dimension the collection expects.
        expected: usize,
        /// The dimension that was actually provided.
        actual: usize,
    },

    /// Invalid vector data
    #[error("Invalid vector: {0}")]
    InvalidVector(String),

    /// Conflict error (e.g., schema evolution violation)
    #[error("Conflict: {0}")]
    Conflict(String),

    /// DML write-lock conflict — another writer holds the table/schema lease.
    /// REST 409 / gRPC ABORTED / pgwire SQLSTATE 55P03.
    #[error("DML lock conflict: {0}")]
    LockConflict(String),

    /// Capability not supported by storage engine
    #[error("Capability not supported: {0}")]
    UnsupportedCapability(String),

    /// Slice 4 of tenant-pod-affinity: the request landed on the wrong pod.
    /// Mapped to HTTP 421 Misdirected Request; the target pod identifier is
    /// carried in the error so the client SDK can retry against the right host.
    #[error("Misdirected request: write must go to pod '{target_pod}'")]
    Misdirected {
        /// Primary pod for the requested `(tenant, collection)`.
        target_pod: String,
        /// The tenant the misroute applies to (echoed back for audit /
        /// client-side logging).
        tenant_id: String,
        /// The collection the misroute applies to.
        collection_id: String,
    },
}

impl ApiError {
    /// Convert from anyhow::Error
    pub fn from_anyhow(err: anyhow::Error) -> Self {
        ApiError::Internal(err.to_string())
    }
}

/// Result type alias for API operations
pub type ApiResult<T> = Result<T, ApiError>;

/// Helper trait for converting various error types to ApiError
pub trait IntoApiError {
    /// Convert this value into an [`ApiError`].
    fn into_api_error(self) -> ApiError;
}

impl IntoApiError for anyhow::Error {
    fn into_api_error(self) -> ApiError {
        ApiError::Internal(self.to_string())
    }
}

impl IntoApiError for std::io::Error {
    fn into_api_error(self) -> ApiError {
        ApiError::Internal(format!("IO error: {}", self))
    }
}

impl IntoApiError for serde_json::Error {
    fn into_api_error(self) -> ApiError {
        ApiError::InvalidArgument(format!("JSON error: {}", self))
    }
}

impl From<String> for ApiError {
    fn from(msg: String) -> Self {
        ApiError::Internal(msg)
    }
}

impl From<&str> for ApiError {
    fn from(msg: &str) -> Self {
        ApiError::Internal(msg.to_string())
    }
}

/// Convert `proximadb_kernel::error::ProtocolError` to ApiError for unified
/// error handling (kernel is a foundation-tier peer, so this From lives here).
impl From<proximadb_kernel::error::ProtocolError> for ApiError {
    fn from(err: proximadb_kernel::error::ProtocolError) -> Self {
        use proximadb_kernel::error::ProtocolError;
        match err {
            ProtocolError::InvalidArgument { msg, field } => {
                let message = if let Some(f) = field {
                    format!("{} (field: {})", msg, f)
                } else {
                    msg
                };
                ApiError::InvalidArgument(message)
            }
            ProtocolError::NotFound { resource, id } => {
                if resource.to_lowercase() == "collection" {
                    ApiError::CollectionNotFound(id)
                } else {
                    ApiError::InvalidArgument(format!("{} not found: {}", resource, id))
                }
            }
            ProtocolError::AlreadyExists { resource, id } => {
                ApiError::AlreadyExists(format!("{}: {}", resource, id))
            }
            ProtocolError::Internal { details } => ApiError::Internal(details),
            ProtocolError::PermissionDenied { action } => {
                ApiError::Unauthorized(format!("Permission denied: {}", action))
            }
            ProtocolError::Timeout {
                operation,
                duration_ms,
            } => ApiError::DeadlineExceeded(format!(
                "Operation '{}' timed out after {}ms",
                operation, duration_ms
            )),
            ProtocolError::ResourceExhausted { details } => ApiError::ResourceExhausted(details),
            ProtocolError::PreconditionFailed { details } => ApiError::InvalidArgument(details),
        }
    }
}

/// Helper function to convert Result<T, ApiError> to Response
pub fn result_into_response<T>(result: Result<T, ApiError>) -> Response
where
    T: IntoResponse,
{
    match result {
        Ok(value) => value.into_response(),
        Err(error) => error.into_response(),
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            ApiError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, "collection_not_found"),
            ApiError::InvalidArgument(_) => (StatusCode::BAD_REQUEST, "invalid_argument"),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
            ApiError::ResourceExhausted(_) => (StatusCode::TOO_MANY_REQUESTS, "resource_exhausted"),
            ApiError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "unauthorized"),
            ApiError::Forbidden(_) => (StatusCode::FORBIDDEN, "forbidden"),
            ApiError::NotImplemented(_) => (StatusCode::NOT_IMPLEMENTED, "not_implemented"),
            ApiError::DeadlineExceeded(_) => (StatusCode::REQUEST_TIMEOUT, "deadline_exceeded"),
            ApiError::AlreadyExists(_) => (StatusCode::CONFLICT, "already_exists"),
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            ApiError::Gone(_) => (StatusCode::GONE, "gone"),
            ApiError::DimensionMismatch { .. } => (StatusCode::BAD_REQUEST, "dimension_mismatch"),
            ApiError::InvalidVector(_) => (StatusCode::BAD_REQUEST, "invalid_vector"),
            ApiError::Conflict(_) => (StatusCode::CONFLICT, "conflict"),
            ApiError::LockConflict(_) => (StatusCode::CONFLICT, "lock_conflict"),
            ApiError::UnsupportedCapability(_) => {
                (StatusCode::BAD_REQUEST, "unsupported_capability")
            }
            ApiError::Misdirected { .. } => {
                (StatusCode::MISDIRECTED_REQUEST, "misdirected_request")
            }
        };

        // Misdirected requests get a structured body with the target pod so
        // the client SDK can re-route.
        if let ApiError::Misdirected {
            target_pod,
            tenant_id,
            collection_id,
        } = &self
        {
            let body = Json(json!({
                "error": {
                    "type": "misdirected_request",
                    "message": format!(
                        "write for ({}, {}) must go to pod '{}'",
                        tenant_id, collection_id, target_pod
                    ),
                    "code": status.as_u16(),
                    "target_pod": target_pod,
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                }
            }));
            return (status, body).into_response();
        }

        // Legacy wire shape preserved verbatim (ADR-094): the root crate's
        // CapabilityError formatting emitted `error_type` (not `type`), no
        // `code`, and a literal `missing_capability` placeholder. Reproduced
        // byte-identically; unify only as a reviewed spec change.
        if let ApiError::UnsupportedCapability(msg) = &self {
            let body = Json(json!({
                "error": {
                    "error_type": "unsupported_capability",
                    "message": format!("Capability not supported: {msg}"),
                    "missing_capability": "capability",
                }
            }));
            return (status, body).into_response();
        }

        let mut error_obj = json!({
            "type": error_type,
            "message": self.to_string(),
            "code": status.as_u16(),
        });
        if let Some(rid) = current_request_id() {
            error_obj["request_id"] = json!(rid);
        }
        (status, Json(json!({ "error": error_obj }))).into_response()
    }
}


/// Walk an error chain looking for the foundation `ApiError::LockConflict`
/// variant (DML write-lock). Returns the conflict message so transport layers
/// can map it (HTTP 409 / gRPC ABORTED / SQLSTATE 55P03) without each
/// reimplementing the chain walk.
pub fn extract_lock_conflict(err: &anyhow::Error) -> Option<String> {
    err.chain().find_map(|cause| match cause.downcast_ref::<ApiError>() {
        Some(ApiError::LockConflict(msg)) => Some(msg.clone()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::Value;

    async fn response_body(error: ApiError) -> (StatusCode, Value) {
        let response = error.into_response();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[test]
    fn variants_map_to_status_type_message_and_code() {
        let cases = [
            (
                ApiError::CollectionNotFound("c1".to_string()),
                StatusCode::NOT_FOUND,
                "collection_not_found",
                "Collection not found: c1",
            ),
            (
                ApiError::InvalidArgument("bad field".to_string()),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "Invalid argument: bad field",
            ),
            (
                ApiError::Internal("boom".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "Internal error: boom",
            ),
            (
                ApiError::NotFound("row".to_string()),
                StatusCode::NOT_FOUND,
                "not_found",
                "Not found: row",
            ),
            (
                ApiError::AlreadyExists("collection".to_string()),
                StatusCode::CONFLICT,
                "already_exists",
                "Already exists: collection",
            ),
            (
                ApiError::Conflict("write-write".to_string()),
                StatusCode::CONFLICT,
                "conflict",
                "Conflict: write-write",
            ),
            (
                ApiError::LockConflict("public.users held by pod-7".to_string()),
                StatusCode::CONFLICT,
                "lock_conflict",
                "DML lock conflict: public.users held by pod-7",
            ),
            (
                ApiError::NotImplemented("feature".to_string()),
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                "Not implemented: feature",
            ),
            (
                ApiError::Unauthorized("missing token".to_string()),
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "Unauthorized: missing token",
            ),
            (
                ApiError::ResourceExhausted("quota".to_string()),
                StatusCode::TOO_MANY_REQUESTS,
                "resource_exhausted",
                "Resource exhausted: quota",
            ),
        ];

        for (error, expected_status, expected_type, expected_message) in cases {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (status, body) = rt.block_on(response_body(error));
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
                response_body(ApiError::NotFound("x".to_string())).await
            })
            .await;
        assert_eq!(body["error"]["request_id"], "req-abc-123");
        assert_eq!(body["error"]["type"], "not_found");
    }

    #[tokio::test]
    async fn error_envelope_omits_request_id_when_unscoped() {
        let (_status, body) = response_body(ApiError::Internal("boom".to_string())).await;
        assert!(body["error"].get("request_id").is_none());
    }

    #[tokio::test]
    async fn unsupported_capability_preserves_legacy_wire_shape() {
        // ADR-094: the root CapabilityError formatting emitted error_type (not
        // type), no code, and the literal missing_capability placeholder.
        let (status, body) = response_body(ApiError::UnsupportedCapability("geo".to_string())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"]["error_type"],
            Value::String("unsupported_capability".to_string())
        );
        assert_eq!(
            body["error"]["message"],
            Value::String("Capability not supported: geo".to_string())
        );
        assert_eq!(
            body["error"]["missing_capability"],
            Value::String("capability".to_string())
        );
        assert!(body["error"].get("code").is_none());
        assert!(body["error"].get("type").is_none());
    }

    #[tokio::test]
    async fn misdirected_carries_structured_reroute_fields() {
        let (status, body) = response_body(ApiError::Misdirected {
            target_pod: "pod-2".to_string(),
            tenant_id: "t1".to_string(),
            collection_id: "c1".to_string(),
        })
        .await;
        assert_eq!(status, StatusCode::MISDIRECTED_REQUEST);
        assert_eq!(body["error"]["type"], "misdirected_request");
        assert_eq!(body["error"]["target_pod"], "pod-2");
        assert_eq!(body["error"]["tenant_id"], "t1");
        assert_eq!(body["error"]["collection_id"], "c1");
        assert_eq!(body["error"]["code"], 421);
    }

    #[tokio::test]
    async fn gone_maps_to_410() {
        let (status, body) = response_body(ApiError::Gone("expired prepared statement".to_string())).await;
        assert_eq!(status, StatusCode::GONE);
        assert_eq!(body["error"]["type"], "gone");
    }

    #[test]
    fn protocol_errors_lower_into_api_error() {
        use proximadb_kernel::error::ProtocolError;
        let err = ApiError::from(ProtocolError::not_found("collection", "c1"));
        assert!(matches!(err, ApiError::CollectionNotFound(id) if id == "c1"));
        let err = ApiError::from(ProtocolError::Timeout {
            operation: "scan".to_string(),
            duration_ms: 1_500,
        });
        assert!(matches!(err, ApiError::DeadlineExceeded(m) if m.contains("1 500ms") || m.contains("1500ms")));
    }

    #[test]
    fn from_anyhow_lowers_to_internal() {
        let err = ApiError::from_anyhow(anyhow::anyhow!("disk gone"));
        assert!(matches!(err, ApiError::Internal(m) if m == "disk gone"));
    }
}

/// Convert ApiError to gRPC Status (ADR-094: the gRPC wire mapping is part of
/// the error contract; tonic lives here because the orphan rules require the
/// conversion to be defined in the enum's own crate).
impl From<ApiError> for tonic::Status {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::CollectionNotFound(msg) => tonic::Status::not_found(msg),
            ApiError::InvalidArgument(msg) => tonic::Status::invalid_argument(msg),
            ApiError::Internal(msg) => tonic::Status::internal(msg),
            ApiError::ResourceExhausted(msg) => tonic::Status::resource_exhausted(msg),
            ApiError::Unauthorized(msg) => tonic::Status::unauthenticated(msg),
            ApiError::Forbidden(msg) => tonic::Status::permission_denied(msg),
            ApiError::NotImplemented(msg) => tonic::Status::unimplemented(msg),
            ApiError::DeadlineExceeded(msg) => tonic::Status::deadline_exceeded(msg),
            ApiError::AlreadyExists(msg) => tonic::Status::already_exists(msg),
            ApiError::NotFound(msg) => tonic::Status::not_found(msg),
            ApiError::Gone(msg) => tonic::Status::not_found(format!("Resource expired: {}", msg)),
            ApiError::DimensionMismatch { expected, actual } => {
                tonic::Status::invalid_argument(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    expected, actual
                ))
            }
            ApiError::InvalidVector(msg) => {
                tonic::Status::invalid_argument(format!("Invalid vector: {}", msg))
            }
            ApiError::Conflict(msg) => tonic::Status::aborted(msg),
            ApiError::LockConflict(msg) => tonic::Status::aborted(format!(
                "DML lock conflict: {msg}. Retry the write once the holder releases."
            )),
            ApiError::UnsupportedCapability(msg) => tonic::Status::invalid_argument(format!(
                "Capability not supported: {}. Please check storage engine capabilities.",
                msg
            )),
            ApiError::Misdirected {
                target_pod,
                tenant_id,
                collection_id,
            } => {
                // gRPC has no direct equivalent of HTTP 421; use
                // FailedPrecondition with a structured message so the
                // client SDK can parse the target pod out for retry.
                let mut status = tonic::Status::failed_precondition(format!(
                    "misdirected_request: write for ({}, {}) must go to pod '{}'",
                    tenant_id, collection_id, target_pod
                ));
                let metadata = status.metadata_mut();
                if let Ok(v) = target_pod.parse() {
                    metadata.insert("x-primary-pod", v);
                }
                if let Ok(v) = tenant_id.parse() {
                    metadata.insert("x-tenant-id", v);
                }
                if let Ok(v) = collection_id.parse() {
                    metadata.insert("x-collection-id", v);
                }
                status
            }
        }
    }
}

/// Convert ProtocolError to gRPC Status via ApiError.
///
/// This stays as a named adapter to avoid implementing a foreign trait for a
/// foundation error type at the API boundary.
pub fn protocol_error_to_grpc_status(
    err: proximadb_kernel::error::ProtocolError,
) -> tonic::Status {
    ApiError::from(err).into()
}

#[cfg(test)]
mod grpc_tests {
    use super::*;

    #[test]
    fn api_error_maps_to_grpc_status() {
        let status: tonic::Status = ApiError::CollectionNotFound("c1".to_string()).into();
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn gone_maps_to_not_found_with_context() {
        let status: tonic::Status = ApiError::Gone("expired".to_string()).into();
        assert_eq!(status.code(), tonic::Code::NotFound);
        assert!(status.message().contains("Resource expired"));
    }

    #[test]
    fn misdirected_carries_trailing_metadata() {
        let status: tonic::Status = ApiError::Misdirected {
            target_pod: "pod-2".to_string(),
            tenant_id: "t1".to_string(),
            collection_id: "c1".to_string(),
        }
        .into();
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
        assert_eq!(status.metadata().get("x-primary-pod").unwrap(), "pod-2");
    }
}
