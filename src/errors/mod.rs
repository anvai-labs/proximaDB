//! Unified error handling for ProximaDB APIs
//!
//! This module provides a single ApiError type that can be converted
//! to both gRPC Status and HTTP responses, ensuring consistent error
//! handling across all API protocols.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Unified API error type for consistent error handling across REST and gRPC
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

    /// A unified-query-port execution exceeded the port deadline
    /// (`proximadb_runtime::PORT_QUERY_DEADLINE`) — HTTP 504, the one status
    /// every port surface uses for it (TD-SPINE-6 round 3).
    #[error("Query deadline exceeded: {0}")]
    QueryDeadline(String),

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

    /// Slice 4 of tenant-pod-affinity: the request landed on the
    /// wrong pod. The primary-pod registry has a binding for this
    /// `(tenant, collection)` that points at a different pod.
    /// Mapped to HTTP 421 Misdirected Request — the canonical
    /// semantics for "this server is not configured to serve this
    /// authority." The target pod identifier is carried in the
    /// error so the client SDK can retry against the right host.
    #[error("Misdirected request: write must go to pod '{target_pod}'")]
    Misdirected {
        /// Primary pod for the requested `(tenant, collection)`.
        target_pod: String,
        /// The tenant the misroute applies to (echoed back for
        /// audit / client-side logging).
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

    /// Map an anyhow error from the write/ingest path to an `ApiError`, promoting
    /// an ADR-069 S4 `WalBackpressure` found anywhere in the error chain to a
    /// **retryable** `ResourceExhausted` (HTTP 429 / gRPC RESOURCE_EXHAUSTED) so
    /// clients back off instead of treating a shed write as a non-retryable 500.
    /// Any other error maps to `Internal` under the `context` prefix. Walking the
    /// chain keeps the mapping robust to `.context(..)` wrapping by intermediate
    /// layers.
    pub fn from_write_error(context: &str, err: anyhow::Error) -> Self {
        use crate::storage::persistence::write_ahead_log::flush_policy::WalBackpressure;
        if let Some(bp) = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<WalBackpressure>())
        {
            ApiError::ResourceExhausted(bp.to_string())
        } else {
            ApiError::Internal(format!("{context}: {err}"))
        }
    }
}

/// Convert ApiError to gRPC Status
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
            ApiError::QueryDeadline(msg) => tonic::Status::deadline_exceeded(msg),
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
                // Trailing metadata makes the structured fields
                // machine-readable without the client having to
                // parse the human message.
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
pub fn protocol_error_to_grpc_status(err: proximadb_kernel::error::ProtocolError) -> tonic::Status {
    ApiError::from(err).into()
}

/// Convert ApiError to HTTP Response
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
            ApiError::QueryDeadline(_) => (StatusCode::GATEWAY_TIMEOUT, "deadline_exceeded"),
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

        // Misdirected requests get a structured body with the target
        // pod so the client SDK can re-route. Other errors fall
        // through to the generic JSON envelope below.
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

        // Check if this is a capability error and use enhanced formatting
        let body = if matches!(&self, ApiError::UnsupportedCapability(_)) {
            // This is a capability error - try to extract details
            let error_msg = self.to_string();
            let cap_error = CapabilityError {
                capability: "capability".to_string(),
                available_alternatives: vec![],
                message: error_msg.clone(),
                error_type: CapabilityErrorType::UnsupportedCapability,
            };

            Json(cap_error.to_rest_response()).into_response()
        } else {
            Json(json!({
                "error": {
                    "type": error_type,
                    "message": self.to_string(),
                    "code": status.as_u16()
                }
            }))
            .into_response()
        };

        (status, body).into_response()
    }
}

/// Capability error module for protocol-aware error mapping
pub mod capability_error;

// Re-export capability error types for convenience
pub use capability_error::{CapabilityError, CapabilityErrorType};

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

/// Convert CapabilityCheckError to ApiError for unified error handling
impl From<crate::query::capability::CapabilityCheckError> for ApiError {
    fn from(err: crate::query::capability::CapabilityCheckError) -> Self {
        use crate::query::capability::CapabilityCheckError;

        match err {
            CapabilityCheckError::UnsupportedCapability {
                capability,
                available_alternatives,
            } => {
                let msg = if available_alternatives.is_empty() {
                    format!(
                        "The requested capability '{}' is not supported by the selected storage engine.",
                        capability
                    )
                } else {
                    format!(
                        "The requested capability '{}' is not supported. Available alternatives: {}",
                        capability,
                        available_alternatives.join(", ")
                    )
                };
                ApiError::UnsupportedCapability(msg)
            }

            CapabilityCheckError::MultipleUnsupportedCapabilities {
                missing_capabilities,
                available_alternatives,
            } => {
                let msg = if available_alternatives.is_empty() {
                    format!(
                        "Multiple capabilities are not supported: {}. Please check the storage engine capabilities.",
                        missing_capabilities.join(", ")
                    )
                } else {
                    format!(
                        "Multiple capabilities are not supported: {}. Available alternatives: {}",
                        missing_capabilities.join(", "),
                        available_alternatives.join(", ")
                    )
                };
                ApiError::UnsupportedCapability(msg)
            }
        }
    }
}

/// Convert ProtocolError to ApiError for unified error handling
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

/// Convert canonical ProximaDBError to ApiError for unified error handling.
/// This enables any service-layer code returning ProximaDBError to be used
/// directly in API handlers via the ? operator.
impl From<crate::core::errors::ProximaDBError> for ApiError {
    fn from(err: crate::core::errors::ProximaDBError) -> Self {
        use crate::core::errors::ProximaDBError as E;
        match err {
            E::NotFound { resource_type, id } => {
                if resource_type.to_lowercase() == "collection" {
                    ApiError::CollectionNotFound(id)
                } else {
                    ApiError::NotFound(format!("{} not found: {}", resource_type, id))
                }
            }
            E::AlreadyExists { resource_type, id } => {
                ApiError::AlreadyExists(format!("{}: {}", resource_type, id))
            }
            E::InvalidInput(msg) => ApiError::InvalidArgument(msg),
            E::InvalidCacheKey(msg) => ApiError::InvalidArgument(msg),
            E::Authentication(msg) => ApiError::Unauthorized(msg),
            E::PermissionDenied(msg) => ApiError::Forbidden(msg),
            E::Timeout(secs) => ApiError::DeadlineExceeded(format!("Timed out after {}s", secs)),
            E::CapacityExceeded { message } => ApiError::ResourceExhausted(message),
            E::TransactionConflict {
                transaction,
                conflicting_with,
            } => ApiError::Conflict(format!(
                "{} conflicts with {}",
                transaction, conflicting_with
            )),
            E::DmlLockConflict { resource, holder } => ApiError::LockConflict(match holder {
                Some(h) => format!("{resource} held by {h}"),
                None => resource,
            }),
            other => ApiError::Internal(other.to_string()),
        }
    }
}

/// Walk an `anyhow::Error` chain looking for a DML lock conflict
/// (`ProximaDBError::DmlLockConflict`). Returns `(resource, holder?)` so
/// protocol boundaries that receive a raw `anyhow::Error` (pgwire, gRPC) can
/// detect a lock conflict and map it to the right code (SQLSTATE 55P03 /
/// `tonic::ABORTED`) without each reimplementing the chain walk. Anyhow's
/// `downcast_ref` only inspects the top error, so we iterate `.chain()` to see
/// through `.context(...)` wrappers added along the way.
pub fn extract_dml_lock_conflict(err: &anyhow::Error) -> Option<(String, Option<String>)> {
    use crate::core::errors::ProximaDBError as E;
    err.chain()
        .find_map(|source| match source.downcast_ref::<E>() {
            Some(E::DmlLockConflict { resource, holder }) => {
                Some((resource.clone(), holder.clone()))
            }
            _ => None,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_to_status() {
        let err = ApiError::CollectionNotFound("test_collection".to_string());
        let status: tonic::Status = err.into();
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn dml_lock_conflict_maps_to_conflict_status_and_http() {
        use crate::core::errors::ProximaDBError;
        let mk = || {
            ApiError::from(ProximaDBError::DmlLockConflict {
                resource: "public.users".into(),
                holder: Some("pod-7".into()),
            })
        };
        // Variant + gRPC → ABORTED (retryable).
        let api = mk();
        assert!(matches!(api, ApiError::LockConflict(_)));
        let status: tonic::Status = api.into();
        assert_eq!(status.code(), tonic::Code::Aborted);
        // REST → 409 Conflict (ApiError isn't Clone, so rebuild).
        let resp = mk().into_response();
        assert_eq!(resp.status().as_u16(), 409);
    }

    #[test]
    fn extract_dml_lock_conflict_walks_anyhow_chain() {
        use crate::core::errors::ProximaDBError;
        // Plain (no context wrapper).
        let e: anyhow::Error = ProximaDBError::DmlLockConflict {
            resource: "public.t".into(),
            holder: None,
        }
        .into();
        let (resource, holder) = extract_dml_lock_conflict(&e).expect("should find the conflict");
        assert_eq!(resource, "public.t");
        assert!(holder.is_none());

        // Through a .context(...) wrapper (the real DmlService path).
        let wrapped: anyhow::Error = anyhow::Error::new(ProximaDBError::DmlLockConflict {
            resource: "s.t".into(),
            holder: Some("pod-1".into()),
        })
        .context("DML failed");
        let (resource, holder) =
            extract_dml_lock_conflict(&wrapped).expect("should see through context");
        assert_eq!(resource, "s.t");
        assert_eq!(holder.as_deref(), Some("pod-1"));

        // Unrelated error → None.
        let other: anyhow::Error = ProximaDBError::InvalidInput("boom".into()).into();
        assert!(extract_dml_lock_conflict(&other).is_none());
    }

    #[test]
    fn test_api_error_to_response() {
        let err = ApiError::InvalidArgument("bad input".to_string());
        let _response = err.into_response();
        // Response will have status 400 and JSON body
    }

    #[test]
    fn test_protocol_error_to_grpc_status() {
        let err = proximadb_kernel::error::ProtocolError::not_found("collection", "c1");

        let status = protocol_error_to_grpc_status(err);

        assert_eq!(status.code(), tonic::Code::NotFound);
        assert!(status.message().contains("c1"));
    }

    #[test]
    fn from_write_error_promotes_wal_backpressure_to_retryable() {
        use crate::storage::persistence::write_ahead_log::flush_policy::WalBackpressure;
        let bp = WalBackpressure {
            collection_id: "c1".to_string(),
            fill_pct: 97.0,
        };
        // Wrapped in context by intermediate layers — the chain-walk still finds it
        // and promotes it to the retryable ResourceExhausted (RESOURCE_EXHAUSTED).
        let wrapped = anyhow::Error::new(bp)
            .context("wal append")
            .context("insert path");
        let mapped = ApiError::from_write_error("Insert failed", wrapped);
        assert!(matches!(&mapped, ApiError::ResourceExhausted(_)));
        assert_eq!(
            tonic::Status::from(mapped).code(),
            tonic::Code::ResourceExhausted
        );
        // A plain error stays a non-retryable Internal.
        let other = ApiError::from_write_error("Insert failed", anyhow::anyhow!("disk gone"));
        assert!(matches!(other, ApiError::Internal(_)));
    }
}
