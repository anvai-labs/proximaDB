//! Unified error handling for ProximaDB APIs — root shim (ADR-094).
//!
//! The error enum, HTTP envelope, request-id task-local, and kernel-protocol
//! conversion live in the foundation crate
//! [`proximadb_api_error`](proximadb_api_error) and are re-exported here so
//! `crate::errors::*` paths keep working across the monolith.
//!
//! What remains in the ROOT crate: conversions from root-local error types
//! (`ProximaDBError`, `CapabilityCheckError`, the WalBackpressure chain-walk)
//! — orphan rules permit `impl From<LocalType> for ForeignType`.

pub use proximadb_api_error::{
    current_request_id, result_into_response, ApiError, ApiResult, IntoApiError, REQUEST_ID,
};

/// Capability error module for protocol-aware error mapping
pub mod capability_error;

// Re-export capability error types for convenience
pub use capability_error::{CapabilityError, CapabilityErrorType};

/// Convert ApiError to gRPC Status.
///
/// The `From<ApiError> for tonic::Status` impl lives in the foundation crate
/// (ADR-094), so `.into()` works workspace-wide through coherence.
pub use proximadb_api_error::protocol_error_to_grpc_status;

/// Map an anyhow error from the write/ingest path to an `ApiError`, promoting
/// an ADR-069 S4 `WalBackpressure` found anywhere in the error chain to a
/// **retryable** `ResourceExhausted` (HTTP 429 / gRPC RESOURCE_EXHAUSTED) so
/// clients back off instead of treating a shed write as a non-retryable 500.
/// Any other error maps to `Internal` under the `context` prefix. Walking the
/// chain keeps the mapping robust to `.context(..)` wrapping by intermediate
/// layers.
///
/// (Root free function — not an inherent `ApiError` method — because
/// `WalBackpressure` is a root type and the enum now lives in foundation;
/// orphan rules forbid inherent impls on foreign types.)
pub fn api_error_from_write_error(context: &str, err: anyhow::Error) -> ApiError {
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

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
        let mapped = crate::errors::api_error_from_write_error("Insert failed", wrapped);
        assert!(matches!(&mapped, ApiError::ResourceExhausted(_)));
        assert_eq!(
            tonic::Status::from(mapped).code(),
            tonic::Code::ResourceExhausted
        );
        // A plain error stays a non-retryable Internal.
        let other =
            crate::errors::api_error_from_write_error("Insert failed", anyhow::anyhow!("disk gone"));
        assert!(matches!(other, ApiError::Internal(_)));
    }
}
