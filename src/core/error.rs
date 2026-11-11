//! Compatibility shim for legacy core error imports.
//!
//! The foundational error contract now lives in the workspace leaf crate
//! `proximadb-kernel`. This module preserves existing imports such as
//! `crate::core::error::ProximaDBError`.

pub use proximadb_kernel::error::*;

// Canonical storage errors (storage_common::StorageError) flow to crate::core::errors::ProximaDBError
// via its #[from] impl. Files returning kernel::VectorDBError should migrate to
// crate::core::errors::ProximaDBError, which has the full conversion chain.

impl From<ProtocolError> for crate::core::errors::ProximaDBError {
    fn from(err: ProtocolError) -> Self {
        match err {
            ProtocolError::InvalidArgument { msg, .. } => {
                crate::core::errors::ProximaDBError::InvalidInput(msg)
            }
            ProtocolError::NotFound { resource, id } => {
                crate::core::errors::ProximaDBError::NotFound {
                    resource_type: resource,
                    id,
                }
            }
            ProtocolError::AlreadyExists { resource, id } => {
                crate::core::errors::ProximaDBError::AlreadyExists {
                    resource_type: resource,
                    id,
                }
            }
            ProtocolError::Internal { details } => {
                crate::core::errors::ProximaDBError::Internal(details)
            }
            ProtocolError::PermissionDenied { action } => {
                crate::core::errors::ProximaDBError::PermissionDenied(action)
            }
            ProtocolError::Timeout { duration_ms, .. } => {
                crate::core::errors::ProximaDBError::Timeout(duration_ms / 1000)
            }
            ProtocolError::ResourceExhausted { details } => {
                crate::core::errors::ProximaDBError::CapacityExceeded { message: details }
            }
            ProtocolError::PreconditionFailed { details } => {
                crate::core::errors::ProximaDBError::InvalidInput(details)
            }
        }
    }
}

impl From<VectorDBError> for crate::core::errors::ProximaDBError {
    fn from(err: VectorDBError) -> Self {
        match err {
            VectorDBError::Storage(e) => crate::core::errors::ProximaDBError::Storage(e.into()),
            VectorDBError::Config(msg) => {
                crate::core::errors::ProximaDBError::Internal(format!("Config: {}", msg))
            }
            VectorDBError::Internal(msg) => crate::core::errors::ProximaDBError::Internal(msg),
            VectorDBError::Quantization(msg) => {
                crate::core::errors::ProximaDBError::Quantization(msg)
            }
            VectorDBError::InvalidInput(msg) => {
                crate::core::errors::ProximaDBError::InvalidInput(msg)
            }
            VectorDBError::Filesystem(msg) => crate::core::errors::ProximaDBError::Io(msg),
            VectorDBError::NotImplemented(msg) => {
                crate::core::errors::ProximaDBError::Internal(format!("Not implemented: {}", msg))
            }
            VectorDBError::Consensus(e) => {
                crate::core::errors::ProximaDBError::Network(e.to_string())
            }
            VectorDBError::Network(e) => {
                crate::core::errors::ProximaDBError::Network(e.to_string())
            }
            VectorDBError::Query(e) => crate::core::errors::ProximaDBError::Internal(e.to_string()),
            VectorDBError::Schema(e) => {
                crate::core::errors::ProximaDBError::Internal(e.to_string())
            }
            VectorDBError::TransactionNotFound { id } => {
                crate::core::errors::ProximaDBError::TransactionNotFound { id }
            }
            VectorDBError::TransactionNotActive { id } => {
                crate::core::errors::ProximaDBError::TransactionNotActive { id }
            }
            VectorDBError::TransactionTimedOut { id } => {
                crate::core::errors::ProximaDBError::TransactionTimedOut { id }
            }
            VectorDBError::TransactionConflict {
                transaction,
                conflicting_with,
            } => crate::core::errors::ProximaDBError::TransactionConflict {
                transaction,
                conflicting_with,
            },
            VectorDBError::TooManyTransactions { max } => {
                crate::core::errors::ProximaDBError::TooManyTransactions { max }
            }
            VectorDBError::TransactionAborted(msg) => {
                crate::core::errors::ProximaDBError::Internal(format!(
                    "Transaction aborted: {}",
                    msg
                ))
            }
            VectorDBError::LockTimeout { resource } => {
                crate::core::errors::ProximaDBError::LockTimeout { resource }
            }
            VectorDBError::DeadlockDetected { transaction } => {
                crate::core::errors::ProximaDBError::DeadlockDetected { transaction }
            }
            VectorDBError::SavepointNotFound { name } => {
                crate::core::errors::ProximaDBError::SavepointNotFound { name }
            }
            VectorDBError::NotFound { resource_type, id } => {
                crate::core::errors::ProximaDBError::NotFound { resource_type, id }
            }
        }
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
