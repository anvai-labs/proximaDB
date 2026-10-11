//! Queue error types.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, QueueError>;

#[derive(Debug, Error)]
pub enum QueueError {
    #[error("topic not found: {0}")]
    TopicNotFound(String),

    #[error("partition not found: topic={topic}, partition={partition}")]
    PartitionNotFound { topic: String, partition: u32 },

    /// Soft backpressure — caller MAY retry or accept slight slowdown.
    /// Returned when memory tier is between soft and hard threshold.
    #[error("queue backpressure (soft): memory tier at {pct:.0}%")]
    BackpressureSoft { pct: f32 },

    /// Hard backpressure — caller MUST back off. Includes suggested
    /// retry-after in milliseconds.
    #[error("queue backpressure (hard): memory tier at {pct:.0}%; retry after {retry_after_ms}ms")]
    Backpressure { pct: f32, retry_after_ms: u64 },

    /// Disk write or fsync failed. Caller should treat as unrecoverable
    /// for the current request.
    #[error("queue persistence failure: {0}")]
    Persistence(String),

    /// Conditional publication may have applied; this consumer must stop.
    /// A failed response is not evidence that a durable ACK was rejected.
    #[error("queue publication outcome indeterminate; close consumer and recover: {0}")]
    PublicationIndeterminate(String),

    /// Typed absence; permission failures and unavailable storage are NOT absence.
    #[error("queue path not found: {0}")]
    NotFound(String),

    /// Consumer lease conflict — another instance owns the partition.
    #[error("lease conflict on topic={topic}, partition={partition}: held by {holder}")]
    LeaseConflict {
        topic: String,
        partition: u32,
        holder: String,
    },

    /// Message acknowledgement targeted an unknown message id.
    #[error("unknown message id: {0}")]
    UnknownMessageId(String),

    /// Wrapper for unexpected I/O or serialization errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_errors_render_actionable_context() {
        let cases = [
            (
                QueueError::TopicNotFound("orders".to_string()),
                "topic not found: orders",
            ),
            (
                QueueError::PartitionNotFound {
                    topic: "orders".to_string(),
                    partition: 3,
                },
                "partition not found: topic=orders, partition=3",
            ),
            (
                QueueError::BackpressureSoft { pct: 78.0 },
                "queue backpressure (soft): memory tier at 78%",
            ),
            (
                QueueError::Backpressure {
                    pct: 99.0,
                    retry_after_ms: 250,
                },
                "queue backpressure (hard): memory tier at 99%; retry after 250ms",
            ),
            (
                QueueError::Persistence("disk unavailable".to_string()),
                "queue persistence failure: disk unavailable",
            ),
            (
                QueueError::LeaseConflict {
                    topic: "orders".to_string(),
                    partition: 1,
                    holder: "consumer-a".to_string(),
                },
                "lease conflict on topic=orders, partition=1: held by consumer-a",
            ),
            (
                QueueError::UnknownMessageId("1:2:3".to_string()),
                "unknown message id: 1:2:3",
            ),
        ];

        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn anyhow_errors_convert_to_other_variant() {
        let error: QueueError = anyhow::anyhow!("boom").into();

        assert_eq!(error.to_string(), "boom");
        assert!(matches!(error, QueueError::Other(_)));
    }
}
