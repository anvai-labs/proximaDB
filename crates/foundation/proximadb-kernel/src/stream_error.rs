//! Error types for streaming operations

use std::fmt;

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur during streaming operations
#[derive(Debug, Clone)]
pub enum StreamError {
    TooManySessions {
        max: usize,
        current: usize,
    },
    SessionNotFound {
        session_id: String,
    },
    SessionClosed {
        session_id: String,
    },
    RateLimited {
        current_rate: u64,
        max_rate: u64,
        retry_after_ms: u64,
    },
    BufferFull {
        capacity: usize,
        dropped: usize,
    },
    InvalidConfig {
        message: String,
    },
    CollectionNotFound {
        collection: String,
    },
    StorageError {
        message: String,
    },
    SerializationError {
        message: String,
    },
    ConnectionError {
        message: String,
    },
    Timeout {
        operation: String,
        timeout_ms: u64,
    },
    Internal {
        message: String,
    },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::TooManySessions { max, current } => {
                write!(
                    f,
                    "Too many concurrent sessions: {} (max: {})",
                    current, max
                )
            }
            StreamError::SessionNotFound { session_id } => {
                write!(f, "Session not found: {}", session_id)
            }
            StreamError::SessionClosed { session_id } => {
                write!(f, "Session is closed: {}", session_id)
            }
            StreamError::RateLimited {
                current_rate,
                max_rate,
                retry_after_ms,
            } => {
                write!(
                    f,
                    "Rate limit exceeded: {}/s (max: {}/s), retry after {}ms",
                    current_rate, max_rate, retry_after_ms
                )
            }
            StreamError::BufferFull { capacity, dropped } => {
                write!(
                    f,
                    "Buffer full (capacity: {}), dropped {} records",
                    capacity, dropped
                )
            }
            StreamError::InvalidConfig { message } => {
                write!(f, "Invalid configuration: {}", message)
            }
            StreamError::CollectionNotFound { collection } => {
                write!(f, "Collection not found: {}", collection)
            }
            StreamError::StorageError { message } => write!(f, "Storage error: {}", message),
            StreamError::SerializationError { message } => {
                write!(f, "Serialization error: {}", message)
            }
            StreamError::ConnectionError { message } => {
                write!(f, "Connection error: {}", message)
            }
            StreamError::Timeout {
                operation,
                timeout_ms,
            } => {
                write!(f, "Timeout after {}ms: {}", timeout_ms, operation)
            }
            StreamError::Internal { message } => write!(f, "Internal error: {}", message),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<String> for StreamError {
    fn from(message: String) -> Self {
        StreamError::Internal { message }
    }
}

impl From<&str> for StreamError {
    fn from(message: &str) -> Self {
        StreamError::Internal {
            message: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_error_display_covers_all_streaming_failure_shapes() {
        let messages = vec![
            StreamError::TooManySessions {
                max: 10,
                current: 11,
            }
            .to_string(),
            StreamError::SessionNotFound {
                session_id: "s1".to_string(),
            }
            .to_string(),
            StreamError::SessionClosed {
                session_id: "s2".to_string(),
            }
            .to_string(),
            StreamError::RateLimited {
                current_rate: 50,
                max_rate: 10,
                retry_after_ms: 250,
            }
            .to_string(),
            StreamError::BufferFull {
                capacity: 1024,
                dropped: 3,
            }
            .to_string(),
            StreamError::InvalidConfig {
                message: "bad".to_string(),
            }
            .to_string(),
            StreamError::CollectionNotFound {
                collection: "docs".to_string(),
            }
            .to_string(),
            StreamError::StorageError {
                message: "disk".to_string(),
            }
            .to_string(),
            StreamError::SerializationError {
                message: "json".to_string(),
            }
            .to_string(),
            StreamError::ConnectionError {
                message: "closed".to_string(),
            }
            .to_string(),
            StreamError::Timeout {
                operation: "flush".to_string(),
                timeout_ms: 500,
            }
            .to_string(),
            StreamError::Internal {
                message: "boom".to_string(),
            }
            .to_string(),
        ];

        assert!(messages.iter().any(|msg| msg.contains("Too many")));
        assert!(messages.iter().any(|msg| msg.contains("retry after 250ms")));
        assert!(
            messages
                .iter()
                .any(|msg| msg.contains("Timeout after 500ms"))
        );
    }

    #[test]
    fn stream_error_from_string_and_str_create_internal_errors() {
        let from_string = StreamError::from("owned".to_string());
        let from_str = StreamError::from("borrowed");
        let result: StreamResult<()> = Err(from_str.clone());

        assert!(matches!(from_string, StreamError::Internal { message } if message == "owned"));
        assert!(matches!(from_str, StreamError::Internal { message } if message == "borrowed"));
        assert!(matches!(result, Err(error) if error.to_string().contains("borrowed")));
    }
}
