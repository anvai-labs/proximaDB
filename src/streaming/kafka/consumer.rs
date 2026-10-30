/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Kafka Vector Consumer
//!
//! High-level consumer for ingesting vectors from Kafka topics.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info};

use super::config::KafkaConsumerConfig;
use super::deserializer::{MessageDeserializer, VectorMessage};
use crate::streaming::{BackpressureLevel, StreamCoordinator, StreamId};

/// Consumer handle for controlling the consumer
pub struct ConsumerHandle {
    /// Consumer ID
    pub id: String,
    /// Shutdown signal sender
    shutdown_tx: mpsc::Sender<()>,
    /// Status receiver
    status_rx: mpsc::Receiver<ConsumerStatus>,
}

impl ConsumerHandle {
    /// Signal the consumer to stop
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Get current consumer status
    pub async fn status(&mut self) -> Option<ConsumerStatus> {
        self.status_rx.try_recv().ok()
    }
}

/// Consumer status
#[derive(Debug, Clone)]
pub struct ConsumerStatus {
    /// Whether consumer is running
    pub running: bool,
    /// Current partition assignments
    pub partitions: Vec<i32>,
    /// Messages consumed
    pub messages_consumed: u64,
    /// Messages processed successfully
    pub messages_processed: u64,
    /// Messages failed
    pub messages_failed: u64,
    /// Current lag (approximate)
    pub lag: Option<u64>,
    /// Last error (if any)
    pub last_error: Option<String>,
    /// Current backpressure level
    pub backpressure: BackpressureLevel,
}

/// Kafka vector consumer
///
/// This consumer integrates with ProximaDB's streaming infrastructure
/// to ingest vectors from Kafka topics.
pub struct KafkaVectorConsumer {
    /// Configuration
    config: KafkaConsumerConfig,
    /// Stream coordinator
    coordinator: Arc<StreamCoordinator>,
    /// Message deserializer
    deserializer: MessageDeserializer,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Metrics
    metrics: Arc<ConsumerMetrics>,
    /// Active session per collection
    sessions: Arc<RwLock<HashMap<String, StreamId>>>,
}

/// Consumer metrics
struct ConsumerMetrics {
    messages_received: AtomicU64,
    messages_processed: AtomicU64,
    messages_failed: AtomicU64,
    bytes_received: AtomicU64,
    batches_processed: AtomicU64,
    deserialization_errors: AtomicU64,
    commits: AtomicU64,
    dlq_messages: AtomicU64,
}

impl Default for ConsumerMetrics {
    fn default() -> Self {
        Self {
            messages_received: AtomicU64::new(0),
            messages_processed: AtomicU64::new(0),
            messages_failed: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            batches_processed: AtomicU64::new(0),
            deserialization_errors: AtomicU64::new(0),
            commits: AtomicU64::new(0),
            dlq_messages: AtomicU64::new(0),
        }
    }
}

impl KafkaVectorConsumer {
    /// Create a new Kafka vector consumer
    pub fn new(config: KafkaConsumerConfig, coordinator: Arc<StreamCoordinator>) -> Self {
        let deserializer = MessageDeserializer::new(config.format);

        Self {
            config,
            coordinator,
            deserializer,
            running: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(ConsumerMetrics::default()),
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start the consumer (mock implementation without rdkafka)
    ///
    /// Note: This is a simulated consumer for testing purposes.
    /// Production deployment requires the rdkafka crate.
    pub async fn start(self: Arc<Self>) -> Result<ConsumerHandle, ConsumerError> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| ConsumerError::ConfigError(e.to_string()))?;

        if self.running.swap(true, Ordering::SeqCst) {
            return Err(ConsumerError::AlreadyRunning);
        }

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let (status_tx, status_rx) = mpsc::channel::<ConsumerStatus>(10);

        let consumer_id = format!(
            "consumer_{}_{}",
            self.config.group.group_id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let consumer = self.clone();
        let id = consumer_id.clone();

        // Spawn consumer task
        tokio::spawn(async move {
            info!(
                "Starting Kafka consumer {} for topic {}",
                id, consumer.config.topic
            );

            // Simulate consumer loop
            let mut interval = tokio::time::interval(consumer.config.poll_timeout);

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Consumer {} received shutdown signal", id);
                        break;
                    }
                    _ = interval.tick() => {
                        // In production, this would poll Kafka
                        // For now, just send status updates
                        let status = ConsumerStatus {
                            running: consumer.running.load(Ordering::Relaxed),
                            partitions: vec![0, 1, 2], // Simulated
                            messages_consumed: consumer.metrics.messages_received.load(Ordering::Relaxed),
                            messages_processed: consumer.metrics.messages_processed.load(Ordering::Relaxed),
                            messages_failed: consumer.metrics.messages_failed.load(Ordering::Relaxed),
                            lag: Some(0),
                            last_error: None,
                            backpressure: BackpressureLevel::None,
                        };

                        let _ = status_tx.try_send(status);
                    }
                }
            }

            consumer.running.store(false, Ordering::SeqCst);
            info!("Consumer {} stopped", id);
        });

        Ok(ConsumerHandle {
            id: consumer_id,
            shutdown_tx,
            status_rx,
        })
    }

    /// Process a batch of messages (for testing or external integration)
    pub async fn process_messages(
        &self,
        messages: Vec<(Vec<u8>, Option<String>)>, // (payload, key)
    ) -> BatchResult {
        let start = Instant::now();
        let mut successful = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        // Deserialize messages
        let mut vectors_by_collection: HashMap<String, Vec<VectorMessage>> = HashMap::new();

        for (payload, _key) in &messages {
            self.metrics
                .messages_received
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .bytes_received
                .fetch_add(payload.len() as u64, Ordering::Relaxed);

            match self.deserializer.deserialize(payload) {
                Ok(msg) => {
                    let collection = msg
                        .collection
                        .clone()
                        .or_else(|| self.config.default_collection.clone())
                        .unwrap_or_else(|| "default".to_string());

                    vectors_by_collection
                        .entry(collection)
                        .or_default()
                        .push(msg);
                }
                Err(e) => {
                    self.metrics
                        .deserialization_errors
                        .fetch_add(1, Ordering::Relaxed);
                    errors.push(ProcessingError {
                        message_index: messages.len() - 1,
                        error: e.to_string(),
                    });
                    failed += 1;
                }
            }
        }

        // Push to coordinator by collection
        for (collection, vectors) in vectors_by_collection {
            match self.push_to_collection(&collection, vectors).await {
                Ok(count) => successful += count,
                Err(e) => {
                    errors.push(ProcessingError {
                        message_index: 0, // Batch error
                        error: e.to_string(),
                    });
                    failed += 1;
                }
            }
        }

        self.metrics
            .messages_processed
            .fetch_add(successful as u64, Ordering::Relaxed);
        self.metrics
            .messages_failed
            .fetch_add(failed as u64, Ordering::Relaxed);
        self.metrics
            .batches_processed
            .fetch_add(1, Ordering::Relaxed);

        BatchResult {
            successful,
            failed,
            errors,
            processing_time: start.elapsed(),
        }
    }

    /// Push vectors to a collection via stream coordinator
    async fn push_to_collection(
        &self,
        collection: &str,
        vectors: Vec<VectorMessage>,
    ) -> Result<usize, ConsumerError> {
        // Get or create session for this collection
        let session_id = {
            let sessions = self.sessions.read().await;
            sessions.get(collection).cloned()
        };

        let session_id = match session_id {
            Some(id) => id,
            None => {
                // Create new session
                let id = self
                    .coordinator
                    .create_session(
                        collection.to_string(),
                        crate::streaming::SessionConfig::default(),
                    )
                    .await
                    .map_err(|e| ConsumerError::StreamError(e.to_string()))?;

                let mut sessions = self.sessions.write().await;
                sessions.insert(collection.to_string(), id.clone());
                id
            }
        };

        // Convert VectorMessage to canonical ProximaRecord at Kafka boundary
        let records: Vec<proximadb_records::ProximaRecord> = vectors
            .into_iter()
            .map(|v| {
                let dim = v.vector.len() as u32;
                let mut props = proximadb_records::ProximaTree::new();
                for (k, jv) in v.metadata {
                    let pv = json_to_proxima_value(&jv);
                    props.insert(k, proximadb_records::ProximaTreeNode::Value(pv));
                }
                let ts_ns = v
                    .timestamp
                    .map(|t| (t as i64) * 1_000_000_000)
                    .unwrap_or_else(|| chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
                proximadb_records::ProximaRecord {
                    oid: v.id,
                    embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                        "default", "vector", dim, v.vector,
                    )],
                    props,
                    record_version: 1,
                    created_at_ns: ts_ns,
                    updated_at_ns: ts_ns,
                    ..Default::default()
                }
            })
            .collect();

        let count = records.len();

        // Push to coordinator
        match self.coordinator.push_records(&session_id, records).await {
            Ok(result) => {
                if result.backpressure != BackpressureLevel::None {
                    debug!(
                        "Backpressure from collection {}: {:?}",
                        collection, result.backpressure
                    );
                }
                Ok(count)
            }
            Err(e) => Err(ConsumerError::StreamError(e.to_string())),
        }
    }

    /// Get consumer metrics
    pub fn metrics(&self) -> ConsumerMetricsSnapshot {
        ConsumerMetricsSnapshot {
            messages_received: self.metrics.messages_received.load(Ordering::Relaxed),
            messages_processed: self.metrics.messages_processed.load(Ordering::Relaxed),
            messages_failed: self.metrics.messages_failed.load(Ordering::Relaxed),
            bytes_received: self.metrics.bytes_received.load(Ordering::Relaxed),
            batches_processed: self.metrics.batches_processed.load(Ordering::Relaxed),
            deserialization_errors: self.metrics.deserialization_errors.load(Ordering::Relaxed),
            commits: self.metrics.commits.load(Ordering::Relaxed),
            dlq_messages: self.metrics.dlq_messages.load(Ordering::Relaxed),
        }
    }

    /// Check if consumer is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// Convert serde_json Value to ProximaValue at the protocol boundary
fn json_to_proxima_value(v: &serde_json::Value) -> proximadb_data_model::ProximaValue {
    match v {
        serde_json::Value::String(s) => proximadb_data_model::ProximaValue::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                proximadb_data_model::ProximaValue::Int64(i)
            } else {
                proximadb_data_model::ProximaValue::Float64(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::Bool(b) => proximadb_data_model::ProximaValue::Boolean(*b),
        _ => proximadb_data_model::ProximaValue::String(v.to_string()),
    }
}

/// Consumer error
#[derive(Debug, Clone)]
pub enum ConsumerError {
    /// Configuration error
    ConfigError(String),
    /// Consumer already running
    AlreadyRunning,
    /// Kafka connection error
    ConnectionError(String),
    /// Stream coordinator error
    StreamError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Commit error
    CommitError(String),
}

impl std::fmt::Display for ConsumerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::AlreadyRunning => write!(f, "Consumer already running"),
            Self::ConnectionError(msg) => write!(f, "Kafka connection error: {}", msg),
            Self::StreamError(msg) => write!(f, "Stream coordinator error: {}", msg),
            Self::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            Self::CommitError(msg) => write!(f, "Commit error: {}", msg),
        }
    }
}

impl std::error::Error for ConsumerError {}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of successfully processed messages
    pub successful: usize,
    /// Number of failed messages
    pub failed: usize,
    /// Individual errors
    pub errors: Vec<ProcessingError>,
    /// Total processing time
    pub processing_time: Duration,
}

/// Individual message processing error
#[derive(Debug, Clone)]
pub struct ProcessingError {
    /// Index of failed message in batch
    pub message_index: usize,
    /// Error description
    pub error: String,
}

/// Snapshot of consumer metrics
#[derive(Debug, Clone)]
pub struct ConsumerMetricsSnapshot {
    /// Total messages received
    pub messages_received: u64,
    /// Messages processed successfully
    pub messages_processed: u64,
    /// Messages failed
    pub messages_failed: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Batches processed
    pub batches_processed: u64,
    /// Deserialization errors
    pub deserialization_errors: u64,
    /// Offset commits
    pub commits: u64,
    /// Messages sent to DLQ
    pub dlq_messages: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamConfig;

    fn create_test_consumer() -> Arc<KafkaVectorConsumer> {
        let config = KafkaConsumerConfig::local("test-topic", "test-group");
        let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
        Arc::new(KafkaVectorConsumer::new(config, coordinator))
    }

    #[tokio::test]
    async fn test_consumer_creation() {
        let consumer = create_test_consumer();
        assert!(!consumer.is_running());
    }

    #[tokio::test]
    async fn test_process_json_messages() {
        let consumer = create_test_consumer();

        let messages = vec![
            (
                br#"{"id": "v1", "vector": [0.1, 0.2, 0.3], "collection": "test"}"#.to_vec(),
                None,
            ),
            (
                br#"{"id": "v2", "vector": [0.4, 0.5, 0.6], "collection": "test"}"#.to_vec(),
                None,
            ),
        ];

        let result = consumer.process_messages(messages).await;

        assert_eq!(result.successful, 2);
        assert_eq!(result.failed, 0);
    }

    #[tokio::test]
    async fn test_process_invalid_messages() {
        let consumer = create_test_consumer();

        let messages = vec![
            (br#"{"id": "v1", "vector": [0.1]}"#.to_vec(), None),
            (br#"invalid json"#.to_vec(), None),
        ];

        let result = consumer.process_messages(messages).await;

        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 1);
    }

    #[tokio::test]
    async fn test_consumer_start_stop() {
        let consumer = create_test_consumer();

        let handle = consumer.clone().start().await.unwrap();

        assert!(consumer.is_running());

        handle.stop().await;

        // Give it a moment to stop
        tokio::time::sleep(Duration::from_millis(200)).await;

        assert!(!consumer.is_running());
    }

    #[tokio::test]
    async fn test_consumer_metrics() {
        let consumer = create_test_consumer();

        let messages = vec![(
            br#"{"id": "v1", "vector": [0.1, 0.2], "collection": "test"}"#.to_vec(),
            None,
        )];

        consumer.process_messages(messages).await;

        let metrics = consumer.metrics();
        assert_eq!(metrics.messages_received, 1);
        assert_eq!(metrics.messages_processed, 1);
        assert_eq!(metrics.batches_processed, 1);
    }

    #[tokio::test]
    async fn test_default_collection_fallback() {
        let mut config = KafkaConsumerConfig::local("test-topic", "test-group");
        config.default_collection = Some("fallback".to_string());

        let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
        let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

        // Message without collection
        let messages = vec![(br#"{"id": "v1", "vector": [0.1]}"#.to_vec(), None)];

        let result = consumer.process_messages(messages).await;
        assert_eq!(result.successful, 1);
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult {
            successful: 10,
            failed: 2,
            errors: vec![ProcessingError {
                message_index: 5,
                error: "test error".to_string(),
            }],
            processing_time: Duration::from_millis(100),
        };

        assert_eq!(result.successful, 10);
        assert_eq!(result.failed, 2);
        assert_eq!(result.errors.len(), 1);
    }
}
