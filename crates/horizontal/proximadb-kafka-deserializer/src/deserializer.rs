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

//! Message deserialization for Kafka consumers

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use proximadb_kafka_config::config::DeserializationFormat;

/// Deserialization error
#[derive(Debug, Clone)]
pub enum DeserializationError {
    /// JSON parsing error
    JsonError(String),
    /// Avro parsing error
    AvroError(String),
    /// Protobuf parsing error
    ProtobufError(String),
    /// Invalid message format
    InvalidFormat(String),
    /// Missing required field
    MissingField(String),
    /// Invalid vector dimension
    InvalidDimension {
        /// Expected vector dimension.
        expected: usize,
        /// Actual dimension received.
        actual: usize,
    },
    /// Schema registry error
    SchemaRegistryError(String),
}

impl std::fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JsonError(msg) => write!(f, "JSON deserialization error: {}", msg),
            Self::AvroError(msg) => write!(f, "Avro deserialization error: {}", msg),
            Self::ProtobufError(msg) => write!(f, "Protobuf deserialization error: {}", msg),
            Self::InvalidFormat(msg) => write!(f, "Invalid message format: {}", msg),
            Self::MissingField(field) => write!(f, "Missing required field: {}", field),
            Self::InvalidDimension { expected, actual } => {
                write!(
                    f,
                    "Invalid vector dimension: expected {}, got {}",
                    expected, actual
                )
            }
            Self::SchemaRegistryError(msg) => write!(f, "Schema registry error: {}", msg),
        }
    }
}

impl std::error::Error for DeserializationError {}

/// Deserialized vector message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMessage {
    /// Vector ID
    pub id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Target collection (optional, uses default if not specified)
    pub collection: Option<String>,
    /// Metadata key-value pairs
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Message operation type
    #[serde(default)]
    pub operation: MessageOperation,
    /// Timestamp (epoch millis)
    pub timestamp: Option<u64>,
    /// Partition key for routing
    pub partition_key: Option<String>,
}

/// Message operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageOperation {
    /// Insert or update vector
    #[default]
    Upsert,
    /// Insert only (fail if exists)
    Insert,
    /// Update only (fail if not exists)
    Update,
    /// Delete vector
    Delete,
}

/// Message deserializer for Kafka messages
pub struct MessageDeserializer {
    /// Deserialization format
    format: DeserializationFormat,
    /// Expected vector dimension (0 = any)
    expected_dimension: usize,
    /// Schema registry URL (for Avro)
    #[allow(dead_code)]
    schema_registry_url: Option<String>,
}

impl MessageDeserializer {
    /// Create a new deserializer
    pub fn new(format: DeserializationFormat) -> Self {
        Self {
            format,
            expected_dimension: 0,
            schema_registry_url: None,
        }
    }

    /// Create a deserializer with dimension validation
    pub fn with_dimension(format: DeserializationFormat, dimension: usize) -> Self {
        Self {
            format,
            expected_dimension: dimension,
            schema_registry_url: None,
        }
    }

    /// Create a deserializer with Avro schema registry
    pub fn with_schema_registry(schema_registry_url: String) -> Self {
        Self {
            format: DeserializationFormat::Avro,
            expected_dimension: 0,
            schema_registry_url: Some(schema_registry_url),
        }
    }

    /// Deserialize a message
    pub fn deserialize(&self, payload: &[u8]) -> Result<VectorMessage, DeserializationError> {
        let message = match self.format {
            DeserializationFormat::Json => self.deserialize_json(payload)?,
            DeserializationFormat::Avro => self.deserialize_avro(payload)?,
            DeserializationFormat::Protobuf => self.deserialize_protobuf(payload)?,
            DeserializationFormat::Raw => self.deserialize_raw(payload)?,
        };

        // Validate dimension if specified
        if self.expected_dimension > 0 && message.vector.len() != self.expected_dimension {
            return Err(DeserializationError::InvalidDimension {
                expected: self.expected_dimension,
                actual: message.vector.len(),
            });
        }

        Ok(message)
    }

    /// Deserialize a batch of messages
    pub fn deserialize_batch(
        &self,
        payloads: &[&[u8]],
    ) -> Vec<Result<VectorMessage, DeserializationError>> {
        payloads.iter().map(|p| self.deserialize(p)).collect()
    }

    /// Deserialize JSON message
    fn deserialize_json(&self, payload: &[u8]) -> Result<VectorMessage, DeserializationError> {
        // Try standard format first
        if let Ok(msg) = serde_json::from_slice::<VectorMessage>(payload) {
            return Ok(msg);
        }

        // Try alternative formats
        if let Ok(alt) = serde_json::from_slice::<AlternativeJsonFormat>(payload) {
            return Ok(VectorMessage {
                id: alt.id.or(alt.vector_id).ok_or_else(|| {
                    DeserializationError::MissingField("id or vector_id".to_string())
                })?,
                vector: alt.vector.or(alt.embedding).or(alt.values).ok_or_else(|| {
                    DeserializationError::MissingField("vector, embedding, or values".to_string())
                })?,
                collection: alt.collection.or(alt.namespace),
                metadata: alt.metadata.unwrap_or_default(),
                operation: alt.operation.unwrap_or_default(),
                timestamp: alt.timestamp,
                partition_key: alt.partition_key,
            });
        }

        Err(DeserializationError::JsonError(
            String::from_utf8_lossy(payload).to_string(),
        ))
    }

    /// Deserialize Avro message (placeholder - requires apache-avro crate)
    fn deserialize_avro(&self, payload: &[u8]) -> Result<VectorMessage, DeserializationError> {
        // For now, try to parse as JSON (Avro JSON encoding)
        // Full Avro support would require the apache-avro crate and schema registry integration

        // Skip Confluent Schema Registry header (5 bytes) if present
        let data = if payload.len() > 5 && payload[0] == 0 {
            &payload[5..]
        } else {
            payload
        };

        self.deserialize_json(data).map_err(|_| {
            DeserializationError::AvroError(
                "Avro deserialization requires apache-avro crate".to_string(),
            )
        })
    }

    /// Deserialize Protobuf message (placeholder)
    fn deserialize_protobuf(&self, payload: &[u8]) -> Result<VectorMessage, DeserializationError> {
        // Protobuf deserialization would use prost or similar
        // For now, return error indicating feature not implemented
        Err(DeserializationError::ProtobufError(format!(
            "Protobuf deserialization not yet implemented (payload size: {})",
            payload.len()
        )))
    }

    /// Deserialize raw bytes as vector
    fn deserialize_raw(&self, payload: &[u8]) -> Result<VectorMessage, DeserializationError> {
        // Interpret bytes as f32 array
        if !payload.len().is_multiple_of(4) {
            return Err(DeserializationError::InvalidFormat(
                "Raw payload size must be multiple of 4 bytes".to_string(),
            ));
        }

        let vector: Vec<f32> = payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();

        // Generate ID from hash of vector
        let id = format!("raw_{:x}", Self::hash_vector(&vector));

        Ok(VectorMessage {
            id,
            vector,
            collection: None,
            metadata: HashMap::new(),
            operation: MessageOperation::Upsert,
            timestamp: None,
            partition_key: None,
        })
    }

    /// Hash a vector for ID generation
    fn hash_vector(vector: &[f32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in vector {
            ((v * 10000.0) as i32).hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Alternative JSON formats for flexibility
#[derive(Debug, Deserialize)]
struct AlternativeJsonFormat {
    /// Vector ID (standard)
    id: Option<String>,
    /// Alternative: vector_id
    vector_id: Option<String>,
    /// Vector data (standard)
    vector: Option<Vec<f32>>,
    /// Alternative: embedding
    embedding: Option<Vec<f32>>,
    /// Alternative: values
    values: Option<Vec<f32>>,
    /// Collection name
    collection: Option<String>,
    /// Alternative: namespace
    namespace: Option<String>,
    /// Metadata
    metadata: Option<HashMap<String, serde_json::Value>>,
    /// Operation type
    operation: Option<MessageOperation>,
    /// Timestamp
    timestamp: Option<u64>,
    /// Partition key
    partition_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_json_standard() {
        let json = r#"{
            "id": "vec_1",
            "vector": [0.1, 0.2, 0.3],
            "collection": "test",
            "metadata": {"key": "value"}
        }"#;

        let deserializer = MessageDeserializer::new(DeserializationFormat::Json);
        let msg = deserializer.deserialize(json.as_bytes()).unwrap();

        assert_eq!(msg.id, "vec_1");
        assert_eq!(msg.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(msg.collection, Some("test".to_string()));
    }

    #[test]
    fn test_deserialize_json_alternative_fields() {
        let json = r#"{
            "vector_id": "vec_2",
            "embedding": [0.4, 0.5, 0.6],
            "namespace": "alt_collection"
        }"#;

        let deserializer = MessageDeserializer::new(DeserializationFormat::Json);
        let msg = deserializer.deserialize(json.as_bytes()).unwrap();

        assert_eq!(msg.id, "vec_2");
        assert_eq!(msg.vector, vec![0.4, 0.5, 0.6]);
        assert_eq!(msg.collection, Some("alt_collection".to_string()));
    }

    #[test]
    fn test_deserialize_json_with_operation() {
        let json = r#"{
            "id": "vec_3",
            "vector": [0.1, 0.2],
            "operation": "delete"
        }"#;

        let deserializer = MessageDeserializer::new(DeserializationFormat::Json);
        let msg = deserializer.deserialize(json.as_bytes()).unwrap();

        assert_eq!(msg.operation, MessageOperation::Delete);
    }

    #[test]
    fn test_dimension_validation() {
        let json = r#"{"id": "vec_1", "vector": [0.1, 0.2, 0.3]}"#;

        let deserializer = MessageDeserializer::with_dimension(DeserializationFormat::Json, 4);
        let result = deserializer.deserialize(json.as_bytes());

        assert!(matches!(
            result,
            Err(DeserializationError::InvalidDimension {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn test_deserialize_raw() {
        // 3 floats: 1.0, 2.0, 3.0
        let bytes: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let deserializer = MessageDeserializer::new(DeserializationFormat::Raw);
        let msg = deserializer.deserialize(&bytes).unwrap();

        assert_eq!(msg.vector, vec![1.0, 2.0, 3.0]);
        assert!(msg.id.starts_with("raw_"));
    }

    #[test]
    fn test_deserialize_raw_invalid_size() {
        let bytes = vec![0u8, 1, 2]; // Not multiple of 4

        let deserializer = MessageDeserializer::new(DeserializationFormat::Raw);
        let result = deserializer.deserialize(&bytes);

        assert!(matches!(
            result,
            Err(DeserializationError::InvalidFormat(_))
        ));
    }

    #[test]
    fn test_deserialize_batch() {
        let payloads: Vec<&[u8]> = vec![
            br#"{"id": "v1", "vector": [0.1]}"#,
            br#"{"id": "v2", "vector": [0.2]}"#,
            br#"invalid json"#,
        ];

        let deserializer = MessageDeserializer::new(DeserializationFormat::Json);
        let results = deserializer.deserialize_batch(&payloads);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
    }

    #[test]
    fn test_message_operation_default() {
        let op = MessageOperation::default();
        assert_eq!(op, MessageOperation::Upsert);
    }

    #[test]
    fn test_vector_message_serialization() {
        let msg = VectorMessage {
            id: "test".to_string(),
            vector: vec![0.1, 0.2],
            collection: Some("col".to_string()),
            metadata: HashMap::new(),
            operation: MessageOperation::Insert,
            timestamp: Some(12345),
            partition_key: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"id\":\"test\""));
        assert!(json.contains("\"operation\":\"insert\""));
    }
}
