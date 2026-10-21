//! Avro serialization for WAL
//!
//! Provides schema evolution support for backward compatibility.

use anyhow::Result;
use proximadb_records::ProximaRecord;

/// Avro serializer - for schema evolution support
#[derive(Debug, Clone, Default)]
pub struct AvroSerializer;

impl AvroSerializer {
    /// Create a new Avro serializer
    pub fn new() -> Self {
        Self
    }
}

impl super::VectorBatchSerializer for AvroSerializer {
    fn serialize_batch(&self, records: &[ProximaRecord]) -> Result<Vec<u8>> {
        bincode::serialize(records).map_err(|e| {
            anyhow::anyhow!("Failed to serialize ProximaRecords to Avro WAL slot: {}", e)
        })
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        bincode::deserialize(data).map_err(|e| {
            anyhow::anyhow!(
                "Failed to deserialize ProximaRecords from Avro WAL slot: {}",
                e
            )
        })
    }

    fn format(&self) -> super::SerializationFormat {
        super::SerializationFormat::Avro
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::serialization::VectorBatchSerializer;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTree, ProximaTreeNode};

    fn create_test_vector() -> ProximaRecord {
        let mut props = ProximaTree::new();
        props.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("test".to_string())),
        );

        ProximaRecord {
            oid: "test_vector_1".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            props,
            record_version: 1,
            ..Default::default()
        }
    }

    #[test]
    fn test_avro_round_trip() {
        let serializer = AvroSerializer::new();
        let vectors = vec![create_test_vector()];

        // Serialize
        let serialized = serializer
            .serialize_batch(&vectors)
            .expect("Failed to serialize batch");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized = serializer
            .deserialize_batch(&serialized)
            .expect("Failed to deserialize batch");
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized[0].oid, vectors[0].oid);
        assert_eq!(
            deserialized[0].embeddings.first().map(|e| e.values.clone()),
            vectors[0].embeddings.first().map(|e| e.values.clone())
        );
    }

    #[test]
    fn test_metadata_preservation() {
        let serializer = AvroSerializer::new();
        let mut vector = create_test_vector();
        {
            let mut props = ProximaTree::new();
            props.insert(
                "key1".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("value1".to_string())),
            );
            props.insert(
                "key2".to_string(),
                ProximaTreeNode::Value(ProximaValue::String("value2".to_string())),
            );
            vector.props = props;
        }

        let vectors = vec![vector];
        let serialized = serializer
            .serialize_batch(&vectors)
            .expect("Failed to serialize batch");
        let deserialized = serializer
            .deserialize_batch(&serialized)
            .expect("Failed to deserialize batch");

        assert_eq!(deserialized[0].props.len(), 2);

        // Check that both keys are present (order doesn't matter due to HashMap)
        assert!(deserialized[0].props.contains_key("key1"));
        assert!(deserialized[0].props.contains_key("key2"));

        // Find and verify each key-value pair
        let key1_value = deserialized[0].props.get("key1").unwrap();
        let key2_value = deserialized[0].props.get("key2").unwrap();
        assert!(
            matches!(key1_value, ProximaTreeNode::Value(ProximaValue::String(s)) if s == "value1")
        );
        assert!(
            matches!(key2_value, ProximaTreeNode::Value(ProximaValue::String(s)) if s == "value2")
        );
    }

    #[test]
    fn test_format_identifier() {
        let serializer = AvroSerializer::new();
        assert_eq!(serializer.format(), super::super::SerializationFormat::Avro);
    }
}
