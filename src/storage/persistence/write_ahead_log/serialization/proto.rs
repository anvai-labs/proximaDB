//! Protocol Buffers serialization for WAL
//!
//! This is the default and recommended format for ProximaDB's proto-first architecture.

use anyhow::{Context, Result};
use proximadb_records::ProximaRecord;

/// Protocol Buffers serializer - the default for proto-first architecture
#[derive(Debug, Clone, Default)]
pub struct ProtocolBuffersSerializer;

impl ProtocolBuffersSerializer {
    /// Create a new Protocol Buffers serializer
    pub fn new() -> Self {
        Self
    }
}

impl super::VectorBatchSerializer for ProtocolBuffersSerializer {
    fn serialize_batch(&self, records: &[ProximaRecord]) -> Result<Vec<u8>> {
        bincode::serialize(records)
            .context("Failed to serialize ProximaRecords for canonical WAL proto slot")
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        bincode::deserialize(data)
            .context("Failed to deserialize ProximaRecords from canonical WAL proto slot")
    }

    fn format(&self) -> super::SerializationFormat {
        super::SerializationFormat::ProtocolBuffers
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
    fn test_protocol_buffers_round_trip() {
        let serializer = ProtocolBuffersSerializer::new();
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
    fn test_empty_batch_handling() {
        let serializer = ProtocolBuffersSerializer::new();
        let vectors: Vec<ProximaRecord> = vec![];

        let serialized = serializer
            .serialize_batch(&vectors)
            .expect("Failed to serialize empty batch");
        let deserialized = serializer
            .deserialize_batch(&serialized)
            .expect("Failed to deserialize empty batch");

        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_format_identifier() {
        let serializer = ProtocolBuffersSerializer::new();
        assert_eq!(
            serializer.format(),
            super::super::SerializationFormat::ProtocolBuffers
        );
    }
}
