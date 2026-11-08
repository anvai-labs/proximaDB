//! "Protocol Buffers" slot serializer for WAL
//!
//! TD-WAL-6 disclosure: despite the name, this serializer emits the shared
//! **v1 bincode encoding** (`serialize_bincode_batch`), not protobuf bytes.
//! The type exists as the named slot for `SerializationFormat::ProtocolBuffers`
//! so config strings and recovery keep selecting it. It also does NOT
//! override the v2 wire path — non-Fp32 records re-trigger the v1
//! fp32-only refuse-error (use `SerializationFormat::Bincode` for
//! precision-aware ingest).

use anyhow::Result;
use proximadb_records::ProximaRecord;

/// Serializer for the canonical "proto" WAL slot (bincode-backed, TD-WAL-6)
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
        super::serialize_bincode_batch(records, "canonical WAL proto slot")
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        super::deserialize_bincode_batch(data, "canonical WAL proto slot")
    }

    fn format(&self) -> super::SerializationFormat {
        super::SerializationFormat::ProtocolBuffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::serialization::VectorBatchSerializer;
    use crate::storage::persistence::write_ahead_log::serialization::test_util::create_test_vector;

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
