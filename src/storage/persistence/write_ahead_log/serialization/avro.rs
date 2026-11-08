//! "Avro" slot serializer for WAL
//!
//! TD-WAL-6 disclosure: despite the name, this serializer emits the shared
//! **v1 bincode encoding** (`serialize_bincode_batch`), not Avro bytes —
//! there is no schema evolution. The type exists as the named slot for
//! `SerializationFormat::Avro` so config strings and the memtable WAL
//! behavior keep selecting it. Like the proto slot it does NOT override the
//! v2 wire path — non-Fp32 records re-trigger the v1 fp32-only
//! refuse-error (use `SerializationFormat::Bincode` for precision-aware
//! ingest).

use anyhow::Result;
use proximadb_records::ProximaRecord;

/// Serializer for the "avro" WAL slot (bincode-backed, TD-WAL-6)
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
        super::serialize_bincode_batch(records, "Avro WAL slot")
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        super::deserialize_bincode_batch(data, "Avro WAL slot")
    }

    fn format(&self) -> super::SerializationFormat {
        super::SerializationFormat::Avro
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::serialization::VectorBatchSerializer;
    use crate::storage::persistence::write_ahead_log::serialization::test_util::create_test_vector;
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{ProximaTree, ProximaTreeNode};

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
