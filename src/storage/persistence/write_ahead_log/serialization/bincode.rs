//! Bincode serialization for WAL
//!
//! Provides maximum performance for native Rust serialization.

use anyhow::{Context, Result};
use proximadb_records::ProximaRecord;

/// Bincode serializer - optimized for performance
#[derive(Debug, Clone, Default)]
pub struct BincodeSerializer;

impl BincodeSerializer {
    /// Create a new Bincode serializer
    pub fn new() -> Self {
        Self
    }
}

impl super::VectorBatchSerializer for BincodeSerializer {
    fn serialize_batch(&self, records: &[ProximaRecord]) -> Result<Vec<u8>> {
        bincode::serialize(records).context("Failed to serialize ProximaRecords to Bincode format")
    }

    fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        bincode::deserialize(data).context("Failed to deserialize Bincode ProximaRecords")
    }

    fn serialize_batch_v2(&self, records: &[ProximaRecord]) -> Result<Vec<u8>> {
        let v2: Vec<proximadb_records::wire_v2::ProximaRecordV2> =
            records.iter().map(Into::into).collect();
        bincode::serialize(&v2).context(
            "Failed to serialize ProximaRecords to Bincode v2 format (enum-aware embeddings)",
        )
    }

    fn deserialize_batch_v2(&self, data: &[u8]) -> Result<Vec<ProximaRecord>> {
        let v2: Vec<proximadb_records::wire_v2::ProximaRecordV2> = bincode::deserialize(data)
            .context("Failed to deserialize Bincode v2 ProximaRecords")?;
        Ok(v2.into_iter().map(Into::into).collect())
    }

    fn format(&self) -> super::SerializationFormat {
        super::SerializationFormat::Bincode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::serialization::VectorBatchSerializer;
    use proximadb_records::{EmbeddingCell, ProximaRecord};

    fn create_test_vector() -> ProximaRecord {
        ProximaRecord {
            oid: "test_vector_1".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            origin: Some("test".to_string()),
            record_version: 1,
            ..Default::default()
        }
    }

    #[test]
    fn test_bincode_round_trip() {
        let serializer = BincodeSerializer::new();
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
    fn test_multiple_vectors_batch() {
        let serializer = BincodeSerializer::new();
        let vectors = vec![
            create_test_vector(),
            create_test_vector(),
            create_test_vector(),
        ];

        let serialized = serializer
            .serialize_batch(&vectors)
            .expect("Failed to serialize batch");
        let deserialized = serializer
            .deserialize_batch(&serialized)
            .expect("Failed to deserialize batch");

        assert_eq!(deserialized.len(), 3);
    }

    #[test]
    fn test_high_dimensional_vector() {
        let serializer = BincodeSerializer::new();
        let mut vector = create_test_vector();
        vector.embeddings = vec![EmbeddingCell::new_fp32(
            "default",
            "vector",
            1024,
            vec![0.1; 1024],
        )];

        let vectors = vec![vector];
        let serialized = serializer
            .serialize_batch(&vectors)
            .expect("Failed to serialize high-dimensional vector");
        let deserialized = serializer
            .deserialize_batch(&serialized)
            .expect("Failed to deserialize high-dimensional vector");

        assert_eq!(
            deserialized[0].embeddings.first().map(|e| e.values.len()),
            Some(1024)
        );
    }

    #[test]
    fn test_format_identifier() {
        let serializer = BincodeSerializer::new();
        assert_eq!(
            serializer.format(),
            super::super::SerializationFormat::Bincode
        );
    }

    // === PR 2: schema_version dispatch (LLD §schema-version-dispatch) ===

    #[test]
    fn deserialize_batch_default_stamps_v1() {
        let serializer = BincodeSerializer::new();
        let bytes = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let records = serializer.deserialize_batch(&bytes).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].schema_version,
            proximadb_records::schema_version::V1,
            "PR 2 WAL frames must read as V1 by default"
        );
    }

    #[test]
    fn deserialize_batch_with_v1_hint_stamps_v1() {
        let serializer = BincodeSerializer::new();
        let bytes = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let records = serializer
            .deserialize_batch_with_schema_version(&bytes, proximadb_records::schema_version::V1)
            .unwrap();
        assert_eq!(
            records[0].schema_version,
            proximadb_records::schema_version::V1
        );
    }

    #[test]
    fn deserialize_batch_with_v2_hint_stamps_v2() {
        // PR 2: on-disk format is unchanged (writer is still V1). The hint is
        // what the caller (segment header in PR 4) declares the segment to
        // be. Stamping V2 here is the dispatch site future PRs will use to
        // pick a precision-aware decoder.
        let serializer = BincodeSerializer::new();
        let bytes = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let records = serializer
            .deserialize_batch_with_schema_version(&bytes, proximadb_records::schema_version::V2)
            .unwrap();
        assert_eq!(
            records[0].schema_version,
            proximadb_records::schema_version::V2
        );
    }

    // === INT-2a: v2 segment header round-trip via the generic trait helpers ===

    fn sample_v2_header()
    -> crate::storage::persistence::write_ahead_log::v2_segment_header::V2SegmentHeader {
        crate::storage::persistence::write_ahead_log::v2_segment_header::V2SegmentHeader {
            flags: 0,
            segment_id: 0xDEAD_BEEF_CAFE_F00D_u128,
            created_at_ns: 1_700_000_000_000_000_000,
            canonical_default_precision: proximadb_records::EmbeddingScalarType::Fp16,
            precision_epoch: 7,
            policy_id: "tenant-acme/precision-fp16".to_string(),
            policy_version: 3,
        }
    }

    #[test]
    fn serialize_with_v2_header_round_trips_via_auto_dispatch() {
        let serializer = BincodeSerializer::new();
        let header = sample_v2_header();
        let records = vec![create_test_vector()];

        let bytes = serializer
            .serialize_batch_with_v2_segment_header(&records, &header)
            .unwrap();
        // Sanity: blob starts with PWAL magic so the auto-dispatch will
        // route to the v2 path.
        assert_eq!(
            &bytes[..4],
            crate::storage::persistence::write_ahead_log::v2_segment_header::PWAL_MAGIC,
            "v2 header must prepend the PWAL magic"
        );

        let (decoded, parsed_header) = serializer.deserialize_batch_auto(&bytes).unwrap();
        let parsed = parsed_header.expect("v2 segment must return a parsed header");
        assert_eq!(parsed, header, "header must round-trip byte-identical");
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded[0].schema_version,
            proximadb_records::schema_version::V2,
            "v2 segment must stamp records as V2"
        );
        // Per-record bytes are unchanged in INT-2a; the values themselves
        // round-trip identically to the v1 path.
        assert_eq!(decoded[0].oid, records[0].oid);
        assert_eq!(
            decoded[0].embeddings[0].values,
            records[0].embeddings[0].values,
        );
    }

    #[test]
    fn auto_dispatch_on_legacy_v1_bytes_stamps_v1() {
        // Existing v1 WAL files have no PWAL prefix. The auto-dispatch
        // must treat them as legacy and stamp V1.
        let serializer = BincodeSerializer::new();
        let bytes = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let (decoded, header) = serializer.deserialize_batch_auto(&bytes).unwrap();
        assert!(header.is_none(), "v1 bytes must not produce a header");
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded[0].schema_version,
            proximadb_records::schema_version::V1,
        );
    }

    #[test]
    fn auto_dispatch_on_empty_input_treats_as_v1_error() {
        // Empty input is too short for the PWAL peek; auto-dispatch
        // routes to the legacy path which fails to deserialize. The
        // failure must surface to the caller, not panic.
        let serializer = BincodeSerializer::new();
        assert!(serializer.deserialize_batch_auto(&[]).is_err());
    }

    #[test]
    fn auto_dispatch_on_short_pwal_like_prefix_treats_as_v1() {
        // Three bytes starting with "PWA" but not 4-byte aligned to
        // PWAL must be treated as v1 (legacy) rather than crash.
        let serializer = BincodeSerializer::new();
        assert!(serializer.deserialize_batch_auto(b"PWA").is_err());
    }

    #[test]
    fn v2_header_round_trip_preserves_canonical_precision_for_int3() {
        // INT-3's PAX writer keys on the parsed header's
        // canonical_default_precision. Verify that survives the
        // round-trip — corruption here would cascade into mis-routed
        // PAX writes.
        let serializer = BincodeSerializer::new();
        let mut header = sample_v2_header();
        header.canonical_default_precision = proximadb_records::EmbeddingScalarType::Int8Scalar;
        let records = vec![create_test_vector()];

        let bytes = serializer
            .serialize_batch_with_v2_segment_header(&records, &header)
            .unwrap();
        let (_, parsed) = serializer.deserialize_batch_auto(&bytes).unwrap();
        let parsed = parsed.unwrap();
        assert_eq!(
            parsed.canonical_default_precision,
            proximadb_records::EmbeddingScalarType::Int8Scalar,
            "INT-3 reads this field to pick the PAX layout — it must round-trip exactly"
        );
        assert_eq!(parsed.precision_epoch, 7);
        assert_eq!(parsed.policy_id, "tenant-acme/precision-fp16");
        assert_eq!(parsed.policy_version, 3);
    }

    #[test]
    fn mixed_v1_v2_files_read_independently_via_auto_dispatch() {
        // The rolling-deploy contract: a single replay/recovery pass
        // must handle both v1 files (pre-INT-2a writers) and v2 files
        // (post-INT-2a writers with the flag on). Each file's records
        // get stamped with the correct schema_version automatically.
        let serializer = BincodeSerializer::new();
        let v1_bytes = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let v2_bytes = serializer
            .serialize_batch_with_v2_segment_header(&[create_test_vector()], &sample_v2_header())
            .unwrap();

        let (v1_decoded, v1_header) = serializer.deserialize_batch_auto(&v1_bytes).unwrap();
        let (v2_decoded, v2_header) = serializer.deserialize_batch_auto(&v2_bytes).unwrap();

        assert!(v1_header.is_none());
        assert!(v2_header.is_some());
        assert_eq!(
            v1_decoded[0].schema_version,
            proximadb_records::schema_version::V1
        );
        assert_eq!(
            v2_decoded[0].schema_version,
            proximadb_records::schema_version::V2
        );
        // Per-record values are byte-identical between v1 and v2 in
        // INT-2a (INT-3 is what changes that).
        assert_eq!(
            v1_decoded[0].embeddings[0].values,
            v2_decoded[0].embeddings[0].values
        );
    }

    #[test]
    fn mixed_v1_v2_segments_read_independently() {
        // Simulates PR 4's mixed-segment-reader contract: two batches written
        // identically (PR 2 keeps the writer on V1), but each read dispatch
        // can label them with their declared schema version.
        let serializer = BincodeSerializer::new();
        let bytes_a = serializer.serialize_batch(&[create_test_vector()]).unwrap();
        let bytes_b = serializer.serialize_batch(&[create_test_vector()]).unwrap();

        let v1_records = serializer
            .deserialize_batch_with_schema_version(&bytes_a, proximadb_records::schema_version::V1)
            .unwrap();
        let v2_records = serializer
            .deserialize_batch_with_schema_version(&bytes_b, proximadb_records::schema_version::V2)
            .unwrap();
        assert_eq!(
            v1_records[0].schema_version,
            proximadb_records::schema_version::V1
        );
        assert_eq!(
            v2_records[0].schema_version,
            proximadb_records::schema_version::V2
        );
        // Payload is structurally identical — PR 4 will change this when the
        // v2 path swaps in the EmbeddingValues decoder.
        assert_eq!(
            v1_records[0].embeddings[0].values,
            v2_records[0].embeddings[0].values
        );
    }
}
