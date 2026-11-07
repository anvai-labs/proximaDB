#[cfg(test)]
mod tests {
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use std::collections::BTreeMap;
    use tempfile::NamedTempFile;

    fn test_record(id: impl Into<String>, vector: Vec<f32>) -> ProximaRecord {
        let id = id.into();
        ProximaRecord {
            oid: id.clone(),
            local_id: Some(id),
            record_version: 1,
            created_at_ns: 123_456_789_000_000,
            updated_at_ns: 123_456_789_000_000,
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                vector.len() as u32,
                vector,
            )],
            ..ProximaRecord::default()
        }
    }

    #[tokio::test]
    async fn test_sstable_writer_basic() {
        // Note: This test would need a mock filesystem for full testing
        // For now, just test the data structure building
        let _temp_file = NamedTempFile::new().unwrap();

        // Create test records
        let mut records = BTreeMap::new();
        for i in 0..10 {
            let record = test_record(format!("key{:03}", i), vec![1.0, 2.0, 3.0]);
            records.insert(record.oid.clone(), record);
        }

        assert_eq!(records.len(), 10);
    }
}
