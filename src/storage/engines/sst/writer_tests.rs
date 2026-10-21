#[cfg(test)]
mod tests {
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use std::collections::BTreeMap;
    use std::sync::Arc;
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

    /// Regression: SstableWriter MUST NOT emit Vector Object Economy
    /// directory updates by default. Opt-in is via
    /// `with_directory_emission`. Flipping this default would silently
    /// touch the read-side cache for every existing call site — a
    /// behaviour change masquerading as a no-op refactor.
    #[tokio::test]
    async fn sstable_writer_default_does_not_emit_directory() {
        use crate::storage::engines::sst::SstableWriter;
        use crate::storage::engines::sst::object_economy_directory::{
            SstableWriterDirectoryHooks, VectorObjectEconomyDirectoryCache,
        };
        use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
        use proximadb_catalog::CatalogAuthorityMode;
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();
        let fs = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .unwrap(),
        );

        // Default construction: no directory emission configured.
        let writer = SstableWriter::new_with_config(
            temp.path().join("test.sst"),
            64 * 1024,
            fs.clone(),
            None,
        );
        assert!(
            !writer.directory_emission_configured(),
            "freshly-constructed writer must NOT emit directory updates"
        );

        // Opt in via builder — accessor flips.
        let cache = Arc::new(VectorObjectEconomyDirectoryCache::new());
        let writer_opted_in =
            SstableWriter::new_with_config(temp.path().join("test.sst"), 64 * 1024, fs, None)
                .with_directory_emission(SstableWriterDirectoryHooks {
                    cache,
                    collection_id: "coll".to_string(),
                    collection_root: "file:///tmp/coll".to_string(),
                    storage_epoch: 1,
                    authority_mode: CatalogAuthorityMode::RebuildableProjection,
                    freshness_lsn: 0,
                    level: 0,
                });
        assert!(writer_opted_in.directory_emission_configured());
    }
}
