/*
 * Copyright 2025 ProximaDB
 *
 * End-to-End Integration Test for ArrowBlock Format
 *
 * This test verifies the Arrow IPC block format:
 * 1. Writes data via ArrowBlockWriter
 * 2. Reads data back via ArrowBlockReader
 * 3. Verifies data integrity and PyArrow compatibility
 */

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tempfile::TempDir;

    use crate::storage::engines::core::formats::arrow_block::{
        ArrowBlockConfig, ArrowBlockReader, ArrowBlockWriter,
    };
    use proximadb_data_model::ProximaValue;
    use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};
    use tracing::info;

    fn create_test_record(
        id: impl Into<String>,
        vector: Vec<f32>,
        timestamp_ms: i64,
    ) -> ProximaRecord {
        let id = id.into();
        let timestamp_ns = timestamp_ms.saturating_mul(1_000_000);
        ProximaRecord {
            oid: id.clone(),
            local_id: Some(id),
            created_at_ns: timestamp_ns,
            updated_at_ns: timestamp_ns,
            record_version: 1,
            embeddings: vec![EmbeddingCell::new_typed(
                "test",
                "dense_vector",
                vector.len() as u32,
                vector,
            )],
            ..ProximaRecord::default()
        }
    }

    fn embedding_values(record: &ProximaRecord) -> &[f32] {
        record
            .embeddings
            .first()
            .map(|embedding| embedding.as_fp32_slice())
            .unwrap_or(&[])
    }

    /// Test direct ArrowBlock write and read without SST engine
    #[tokio::test]
    async fn test_arrow_block_direct_write_read() -> Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        info!("🏹 Starting ArrowBlock direct write/read test");

        let temp_dir = TempDir::new()?;
        let arrow_path = temp_dir.path().join("test_vectors.arrow");

        // Create test vectors
        let dimension = 128;
        let num_vectors = 50;
        let mut vectors = Vec::new();

        for i in 0..num_vectors {
            let mut values = vec![0.0f32; dimension];
            for j in 0..dimension {
                values[j] = ((i as f32) * 0.1 + (j as f32) * 0.01).sin();
            }

            let mut record = create_test_record(format!("arrow_vec_{}", i), values, i as i64);
            record.props.insert(
                "category".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("cat_{}", i % 5))),
            );
            vectors.push(record);
        }

        // Write using ArrowBlockWriter
        info!("📝 Writing {} vectors to Arrow file", vectors.len());
        let config = ArrowBlockConfig::new(dimension as u32);
        let mut writer = ArrowBlockWriter::new(&arrow_path, config)?;
        writer.write_block(&vectors)?;
        let metadata = writer.finalize()?;

        info!(
            "✅ Written {} records, {} blocks",
            metadata.total_records, metadata.num_blocks
        );

        // Read back using ArrowBlockReader
        info!("📖 Reading vectors from Arrow file");
        let reader = ArrowBlockReader::open(&arrow_path)?;
        let read_records = reader.read_all()?;

        assert_eq!(
            read_records.len(),
            num_vectors,
            "Should read back same number of vectors"
        );

        // Verify specific records
        for (i, record) in read_records.iter().enumerate() {
            assert_eq!(record.oid, format!("arrow_vec_{}", i));
            assert_eq!(embedding_values(record).len(), dimension);
            assert!(record.props.contains_key("category"));
        }

        // Test lookup by ID
        let lookup_result = reader.lookup_by_id("arrow_vec_25")?;
        assert!(lookup_result.is_some(), "Should find vector by ID");
        let found = lookup_result.unwrap();
        assert_eq!(found.oid, "arrow_vec_25");

        info!("✅ ArrowBlock direct write/read test passed");
        Ok(())
    }

    /// Test PyArrow interoperability - Arrow file should be readable by PyArrow
    #[tokio::test]
    async fn test_arrow_block_pyarrow_compatibility() -> Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        info!("🐍 Testing PyArrow compatibility");

        let temp_dir = TempDir::new()?;
        let arrow_path = temp_dir.path().join("pyarrow_compatible.arrow");

        // Create test vectors
        let dimension = 64;
        let num_vectors = 10;
        let mut vectors = Vec::new();

        for i in 0..num_vectors {
            let values: Vec<f32> = (0..dimension)
                .map(|j| (i * dimension + j) as f32 / 1000.0)
                .collect();

            let mut record = create_test_record(format!("pyarrow_vec_{}", i), values, i as i64);
            record.props.insert(
                "name".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("item_{}", i))),
            );
            vectors.push(record);
        }

        // Write using ArrowBlockWriter
        let config = ArrowBlockConfig::new(dimension as u32);
        let mut writer = ArrowBlockWriter::new(&arrow_path, config)?;
        writer.write_block(&vectors)?;
        writer.finalize()?;

        // Verify the file exists and has non-zero size
        let file_metadata = std::fs::metadata(&arrow_path)?;
        assert!(file_metadata.len() > 0, "Arrow file should have content");
        info!(
            "📄 Created Arrow file: {} ({} bytes)",
            arrow_path.display(),
            file_metadata.len()
        );

        // Verify we can read it back with standard Arrow IPC reader
        // This tests that the file is valid Arrow IPC format
        let file = std::fs::File::open(&arrow_path)?;
        let arrow_reader = arrow_ipc::reader::FileReader::try_new(file, None)?;

        let schema = arrow_reader.schema();
        info!("📊 Arrow schema fields:");
        for field in schema.fields() {
            info!("  - {}: {:?}", field.name(), field.data_type());
        }

        // Verify expected fields exist
        assert!(
            schema.field_with_name("id").is_ok(),
            "Schema should have 'id' field"
        );
        assert!(
            schema.field_with_name("vector").is_ok(),
            "Schema should have 'vector' field"
        );

        info!("✅ Arrow file is PyArrow compatible (valid Arrow IPC format)");
        Ok(())
    }

    /// Test batch lookup functionality
    #[tokio::test]
    async fn test_arrow_block_batch_lookup() -> Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        info!("🏹 Testing batch lookup");

        let temp_dir = TempDir::new()?;
        let arrow_path = temp_dir.path().join("batch_lookup.arrow");

        // Create test vectors
        let dimension = 32;
        let num_vectors = 100;
        let mut vectors = Vec::new();

        for i in 0..num_vectors {
            let values: Vec<f32> = (0..dimension).map(|j| (i + j) as f32).collect();
            vectors.push(create_test_record(
                format!("batch_vec_{}", i),
                values,
                i as i64,
            ));
        }

        // Write
        let config = ArrowBlockConfig::new(dimension as u32);
        let mut writer = ArrowBlockWriter::new(&arrow_path, config)?;
        writer.write_block(&vectors)?;
        writer.finalize()?;

        // Read and batch lookup
        let reader = ArrowBlockReader::open(&arrow_path)?;
        let ids = vec![
            "batch_vec_10",
            "batch_vec_50",
            "batch_vec_99",
            "nonexistent",
        ];
        let results = reader.lookup_batch(&ids)?;

        // Should find 3 out of 4 IDs
        assert_eq!(results.len(), 3, "Should find 3 vectors");

        let found_ids: Vec<_> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(found_ids.contains(&"batch_vec_10"));
        assert!(found_ids.contains(&"batch_vec_50"));
        assert!(found_ids.contains(&"batch_vec_99"));

        info!("✅ Batch lookup test passed");
        Ok(())
    }
}
