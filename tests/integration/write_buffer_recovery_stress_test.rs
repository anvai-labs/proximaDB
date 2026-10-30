// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Simplified WAL Recovery Stress Tests
//!
//! These tests verify WAL file creation, reading, and recovery performance
//! without depending on VectorOperationsService.

use anyhow::Result;
use proximadb::storage::persistence::write_ahead_log::serialization::{
    SerializationFormat, SerializerFactory, VectorBatchSerializer,
};
use proximadb_records::ProximaRecord;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use tracing::info;

/// Test helper to create test vectors with metadata
fn create_test_vectors_with_metadata(
    start_id: usize,
    count: usize,
    dimension: usize,
) -> Vec<ProximaRecord> {
    (start_id..start_id + count)
        .map(|i| {
            let mut props = proximadb_records::ProximaTree::new();
            props.insert(
                "batch_id".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String((i / 100).to_string()),
                ),
            );
            props.insert(
                "timestamp".to_string(),
                proximadb_records::ProximaTreeNode::Value(
                    proximadb_data_model::ProximaValue::String(
                        chrono::Utc::now().timestamp().to_string(),
                    ),
                ),
            );

            ProximaRecord {
                oid: format!("vec_{}", i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    dimension as u32,
                    vec![(i % 256) as f32; dimension],
                )],
                props,
                record_version: 1,
                created_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                updated_at_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                ..Default::default()
            }
        })
        .collect()
}

/// Create test WAL files
async fn create_test_wal_files(
    wal_dir: &Path,
    collection_id: &str,
    num_files: usize,
) -> Result<Vec<PathBuf>> {
    let collection_dir = wal_dir.join(collection_id);
    let logs_dir = collection_dir.join("logs");
    std::fs::create_dir_all(&logs_dir)?;

    let mut wal_files = Vec::new();

    let serializer = SerializerFactory::create(SerializationFormat::ProtocolBuffers);

    for idx in 0..num_files {
        let vectors = create_test_vectors_with_metadata(idx * 100, 100, 128);
        let serialized = serializer.serialize_batch(&vectors)?;

        let filename = format!(
            "wal_20250717_120000_{:010}_{:010}_test_{}.data",
            idx * 100,
            (idx + 1) * 100 - 1,
            idx
        );

        let wal_path = logs_dir.join(&filename);
        std::fs::write(&wal_path, serialized)?;
        wal_files.push(wal_path);
    }

    Ok(wal_files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn stress_test_concurrent_recovery() -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();

        // Create multiple temp directories for different collections
        let temp_dirs: Vec<TempDir> = (0..5).map(|_| TempDir::new().unwrap()).collect();

        // Create WAL files for each collection
        let mut tasks = JoinSet::new();

        for (idx, temp_dir) in temp_dirs.iter().enumerate() {
            let collection_id = format!("stress_collection_{}", idx);
            let wal_dir = temp_dir.path().to_path_buf();

            tasks.spawn(async move { create_test_wal_files(&wal_dir, &collection_id, 3).await });
        }

        // Wait for all WAL files to be created
        let mut all_files = Vec::new();
        while let Some(result) = tasks.join_next().await {
            all_files.extend(result??);
        }

        assert_eq!(
            all_files.len(),
            15,
            "Should have created 15 WAL files (5 collections * 3 files each)"
        );

        // Now test concurrent recovery by reading all files
        let mut recovery_tasks = JoinSet::new();

        for (idx, temp_dir) in temp_dirs.iter().enumerate() {
            let collection_id = format!("stress_collection_{}", idx);
            let wal_dir = temp_dir.path().to_path_buf();

            recovery_tasks.spawn(async move {
                let logs_dir = wal_dir.join(&collection_id).join("logs");
                let mut total_vectors = 0;

                // Read all WAL files in this collection
                for entry in std::fs::read_dir(&logs_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("data") {
                        let data = std::fs::read(&path)?;
                        let serializer =
                            SerializerFactory::create(SerializationFormat::ProtocolBuffers);
                        let vectors = serializer.deserialize_batch(&data)?;
                        total_vectors += vectors.len();
                    }
                }

                Ok::<(String, usize), anyhow::Error>((collection_id, total_vectors))
            });
        }

        // Wait for all recoveries to complete
        let mut recovered_collections = Vec::new();
        while let Some(result) = recovery_tasks.join_next().await {
            let (collection_id, vector_count) = result??;
            recovered_collections.push(collection_id);
            assert_eq!(vector_count, 300, "Each collection should have 300");
        }

        assert_eq!(recovered_collections.len(), 5, "Should recover all 5");

        let duration = start.elapsed();
        info!(
            "Concurrent recovery stress test completed in {:?}",
            duration
        );

        Ok(())
    }

    #[tokio::test]
    async fn stress_test_large_wal_files() -> Result<()> {
        use std::time::Instant;
        use tracing::info;

        let start = Instant::now();
        let temp_dir = TempDir::new()?;
        let collection_id = "large_wal_test";

        info!("Creating large WAL files for stress testing...");

        // Create a large batch of vectors (100k vectors with 512 dimensions)
        let batch_size = 10000;
        let num_batches = 10; // Total 100k vectors
        let dimension = 512;

        let wal_dir = temp_dir.path();
        let collection_dir = wal_dir.join(collection_id);
        let logs_dir = collection_dir.join("logs");
        std::fs::create_dir_all(&logs_dir)?;

        // Create large WAL files
        for batch_idx in 0..num_batches {
            let vectors =
                create_test_vectors_with_metadata(batch_idx * batch_size, batch_size, dimension);

            let serializer = SerializerFactory::create(SerializationFormat::ProtocolBuffers);
            let serialized = serializer.serialize_batch(&vectors)?;

            let filename = format!(
                "wal_20250717_120000_{:010}_{:010}_batch_{}.data",
                batch_idx * batch_size,
                (batch_idx + 1) * batch_size - 1,
                batch_idx
            );
            let file_path = logs_dir.join(&filename);
            tokio::fs::write(&file_path, &serialized).await?;

            info!(
                "Created WAL file {} with {} vectors ({} MB)",
                filename,
                batch_size,
                serialized.len() / (1024 * 1024)
            );
        }

        // Now test recovery of these large files
        let recovery_start = Instant::now();

        let mut total_vectors_recovered = 0;
        let mut total_bytes_processed = 0;

        // Read all files back
        for entry in std::fs::read_dir(&logs_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("data") {
                let data = tokio::fs::read(&path).await?;
                total_bytes_processed += data.len();

                // Deserialize to verify data integrity
                let serializer = SerializerFactory::create(SerializationFormat::ProtocolBuffers);
                let vectors = serializer.deserialize_batch(&data)?;
                total_vectors_recovered += vectors.len();
            }
        }

        assert_eq!(
            total_vectors_recovered,
            batch_size * num_batches,
            "Recovery check"
        );

        let recovery_duration = recovery_start.elapsed();
        let total_duration = start.elapsed();

        info!(
            "Completed: {} vectors, {} MB, recovery {:?}, total {:?}, {} vec/s",
            total_vectors_recovered,
            total_bytes_processed / (1024 * 1024),
            recovery_duration,
            total_duration,
            (total_vectors_recovered as f64 / recovery_duration.as_secs_f64()) as u64
        );

        Ok(())
    }
}
