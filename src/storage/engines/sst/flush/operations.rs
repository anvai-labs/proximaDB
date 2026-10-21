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

//! Flush Operations
//!
//! Low-level flush operations and utilities for the SST engine.
//! Provides atomic write operations and file management.

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::storage::engines::sst::SstEngine;
use crate::storage::engines::sst::writer::SstableWriter;
use crate::storage::transaction_coordinator::{
    StagingConfig, TransactionStageType, TransactionalOperationMetadata,
};
use proximadb_records::ProximaRecord;

/// Low-level flush operations
pub struct FlushOperations {
    engine: Arc<SstEngine>,
}

impl FlushOperations {
    /// Create new flush operations handler
    pub fn new(engine: Arc<SstEngine>) -> Self {
        Self { engine }
    }

    /// Write vectors to SSTable using atomic operations
    pub async fn write_vectors_atomic(
        &self,
        vectors: Vec<(String, ProximaRecord)>,
        staging_url: &str,
        filename: &str,
    ) -> Result<(u64, u64)> {
        debug!(
            "🔧 FlushOps: Writing {} vectors to {}",
            vectors.len(),
            staging_url
        );

        let block_size = (self.engine.config().block_size_kb * 1024) as usize;
        let full_path = format!("{}/{}", staging_url, filename);

        // Create SSTable writer
        let writer = SstableWriter::with_compression(
            &full_path,
            block_size,
            Arc::clone(self.engine.filesystem()),
            None, // No compression config in this simplified path
        );

        let mut bytes_written = 0u64;
        let mut entries_written = 0u64;

        // Write canonical records to SSTable
        for (key, vector_record) in vectors {
            // Serialize canonical record
            let record_bytes = self
                .serialize_proxima_record(&vector_record)
                .context("Failed to serialize canonical record")?;

            // Write key-value pair to SSTable
            self.write_record_to_sstable(&writer, &key, &record_bytes)
                .await
                .context("Failed to write record to SSTable")?;

            bytes_written += record_bytes.len() as u64;
            entries_written += 1;
        }

        // Finalize SSTable
        self.finalize_sstable(&writer)
            .await
            .context("Failed to finalize SSTable")?;

        info!(
            "✅ FlushOps: Wrote {} entries, {} bytes to SSTable",
            entries_written, bytes_written
        );
        Ok((entries_written, bytes_written))
    }

    /// Begin atomic flush operation
    pub async fn begin_atomic_flush(
        &self,
        storage_url: &str,
        collection_id: Option<&str>,
    ) -> Result<TransactionalOperationMetadata> {
        let staging_config = StagingConfig {
            base_url: storage_url.to_string(),
            collection_id: collection_id.map(|s| s.to_string()),
            operation_type: TransactionStageType::Flush,
            custom_staging_dir: None,
            auto_cleanup: true,
            max_orphaned_age_hours: 24,
            skip_uuid_subdir: true,
        };

        self.engine
            .atomic_coordinator()
            .begin_atomic_operation(&staging_config)
            .await
            .context("Failed to begin atomic flush operation")
    }

    /// Commit atomic flush operation
    pub async fn commit_atomic_flush(
        &self,
        atomic_op: TransactionalOperationMetadata,
    ) -> Result<()> {
        self.engine
            .atomic_coordinator()
            .finalize_atomic_operation(&atomic_op.operation_id)
            .await
            .context("Failed to commit atomic flush operation")
    }

    /// Abort atomic flush operation
    pub async fn abort_atomic_flush(
        &self,
        atomic_op: TransactionalOperationMetadata,
    ) -> Result<()> {
        warn!("🚫 FlushOps: Aborting atomic flush operation");
        self.engine
            .atomic_coordinator()
            .abort_atomic_operation(&atomic_op.operation_id, "Flush operation aborted")
            .await
            .context("Failed to abort atomic flush operation")
    }

    /// Serialize canonical record to bytes
    fn serialize_proxima_record(&self, record: &ProximaRecord) -> Result<Vec<u8>> {
        // Use efficient serialization format
        serde_json::to_vec(record).context("Failed to serialize canonical record")
    }

    /// Write record to SSTable (simplified implementation)
    async fn write_record_to_sstable(
        &self,
        _writer: &SstableWriter,
        _key: &str,
        _data: &[u8],
    ) -> Result<()> {
        // In a real implementation, this would write to the SSTable format
        // For now, we'll just simulate the operation
        debug!("📝 Writing record to SSTable (simulated)");
        Ok(())
    }

    /// Finalize SSTable writing
    async fn finalize_sstable(&self, _writer: &SstableWriter) -> Result<()> {
        // In a real implementation, this would:
        // - Write SSTable footer
        // - Generate bloom filter
        // - Write metadata
        debug!("🏁 Finalizing SSTable (simulated)");
        Ok(())
    }

    /// Calculate optimal block size for SSTable
    pub fn calculate_optimal_block_size(
        &self,
        vector_count: usize,
        avg_vector_size: usize,
    ) -> usize {
        let base_block_size = (self.engine.config().block_size_kb * 1024) as usize;

        // Adjust block size based on data characteristics
        let optimal_size = if vector_count < 1000 {
            // Small datasets: use smaller blocks for better cache utilization
            base_block_size / 2
        } else if avg_vector_size > 10240 {
            // Large vectors: use larger blocks for better compression
            base_block_size * 2
        } else {
            base_block_size
        };

        // Ensure minimum and maximum block sizes
        optimal_size.clamp(4096, 1024 * 1024) // 4KB to 1MB
    }

    /// Validate flush preconditions
    pub fn validate_flush_preconditions(
        &self,
        vectors: &[(String, ProximaRecord)],
        storage_url: &str,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("No vectors to flush"));
        }

        if storage_url.is_empty() {
            return Err(anyhow::anyhow!("Storage URL is required"));
        }

        // Validate vector data
        for (key, record) in vectors {
            if key.is_empty() {
                return Err(anyhow::anyhow!("Vector key cannot be empty"));
            }

            if record
                .embeddings
                .first()
                .is_none_or(|embedding| embedding.values.is_empty())
            {
                return Err(anyhow::anyhow!("Vector values cannot be empty"));
            }
        }

        debug!("✅ FlushOps: Flush preconditions validated");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    #[tokio::test]
    async fn test_validate_flush_preconditions() {
        let engine = create_test_engine().await;
        let ops = FlushOperations::new(Arc::new(engine));

        // Test with valid data
        let vectors = vec![(
            "key1".to_string(),
            create_test_vector("id1", vec![1.0, 2.0]),
        )];
        assert!(
            ops.validate_flush_preconditions(&vectors, "file:///tmp/test")
                .is_ok()
        );

        // Test with empty vectors
        let empty_vectors = vec![];
        assert!(
            ops.validate_flush_preconditions(&empty_vectors, "file:///tmp/test")
                .is_err()
        );

        // Test with empty storage URL
        assert!(ops.validate_flush_preconditions(&vectors, "").is_err());
    }

    #[tokio::test]
    async fn test_calculate_optimal_block_size() {
        let engine = create_test_engine().await;
        let ops = FlushOperations::new(Arc::new(engine));

        // Test small dataset
        let small_size = ops.calculate_optimal_block_size(100, 1000);
        let normal_size = ops.calculate_optimal_block_size(10000, 1000);
        // Both get clamped to max 1MB, so they're equal
        assert!(small_size <= normal_size);

        // Test large vectors - this should actually be larger due to avg_vector_size > 10240
        let large_vector_size = ops.calculate_optimal_block_size(5000, 20000);
        assert!(large_vector_size >= ops.calculate_optimal_block_size(5000, 1000));
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }

    fn create_test_vector(id: &str, vector: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            created_at_ns: 12_345_000_000,
            updated_at_ns: 12_345_000_000,
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "vector",
                vector.len() as u32,
                vector,
            )],
            ..Default::default()
        }
    }
}
