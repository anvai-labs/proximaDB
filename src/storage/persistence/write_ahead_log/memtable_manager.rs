//! Memtable Manager for WAL operations
//!
//! This module centralizes all memtable operations, removing them from batch strategies.
//! It provides a clean interface for adding vectors to the memtable and retrieving them.

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

use crate::storage::memtable::core::MemtableConfig;
use crate::storage::memtable::specialized::wal_behavior::{WALBehaviorWrapper, WALVectorBatch};
use proximadb_records::ProximaRecord;

/// Centralized manager for all memtable operations
pub struct MemtableManager {
    /// The WAL behavior wrapper containing GlobalPartitionedMemtable
    wal_behavior: WALBehaviorWrapper,
    /// Statistics
    stats: Arc<tokio::sync::RwLock<MemtableStats>>,
}

/// Statistics for memtable operations
#[derive(Debug, Clone, Default)]
pub struct MemtableStats {
    pub total_vectors_added: u64,
    pub total_batches_added: u64,
    pub total_collections: usize,
    pub memory_usage_bytes: u64,
    pub per_collection_bytes: std::collections::HashMap<String, u64>,
}

impl MemtableManager {
    /// Create a new memtable manager
    pub fn new(config: MemtableConfig) -> Self {
        info!("🎯 Creating MemtableManager with config: {:?}", config);

        Self {
            wal_behavior: WALBehaviorWrapper::new(config),
            stats: Arc::new(tokio::sync::RwLock::new(MemtableStats::default())),
        }
    }

    /// Add a vector batch to the memtable
    pub async fn add_vector_batch(
        &self,
        collection_id: &str,
        batch: WALVectorBatch,
    ) -> Result<Vec<u64>> {
        debug!(
            "📝 Adding batch {} with {} vectors to collection {}",
            batch.batch_id.to_base62(),
            batch.vector_records.len(),
            collection_id
        );

        let vector_count = batch.vector_records.len() as u64;
        let batch_size_bytes = batch.total_size_bytes as u64;

        // Add to memtable
        let sequences = self
            .wal_behavior
            .add_vector_batch(collection_id, batch)
            .await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_vectors_added += vector_count;
            stats.total_batches_added += 1;
            stats.memory_usage_bytes += batch_size_bytes;
            let e = stats
                .per_collection_bytes
                .entry(collection_id.to_string())
                .or_insert(0);
            *e += batch_size_bytes;
            stats.total_collections = stats.per_collection_bytes.len();
        }

        debug!(
            "✅ Added {} vectors to collection {}, sequences: {:?}",
            vector_count, collection_id, sequences
        );

        Ok(sequences)
    }

    /// Add a single vector to the memtable
    pub async fn add_vector(&self, collection_id: &str, vector: ProximaRecord) -> Result<u64> {
        // Create a batch of one
        let batch = WALVectorBatch {
            batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
            vector_records: Arc::new(vec![vector]),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: 256, // Approximate
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        let sequences = self.add_vector_batch(collection_id, batch).await?;
        Ok(sequences.into_iter().next().unwrap_or(0))
    }

    /// Get all vectors for a collection
    pub async fn get_collection_vectors(&self, collection_id: &str) -> Result<Vec<ProximaRecord>> {
        let collection_id_string = crate::core::String::from(collection_id.to_string());
        self.wal_behavior
            .get_collection_vectors(&collection_id_string)
            .await
    }

    /// Search for a specific vector by ID
    pub async fn search_vector_by_id(
        &self,
        collection_id: &str,
        vector_id: &str,
    ) -> Result<Option<ProximaRecord>> {
        self.wal_behavior
            .vector_by_id(collection_id, vector_id)
            .await
    }

    /// Get all unflushed batches for a collection
    pub async fn get_unflushed_batches(&self, collection_id: &str) -> Result<Vec<WALVectorBatch>> {
        self.wal_behavior.get_unflushed_batches(collection_id).await
    }

    /// Mark batches as flushed
    pub async fn mark_batches_flushed(
        &self,
        collection_id: &str,
        batch_ids: &[crate::storage::persistence::write_ahead_log::BatchId],
    ) -> Result<()> {
        // Mark each batch as flushed individually
        for batch_id in batch_ids {
            self.wal_behavior
                .mark_batch_flushed(collection_id, &batch_id.to_base62())
                .await?;
        }
        Ok(())
    }

    /// Get memtable statistics
    pub async fn stats(&self) -> Result<MemtableStats> {
        Ok(self.stats.read().await.clone())
    }

    /// Remove flushed batches from memory
    pub async fn remove_flushed_batches(
        &self,
        collection_id: &str,
        batch_ids: &[crate::storage::persistence::write_ahead_log::BatchId],
    ) -> Result<()> {
        // Remove each batch individually
        for batch_id in batch_ids {
            self.wal_behavior
                .remove_batch(collection_id, &batch_id.to_base62())
                .await?;
        }
        Ok(())
    }

    /// Get memory usage for a collection
    pub async fn get_collection_memory_usage(&self, collection_id: &str) -> Result<u64> {
        // Prefer cached stats, fall back to sum
        let stats = self.stats.read().await;
        if let Some(b) = stats.per_collection_bytes.get(collection_id) {
            return Ok(*b);
        }
        drop(stats);
        let batches = self
            .wal_behavior
            .get_unflushed_batches(collection_id)
            .await?;
        Ok(batches.iter().map(|b| b.total_size_bytes as u64).sum())
    }

    /// Check if collection should be flushed based on memory usage
    pub async fn should_flush_collection(
        &self,
        collection_id: &str,
        threshold_bytes: u64,
    ) -> Result<bool> {
        let usage = self.get_collection_memory_usage(collection_id).await?;
        Ok(usage >= threshold_bytes)
    }

    /// Get all collections in the memtable
    pub async fn get_all_collections(&self) -> Result<Vec<String>> {
        Ok(self.wal_behavior.list_collections().await)
    }

    /// Get statistics
    pub async fn get_stats(&self) -> Result<MemtableStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Clear all data for a collection
    pub async fn clear_collection(&self, collection_id: &str) -> Result<()> {
        // Clear flushed batches for the collection
        self.wal_behavior
            .clear_flushed_batches(collection_id)
            .await?;
        Ok(())
    }

    /// Get the underlying WAL behavior wrapper (for advanced operations)
    pub fn get_wal_behavior(&self) -> &WALBehaviorWrapper {
        &self.wal_behavior
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::{EmbeddingCell, ProximaRecord};

    fn create_test_config() -> MemtableConfig {
        MemtableConfig {
            max_size_bytes: 10 * 1024 * 1024,       // 10MB
            flush_threshold_bytes: 5 * 1024 * 1024, // 5MB
            enable_mvcc: true,
            mvcc_cleanup_interval_secs: 60,
            max_versions_per_key: 10,
        }
    }

    fn create_test_vector(id: &str) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            created_at_ns: 1_234_567_890_000_000_000,
            updated_at_ns: 1_234_567_890_000_000_000,
            record_version: 1,
            origin: Some("test".to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_memtable_manager_basic_operations() {
        let manager = MemtableManager::new(create_test_config());
        let collection_id = "test_collection";

        // Add a single vector
        let vector = create_test_vector("test1");
        let seq = manager
            .add_vector(collection_id, vector.clone())
            .await
            .expect("Failed to add vector");
        assert!(seq > 0);

        // Retrieve the vector
        let retrieved = manager
            .search_vector_by_id(collection_id, "test1")
            .await
            .expect("Failed to search vector");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().oid, vector.oid);

        // Get all vectors
        let all_vectors = manager
            .get_collection_vectors(collection_id)
            .await
            .expect("Failed to get collection vectors");
        assert_eq!(all_vectors.len(), 1);

        // Check stats
        let stats = manager.stats().await.expect("Failed to get stats");
        assert_eq!(stats.total_vectors_added, 1);
        assert_eq!(stats.total_batches_added, 1);
    }

    #[tokio::test]
    async fn test_memtable_manager_batch_operations() {
        let manager = MemtableManager::new(create_test_config());
        let collection_id = "test_collection";

        // Create a batch
        let vectors = vec![
            create_test_vector("test1"),
            create_test_vector("test2"),
            create_test_vector("test3"),
        ];

        let batch = WALVectorBatch {
            batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
            vector_records: Arc::new(vectors.into_iter().collect()),
            timestamp: std::time::SystemTime::now(),
            total_size_bytes: 1024,
            is_flushed: false,
            metadata_bloom_filter: None,
        };

        // Add batch
        let sequences = manager
            .add_vector_batch(collection_id, batch.clone())
            .await
            .expect("Failed to add batch");
        assert_eq!(sequences.len(), 3);

        // Get unflushed batches
        let unflushed = manager
            .get_unflushed_batches(collection_id)
            .await
            .expect("Failed to get unflushed batches");
        assert_eq!(unflushed.len(), 1);

        // Mark as flushed
        manager
            .mark_batches_flushed(collection_id, &[batch.batch_id])
            .await
            .expect("Failed to mark batches flushed");

        // Remove flushed batches
        manager
            .remove_flushed_batches(collection_id, &[batch.batch_id])
            .await
            .expect("Failed to remove flushed batches");

        // Verify removed
        let remaining = manager
            .get_collection_vectors(collection_id)
            .await
            .expect("Failed to get remaining vectors");
        assert_eq!(remaining.len(), 0);
    }
}
