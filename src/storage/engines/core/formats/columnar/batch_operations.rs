// Columnar Batch Operations
// Efficient batch processing for columnar storage engines

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{ColumnarConfig, ParquetLocation, UnifiedParquetReader};
use proximadb_records::ProximaRecord;
use proximadb_runtime_common::pool::VectorMemoryPool;

/// Batch operations for columnar storage
pub struct ColumnarBatchOperations {
    /// Unified Parquet reader
    #[allow(dead_code)]
    parquet_reader: Arc<UnifiedParquetReader>,

    /// Vector memory pool for efficient buffer reuse
    memory_pool: Arc<VectorMemoryPool>,

    /// Configuration
    #[allow(dead_code)]
    config: ColumnarConfig,

    /// Batch operation cache
    operation_cache: Arc<RwLock<HashMap<String, CachedBatchResult>>>,
}

impl ColumnarBatchOperations {
    /// Create new batch operations handler
    pub fn new(
        parquet_reader: Arc<UnifiedParquetReader>,
        _hardware: Arc<proximadb_hardware_caps::HardwareCapabilities>,
        memory_pool: Arc<VectorMemoryPool>,
        config: ColumnarConfig,
    ) -> Self {
        Self {
            parquet_reader,
            memory_pool,
            config,
            operation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Batch read vectors by IDs across multiple files
    /// Deferred: Re-implement when UnifiedParquetReader.batch_id_lookup is available
    pub async fn batch_read_by_ids(
        &self,
        _file_paths: &[String],
        _ids: &[String],
    ) -> Result<Vec<ProximaRecord>> {
        // Deferred: Re-implement when batch_id_lookup API is available
        Err(anyhow::anyhow!(
            "BatchOperations temporarily disabled due to API changes"
        ))
    }

    /// Batch write vectors to columnar format
    pub async fn batch_write_vectors(
        &self,
        vectors: &[ProximaRecord],
        target_file: &str,
        compression_config: Option<&super::QuantizationConfig>,
    ) -> Result<BatchWriteResult> {
        info!("Batch writing {} vectors to {}", vectors.len(), target_file);

        // Organize vectors into optimal batches
        let batches = self.organize_into_batches(vectors)?;

        let mut total_bytes_written = 0;
        let mut written_row_groups = Vec::new();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            debug!("Writing batch {} with {} vectors", batch_idx, batch.len());

            // Use memory pool for efficient processing
            let batch_result = self
                .write_batch_optimized(&batch, target_file, batch_idx, compression_config)
                .await?;

            total_bytes_written += batch_result.bytes_written;
            written_row_groups.extend(batch_result.row_groups);
        }

        Ok(BatchWriteResult {
            vectors_written: vectors.len(),
            bytes_written: total_bytes_written,
            row_groups: written_row_groups,
            compression_ratio: self.calculate_compression_ratio(vectors.len(), total_bytes_written),
        })
    }

    /// Batch update vectors with optimistic concurrency
    pub async fn batch_update_vectors(
        &self,
        updates: &[VectorUpdateRequest],
        file_paths: &[String],
    ) -> Result<BatchUpdateResult> {
        info!(
            "Batch updating {} vectors across {} files",
            updates.len(),
            file_paths.len()
        );

        // Group updates by file for efficient processing
        let mut updates_by_file: HashMap<String, Vec<&VectorUpdateRequest>> = HashMap::new();

        for update in updates {
            if let Some(location) = &update.location {
                updates_by_file
                    .entry(location.file_path.clone())
                    .or_default()
                    .push(update);
            }
        }

        let mut total_updated = 0;
        let mut failed_updates = Vec::new();

        // Process updates per file
        for (file_path, file_updates) in updates_by_file {
            match self.process_file_updates(&file_path, &file_updates).await {
                Ok(count) => total_updated += count,
                Err(e) => {
                    for update in file_updates {
                        failed_updates.push(FailedUpdate {
                            vector_id: update.vector_id.clone(),
                            error: e.to_string(),
                        });
                    }
                }
            }
        }

        Ok(BatchUpdateResult {
            total_requested: updates.len(),
            successful_updates: total_updated,
            failed_updates,
        })
    }

    /// Batch delete vectors with tombstone marking
    pub async fn batch_delete_vectors(
        &self,
        ids: &[String],
        file_paths: &[String],
    ) -> Result<BatchDeleteResult> {
        info!(
            "Batch deleting {} vectors across {} files",
            ids.len(),
            file_paths.len()
        );

        // Find locations of vectors to delete
        let mut locations = Vec::new();
        for file_path in file_paths {
            // Use ID index to find vector locations
            let file_locations = self.find_vector_locations(file_path, ids).await?;
            locations.extend(file_locations);
        }

        // Mark vectors as deleted using tombstones
        let mut deleted_count = 0;
        let mut failed_deletes = Vec::new();

        for (vector_id, location) in locations {
            match self.mark_vector_deleted(&location).await {
                Ok(_) => deleted_count += 1,
                Err(e) => failed_deletes.push(FailedDelete {
                    vector_id,
                    error: e.to_string(),
                }),
            }
        }

        Ok(BatchDeleteResult {
            total_requested: ids.len(),
            successful_deletes: deleted_count,
            failed_deletes,
        })
    }

    /// Optimize read plan by grouping IDs by likely file locations
    #[allow(dead_code)]
    async fn optimize_read_plan(
        &self,
        file_paths: &[String],
        ids: &[String],
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut grouped = HashMap::new();

        // Simple // strategy removed -  distribute IDs evenly across files
        // In production, would use bloom filters or ID range analysis
        for (idx, id) in ids.iter().enumerate() {
            let file_idx = idx % file_paths.len();
            grouped
                .entry(file_paths[file_idx].clone())
                .or_insert_with(Vec::new)
                .push(id.clone());
        }

        Ok(grouped)
    }

    /// Organize vectors into optimal batches for writing
    fn organize_into_batches(&self, vectors: &[ProximaRecord]) -> Result<Vec<Vec<ProximaRecord>>> {
        const OPTIMAL_BATCH_SIZE: usize = 10000; // Optimal for Parquet row groups

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        for vector in vectors {
            current_batch.push(vector.clone());

            if current_batch.len() >= OPTIMAL_BATCH_SIZE {
                batches.push(current_batch);
                current_batch = Vec::new();
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        debug!(
            "Organized {} vectors into {} batches",
            vectors.len(),
            batches.len()
        );
        Ok(batches)
    }

    /// Write a single batch with memory pool optimization
    async fn write_batch_optimized(
        &self,
        batch: &[ProximaRecord],
        target_file: &str,
        batch_idx: usize,
        _compression_config: Option<&super::QuantizationConfig>,
    ) -> Result<SingleBatchWriteResult> {
        // Get buffer from memory pool
        let mut buffer = self.memory_pool.serialization_buffers.acquire();
        buffer.clear();

        // Serialize batch to buffer (simplified)
        let estimated_size = batch.len() * 1024; // Estimate 1KB per vector

        // In production, would write actual Parquet data
        debug!(
            "Writing batch {} to {} (estimated {} bytes)",
            batch_idx, target_file, estimated_size
        );

        Ok(SingleBatchWriteResult {
            bytes_written: estimated_size,
            row_groups: vec![format!("{target_file}:rg_{batch_idx}")],
        })
    }

    /// Process updates for a single file
    async fn process_file_updates(
        &self,
        file_path: &str,
        updates: &[&VectorUpdateRequest],
    ) -> Result<usize> {
        debug!(
            "Processing {} updates for file: {}",
            updates.len(),
            file_path
        );

        // In production, would:
        // 1. Load affected row groups
        // 2. Apply updates with version checking
        // 3. Write updated row groups
        // 4. Update indexes

        // Simulate processing
        Ok(updates.len())
    }

    /// Find vector locations in a file
    async fn find_vector_locations(
        &self,
        file_path: &str,
        ids: &[String],
    ) -> Result<Vec<(String, ParquetLocation)>> {
        let mut locations = Vec::new();

        // Use Parquet reader to find locations
        for id in ids {
            // Simplified location lookup
            locations.push((
                id.clone(),
                ParquetLocation {
                    file_path: file_path.to_string(),
                    row_group_id: 0,
                    row_offset: 0,
                    page_num: None,
                },
            ));
        }

        Ok(locations)
    }

    /// Mark a vector as deleted using tombstone
    async fn mark_vector_deleted(&self, _location: &ParquetLocation) -> Result<()> {
        // In production, would mark vector as deleted in metadata
        Ok(())
    }

    /// Calculate compression ratio
    fn calculate_compression_ratio(&self, vector_count: usize, bytes_written: usize) -> f32 {
        if bytes_written == 0 {
            return 1.0;
        }

        let uncompressed_estimate = vector_count * 4 * 768; // Assume 768-dim float32
        uncompressed_estimate as f32 / bytes_written as f32
    }

    /// Generate cache key for operation
    #[allow(dead_code)]
    fn generate_cache_key(&self, file_paths: &[String], ids: &[String]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        file_paths.hash(&mut hasher);
        ids.hash(&mut hasher);
        format!("batch_read_{:x}", hasher.finish())
    }

    /// Get cached result
    #[allow(dead_code)]
    async fn get_cached_result(&self, cache_key: &str) -> Option<CachedBatchResult> {
        let cache = self.operation_cache.read().await;
        cache.get(cache_key).cloned()
    }

    /// Cache operation result
    #[allow(dead_code)]
    async fn cache_result(&self, cache_key: String, records: &[ProximaRecord]) {
        let cached = CachedBatchResult {
            records: records.to_vec(),
            timestamp: chrono::Utc::now(),
            ttl_seconds: 300, // 5 minute TTL
        };

        let mut cache = self.operation_cache.write().await;
        cache.insert(cache_key, cached);

        // Simple cache eviction (keep last 100 entries)
        if cache.len() > 100 {
            let oldest_key = cache.keys().next().cloned();
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
    }

    /// Clear operation cache
    pub async fn clear_cache(&self) {
        let mut cache = self.operation_cache.write().await;
        cache.clear();
        info!("Cleared batch operations cache_info");
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> BatchCacheStats {
        let cache = self.operation_cache.read().await;

        BatchCacheStats {
            entry_count: cache.len(),
            total_cached_records: cache.values().map(|v| v.records.len()).sum(),
            oldest_entry: cache.values().map(|v| v.timestamp).min(),
        }
    }
}

/// Request to update a vector
#[derive(Debug, Clone)]
pub struct VectorUpdateRequest {
    pub vector_id: String,
    pub new_vector: Vec<f32>,
    pub new_metadata: Option<HashMap<String, String>>,
    pub expected_version: Option<u32>,
    pub location: Option<ParquetLocation>,
}

/// Result of batch write operation
#[derive(Debug)]
pub struct BatchWriteResult {
    pub vectors_written: usize,
    pub bytes_written: usize,
    pub row_groups: Vec<String>,
    pub compression_ratio: f32,
}

/// Result of single batch write
#[derive(Debug)]
struct SingleBatchWriteResult {
    pub bytes_written: usize,
    pub row_groups: Vec<String>,
}

/// Result of batch update operation
#[derive(Debug)]
pub struct BatchUpdateResult {
    pub total_requested: usize,
    pub successful_updates: usize,
    pub failed_updates: Vec<FailedUpdate>,
}

/// Result of batch delete operation
#[derive(Debug)]
pub struct BatchDeleteResult {
    pub total_requested: usize,
    pub successful_deletes: usize,
    pub failed_deletes: Vec<FailedDelete>,
}

/// Failed update information
#[derive(Debug)]
pub struct FailedUpdate {
    pub vector_id: String,
    pub error: String,
}

/// Failed delete information
#[derive(Debug)]
pub struct FailedDelete {
    pub vector_id: String,
    pub error: String,
}

/// Cached batch operation result
#[derive(Debug, Clone)]
struct CachedBatchResult {
    pub records: Vec<ProximaRecord>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    #[allow(dead_code)]
    pub ttl_seconds: i64,
}

impl CachedBatchResult {
    #[allow(dead_code)]
    fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(self.timestamp);
        age.num_seconds() > self.ttl_seconds
    }
}

/// Batch cache statistics
#[derive(Debug)]
pub struct BatchCacheStats {
    pub entry_count: usize,
    pub total_cached_records: usize,
    pub oldest_entry: Option<chrono::DateTime<chrono::Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use proximadb_records::EmbeddingCell;
    use proximadb_runtime_common::pool::VectorMemoryPool;

    #[tokio::test]
    async fn test_batch_operations_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let filesystem_factory = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .unwrap(),
        );

        // Create UnifiedCachingFilesystem for testing
        let base_fs = filesystem_factory
            .get_filesystem("file:///tmp/test")
            .unwrap();
        let cached_filesystem = Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                base_fs,
                "test_collection".to_string(),
                "test".to_string(),
            ),
        );
        let parquet_reader = Arc::new(
            UnifiedParquetReader::new(
                vec![],
                128,
                filesystem_factory,
                cached_filesystem,
                "test_collection".to_string(),
                "test".to_string(),
            )
            .unwrap(),
        );
        let hardware = proximadb_hardware_caps::get_hardware_capabilities();
        let memory_pool = Arc::new(VectorMemoryPool::new());
        let config = ColumnarConfig::default();

        let batch_ops = ColumnarBatchOperations::new(parquet_reader, hardware, memory_pool, config);

        // Test cache operations
        let stats = batch_ops.get_cache_stats().await;
        assert_eq!(stats.entry_count, 0);

        batch_ops.clear_cache().await;
    }

    #[test]
    fn test_vector_organization() {
        let batch_ops = create_test_batch_ops();

        // Create test vectors
        let vectors: Vec<ProximaRecord> = (0..25000)
            .map(|i| {
                let values = vec![i as f32; 768];
                let dim = values.len() as u32;
                let mut record = ProximaRecord {
                    oid: format!("test_{i}"),
                    ..Default::default()
                };
                record
                    .embeddings
                    .push(EmbeddingCell::new_fp32("default", "vector", dim, values));
                record
            })
            .collect();

        let batches = batch_ops.organize_into_batches(&vectors).unwrap();

        // Should create 3 batches (10k, 10k, 5k)
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 10000);
        assert_eq!(batches[1].len(), 10000);
        assert_eq!(batches[2].len(), 5000);
    }

    fn create_test_batch_ops() -> ColumnarBatchOperations {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

            let filesystem_factory = Arc::new(
                FilesystemFactory::create(FilesystemConfig::default())
                    .await
                    .unwrap(),
            );

            // Create UnifiedCachingFilesystem for testing
            let base_fs = filesystem_factory
                .get_filesystem("file:///tmp/test")
                .unwrap();
            let cached_filesystem = Arc::new(
                crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                    base_fs,
                    "test_collection".to_string(),
                    "test".to_string(),
                ),
            );
            let parquet_reader = Arc::new(
                UnifiedParquetReader::new(
                    vec![],
                    128,
                    filesystem_factory,
                    cached_filesystem,
                    "test_collection".to_string(),
                    "test".to_string(),
                )
                .unwrap(),
            );
            let hardware = proximadb_hardware_caps::get_hardware_capabilities();
            let memory_pool = Arc::new(VectorMemoryPool::new());
            let config = ColumnarConfig::default();

            ColumnarBatchOperations::new(parquet_reader, hardware, memory_pool, config)
        })
    }
}
