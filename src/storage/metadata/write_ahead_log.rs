// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Metadata Write Buffer System - Optimized for metadata workloads
//!
//! Reuses the vector write buffer infrastructure but with different optimization:
//!
//! ## Key Differences from Vector Write Buffer:
//! - **Avro schema**: Different schema optimized for collection metadata
//! - **Separate directory**: `./data/metadata/write_buffer` vs `./data/write_buffer`
//! - **B+Tree memtable**: Better for range queries (list collections) vs ART for vectors
//! - **Smaller batches**: 100 vs 1000+ for vectors (metadata operations are fewer)
//! - **Keep in memory**: All metadata kept in memory for fast atomic operations
//! - **More MVCC versions**: 10 vs 3 for vectors (metadata has concurrent updates)
//!
//! ## Flush Strategy:
//! - **Vector Write Buffer**: Flushes to collection-specific indexing/storage layout
//! - **Metadata Write Buffer**: Flushes to optimal B+Tree storage for reload into memory

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::CompressionAlgorithm;
use crate::storage::persistence::filesystem::FilesystemFactory;

use crate::storage::persistence::write_ahead_log::{
    WALConfig,
    // Using modern batch architecture - no more WalEntry
    config::{MemTableType, WriteBufferStrategyType},
};

/// Metadata-specific write buffer configuration
#[derive(Debug, Clone)]
pub struct MetadataWALConfig {
    /// Base write buffer configuration
    pub base_config: WALConfig,

    /// Keep all metadata in memory for fast access
    pub keep_all_in_memory: bool,

    /// Metadata-specific flush threshold (number of operations)
    pub metadata_flush_threshold: usize,

    /// Enable metadata caching
    pub enable_metadata_cache: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for MetadataWALConfig {
    fn default() -> Self {
        // Optimized defaults for metadata workloads (different from vector write buffer)
        let mut base_config = WALConfig {
            // Use Avro for schema evolution (metadata schemas change more than vector schemas)
            strategy_type: WriteBufferStrategyType::AvroBatch,
            ..Default::default()
        };

        // Use B+Tree for sorted iteration and range queries on collection metadata
        // Better than ART since we need range scans for list operations
        base_config.memtable.memtable_type = MemTableType::BTree;

        // Separate directory from vector write buffer data
        base_config.multi_disk.data_directories = vec!["./data/metadata/write_buffer".to_string()];

        // Smaller memory limit for metadata (fewer operations than vectors)
        base_config.performance.memory_flush_size_bytes = 32 * 1024 * 1024; // 32MB size threshold

        // Lower memory limit since metadata is much smaller than vectors
        base_config.memtable.global_memory_limit = 128 * 1024 * 1024; // 128MB

        // Keep more MVCC versions for metadata due to concurrent updates
        base_config.memtable.mvcc_versions_retained = 10;

        // Use Snappy for fast compression (metadata needs fast read/write)
        base_config.compression.algorithm = CompressionAlgorithm::Snappy;

        // Configuration optimized for metadata operations

        Self {
            base_config,
            keep_all_in_memory: true, // Keep metadata in memory for fast lookups
            metadata_flush_threshold: 2_500, // Flush more frequently for durability
            enable_metadata_cache: true,
            cache_ttl_seconds: 600, // 10 minutes (metadata changes less frequently)
        }
    }
}

/// Collection metadata with versioning
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VersionedCollectionMetadata {
    /// Collection ID
    pub id: String,
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric type
    pub distance_metric: String,
    /// Indexing algorithm
    pub indexing_algorithm: String,
    /// Last modification timestamp (seconds since epoch)
    pub timestamp: u32,
    /// Schema version
    pub version: Option<u32>,
    /// Number of vectors
    pub vector_count: u64,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Collection configuration
    pub config: HashMap<String, serde_json::Value>,
    /// Collection description
    pub description: Option<String>,
    /// Collection tags
    pub tags: Vec<String>,
    /// Collection owner
    pub owner: Option<String>,
    /// Access pattern for optimization
    pub access_pattern: AccessPattern,
    /// Retention policy
    pub retention_policy: Option<RetentionPolicy>,
}

/// Access pattern hints for optimization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AccessPattern {
    /// Frequently accessed, keep hot
    Hot,
    /// Normal access pattern
    Normal,
    /// Rarely accessed
    Cold,
    /// Archive, very rare access
    Archive,
}

/// Retention policy for collections
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetentionPolicy {
    /// Number of days to retain data
    pub retain_days: u32,
    /// Enable automatic archiving
    pub auto_archive: bool,
    /// Enable automatic deletion
    pub auto_delete: bool,
}

/// Metadata write buffer manager with caching
pub struct MetadataWriteAheadLog {
    /// Underlying write buffer manager with modern batch strategy (Avro)
    write_buffer_strategy: crate::storage::persistence::write_ahead_log::WriteAheadLogManager,

    /// Configuration
    config: MetadataWALConfig,

    /// In-memory cache of all metadata
    metadata_cache: Arc<tokio::sync::RwLock<HashMap<String, VersionedCollectionMetadata>>>,

    /// Cache timestamps for TTL
    cache_timestamps: Arc<tokio::sync::RwLock<HashMap<String, DateTime<Utc>>>>,

    /// Statistics
    stats: Arc<tokio::sync::RwLock<WalMetadataStats>>,
}

impl std::fmt::Debug for MetadataWriteAheadLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataWriteAheadLog")
            .field("config", &self.config)
            .field("metadata_cache_info", &"<cached data>")
            .field("cache_timestamps", &"<timestamps>")
            .field("stats", &"<stats>")
            .finish()
    }
}

/// Backwards-compat alias for [`WalMetadataStats`].
pub type MetadataStats = WalMetadataStats;

#[derive(Debug, Default, Clone)]
pub struct WalMetadataStats {
    /// Total number of collections
    pub total_collections: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
    /// Total write buffer writes
    pub write_buffer_writes: u64,
    /// Total write buffer reads
    pub write_buffer_reads: u64,
}

impl MetadataWriteAheadLog {
    /// Create new metadata write buffer manager
    pub async fn new(
        config: MetadataWALConfig,
        filesystem: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        tracing::debug!("🚀 Creating MetadataWriteAheadLog with B+Tree memtable for sorted access");

        // Create write buffer manager using modern batch factory pattern
        let write_buffer_manager = crate::storage::persistence::write_ahead_log::WriteAheadLogManager::create_with_batch_factory(
            config.base_config.strategy_type,
            config.base_config.clone(),
            filesystem,
        )
        .await?;

        // Initialize metadata write buffer
        tracing::debug!("📋 Initializing metadata write buffer with modern batch strategy");

        let manager = Self {
            write_buffer_strategy: write_buffer_manager,
            config,
            metadata_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            cache_timestamps: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            stats: Arc::new(tokio::sync::RwLock::new(WalMetadataStats::default())),
        };

        // Initialize by recovering any existing metadata from write buffer
        manager.recover_metadata_from_write_buffer().await?;

        Ok(manager)
    }

    /// Create or update collection metadata
    pub async fn upsert_collection(&self, metadata: VersionedCollectionMetadata) -> Result<()> {
        let collection_id = metadata.id.clone();
        tracing::debug!("📝 Upserting metadata for collection: {}", collection_id);

        // Convert metadata to canonical record.
        let metadata_record = self.metadata_to_proxima_record(&metadata)?;

        // Create modern batch operation using canonical records.
        let batch_records = vec![metadata_record];

        // Write to write buffer using modern batch architecture through WALBehaviorWrapper
        {
            let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
            // Create WALVectorBatch for the metadata
            let batch = crate::storage::memtable::specialized::wal_behavior::WALVectorBatch {
                batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
                vector_records: Arc::new(batch_records),
                timestamp: std::time::SystemTime::now(),
                total_size_bytes: 1024, // Approximate for metadata
                is_flushed: false,
                metadata_bloom_filter: None,
            };
            let _sequence = behavior_wrapper
                .add_vector_batch(&collection_id, batch)
                .await?;
        }

        // Update cache if enabled
        if self.config.enable_metadata_cache {
            let mut cache = self.metadata_cache.write().await;
            let mut timestamps = self.cache_timestamps.write().await;

            cache.insert(collection_id.clone(), metadata);
            timestamps.insert(collection_id.clone(), Utc::now());

            tracing::debug!(
                "✅ Updated metadata cache for collection: {}",
                collection_id
            );
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.write_buffer_writes += 1;
        if self.config.enable_metadata_cache {
            let cache = self.metadata_cache.read().await;
            stats.total_collections = cache.len() as u64;
        }

        Ok(())
    }

    /// Get collection metadata (alias for get_collection)
    pub async fn collection(
        &self,
        collection_id: &str,
    ) -> Result<Option<VersionedCollectionMetadata>> {
        self.get_collection(collection_id).await
    }

    /// Get collection metadata
    pub async fn get_collection(
        &self,
        collection_id: &str,
    ) -> Result<Option<VersionedCollectionMetadata>> {
        tracing::debug!("🔍 Getting metadata for collection: {}", collection_id);

        // Check cache first
        if self.config.enable_metadata_cache {
            let cache = self.metadata_cache.read().await;
            let timestamps = self.cache_timestamps.read().await;

            if let Some(metadata) = cache.get(collection_id) {
                // Check TTL
                if let Some(timestamp) = timestamps.get(collection_id) {
                    let age = Utc::now().signed_duration_since(*timestamp);
                    if age.num_seconds() < self.config.cache_ttl_seconds as i64 {
                        let mut stats = self.stats.write().await;
                        stats.cache_hits += 1;
                        tracing::debug!("✅ Cache hit for collection: {}", collection_id);
                        return Ok(Some(metadata.clone()));
                    }
                }
            }
        }

        // Cache miss, read from write buffer
        let mut stats = self.stats.write().await;
        stats.cache_misses += 1;
        stats.write_buffer_reads += 1;
        drop(stats);

        tracing::debug!(
            "💾 Cache miss, reading from write buffer for collection: {}",
            collection_id
        );

        // Search in write buffer using modern vector_by_id through WALBehaviorWrapper
        let vector_id = format!("metadata_{}", collection_id);
        let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
        let vector_record = behavior_wrapper
            .vector_by_id(collection_id, &vector_id)
            .await?;

        if let Some(record) = vector_record {
            let metadata = self.vector_record_to_metadata(&record)?;

            // Update cache
            if self.config.enable_metadata_cache {
                let mut cache = self.metadata_cache.write().await;
                let mut timestamps = self.cache_timestamps.write().await;

                cache.insert(collection_id.to_string(), metadata.clone());
                timestamps.insert(collection_id.to_string(), Utc::now());
            }

            return Ok(Some(metadata));
        }

        Ok(None)
    }

    /// List all collections with optional filtering
    pub async fn list_collections(
        &self,
        filter: Option<Box<dyn Fn(&VersionedCollectionMetadata) -> bool + Send>>,
    ) -> Result<Vec<VersionedCollectionMetadata>> {
        tracing::debug!(
            "📋 Listing all collections with filter={}",
            filter.is_some()
        );

        let mut collections = Vec::new();

        // If cache is complete and valid, use it
        if self.config.keep_all_in_memory && self.config.enable_metadata_cache {
            let cache = self.metadata_cache.read().await;
            for metadata in cache.values() {
                if let Some(ref filter_fn) = filter {
                    if filter_fn(metadata) {
                        collections.push(metadata.clone());
                    }
                } else {
                    collections.push(metadata.clone());
                }
            }

            // Sort by name for consistent ordering (B+Tree advantage)
            collections.sort_by(|a, b| a.name.cmp(&b.name));

            tracing::debug!(
                "✅ Listed {} collections from cache_info",
                collections.len()
            );
            return Ok(collections);
        }

        // Otherwise, scan write buffer (this is where B+Tree memtable helps with sorted iteration)
        tracing::debug!("📂 Scanning write buffer for collection metadata...");

        // For now, return empty list when cache is not populated
        // The cache will be populated on-demand when collections are accessed
        // This is a simplified approach - in production, we'd implement full write buffer scanning
        tracing::warn!(
            "⚠️ Cache empty and full write buffer scan not implemented - collections will be loaded on-demand"
        );

        // Try to get stats to see if there's any data using modern interface
        let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
        if let Ok(stats_map) = behavior_wrapper.stats().await {
            let total_entries: u64 = stats_map.values().map(|s| s.total_entries).sum();
            if total_entries > 0 {
                tracing::info!(
                    "📊 Write buffer contains {} entries - metadata will be loaded when accessed",
                    total_entries
                );
            }
        }

        Ok(collections)
    }

    /// Recover metadata from write buffer on startup and populate cache
    async fn recover_metadata_from_write_buffer(&self) -> Result<()> {
        tracing::info!("🔄 Recovering collection metadata from write buffer...");

        // Call write buffer strategy recover to load from disk using modern interface
        let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
        let recovered_entries = if let Ok(stats_map) = behavior_wrapper.stats().await {
            stats_map.values().map(|s| s.total_entries).sum()
        } else {
            0
        };
        tracing::info!(
            "📂 Write buffer recovery found {} total entries",
            recovered_entries
        );

        if recovered_entries == 0 {
            tracing::info!("📭 No existing metadata found in write buffer - starting fresh");
            return Ok(());
        }

        // Now populate our cache by reading all metadata entries from write buffer memtable
        let mut recovered_collections = 0;

        // Get write buffer statistics to see what collections exist using modern interface
        let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
        if let Ok(stats_map) = behavior_wrapper.stats().await {
            let total_entries: u64 = stats_map.values().map(|s| s.total_entries).sum();
            let collections_count = stats_map.len();
            tracing::info!(
                "📊 Write buffer stats: {} total entries, {} collections",
                total_entries,
                collections_count
            );

            // Try to enumerate collections by looking at write buffer entries
            // Since we store metadata with vector_id = "metadata_{collection_id}"
            // we can scan for these entries and rebuild our cache
            if let Ok(collection_ids) = self.get_all_collection_ids().await {
                for collection_id in collection_ids {
                    if let Ok(Some(metadata)) = self.collection(&collection_id).await {
                        tracing::debug!(
                            "📦 Recovered collection: {} ({})",
                            metadata.name,
                            collection_id
                        );
                        recovered_collections += 1;
                    }
                }
            }
        }

        tracing::info!(
            "✅ Metadata recovery completed - {} collections recovered into cache_info",
            recovered_collections
        );
        Ok(())
    }

    /// Get all collection IDs that have metadata stored in write buffer
    async fn get_all_collection_ids(&self) -> Result<Vec<String>> {
        // If cache is populated, use it
        if self.config.enable_metadata_cache {
            let cache = self.metadata_cache.read().await;
            if !cache.is_empty() {
                return Ok(cache.keys().cloned().collect());
            }
        }

        // Get stats to find collections using modern interface
        let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
        if let Ok(stats_map) = behavior_wrapper.stats().await {
            // Extract collection IDs from write buffer stats
            let collection_ids: Vec<String> = stats_map.keys().cloned().collect();

            let total_entries: u64 = stats_map.values().map(|s| s.total_entries).sum();
            let collections_count = stats_map.len();

            tracing::debug!(
                "📊 Write buffer has {} total entries across {} collections",
                total_entries,
                collections_count
            );

            Ok(collection_ids)
        } else {
            Ok(Vec::new())
        }
    }

    /// Delete collection metadata
    pub async fn delete_collection(&self, collection_id: &str) -> Result<bool> {
        tracing::debug!("🗑️ Deleting metadata for collection: {}", collection_id);

        // Check if exists
        let exists = self.collection(collection_id).await?.is_some();

        if exists {
            // Create canonical record for delete operation using MVCC logical delete.
            let vector_id = format!("metadata_{}", collection_id);
            let current_time = chrono::Utc::now().timestamp_micros();

            let current_time_secs = (current_time / 1_000_000) as u32; // Convert microseconds to seconds
            let delete_record = proximadb_records::ProximaRecord {
                oid: vector_id,
                embeddings: vec![proximadb_records::EmbeddingCell {
                    model_id: "metadata".to_string(),
                    modality: "tombstone".to_string(),
                    values: proximadb_records::EmbeddingValues::Fp32(vec![0.0]), // Content irrelevant for delete.
                    dim: 1,
                    ..Default::default()
                }],
                created_at_ns: current_time_secs as i64 * 1_000_000_000,
                updated_at_ns: current_time_secs as i64 * 1_000_000_000,
                valid_to_ns: Some(current_time_secs.saturating_sub(1) as i64 * 1_000_000_000),
                record_version: 1,
                ..Default::default()
            };

            // Write delete record to write buffer using modern batch architecture through WALBehaviorWrapper
            let delete_batch_records = vec![delete_record];
            {
                let behavior_wrapper = self.write_buffer_strategy.get_wal_behavior_wrapper();
                // Create WALVectorBatch for the delete
                let vector_count = delete_batch_records.len() as u64;
                let _end_sequence = if vector_count > 0 {
                    2 + vector_count - 1
                } else {
                    2
                };
                let delete_batch =
                    crate::storage::memtable::specialized::wal_behavior::WALVectorBatch {
                        batch_id: crate::storage::persistence::write_ahead_log::BatchId::new(),
                        vector_records: Arc::new(delete_batch_records),
                        timestamp: std::time::SystemTime::now(),
                        total_size_bytes: 1024, // Approximate for metadata
                        is_flushed: false,
                        metadata_bloom_filter: None,
                    };
                let _sequence = behavior_wrapper
                    .add_vector_batch(collection_id, delete_batch)
                    .await?;
            }

            // Remove from cache
            if self.config.enable_metadata_cache {
                let mut cache = self.metadata_cache.write().await;
                let mut timestamps = self.cache_timestamps.write().await;

                cache.remove(collection_id);
                timestamps.remove(collection_id);
            }

            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_collections = stats.total_collections.saturating_sub(1);

            tracing::debug!("✅ Deleted metadata for collection: {}", collection_id);
        }

        Ok(exists)
    }

    /// Update collection statistics (vector count, size)
    pub async fn update_stats(
        &self,
        collection_id: &str,
        vector_delta: i64,
        size_delta: i64,
    ) -> Result<()> {
        tracing::debug!(
            "📊 Updating stats for collection {}: vectors={:+}, size={:+}",
            collection_id,
            vector_delta,
            size_delta
        );

        if let Some(mut metadata) = self.collection(collection_id).await? {
            // Update stats
            if vector_delta > 0 {
                metadata.vector_count += vector_delta as u64;
            } else {
                metadata.vector_count =
                    metadata.vector_count.saturating_sub((-vector_delta) as u64);
            }

            if size_delta > 0 {
                metadata.total_size_bytes += size_delta as u64;
            } else {
                metadata.total_size_bytes = metadata
                    .total_size_bytes
                    .saturating_sub((-size_delta) as u64);
            }

            metadata.timestamp = Utc::now().timestamp() as u32;
            metadata.version = Some(metadata.version.unwrap_or(0) + 1);

            // Save updated metadata
            self.upsert_collection(metadata).await?;
        }

        Ok(())
    }

    /// Flush metadata to disk
    pub async fn flush(&self) -> Result<()> {
        tracing::debug!("💾 Flushing metadata write buffer");
        // Modern WriteAheadLogManager doesn't have flush method - data is automatically persisted
        // In the modern architecture, persistence happens automatically through the batch strategies
        tracing::debug!("✅ Metadata write buffer uses automatic persistence via batch strategies");
        Ok(())
    }

    /// Get metadata statistics
    pub async fn get_stats(&self) -> Result<WalMetadataStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Get metadata statistics (alias for get_stats)
    pub async fn stats(&self) -> Result<WalMetadataStats> {
        self.get_stats().await
    }

    /// Convert metadata to canonical record for write buffer storage.
    fn metadata_to_proxima_record(
        &self,
        metadata: &VersionedCollectionMetadata,
    ) -> Result<proximadb_records::ProximaRecord> {
        // Serialize metadata to JSON, then to bytes as a "vector"
        let json = serde_json::to_vec(metadata)?;
        let vector: Vec<f32> = json.iter().map(|&b| b as f32).collect();

        let timestamp_secs = metadata.timestamp; // Already in seconds
        Ok(proximadb_records::ProximaRecord {
            oid: format!("metadata_{}", metadata.id),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "metadata",
                "collection_metadata",
                vector.len() as u32,
                vector,
            )],
            created_at_ns: timestamp_secs as i64 * 1_000_000_000,
            updated_at_ns: timestamp_secs as i64 * 1_000_000_000,
            record_version: 1,
            ..Default::default()
        })
    }

    /// Convert canonical record back to metadata.
    fn vector_record_to_metadata(
        &self,
        record: &proximadb_records::ProximaRecord,
    ) -> Result<VersionedCollectionMetadata> {
        // Convert float vector back to bytes
        let bytes: Vec<u8> = record
            .embeddings
            .first()
            .map(|embedding| embedding.as_fp32_cow().iter().map(|&f| f as u8).collect())
            .unwrap_or_default();

        // Deserialize from JSON
        let metadata: VersionedCollectionMetadata = serde_json::from_slice(&bytes)?;
        Ok(metadata)
    }
}

/// System metadata with write buffer backing
#[derive(Debug, Clone)]
pub struct SystemMetadata {
    /// System version
    pub version: String,
    /// Node ID
    pub node_id: String,
    /// Cluster name
    pub cluster_name: String,
    /// Creation timestamp (seconds since epoch)
    pub timestamp: u32,
    /// Last update timestamp (seconds since epoch)
    pub updated_at: Option<u32>,
    /// Total number of collections
    pub total_collections: u64,
    /// Total number of vectors
    pub total_vectors: u64,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// System configuration
    pub config: HashMap<String, serde_json::Value>,
}

impl Default for SystemMetadata {
    fn default() -> Self {
        Self {
            version: "0.1.0".to_string(),
            node_id: proximadb_kernel::uuid::Uuid::new_v4().to_string(),
            cluster_name: "default".to_string(),
            timestamp: Utc::now().timestamp() as u32,
            updated_at: None,
            total_collections: 0,
            total_vectors: 0,
            total_size_bytes: 0,
            config: HashMap::new(),
        }
    }
}

impl SystemMetadata {
    /// Create default system metadata
    pub fn default_with_node_id(node_id: String) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            node_id,
            cluster_name: "proximadb-cluster".to_string(),
            timestamp: Utc::now().timestamp() as u32,
            updated_at: Some(Utc::now().timestamp() as u32),
            total_collections: 0,
            total_vectors: 0,
            total_size_bytes: 0,
            config: HashMap::new(),
        }
    }
}
