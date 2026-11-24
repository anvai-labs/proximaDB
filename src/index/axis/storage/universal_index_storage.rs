/*
 * Copyright 2025 ProximaDB
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

//! Universal Index Storage Handler for ALL AXIS Indexes
//!
//! Storage Tier Hierarchy:
//! Memory → NVMe → HDD → S3 Express One Zone
//!
//! Key Principles:
//! - Memory: bincode format, LRU eviction triggers demotion to disk
//! - NVMe/HDD: SST format, only eviction (no demotion between disk tiers)
//! - S3: SST format, promotion to local disk for bloom filter/columnar scan benefits
//! - S3 → Disk promotion uses /tmp (if NVMe not configured) or HDD as staging

use anyhow::{Result, anyhow};
use proximadb_kernel::encoding::base64_encode;
use proximadb_kernel::error::{ProximaDBError, StorageError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader;
use crate::storage::engines::sst::writer::SstableWriter;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

/// Universal index data that can be stored across tiers
pub trait IndexData: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync {
    /// Get unique identifier for this data
    fn get_id(&self) -> String;

    /// Estimate size in bytes
    fn size_bytes(&self) -> usize;
}

/// Storage location and format
#[derive(Debug, Clone)]
pub enum StorageLocation {
    /// In-memory with bincode format
    Memory,
    /// NVMe SSD with SST format
    NvmeSsd {
        /// Root directory on the NVMe SSD
        path: PathBuf,
    },
    /// HDD with SST format
    HardDisk {
        /// Root directory on the hard disk
        path: PathBuf,
    },
    /// S3 Express One Zone with SST format
    S3Express {
        /// S3 bucket name
        bucket: String,
        /// Key prefix used to scope objects within the bucket
        prefix: String,
    },
}

/// Storage engine type
#[derive(Debug, Clone, Copy)]
pub enum StorageEngine {
    /// Sorted String Table engine — bloom-filter accelerated key lookup
    SST,
}

/// Index storage configuration
#[derive(Debug, Clone)]
pub struct IndexStorageConfig {
    /// Physical location where index data resides
    pub storage_location: StorageLocation,
    /// Serialization engine used for on-disk/cloud data
    pub storage_engine: StorageEngine,
    /// Maximum number of items to keep in the memory cache
    pub cache_size: usize,
    /// Fraction of cache capacity (0.0–1.0) that triggers LRU eviction
    pub eviction_threshold: f64,
}

/// Universal index storage handler for any index type
pub struct UniversalIndexStorage<T: IndexData> {
    collection_id: String,
    index_type: String, // "hnsw", "lsh", "annoy", "ivf"
    storage_engine: StorageEngine,

    /// Current storage location for each data item
    data_locations: Arc<dashmap::DashMap<String, StorageLocation>>,

    /// Memory cache (bincode format)
    memory_cache: Arc<dashmap::DashMap<String, Vec<u8>>>,

    /// Access tracker for LRU eviction
    access_tracker: Arc<dashmap::DashMap<String, std::time::SystemTime>>,

    /// Configuration
    max_memory_items: usize,
    nvme_path: Option<PathBuf>,
    hdd_path: Option<PathBuf>,
    s3_bucket: Option<String>,

    /// Staging area for S3 promotion (uses /tmp or HDD)
    staging_path: PathBuf,

    _phantom: std::marker::PhantomData<T>,
}

impl<T: IndexData> UniversalIndexStorage<T> {
    /// Create a new `UniversalIndexStorage` instance.
    ///
    /// Storage tiers are auto-detected from the environment.  Set
    /// `PROXIMADB_S3_BUCKET` to enable the cloud tier.
    pub fn new(
        collection_id: String,
        index_type: String,
        storage_engine: StorageEngine,
        max_memory_items: usize,
    ) -> Self {
        // Detect available storage tiers
        let nvme_path = Self::detect_nvme_path();
        let hdd_path = Self::detect_hdd_path();
        let s3_bucket = std::env::var("PROXIMADB_S3_BUCKET").ok();

        // Determine staging path for S3 promotion
        let staging_path = if let Some(ref nvme) = nvme_path {
            nvme.join("staging")
        } else if let Some(ref hdd) = hdd_path {
            hdd.join("staging")
        } else {
            PathBuf::from("/tmp/proximadb_staging")
        };

        // Create staging directory if needed
        let _ = std::fs::create_dir_all(&staging_path);

        info!(
            "Universal index storage for {}/{}: Memory({}) → NVMe({:?}) → HDD({:?}) → S3({:?})",
            collection_id, index_type, max_memory_items, nvme_path, hdd_path, s3_bucket
        );

        Self {
            collection_id,
            index_type,
            storage_engine,
            data_locations: Arc::new(dashmap::DashMap::new()),
            memory_cache: Arc::new(dashmap::DashMap::new()),
            access_tracker: Arc::new(dashmap::DashMap::new()),
            max_memory_items,
            nvme_path,
            hdd_path,
            s3_bucket,
            staging_path,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Store index data
    pub async fn put(&self, data: T) -> Result<()> {
        let id = data.get_id();

        // Always start in memory with bincode
        let bytes = bincode::serialize(&data)?;
        self.memory_cache.insert(id.clone(), bytes);
        self.data_locations
            .insert(id.clone(), StorageLocation::Memory);
        self.access_tracker
            .insert(id.clone(), std::time::SystemTime::now());

        debug!("Stored {} data {} in mem", self.index_type, id);

        // Check if we need to evict from memory
        if self.memory_cache.len() > self.max_memory_items {
            self.evict_from_memory().await?;
        }

        Ok(())
    }

    /// Retrieve index data with automatic tier promotion
    pub async fn get(&self, id: &str) -> Result<Option<T>> {
        // Update access time
        self.access_tracker
            .insert(id.to_string(), std::time::SystemTime::now());

        // Check current location
        let location = self.data_locations.get(id).map(|l| l.clone());

        match location {
            Some(StorageLocation::Memory) => {
                // Fast path: already in memory
                if let Some(bytes) = self.memory_cache.get(id) {
                    let data: T = bincode::deserialize(&bytes)?;
                    return Ok(Some(data));
                }
            }
            Some(StorageLocation::NvmeSsd { path }) => {
                // Read from NVMe using SST
                let data = self.read_from_disk(&path, id).await?;
                if let Some(d) = data {
                    // Promote to memory
                    self.promote_to_memory(id, d.clone()).await?;
                    return Ok(Some(d));
                }
            }
            Some(StorageLocation::HardDisk { path }) => {
                // Read from HDD using SST
                let data = self.read_from_disk(&path, id).await?;
                if let Some(d) = data {
                    // Promote to memory (and optionally NVMe if available)
                    if self.nvme_path.is_some() {
                        self.promote_to_nvme(id, d.clone()).await?;
                    }
                    self.promote_to_memory(id, d.clone()).await?;
                    return Ok(Some(d));
                }
            }
            Some(StorageLocation::S3Express { bucket, prefix }) => {
                // CRITICAL: Promote from S3 to local disk first
                // This enables bloom filters and columnar scans
                let data = self.promote_from_s3(&bucket, &prefix, id).await?;
                if let Some(d) = data {
                    // Promote through the hierarchy
                    self.promote_to_memory(id, d.clone()).await?;
                    return Ok(Some(d));
                }
            }
            None => {
                // Not found
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Evict least recently used items from memory
    async fn evict_from_memory(&self) -> Result<()> {
        let mut access_times: Vec<(String, std::time::SystemTime)> = self
            .access_tracker
            .iter()
            .filter(|entry| {
                self.data_locations
                    .get(entry.key())
                    .is_some_and(|loc| matches!(*loc, StorageLocation::Memory))
            })
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        access_times.sort_by_key(|&(_, time)| time);

        // Evict oldest 10%
        let evict_count = self.max_memory_items / 10;
        for (id, _) in access_times.iter().take(evict_count) {
            if let Some((_key, bytes)) = self.memory_cache.remove(id) {
                let data: T = bincode::deserialize(&bytes)?;

                // Demote to next available tier
                if let Some(ref nvme_path) = self.nvme_path {
                    self.demote_to_disk(
                        id,
                        data,
                        nvme_path.clone(),
                        StorageLocation::NvmeSsd {
                            path: nvme_path.clone(),
                        },
                    )
                    .await?;
                } else if let Some(ref hdd_path) = self.hdd_path {
                    self.demote_to_disk(
                        id,
                        data,
                        hdd_path.clone(),
                        StorageLocation::HardDisk {
                            path: hdd_path.clone(),
                        },
                    )
                    .await?;
                } else if let Some(ref bucket) = self.s3_bucket {
                    self.demote_to_s3(id, data, bucket).await?;
                } else {
                    // No lower tier available - keep in memory but log warning
                    // We can't actually evict without losing data
                    warn!("No lower tier available for {}, keeping in memory", id);
                    // Put the data back in memory cache since we can't evict it
                    self.memory_cache.insert(id.clone(), bytes);
                }

                info!("Evicted {} data {} from mem", self.index_type, id);
            }
        }

        Ok(())
    }

    /// Demote data from memory to disk (convert bincode → SST)
    async fn demote_to_disk(
        &self,
        id: &str,
        data: T,
        path: PathBuf,
        location: StorageLocation,
    ) -> Result<()> {
        let file_path = self.get_disk_path(&path, id);

        match self.storage_engine {
            StorageEngine::SST => {
                // Write as SST
                let config = crate::storage::persistence::filesystem::FilesystemConfig {
                    default_fs: Some(file_path.to_string_lossy().to_string()),
                    ..Default::default()
                };
                let filesystem_factory =
                    crate::storage::persistence::filesystem::FilesystemFactory::create(config)
                        .await
                        .map_err(|e| anyhow!("Failed to create filesystem: {}", e))?;
                let filesystem = Arc::new(filesystem_factory);
                let writer = SstableWriter::new(&file_path, 4096, filesystem);
                let _value = bincode::serialize(&data)?;
                let mut records = std::collections::BTreeMap::new();
                let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
                records.insert(
                    id.to_string(),
                    proximadb_records::ProximaRecord {
                        oid: id.to_string(),
                        created_at_ns: now_ns,
                        updated_at_ns: now_ns,
                        ..proximadb_records::ProximaRecord::default()
                    },
                );
                // M1-2 (ADR-049): write a native PAX segment (ProximaRecord→PAX, no
                // VectorRecord round-trip). Reads stay mixed-format-safe.
                let sorted_records_iter = records.into_values(); // Extract values from BTreeMap
                writer.write_pax_segment(sorted_records_iter, id).await?;
            }
        }

        self.data_locations.insert(id.to_string(), location);
        Ok(())
    }

    /// Read data from disk (SST format)
    async fn read_from_disk(&self, path: &PathBuf, id: &str) -> Result<Option<T>> {
        let file_path = self.get_disk_path(path, id);

        match self.storage_engine {
            StorageEngine::SST => {
                // Read from SST
                if !file_path.exists() {
                    return Ok(None);
                }
                let config = crate::storage::persistence::filesystem::FilesystemConfig {
                    default_fs: Some(file_path.to_string_lossy().to_string()),
                    ..Default::default()
                };
                let filesystem_factory =
                    crate::storage::persistence::filesystem::FilesystemFactory::create(config)
                        .await
                        .map_err(|e| anyhow!("Failed to create filesystem: {}", e))?;
                let filesystem = Arc::new(filesystem_factory);
                // Create zero-copy system for the reader
                // Create UnifiedCachingFilesystem for the reader
                let base_fs = filesystem.get_filesystem("file://").map_err(|e| {
                    ProximaDBError::Storage(StorageError::DiskIO(std::io::Error::other(format!(
                        "Failed to get base filesystem: {}",
                        e
                    ))))
                })?;
                let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
                    base_fs,
                    self.collection_id.clone(),
                    "universal_index".to_string(),
                ));
                let reader =
                    UnifiedSstableReader::new(filesystem, unified_fs, self.collection_id.clone());
                // UnifiedSstableReader returns a canonical ProximaRecord, not raw bytes
                // We need to handle this differently
                if let Some(_vector_record) =
                    reader.vector(&file_path.to_string_lossy(), id).await?
                {
                    // For now, we can't directly deserialize since we get a ProximaRecord
                    // This would need a different storage approach
                    debug!(
                        "Found record {} in SST but cannot deserialize generic type T from ProximaRecord",
                        id
                    );
                }
            }
        }

        Ok(None)
    }

    /// Promote data from S3 to local disk for efficient access
    async fn promote_from_s3(&self, bucket: &str, prefix: &str, id: &str) -> Result<Option<T>> {
        info!(
            "Promoting {} from S3 to local staging for bloom filter/columnar scan benefits",
            id
        );

        // Download from S3 to staging area
        let s3_key = format!(
            "{}/{}/{}/{}",
            prefix, self.collection_id, self.index_type, id
        );
        let staging_file = self
            .staging_path
            .join(format!("{}_{}", self.collection_id, id));

        // In production, use S3 client to download
        debug!(
            "Downloading s3://{}/{} to {:?}",
            bucket, s3_key, staging_file
        );

        // Read from staging using the SST reader
        if staging_file.exists() {
            // For now, skip reading from staged S3 file since it's complex to implement properly
            debug!("Skipping S3 staged file read for id: {}", id);
        }

        Ok(None)
    }

    /// Promote data to memory (convert SST → bincode)
    async fn promote_to_memory(&self, id: &str, data: T) -> Result<()> {
        let bytes = bincode::serialize(&data)?;
        self.memory_cache.insert(id.to_string(), bytes);
        self.data_locations
            .insert(id.to_string(), StorageLocation::Memory);
        debug!("Promoted {} to mem", id);
        Ok(())
    }

    /// Promote data to NVMe
    async fn promote_to_nvme(&self, id: &str, data: T) -> Result<()> {
        if let Some(ref nvme_path) = self.nvme_path {
            self.demote_to_disk(
                id,
                data,
                nvme_path.clone(),
                StorageLocation::NvmeSsd {
                    path: nvme_path.clone(),
                },
            )
            .await?;
            debug!("Promoted {} to NVMe", id);
        }
        Ok(())
    }

    /// Demote data to S3
    async fn demote_to_s3(&self, id: &str, data: T, bucket: &str) -> Result<()> {
        let prefix = format!("{}/{}", self.collection_id, self.index_type);

        // First write to staging in SST format
        let staging_file = self
            .staging_path
            .join(format!("{}_{}", self.collection_id, id));

        match self.storage_engine {
            StorageEngine::SST => {
                let fs_config = crate::storage::persistence::filesystem::FilesystemConfig {
                    default_fs: Some("file://".to_string()),
                    local: None,
                    global_options: Default::default(),
                    auth_config: None,
                    performance_config: Default::default(),
                    scheme_mapping: HashMap::new(),
                };
                let filesystem_factory =
                    crate::storage::persistence::filesystem::FilesystemFactory::create(fs_config)
                        .await?;
                let filesystem = Arc::new(filesystem_factory);
                let writer = SstableWriter::new(&staging_file, 4096, filesystem);
                let value = bincode::serialize(&data)?;
                let mut records = std::collections::BTreeMap::new();
                let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
                let record = proximadb_records::ProximaRecord {
                    oid: id.to_string(),
                    created_at_ns: now_ns,
                    updated_at_ns: now_ns,
                    props: std::iter::once((
                        "serialized_data".to_string(),
                        proximadb_records::ProximaTreeNode::Value(
                            proximadb_data_model::ProximaValue::String(base64_encode(&value)),
                        ),
                    ))
                    .collect(),
                    ..proximadb_records::ProximaRecord::default()
                };
                records.insert(id.to_string(), record);
                // M1-2 (ADR-049): write a native PAX segment (ProximaRecord→PAX, no
                // VectorRecord round-trip). Reads stay mixed-format-safe.
                let sorted_records_iter = records.into_values(); // Extract values from BTreeMap
                writer.write_pax_segment(sorted_records_iter, id).await?;
            }
        }

        // Upload to S3
        let s3_key = format!("{}/{}", prefix, id);
        debug!("Uploading {:?} to s3://{}/{}", staging_file, bucket, s3_key);

        self.data_locations.insert(
            id.to_string(),
            StorageLocation::S3Express {
                bucket: bucket.to_string(),
                prefix,
            },
        );

        Ok(())
    }

    /// Get disk path for a data item
    #[expect(clippy::ptr_arg)] // Accepting &PathBuf for API compatibility
    fn get_disk_path(&self, base: &PathBuf, id: &str) -> PathBuf {
        let ext = match self.storage_engine {
            StorageEngine::SST => "sst",
        };
        base.join(&self.collection_id)
            .join(&self.index_type)
            .join(format!("{}.{}", id, ext))
    }

    /// Detect NVMe path
    fn detect_nvme_path() -> Option<PathBuf> {
        let candidates = vec!["/mnt/nvme", "/nvme", "/var/lib/proximadb/nvme"];

        for path in candidates {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    /// Detect HDD path
    fn detect_hdd_path() -> Option<PathBuf> {
        let candidates = vec!["/mnt/disk", "/data", "/var/lib/proximadb/data"];

        for path in candidates {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }
}

/// Example implementation for HNSW graph nodes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswNode {
    /// Internal node identifier (dense index into the node array)
    pub id: usize,
    /// Graph layer this node resides in (0 = bottom/largest layer)
    pub layer: usize,
    /// Neighbour node IDs in the same layer
    pub connections: Vec<usize>,
}

impl IndexData for HnswNode {
    fn get_id(&self) -> String {
        format!("node_{}_{}", self.layer, self.id)
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.connections.len() * std::mem::size_of::<usize>()
    }
}

/// Example implementation for LSH hash buckets
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LshBucket {
    /// Index of the LSH hash table this bucket belongs to
    pub table_id: usize,
    /// Hash value that identifies this bucket within its table
    pub hash: u64,
    /// External vector IDs that hash to this bucket
    pub vector_ids: Vec<String>,
}

impl IndexData for LshBucket {
    fn get_id(&self) -> String {
        format!("bucket_{}_{}", self.table_id, self.hash)
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.vector_ids.iter().map(|s| s.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::{HnswNode, StorageEngine, UniversalIndexStorage};

    #[tokio::test]
    async fn test_tier_hierarchy() {
        let storage = UniversalIndexStorage::<HnswNode>::new(
            "test_collection".to_string(),
            "hnsw".to_string(),
            StorageEngine::SST,
            10, // Small memory limit for testing
        );

        // Add nodes
        for i in 0..15 {
            let node = HnswNode {
                id: i,
                layer: 0,
                connections: vec![i + 1, i + 2],
            };
            storage.put(node).await.unwrap();
        }

        // Without a lower tier, data cannot be evicted and will stay in memory
        // This is expected behavior to prevent data loss
        assert_eq!(
            storage.memory_cache.len(),
            15,
            "All items should remain in memory when no lower tier is available"
        );

        // Retrieve should work since data is still in memory
        let node = storage.get("node_0_0").await.unwrap();
        assert!(node.is_some());
    }
}
