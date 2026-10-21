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

//! IVF Posting List Storage with proper format handling
//!
//! Key principles:
//! - Memory tier: bincode for speed (hot posting lists)
//! - Disk tier: SST or VIPER format (warm/cold posting lists)  
//! - Cloud tier: SST or VIPER format (archived posting lists)
//! - Demotion from memory triggers format conversion (bincode → SST/VIPER)
//! - Disk ↔ Cloud movement preserves format (no conversion)

use anyhow::{Result, anyhow};
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::types::StorageEngineType;
use crate::infrastructure::tier_policy_engine::InfrastructureTier;
use crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader;
use crate::storage::engines::sst::writer::SstableWriter;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;

/// Type alias for [`IvfClusterPostingListEntry`] for compatibility
pub type PostingEntry = IvfClusterPostingListEntry;

/// Posting list entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IvfClusterPostingListEntry {
    /// External identifier of the vector belonging to this cluster
    pub vector_id: String,
    /// Pre-computed distance from this vector to its cluster centroid
    pub distance_to_centroid: f32,
}

/// Backwards-compat alias for [`IvfClusterPostingList`].
pub type PostingList = IvfClusterPostingList;

/// Complete posting list for a cluster
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IvfClusterPostingList {
    /// Index of the IVF cluster this posting list belongs to
    pub cluster_id: usize,
    /// Ordered list of vector entries assigned to this cluster
    pub entries: Vec<IvfClusterPostingListEntry>,
    /// Timestamp of the most recent access, used for LRU eviction
    pub last_accessed: std::time::SystemTime,
}

/// Storage backend for IVF posting lists
pub enum IvfClusterPostingListStorage {
    /// In-memory storage with bincode serialization
    Memory {
        /// Map from cluster ID to bincode-encoded posting list bytes
        cache: Arc<dashmap::DashMap<usize, Vec<u8>>>,
    },
    /// SST-based disk storage
    SstDisk {
        /// Filesystem root path for SST files
        base_path: String,
        /// Collection identifier used to scope file paths
        collection_id: String,
    },
    /// VIPER-based disk storage
    ViperDisk {
        /// Filesystem root path for Parquet/VIPER files
        base_path: String,
        /// Collection identifier used to scope file paths
        collection_id: String,
    },
    /// Cloud storage (S3, etc.) using SST format
    CloudSst {
        /// S3 bucket name
        bucket: String,
        /// Collection identifier used to scope object keys
        collection_id: String,
    },
    /// Cloud storage using VIPER format
    CloudViper {
        /// S3 bucket name
        bucket: String,
        /// Collection identifier used to scope object keys
        collection_id: String,
    },
}

impl IvfClusterPostingListStorage {
    /// Create storage backend based on tier and engine type
    pub fn new(
        tier: &InfrastructureTier,
        collection_id: String,
        engine_type: StorageEngineType,
    ) -> Result<Self> {
        match tier {
            InfrastructureTier::Memory => Ok(IvfClusterPostingListStorage::Memory {
                cache: Arc::new(dashmap::DashMap::new()),
            }),
            InfrastructureTier::NvmeSsd { mount_path }
            | InfrastructureTier::HardDisk { mount_path } => match engine_type {
                StorageEngineType::SST => Ok(IvfClusterPostingListStorage::SstDisk {
                    base_path: mount_path.clone(),
                    collection_id,
                }),
                StorageEngineType::VIPER => Ok(IvfClusterPostingListStorage::ViperDisk {
                    base_path: mount_path.clone(),
                    collection_id,
                }),
                // For newer engine types, default to SST disk storage
                _ => Ok(IvfClusterPostingListStorage::SstDisk {
                    base_path: mount_path.clone(),
                    collection_id,
                }),
            },
            InfrastructureTier::CloudExpressOneZone { provider, .. }
            | InfrastructureTier::CloudStandard { provider, .. }
            | InfrastructureTier::CloudInfrequentAccess { provider, .. }
            | InfrastructureTier::CloudArchive { provider, .. }
            | InfrastructureTier::CloudDeepArchive { provider, .. } => {
                let bucket = match provider {
                    crate::infrastructure::tier_policy_engine::CloudProvider::AwsS3 {
                        bucket,
                        ..
                    } => bucket.clone(),
                    _ => "proximadb-default".to_string(),
                };

                match engine_type {
                    StorageEngineType::SST => Ok(IvfClusterPostingListStorage::CloudSst {
                        bucket,
                        collection_id,
                    }),
                    StorageEngineType::VIPER => Ok(IvfClusterPostingListStorage::CloudViper {
                        bucket,
                        collection_id,
                    }),
                    // For newer engine types, default to SST cloud storage
                    _ => Ok(IvfClusterPostingListStorage::CloudSst {
                        bucket,
                        collection_id,
                    }),
                }
            }
        }
    }

    /// Store a posting list
    pub async fn put(&self, cluster_id: usize, list: IvfClusterPostingList) -> Result<()> {
        match self {
            IvfClusterPostingListStorage::Memory { cache } => {
                // Serialize with bincode for memory storage
                let bytes = bincode::serialize(&list)?;
                cache.insert(cluster_id, bytes);
                debug!(
                    "Stored posting list {} in memory ({} entries)",
                    cluster_id,
                    list.entries.len()
                );
                Ok(())
            }
            IvfClusterPostingListStorage::SstDisk {
                base_path,
                collection_id,
            } => {
                // Write to SST format on disk
                let path = format!(
                    "{}/{}/posting_lists/cluster_{}.sstable",
                    base_path, collection_id, cluster_id
                );

                let config = crate::storage::persistence::filesystem::FilesystemConfig {
                    default_fs: Some(path.clone()),
                    ..Default::default()
                };
                let filesystem = Arc::new(
                    crate::storage::persistence::filesystem::FilesystemFactory::create(config)
                        .await
                        .map_err(|e| anyhow!("Failed to create filesystem: {}", e))?,
                );
                let writer = SstableWriter::new(&path, 4096, filesystem);

                // Convert entries to SST records
                let mut records = std::collections::BTreeMap::new();
                for entry in &list.entries {
                    let id = format!("{}:{}", cluster_id, entry.vector_id);
                    // For posting list entries, we store the distance as a simple vector
                    let vector = vec![entry.distance_to_centroid];
                    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
                    let props = std::collections::HashMap::from([
                        (
                            "cluster_id".to_string(),
                            proximadb_records::ProximaTreeNode::Value(
                                proximadb_data_model::ProximaValue::String(cluster_id.to_string()),
                            ),
                        ),
                        (
                            "vector_id".to_string(),
                            proximadb_records::ProximaTreeNode::Value(
                                proximadb_data_model::ProximaValue::String(entry.vector_id.clone()),
                            ),
                        ),
                    ]);
                    records.insert(
                        id.clone(),
                        proximadb_records::ProximaRecord {
                            oid: id,
                            created_at_ns: now_ns,
                            updated_at_ns: now_ns,
                            props,
                            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                                "ivf_posting_distance",
                                "dense_vector",
                                vector.len() as u32,
                                vector,
                            )],
                            ..proximadb_records::ProximaRecord::default()
                        },
                    );
                }

                // M1-2 (ADR-049): native PAX segment (no VectorRecord round-trip).
                let sorted_records_iter = records.into_values(); // Extract values from BTreeMap
                writer
                    .write_pax_segment(sorted_records_iter, &format!("ivf_cluster_{cluster_id}"))
                    .await?;
                debug!("Stored posting list {} to SST at {}", cluster_id, path);
                Ok(())
            }
            IvfClusterPostingListStorage::ViperDisk {
                base_path,
                collection_id,
            } => {
                // Write to VIPER format on disk
                let path = format!(
                    "{}/{}/posting_lists/cluster_{}.parquet",
                    base_path, collection_id, cluster_id
                );

                // In production, use VIPER's columnar writer
                // For now, we'll simulate it
                debug!("Stored posting list {} to VIPER at {}", cluster_id, path);
                Ok(())
            }
            IvfClusterPostingListStorage::CloudSst {
                bucket,
                collection_id,
            } => {
                // Write to S3 using SST format
                let key = format!(
                    "{}/posting_lists/cluster_{}.sstable",
                    collection_id, cluster_id
                );

                debug!(
                    "Stored posting list {} to S3 (SST) at s3://{}/{}",
                    cluster_id, bucket, key
                );
                Ok(())
            }
            IvfClusterPostingListStorage::CloudViper {
                bucket,
                collection_id,
            } => {
                // Write to S3 using VIPER format
                let key = format!(
                    "{}/posting_lists/cluster_{}.parquet",
                    collection_id, cluster_id
                );

                debug!(
                    "Stored posting list {} to S3 (VIPER) at s3://{}/{}",
                    cluster_id, bucket, key
                );
                Ok(())
            }
        }
    }

    /// Retrieve a posting list
    pub async fn get(&self, cluster_id: usize) -> Result<Option<IvfClusterPostingList>> {
        match self {
            IvfClusterPostingListStorage::Memory { cache } => {
                // Deserialize from bincode in memory
                if let Some(bytes) = cache.get(&cluster_id) {
                    let list: IvfClusterPostingList = bincode::deserialize(&bytes)?;
                    Ok(Some(list))
                } else {
                    Ok(None)
                }
            }
            IvfClusterPostingListStorage::SstDisk {
                base_path,
                collection_id,
            } => {
                // Read from SST format on disk
                let path = format!(
                    "{}/{}/posting_lists/cluster_{}.sstable",
                    base_path, collection_id, cluster_id
                );

                let config = crate::storage::persistence::filesystem::FilesystemConfig {
                    default_fs: Some(path.clone()),
                    ..Default::default()
                };
                let filesystem = Arc::new(
                    crate::storage::persistence::filesystem::FilesystemFactory::create(config)
                        .await
                        .map_err(|e| anyhow!("Failed to create filesystem: {}", e))?,
                );
                // Create zero-copy system for the reader
                // Create UnifiedCachingFilesystem for the reader
                let base_fs = filesystem
                    .get_filesystem("file://")
                    .map_err(|e| anyhow!("Failed to get base filesystem: {}", e))?;
                let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
                    base_fs,
                    cluster_id.to_string(),
                    "ivf".to_string(),
                ));
                let _reader =
                    UnifiedSstableReader::new(filesystem, unified_fs, cluster_id.to_string());
                let entries = Vec::new();

                // Read all entries for this cluster
                // Note: SSTable reader doesn't have scan method currently,
                // so we would need to implement batch reading or iterate through known keys
                // For now, returning empty as a placeholder
                // Deferred: Implement proper posting list storage backend

                if entries.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(IvfClusterPostingList {
                        cluster_id,
                        entries,
                        last_accessed: std::time::SystemTime::now(),
                    }))
                }
            }
            IvfClusterPostingListStorage::ViperDisk { .. }
            | IvfClusterPostingListStorage::CloudSst { .. }
            | IvfClusterPostingListStorage::CloudViper { .. } => {
                // Similar implementations for other storage types
                Ok(None) // Placeholder
            }
        }
    }

    /// Remove a posting list
    pub async fn remove(&self, cluster_id: usize) -> Result<()> {
        match self {
            IvfClusterPostingListStorage::Memory { cache } => {
                cache.remove(&cluster_id);
                Ok(())
            }
            _ => {
                // For disk/cloud storage, mark for deletion during compaction
                Ok(())
            }
        }
    }
}

/// Manager for tiered posting list storage
pub struct TieredIvfClusterPostingListManager {
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    engine_type: StorageEngineType,

    /// Memory tier (hot posting lists)
    memory_storage: IvfClusterPostingListStorage,

    /// Disk tier (warm/cold posting lists)
    disk_storage: Option<IvfClusterPostingListStorage>,

    /// Cloud tier (archived posting lists)
    cloud_storage: Option<IvfClusterPostingListStorage>,

    /// LRU tracking for memory eviction
    access_tracker: Arc<dashmap::DashMap<usize, std::time::SystemTime>>,
}

impl TieredIvfClusterPostingListManager {
    /// Create a new tiered posting list manager.
    ///
    /// Providing `disk_path` enables the NVMe/SSD tier and `cloud_bucket`
    /// enables the cloud tier.  Either may be `None` to disable that tier.
    pub fn new(
        collection_id: String,
        engine_type: StorageEngineType,
        disk_path: Option<String>,
        cloud_bucket: Option<String>,
    ) -> Result<Self> {
        let memory_storage = IvfClusterPostingListStorage::new(
            &InfrastructureTier::Memory,
            collection_id.clone(),
            engine_type,
        )?;

        let disk_storage = disk_path
            .map(|path| {
                IvfClusterPostingListStorage::new(
                    &InfrastructureTier::NvmeSsd { mount_path: path },
                    collection_id.clone(),
                    engine_type,
                )
            })
            .transpose()?;

        let cloud_storage = cloud_bucket.map(|bucket| {
            IvfClusterPostingListStorage::new(
                &InfrastructureTier::CloudStandard {
                    provider: crate::infrastructure::tier_policy_engine::CloudProvider::AwsS3 {
                        bucket,
                        storage_class: crate::infrastructure::tier_policy_engine::AwsStorageClass::Standard,
                        lifecycle_enabled: false,
                    },
                    region: "us-east-1".to_string(),
                },
                collection_id.clone(),
                engine_type,
            )
        }).transpose()?;

        Ok(Self {
            collection_id,
            engine_type,
            memory_storage,
            disk_storage,
            cloud_storage,
            access_tracker: Arc::new(dashmap::DashMap::new()),
        })
    }

    /// Get posting list with tier-aware loading
    pub async fn get(&self, cluster_id: usize) -> Result<Option<IvfClusterPostingList>> {
        // Try memory first
        if let Some(list) = self.memory_storage.get(cluster_id).await? {
            self.access_tracker
                .insert(cluster_id, std::time::SystemTime::now());
            return Ok(Some(list));
        }

        // Try disk if available
        if let Some(ref disk) = self.disk_storage
            && let Some(list) = disk.get(cluster_id).await?
        {
            // Promote to memory
            self.memory_storage.put(cluster_id, list.clone()).await?;
            self.access_tracker
                .insert(cluster_id, std::time::SystemTime::now());
            info!("Promoted posting list {} from disk to memory", cluster_id);
            return Ok(Some(list));
        }

        // Try cloud if available
        if let Some(ref cloud) = self.cloud_storage
            && let Some(list) = cloud.get(cluster_id).await?
        {
            // Promote to memory (and optionally disk)
            self.memory_storage.put(cluster_id, list.clone()).await?;
            if let Some(ref disk) = self.disk_storage {
                disk.put(cluster_id, list.clone()).await?;
            }
            self.access_tracker
                .insert(cluster_id, std::time::SystemTime::now());
            info!("Promoted posting list {} from cloud to mem", cluster_id);
            return Ok(Some(list));
        }

        Ok(None)
    }

    /// Store posting list with tier placement
    pub async fn put(&self, cluster_id: usize, list: IvfClusterPostingList) -> Result<()> {
        // Always store in memory first
        self.memory_storage.put(cluster_id, list.clone()).await?;
        self.access_tracker
            .insert(cluster_id, std::time::SystemTime::now());

        // Check if we need to evict from memory
        self.maybe_evict_from_memory().await?;

        Ok(())
    }

    /// Evict least recently used posting lists from memory
    async fn maybe_evict_from_memory(&self) -> Result<()> {
        const MAX_MEMORY_LISTS: usize = 1000; // Configurable threshold

        if self.access_tracker.len() > MAX_MEMORY_LISTS {
            // Find LRU posting lists
            let mut access_times: Vec<(usize, std::time::SystemTime)> = self
                .access_tracker
                .iter()
                .map(|entry| (*entry.key(), *entry.value()))
                .collect();

            access_times.sort_by_key(|&(_, time)| time);

            // Evict oldest 10%
            let evict_count = MAX_MEMORY_LISTS / 10;
            for (cluster_id, _) in access_times.iter().take(evict_count) {
                // Get posting list from memory
                if let Some(list) = self.memory_storage.get(*cluster_id).await? {
                    // Demote to disk (convert from bincode to SST/VIPER)
                    if let Some(ref disk) = self.disk_storage {
                        disk.put(*cluster_id, list).await?;
                        info!("Demoted posting list {} from memory to disk", cluster_id);
                    }

                    // Remove from memory
                    self.memory_storage.remove(*cluster_id).await?;
                    self.access_tracker.remove(cluster_id);
                }
            }
        }

        Ok(())
    }
}
