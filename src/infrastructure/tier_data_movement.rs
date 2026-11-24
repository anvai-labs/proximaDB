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

//! Tier data movement with proper format handling
//!
//! Key principle: Each tier uses its native storage format
//! - Memory: bincode for fast serialization
//! - Disk (NVMe/HDD): SST native format
//! - Cloud (S3): SST native format
//!
//! Data movement between tiers respects format boundaries

use anyhow::{Result, anyhow};
use std::sync::Arc;
use tracing::{debug, info};

use crate::infrastructure::tier_policy_engine::InfrastructureTier;
use crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader;
use crate::storage::engines::sst::writer::SstableWriter;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use proximadb_records::ProximaRecord;
/// Data format used by each tier
#[derive(Debug, Clone)]
pub enum TierDataFormat {
    /// Memory tier uses bincode for speed
    Bincode,
    /// SST format for hybrid columnar storage (ProximaBlocks)
    SST,
}

/// Tier data movement coordinator
pub struct TierDataMovement {
    /// Collection this coordinator operates on
    collection_id: String,
}

/// Canonical storage engine type (re-exported from `proximadb-storage-common` via
/// `core::types`). The previous local 2-variant copy was
/// consolidated into the single source of truth.
pub use crate::core::types::StorageEngineType;

impl TierDataMovement {
    /// Create a new coordinator for the given collection and storage engine type
    pub fn new(collection_id: String) -> Self {
        Self { collection_id }
    }

    /// Get the native format for a tier
    pub fn get_tier_format(&self, tier: &InfrastructureTier) -> TierDataFormat {
        match tier {
            InfrastructureTier::Memory => TierDataFormat::Bincode,
            InfrastructureTier::NvmeSsd { .. }
            | InfrastructureTier::HardDisk { .. }
            | InfrastructureTier::CloudExpressOneZone { .. }
            | InfrastructureTier::CloudStandard { .. }
            | InfrastructureTier::CloudInfrequentAccess { .. }
            | InfrastructureTier::CloudArchive { .. }
            | InfrastructureTier::CloudDeepArchive { .. } => {
                // Every remaining engine defaults to the SST (ProximaBlocks)
                // tier format — consistent with the strategy→type mapping in
                // `storage/traits` (Cedar/Chrono→SST). VIPER was removed
                // (ADR-093).
                TierDataFormat::SST
            }
        }
    }

    /// Promote data from lower tier to higher tier
    pub async fn promote_data(
        &self,
        items: Vec<String>, // Vector IDs to promote
        from_tier: &InfrastructureTier,
        to_tier: &InfrastructureTier,
    ) -> Result<PromotionResult> {
        let from_format = self.get_tier_format(from_tier);
        let to_format = self.get_tier_format(to_tier);

        info!(
            "Promoting {} items from {:?} ({:?}) to {:?} ({:?})",
            items.len(),
            from_tier,
            from_format,
            to_tier,
            to_format
        );

        // Read data from source tier
        let vectors = self.read_from_tier(&items, from_tier, &from_format).await?;

        // Write data to destination tier
        let bytes_written = self.write_to_tier(vectors, to_tier, &to_format).await?;

        // Remove from source tier after successful promotion
        self.remove_from_tier(&items, from_tier).await?;

        Ok(PromotionResult {
            items_promoted: items.len(),
            bytes_written,
        })
    }

    /// Demote data from higher tier to lower tier
    pub async fn demote_data(
        &self,
        items: Vec<String>, // Vector IDs to demote
        from_tier: &InfrastructureTier,
        to_tier: &InfrastructureTier,
    ) -> Result<DemotionResult> {
        let from_format = self.get_tier_format(from_tier);
        let to_format = self.get_tier_format(to_tier);

        info!(
            "Demoting {} items from {:?} ({:?}) to {:?} ({:?})",
            items.len(),
            from_tier,
            from_format,
            to_tier,
            to_format
        );

        // Read data from source tier
        let vectors = self.read_from_tier(&items, from_tier, &from_format).await?;

        // Write data to destination tier
        let bytes_written = self.write_to_tier(vectors, to_tier, &to_format).await?;

        // Remove from source tier after successful demotion
        self.remove_from_tier(&items, from_tier).await?;

        Ok(DemotionResult {
            items_demoted: items.len(),
            bytes_written,
            bytes_freed: self.estimate_tier_bytes(&items, from_tier),
        })
    }

    /// Read vectors from a tier using its native format
    async fn read_from_tier(
        &self,
        ids: &[String],
        tier: &InfrastructureTier,
        format: &TierDataFormat,
    ) -> Result<Vec<ProximaRecord>> {
        match format {
            TierDataFormat::Bincode => {
                // Read from memory tier using bincode deserialization
                self.read_from_memory_tier(ids, tier).await
            }
            TierDataFormat::SST => {
                // Read from SST format using SSTable reader
                self.read_from_sst_tier(ids, tier).await
            }
        }
    }

    /// Write vectors to a tier using its native format
    async fn write_to_tier(
        &self,
        records: Vec<ProximaRecord>,
        tier: &InfrastructureTier,
        format: &TierDataFormat,
    ) -> Result<usize> {
        match format {
            TierDataFormat::Bincode => {
                // Write to memory tier using bincode serialization
                self.write_to_memory_tier(records, tier).await
            }
            TierDataFormat::SST => {
                // Write to SST format using SSTable writer
                self.write_to_sst_tier(records, tier).await
            }
        }
    }

    /// Read from memory tier (bincode format)
    async fn read_from_memory_tier(
        &self,
        ids: &[String],
        _tier: &InfrastructureTier,
    ) -> Result<Vec<ProximaRecord>> {
        debug!(
            "Reading {} vectors from memory tier using bincode",
            ids.len()
        );

        // In production, this would:
        // 1. Look up vectors in memory cache
        // 2. Deserialize using bincode
        // For now, return placeholder
        Ok(vec![])
    }

    /// Write to memory tier (bincode format)
    async fn write_to_memory_tier(
        &self,
        records: Vec<ProximaRecord>,
        _tier: &InfrastructureTier,
    ) -> Result<usize> {
        debug!(
            "Writing {} records to memory tier using bincode",
            records.len()
        );

        // Serialize canonical records using bincode.
        let mut total_bytes = 0;
        for record in &records {
            let serialized = bincode::serialize(record)?;
            total_bytes += serialized.len();
            // In production, store in memory cache
        }

        Ok(total_bytes)
    }

    /// Read from SST tier
    async fn read_from_sst_tier(
        &self,
        ids: &[String],
        tier: &InfrastructureTier,
    ) -> Result<Vec<ProximaRecord>> {
        let path = self.get_tier_path(tier)?;
        debug!("Reading {} vectors from SST at {}", ids.len(), path);

        // Use SSTable reader to load vectors
        let config = crate::storage::persistence::filesystem::FilesystemConfig {
            default_fs: Some(path.clone()),
            ..Default::default()
        };
        let filesystem = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(config)
                .await
                .map_err(|e| anyhow!("Failed to create filesystem: {}", e))?,
        );
        // Create UnifiedCachingFilesystem for the reader
        let base_fs = filesystem
            .get_filesystem("file://")
            .map_err(|e| anyhow!("Failed to get base filesystem: {}", e))?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            self.collection_id.clone(),
            "sst".to_string(),
        ));
        let reader = UnifiedSstableReader::new(filesystem, unified_fs, self.collection_id.clone());
        let mut records = Vec::new();

        // SSTable reader reads individual vectors
        // Note: We're using a dummy SST file path for each vector since SSTable reader
        // expects file paths, not directory paths
        let sst_file = format!("{}/data.sstable", path);
        for id in ids {
            if let Some(record) = reader.vector(&sst_file, id).await? {
                records.push(record);
            }
        }

        Ok(records)
    }

    /// Write to SST tier
    async fn write_to_sst_tier(
        &self,
        records: Vec<ProximaRecord>,
        tier: &InfrastructureTier,
    ) -> Result<usize> {
        let path = self.get_tier_path(tier)?;
        debug!("Writing {} records to SST at {}", records.len(), path);

        // Use SSTable writer to store vectors
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

        // Pass records through without any ID manipulation or deduplication.
        // Collection service handles ID validation when needed
        let mut total_bytes = 0;
        for record in &records {
            total_bytes += record
                .embeddings
                .first()
                .map(|embedding| embedding.values.len() * 4)
                .unwrap_or_default();
        }

        // M1-2 (ADR-049): native PAX segment (no VectorRecord round-trip).
        writer
            .write_pax_segment(records.into_iter(), &self.collection_id)
            .await?;
        Ok(total_bytes)
    }

    /// Remove vectors from a tier
    async fn remove_from_tier(&self, ids: &[String], tier: &InfrastructureTier) -> Result<()> {
        debug!("Removing {} vectors from {:?}", ids.len(), tier);

        // In production, this would:
        // 1. Mark vectors as deleted in the tier
        // 2. Schedule compaction to reclaim space

        Ok(())
    }

    /// Get the storage path for a tier
    fn get_tier_path(&self, tier: &InfrastructureTier) -> Result<String> {
        match tier {
            InfrastructureTier::Memory => Err(anyhow!("Memory tier has no file path")),
            InfrastructureTier::NvmeSsd { mount_path } => {
                Ok(format!("{}/{}", mount_path, self.collection_id))
            }
            InfrastructureTier::HardDisk { mount_path } => {
                Ok(format!("{}/{}", mount_path, self.collection_id))
            }
            InfrastructureTier::CloudExpressOneZone { provider, .. }
            | InfrastructureTier::CloudStandard { provider, .. }
            | InfrastructureTier::CloudInfrequentAccess { provider, .. }
            | InfrastructureTier::CloudArchive { provider, .. }
            | InfrastructureTier::CloudDeepArchive { provider, .. } => {
                // Extract bucket/container from provider
                match provider {
                    crate::infrastructure::tier_policy_engine::CloudProvider::AwsS3 {
                        bucket,
                        ..
                    } => Ok(format!("s3://{}/{}", bucket, self.collection_id)),
                    _ => Ok(format!("cloud://{}", self.collection_id)),
                }
            }
        }
    }

    /// Estimate bytes used by items in a tier
    fn estimate_tier_bytes(&self, ids: &[String], _tier: &InfrastructureTier) -> usize {
        // Rough estimate: 1KB per vector + overhead
        ids.len() * 1024
    }
}

/// Result of a tier promotion operation
#[derive(Debug)]
pub struct PromotionResult {
    /// Number of items successfully promoted to the faster tier
    pub items_promoted: usize,
    /// Total bytes written to the destination tier
    pub bytes_written: usize,
}

/// Result of a tier demotion operation
#[derive(Debug)]
pub struct DemotionResult {
    /// Number of items successfully demoted to the slower tier
    pub items_demoted: usize,
    /// Total bytes written to the destination tier
    pub bytes_written: usize,
    /// Total bytes freed from the source tier
    pub bytes_freed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_format_selection() {
        let movement = TierDataMovement::new("test_collection".to_string());

        // Memory always uses bincode
        assert!(matches!(
            movement.get_tier_format(&InfrastructureTier::Memory),
            TierDataFormat::Bincode
        ));

        // Disk tiers use engine format
        assert!(matches!(
            movement.get_tier_format(&InfrastructureTier::NvmeSsd {
                mount_path: "/mnt/nvme".to_string()
            }),
            TierDataFormat::SST
        ));
    }
}
