//! Leveled compaction with liquid clustering for HELIX engine
//!
//! This module implements the core compaction logic that clusters similar
//! vectors together using PCA + Hilbert curves during compaction.

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::common::compaction_orchestrator::FilenameCodec;
use crate::storage::persistence::filesystem::{
    FileSystem, caching_filesystem::UnifiedCachingFilesystem,
};
use proximadb_records::ProximaRecord;

use super::clustering::{
    HilbertKey, LiquidClusteringConfig, PCAModel, QueryPatternTracker, sort_by_hilbert,
};
use super::liquid_clustering::LiquidClusteringCoordinator;
use super::proxima::{extract_helix_metadata, write_helix_sstable};
use super::{HelixConfig, SStableMetadata};

/// Leveled compactor for HELIX engine
pub struct LeveledCompactor {
    config: HelixConfig,
    filesystem: Arc<UnifiedCachingFilesystem>,
    data_dir: PathBuf,
    liquid_config: LiquidClusteringConfig,
    query_tracker: Arc<RwLock<QueryPatternTracker>>,
    filename_codec: FilenameCodec,
}

impl LeveledCompactor {
    /// Create a new compactor
    pub fn new(
        config: HelixConfig,
        filesystem: Arc<UnifiedCachingFilesystem>,
        data_dir: PathBuf,
    ) -> Self {
        Self {
            config,
            filesystem,
            data_dir,
            liquid_config: LiquidClusteringConfig::default(),
            query_tracker: Arc::new(RwLock::new(QueryPatternTracker::default())),
            filename_codec: FilenameCodec::new(),
        }
    }

    /// Compact L0 to L1 with optimized merge for pre-sorted files
    pub async fn compact_l0_to_l1(
        &self,
        levels: Arc<RwLock<HashMap<usize, Vec<SStableMetadata>>>>,
    ) -> Result<(usize, u64)> {
        info!("Starting L0 to L1 compaction (optimized for pre-sorted files)");

        // Get L0 files
        let l0_files = {
            let levels_read = levels.read().await;
            levels_read.get(&0).cloned().unwrap_or_default()
        };

        if l0_files.len() < self.config.level0_file_num_compaction_trigger {
            return Ok((0, 0));
        }

        // Check if all L0 files have Hilbert ranges (sorted flush optimization)
        let all_sorted = l0_files.iter().all(|f| f.hilbert_range.is_some());

        if all_sorted {
            info!(
                "All {} L0 files are pre-sorted by Hilbert key - using optimized merge",
                l0_files.len()
            );
            // Optimized path: merge pre-sorted files
            return self.merge_sorted_l0_files(levels, l0_files).await;
        }

        info!("Mixed sorted/unsorted L0 files - using full re-clustering");

        // Fallback path: read all records and re-cluster
        // Newest copy per id across the whole input (TD-ENGINE-MVCC-1).
        let input_paths: Vec<&PathBuf> = l0_files.iter().map(|s| &s.path).collect();
        let (mut all_records, dead_ids) = self.read_sstables_resolved(&input_paths).await?;

        info!(
            "Read {} vectors from {} L0 files",
            all_records.len(),
            l0_files.len()
        );

        // Train PCA model if we have enough data
        let trainable = pca_training_records(&all_records, &dead_ids);
        let pca_model = if trainable.len() > 100 {
            Some(PCAModel::train(&trainable, self.config.pca_dimensions)?)
        } else {
            None
        };

        // Compute Hilbert keys using configured bits per dimension
        let hilbert_keys: Vec<HilbertKey> = if let Some(ref model) = pca_model {
            all_records
                .iter()
                .map(|record| {
                    model
                        .project_and_compute_hilbert_with_config(
                            &record.vector,
                            self.config.hilbert_bits_per_dimension,
                        )
                        .unwrap_or(0)
                })
                .collect()
        } else {
            // No PCA, use simple hash
            all_records
                .iter()
                .enumerate()
                .map(|(i, _)| i as u64)
                .collect()
        };

        // Sort records by Hilbert key
        sort_by_hilbert(&mut all_records, &hilbert_keys)?;

        // Split into L1 files (target size based on config)
        let target_file_size = 100; // vectors per file
        let mut bytes_written = 0u64;
        let mut new_l1_files = Vec::new();

        for (chunk_idx, chunk) in all_records.chunks(target_file_size).enumerate() {
            let filename = self.filename_codec.generate(1, "helix");
            let file_path = self.data_dir.join(&filename);

            // Get Hilbert keys for this chunk
            let chunk_start = chunk_idx * target_file_size;
            let chunk_end = std::cmp::min(chunk_start + chunk.len(), hilbert_keys.len());
            let chunk_keys = &hilbert_keys[chunk_start..chunk_end];

            // Write L1 file with clustering
            let bytes = write_helix_sstable(
                &self.filesystem,
                &file_path,
                chunk,
                self.config.proxima_block_size,
                crate::storage::engines::constants::HELIX_MAGIC,
                Some(chunk_keys),
                Some(256), // Default Hilbert curve size
            )
            .await?;

            bytes_written += bytes;

            // Generate bloom filter for chunk IDs
            let bloom_filter = if !chunk.is_empty() {
                use crate::core::bloom::{BloomFilterConfig, factory::BloomFilterFactory};

                let bloom_config = BloomFilterConfig::for_sstable(chunk.len());
                let mut bloom = BloomFilterFactory::create(&bloom_config);

                for record in chunk {
                    bloom.insert(record.id.as_bytes());
                }

                bloom.serialize().ok()
            } else {
                None
            };

            // Create metadata
            let metadata = SStableMetadata {
                path: file_path,
                level: 1,
                hilbert_range: Some((
                    *chunk_keys.iter().min().unwrap_or(&0),
                    *chunk_keys.iter().max().unwrap_or(&0),
                )),
                num_vectors: chunk.len(),
                size_bytes: bytes,
                created_at: chrono::Utc::now(),
                blocks: extract_helix_metadata(
                    chunk,
                    self.config.proxima_block_size,
                    Some(chunk_keys),
                )
                .into_iter()
                .map(|h| h.proxima_metadata)
                .collect(),
                bloom_filter,
            };

            new_l1_files.push(metadata);
        }

        // Update levels
        {
            let mut levels_write = levels.write().await;

            debug!(
                "HELIX Compaction: Removing {} L0 files from metadata",
                l0_files.len()
            );
            for sstable in &l0_files {
                debug!("  - L0 file marked for removal: {}", sstable.path.display());
            }

            // Remove L0 files from metadata
            levels_write.remove(&0);

            debug!(
                "HELIX Compaction: Adding {} new L1 files to metadata",
                new_l1_files.len()
            );
            for sstable in &new_l1_files {
                debug!(
                    "  + L1 file created: {} (Hilbert range: {:?})",
                    sstable.path.display(),
                    sstable.hilbert_range
                );
            }

            // Add new L1 files
            levels_write.entry(1).or_default().extend(new_l1_files);
        }

        // Delete old L0 files from disk
        debug!(
            "HELIX Compaction: Deleting {} L0 files from disk",
            l0_files.len()
        );
        for sstable in &l0_files {
            let file_path = sstable.path.to_str().unwrap_or("");
            debug!("  - Deleting L0 file: {}", file_path);
            self.filesystem.delete(file_path).await.map_err(|e| {
                warn!("Failed to delete L0 file {}: {}", file_path, e);
                e
            })?;
            info!("  ✓ Deleted L0 file: {}", file_path);
        }

        info!(
            "✓ HELIX Compaction complete: {} L0 files → L1, {} bytes written, {} L0 files deleted",
            l0_files.len(),
            bytes_written,
            l0_files.len()
        );

        Ok((l0_files.len(), bytes_written))
    }

    /// Optimized merge for pre-sorted L0 files
    async fn merge_sorted_l0_files(
        &self,
        levels: Arc<RwLock<HashMap<usize, Vec<SStableMetadata>>>>,
        l0_files: Vec<SStableMetadata>,
    ) -> Result<(usize, u64)> {
        info!(
            "Performing optimized k-way merge of {} pre-sorted L0 files",
            l0_files.len()
        );

        // Since files are already sorted by Hilbert key, we can do a k-way merge
        // This is much more efficient than reading all records and re-sorting

        // (record, hilbert key) pairs, so resolution keeps the pairing.
        let mut pairs: Vec<(ProximaRecord, u64)> = Vec::new();

        // For simplicity, read all records but preserve sorting
        // In production, implement proper k-way merge with iterators
        for sstable in &l0_files {
            let records = self.read_sstable(&sstable.path).await?;

            // Records are already sorted, just append
            if let Some((min_key, max_key)) = sstable.hilbert_range {
                let num_records = records.len();
                if num_records > 0 {
                    // Generate keys based on range
                    let step = if num_records > 1 {
                        (max_key - min_key) / (num_records as u64 - 1)
                    } else {
                        0
                    };

                    for (i, record) in records.into_iter().enumerate() {
                        pairs.push((record, min_key + (i as u64 * step)));
                    }
                }
            } else {
                // Shouldn't happen with sorted flush
                pairs.extend(records.into_iter().map(|r| (r, 0)));
            }
        }

        // Newest copy per id across the whole input, dead winners kept
        // (TD-ENGINE-MVCC-1; see `read_sstables_resolved`).
        let pairs =
            proximadb_records::keep_newest_by(pairs, |p| p.0.oid.as_str(), |p| p.0.version_key());
        let (records, all_keys): (Vec<ProximaRecord>, Vec<u64>) = pairs.into_iter().unzip();
        let (all_records, _dead_ids) = materialize_for_write(records);

        info!(
            "Merged {} records from pre-sorted L0 files",
            all_records.len()
        );

        // Write merged L1 files
        let target_file_size = 100; // vectors per file
        let mut bytes_written = 0u64;
        let mut new_l1_files = Vec::new();

        for (chunk_idx, chunk) in all_records.chunks(target_file_size).enumerate() {
            let filename = self.filename_codec.generate(1, "helix");
            let file_path = self.data_dir.join(&filename);

            let chunk_start = chunk_idx * target_file_size;
            let chunk_end = std::cmp::min(chunk_start + chunk.len(), all_keys.len());
            let chunk_keys = &all_keys[chunk_start..chunk_end];

            let bytes = write_helix_sstable(
                &self.filesystem,
                &file_path,
                chunk,
                self.config.proxima_block_size,
                crate::storage::engines::constants::HELIX_MAGIC,
                Some(chunk_keys),
                Some(256), // Default Hilbert curve size
            )
            .await?;

            bytes_written += bytes;

            // Generate bloom filter for chunk IDs
            let bloom_filter = if !chunk.is_empty() {
                use crate::core::bloom::{BloomFilterConfig, factory::BloomFilterFactory};

                let bloom_config = BloomFilterConfig::for_sstable(chunk.len());
                let mut bloom = BloomFilterFactory::create(&bloom_config);

                for record in chunk {
                    bloom.insert(record.id.as_bytes());
                }

                bloom.serialize().ok()
            } else {
                None
            };

            let metadata = SStableMetadata {
                path: file_path,
                level: 1,
                hilbert_range: Some((
                    *chunk_keys.iter().min().unwrap_or(&0),
                    *chunk_keys.iter().max().unwrap_or(&0),
                )),
                num_vectors: chunk.len(),
                size_bytes: bytes,
                created_at: chrono::Utc::now(),
                blocks: extract_helix_metadata(
                    chunk,
                    self.config.proxima_block_size,
                    Some(chunk_keys),
                )
                .into_iter()
                .map(|h| h.proxima_metadata)
                .collect(),
                bloom_filter,
            };

            new_l1_files.push(metadata);
        }

        // Update levels
        let num_l1_files = new_l1_files.len();
        let mut levels_write = levels.write().await;

        debug!(
            "HELIX Optimized Merge: Removing {} L0 files from metadata",
            l0_files.len()
        );
        for sstable in &l0_files {
            debug!("  - L0 file marked for removal: {}", sstable.path.display());
        }

        levels_write.remove(&0);

        debug!(
            "HELIX Optimized Merge: Adding {} new L1 files to metadata",
            num_l1_files
        );
        for sstable in &new_l1_files {
            debug!(
                "  + L1 file created: {} (Hilbert range: {:?})",
                sstable.path.display(),
                sstable.hilbert_range
            );
        }

        levels_write.entry(1).or_default().extend(new_l1_files);

        drop(levels_write); // Release lock before filesystem operations

        // Delete L0 files from disk
        debug!(
            "HELIX Optimized Merge: Deleting {} L0 files from disk",
            l0_files.len()
        );
        for sstable in &l0_files {
            let file_path = sstable.path.to_string_lossy();
            debug!("  - Deleting L0 file: {}", file_path);
            self.filesystem.delete(&file_path).await.map_err(|e| {
                warn!("Failed to delete L0 file {}: {}", file_path, e);
                e
            })?;
            info!("  ✓ Deleted L0 file: {}", file_path);
        }

        info!(
            "✓ HELIX Optimized Merge complete: {} L0 files → {} L1 files, {} bytes written, {} L0 files deleted",
            l0_files.len(),
            num_l1_files,
            bytes_written,
            l0_files.len()
        );

        Ok((l0_files.len(), bytes_written))
    }

    /// Compact level i to level i+1 with liquid clustering
    pub async fn compact_level_to_next(
        &self,
        levels: Arc<RwLock<HashMap<usize, Vec<SStableMetadata>>>>,
        level: usize,
        pca_model: Arc<RwLock<Option<PCAModel>>>,
    ) -> Result<(usize, u64)> {
        info!(
            "Starting L{} to L{} compaction with liquid clustering",
            level,
            level + 1
        );

        // Get files from current and next level
        let (curr_files, next_files) = {
            let levels_read = levels.read().await;
            (
                levels_read.get(&level).cloned().unwrap_or_default(),
                levels_read.get(&(level + 1)).cloned().unwrap_or_default(),
            )
        };

        if curr_files.is_empty() {
            return Ok((0, 0));
        }

        // Select files to compact based on overlapping Hilbert ranges
        let files_to_compact = self.select_files_for_compaction(&curr_files, &next_files)?;

        // Read all vectors
        // Newest copy per id across the whole input (TD-ENGINE-MVCC-1).
        let input_paths: Vec<&PathBuf> = files_to_compact.iter().map(|s| &s.path).collect();
        let (mut all_records, _dead_ids) = self.read_sstables_resolved(&input_paths).await?;

        // Apply liquid clustering if enabled (after computing Hilbert keys)
        // This will be done after Hilbert key computation below

        // Re-cluster with PCA if model is available using configured bits
        let mut hilbert_keys: Vec<HilbertKey> = if let Some(ref model) = *pca_model.read().await {
            all_records
                .iter()
                .map(|record| {
                    model
                        .project_and_compute_hilbert_with_config(
                            &record.vector,
                            self.config.hilbert_bits_per_dimension,
                        )
                        .unwrap_or(0)
                })
                .collect()
        } else {
            vec![0; all_records.len()]
        };

        // Apply liquid clustering if enabled (now that we have Hilbert keys)
        if self.liquid_config.enabled {
            let (reorganized_records, reorganized_keys) = self
                .apply_liquid_clustering(all_records, hilbert_keys)
                .await?;
            all_records = reorganized_records;
            hilbert_keys = reorganized_keys;
        }

        // Sort by Hilbert key
        sort_by_hilbert(&mut all_records, &hilbert_keys)?;

        // Write new files at next level
        let mut bytes_written = 0u64;
        let mut new_files = Vec::new();
        let target_file_size = 200 * (level + 1); // Larger files at higher levels

        for (chunk_idx, chunk) in all_records.chunks(target_file_size).enumerate() {
            let filename = self.filename_codec.generate((level + 1) as u32, "helix");
            let file_path = self.data_dir.join(&filename);

            // Get Hilbert keys for this chunk
            let chunk_start = chunk_idx * target_file_size;
            let chunk_end = std::cmp::min(chunk_start + chunk.len(), hilbert_keys.len());
            let chunk_keys = &hilbert_keys[chunk_start..chunk_end];

            // Write file
            let bytes = write_helix_sstable(
                &self.filesystem,
                &file_path,
                chunk,
                self.config.proxima_block_size,
                crate::storage::engines::constants::HELIX_MAGIC,
                Some(chunk_keys),
                Some(256), // Default Hilbert curve size
            )
            .await?;

            bytes_written += bytes;

            // Generate bloom filter for chunk IDs
            let bloom_filter = if !chunk.is_empty() {
                use crate::core::bloom::{BloomFilterConfig, factory::BloomFilterFactory};

                let bloom_config = BloomFilterConfig::for_sstable(chunk.len());
                let mut bloom = BloomFilterFactory::create(&bloom_config);

                for record in chunk {
                    bloom.insert(record.id.as_bytes());
                }

                bloom.serialize().ok()
            } else {
                None
            };

            // Create metadata
            let metadata = SStableMetadata {
                path: file_path,
                level: level + 1,
                hilbert_range: if !chunk_keys.is_empty() {
                    // Safe min/max - we already checked chunk_keys is not empty
                    match (chunk_keys.iter().min(), chunk_keys.iter().max()) {
                        (Some(&min), Some(&max)) => Some((min, max)),
                        _ => None, // Should not happen but handle gracefully
                    }
                } else {
                    None
                },
                num_vectors: chunk.len(),
                size_bytes: bytes,
                created_at: chrono::Utc::now(),
                blocks: extract_helix_metadata(
                    chunk,
                    self.config.proxima_block_size,
                    Some(chunk_keys),
                )
                .into_iter()
                .map(|h| h.proxima_metadata)
                .collect(),
                bloom_filter,
            };

            new_files.push(metadata);
        }

        // Update levels
        {
            let mut levels_write = levels.write().await;

            // Remove compacted files from current level
            if let Some(curr_level_files) = levels_write.get_mut(&level) {
                curr_level_files.retain(|f| !files_to_compact.iter().any(|cf| cf.path == f.path));
            }

            // Remove compacted files from next level
            if let Some(next_level_files) = levels_write.get_mut(&(level + 1)) {
                next_level_files.retain(|f| !files_to_compact.iter().any(|cf| cf.path == f.path));
            }

            // Add new files to next level
            levels_write.entry(level + 1).or_default().extend(new_files);
        }

        // Delete old files
        for sstable in &files_to_compact {
            self.filesystem
                .delete(sstable.path.to_str().unwrap_or(""))
                .await?;
        }

        info!(
            "Compacted {} files from L{}/L{} into L{}, wrote {} bytes",
            files_to_compact.len(),
            level,
            level + 1,
            level + 1,
            bytes_written
        );

        Ok((files_to_compact.len(), bytes_written))
    }

    /// Select files for compaction based on overlapping ranges
    fn select_files_for_compaction(
        &self,
        curr_files: &[SStableMetadata],
        next_files: &[SStableMetadata],
    ) -> Result<Vec<SStableMetadata>> {
        let mut files_to_compact = Vec::new();

        // Add all current level files
        files_to_compact.extend_from_slice(curr_files);

        // Add overlapping files from next level
        for curr_file in curr_files {
            if let Some((curr_min, curr_max)) = curr_file.hilbert_range {
                for next_file in next_files {
                    if let Some((next_min, next_max)) = next_file.hilbert_range {
                        // Check for overlap
                        if !(curr_max < next_min
                            || curr_min > next_max
                            || files_to_compact.iter().any(|f| f.path == next_file.path))
                        {
                            files_to_compact.push(next_file.clone());
                        }
                    }
                }
            }
        }

        Ok(files_to_compact)
    }

    /// Apply liquid clustering based on query patterns
    async fn apply_liquid_clustering(
        &self,
        records: Vec<VectorRecord>,
        hilbert_keys: Vec<HilbertKey>,
    ) -> Result<(Vec<VectorRecord>, Vec<HilbertKey>)> {
        // Create liquid clustering coordinator
        let coordinator = LiquidClusteringCoordinator::new(
            self.liquid_config.clone(),
            self.query_tracker.clone(),
        );

        // Apply liquid clustering to reorganize based on access patterns
        let (reorganized_records, reorganized_keys) = coordinator
            .apply_liquid_clustering(records, &hilbert_keys)
            .await?;

        // Check if we should trigger re-clustering
        if coordinator.should_recluster().await {
            info!("Liquid clustering: Re-clustering triggered based on access patterns");
        }

        // Calculate and log clustering quality
        let quality = coordinator
            .calculate_clustering_quality(&reorganized_records, &reorganized_keys)
            .await;
        info!("Liquid clustering quality: {:.2}", quality);

        // Get optimization suggestions
        let suggestions = coordinator.get_optimization_suggestions().await;
        for suggestion in suggestions {
            debug!(
                "Optimization suggestion: {:?} - {} (estimated improvement: {:.1}%)",
                suggestion.suggestion_type,
                suggestion.reason,
                suggestion.estimated_improvement * 100.0
            );
        }

        Ok((reorganized_records, reorganized_keys))
    }

    /// Read every input SSTable and keep only each id's NEWEST copy across the
    /// whole compaction input (the shared order, `proximadb_records::VersionKey`;
    /// TD-ENGINE-MVCC-1), preserving input order.
    ///
    /// Dead winners (tombstones and TTL-expired copies) are KEPT. A compaction
    /// input is a subset of the collection's SSTables, so a dropped dead winner
    /// would stop shadowing an older live copy in a level outside the input,
    /// and search would resurface it. This used to drop every dead row before
    /// any resolution, which did exactly that. Purging dead winners needs a
    /// full-coverage watermark (TD-DELVEC-2).
    async fn read_sstables_resolved(
        &self,
        paths: &[&PathBuf],
    ) -> Result<(Vec<VectorRecord>, std::collections::HashSet<String>)> {
        let mut all = Vec::new();
        for path in paths {
            all.extend(self.read_sstable(path).await?);
        }
        Ok(materialize_for_write(proximadb_records::keep_newest_by(
            all,
            |r| r.oid.as_str(),
            ProximaRecord::version_key,
        )))
    }

    /// Every row of one SSTable, dead or alive, unresolved.
    async fn read_sstable(&self, path: &std::path::Path) -> Result<Vec<ProximaRecord>> {
        // Read file data
        let file_data = self.filesystem.read(path.to_str().unwrap_or("")).await?;
        let mut cursor = std::io::Cursor::new(file_data);

        // Skip magic and version
        cursor.set_position(8);

        // Read number of blocks
        let mut num_blocks_bytes = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut num_blocks_bytes)?;
        let num_blocks = u32::from_le_bytes(num_blocks_bytes);

        let mut records = Vec::new();

        // Read blocks
        for _ in 0..num_blocks {
            // Read block size
            let mut size_bytes = [0u8; 4];
            std::io::Read::read_exact(&mut cursor, &mut size_bytes)?;
            let block_size = u32::from_le_bytes(size_bytes) as usize;

            // Read block data
            let mut block_data = vec![0u8; block_size];
            std::io::Read::read_exact(&mut cursor, &mut block_data)?;

            // Deserialize block
            use crate::storage::engines::core::formats::proximablocks::ProximaDataBlock;
            let block = ProximaDataBlock::deserialize(&block_data, None)?;

            records.extend(block.records);
        }

        Ok(records)
    }

    /// Trigger re-clustering at max level
    pub async fn recluster_max_level(
        &self,
        levels: Arc<RwLock<HashMap<usize, Vec<SStableMetadata>>>>,
        pca_model: Arc<RwLock<Option<PCAModel>>>,
    ) -> Result<()> {
        let max_level = self.config.max_levels - 1;

        info!("Starting re-clustering at L{}", max_level);

        // Get files at max level
        let max_level_files = {
            let levels_read = levels.read().await;
            levels_read.get(&max_level).cloned().unwrap_or_default()
        };

        if max_level_files.is_empty() {
            return Ok(());
        }

        // Read all vectors
        // Newest copy per id across the whole input (TD-ENGINE-MVCC-1).
        let input_paths: Vec<&PathBuf> = max_level_files.iter().map(|s| &s.path).collect();
        let (all_records, dead_ids) = self.read_sstables_resolved(&input_paths).await?;

        // Update PCA model with current data
        let trainable = pca_training_records(&all_records, &dead_ids);
        if !trainable.is_empty() {
            let new_model = PCAModel::train(&trainable, self.config.pca_dimensions)?;
            *pca_model.write().await = Some(new_model);
        }

        // Re-compact with new model
        self.compact_level_to_next(levels, max_level - 1, pca_model)
            .await?;

        Ok(())
    }
}

/// Convert resolved compaction survivors for the HELIX writer.
///
/// Compaction keeps dead winners (TD-ENGINE-MVCC-1), and a tombstone carries no
/// vector. The HELIX block encoder requires one dimension per block, so an
/// empty vector next to live rows fails the WHOLE compaction. A dead winner
/// without a vector therefore gets an all-zero vector of the input's
/// dimension. It is never scored: every HELIX read path skips dead records
/// before scoring, and get-by-id returns `None` for them. It only shadows
/// older copies, which is what it is kept for.
fn materialize_for_write(
    records: Vec<ProximaRecord>,
) -> (Vec<VectorRecord>, std::collections::HashSet<String>) {
    // Deadness on the canonical `valid_to_ns` (16a), before conversion.
    let now_ns = proximadb_kernel::time::now_unix_ns();
    let dead_ids: std::collections::HashSet<String> = records
        .iter()
        .filter(|r| proximadb_records::is_record_dead(r.valid_to_ns, now_ns))
        .map(|r| r.oid.clone())
        .collect();
    let mut out: Vec<VectorRecord> = records.iter().map(VectorRecord::from).collect();
    let dimension = out
        .iter()
        .map(|r| r.vector.len())
        .find(|&d| d > 0)
        .unwrap_or(0);
    if dimension > 0 {
        for r in out.iter_mut().filter(|r| r.vector.is_empty()) {
            r.vector = vec![0.0; dimension];
        }
    }
    (out, dead_ids)
}

/// The records PCA may train on: live ones. Dead winners kept by compaction
/// (TD-ENGINE-MVCC-1) carry an all-zero vector or a vector nobody will
/// read again; they must not shape the model. Borrows when nothing is dead.
fn pca_training_records<'a>(
    records: &'a [VectorRecord],
    dead_ids: &std::collections::HashSet<String>,
) -> std::borrow::Cow<'a, [VectorRecord]> {
    if dead_ids.is_empty() {
        std::borrow::Cow::Borrowed(records)
    } else {
        std::borrow::Cow::Owned(
            records
                .iter()
                .filter(|r| !dead_ids.contains(&r.id))
                .cloned()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_compactor_creation() {
        let config = HelixConfig::default();

        // Create filesystem factory with proper config
        let mut fs_config = crate::storage::persistence::filesystem::FilesystemConfig::default();
        fs_config.default_fs = Some("file:///tmp/helix_test".to_string());
        let factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(fs_config)
                .await
                .unwrap(),
        );
        let base_filesystem = factory.get_filesystem("file:///tmp/helix_test").unwrap();

        // Wrap in UnifiedCachingFilesystem like production code
        let filesystem = Arc::new(UnifiedCachingFilesystem::new(
            base_filesystem,
            "test_collection".to_string(),
            "helix".to_string(),
        ));

        let data_dir = PathBuf::from("/tmp/helix_test");

        let compactor = LeveledCompactor::new(config, filesystem, data_dir);
        assert!(compactor.liquid_config.enabled);
    }

    // ---- TD-ENGINE-MVCC-1 / #2521 review H1 ----
    // Compaction must resolve each id's newest copy across its whole input and
    // KEEP a dead winner. Dropping dead rows first rewrote the older live copy
    // with nothing left to shadow it, so search resurfaced it.

    const DIM: usize = 8;

    fn vrec(id: &str, ms: i64, version: u32, expires_at: Option<i64>, live: bool) -> VectorRecord {
        let mut vector = vec![0.0_f32; DIM];
        vector[0] = 1.0;
        VectorRecord {
            id: id.to_string(),
            // A tombstone carries no vector (as flush writes it).
            vector: if live { vector } else { Vec::new() },
            metadata: HashMap::new(),
            timestamp: Some(ms),
            updated_at: Some(ms),
            expires_at,
            version: Some(version),
            source: None,
        }
    }

    fn pad(name: &str, i: usize) -> VectorRecord {
        let mut v = vrec(&format!("{name}_{i}"), 500, 1, None, true);
        v.vector[DIM - 1] = 1.0 + i as f32;
        v
    }

    async fn compactor_in(
        dir: &std::path::Path,
    ) -> (LeveledCompactor, Arc<UnifiedCachingFilesystem>) {
        let url = format!("file://{}", dir.display());
        let mut fs_config = crate::storage::persistence::filesystem::FilesystemConfig::default();
        fs_config.default_fs = Some(url.clone());
        let factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(fs_config)
                .await
                .unwrap_or_else(|e| panic!("fs factory: {e}")),
        );
        let base = factory
            .get_filesystem(&url)
            .unwrap_or_else(|e| panic!("fs: {e}"));
        let filesystem = Arc::new(UnifiedCachingFilesystem::new(
            base,
            "mvcc_compaction_test".to_string(),
            "helix".to_string(),
        ));
        let compactor = LeveledCompactor::new(
            HelixConfig::default(),
            filesystem.clone(),
            dir.to_path_buf(),
        );
        (compactor, filesystem)
    }

    async fn l0_file(
        filesystem: &Arc<UnifiedCachingFilesystem>,
        dir: &std::path::Path,
        name: &str,
        records: &[VectorRecord],
        sorted: bool,
    ) -> SStableMetadata {
        let path = dir.join(name);
        let keys: Vec<u64> = (0..records.len() as u64).collect();
        let size_bytes = write_helix_sstable(
            filesystem,
            &path,
            records,
            HelixConfig::default().proxima_block_size,
            crate::storage::engines::constants::HELIX_MAGIC,
            if sorted { Some(&keys) } else { None },
            Some(256),
        )
        .await
        .unwrap_or_else(|e| panic!("write {name}: {e}"));
        SStableMetadata {
            path,
            level: 0,
            hilbert_range: sorted.then(|| (0, records.len().saturating_sub(1) as u64)),
            num_vectors: records.len(),
            size_bytes,
            created_at: chrono::Utc::now(),
            blocks: Vec::new(),
            bloom_filter: None,
        }
    }

    /// Run one L0->L1 compaction over [v1 live] + [newest dead copy] + 2 pad
    /// files; return every copy of `victim` found in the L1 output.
    async fn compact_and_find_victim(
        sorted: bool,
        dead_newest: VectorRecord,
    ) -> Vec<proximadb_records::ProximaRecord> {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let dir = tmp.path().to_path_buf();
        let (compactor, fs) = compactor_in(&dir).await;
        let v1 = vrec("victim", 1_000, 1, None, true);
        let mut f0 = vec![v1];
        f0.extend((0..3).map(|i| pad("a", i)));
        // A tombstone carries no vector, so (as a flush writes it) it sits in a
        // file of its own; an expired copy still has its vector and shares one.
        let tombstone = dead_newest.vector.is_empty();
        let mut f1 = vec![dead_newest];
        if !tombstone {
            f1.extend((0..3).map(|i| pad("b", i)));
        }
        let files = vec![
            l0_file(&fs, &dir, "l0_0.helix", &f0, sorted).await,
            l0_file(&fs, &dir, "l0_1.helix", &f1, sorted).await,
            l0_file(
                &fs,
                &dir,
                "l0_2.helix",
                &(0..4).map(|i| pad("c", i)).collect::<Vec<_>>(),
                sorted,
            )
            .await,
            l0_file(
                &fs,
                &dir,
                "l0_3.helix",
                &(0..4).map(|i| pad("d", i)).collect::<Vec<_>>(),
                sorted,
            )
            .await,
        ];
        let levels = Arc::new(RwLock::new(HashMap::from([(0_usize, files)])));
        let (compacted, _) = compactor
            .compact_l0_to_l1(levels.clone())
            .await
            .unwrap_or_else(|e| panic!("compact: {e}"));
        assert!(
            compacted > 0,
            "positive control: compaction must run (sorted={sorted})"
        );
        let l1 = levels.read().await.get(&1).cloned().unwrap_or_default();
        assert!(
            !l1.is_empty(),
            "positive control: L1 output written (sorted={sorted})"
        );
        let mut copies = Vec::new();
        for file in &l1 {
            for r in compactor
                .read_sstable(&file.path)
                .await
                .unwrap_or_else(|e| panic!("read L1: {e}"))
            {
                if r.oid == "victim" {
                    copies.push(r);
                }
            }
        }
        std::mem::forget(tmp);
        copies
    }

    async fn assert_only_dead_winner(sorted: bool, dead_newest: VectorRecord, what: &str) {
        let copies = compact_and_find_victim(sorted, dead_newest).await;
        let now = proximadb_kernel::time::now_unix_ns();
        assert_eq!(
            copies.len(),
            1,
            "[{what}, sorted={sorted}] exactly the newest copy survives; got {:?}",
            copies
                .iter()
                .map(|c| (c.record_version, c.valid_to_ns))
                .collect::<Vec<_>>()
        );
        assert!(
            proximadb_records::is_record_dead(copies[0].valid_to_ns, now),
            "[{what}, sorted={sorted}] the survivor must be the DEAD newest copy, not the \
             older live v1 (which search would resurface): {:?}",
            (copies[0].record_version, copies[0].valid_to_ns)
        );
    }

    #[tokio::test]
    async fn compaction_keeps_expired_winner_and_drops_older_live_copy() {
        for sorted in [true, false] {
            // v2: newer, TTL already expired (1 ms after the epoch).
            assert_only_dead_winner(sorted, vrec("victim", 2_000, 2, Some(1), true), "expired")
                .await;
        }
    }

    #[tokio::test]
    async fn compaction_keeps_tombstone_winner_and_drops_older_live_copy() {
        for sorted in [true, false] {
            assert_only_dead_winner(
                sorted,
                vrec("victim", 2_000, 0, Some(0), false),
                "tombstone",
            )
            .await;
        }
    }
}
