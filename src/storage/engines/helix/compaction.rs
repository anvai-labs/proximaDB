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
            owned(levels_read.get(&0), &self.data_dir)
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

        // Fallback path: read all records and re-cluster. Newest copy per
        // (collection, id) across the L0 input, dead winners KEPT: older copies
        // may live in L1, outside this input (TD-ENGINE-MVCC-1). Physical GC of
        // dead winners needs collection-scoped compaction plus a compaction
        // mutex (TD-HELIX-COMPACT-SCOPE-1); it is deliberately not done here.
        let l0_paths: Vec<&PathBuf> = l0_files.iter().map(|s| &s.path).collect();
        let (mut all_records, dead_ids) = self.read_sstables_resolved(&l0_paths).await?;

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

            // Remove exactly the consumed files. `remove(&0)` also dropped L0
            // files flushed after this compaction's snapshot (e.g. a tombstone
            // file), which search then never saw again.
            retain_unconsumed(&mut levels_write, 0, &l0_files);

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
        let mut pairs: Vec<(String, ProximaRecord, u64)> = Vec::new();

        // For simplicity, read all records but preserve sorting
        // In production, implement proper k-way merge with iterators
        for sstable in &l0_files {
            let records = self.read_sstable(&sstable.path).await?;
            let scope = collection_scope(&sstable.path);

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
                        let id = scoped_id(&scope, &record.oid);
                        pairs.push((id, record, min_key + (i as u64 * step)));
                    }
                }
            } else {
                // Shouldn't happen with sorted flush
                pairs.extend(
                    records
                        .into_iter()
                        .map(|r| (scoped_id(&scope, &r.oid), r, 0)),
                );
            }
        }

        // Newest copy per id across the whole input, dead winners kept
        // (TD-ENGINE-MVCC-1; see `read_sstables_resolved`).
        let shadowed = shadowed_vectors(pairs.iter().map(|p| (p.0.as_str(), &p.1)));
        let pairs =
            proximadb_records::keep_newest_by(pairs, |p| p.0.as_str(), |p| p.1.version_key());
        let mut all_keys = Vec::with_capacity(pairs.len());
        let mut scoped = Vec::with_capacity(pairs.len());
        for (id, record, key) in pairs {
            all_keys.push(key);
            scoped.push((id, record));
        }
        let (all_records, _dead_ids) = materialize_for_write(scoped, &shadowed);

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

        // Remove exactly the consumed files: `remove(&0)` also dropped L0 files
        // flushed after this compaction's snapshot (#2521 review round 4).
        retain_unconsumed(&mut levels_write, 0, &l0_files);

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
                owned(levels_read.get(&level), &self.data_dir),
                owned(levels_read.get(&(level + 1)), &self.data_dir),
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
    ) -> Result<(Vec<VectorRecord>, Vec<bool>)> {
        let mut all = Vec::new();
        for path in paths {
            let scope = collection_scope(path);
            all.extend(
                self.read_sstable(path)
                    .await?
                    .into_iter()
                    .map(|r| (scoped_id(&scope, &r.oid), r)),
            );
        }
        let shadowed = shadowed_vectors(all.iter().map(|p| (p.0.as_str(), &p.1)));
        Ok(materialize_for_write(
            proximadb_records::keep_newest_by(all, |p| p.0.as_str(), |p| p.1.version_key()),
            &shadowed,
        ))
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
            owned(levels_read.get(&max_level), &self.data_dir)
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
    scoped: Vec<(String, ProximaRecord)>,
    shadowed: &HashMap<String, Vec<f32>>,
) -> (Vec<VectorRecord>, Vec<bool>) {
    let scoped_ids: Vec<String> = scoped.iter().map(|(id, _)| id.clone()).collect();
    let records: Vec<ProximaRecord> = scoped.into_iter().map(|(_, r)| r).collect();
    // Deadness on the canonical `valid_to_ns` (16a), before conversion.
    let now_ns = proximadb_kernel::time::now_unix_ns();
    // Positional (one flag per output row), never keyed on the bare id: one
    // compaction input can hold several collections' records with the same id
    // (review round 5, LOW-3).
    let dead: Vec<bool> = records
        .iter()
        .map(|r| proximadb_records::is_record_dead(r.valid_to_ns, now_ns))
        .collect();
    let mut out: Vec<VectorRecord> = records.iter().map(VectorRecord::from).collect();
    let dimension = out
        .iter()
        .map(|r| r.vector.len())
        .find(|&d| d > 0)
        .unwrap_or(0);
    for (r, scoped_id) in out
        .iter_mut()
        .zip(scoped_ids.iter())
        .filter(|(r, _)| r.vector.is_empty())
    {
        // Prefer the vector of the copy this dead winner shadows (review round
        // 2, M-d): Hilbert keys, block centroids and SSTable ranges are computed
        // from it, so the tombstone lands where the shadowed copy lives and
        // pruning cannot read one without the other. Zero only when no copy of
        // the id is in this input.
        if let Some(v) = shadowed.get(scoped_id) {
            r.vector = v.clone();
        } else if dimension > 0 {
            r.vector = vec![0.0; dimension];
        }
    }
    (out, dead)
}

/// For each (collection, id) whose NEWEST copy in a compaction input is dead
/// and carries no vector, the newest LIVE vector of that id: what the dead
/// winner borrows so it stays co-located with the copy it shadows. Copies only
/// the vectors actually needed (review round 5, LOW-4).
fn shadowed_vectors<'a, I>(records: I) -> HashMap<String, Vec<f32>>
where
    I: IntoIterator<Item = (&'a str, &'a ProximaRecord)>,
    I::IntoIter: Clone,
{
    let records = records.into_iter();
    let now_ns = proximadb_kernel::time::now_unix_ns();
    let mut newest = proximadb_records::NewestVersionIndex::new();
    for (id, r) in records.clone() {
        newest.observe(id, r.version_key());
    }
    let needed: std::collections::HashSet<&str> = records
        .clone()
        .filter(|(id, r)| {
            r.embeddings.is_empty()
                && proximadb_records::is_record_dead(r.valid_to_ns, now_ns)
                && newest.is_newest(id, &r.version_key())
        })
        .map(|(id, _)| id)
        .collect();
    let mut best: HashMap<String, (proximadb_records::VersionKey, Vec<f32>)> = HashMap::new();
    for (id, r) in records {
        if !needed.contains(id) || proximadb_records::is_record_dead(r.valid_to_ns, now_ns) {
            continue;
        }
        let Some(cell) = r.embeddings.first() else {
            continue;
        };
        let key = r.version_key();
        if best.get(id).is_none_or(|(k, _)| key.supersedes(k)) {
            best.insert(id.to_string(), (key, cell.values.to_fp32_owned()));
        }
    }
    best.into_iter().map(|(id, (_, v))| (id, v)).collect()
}

/// True when `path` is an SSTable of the collection whose data directory is
/// `data_dir` (its direct parent; component-wise, so `coll_1` never matches
/// `coll_10`).
///
/// One HELIX engine and one level map serve every collection of every tenant
/// (the flush coordinator caches one engine per engine type). A compaction
/// must therefore select ONLY its own collection's files. Selecting the whole
/// map merged every collection's L0 into the triggering collection's
/// directory: cross-tenant data movement, a path-isolation violation
/// (TD-HELIX-COMPACT-SCOPE-1).
pub(super) fn owns_sstable(data_dir: &std::path::Path, path: &std::path::Path) -> bool {
    path.parent()
        .is_some_and(|parent| canonical_location(parent) == canonical_location(data_dir))
}

/// One canonical form for a storage location, so the SAME directory compares
/// equal however it was spelled (#2530 review HIGH-1). Flushed SSTable paths are
/// built from the raw `data_dir` string (e.g. the default `file://./data`), but
/// SSTables rediscovered from a directory listing come back as absolute
/// `file:///abs/...` URLs. Comparing raw forms made every rediscovered file
/// "not owned", and that collection never compacted again.
///
/// `file://` is stripped and relative paths are absolutized (as the local
/// filesystem's listing does). The Azure scheme aliases (`adls`/`abfs`/`azure`)
/// fold to `az`, the one flat Blob backend they all resolve to (ADR-036). `.`
/// components and trailing separators are dropped.
pub(super) fn canonical_location(location: &std::path::Path) -> std::path::PathBuf {
    let raw = location.to_string_lossy();
    let mut s: String = raw.into_owned();
    for alias in ["adls://", "abfs://", "azure://"] {
        if let Some(rest) = s.strip_prefix(alias) {
            s = format!("az://{rest}");
            break;
        }
    }
    let local = match s.strip_prefix("file://") {
        Some(rest) => Some(rest.to_string()),
        None if !s.contains("://") => Some(s.clone()),
        None => None,
    };
    let path = match local {
        Some(p) => {
            let p = std::path::PathBuf::from(p);
            if p.is_absolute() {
                p
            } else {
                std::path::absolute(&p).unwrap_or(p)
            }
        }
        None => std::path::PathBuf::from(s),
    };
    path.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect()
}

/// The L0 compaction trigger for ONE collection: its own L0 files only. The
/// level map is shared by every collection (TD-HELIX-COMPACT-SCOPE-1).
pub(super) fn l0_trigger_reached(
    levels: &HashMap<usize, Vec<SStableMetadata>>,
    data_dir: &std::path::Path,
    trigger: usize,
) -> bool {
    levels.get(&0).is_some_and(|l0| {
        l0.iter()
            .filter(|f| owns_sstable(data_dir, &f.path))
            .count()
            >= trigger
    })
}

/// Acquire a collection's compaction guard. `wait = false` (flush-triggered):
/// skip when a compaction is already running. `wait = true` (explicit
/// `do_compact`): wait for it. Returns `None` only when skipping.
pub(super) async fn acquire_compaction(
    guard: Arc<tokio::sync::Mutex<()>>,
    wait: bool,
) -> Option<tokio::sync::OwnedMutexGuard<()>> {
    if wait {
        Some(guard.lock_owned().await)
    } else {
        guard.try_lock_owned().ok()
    }
}

/// The files of one level that belong to the collection at `data_dir`.
fn owned(files: Option<&Vec<SStableMetadata>>, data_dir: &std::path::Path) -> Vec<SStableMetadata> {
    files
        .map(|v| {
            v.iter()
                .filter(|f| owns_sstable(data_dir, &f.path))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Drop exactly the `consumed` files from `level` of the level map, keeping
/// any file registered after the compaction's snapshot (e.g. an L0 file flushed
/// mid-compaction, possibly a tombstone file). `remove(&level)` dropped those
/// too, hiding them from search (#2521 review rounds 3-4).
fn retain_unconsumed(
    levels: &mut HashMap<usize, Vec<SStableMetadata>>,
    level: usize,
    consumed: &[SStableMetadata],
) {
    let consumed: std::collections::HashSet<&PathBuf> = consumed.iter().map(|s| &s.path).collect();
    if let Some(files) = levels.get_mut(&level) {
        files.retain(|f| !consumed.contains(&f.path));
    }
}

/// The collection an SSTable belongs to: its directory. One HELIX engine (and
/// one level map) serves every collection of every tenant
/// (TD-HELIX-COMPACT-SCOPE-1), so a compaction input can mix collections.
/// Version resolution must never let one collection's copy supersede another
/// collection's record with the same id (#2521 review round 4).
fn collection_scope(path: &std::path::Path) -> String {
    path.parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Resolution key: (collection directory, id).
fn scoped_id(scope: &str, oid: &str) -> String {
    format!("{scope}\u{0}{oid}")
}

/// The records PCA may train on: live ones. Dead winners kept by compaction
/// (TD-ENGINE-MVCC-1) carry an all-zero vector or a vector nobody will
/// read again; they must not shape the model. Borrows when nothing is dead.
fn pca_training_records<'a>(
    records: &'a [VectorRecord],
    dead: &[bool],
) -> std::borrow::Cow<'a, [VectorRecord]> {
    if !dead.iter().any(|&d| d) {
        std::borrow::Cow::Borrowed(records)
    } else {
        std::borrow::Cow::Owned(
            records
                .iter()
                .zip(dead.iter())
                .filter(|(_, d)| !**d)
                .map(|(r, _)| r.clone())
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

    struct Outcome {
        /// (collection dir, record) for every L1 output row.
        rows: Vec<(String, proximadb_records::ProximaRecord)>,
    }

    impl Outcome {
        fn copies(&self, oid: &str) -> Vec<&proximadb_records::ProximaRecord> {
            self.rows
                .iter()
                .filter(|(_, r)| r.oid == oid)
                .map(|(_, r)| r)
                .collect()
        }
    }

    /// Compact `l0` (each entry: sub-directory, file name, records) with the
    /// compactor rooted at `dir/<root>`; return every L1 output row.
    async fn compact_l0(
        sorted: bool,
        root: &str,
        l0: Vec<(&str, &str, Vec<VectorRecord>)>,
    ) -> Outcome {
        let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let base = tmp.path().to_path_buf();
        let (compactor, fs) = compactor_in(&base.join(root)).await;
        let mut files = Vec::new();
        for (sub, name, records) in &l0 {
            let dir = base.join(sub);
            std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir: {e}"));
            files.push(l0_file(&fs, &dir, name, records, sorted).await);
        }
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
        let mut rows = Vec::new();
        for file in &l1 {
            let scope = super::collection_scope(&file.path);
            for r in compactor
                .read_sstable(&file.path)
                .await
                .unwrap_or_else(|e| panic!("read L1: {e}"))
            {
                rows.push((scope.clone(), r));
            }
        }
        std::mem::forget(tmp);
        Outcome { rows }
    }

    fn deads() -> [(&'static str, VectorRecord); 2] {
        [
            // v2: newer, TTL already expired (1 ms after the epoch).
            ("expired", vrec("victim", 2_000, 2, Some(1), true)),
            ("tombstone", vrec("victim", 2_000, 0, Some(0), false)),
        ]
    }

    fn pads(name: &str, n: usize) -> Vec<VectorRecord> {
        (0..n).map(|i| pad(name, i)).collect()
    }

    #[tokio::test]
    async fn l0_compaction_keeps_dead_winner_and_drops_older_live_copy() {
        // H1: dropping dead rows before resolving rewrote v1 with nothing left
        // to shadow it. The dead winner must survive (older copies may live in
        // L1, outside the input) and v1 must be gone.
        for sorted in [true, false] {
            for (what, dead) in deads() {
                let tombstone = dead.vector.is_empty();
                let mut f0 = vec![vrec("victim", 1_000, 1, None, true)];
                f0.extend(pads("a", 3));
                let mut f1 = vec![dead];
                if !tombstone {
                    f1.extend(pads("b", 3));
                }
                let out = compact_l0(
                    sorted,
                    "c1",
                    vec![
                        ("c1", "l0_0.helix", f0),
                        ("c1", "l0_1.helix", f1),
                        ("c1", "l0_2.helix", pads("c", 4)),
                        ("c1", "l0_3.helix", pads("d", 4)),
                    ],
                )
                .await;
                let copies = out.copies("victim");
                let now = proximadb_kernel::time::now_unix_ns();
                assert_eq!(
                    copies.len(),
                    1,
                    "[{what}, sorted={sorted}] exactly the newest copy survives"
                );
                assert!(
                    proximadb_records::is_record_dead(copies[0].valid_to_ns, now),
                    "[{what}, sorted={sorted}] the survivor must be the DEAD newest copy, not v1: {:?}",
                    (copies[0].record_version, copies[0].valid_to_ns)
                );
            }
        }
    }

    #[tokio::test]
    async fn resolution_never_lets_one_collection_supersede_another() {
        // Round 4 HIGH-1 (defence in depth): compaction no longer mixes
        // collections (TD-HELIX-COMPACT-SCOPE-1), but the resolver itself must
        // never let one collection's tombstone supersede another collection's
        // record with the same id, whatever input it is handed.
        for sorted in [true, false] {
            let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
            let base = tmp.path().to_path_buf();
            let (dir_a, dir_b) = (base.join("coll_a"), base.join("coll_b"));
            std::fs::create_dir_all(&dir_a).unwrap_or_else(|e| panic!("mkdir: {e}"));
            std::fs::create_dir_all(&dir_b).unwrap_or_else(|e| panic!("mkdir: {e}"));
            let (compactor, fs) = compactor_in(&dir_a).await;
            let a = l0_file(
                &fs,
                &dir_a,
                "l0_a.helix",
                &[vrec("shared", 2_000, 0, Some(0), false)],
                sorted,
            )
            .await;
            let mut b_records = vec![vrec("shared", 1_000, 1, None, true)];
            b_records.extend(pads("b", 3));
            let b = l0_file(&fs, &dir_b, "l0_b.helix", &b_records, sorted).await;
            let (out, _) = compactor
                .read_sstables_resolved(&[&a.path, &b.path])
                .await
                .unwrap_or_else(|e| panic!("resolve: {e}"));
            let now = proximadb_kernel::time::now_unix_ns();
            let shared: Vec<_> = out.iter().filter(|r| r.id == "shared").collect();
            assert_eq!(
                shared.len(),
                2,
                "[sorted={sorted}] both collections' 'shared' survive resolution"
            );
            assert!(
                shared
                    .iter()
                    .any(|r| r.expires_at.is_none() && !r.vector.iter().all(|v| *v == 0.0)),
                "[sorted={sorted}] collection B's LIVE 'shared' must survive A's tombstone"
            );
            let _ = now;
            std::mem::forget(tmp);
        }
    }

    #[test]
    fn retain_unconsumed_keeps_files_registered_after_the_snapshot() {
        // Round 3/4 HIGH-2: `remove(&0)` dropped L0 files flushed mid-compaction.
        let meta = |name: &str| SStableMetadata {
            path: PathBuf::from(name),
            level: 0,
            hilbert_range: None,
            num_vectors: 0,
            size_bytes: 0,
            created_at: chrono::Utc::now(),
            blocks: Vec::new(),
            bloom_filter: None,
        };
        let consumed = vec![meta("a"), meta("b")];
        let mut levels = HashMap::from([(0_usize, vec![meta("a"), meta("late"), meta("b")])]);
        super::retain_unconsumed(&mut levels, 0, &consumed);
        let left: Vec<_> = levels[&0].iter().map(|f| f.path.clone()).collect();
        assert_eq!(left, vec![PathBuf::from("late")]);
    }

    #[test]
    fn pca_trains_on_live_records_only() {
        let live = vrec("live", 1, 1, None, true);
        let dead = vrec("dead", 1, 1, Some(0), true);
        // A live record sharing an id with a dead one (another collection)
        // must still train: the mask is positional, not keyed on the id.
        let mut same_id = vrec("dead", 1, 1, None, true);
        same_id.vector[1] = 0.5;
        let records = vec![live, dead, same_id];
        let trainable = pca_training_records(&records, &[false, true, false]);
        assert_eq!(trainable.len(), 2);
        assert!(trainable.iter().all(|r| r.expires_at != Some(0)));
        assert!(matches!(
            pca_training_records(&records, &[false, false, false]),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn tombstone_winner_takes_the_shadowed_copys_vector_within_its_collection() {
        let scoped = |scope: &str, r: VectorRecord| {
            let r = proximadb_records::ProximaRecord::from(r);
            (super::scoped_id(scope, &r.oid), r)
        };
        let mut v1 = vrec("victim", 1_000, 1, None, true);
        v1.vector = vec![0.5; DIM];
        let mut other_coll = vrec("victim", 1_500, 1, None, true);
        other_coll.vector = vec![0.9; DIM];
        let input = vec![
            scoped("A", v1),
            scoped("A", vrec("victim", 2_000, 0, Some(0), false)),
            scoped("B", other_coll),
        ];
        let shadowed = super::shadowed_vectors(input.iter().map(|p| (p.0.as_str(), &p.1)));
        let resolved =
            proximadb_records::keep_newest_by(input, |p| p.0.as_str(), |p| p.1.version_key());
        let (out, _) = materialize_for_write(resolved, &shadowed);
        let tomb = out
            .iter()
            .find(|r| r.id == "victim" && r.expires_at == Some(0))
            .unwrap_or_else(|| panic!("tombstone kept"));
        assert_eq!(
            tomb.vector,
            vec![0.5; DIM],
            "borrows its OWN collection's shadowed vector"
        );
        assert_eq!(
            out.iter().filter(|r| r.id == "victim").count(),
            2,
            "B's live copy untouched"
        );
    }

    #[test]
    fn no_production_path_clears_a_whole_level() {
        // Round 5 LOW-1: the call sites of `retain_unconsumed` cannot be driven
        // deterministically mid-compaction, and `levels_write.remove(&0)` came
        // back once already (rounds 3-4, HIGH-2). Guard the source structurally:
        // no production code may drop a whole level from the map.
        let source = include_str!("compaction.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        assert!(
            !production.contains("levels_write.remove(&"),
            "a production path removes a whole level; use retain_unconsumed"
        );
        // The engine (`do_compact`) shares the same level map (round 6 LOW-1).
        let engine = include_str!("mod.rs");
        let engine = engine.split("#[cfg(test)]").next().unwrap_or(engine);
        assert!(
            !engine.contains("levels_write.remove(&"),
            "helix/mod.rs removes a whole level of the shared map"
        );
        assert!(
            production
                .matches("retain_unconsumed(&mut levels_write")
                .count()
                >= 2,
            "both L0 paths must update the level map through retain_unconsumed"
        );
    }

    #[test]
    fn owns_sstable_is_exact_per_collection_directory() {
        use std::path::Path;
        let dir = Path::new("/data/t1/ns/coll_1");
        assert!(super::owns_sstable(
            dir,
            Path::new("/data/t1/ns/coll_1/L0_a.helix")
        ));
        assert!(!super::owns_sstable(
            dir,
            Path::new("/data/t1/ns/coll_10/L0_a.helix")
        ));
        assert!(!super::owns_sstable(
            dir,
            Path::new("/data/t2/ns/coll_1/L0_a.helix")
        ));
        assert!(!super::owns_sstable(
            dir,
            Path::new("/data/t1/ns/coll_1/__model/x.helix")
        ));
    }

    #[tokio::test]
    async fn compaction_never_reads_moves_or_deletes_another_collections_files() {
        // TD-HELIX-COMPACT-SCOPE-1: one level map holds every collection's
        // SSTables. Compacting A must leave B's L0 files in the map and on
        // disk, and write only A's records, only under A's directory.
        for sorted in [true, false] {
            let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
            let base = tmp.path().to_path_buf();
            let dir_a = base.join("tenant_a").join("coll");
            let dir_b = base.join("tenant_b").join("coll");
            std::fs::create_dir_all(&dir_a).unwrap_or_else(|e| panic!("mkdir: {e}"));
            std::fs::create_dir_all(&dir_b).unwrap_or_else(|e| panic!("mkdir: {e}"));
            let (compactor, fs) = compactor_in(&dir_a).await;
            let mut l0 = Vec::new();
            for i in 0..4 {
                l0.push(
                    l0_file(
                        &fs,
                        &dir_a,
                        &format!("a_{i}.helix"),
                        &pads(&format!("a{i}"), 4),
                        sorted,
                    )
                    .await,
                );
            }
            let mut b_files = Vec::new();
            for i in 0..4 {
                let f = l0_file(
                    &fs,
                    &dir_b,
                    &format!("b_{i}.helix"),
                    &pads(&format!("b{i}"), 4),
                    sorted,
                )
                .await;
                b_files.push(f.path.clone());
                l0.push(f);
            }
            let levels = Arc::new(RwLock::new(HashMap::from([(0_usize, l0)])));
            let (compacted, _) = compactor
                .compact_l0_to_l1(levels.clone())
                .await
                .unwrap_or_else(|e| panic!("compact: {e}"));
            assert_eq!(
                compacted, 4,
                "[sorted={sorted}] only A's 4 L0 files are compacted"
            );
            let map = levels.read().await;
            let l0_left: Vec<_> = map
                .get(&0)
                .map(|v| v.iter().map(|f| f.path.clone()).collect())
                .unwrap_or_default();
            for b in &b_files {
                assert!(
                    l0_left.contains(b),
                    "[sorted={sorted}] B's L0 file left the map: {}",
                    b.display()
                );
                assert!(
                    b.exists(),
                    "[sorted={sorted}] B's L0 file was deleted: {}",
                    b.display()
                );
            }
            let l1 = map.get(&1).cloned().unwrap_or_default();
            assert!(
                !l1.is_empty(),
                "[sorted={sorted}] positive control: A's L1 written"
            );
            for f in &l1 {
                assert!(
                    super::owns_sstable(&dir_a, &f.path),
                    "[sorted={sorted}] an output landed outside A's directory: {}",
                    f.path.display()
                );
                for r in compactor
                    .read_sstable(&f.path)
                    .await
                    .unwrap_or_else(|e| panic!("read: {e}"))
                {
                    assert!(
                        r.oid.starts_with('a'),
                        "[sorted={sorted}] B's record '{}' was moved into A's L1",
                        r.oid
                    );
                }
            }
            drop(map);
            std::mem::forget(tmp);
        }
    }

    #[test]
    fn owns_sstable_matches_rediscovered_files_in_every_data_dir_form() {
        // #2530 review HIGH-1: rediscovered SSTables come back as absolute
        // `file://` URLs, whatever form the collection's data_dir was written
        // in. The default storage URL is the RELATIVE `file://./data`.
        use std::path::{Path, PathBuf};
        let cwd = std::env::current_dir().unwrap_or_else(|e| panic!("cwd: {e}"));
        let abs = cwd.join("rel_data").join("c1");
        let discovered = PathBuf::from(format!("file://{}/L0_x.helix", abs.display()));
        for data_dir in [
            "file://./rel_data/c1".to_string(),
            "./rel_data/c1".to_string(),
            "rel_data/c1".to_string(),
            format!("file://{}", abs.display()),
            format!("file://{}/", abs.display()),
            abs.display().to_string(),
        ] {
            assert!(
                super::owns_sstable(Path::new(&data_dir), &discovered),
                "data_dir {data_dir} must own {}",
                discovered.display()
            );
        }
        // Flushed form (built from the raw data_dir) still matches.
        let raw = Path::new("file://./rel_data/c1");
        assert!(super::owns_sstable(raw, &raw.join("L0_y.helix")));
        // Still exact: another collection is never owned.
        assert!(!super::owns_sstable(
            Path::new("file://./rel_data/c1"),
            &PathBuf::from(format!(
                "file://{}/L0_x.helix",
                cwd.join("rel_data").join("c10").display()
            ))
        ));
        // Azure aliases resolve to the one flat Blob backend.
        assert!(super::owns_sstable(
            Path::new("adls://cont/t/ns/c1/data"),
            Path::new("az://cont/t/ns/c1/data/L0_z.helix")
        ));
    }

    #[test]
    fn l0_trigger_counts_only_the_collections_own_files_in_any_form() {
        // Round review MEDIUM-1 (M3): the trigger must count only this
        // collection, including its rediscovered (URL-form) files.
        let meta = |p: &str| SStableMetadata {
            path: PathBuf::from(p),
            level: 0,
            hilbert_range: None,
            num_vectors: 0,
            size_bytes: 0,
            created_at: chrono::Utc::now(),
            blocks: Vec::new(),
            bloom_filter: None,
        };
        let dir = std::path::Path::new("/d/t1/c1");
        let mut l0 = vec![
            meta("/d/t1/c1/a.helix"),
            meta("file:///d/t1/c1/b.helix"),
            meta("/d/t1/c1/c.helix"),
        ];
        l0.extend((0..5).map(|i| meta(&format!("/d/t2/c9/{i}.helix"))));
        let levels = HashMap::from([(0_usize, l0)]);
        assert!(
            !super::l0_trigger_reached(&levels, dir, 4),
            "3 own files < 4 (others ignored)"
        );
        assert!(
            super::l0_trigger_reached(&levels, dir, 3),
            "rediscovered URL form counts"
        );
    }

    #[tokio::test]
    async fn acquire_compaction_skips_or_waits_while_one_runs() {
        // MEDIUM-1 (M4/M6): one compaction per collection at a time.
        let guard = Arc::new(tokio::sync::Mutex::new(()));
        let held = super::acquire_compaction(guard.clone(), true)
            .await
            .unwrap_or_else(|| panic!("first acquire"));
        assert!(
            super::acquire_compaction(guard.clone(), false)
                .await
                .is_none(),
            "flush-triggered compaction must skip while one runs"
        );
        let waiter = tokio::spawn(super::acquire_compaction(guard.clone(), true));
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "explicit compaction must wait");
        drop(held);
        assert!(
            waiter
                .await
                .unwrap_or_else(|e| panic!("join: {e}"))
                .is_some()
        );
    }

    #[test]
    fn engine_sites_use_the_scoped_trigger_and_the_guard() {
        // Structural: both compaction entry points in helix/mod.rs go through
        // the guard (skip for flush, wait for do_compact) and the scoped trigger.
        let engine = include_str!("mod.rs");
        let engine = engine.split("#[cfg(test)]").next().unwrap_or(engine);
        assert!(
            engine.contains("compaction::acquire_compaction(guard, false)"),
            "flush must skip"
        );
        assert!(
            engine.contains("compaction::acquire_compaction(guard, true)"),
            "do_compact must wait"
        );
        assert!(
            engine.contains("compaction::l0_trigger_reached("),
            "per-collection trigger"
        );
    }
}
