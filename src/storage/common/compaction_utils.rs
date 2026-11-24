/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Common utilities for compaction across storage engines
//! Provides unified file discovery and filtering logic

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use super::flush_handler_trait::FlushHandlerFactory;
use crate::core::config::CompactionConfig;
use crate::storage::common::compaction_memory::current_compaction_input_budget;
use crate::storage::common::compaction_orchestrator::{GenericFileMetadata, TieredFileRegistry};
use crate::storage::persistence::filesystem::FilesystemFactory;

/// Canonical storage engine type (re-exported from `proximadb-storage-common` via
/// `core::types`). The previous local 7-variant copy was consolidated into the
/// single source of truth — variant sets were identical (SST/NOVA/SWIFT/
/// HELIX/RAPTOR/TST). The dead `from_proto`/`from_str_uppercase`/`file_extension`
/// helpers had no external callers and were dropped (proto bridging lives on the
/// canonical type and the proto-bridge impls in `background_flush_context`).
pub use crate::core::types::StorageEngineType;

/// Result of file discovery with EventLog filtering
#[derive(Debug, Clone)]
pub struct FilteredCompactionFiles {
    /// Files ready for compaction (not pending in EventLog)
    pub compactable_files: HashMap<u32, Vec<GenericFileMetadata>>,
    /// Files pending AXIS processing (excluded from compaction)
    pub pending_files: HashMap<u32, Vec<String>>,
    /// Total file count across all levels
    pub total_files: usize,
    /// Count of compactable files
    pub compactable_count: usize,
    /// Count of pending files
    pub pending_count: usize,
}

/// Unified file discovery with EventLog filtering
/// This eliminates duplicate logic across the SST-family engines
pub struct CompactionFileDiscovery {
    registry: TieredFileRegistry,
    filesystem: Arc<FilesystemFactory>,
}

impl CompactionFileDiscovery {
    /// Create new file discovery instance
    pub fn new(filesystem: Arc<FilesystemFactory>) -> Self {
        Self {
            registry: TieredFileRegistry::new(),
            filesystem,
        }
    }

    /// Discover and filter compactable files for a collection
    /// Files pending in EventLog (AXIS processing) are excluded
    pub async fn discover_compactable_files(
        &self,
        collection_id: &str,
        data_directory: &str,
        extension: &str,
        engine_type: StorageEngineType,
    ) -> Result<FilteredCompactionFiles> {
        debug!(
            "📁 COMPACTION: Discovering {} files in {} for collection {}",
            extension, data_directory, collection_id
        );

        // Discover all files using the unified registry
        let all_files = self
            .registry
            .discover_files(&self.filesystem, data_directory, extension)
            .await?;

        // Get appropriate EventLog handler based on engine type
        let mut compactable_files = HashMap::new();
        let mut pending_files = HashMap::new();
        let mut total_files = 0;
        let mut compactable_count = 0;
        let mut pending_count = 0;

        // Process each level
        for (level, files) in all_files {
            let mut level_compactable = Vec::new();
            let mut level_pending = Vec::new();

            for file_meta in files {
                total_files += 1;
                let file_path = file_meta.path.clone();

                // Check if file can be compacted using unified handler
                let handler = FlushHandlerFactory::create(engine_type);
                let can_compact = handler
                    .can_compact_files(collection_id, std::slice::from_ref(&file_path))
                    .await
                    .unwrap_or(false);

                if can_compact {
                    level_compactable.push(file_meta);
                    compactable_count += 1;
                    debug!(
                        "  ✅ Level {} file {} is ready for compaction_info",
                        level, file_path
                    );
                } else {
                    level_pending.push(file_path.clone());
                    pending_count += 1;
                    debug!(
                        "  ⏸️ Level {} file {} is pending AXIS processing",
                        level, file_path
                    );
                }
            }

            if !level_compactable.is_empty() {
                compactable_files.insert(level, level_compactable);
            }
            if !level_pending.is_empty() {
                pending_files.insert(level, level_pending);
            }
        }

        info!(
            "🔍 COMPACTION: Discovery complete for collection {}:\n  Total files: {}\n  Compactable: {}\n  Pending AXIS: {}",
            collection_id, total_files, compactable_count, pending_count
        );

        // Log details by level if there are pending files
        if pending_count > 0 {
            for (level, files) in &pending_files {
                debug!(
                    "  Level {} has {} pending files: {:?}",
                    level,
                    files.len(),
                    files
                );
            }
        }

        Ok(FilteredCompactionFiles {
            compactable_files,
            pending_files,
            total_files,
            compactable_count,
            pending_count,
        })
    }

    /// Check if compaction should be triggered based on filtered files
    pub fn should_trigger_compaction(
        &self,
        filtered_files: &FilteredCompactionFiles,
        level: u32,
        threshold: usize,
    ) -> bool {
        if let Some(level_files) = filtered_files.compactable_files.get(&level) {
            let should_compact = level_files.len() >= threshold;

            if should_compact {
                info!(
                    "✅ COMPACTION: Level {} has {} compactable files (>= threshold {})",
                    level,
                    level_files.len(),
                    threshold
                );
            } else if filtered_files.pending_files.contains_key(&level) {
                debug!(
                    "⏸️ COMPACTION: Level {} has only {} compactable files (< threshold {}), some files pending",
                    level,
                    level_files.len(),
                    threshold
                );
            }

            should_compact
        } else {
            false
        }
    }

    /// Get compaction task files for a specific level
    pub fn get_compaction_files(
        &self,
        filtered_files: &FilteredCompactionFiles,
        level: u32,
    ) -> Vec<String> {
        filtered_files
            .compactable_files
            .get(&level)
            .map(|files| files.iter().map(|f| f.path.clone()).collect())
            .unwrap_or_default()
    }
}

/// Unified compaction task builder
/// Creates compaction tasks based on discovered files and thresholds
pub struct CompactionTaskBuilder;

fn higher_level_compaction_reduces_fanout(
    compactable_file_count: usize,
    threshold_triggered: bool,
) -> bool {
    threshold_triggered && compactable_file_count >= 2
}

fn higher_level_file_threshold(config: &CompactionConfig, level: u32) -> usize {
    (config.higher_level_file_threshold as f64 * config.level_multiplier.powi(level as i32 - 1))
        .ceil() as usize
}

fn l0_minimum_morsel_files(
    config: &CompactionConfig,
    engine_type: &StorageEngineType,
    extension: &str,
) -> usize {
    if config.l0_file_threshold == 1
        && *engine_type == StorageEngineType::SST
        && extension.eq_ignore_ascii_case("pax")
    {
        1
    } else {
        2
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompactionMorsel {
    input_files: Vec<String>,
    input_bytes: u64,
    total_candidate_bytes: u64,
    total_candidate_files: usize,
}

/// Pack the oldest files that fit the current input-byte budget. Oversized
/// files remain horizontal; smaller, mutually exclusive peers may still make
/// progress around them.
fn select_compaction_morsel_with_minimum(
    files: &[GenericFileMetadata],
    max_input_bytes: u64,
    minimum_files: usize,
) -> Option<CompactionMorsel> {
    let minimum_files = minimum_files.max(1);
    if max_input_bytes == 0 || files.len() < minimum_files {
        return None;
    }

    let mut ordered = files.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        left.timestamp
            .cmp(&right.timestamp)
            .then_with(|| left.path.cmp(&right.path))
    });

    let total_candidate_bytes = ordered
        .iter()
        .fold(0u64, |total, file| total.saturating_add(file.size_bytes));
    let mut input_files = Vec::new();
    let mut input_bytes = 0u64;
    for file in ordered {
        if file.size_bytes <= max_input_bytes.saturating_sub(input_bytes) {
            input_bytes = input_bytes.saturating_add(file.size_bytes);
            input_files.push(file.path.clone());
        }
    }

    (input_files.len() >= minimum_files).then_some(CompactionMorsel {
        input_files,
        input_bytes,
        total_candidate_bytes,
        total_candidate_files: files.len(),
    })
}

/// Ordinary compaction must reduce query-visible fanout, so it requires at
/// least two input files. The L0 training path calls the configurable helper
/// directly because its threshold-one rewrite changes the on-disk layout.
fn select_compaction_morsel(
    files: &[GenericFileMetadata],
    max_input_bytes: u64,
) -> Option<CompactionMorsel> {
    select_compaction_morsel_with_minimum(files, max_input_bytes, 2)
}

impl CompactionTaskBuilder {
    /// Check if compaction is needed and build task if necessary
    /// This unifies logic from SST's check_compaction_needed and the columnar engines' discover_compactable_files
    pub async fn check_and_build_compaction_task(
        collection_id: &str,
        data_directory: &str,
        extension: &str,
        engine_type: StorageEngineType,
        config: &CompactionConfig,
        filesystem: Arc<FilesystemFactory>,
    ) -> Result<Option<CompactionTaskInfo>> {
        debug!(
            "🔍 UNIFIED COMPACTION: Checking compaction for {:?} collection {} in {}",
            engine_type, collection_id, data_directory
        );

        // Use unified file discovery with EventLog filtering
        let file_discovery = CompactionFileDiscovery::new(filesystem);
        let filtered_files = file_discovery
            .discover_compactable_files(collection_id, data_directory, extension, engine_type)
            .await?;

        // Determine thresholds based on strategy
        let should_compact_l0 = match config.strategy.as_str() {
            "count" => file_discovery.should_trigger_compaction(
                &filtered_files,
                0,
                config.l0_file_threshold,
            ),
            "size" => {
                // Calculate total size at L0
                let l0_total_size_mb =
                    filtered_files.compactable_files.get(&0).map_or(0, |files| {
                        files
                            .iter()
                            .map(|f| f.size_bytes / (1024 * 1024))
                            .sum::<u64>() as usize
                    });
                l0_total_size_mb >= config.l0_size_threshold_mb
            }
            _ => {
                // Use both count and size thresholds
                let count_triggered = file_discovery.should_trigger_compaction(
                    &filtered_files,
                    0,
                    config.l0_file_threshold,
                );
                let l0_total_size_mb =
                    filtered_files.compactable_files.get(&0).map_or(0, |files| {
                        files
                            .iter()
                            .map(|f| f.size_bytes / (1024 * 1024))
                            .sum::<u64>() as usize
                    });
                let size_triggered = l0_total_size_mb >= config.l0_size_threshold_mb;
                count_triggered || size_triggered
            }
        };

        // Check if Level 0 compaction is needed
        if should_compact_l0 {
            let memory_budget = current_compaction_input_budget(config);
            // The flush training arm deliberately overrides the L0 threshold
            // to one so an untrained PAX segment can be promoted to the
            // searchable v3 layout. It is useful work even though it does not
            // reduce file count; admission still happens before training.
            let minimum_files = l0_minimum_morsel_files(config, &engine_type, extension);
            let morsel = filtered_files.compactable_files.get(&0).and_then(|files| {
                select_compaction_morsel_with_minimum(
                    files,
                    memory_budget.max_input_bytes,
                    minimum_files,
                )
            });

            if let Some(morsel) = morsel {
                info!(
                    input_files = morsel.input_files.len(),
                    input_bytes = morsel.input_bytes,
                    candidate_files = morsel.total_candidate_files,
                    candidate_bytes = morsel.total_candidate_bytes,
                    projected_budget_bytes = memory_budget.remaining_bytes,
                    layout_promotion = minimum_files == 1,
                    "✅ {} COMPACTION: Triggering bounded L0 morsel for collection {} (excluded {} pending files)",
                    engine_type,
                    collection_id,
                    filtered_files.pending_count
                );

                return Ok(Some(CompactionTaskInfo {
                    collection_id: collection_id.to_string(),
                    source_level: 0,
                    target_level: 1,
                    input_files: morsel.input_files,
                    input_bytes: morsel.input_bytes,
                    extension: extension.to_string(),
                    pending_files_count: filtered_files.pending_count,
                    total_files_count: filtered_files.total_files,
                }));
            }

            info!(
                max_input_bytes = memory_budget.max_input_bytes,
                effective_budget_bytes = memory_budget.effective_budget_bytes,
                reserved_bytes = memory_budget.reserved_bytes,
                "⏸️ {} COMPACTION: L0 remains horizontal because no file pair fits the current memory budget for collection {}",
                engine_type,
                collection_id
            );
        }

        // Check higher levels if needed (using configured max_levels)
        for level in 1..config.max_levels as u32 {
            // L0 admission batches writes; query-visible higher levels use a
            // lower base so any pair is consolidated rather than left as two
            // permanent query cascades. The multiplier applies between higher levels (L1 is
            // exponent zero), preserving the operator's write-amplification
            // trade-off without inheriting the L0 batching threshold.
            let level_file_threshold = higher_level_file_threshold(config, level);
            let level_size_threshold_mb = (config.l0_size_threshold_mb as f64
                * config.level_multiplier.powi(level as i32))
                as usize;
            let level_file_count = filtered_files
                .compactable_files
                .get(&level)
                .map_or(0, Vec::len);

            let should_compact =
                match config.strategy.as_str() {
                    "count" => file_discovery.should_trigger_compaction(
                        &filtered_files,
                        level,
                        level_file_threshold,
                    ),
                    "size" => {
                        let level_total_size_mb = filtered_files
                            .compactable_files
                            .get(&level)
                            .map_or(0, |files| {
                                files
                                    .iter()
                                    .map(|f| f.size_bytes / (1024 * 1024))
                                    .sum::<u64>() as usize
                            });
                        level_total_size_mb >= level_size_threshold_mb
                    }
                    _ => {
                        let count_triggered = file_discovery.should_trigger_compaction(
                            &filtered_files,
                            level,
                            level_file_threshold,
                        );
                        let level_total_size_mb = filtered_files
                            .compactable_files
                            .get(&level)
                            .map_or(0, |files| {
                                files
                                    .iter()
                                    .map(|f| f.size_bytes / (1024 * 1024))
                                    .sum::<u64>() as usize
                            });
                        let size_triggered = level_total_size_mb >= level_size_threshold_mb;
                        count_triggered || size_triggered
                    }
                };

            // Rewriting a single immutable segment cannot reduce read
            // amplification. In particular, a size-only trigger must not
            // cascade one large file through every level.
            if higher_level_compaction_reduces_fanout(level_file_count, should_compact) {
                let memory_budget = current_compaction_input_budget(config);
                let morsel = filtered_files
                    .compactable_files
                    .get(&level)
                    .and_then(|files| {
                        select_compaction_morsel(files, memory_budget.max_input_bytes)
                    });

                if let Some(morsel) = morsel {
                    info!(
                        input_files = morsel.input_files.len(),
                        input_bytes = morsel.input_bytes,
                        candidate_files = morsel.total_candidate_files,
                        candidate_bytes = morsel.total_candidate_bytes,
                        "✅ {} COMPACTION: Level {} triggering bounded morsel for collection {}",
                        engine_type,
                        level,
                        collection_id
                    );

                    return Ok(Some(CompactionTaskInfo {
                        collection_id: collection_id.to_string(),
                        source_level: level,
                        target_level: level + 1,
                        input_files: morsel.input_files,
                        input_bytes: morsel.input_bytes,
                        extension: extension.to_string(),
                        pending_files_count: filtered_files.pending_count,
                        total_files_count: filtered_files.total_files,
                    }));
                }

                info!(
                    level,
                    max_input_bytes = memory_budget.max_input_bytes,
                    effective_budget_bytes = memory_budget.effective_budget_bytes,
                    reserved_bytes = memory_budget.reserved_bytes,
                    "⏸️ {} COMPACTION: level remains horizontal because no file pair fits the current memory budget for collection {}",
                    engine_type,
                    collection_id
                );
            }
        }

        if filtered_files.pending_count > 0 {
            debug!(
                "⏸️ {} COMPACTION: Not enough compactable files for collection {} ({} ready, {} pending AXIS)",
                engine_type,
                collection_id,
                filtered_files.compactable_count,
                filtered_files.pending_count
            );
        } else {
            debug!(
                "📋 {} COMPACTION: No compaction needed for collection {} ({} total files)",
                engine_type, collection_id, filtered_files.total_files
            );
        }

        Ok(None)
    }

    /// Get all compactable files for a collection (used by the columnar engines for size-based compaction)
    pub async fn get_all_compactable_files(
        collection_id: &str,
        data_directory: &str,
        extension: &str,
        engine_type: StorageEngineType,
        filesystem: Arc<FilesystemFactory>,
    ) -> Result<Vec<String>> {
        let file_discovery = CompactionFileDiscovery::new(filesystem);
        let filtered_files = file_discovery
            .discover_compactable_files(collection_id, data_directory, extension, engine_type)
            .await?;

        // Get all compactable files across all levels
        let mut all_files = Vec::new();
        for (_level, files) in filtered_files.compactable_files {
            for file_meta in files {
                all_files.push(file_meta.path);
            }
        }

        if all_files.is_empty() && filtered_files.pending_count > 0 {
            info!(
                "⏸️ {} COMPACTION: All {} files are pending AXIS processing for collection {}",
                engine_type, filtered_files.pending_count, collection_id
            );
        }

        Ok(all_files)
    }
}

/// Information about a compaction task
#[derive(Debug, Clone)]
pub struct CompactionTaskInfo {
    pub collection_id: String,
    pub source_level: u32,
    pub target_level: u32,
    pub input_files: Vec<String>,
    /// Immutable segment bytes used for weighted memory admission.
    pub input_bytes: u64,
    pub extension: String,
    pub pending_files_count: usize,
    pub total_files_count: usize,
}

/// Self-healing behavior for compaction
/// When old files become available (AXIS completes), they automatically
/// become eligible for the next compaction cycle
pub struct CompactionSelfHealing;

impl CompactionSelfHealing {
    /// Log self-healing behavior when files transition from pending to ready
    pub fn log_file_transition(collection_id: &str, file_path: &str) {
        info!(
            "🔄 SELF-HEALING: File {} for collection {} is now ready for compaction_info",
            file_path, collection_id
        );
    }

    /// Check if self-healing occurred (previously pending files now ready)
    pub fn check_self_healing(
        previous_pending: &[String],
        current_compactable: &[String],
    ) -> Vec<String> {
        let mut healed_files = Vec::new();

        for file in previous_pending {
            if current_compactable.contains(file) {
                healed_files.push(file.clone());
            }
        }

        if !healed_files.is_empty() {
            info!(
                "🔄 SELF-HEALING: {} previously pending files are now compactable: {:?}",
                healed_files.len(),
                healed_files
            );
        }

        healed_files
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn file(level: u32) -> GenericFileMetadata {
        GenericFileMetadata {
            path: format!("L{level}.pax"),
            size_bytes: 1,
            level,
            timestamp: 0,
            extension: "pax".to_string(),
        }
    }

    fn sized_file(path: &str, timestamp: u64, size_mb: u64) -> GenericFileMetadata {
        GenericFileMetadata {
            path: path.to_string(),
            size_bytes: size_mb * MIB,
            level: 0,
            timestamp,
            extension: "pax".to_string(),
        }
    }

    #[test]
    fn compaction_morsel_bounds_input_bytes_oldest_first() {
        let files = (0..5)
            .map(|index| sized_file(&format!("L0_{index}.pax"), index, 180))
            .collect::<Vec<_>>();

        let morsel = select_compaction_morsel(&files, 512 * MIB)
            .expect("two oldest files fit the compaction budget");

        assert_eq!(morsel.input_files, vec!["L0_0.pax", "L0_1.pax"]);
        assert_eq!(morsel.input_bytes, 360 * MIB);
        assert_eq!(morsel.total_candidate_bytes, 900 * MIB);
        assert_eq!(morsel.total_candidate_files, 5);
    }

    #[test]
    fn compaction_morsel_skips_unpairable_oversized_oldest_file() {
        let files = vec![
            sized_file("oversized.pax", 0, 600),
            sized_file("oldest-fitting.pax", 1, 200),
            sized_file("second-fitting.pax", 2, 200),
            sized_file("third-fitting.pax", 3, 100),
        ];

        let morsel = select_compaction_morsel(&files, 512 * MIB)
            .expect("smaller peers should compact around an oversized file");

        assert_eq!(
            morsel.input_files,
            vec![
                "oldest-fitting.pax",
                "second-fitting.pax",
                "third-fitting.pax"
            ]
        );
        assert_eq!(morsel.input_bytes, 500 * MIB);
    }

    #[test]
    fn compaction_morsel_requires_two_files_within_budget() {
        let files = vec![
            sized_file("first.pax", 0, 180),
            sized_file("second.pax", 1, 180),
        ];

        assert!(select_compaction_morsel(&files, 300 * MIB).is_none());
        assert!(select_compaction_morsel(&files[..1], 512 * MIB).is_none());
    }

    #[test]
    fn threshold_one_training_morsel_can_promote_one_untrained_file() {
        let files = vec![sized_file("untrained-L0.pax", 0, 180)];
        let mut config = CompactionConfig::default();
        config.l0_file_threshold = 1;

        let minimum_files = l0_minimum_morsel_files(&config, &StorageEngineType::SST, "pax");
        let morsel = select_compaction_morsel_with_minimum(&files, 512 * MIB, minimum_files)
            .expect("the explicit threshold-one training arm must promote its one L0 file");

        assert_eq!(minimum_files, 1);
        assert_eq!(morsel.input_files, vec!["untrained-L0.pax"]);
        assert_eq!(morsel.input_bytes, 180 * MIB);

        assert_eq!(
            l0_minimum_morsel_files(&config, &StorageEngineType::NOVA, "parquet"),
            2,
            "threshold one must not turn non-PAX compaction into one-file rewrites"
        );
    }

    #[test]
    fn zero_compaction_input_budget_fails_closed() {
        let files = vec![
            sized_file("first.pax", 0, 600),
            sized_file("second.pax", 1, 600),
            sized_file("third.pax", 2, 600),
        ];

        assert!(select_compaction_morsel(&files, 0).is_none());
    }

    #[tokio::test]
    async fn count_trigger_reads_the_requested_higher_level() {
        let filesystem = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("test filesystem"),
        );
        let discovery = CompactionFileDiscovery::new(filesystem);
        let filtered = FilteredCompactionFiles {
            compactable_files: HashMap::from([(1, vec![file(1), file(1), file(1)])]),
            pending_files: HashMap::new(),
            total_files: 3,
            compactable_count: 3,
            pending_count: 0,
        };

        assert!(discovery.should_trigger_compaction(&filtered, 1, 3));
        assert!(
            !discovery.should_trigger_compaction(&filtered, 0, 3),
            "L1 files must not be counted as L0"
        );
    }

    #[test]
    fn a_single_large_file_is_not_rewritten_to_reduce_gets() {
        assert!(!higher_level_compaction_reduces_fanout(1, true));
        assert!(higher_level_compaction_reduces_fanout(2, true));
        assert!(!higher_level_compaction_reduces_fanout(3, false));
    }

    #[test]
    fn default_higher_levels_consolidate_a_pair() {
        let config = CompactionConfig::default();

        assert_eq!(higher_level_file_threshold(&config, 1), 2);
        assert_eq!(higher_level_file_threshold(&config, 2), 2);
    }

    #[test]
    fn test_self_healing_detection() {
        let previous_pending = vec![
            "file1.sstable".to_string(),
            "file2.sstable".to_string(),
            "file3.sstable".to_string(),
        ];

        let current_compactable = vec![
            "file1.sstable".to_string(),
            "file3.sstable".to_string(),
            "file4.sstable".to_string(),
        ];

        let healed =
            CompactionSelfHealing::check_self_healing(&previous_pending, &current_compactable);

        assert_eq!(healed.len(), 2);
        assert!(healed.contains(&"file1.sstable".to_string()));
        assert!(healed.contains(&"file3.sstable".to_string()));
    }
}
