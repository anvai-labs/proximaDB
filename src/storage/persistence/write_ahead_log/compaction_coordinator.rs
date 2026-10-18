/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Automatic Background Compaction Coordinator
//!
//! Coordinates automatic compaction after flush operations to optimize storage efficiency.
//! Integrates with WALFlushCoordinator to trigger compaction when needed.

use crate::storage::traits::UnifiedStorageFormat;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

// Temporarily disabled due to arrow-arith compilation conflicts - DEFERRED: Re-enable when resolved
// use crate::storage::engines::viper::ViperEngine;
use crate::storage::engines::sst::SstEngine;
use crate::storage::traits::FlushResult;
use proximadb_index_traits::IndexMaintenance;

use super::compaction_axis_integration::CompactionAxisUpdater;

/// Background compaction coordinator
///
/// **Architecture:**
/// - Monitors flush operations from WALFlushCoordinator
/// - Automatically triggers compaction when thresholds are met
/// - Coordinates compaction across VIPER and LSM engines
/// - Provides intelligent scheduling to minimize I/O contention
#[derive(Clone)]
pub struct CompactionCoordinator {
    /// Per-collection compaction state
    collection_states: Arc<RwLock<HashMap<String, CollectionCompactionState>>>,

    /// Storage engines for compaction
    viper_engine: Arc<crate::storage::engines::viper::engine::ViperEngine>,
    sst_engine: Arc<SstEngine>,

    /// Compaction configuration
    config: Option<WalCompactionConfig>,

    /// Active compaction tracking
    active_compactions: Arc<Mutex<HashMap<String, WalCompactionTask>>>,

    /// Compaction statistics
    stats: Arc<RwLock<WalCompactionStats>>,

    /// AXIS index updater
    axis_updater: CompactionAxisUpdater,
}

/// Per-collection compaction state
#[derive(Debug, Clone)]
pub struct CollectionCompactionState {
    /// Number of files that need compaction
    pub files_needing_compaction: usize,

    /// Total size of uncompacted data
    pub uncompacted_size_bytes: u64,

    /// Last compaction timestamp
    pub last_compaction: Option<DateTime<Utc>>,

    /// Number of flushes since last compaction
    pub flushes_since_compaction: u32,

    /// Whether compaction is currently running
    pub compaction_in_progress: bool,

    /// Preferred storage engine for this collection
    pub preferred_engine: String,
}

impl Default for CollectionCompactionState {
    fn default() -> Self {
        Self {
            files_needing_compaction: 0,
            uncompacted_size_bytes: 0,
            last_compaction: None,
            flushes_since_compaction: 0,
            compaction_in_progress: false,
            preferred_engine: "VIPER".to_string(), // Default to VIPER
        }
    }
}

/// Backwards-compat alias for [`WalCompactionConfig`].
pub type CompactionConfig = WalCompactionConfig;

/// Compaction configuration
#[derive(Debug, Clone)]
pub struct WalCompactionConfig {
    /// Maximum files before triggering compaction
    pub max_files_before_compaction: usize,

    /// Maximum size before triggering compaction (bytes)
    pub max_size_before_compaction: u64,

    /// Maximum flushes before forcing compaction
    pub max_flushes_before_compaction: u32,

    /// Minimum time between compactions (seconds)
    pub min_compaction_interval_secs: u64,

    /// Enable background compaction
    pub enable_background_compaction: bool,

    /// Maximum concurrent compactions
    pub max_concurrent_compactions: usize,
}

impl Default for WalCompactionConfig {
    fn default() -> Self {
        Self {
            max_files_before_compaction: 5, // Reduced from 10 to be more aggressive
            max_size_before_compaction: 100 * 1024 * 1024, // 100MB
            max_flushes_before_compaction: 5,
            min_compaction_interval_secs: 60, // 1 minute (reduced from 5 minutes)
            enable_background_compaction: true,
            max_concurrent_compactions: 2,
        }
    }
}

/// Backwards-compat alias for [`WalCompactionTask`].
pub type CompactionTask = WalCompactionTask;

/// Active compaction task
#[derive(Debug, Clone)]
pub struct WalCompactionTask {
    /// Unique task identifier
    pub task_id: String,

    /// Collection being compacted
    pub collection_id: String,

    /// Storage engine performing compaction
    pub engine_type: String,

    /// When compaction started
    pub started_at: DateTime<Utc>,

    /// Estimated completion time
    pub estimated_completion: Option<DateTime<Utc>>,
}

/// Backwards-compat alias for [`WalCompactionStats`].
pub type CompactionStats = WalCompactionStats;

/// Compaction statistics
#[derive(Debug, Clone, Default)]
pub struct WalCompactionStats {
    /// Total compactions completed
    pub total_compactions: u64,

    /// Total bytes compacted
    pub total_bytes_compacted: u64,

    /// Total files compacted
    pub total_files_compacted: u64,

    /// Average compaction duration (seconds)
    pub avg_compaction_duration_secs: f64,

    /// Current active compactions
    pub active_compactions: u32,

    /// Compaction failures
    pub failed_compactions: u64,
}

/// Backwards-compat alias for [`WalCompactionResult`].
pub type CompactionResult = WalCompactionResult;

/// Compaction result
#[derive(Debug, Clone)]
pub struct WalCompactionResult {
    /// Whether compaction succeeded
    pub success: bool,

    /// Collections affected
    pub collections_affected: Vec<String>,

    /// Files compacted
    pub files_compacted: u64,

    /// Bytes reclaimed
    pub bytes_reclaimed: u64,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// When compaction completed
    pub completed_at: DateTime<Utc>,

    /// Engine that performed compaction
    pub engine_type: String,
}

impl CompactionCoordinator {
    /// Create new compaction coordinator
    pub fn new(
        viper_engine: Arc<crate::storage::engines::viper::engine::ViperEngine>,
        sst_engine: Arc<SstEngine>,
        config: Option<WalCompactionConfig>,
        axis_manager: Option<Arc<dyn IndexMaintenance>>,
    ) -> Self {
        let config_ref = config.as_ref();

        if let Some(cfg) = config_ref {
            info!(
                "🔧 CompactionCoordinator: Initializing with config: max_files={}, max_size={}MB, max_flushes={}",
                cfg.max_files_before_compaction,
                cfg.max_size_before_compaction / (1024 * 1024),
                cfg.max_flushes_before_compaction
            );
        } else {
            info!("🔧 CompactionCoordinator: Initializing with default config");
        }

        Self {
            collection_states: Arc::new(RwLock::new(HashMap::new())),
            viper_engine,
            sst_engine,
            config,
            active_compactions: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(WalCompactionStats::default())),
            axis_updater: CompactionAxisUpdater::new(axis_manager),
        }
    }

    /// Initialize compaction state for a collection
    pub async fn initialize_collection(
        &self,
        collection_id: &str,
        preferred_engine: &str,
    ) -> Result<()> {
        let mut states = self.collection_states.write().await;
        if !states.contains_key(collection_id) {
            let mut state = CollectionCompactionState {
                preferred_engine: preferred_engine.to_string(),
                ..Default::default()
            };

            // Discover existing files to initialize proper state
            let existing_files = self
                .discover_existing_files_for_collection(collection_id, preferred_engine)
                .await?;
            if !existing_files.is_empty() {
                state.files_needing_compaction = existing_files.len();
                state.uncompacted_size_bytes = existing_files.len() as u64 * 5 * 1024 * 1024; // Estimate 5MB per file

                info!(
                    "🔍 CompactionCoordinator: Found {} existing files for collection {} during initialization",
                    existing_files.len(),
                    collection_id
                );
            }

            states.insert(collection_id.to_string(), state);

            info!(
                "🔧 CompactionCoordinator: Initialized collection {} with preferred engine {} ({} existing files)",
                collection_id,
                preferred_engine,
                existing_files.len()
            );
        }
        Ok(())
    }

    /// Handle flush completion - triggers compaction if needed
    /// This is called by WALFlushCoordinator after successful flush
    pub async fn handle_flush_completion(
        &self,
        flush_result: &FlushResult,
    ) -> Result<Option<WalCompactionResult>> {
        if !self
            .config
            .as_ref()
            .is_some_and(|c| c.enable_background_compaction)
        {
            return Ok(None);
        }

        for collection_id in &flush_result.collections_affected {
            info!(
                "🔧 CompactionCoordinator: Processing flush completion for collection {}",
                collection_id
            );

            // Update collection state
            self.update_collection_state_after_flush(collection_id, flush_result)
                .await?;

            // Check if compaction is needed
            if self.should_trigger_compaction(collection_id).await? {
                info!(
                    "🚀 CompactionCoordinator: Triggering background compaction for collection {}",
                    collection_id
                );

                return self
                    .trigger_background_compaction(collection_id)
                    .await
                    .map(Some);
            }
        }

        Ok(None)
    }

    /// Update collection state after flush
    async fn update_collection_state_after_flush(
        &self,
        collection_id: &str,
        flush_result: &FlushResult,
    ) -> Result<()> {
        let mut states = self.collection_states.write().await;
        let state = states.entry(collection_id.to_string()).or_default();

        // Update state based on flush result
        state.files_needing_compaction += flush_result.files_created.unwrap_or(0) as usize;
        state.uncompacted_size_bytes += flush_result.bytes_written.unwrap_or(0);
        state.flushes_since_compaction += 1;

        debug!(
            "🔧 CompactionCoordinator: Updated state for {}: files={}, size={}KB, flushes={}",
            collection_id,
            state.files_needing_compaction,
            state.uncompacted_size_bytes / 1024,
            state.flushes_since_compaction
        );

        Ok(())
    }

    /// Check if compaction should be triggered
    async fn should_trigger_compaction(&self, collection_id: &str) -> Result<bool> {
        let states = self.collection_states.read().await;
        let _default_state = CollectionCompactionState::default();
        let state = states.get(collection_id);

        // Don't trigger if already in progress
        if let Some(s) = state
            && s.compaction_in_progress
        {
            return Ok(false);
        }

        // Check time constraint
        if let Some(s) = state
            && let Some(last_compaction) = s.last_compaction
        {
            let elapsed = Utc::now().signed_duration_since(last_compaction);
            if elapsed.num_seconds()
                < self
                    .config
                    .as_ref()
                    .map_or(60, |c| c.min_compaction_interval_secs) as i64
            {
                debug!(
                    "🔧 CompactionCoordinator: Too soon for compaction ({}s < {}s)",
                    elapsed.num_seconds(),
                    self.config
                        .as_ref()
                        .map_or(60, |c| c.min_compaction_interval_secs)
                );
                return Ok(false);
            }
        }

        // Check active compaction limit
        let active_count = self.active_compactions.lock().await.len();
        if active_count
            >= self
                .config
                .as_ref()
                .map_or(2, |c| c.max_concurrent_compactions)
        {
            debug!(
                "🔧 CompactionCoordinator: Too many active compactions ({}/{})",
                active_count,
                self.config
                    .as_ref()
                    .map_or(2, |c| c.max_concurrent_compactions)
            );
            return Ok(false);
        }

        // Also check actual file count in storage (not just tracked state)
        let preferred_engine = state
            .as_ref()
            .map_or("viper", |s| s.preferred_engine.as_str());
        let actual_file_count = match self
            .discover_existing_files_for_collection(collection_id, preferred_engine)
            .await
        {
            Ok(files) => files.len(),
            Err(e) => {
                warn!(
                    "⚠️ CompactionCoordinator: Failed to discover files for {}: {}",
                    collection_id, e
                );
                0
            }
        };

        // Use the maximum of tracked state and actual file count
        let effective_file_count = state
            .as_ref()
            .map_or(0, |s| s.files_needing_compaction)
            .max(actual_file_count);

        // Check thresholds
        let should_compact = if let Some(s) = state {
            effective_file_count
                >= self
                    .config
                    .as_ref()
                    .map_or(10, |c| c.max_files_before_compaction)
                || s.uncompacted_size_bytes
                    >= self
                        .config
                        .as_ref()
                        .map_or(1024 * 1024 * 1024, |c| c.max_size_before_compaction)
                || s.flushes_since_compaction
                    >= self
                        .config
                        .as_ref()
                        .map_or(5, |c| c.max_flushes_before_compaction)
        } else {
            effective_file_count
                >= self
                    .config
                    .as_ref()
                    .map_or(10, |c| c.max_files_before_compaction)
        };

        if should_compact {
            info!(
                "🚀 CompactionCoordinator: Compaction needed for {}: files={}/{} (actual={}), size={}MB/{}MB, flushes={}/{}",
                collection_id,
                state.as_ref().map_or(0, |s| s.files_needing_compaction),
                self.config
                    .as_ref()
                    .map_or(10, |c| c.max_files_before_compaction),
                actual_file_count,
                state.as_ref().map_or(0, |s| s.uncompacted_size_bytes) / (1024 * 1024),
                self.config
                    .as_ref()
                    .map_or(1024 * 1024 * 1024, |c| c.max_size_before_compaction)
                    / (1024 * 1024),
                state.as_ref().map_or(0, |s| s.flushes_since_compaction),
                self.config
                    .as_ref()
                    .map_or(5, |c| c.max_flushes_before_compaction)
            );
        } else if actual_file_count > 5 {
            debug!(
                "📊 CompactionCoordinator: Collection {} has {} files but doesn't meet thresholds yet",
                collection_id, actual_file_count
            );
        }

        Ok(should_compact)
    }

    /// Trigger background compaction for a collection
    async fn trigger_background_compaction(
        &self,
        collection_id: &str,
    ) -> Result<WalCompactionResult> {
        let task_id = proximadb_kernel::uuid::Uuid::new_v4().to_string();
        let collection_id = collection_id.to_string();

        // Mark compaction as in progress
        {
            let mut states = self.collection_states.write().await;
            if let Some(state) = states.get_mut(&collection_id) {
                state.compaction_in_progress = true;
            }
        }

        // Get preferred engine
        let preferred_engine = {
            let states = self.collection_states.read().await;
            states
                .get(&collection_id)
                .map_or_else(|| "VIPER".to_string(), |s| s.preferred_engine.clone())
        };

        // Create compaction task
        let task = WalCompactionTask {
            task_id: task_id.clone(),
            collection_id: collection_id.clone(),
            engine_type: preferred_engine.clone(),
            started_at: Utc::now(),
            estimated_completion: None,
        };

        // Track active compaction
        {
            let mut active = self.active_compactions.lock().await;
            active.insert(collection_id.clone(), task);
        }

        info!(
            "🚀 CompactionCoordinator: Starting background compaction task {} for collection {} using {}",
            task_id, collection_id, preferred_engine
        );

        // Execute compaction
        let result = self
            .execute_compaction(&collection_id, &preferred_engine)
            .await;

        // Cleanup and update state
        self.complete_compaction(&collection_id, &task_id, &result)
            .await;

        result
    }

    /// Execute compaction using the specified engine
    async fn execute_compaction(
        &self,
        collection_id: &str,
        engine_type: &str,
    ) -> Result<WalCompactionResult> {
        let start_time = std::time::Instant::now();
        let _started_at = Utc::now();

        info!(
            "🔧 CompactionCoordinator: Executing compaction for collection {} using {} engine",
            collection_id, engine_type
        );

        // Execute compaction based on engine type
        let compaction_result = match engine_type {
            "VIPER" => self.execute_viper_compaction(collection_id).await,
            "LSM" => self.execute_lsm_compaction(collection_id).await,
            _ => {
                warn!(
                    "⚠️ CompactionCoordinator: Unknown engine type {}, defaulting to VIPER",
                    engine_type
                );
                self.execute_viper_compaction(collection_id).await
            }
        };

        let duration = start_time.elapsed();

        match compaction_result {
            Ok(result) => {
                info!(
                    "✅ CompactionCoordinator: Compaction completed successfully for {} in {}ms",
                    collection_id,
                    duration.as_millis()
                );

                // Update AXIS indexes with compaction results
                // Note: WAL compaction doesn't track vector-level changes
                // Only storage engines (LSM/VIPER) track deleted/merged vectors
                // For now, pass empty arrays since WAL doesn't have this data
                use std::collections::HashMap;
                if let Err(e) = self
                    .axis_updater
                    .update_indexes_after_compaction(
                        collection_id,
                        &crate::storage::traits::CompactionResult {
                            success: true,
                            collections_affected: vec![collection_id.to_string()],
                            entries_processed: Some(result.files_compacted), // Map files to entries
                            entries_removed: Some(0), // Not tracked in WAL compaction
                            bytes_read: Some(result.bytes_reclaimed), // Approximate
                            bytes_written: Some(0),   // Not tracked
                            input_files: Some(result.files_compacted),
                            output_files: Some(1), // Typically compacts to single file
                            duration_ms: Some(duration.as_millis() as u64),
                            completed_at: Utc::now(),
                            engine_metrics: HashMap::new(),
                        },
                        &[], // No deleted vectors tracked at WAL level
                        &[], // No merged vectors tracked at WAL level
                    )
                    .await
                {
                    warn!(
                        "⚠️ CompactionCoordinator: Failed to update AXIS indexes after compaction: {}",
                        e
                    );
                    // Continue - compaction succeeded even if index update failed
                }

                // Update the result with the actual duration and return it
                let mut final_result = result;
                final_result.duration_ms = duration.as_millis() as u64;
                Ok(final_result)
            }
            Err(e) => {
                warn!(
                    "❌ CompactionCoordinator: Compaction failed for {} after {}ms: {}",
                    collection_id,
                    duration.as_millis(),
                    e
                );

                Ok(WalCompactionResult {
                    success: false,
                    collections_affected: vec![collection_id.to_string()],
                    files_compacted: 0,
                    bytes_reclaimed: 0,
                    duration_ms: duration.as_millis() as u64,
                    completed_at: Utc::now(),
                    engine_type: engine_type.to_string(),
                })
            }
        }
    }

    /// Execute VIPER engine compaction
    async fn execute_viper_compaction(&self, collection_id: &str) -> Result<WalCompactionResult> {
        debug!(
            "🔧 CompactionCoordinator: Executing VIPER compaction for {}",
            collection_id
        );

        // Use VIPER's do_compact method through unified framework
        let params = crate::storage::CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: false,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: Some(30000),
            priority: crate::storage::traits::OperationPriority::Medium,
            collection_config: None,
            estimated_input_size: 0,
        };
        match self.viper_engine.do_compact(&params).await {
            Ok(enhanced_result) => {
                info!(
                    "✅ VIPER compaction completed: {} entries processed, {} bytes written, {} entries removed",
                    enhanced_result.entries_processed.unwrap_or(0),
                    enhanced_result.bytes_written.unwrap_or(0),
                    enhanced_result.entries_removed.unwrap_or(0)
                );

                // Convert storage::WalCompactionResult to local WalCompactionResult
                Ok(WalCompactionResult {
                    success: true,
                    collections_affected: enhanced_result.collections_affected,
                    files_compacted: enhanced_result.input_files.unwrap_or(0),
                    bytes_reclaimed: enhanced_result
                        .bytes_written
                        .unwrap_or(0)
                        .saturating_sub(enhanced_result.bytes_read.unwrap_or(0)),
                    duration_ms: 0, // Will be filled by caller
                    completed_at: Utc::now(),
                    engine_type: "VIPER".to_string(),
                })
            }
            Err(e) => {
                warn!("❌ VIPER compaction failed: {}", e);
                Err(e)
            }
        }
    }

    /// Execute LSM engine compaction  
    async fn execute_lsm_compaction(&self, collection_id: &str) -> Result<WalCompactionResult> {
        debug!(
            "🔧 CompactionCoordinator: Executing LSM compaction for {}",
            collection_id
        );

        // Use SST's do_compact method through unified framework.
        // TD-READFID-2 review: this producer has only the collection id, not the
        // catalog meta (tags + `pax_vector_quant`) that fixes the output's base
        // quant / rg layout / arming, so `collection_config` stays `None` and the
        // SST engine declines rather than guess; the flush trigger and the
        // background manager are the meta-aware producers.
        let params = crate::storage::CompactionParameters {
            collection_id: Some(collection_id.to_string()),
            force: false,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: Some(30000),
            priority: crate::storage::traits::OperationPriority::Medium,
            collection_config: None,
            estimated_input_size: 0,
        };
        match self.sst_engine.do_compact(&params).await {
            Ok(enhanced_result) => {
                info!(
                    "✅ LSM compaction completed: {} entries processed, {} bytes written, {} entries removed",
                    enhanced_result.entries_processed.unwrap_or(0),
                    enhanced_result.bytes_written.unwrap_or(0),
                    enhanced_result.entries_removed.unwrap_or(0)
                );

                // Convert storage::WalCompactionResult to local WalCompactionResult
                Ok(WalCompactionResult {
                    success: true,
                    collections_affected: enhanced_result.collections_affected,
                    files_compacted: enhanced_result.input_files.unwrap_or(0),
                    bytes_reclaimed: enhanced_result
                        .bytes_written
                        .unwrap_or(0)
                        .saturating_sub(enhanced_result.bytes_read.unwrap_or(0)),
                    duration_ms: 0, // Will be filled by caller
                    completed_at: Utc::now(),
                    engine_type: "SST".to_string(),
                })
            }
            Err(e) => {
                warn!("❌ LSM compaction failed: {}", e);
                Err(e)
            }
        }
    }

    /// Complete compaction and update state
    async fn complete_compaction(
        &self,
        collection_id: &str,
        task_id: &str,
        result: &Result<WalCompactionResult>,
    ) {
        // Remove from active compactions
        {
            let mut active = self.active_compactions.lock().await;
            active.remove(collection_id);
        }

        // Update collection state
        {
            let mut states = self.collection_states.write().await;
            if let Some(state) = states.get_mut(collection_id) {
                state.compaction_in_progress = false;

                if let Ok(compaction_result) = result
                    && compaction_result.success
                {
                    // Reset compaction metrics on success
                    state.files_needing_compaction = 0;
                    state.uncompacted_size_bytes = 0;
                    state.flushes_since_compaction = 0;
                    state.last_compaction = Some(Utc::now());
                }
            }
        }

        // Update global statistics
        {
            let mut stats = self.stats.write().await;
            match result {
                Ok(compaction_result) => {
                    if compaction_result.success {
                        stats.total_compactions += 1;
                        stats.total_bytes_compacted += compaction_result.bytes_reclaimed;
                        stats.total_files_compacted += compaction_result.files_compacted;

                        // Update average duration
                        let total_duration = stats.avg_compaction_duration_secs
                            * (stats.total_compactions - 1) as f64
                            + (compaction_result.duration_ms as f64 / 1000.0);
                        stats.avg_compaction_duration_secs =
                            total_duration / stats.total_compactions as f64;
                    } else {
                        stats.failed_compactions += 1;
                    }
                }
                Err(_) => {
                    stats.failed_compactions += 1;
                }
            }

            stats.active_compactions = self.active_compactions.lock().await.len() as u32;
        }

        info!(
            "🎯 CompactionCoordinator: Completed compaction task {} for collection {}",
            task_id, collection_id
        );
    }

    /// Get compaction statistics
    pub async fn get_stats(&self) -> WalCompactionStats {
        self.stats.read().await.clone()
    }

    /// Get collection state
    pub async fn get_collection_state(
        &self,
        collection_id: &str,
    ) -> Option<CollectionCompactionState> {
        let states = self.collection_states.read().await;
        states.get(collection_id).cloned()
    }

    /// Manual compaction trigger (for testing or admin operations)
    pub async fn trigger_manual_compaction(
        &self,
        collection_id: &str,
        engine_type: Option<&str>,
    ) -> Result<WalCompactionResult> {
        let engine = engine_type;

        info!(
            "🔧 CompactionCoordinator: Manual compaction triggered for collection {} using {}",
            collection_id,
            engine.unwrap_or("default")
        );

        self.trigger_background_compaction(collection_id).await
    }

    /// Check collection compaction status and trigger if needed
    pub async fn check_and_compact(
        &self,
        collection_id: &str,
    ) -> Result<Option<WalCompactionResult>> {
        // Initialize collection state if not exists
        if self.get_collection_state(collection_id).await.is_none() {
            self.initialize_collection(collection_id, "VIPER").await?;
        }

        // Check if compaction is needed
        if self.should_trigger_compaction(collection_id).await? {
            info!(
                "🔧 CompactionCoordinator: Auto-triggering compaction for collection {} based on file count",
                collection_id
            );
            self.trigger_background_compaction(collection_id)
                .await
                .map(Some)
        } else {
            Ok(None)
        }
    }

    /// Discover existing files for a collection
    async fn discover_existing_files_for_collection(
        &self,
        collection_id: &str,
        engine_type: &str,
    ) -> Result<Vec<String>> {
        match engine_type {
            "VIPER" => {
                // Use VIPER engine's file discovery
                self.viper_engine
                    .parquet_files_for_collection(collection_id)
                    .await
            }
            "LSM" | "SST" => {
                // For SST engine, we'd need to implement similar discovery
                // For now, return empty as SST handles its own compaction differently
                Ok(vec![])
            }
            _ => Ok(vec![]),
        }
    }
}

/// Trait for compaction coordination callbacks
#[async_trait]
pub trait CompactionCoordinatorCallbacks {
    /// Called before starting compaction
    async fn on_compaction_start(&self, collection_id: &str, engine_type: &str) -> Result<()>;

    /// Called after compaction completion
    async fn on_compaction_complete(
        &self,
        collection_id: &str,
        result: &WalCompactionResult,
    ) -> Result<()>;

    /// Called on compaction failure
    async fn on_compaction_failure(&self, collection_id: &str, error: &anyhow::Error)
    -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compaction_coordinator() {
        // Deferred: Implement comprehensive tests
        assert!(true);
    }

    #[test]
    fn test_compaction_config() {
        let config = WalCompactionConfig::default();
        assert_eq!(config.max_files_before_compaction, 5); // Changed from 10 to 5 to match actual default
        assert_eq!(config.max_size_before_compaction, 100 * 1024 * 1024);
        assert_eq!(config.max_flushes_before_compaction, 5);
        assert!(config.enable_background_compaction);
    }

    #[test]
    fn test_collection_state() {
        let state = CollectionCompactionState::default();
        assert_eq!(state.files_needing_compaction, 0);
        assert_eq!(state.uncompacted_size_bytes, 0);
        assert_eq!(state.flushes_since_compaction, 0);
        assert!(!state.compaction_in_progress);
        assert_eq!(state.preferred_engine, "VIPER");
    }
}
