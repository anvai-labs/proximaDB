//! Recovery Manager for WAL operations
//!
//! This module handles WAL recovery with proper separation of concerns:
//! - Disk manager for reading WAL files
//! - Serialization adapters for deserializing data
//! - Storage engines (LSM/VIPER) as primary recovery destination
//! - Flush coordinator for coordinating recovery and flush operations
//!
//! Key Design Decision: Recovery goes directly to storage engines, not memtable.
//! This ensures durability and prevents memory pressure during recovery.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

use crate::storage::BatchId;
use crate::storage::persistence::write_ahead_log::{
    WALFlushCoordinator, WalFileInfo, WriteAheadLogDiskManager,
    recovery_thread_pool::get_recovery_thread_pool, serialization::SerializerFactory,
};
use crate::storage::traits::UnifiedStorageFormat;

/// Recovery destination configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryMode {
    /// Recover directly to storage engine (recommended)
    DirectToStorage,
    /// Recover to memtable then flush (alternative mode)
    ViaMemtable,
}

/// Manager for WAL recovery operations
pub struct RecoveryManager {
    /// Disk manager for reading WAL files
    disk_manager: Arc<WriteAheadLogDiskManager>,
    /// Storage engines by collection (LSM or VIPER)
    storage_engines: Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
    /// Flush coordinator for managing recovery coordination
    flush_coordinator: Arc<WALFlushCoordinator>,
    /// Recovery mode
    recovery_mode: RecoveryMode,
    /// Statistics
    stats: Arc<tokio::sync::RwLock<WalRecoveryStats>>,
    /// ADR-069 S1 local WAL root, resolved ONCE from the same `WALConfig` the
    /// write path uses (`local_wal_dir`: env override > TOML > None). TD-WAL-5:
    /// recovery used to consult only the env, so WAL segments written under a
    /// config-file `wal_local_dir` were invisible to replay until a manifest
    /// entry happened to record their base — a crash right after the ack lost
    /// acknowledged writes.
    local_wal_dir: Option<String>,
}

/// Backwards-compat alias for [`WalRecoveryStats`].
pub type RecoveryStats = WalRecoveryStats;

/// Statistics for recovery operations
#[derive(Debug, Clone, Default)]
pub struct WalRecoveryStats {
    pub total_files_recovered: u64,
    pub total_vectors_recovered: u64,
    pub total_collections_recovered: usize,
    pub recovery_errors: u64,
    pub total_bytes_processed: u64,
}

/// Recovery progress callback
pub type RecoveryProgressCallback = Box<dyn Fn(RecoveryProgress) + Send + Sync>;

/// Recovery progress information
#[derive(Debug, Clone)]
pub struct RecoveryProgress {
    pub current_file: usize,
    pub total_files: usize,
    pub current_collection: String,
    pub vectors_recovered: u64,
    pub bytes_processed: u64,
}

impl Clone for RecoveryManager {
    fn clone(&self) -> Self {
        Self {
            disk_manager: self.disk_manager.clone(),
            storage_engines: self.storage_engines.clone(),
            flush_coordinator: self.flush_coordinator.clone(),
            recovery_mode: self.recovery_mode,
            stats: self.stats.clone(),
            local_wal_dir: self.local_wal_dir.clone(),
        }
    }
}

#[allow(dead_code)]
impl RecoveryManager {
    /// Create a new recovery manager with direct-to-storage recovery
    pub fn new(
        config: crate::storage::persistence::write_ahead_log::config::WALConfig,
        _wal_behavior: Arc<crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper>,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Self {
        info!(
            "🎯 Creating RecoveryManager with direct-to-storage recovery (catalog-backed config resolution)"
        );
        let local_wal_dir = crate::storage::persistence::write_ahead_log::local_wal_dir(&config);

        // Create disk manager
        let disk_manager = Arc::new(WriteAheadLogDiskManager::new(
            filesystem_factory.clone(),
            // Use the first data directory from WAL config as base for disk manager
            config
                .multi_disk
                .data_directories
                .first()
                .cloned()
                .unwrap_or_default(),
        ));

        Self {
            disk_manager,
            storage_engines: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            flush_coordinator: Arc::new(WALFlushCoordinator::new()), // Flush coordinator is internal to recovery
            // CRITICAL FIX: Use ViaMemtable for startup recovery since storage engines aren't initialized yet
            // This recovers vectors to memtable where they're immediately searchable
            // Can be flushed to storage later when engines are ready
            recovery_mode: RecoveryMode::ViaMemtable,
            stats: Arc::new(tokio::sync::RwLock::new(WalRecoveryStats::default())),
            local_wal_dir,
        }
    }

    /// Set recovery mode
    pub fn set_recovery_mode(&mut self, mode: RecoveryMode) {
        self.recovery_mode = mode;
        info!("Set recovery mode to {:?}", mode);
    }

    /// Register a storage engine for a collection
    pub async fn register_storage_engine(
        &self,
        collection_id: &str,
        engine: Arc<dyn UnifiedStorageFormat>,
    ) -> Result<()> {
        // Register with our internal map
        let mut engines = self.storage_engines.write().await;
        engines.insert(collection_id.to_string(), engine.clone());

        // Also register with flush coordinator by engine type
        // The flush coordinator needs engines registered by type (VIPER, LSM), not collection
        let engine_type = engine.format_name(); // Get the engine name
        self.flush_coordinator
            .register_storage_engine(engine_type, engine)
            .await;

        info!(
            "✅ Registered {} storage engine for collection {}",
            engine_type, collection_id
        );
        Ok(())
    }

    /// Recover all collections from disk to storage engines in parallel
    pub async fn recover_all(&self) -> Result<WalRecoveryStats> {
        info!(
            "🔄 Starting WAL recovery using global manifest (mode: {:?})",
            self.recovery_mode
        );
        // TD-WAL-1 S6: measure this boot's replay wall-clock for the
        // `proximadb_wal_replay_duration_seconds` gauge.
        let replay_started_at = std::time::Instant::now();

        // Get the recovery thread pool
        let thread_pool = get_recovery_thread_pool();

        // Start recovery phase - this acquires all CPU resources
        let recovery_guard = thread_pool
            .start_recovery()
            .await
            .context("Failed to start recovery phase")?;

        // Use global manifest for proper LSN ordering
        trace!("🔍 : Getting active entries from global manifest...");
        let all_entries =
            crate::storage::persistence::write_ahead_log::manifest::get_active_entries().await;
        info!(
            "🔍 DEBUG: Global manifest returned {} active entries",
            all_entries.len()
        );

        info!(
            "📂 Found {} WAL batches across collections (sorted by global LSN)",
            all_entries.len()
        );
        for (i, entry) in all_entries.iter().take(5).enumerate() {
            info!(
                "  Entry {}: LSN={}, collection={}, batch={}, status={:?}",
                i + 1,
                entry.global_lsn,
                entry.collection_id,
                entry.batch_id,
                entry.status
            );
        }

        // Group by collection for organized recovery
        let mut collections_to_recover: std::collections::HashSet<String> = all_entries
            .iter()
            .map(|e| e.collection_id.clone())
            .collect();
        // The manifest is only a cache. Catalog boot registers engines before
        // recovery, so their canonical collection ids must also be LISTed even
        // when the async manifest pointer is empty.
        collections_to_recover.extend(self.storage_engines.read().await.keys().cloned());
        let collections: Vec<String> = collections_to_recover.into_iter().collect();
        info!(
            "Found {} collections to recover using {} threads",
            collections.len(),
            num_cpus::get()
        );

        if collections.is_empty() {
            // Recovery phase complete
            recovery_guard.complete(0, 0).await;
            crate::metrics::wal_flush_metrics::set_replay_duration(
                replay_started_at.elapsed().as_secs_f64(),
            );
            return Ok(WalRecoveryStats::default());
        }

        // Create recovery tasks for all collections
        let mut recovery_tasks = Vec::new();

        for collection_id in collections {
            let collection_id_clone = collection_id.clone();
            let disk_manager = self.disk_manager.clone();
            let storage_engines = self.storage_engines.clone();
            let flush_coordinator = self.flush_coordinator.clone();
            let recovery_mode = self.recovery_mode;
            let local_wal_dir = self.local_wal_dir.clone();

            // Execute recovery task through thread pool
            let task = thread_pool.execute_recovery_task("recover_collection", async move {
                info!(
                    "🧵 Starting recovery for collection: {}",
                    collection_id_clone
                );

                let result = Self::recover_collection_internal(
                    &collection_id_clone,
                    disk_manager,
                    storage_engines,
                    flush_coordinator,
                    recovery_mode,
                    local_wal_dir.as_deref(),
                    None,
                )
                .await;

                match &result {
                    Ok((vectors, files)) => info!(
                        "✅ Collection {} recovered: {} vectors from {} files",
                        collection_id_clone, vectors, files
                    ),
                    Err(e) => warn!(
                        "❌ Collection {} recovery failed: {}",
                        collection_id_clone, e
                    ),
                }

                Ok((collection_id_clone, result))
            });

            recovery_tasks.push(task);
        }

        // Wait for all recovery tasks to complete
        info!(
            "⏳ Waiting for {} parallel recovery tasks to complete...",
            recovery_tasks.len()
        );
        let recovery_results = futures::future::join_all(recovery_tasks).await;

        // Process results
        let mut total_vectors = 0u64;
        let mut total_files = 0u64;
        let mut recovery_errors = 0u64;
        let mut successful_collections = 0usize;

        for result in recovery_results {
            match result {
                Ok((collection_id, Ok((vectors, files)))) => {
                    total_vectors += vectors;
                    total_files += files;
                    successful_collections += 1;
                    debug!(
                        "Collection {} recovered successfully with {} vectors from {} files",
                        collection_id, vectors, files
                    );
                }
                Ok((collection_id, Err(e))) => {
                    warn!("Collection {} recovery failed: {}", collection_id, e);
                    recovery_errors += 1;
                }
                Err(e) => {
                    warn!("Recovery task panicked: {}", e);
                    recovery_errors += 1;
                }
            }
        }

        // Notify flush coordinator that recovery is complete
        // Recovery complete

        // Complete recovery phase and release all threads
        recovery_guard
            .complete(successful_collections, total_vectors)
            .await;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_collections_recovered = successful_collections;
            stats.total_vectors_recovered = total_vectors;
            stats.total_files_recovered = total_files;
            stats.recovery_errors = recovery_errors;
        }

        let stats = self.stats.read().await.clone();
        let pool_stats = thread_pool.get_stats().await;

        info!(
            "✅ Parallel WAL recovery completed: {} collections, {} vectors, {} errors (peak {} threads, {}ms)",
            successful_collections,
            total_vectors,
            recovery_errors,
            pool_stats.peak_concurrent_threads,
            pool_stats.total_recovery_time_ms
        );

        // Cleanup manifest: Remove Flushed entries and old segments
        info!("🧹 Cleaning up global manifest after recovery...");
        if let Ok(removed) =
            crate::storage::persistence::write_ahead_log::manifest::cleanup_checkpointed().await
            && removed > 0
        {
            info!("🧹 Removed {} flushed manifest entries", removed);
        }

        crate::metrics::wal_flush_metrics::set_replay_duration(
            replay_started_at.elapsed().as_secs_f64(),
        );
        Ok(stats)
    }

    /// Recover a specific collection (public API)
    /// Returns WalRecoveryStats with detailed recovery information
    pub async fn recover_collection(&self, collection_id: &str) -> Result<WalRecoveryStats> {
        let (vectors_recovered, files_recovered) = Self::recover_collection_internal(
            collection_id,
            self.disk_manager.clone(),
            self.storage_engines.clone(),
            self.flush_coordinator.clone(),
            self.recovery_mode,
            self.local_wal_dir.as_deref(),
            None, // No progress callback for startup recovery
        )
        .await?;

        // Update global stats
        if vectors_recovered > 0 {
            let mut stats = self.stats.write().await;
            stats.total_collections_recovered += 1;
            stats.total_vectors_recovered += vectors_recovered;
            stats.total_files_recovered += files_recovered;
        }

        // Return recovery stats for this collection
        Ok(WalRecoveryStats {
            total_files_recovered: files_recovered,
            total_vectors_recovered: vectors_recovered,
            total_collections_recovered: if vectors_recovered > 0 { 1 } else { 0 },
            recovery_errors: 0,
            total_bytes_processed: 0, // Deferred: Track bytes if needed
        })
    }

    /// Recover a specific collection with progress callback (for manual recovery)
    pub async fn recover_collection_with_progress(
        &self,
        collection_id: &str,
        progress_callback: Option<RecoveryProgressCallback>,
    ) -> Result<u64> {
        let (vectors_recovered, files_recovered) = Self::recover_collection_internal(
            collection_id,
            self.disk_manager.clone(),
            self.storage_engines.clone(),
            self.flush_coordinator.clone(),
            self.recovery_mode,
            self.local_wal_dir.as_deref(),
            progress_callback,
        )
        .await?;

        // Update stats
        if vectors_recovered > 0 {
            let mut stats = self.stats.write().await;
            stats.total_collections_recovered += 1;
            stats.total_vectors_recovered += vectors_recovered;
            stats.total_files_recovered += files_recovered;
        }

        Ok(vectors_recovered)
    }

    /// Internal collection recovery logic (can be called from parallel tasks)
    async fn recover_collection_internal(
        collection_id: &str,
        disk_manager: Arc<WriteAheadLogDiskManager>,
        storage_engines: Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        _flush_coordinator: Arc<WALFlushCoordinator>,
        recovery_mode: RecoveryMode,
        local_wal_dir: Option<&str>,
        progress_callback: Option<RecoveryProgressCallback>,
    ) -> Result<(u64, u64)> {
        info!(
            "🔄 Recovering collection: {} (mode: {:?})",
            collection_id, recovery_mode
        );

        Self::recover_collection_authoritative(
            collection_id,
            disk_manager.clone(),
            storage_engines.clone(),
            recovery_mode,
            local_wal_dir,
            &progress_callback,
        )
        .await
    }

    async fn recover_collection_authoritative(
        collection_id: &str,
        disk_manager: Arc<WriteAheadLogDiskManager>,
        storage_engines: Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        _recovery_mode: RecoveryMode,
        local_wal_dir: Option<&str>,
        progress_callback: &Option<RecoveryProgressCallback>,
    ) -> Result<(u64, u64)> {
        if !storage_engines.read().await.contains_key(collection_id) {
            anyhow::bail!(
                "no storage engine registered for WAL recovery collection {collection_id}"
            );
        }
        let manifest_entries =
            crate::storage::persistence::write_ahead_log::manifest::get_collection_entries(
                collection_id,
            )
            .await;

        // Issue #1125: the write path roots each collection's WAL under its
        // catalog-assigned base_location (load-balanced across the configured
        // storage locations), while this recovery manager's disk_manager is
        // rooted at the single configured write-buffer directory. Listing only
        // that one base silently missed fsync'd, manifest-Active WAL objects
        // (restart data loss: "0 vectors from 0 files" with the .bcwal on disk).
        // List every candidate base — the catalog assignment, every distinct
        // manifest storage_url, and the configured base — and union the results.
        let catalog_base =
            crate::storage::persistence::write_ahead_log::list_collections_from_catalog()
                .await
                .into_iter()
                .find(|collection| collection.id == collection_id)
                .and_then(|collection| collection.storage_assignment)
                .map(|assignment| assignment.base_location);
        let manifest_bases: Vec<String> = manifest_entries
            .iter()
            .map(|entry| entry.storage_url.clone())
            .collect();
        let candidate_bases = Self::wal_candidate_bases(
            local_wal_dir,
            catalog_base.as_deref(),
            &manifest_bases,
            disk_manager.get_base_wal_url(),
        );
        let listed =
            Self::list_wal_files_across_bases(&disk_manager, &candidate_bases, collection_id)
                .await?;
        if listed.is_empty() {
            return Ok((0, 0));
        }

        struct ReplayObject {
            file: WalFileInfo,
            data: Vec<u8>,
            token: Option<crate::storage::persistence::write_ahead_log::RecoveryToken>,
            manifest_lsn: Option<u64>,
            manifested: bool,
            checksum: u32,
            records: Vec<proximadb_records::ProximaRecord>,
        }

        let manifests_by_name: HashMap<_, _> = manifest_entries
            .iter()
            .filter_map(|entry| {
                entry
                    .file_path
                    .split('/')
                    .next_back()
                    .map(|name| (name.to_string(), entry))
            })
            .collect();
        let mut replay = Vec::with_capacity(listed.len());
        for file in listed {
            let file_name = file.file_url.split('/').next_back().unwrap_or("");
            let manifest = manifests_by_name.get(file_name).copied();
            if let Some(entry) = manifest
                && entry.status
                    != crate::storage::persistence::write_ahead_log::manifest::WalEntryStatus::Active
            {
                // Flushed is the cache's durable skip-list: materialization
                // committed before this status was published. Never replay it,
                // because later compaction may legitimately consume the original
                // deterministic L0 segment while a failed WAL delete leaves this
                // object behind.
                if matches!(
                    entry.status,
                    crate::storage::persistence::write_ahead_log::manifest::WalEntryStatus::Flushed
                        | crate::storage::persistence::write_ahead_log::manifest::WalEntryStatus::Archived
                        | crate::storage::persistence::write_ahead_log::manifest::WalEntryStatus::RolledBack
                ) && let Err(error) = disk_manager.delete_wal_file_url(&file.file_url).await
                {
                    warn!(file = %file.file_url, %error, "failed to retire skipped WAL object");
                }
                continue;
            }
            let read = disk_manager.read_batch_with_envelope(&file).await?;
            if let Some(entry) = manifest
                && read.checksum_crc32 != entry.checksum_crc32
            {
                anyhow::bail!("manifest checksum mismatch for {}", file.file_url);
            }
            let records = SerializerFactory::create(file.format)
                .deserialize_batch(&read.data)
                .with_context(|| format!("deserializing {}", file.file_url))?;
            if !read.record_ordinals.is_empty() && read.record_ordinals.len() != records.len() {
                anyhow::bail!("record ordinal count mismatch for {}", file.file_url);
            }
            replay.push(ReplayObject {
                file,
                data: read.data,
                token: read.recovery_token,
                manifest_lsn: manifest.map(|entry| entry.global_lsn),
                manifested: manifest.is_some(),
                checksum: read.checksum_crc32,
                records,
            });
        }
        if replay.is_empty() {
            return Ok((0, 0));
        }

        let storage_url = Self::storage_base_from_wal_url(&replay[0].file)?;
        if replay
            .iter()
            .any(|object| object.token.is_none() && !object.manifested)
        {
            let fresh = replay.len() == 1
                && manifest_entries.is_empty()
                && !Self::collection_has_segments(collection_id, &storage_url, &disk_manager)
                    .await?;
            if !fresh {
                anyhow::bail!(
                    "tokenless orphan WAL for collection {collection_id} overlaps existing WAL/segments; explicit operator ordering is required"
                );
            }
        }

        if let Some(tenant_id) = replay
            .iter()
            .filter_map(|object| object.token.as_ref())
            .map(|token| token.tenant_id.as_str())
            .find(|tenant| !tenant.is_empty())
        {
            if let Some(manager) = crate::cluster::partition_lease::global_partition_lease_manager()
            {
                if !manager
                    .begin_writer_incarnation(
                        tenant_id,
                        collection_id,
                        chrono::Utc::now().timestamp_millis(),
                    )
                    .await?
                {
                    anyhow::bail!("collection {collection_id} recovery is fenced by another pod");
                }
            } else if crate::storage::persistence::write_ahead_log::recovery_token::certified_mode()
            {
                anyhow::bail!("certified recovery has no partition lease manager");
            }
        }

        replay.sort_by(|left, right| match (&left.token, &right.token) {
            (None, None) => left.manifest_lsn.cmp(&right.manifest_lsn),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(left), Some(right)) => left.cmp(right),
        });

        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        let mut latest_by_oid: HashMap<String, (usize, proximadb_records::ProximaRecord)> =
            HashMap::new();
        let mut order = 0usize;
        // TD-DELVEC-1 WI-5 P1: retain each record's durable per-batch manifest_lsn
        // (the dedup map drops the batch→record association) so the post-flush
        // reconciliation pass can re-mark DV bits at the correct generation.
        #[cfg(feature = "cold-deletion-vectors")]
        let mut oid_to_lsn: HashMap<String, u64> = HashMap::new();
        for object in &replay {
            if let Some(token) = &object.token {
                digest.update(token.epoch.to_be_bytes());
                digest.update(token.sequence.to_be_bytes());
            } else {
                digest.update(object.manifest_lsn.unwrap_or_default().to_be_bytes());
            }
            digest.update(&object.data);
            for record in &object.records {
                // Latest mutation wins in durable token order. Tombstones remain
                // materialized so they continue suppressing values in older segments.
                latest_by_oid.insert(record.oid.clone(), (order, record.clone()));
                #[cfg(feature = "cold-deletion-vectors")]
                if let Some(lsn) = object.manifest_lsn {
                    oid_to_lsn.insert(record.oid.clone(), lsn);
                }
                order = order.saturating_add(1);
            }
        }
        let mut ordered = latest_by_oid.into_values().collect::<Vec<_>>();
        ordered.sort_by_key(|(record_order, _)| *record_order);
        let vectors = ordered
            .into_iter()
            .map(|(_, record)| record)
            .collect::<Vec<_>>();
        // TD-DELVEC-1 WI-5 P1: collect the recovery tombstones (with their durable
        // manifest_lsn) so the post-flush pass can re-mark any DV bits a crash
        // stranded between the WAL append and mark_deleted.
        #[cfg(feature = "cold-deletion-vectors")]
        let recovery_tombstones: Vec<(String, u64)> = vectors
            .iter()
            .filter(|r| r.valid_to_ns == Some(0) && r.origin.as_deref() == Some("delete"))
            .filter_map(|r| oid_to_lsn.get(&r.oid).map(|&lsn| (r.oid.clone(), lsn)))
            .collect();
        let digest = format!("{:x}", digest.finalize());
        let range = match (
            replay.first().and_then(|object| object.token.as_ref()),
            replay.last().and_then(|object| object.token.as_ref()),
        ) {
            (Some(first), Some(last)) => format!(
                "{:020}-{:020}_{:020}-{:020}",
                first.epoch, first.sequence, last.epoch, last.sequence
            ),
            _ => format!("legacy-{}", replay[0].file.batch_id.to_base62()),
        };
        let materialization_id = format!("{range}-{digest}");
        let batch_ids = replay.iter().map(|object| object.file.batch_id).collect();
        let result = Self::flush_recovered_range(
            collection_id,
            vectors.clone(),
            batch_ids,
            &storage_engines,
            &storage_url,
            &materialization_id,
            &digest,
        )
        .await?;
        if !result.success {
            anyhow::bail!("recovery materialization failed for collection {collection_id}");
        }

        // TD-DELVEC-1 WI-5 P1: re-mark deletion-vector bits for the recovery
        // tombstones (a crash may have stranded them between the WAL append and
        // mark_deleted). The just-flushed `L0_recovery_*` segments now exist on
        // disk, so resolve_oid_positions can find the target rows. Disk-authoritative:
        // the recovery engine's marks persist to `{segment}.dv`, which the canonical
        // serving engine reads. Best-effort — failure logs + continues.
        #[cfg(feature = "cold-deletion-vectors")]
        if !recovery_tombstones.is_empty()
            && let Some(engine) = storage_engines.read().await.get(collection_id).cloned()
            && let Err(e) = engine
                .reconcile_deletion_vectors(collection_id, &recovery_tombstones)
                .await
        {
            tracing::warn!("recovery DV reconcile failed for {collection_id}: {e:?}");
        }

        // Test-only crash-point seam (default unset ⇒ normal retirement, no runtime cost
        // on the hot path — recovery is infrequent). Simulates a process crash at a
        // precise point in the post-materialization retirement sequence so the two
        // recovery double-replay crash windows are deterministically reproducible in the
        // full-server restart harness (TD-OBJSTORE-4 S3). `after_materialize` = crash right
        // after the segment `write_if_absent` commit, before the manifest is marked flushed
        // (W1: next boot re-replays → `AlreadyExists` idempotency).
        let recovery_crash_point =
            std::env::var("PROXIMADB_TEST_RECOVERY_CRASH_POINT").unwrap_or_default();
        if recovery_crash_point == "after_materialize" {
            warn!(
                collection = %collection_id,
                "PROXIMADB_TEST_RECOVERY_CRASH_POINT=after_materialize: skipping manifest \
                 mark-flushed and WAL retirement to simulate a crash after materialization"
            );
            return Ok((vectors.len() as u64, replay.len() as u64));
        }

        if crate::storage::persistence::write_ahead_log::manifest::get_service().is_some() {
            for object in replay.iter().filter(|object| !object.manifested) {
                let file_name = object
                    .file
                    .file_url
                    .split('/')
                    .next_back()
                    .unwrap_or("")
                    .to_string();
                let entry =
                crate::storage::persistence::write_ahead_log::manifest::GlobalManifestEntry::new(
                    0,
                    collection_id.to_string(),
                    &object.file.batch_id,
                    file_name,
                    object.data.len() as u64,
                    object.checksum,
                    object.file.format,
                    object.records.len() as u64,
                    storage_url.clone(),
                );
                crate::storage::persistence::write_ahead_log::manifest::append_sync(entry).await?;
            }
            let batch_id_strings = replay
                .iter()
                .map(|object| object.file.batch_id.to_base62())
                .collect::<Vec<_>>();
            crate::storage::persistence::write_ahead_log::manifest::mark_flushed(&batch_id_strings)
                .await?;
        } else if crate::storage::persistence::write_ahead_log::recovery_token::certified_mode() {
            anyhow::bail!("certified recovery cannot repair an uninitialized manifest cache");
        }
        // W2: crash after the manifest is marked flushed but before WAL retirement — next
        // boot sees the WAL present + manifest `Flushed` and skips it via the durable
        // skip-list, retiring the stranded object. Test-only (default unset ⇒ normal path).
        if recovery_crash_point == "after_mark_flushed" {
            warn!(
                collection = %collection_id,
                "PROXIMADB_TEST_RECOVERY_CRASH_POINT=after_mark_flushed: skipping WAL \
                 retirement to simulate a crash after mark-flushed"
            );
            return Ok((vectors.len() as u64, replay.len() as u64));
        }
        for object in &replay {
            if let Err(error) = disk_manager
                .delete_wal_file_url(&object.file.file_url)
                .await
            {
                warn!(file = %object.file.file_url, %error, "recovery committed; WAL retirement will retry next boot");
            }
        }
        if let Some(callback) = progress_callback {
            callback(RecoveryProgress {
                current_file: replay.len(),
                total_files: replay.len(),
                current_collection: collection_id.to_string(),
                vectors_recovered: vectors.len() as u64,
                bytes_processed: replay.iter().map(|object| object.data.len() as u64).sum(),
            });
        }
        Ok((vectors.len() as u64, replay.len() as u64))
    }

    /// Issue #1125: assemble the ordered, deduplicated list of WAL base URLs a
    /// collection's files may live under. Priority: the resolved local WAL
    /// root (ADR-069 S1 / TD-WAL-5), then the catalog-assigned
    /// base_location (what the write path uses), then every distinct manifest
    /// storage_url (where files were actually recorded), then the configured
    /// primary base (this recovery manager's disk_manager root). Trailing
    /// slashes are normalized so `file:///x/` and `file:///x` dedupe.
    fn wal_candidate_bases(
        local_wal_dir: Option<&str>,
        catalog_base: Option<&str>,
        manifest_bases: &[String],
        primary_base: &str,
    ) -> Vec<String> {
        let mut bases: Vec<String> = Vec::new();
        let push_base = |bases: &mut Vec<String>, base: &str| {
            let normalized = base.trim_end_matches('/').to_string();
            if !normalized.is_empty() && !bases.contains(&normalized) {
                bases.push(normalized);
            }
        };
        // ADR-069 S1: when an opt-in local WAL root is set (env OR config — the
        // caller passes the value resolved by the write path's own resolver,
        // TD-WAL-5), the write path roots the WAL there — prepend it as a
        // candidate base so replay lists it (otherwise the .bcwal on /waldisk is
        // missed → restart data loss). Every base is listed and unioned. The
        // order decides which duplicate URL is kept and which listed file comes
        // first; the first REPLAYED file's base becomes `storage_url`, which is
        // the recovery flush target, the new manifest entries' `storage_url`,
        // and the tokenless-orphan segment probe root (TD-WAL-5 residual: under
        // a local WAL dir that is the WAL disk, not the data root).
        if let Some(local) = local_wal_dir {
            push_base(&mut bases, local);
        }
        if let Some(base) = catalog_base {
            push_base(&mut bases, base);
        }
        for base in manifest_bases {
            push_base(&mut bases, base);
        }
        push_base(&mut bases, primary_base);
        bases
    }

    /// Issue #1125: list a collection's WAL files across every candidate base,
    /// unioned and deduplicated by file URL. Listing failures stay fail-closed
    /// (`?`), preserving the TD-OBJSTORE-4 posture: an unlistable base aborts
    /// recovery rather than silently dropping data.
    async fn list_wal_files_across_bases(
        disk_manager: &Arc<WriteAheadLogDiskManager>,
        candidate_bases: &[String],
        collection_id: &str,
    ) -> Result<Vec<WalFileInfo>> {
        let primary_base = disk_manager.get_base_wal_url().trim_end_matches('/');
        let mut listed: Vec<WalFileInfo> = Vec::new();
        let mut seen_file_urls: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for base in candidate_bases {
            let base_manager = if base == primary_base {
                disk_manager.clone()
            } else {
                Arc::new(WriteAheadLogDiskManager::new(
                    disk_manager.filesystem_factory().clone(),
                    base.clone(),
                ))
            };
            for file in base_manager.list_collection_files(collection_id).await? {
                if seen_file_urls.insert(file.file_url.clone()) {
                    listed.push(file);
                }
            }
        }
        Ok(listed)
    }

    /// Recover a single WAL file (public API)
    async fn recover_file(&self, file_info: &WalFileInfo) -> Result<u64> {
        Self::recover_file_internal(
            file_info,
            &self.disk_manager,
            &self.storage_engines,
            self.recovery_mode,
        )
        .await
    }

    /// Internal file recovery logic (can be called from parallel tasks)
    async fn recover_file_internal(
        file_info: &WalFileInfo,
        disk_manager: &Arc<WriteAheadLogDiskManager>,
        storage_engines: &Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        recovery_mode: RecoveryMode,
    ) -> Result<u64> {
        debug!(
            "Recovering file: {} (format: {:?})",
            file_info.file_url, file_info.format
        );

        // Read the file data
        let data = disk_manager
            .read_batch(file_info)
            .await
            .context("Failed to read WAL file")?;

        // Create serializer for the format
        let serializer = SerializerFactory::create(file_info.format);

        // Deserialize vectors
        let vectors = serializer
            .deserialize_batch(&data)
            .context("Failed to deserialize WAL data")?;

        let vector_count = vectors.len() as u64;

        // Extract storage URL from file path
        // WAL files are at: {base_location}/{collection_id}/wal/{filename}
        // So we extract everything before /{collection_id}/wal/
        let storage_url = file_info
            .file_url
            .split(&format!("/{}/wal/", file_info.collection_id))
            .next()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not extract storage URL from WAL file path: {}",
                    file_info.file_url
                )
            })?
            .to_string();

        // Pass recovered vectors to flush coordinator
        // It will use the storage engine's do_flush method properly.
        let flush_result = Self::flush_recovered_vectors(
            file_info,
            vectors,
            disk_manager,
            storage_engines,
            recovery_mode,
            &storage_url,
        )
        .await
        .context("Failed to flush recovered vectors")?;

        // If flush was successful, mark the WAL file for deletion
        if flush_result.success {
            debug!(
                "✅ Successfully flushed {} vectors from WAL file {} - marking for deletion_info",
                flush_result.entries_flushed.unwrap_or(0),
                file_info.file_url
            );

            // Delete the WAL file since data is now safely in storage engine
            if let Err(e) = disk_manager.delete_wal_file_url(&file_info.file_url).await {
                warn!("Failed to delete WAL file after successful recovery: {}", e);
                // Continue - data is safely recovered even if cleanup fails
            }
        } else {
            warn!(
                "⚠️ Flush failed for WAL file {} - keeping file for retry",
                file_info.file_url
            );
        }

        Ok(vector_count)
    }

    fn storage_base_from_wal_url(file: &WalFileInfo) -> Result<String> {
        file.file_url
            .split(&format!("/{}/wal/", file.collection_id))
            .next()
            .filter(|base| !base.is_empty())
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("cannot derive storage base from {}", file.file_url))
    }

    async fn collection_has_segments(
        collection_id: &str,
        storage_url: &str,
        disk_manager: &Arc<WriteAheadLogDiskManager>,
    ) -> Result<bool> {
        let data_url = format!(
            "{}/{}/data/",
            storage_url.trim_end_matches('/'),
            collection_id
        );
        let filesystem = disk_manager
            .filesystem_factory()
            .get_filesystem(&data_url)?;
        let entries =
            WriteAheadLogDiskManager::list_prefix_entries(&*filesystem, &data_url).await?;
        Ok(entries.iter().any(|entry| {
            let name = entry.name.to_ascii_lowercase();
            name.ends_with(".pax") || name.ends_with(".sst") || name.ends_with(".arrow")
        }))
    }

    async fn flush_recovered_range(
        collection_id: &str,
        vectors: Vec<proximadb_records::ProximaRecord>,
        batch_ids: Vec<BatchId>,
        storage_engines: &Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        storage_url: &str,
        materialization_id: &str,
        content_digest: &str,
    ) -> Result<crate::storage::traits::FlushResult> {
        let engines = storage_engines.read().await;
        let engine = engines.get(collection_id).ok_or_else(|| {
            anyhow::anyhow!("No storage engine registered for collection {collection_id}")
        })?;
        let collection_config =
            if let Some(collection) = super::resolve_collection_from_catalog(collection_id).await {
                Some(collection)
            } else {
                Self::create_minimal_collection_config(collection_id, storage_url)
            };
        let mut hints = HashMap::new();
        hints.insert(
            "recovery_materialization_id".to_string(),
            serde_json::Value::String(materialization_id.to_string()),
        );
        hints.insert(
            "recovery_content_digest".to_string(),
            serde_json::Value::String(content_digest.to_string()),
        );
        hints.insert(
            "suppress_compaction_until_wal_retired".to_string(),
            serde_json::Value::Bool(true),
        );
        let params = crate::storage::traits::FlushParameters {
            collection_id: Some(collection_id.to_string()),
            collection_config,
            force: true,
            synchronous: true,
            vector_records: vectors,
            batch_ids,
            hints,
            trigger_compaction: false,
            ..Default::default()
        };
        engine.do_flush(&params).await
    }

    /// Flush recovered vectors to storage engine
    async fn flush_recovered_vectors(
        file_info: &WalFileInfo,
        vectors: Vec<proximadb_records::ProximaRecord>,
        _disk_manager: &Arc<WriteAheadLogDiskManager>,
        storage_engines: &Arc<tokio::sync::RwLock<HashMap<String, Arc<dyn UnifiedStorageFormat>>>>,
        _recovery_mode: RecoveryMode,
        storage_url: &str,
    ) -> Result<crate::storage::traits::FlushResult> {
        // Get the storage engine for this collection
        let engines = storage_engines.read().await;
        let engine = engines.get(&file_info.collection_id).ok_or_else(|| {
            anyhow::anyhow!(
                "No storage engine registered for collection {}",
                file_info.collection_id
            )
        })?;

        // The catalog is the sole read authority; resolve collection config from
        // it, otherwise fall back to a minimal config.
        let collection_config = if let Some(collection) =
            super::resolve_collection_from_catalog(&file_info.collection_id).await
        {
            info!(
                "✅ Using collection config from catalog for {}",
                file_info.collection_id
            );
            Some(collection)
        } else {
            warn!(
                "⚠️ Collection {} not found in catalog, using minimal config",
                file_info.collection_id
            );
            Self::create_minimal_collection_config(&file_info.collection_id, storage_url)
        };

        // Create flush parameters with REAL collection config
        let flush_params = crate::storage::traits::FlushParameters {
            collection_id: Some(file_info.collection_id.clone()),
            collection_config,
            force: true,
            synchronous: true,
            vector_records: vectors,
            batch_ids: vec![file_info.batch_id],
            ..Default::default()
        };

        // Execute flush using the storage engine's do_flush method
        let result = engine.do_flush(&flush_params).await?;

        Ok(result)
    }

    /// Create minimal collection config as fallback
    fn create_minimal_collection_config(
        collection_id: &str,
        storage_url: &str,
    ) -> Option<crate::proto::proximadb_v1::Collection> {
        Some(crate::proto::proximadb_v1::Collection {
            id: collection_id.to_string(),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                primary_path: storage_url.to_string(),
                backup_paths: vec![],
                engine: crate::proto::proximadb_v1::StorageEngine::Sst as i32,
                engine_config: std::collections::HashMap::new(),
                base_location: storage_url.to_string(),
                assigned_at: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Discover all collections by scanning the filesystem
    async fn discover_collections(&self) -> Result<Vec<String>> {
        debug!("Discovering collections from WAL directory");

        let mut collections = Vec::new();

        // Get the base WAL directory from disk_manager
        let base_wal_url = self.disk_manager.get_base_wal_url();

        // List directories within the base WAL URL
        let entries = self
            .disk_manager
            .filesystem_factory()
            .get_filesystem(base_wal_url)
            .context("Failed to get filesystem for base WAL URL")?
            .list(base_wal_url)
            .await
            .context("Failed to list base WAL URL")?;

        for entry in entries {
            if entry.metadata.is_directory {
                // Assume directories are collection IDs for now
                // Deferred: Add more robust validation for collection IDs
                collections.push(entry.name);
            }
        }

        info!("Discovered {} collections for recovery", collections.len());
        Ok(collections)
    }

    /// Get recovery statistics
    pub async fn get_stats(&self) -> Result<WalRecoveryStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }

    /// Clear recovery statistics
    pub async fn clear_stats(&self) -> Result<()> {
        let mut stats = self.stats.write().await;
        *stats = WalRecoveryStats::default();
        Ok(())
    }
}

/// Parallel recovery manager for improved performance
pub struct ParallelRecoveryManager {
    /// Base recovery manager
    recovery_manager: Arc<RecoveryManager>,
    /// Number of parallel workers
    num_workers: usize,
}

impl ParallelRecoveryManager {
    /// Create a new parallel recovery manager
    pub fn new(
        _disk_manager: Arc<WriteAheadLogDiskManager>,
        _flush_coordinator: Arc<WALFlushCoordinator>,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
        num_workers: Option<usize>,
    ) -> Self {
        // Create a default WAL config for recovery
        let config = crate::storage::persistence::write_ahead_log::config::WALConfig::default();

        // Create WAL behavior wrapper using MemtableConfig
        let memtable_config = crate::storage::memtable::MemtableConfig::default();
        let wal_behavior = Arc::new(
            crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper::new(
                memtable_config,
            ),
        );

        let recovery_manager = Arc::new(RecoveryManager::new(
            config,
            wal_behavior,
            filesystem_factory,
        ));
        let num_workers = num_workers.unwrap_or_else(|| num_cpus::get().min(8));

        info!(
            "🎯 Creating ParallelRecoveryManager with {} workers",
            num_workers
        );

        Self {
            recovery_manager,
            num_workers,
        }
    }

    /// Register storage engine for a collection
    pub async fn register_storage_engine(
        &self,
        collection_id: &str,
        engine: Arc<dyn UnifiedStorageFormat>,
    ) -> Result<()> {
        self.recovery_manager
            .register_storage_engine(collection_id, engine)
            .await
    }

    /// Recover all collections in parallel
    pub async fn recover_all_parallel(&self) -> Result<WalRecoveryStats> {
        info!(
            "🔄 Starting parallel WAL recovery with {} workers",
            self.num_workers
        );

        // The base recovery manager already does parallel recovery
        self.recovery_manager.recover_all().await
    }

    /// Recover a collection with parallel file processing
    pub async fn recover_collection_parallel(&self, collection_id: &str) -> Result<u64> {
        info!(
            "🔄 Starting parallel recovery for collection {} with {} workers",
            collection_id, self.num_workers
        );

        // Get all WAL files for the collection
        let wal_files = self
            .recovery_manager
            .disk_manager
            .list_collection_files(collection_id)
            .await?;

        if wal_files.is_empty() {
            debug!("No WAL files found for collection {}", collection_id);
            return Ok(0);
        }

        // Split files into chunks for parallel processing
        let chunk_size = wal_files.len().div_ceil(self.num_workers);
        let file_chunks: Vec<Vec<_>> = wal_files
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        info!(
            "🧵 Processing {} WAL files in {} parallel chunks",
            wal_files.len(),
            file_chunks.len()
        );

        // Create tasks for parallel file processing
        let mut recovery_tasks = Vec::new();

        for (chunk_idx, chunk) in file_chunks.into_iter().enumerate() {
            let recovery_manager = self.recovery_manager.clone();
            let collection_id_clone = collection_id.to_string();

            let task = tokio::spawn(async move {
                let mut chunk_vectors = 0u64;

                debug!(
                    "Worker {} processing {} files for collection {}",
                    chunk_idx,
                    chunk.len(),
                    collection_id_clone
                );

                for file_info in chunk {
                    match RecoveryManager::recover_file_internal(
                        &file_info,
                        &recovery_manager.disk_manager,
                        &recovery_manager.storage_engines,
                        recovery_manager.recovery_mode,
                    )
                    .await
                    {
                        Ok(count) => {
                            chunk_vectors += count;
                            debug!(
                                "Worker {} recovered {} vectors from batch {}",
                                chunk_idx,
                                count,
                                file_info.batch_id.to_base62()
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Worker {} failed to recover batch {}: {}",
                                chunk_idx,
                                file_info.batch_id.to_base62(),
                                e
                            );
                        }
                    }
                }

                debug!(
                    "Worker {} completed: recovered {} vectors",
                    chunk_idx, chunk_vectors
                );

                chunk_vectors
            });

            recovery_tasks.push(task);
        }

        // Wait for all workers to complete
        let results = futures::future::join_all(recovery_tasks).await;

        // Sum up results
        let mut total_vectors = 0u64;
        for (idx, result) in results.iter().enumerate() {
            match result {
                Ok(count) => {
                    total_vectors += count;
                    debug!("Worker {} recovered {} vectors", idx, count);
                }
                Err(e) => {
                    warn!("Worker {} panicked: {}", idx, e);
                }
            }
        }

        info!(
            "✅ Parallel recovery completed for collection {}: {} vectors recovered",
            collection_id, total_vectors
        );

        Ok(total_vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::persistence::write_ahead_log::{BatchId, SerializationFormat};
    use proximadb_records::{EmbeddingCell, ProximaRecord};
    use tempfile::TempDir;

    async fn create_test_managers() -> (
        Arc<WriteAheadLogDiskManager>,
        Arc<WALFlushCoordinator>,
        RecoveryManager,
        TempDir,
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create filesystem factory
        let filesystem_config = FilesystemConfig::default();
        let filesystem_factory = Arc::new(
            FilesystemFactory::create(filesystem_config)
                .await
                .expect("Failed to create filesystem factory"),
        );

        // Create disk manager
        let disk_manager = Arc::new(WriteAheadLogDiskManager::new(
            filesystem_factory.clone(),
            temp_dir.path().to_str().unwrap(),
        ));

        // Create flush coordinator
        let flush_coordinator = Arc::new(WALFlushCoordinator::new());

        // Create recovery manager with temp_dir path
        let mut config = crate::storage::persistence::write_ahead_log::config::WALConfig::default();
        config.multi_disk.data_directories = vec![temp_dir.path().to_str().unwrap().to_string()];
        let wal_behavior = Arc::new(
            crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper::new(
                crate::storage::memtable::MemtableConfig::default(),
            ),
        );
        let recovery_manager =
            RecoveryManager::new(config, wal_behavior, filesystem_factory.clone());

        (disk_manager, flush_coordinator, recovery_manager, temp_dir)
    }

    fn create_test_vector(id: &str) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                4,
                vec![0.1, 0.2, 0.3, 0.4],
            )],
            record_version: 1,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_recovery_manager_direct_to_storage() {
        let (disk_manager, _flush_coordinator, recovery_manager, temp_dir) =
            create_test_managers().await;
        let collection_id = "test_collection";

        // Create WAL directory structure for collection
        // Use collection_id directly to match the actual directory structure used by disk_manager
        let wal_dir = temp_dir.path().join(collection_id).join("wal");
        tokio::fs::create_dir_all(&wal_dir)
            .await
            .expect("Failed to create WAL directory");

        // Create a mock storage engine
        let storage_engine = create_mock_storage_engine();
        recovery_manager
            .register_storage_engine(collection_id, storage_engine.clone())
            .await
            .expect("Failed to register storage engine");

        // Write some test data
        use crate::storage::persistence::write_ahead_log::serialization::{
            ProtocolBuffersSerializer, VectorBatchSerializer,
        };
        let serializer = ProtocolBuffersSerializer::new();
        for i in 0..3 {
            let vector = create_test_vector(&format!("test{}", i));
            let batch = WALVectorBatch {
                batch_id: BatchId::new(),
                vector_records: Arc::new(vec![vector.clone()]),
                timestamp: std::time::SystemTime::now(),
                total_size_bytes: 256,
                is_flushed: false,
                metadata_bloom_filter: None,
            };
            let data = serializer
                .serialize_batch(std::slice::from_ref(&vector))
                .expect("Failed to serialize");
            disk_manager
                .write_batch(
                    collection_id,
                    &batch.batch_id,
                    &data,
                    SerializationFormat::ProtocolBuffers,
                    1,
                )
                .await
                .expect("Failed to write batch");
        }

        // Verify files were written
        let written_files = disk_manager
            .list_collection_files(collection_id)
            .await
            .expect("Failed to list files after writing");
        assert_eq!(written_files.len(), 3);

        // Recover the collection (should go to storage engine)
        let recovered = recovery_manager
            .recover_collection(collection_id)
            .await
            .expect("Failed to recover collection");
        assert_eq!(recovered.total_vectors_recovered, 3);

        // Verify WAL files were cleaned up
        let remaining_files = disk_manager
            .list_collection_files(collection_id)
            .await
            .expect("Failed to list files");
        assert_eq!(
            remaining_files.len(),
            0,
            "WAL files should be deleted after recovery"
        );

        // Check stats
        let stats = recovery_manager
            .get_stats()
            .await
            .expect("Failed to get stats");
        assert_eq!(stats.total_vectors_recovered, 3);
        assert_eq!(stats.total_files_recovered, 3);
    }

    /// Issue #1125 regression: candidate bases must union the catalog-assigned
    /// base, every manifest storage_url, and the configured primary base —
    /// deduplicated with trailing slashes normalized.
    #[test]
    fn wal_candidate_bases_unions_catalog_manifest_and_primary() {
        let bases = RecoveryManager::wal_candidate_bases(
            None,
            Some("file:///tmp/pdb/d2/"),
            &[
                "file:///tmp/pdb/d2".to_string(),
                "file:///tmp/pdb/d3".to_string(),
                String::new(),
            ],
            "file:///tmp/pdb/wal-primary",
        );
        assert_eq!(
            bases,
            vec![
                "file:///tmp/pdb/d2".to_string(),
                "file:///tmp/pdb/d3".to_string(),
                "file:///tmp/pdb/wal-primary".to_string(),
            ],
            "catalog base first, manifest bases next, primary last; dedup + slash-normalized"
        );

        // No catalog / no manifest degrades to the primary base only (pre-#1125 behavior).
        let primary_only =
            RecoveryManager::wal_candidate_bases(None, None, &[], "file:///tmp/pdb/wal-primary");
        assert_eq!(
            primary_only,
            vec!["file:///tmp/pdb/wal-primary".to_string()]
        );
    }

    /// TD-WAL-5 regression: a config-file `wal_local_dir` (no env) must reach
    /// replay. The recovery manager resolves it with the write path's own
    /// resolver and lists it FIRST, ahead of catalog/manifest/primary bases.
    #[tokio::test]
    async fn config_file_wal_local_dir_is_a_replay_candidate() {
        if std::env::var("PROXIMADB_WAL_LOCAL_DIR").is_ok() {
            return; // env override would win; this test pins the TOML-only path
        }
        let mut config = crate::storage::persistence::write_ahead_log::config::WALConfig::default();
        config.wal_local_dir = Some("file:///waldisk/".to_string());
        let wal_behavior = Arc::new(
            crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper::new(
                crate::storage::memtable::MemtableConfig::default(),
            ),
        );
        let filesystem = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .expect("filesystem factory"),
        );
        let manager = RecoveryManager::new(config, wal_behavior, filesystem);
        assert_eq!(manager.local_wal_dir.as_deref(), Some("file:///waldisk"));

        let bases = RecoveryManager::wal_candidate_bases(
            manager.local_wal_dir.as_deref(),
            Some("file:///tmp/pdb/d2"),
            &[],
            "file:///tmp/pdb/wal-primary",
        );
        assert_eq!(
            bases,
            vec![
                "file:///waldisk".to_string(),
                "file:///tmp/pdb/d2".to_string(),
                "file:///tmp/pdb/wal-primary".to_string(),
            ]
        );
    }

    /// Issue #1125 regression: a WAL batch written under a NON-primary base
    /// (the collection's assigned storage location — e.g. `d2` of a multi-disk
    /// config) must be found by the cross-base listing. The single-base listing
    /// this replaces returned zero files for exactly this layout, so an
    /// acknowledged fsync'd batch was silently dropped on restart.
    #[tokio::test]
    async fn recovery_lists_wal_written_under_non_primary_base() {
        let (primary_disk_manager, _flush_coordinator, _recovery_manager, primary_dir) =
            create_test_managers().await;
        let collection_id = "c1125";

        // Simulate the write path: the collection's WAL lands under its
        // assigned base_location (a different directory from the recovery
        // manager's configured base).
        let assigned_dir = TempDir::new().expect("Failed to create assigned dir");
        let assigned_base = assigned_dir.path().to_str().unwrap().to_string();
        let write_disk_manager = Arc::new(WriteAheadLogDiskManager::new(
            primary_disk_manager.filesystem_factory().clone(),
            assigned_base.clone(),
        ));

        use crate::storage::persistence::write_ahead_log::serialization::{
            ProtocolBuffersSerializer, VectorBatchSerializer,
        };
        let serializer = ProtocolBuffersSerializer::new();
        let vector = create_test_vector("r1125");
        let data = serializer
            .serialize_batch(std::slice::from_ref(&vector))
            .expect("Failed to serialize");
        write_disk_manager
            .write_batch(
                collection_id,
                &BatchId::new(),
                &data,
                SerializationFormat::ProtocolBuffers,
                1,
            )
            .await
            .expect("Failed to write batch");

        // Old behavior (primary base only): the batch is invisible.
        let primary_only = RecoveryManager::list_wal_files_across_bases(
            &primary_disk_manager,
            &RecoveryManager::wal_candidate_bases(
                None,
                None,
                &[],
                primary_disk_manager.get_base_wal_url(),
            ),
            collection_id,
        )
        .await
        .expect("primary-only listing failed");
        assert!(
            primary_only.is_empty(),
            "sanity: the batch does NOT live under the primary base"
        );

        // Fixed behavior: the assigned base (as the catalog/manifest would
        // report it) is included, so the batch is found exactly once.
        let unioned = RecoveryManager::list_wal_files_across_bases(
            &primary_disk_manager,
            &RecoveryManager::wal_candidate_bases(
                None,
                Some(&assigned_base),
                &[assigned_base.clone()], // manifest repeats it — must dedupe
                primary_disk_manager.get_base_wal_url(),
            ),
            collection_id,
        )
        .await
        .expect("cross-base listing failed");
        assert_eq!(
            unioned.len(),
            1,
            "the non-primary-base WAL batch must be listed exactly once (deduped)"
        );
        assert_eq!(unioned[0].collection_id, collection_id);
        drop(primary_dir);
    }

    // Mock storage engine for testing
    fn create_mock_storage_engine() -> Arc<dyn UnifiedStorageFormat> {
        use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
        use crate::storage::traits::{
            CompactionParameters, CompactionResult, FlushParameters, FlushResult,
            StorageFormatStrategy, UnifiedStorageFormat,
        };
        use async_trait::async_trait;
        use std::collections::HashMap;

        struct MockStorageEngine {
            vectors_received: Arc<tokio::sync::Mutex<Vec<proximadb_records::ProximaRecord>>>,
            filesystem_factory: FilesystemFactory,
        }

        #[async_trait]
        impl UnifiedStorageFormat for MockStorageEngine {
            fn engine_name(&self) -> &'static str {
                "MockEngine"
            }

            fn engine_version(&self) -> &'static str {
                "1.0.0"
            }

            fn strategy(&self) -> StorageFormatStrategy {
                StorageFormatStrategy::Sst
            }

            async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
                // Store the vectors we receive during recovery
                let mut vectors = self.vectors_received.lock().await;
                vectors.extend(params.vector_records.clone());

                Ok(FlushResult {
                    success: true,
                    collections_affected: vec![params.collection_id.clone().unwrap_or_default()],
                    entries_flushed: Some(params.vector_records.len() as u64),
                    bytes_written: Some(params.vector_records.len() as u64 * 256),
                    files_created: Some(1),
                    file_paths: vec![],
                    duration_ms: Some(10),
                    completed_at: chrono::Utc::now(),
                    engine_metrics: HashMap::new(),
                    compaction_triggered: false,
                    compaction_error: None,
                    flushed_batch_ids: params.batch_ids.clone(),
                })
            }

            async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
                Ok(CompactionResult {
                    success: true,
                    collections_affected: vec![params.collection_id.clone().unwrap_or_default()],
                    entries_processed: Some(0),
                    entries_removed: Some(0),
                    bytes_read: Some(0),
                    bytes_written: Some(0),
                    input_files: Some(0),
                    output_files: Some(0),
                    duration_ms: Some(10),
                    completed_at: chrono::Utc::now(),
                    engine_metrics: HashMap::new(),
                })
            }

            async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
                Ok(HashMap::new())
            }

            async fn vector_by_id(
                &self,
                _collection_id: &str,
                _base_path: &str,
                _vector_id: &str,
            ) -> Result<Option<proximadb_records::ProximaRecord>> {
                Ok(None)
            }

            async fn search_vectors_unified(
                &self,
                _query_context: &crate::storage::traits::StorageQueryContext,
            ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
                Ok(Vec::new())
            }
        }

        // Create a filesystem factory for the mock
        let filesystem_factory = futures::executor::block_on(async {
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .expect("Failed to create filesystem factory")
        });

        Arc::new(MockStorageEngine {
            vectors_received: Arc::new(tokio::sync::Mutex::new(Vec::<
                proximadb_records::ProximaRecord,
            >::new())),
            filesystem_factory,
        })
    }
}
