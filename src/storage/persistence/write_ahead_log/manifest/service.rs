//! Global Manifest Service
//!
//! High-performance, thread-safe service for managing the global WAL manifest.
//!
//! Architecture:
//! - Centralized service (singleton) with async write-behind queue
//! - Lock-free append operations via channels
//! - Batched disk writes for performance
//! - Atomic manifest updates with double-buffering
//! - Crash recovery via write-ahead manifest staging
//!
//! Performance characteristics:
//! - O(1) append via channel send
//! - Batched disk writes (configurable interval)
//! - Zero contention between collections
//! - Scales to 1000s of concurrent collections

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, trace, warn};

use super::types::{
    CheckpointCollectionState, GlobalCheckpoint, GlobalLsnAllocator, GlobalManifestEntry,
    WalEntryStatus,
};

/// Configuration for the global manifest service
#[derive(Debug, Clone)]
pub struct GlobalManifestServiceConfig {
    /// Batch writes every N milliseconds
    pub batch_interval_ms: u64,

    /// Maximum batch size before forcing a write
    pub max_batch_size: usize,

    /// Channel buffer size for append requests
    pub channel_buffer_size: usize,
}

impl Default for GlobalManifestServiceConfig {
    fn default() -> Self {
        Self {
            batch_interval_ms: 100,     // Write every 100ms
            max_batch_size: 1000,       // Or when 1000 entries accumulated
            channel_buffer_size: 10000, // Can buffer 10k pending entries
        }
    }
}

/// Request sent to the background manifest worker.
///
/// Issue #1126 (L1): the worker batches appends (write-behind), so readers of
/// `entries` could race in-flight appends — a shutdown flush inside the
/// batching window under-marked the freshest batch, leaving live WAL files
/// that the next boot re-materialized as duplicate `L0_recovery_*` segments.
/// `Barrier` lets a caller force the worker to drain its pending batch and
/// ack, making "append happened-before lookup" enforceable.
#[derive(Debug)]
enum AppendRequest {
    Append {
        entry: GlobalManifestEntry,
        /// Optional response channel for synchronous append
        response: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    },
    /// Flush the worker's pending batch, then ack. Never enters the batch.
    Barrier(tokio::sync::oneshot::Sender<()>),
}

/// Global manifest service - centralized singleton for all collections
pub struct GlobalManifestService {
    /// Configuration
    config: GlobalManifestServiceConfig,

    /// Filesystem factory for I/O
    filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,

    /// Base WAL directory
    wal_base_url: String,

    /// In-memory manifest entries (sorted by LSN)
    entries: Arc<RwLock<Vec<GlobalManifestEntry>>>,

    /// Latest checkpoint
    latest_checkpoint: Arc<RwLock<Option<GlobalCheckpoint>>>,

    /// Global LSN allocator
    lsn_allocator: Arc<GlobalLsnAllocator>,

    /// Append request channel (sender half)
    append_tx: mpsc::Sender<AppendRequest>,

    /// Background worker handle
    worker_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl GlobalManifestService {
    /// Create a new global manifest service
    pub async fn new(
        config: GlobalManifestServiceConfig,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
        wal_base_url: String,
    ) -> Result<Arc<Self>> {
        info!("🌐 Initializing GlobalManifestService at {}", wal_base_url);

        // Create append channel
        let (append_tx, append_rx) = mpsc::channel(config.channel_buffer_size);

        let service = Arc::new(Self {
            config: config.clone(),
            filesystem_factory: filesystem_factory.clone(),
            wal_base_url: wal_base_url.clone(),
            entries: Arc::new(RwLock::new(Vec::new())),
            latest_checkpoint: Arc::new(RwLock::new(None)),
            lsn_allocator: Arc::new(GlobalLsnAllocator::new(1)),
            append_tx,
            worker_handle: Arc::new(Mutex::new(None)),
        });

        // Load existing manifest from disk
        service.load_from_disk().await?;

        // Start background worker
        service.start_background_worker(append_rx).await;

        info!(
            "✅ GlobalManifestService started with {} existing entries",
            service.entries.read().await.len()
        );

        Ok(service)
    }

    /// Start the background worker for batched writes
    async fn start_background_worker(
        self: &Arc<Self>,
        mut append_rx: mpsc::Receiver<AppendRequest>,
    ) {
        let service = Arc::clone(self);
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            info!("🔄 GlobalManifest background worker started");

            let mut write_interval = interval(Duration::from_millis(config.batch_interval_ms));
            let mut pending_batch: Vec<AppendRequest> = Vec::new();

            loop {
                tokio::select! {
                    // Receive append requests
                    Some(request) = append_rx.recv() => {
                        match request {
                            AppendRequest::Barrier(ack) => {
                                // Issue #1126 L1: drain everything queued ahead of
                                // the barrier so post-barrier readers see it all.
                                if !pending_batch.is_empty()
                                    && let Err(e) = service.flush_pending_batch(&mut pending_batch).await {
                                        error!("❌ Failed to flush manifest batch at barrier: {}", e);
                                    }
                                let _ = ack.send(());
                            }
                            append @ AppendRequest::Append { .. } => {
                                pending_batch.push(append);

                                // Force write if batch is full
                                if pending_batch.len() >= config.max_batch_size
                                    && let Err(e) = service.flush_pending_batch(&mut pending_batch).await {
                                        error!("❌ Failed to flush manifest batch: {}", e);
                                    }
                            }
                        }
                    }

                    // Periodic batch write
                    _ = write_interval.tick() => {
                        if !pending_batch.is_empty()
                            && let Err(e) = service.flush_pending_batch(&mut pending_batch).await {
                                error!("❌ Failed to flush manifest batch: {}", e);
                            }
                    }

                    // Channel closed
                    else => {
                        info!("📛 GlobalManifest worker shutting down");
                        // Flush any remaining entries
                        if !pending_batch.is_empty() {
                            let _ = service.flush_pending_batch(&mut pending_batch).await;
                        }
                        break;
                    }
                }
            }

            info!("✅ GlobalManifest background worker stopped");
        });

        *self.worker_handle.lock().await = Some(handle);
    }

    /// Flush pending batch to disk (called by background worker)
    async fn flush_pending_batch(&self, batch: &mut Vec<AppendRequest>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        debug!("💾 Flushing {} manifest entries to disk", batch.len());

        // Extract entries and sort by LSN (the batch only ever holds Append
        // requests — Barriers are handled inline by the worker loop).
        let mut entries_to_write: Vec<GlobalManifestEntry> = batch
            .iter()
            .filter_map(|req| match req {
                AppendRequest::Append { entry, .. } => Some(entry.clone()),
                AppendRequest::Barrier(_) => None,
            })
            .collect();
        entries_to_write.sort_by_key(|e| e.global_lsn);

        // Write to staging file first (crash safety)
        let staging_result = self.write_to_staging(&entries_to_write).await;

        // Update in-memory manifest
        {
            let mut entries = self.entries.write().await;
            entries.extend(entries_to_write.clone());
            entries.sort_by_key(|e| e.global_lsn);
        }

        // Staging write is the final write (no promotion needed with sequential files)
        let write_result = staging_result;

        // Send responses to synchronous callers
        for request in batch.drain(..) {
            match request {
                AppendRequest::Append {
                    response: Some(tx), ..
                } => {
                    // Convert Result to cloneable form
                    let response = match &write_result {
                        Ok(()) => Ok(()),
                        Err(e) => Err(anyhow::anyhow!("{}", e)),
                    };
                    let _ = tx.send(response);
                }
                AppendRequest::Append { response: None, .. } => {}
                AppendRequest::Barrier(ack) => {
                    let _ = ack.send(());
                }
            }
        }

        write_result
    }

    /// Write entries as sequential LSN-based files
    ///
    /// Unified approach for all storage (local, S3, Azure, GCS):
    /// Creates immutable sequential files: manifest_{min_lsn}_{max_lsn}.jsonl
    ///
    /// Benefits:
    /// - Works efficiently on ALL storage types (no read-modify-write)
    /// - Immutable files (easier to cache, replicate, backup)
    /// - Parallel writes possible (different LSN ranges)
    /// - Clean deletion of old segments after checkpoint
    async fn write_to_staging(&self, new_entries: &[GlobalManifestEntry]) -> Result<()> {
        let fs = self.filesystem_factory.get_filesystem(&self.wal_base_url)?;

        // Sequential LSN-based file naming (works for all storage types)
        // Format: manifest_{min_lsn}_{max_lsn}.jsonl
        let min_lsn = new_entries.iter().map(|e| e.global_lsn).min().unwrap_or(0);
        let max_lsn = new_entries.iter().map(|e| e.global_lsn).max().unwrap_or(0);
        let segment_url = format!(
            "{}/manifest_{:020}_{:020}.jsonl",
            self.wal_base_url, min_lsn, max_lsn
        );

        // Serialize entries
        let mut content = Vec::new();
        for entry in new_entries {
            let mut line =
                serde_json::to_vec(entry).context("Failed to serialize manifest entry")?;
            line.push(b'\n');
            content.extend_from_slice(&line);
        }

        // Check if file already exists (status update for existing entry)
        let file_exists = fs.exists(&segment_url).await.unwrap_or(false);

        if file_exists {
            // File exists - this is a status update
            // Append to existing file instead of overwriting
            debug!(
                "📝 Appending status update to existing manifest segment: {}",
                segment_url
            );

            // Read existing content
            let existing_content = fs
                .read(&segment_url)
                .await
                .context("Failed to read existing manifest segment")?;

            // Append new entries
            let mut combined_content = existing_content;
            combined_content.extend_from_slice(&content);

            // Write combined content (overwrite with appended data)
            let strategy = crate::storage::persistence::filesystem::write_strategy::WriteStrategyFactory
                ::create_metadata_strategy(&*fs, None)?;
            let opts = strategy.create_file_options(&*fs, &segment_url)?;
            fs.write(&segment_url, &combined_content, Some(opts))
                .await
                .context("Failed to write manifest segment with status update")?;

            debug!(
                "💾 Appended {} status updates to manifest segment: {} (LSN {}-{})",
                new_entries.len(),
                segment_url,
                min_lsn,
                max_lsn
            );
        } else {
            // New file - write normally
            let strategy = crate::storage::persistence::filesystem::write_strategy::WriteStrategyFactory
                ::create_metadata_strategy(&*fs, None)?;
            let opts = strategy.create_file_options(&*fs, &segment_url)?;
            fs.write(&segment_url, &content, Some(opts))
                .await
                .context("Failed to write manifest segment")?;

            debug!(
                "💾 Wrote new manifest segment: {} ({} entries, LSN {}-{})",
                segment_url,
                new_entries.len(),
                min_lsn,
                max_lsn
            );
        }

        // Best-effort sync (no-op for cloud storage)
        let _ = fs.sync_file(&segment_url).await;

        Ok(())
    }

    /// Get the global manifest URL
    fn global_manifest_url(&self) -> String {
        format!("{}/global_manifest.log", self.wal_base_url)
    }

    /// Load manifest from disk
    ///
    /// Unified approach: Reads all manifest_*.jsonl files (sorted by LSN)
    /// Works consistently for local, S3, Azure, GCS storage
    async fn load_from_disk(&self) -> Result<()> {
        trace!("🔍 : Loading manifest from: {}", self.wal_base_url);
        let fs = self.filesystem_factory.get_filesystem(&self.wal_base_url)?;
        trace!("🔍 : Got filesystem for manifest");

        // List all manifest segment files: manifest_{min_lsn}_{max_lsn}.jsonl
        let dir_entries = fs.list(&self.wal_base_url).await.unwrap_or_default();
        trace!("🔍 : Listed {} directory entries", dir_entries.len());

        let mut manifest_files: Vec<String> = dir_entries
            .into_iter()
            .filter(|entry| {
                let matches = entry.name.contains("manifest_") && entry.name.ends_with(".jsonl");
                if matches {
                    trace!("🔍 : Found manifest file: {}", entry.name);
                }
                matches
            })
            .map(|entry| entry.url)
            .collect();
        manifest_files.sort(); // Lexicographic sort works due to zero-padded LSN

        if manifest_files.is_empty() {
            info!("📝 No existing manifest segments found, starting fresh");
            return Ok(());
        }

        info!("📂 Loading {} manifest segments", manifest_files.len());

        let mut entries = Vec::new();
        let mut max_lsn = 0u64;

        for file_url in &manifest_files {
            let data = match fs.read(file_url).await {
                Ok(d) => d,
                Err(e) => {
                    warn!("⚠️  Failed to read manifest segment {}: {}", file_url, e);
                    continue;
                }
            };

            for line in data.split(|b| *b == b'\n') {
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_slice::<GlobalManifestEntry>(line) {
                    Ok(entry) => {
                        max_lsn = max_lsn.max(entry.global_lsn);
                        entries.push(entry);
                    }
                    Err(e) => {
                        warn!("⚠️  Failed to parse manifest entry: {}", e);
                    }
                }
            }
        }

        // Sort by global LSN
        entries.sort_by_key(|e| e.global_lsn);

        // Deduplicate: For same batch_id, keep entry with highest LSN (latest status)
        let mut deduped: std::collections::HashMap<String, GlobalManifestEntry> =
            std::collections::HashMap::new();
        for entry in entries {
            deduped
                .entry(entry.batch_id.clone())
                .and_modify(|existing| {
                    if should_replace_manifest_entry(existing, &entry) {
                        *existing = entry.clone();
                    }
                })
                .or_insert(entry);
        }

        let final_entries: Vec<GlobalManifestEntry> = deduped.into_values().collect();
        let mut sorted_entries = final_entries;
        sorted_entries.sort_by_key(|e| e.global_lsn);

        info!(
            "✅ Loaded {} unique manifest entries (after deduplication)",
            sorted_entries.len()
        );

        *self.entries.write().await = sorted_entries;

        // Set next LSN
        self.lsn_allocator.set_next(max_lsn + 1).await;

        info!("✅ Next LSN: {}", max_lsn + 1);

        // Load checkpoint
        self.load_checkpoint().await?;

        // Reaped entries must STAY reaped across restarts. Segment cleanup
        // deletes only segments whose max LSN is below the watermark, but a
        // surviving status-update segment (written at a later LSN) can still
        // carry records for entries the reaper removed — which resurrected
        // them here as Flushed. Apply the reaper's exact retention predicate
        // at load so the rewritten manifest + checkpoint stay authoritative.
        if let Some(checkpoint) = self.latest_checkpoint.read().await.clone() {
            let mut entries = self.entries.write().await;
            let before = entries.len();
            entries.retain(|e| {
                e.global_lsn >= checkpoint.safe_to_delete_before_lsn
                    || e.status == WalEntryStatus::Active
            });
            let dropped = before - entries.len();
            if dropped > 0 {
                info!(
                    "🧹 Dropped {} pre-checkpoint entries resurrected by surviving segments",
                    dropped
                );
            }
        }

        Ok(())
    }

    /// Load checkpoint from disk
    async fn load_checkpoint(&self) -> Result<()> {
        let url = format!("{}/checkpoint.state", self.wal_base_url);
        let fs = self.filesystem_factory.get_filesystem(&url)?;

        if !fs.exists(&url).await? {
            debug!("No existing checkpoint found");
            return Ok(());
        }

        let data = fs.read(&url).await.context("Failed to read checkpoint")?;

        let checkpoint: GlobalCheckpoint =
            serde_json::from_slice(&data).context("Failed to parse checkpoint")?;

        *self.latest_checkpoint.write().await = Some(checkpoint.clone());

        info!(
            "✅ Loaded checkpoint {} at LSN {}",
            checkpoint.checkpoint_id, checkpoint.checkpoint_lsn
        );

        Ok(())
    }

    /// Append an entry asynchronously (the LSN is allocated synchronously; the
    /// entry is persisted by the background worker). Returns the allocated durable
    /// `global_lsn` — TD-DELVEC-1 WI-5 P0 keys deletion-vector bits on this LSN so
    /// the DV generation shares the read side's `snapshot_lsn` number space
    /// (correct MVCC) and is recoverable.
    pub async fn append_async(&self, mut entry: GlobalManifestEntry) -> Result<u64> {
        // Allocate global LSN
        entry.global_lsn = self.lsn_allocator.allocate().await;
        let global_lsn = entry.global_lsn;

        // Send to background worker (non-blocking)
        self.append_tx
            .send(AppendRequest::Append {
                entry,
                response: None,
            })
            .await
            .context("Failed to send append request")?;

        Ok(global_lsn)
    }

    /// Append an entry synchronously (waits for disk write)
    pub async fn append_sync(&self, mut entry: GlobalManifestEntry) -> Result<()> {
        // Allocate global LSN
        entry.global_lsn = self.lsn_allocator.allocate().await;

        // Create response channel
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Send to background worker
        self.append_tx
            .send(AppendRequest::Append {
                entry,
                response: Some(tx),
            })
            .await
            .context("Failed to send append request")?;

        // Wait for response
        rx.await.context("Failed to receive append response")?
    }

    /// Get all entries for a collection
    pub async fn get_collection_entries(&self, collection_id: &str) -> Vec<GlobalManifestEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.collection_id == collection_id)
            .cloned()
            .collect()
    }

    /// Get all active entries
    pub async fn get_active_entries(&self) -> Vec<GlobalManifestEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.status == WalEntryStatus::Active)
            .cloned()
            .collect()
    }

    /// Get all entries sorted by LSN
    pub async fn get_all_entries(&self) -> Vec<GlobalManifestEntry> {
        self.entries.read().await.clone()
    }

    /// Get the LSN allocator
    pub fn lsn_allocator(&self) -> Arc<GlobalLsnAllocator> {
        self.lsn_allocator.clone()
    }

    /// Get the latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Option<GlobalCheckpoint> {
        self.latest_checkpoint.read().await.clone()
    }

    /// Issue #1126 L1: drain the worker's write-behind queue so every append
    /// enqueued before this call is visible in `entries`. Errors are swallowed:
    /// a gone worker means the channel is closed and nothing is pending.
    pub async fn drain_pending(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self
            .append_tx
            .send(AppendRequest::Barrier(tx))
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Mark entries as flushed
    ///
    /// Cloud-optimized: Writes status update as new manifest segment
    /// instead of editing existing files (append-only for S3/Azure/GCS)
    pub async fn mark_flushed(&self, batch_ids: &[String]) -> Result<()> {
        // Issue #1126 L1: the append path is write-behind; without this
        // barrier a flush racing the batching window under-marks the
        // freshest batch → leftover WAL → duplicate L0_recovery segments
        // on the next boot.
        self.drain_pending().await;

        // Batch-id membership as a set (was: a linear scan per manifest
        // entry on every flush acknowledgement).
        let batch_id_set: std::collections::HashSet<&String> = batch_ids.iter().collect();

        // Update in-memory state
        let mut entries = self.entries.write().await;
        let mut status_updates = Vec::new();

        for entry in entries.iter_mut() {
            if batch_id_set.contains(&entry.batch_id) && entry.status == WalEntryStatus::Active {
                entry.status = WalEntryStatus::Flushed;

                // Create status update entry for append-only cloud storage
                let mut update_entry = entry.clone();
                update_entry.status = WalEntryStatus::Flushed;
                status_updates.push(update_entry);
            }
        }

        drop(entries);

        if !status_updates.is_empty() {
            // Write status updates as new manifest segment (append-only)
            // This works efficiently on cloud storage without read-modify-write
            self.write_to_staging(&status_updates).await?;

            info!(
                "✅ Marked {} entries as Flushed (wrote status update segment)",
                status_updates.len()
            );
        }

        Ok(())
    }

    /// Mark entries as flushed AND delete the actual WAL files from disk
    ///
    /// This is the safe pattern: mark as flushed first, then delete files.
    /// If deletion fails, the entry is already marked as flushed so we won't
    /// try to recover it again on restart.
    pub async fn mark_flushed_and_delete_files(&self, batch_ids: &[String]) -> Result<usize> {
        if batch_ids.is_empty() {
            return Ok(0);
        }

        // Issue #1126 L1: drain the write-behind queue BEFORE the lookup —
        // otherwise a batch appended moments ago is invisible here, its WAL
        // file survives, and the next boot re-materializes it as a duplicate
        // L0_recovery segment.
        self.drain_pending().await;

        // Batch-id membership as a set (was: a linear scan per manifest entry).
        let batch_id_set: std::collections::HashSet<&String> = batch_ids.iter().collect();

        // Collect file URLs before marking as flushed (need Active status entries)
        let file_urls: Vec<String> = {
            let entries = self.entries.read().await;
            entries
                .iter()
                .filter(|e| {
                    batch_id_set.contains(&e.batch_id) && e.status == WalEntryStatus::Active
                })
                .map(|e| {
                    // Construct full file URL from storage_url + file_path
                    format!("{}/{}", e.storage_url.trim_end_matches('/'), e.file_path)
                })
                .collect()
        };

        if file_urls.is_empty() {
            debug!(
                "🔍 No active WAL files found for batch IDs: {:?}",
                batch_ids
            );
            return Ok(0);
        }

        // Mark as flushed first (CRITICAL: must happen before file deletion)
        self.mark_flushed(batch_ids).await?;

        // Now delete the actual WAL files
        let mut deleted_count = 0;
        for file_url in &file_urls {
            match self.filesystem_factory.get_filesystem(file_url) {
                Ok(fs) => {
                    match fs.delete(file_url).await {
                        Ok(_) => {
                            debug!("🗑️ Deleted WAL file: {}", file_url);
                            deleted_count += 1;
                        }
                        Err(e) => {
                            // File deletion failed, but entry is already marked as flushed
                            // This is safe - file will be orphaned but won't be recovered
                            warn!(
                                "⚠️ Failed to delete WAL file {} (already marked flushed): {}",
                                file_url, e
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to get filesystem for WAL file {}: {}",
                        file_url, e
                    );
                }
            }
        }

        info!(
            "🧹 Deleted {}/{} WAL files after flush (batch IDs: {})",
            deleted_count,
            file_urls.len(),
            batch_ids.len()
        );

        Ok(deleted_count)
    }

    /// Rewrite the entire manifest (used after status updates).
    /// Uses write_atomic (temp file + rename) to prevent corruption if the process
    /// crashes mid-write. A direct truncate+write would leave an empty or partial file.
    async fn rewrite_manifest(&self) -> Result<()> {
        let entries = self.entries.read().await;

        let url = self.global_manifest_url();
        let fs = self.filesystem_factory.get_filesystem(&url)?;

        let mut buf = Vec::new();
        for entry in entries.iter() {
            let mut line =
                serde_json::to_vec(entry).context("Failed to serialize manifest entry")?;
            line.push(b'\n');
            buf.extend_from_slice(&line);
        }

        let strategy = crate::storage::persistence::filesystem::write_strategy::WriteStrategyFactory
            ::create_metadata_strategy(&*fs, None)?;
        let opts = strategy.create_file_options(&*fs, &url)?;
        fs.write_atomic(&url, &buf, Some(opts))
            .await
            .context("Failed to rewrite global manifest")?;

        let _ = fs.sync_file(&url).await;

        info!("✅ Rewrote global manifest with {} entries", entries.len());

        Ok(())
    }

    /// Create a new checkpoint
    pub async fn create_checkpoint(&self) -> Result<GlobalCheckpoint> {
        let entries = self.entries.read().await;

        // Get the latest checkpoint ID
        let checkpoint_id = {
            let latest = self.latest_checkpoint.read().await;
            latest.as_ref().map_or(1, |c| c.checkpoint_id + 1)
        };

        // Find the highest flushed LSN
        let checkpoint_lsn = entries
            .iter()
            .filter(|e| e.status == WalEntryStatus::Flushed)
            .map(|e| e.global_lsn)
            .max()
            .unwrap_or(0);

        // Group by collection
        let mut collection_map: std::collections::HashMap<String, CheckpointCollectionState> =
            std::collections::HashMap::new();
        for entry in entries.iter() {
            if entry.status == WalEntryStatus::Flushed && entry.global_lsn <= checkpoint_lsn {
                collection_map
                    .entry(entry.collection_id.clone())
                    .and_modify(|state| {
                        state.last_flushed_lsn = state.last_flushed_lsn.max(entry.global_lsn);
                        state.vector_count += entry.vector_count;
                    })
                    .or_insert(CheckpointCollectionState {
                        collection_id: entry.collection_id.clone(),
                        last_flushed_lsn: entry.global_lsn,
                        vector_count: entry.vector_count,
                    });
            }
        }

        let checkpoint = GlobalCheckpoint {
            checkpoint_id,
            checkpoint_lsn,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis() as u64)
                .unwrap_or(0),
            collections: collection_map.into_values().collect(),
            safe_to_delete_before_lsn: checkpoint_lsn,
        };

        drop(entries);

        // Save checkpoint to disk
        self.save_checkpoint(&checkpoint).await?;

        // Update in-memory checkpoint
        *self.latest_checkpoint.write().await = Some(checkpoint.clone());

        info!(
            "✅ Created checkpoint {} at LSN {}",
            checkpoint.checkpoint_id, checkpoint.checkpoint_lsn
        );

        Ok(checkpoint)
    }

    /// Save checkpoint to disk
    async fn save_checkpoint(&self, checkpoint: &GlobalCheckpoint) -> Result<()> {
        let url = format!("{}/checkpoint.state", self.wal_base_url);
        let fs = self.filesystem_factory.get_filesystem(&url)?;

        let data =
            serde_json::to_vec_pretty(checkpoint).context("Failed to serialize checkpoint")?;

        let strategy = crate::storage::persistence::filesystem::write_strategy::WriteStrategyFactory
            ::create_metadata_strategy(&*fs, None)?;
        let opts = strategy.create_file_options(&*fs, &url)?;
        fs.write(&url, &data, Some(opts))
            .await
            .context("Failed to write checkpoint")?;

        let _ = fs.sync_file(&url).await;

        Ok(())
    }

    /// Clean up old WAL entries that have been checkpointed
    pub async fn cleanup_checkpointed_entries(&self) -> Result<usize> {
        let checkpoint = match self.latest_checkpoint.read().await.clone() {
            Some(cp) => cp,
            None => {
                debug!("No checkpoint exists, skipping cleanup");
                return Ok(0);
            }
        };

        let mut entries = self.entries.write().await;
        let original_count = entries.len();

        // Remove entries that are checkpointed and can be safely deleted
        entries.retain(|e| {
            e.global_lsn >= checkpoint.safe_to_delete_before_lsn
                || e.status == WalEntryStatus::Active
        });

        let removed_count = original_count - entries.len();
        drop(entries);

        if removed_count > 0 {
            self.rewrite_manifest().await?;

            // Cloud storage: Delete old manifest segment files
            self.cleanup_old_manifest_segments(checkpoint.safe_to_delete_before_lsn)
                .await?;

            info!("🧹 Cleaned up {} checkpointed WAL entries", removed_count);
        }

        Ok(removed_count)
    }

    /// Cleanup old manifest segment files
    /// Deletes segments where max_lsn < safe_to_delete_before_lsn
    async fn cleanup_old_manifest_segments(&self, safe_to_delete_before_lsn: u64) -> Result<()> {
        let fs = self.filesystem_factory.get_filesystem(&self.wal_base_url)?;

        // List all manifest segment files
        let dir_entries = fs.list(&self.wal_base_url).await.unwrap_or_default();
        let mut deleted_count = 0;

        for entry in dir_entries {
            if !entry.name.contains("manifest_") || !entry.name.ends_with(".jsonl") {
                continue;
            }

            // Parse max_lsn from filename: manifest_{min_lsn}_{max_lsn}.jsonl
            if let Some(filename) = entry.url.split('/').next_back()
                && let Some(max_lsn_str) = filename
                    .strip_prefix("manifest_")
                    .and_then(|s| s.strip_suffix(".jsonl"))
                    .and_then(|s| s.split('_').nth(1))
                && let Ok(max_lsn) = max_lsn_str.parse::<u64>()
                && max_lsn < safe_to_delete_before_lsn
            {
                // Delete this segment
                match fs.delete(&entry.url).await {
                    Ok(_) => {
                        debug!("🗑️  Deleted old manifest segment: {}", entry.url);
                        deleted_count += 1;
                    }
                    Err(e) => {
                        warn!("⚠️  Failed to delete manifest segment {}: {}", entry.url, e);
                    }
                }
            }
        }

        if deleted_count > 0 {
            info!("🧹 Deleted {} old manifest segments", deleted_count);
        }

        Ok(())
    }

    // ========================================================================
    // PITR (Point-in-Time Recovery) Support Methods
    // ========================================================================

    /// Get the current LSN (last allocated LSN, or 0 if none allocated)
    pub async fn current_lsn(&self) -> u64 {
        let next = self.lsn_allocator.current().await;
        next.saturating_sub(1)
    }

    /// Get all entries up to (and including) a specific LSN
    pub async fn get_entries_up_to_lsn(&self, target_lsn: u64) -> Vec<GlobalManifestEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.global_lsn <= target_lsn)
            .cloned()
            .collect()
    }

    /// Get all entries in a LSN range [start_lsn, end_lsn] (inclusive)
    pub async fn get_entries_between_lsn(
        &self,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Vec<GlobalManifestEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.global_lsn >= start_lsn && e.global_lsn <= end_lsn)
            .cloned()
            .collect()
    }

    /// Mark all entries after a given LSN as rolled back (for PITR)
    ///
    /// This is used during point-in-time recovery to mark entries that should
    /// not be recovered. Returns the number of entries marked.
    pub async fn mark_entries_after_lsn_rolled_back(&self, target_lsn: u64) -> Result<usize> {
        let mut entries = self.entries.write().await;
        let mut marked_count = 0;
        let mut status_updates = Vec::new();

        for entry in entries.iter_mut() {
            if entry.global_lsn > target_lsn && entry.status == WalEntryStatus::Active {
                entry.status = WalEntryStatus::RolledBack;
                marked_count += 1;

                // Create status update entry for append-only storage
                let mut update_entry = entry.clone();
                update_entry.status = WalEntryStatus::RolledBack;
                status_updates.push(update_entry);
            }
        }

        drop(entries);

        if !status_updates.is_empty() {
            // Write status updates as new manifest segment
            self.write_to_staging(&status_updates).await?;

            info!(
                "📛 PITR: Marked {} entries after LSN {} as RolledBack",
                marked_count, target_lsn
            );
        }

        Ok(marked_count)
    }

    /// Get entries that need to be replayed for a PITR recovery
    ///
    /// Returns active entries between current state and target LSN,
    /// sorted by LSN for replay order.
    pub async fn get_entries_for_pitr_replay(
        &self,
        current_lsn: u64,
        target_lsn: u64,
    ) -> Vec<GlobalManifestEntry> {
        let entries = self.entries.read().await;

        // For forward replay: current_lsn < target_lsn
        // For rollback: current_lsn > target_lsn (return empty, handled separately)
        if current_lsn >= target_lsn {
            return Vec::new();
        }

        let mut replay_entries: Vec<_> = entries
            .iter()
            .filter(|e| {
                e.global_lsn > current_lsn
                    && e.global_lsn <= target_lsn
                    && e.status == WalEntryStatus::Active
            })
            .cloned()
            .collect();

        replay_entries.sort_by_key(|e| e.global_lsn);
        replay_entries
    }

    /// Check if a LSN exists in the manifest
    pub async fn lsn_exists(&self, target_lsn: u64) -> bool {
        let entries = self.entries.read().await;
        entries.iter().any(|e| e.global_lsn == target_lsn)
    }

    /// Find the closest LSN at or before a given timestamp
    pub async fn find_lsn_at_timestamp(&self, timestamp_ms: u64) -> Option<u64> {
        let entries = self.entries.read().await;

        // Find entries at or before the timestamp, get the highest LSN
        entries
            .iter()
            .filter(|e| e.timestamp_ms <= timestamp_ms)
            .map(|e| e.global_lsn)
            .max()
    }

    /// Get entry count by status (for PITR diagnostics)
    pub async fn get_entry_counts_by_status(
        &self,
    ) -> std::collections::HashMap<WalEntryStatus, usize> {
        let entries = self.entries.read().await;
        let mut counts = std::collections::HashMap::new();

        for entry in entries.iter() {
            *counts.entry(entry.status).or_insert(0) += 1;
        }

        counts
    }

    // ========================================================================
    // End PITR Support Methods
    // ========================================================================

    /// Shutdown the service gracefully
    pub async fn shutdown(&self) -> Result<()> {
        debug!("Shutting down GlobalManifestService");

        // Close append channel by dropping the sender
        // This signals the background worker to exit after flushing pending entries
        drop(self.append_tx.clone()); // Close the channel

        // Give worker a moment to process the channel closure
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Wait for background worker to finish with timeout
        if let Some(handle) = self.worker_handle.lock().await.take() {
            // Use timeout to prevent indefinite hang
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(3), // Reduced from 5s to 3s
                handle,
            )
            .await
            {
                Ok(Ok(())) => {
                    debug!("GlobalManifestService worker exited cleanly");
                }
                Ok(Err(e)) => {
                    warn!("GlobalManifestService worker error: {}", e);
                }
                Err(_) => {
                    warn!("GlobalManifestService shutdown timeout - forcing exit");
                    // JoinHandle will be dropped, cancelling the task
                }
            }
        }

        debug!("GlobalManifestService shut down");
        Ok(())
    }
}

fn status_rank(status: WalEntryStatus) -> u8 {
    match status {
        WalEntryStatus::Active => 0,
        WalEntryStatus::Flushed => 1,
        WalEntryStatus::Archived | WalEntryStatus::RolledBack => 2,
    }
}

fn should_replace_manifest_entry(
    existing: &GlobalManifestEntry,
    incoming: &GlobalManifestEntry,
) -> bool {
    incoming.global_lsn > existing.global_lsn
        || (incoming.global_lsn == existing.global_lsn
            && status_rank(incoming.status) > status_rank(existing.status))
}

#[cfg(test)]
mod manifest_status_tests {
    use super::*;
    use crate::storage::persistence::write_ahead_log::BatchId;
    use crate::storage::persistence::write_ahead_log::serialization::SerializationFormat;

    fn entry(status: WalEntryStatus) -> GlobalManifestEntry {
        let mut entry = GlobalManifestEntry::new(
            7,
            "collection".to_string(),
            &BatchId::new(),
            "batch.bcwal".to_string(),
            1,
            1,
            SerializationFormat::Bincode,
            1,
            "file:///tmp".to_string(),
        );
        entry.status = status;
        entry
    }

    #[test]
    fn equal_lsn_flushed_supersedes_active() {
        assert!(should_replace_manifest_entry(
            &entry(WalEntryStatus::Active),
            &entry(WalEntryStatus::Flushed)
        ));
        assert!(!should_replace_manifest_entry(
            &entry(WalEntryStatus::Flushed),
            &entry(WalEntryStatus::Active)
        ));
    }
}

/// Vector Object Economy `FreshnessLsnSource` adapter backed by the
/// global manifest service's LSN allocator.
///
/// The current impl returns the *global* current LSN regardless of the
/// requested `collection_id`. That is a safe over-approximation for the
/// strong-route decision: a watermark advanced past unrelated writes
/// means the planner does at most more delta-merge work, never less
/// correctness. A per-collection cursor can be added later without
/// changing the trait signature.
#[derive(Clone)]
pub struct WalCursorLsnSource {
    manifest: Arc<GlobalManifestService>,
}

impl std::fmt::Debug for WalCursorLsnSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalCursorLsnSource")
            .field("manifest", &"Arc<GlobalManifestService>")
            .finish()
    }
}

impl WalCursorLsnSource {
    pub fn new(manifest: Arc<GlobalManifestService>) -> Self {
        Self { manifest }
    }
}

#[async_trait::async_trait]
impl crate::storage::engines::sst::object_economy_directory::FreshnessLsnSource
    for WalCursorLsnSource
{
    async fn current_lsn(&self, _collection_id: &str) -> u64 {
        self.manifest.current_lsn().await
    }
}

#[cfg(test)]
mod barrier_tests {
    use super::*;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::persistence::write_ahead_log::manifest::types::WalEntryStatus;
    use crate::storage::persistence::write_ahead_log::serialization::SerializationFormat;

    /// Stamped prologue: fresh tempdir-backed local WAL directory + a
    /// filesystem factory over it. Keep the `TempDir` alive for the test's
    /// length.
    async fn tempdir_wal_dir() -> (Arc<FilesystemFactory>, String, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let wal_url = format!("file://{}", temp_dir.path().display());
        let fs_factory = Arc::new(
            FilesystemFactory::create(FilesystemConfig::default())
                .await
                .expect("fs factory"),
        );
        (fs_factory, wal_url, temp_dir)
    }

    /// A service over [`tempdir_wal_dir`]'s directory (fresh instance per
    /// call — the restart test opens a second one over the same dir).
    async fn service_for(
        fs_factory: &Arc<FilesystemFactory>,
        wal_url: &str,
    ) -> Arc<GlobalManifestService> {
        GlobalManifestService::new(
            GlobalManifestServiceConfig::default(),
            fs_factory.clone(),
            wal_url.to_string(),
        )
        .await
        .expect("service")
    }

    /// Stamped active-entry fixture for collection `c1`.
    fn active_entry(batch_id: &str, wal_url: &str) -> GlobalManifestEntry {
        GlobalManifestEntry {
            global_lsn: 0, // allocated by append_async
            collection_id: "c1".to_string(),
            batch_id: batch_id.to_string(),
            file_path: format!("c1/wal/{batch_id}.bcwal"),
            storage_url: wal_url.to_string(),
            size_bytes: 42,
            checksum_crc32: 0,
            timestamp_ms: 1,
            format: SerializationFormat::Bincode,
            vector_count: 1,
            status: WalEntryStatus::Active,
            checkpoint_id: None,
        }
    }

    /// Issue #1126 L1 regression: an entry appended via the WRITE-BEHIND path
    /// must be visible to `mark_flushed_and_delete_files` immediately — the
    /// drain barrier makes append happen-before the flush-time lookup. Without
    /// the barrier this test flakes/fails whenever the worker's batch window
    /// (batch_interval_ms) hasn't elapsed — which was the restart data-loss
    /// trigger (leftover WAL → duplicate L0_recovery segments on next boot).
    #[tokio::test]
    async fn write_behind_append_visible_to_flush_lookup_without_sleep() {
        let (fs_factory, wal_url, _dir) = tempdir_wal_dir().await;
        let service = service_for(&fs_factory, &wal_url).await;

        let entry = active_entry("b1126", &wal_url);

        // Async (write-behind) append, then IMMEDIATELY mark — no sleep.
        service.append_async(entry).await.expect("append");
        service
            .mark_flushed_and_delete_files(&["b1126".to_string()])
            .await
            .expect("mark+delete");

        let entries = service.get_collection_entries("c1").await;
        assert_eq!(entries.len(), 1, "entry must be visible post-barrier");
        assert_eq!(
            entries[0].status,
            WalEntryStatus::Flushed,
            "freshly appended batch must be marked Flushed, not missed"
        );
    }

    /// TD-DELVEC-1 WI-5 P0: `append_async` returns the durable per-batch
    /// `global_lsn` it allocated (previously a discarded `()`), so the cold-delete
    /// path can key deletion-vector bits on a generation in the read side's
    /// `snapshot_lsn` number space. The LSN is allocated synchronously, monotonic
    /// across appends, and matches `current_lsn`.
    #[tokio::test]
    async fn append_async_returns_durable_global_lsn() {
        let (fs_factory, wal_url, _dir) = tempdir_wal_dir().await;
        let service = service_for(&fs_factory, &wal_url).await;

        let mk_entry = |batch_id: &str| active_entry(batch_id, &wal_url);

        let lsn1 = service
            .append_async(mk_entry("b1"))
            .await
            .expect("append b1");
        let lsn2 = service
            .append_async(mk_entry("b2"))
            .await
            .expect("append b2");
        assert!(
            lsn1 > 0,
            "append_async must return an allocated LSN, got {lsn1}"
        );
        assert!(
            lsn2 > lsn1,
            "returned LSNs must be monotonic: {lsn1} then {lsn2}"
        );
        assert_eq!(
            service.current_lsn().await,
            lsn2,
            "the returned LSN must match the manifest's current_lsn"
        );
    }

    /// WAL-retention slice: the reaper (checkpoint + cleanup) — which
    /// previously had ZERO production callers, so `data/wal` grew one
    /// `manifest_*.jsonl` per append for the life of the data dir — must
    /// (1) drop checkpointed/flushed entries, (2) shrink the on-disk
    /// segment count, and (3) survive a restart without duplicating or
    /// resurrecting entries that appear in both the rewritten manifest and
    /// a surviving segment.
    #[tokio::test]
    async fn reaper_reclaims_segments_and_restart_dedups() {
        let (fs_factory, wal_url, temp_dir) = tempdir_wal_dir().await;
        let count_segments = |dir: &std::path::Path| -> usize {
            std::fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| {
                            let name = e.file_name();
                            let name = name.to_string_lossy();
                            name.starts_with("manifest_") && name.ends_with(".jsonl")
                        })
                        .count()
                })
                .unwrap_or(0)
        };

        let mk_entry = |batch_id: &str| active_entry(batch_id, &wal_url);

        {
            let service = service_for(&fs_factory, &wal_url).await;
            // append_sync: one durable segment per append, so the segments
            // land with distinct LSN ranges and the watermark can strand the
            // oldest one (append_async may coalesce all three into a single
            // segment, which then legitimately cannot be reclaimed).
            for id in ["b1", "b2", "b3"] {
                service.append_sync(mk_entry(id)).await.expect("append");
            }
            // Flush b1 and b2 (their .bcwal files don't exist; deletion of
            // missing files is tolerated), leaving b3 active.
            service
                .mark_flushed_and_delete_files(&["b1".to_string(), "b2".to_string()])
                .await
                .expect("mark+delete");
            let before = count_segments(temp_dir.path());
            // The write-behind worker may coalesce appends into fewer files;
            // the property under test is monotonic growth WITHOUT the reaper
            // and shrinkage WITH it, not an exact per-append count.
            assert!(
                before >= 2,
                "appends + status updates write segments, saw {before}"
            );

            service.create_checkpoint().await.expect("checkpoint");
            let removed = service
                .cleanup_checkpointed_entries()
                .await
                .expect("cleanup");
            // Designed semantics: `safe_to_delete_before_lsn` is the HIGHEST
            // flushed LSN and cleanup retains entries at/above it, so the
            // newest flushed entry (b2) survives as the checkpoint watermark
            // and only strictly-older flushed entries (b1) are reaped.
            assert!(
                removed >= 1,
                "flushed entries below the watermark must be reaped, got {removed}"
            );
            let after = count_segments(temp_dir.path());
            assert!(
                after < before,
                "on-disk manifest segments must shrink ({before} -> {after})"
            );
        }

        // Restart: a fresh service over the same dir must see exactly the one
        // still-active entry — never a duplicate or a resurrected flushed one.
        let reopened = service_for(&fs_factory, &wal_url).await;
        let entries = reopened.get_collection_entries("c1").await;
        let active: Vec<_> = entries
            .iter()
            .filter(|e| e.status == WalEntryStatus::Active)
            .collect();
        assert_eq!(
            active.len(),
            1,
            "exactly b3 must survive as active: {entries:?}"
        );
        assert_eq!(active[0].batch_id, "b3");
        assert!(
            !entries.iter().any(|e| e.batch_id == "b1"),
            "b1 (below the watermark) must stay reaped after restart: {entries:?}"
        );
        for id in ["b2", "b3"] {
            assert!(
                entries.iter().filter(|e| e.batch_id == id).count() <= 1,
                "no duplicate {id} after reap+restart: {entries:?}"
            );
        }
    }
}

// Deferred: Fix compilation errors - global_manifest renamed to manifest, import paths changed
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use super::super::global_manifest::SerializationFormat;
//     use super::super::BatchId;
//
//     #[tokio::test]
//     async fn test_concurrent_appends() {
//         let temp_dir = tempfile::tempdir().ok().unwrap_or_default();
//         let wal_url = format!("file://{}", temp_dir.path().display());
//
//         let fs_factory = Arc::new(
//             crate::storage::persistence::filesystem::FilesystemFactory::create_default()
//                 .await
//                 .ok()
//         );
//
//         let service = GlobalManifestService::new(
//             GlobalManifestServiceConfig::default(),
//             fs_factory,
//             wal_url,
//         ).await.ok();
//
//         // Spawn 10 concurrent append tasks
//         let mut handles = vec![];
//         for i in 0..10 {
//             let service = service.clone();
//             let handle = tokio::spawn(async move {
//                 for j in 0..100 {
//                     let batch_id = BatchId::new();
//                     let entry = GlobalManifestEntry::new(
//                         0,  // Will be allocated
//                         format!("collection_{}", i),
//                         &batch_id,
//                         format!("batch_{}.bcwal", j),
//                         1024,
//                         12345,
//                         SerializationFormat::Bincode,
//                         10,
//                         format!("file://{}", temp_dir.path().display()),
//                     );
//                     service.append_async(entry).await.ok();
//                 }
//             });
//             handles.push(handle);
//         }
//
//         // Wait for all tasks
//         for handle in handles {
//             handle.await.ok();
//         }
//
//         // Give background worker time to flush
//         tokio::time::sleep(Duration::from_millis(500)).await;
//
//         // Verify all entries were written
//         let entries = service.get_all_entries().await;
//         assert_eq!(entries.len(), 1000);
//
//         // Verify LSNs are unique and sequential
//         // for (i, entry) in entries.iter().enumerate() {
//         //     assert_eq!(entry.global_lsn, (i + 1) as u64);
//         // }
//     // }
// }
