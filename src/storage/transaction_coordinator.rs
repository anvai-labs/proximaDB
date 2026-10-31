// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Unified Atomic Operations Package for ProximaDB Storage Components
//!
//! This module provides common atomic operations infrastructure that can be used
//! across all storage components: VIPER, WAL, and the system catalog.
//!
//! ## Key Features:
//! - **Staging Pattern**: Write to `__flush/`, `__temp/`, or custom staging directories
//! - **Atomic Moves**: Use filesystem's atomic move operations
//! - **Strategy-Based**: Different strategies for local vs cloud storage
//! - **Cleanup**: Automatic cleanup of failed/orphaned operations
//! - **Unified Interface**: Single API for all storage components

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use proximadb_kernel::uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, trace, warn};

use crate::storage::persistence::filesystem::{
    FilesystemFactory,
    atomic_strategy::{AtomicWriteConfig, AtomicWriteExecutor, AtomicWriteExecutorFactory},
    write_strategy::{MetadataWriteStrategy, WriteStrategyFactory},
};

// Re-export metadata transaction types for backward compatibility
pub use crate::storage::metadata::atomic::{
    AtomicMetadataStore, IsolationLevel, MetadataTransaction,
};
// Note: TransactionState is defined locally to avoid conflicts

/// Atomic operation ID for tracking operations
pub type OperationId = String;

/// Transaction ID for ACID operations
pub type TransactionId = String;

/// Transaction ID generator
static TRANSACTION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Staging operation types
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionStageType {
    /// Flush operation (e.g., memtable to storage)
    Flush,
    /// Compaction operation (e.g., merge files)
    Compaction,
    /// Metadata operation (e.g., collection updates)
    Metadata,
    /// WAL operation (e.g., segment rotation)
    Wal,
    /// Custom operation type
    Custom(String),
    /// ACID transaction operation
    Transaction,
}

impl TransactionStageType {
    /// Get staging directory name for this operation type
    pub fn staging_dir_name(&self) -> &str {
        match self {
            TransactionStageType::Flush => "__flush",
            TransactionStageType::Compaction => "__compact",
            TransactionStageType::Metadata => "__metadata_info",
            TransactionStageType::Wal => "__wal",
            TransactionStageType::Transaction => "__transaction",
            TransactionStageType::Custom(name) => name,
        }
    }
}

/// Staging directory configuration
#[derive(Debug, Clone)]
pub struct StagingConfig {
    /// Base storage URL (e.g., "file:///data", "s3://bucket")
    pub base_url: String,

    /// Collection ID (optional, for collection-specific operations)
    #[allow(dead_code)]
    pub collection_id: Option<String>,

    /// Operation type
    #[allow(dead_code)]
    pub operation_type: TransactionStageType,

    /// Custom staging directory name (overrides default)
    pub custom_staging_dir: Option<String>,

    /// Enable automatic cleanup of successful operations
    pub auto_cleanup: bool,

    /// Maximum age for cleanup of orphaned operations (hours)
    pub max_orphaned_age_hours: u64,

    /// Skip UUID subdirectory in staging path (useful for compaction to avoid cleanup issues)
    pub skip_uuid_subdir: bool,
}

impl Default for StagingConfig {
    fn default() -> Self {
        Self {
            base_url: "file://./data".to_string(),
            collection_id: None,
            operation_type: TransactionStageType::Flush,
            custom_staging_dir: None,
            auto_cleanup: true,
            max_orphaned_age_hours: 24,
            skip_uuid_subdir: false,
        }
    }
}

/// Atomic operation metadata
#[derive(Debug, Clone)]
pub struct TransactionalOperationMetadata {
    pub operation_id: OperationId,
    pub operation_type: TransactionStageType,
    pub collection_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub staging_url: String,
    pub final_url: String,
    pub status: TransactionalOperationStatus,
    /// Flag indicating if this operation is managed by ZeroCopyFilesystem
    /// When true, AtomicCoordinator will skip staging operations and not error
    /// if files are not found during finalize/cleanup operations
    pub zero_copy_managed: bool,
    /// TD-TIER-1 SC5: the access tier finalize applies to every object it
    /// publishes from staging (`None` ⇒ the account default — today's PUT).
    /// Set by [`TransactionCoordinator::set_operation_access_tier`].
    pub access_tier: Option<proximadb_storage_filesystem_types::ObjectAccessTier>,
}

/// Atomic operation status
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionalOperationStatus {
    /// Operation is being prepared
    Preparing,
    /// Operation is writing to staging
    Staging,
    /// Operation is performing atomic move
    Finalizing,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed(String),
}

/// Transaction state for ACID compliance
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionState {
    /// Transaction initialized but not started
    Initialized,
    /// Transaction is active and running
    Active,
    /// Phase 1: Preparing all participants
    Preparing,
    /// All participants prepared successfully
    Prepared,
    /// Phase 2: Committing all participants
    Committing,
    /// Transaction committed successfully
    Committed,
    /// Transaction is aborting
    Aborting,
    /// Transaction aborted/rolled back
    Aborted,
    /// Transaction timed out
    TimedOut,
    /// Transaction failed
    Failed(String),
}

/// Participant in the atomic transaction
#[derive(Debug, Clone)]
pub struct TransactionParticipant {
    /// Unique participant ID
    pub id: String,
    /// Current state of this participant
    pub state: ParticipantState,
    /// Rollback action if needed
    pub rollback_action: Option<RollbackAction>,
    /// Timestamp when participant was added
    pub joined_at: Instant,
}

/// State of individual participant
#[derive(Debug, Clone, PartialEq)]
pub enum ParticipantState {
    /// Participant registered but not prepared
    Registered,
    /// Participant is preparing
    Preparing,
    /// Participant prepared successfully
    Prepared,
    /// Participant is committing
    Committing,
    /// Participant committed successfully
    Committed,
    /// Participant is rolling back
    RollingBack,
    /// Participant rolled back successfully
    RolledBack,
    /// Participant failed
    Failed(String),
}

/// Rollback action for a participant
#[derive(Debug, Clone)]
pub enum RollbackAction {
    /// Remove entry from memtable by key
    RemoveFromMemtable { key: String },
    /// Restore previous value in memtable
    RestoreMemtableValue {
        key: String,
        previous_value: Vec<u8>,
    },
    /// Remove entry from secondary index
    RemoveFromSecondaryIndex { name: String, uuid: String },
    /// Delete file from disk
    DeleteFile { path: String },
    /// Restore file on disk
    RestoreFile { path: String, content: Vec<u8> },
    /// Custom rollback action
    Custom(String),
}

/// Active transaction tracking
#[derive(Debug)]
pub struct ActiveTransaction {
    /// Unique transaction ID
    pub id: String,
    /// Transaction state
    pub state: TransactionState,
    /// Participants in this transaction
    pub participants: HashMap<String, TransactionParticipant>,
    /// Transaction start time
    pub started_at: Instant,
    /// Transaction deadline
    pub deadline: Option<Instant>,
    /// Transaction metadata
    pub metadata: HashMap<String, String>,
}

impl ActiveTransaction {
    /// Create new transaction
    pub fn new(id: String, deadline_ms: Option<u64>) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: TransactionState::Initialized,
            participants: HashMap::new(),
            started_at: now,
            deadline: deadline_ms.map(|ms| now + Duration::from_millis(ms)),
            metadata: HashMap::new(),
        }
    }

    /// Check if transaction has expired
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() > d)
    }

    /// Add participant to transaction
    pub fn add_participant(&mut self, id: String, rollback_action: Option<RollbackAction>) {
        self.participants.insert(
            id.clone(),
            TransactionParticipant {
                id,
                state: ParticipantState::Registered,
                rollback_action,
                joined_at: Instant::now(),
            },
        );
    }

    /// Check if all participants are in the given state
    pub fn all_participants_in_state(&self, state: &ParticipantState) -> bool {
        self.participants.values().all(|p| &p.state == state)
    }

    /// Update participant state
    pub fn update_participant_state(&mut self, id: &str, state: ParticipantState) -> Result<()> {
        self.participants
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("Participant {} not found", id))
            .map(|p| p.state = state)
    }
}

/// Backwards-compat alias for [`UnifiedTransactionCoordinator`].
pub type TransactionCoordinator = UnifiedTransactionCoordinator;

/// Unified atomic operations coordinator with ACID support
pub struct UnifiedTransactionCoordinator {
    /// Filesystem factory for multi-cloud support
    filesystem: Arc<FilesystemFactory>,

    /// Atomic write executor for robust operations
    atomic_executor: Box<dyn AtomicWriteExecutor>,

    /// Write strategy for optimized operations
    write_strategy: MetadataWriteStrategy,

    /// Active operations tracker using lock-free DashMap
    active_operations: Arc<DashMap<OperationId, TransactionalOperationMetadata>>,

    /// Active ACID transactions
    transactions: Arc<DashMap<TransactionId, Arc<RwLock<ActiveTransaction>>>>,

    /// Transaction timeout (default: 30 seconds)
    default_timeout_ms: u64,

    /// Cleanup task handle
    cleanup_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Configuration
    config: AtomicWriteConfig,
}

impl std::fmt::Debug for UnifiedTransactionCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedTransactionCoordinator")
            .field("filesystem", &self.filesystem)
            .field("active_operations_count", &self.active_operations.len())
            .field("config", &self.config)
            .finish()
    }
}

impl UnifiedTransactionCoordinator {
    /// Create new unified atomic coordinator with ACID support
    pub async fn new(
        filesystem: Arc<FilesystemFactory>,
        temp_directory: Option<String>,
    ) -> Result<Self> {
        let config = AtomicWriteConfig::default();

        // Create appropriate atomic executor based on environment
        let atomic_executor = if cfg!(test) {
            AtomicWriteExecutorFactory::create_dev_executor()
        } else {
            AtomicWriteExecutorFactory::create_executor(&config)
        };

        // Create metadata-optimized write strategy
        let fs = filesystem
            .get_filesystem("file://")
            .map_err(|_| anyhow::anyhow!("No filesystem available for transaction coordinator"))?;
        let write_strategy =
            WriteStrategyFactory::create_metadata_strategy(fs.as_ref(), temp_directory.as_deref())?;

        let coordinator = Self {
            filesystem,
            atomic_executor,
            write_strategy,
            active_operations: Arc::new(DashMap::new()),
            transactions: Arc::new(DashMap::new()),
            default_timeout_ms: 30000, // 30 seconds
            cleanup_handle: Arc::new(Mutex::new(None)),
            config,
        };

        // Start cleanup task for expired transactions
        coordinator.start_cleanup_task();

        Ok(coordinator)
    }

    /// Begin atomic operation - creates staging directory and returns operation metadata
    pub async fn begin_atomic_operation(
        &self,
        staging_config: &StagingConfig,
    ) -> Result<TransactionalOperationMetadata> {
        let operation_id = Uuid::new_v4().to_string();

        debug!("Beginning atomic operation: {}", operation_id);
        trace!(
            "Staging config: base_url={}, operation_type={:?}, custom_staging_dir={:?}",
            staging_config.base_url,
            staging_config.operation_type,
            staging_config.custom_staging_dir
        );

        // Build staging and final URLs
        let (staging_url, final_url) = self.build_operation_urls(staging_config, &operation_id)?;

        trace!("Staging URL: {}", staging_url);
        trace!("Final URL: {}", final_url);

        // Create both staging and final directories upfront for robustness
        // Note: staging_url and final_url are directory URLs, not file URLs
        trace!("Creating staging directory: {}", staging_url);
        self.filesystem
            .create_dir_all(&staging_url)
            .await
            .context("Failed to create staging directory")?;

        trace!("Creating final directory: {}", final_url);
        self.filesystem
            .create_dir_all(&final_url)
            .await
            .context("Failed to create final directory")?;

        // Create operation metadata
        let metadata = TransactionalOperationMetadata {
            operation_id: operation_id.clone(),
            operation_type: staging_config.operation_type.clone(),
            collection_id: staging_config.collection_id.clone(),
            started_at: Utc::now(),
            staging_url,
            final_url,
            status: TransactionalOperationStatus::Preparing,
            zero_copy_managed: false, // Default to false for traditional operations
            access_tier: None,
        };

        // Track operation using DashMap
        self.active_operations
            .insert(operation_id.clone(), metadata.clone());

        info!("✅ Atomic operation prepared: {}", operation_id);
        Ok(metadata)
    }

    /// Begin zero-copy managed atomic operation - for ZeroCopyFilesystem
    /// This creates an operation metadata but lets ZeroCopyFilesystem handle staging/moving
    pub async fn begin_zero_copy_managed_operation(
        &self,
        staging_config: &StagingConfig,
    ) -> Result<TransactionalOperationMetadata> {
        let operation_id = Uuid::new_v4().to_string();

        info!("🚀 Beginning zero-copy managed operation: {}", operation_id);
        debug!("Configuration: {:?}", staging_config);

        // Generate staging and final URLs (for tracking purposes)
        let staging_url = self.generate_staging_url(staging_config, &operation_id);
        let final_url = self.generate_final_url(staging_config);

        debug!("📁 Staging URL: {}", staging_url);
        debug!("📁 Final URL: {}", final_url);

        // Note: We don't create directories here since ZeroCopyFilesystem handles everything

        // Create operation metadata with zero_copy_managed flag set to true
        let metadata = TransactionalOperationMetadata {
            operation_id: operation_id.clone(),
            operation_type: staging_config.operation_type.clone(),
            collection_id: staging_config.collection_id.clone(),
            started_at: Utc::now(),
            staging_url,
            final_url,
            status: TransactionalOperationStatus::Preparing,
            zero_copy_managed: true, // This operation is managed by ZeroCopyFilesystem
            access_tier: None,
        };

        // Track operation using DashMap
        self.active_operations
            .insert(operation_id.clone(), metadata.clone());

        info!(
            "✅ Zero-copy managed atomic operation prepared: {}",
            operation_id
        );
        Ok(metadata)
    }

    /// Write file to staging area
    pub async fn write_to_staging(
        &self,
        operation_id: &OperationId,
        relative_path: &str,
        data: &[u8],
    ) -> Result<()> {
        trace!(
            "write_to_staging: {} bytes to {}",
            data.len(),
            relative_path
        );

        // Get operation metadata from DashMap
        let metadata = self
            .active_operations
            .get(operation_id)
            .ok_or_else(|| anyhow::anyhow!("Operation not found: {}", operation_id))?
            .clone();

        // Update status to staging
        self.update_operation_status(operation_id, TransactionalOperationStatus::Staging)
            .await?;

        // Build staging file URL
        let staging_file_url = format!(
            "{}/{}",
            metadata.staging_url.trim_end_matches('/'),
            relative_path
        );

        // Create file options using write strategy
        let fs = self.filesystem.get_filesystem(&staging_file_url)?;

        let file_options = self
            .write_strategy
            .create_file_options(fs.as_ref(), &staging_file_url)?;

        // Extract path for write
        let staging_path = FilesystemFactory::resolve_path(&staging_file_url)?;

        // Write to staging using atomic executor
        match self
            .atomic_executor
            .write_atomic(fs.as_ref(), &staging_path, data, Some(file_options))
            .await
        {
            Ok(_) => {
                trace!("Written to staging: {}", staging_file_url);
                Ok(())
            }
            Err(e) => {
                error!("Failed to write to staging: {}", e);
                Err(anyhow::anyhow!("Failed to write to staging: {}", e))
            }
        }
    }

    /// Finalize atomic operation - move from staging to final location
    pub async fn finalize_atomic_operation(&self, operation_id: &OperationId) -> Result<()> {
        trace!("🔄 [] Finalizing atomic operation: {}", operation_id);
        debug!("🔄 [DEBUG] Finalizing atomic operation: {}", operation_id);

        // Get operation metadata from DashMap
        let metadata = self
            .active_operations
            .get(operation_id)
            .ok_or_else(|| anyhow::anyhow!("Operation not found: {}", operation_id))?
            .clone();

        trace!(
            "Operation metadata: id={}, type={:?}, zero_copy={}",
            metadata.operation_id, metadata.operation_type, metadata.zero_copy_managed
        );

        // Check if this operation is managed by ZeroCopyFilesystem
        if metadata.zero_copy_managed {
            info!(
                "🚀 [DEBUG] Operation is managed by ZeroCopyFilesystem - skipping staging operations"
            );
            debug!("🚀 [DEBUG] ZeroCopyFilesystem has already handled staging and atomic move");

            // Update status directly to completed since ZeroCopyFilesystem handled the operation
            self.update_operation_status(operation_id, TransactionalOperationStatus::Completed)
                .await?;

            // Remove from active operations
            self.active_operations.remove(operation_id);

            info!("🎉 Zero-copy managed operation completed: {}", operation_id);
            return Ok(());
        }

        // Update status to finalizing for traditional operations
        self.update_operation_status(operation_id, TransactionalOperationStatus::Finalizing)
            .await?;

        // List all files in staging directory
        trace!("Listing staging directory: {}", metadata.staging_url);

        let staging_entries = self
            .filesystem
            .list(&metadata.staging_url)
            .await
            .context("Failed to list staging directory")?;

        trace!("Found {} files in staging", staging_entries.len());

        // Individual file listing removed - too verbose for production

        // Move each file atomically from staging to final location
        // Note: Final directory was already created during begin_atomic_operation
        for entry in staging_entries {
            if !entry.metadata.is_directory && !entry.name.starts_with(".") {
                // Use the full URL from entry, not just the name
                let staging_file_url = entry.url.clone();
                let final_file_url = format!(
                    "{}/{}",
                    metadata.final_url.trim_end_matches('/'),
                    entry.name
                );

                trace!("Moving file: {} -> {}", staging_file_url, final_file_url);

                // Use FilesystemFactory's atomic move which handles cross-storage scenarios
                match self
                    .filesystem
                    .move_atomic_with_options(
                        &staging_file_url,
                        &final_file_url,
                        proximadb_storage_filesystem_types::FileOptions::for_access_tier(
                            metadata.access_tier,
                        ),
                    )
                    .await
                {
                    Ok(_) => {
                        trace!("Move successful: {}", entry.name);
                        // File existence verification removed - move_atomic already guarantees this
                    }
                    Err(e) => {
                        // TD-FLUSH-6: preserve the full error chain. The prior
                        // `anyhow::anyhow!("…{e}")` consumed `e` via Display and
                        // dropped its `.source()`, hiding the real object-store
                        // cause under the generic outer wrapper. `Error::from(e)
                        // .context(..)` keeps the inner error traversable so it
                        // surfaces in `{:#}` / Debug log output.
                        error!("Failed to move file {}: {:#}", entry.name, e);
                        return Err(anyhow::Error::from(e).context(format!(
                            "Failed to move {staging_file_url} to {final_file_url}"
                        )));
                    }
                }

                debug!("✅ Moved: {}", entry.name);
            }
        }

        // Clean up staging directory. Best-effort (TD-FLUSH-6): every staged
        // segment has already been moved to the final prefix above — the move
        // loop returns `Err` on the first failure, so reaching here means the
        // commit is already durable. On object stores the staging "directory"
        // is a *prefix* with no blob of its own, so once the move loop drains
        // it the delete returns NotFound (404); that is the expected end state,
        // not a failure. Never fail the commit on cleanup — orphaned staging
        // objects are reaped by the staging config's `auto_cleanup` /
        // `max_orphaned_age_hours` sweep.
        if metadata.operation_type != TransactionStageType::Custom("preserve_staging".to_string()) {
            if let Err(e) = self.filesystem.delete(&metadata.staging_url).await {
                warn!(
                    staging_url = %metadata.staging_url,
                    "staging cleanup failed (non-fatal — segments already committed to final prefix): {e:#}"
                );
            } else {
                debug!("🧹 Cleaned up staging directory");
            }
        }

        // Update status to completed
        self.update_operation_status(operation_id, TransactionalOperationStatus::Completed)
            .await?;

        // Remove from active operations using DashMap
        self.active_operations.remove(operation_id);

        // List the final directory to confirm files are there
        trace!(
            "Listing final directory after operation: {}",
            metadata.final_url
        );
        if let Ok(final_entries) = self.filesystem.list(&metadata.final_url).await {
            debug!("Found {} files in final location", final_entries.len());
            // Individual file listing removed - way too verbose for production
            // Use trace if needed for deep debugging
        }

        debug!("Atomic operation completed: {}", operation_id);
        Ok(())
    }

    /// Abort atomic operation - cleanup staging area
    pub async fn abort_atomic_operation(
        &self,
        operation_id: &OperationId,
        reason: &str,
    ) -> Result<()> {
        warn!(
            "❌ Aborting atomic operation: {} ({})",
            operation_id, reason
        );

        // Get operation metadata from DashMap
        let metadata = self
            .active_operations
            .get(operation_id)
            .map(|entry| entry.clone());

        if let Some(metadata) = metadata {
            // Update status to failed
            self.update_operation_status(
                operation_id,
                TransactionalOperationStatus::Failed(reason.to_string()),
            )
            .await?;

            // Check if this operation is managed by ZeroCopyFilesystem
            if metadata.zero_copy_managed {
                info!(
                    "🚀 [DEBUG] Abort operation is managed by ZeroCopyFilesystem - skipping staging cleanup"
                );
                debug!("🚀 [DEBUG] ZeroCopyFilesystem will handle its own cleanup if needed");
            } else {
                // Cleanup staging directory for traditional operations
                if let Err(e) = self.filesystem.delete(&metadata.staging_url).await {
                    warn!("Failed to cleanup staging directory during abort: {}", e);
                } else {
                    debug!("🧹 Cleaned up staging directory after abort");
                }
            }

            // Remove from active operations using DashMap
            self.active_operations.remove(operation_id);
        }

        Ok(())
    }

    /// Get operation status
    pub async fn get_operation_status(
        &self,
        operation_id: &OperationId,
    ) -> Option<TransactionalOperationStatus> {
        self.active_operations
            .get(operation_id)
            .map(|entry| entry.status.clone())
    }

    /// List active operations
    pub async fn list_active_operations(&self) -> Vec<TransactionalOperationMetadata> {
        self.active_operations
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Cleanup orphaned operations older than configured age
    pub async fn cleanup_orphaned_operations(
        &self,
        base_url: &str,
        max_age_hours: u64,
    ) -> Result<usize> {
        debug!(
            "🧹 Cleaning up orphaned operations older than {} hours",
            max_age_hours
        );

        let cutoff_time = Utc::now() - chrono::Duration::hours(max_age_hours as i64);
        let mut cleaned_count = 0;

        // Find directories with staging patterns
        let staging_patterns = ["__flush", "__compact", "__metadata_info", "__wal"];

        for pattern in &staging_patterns {
            let pattern_url = format!("{}/{}", base_url.trim_end_matches('/'), pattern);

            if let Ok(entries) = self.filesystem.list(&pattern_url).await {
                for entry in entries {
                    if entry.metadata.is_directory {
                        // Check if directory is older than cutoff
                        if let Some(created) = entry.metadata.created
                            && created < cutoff_time
                        {
                            let old_dir_url = format!("{}/{}", pattern_url, entry.name);
                            if let Ok(()) = self.filesystem.delete(&old_dir_url).await {
                                cleaned_count += 1;
                                debug!("🧹 Cleaned orphaned staging dir: {}", old_dir_url);
                            }
                        }
                    }
                }
            }
        }

        info!("🧹 Cleaned {} orphaned operations", cleaned_count);
        Ok(cleaned_count)
    }

    // ACID Transaction Support Methods

    /// Begin a new atomic transaction
    pub async fn begin_transaction(
        &self,
        tx_id: &str,
        participants: Vec<String>,
    ) -> Result<TransactionHandle<'_>> {
        info!("🔄 Beginning transaction: {}", tx_id);

        // Check if transaction already exists
        if self.transactions.contains_key(tx_id) {
            return Err(anyhow::anyhow!("Transaction {} already exists", tx_id));
        }

        // Create new transaction
        let mut transaction =
            ActiveTransaction::new(tx_id.to_string(), Some(self.default_timeout_ms));

        // Register participants
        for participant in participants {
            transaction.add_participant(participant, None);
        }

        transaction.state = TransactionState::Initialized;

        // Store transaction
        let tx_arc = Arc::new(RwLock::new(transaction));
        self.transactions.insert(tx_id.to_string(), tx_arc.clone());

        Ok(TransactionHandle {
            id: tx_id.to_string(),
            coordinator: self,
            transaction: tx_arc,
        })
    }

    /// Prepare phase of two-phase commit
    pub async fn prepare_transaction(&self, tx_id: &str) -> Result<bool> {
        let tx = self.get_transaction(tx_id).await?;
        let mut tx_guard = tx.write().await;

        // Check current state
        match &tx_guard.state {
            TransactionState::Initialized => {
                tx_guard.state = TransactionState::Preparing;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Transaction {} not in correct state for prepare",
                    tx_id
                ));
            }
        }

        // Mark all participants as preparing
        let participant_ids: Vec<String> = tx_guard.participants.keys().cloned().collect();
        for pid in &participant_ids {
            tx_guard.update_participant_state(pid, ParticipantState::Preparing)?;
        }

        drop(tx_guard); // Release lock during preparation

        // Simulate prepare phase (in real implementation, would call each participant)
        info!(
            "📝 Preparing {} participants for transaction {}",
            participant_ids.len(),
            tx_id
        );

        // Re-acquire lock and check results
        let mut tx_guard = tx.write().await;

        // Mark all participants as prepared (in real implementation, based on actual results)
        for pid in &participant_ids {
            tx_guard.update_participant_state(pid, ParticipantState::Prepared)?;
        }

        // Update transaction state
        if tx_guard.all_participants_in_state(&ParticipantState::Prepared) {
            tx_guard.state = TransactionState::Prepared;
            info!("✅ All participants prepared for transaction {}", tx_id);
            Ok(true)
        } else {
            tx_guard.state = TransactionState::Failed("Not all participants prepared".to_string());
            Ok(false)
        }
    }

    /// Commit phase of two-phase commit
    pub async fn commit_transaction(&self, tx_id: &str) -> Result<()> {
        let tx = self.get_transaction(tx_id).await?;
        let mut tx_guard = tx.write().await;

        // Check current state
        match &tx_guard.state {
            TransactionState::Prepared => {
                tx_guard.state = TransactionState::Committing;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Transaction {} not prepared for commit",
                    tx_id
                ));
            }
        }

        // Mark all participants as committing
        let participant_ids: Vec<String> = tx_guard.participants.keys().cloned().collect();
        for pid in &participant_ids {
            tx_guard.update_participant_state(pid, ParticipantState::Committing)?;
        }

        drop(tx_guard); // Release lock during commit

        // Simulate commit phase
        info!(
            "💾 Committing {} participants for transaction {}",
            participant_ids.len(),
            tx_id
        );

        // Re-acquire lock and mark as committed
        let mut tx_guard = tx.write().await;

        // Mark all participants as committed
        for pid in &participant_ids {
            tx_guard.update_participant_state(pid, ParticipantState::Committed)?;
        }

        tx_guard.state = TransactionState::Committed;
        info!("✅ Transaction {} committed successfully", tx_id);

        // Remove from active transactions after short delay
        let transactions = self.transactions.clone();
        let tx_id_clone = tx_id.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(5)).await;
            transactions.remove(&tx_id_clone);
        });

        Ok(())
    }

    /// Rollback/abort transaction
    pub async fn rollback_transaction(&self, tx_id: &str) -> Result<()> {
        let tx = self.get_transaction(tx_id).await?;
        let mut tx_guard = tx.write().await;

        info!("🔙 Rolling back transaction: {}", tx_id);
        tx_guard.state = TransactionState::Aborting;

        // Get rollback actions
        let rollback_actions: Vec<(String, Option<RollbackAction>)> = tx_guard
            .participants
            .values()
            .map(|p| (p.id.clone(), p.rollback_action.clone()))
            .collect();

        // Mark participants as rolling back
        for (pid, _) in &rollback_actions {
            tx_guard.update_participant_state(pid, ParticipantState::RollingBack)?;
        }

        drop(tx_guard); // Release lock during rollback

        // Execute rollback actions
        for (pid, action) in rollback_actions {
            if let Some(action) = action {
                self.execute_rollback_action(&pid, action).await?;
            }
        }

        // Mark transaction as aborted
        let mut tx_guard = tx.write().await;
        for pid in tx_guard.participants.keys().cloned().collect::<Vec<_>>() {
            tx_guard.update_participant_state(&pid, ParticipantState::RolledBack)?;
        }

        tx_guard.state = TransactionState::Aborted;
        info!("✅ Transaction {} rolled back successfully", tx_id);

        // Remove from active transactions
        self.transactions.remove(tx_id);

        Ok(())
    }

    /// Execute a rollback action
    async fn execute_rollback_action(
        &self,
        participant_id: &str,
        action: RollbackAction,
    ) -> Result<()> {
        match action {
            RollbackAction::RemoveFromMemtable { key } => {
                debug!("🔙 Rolling back memtable: removing key {}", key);
                // In real implementation, would call memtable remove
                Ok(())
            }
            RollbackAction::RestoreMemtableValue {
                key,
                previous_value,
            } => {
                debug!(
                    "🔙 Rolling back memtable: restoring key {} with {} bytes",
                    key,
                    previous_value.len()
                );
                // In real implementation, would call memtable restore
                Ok(())
            }
            RollbackAction::RemoveFromSecondaryIndex { name, uuid } => {
                debug!(
                    "🔙 Rolling back secondary index: removing {} -> {}",
                    name, uuid
                );
                // In real implementation, would call secondary index remove
                Ok(())
            }
            RollbackAction::DeleteFile { path } => {
                debug!("🔙 Rolling back disk: deleting file {}", path);
                // Use filesystem to delete
                self.filesystem.delete(&path).await?;
                Ok(())
            }
            RollbackAction::RestoreFile { path, content } => {
                debug!(
                    "🔙 Rolling back disk: restoring file {} with {} bytes",
                    path,
                    content.len()
                );
                // Use filesystem to restore
                let fs = self.filesystem.get_filesystem(&path)?;
                fs.write(&FilesystemFactory::resolve_path(&path)?, &content, None)
                    .await?;
                Ok(())
            }
            RollbackAction::Custom(desc) => {
                debug!("🔙 Rolling back {}: {}", participant_id, desc);
                Ok(())
            }
        }
    }

    /// Register rollback action for a participant
    pub async fn register_rollback(
        &self,
        tx_id: &str,
        participant_id: &str,
        action: RollbackAction,
    ) -> Result<()> {
        let tx = self.get_transaction(tx_id).await?;
        let mut tx_guard = tx.write().await;

        if let Some(participant) = tx_guard.participants.get_mut(participant_id) {
            participant.rollback_action = Some(action);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Participant {} not found in transaction {}",
                participant_id,
                tx_id
            ))
        }
    }

    /// Get transaction by ID
    async fn get_transaction(&self, tx_id: &str) -> Result<Arc<RwLock<ActiveTransaction>>> {
        self.transactions
            .get(tx_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))
    }

    /// Get transaction state
    pub async fn get_transaction_state(&self, tx_id: &str) -> Result<TransactionState> {
        let tx = self.get_transaction(tx_id).await?;
        let tx_guard = tx.read().await;
        Ok(tx_guard.state.clone())
    }

    /// Start background cleanup task for expired transactions
    fn start_cleanup_task(&self) {
        let transactions = self.transactions.clone();
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30)); // Optimized: 5s -> 30s

            loop {
                interval.tick().await;

                // Find and clean up expired transactions
                let mut expired = Vec::new();
                for entry in transactions.iter() {
                    let tx = entry.value().clone();
                    let is_expired = {
                        let tx_guard = tx.read().await;
                        tx_guard.is_expired()
                    };

                    if is_expired {
                        expired.push(entry.key().clone());
                    }
                }

                // Abort expired transactions
                for tx_id in expired {
                    warn!("🕐 Transaction {} expired, aborting", tx_id);
                    if let Some((_, tx)) = transactions.remove(&tx_id) {
                        let mut tx_guard = tx.write().await;
                        tx_guard.state =
                            TransactionState::Failed("Transaction expired".to_string());
                        // In production, we'd trigger rollback here
                    }
                }
            }
        });

        let cleanup_handle = self.cleanup_handle.clone();
        tokio::spawn(async move {
            *cleanup_handle.lock().await = Some(handle);
        });
    }

    // Private helper methods

    /// Build staging and final URLs for operation
    fn build_operation_urls(
        &self,
        config: &StagingConfig,
        operation_id: &OperationId,
    ) -> Result<(String, String)> {
        let base_url = config.base_url.trim_end_matches('/');

        trace!(
            "build_operation_urls: base={}, type={:?}, collection={:?}",
            base_url, config.operation_type, config.collection_id
        );

        // Build collection-specific path if provided
        let collection_path = if let Some(ref collection_id) = config.collection_id {
            format!("/collections/{}", collection_id)
        } else {
            String::new()
        };

        // Build staging directory name
        let staging_dir = config
            .custom_staging_dir
            .as_ref()
            .map_or_else(|| config.operation_type.staging_dir_name(), |s| s.as_str());

        // For metadata operations with custom staging dir containing path separators,
        // we need special handling
        let (staging_url, final_url) = if config.operation_type == TransactionStageType::Metadata
            && staging_dir.starts_with("../")
        {
            trace!("Using metadata with relative staging path");
            // For relative paths like "../staging", we need to resolve them properly
            // base_url is like "file:///path/to/metadata/current"
            // We want staging to be "file:///path/to/metadata/staging/{operation_id}"

            // Parse the base URL to extract scheme and path
            if let Some((scheme, path)) = base_url.split_once("://") {
                // Remove the last path component (e.g., "current")
                let parent_path = if let Some(last_slash) = path.rfind('/') {
                    &path[..last_slash]
                } else {
                    path
                };

                // Extract the staging directory name (remove "../")
                let staging_dirname = staging_dir.trim_start_matches("../");

                let staging_url = format!(
                    "{}://{}/{}/{}",
                    scheme, parent_path, staging_dirname, operation_id
                );
                let final_url = base_url.to_string();

                (staging_url, final_url)
            } else {
                // Fallback for non-URL paths
                let staging_url = format!("{}/{}/{}", base_url, staging_dir, operation_id);
                let final_url = base_url.to_string();
                (staging_url, final_url)
            }
        } else if config.operation_type == TransactionStageType::Metadata {
            let staging_url = format!("{}/{}/{}", base_url, staging_dir, operation_id);
            let final_url = base_url.to_string();
            (staging_url, final_url)
        } else {
            // Check if UUID subdirectory should be skipped (useful for compaction to avoid cleanup issues)
            if config.skip_uuid_subdir {
                let staging_url = format!("{}{}/{}", base_url, collection_path, staging_dir);
                let final_url = format!("{}{}", base_url, collection_path);
                (staging_url, final_url)
            } else {
                // Standard behavior: use UUID subdirectory for operation isolation
                let staging_url = format!(
                    "{}{}/{}/{}",
                    base_url, collection_path, staging_dir, operation_id
                );
                let final_url = format!("{}{}", base_url, collection_path);
                (staging_url, final_url)
            }
        };

        trace!("URLs: staging={}, final={}", staging_url, final_url);
        Ok((staging_url, final_url))
    }

    /// Generate staging URL for an operation
    fn generate_staging_url(&self, config: &StagingConfig, operation_id: &str) -> String {
        let base = &config.base_url;
        let staging_dir = match &config.operation_type {
            TransactionStageType::Flush => format!("__flush/{}", operation_id),
            TransactionStageType::Compaction => format!("__compact/{}", operation_id),
            TransactionStageType::Metadata => format!("__metadata_info/{}", operation_id),
            TransactionStageType::Wal => format!("__wal/{}", operation_id),
            TransactionStageType::Transaction => format!("__transaction/{}", operation_id),
            TransactionStageType::Custom(name) => format!("{}_{}", name, operation_id),
        };
        format!("{}/{}", base, staging_dir)
    }

    /// Generate final URL for an operation
    fn generate_final_url(&self, config: &StagingConfig) -> String {
        match &config.collection_id {
            Some(collection) => format!("{}/{}", config.base_url, collection),
            None => config.base_url.clone(),
        }
    }

    /// TD-TIER-1 SC5: apply `tier` to every object this operation publishes
    /// at finalize (a staged segment's final key must carry the collection's
    /// tier: the move is a copy, and a copy without options would land on the
    /// account default). Unknown operation ⇒ error, never a silent no-op.
    pub fn set_operation_access_tier(
        &self,
        operation_id: &OperationId,
        tier: Option<proximadb_storage_filesystem_types::ObjectAccessTier>,
    ) -> Result<()> {
        let mut entry = self
            .active_operations
            .get_mut(operation_id)
            .ok_or_else(|| anyhow::anyhow!("Operation not found: {}", operation_id))?;
        entry.access_tier = tier;
        Ok(())
    }

    /// Update operation status
    async fn update_operation_status(
        &self,
        operation_id: &OperationId,
        status: TransactionalOperationStatus,
    ) -> Result<()> {
        // Update using DashMap's entry API
        if let Some(mut entry) = self.active_operations.get_mut(operation_id) {
            entry.status = status;
        }
        Ok(())
    }
}

/// Handle for an active transaction
pub struct TransactionHandle<'a> {
    id: String,
    coordinator: &'a UnifiedTransactionCoordinator,
    #[allow(dead_code)]
    transaction: Arc<RwLock<ActiveTransaction>>,
}

impl<'a> TransactionHandle<'a> {
    /// Prepare the transaction (phase 1 of 2PC)
    pub async fn prepare(&self) -> Result<bool> {
        self.coordinator.prepare_transaction(&self.id).await
    }

    /// Commit the transaction (phase 2 of 2PC)
    pub async fn commit(&self) -> Result<()> {
        self.coordinator.commit_transaction(&self.id).await
    }

    /// Rollback the transaction
    pub async fn rollback(&self) -> Result<()> {
        self.coordinator.rollback_transaction(&self.id).await
    }

    /// Register a rollback action
    pub async fn register_rollback(
        &self,
        participant_id: &str,
        action: RollbackAction,
    ) -> Result<()> {
        self.coordinator
            .register_rollback(&self.id, participant_id, action)
            .await
    }

    /// Get transaction ID
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Generate unique transaction ID
pub fn generate_transaction_id(prefix: &str) -> String {
    let counter = TRANSACTION_COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => duration.as_millis(),
        Err(error) => {
            warn!(
                error = %error,
                "System clock is before Unix epoch while generating transaction ID; using timestamp 0"
            );
            0
        }
    };
    format!("{}_{}_{}", prefix, timestamp, counter)
}

/// Convenience wrapper for VIPER-specific atomic operations
pub struct ViperTransactionalOperations {
    coordinator: Arc<UnifiedTransactionCoordinator>,
}

impl ViperTransactionalOperations {
    pub fn new(coordinator: Arc<UnifiedTransactionCoordinator>) -> Self {
        Self { coordinator }
    }

    /// Begin flush operation for a collection
    pub async fn begin_flush_operation(
        &self,
        collection_id: &str,
        storage_url: &str,
    ) -> Result<TransactionalOperationMetadata> {
        let config = StagingConfig {
            base_url: storage_url.to_string(),
            collection_id: Some(collection_id.to_string()),
            operation_type: TransactionStageType::Flush,
            auto_cleanup: true,
            skip_uuid_subdir: true, // Avoid creating subdirectories that get left behind
            ..Default::default()
        };

        self.coordinator.begin_atomic_operation(&config).await
    }

    /// Write Parquet file to flush staging
    pub async fn write_parquet_to_staging(
        &self,
        operation_id: &OperationId,
        parquet_filename: &str,
        parquet_data: &[u8],
    ) -> Result<()> {
        let relative_path = format!("vectors/{}", parquet_filename);
        self.coordinator
            .write_to_staging(operation_id, &relative_path, parquet_data)
            .await
    }

    /// Finalize flush operation
    pub async fn finalize_flush(&self, operation_id: &OperationId) -> Result<()> {
        self.coordinator
            .finalize_atomic_operation(operation_id)
            .await
    }

    /// Abort flush operation
    pub async fn abort_flush(&self, operation_id: &OperationId, reason: &str) -> Result<()> {
        self.coordinator
            .abort_atomic_operation(operation_id, reason)
            .await
    }
}

/// Convenience wrapper for Write Buffer-specific atomic operations
pub struct WalTransactionalOperations {
    coordinator: Arc<UnifiedTransactionCoordinator>,
}

impl WalTransactionalOperations {
    pub fn new(coordinator: Arc<UnifiedTransactionCoordinator>) -> Self {
        Self { coordinator }
    }

    /// Begin WAL segment rotation operation
    pub async fn begin_segment_rotation(
        &self,
        write_buffer_url: &str,
    ) -> Result<TransactionalOperationMetadata> {
        let config = StagingConfig {
            base_url: write_buffer_url.to_string(),
            collection_id: None,
            operation_type: TransactionStageType::Wal,
            auto_cleanup: true,
            ..Default::default()
        };

        self.coordinator.begin_atomic_operation(&config).await
    }

    /// Write WAL segment to staging
    pub async fn write_segment_to_staging(
        &self,
        operation_id: &OperationId,
        segment_filename: &str,
        segment_data: &[u8],
    ) -> Result<()> {
        self.coordinator
            .write_to_staging(operation_id, segment_filename, segment_data)
            .await
    }

    /// Finalize segment rotation
    pub async fn finalize_rotation(&self, operation_id: &OperationId) -> Result<()> {
        self.coordinator
            .finalize_atomic_operation(operation_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::FilesystemConfig;
    use tempfile::TempDir;

    async fn create_test_coordinator() -> (UnifiedTransactionCoordinator, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_string_lossy().to_string();

        let fs_config = FilesystemConfig {
            default_fs: Some(format!("file://{}", temp_path)),
            ..Default::default()
        };

        let filesystem = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());
        let coordinator = UnifiedTransactionCoordinator::new(filesystem, None)
            .await
            .unwrap();

        (coordinator, temp_dir)
    }

    #[tokio::test]
    async fn test_atomic_operation_lifecycle() {
        let (coordinator, temp_dir) = create_test_coordinator().await;
        let temp_path = temp_dir.path().to_string_lossy().to_string();

        let config = StagingConfig {
            base_url: format!("file://{}", temp_path),
            collection_id: Some("test_collection".to_string()),
            operation_type: TransactionStageType::Flush,
            ..Default::default()
        };

        // Begin operation
        let metadata = coordinator.begin_atomic_operation(&config).await.unwrap();
        assert_eq!(metadata.status, TransactionalOperationStatus::Preparing);

        // Write to staging
        let test_data = b"test parquet data";
        coordinator
            .write_to_staging(&metadata.operation_id, "test.parquet", test_data)
            .await
            .unwrap();

        // Check status
        let status = coordinator
            .get_operation_status(&metadata.operation_id)
            .await
            .unwrap();
        assert_eq!(status, TransactionalOperationStatus::Staging);

        // Finalize operation
        coordinator
            .finalize_atomic_operation(&metadata.operation_id)
            .await
            .unwrap();

        // Operation should be removed from active operations
        assert!(
            coordinator
                .get_operation_status(&metadata.operation_id)
                .await
                .is_none()
        );
    }

    /// TD-TIER-1 SC5: a tiered operation still publishes on an untiered
    /// (local) backend, and tiering an unknown operation is an error, never a
    /// silent no-op.
    #[tokio::test]
    #[serial_test::serial]
    async fn access_tier_is_recorded_and_finalize_still_publishes() {
        // Serialized: finalize's tiered publish takes copy_atomic's same-backend
        // fast path only when the filesystem Arcs are identical — a concurrent
        // test arming the (per-call-minted) fault/counting wrappers would
        // reroute it to the streaming path, which refuses a tier.

        let (coordinator, temp_dir) = create_test_coordinator().await;
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        let config = StagingConfig {
            base_url: format!("file://{}", temp_path),
            collection_id: Some("tiered".to_string()),
            operation_type: TransactionStageType::Flush,
            ..Default::default()
        };
        let metadata = coordinator.begin_atomic_operation(&config).await.unwrap();
        assert_eq!(metadata.access_tier, None, "account default unless set");
        coordinator
            .set_operation_access_tier(
                &metadata.operation_id,
                Some(proximadb_storage_filesystem_types::ObjectAccessTier::Cool),
            )
            .unwrap();
        coordinator
            .write_to_staging(&metadata.operation_id, "L0.pax", b"segment")
            .await
            .unwrap();
        coordinator
            .finalize_atomic_operation(&metadata.operation_id)
            .await
            .unwrap();
        let published = std::fs::read(
            std::path::Path::new(metadata.final_url.trim_start_matches("file://")).join("L0.pax"),
        )
        .unwrap();
        assert_eq!(published, b"segment");
        assert!(
            coordinator
                .set_operation_access_tier(&"no-such-op".to_string(), None)
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_viper_atomic_operations() {
        let (coordinator, temp_dir) = create_test_coordinator().await;
        let temp_path = temp_dir.path().to_string_lossy().to_string();
        let coordinator = Arc::new(coordinator);

        let viper_ops = ViperTransactionalOperations::new(coordinator);

        // Begin flush operation
        let metadata = viper_ops
            .begin_flush_operation("test_collection", &format!("file://{}", temp_path))
            .await
            .unwrap();

        // Write Parquet data
        let parquet_data = b"mock parquet data";
        viper_ops
            .write_parquet_to_staging(&metadata.operation_id, "batch_001.parquet", parquet_data)
            .await
            .unwrap();

        // Finalize flush
        viper_ops
            .finalize_flush(&metadata.operation_id)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_operation_abort() {
        let (coordinator, temp_dir) = create_test_coordinator().await;
        let temp_path = temp_dir.path().to_string_lossy().to_string();

        let config = StagingConfig {
            base_url: format!("file://{}", temp_path),
            operation_type: TransactionStageType::Flush,
            ..Default::default()
        };

        // Begin operation
        let metadata = coordinator.begin_atomic_operation(&config).await.unwrap();

        // Write some data
        coordinator
            .write_to_staging(&metadata.operation_id, "test.dat", b"test")
            .await
            .unwrap();

        // Abort operation
        coordinator
            .abort_atomic_operation(&metadata.operation_id, "test abort")
            .await
            .unwrap();

        // Operation should be removed
        assert!(
            coordinator
                .get_operation_status(&metadata.operation_id)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_acid_transaction_lifecycle() {
        let (coordinator, _temp_dir) = create_test_coordinator().await;

        // Begin transaction
        let tx = coordinator
            .begin_transaction(
                "test_tx_1",
                vec!["memtable".to_string(), "disk".to_string()],
            )
            .await
            .unwrap();

        // Register rollback actions
        tx.register_rollback(
            "memtable",
            RollbackAction::RemoveFromMemtable {
                key: "test_key".to_string(),
            },
        )
        .await
        .unwrap();

        // Prepare
        let prepared = tx.prepare().await.unwrap();
        assert!(prepared);

        // Commit
        tx.commit().await.unwrap();

        // Verify state
        let state = coordinator
            .get_transaction_state("test_tx_1")
            .await
            .unwrap();
        assert_eq!(state, TransactionState::Committed);
    }

    #[tokio::test]
    async fn test_acid_transaction_rollback() {
        let (coordinator, _temp_dir) = create_test_coordinator().await;

        // Begin transaction
        let tx = coordinator
            .begin_transaction("test_tx_2", vec!["memtable".to_string()])
            .await
            .unwrap();

        // Register rollback action
        tx.register_rollback(
            "memtable",
            RollbackAction::RestoreMemtableValue {
                key: "test_key".to_string(),
                previous_value: vec![1, 2, 3],
            },
        )
        .await
        .unwrap();

        // Prepare
        tx.prepare().await.unwrap();

        // Rollback instead of commit
        tx.rollback().await.unwrap();

        // Transaction should be removed after rollback
        let result = coordinator.get_transaction_state("test_tx_2").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_transaction_id() {
        let id1 = generate_transaction_id("collection");
        let id2 = generate_transaction_id("collection");

        // IDs should be unique
        assert_ne!(id1, id2);

        // IDs should follow expected format
        assert!(id1.starts_with("collection_"));
        assert!(id2.starts_with("collection_"));
    }
}
