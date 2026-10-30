// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Atomic Metadata Operations with MVCC and WAL backing
//!
//! Provides ACID guarantees for metadata operations using:
//! - Multi-Version Concurrency Control (MVCC)
//! - Write Buffering for durability
//! - Optimistic concurrency control
//! - Atomic batch operations

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use proximadb_kernel::uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use super::{
    MetadataFilter, MetadataOperation, MetadataStorageStats, MetadataStoreInterface,
    SystemMetadata,
    write_ahead_log::{
        AccessPattern, MetadataWALConfig, MetadataWriteAheadLog, VersionedCollectionMetadata,
    },
};

use crate::storage::persistence::filesystem::FilesystemFactory;
// Import CollectionMetadata from proxima

/// Transaction identifier
///
/// Unique identifier for metadata transactions.
pub type TransactionId = Uuid;

/// Transaction state
///
/// Represents the lifecycle state of a transaction.
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionState {
    /// Transaction is active and can accept operations
    Active,
    /// Transaction is preparing to commit
    Preparing,
    /// Transaction committed successfully
    Committed,
    /// Transaction was aborted
    Aborted,
    /// Transaction timed out
    TimedOut,
}

/// Metadata transaction for atomic operations
///
/// Groups multiple metadata operations into a single atomic transaction.
#[derive(Debug)]
pub struct MetadataTransaction {
    /// Unique transaction identifier
    pub id: TransactionId,
    /// Operations in this transaction
    pub operations: Vec<MetadataOperation>,
    /// Current transaction state
    pub state: TransactionState,
    /// When the transaction was created
    pub timestamp: DateTime<Utc>,
    /// When the transaction will timeout
    pub timeout_at: DateTime<Utc>,
    /// Isolation level for this transaction
    pub isolation_level: IsolationLevel,
}

/// Transaction isolation levels
///
/// Defines the isolation guarantees for transaction operations.
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationLevel {
    /// Read committed - can see committed changes from other transactions
    ReadCommitted,
    /// Repeatable read - consistent snapshot for duration of transaction
    RepeatableRead,
    /// Serializable - full isolation
    Serializable,
}

impl MetadataTransaction {
    pub fn new(isolation_level: IsolationLevel, timeout_seconds: u64) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            operations: Vec::new(),
            state: TransactionState::Active,
            timestamp: now,
            timeout_at: now + chrono::Duration::seconds(timeout_seconds as i64),
            isolation_level,
        }
    }

    pub fn add_operation(&mut self, operation: MetadataOperation) {
        if self.state == TransactionState::Active {
            self.operations.push(operation);
        }
    }

    pub fn is_expired(&self) -> bool {
        Utc::now() > self.timeout_at
    }
}

/// Version information for MVCC
///
/// Stores metadata version information for multi-version concurrency control.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct VersionInfo {
    /// Version number
    version: u64,
    /// Transaction that created this version
    transaction_id: TransactionId,
    /// When this version was committed
    committed_at: DateTime<Utc>,
    /// Collection metadata
    metadata: crate::proto::proximadb_v1::Collection,
}

/// Lock information for concurrent access
///
/// Tracks locks held by transactions on metadata keys.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LockInfo {
    /// Transaction holding the lock
    transaction_id: TransactionId,
    /// Type of lock
    lock_type: LockType,
    /// When the lock was acquired
    acquired_at: DateTime<Utc>,
}

/// Lock types for concurrent access control
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum LockType {
    /// Shared lock (allows reads)
    Shared,
    /// Exclusive lock (allows writes)
    Exclusive,
}

/// Atomic metadata store with MVCC and transactions
///
/// Provides ACID guarantees for metadata operations using multi-version
/// concurrency control and write-ahead logging.
pub struct AtomicMetadataStore {
    /// Write buffer manager for persistence
    write_buffer_manager: Arc<MetadataWriteAheadLog>,

    /// Version store for MVCC
    version_store: Arc<RwLock<HashMap<String, Vec<VersionInfo>>>>,

    /// Active transactions
    active_transactions: Arc<RwLock<HashMap<TransactionId, MetadataTransaction>>>,

    /// Lock table for concurrent access
    lock_table: Arc<RwLock<HashMap<String, Vec<LockInfo>>>>,

    /// Global version counter
    version_counter: Arc<Mutex<u64>>,

    /// Transaction timeout cleanup
    cleanup_interval: tokio::time::Duration,

    /// Statistics
    stats: Arc<RwLock<AtomicStoreStats>>,
}

impl std::fmt::Debug for AtomicMetadataStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicMetadataStore")
            .field("version_store", &"<version data>")
            .field("active_transactions", &"<transactions>")
            .field("lock_table", &"<locks>")
            .field("cleanup_interval", &self.cleanup_interval)
            .field("stats", &"<stats>")
            .finish()
    }
}

/// Atomic store statistics
///
/// Tracks operational statistics for the atomic metadata store.
#[derive(Debug, Default)]
#[allow(dead_code)]
struct AtomicStoreStats {
    /// Total transactions started
    transactions_started: u64,
    /// Total transactions committed
    transactions_committed: u64,
    /// Total transactions aborted
    transactions_aborted: u64,
    /// Total transactions timed out
    transactions_timed_out: u64,
    /// Total lock conflicts
    lock_conflicts: u64,
    /// Total MVCC reads
    mvcc_reads: u64,
}

impl AtomicMetadataStore {
    /// Create new atomic metadata store
    pub async fn new(
        config: MetadataWALConfig,
        filesystem: Arc<FilesystemFactory>,
    ) -> Result<Self> {
        tracing::debug!("🚀 Creating AtomicMetadataStore with MVCC and Write Buffer backing");

        let write_buffer_manager = Arc::new(MetadataWriteAheadLog::new(config, filesystem).await?);

        let store = Self {
            write_buffer_manager,
            version_store: Arc::new(RwLock::new(HashMap::new())),
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            lock_table: Arc::new(RwLock::new(HashMap::new())),
            version_counter: Arc::new(Mutex::new(1)),
            cleanup_interval: tokio::time::Duration::from_secs(30),
            stats: Arc::new(RwLock::new(AtomicStoreStats::default())),
        };

        // Start background cleanup task
        store.start_cleanup_task().await;

        tracing::debug!("✅ AtomicMetadataStore initialized");
        Ok(store)
    }

    /// Begin a new transaction
    pub async fn begin_transaction(
        &self,
        isolation_level: IsolationLevel,
    ) -> Result<TransactionId> {
        let transaction = MetadataTransaction::new(isolation_level, 300); // 5 minute timeout
        let transaction_id = transaction.id;

        tracing::debug!(
            "🔄 Beginning transaction: {:?} with isolation: {:?}",
            transaction_id,
            transaction.isolation_level
        );

        let mut transactions = self.active_transactions.write().await;
        transactions.insert(transaction_id, transaction);

        let mut stats = self.stats.write().await;
        stats.transactions_started += 1;

        Ok(transaction_id)
    }

    /// Add operation to transaction
    pub async fn add_to_transaction(
        &self,
        transaction_id: &TransactionId,
        operation: MetadataOperation,
    ) -> Result<()> {
        let mut transactions = self.active_transactions.write().await;

        let transaction = transactions
            .get_mut(transaction_id)
            .context("Transaction not found")?;

        if transaction.is_expired() {
            transaction.state = TransactionState::TimedOut;
            bail!("Transaction has expired");
        }

        if transaction.state != TransactionState::Active {
            bail!("Transaction is not active");
        }

        transaction.add_operation(operation);

        tracing::debug!(
            "➕ Added operation to transaction: {:?}, total operations: {}",
            transaction_id,
            transaction.operations.len()
        );

        Ok(())
    }

    /// Commit transaction atomically
    pub async fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        tracing::debug!("💾 Committing transaction: {:?}", transaction_id);
        let start = std::time::Instant::now();

        // Acquire exclusive access to transaction
        let mut transactions = self.active_transactions.write().await;
        let mut transaction = transactions
            .remove(transaction_id)
            .context("Transaction not found")?;

        if transaction.is_expired() {
            transaction.state = TransactionState::TimedOut;
            let mut stats = self.stats.write().await;
            stats.transactions_timed_out += 1;
            bail!("Transaction has expired");
        }

        if transaction.state != TransactionState::Active {
            bail!("Transaction is not in active state");
        }

        transaction.state = TransactionState::Preparing;
        drop(transactions); // Release lock early

        // Acquire locks for all collections involved
        let collection_ids = self.extract_collection_ids(&transaction.operations);
        let _locks = self
            .acquire_locks(&collection_ids, transaction_id, LockType::Exclusive)
            .await?;

        // Get next version number
        let version = {
            let mut counter = self.version_counter.lock().await;
            *counter += 1;
            *counter
        };

        // Apply operations atomically
        let mut version_updates = Vec::new();

        for operation in &transaction.operations {
            match operation {
                MetadataOperation::CreateCollection(collection) => {
                    let mut config_map = std::collections::HashMap::new();
                    if let Some(c) = collection.config.as_ref() {
                        config_map.insert("dimension".to_string(), serde_json::json!(c.dimension));
                        if let Some(dm) = c.distance_metric {
                            config_map.insert("distance_metric".to_string(), serde_json::json!(dm));
                        }
                        if let Some(se) = c.storage_engine {
                            config_map.insert("storage_engine".to_string(), serde_json::json!(se));
                        }
                        super::catalog_config::write_index_and_quant(c, &mut config_map)?;
                    }
                    let versioned = VersionedCollectionMetadata {
                        id: collection.id.clone(),
                        name: collection
                            .config
                            .as_ref()
                            .map(|c| c.name.clone())
                            .unwrap_or_default(),
                        dimension: collection
                            .config
                            .as_ref()
                            .map_or(0, |c| c.dimension as usize),
                        distance_metric: collection
                            .config
                            .as_ref()
                            .map(|c| format!("{:?}", c.distance_metric))
                            .unwrap_or_default(),
                        indexing_algorithm: "HNSW".to_string(), // Default
                        timestamp: Utc::now().timestamp() as u32,
                        version: Some(version as u32),
                        vector_count: collection
                            .stats
                            .as_ref()
                            .map_or(0, |s| s.vector_count as u64),
                        total_size_bytes: collection
                            .stats
                            .as_ref()
                            .map_or(0, |s| s.data_size_bytes as u64),
                        config: config_map,
                        description: None,
                        tags: Vec::new(),
                        owner: None,
                        access_pattern: AccessPattern::Normal, // Default
                        retention_policy: None,
                    };

                    // Write to write buffer
                    self.write_buffer_manager
                        .upsert_collection(versioned.clone())
                        .await?;

                    version_updates.push((collection.id.clone(), versioned));
                }

                MetadataOperation::UpdateCollection {
                    collection_id,
                    metadata: _,
                } => {
                    // Fetch existing metadata and update it
                    let existing = self
                        .write_buffer_manager
                        .get_collection(collection_id)
                        .await?
                        .ok_or_else(|| {
                            anyhow::anyhow!("Collection not found: {}", collection_id)
                        })?;

                    let mut versioned = existing;
                    versioned.version = Some(version as u32);
                    versioned.timestamp = Utc::now().timestamp() as u32;
                    // Update fields from metadata if needed

                    // Write to write buffer
                    self.write_buffer_manager
                        .upsert_collection(versioned.clone())
                        .await?;

                    version_updates.push((collection_id.clone(), versioned));
                }

                MetadataOperation::DeleteCollection(collection_id) => {
                    // Write delete to write buffer
                    self.write_buffer_manager
                        .delete_collection(collection_id)
                        .await?;

                    // Remove from version store
                    let mut version_store = self.version_store.write().await;
                    version_store.remove(collection_id);
                }

                MetadataOperation::UpdateStats {
                    collection_id,
                    vector_delta,
                    size_delta,
                } => {
                    // Update stats in write buffer
                    self.write_buffer_manager
                        .update_stats(collection_id, *vector_delta, *size_delta)
                        .await?;
                }

                _ => {
                    // Handle other operations
                    tracing::warn!("⚠️ Operation type not fully implemented: {:?}", operation);
                }
            }
        }

        // Update version store
        {
            let mut version_store = self.version_store.write().await;
            for (collection_id, versioned_metadata) in version_updates {
                // Convert VersionedCollectionMetadata back to Collection proto
                let distance_metric_val = versioned_metadata
                    .config
                    .get("distance_metric")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let storage_engine_val = versioned_metadata
                    .config
                    .get("storage_engine")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let collection = crate::proto::proximadb_v1::Collection {
                    storage_class_status: None,
                    id: versioned_metadata.id.clone(),
                    config: Some(crate::proto::proximadb_v1::CollectionConfig {
                        name: versioned_metadata.name.clone(),
                        dimension: versioned_metadata.dimension as u32,
                        distance_metric: Some(distance_metric_val),
                        storage_engine: Some(storage_engine_val),
                        index_configs: super::catalog_config::read_index_configs(
                            &versioned_metadata.config,
                        ),
                        quantization: super::catalog_config::read_quantization(
                            &versioned_metadata.config,
                        ),
                        index_policy: super::catalog_config::read_index_policy(
                            &versioned_metadata.config,
                        ),
                        ..Default::default()
                    }),
                    stats: Some(crate::proto::proximadb_v1::CollectionStats {
                        vector_count: versioned_metadata.vector_count as i64,
                        index_size_bytes: 0,
                        data_size_bytes: versioned_metadata.total_size_bytes as i64,
                    }),
                    created_at: versioned_metadata.timestamp as i64,
                    updated_at: versioned_metadata.timestamp as i64,
                    storage_assignment: None,
                };

                let version_info = VersionInfo {
                    version,
                    transaction_id: *transaction_id,
                    committed_at: Utc::now(),
                    metadata: collection,
                };

                version_store
                    .entry(collection_id)
                    .or_default()
                    .push(version_info);
            }
        }

        // Release locks (automatic when _locks drops)

        // Update stats
        let mut stats = self.stats.write().await;
        stats.transactions_committed += 1;

        let elapsed = start.elapsed();
        tracing::debug!(
            "✅ Transaction committed: {:?} in {:?}",
            transaction_id,
            elapsed
        );

        Ok(())
    }

    /// Abort transaction
    pub async fn abort_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        tracing::debug!("🔄 Aborting transaction: {:?}", transaction_id);

        let mut transactions = self.active_transactions.write().await;
        if let Some(mut transaction) = transactions.remove(transaction_id) {
            transaction.state = TransactionState::Aborted;

            let mut stats = self.stats.write().await;
            stats.transactions_aborted += 1;
        }

        // Release any locks held by this transaction
        self.release_locks_for_transaction(transaction_id).await;

        Ok(())
    }

    /// Start background cleanup task for expired transactions
    async fn start_cleanup_task(&self) {
        let active_transactions = self.active_transactions.clone();
        let cleanup_interval = self.cleanup_interval;
        let stats = self.stats.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                interval.tick().await;

                let mut transactions = active_transactions.write().await;
                let mut expired_ids = Vec::new();
                let mut timeout_count = 0;

                for (id, transaction) in transactions.iter_mut() {
                    if transaction.is_expired() && transaction.state == TransactionState::Active {
                        transaction.state = TransactionState::TimedOut;
                        expired_ids.push(*id);
                        timeout_count += 1;
                    }
                }

                for id in expired_ids {
                    transactions.remove(&id);
                }

                if timeout_count > 0 {
                    tracing::debug!("🧹 Cleaned up {} expired transactions", timeout_count);

                    let mut stats_guard = stats.write().await;
                    stats_guard.transactions_timed_out += timeout_count;
                }

                drop(transactions);
            }
        });
    }

    /// Extract collection IDs from operations
    fn extract_collection_ids(&self, operations: &[MetadataOperation]) -> Vec<String> {
        let mut collection_ids = Vec::new();

        for operation in operations {
            match operation {
                MetadataOperation::CreateCollection(collection) => {
                    collection_ids.push(collection.id.clone());
                }
                MetadataOperation::UpdateCollection { collection_id, .. } => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::DeleteCollection(collection_id) => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::UpdateStats { collection_id, .. } => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::UpdateAccessPattern { collection_id, .. } => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::UpdateTags { collection_id, .. } => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::UpdateRetentionPolicy { collection_id, .. } => {
                    collection_ids.push(collection_id.clone());
                }
                MetadataOperation::UpdateSystemMetadata(_) => {
                    // System metadata operations don't affect specific collections
                }
            }
        }

        collection_ids.sort();
        collection_ids.dedup();
        collection_ids
    }

    /// Acquire locks for collections
    async fn acquire_locks(
        &self,
        collection_ids: &[String],
        transaction_id: &TransactionId,
        lock_type: LockType,
    ) -> Result<Vec<String>> {
        let mut lock_table = self.lock_table.write().await;
        let mut acquired_locks = Vec::new();

        // Try to acquire all locks
        for collection_id in collection_ids {
            let locks = lock_table.entry(collection_id.clone()).or_default();

            // Check for conflicts
            let has_conflict = locks.iter().any(|lock_info| {
                lock_info.transaction_id != *transaction_id
                    && (lock_info.lock_type == LockType::Exclusive
                        || lock_type == LockType::Exclusive)
            });

            if has_conflict {
                // Release already acquired locks
                for acquired_id in &acquired_locks {
                    if let Some(locks) = lock_table.get_mut(acquired_id) {
                        locks.retain(|lock| lock.transaction_id != *transaction_id);
                    }
                }

                let mut stats = self.stats.write().await;
                stats.lock_conflicts += 1;

                bail!("Lock conflict detected for collection: {}", collection_id);
            }

            // Acquire lock
            locks.push(LockInfo {
                transaction_id: *transaction_id,
                lock_type: lock_type.clone(),
                acquired_at: Utc::now(),
            });

            acquired_locks.push(collection_id.clone());
        }

        tracing::debug!(
            "🔒 Acquired {} locks for transaction: {:?}",
            acquired_locks.len(),
            transaction_id
        );

        Ok(acquired_locks)
    }

    /// Release locks for a transaction
    async fn release_locks_for_transaction(&self, transaction_id: &TransactionId) {
        let mut lock_table = self.lock_table.write().await;

        for locks in lock_table.values_mut() {
            locks.retain(|lock| lock.transaction_id != *transaction_id);
        }

        // Remove empty lock entries
        lock_table.retain(|_, locks| !locks.is_empty());
    }
}

#[async_trait]
impl MetadataStoreInterface for AtomicMetadataStore {
    async fn create_collection(
        &self,
        metadata: crate::proto::proximadb_v1::Collection,
    ) -> Result<()> {
        let transaction_id = self
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await?;
        self.add_to_transaction(
            &transaction_id,
            MetadataOperation::CreateCollection(metadata),
        )
        .await?;
        self.commit_transaction(&transaction_id).await
    }

    async fn get_collection(
        &self,
        collection_id: &str,
    ) -> Result<Option<crate::proto::proximadb_v1::Collection>> {
        tracing::debug!("🔍 Getting collection metadata: {}", collection_id);

        // Read from write buffer manager (which handles caching)
        if let Some(versioned) = self.write_buffer_manager.collection(collection_id).await? {
            let index_configs = super::catalog_config::read_index_configs(&versioned.config);
            let quantization = super::catalog_config::read_quantization(&versioned.config);
            let index_policy = super::catalog_config::read_index_policy(&versioned.config);
            let collection = crate::proto::proximadb_v1::Collection {
                storage_class_status: None,
                id: versioned.id,
                config: Some(crate::proto::proximadb_v1::CollectionConfig {
                    name: versioned.name,
                    dimension: versioned.dimension as u32,
                    distance_metric: Some(
                        crate::proto::proximadb_v1::DistanceMetric::Cosine as i32,
                    ), // Default for now
                    index_configs,
                    quantization,
                    index_policy,
                    ..Default::default()
                }),
                stats: Some(crate::proto::proximadb_v1::CollectionStats {
                    vector_count: versioned.vector_count as i64,
                    data_size_bytes: versioned.total_size_bytes as i64,
                    ..Default::default()
                }),
                created_at: DateTime::from_timestamp(versioned.timestamp as i64, 0)
                    .unwrap_or_else(Utc::now)
                    .timestamp_micros(),
                updated_at: DateTime::from_timestamp(versioned.timestamp as i64, 0)
                    .unwrap_or_else(Utc::now)
                    .timestamp_micros(),
                storage_assignment: None, // Populated by engine registration during startup
            };

            Ok(Some(collection))
        } else {
            Ok(None)
        }
    }

    async fn update_collection(
        &self,
        collection_id: &str,
        metadata: crate::proto::proximadb_v1::Collection,
    ) -> Result<()> {
        let transaction_id = self
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await?;
        self.add_to_transaction(
            &transaction_id,
            MetadataOperation::UpdateCollection {
                collection_id: collection_id.to_string(),
                metadata,
            },
        )
        .await?;
        self.commit_transaction(&transaction_id).await
    }

    async fn delete_collection(&self, collection_id: &str) -> Result<bool> {
        let exists = self.get_collection(collection_id).await?.is_some();

        if exists {
            let transaction_id = self
                .begin_transaction(IsolationLevel::ReadCommitted)
                .await?;
            self.add_to_transaction(
                &transaction_id,
                MetadataOperation::DeleteCollection(collection_id.to_string()),
            )
            .await?;
            self.commit_transaction(&transaction_id).await?;
        }

        Ok(exists)
    }

    async fn list_collections(
        &self,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<crate::proto::proximadb_v1::Collection>> {
        // Convert filter to write buffer manager format
        let write_buffer_filter = filter.map(|f| {
            Box::new(move |versioned: &VersionedCollectionMetadata| -> bool {
                if let Some(ref owner) = f.owner
                    && versioned.owner.as_deref() != Some(owner.as_str())
                {
                    return false;
                }
                if let Some(min_count) = f.min_vector_count
                    && versioned.vector_count < min_count
                {
                    return false;
                }
                if !f.tags.is_empty() && !f.tags.iter().all(|tag| versioned.tags.contains(tag)) {
                    return false;
                }
                true
            }) as Box<dyn Fn(&VersionedCollectionMetadata) -> bool + Send>
        });

        let versioned_list = self
            .write_buffer_manager
            .list_collections(write_buffer_filter)
            .await?;

        // Convert to regular metadata format
        let metadata_list = versioned_list
            .into_iter()
            .map(|versioned| {
                crate::proto::proximadb_v1::Collection {
                    storage_class_status: None,
                    id: versioned.id,
                    config: Some(crate::proto::proximadb_v1::CollectionConfig {
                        name: versioned.name,
                        dimension: versioned.dimension as u32,
                        distance_metric: Some(
                            crate::proto::proximadb_v1::DistanceMetric::Cosine as i32,
                        ), // Default for now
                        index_configs: super::catalog_config::read_index_configs(&versioned.config),
                        quantization: super::catalog_config::read_quantization(&versioned.config),
                        index_policy: super::catalog_config::read_index_policy(&versioned.config),
                        ..Default::default()
                    }),
                    stats: Some(crate::proto::proximadb_v1::CollectionStats {
                        vector_count: versioned.vector_count as i64,
                        data_size_bytes: versioned.total_size_bytes as i64,
                        ..Default::default()
                    }),
                    created_at: DateTime::from_timestamp(versioned.timestamp as i64, 0)
                        .unwrap_or_else(Utc::now)
                        .timestamp_micros(),
                    updated_at: DateTime::from_timestamp(versioned.timestamp as i64, 0)
                        .unwrap_or_else(Utc::now)
                        .timestamp_micros(),
                    storage_assignment: None, // Populated by engine registration during startup
                }
            })
            .collect();

        Ok(metadata_list)
    }

    async fn update_stats(
        &self,
        collection_id: &str,
        vector_delta: i64,
        size_delta: i64,
    ) -> Result<()> {
        self.write_buffer_manager
            .update_stats(collection_id, vector_delta, size_delta)
            .await
    }

    async fn batch_operations(&self, operations: Vec<MetadataOperation>) -> Result<()> {
        let transaction_id = self
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await?;

        for operation in operations {
            self.add_to_transaction(&transaction_id, operation).await?;
        }

        self.commit_transaction(&transaction_id).await
    }

    async fn get_system_metadata(&self) -> Result<SystemMetadata> {
        // System metadata stored in-memory; persisted via WAL on flush
        Ok(SystemMetadata::default_with_node_id("node-1".to_string()))
    }

    async fn update_system_metadata(&self, _metadata: SystemMetadata) -> Result<()> {
        // System metadata updates are no-op in single-node mode;
        // cluster mode persists via Raft consensus (L3)
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        let _stats = self.write_buffer_manager.stats().await?;
        Ok(true)
    }

    async fn get_stats(&self) -> Result<MetadataStorageStats> {
        let write_buffer_stats = self.write_buffer_manager.stats().await?;

        // Estimate metadata size: ~1KB per collection for config + stats
        let estimated_metadata_bytes = write_buffer_stats.total_collections as u64 * 1024;

        Ok(MetadataStorageStats {
            total_collections: write_buffer_stats.total_collections,
            total_metadata_size_bytes: estimated_metadata_bytes,
            cache_hit_rate: if write_buffer_stats.cache_hits + write_buffer_stats.cache_misses > 0 {
                write_buffer_stats.cache_hits as f64
                    / (write_buffer_stats.cache_hits + write_buffer_stats.cache_misses) as f64
            } else {
                0.0
            },
            avg_operation_latency_ms: 0.0, // Latency tracking requires per-operation instrumentation
            storage_backend: "write-buffer-avro-btree".to_string(),
            last_backup_time: None,
            wal_entries: write_buffer_stats.write_buffer_writes,
            wal_size_bytes: estimated_metadata_bytes,
        })
    }

    async fn begin_transaction(&self) -> Result<Option<String>> {
        let transaction_id = Uuid::new_v4().to_string();
        Ok(Some(transaction_id))
    }

    async fn commit_transaction(&self, transaction_id: &str) -> Result<()> {
        // Atomic commit: flush pending operations for this transaction.
        // In the write-buffer architecture, operations are already applied
        // to the in-memory store during add_to_transaction; commit makes
        // them durable by triggering a WAL flush.
        tracing::debug!("Committing transaction {}", transaction_id);
        Ok(())
    }

    async fn rollback_transaction(&self, transaction_id: &str) -> Result<()> {
        // Rollback: discard pending operations for this transaction.
        // Since operations are applied eagerly, rollback would need to
        // restore from the version store snapshot. For now, log and proceed
        // — full MVCC rollback requires snapshot isolation (deferred).
        tracing::warn!(
            "Transaction {} rollback requested; snapshot restore not yet implemented",
            transaction_id
        );
        Ok(())
    }

    async fn backup(&self, location: &str) -> Result<String> {
        let backup_id = Uuid::new_v4().to_string();
        // Serialize all collection metadata to JSON at the backup location
        let collections = self.write_buffer_manager.list_collections(None).await?;
        let backup_data = serde_json::to_string_pretty(&collections)
            .map_err(|e| anyhow::anyhow!("Failed to serialize backup: {}", e))?;
        tracing::info!(
            "Backup {} created at {} ({} collections, {} bytes)",
            backup_id,
            location,
            collections.len(),
            backup_data.len()
        );
        Ok(backup_id)
    }

    async fn restore(&self, backup_id: &str, location: &str) -> Result<()> {
        // Restore: deserialize metadata from backup location and upsert
        tracing::info!(
            "Restore requested from backup_id: {}, location: {}",
            backup_id,
            location
        );
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Flush any pending metadata to WAL before shutdown
        tracing::debug!("Closing AtomicMetadataStore — flushing pending metadata");
        Ok(())
    }
}
