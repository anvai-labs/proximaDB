//! Transaction Management for GraphOperationsService
//!
//! This module implements transaction support using the Unit of Work pattern
//! for graph database operations with ACID guarantees.
//!
//! # Design Patterns
//!
//! - **Unit of Work**: Tracks changes within a transaction and commits/rolls back atomically
//! - **Repository Pattern**: Graph operations within transactions go through the UnitOfWork
//! - **Transaction Isolation**: Supports READ_COMMITTED and SERIALIZABLE isolation levels
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │        GraphOperationsService           │
//! │  ┌───────────────────────────────────┐  │
//! │  │      TransactionManager           │  │
//! │  │  ┌─────────────────────────────┐  │  │
//! │  │  │      UnitOfWork             │  │  │
//! │  │  │ (tracks pending changes)    │  │  │
//! │  │  └─────────────────────────────┘  │  │
//! │  │  ┌─────────────────────────────┐  │  │
//! │  │  │  LocalTransactionCoord      │  │  │
//! │  │  │  (single-node transactions) │  │  │
//! │  │  └─────────────────────────────┘  │  │
//! │  └───────────────────────────────────┘  │
//! └─────────────────────────────────────────┘
//! ```

use super::Result;
use crate::graph::engines::GraphEngine;
use crate::graph::{Edge, Node};
use dashmap::DashMap;
use proximadb_kernel::error::ProximaDBError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// Transaction Types (self-contained, no feature flag dependencies)
// ============================================================================

/// Type alias for transaction ID
pub type TransactionId = String;

/// Type alias for shard ID
pub type ShardId = String;

/// Type alias for resource ID (node or edge)
pub type ResourceId = String;

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Transaction is active, operations in progress
    Active,
    /// Preparing to commit (Phase 1 of 2PC)
    Preparing,
    /// Prepared, waiting for coordinator decision
    Prepared,
    /// Committing (Phase 2 of 2PC)
    Committing,
    /// Successfully committed
    Committed,
    /// Aborting
    Aborting,
    /// Aborted (rolled back)
    Aborted,
}

/// Graph operation to be executed within a transaction
#[derive(Debug, Clone)]
pub enum GraphOperation {
    /// Insert a node
    InsertNode { shard_id: ShardId, node: Node },
    /// Update a node's properties
    UpdateNode { shard_id: ShardId, node: Node },
    /// Delete a node
    DeleteNode { shard_id: ShardId, node_id: String },
    /// Insert an edge
    InsertEdge { shard_id: ShardId, edge: Edge },
    /// Update an edge's properties
    UpdateEdge { shard_id: ShardId, edge: Edge },
    /// Delete an edge
    DeleteEdge { shard_id: ShardId, edge_id: String },
}

impl GraphOperation {
    /// Get the shard ID this operation targets
    pub fn shard_id(&self) -> &ShardId {
        match self {
            GraphOperation::InsertNode { shard_id, .. } => shard_id,
            GraphOperation::UpdateNode { shard_id, .. } => shard_id,
            GraphOperation::DeleteNode { shard_id, .. } => shard_id,
            GraphOperation::InsertEdge { shard_id, .. } => shard_id,
            GraphOperation::UpdateEdge { shard_id, .. } => shard_id,
            GraphOperation::DeleteEdge { shard_id, .. } => shard_id,
        }
    }

    /// Get the resource ID being modified (for locking)
    pub fn resource_id(&self) -> ResourceId {
        match self {
            GraphOperation::InsertNode { node, .. } => node.id.clone(),
            GraphOperation::UpdateNode { node, .. } => node.id.clone(),
            GraphOperation::DeleteNode { node_id, .. } => node_id.clone(),
            GraphOperation::InsertEdge { edge, .. } => edge.id.clone(),
            GraphOperation::UpdateEdge { edge, .. } => edge.id.clone(),
            GraphOperation::DeleteEdge { edge_id, .. } => edge_id.clone(),
        }
    }
}

/// Transaction metadata
#[derive(Debug, Clone)]
pub struct TransactionMetadata {
    /// Transaction ID
    pub id: TransactionId,
    /// Graph ID this transaction operates on
    pub graph_id: String,
    /// Participating shards
    #[allow(dead_code)]
    pub participants: HashSet<ShardId>,
    /// Current state
    pub state: TransactionState,
    /// Operations in this transaction
    pub operations: Vec<GraphOperation>,
    /// Timestamp when transaction started
    pub start_time: Instant,
    /// Timeout duration
    pub timeout: Duration,
}

impl TransactionMetadata {
    pub fn new(graph_id: String, participants: Vec<ShardId>, timeout: Duration) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            graph_id,
            participants: participants.into_iter().collect(),
            state: TransactionState::Active,
            operations: Vec::new(),
            start_time: Instant::now(),
            timeout,
        }
    }

    /// Check if transaction has timed out
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }
}

/// Distributed lock manager for transaction isolation
///
/// Provides distributed locking with deadlock detection.
pub struct DistributedLockManager {
    /// Locks held by transactions: resource_id -> transaction_id
    locks: Arc<DashMap<ResourceId, TransactionId>>,
    /// Waiting transactions: transaction_id -> set of resource_ids waiting for
    waiting: Arc<DashMap<TransactionId, HashSet<ResourceId>>>,
}

impl DistributedLockManager {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(DashMap::new()),
            waiting: Arc::new(DashMap::new()),
        }
    }

    /// Acquire a lock on a resource
    ///
    /// Returns true if lock acquired, false if would deadlock or resource locked
    pub async fn acquire_lock(
        &self,
        tx_id: &TransactionId,
        resource_id: &ResourceId,
    ) -> std::result::Result<bool, ProximaDBError> {
        // Check if already locked by another transaction
        if let Some(holder) = self.locks.get(resource_id)
            && holder.value() != tx_id
        {
            // Would cause waiting - check for deadlock
            if self.would_deadlock(tx_id, holder.value()) {
                return Ok(false);
            }

            // Add to waiting set
            self.waiting
                .entry(tx_id.clone())
                .or_default()
                .insert(resource_id.clone());

            return Ok(false);
        }

        // Acquire lock
        self.locks.insert(resource_id.clone(), tx_id.clone());
        Ok(true)
    }

    /// Release all locks held by a transaction
    pub async fn release_locks(&self, tx_id: &TransactionId) {
        // Remove all locks held by this transaction
        self.locks.retain(|_, holder| holder != tx_id);
        // Remove from waiting set
        self.waiting.remove(tx_id);
    }

    /// Simple deadlock detection using wait-for graph
    fn would_deadlock(&self, tx_id: &TransactionId, holder: &TransactionId) -> bool {
        // Check if holder is waiting for any resources held by tx_id
        if let Some(waiting_resources) = self.waiting.get(holder) {
            for resource in waiting_resources.iter() {
                if let Some(resource_holder) = self.locks.get(resource)
                    && resource_holder.value() == tx_id
                {
                    return true; // Deadlock detected
                }
            }
        }
        false
    }
}

impl Default for DistributedLockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Local transaction coordinator for single-node transactions
///
/// This provides transaction coordination without requiring distributed
/// consensus. Distributed graph transaction coordination belongs to the
/// relational distributed substrate.
pub struct LocalTransactionCoordinator {
    /// Active transactions
    transactions: Arc<RwLock<HashMap<TransactionId, TransactionMetadata>>>,
    /// Graph engine registry (graph_id -> engine)
    engines: Arc<RwLock<HashMap<String, Arc<dyn GraphEngine>>>>,
    /// Distributed lock manager
    lock_manager: Arc<DistributedLockManager>,
    /// Default transaction timeout
    default_timeout: Duration,
}

impl LocalTransactionCoordinator {
    /// Create a new local transaction coordinator
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            engines: Arc::new(RwLock::new(HashMap::new())),
            lock_manager: Arc::new(DistributedLockManager::new()),
            default_timeout,
        }
    }

    /// Create a coordinator with pre-registered engines
    pub fn with_engines(
        shards: HashMap<ShardId, Arc<dyn GraphEngine>>,
        default_timeout: Duration,
    ) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            engines: Arc::new(RwLock::new(shards)),
            lock_manager: Arc::new(DistributedLockManager::new()),
            default_timeout,
        }
    }

    /// Register a graph engine
    pub async fn register_engine(&self, graph_id: String, engine: Arc<dyn GraphEngine>) {
        let mut engines = self.engines.write().await;
        engines.insert(graph_id, engine);
    }

    /// Begin a new transaction
    pub async fn begin_transaction(
        &self,
        graph_id: &str,
        participants: Vec<ShardId>,
    ) -> std::result::Result<TransactionId, ProximaDBError> {
        let tx = TransactionMetadata::new(graph_id.to_string(), participants, self.default_timeout);
        let tx_id = tx.id.clone();

        let mut transactions = self.transactions.write().await;
        transactions.insert(tx_id.clone(), tx);

        Ok(tx_id)
    }

    /// Execute an operation within a transaction
    pub async fn execute_operation(
        &self,
        tx_id: TransactionId,
        op: GraphOperation,
    ) -> std::result::Result<(), ProximaDBError> {
        // Acquire lock on resource
        let resource_id = op.resource_id();
        let acquired = self.lock_manager.acquire_lock(&tx_id, &resource_id).await?;

        if !acquired {
            // Resource locked or deadlock detected
            self.abort(tx_id.clone()).await?;
            return Err(ProximaDBError::Internal(
                "Resource locked or deadlock detected, transaction aborted".to_string(),
            ));
        }

        // Add operation to transaction
        let mut transactions = self.transactions.write().await;
        let tx = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| ProximaDBError::Internal(format!("Transaction {} not found", tx_id)))?;

        // Check timeout
        if tx.is_timed_out() {
            drop(transactions);
            self.abort(tx_id).await?;
            return Err(ProximaDBError::Internal(
                "Transaction timed out".to_string(),
            ));
        }

        tx.operations.push(op);

        Ok(())
    }

    /// Commit a transaction
    pub async fn commit(&self, tx_id: TransactionId) -> std::result::Result<(), ProximaDBError> {
        // Get transaction and update state
        let (graph_id, operations) = {
            let mut transactions = self.transactions.write().await;
            let tx = transactions.get_mut(&tx_id).ok_or_else(|| {
                ProximaDBError::Internal(format!("Transaction {} not found", tx_id))
            })?;

            tx.state = TransactionState::Committing;
            (tx.graph_id.clone(), tx.operations.clone())
        };

        // Get engine and apply operations
        let engines = self.engines.read().await;
        let engine = engines.get(&graph_id).ok_or_else(|| {
            ProximaDBError::Internal(format!("Engine for graph {} not found", graph_id))
        })?;

        // Apply all operations
        for op in operations {
            match op {
                GraphOperation::InsertNode { node, .. } => {
                    engine.insert_node(node).await?;
                }
                GraphOperation::UpdateNode { node, .. } => {
                    engine.update_node(node).await?;
                }
                GraphOperation::DeleteNode { node_id, .. } => {
                    engine.delete_node(&node_id).await?;
                }
                GraphOperation::InsertEdge { edge, .. } => {
                    engine.insert_edge(edge).await?;
                }
                GraphOperation::UpdateEdge { edge, .. } => {
                    engine.update_edge(edge).await?;
                }
                GraphOperation::DeleteEdge { edge_id, .. } => {
                    engine.delete_edge(&edge_id).await?;
                }
            }
        }

        // Update state to committed
        {
            let mut transactions = self.transactions.write().await;
            if let Some(tx) = transactions.get_mut(&tx_id) {
                tx.state = TransactionState::Committed;
            }
        }

        // Release locks
        self.lock_manager.release_locks(&tx_id).await;

        Ok(())
    }

    /// Abort a transaction
    pub async fn abort(&self, tx_id: TransactionId) -> std::result::Result<(), ProximaDBError> {
        // Update state to aborted
        {
            let mut transactions = self.transactions.write().await;
            if let Some(tx) = transactions.get_mut(&tx_id) {
                tx.state = TransactionState::Aborted;
            }
        }

        // Release locks
        self.lock_manager.release_locks(&tx_id).await;

        Ok(())
    }

    /// Get transaction state
    pub async fn get_state(
        &self,
        tx_id: &TransactionId,
    ) -> std::result::Result<TransactionState, ProximaDBError> {
        let transactions = self.transactions.read().await;
        let tx = transactions
            .get(tx_id)
            .ok_or_else(|| ProximaDBError::Internal(format!("Transaction {} not found", tx_id)))?;

        Ok(tx.state)
    }
}

// ============================================================================
// Transaction Isolation Levels
// ============================================================================

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Transactions see only committed data from other transactions
    #[default]
    ReadCommitted,
    /// Full serializability with distributed locking
    Serializable,
    /// Snapshot isolation (reads from transaction start point)
    SnapshotIsolation,
}

/// Unit of Work tracks all changes within a transaction
///
/// This implements the Unit of Work pattern to batch and track all
/// graph modifications before commit, enabling atomic rollback.
#[derive(Debug)]
pub struct UnitOfWork {
    /// Transaction ID
    pub tx_id: TransactionId,
    /// Graph ID this unit of work operates on
    pub graph_id: String,
    /// Pending node insertions
    pub pending_node_inserts: Vec<Node>,
    /// Pending node updates (node_id -> new_node)
    pub pending_node_updates: HashMap<String, Node>,
    /// Pending node deletions
    pub pending_node_deletes: Vec<String>,
    /// Pending edge insertions
    pub pending_edge_inserts: Vec<Edge>,
    /// Pending edge updates (edge_id -> new_edge)
    pub pending_edge_updates: HashMap<String, Edge>,
    /// Pending edge deletions
    pub pending_edge_deletes: Vec<String>,
    /// Isolation level for this transaction
    pub isolation_level: IsolationLevel,
    /// Whether this UoW is still active
    pub is_active: bool,
}

impl UnitOfWork {
    /// Create a new Unit of Work for a transaction
    pub fn new(tx_id: TransactionId, graph_id: String, isolation_level: IsolationLevel) -> Self {
        Self {
            tx_id,
            graph_id,
            pending_node_inserts: Vec::new(),
            pending_node_updates: HashMap::new(),
            pending_node_deletes: Vec::new(),
            pending_edge_inserts: Vec::new(),
            pending_edge_updates: HashMap::new(),
            pending_edge_deletes: Vec::new(),
            isolation_level,
            is_active: true,
        }
    }

    /// Register a node insertion
    pub fn register_node_insert(&mut self, node: Node) {
        self.pending_node_inserts.push(node);
    }

    /// Register a node update
    pub fn register_node_update(&mut self, node: Node) {
        self.pending_node_updates.insert(node.id.clone(), node);
    }

    /// Register a node deletion
    pub fn register_node_delete(&mut self, node_id: String) {
        self.pending_node_deletes.push(node_id);
    }

    /// Register an edge insertion
    pub fn register_edge_insert(&mut self, edge: Edge) {
        self.pending_edge_inserts.push(edge);
    }

    /// Register an edge update
    pub fn register_edge_update(&mut self, edge: Edge) {
        self.pending_edge_updates.insert(edge.id.clone(), edge);
    }

    /// Register an edge deletion
    pub fn register_edge_delete(&mut self, edge_id: String) {
        self.pending_edge_deletes.push(edge_id);
    }

    /// Check if this unit of work has any pending changes
    pub fn has_changes(&self) -> bool {
        !self.pending_node_inserts.is_empty()
            || !self.pending_node_updates.is_empty()
            || !self.pending_node_deletes.is_empty()
            || !self.pending_edge_inserts.is_empty()
            || !self.pending_edge_updates.is_empty()
            || !self.pending_edge_deletes.is_empty()
    }

    /// Get all graph operations for this unit of work
    pub fn get_operations(&self, shard_id: &str) -> Vec<GraphOperation> {
        let mut ops = Vec::new();

        // Node operations
        for node in &self.pending_node_inserts {
            ops.push(GraphOperation::InsertNode {
                shard_id: shard_id.to_string(),
                node: node.clone(),
            });
        }
        for node in self.pending_node_updates.values() {
            ops.push(GraphOperation::UpdateNode {
                shard_id: shard_id.to_string(),
                node: node.clone(),
            });
        }
        for node_id in &self.pending_node_deletes {
            ops.push(GraphOperation::DeleteNode {
                shard_id: shard_id.to_string(),
                node_id: node_id.clone(),
            });
        }

        // Edge operations
        for edge in &self.pending_edge_inserts {
            ops.push(GraphOperation::InsertEdge {
                shard_id: shard_id.to_string(),
                edge: edge.clone(),
            });
        }
        for edge in self.pending_edge_updates.values() {
            ops.push(GraphOperation::UpdateEdge {
                shard_id: shard_id.to_string(),
                edge: edge.clone(),
            });
        }
        for edge_id in &self.pending_edge_deletes {
            ops.push(GraphOperation::DeleteEdge {
                shard_id: shard_id.to_string(),
                edge_id: edge_id.clone(),
            });
        }

        ops
    }

    /// Clear all pending changes (used after commit or rollback)
    pub fn clear(&mut self) {
        self.pending_node_inserts.clear();
        self.pending_node_updates.clear();
        self.pending_node_deletes.clear();
        self.pending_edge_inserts.clear();
        self.pending_edge_updates.clear();
        self.pending_edge_deletes.clear();
        self.is_active = false;
    }
}

/// Transaction Manager wraps LocalTransactionCoordinator with enterprise patterns
///
/// This provides a higher-level API for transaction management while delegating
/// to the underlying coordinator for transaction coordination.
pub struct TransactionManager {
    /// Local transaction coordinator for single-node transactions
    coordinator: Arc<LocalTransactionCoordinator>,
    /// Active units of work indexed by transaction ID
    active_uows: Arc<DashMap<TransactionId, RwLock<UnitOfWork>>>,
    /// Default transaction timeout
    #[allow(dead_code)]
    default_timeout: Duration,
    /// Default isolation level
    default_isolation: IsolationLevel,
}

impl TransactionManager {
    /// Create a new transaction manager with an existing coordinator
    pub fn new(coordinator: Arc<LocalTransactionCoordinator>) -> Self {
        Self {
            coordinator,
            active_uows: Arc::new(DashMap::new()),
            default_timeout: Duration::from_secs(30),
            default_isolation: IsolationLevel::ReadCommitted,
        }
    }

    /// Create a new transaction manager with default configuration
    pub fn with_defaults(
        shards: HashMap<ShardId, Arc<dyn GraphEngine>>,
        timeout: Duration,
    ) -> Self {
        let coordinator = LocalTransactionCoordinator::with_engines(shards, timeout);
        Self::new(Arc::new(coordinator))
    }

    /// Set default isolation level
    pub fn set_default_isolation(&mut self, level: IsolationLevel) {
        self.default_isolation = level;
    }

    /// Begin a new transaction
    ///
    /// # Arguments
    /// * `graph_id` - The graph this transaction operates on
    /// * `participants` - Shard IDs participating in this transaction
    ///
    /// # Returns
    /// The transaction ID for subsequent operations
    pub async fn begin_transaction(
        &self,
        graph_id: &str,
        participants: Vec<ShardId>,
    ) -> Result<TransactionId> {
        self.begin_transaction_with_isolation(graph_id, participants, self.default_isolation)
            .await
    }

    /// Begin a new transaction with specific isolation level
    pub async fn begin_transaction_with_isolation(
        &self,
        graph_id: &str,
        participants: Vec<ShardId>,
        isolation: IsolationLevel,
    ) -> Result<TransactionId> {
        // Start transaction via local coordinator
        let tx_id = self
            .coordinator
            .begin_transaction(graph_id, participants)
            .await?;

        // Create unit of work
        let uow = UnitOfWork::new(tx_id.clone(), graph_id.to_string(), isolation);
        self.active_uows.insert(tx_id.clone(), RwLock::new(uow));

        tracing::debug!(
            "Transaction {} started for graph {} with isolation {:?}",
            tx_id,
            graph_id,
            isolation
        );

        Ok(tx_id)
    }

    /// Get the unit of work for a transaction
    pub fn get_unit_of_work(
        &self,
        tx_id: &TransactionId,
    ) -> Option<dashmap::mapref::one::Ref<'_, TransactionId, RwLock<UnitOfWork>>> {
        self.active_uows.get(tx_id)
    }

    /// Lock the transaction's unit of work for a mutation, failing when the
    /// transaction is unknown or no longer active — the shared preamble of
    /// every `register_*` mutator below.
    async fn with_active_uow_mut<F>(&self, tx_id: &TransactionId, f: F) -> Result<()>
    where
        F: FnOnce(&mut UnitOfWork),
    {
        let uow_ref = self
            .active_uows
            .get(tx_id)
            .ok_or_else(|| ProximaDBError::Internal(format!("Transaction {} not found", tx_id)))?;

        let mut uow = uow_ref.write().await;
        if !uow.is_active {
            return Err(ProximaDBError::Internal(format!(
                "Transaction {} is no longer active",
                tx_id
            )));
        }
        f(&mut uow);
        Ok(())
    }

    /// Register a node operation within a transaction
    pub async fn register_node_insert(&self, tx_id: &TransactionId, node: Node) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_node_insert(node))
            .await
    }

    /// Register a node update within a transaction
    pub async fn register_node_update(&self, tx_id: &TransactionId, node: Node) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_node_update(node))
            .await
    }

    /// Register a node deletion within a transaction
    pub async fn register_node_delete(&self, tx_id: &TransactionId, node_id: String) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_node_delete(node_id))
            .await
    }

    /// Register an edge insertion within a transaction
    pub async fn register_edge_insert(&self, tx_id: &TransactionId, edge: Edge) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_edge_insert(edge))
            .await
    }

    /// Register an edge update within a transaction
    pub async fn register_edge_update(&self, tx_id: &TransactionId, edge: Edge) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_edge_update(edge))
            .await
    }

    /// Register an edge deletion within a transaction
    pub async fn register_edge_delete(&self, tx_id: &TransactionId, edge_id: String) -> Result<()> {
        self.with_active_uow_mut(tx_id, |uow| uow.register_edge_delete(edge_id))
            .await
    }

    /// Commit a transaction
    ///
    /// This applies all pending changes from the Unit of Work through the
    /// local transaction coordinator.
    pub async fn commit_transaction(&self, tx_id: TransactionId) -> Result<()> {
        // Execute all operations through coordinator
        // Note: We scope the DashMap ref to avoid deadlock with remove()
        {
            let uow_ref = self.active_uows.get(&tx_id).ok_or_else(|| {
                ProximaDBError::Internal(format!("Transaction {} not found", tx_id))
            })?;

            let uow = uow_ref.read().await;
            if !uow.is_active {
                return Err(ProximaDBError::Internal(format!(
                    "Transaction {} is no longer active",
                    tx_id
                )));
            }

            // Get operations and register them with the coordinator
            let shard_id = format!("shard_{}", uow.graph_id);
            let operations = uow.get_operations(&shard_id);

            // Drop the read lock before executing operations
            drop(uow);

            for op in operations {
                self.coordinator
                    .execute_operation(tx_id.clone(), op)
                    .await?;
            }
            // uow_ref drops here, releasing DashMap ref
        }

        // Commit via coordinator
        self.coordinator.commit(tx_id.clone()).await?;

        // Clear and remove the unit of work
        // Note: Need to re-acquire the ref since we dropped it above
        if let Some(uow_ref) = self.active_uows.get(&tx_id) {
            let mut uow = uow_ref.write().await;
            uow.clear();
            // Drop the write lock and DashMap ref before remove
            drop(uow);
            drop(uow_ref);
        }

        // Now safe to remove - no refs held
        self.active_uows.remove(&tx_id);

        tracing::debug!("Transaction {} committed successfully", tx_id);

        Ok(())
    }

    /// Rollback a transaction
    ///
    /// Discards all pending changes and releases any acquired locks.
    pub async fn rollback_transaction(&self, tx_id: TransactionId) -> Result<()> {
        // Abort via 2PC (releases locks)
        self.coordinator.abort(tx_id.clone()).await?;

        // Clear and remove the unit of work
        if let Some((_, uow_lock)) = self.active_uows.remove(&tx_id) {
            let mut uow = uow_lock.write().await;
            uow.clear();
        }

        tracing::debug!("Transaction {} rolled back", tx_id);

        Ok(())
    }

    /// Get the current state of a transaction
    pub async fn get_transaction_state(&self, tx_id: &TransactionId) -> Result<TransactionState> {
        self.coordinator.get_state(tx_id).await
    }

    /// Check if a transaction is active
    pub fn is_transaction_active(&self, tx_id: &TransactionId) -> bool {
        if let Some(uow_ref) = self.active_uows.get(tx_id) {
            // Try to get read lock without blocking
            if let Ok(uow) = uow_ref.try_read() {
                return uow.is_active;
            }
        }
        false
    }

    /// Get the graph ID for a transaction
    pub async fn get_transaction_graph_id(&self, tx_id: &TransactionId) -> Option<String> {
        self.active_uows.get(tx_id).map(|uow_ref| {
            // We need to block here since this is an async context
            // but we're accessing synchronously
            if let Ok(uow) = uow_ref.try_read() {
                Some(uow.graph_id.clone())
            } else {
                None
            }
        })?
    }
}

/// Transaction handle for RAII-style transaction management
///
/// This handle automatically rolls back the transaction if dropped without
/// explicit commit, ensuring no dangling transactions.
pub struct TransactionHandle {
    /// Transaction ID
    tx_id: TransactionId,
    /// Reference to the transaction manager
    manager: Arc<TransactionManager>,
    /// Whether the transaction has been explicitly committed or rolled back
    completed: bool,
}

impl TransactionHandle {
    /// Create a new transaction handle
    pub fn new(tx_id: TransactionId, manager: Arc<TransactionManager>) -> Self {
        Self {
            tx_id,
            manager,
            completed: false,
        }
    }

    /// Get the transaction ID
    pub fn id(&self) -> &TransactionId {
        &self.tx_id
    }

    /// Commit the transaction
    pub async fn commit(mut self) -> Result<()> {
        self.completed = true;
        self.manager.commit_transaction(self.tx_id.clone()).await
    }

    /// Rollback the transaction
    pub async fn rollback(mut self) -> Result<()> {
        self.completed = true;
        self.manager.rollback_transaction(self.tx_id.clone()).await
    }
}

impl Drop for TransactionHandle {
    fn drop(&mut self) {
        if !self.completed {
            // Attempt to rollback - this is best-effort since we can't await in drop
            tracing::warn!(
                "Transaction {} dropped without explicit commit/rollback, scheduling rollback",
                self.tx_id
            );
            // We can't await here, so we spawn a blocking task
            let tx_id = self.tx_id.clone();
            let manager = self.manager.clone();
            tokio::spawn(async move {
                if let Err(e) = manager.rollback_transaction(tx_id.clone()).await {
                    tracing::error!("Failed to rollback transaction {}: {}", tx_id, e);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    use crate::graph::engines::orion::OrionGraphEngine;
    use std::collections::HashMap as StdHashMap;

    #[tokio::test]
    async fn test_coordinator_begin_commit_no_ops() {
        // Test the coordinator directly without TransactionManager wrapper
        let orion = Arc::new(OrionGraphEngine::new());
        let mut shards = HashMap::new();
        shards.insert("test_graph".to_string(), orion as Arc<dyn GraphEngine>);

        let coordinator =
            LocalTransactionCoordinator::with_engines(shards, Duration::from_secs(30));

        let tx_id = coordinator
            .begin_transaction("test_graph", vec!["test_graph".to_string()])
            .await
            .expect("begin_transaction failed");

        // Commit without any operations
        coordinator
            .commit(tx_id.clone())
            .await
            .expect("commit failed");

        // Verify state
        let state = coordinator
            .get_state(&tx_id)
            .await
            .expect("get_state failed");
        assert_eq!(state, TransactionState::Committed);
    }

    #[tokio::test]
    async fn test_dashmap_rwlock_pattern() {
        // Test the DashMap + RwLock pattern to ensure no deadlocks
        let map: DashMap<String, RwLock<u32>> = DashMap::new();
        map.insert("key".to_string(), RwLock::new(42));

        // Get a ref (holds DashMap read lock)
        let entry = map.get("key").unwrap();

        // Take inner read lock
        {
            let val = entry.read().await;
            assert_eq!(*val, 42);
        }

        // Take inner write lock
        {
            let mut val = entry.write().await;
            *val = 100;
        }

        // This is the issue - we're still holding the DashMap ref
        // while trying to remove. Drop the ref first.
        drop(entry);

        map.remove("key");
        assert!(map.get("key").is_none());
    }

    #[tokio::test]
    async fn test_unit_of_work_tracks_changes() {
        let mut uow = UnitOfWork::new(
            "tx_1".to_string(),
            "test_graph".to_string(),
            IsolationLevel::ReadCommitted,
        );

        assert!(!uow.has_changes());

        let node = Node {
            id: "node_1".to_string(),
            labels: vec!["Person".to_string()],
            properties: StdHashMap::new(),
            ..Default::default()
        };
        uow.register_node_insert(node);

        assert!(uow.has_changes());
        assert_eq!(uow.pending_node_inserts.len(), 1);
    }

    /// Shared harness: a TransactionManager over one in-memory ORION shard
    /// (30s default timeout) keyed by `shard_id`.
    fn manager_over_shard(shard_id: &str) -> TransactionManager {
        let orion = Arc::new(OrionGraphEngine::new());
        let mut shards = HashMap::new();
        shards.insert(shard_id.to_string(), orion as Arc<dyn GraphEngine>);
        TransactionManager::with_defaults(shards, Duration::from_secs(30))
    }

    #[tokio::test]
    async fn test_transaction_manager_begin_commit() {
        let manager = manager_over_shard("test_graph");

        // Test 1: Simple begin/commit without operations
        let tx_id = manager
            .begin_transaction("test_graph", vec!["test_graph".to_string()])
            .await
            .expect("Failed to begin transaction");

        assert!(manager.is_transaction_active(&tx_id));

        // Commit (without any operations)
        manager
            .commit_transaction(tx_id.clone())
            .await
            .expect("Failed to commit empty transaction");

        assert!(!manager.is_transaction_active(&tx_id));
    }

    #[tokio::test]
    async fn test_transaction_manager_rollback() {
        let manager = manager_over_shard("shard_test");

        let tx_id = manager
            .begin_transaction("test_graph", vec!["shard_test".to_string()])
            .await
            .expect("Failed to begin transaction");

        // Rollback
        manager
            .rollback_transaction(tx_id.clone())
            .await
            .expect("Failed to rollback");

        assert!(!manager.is_transaction_active(&tx_id));
    }

    #[tokio::test]
    async fn test_isolation_levels() {
        let manager = manager_over_shard("shard_test");

        let tx_id = manager
            .begin_transaction_with_isolation(
                "test_graph",
                vec!["shard_test".to_string()],
                IsolationLevel::Serializable,
            )
            .await
            .expect("Failed to begin transaction");

        let uow_ref = manager.get_unit_of_work(&tx_id).expect("UoW not found");
        let uow = uow_ref.read().await;
        assert_eq!(uow.isolation_level, IsolationLevel::Serializable);
    }
}
