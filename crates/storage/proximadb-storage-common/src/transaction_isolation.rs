//! # Transaction Isolation Levels
//!
//! Provides isolation level management for multi-model transactions.
//! Supports standard SQL isolation levels with MVCC semantics.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::debug;

/// Transaction isolation level
/// Canonical isolation-level vocabulary (dedup wave 5): previously defined
/// independently (with drifted variant sets) in storage-transaction,
/// metadata/atomic, graph/service_transactions and catalog_bulk_write.
/// `Snapshot` is the snapshot-isolation level those sites spelled
/// `SnapshotIsolation`; variant names are the wire vocabulary and are
/// unchanged from every former definition.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum IsolationLevel {
    /// Read uncommitted - can see uncommitted changes from other transactions
    ReadUncommitted,
    /// Read committed - only see committed changes (default)
    #[default]
    ReadCommitted,
    /// Repeatable read - consistent reads within transaction
    RepeatableRead,
    /// Snapshot isolation - transactions read from a consistent snapshot
    Snapshot,
    /// Serializable - full isolation, transactions appear to run serially
    Serializable,
}

impl IsolationLevel {
    /// Get the strictness level (higher = more strict)
    pub fn strictness(&self) -> u8 {
        match self {
            IsolationLevel::ReadUncommitted => 0,
            IsolationLevel::ReadCommitted => 1,
            IsolationLevel::RepeatableRead => 2,
            IsolationLevel::Snapshot => 3,
            IsolationLevel::Serializable => 4,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            IsolationLevel::ReadUncommitted => "READ UNCOMMITTED",
            IsolationLevel::ReadCommitted => "READ COMMITTED",
            IsolationLevel::RepeatableRead => "REPEATABLE READ",
            IsolationLevel::Snapshot => "SNAPSHOT",
            IsolationLevel::Serializable => "SERIALIZABLE",
        }
    }

    /// Check if this level allows dirty reads
    pub fn allows_dirty_reads(&self) -> bool {
        self.strictness() < 1
    }

    /// Check if this level prevents dirty reads
    pub fn prevents_dirty_reads(&self) -> bool {
        self.strictness() >= 1
    }

    /// Check if this level prevents non-repeatable reads
    pub fn prevents_non_repeatable_reads(&self) -> bool {
        self.strictness() >= 2
    }

    /// Check if this level prevents phantom reads
    pub fn prevents_phantom_reads(&self) -> bool {
        self.strictness() >= 4
    }
}

/// Read snapshot for MVCC
#[derive(Debug, Clone)]
pub struct ReadSnapshot {
    /// Transaction ID that created this snapshot
    pub transaction_id: String,
    /// Snapshot timestamp in nanoseconds
    pub snapshot_time_ns: i64,
    /// Active transaction IDs at snapshot time
    pub active_transactions: HashSet<String>,
    /// Committed transaction IDs visible to this snapshot
    pub committed_transactions: HashSet<String>,
    /// Minimum visible timestamp
    pub min_visible_ts: i64,
    /// Maximum visible timestamp
    pub max_visible_ts: i64,
}

impl ReadSnapshot {
    /// Create a new read snapshot
    pub fn new(transaction_id: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        Self {
            transaction_id,
            snapshot_time_ns: now,
            active_transactions: HashSet::new(),
            committed_transactions: HashSet::new(),
            min_visible_ts: 0,
            max_visible_ts: now,
        }
    }

    /// Check if a record version is visible in this snapshot
    pub fn is_visible(
        &self,
        record_ts: i64,
        created_by_txn: Option<&str>,
        committed: bool,
    ) -> bool {
        // Same transaction can see its own changes
        if let Some(txn_id) = created_by_txn {
            if txn_id == self.transaction_id {
                return true;
            }

            // Can't see changes from active (uncommitted) transactions
            if self.active_transactions.contains(txn_id) {
                return false;
            }

            // Must be committed and in our committed set
            if !committed {
                return false;
            }
        }

        // Check timestamp bounds
        record_ts >= self.min_visible_ts && record_ts <= self.max_visible_ts
    }

    /// Add an active transaction
    pub fn add_active_transaction(&mut self, txn_id: String) {
        self.active_transactions.insert(txn_id);
    }

    /// Mark a transaction as committed
    pub fn mark_committed(&mut self, txn_id: String) {
        self.active_transactions.remove(&txn_id);
        self.committed_transactions.insert(txn_id);
    }
}

/// Write set tracking for transaction
#[derive(Debug, Clone, Default)]
pub struct WriteSet {
    /// Written records (store_type -> record_ids)
    pub writes: HashMap<String, HashSet<String>>,
    /// Deleted records (store_type -> record_ids)
    pub deletes: HashMap<String, HashSet<String>>,
    /// Total write count
    pub total_writes: usize,
    /// Total delete count
    pub total_deletes: usize,
}

impl WriteSet {
    /// Create a new empty write set
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a write operation
    pub fn record_write(&mut self, store_type: &str, record_id: &str) {
        self.writes
            .entry(store_type.to_string())
            .or_default()
            .insert(record_id.to_string());
        self.total_writes += 1;
    }

    /// Record a delete operation
    pub fn record_delete(&mut self, store_type: &str, record_id: &str) {
        self.deletes
            .entry(store_type.to_string())
            .or_default()
            .insert(record_id.to_string());
        self.total_deletes += 1;
    }

    /// Check if a record was written in this transaction
    pub fn has_write(&self, store_type: &str, record_id: &str) -> bool {
        self.writes
            .get(store_type)
            .is_some_and(|ids| ids.contains(record_id))
    }

    /// Check if a record was deleted in this transaction
    pub fn has_delete(&self, store_type: &str, record_id: &str) -> bool {
        self.deletes
            .get(store_type)
            .is_some_and(|ids| ids.contains(record_id))
    }

    /// Check for write conflicts with another write set
    pub fn conflicts_with(&self, other: &WriteSet) -> bool {
        // Check if any writes in self conflict with writes or deletes in other
        for (store_type, ids) in &self.writes {
            if let Some(other_writes) = other.writes.get(store_type)
                && !ids.is_disjoint(other_writes)
            {
                return true;
            }
            if let Some(other_deletes) = other.deletes.get(store_type)
                && !ids.is_disjoint(other_deletes)
            {
                return true;
            }
        }

        // Check if any deletes in self conflict with writes in other
        for (store_type, ids) in &self.deletes {
            if let Some(other_writes) = other.writes.get(store_type)
                && !ids.is_disjoint(other_writes)
            {
                return true;
            }
        }

        false
    }

    /// Get all affected store types
    pub fn affected_stores(&self) -> HashSet<String> {
        let mut stores: HashSet<String> = self.writes.keys().cloned().collect();
        stores.extend(self.deletes.keys().cloned());
        stores
    }

    /// Is the write set empty?
    pub fn is_empty(&self) -> bool {
        self.total_writes == 0 && self.total_deletes == 0
    }

    /// Merge another write set into this one
    pub fn merge(&mut self, other: &WriteSet) {
        for (store_type, ids) in &other.writes {
            self.writes
                .entry(store_type.clone())
                .or_default()
                .extend(ids.iter().cloned());
        }
        for (store_type, ids) in &other.deletes {
            self.deletes
                .entry(store_type.clone())
                .or_default()
                .extend(ids.iter().cloned());
        }
        self.total_writes += other.total_writes;
        self.total_deletes += other.total_deletes;
    }
}

/// Isolation manager handles MVCC and conflict detection
pub struct IsolationManager {
    /// Default isolation level
    default_level: IsolationLevel,
    /// Active transactions and their snapshots
    active_snapshots: RwLock<HashMap<String, (IsolationLevel, ReadSnapshot)>>,
    /// Active write sets for conflict detection
    active_write_sets: RwLock<HashMap<String, WriteSet>>,
    /// Committed transaction history for serializable isolation
    commit_history: RwLock<Vec<(String, i64, WriteSet)>>,
    /// Maximum history size
    max_history_size: usize,
}

impl IsolationManager {
    /// Create a new isolation manager
    pub fn new(default_level: IsolationLevel) -> Self {
        Self {
            default_level,
            active_snapshots: RwLock::new(HashMap::new()),
            active_write_sets: RwLock::new(HashMap::new()),
            commit_history: RwLock::new(Vec::new()),
            max_history_size: 10000,
        }
    }

    /// Begin a transaction with specified isolation level
    pub async fn begin_transaction(
        &self,
        transaction_id: &str,
        level: Option<IsolationLevel>,
    ) -> ReadSnapshot {
        let level = level.unwrap_or(self.default_level);
        let mut snapshot = ReadSnapshot::new(transaction_id.to_string());

        // For repeatable read and serializable, capture active transactions
        if level.prevents_non_repeatable_reads() {
            let active = self.active_snapshots.read().await;
            for txn_id in active.keys() {
                if txn_id != transaction_id {
                    snapshot.add_active_transaction(txn_id.clone());
                }
            }
        }

        // Store the snapshot
        {
            let mut snapshots = self.active_snapshots.write().await;
            snapshots.insert(transaction_id.to_string(), (level, snapshot.clone()));
        }

        // Initialize empty write set
        {
            let mut write_sets = self.active_write_sets.write().await;
            write_sets.insert(transaction_id.to_string(), WriteSet::new());
        }

        debug!(
            "Transaction {} started with isolation level {:?}",
            transaction_id, level
        );
        snapshot
    }

    /// Get snapshot for a transaction
    pub async fn get_snapshot(&self, transaction_id: &str) -> Option<ReadSnapshot> {
        let snapshots = self.active_snapshots.read().await;
        snapshots.get(transaction_id).map(|(_, s)| s.clone())
    }

    /// Get isolation level for a transaction
    pub async fn get_isolation_level(&self, transaction_id: &str) -> Option<IsolationLevel> {
        let snapshots = self.active_snapshots.read().await;
        snapshots.get(transaction_id).map(|(l, _)| *l)
    }

    /// Record a write in the transaction's write set
    pub async fn record_write(&self, transaction_id: &str, store_type: &str, record_id: &str) {
        let mut write_sets = self.active_write_sets.write().await;
        if let Some(write_set) = write_sets.get_mut(transaction_id) {
            write_set.record_write(store_type, record_id);
        }
    }

    /// Record a delete in the transaction's write set
    pub async fn record_delete(&self, transaction_id: &str, store_type: &str, record_id: &str) {
        let mut write_sets = self.active_write_sets.write().await;
        if let Some(write_set) = write_sets.get_mut(transaction_id) {
            write_set.record_delete(store_type, record_id);
        }
    }

    /// Check for conflicts before commit (for serializable isolation)
    pub async fn check_conflicts(&self, transaction_id: &str) -> Result<(), String> {
        let (level, snapshot) = {
            let snapshots = self.active_snapshots.read().await;
            match snapshots.get(transaction_id) {
                Some(entry) => entry.clone(),
                None => return Err(format!("Transaction {} not found", transaction_id)),
            }
        };

        // Only serializable needs conflict checking
        if level != IsolationLevel::Serializable {
            return Ok(());
        }

        let my_write_set = {
            let write_sets = self.active_write_sets.read().await;
            match write_sets.get(transaction_id) {
                Some(ws) => ws.clone(),
                None => return Err(format!("Write set for {} not found", transaction_id)),
            }
        };

        // Check against committed transactions since our snapshot
        let history = self.commit_history.read().await;
        for (committed_txn, commit_ts, committed_writes) in history.iter() {
            // Skip if committed before our snapshot
            if *commit_ts <= snapshot.snapshot_time_ns {
                continue;
            }

            // Check for conflicts
            if my_write_set.conflicts_with(committed_writes) {
                return Err(format!(
                    "Serialization conflict: transaction {} conflicts with {}",
                    transaction_id, committed_txn
                ));
            }
        }

        Ok(())
    }

    /// Commit a transaction
    pub async fn commit_transaction(&self, transaction_id: &str) -> Result<(), String> {
        // Check conflicts first
        self.check_conflicts(transaction_id).await?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        // Get and remove write set
        let write_set = {
            let mut write_sets = self.active_write_sets.write().await;
            write_sets.remove(transaction_id)
        };

        // Add to commit history
        if let Some(ws) = write_set
            && !ws.is_empty()
        {
            let mut history = self.commit_history.write().await;
            history.push((transaction_id.to_string(), now, ws));

            // Trim history if too large
            let history_len = history.len();
            if history_len > self.max_history_size {
                let drain_count = history_len - self.max_history_size;
                history.drain(0..drain_count);
            }
        }

        // Remove snapshot
        {
            let mut snapshots = self.active_snapshots.write().await;
            snapshots.remove(transaction_id);
        }

        debug!("Transaction {} committed", transaction_id);
        Ok(())
    }

    /// Abort a transaction
    pub async fn abort_transaction(&self, transaction_id: &str) {
        // Remove write set
        {
            let mut write_sets = self.active_write_sets.write().await;
            write_sets.remove(transaction_id);
        }

        // Remove snapshot
        {
            let mut snapshots = self.active_snapshots.write().await;
            snapshots.remove(transaction_id);
        }

        debug!("Transaction {} aborted", transaction_id);
    }

    /// Get active transaction count
    pub async fn active_transaction_count(&self) -> usize {
        let snapshots = self.active_snapshots.read().await;
        snapshots.len()
    }

    /// Get default isolation level
    pub fn default_level(&self) -> IsolationLevel {
        self.default_level
    }
}

impl Default for IsolationManager {
    fn default() -> Self {
        Self::new(IsolationLevel::ReadCommitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_level_default() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::ReadCommitted);
    }

    #[test]
    fn test_isolation_level_properties() {
        assert!(IsolationLevel::ReadUncommitted.allows_dirty_reads());
        assert!(!IsolationLevel::ReadCommitted.allows_dirty_reads());

        assert!(!IsolationLevel::ReadCommitted.prevents_non_repeatable_reads());
        assert!(IsolationLevel::RepeatableRead.prevents_non_repeatable_reads());

        assert!(!IsolationLevel::RepeatableRead.prevents_phantom_reads());
        assert!(IsolationLevel::Serializable.prevents_phantom_reads());
    }

    #[test]
    fn test_read_snapshot_visibility() {
        let mut snapshot = ReadSnapshot::new("txn1".to_string());
        snapshot.add_active_transaction("txn2".to_string());

        // Own changes are visible
        assert!(snapshot.is_visible(100, Some("txn1"), false));

        // Active transaction changes not visible
        assert!(!snapshot.is_visible(100, Some("txn2"), false));

        // Committed changes visible
        snapshot.mark_committed("txn3".to_string());
        assert!(snapshot.is_visible(100, Some("txn3"), true));
    }

    #[test]
    fn test_write_set_operations() {
        let mut ws = WriteSet::new();

        ws.record_write("vector", "vec1");
        ws.record_delete("document", "doc1");

        assert!(ws.has_write("vector", "vec1"));
        assert!(!ws.has_write("vector", "vec2"));
        assert!(ws.has_delete("document", "doc1"));

        assert_eq!(ws.total_writes, 1);
        assert_eq!(ws.total_deletes, 1);
    }

    #[test]
    fn test_write_set_conflicts() {
        let mut ws1 = WriteSet::new();
        ws1.record_write("vector", "vec1");
        ws1.record_write("vector", "vec2");

        let mut ws2 = WriteSet::new();
        ws2.record_write("vector", "vec2"); // Conflicts with ws1
        ws2.record_write("vector", "vec3");

        assert!(ws1.conflicts_with(&ws2));

        let mut ws3 = WriteSet::new();
        ws3.record_write("document", "doc1"); // No conflict
        ws3.record_delete("graph", "edge1");

        assert!(!ws1.conflicts_with(&ws3));
    }

    #[test]
    fn test_write_set_merge() {
        let mut ws1 = WriteSet::new();
        ws1.record_write("vector", "vec1");

        let mut ws2 = WriteSet::new();
        ws2.record_write("vector", "vec2");
        ws2.record_delete("document", "doc1");

        ws1.merge(&ws2);

        assert!(ws1.has_write("vector", "vec1"));
        assert!(ws1.has_write("vector", "vec2"));
        assert!(ws1.has_delete("document", "doc1"));
        assert_eq!(ws1.total_writes, 2);
        assert_eq!(ws1.total_deletes, 1);
    }

    #[tokio::test]
    async fn test_isolation_manager_basic() {
        let manager = IsolationManager::new(IsolationLevel::ReadCommitted);

        let snapshot = manager.begin_transaction("txn1", None).await;
        assert_eq!(snapshot.transaction_id, "txn1");

        let level = manager.get_isolation_level("txn1").await;
        assert_eq!(level, Some(IsolationLevel::ReadCommitted));

        assert_eq!(manager.active_transaction_count().await, 1);

        manager.commit_transaction("txn1").await.unwrap();
        assert_eq!(manager.active_transaction_count().await, 0);
    }

    #[tokio::test]
    async fn test_isolation_manager_write_tracking() {
        let manager = IsolationManager::new(IsolationLevel::ReadCommitted);

        manager.begin_transaction("txn1", None).await;

        manager.record_write("txn1", "vector", "vec1").await;
        manager.record_delete("txn1", "document", "doc1").await;

        manager.commit_transaction("txn1").await.unwrap();
    }

    #[tokio::test]
    async fn test_serializable_conflict_detection() {
        let manager = IsolationManager::new(IsolationLevel::Serializable);

        // Start two concurrent transactions
        manager
            .begin_transaction("txn1", Some(IsolationLevel::Serializable))
            .await;
        manager
            .begin_transaction("txn2", Some(IsolationLevel::Serializable))
            .await;

        // Both write to same record
        manager.record_write("txn1", "vector", "vec1").await;
        manager.record_write("txn2", "vector", "vec1").await;

        // First commit succeeds
        manager.commit_transaction("txn1").await.unwrap();

        // Second commit should fail due to conflict
        let result = manager.commit_transaction("txn2").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_abort_transaction() {
        let manager = IsolationManager::new(IsolationLevel::ReadCommitted);

        manager.begin_transaction("txn1", None).await;
        manager.record_write("txn1", "vector", "vec1").await;

        assert_eq!(manager.active_transaction_count().await, 1);

        manager.abort_transaction("txn1").await;

        assert_eq!(manager.active_transaction_count().await, 0);
    }
}
