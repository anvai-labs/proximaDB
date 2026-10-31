/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Transaction Isolation Levels and Conflict Resolution

use serde::{Deserialize, Serialize};

/// Transaction isolation level
/// Canonical isolation levels live in `proximadb-storage-common`
/// (dedup wave 5): this crate's former copy was identical in spirit with a
/// drifted variant set and is now a re-export.
pub use proximadb_storage_common::IsolationLevel;

/// Conflict resolution strategy for concurrent transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConflictResolution {
    /// First writer wins - later transactions abort on conflict
    #[default]
    FirstWriterWins,

    /// Last writer wins - later transactions overwrite
    LastWriterWins,

    /// Abort on conflict - all conflicting transactions abort
    AbortOnConflict,

    /// Merge if possible - attempt to merge non-conflicting changes
    MergeIfPossible,

    /// Wait and retry - wait for conflicting transaction to complete
    WaitAndRetry,
}

/// Lock mode for resources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared lock - allows concurrent reads
    Shared,
    /// Exclusive lock - prevents all concurrent access
    Exclusive,
    /// Update lock - shared for read, upgradable to exclusive
    Update,
    /// Intent shared - signals intent to acquire shared locks on children
    IntentShared,
    /// Intent exclusive - signals intent to acquire exclusive locks on children
    IntentExclusive,
}

impl LockMode {
    /// Check if this lock mode is compatible with another
    pub fn is_compatible(&self, other: &LockMode) -> bool {
        match (self, other) {
            // Shared locks are compatible with each other
            (LockMode::Shared, LockMode::Shared) => true,
            (LockMode::Shared, LockMode::IntentShared) => true,
            (LockMode::IntentShared, LockMode::Shared) => true,
            (LockMode::IntentShared, LockMode::IntentShared) => true,

            // Intent exclusive is compatible with intent shared
            (LockMode::IntentExclusive, LockMode::IntentShared) => true,
            (LockMode::IntentShared, LockMode::IntentExclusive) => true,
            (LockMode::IntentExclusive, LockMode::IntentExclusive) => true,

            // Update locks are compatible with shared
            (LockMode::Update, LockMode::Shared) => true,
            (LockMode::Shared, LockMode::Update) => true,

            // Everything else conflicts
            _ => false,
        }
    }
}

/// A lock held by a transaction
#[derive(Debug, Clone)]
pub struct Lock {
    /// Transaction holding the lock
    pub transaction_id: String,
    /// Resource being locked
    pub resource_id: String,
    /// Lock mode
    pub mode: LockMode,
    /// When lock was acquired
    pub acquired_at: std::time::Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_level_strictness() {
        assert!(IsolationLevel::Serializable.strictness() > IsolationLevel::Snapshot.strictness());
        assert!(
            IsolationLevel::Snapshot.strictness() > IsolationLevel::RepeatableRead.strictness()
        );
        assert!(
            IsolationLevel::RepeatableRead.strictness()
                > IsolationLevel::ReadCommitted.strictness()
        );
        assert!(
            IsolationLevel::ReadCommitted.strictness()
                > IsolationLevel::ReadUncommitted.strictness()
        );
    }

    #[test]
    fn test_isolation_level_properties() {
        assert!(!IsolationLevel::ReadUncommitted.prevents_dirty_reads());
        assert!(IsolationLevel::ReadCommitted.prevents_dirty_reads());
        assert!(IsolationLevel::RepeatableRead.prevents_non_repeatable_reads());
        assert!(IsolationLevel::Serializable.prevents_phantom_reads());
    }

    #[test]
    fn test_lock_compatibility() {
        assert!(LockMode::Shared.is_compatible(&LockMode::Shared));
        assert!(!LockMode::Shared.is_compatible(&LockMode::Exclusive));
        assert!(!LockMode::Exclusive.is_compatible(&LockMode::Exclusive));
        assert!(LockMode::Update.is_compatible(&LockMode::Shared));
    }
}
