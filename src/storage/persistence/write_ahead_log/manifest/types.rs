#![allow(dead_code)]
//! Global WAL Manifest
//!
//! Provides a centralized manifest for tracking WAL files across all collections.
//! This enables:
//! - Global recovery ordering via monotonic LSN
//! - Cross-collection consistency
//! - Efficient checkpoint management
//! - Simplified disaster recovery
//!
//! Architecture:
//! ```text
//! Multi-disk configuration with centralized global manifest:
//!
//! DISK 1 (Primary - hosts global manifest):
//! /tmp/proximadb1/data/
//!   ├── wal/
//!   │   ├── global_manifest.log      # ✨ GLOBAL: Tracks ALL collections
//!   │   └── checkpoint.state         # ✨ GLOBAL: Latest checkpoint
//!   ├── {collection_A}/
//!   │   ├── wal/{batch_id}.bcwal     # Collection A WAL (if assigned here)
//!   │   └── data/*.sst               # Collection A data
//!   └── ...
//!
//! DISK 2 (Secondary - collection WAL files only):
//! /tmp/proximadb2/data/
//!   ├── {collection_B}/
//!   │   ├── wal/{batch_id}.bcwal     # Collection B WAL (if assigned here)
//!   │   └── data/*.parquet           # Collection B data
//!   └── ...
//!
//! The global manifest at DISK 1 tracks WAL files on ALL disks.
//! ```
//!
//! The runtime surface is [`super::service::GlobalManifestService`] (the
//! write-behind service these types describe); the legacy `GlobalManifest`
//! struct that duplicated its logic was removed as dead code (never
//! constructed outside its own impl, hidden by the blanket
//! `allow(dead_code)` above).

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::persistence::write_ahead_log::{BatchId, serialization::SerializationFormat};

/// Status of a WAL entry in the global manifest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WalEntryStatus {
    /// Active WAL file, not yet flushed to storage engine
    Active,
    /// Flushed to storage engine, safe to archive
    Flushed,
    /// Archived to long-term storage, can be deleted
    Archived,
    /// Rolled back during PITR recovery (not to be recovered)
    RolledBack,
}

/// Global manifest entry tracking a single WAL batch across all collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalManifestEntry {
    /// Global LSN (monotonically increasing across all collections)
    pub global_lsn: u64,

    /// Collection identifier
    pub collection_id: String,

    /// Batch identifier (base62 encoded)
    pub batch_id: String,

    /// Relative path from base_url: {collection_id}/wal/{filename}
    /// Example: "1v5XVSY/wal/8WBT...bcwal"
    /// This path is relative to the storage location where the collection is assigned
    pub file_path: String,

    /// Storage location URL where this WAL file resides
    /// Example: "file:///tmp/proximadb1/data" or "file:///tmp/proximadb2/data"
    /// This allows the manifest to track files across multiple disks
    pub storage_url: String,

    /// File size in bytes
    pub size_bytes: u64,

    /// CRC32 checksum
    pub checksum_crc32: u32,

    /// Timestamp when batch was created (milliseconds since epoch)
    pub timestamp_ms: u64,

    /// Serialization format (bincode, avro, proto)
    pub format: SerializationFormat,

    /// Number of vectors in this batch
    pub vector_count: u64,

    /// Status: Active, Flushed, Archived
    pub status: WalEntryStatus,

    /// Optional: checkpoint ID if this batch is part of a checkpoint
    pub checkpoint_id: Option<u64>,
}

impl GlobalManifestEntry {
    /// Create a new active manifest entry
    pub fn new(
        global_lsn: u64,
        collection_id: String,
        batch_id: &BatchId,
        file_name: String,
        size_bytes: u64,
        checksum_crc32: u32,
        format: SerializationFormat,
        vector_count: u64,
        storage_url: String, // Which disk this WAL file is on
    ) -> Self {
        let timestamp_ms = batch_id.timestamp_ms();
        // Keep existing structure: {collection_id}/wal/{file_name}
        let file_path = format!("{}/wal/{}", collection_id, file_name);

        Self {
            global_lsn,
            collection_id,
            batch_id: batch_id.to_base62(),
            file_path,
            storage_url,
            size_bytes,
            checksum_crc32,
            timestamp_ms,
            format,
            vector_count,
            status: WalEntryStatus::Active,
            checkpoint_id: None,
        }
    }

    /// Get the full URL to this WAL file
    pub fn full_url(&self) -> String {
        format!("{}/{}", self.storage_url, self.file_path)
    }
}

/// Global checkpoint state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCheckpoint {
    /// Checkpoint identifier (monotonically increasing)
    pub checkpoint_id: u64,

    /// Global LSN at checkpoint time
    pub checkpoint_lsn: u64,

    /// Timestamp of checkpoint
    pub timestamp_ms: u64,

    /// Collections included in this checkpoint
    pub collections: Vec<CheckpointCollectionState>,

    /// All WAL entries with global_lsn < safe_to_delete_before_lsn can be safely deleted
    pub safe_to_delete_before_lsn: u64,
}

/// Per-collection state at checkpoint time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointCollectionState {
    pub collection_id: String,
    pub last_flushed_lsn: u64,
    pub vector_count: u64,
}

/// Global LSN allocator - ensures monotonically increasing LSN across all collections.
/// Uses AtomicU64 for lock-free allocation, eliminating contention on the write path.
/// Per-POD monotonic LSN source (`AtomicU64`, seeded at recovery). The LSNs
/// it allocates are **never comparable across pods** — each process owns an
/// independent counter (TD-CACHE-10: cross-pod invalidation events therefore
/// carry `(pod_id, lsn)` and dedup/order per pod). Any design that needs a
/// globally comparable ordering must use a cross-pod watermark (lease
/// generations, the directory's `freshness_watermark_lsn`) or a replicated
/// counter — never raw allocator output.
pub struct GlobalLsnAllocator {
    next_lsn: AtomicU64,
}

impl GlobalLsnAllocator {
    /// Create a new LSN allocator starting from a given LSN
    pub fn new(start_lsn: u64) -> Self {
        Self {
            next_lsn: AtomicU64::new(start_lsn),
        }
    }

    /// Allocate the next global LSN (lock-free via fetch_add)
    pub async fn allocate(&self) -> u64 {
        self.next_lsn.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the current LSN without allocating
    pub async fn current(&self) -> u64 {
        self.next_lsn.load(Ordering::Relaxed)
    }

    /// Set the next LSN (used during recovery)
    pub async fn set_next(&self, next_lsn: u64) {
        self.next_lsn.store(next_lsn, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lsn_allocator() {
        let allocator = GlobalLsnAllocator::new(100);

        assert_eq!(allocator.allocate().await, 100);
        assert_eq!(allocator.allocate().await, 101);
        assert_eq!(allocator.allocate().await, 102);
        assert_eq!(allocator.current().await, 103);
    }

    #[tokio::test]
    async fn test_manifest_entry_creation() {
        let batch_id = BatchId::new();
        let entry = GlobalManifestEntry::new(
            1,
            "test_collection".to_string(),
            &batch_id,
            "test.bcwal".to_string(),
            1024,
            12345,
            SerializationFormat::Bincode,
            100,
            "file:///tmp/proximadb1/data".to_string(),
        );

        assert_eq!(entry.global_lsn, 1);
        assert_eq!(entry.collection_id, "test_collection");
        assert_eq!(entry.size_bytes, 1024);
        assert_eq!(entry.vector_count, 100);
        assert_eq!(entry.status, WalEntryStatus::Active);
        assert_eq!(entry.file_path, "test_collection/wal/test.bcwal");
        assert_eq!(entry.storage_url, "file:///tmp/proximadb1/data");
        assert_eq!(
            entry.full_url(),
            "file:///tmp/proximadb1/data/test_collection/wal/test.bcwal"
        );
    }
}
