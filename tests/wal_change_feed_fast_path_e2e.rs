// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-OLAP-17 fast path: a never-written collection must not decode the WAL.
//!
//! The OLAP delta merge probes both the name-keyed and the object_id-keyed WAL
//! collection (ADR-031 dual-read). The object_id key is never written, so before
//! this fix its watermark was `None` ("unknown") and the whole canonical WAL was
//! decoded for every table on every query.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proximadb::services::MemoryTableWalAppender;
use proximadb::services::record_store::{
    DirectWalTableRecordStore, TableRecordStore, TableWalAppender,
};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};

struct CountingWal {
    inner: MemoryTableWalAppender,
    reads: AtomicUsize,
}

#[async_trait::async_trait]
impl TableWalAppender for CountingWal {
    async fn append_operations(
        &self,
        operations: Vec<CanonicalOperation>,
        tenant_id: Option<String>,
    ) -> anyhow::Result<Vec<CanonicalWalEntry>> {
        self.inner.append_operations(operations, tenant_id).await
    }
    async fn read_all_entries(&self) -> anyhow::Result<Vec<CanonicalWalEntry>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read_all_entries().await
    }
    fn collection_max_sequence_for(&self, c: &str) -> Option<u64> {
        self.inner.collection_max_sequence_for(c)
    }
    fn tracks_collection_high_water(&self) -> bool {
        self.inner.tracks_collection_high_water()
    }
}

#[tokio::test]
async fn never_written_collection_skips_wal_decode() {
    let wal = Arc::new(CountingWal {
        inner: MemoryTableWalAppender::new(),
        reads: AtomicUsize::new(0),
    });
    wal.append_operations(
        vec![CanonicalOperation::RecordUpsert {
            collection_id: "t1".to_string(),
            record: Box::new(ProximaRecord {
                oid: "k1".to_string(),
                ..Default::default()
            }),
            projections: vec![],
        }],
        None,
    )
    .await
    .unwrap();
    let store = DirectWalTableRecordStore::new_partitioned(wal.clone());

    assert!(
        store
            .read_changes_since("never-written", 0)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .read_changes_since_scoped("never-written", None, 0)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        wal.reads.load(Ordering::SeqCst),
        0,
        "no WAL decode for an unwritten key"
    );

    // A written collection still scans and returns its change.
    assert_eq!(store.read_changes_since("t1", 0).await.unwrap().len(), 1);
    assert_eq!(wal.reads.load(Ordering::SeqCst), 1);
}
