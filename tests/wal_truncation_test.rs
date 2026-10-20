//! P2.2a (TD-SEAM-2 / TD-SEAM-12): canonical-WAL truncation primitives.
//!
//! The mechanism only — nothing in production calls `truncate_below` until
//! every consumer in TD-SEAM-12 has a low-water mark. These tests pin:
//!
//! * only a contiguous oldest prefix of sealed segments is removed, never the
//!   active file, and the retained history reads back gap-free;
//! * the floor is durable: a restart never reuses a sequence, even when every
//!   segment is gone and the active file is empty, and a crash between the
//!   floor write and the deletes is finished on reopen;
//! * a damaged floor marker fails the open;
//! * readers racing truncation (and rotation) never error;
//! * the change feed reports truncated history instead of returning a gap;
//! * a write that is appended but not yet applied holds an in-flight
//!   reservation the truncation floor must respect.
//!
//!   cargo test --test wal_truncation_test

use anyhow::Result;
use proximadb::services::FramedTableWalAppender;
use proximadb::services::canonical_wal::WalHistoryTruncated;
use proximadb::services::record_store::{
    DirectWalTableRecordStore, TableRecordStore, TableWalAppender,
};
use proximadb_records::ProximaRecord;
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;

fn upsert(collection_id: &str, oid: &str) -> CanonicalOperation {
    CanonicalOperation::RecordUpsert {
        collection_id: collection_id.to_string(),
        record: Box::new(ProximaRecord {
            oid: oid.to_string(),
            ..ProximaRecord::default()
        }),
        projections: Vec::new(),
    }
}

fn seqs(entries: &[CanonicalWalEntry]) -> Vec<u64> {
    entries.iter().map(|e| e.sequence_number).collect()
}

fn sealed_segments(wal: &Path) -> Vec<PathBuf> {
    let name = wal.file_name().unwrap().to_string_lossy().to_string();
    let mut out: Vec<PathBuf> = std::fs::read_dir(wal.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let f = p.file_name().unwrap().to_string_lossy().to_string();
            f.starts_with(&format!("{name}.")) && f.ends_with(".seg")
        })
        .collect();
    out.sort();
    out
}

/// Five appends, each sealed into its own segment (seq 1..=5), then one more
/// left in the active file (seq 6).
async fn five_sealed_one_active(path: &Path) -> Result<FramedTableWalAppender> {
    let wal = FramedTableWalAppender::open(path).await?;
    wal.enable_segment_rotation(1);
    for i in 1..=5 {
        wal.append_operations(vec![upsert("a", &format!("r{i}"))], None)
            .await?;
    }
    wal.enable_segment_rotation(0);
    wal.append_operations(vec![upsert("a", "r6")], None).await?;
    assert_eq!(sealed_segments(path).len(), 5);
    Ok(wal)
}

#[tokio::test]
async fn truncation_removes_only_the_oldest_whole_segments() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = five_sealed_one_active(&path).await?;
    assert_eq!(wal.retained_from(), 0, "nothing truncated yet");

    // Floor 3: segments holding seq 1 and 2 go; seq 3.. stay.
    let report = wal.truncate_below(3).await?;
    assert_eq!((report.segments, report.truncated_through), (2, 2));
    assert_eq!(wal.retained_from(), 3);
    assert_eq!(sealed_segments(&path).len(), 3);
    assert_eq!(seqs(&wal.read_entries().await?), [3, 4, 5, 6]);
    assert_eq!(seqs(&wal.read_entries_since("a", 0).await?), [3, 4, 5, 6]);

    // A floor past everything still never touches the active file.
    let report = wal.truncate_below(u64::MAX).await?;
    assert_eq!((report.segments, report.truncated_through), (3, 5));
    assert_eq!(seqs(&wal.read_entries().await?), [6]);
    assert!(path.exists(), "the active file is never truncated");

    // Nothing left to remove is a no-op.
    assert_eq!(wal.truncate_below(u64::MAX).await?.segments, 0);
    Ok(())
}

#[tokio::test]
async fn a_restart_never_reuses_a_truncated_sequence() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1);
    for i in 1..=3 {
        wal.append_operations(vec![upsert("a", &format!("r{i}"))], None)
            .await?;
    }
    // Every entry is sealed; the active file is empty. Truncate it all.
    assert_eq!(wal.truncate_below(u64::MAX).await?.truncated_through, 3);
    assert!(sealed_segments(&path).is_empty());
    drop(wal);

    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(
        reopened.last_sequence(),
        3,
        "the floor marker carries the sequence"
    );
    assert_eq!(reopened.retained_from(), 4);
    let next = reopened
        .append_operations(vec![upsert("a", "r4")], None)
        .await?;
    assert_eq!(next[0].sequence_number, 4, "no sequence is ever reissued");
    Ok(())
}

#[tokio::test]
async fn an_interrupted_truncation_is_finished_on_reopen() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = five_sealed_one_active(&path).await?;
    drop(wal);
    // The floor was made durable, then the process died before deleting.
    let mut floor = path.as_os_str().to_owned();
    floor.push(".floor");
    std::fs::write(&floor, "2")?;

    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(reopened.retained_from(), 3);
    assert_eq!(
        sealed_segments(&path).len(),
        3,
        "segments at or below the floor removed"
    );
    assert_eq!(seqs(&reopened.read_entries().await?), [3, 4, 5, 6]);
    assert_eq!(
        seqs(&FramedTableWalAppender::read_entries_from_path(&path).await?),
        [3, 4, 5, 6]
    );

    // A damaged marker fails the open rather than guessing.
    drop(reopened);
    std::fs::write(&floor, "not-a-number")?;
    let err = FramedTableWalAppender::open(&path)
        .await
        .err()
        .expect("a damaged floor marker must fail the open");
    assert!(format!("{err:#}").contains("floor marker"), "{err:#}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_racing_truncation_and_rotation_never_error() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = Arc::new(FramedTableWalAppender::open(&path).await?);
    wal.enable_segment_rotation(1);

    let writer = {
        let wal = wal.clone();
        tokio::spawn(async move {
            for i in 0..150u64 {
                wal.append_operations(vec![upsert("a", &format!("r{i}"))], None)
                    .await?;
                if i % 10 == 9 {
                    wal.truncate_below(wal.last_sequence().saturating_sub(5))
                        .await?;
                }
            }
            anyhow::Ok(())
        })
    };
    let mut readers = Vec::new();
    for _ in 0..3 {
        let (wal, path) = (wal.clone(), path.clone());
        readers.push(tokio::spawn(async move {
            for _ in 0..80 {
                for entries in [
                    wal.read_entries_since("a", 0).await?,
                    FramedTableWalAppender::read_entries_from_path(&path).await?,
                ] {
                    let s = seqs(&entries);
                    // Whatever is retained is contiguous and ordered.
                    assert!(
                        s.windows(2).all(|w| w[1] == w[0] + 1),
                        "retained history has a gap or reorder: {s:?}"
                    );
                }
            }
            anyhow::Ok(())
        }));
    }
    writer.await??;
    for r in readers {
        r.await??;
    }
    Ok(())
}

#[tokio::test]
async fn the_change_feed_reports_truncated_history() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = Arc::new(five_sealed_one_active(&path).await?);
    wal.truncate_below(3).await?;
    let store = DirectWalTableRecordStore::new_partitioned(wal.clone());

    let err = store
        .read_changes_since("a", 0)
        .await
        .expect_err("history before seq 3 is gone");
    let truncated = err
        .downcast_ref::<WalHistoryTruncated>()
        .expect("a typed truncated-history error");
    assert_eq!((truncated.since, truncated.retained_from), (0, 3));
    assert!(store.read_changes_since_scoped("a", None, 0).await.is_err());

    // A cursor at or past the floor is served in full.
    assert_eq!(store.read_changes_since("a", 2).await?.len(), 4);
    Ok(())
}

/// Blocks inside `append_operations` until released, after the entry is
/// durable — the window between "appended" and "applied to the partition".
struct ParkedWal {
    inner: FramedTableWalAppender,
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl TableWalAppender for ParkedWal {
    async fn append_operations(
        &self,
        operations: Vec<CanonicalOperation>,
        tenant_id: Option<String>,
    ) -> Result<Vec<CanonicalWalEntry>> {
        let entries = self.inner.append_operations(operations, tenant_id).await?;
        self.reached.notify_one();
        self.release.notified().await;
        Ok(entries)
    }
    fn last_sequence(&self) -> Option<u64> {
        Some(self.inner.last_sequence())
    }
}

#[tokio::test]
async fn an_in_flight_write_holds_the_truncation_floor() -> Result<()> {
    let dir = TempDir::new()?;
    let inner = FramedTableWalAppender::open(dir.path().join("canonical-records.wal")).await?;
    inner
        .append_operations(vec![upsert("t", "seed")], None)
        .await?;
    let wal = Arc::new(ParkedWal {
        inner,
        reached: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    let store = Arc::new(DirectWalTableRecordStore::new_partitioned(wal.clone()));
    assert_eq!(store.inflight_floor(), None);

    let writer = {
        let store = store.clone();
        tokio::spawn(async move {
            store
                .drop_table_records(&proximadb_catalog::CatalogTableSchema::new("t"), None)
                .await
        })
    };
    wal.reached.notified().await;
    // Appended (seq 2) but not yet applied: the floor must not pass it.
    assert_eq!(store.inflight_floor(), Some(2));
    wal.release.notify_one();
    writer.await??;
    assert_eq!(store.inflight_floor(), None, "released on completion");
    Ok(())
}
