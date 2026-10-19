//! P2.1 (ADR-097 S2): segmented canonical WAL.
//!
//! The shared canonical WAL can seal its active file into immutable segments
//! (`{path}.{first_seq:020}.seg`). These tests pin the contract:
//!
//! * every entry reads back once, oldest first, across segments and restarts;
//! * the sequence counter continues past sealed segments after a reopen;
//! * a range read opens only segments whose LSN summary can contain a match;
//! * a corrupt sealed segment fails closed, while the active file's torn tail is
//!   still repaired;
//! * rewrite-compaction refuses a WAL that has sealed segments.
//!
//!   cargo test --test canonical_wal_segments_test

use anyhow::Result;
use proximadb::services::FramedTableWalAppender;
use proximadb::services::record_store::TableWalAppender;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use std::path::{Path, PathBuf};
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

fn oids(entries: &[CanonicalWalEntry]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|e| match &e.operation {
            CanonicalOperation::RecordUpsert { record, .. } => Some(record.oid.clone()),
            _ => None,
        })
        .collect()
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

/// Append one entry per call to `collection`, rotating after every append.
async fn append_each(wal: &FramedTableWalAppender, collection: &str, oids: &[&str]) -> Result<()> {
    for oid in oids {
        wal.append_operations(vec![upsert(collection, oid)], None)
            .await?;
    }
    Ok(())
}

#[tokio::test]
async fn rotated_segments_read_back_in_order_across_restart() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("pgwire").join("canonical-records.wal");

    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1); // seal after every append
    append_each(&wal, "a", &["a1", "a2"]).await?;
    append_each(&wal, "b", &["b1"]).await?;
    drop(wal);

    let segments = sealed_segments(&path);
    assert_eq!(
        segments.len(),
        3,
        "one sealed segment per append: {segments:?}"
    );
    assert!(
        segments[0]
            .to_string_lossy()
            .ends_with("canonical-records.wal.00000000000000000001.seg"),
        "segments are named by their first sequence: {segments:?}"
    );

    // A fresh appender WITHOUT rotation still reads the sealed history and
    // continues the sequence after it.
    let reopened = FramedTableWalAppender::open(&path).await?;
    let appended = reopened
        .append_operations(vec![upsert("a", "a3")], None)
        .await?;
    assert_eq!(
        appended[0].sequence_number, 4,
        "sequence continues past segments"
    );

    let all = reopened.read_entries().await?;
    assert_eq!(oids(&all), ["a1", "a2", "b1", "a3"]);
    let seqs: Vec<u64> = all.iter().map(|e| e.sequence_number).collect();
    assert_eq!(seqs, [1, 2, 3, 4], "oldest first, each entry exactly once");

    // The path-level reader (graph checkpoint correlation, branch merge) agrees.
    let by_path = FramedTableWalAppender::read_entries_from_path(&path).await?;
    assert_eq!(oids(&by_path), ["a1", "a2", "b1", "a3"]);
    Ok(())
}

#[tokio::test]
async fn range_read_opens_only_matching_segments() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1);
    append_each(&wal, "a", &["a1", "a2"]).await?; // seq 1, 2
    append_each(&wal, "b", &["b1"]).await?; // seq 3
    append_each(&wal, "a", &["a3"]).await?; // seq 4

    // Only segments holding `a` beyond seq 1: a2 (seq 2) and a3 (seq 4). The
    // `b`-only segment is never read, so `b1` cannot appear.
    let a_since_1 = wal.read_entries_since("a", 1).await?;
    assert_eq!(oids(&a_since_1), ["a2", "a3"]);

    let b_all = TableWalAppender::read_entries_since(&wal, "b", 0).await?;
    assert_eq!(oids(&b_all), ["b1"]);

    assert!(wal.read_entries_since("a", 4).await?.is_empty());
    assert!(wal.read_entries_since("missing", 0).await?.is_empty());

    // After a restart the summaries are rebuilt from the segments.
    drop(wal);
    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(
        oids(&reopened.read_entries_since("a", 1).await?),
        ["a2", "a3"]
    );
    Ok(())
}

#[tokio::test]
async fn unrotated_wal_range_read_matches_full_scan() -> Result<()> {
    // Rotation off (the default): one active file; the range read is the full
    // scan filtered by sequence, exactly as before P2.1.
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    append_each(&wal, "a", &["a1"]).await?;
    append_each(&wal, "b", &["b1"]).await?;
    append_each(&wal, "a", &["a2"]).await?;
    assert!(sealed_segments(&path).is_empty());
    let since: Vec<String> = oids(&wal.read_entries_since("a", 1).await?);
    assert!(since.contains(&"a2".to_string()) && !since.contains(&"a1".to_string()));
    Ok(())
}

#[tokio::test]
async fn corrupt_sealed_segment_fails_closed_but_active_torn_tail_is_repaired() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1);
    append_each(&wal, "a", &["a1", "a2"]).await?;
    wal.enable_segment_rotation(0);
    append_each(&wal, "a", &["a3"]).await?; // stays in the active file
    drop(wal);

    // A torn (zero-extended) tail on the ACTIVE file is repaired on open.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path)?;
        f.write_all(&[0u8; 7])?;
    }
    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(oids(&reopened.read_entries().await?), ["a1", "a2", "a3"]);
    drop(reopened);

    // A truncated SEALED segment is corruption: open refuses instead of
    // silently dropping acknowledged entries.
    let first = sealed_segments(&path).remove(0);
    let bytes = std::fs::read(&first)?;
    std::fs::write(&first, &bytes[..bytes.len() - 3])?;
    let err = FramedTableWalAppender::open(&path)
        .await
        .err()
        .expect("a truncated sealed segment must fail the open");
    assert!(
        format!("{err:#}").contains("sealed canonical WAL segment"),
        "{err:#}"
    );
    Ok(())
}

#[tokio::test]
async fn rewrite_compaction_refuses_a_rotated_wal() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1);
    append_each(&wal, "a", &["a1"]).await?;
    let entries = wal.read_entries().await?;
    let err = proximadb::services::canonical_wal::rewrite_canonical_wal(&path, &entries)
        .await
        .err()
        .expect("rewrite must refuse a WAL with sealed segments");
    assert!(format!("{err:#}").contains("sealed segments"), "{err:#}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn range_reads_racing_rotation_never_lose_acknowledged_entries() -> Result<()> {
    // Rotation after every append, with readers running concurrently: every
    // read must return a gap-free prefix that includes each entry acknowledged
    // before the read began (a reader overlapping a rename re-reads).
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = std::sync::Arc::new(FramedTableWalAppender::open(&path).await?);
    wal.enable_segment_rotation(1);
    let acked = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    let writer = {
        let (wal, acked) = (wal.clone(), acked.clone());
        tokio::spawn(async move {
            for i in 0..200u64 {
                let out = wal
                    .append_operations(vec![upsert("a", &format!("a{i}"))], None)
                    .await?;
                acked.store(out[0].sequence_number, std::sync::atomic::Ordering::SeqCst);
            }
            anyhow::Ok(())
        })
    };
    let mut readers = Vec::new();
    for _ in 0..3 {
        let (wal, acked) = (wal.clone(), acked.clone());
        readers.push(tokio::spawn(async move {
            for _ in 0..100 {
                let floor = acked.load(std::sync::atomic::Ordering::SeqCst);
                let seqs: Vec<u64> = wal
                    .read_entries_since("a", 0)
                    .await?
                    .iter()
                    .map(|e| e.sequence_number)
                    .collect();
                let expected: Vec<u64> = (1..=seqs.len() as u64).collect();
                assert_eq!(seqs, expected, "gap-free, oldest first, no duplicates");
                assert!(
                    seqs.len() as u64 >= floor,
                    "read lost acknowledged entries: {} < {floor}",
                    seqs.len()
                );
            }
            anyhow::Ok(())
        }));
    }
    writer.await??;
    for r in readers {
        r.await??;
    }
    assert_eq!(wal.read_entries().await?.len(), 200);
    Ok(())
}
