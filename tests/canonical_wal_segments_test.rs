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
//! * rewrite-compaction refuses a WAL that has sealed segments;
//! * the per-segment branch summary (TD-SEAM-12) is rebuilt on reopen, and an
//!   armed branch fence refuses a truncation that would strip a collection
//!   branched after the merge-base image was taken.
//!
//!   cargo test --test canonical_wal_segments_test

use anyhow::Result;
use proximadb::services::FramedTableWalAppender;
use proximadb::services::canonical_wal::BranchedCollection;
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

#[tokio::test]
async fn open_shared_returns_one_appender_per_file() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");

    let first = FramedTableWalAppender::open_shared(&path).await?;
    let second =
        FramedTableWalAppender::open_shared(dir.path().join("canonical-records.wal")).await?;
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "one live appender per file"
    );
    let a = first
        .append_operations(vec![upsert("a", "a1")], None)
        .await?;
    let b = second
        .append_operations(vec![upsert("a", "a2")], None)
        .await?;
    assert_eq!((a[0].sequence_number, b[0].sequence_number), (1, 2));

    // A different file is a different appender.
    let other = FramedTableWalAppender::open_shared(dir.path().join("other.wal")).await?;
    assert!(!std::sync::Arc::ptr_eq(&first, &other));

    // Once every handle is gone the file is released; the next open recovers
    // it afresh and continues the sequence.
    drop((first, second));
    let reopened = FramedTableWalAppender::open_shared(&path).await?;
    let c = reopened
        .append_operations(vec![upsert("a", "a3")], None)
        .await?;
    assert_eq!(c[0].sequence_number, 3);
    Ok(())
}

fn branch_upsert(collection_id: &str, oid: &str, branch: &str) -> CanonicalOperation {
    CanonicalOperation::RecordUpsert {
        collection_id: collection_id.to_string(),
        record: Box::new(ProximaRecord {
            oid: oid.to_string(),
            branch_id: Some(branch.to_string()),
            ..ProximaRecord::default()
        }),
        projections: Vec::new(),
    }
}

/// seq 1 `a` main, seq 2 `b` main, seq 3 `a` on branch `x` (tenant `t1`), seq 4
/// `c` main — each sealed alone — then seq 5 `b` on branch `y` in the active file.
async fn branched_wal(path: &Path) -> Result<FramedTableWalAppender> {
    let wal = FramedTableWalAppender::open(path).await?;
    wal.enable_segment_rotation(1);
    wal.append_operations(vec![upsert("a", "r1")], None).await?;
    wal.append_operations(vec![upsert("b", "r2")], None).await?;
    wal.append_operations(vec![branch_upsert("a", "r3", "x")], Some("t1".into()))
        .await?;
    wal.append_operations(vec![upsert("c", "r4")], None).await?;
    wal.enable_segment_rotation(0);
    wal.append_operations(vec![branch_upsert("b", "r5", "y")], None)
        .await?;
    assert_eq!(sealed_segments(path).len(), 4);
    Ok(wal)
}

#[tokio::test]
async fn branch_summary_and_collections_below_survive_reopen() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = branched_wal(&path).await?;

    let summary = wal.branch_summary();
    assert_eq!(summary.oldest_branch_seq, Some(3));
    assert_eq!(
        summary.collections.iter().cloned().collect::<Vec<_>>(),
        vec![
            BranchedCollection::new(None, "b"),
            BranchedCollection::new(Some("t1"), "a"),
        ]
    );
    let below = wal.collections_below(4);
    assert_eq!(
        below.into_iter().collect::<Vec<_>>(),
        vec![("a".to_string(), 1), ("b".to_string(), 2)],
        "seq 1..=3 sit in segments wholly below 4; each collection maps to its oldest one"
    );
    assert!(wal.collections_below(1).is_empty());
    drop(wal);

    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(reopened.branch_summary(), summary);
    assert_eq!(reopened.collections_below(4), {
        let mut m = std::collections::BTreeMap::new();
        m.insert("a".to_string(), 1);
        m.insert("b".to_string(), 2);
        m
    });

    // Truncating the main-line prefix keeps every branch entry.
    reopened.truncate_below(3).await?;
    assert_eq!(reopened.retained_from(), 3);
    drop(reopened);
    let reopened = FramedTableWalAppender::open(&path).await?;
    assert_eq!(reopened.branch_summary(), summary);
    assert_eq!(
        reopened
            .collections_below(5)
            .into_iter()
            .collect::<Vec<_>>(),
        vec![("a".to_string(), 3), ("c".to_string(), 4)]
    );
    Ok(())
}

#[tokio::test]
async fn a_branch_fence_refuses_stripping_a_collection_branched_since() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = FramedTableWalAppender::open(&path).await?;
    wal.enable_segment_rotation(1);
    wal.append_operations(vec![upsert("a", "r1")], None).await?;
    wal.append_operations(vec![upsert("b", "r2")], None).await?;
    wal.append_operations(vec![branch_upsert("a", "r3", "x")], None)
        .await?;

    // The image covers `a`; `b` branches before the truncation runs.
    let covered: std::collections::BTreeSet<_> =
        [BranchedCollection::new(None, "a")].into_iter().collect();
    wal.fence_branches(covered.clone());
    wal.append_operations(vec![branch_upsert("b", "r4", "y")], None)
        .await?;
    let refused = wal.truncate_below(3).await.unwrap_err();
    assert!(
        format!("{refused:#}").contains("/b"),
        "the refusal names the uncovered collection: {refused:#}"
    );
    assert_eq!(
        wal.retained_from(),
        0,
        "a refused truncation removes nothing"
    );

    // The fence is consumed: an unfenced run truncates.
    assert_eq!(wal.truncate_below(3).await?.truncated_through, 2);

    // A collection branched since with no history below the floor does not
    // trip the fence.
    wal.fence_branches(covered);
    wal.append_operations(vec![branch_upsert("d", "r5", "z")], None)
        .await?;
    assert_eq!(wal.truncate_below(4).await?.truncated_through, 3);
    Ok(())
}
