//! P2.2c (TD-SEAM-2 / TD-SEAM-12): the canonical-WAL truncator composes every
//! consumer's floor and truncates below the minimum — or refuses.
//!
//!   cargo test --test wal_truncator_test

use anyhow::Result;
use proximadb::services::canonical_wal::WalHistoryTruncated;
use proximadb::services::record_store::{
    DirectWalTableRecordStore, TableRecordStore, TableWalAppender,
};
use proximadb::services::wal_truncator::{
    self, BranchEntriesGuard, ChangeFeedRetention, FunctionCheckpoint, PublishedSnapshotFloor,
    RankProfileCheckpoint, RecordTierFloor, TruncationRun, WalFloorSource,
};
use proximadb::services::{
    CanonicalWalFunctionStore, CanonicalWalRankProfileStore, FramedTableWalAppender, FunctionStore,
    RankProfileStore, StoredFunction,
};
use proximadb_data_model::ProximaType;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::CanonicalOperation;
use std::sync::Arc;
use tempfile::TempDir;

const PROFILE: &str = "[first_phase]\nexpression = \"1.0\"\nheap_size = 100\n";

fn upsert(oid: &str, branch: Option<&str>) -> CanonicalOperation {
    CanonicalOperation::RecordUpsert {
        collection_id: "orders".to_string(),
        record: Box::new(ProximaRecord {
            oid: oid.to_string(),
            branch_id: branch.map(str::to_string),
            ..ProximaRecord::default()
        }),
        projections: Vec::new(),
    }
}

/// A WAL where every append seals its own segment.
async fn rotating_wal(dir: &TempDir) -> Result<Arc<FramedTableWalAppender>> {
    let wal =
        Arc::new(FramedTableWalAppender::open(dir.path().join("canonical-records.wal")).await?);
    wal.enable_segment_rotation(1);
    Ok(wal)
}

/// A fixed answer, to isolate the composition logic.
struct Fixed(&'static str, Result<Option<u64>, &'static str>);

#[async_trait::async_trait]
impl WalFloorSource for Fixed {
    fn name(&self) -> &str {
        self.0
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        self.1.map_err(|reason| anyhow::anyhow!(reason))
    }
}

#[tokio::test]
async fn the_floor_is_the_minimum_and_one_refusal_stops_the_run() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    for i in 1..=10 {
        wal.append_operations(vec![upsert(&format!("r{i}"), None)], None)
            .await?;
    }

    // A refusal deletes nothing, even when another source would allow it.
    let run = wal_truncator::run_once(
        &wal,
        &[
            Arc::new(Fixed("open", Ok(None))),
            Arc::new(Fixed("blocker", Err("not yet"))),
        ],
    )
    .await?;
    assert_eq!(
        run,
        TruncationRun::Refused {
            source: "blocker".into(),
            reason: "not yet".into()
        }
    );
    assert_eq!(wal.retained_from(), 0);

    // The lowest floor wins.
    let run = wal_truncator::run_once(
        &wal,
        &[
            Arc::new(Fixed("a", Ok(Some(8)))),
            Arc::new(Fixed("b", Ok(Some(5)))),
            Arc::new(Fixed("none", Ok(None))),
        ],
    )
    .await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor: 5, .. }),
        "{run:?}"
    );
    assert_eq!(wal.retained_from(), 5);

    // Change-feed retention keeps the newest N sequences.
    let run = wal_truncator::run_once(&wal, &[Arc::new(ChangeFeedRetention(3))]).await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor: 8, .. }),
        "{run:?}"
    );
    let seqs: Vec<u64> = wal
        .read_entries()
        .await?
        .iter()
        .map(|e| e.sequence_number)
        .collect();
    assert_eq!(seqs, [8, 9, 10]);

    // Nothing new below the floor is a no-op.
    let run = wal_truncator::run_once(&wal, &[Arc::new(ChangeFeedRetention(3))]).await?;
    assert!(matches!(run, TruncationRun::NothingToDo { .. }), "{run:?}");
    Ok(())
}

#[tokio::test]
async fn branch_tagged_records_block_truncation() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    wal.append_operations(vec![upsert("main", None)], None)
        .await?;
    let guard: Arc<dyn WalFloorSource> = Arc::new(BranchEntriesGuard(wal.clone()));
    assert_eq!(guard.floor(wal.last_sequence()).await?, None);

    wal.append_operations(vec![upsert("b", Some("feature"))], None)
        .await?;
    let run = wal_truncator::run_once(&wal, &[guard, Arc::new(ChangeFeedRetention(0))]).await?;
    assert!(
        matches!(&run, TruncationRun::Refused { source, .. } if source == "branch-merge"),
        "{run:?}"
    );
    assert_eq!(wal.retained_from(), 0, "nothing deleted");

    // The guard survives a restart (rebuilt from the retained segments).
    drop(wal);
    let reopened =
        Arc::new(FramedTableWalAppender::open(dir.path().join("canonical-records.wal")).await?);
    assert!(reopened.has_branch_entries());
    Ok(())
}

#[tokio::test]
async fn a_memtable_record_tier_refuses() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    let store = Arc::new(DirectWalTableRecordStore::new_partitioned(wal.clone()));
    store
        .replay_wal_entries(vec![
            wal.append_operations(vec![upsert("r1", None)], None)
                .await?
                .remove(0),
        ])
        .await?;
    let run = wal_truncator::run_once(
        &wal,
        &[
            Arc::new(RecordTierFloor(store)),
            Arc::new(ChangeFeedRetention(0)),
        ],
    )
    .await?;
    assert!(
        matches!(&run, TruncationRun::Refused { source, .. } if source == "record-tier"),
        "a memtable partition's rows exist only in memory: {run:?}"
    );
    Ok(())
}

#[tokio::test]
async fn an_empty_catalog_manager_imposes_no_snapshot_floor() -> Result<()> {
    let source = PublishedSnapshotFloor(Arc::new(proximadb::catalog::CatalogManager::new()));
    assert_eq!(source.floor(0).await?, None);
    Ok(())
}

#[tokio::test]
async fn catalog_checkpoints_let_a_run_drop_their_history() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    let profiles: Arc<dyn RankProfileStore> =
        Arc::new(CanonicalWalRankProfileStore::new(wal.clone()));
    let functions: Arc<dyn FunctionStore> = Arc::new(CanonicalWalFunctionStore::new(wal.clone()));
    for _ in 0..3 {
        profiles
            .install("p", PROFILE.to_string(), None, None)
            .await?;
    }
    functions
        .put(StoredFunction {
            name: "double".into(),
            params: vec![("x".into(), ProximaType::Int64)],
            return_ty: ProximaType::Int64,
            body: "x * 2".into(),
            created_at_ms: 0,
        })
        .await?;
    let before_head = wal.last_sequence();

    let run = wal_truncator::run_once(
        &wal,
        &[
            Arc::new(BranchEntriesGuard(wal.clone())),
            Arc::new(ChangeFeedRetention(0)),
            Arc::new(RankProfileCheckpoint(profiles.clone())),
            Arc::new(FunctionCheckpoint(functions.clone())),
        ],
    )
    .await?;
    // The checkpoints appended after the head the run read, so the floor is
    // the head + 1: every pre-checkpoint entry goes, the checkpoints stay.
    assert!(
        matches!(run, TruncationRun::Truncated { floor, .. } if floor == before_head + 1),
        "{run:?}"
    );

    let entries =
        FramedTableWalAppender::read_entries_from_path(dir.path().join("canonical-records.wal"))
            .await?;
    let profiles_again = CanonicalWalRankProfileStore::from_wal_entries(wal.clone(), &entries);
    let functions_again = CanonicalWalFunctionStore::from_wal_entries(wal.clone(), &entries);
    assert_eq!(profiles_again.get("p").await?.expect("profile").version, 3);
    assert_eq!(functions_again.list_all().await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_snapshot_pin_holds_the_floor_until_dropped() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    for i in 1..=5 {
        wal.append_operations(vec![upsert(&format!("r{i}"), None)], None)
            .await?;
    }
    let (lsn, pin) = wal.pin_history(&|head| head - 2).await?;
    assert_eq!((lsn, pin.floor()), (3, 4));

    // Nothing above the pinned LSN goes, whatever the caller's floor.
    let run = wal_truncator::run_once(&wal, &[Arc::new(ChangeFeedRetention(0))]).await?;
    assert!(
        matches!(&run, TruncationRun::Truncated { report, .. } if report.truncated_through == 3),
        "{run:?}"
    );
    assert_eq!(wal.retained_from(), 4);

    // History below the retained floor can no longer be pinned.
    let err = wal
        .pin_history(&|_| 1)
        .await
        .expect_err("entries after 1 are gone");
    assert!(
        err.downcast_ref::<WalHistoryTruncated>().is_some(),
        "{err:#}"
    );

    drop(pin);
    wal.truncate_below(u64::MAX).await?;
    assert_eq!(wal.retained_from(), 6, "released: the rest can go");
    Ok(())
}

#[tokio::test]
async fn a_snapshot_capture_after_truncation_is_the_head_not_zero() -> Result<()> {
    // Before P2.2c, MATERIALIZE derived its snapshot LSN from the change feed
    // read from 0 and swallowed errors as 0: after a truncation that would
    // record `snapshot_lsn = 0`, which no retained history can serve.
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    for i in 1..=3 {
        wal.append_operations(vec![upsert(&format!("r{i}"), None)], None)
            .await?;
    }
    wal.truncate_below(u64::MAX).await?;
    let store = DirectWalTableRecordStore::new_partitioned(wal.clone());

    let snapshot = store.capture_snapshot_lsn("orders").await?;
    assert_eq!(snapshot.lsn, 3);
    assert_eq!(snapshot.pin.as_ref().map(|p| p.floor()), Some(4));
    assert!(
        store
            .read_changes_since("orders", snapshot.lsn)
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn a_branch_fence_refusal_is_reported_as_a_refusal() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = rotating_wal(&dir).await?;
    wal.append_operations(vec![upsert("r1", None)], None)
        .await?;
    wal.append_operations(vec![upsert("r2", None)], None)
        .await?;
    // Armed with nothing recorded: every victim's loss would go unrecorded.
    wal.fence_branches(Default::default(), Default::default());
    let run = wal_truncator::run_once(&wal, &[Arc::new(ChangeFeedRetention(0))]).await?;
    assert!(
        matches!(&run, TruncationRun::Refused { source, .. } if source == "branch-fence"),
        "{run:?}"
    );
    assert_eq!(wal.retained_from(), 0, "nothing deleted");
    Ok(())
}
