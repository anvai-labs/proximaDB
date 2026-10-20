//! P2.2b (TD-SEAM-12 hazard 1): the rank-profile and SQL-function stores
//! rebuild from the canonical WAL, so they need a way to stop depending on
//! history before the WAL can be truncated. `checkpoint()` re-appends the live
//! set and returns the store's floor; truncating every segment below it must
//! leave a reopened store identical to the live one.
//!
//!   cargo test --test wal_system_catalog_checkpoint_test

use anyhow::Result;
use proximadb::services::record_store::TableWalAppender;
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

fn function(name: &str, body: &str) -> StoredFunction {
    StoredFunction {
        name: name.to_string(),
        params: vec![("x".to_string(), ProximaType::Int64)],
        return_ty: ProximaType::Int64,
        body: body.to_string(),
        created_at_ms: 0,
    }
}

/// Unrelated table traffic, so the system entries are interleaved with other
/// segments exactly as on a live server.
async fn table_write(wal: &FramedTableWalAppender, oid: &str) -> Result<()> {
    wal.append_operations(
        vec![CanonicalOperation::RecordUpsert {
            collection_id: "orders".to_string(),
            record: Box::new(ProximaRecord {
                oid: oid.to_string(),
                ..ProximaRecord::default()
            }),
            projections: Vec::new(),
        }],
        None,
    )
    .await?;
    Ok(())
}

#[tokio::test]
async fn a_checkpoint_lets_truncation_drop_all_older_catalog_history() -> Result<()> {
    let dir = TempDir::new()?;
    let path = dir.path().join("canonical-records.wal");
    let wal = Arc::new(FramedTableWalAppender::open(&path).await?);
    wal.enable_segment_rotation(1); // every append seals a segment

    let profiles = CanonicalWalRankProfileStore::new(wal.clone());
    let functions = CanonicalWalFunctionStore::new(wal.clone());

    profiles
        .install("keep", PROFILE.to_string(), Some("t1".into()), None)
        .await?;
    profiles
        .install("keep", PROFILE.to_string(), Some("t1".into()), None)
        .await?; // version 2
    profiles
        .install("gone", PROFILE.to_string(), None, None)
        .await?;
    profiles.remove("gone").await?;
    functions.put(function("double", "x * 2")).await?;
    functions.put(function("double", "x + x")).await?; // replaced
    functions.put(function("dropped", "x")).await?;
    functions.remove("dropped").await?;
    table_write(&wal, "o1").await?;

    // Each store's floor; the WAL may drop everything below the lower one.
    let profile_floor = profiles.checkpoint().await?.expect("a live profile");
    let function_floor = functions.checkpoint().await?.expect("a live function");
    let floor = profile_floor.min(function_floor);
    let truncated = wal.truncate_below(floor).await?;
    assert!(
        truncated.segments >= 9,
        "the pre-checkpoint history went: {truncated:?}"
    );

    // Restart: rebuild both stores from the retained WAL only.
    drop((profiles, functions));
    let entries = FramedTableWalAppender::read_entries_from_path(&path).await?;
    assert!(entries.iter().all(|e| e.sequence_number >= floor));
    let profiles = CanonicalWalRankProfileStore::from_wal_entries(wal.clone(), &entries);
    let functions = CanonicalWalFunctionStore::from_wal_entries(wal.clone(), &entries);

    let keep = profiles.get("keep").await?.expect("live profile survives");
    assert_eq!(
        (keep.version, keep.tenant.as_deref()),
        (2, Some("t1")),
        "latest version and tenant"
    );
    assert!(
        profiles.get("gone").await?.is_none(),
        "removed stays removed"
    );
    let all = functions.list_all().await?;
    assert_eq!(all.len(), 1);
    assert_eq!(
        (all[0].name.as_str(), all[0].body.as_str()),
        ("double", "x + x")
    );
    Ok(())
}

#[tokio::test]
async fn an_empty_store_imposes_no_floor() -> Result<()> {
    let dir = TempDir::new()?;
    let wal = Arc::new(FramedTableWalAppender::open(dir.path().join("w.wal")).await?);
    let profiles = CanonicalWalRankProfileStore::new(wal.clone());
    let functions = CanonicalWalFunctionStore::new(wal.clone());
    profiles
        .install("p", PROFILE.to_string(), None, None)
        .await?;
    profiles.remove("p").await?;
    assert_eq!(profiles.checkpoint().await?, None);
    assert_eq!(functions.checkpoint().await?, None);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_checkpoint_never_replays_an_older_value_over_a_concurrent_install() -> Result<()> {
    // Installs race checkpoints; whatever interleaving happens, replaying the
    // WAL must end on the latest installed version.
    let dir = TempDir::new()?;
    let path = dir.path().join("w.wal");
    let wal = Arc::new(FramedTableWalAppender::open(&path).await?);
    let profiles = Arc::new(CanonicalWalRankProfileStore::new(wal.clone()));
    profiles
        .install("p", PROFILE.to_string(), None, None)
        .await?;

    let installer = {
        let profiles = profiles.clone();
        tokio::spawn(async move {
            for _ in 0..50 {
                profiles
                    .install("p", PROFILE.to_string(), None, None)
                    .await?;
            }
            anyhow::Ok(())
        })
    };
    let checkpointer = {
        let profiles = profiles.clone();
        tokio::spawn(async move {
            for _ in 0..50 {
                profiles.checkpoint().await?;
            }
            anyhow::Ok(())
        })
    };
    installer.await??;
    checkpointer.await??;

    let live = profiles.get("p").await?.expect("live").version;
    assert_eq!(live, 51);
    let entries = FramedTableWalAppender::read_entries_from_path(&path).await?;
    let replayed = CanonicalWalRankProfileStore::from_wal_entries(wal, &entries);
    assert_eq!(
        replayed.get("p").await?.expect("replayed").version,
        live,
        "replay must end on the newest install, never a checkpoint's older copy"
    );
    Ok(())
}
