//! P2.2 gate (TD-SEAM-2 / TD-SEAM-3): with Layer A spilling, segment rotation
//! and the scheduled truncator all on, a crash after truncation recovers every
//! row — flushed rows from the spill segments, the rest from the retained WAL —
//! and no delete or update is lost or undone. This is the end-to-end check that
//! must hold before rotation and truncation can default on.
//!
//! A memory-only tombstone holds the floor at its delete's LSN until a flush
//! persists it.
//!
//! The seeded soak below interleaves the same operations at random and checks
//! every restart against a reference model.
//!
//!   cargo test --test wal_truncation_recovery_test

use anyhow::{Context, Result};
use proximadb::services::FramedTableWalAppender;
use proximadb::services::branch_base_store::BranchBaseStore;
use proximadb::services::record_store::{
    DirectWalTableRecordStore, TableRecordMutation, TableRecordMutationKind,
    TableRecordScanRequest, TableRecordStore, TableWalAppender,
};
use proximadb::services::wal_truncator::{
    self, BranchMergeBase, ChangeFeedRetention, RecordTierFloor, TruncationRun, WalFloorSource,
};
use proximadb::storage::tenant::context::StorageTenantContext;
use proximadb_catalog::CatalogTableSchema;
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};
use proximadb_storage_common::CanonicalOperation;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Once};
use tempfile::TempDir;

const TABLE: &str = "orders";
/// One untenanted partition (a single-tenant server's only key) and one tenant.
const TENANTS: [&str; 2] = ["", "t1"];

fn upsert(oid: &str, tenant: &str, version: i64) -> CanonicalOperation {
    let mut record = ProximaRecord {
        oid: oid.to_string(),
        tenant_id: tenant.to_string(),
        ..ProximaRecord::default()
    };
    record.props.insert(
        "version".to_string(),
        ProximaTreeNode::Value(ProximaValue::Int64(version)),
    );
    CanonicalOperation::RecordUpsert {
        collection_id: TABLE.to_string(),
        record: Box::new(record),
        projections: Vec::new(),
    }
}

fn delete(oid: &str) -> CanonicalOperation {
    CanonicalOperation::RecordDelete {
        collection_id: TABLE.to_string(),
        oid: oid.to_string(),
        projections: Vec::new(),
    }
}

fn tenant_arg(tenant: &str) -> Option<String> {
    (!tenant.is_empty()).then(|| tenant.to_string())
}

/// Append `op` for `tenant` and apply it — the order every server write uses.
async fn write(
    wal: &FramedTableWalAppender,
    store: &DirectWalTableRecordStore,
    tenant: &str,
    op: CanonicalOperation,
) -> Result<u64> {
    let entries = wal.append_operations(vec![op], tenant_arg(tenant)).await?;
    let lsn = entries[0].sequence_number;
    store.replay_wal_entries(entries).await?;
    Ok(lsn)
}

/// Every visible row of `tenant`'s partition, as `oid → version`.
async fn rows(store: &DirectWalTableRecordStore, tenant: &str) -> Result<BTreeMap<String, i64>> {
    let context = StorageTenantContext::for_tenant_id(tenant);
    #[cfg(feature = "abac-policy")]
    let read_context = proximadb_abac::ReadContext::system(
        proximadb_abac::SystemReadReason::Statistics,
        "wal_truncation_recovery_test",
    );
    let records = store
        .scan_records(
            &CatalogTableSchema::new(TABLE),
            TableRecordScanRequest {
                table_id: TABLE.to_string(),
                include_props: true,
                ..TableRecordScanRequest::default()
            },
            (!tenant.is_empty()).then_some(&context),
            #[cfg(feature = "abac-policy")]
            &read_context,
        )
        .await?;
    Ok(records
        .into_iter()
        .map(|record| {
            let version = match record.props.get("version") {
                Some(ProximaTreeNode::Value(ProximaValue::Int64(v))) => *v,
                other => panic!("row {} lost its version prop: {other:?}", record.oid),
            };
            (record.oid, version)
        })
        .collect())
}

/// Turn the spill gate on once for the whole binary. A threshold well above
/// any partition's row count, so only explicit flushes spill and each test
/// controls exactly which rows are durable in segments. Set once and never
/// removed: `cargo test` runs these tests concurrently in one process.
fn spill_gate() {
    static GATE: Once = Once::new();
    // SAFETY: inside `call_once`, before any test reads the variable.
    GATE.call_once(|| unsafe {
        std::env::set_var("PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT", "1000")
    });
}

fn truncator_sources(
    wal: &Arc<FramedTableWalAppender>,
    store: &Arc<DirectWalTableRecordStore>,
) -> Vec<Arc<dyn WalFloorSource>> {
    vec![
        Arc::new(ChangeFeedRetention(0)),
        Arc::new(RecordTierFloor(store.clone())),
        Arc::new(BranchMergeBase(Arc::new(BranchBaseStore::new(wal.clone())))),
    ]
}

async fn open(dir: &Path) -> Result<(Arc<FramedTableWalAppender>, DirectWalTableRecordStore)> {
    let wal = Arc::new(FramedTableWalAppender::open(dir.join("canonical-records.wal")).await?);
    // Every append seals its own segment, so truncation can drop any prefix.
    wal.enable_segment_rotation(1);
    let store = DirectWalTableRecordStore::open_for_server(
        wal.clone() as Arc<dyn TableWalAppender>,
        dir.join("record-spill"),
    )
    .await?;
    Ok((wal, store))
}

#[tokio::test]
async fn a_crash_after_truncation_recovers_every_row_from_segments_and_the_wal_tail() -> Result<()>
{
    spill_gate();
    let dir = TempDir::new()?;
    let (wal, store) = open(dir.path()).await?;
    let store = Arc::new(store);

    // Phase 1, made durable in Layer A: r0..r9 at version 1, r0 deleted.
    for tenant in TENANTS {
        for i in 0..10 {
            write(&wal, &store, tenant, upsert(&format!("r{i}"), tenant, 1)).await?;
        }
        write(&wal, &store, tenant, delete("r0")).await?;
    }
    store.flush_partitions().await?;

    let sources = truncator_sources(&wal, &store);

    // Phase 2: a delete of a flushed row, still memory-only. Its tombstone
    // keeps the delete's LSN, so the record tier retains the WAL from there —
    // truncating that entry would resurrect the row on restart — and drops
    // everything before it.
    let mut first_delete = None;
    for tenant in TENANTS {
        let lsn = write(&wal, &store, tenant, delete("r2")).await?;
        first_delete.get_or_insert(lsn);
    }
    let first_delete = first_delete.expect("phase 2 wrote");
    let run = wal_truncator::run_once(&wal, &sources).await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor, .. } if floor == first_delete),
        "an unpersisted tombstone is a floor at its delete: {run:?}"
    );
    assert_eq!(wal.retained_from(), first_delete);
    store.flush_partitions().await?;

    // Phase 3, only in the WAL and memory: an update and a new row.
    let mut first_unflushed = None;
    for tenant in TENANTS {
        let lsn = write(&wal, &store, tenant, upsert("r1", tenant, 2)).await?;
        first_unflushed.get_or_insert(lsn);
        write(&wal, &store, tenant, upsert("r10", tenant, 1)).await?;
    }
    let first_unflushed = first_unflushed.expect("phase 3 wrote");

    // The truncator drops everything the segments hold and nothing after.
    let run = wal_truncator::run_once(&wal, &sources).await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor, .. } if floor == first_unflushed),
        "the floor is the oldest unflushed write: {run:?}"
    );
    assert_eq!(wal.retained_from(), first_unflushed);

    let mut expected = BTreeMap::new();
    for i in 1..=10 {
        if i != 2 {
            expected.insert(format!("r{i}"), if i == 1 { 2 } else { 1 });
        }
    }
    for tenant in TENANTS {
        assert_eq!(rows(&store, tenant).await?, expected, "before the crash");
    }

    // Crash: no shutdown flush. Recovery is the server's boot sequence —
    // reopen, replay the retained WAL, serve.
    drop(sources);
    drop(store);
    drop(wal);
    let (wal, store) = open(dir.path()).await?;
    assert_eq!(
        wal.retained_from(),
        first_unflushed,
        "the floor marker survives"
    );
    store.replay_wal_entries(wal.read_entries().await?).await?;
    for tenant in TENANTS {
        assert_eq!(
            rows(&store, tenant).await?,
            expected,
            "after recovery, tenant {tenant:?}: segments + WAL tail, deletes and updates kept"
        );
    }

    // A second truncation after recovery: flush, then everything goes, and a
    // further restart still serves every row from segments alone.
    store.flush_partitions().await?;
    let store = Arc::new(store);
    let run = wal_truncator::run_once(
        &wal,
        &[
            Arc::new(ChangeFeedRetention(0)) as Arc<dyn WalFloorSource>,
            Arc::new(RecordTierFloor(store.clone())),
        ],
    )
    .await?;
    assert!(
        matches!(run, TruncationRun::Truncated { .. }),
        "after the flush nothing pins the WAL: {run:?}"
    );
    drop(store);
    drop(wal);
    let (wal, store) = open(dir.path()).await?;
    assert!(
        wal.read_entries().await?.is_empty(),
        "every WAL entry was truncated"
    );
    for tenant in TENANTS {
        assert_eq!(
            rows(&store, tenant).await?,
            expected,
            "segments alone, tenant {tenant:?}"
        );
    }
    Ok(())
}

/// Seeded interleavings of upserts, deletes (of resident, flushed and absent
/// rows), flushes, truncator runs and crash-restarts over both partitions. Each
/// restart replays the retained WAL over the segments and must reproduce the
/// reference model exactly. Every write carries its LSN, so the record tier
/// must never refuse a run.
#[tokio::test]
async fn seeded_crash_truncate_interleavings_match_a_reference_model() -> Result<()> {
    spill_gate();
    let mut over_pending_delete = 0;
    for seed in 0..8 {
        over_pending_delete += soak(seed, 300)
            .await
            .with_context(|| format!("soak seed {seed}"))?;
    }
    // The case #2132 exists for: a truncation that stops at a memory-only
    // tombstone, then a crash. Some seed must have hit it.
    assert!(
        over_pending_delete > 0,
        "no seed truncated while a delete was still memory-only"
    );
    Ok(())
}

/// Returns how many truncations ran while a delete was not yet flushed.
async fn soak(seed: u64, steps: usize) -> Result<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dir = TempDir::new()?;
    let (mut wal, store) = open(dir.path()).await?;
    let mut store = Arc::new(store);
    let mut model: BTreeMap<&str, BTreeMap<String, i64>> = TENANTS
        .iter()
        .map(|tenant| (*tenant, BTreeMap::new()))
        .collect();
    let mut version = 0;
    let (mut truncations, mut restarts, mut over_pending_delete) = (0, 0, 0);
    // A delete applied since the last flush: its tombstone is memory-only.
    let mut pending_delete = false;

    for step in 0..=steps {
        let tenant = TENANTS[rng.gen_range(0..TENANTS.len())];
        let oid = format!("r{}", rng.gen_range(0..16));
        // The last step is always a restart, so every seed ends verified.
        let roll = if step == steps {
            99
        } else {
            rng.gen_range(0..100)
        };
        match roll {
            0..=52 => {
                version += 1;
                write(&wal, &store, tenant, upsert(&oid, tenant, version)).await?;
                model.get_mut(tenant).expect("tenant").insert(oid, version);
            }
            53..=72 => {
                write(&wal, &store, tenant, delete(&oid)).await?;
                model.get_mut(tenant).expect("tenant").remove(&oid);
                pending_delete = true;
            }
            // DROP TABLE: purges the partition's segments; replay of a
            // retained drop must purge too.
            73..=74 => {
                let drop = CanonicalOperation::RecordPartitionDrop {
                    collection_id: TABLE.to_string(),
                };
                write(&wal, &store, tenant, drop).await?;
                model.get_mut(tenant).expect("tenant").clear();
            }
            75..=84 => {
                store.flush_partitions().await?;
                pending_delete = false;
            }
            85..=94 => {
                let run = wal_truncator::run_once(&wal, &truncator_sources(&wal, &store)).await?;
                assert!(
                    !matches!(run, TruncationRun::Refused { .. }),
                    "seed {seed} step {step}: every write is LSN-tracked: {run:?}"
                );
                let truncated = matches!(run, TruncationRun::Truncated { .. });
                truncations += truncated as usize;
                over_pending_delete += (truncated && pending_delete) as usize;
            }
            _ => {
                for tenant in TENANTS {
                    assert_eq!(
                        rows(&store, tenant).await?,
                        model[tenant],
                        "seed {seed} step {step}: live state, tenant {tenant:?}"
                    );
                }
                // Crash: no shutdown flush.
                drop(store);
                drop(wal);
                let (reopened_wal, reopened) = open(dir.path()).await?;
                reopened
                    .replay_wal_entries(reopened_wal.read_entries().await?)
                    .await?;
                (wal, store) = (reopened_wal, Arc::new(reopened));
                restarts += 1;
                // Replay rebuilt any memory-only tombstone from the WAL.
                for tenant in TENANTS {
                    assert_eq!(
                        rows(&store, tenant).await?,
                        model[tenant],
                        "seed {seed} step {step}: after restart {restarts}, tenant {tenant:?}"
                    );
                }
            }
        }
    }
    assert!(
        truncations > 0 && restarts > 1,
        "seed {seed} exercised too little: {truncations} truncation(s), {restarts} restart(s)"
    );
    Ok(over_pending_delete)
}

/// The production write path (`write_mutations`), not replay: a DELETE there
/// must reach the spill partition with its WAL LSN, so the record tier answers
/// a floor at it. Were the delete applied untracked, the truncator would
/// refuse; were it tracked wrongly, recovery would resurrect the row.
#[tokio::test]
async fn a_delete_through_the_write_path_is_a_truncation_floor_and_recovers() -> Result<()> {
    spill_gate();
    let dir = TempDir::new()?;
    let (wal, store) = open(dir.path()).await?;
    let store = Arc::new(store);
    let schema = CatalogTableSchema::new(TABLE);
    let tenant = StorageTenantContext::for_tenant_id("t1");
    let row = |oid: &str, version: i64| match upsert(oid, "t1", version) {
        CanonicalOperation::RecordUpsert { record, .. } => *record,
        _ => unreachable!(),
    };

    let mutations = (0..4)
        .map(|i| {
            TableRecordMutation::new(TableRecordMutationKind::Upsert, row(&format!("r{i}"), 1))
        })
        .collect();
    store
        .write_mutations(&schema, mutations, Some(&tenant))
        .await?;
    store.flush_partitions().await?;

    store
        .write_mutations(
            &schema,
            vec![TableRecordMutation::new(
                TableRecordMutationKind::Delete,
                row("r1", 1),
            )],
            Some(&tenant),
        )
        .await?;
    let delete_lsn = wal.last_sequence();

    let run = wal_truncator::run_once(&wal, &truncator_sources(&wal, &store)).await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor, .. } if floor == delete_lsn),
        "the write-path delete is the floor: {run:?}"
    );

    drop(store);
    drop(wal);
    let (wal, store) = open(dir.path()).await?;
    store.replay_wal_entries(wal.read_entries().await?).await?;
    let expected: BTreeMap<String, i64> = ["r0", "r2", "r3"]
        .iter()
        .map(|oid| (oid.to_string(), 1))
        .collect();
    assert_eq!(
        rows(&store, "t1").await?,
        expected,
        "the delete survived the crash"
    );
    Ok(())
}
