//! TD-SEAM-12 hazard 2: branch merge after canonical-WAL truncation.
//!
//! The truncator's branch merge-base source images every branched
//! collection's main-line base before truncating below the oldest branch
//! entry; the merge endpoint then takes its base from that image. These tests
//! pin that a merge after truncating below a branch point is identical to the
//! untruncated merge, survives a restart, and that a collection that lost
//! history before it was branched refuses (410) instead of merging against a
//! partial base.
//!
//!   cargo test --test wal_branch_merge_base_test

use anyhow::Result;
use proximadb::errors::ApiError;
use proximadb::network::rest::handlers::{GraphBranchMergeRequest, merge_graph_branch_inner};
use proximadb::services::FramedTableWalAppender;
use proximadb::services::branch_base_store::BranchBaseStore;
use proximadb::services::record_store::TableWalAppender;
use proximadb::services::wal_truncator::{
    self, BranchMergeBase, ChangeFeedRetention, TruncationRun, WalFloorSource,
};
use proximadb_data_model::ProximaValue;
use proximadb_records::{LabelSet, ProximaRecord, ProximaTreeNode};
use proximadb_storage_common::CanonicalOperation;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;

const COLLECTION: &str = "col";

fn wal_path(data_dir: &Path) -> PathBuf {
    data_dir.join("pgwire").join("canonical-records.wal")
}

fn record(oid: &str, branch: Option<&str>, name: &str, labels: &[&str]) -> ProximaRecord {
    let mut record = ProximaRecord {
        oid: oid.to_string(),
        branch_id: branch.map(str::to_string),
        labels: LabelSet::from(labels.iter().map(|l| l.to_string()).collect::<Vec<_>>()),
        ..ProximaRecord::default()
    };
    record.props.insert(
        "name".to_string(),
        ProximaTreeNode::Value(ProximaValue::String(name.to_string())),
    );
    record
}

async fn put(wal: &FramedTableWalAppender, collection: &str, record: ProximaRecord) -> Result<u64> {
    let op = CanonicalOperation::RecordUpsert {
        collection_id: collection.to_string(),
        record: Box::new(record),
        projections: Vec::new(),
    };
    Ok(wal.append_operations(vec![op], None).await?[0].sequence_number)
}

async fn delete(wal: &FramedTableWalAppender, oid: &str) -> Result<()> {
    let op = CanonicalOperation::RecordDelete {
        collection_id: COLLECTION.to_string(),
        oid: oid.to_string(),
        projections: Vec::new(),
    };
    wal.append_operations(vec![op], None).await?;
    Ok(())
}

/// Main-line history (with a partition drop and a delete) the base
/// classification depends on,
/// then two branches touching the same oids, with a main write in between.
/// Every append seals its own segment, so truncation can remove any prefix.
/// Returns the first branch write's sequence.
async fn seed(wal: &FramedTableWalAppender) -> Result<u64> {
    wal.enable_segment_rotation(1);
    put(wal, COLLECTION, record("n0", None, "gone", &[])).await?;
    put(wal, COLLECTION, record("n1", None, "dropped", &[])).await?;
    let drop = CanonicalOperation::RecordPartitionDrop {
        collection_id: COLLECTION.to_string(),
    };
    wal.append_operations(vec![drop], None).await?;
    put(wal, COLLECTION, record("n1", None, "v0", &["x"])).await?;
    put(wal, COLLECTION, record("n2", None, "v0", &[])).await?;
    put(wal, COLLECTION, record("n1", None, "v1", &["x", "y"])).await?;
    put(wal, COLLECTION, record("n3", None, "v0", &[])).await?;
    delete(wal, "n3").await?;
    put(wal, "other", record("o1", None, "v0", &[])).await?;
    let fork = put(
        wal,
        COLLECTION,
        record("n1", Some("a"), "left", &["x", "y"]),
    )
    .await?;
    put(wal, COLLECTION, record("n2", None, "main-after", &[])).await?;
    put(
        wal,
        COLLECTION,
        record("n1", Some("b"), "right", &["x", "y", "z"]),
    )
    .await?;
    put(wal, COLLECTION, record("n2", Some("a"), "left", &[])).await?;
    put(wal, COLLECTION, record("n3", Some("b"), "back", &[])).await?;
    Ok(fork)
}

async fn merge(data_dir: &Path, collection: &str) -> Result<serde_json::Value, ApiError> {
    let request: GraphBranchMergeRequest =
        serde_json::from_value(serde_json::json!({ "target_branch": "b", "dry_run": true }))
            .expect("request");
    merge_graph_branch_inner(data_dir, "default", collection, "a", request)
        .await
        .map(|json| json.0)
}

fn sources(wal: &Arc<FramedTableWalAppender>) -> Vec<Arc<dyn WalFloorSource>> {
    vec![
        Arc::new(ChangeFeedRetention(0)),
        Arc::new(BranchMergeBase(Arc::new(BranchBaseStore::new(wal.clone())))),
    ]
}

#[tokio::test]
async fn a_merge_after_truncating_below_the_branch_point_equals_the_untruncated_merge() -> Result<()>
{
    let dir = TempDir::new()?;
    let data_dir = dir.path().to_path_buf();
    let wal = FramedTableWalAppender::open_shared(wal_path(&data_dir)).await?;
    let fork = seed(&wal).await?;

    let untruncated = merge(&data_dir, COLLECTION)
        .await
        .expect("full history merges");
    assert!(
        untruncated["summary"]["conflict_count"].as_u64() > Some(0),
        "{untruncated}"
    );

    let run = wal_truncator::run_once(&wal, &sources(&wal)).await?;
    assert!(
        matches!(run, TruncationRun::Truncated { floor, .. } if floor == fork),
        "the truncation stops at the first branch write: {run:?}"
    );
    assert_eq!(
        wal.retained_from(),
        fork,
        "the whole main-line base is gone"
    );

    let truncated = merge(&data_dir, COLLECTION)
        .await
        .expect("the image gives the base");
    assert_eq!(truncated, untruncated);

    // And across a restart, read back from the WAL alone.
    drop(wal);
    let truncated = merge(&data_dir, COLLECTION).await.expect("after a restart");
    assert_eq!(truncated, untruncated);
    Ok(())
}

#[tokio::test]
async fn a_collection_branched_after_its_history_was_truncated_refuses() -> Result<()> {
    let dir = TempDir::new()?;
    let data_dir = dir.path().to_path_buf();
    let wal = FramedTableWalAppender::open_shared(wal_path(&data_dir)).await?;
    seed(&wal).await?;
    wal_truncator::run_once(&wal, &sources(&wal)).await?;

    // A collection whose history was truncated but that has nothing to merge
    // is a 404, not a 410.
    assert!(matches!(
        merge(&data_dir, "other").await,
        Err(ApiError::NotFound(_))
    ));

    // `other` lost its main-line history while unbranched.
    put(&wal, "other", record("o1", Some("a"), "left", &[])).await?;
    put(&wal, "other", record("o1", Some("b"), "right", &[])).await?;
    match merge(&data_dir, "other").await {
        Err(ApiError::Gone(reason)) => assert!(reason.contains("'other'"), "{reason}"),
        other => panic!("expected 410 Gone, got {other:?}"),
    }
    // The imaged collection still merges.
    assert!(merge(&data_dir, COLLECTION).await.is_ok());

    Ok(())
}
