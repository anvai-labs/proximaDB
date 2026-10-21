//! TD-WLP-2 (ADR-061 D5) integration gate: per-collection compaction override.
//!
//! Compaction arming must resolve per-collection through the tag cascade while
//! the global env (`PROXIMADB_L0_COMPACTION_ENABLED`) stays the master
//! kill-switch:
//!
//! * an `AppendBulk` collection with an explicit `compaction:on` opt-in arms
//!   compaction (at its per-collection `l0_threshold:N`) while the global gate
//!   is OFF;
//! * an untagged collection under the same global-off never arms (today's
//!   default, byte-for-byte);
//! * an explicitly falsy global env hard-disables even opted-in collections.
//!
//! Env-sensitive tests rely on nextest's process-per-test isolation.

use anyhow::Result;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::{FlushParameters, FlushResult, UnifiedStorageEngine};
use proximadb_proto::v1::{Collection, CollectionConfig, StorageAssignment};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

const GLOBAL_GATE_ENV: &str = "PROXIMADB_L0_COMPACTION_ENABLED";

fn record(id: &str) -> ProximaRecord {
    ProximaRecord {
        oid: id.to_string(),
        created_at_ns: 12_345_000_000,
        updated_at_ns: 12_345_000_000,
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32(
            "test",
            "dense_vector",
            4,
            vec![1.0, 0.0, 0.0, 0.0],
        )],
        ..ProximaRecord::default()
    }
}

fn collection(id: &str, base_location: &str, tags: &[&str]) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: 4,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: base_location.to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

async fn flush_once(engine: &SstEngine, coll: &Collection, ids: &[&str]) -> Result<FlushResult> {
    let params = FlushParameters {
        collection_id: Some(coll.id.clone()),
        vector_records: ids.iter().map(|id| record(id)).collect(),
        force: true,
        synchronous: true,
        collection_config: Some(coll.clone()),
        ..Default::default()
    };
    engine.do_flush(&params).await
}

/// TD-WLP-2/TD-WLP-4 gate: with the global env unset, an `AppendBulk`
/// collection (tagged or untagged — AppendBulk is the default profile) arms
/// compaction at its L0 threshold **by default**; a `Churn`-tagged collection
/// never arms (so it never re-clusters/trains — ADR-061 D4/D5).
#[tokio::test]
async fn test_append_collection_arms_compaction_while_global_off() -> Result<()> {
    unsafe {
        std::env::remove_var(GLOBAL_GATE_ENV);
        std::env::remove_var("PROXIMADB_STORAGE_PROFILE");
    }
    let engine = SstEngine::new().await?;

    // AppendBulk collection (explicit tag + tight threshold): arms once L0
    // reaches its own threshold, with NO compaction:on opt-in needed.
    let append_dir = tempfile::tempdir()?;
    let append = collection(
        "wlp2_append_default",
        append_dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:append", "l0_threshold:2"],
    );
    let first = flush_once(&engine, &append, &["a0", "a1"]).await?;
    assert!(first.success, "first flush must succeed");
    assert!(
        !first.compaction_triggered,
        "1 L0 segment < l0_threshold:2 — compaction must not fire yet"
    );
    let second = flush_once(&engine, &append, &["a2", "a3"]).await?;
    assert!(second.success, "second flush must succeed");
    assert!(
        second.compaction_triggered,
        "AppendBulk collection at L0 >= threshold must arm compaction BY \
         DEFAULT (TD-WLP-4 arm-defaults)"
    );

    // Untagged collection = AppendBulk profile: arms at the default legacy
    // threshold of 5 once enough L0 segments accumulate.
    let untagged_dir = tempfile::tempdir()?;
    let untagged = collection(
        "wlp2_untagged",
        untagged_dir.path().to_str().expect("utf8 tempdir"),
        &[],
    );
    let mut untagged_triggered = false;
    for i in 0..6 {
        let ids = [format!("u{i}")];
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let result = flush_once(&engine, &untagged, &id_refs).await?;
        assert!(result.success, "untagged flush {i} must succeed");
        if i < 3 {
            assert!(
                !result.compaction_triggered,
                "below the default threshold of 5 — must not fire (flush {i})"
            );
        }
        untagged_triggered |= result.compaction_triggered;
    }
    assert!(
        untagged_triggered,
        "untagged (AppendBulk-default) collection must arm compaction at the \
         default L0 threshold (TD-WLP-4 arm-defaults)"
    );

    // Churn collection: never arms by default, even past every threshold.
    let churn_dir = tempfile::tempdir()?;
    let churn = collection(
        "wlp2_churn",
        churn_dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:churn", "l0_threshold:1"],
    );
    for i in 0..3 {
        let ids = [format!("c{i}")];
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let result = flush_once(&engine, &churn, &id_refs).await?;
        assert!(result.success, "churn flush {i} must succeed");
        assert!(
            !result.compaction_triggered,
            "Churn profile must never arm compaction by default (flush {i})"
        );
    }

    // compaction:off opts an AppendBulk collection back out.
    let optout_dir = tempfile::tempdir()?;
    let optout = collection(
        "wlp2_optout",
        optout_dir.path().to_str().expect("utf8 tempdir"),
        &["compaction:off", "l0_threshold:1"],
    );
    for i in 0..2 {
        let ids = [format!("o{i}")];
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let result = flush_once(&engine, &optout, &id_refs).await?;
        assert!(
            !result.compaction_triggered,
            "compaction:off must opt out of the armed default (flush {i})"
        );
    }
    Ok(())
}

/// The global env explicitly falsy is the master kill-switch: a per-collection
/// `compaction:on` opt-in cannot arm compaction under it (ADR-061 D5).
#[tokio::test]
async fn test_global_hard_disable_wins_over_collection_opt_in() -> Result<()> {
    unsafe {
        std::env::set_var(GLOBAL_GATE_ENV, "0");
    }
    let engine = SstEngine::new().await?;
    let dir = tempfile::tempdir()?;
    let opted = collection(
        "wlp2_hard_disable",
        dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:append", "compaction:on", "l0_threshold:1"],
    );
    for i in 0..2 {
        let ids = [format!("h{i}")];
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let result = flush_once(&engine, &opted, &id_refs).await?;
        assert!(result.success, "flush {i} must succeed");
        assert!(
            !result.compaction_triggered,
            "global hard-disable must win over the per-collection opt-in \
             (flush {i})"
        );
    }
    unsafe {
        std::env::remove_var(GLOBAL_GATE_ENV);
    }
    Ok(())
}
