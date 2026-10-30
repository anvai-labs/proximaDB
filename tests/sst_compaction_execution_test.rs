//! TD-WLP-7 (ADR-061 D3) integration gate: the armed re-cluster compaction
//! actually EXECUTES on the live flush path.
//!
//! Before TD-WLP-7 the flush only set `compaction_triggered` and nothing
//! scheduled the compactor (a historical stub), so TD-WLP-4's armed-by-default
//! re-cluster never ran. This gate proves that an `AppendBulk` collection whose
//! L0 crosses its threshold runs compaction inline — without any manual
//! compaction call — surfaced via the `compaction_ran` engine metric; and that
//! a `Churn` collection never compacts (never re-clusters/trains).

use anyhow::Result;
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::{FlushParameters, FlushResult, UnifiedStorageEngine};
use proximadb_proto::v1::{Collection, CollectionConfig, StorageAssignment};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

const GLOBAL_GATE_ENV: &str = "PROXIMADB_L0_COMPACTION_ENABLED";

fn record(id: &str, seed: f32) -> ProximaRecord {
    ProximaRecord {
        oid: id.to_string(),
        created_at_ns: 12_345_000_000,
        updated_at_ns: 12_345_000_000,
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32(
            "test",
            "dense_vector",
            4,
            vec![seed, 1.0 - seed, seed * 0.5, 0.25],
        )],
        ..ProximaRecord::default()
    }
}

/// Mint a catalog-style decimal collection object id (ADR-075: `CollectionObjectId = u64`).
/// The SST flush / compaction-admission path is fail-closed on a decimal id
/// (`FlushParams::get_collection_object_id`), so a bare-engine test must supply
/// one — the human name lives in `CollectionConfig.name`. This mirrors the
/// production catalog's monotonic mint without standing up the full catalog.
fn next_object_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed).to_string()
}

fn collection(name: &str, base_location: &str, tags: &[&str]) -> Collection {
    Collection {
        id: next_object_id(),
        config: Some(CollectionConfig {
            name: name.to_string(),
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

async fn flush_batch(
    engine: &SstEngine,
    coll: &Collection,
    ids: &[(&str, f32)],
) -> Result<FlushResult> {
    let params = FlushParameters {
        collection_id: Some(coll.id.clone()),
        vector_records: ids.iter().map(|(id, s)| record(id, *s)).collect(),
        force: true,
        synchronous: true,
        collection_config: Some(coll.clone()),
        ..Default::default()
    };
    engine.do_flush(&params).await
}

fn compaction_ran(result: &FlushResult) -> bool {
    result
        .engine_metrics
        .get("compaction_ran")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// TD-WLP-7 gate: an `AppendBulk` collection at L0 ≥ threshold runs compaction
/// inline on the flush path (no manual call), and a `Churn` collection never
/// does.
#[tokio::test]
async fn test_armed_flush_executes_compaction_inline() -> Result<()> {
    unsafe {
        std::env::remove_var(GLOBAL_GATE_ENV);
        std::env::remove_var("PROXIMADB_STORAGE_PROFILE");
    }
    let engine = SstEngine::new().await?;

    // AppendBulk collection, tiny threshold: the flush that crosses L0 ≥ 2 must
    // execute compaction inline.
    let dir = tempfile::tempdir()?;
    let append = collection(
        "wlp7_append",
        dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:append", "l0_threshold:2"],
    );
    let first = flush_batch(&engine, &append, &[("a0", 0.1), ("a1", 0.2)]).await?;
    assert!(first.success, "first flush must succeed");
    assert!(
        !compaction_ran(&first),
        "1 L0 segment < threshold — compaction must not run yet"
    );

    let second = flush_batch(&engine, &append, &[("a2", 0.3), ("a3", 0.4)]).await?;
    assert!(second.success, "second flush must succeed");
    assert!(
        second.compaction_triggered,
        "at L0 >= threshold the flush must arm compaction"
    );
    assert!(
        second.compaction_error.is_none(),
        "compaction must not error: {:?}",
        second.compaction_error
    );
    assert!(
        compaction_ran(&second),
        "TD-WLP-7: the armed compaction must EXECUTE inline on the flush path \
         (compaction_ran metric), not just be flagged"
    );

    // Churn collection: never arms, so never runs compaction — even past the
    // threshold (ADR-061 D4/D5: churn never re-clusters/trains).
    let churn_dir = tempfile::tempdir()?;
    let churn = collection(
        "wlp7_churn",
        churn_dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:churn", "l0_threshold:1"],
    );
    for i in 0..3 {
        let id = format!("c{i}");
        let result = flush_batch(&engine, &churn, &[(id.as_str(), 0.5)]).await?;
        assert!(result.success, "churn flush {i} must succeed");
        assert!(
            !compaction_ran(&result),
            "Churn must never execute compaction (flush {i})"
        );
    }
    Ok(())
}

/// The global hard-disable kill-switch prevents execution even for an opted-in
/// AppendBulk collection (ADR-061 D5 master kill-switch, TD-WLP-2).
#[tokio::test]
async fn test_global_hard_disable_prevents_execution() -> Result<()> {
    unsafe {
        std::env::set_var(GLOBAL_GATE_ENV, "0");
    }
    let engine = SstEngine::new().await?;
    let dir = tempfile::tempdir()?;
    let append = collection(
        "wlp7_hard_disable",
        dir.path().to_str().expect("utf8 tempdir"),
        &["workload_profile:append", "compaction:on", "l0_threshold:1"],
    );
    for i in 0..2 {
        let id = format!("h{i}");
        let result = flush_batch(&engine, &append, &[(id.as_str(), 0.5)]).await?;
        assert!(result.success, "flush {i} must succeed");
        assert!(
            !compaction_ran(&result),
            "global hard-disable must prevent compaction execution (flush {i})"
        );
    }
    unsafe {
        std::env::remove_var(GLOBAL_GATE_ENV);
    }
    Ok(())
}
