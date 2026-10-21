//! TD-COMPACT-13: the compaction live-set invariant — PERMANENT regression
//! safeguard.
//!
//! After the training-compaction chain quiesces, the storage location must
//! hold exactly ONE segment (no superseded level may survive), every ingested
//! id must be searchable, and no retirement-obligation sidecar may remain.
//! This is the clean happy-path arm; the fault-injected arm lives in
//! `sst_compaction_retirement_fault_test.rs` (own process: the fault env must
//! not leak into this test under plain cargo test).
//!
//! Drives the real production trigger: `l0_threshold:2` + flush rounds with
//! quiescence between rounds, so the worker follow-up chain walks
//! L0→L1→L2→L3 exactly as production does (no manual compaction call).

use anyhow::Result;
use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchParams};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use proximadb_proto::v1::{Collection, CollectionConfig, StorageAssignment, StorageEngine};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DIM: usize = 8;
const BATCH: usize = 64;
const ROUNDS: usize = 5;
const TOP_K: usize = 10;

fn vector_for(row: usize) -> Vec<f32> {
    let mut s = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        })
        .collect()
}

fn record(row: usize) -> ProximaRecord {
    ProximaRecord {
        oid: format!("v{row:05}"),
        created_at_ns: 1_000 + row as i64,
        updated_at_ns: 1_000 + row as i64,
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32(
            "test",
            "dense_vector",
            DIM as u32,
            vector_for(row),
        )],
        ..ProximaRecord::default()
    }
}

fn next_object_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed).to_string()
}

fn collection(name: &str, base_location: &str) -> Collection {
    Collection {
        id: next_object_id(),
        config: Some(CollectionConfig {
            name: name.to_string(),
            dimension: DIM as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            tags: vec!["workload_profile:append".into(), "l0_threshold:2".into()],
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
    rows: std::ops::Range<usize>,
) -> Result<()> {
    let params = FlushParameters {
        collection_id: Some(coll.id.clone()),
        vector_records: rows.map(record).collect(),
        force: true,
        synchronous: true,
        collection_config: Some(coll.clone()),
        ..Default::default()
    };
    engine.do_flush(&params).await?;
    Ok(())
}

async fn quiesce(engine: &SstEngine) {
    if let Some(cm) = engine.compaction_manager() {
        let quiet = cm
            .await_compaction_quiescence(std::time::Duration::from_secs(60))
            .await;
        assert!(quiet, "compaction did not quiesce within 60s");
    }
}

fn pax_segments(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "pax") {
                out.push(p);
            }
        }
    }
    out
}

async fn search_ids(engine: &SstEngine, coll: &Collection, query: Vec<f32>) -> Vec<String> {
    let ctx = StorageQueryContext {
        search_params: Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            block_prune: BlockPruneConfig {
                radius_k: 0.0,
                force_exact: false,
                mode: BlockPruneMode::Ratio,
                ratio: 1.0,
                min_keep: 1,
                max_keep: 0,
                min_blocks_override: Some(0),
            },
            ..Default::default()
        }),
        collection: Arc::new(coll.clone()),
        metadata: StorageQueryMetadata {
            collection_id: coll.id.clone(),
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };
    engine
        .search_vectors_unified(&ctx)
        .await
        .expect("search succeeds")
        .into_iter()
        .map(|r| r.id)
        .collect()
}

/// Drive the full training chain: ROUNDS rounds of two flushes each, with
/// quiescence after every round so the worker follow-up chain walks
/// L0→L1→L2→L3 through the real production trigger.
async fn run_chain(id: &str) -> Result<(tempfile::TempDir, SstEngine, Collection)> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    unsafe {
        std::env::remove_var("PROXIMADB_L0_COMPACTION_ENABLED");
        std::env::remove_var("PROXIMADB_STORAGE_PROFILE");
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_IVF_K", "4");
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
    }
    let engine = SstEngine::new().await?;
    let dir = tempfile::tempdir()?;
    let coll = collection(id, dir.path().to_str().expect("utf8 tempdir"));

    for round in 0..ROUNDS {
        let base = round * 2 * BATCH;
        flush_batch(&engine, &coll, base..base + BATCH).await?;
        flush_batch(&engine, &coll, base + BATCH..base + 2 * BATCH).await?;
        quiesce(&engine).await;
    }
    // The final level merges ride the worker follow-up chain — quiesce once
    // more after the last re-arm so the settled state is observed.
    quiesce(&engine).await;
    Ok((dir, engine, coll))
}

/// TD-COMPACT-13 live-set invariant (permanent): the quiesced chain leaves
/// exactly ONE segment — no superseded level survives in the storage
/// location — and every ingested id stays searchable.
#[tokio::test]
async fn compaction_chain_leaves_exactly_one_segment_and_all_rows_searchable() -> Result<()> {
    let (dir, engine, coll) = run_chain("retire_invariant").await?;

    let segments = pax_segments(dir.path());
    // THE INVARIANT is row accounting, not segment count: a two-level state is
    // legitimate when the lower level holds NEW rows (a late flush after the
    // compaction snapshot); it is the TD-COMPACT-13 defect when rows are
    // DUPLICATED across levels (sum > ingested — the measured 3.03x case:
    // 500k+500k+1M = 2M live rows for 1M ingested) and data loss when the sum
    // is short.
    let total_rows = live_row_total(&segments);
    assert_eq!(
        total_rows,
        ROUNDS as u64 * 2 * BATCH as u64,
        "TD-COMPACT-13: live-set rows must equal ingested rows exactly — \
         a surplus proves duplicated coverage across levels, a deficit proves loss \
         (segments: {segments:?})"
    );

    // Every round's ids stay searchable across the whole chain.
    for probe_row in [3usize, ROUNDS * BATCH / 2, ROUNDS * 2 * BATCH - 3] {
        let ids = search_ids(&engine, &coll, vector_for(probe_row)).await;
        assert!(
            ids.contains(&format!("v{:05}", probe_row)),
            "row {probe_row} missing from search after the chain (got {ids:?})"
        );
    }

    // No retirement obligation may linger after a clean local chain.
    let sidecars: Vec<PathBuf> = pax_sidecars(dir.path());
    assert!(
        sidecars.is_empty(),
        "retire-pending sidecar must not outlive a clean chain: {sidecars:?}"
    );
    Ok(())
}

/// Sum of live rows across every `.pax` segment, read from each footer
/// (tail 16 B = footer_len u64 LE + PAXSEG01 magic; the footer's first 9 B =
/// version u8 + row count u64 LE). Same authority the benchmark harness uses.
fn live_row_total(segments: &[PathBuf]) -> u64 {
    segments
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
            assert!(bytes.len() >= 25, "{} too short", p.display());
            let tail = &bytes[bytes.len() - 16..];
            assert_eq!(&tail[8..], b"PAXSEG01", "{} missing PAX tail", p.display());
            let footer_len = u64::from_le_bytes(tail[0..8].try_into().unwrap()) as usize;
            let footer_start = bytes.len() - 16 - footer_len;
            assert_eq!(bytes[footer_start], 1, "{} footer version", p.display());
            u64::from_le_bytes(
                bytes[footer_start + 1..footer_start + 9]
                    .try_into()
                    .unwrap(),
            )
        })
        .sum()
}

fn pax_sidecars(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with("retire-pending.json"))
            {
                out.push(p);
            }
        }
    }
    out
}
