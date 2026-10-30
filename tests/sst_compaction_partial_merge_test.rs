//! TD-COMPACT-13 T1c: a partial merge must FAIL CLOSED — an unreadable input
//! aborts the compaction BEFORE publication, leaving every input intact (a
//! publish-then-retire-all after a partial read would delete the only copy of
//! the unread rows). Own binary: the read-fault env arms cleanly at process
//! start (no gate sharing with the delete-fault arm).

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

/// Sum of live rows via the footer (tail 16 B = footer_len + PAXSEG01; footer
/// prefix 9 B = version u8 + rows u64 LE).
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

/// Fail-closed: a MISSING input file (external deletion/corruption) must
/// abort the compaction BEFORE publication — the surviving input remains the
/// complete live set and stays searchable (publish-then-retire-all after a
/// partial read would delete the only copy of the unread rows).
#[tokio::test]
async fn missing_input_fails_compaction_without_publishing_or_retiring() -> Result<()> {
    unsafe {
        std::env::remove_var("PROXIMADB_L0_COMPACTION_ENABLED");
        std::env::remove_var("PROXIMADB_STORAGE_PROFILE");
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_IVF_K", "4");
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let engine = SstEngine::new().await?;
    let dir = tempfile::tempdir()?;
    let coll = collection(
        "retire_readfault",
        dir.path().to_str().expect("utf8 tempdir"),
    );

    // Flush batch 1 -> one L0. Make it UNREADABLE (permission bits deny the
    // owner too on macOS/Linux) so flush batch 2's inline compaction hits a
    // read failure on that input — the fail-closed path.
    flush_batch(&engine, &coll, 0..BATCH).await?;
    let segments = pax_segments(dir.path());
    assert_eq!(segments.len(), 1, "one L0 after the first flush");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&segments[0], std::fs::Permissions::from_mode(0o000))?;
    }

    // Flush batch 2 arms the inline compaction; its read of the unreadable
    // input must FAIL CLOSED — no partial merge published, no input retired.
    flush_batch(&engine, &coll, BATCH..2 * BATCH).await?;
    quiesce(&engine).await;

    let after = pax_segments(dir.path());
    assert_eq!(
        after.len(),
        2,
        "fail-closed: both inputs intact — no partial merge published: {after:?}"
    );

    // Restore permissions; both rounds stay searchable and the row accounting
    // holds (nothing lost to a partial merge).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&segments[0], std::fs::Permissions::from_mode(0o644))?;
    }
    assert_eq!(
        live_row_total(&after),
        (2 * BATCH) as u64,
        "both inputs hold their rows (nothing lost to a partial merge)"
    );
    for probe_row in [7usize, BATCH + 7] {
        let ids = search_ids(&engine, &coll, vector_for(probe_row)).await;
        assert!(
            ids.contains(&format!("v{:05}", probe_row)),
            "row {probe_row} searchable (got {ids:?})"
        );
    }

    Ok(())
}
