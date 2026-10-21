//! TD-RDSTRAT-8 PR-A route proof: the PRODUCTION compaction trigger emits the
//! persisted-IVF-probe (v3) layout under `PROXIMADB_PAX_WRITE_A0_TRAIN=1` — and only then.
//!
//! The SIFT recall harness drives flush only, and v3 is written exclusively by
//! `write_pax_segment_compacted`, so without this gate the v3 emission path
//! would be reachable in production yet never exercised by any test that goes
//! through the real trigger. This test mirrors the TD-WLP-7 execution gate
//! (`sst_compaction_execution_test.rs`): flush an armed AppendBulk collection
//! past its L0 threshold so compaction runs INLINE on the flush path (no
//! manual compaction call), then proves the compacted product on disk carries
//! `layout_version=3` + a parsable Region A0, and that the production search
//! path still returns exact top-k over it (v3 reads as a single-level full
//! scan until the PR-B probe reader lands).
//!
//! Env-scoped (nextest process-per-test isolation): the flag-off control runs
//! as its own test/process and must produce NO v3 segment.

use anyhow::Result;
use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchParams};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::{
    FlushParameters, FlushResult, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use proximadb_proto::v1::{Collection, CollectionConfig, StorageAssignment, StorageEngine};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use proximadb_storage_common::coarse_directory::CoarseDirectory;
use proximadb_storage_common::segment_layout::{
    SEG_LAYOUT_VERSION, SegmentHeaderPrefix, is_coalesced_segment,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DIM: usize = 8;
/// Per-flush batch size — two batches merge to 128 rows at compaction, above
/// the 64-usable-row training floor so the IVF probe model actually trains.
const BATCH: usize = 64;
const TOP_K: usize = 5;

fn vector_for(row: usize) -> Vec<f32> {
    // Deterministic xorshift per row — distinct, well-spread vectors so the
    // top-1 self-match assertion is tie-free.
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
        oid: format!("v{row:04}"),
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

/// Mint a catalog-style decimal collection object id (ADR-075: `CollectionObjectId = u64`).
/// The SST flush / compaction-admission path is fail-closed on a decimal id
/// (`FlushParams::get_collection_object_id`), so a bare-engine test must supply
/// one — the human name lives in `CollectionConfig.name`. Mirrors the production
/// catalog's monotonic mint without standing up the full catalog.
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
            // Armed AppendBulk with a tiny threshold: the second flush crosses
            // L0 >= 2 and executes compaction inline (the production trigger).
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
) -> Result<FlushResult> {
    let params = FlushParameters {
        collection_id: Some(coll.id.clone()),
        vector_records: rows.map(record).collect(),
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

/// Every `.pax` segment under `dir`, recursively.
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

/// Layout versions of every coalesced `.pax` segment under `dir`.
fn coalesced_versions(dir: &Path) -> Vec<(PathBuf, u8)> {
    pax_segments(dir)
        .into_iter()
        .filter_map(|p| {
            let bytes = std::fs::read(&p).ok()?;
            if !is_coalesced_segment(&bytes) {
                return None;
            }
            let h = SegmentHeaderPrefix::parse(&bytes).ok()?;
            Some((p, h.layout_version))
        })
        .collect()
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

/// Shared flow: flush twice (crossing L0 >= 2 arms + runs compaction inline),
/// return the collection dir + engine + collection for assertions.
async fn flush_to_compaction(id: &str) -> Result<(tempfile::TempDir, SstEngine, Collection)> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    unsafe {
        std::env::remove_var("PROXIMADB_L0_COMPACTION_ENABLED");
        std::env::remove_var("PROXIMADB_STORAGE_PROFILE");
        // Flush writes `.pax` + RaBitQ so compaction re-emits the coalesced
        // layout (the substrate v3 extends).
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        // Small cell count for the 128-row fixture (the shared IOP-derived
        // override; the natural N·dim/IOP count is ~2 here — legal but coarse).
        std::env::set_var("PROXIMADB_IVF_K", "4");
    }
    let engine = SstEngine::new().await?;
    let dir = tempfile::tempdir()?;
    let coll = collection(id, dir.path().to_str().expect("utf8 tempdir"));

    let first = flush_batch(&engine, &coll, 0..BATCH).await?;
    assert!(
        first.success && !compaction_ran(&first),
        "no compaction at L0=1"
    );
    let second = flush_batch(&engine, &coll, BATCH..2 * BATCH).await?;
    assert!(second.success, "second flush must succeed");
    assert!(
        compaction_ran(&second),
        "the armed compaction must execute inline (compaction_error: {:?})",
        second.compaction_error
    );
    // TD-COMPACT-6 D1: compaction now runs on the background worker pool
    // (enqueued by the flush, not awaited inline). The compacted product (the
    // v3/v1 .pax segment the assertions below read) isn't written until the
    // worker drains the task — await quiescence before returning so the callers
    // observe the post-compaction layout.
    if let Some(cm) = engine.compaction_manager() {
        let quiet = cm
            .await_compaction_quiescence(std::time::Duration::from_secs(30))
            .await;
        assert!(quiet, "enqueued compaction did not complete within 30s");
    }
    Ok((dir, engine, coll))
}

/// `PROXIMADB_PAX_WRITE_A0_TRAIN=1`: the production compaction product is a v3 segment with
/// a parsable Region A0, and production search over it stays exact.
#[tokio::test]
async fn test_ivf2_compaction_route_emits_v3_and_searches() -> Result<()> {
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
    }
    let (dir, engine, coll) = flush_to_compaction("ivf2_route_on").await?;

    let versions = coalesced_versions(dir.path());
    assert!(
        !versions.is_empty(),
        "compaction must leave coalesced .pax segments under {:?}",
        dir.path()
    );
    let v3: Vec<_> = versions
        .iter()
        .filter(|(_, v)| *v == SEG_LAYOUT_VERSION)
        .collect();
    assert!(
        !v3.is_empty(),
        "PROXIMADB_PAX_WRITE_A0_TRAIN=1: the compacted product must carry layout_version=3 \
         (found versions: {versions:?})"
    );
    // The v3 segment's Region A0 parses and covers the merged corpus.
    let (v3_path, _) = v3[0];
    let bytes = std::fs::read(v3_path)?;
    let h = SegmentHeaderPrefix::parse(&bytes)?;
    let a0 = CoarseDirectory::parse(&bytes[h.a0_off as usize..(h.a0_off + h.a0_len) as usize])?;
    assert_eq!(
        a0.model.rows_covered(),
        (2 * BATCH) as u64,
        "A0 cells must cover every merged row"
    );

    // Production search over the v3 product: the query vector's own id is the
    // top hit (exact self-match; v3 reads as a single-level scan until PR-B).
    let ids = search_ids(&engine, &coll, vector_for(17)).await;
    assert_eq!(
        ids.first().map(String::as_str),
        Some("v0017"),
        "top-1 self-match over the compacted v3 segment (got {ids:?})"
    );
    Ok(())
}

/// Flag-off control (own process under nextest): the same production trigger
/// leaves every segment at layout v1 — v3 is strictly opt-in.
#[tokio::test]
async fn test_ivf2_off_compaction_route_stays_v1() -> Result<()> {
    // #1234: PROXIMADB_PAX_WRITE_A0_TRAIN is now default-ON (enable_write_train
    // in CoarseProbeConfig). Explicitly set "0" to test the OFF -> v1 path.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "0");
    }
    let (dir, engine, coll) = flush_to_compaction("ivf2_route_off").await?;

    let versions = coalesced_versions(dir.path());
    assert!(!versions.is_empty(), "compaction must leave coalesced .pax");
    // TD-PAXRG-1 collapse: ONE layout version for all coalesced segments; the
    // A0-presence signal is the header's `a0_len` extent (pre-collapse this
    // asserted the version byte != TWO_LEVEL). Without A0 training: version 1,
    // a0_len == 0.
    assert!(
        versions.iter().all(|(_, v)| *v == SEG_LAYOUT_VERSION),
        "every coalesced segment carries the single collapsed layout version"
    );
    for (p, _) in &versions {
        let bytes = std::fs::read(p)?;
        let h = SegmentHeaderPrefix::parse(&bytes)?;
        assert_eq!(
            h.a0_len, 0,
            "A0-off compaction must not emit a coarse directory: {p:?}"
        );
    }
    let ids = search_ids(&engine, &coll, vector_for(17)).await;
    assert_eq!(ids.first().map(String::as_str), Some("v0017"));
    Ok(())
}
