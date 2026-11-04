//! PAX Phase 2 (read-side wiring) — the production SST search dispatches a
//! `.pax` segment to the RaBitQ→SQ8 cascade.
//!
//! Before this wiring, `rabitq_search_segment` had **zero non-test callers** — a
//! PAX segment written with `PROXIMADB_PAX_VECTOR_SEGMENTS=1` was read in
//! production only via the generic block scan (no RaBitQ acceleration), so the
//! recall ratchet never reflected production behavior. This test exercises the
//! wired dispatch end-to-end: flush a PAX+RaBitQ segment, search it through the
//! production `search_vectors_unified` path under the Euclidean metric, and
//! assert the cascade returns an accurate top-k. Recall@10 ≥ 0.90 vs the exact
//! nearest neighbours proves the cascade served (the generic scan alone cannot
//! rank a RaBitQ-coded segment without the cascade's rerank).
//!
//! Env-scoped (nextest runs each test in its own process, so the PAX env vars set
//! here don't leak to other tests): `PROXIMADB_PAX_VECTOR_SEGMENTS=1` writes
//! `.pax` segments on flush; `PROXIMADB_PAX_VECTOR_QUANT=rabitq` codes them with
//! RaBitQ (no f32 tier) so the cascade is the only path that can rank them.

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchParams};
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
};
use proximadb::storage::engines::sst::SstEngine;
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

const DIMENSION: usize = 32;
const N: usize = 512;
const TOP_K: usize = 10;

fn collection(id: &str, temp_dir: &TempDir) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: DIMENSION as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Deterministic pseudo-random vectors (LCG) — the regime the RaBitQ cascade is
/// validated on (recall 0.932 @ N=100k). Clustered data is a known hard case for
/// quantization and is covered by the separate SIFT real-dataset validation gap,
/// not this wiring test.
fn vectors() -> Vec<VectorRecord> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_f32 = || {
        // xorshift64 → f32 in [0, 1)
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..N)
        .map(|i| VectorRecord {
            id: format!("v{i:04}"),
            vector: (0..DIMENSION).map(|_| next_f32()).collect(),
            metadata: HashMap::new(),
            version: Some(1),
            timestamp: Some(i as i64),
            updated_at: None,
            expires_at: None,
            source: None,
        })
        .collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

async fn search(
    engine: &SstEngine,
    collection: &Collection,
    query: Vec<f32>,
    force_exact: bool,
) -> Vec<String> {
    let ctx = StorageQueryContext {
        search_params: Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            block_prune: BlockPruneConfig {
                radius_k: 0.0,
                force_exact,
                mode: BlockPruneMode::Ratio,
                ratio: 1.0,
                min_keep: 1,
                max_keep: 0,
                min_blocks_override: Some(0),
            },
            ..Default::default()
        }),
        collection: Arc::new(collection.clone()),
        metadata: StorageQueryMetadata {
            collection_id: collection.id.clone(),
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

async fn flush(engine: &SstEngine, collection: &Collection, vectors: Vec<VectorRecord>) {
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: vectors.into_iter().map(|v| v.into()).collect(),
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: None,
        trigger_compaction: false,
        batch_ids: vec![],
        collection_config: Some(collection.clone()),
        estimated_size: 0,
        collection_properties: std::collections::HashMap::new(),
    };
    let result = engine.do_flush(&params).await.expect("flush succeeds");
    assert!(result.success, "flush should succeed");
    assert!(
        result.entries_flushed.unwrap_or(0) > 0,
        "flush should write vectors"
    );
}

/// A PAX+RaBitQ segment searched through the production path returns an accurate
/// top-k (recall@10 ≥ 0.90) — proving the RaBitQ→SQ8 cascade is wired into the
/// dispatch (the only path that can rank a RaBitQ-coded segment).
#[tokio::test]
async fn pax_segment_cascade_serves_accurate_topk() {
    // PAX write-default stays OFF in prod; opt this collection in via env so a
    // `.pax` segment is flushed. nextest isolates each test in its own process.
    // `set_var` is `unsafe` (edition 2024) — confined to this test process.
    unsafe {
        std::env::set_var("PROXIMADB_PAX_VECTOR_SEGMENTS", "1");
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
    }

    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    let temp_dir = TempDir::new().unwrap();
    let collection = collection("pax_cascade_e2e", &temp_dir);

    let engine = SstEngine::new().await.unwrap();
    let vectors = vectors();
    let query = vectors[0].vector.clone(); // a known vector; its true NN are well-defined
    flush(&engine, &collection, vectors.clone()).await;

    // A `.pax` segment was written (otherwise the cascade branch can't apply).
    let mut all_files: Vec<String> = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p.to_string_lossy().to_string());
            }
        }
    }
    walk(temp_dir.path(), &mut all_files);
    let wrote_pax = all_files.iter().any(|f| f.ends_with(".pax"));
    assert!(
        wrote_pax,
        "flush should write a .pax segment under PAX_VECTOR_SEGMENTS=1 (wrote: {all_files:?})"
    );

    // True nearest neighbours by brute-force L2.
    let mut exact: Vec<(String, f32)> = vectors
        .iter()
        .map(|v| (v.id.clone(), l2(&query, &v.vector)))
        .collect();
    exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let exact_ids: std::collections::HashSet<String> =
        exact.iter().take(TOP_K).map(|(id, _)| id.clone()).collect();

    // Production search over the PAX segment routes through the cascade (`.pax` +
    // Euclidean) and returns a full top-k.
    let got = search(&engine, &collection, query, false).await;
    assert_eq!(
        got.len(),
        TOP_K,
        "cascade should return a full top-k (got {})",
        got.len()
    );
    let got_ids: std::collections::HashSet<String> = got.into_iter().collect();
    let overlap = got_ids.intersection(&exact_ids).count();
    let recall = overlap as f64 / TOP_K as f64;
    assert!(
        recall >= 0.90,
        "PAX cascade recall@{TOP_K} = {recall:.2} (overlap {overlap}/{TOP_K}) < 0.90 ratchet"
    );
}
