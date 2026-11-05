//! P4 — end-to-end distance-metric consistency: a collection's configured metric
//! threads through flush → search → distance computation. Proves the metric is
//! actually consulted (Cosine ≠ Euclidean rankings on the same vectors).
//!
//! Uses the same Euclidean `SstEngine` pattern as `pax_cascade_prod_search_test`
//! but varies the metric (Cosine vs Euclidean) and asserts each search's ranking
//! matches a brute-force oracle with the SAME metric.

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

const DIM: usize = 4;
const TOP_K: usize = 3;

/// Well-separated vectors (no colinear pairs → no exact cosine ties) where
/// Cosine and Euclidean rankings DIFFER. Query = [3, 4, 0, 0] (norm 5):
///   v0 = [3, 4, 3, 0]  (cos ≈ 0.858, L2 = 3)
///   v1 = [6, 8, 0, 0]  (cos = 1.0    , L2 = 5)   — far but aligned
///   v2 = [1, 0, 0, 0]  (cos = 0.6    , L2 ≈ 4.47) — near in L2, low cosine
///   v3 = [0, 0, 3, 4]  (cos = 0      , L2 ≈ 7.07)
///
/// Cosine ranking:    v1 (1.0), v0 (0.858), v2 (0.6)   → [v1, v0, v2]
/// Euclidean ranking: v0 (3)  , v2 (4.47) , v1 (5)     → [v0, v2, v1]
/// → every position differs — proves the metric is consulted. All four cosines
/// and all four L2 distances are distinct, so the exact ranking is unambiguous
/// under either metric (no near-tie misrank — see the TD-163 lesson).
const VECTORS: [[f32; DIM]; 4] = [
    [3.0, 4.0, 3.0, 0.0],
    [6.0, 8.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 4.0],
];
const QUERY: [f32; DIM] = [3.0, 4.0, 0.0, 0.0];

fn collection(id: &str, metric: DistanceMetric, temp_dir: &TempDir) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: DIM as u32,
            distance_metric: Some(metric as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            // M1-3 (ADR-049): this test asserts EXACT metric ranking against a
            // brute-force oracle over 4 vectors. RaBitQ is degenerate at that scale
            // (it needs many vectors), so opt to the recall-exact RawF32-PAX quant
            // via the `pax_vector_format:off` tag. The flushed `.pax` segment carries
            // raw f32 vectors, and the search dispatch's exact PAX scan
            // (`search_pax_file_exact`) reads them back losslessly — so the ranking
            // matches the oracle exactly. (Pre-M1-3 this tag forced legacy
            // ProximaBlocks `.sst`; the streaming write path is now retired.)
            tags: vec!["pax_vector_format:off".to_string()],
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn vectors() -> Vec<VectorRecord> {
    VECTORS
        .iter()
        .enumerate()
        .map(|(i, v)| VectorRecord {
            id: format!("v{i}"),
            vector: v.to_vec(),
            metadata: HashMap::new(),
            version: Some(1),
            timestamp: Some(i as i64),
            updated_at: None,
            expires_at: None,
            source: None,
        })
        .collect()
}

async fn flush(engine: &SstEngine, collection: &Collection) {
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: vectors().into_iter().map(Into::into).collect(),
        force: true,
        synchronous: true,
        hints: HashMap::new(),
        timeout_ms: None,
        trigger_compaction: false,
        batch_ids: vec![],
        collection_config: Some(collection.clone()),
        estimated_size: 0,
        collection_properties: None,
    };
    let result = engine.do_flush(&params).await.expect("flush succeeds");
    assert!(result.success);
}

async fn search(
    engine: &SstEngine,
    collection: &Collection,
    metric: DistanceMetric,
) -> Vec<String> {
    let ctx = StorageQueryContext {
        search_params: Arc::new(SearchParams {
            query_vectors: Some(vec![QUERY.to_vec()]),
            top_k: Some(TOP_K as u16),
            distance_metric: Some(metric),
            block_prune: BlockPruneConfig {
                radius_k: 0.0,
                force_exact: true, // exact scan — exercise the distance compute path
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

/// Brute-force ranking with a given metric.
fn brute_force_ranking(metric: DistanceMetric) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = VECTORS
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let score = match metric {
                DistanceMetric::Euclidean => {
                    -(v.iter()
                        .zip(&QUERY)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>())
                }
                DistanceMetric::Cosine => {
                    let dot: f32 = v.iter().zip(&QUERY).map(|(a, b)| a * b).sum();
                    let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nq: f32 = QUERY.iter().map(|x| x * x).sum::<f32>().sqrt();
                    dot / (nv * nq)
                }
                _ => unreachable!("test uses only Euclidean + Cosine"),
            };
            (i, score)
        })
        .collect();
    // Sort descending (higher score = better for both: negated -L2 for Euclidean,
    // cosine sim for Cosine).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(TOP_K)
        .map(|(i, _)| format!("v{i}"))
        .collect()
}

/// A collection's configured metric threads through search → distance computation.
/// Cosine and Euclidean produce DIFFERENT rankings on the same vectors (proving
/// the metric is actually consulted, not hardcoded).
#[tokio::test]
async fn distance_metric_threads_through_search_and_ranks_differently() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // --- Cosine collection ---
    let temp_cos = TempDir::new().unwrap();
    let coll_cos = collection("metric_cosine", DistanceMetric::Cosine, &temp_cos);
    let engine = SstEngine::new().await.unwrap();
    flush(&engine, &coll_cos).await;
    let got_cos = search(&engine, &coll_cos, DistanceMetric::Cosine).await;

    // --- Euclidean collection ---
    let temp_l2 = TempDir::new().unwrap();
    let coll_l2 = collection("metric_l2", DistanceMetric::Euclidean, &temp_l2);
    flush(&engine, &coll_l2).await;
    let got_l2 = search(&engine, &coll_l2, DistanceMetric::Euclidean).await;

    // Each search ranking must match the brute-force oracle with the SAME metric.
    let truth_cos = brute_force_ranking(DistanceMetric::Cosine);
    let truth_l2 = brute_force_ranking(DistanceMetric::Euclidean);

    assert_eq!(
        got_cos, truth_cos,
        "Cosine collection must produce Cosine-ranked results (got {got_cos:?}, want {truth_cos:?})"
    );
    assert_eq!(
        got_l2, truth_l2,
        "Euclidean collection must produce L2-ranked results (got {got_l2:?}, want {truth_l2:?})"
    );

    // The two rankings MUST differ — proves the metric is consulted, not hardcoded.
    assert_ne!(
        got_cos, got_l2,
        "Cosine and Euclidean must produce different rankings on the same vectors \
         (if equal, the metric is being ignored)"
    );
}
