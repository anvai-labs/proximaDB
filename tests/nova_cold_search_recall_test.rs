//! TD-040 NOVA — recall-preserving, I/O-saving vector-bounds row-group pruning.
//!
//! The NOVA cold L2 search now skips whole parquet row groups whose per-dimension
//! bounding box (persisted in the `{file}.nova_meta` sidecar at flush) cannot hold
//! a top-k candidate, reading only the survivors via object_store ranged GETs.
//! The prune is recall-preserving by construction; this test pins that invariant:
//! bounds-pruned search must return the EXACT same top-k (ids + scores) as a full
//! `force_exact` scan, AND must actually fire (≥1 row group skipped) on separable
//! multi-row-group data.
//!
//! `PROXIMADB_NOVA_MAX_ROW_GROUP_SIZE` forces small parquet row groups so a modest
//! vector count yields multiple row groups (production default is large). The
//! pruned-row-group count is observed via the `predicate_diagnostics` task-local
//! bus (the same channel that surfaces it in EXPLAIN).

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchParams};
use proximadb::observability::predicate_diagnostics;
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
};
use proximadb::storage::engines::nova::NovaEngine;
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

const DIMENSION: usize = 16;
const CLUSTERS: usize = 3;
const PER_CLUSTER: usize = 100;
const ROW_GROUP_SIZE: usize = 100; // == PER_CLUSTER ⇒ one cluster per row group
const TOP_K: usize = 10;

fn collection_config(id: &str, temp_dir: &TempDir, metric: DistanceMetric) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: DIMENSION as u32,
            distance_metric: Some(metric as i32),
            storage_engine: Some(StorageEngine::Nova as i32),
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Well-separated clusters: cluster `c` near coordinate `c * 1000` in every dim.
/// Zero-padded, cluster-major ids keep each row group cluster-local (so the
/// per-row-group bounding boxes are tight and far clusters get pruned).
fn clustered_vectors() -> Vec<VectorRecord> {
    let mut out = Vec::with_capacity(CLUSTERS * PER_CLUSTER);
    for c in 0..CLUSTERS {
        let center = c as f32 * 1000.0;
        for r in 0..PER_CLUSTER {
            let jitter = r as f32 * 0.001;
            let vector: Vec<f32> = (0..DIMENSION)
                .map(|d| center + jitter + d as f32 * 0.0001)
                .collect();
            let global = c * PER_CLUSTER + r;
            out.push(VectorRecord {
                id: format!("vec_{global:05}"),
                vector,
                metadata: HashMap::new(),
                version: Some(1),
                timestamp: Some(global as i64),
                updated_at: None,
                expires_at: None,
                source: None,
            });
        }
    }
    out
}

fn search_params(query: Vec<f32>, metric: DistanceMetric, force_exact: bool) -> Arc<SearchParams> {
    Arc::new(SearchParams {
        query_vectors: Some(vec![query]),
        top_k: Some(TOP_K as u16),
        distance_metric: Some(metric),
        block_prune: BlockPruneConfig {
            radius_k: 0.0,
            force_exact,
            mode: BlockPruneMode::Sqrt,
            ratio: 0.2,
            min_keep: 1,
            max_keep: 0,
            min_blocks_override: Some(0),
        },
        ..Default::default()
    })
}

async fn flush_collection(
    engine: &NovaEngine,
    collection: &Collection,
    vectors: Vec<VectorRecord>,
) {
    let flush_params = FlushParameters {
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
        collection_properties: None,
    };
    let result = engine
        .do_flush(&flush_params)
        .await
        .expect("flush succeeds");
    assert!(result.success, "flush should succeed");
}

/// Search and also return the number of row groups pruned (captured via the
/// per-request diagnostics bus inside the scope).
async fn search_collect(
    engine: &NovaEngine,
    collection: &Collection,
    query: Vec<f32>,
    metric: DistanceMetric,
    force_exact: bool,
) -> (Vec<(String, f32)>, u64) {
    let ctx = StorageQueryContext {
        search_params: search_params(query, metric, force_exact),
        collection: Arc::new(collection.clone()),
        metadata: StorageQueryMetadata {
            collection_id: collection.id.clone(),
            ..Default::default()
        },
        user_context: None,
        tenant_context: None,
    };
    predicate_diagnostics::scope(async {
        let results = engine
            .search_vectors_unified(&ctx)
            .await
            .expect("search succeeds");
        let pruned = predicate_diagnostics::take_vector_bounds_pruned();
        let topk: Vec<(String, f32)> = results.into_iter().map(|r| (r.id, r.score)).collect();
        (topk, pruned)
    })
    .await
}

/// All three scenarios run in ONE test (sequentially) because the kill-switch
/// and row-group-size env vars are process-global — separate parallel `#[test]`s
/// would race on them.
#[tokio::test]
async fn nova_vector_bounds_prune_recall_and_guards() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    unsafe {
        std::env::set_var(
            "PROXIMADB_NOVA_MAX_ROW_GROUP_SIZE",
            ROW_GROUP_SIZE.to_string(),
        );
        std::env::remove_var("PROXIMADB_VECTOR_BOUNDS_PRUNE_DISABLE");
    }

    // ── Scenario 1: Euclidean — recall preserved AND pruning fires ──────────
    {
        let temp_dir = TempDir::new().unwrap();
        let collection =
            collection_config("td040_nova_recall", &temp_dir, DistanceMetric::Euclidean);
        let engine = NovaEngine::new().await.unwrap();
        let vectors = clustered_vectors();
        // Query inside cluster 0; the true top-k are all cluster-0 vectors, and
        // the other clusters' row groups are provably farther than the k-th best.
        let query = vectors[0].vector.clone();
        flush_collection(&engine, &collection, vectors).await;

        let (pruned, pruned_count) = search_collect(
            &engine,
            &collection,
            query.clone(),
            DistanceMetric::Euclidean,
            false,
        )
        .await;
        let (exact, _) =
            search_collect(&engine, &collection, query, DistanceMetric::Euclidean, true).await;

        assert!(!exact.is_empty(), "exact search returned no results");
        assert_eq!(
            pruned.len(),
            exact.len(),
            "pruned top-k size must match exact"
        );
        for (i, ((p_id, p_score), (e_id, e_score))) in pruned.iter().zip(exact.iter()).enumerate() {
            assert_eq!(
                p_id, e_id,
                "rank {i}: id mismatch (pruned={p_id}, exact={e_id})"
            );
            assert!(
                (p_score - e_score).abs() < f32::EPSILON,
                "rank {i}: score mismatch (pruned={p_score}, exact={e_score})"
            );
        }
        assert!(
            pruned_count >= 1,
            "vector-bounds pruning must skip ≥1 row group on separable data (got {pruned_count})"
        );
    }

    // ── Scenario 2: Cosine — L2 bound invalid ⇒ no pruning ──────────────────
    {
        let temp_dir = TempDir::new().unwrap();
        let collection = collection_config("td040_nova_cosine", &temp_dir, DistanceMetric::Cosine);
        let engine = NovaEngine::new().await.unwrap();
        let vectors = clustered_vectors();
        let query = vectors[0].vector.clone();
        flush_collection(&engine, &collection, vectors).await;

        let (_, pruned_count) =
            search_collect(&engine, &collection, query, DistanceMetric::Cosine, false).await;
        assert_eq!(
            pruned_count, 0,
            "cosine search must not engage L2 bounds pruning"
        );
    }

    // ── Scenario 3: kill-switch ⇒ full-read path even for exact-L2 ──────────
    {
        unsafe {
            std::env::set_var("PROXIMADB_VECTOR_BOUNDS_PRUNE_DISABLE", "1");
        }
        let temp_dir = TempDir::new().unwrap();
        let collection = collection_config(
            "td040_nova_killswitch",
            &temp_dir,
            DistanceMetric::Euclidean,
        );
        let engine = NovaEngine::new().await.unwrap();
        let vectors = clustered_vectors();
        let query = vectors[0].vector.clone();
        flush_collection(&engine, &collection, vectors).await;

        let (results, pruned_count) = search_collect(
            &engine,
            &collection,
            query,
            DistanceMetric::Euclidean,
            false,
        )
        .await;
        unsafe {
            std::env::remove_var("PROXIMADB_VECTOR_BOUNDS_PRUNE_DISABLE");
        }
        assert!(!results.is_empty(), "search returned no results");
        assert_eq!(pruned_count, 0, "kill-switch must disable pruning");
    }
}
