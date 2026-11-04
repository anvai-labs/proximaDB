//! TD-112 — post-flush ANN recall over flushed/compacted segments.
//!
//! Acceptance #1: insert enough vectors to drive **≥2 flush cycles** plus a
//! compaction, then assert top-k recall stays within a target band against a
//! brute-force oracle. With the index-on-flush fix (`flush_implementation` ->
//! `AxisManager::handle_flushed_vectors`), post-flush search is served by the
//! AXIS ANN index — which is attached here — rather than a brute-force segment
//! scan, and the flushed vectors must be retrievable across every flush batch.
//!
//! The clusters are well-separated so the *expected* recall is high regardless
//! of the approximate index's internals; the band below is the regression gate.
//
// This test exercises the AXIS post-flush indexing path, so the whole file is
// elided when the `axis` feature is off (PAX-exact-scan build has no AXIS index).
#![cfg(feature = "axis")]

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchMode, SearchParams};
#[cfg(feature = "axis")]
use proximadb::index::axis::management::manager::AxisManager;
#[cfg(feature = "axis")]
use proximadb::index::axis::types::AxisConfig;
use proximadb::proto::proximadb_v1::{
    Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
};
use proximadb::storage::engines::sst::SstEngine;
#[cfg(feature = "axis")]
use proximadb::storage::engines::sst::core::set_sst_axis_manager;
use proximadb::storage::traits::{
    FlushParameters, StorageQueryContext, StorageQueryMetadata, UnifiedStorageEngine,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

#[cfg(feature = "axis")]
const DIMENSION: usize = 16;
#[cfg(feature = "axis")]
const CLUSTERS: usize = 6;
#[cfg(feature = "axis")]
const PER_CLUSTER: usize = 24;
#[cfg(feature = "axis")]
const TOP_K: usize = 10;
/// Conservative recall band: well-separated clusters should give near-perfect
/// recall; this gate catches a collapse toward zero (e.g. the index never being
/// populated and search returning nothing useful).
#[cfg(feature = "axis")]
const RECALL_BAND: f32 = 0.8;

#[cfg(feature = "axis")]
fn collection_config(id: &str, temp_dir: &TempDir) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: DIMENSION as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            // M1-1b: opt out of the PAX default. AXIS rebuild-from-SST reads
            // durable segments; a RaBitQ-PAX segment reconstructs COARSE vectors
            // (reader.rs: "RaBitQ is a search representation; reconstruction is
            // coarse"), so an index rebuilt from it has lower recall than the
            // exact vectors the flush path indexes. These tests assert exact
            // rebuild/recall, so they stay on legacy `.sst`. The RaBitQ-PAX
            // rebuild-recall concern is tracked as a follow-up.
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

/// Well-separated clusters with a **distinct sign signature per cluster** so the
/// quantized (binary/INT8) ANN stages can separate them: cluster `c` is centered
/// at `+3` in dimension `c` and `-1` in every other dimension, with a tiny
/// deterministic per-record perturbation. (All-positive, high-magnitude clusters
/// collapse under sign quantization — every vector shares one signature — so this
/// scheme is what gives a *fair* recall measurement of the index.)
#[cfg(feature = "axis")]
fn clustered_vectors() -> Vec<VectorRecord> {
    let mut out = Vec::with_capacity(CLUSTERS * PER_CLUSTER);
    for c in 0..CLUSTERS {
        for r in 0..PER_CLUSTER {
            let jitter = (r as f32) * 0.02;
            let vector: Vec<f32> = (0..DIMENSION)
                .map(|d| {
                    let base = if d == c { 3.0 } else { -1.0 };
                    // Small signed perturbation keeps the cluster tight without
                    // crossing zero (preserving the sign signature).
                    base + jitter * (if d % 2 == 0 { 0.1 } else { -0.1 })
                })
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

#[cfg(feature = "axis")]
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact top-k ids by Euclidean distance — the brute-force oracle.
#[cfg(feature = "axis")]
fn oracle_topk(all: &[VectorRecord], query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = all
        .iter()
        .map(|v| (euclidean_sq(&v.vector, query), v.id.as_str()))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

#[cfg(feature = "axis")]
async fn flush_batch(engine: &SstEngine, collection: &Collection, batch: Vec<VectorRecord>) {
    let params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: batch.into_iter().map(|v| v.into()).collect(),
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
}

#[cfg(feature = "axis")]
async fn search(engine: &SstEngine, collection: &Collection, query: Vec<f32>) -> Vec<String> {
    let params = Arc::new(SearchParams {
        query_vectors: Some(vec![query]),
        top_k: Some(TOP_K as u16),
        distance_metric: Some(DistanceMetric::Euclidean),
        // TD-184: use Approximate so the search takes the AXIS/orchestrated path,
        // which triggers the rebuild-from-SST when the in-memory index is absent.
        // The default Adaptive mode routes small collections to an Exact segment
        // scan that bypasses AXIS entirely — so the rebuild never runs.
        search_mode: SearchMode::Approximate { nprobe: None },
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
    });
    let ctx = StorageQueryContext {
        search_params: params,
        collection: Arc::new(collection.clone()),
        metadata: StorageQueryMetadata {
            collection_id: collection.id.clone(),
            // This test drives the SST engine directly, bypassing the service-layer
            // context builder where the per-collection gate infers use_axis_indexes
            // from index_configs / the pax_vector_format tag. These collections opt
            // out of PAX and run the legacy .sst + AXIS path, so opt into AXIS here:
            // with search_mode=Approximate this routes through the orchestrated path
            // that rebuilds AXIS from SST on a miss (the behavior under test).
            use_axis_indexes: true,
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

/// Post-flush (≥2 flush cycles + compaction) top-k recall must stay within the
/// target band against a brute-force oracle, and the query's own vector must be
/// retrievable — proving flushed vectors are served by the populated ANN index.
#[cfg(feature = "axis")]
#[tokio::test]
async fn postflush_recall_within_band_after_multi_flush_and_compaction() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Attach an AXIS manager so the orchestrated (ANN) search path is engaged;
    // each tests/ file is its own process, so this global is isolated here.
    set_sst_axis_manager(Arc::new(
        AxisManager::new(AxisConfig::default()).await.unwrap(),
    ));

    let temp_dir = TempDir::new().unwrap();
    let collection = collection_config("td112_recall", &temp_dir);
    let engine = SstEngine::new().await.unwrap();

    let all = clustered_vectors();

    // ≥2 flush cycles: split the data into two halves, flush each separately so
    // the index-on-flush hook fires twice over distinct flush batches.
    let mid = all.len() / 2;
    flush_batch(&engine, &collection, all[..mid].to_vec()).await;
    flush_batch(&engine, &collection, all[mid..].to_vec()).await;

    // Compaction cycle.
    engine
        .compact_collection(&collection.id, None)
        .await
        .expect("compaction succeeds");

    // Query an actual inserted vector from each cluster and measure recall.
    let mut total_recall = 0.0f32;
    let mut probes = 0;
    for c in 0..CLUSTERS {
        let probe = &all[c * PER_CLUSTER + PER_CLUSTER / 2];
        let query = probe.vector.clone();

        let got = search(&engine, &collection, query.clone()).await;
        assert!(
            !got.is_empty(),
            "post-flush search returned no hits for cluster {c}"
        );
        assert!(
            got.contains(&probe.id),
            "the query's own vector {} must be retrievable post-flush (cluster {c})",
            probe.id
        );
        let oracle = oracle_topk(&all, &query, TOP_K);
        let hits = got.iter().filter(|id| oracle.contains(id)).count();
        total_recall += hits as f32 / oracle.len() as f32;
        probes += 1;
    }

    let recall = total_recall / probes as f32;
    assert!(
        recall >= RECALL_BAND,
        "post-flush top-k recall {recall:.3} below band {RECALL_BAND} \
         (≥2 flush cycles + compaction)"
    );
}

/// TD-112 local post-loss recovery: when the in-memory AXIS index is lost —
/// simulated here via `drop_collection`, matching the empty-index side of a
/// restart without asserting boot-time prewarm — the next search must rebuild it
/// from durable local SST segments and recover recall, rather than silently
/// degrading to a brute-force scan.
#[cfg(feature = "axis")]
#[tokio::test]
async fn axis_index_rebuilt_from_sst_after_loss() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    set_sst_axis_manager(Arc::new(
        AxisManager::new(AxisConfig::default()).await.unwrap(),
    ));

    let temp_dir = TempDir::new().unwrap();
    let collection = collection_config("td112_restart", &temp_dir);
    let engine = SstEngine::new().await.unwrap();
    let axis = proximadb::storage::engines::sst::core::get_sst_axis_manager()
        .expect("axis manager attached");

    let all = clustered_vectors();
    flush_batch(&engine, &collection, all.clone()).await;

    let probe = &all[PER_CLUSTER / 2];
    let warm = search(&engine, &collection, probe.vector.clone()).await;
    assert!(warm.contains(&probe.id), "warm search must find the probe");

    // Simulate a restart: drop the in-memory AXIS index for the collection. The
    // durable SST segments remain on disk.
    axis.drop_collection(&collection.id).await.unwrap();
    assert_eq!(
        axis.registered_vector_count(&collection.id).await,
        0,
        "drop_collection must clear the in-memory AXIS store"
    );

    // The next search must rebuild the AXIS store from the SST segments and
    // recover. The store being repopulated (vs. an empty store served only by a
    // brute-force segment scan) is what proves the rebuild ran.
    let recovered = search(&engine, &collection, probe.vector.clone()).await;
    assert!(
        recovered.contains(&probe.id),
        "post-loss search must find the probe"
    );
    assert_eq!(
        axis.registered_vector_count(&collection.id).await,
        all.len(),
        "the AXIS store must be rebuilt from SST after the recovering search"
    );
}

// Risk-coverage scenario (docs/10-quality/RISK_CONTRACT.toml :: cold-read-recall):
//   scenario: postflush-recall-band — post-flush recall stays in band + AXIS rebuilds from SST (TD-184)
