//! M1-3 (ADR-049) — kill-switch → RawF32-PAX → exact read end-to-end.
//!
//! Pre-M1-3 the global kill-switch `PROXIMADB_PAX_QUANT_DISABLE`
//! forced the retired legacy `.sst` streaming format. Post-M1-3 flush ALWAYS
//! writes PAX, so the kill-switch instead selects the recall-exact `RawF32`
//! quant: the flushed segment is still `.pax`, but it carries raw f32 vectors
//! (no RaBitQ), and the search dispatch's exact PAX scan
//! (`SstEngine::search_pax_file_exact`) reads them back losslessly.
//!
//! This test proves that escape holds end-to-end: under the kill-switch a
//! collection flushes a RawF32-PAX `.pax` segment AND a subsequent exact search
//! returns the TRUE ranking (recall@k = 1.0 vs a brute-force oracle). Uses
//! well-separated vectors so the exact ranking is unambiguous.

use proximadb::compute::distance_computation::DistanceMetric;
use proximadb::core::search::{BlockPruneConfig, BlockPruneMode, SearchMode, SearchParams};
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
const KILL_SWITCH: &str = "PROXIMADB_PAX_QUANT_DISABLE";

/// Query = [1, 0, 0, 0]. Vectors at DISTINCT Euclidean distances from the query:
///   v0 = [1, 0, 0, 0]  (L2 = 0)
///   v1 = [0, 1, 0, 0]  (L2 = 1)
///   v2 = [0, 1, 1, 0]  (L2 = √2 ≈ 1.41)
///   v3 = [0, 1, 1, 1]  (L2 = √3 ≈ 1.73)
/// → exact Euclidean ranking is v0, v1, v2, v3 (no ties among the top-K).
const VECTORS: [[f32; DIM]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
];
const QUERY: [f32; DIM] = [1.0, 0.0, 0.0, 0.0];

fn collection(id: &str, temp_dir: &TempDir) -> Collection {
    collection_with_dim(id, temp_dir, DIM)
}

fn collection_with_dim(id: &str, temp_dir: &TempDir, dimension: usize) -> Collection {
    Collection {
        id: id.to_string(),
        config: Some(CollectionConfig {
            name: id.to_string(),
            dimension: dimension as u32,
            distance_metric: Some(DistanceMetric::Euclidean as i32),
            storage_engine: Some(StorageEngine::Sst as i32),
            // NOTE: no `pax_vector_format` tag — the GLOBAL kill-switch alone
            // selects RawF32 here.
            ..Default::default()
        }),
        storage_assignment: Some(StorageAssignment {
            base_location: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn vector_records() -> Vec<VectorRecord> {
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

async fn flush_records(engine: &SstEngine, collection: &Collection, records: Vec<VectorRecord>) {
    let flush_params = FlushParameters {
        collection_id: Some(collection.id.clone()),
        vector_records: records.into_iter().map(Into::into).collect(),
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

fn exact_context(collection: &Collection, query: Vec<f32>, top_k: usize) -> StorageQueryContext {
    StorageQueryContext {
        search_params: Arc::new(SearchParams {
            query_vectors: Some(vec![query]),
            top_k: Some(top_k as u16),
            distance_metric: Some(DistanceMetric::Euclidean),
            search_mode: SearchMode::Exact,
            block_prune: BlockPruneConfig {
                radius_k: 0.0,
                force_exact: true,
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
    }
}

/// Recursively collect durable segment paths under `dir` (the SST engine writes
/// them below a `collection_data_path` subdirectory, not the base directly).
fn collect_segment_paths(dir: &std::path::Path, paths: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_segment_paths(&path, paths);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("pax" | "sst")
        ) {
            paths.push(path);
        }
    }
}

fn has_extension(path: &std::path::Path, extension: &str) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some(extension)
}

/// Brute-force exact Euclidean ranking — the recall oracle.
fn brute_force_top_k() -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = VECTORS
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let l2: f32 = v
                .iter()
                .zip(&QUERY)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (i, l2)
        })
        .collect();
    // Ascending by distance (nearest first).
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(TOP_K)
        .map(|(i, _)| format!("v{i}"))
        .collect()
}

/// Under the kill-switch, a collection flushes a RawF32-PAX `.pax` segment (not
/// legacy `.sst`) AND an exact search returns the true ranking (recall@k = 1.0).
#[tokio::test]
async fn kill_switch_flushes_rawf32_pax_and_reads_back_exact() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();

    // Arm the kill-switch for this test (nextest isolates each test process).
    // `set_var`/`remove_var` are `unsafe` under edition 2024.
    unsafe {
        std::env::set_var(KILL_SWITCH, "1");
    }

    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path().to_str().unwrap().to_string();
    let collection = collection("kill_switch_rawf32", &temp_dir);
    let engine = SstEngine::new().await.unwrap();

    // Flush under the kill-switch → must write a `.pax` (RawF32-PAX) segment.
    flush_records(&engine, &collection, vector_records()).await;

    // The flushed segment must be `.pax` (RawF32-PAX), NOT legacy `.sst`.
    let mut segments = Vec::new();
    collect_segment_paths(std::path::Path::new(&base), &mut segments);
    assert!(
        segments.iter().any(|path| has_extension(path, "pax")),
        "kill-switch must still write a .pax segment (RawF32-PAX)"
    );
    assert!(
        !segments.iter().any(|path| has_extension(path, "sst")),
        "kill-switch must NOT write legacy .sst (got a .sst under {base})"
    );

    // Exact search → must match the brute-force oracle (RawF32 = no quantization
    // loss; the search dispatch's exact PAX scan reads the raw vectors back).
    let ctx = exact_context(&collection, QUERY.to_vec(), TOP_K);
    let got: Vec<String> = engine
        .search_vectors_unified(&ctx)
        .await
        .expect("search succeeds")
        .into_iter()
        .map(|r| r.id)
        .collect();

    // Always disarm the kill-switch before asserting (so a panic can't leak it).
    unsafe {
        std::env::remove_var(KILL_SWITCH);
    }

    let truth = brute_force_top_k();
    assert_eq!(
        got, truth,
        "RawF32-PAX exact scan must return the brute-force ranking (got {got:?}, want {truth:?}) — \
         if this fails, the kill-switch is no longer producing a recall-exact segment",
    );
}

/// Without the kill-switch, the same collection writes a RaBitQ-PAX `.pax` segment
/// (the default) — a guard that the kill-switch path is opt-in, not the default.
#[tokio::test]
async fn default_without_kill_switch_writes_pax_segment() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    unsafe {
        std::env::remove_var(KILL_SWITCH);
    }
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path().to_str().unwrap().to_string();
    let collection = collection("default_pax", &temp_dir);
    let engine = SstEngine::new().await.unwrap();
    flush_records(&engine, &collection, vector_records()).await;
    let mut segments = Vec::new();
    collect_segment_paths(std::path::Path::new(&base), &mut segments);
    assert!(
        segments.iter().any(|path| has_extension(path, "pax")),
        "default flush must write a .pax segment"
    );
    assert!(
        !segments.iter().any(|path| has_extension(path, "sst")),
        "default flush must not write legacy .sst"
    );
}

/// `SearchMode::Exact` is a 100%-recall contract, not a synonym for scanning
/// every lossy code. A default RaBitQ/SQ8 segment without raw-f32 authority must
/// reject the request instead of returning approximate rows labeled exact.
#[tokio::test]
async fn lossy_pax_rejects_exact_search() {
    let _ = proximadb::core::hardware_capabilities::initialize_hardware_capabilities_default();
    unsafe {
        std::env::remove_var(KILL_SWITCH);
        std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rabitq");
        std::env::set_var("PROXIMADB_PAX_F32_TIER", "0");
    }

    let temp_dir = TempDir::new().unwrap();
    const LOSSY_DIM: usize = 64;
    let collection = collection_with_dim("lossy_pax_exact_rejected", &temp_dir, LOSSY_DIM);
    let engine = SstEngine::new().await.unwrap();
    let records = (0..8)
        .map(|i| VectorRecord {
            id: format!("lossy-{i}"),
            vector: (0..LOSSY_DIM)
                .map(|d| ((i * LOSSY_DIM + d) as f32 * 0.017).sin())
                .collect(),
            metadata: HashMap::new(),
            version: Some(1),
            timestamp: Some(i as i64),
            updated_at: None,
            expires_at: None,
            source: None,
        })
        .collect();
    flush_records(&engine, &collection, records).await;

    let mut segments = Vec::new();
    collect_segment_paths(temp_dir.path(), &mut segments);
    let pax_paths: Vec<_> = segments
        .iter()
        .filter(|path| has_extension(path, "pax"))
        .collect();
    assert_eq!(pax_paths.len(), 1, "fixture must flush one PAX segment");
    let segment = std::fs::read(&pax_paths[0]).unwrap();
    assert!(
        !proximadb::storage::engines::sst::segment_format::pax_segment_has_exact_vector_authority(
            &segment
        ),
        "fixture must prove the on-disk PAX segment is lossy"
    );

    let error = engine
        .search_vectors_unified(&exact_context(&collection, vec![0.0; LOSSY_DIM], TOP_K))
        .await
        .expect_err("lossy PAX cannot satisfy SearchMode::Exact");
    let message = error.to_string();
    assert!(
        message.contains("authoritative raw-f32") && message.contains("pax_f32_tier"),
        "exact-search rejection must explain how to satisfy the contract: {message}"
    );
}
