// Copyright 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Phase 2F-b end-to-end test for NOVA's `ingest_sorted_segment`
//! override.
//!
//! Validates that calling the engine's LSM-bypass entry point with a
//! pre-sorted batch:
//! - Returns `used_engine_specific_path: true` (engine took the
//!   optimized path, not the trait default fallback).
//! - Reports the actual record count.
//! - Writes a real segment file under the requested `base_path`
//!   (proves the override isn't a no-op).
//!
//! This is the customer-relevant assurance: when the queue drainer
//! flushes a batch into a NOVA-backed collection, records land via
//! a single Parquet write instead of N WAL fsyncs + N memtable
//! inserts. Skipping the WAL is the 30% storage-side win the queue
//! README commits to.
//!
//! The test does NOT spin up a full ProximaDB server (too heavy for
//! a focused override test); it exercises NOVA directly through the
//! `UnifiedStorageEngine` trait. End-to-end tests covering the full
//! REST-async → queue → drainer → bulk-load → search chain live in
//! a separate integration suite (Phase 2G).

use proximadb::storage::engines::nova::NovaEngine;
use proximadb::storage::traits::UnifiedStorageEngine;
use proximadb_data_model::ProximaValue;
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

/// Build a small batch of synthetic records suitable for the NOVA
/// writer. Each record carries a dense fp32 embedding so the writer
/// has something to encode into the Parquet column.
fn synthetic_batch(count: usize, dim: u32) -> Vec<ProximaRecord> {
    (0..count)
        .map(|i| {
            let oid = format!("rec-{i:06}");
            let mut props = std::collections::HashMap::new();
            props.insert(
                "text".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(format!("record {i}"))),
            );
            ProximaRecord {
                oid: oid.clone(),
                local_id: Some(oid),
                tenant_id: "test-tenant".to_string(),
                created_at_ns: 1_700_000_000_000_000_000 + i as i64,
                updated_at_ns: 1_700_000_000_000_000_000 + i as i64,
                origin: Some("phase_2f_b_test".to_string()),
                props,
                embeddings: vec![EmbeddingCell::new_fp32(
                    "test-model",
                    "dense_vector",
                    dim,
                    (0..dim).map(|j| (i + j as usize) as f32 * 0.01).collect(),
                )],
                ..ProximaRecord::default()
            }
        })
        .collect()
}

/// Core assertion: the override actually does the LSM bypass, not
/// just compile. Returns `used_engine_specific_path: true` and
/// produces a file on disk under `base_path`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nova_ingest_sorted_segment_writes_parquet_and_reports_engine_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_path = format!("file://{}", temp.path().display());
    let collection_id = "42";
    let records = synthetic_batch(8, 4);

    let engine = NovaEngine::new()
        .await
        .expect("NOVA engine construction should succeed in tests");

    // Sort by oid up-front — the override's documented contract is
    // that input is already sorted (drainer sorts before calling).
    // Our synthetic_batch is already sorted, but locking the
    // assertion explicitly here documents the invariant.
    assert!(
        records.windows(2).all(|w| w[0].oid <= w[1].oid),
        "synthetic_batch must produce sorted records by oid"
    );

    let result = engine
        .ingest_sorted_segment(collection_id, &base_path, records)
        .await
        .expect("ingest_sorted_segment should succeed on a fresh tempdir");

    // The single assertion that makes this a Phase 2F-b smoke test
    // rather than a trivial trait-shape check: NOVA must report that
    // it took the engine-specific path. If this flips to false, the
    // override has silently regressed to the default fallback.
    assert!(
        result.used_engine_specific_path,
        "NOVA must return used_engine_specific_path=true; got {result:?}"
    );
    assert_eq!(
        result.record_count, 8,
        "all 8 records should be reported as written"
    );
    assert_eq!(result.collection_id, collection_id);
    assert!(
        !result.synthetic_segment_id.is_empty()
            && result.synthetic_segment_id != "default-fallback",
        "synthetic_segment_id should name the written file or a stable id, got {:?}",
        result.synthetic_segment_id
    );

    // Prove the override actually persisted bytes. NOVA writes
    // `nova_{collection_id}_{ts}_{uuid}.parquet` under
    // `{base_path}/{collection_id}/data/`. We don't validate the
    // exact name (depends on UUID + timestamp) — just that at least
    // one .parquet file exists under the collection's data dir.
    let collection_data_dir = temp.path().join(collection_id).join("data");
    let parquet_files = count_files_with_extension(&collection_data_dir, "parquet");
    assert!(
        parquet_files >= 1,
        "expected at least one .parquet file under {:?}, found {}",
        collection_data_dir,
        parquet_files
    );
}

/// Recursively count files with the given extension under `dir`.
/// Returns 0 when `dir` doesn't exist. Avoids the `walkdir` dep —
/// the tree we care about is shallow (collection_id/data/*.parquet).
fn count_files_with_extension(dir: &std::path::Path, ext: &str) -> usize {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut count = 0;
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            count += count_files_with_extension(&path, ext);
        } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            count += 1;
        }
    }
    count
}

/// Empty batch short-circuits without touching the engine writer.
/// Locks the contract the drainer relies on for empty poll batches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nova_ingest_sorted_segment_short_circuits_empty_batch() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_path = format!("file://{}", temp.path().display());

    let engine = NovaEngine::new()
        .await
        .expect("NOVA engine construction should succeed in tests");

    let result = engine
        .ingest_sorted_segment("empty_collection", &base_path, Vec::new())
        .await
        .expect("empty batch should not error");

    assert!(
        result.used_engine_specific_path,
        "even the empty short-circuit reports the engine path"
    );
    assert_eq!(result.record_count, 0);
    assert_eq!(result.synthetic_segment_id, "empty");

    // No data files should have been created.
    let data_dir = temp.path().join("empty_collection").join("data");
    assert!(
        !data_dir.exists()
            || std::fs::read_dir(&data_dir)
                .map(|d| d.count() == 0)
                .unwrap_or(true),
        "empty batch must not create any files",
    );
}
