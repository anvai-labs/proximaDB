//! End-to-end test for `proximadb_embedding_precision_canonical_bytes`.
//!
//! Boots `EmbeddedProximaDB` (the in-process equivalent of the running
//! server — same `SharedServices` + storage stack as
//! `proximadb-server`), inserts records carrying typed
//! `EmbeddingValues::Fp16` cells through `insert_proxima_records`,
//! forces a WAL flush, then scrapes the
//! `proximadb_embedding_precision_canonical_bytes` gauge directly from
//! the precision-metrics registry (same source the
//! `/metrics/prometheus` HTTP endpoint reads).
//!
//! What this proves:
//! - The WAL bincode serialization strategy walks every embedding
//!   cell's `.precision` field and increments the per-precision
//!   counter at flush boundary.
//! - Records with `EmbeddingValues::Fp16` payloads are accounted as
//!   `precision="fp16"` with the bytes-per-element budget from
//!   `EmbeddingValues::byte_size` (2 bytes/elem for fp16 vs 4 for fp32).
//! - A parallel fp32-cell insert into a separate collection accounts
//!   as `precision="fp32"`, so the two precisions are independently
//!   tracked.
//!
//! What this doesn't prove:
//! - The full ingest path through the embedding drainer (the embedded
//!   API skips the drainer; the runtime catalog-resolver wire-up in
//!   #74 only matters for the queue-driven ingest path). The drainer
//!   integration is covered by `services::embedding_drainer::tests::drainer_stamps_target_precision_from_resolver`.
//! - The byte-halving ratio assertion (covered by
//!   `storage::engines::core::formats::arrow_block::writer::tests::fp16_file_is_approximately_half_the_fp32_size`).
//!
//! Together with those two, this test closes the loop: the production
//! metric pipeline actually records fp16 byte accounting end-to-end.

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use proximadb_records::{EmbeddingCell, EmbeddingScalarType, EmbeddingValues, ProximaRecord};

fn make_fp16_record(oid: &str, dim: usize) -> ProximaRecord {
    let f16s: Vec<half::f16> = (0..dim)
        .map(|i| half::f16::from_f32((i as f32) * 0.125))
        .collect();
    ProximaRecord {
        oid: oid.to_string(),
        local_id: Some(oid.to_string()),
        embeddings: vec![EmbeddingCell {
            model_id: "test".to_string(),
            modality: "dense_vector".to_string(),
            dim: dim as u32,
            values: EmbeddingValues::Fp16(f16s),
            precision: EmbeddingScalarType::Fp16,
            ..Default::default()
        }],
        ..ProximaRecord::default()
    }
}

fn make_fp32_record(oid: &str, dim: usize) -> ProximaRecord {
    let vs: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.125).collect();
    ProximaRecord {
        oid: oid.to_string(),
        local_id: Some(oid.to_string()),
        embeddings: vec![EmbeddingCell::new_fp32(
            "test",
            "dense_vector",
            dim as u32,
            vs,
        )],
        ..ProximaRecord::default()
    }
}

/// End-to-end proof: a fp16 ProximaRecord, written through the embedded
/// DB's WAL flush, increments
/// `proximadb_embedding_precision_canonical_bytes{precision="fp16"}` by
/// exactly its on-disk byte cost (2 B/elem × dim × record count).
///
/// Routes:
/// - `PROXIMADB_EMBED_PRECISION_SCHEMA_V2=true` flips the WAL bincode
///   strategy to `serialize_batch_with_v2_segment_header`, which now
///   uses the v2 wire shape (`ProximaRecordV2` with natural enum-aware
///   embeddings serde) instead of the legacy fp32-refusing v1 impl.
/// - The per-precision canonical_bytes counter at
///   `src/storage/persistence/write_ahead_log/bincode_serialization_strategy.rs:247`
///   walks each cell's `.precision` field and accumulates
///   `cell.values_byte_size()` keyed by `precision_label(cell.precision)`.
///
/// Earlier this test stood `#[ignore]`'d to surface the WAL v2 record
/// encoding gap. The gap is now filled by the
/// `proximadb_records::wire_v2` module and the
/// `VectorBatchSerializer::serialize_batch_v2` trait method.
#[test]
fn fp16_records_increment_canonical_bytes_metric_at_flush() {
    // Enable WAL schema v2 BEFORE anything touches the cached config —
    // the v1 bincode serializer for EmbeddingCell deliberately refuses
    // non-Fp32 variants (INT-2.5b step 2 Q1) so an fp16 record on the
    // v1 path errors with "Failed to serialize batch for WAL". Schema
    // v2 emits the v2 segment header and (once the per-record v2
    // encoding lands) will accept typed records.
    //
    // `EmbeddingPrecisionConfig::cached()` reads the env var via
    // OnceLock; each `tests/*.rs` file compiles as its own binary so
    // the lock here is isolated to this test process.
    // SAFETY: setting an env var before any threads are spawned by
    // the embedded DB; the OnceLock read happens inside the WAL
    // codepath we'll trigger below.
    unsafe {
        std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true");
    }

    // Init the precision metrics registry — production does this at
    // server boot. The metric is process-global so tests sharing this
    // crate may see counts from other tests; that's OK since we
    // assert presence + non-zero on a per-collection label, not equality.
    proximadb::observability::precision_metrics::init_precision_metrics();

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("fp16_metric_test");
    std::fs::create_dir_all(&data_path).expect("create data dir");

    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    let dim: usize = 64;
    let fp16_collection = format!(
        "fp16_metric_coll_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let fp32_collection = format!(
        "fp32_metric_coll_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .wrapping_add(1)
    );

    db.create_collection(&fp16_collection, dim as u32, Some("sst"))
        .expect("create fp16 collection");
    db.create_collection(&fp32_collection, dim as u32, Some("sst"))
        .expect("create fp32 collection");

    // Insert fp16-typed records into the fp16 collection.
    let fp16_records: Vec<ProximaRecord> = (0..50)
        .map(|i| make_fp16_record(&format!("rec_fp16_{:03}", i), dim))
        .collect();
    db.insert_proxima_records(&fp16_collection, fp16_records)
        .expect("insert fp16");

    // Insert fp32-typed records into the fp32 collection (control group).
    let fp32_records: Vec<ProximaRecord> = (0..50)
        .map(|i| make_fp32_record(&format!("rec_fp32_{:03}", i), dim))
        .collect();
    db.insert_proxima_records(&fp32_collection, fp32_records)
        .expect("insert fp32");

    // Force a flush so the WAL bincode serialization strategy walks
    // every record and increments the per-precision canonical_bytes
    // counter. Without this, records may still be in the in-memory
    // memtable and the metric won't have been written yet.
    db.flush().expect("flush");

    // Scrape the precision-metrics registry directly — same content
    // the /metrics/prometheus endpoint appends.
    let scrape = proximadb::observability::precision_metrics::scrape_text();

    // Locate the per-(collection, precision) gauge lines. Prometheus
    // text format: `metric{label="value",label="value"} value`.
    let metric_prefix = "proximadb_embedding_precision_canonical_bytes";
    let fp16_label_fragment = format!(r#"collection="{}""#, fp16_collection);
    let fp32_label_fragment = format!(r#"collection="{}""#, fp32_collection);

    let mut fp16_value: Option<i64> = None;
    let mut fp32_value: Option<i64> = None;

    for line in scrape.lines() {
        if !line.starts_with(metric_prefix) {
            continue;
        }
        let after_braces = match line.split_once('}') {
            Some((_, tail)) => tail.trim(),
            None => continue,
        };
        let value: i64 = match after_braces.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        if line.contains(&fp16_label_fragment) && line.contains(r#"precision="fp16""#) {
            fp16_value = Some(value);
        }
        if line.contains(&fp32_label_fragment) && line.contains(r#"precision="fp32""#) {
            fp32_value = Some(value);
        }
    }

    // Expected canonical bytes per record:
    //   fp16: 64 elems × 2 B/elem = 128 B
    //   fp32: 64 elems × 4 B/elem = 256 B
    // Expected total over 50 records:
    //   fp16: 50 × 128 = 6400 B
    //   fp32: 50 × 256 = 12_800 B
    // Use exact-equality assertions — the metric is a counter that
    // sums byte_size() per cell, deterministically.
    let expected_fp16: i64 = 50 * 64 * 2;
    let expected_fp32: i64 = 50 * 64 * 4;

    assert_eq!(
        fp16_value,
        Some(expected_fp16),
        "canonical_bytes{{collection={fp16_collection},precision=fp16}} = {fp16_value:?}, expected {expected_fp16}. Full scrape:\n{scrape}"
    );
    assert_eq!(
        fp32_value,
        Some(expected_fp32),
        "canonical_bytes{{collection={fp32_collection},precision=fp32}} = {fp32_value:?}, expected {expected_fp32}. Full scrape:\n{scrape}"
    );

    // Sanity: byte ratio between the two collections should be exactly 0.5
    // (fp16 is half of fp32 element-size; equal element counts means
    // equal halving). Confirms the metric layer agrees with the storage
    // layer's byte-halving assertion in
    // arrow_block::writer::fp16_file_is_approximately_half_the_fp32_size.
    let ratio = fp16_value.unwrap() as f64 / fp32_value.unwrap() as f64;
    assert!(
        (0.49..=0.51).contains(&ratio),
        "fp16/fp32 byte ratio {ratio:.4} not within [0.49, 0.51]"
    );
}
