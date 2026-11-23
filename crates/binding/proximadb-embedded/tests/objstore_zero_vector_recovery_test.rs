//! Regression test for TD-OBJSTORE-1 batch 3: dim-1 "zero-vector" KV records
//! (written with `vector: [0.0]`) must survive a persist/restart cycle exactly
//! like embedded-vector records.
//!
//! anvaiops uses zero-vector rows as a durable KV store (API keys, tenant
//! registry, workspace/billing rows): the engine stores them and serves them by
//! id, but the recovery path was observed to bring the collection back EMPTY
//! after an `az vm restart`. The control record (a non-zero dim-1 vector) isolates
//! the zero-vector-ness as the single variable — if the control survives but the
//! zero-vector record does not, the loss is specific to zero vectors, not to
//! recovery in general.
//!
//! Note the collection defaults to Cosine distance, under which a zero vector has
//! an undefined (norm-0) direction — the suspected reason a flush/index build path
//! would drop it.

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use std::collections::HashMap;

const COLLECTION: &str = "kv_recovery";
const NORMAL_ID: &str = "rec-normal";
const ZERO_ID: &str = "rec-zero";

fn seed(data_path: &str) {
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create db");

    db.create_collection(COLLECTION, 1, Some("sst"))
        .expect("create dim-1 collection");

    // Control (non-zero) + subject (zero-vector KV record), same dim-1 collection.
    let ids = vec![NORMAL_ID.to_string(), ZERO_ID.to_string()];
    let vectors = vec![vec![0.5_f32], vec![0.0_f32]];
    let metadata = vec![
        HashMap::from([("kind".to_string(), serde_json::json!("embedded"))]),
        HashMap::from([
            ("kind".to_string(), serde_json::json!("api_key")),
            ("secret".to_string(), serde_json::json!("sk-anvaiops-123")),
        ]),
    ];
    db.insert(COLLECTION, ids, vectors, Some(metadata))
        .expect("insert records");

    // Both records must be retrievable BEFORE restart (matches the TD: the KV
    // record is queryable at seed time; it only 404s after restart).
    assert!(
        db.get_vector(COLLECTION, NORMAL_ID)
            .expect("get normal pre-restart")
            .is_some(),
        "control record must exist pre-restart"
    );
    assert!(
        db.get_vector(COLLECTION, ZERO_ID)
            .expect("get zero pre-restart")
            .is_some(),
        "zero-vector KV record must exist pre-restart"
    );

    // Flush so the records are durable in the on-disk segment (the persist half of
    // the durability cycle). Recovery then reads them back from storage.
    db.flush().expect("flush");
    drop(db);
}

fn reopen_and_check(data_path: &str) {
    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("reopen db");

    // Post-restart the recovered records are served from the flushed segment.
    // (In the embedded harness `get_vector`/`scan_records` read the reset
    // memtable, not the reopened segment, so they are NOT reliable recovery
    // signals here — `search` and collection stats are. This mirrors the
    // existing `embedded_flush_recovery` test, which also verifies via search.)
    let stats = db.stats().expect("stats");
    assert_eq!(
        stats.total_vectors, 2,
        "both records (control + zero-vector) must survive flush+restart; got {}",
        stats.total_vectors
    );

    // The control (non-zero) record must be searchable after restart.
    let hits = db
        .search(COLLECTION, vec![0.5_f32], 10, None)
        .expect("search after restart");
    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    assert!(
        ids.iter().any(|id| id == NORMAL_ID),
        "control record must be recoverable after restart; got {ids:?}"
    );
    // The zero-vector KV record must also come back from the flushed segment.
    assert!(
        ids.iter().any(|id| id == ZERO_ID),
        "zero-vector KV record must be recoverable after restart; got {ids:?}"
    );
}

#[test]
fn zero_vector_kv_record_survives_flush_and_restart() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");
    let data_path = data_path.to_string_lossy().to_string();

    seed(&data_path);
    reopen_and_check(&data_path);
}
