use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use tempfile::TempDir;

#[test]
fn test_embedded_create_collection_reports_tst_engine() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let mut config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
    config.enable_wal = false;

    let db = EmbeddedProximaDB::new(config).expect("create embedded db");
    db.create_collection("tst_parity_collection", 16, Some("tst"))
        .expect("create tst collection");

    let collection = db
        .get_collection("tst_parity_collection")
        .expect("get collection")
        .expect("collection exists");

    assert_eq!(collection.engine, "tst");
}
