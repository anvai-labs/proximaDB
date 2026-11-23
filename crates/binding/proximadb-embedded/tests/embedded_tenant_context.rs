//! Embedded co-design (Pillar D, write-side): the governance boundary contract is
//! threaded through the embedded insert path. Configuring a tenant
//! (`EmbeddedConfig::with_tenant`) routes inserts through the same tenant-scoped
//! path the networked server uses, stamping `tenant_id` on records (co-design
//! tenet 3 — the boundary is the contract). The default (no tenant) keeps legacy
//! single-tenant behavior byte-identical: records carry an empty `tenant_id` and
//! the collection-keyed on-disk layout is unchanged (non-breaking — embedded
//! vector storage is not tenant-keyed).

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use tempfile::TempDir;

fn insert_one(db: &EmbeddedProximaDB, coll: &str) {
    db.create_collection(coll, 4, Some("tst"))
        .expect("create collection");
    let n = db
        .insert(
            coll,
            vec!["r1".to_string()],
            vec![vec![0.1, 0.2, 0.3, 0.4]],
            None,
        )
        .expect("insert");
    assert_eq!(n, 1, "insert should accept the record");
}

#[test]
fn configured_tenant_is_stamped_on_records() {
    let tmp = TempDir::new().expect("tempdir");
    let config =
        EmbeddedConfig::for_low_memory(tmp.path().to_string_lossy().as_ref()).with_tenant("acme");
    let db = EmbeddedProximaDB::new(config).expect("embedded db");

    insert_one(&db, "scoped");
    let rec = db
        .get_vector("scoped", "r1")
        .expect("get_vector")
        .expect("record exists");
    assert_eq!(
        rec.tenant_id, "acme",
        "configured tenant must be stamped on the stored record"
    );
}

#[test]
fn default_is_single_tenant_unchanged() {
    let tmp = TempDir::new().expect("tempdir");
    let config = EmbeddedConfig::for_low_memory(tmp.path().to_string_lossy().as_ref());
    let db = EmbeddedProximaDB::new(config).expect("embedded db");

    insert_one(&db, "legacy");
    let rec = db
        .get_vector("legacy", "r1")
        .expect("get_vector")
        .expect("record exists");
    assert!(
        rec.tenant_id.is_empty(),
        "default (no configured tenant) must keep an empty tenant_id (legacy behavior)"
    );
}

// Risk-coverage scenarios (docs/10-quality/RISK_CONTRACT.toml :: tenant-isolation):
//   scenario: tenant-stamped        — configured tenant is stamped on stored records
//   scenario: default-single-tenant — default (no tenant) keeps empty tenant_id (non-breaking)
