//! Engine TTL conformance (CLAUDE.md directive 16a): a TTL-expired record —
//! `0 < valid_to_ns < now_ns` — must never be returned by vector search, for
//! every selectable storage engine, both while it is still in the WAL /
//! memtable (unflushed) and after it has been flushed into the engine's
//! segment.
//!
//! Each engine x path combination is its own `#[test]` so passing ones can be
//! wired into a CI lane independently of the ones that document a leak.
//!
//! Procedure per test:
//!   1. create a collection on the engine under test;
//!   2. insert N live filler records + one record `ttl_victim` with
//!      `valid_to_ns = now + TTL`;
//!   3. (flushed variant) flush to the engine segment;
//!   4. search at the victim's vector BEFORE expiry — it must be found
//!      (positive control) and we record whether the returned hit carries
//!      `OptimizedSearchRecord.valid_to_ns`;
//!   5. sleep past `valid_to_ns`, search again — the victim must be absent
//!      while live fillers are still returned (second positive control).
//!
//! Search goes through `VectorOperationsService::unified_search_native` (the
//! WAL + AXIS + storage two-stage path, no query-result cache) with a query
//! vector perturbed per call so no cache layer can answer it.
//!
//! Every test prints one `TTL_CONFORMANCE` line for the report table:
//!   `TTL_CONFORMANCE engine=<e> path=<p> verdict=<hidden|LEAKS|n/a> ...`

use proximadb_embedded::{AccessMode, EmbeddedConfig, EmbeddedProximaDB, StorageLocationConfig};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

const DIM: usize = 8;
const FILLERS: usize = 32;
const VICTIM: &str = "ttl_victim";

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

fn create_test_db(default_engine: &str) -> (TempDir, EmbeddedProximaDB) {
    let temp_dir = TempDir::new().expect("tempdir");
    let root = temp_dir.path().to_str().expect("utf8 path").to_string();
    let config = EmbeddedConfig {
        storage_locations: vec![StorageLocationConfig::new(&root)],
        metadata_path: format!("{root}/metadata"),
        cache_size_mb: 64,
        default_engine: default_engine.to_string(),
        enable_wal: true,
        wal_sync_mode: "batch".to_string(),
        block_prune_mode: "sqrt".to_string(),
        block_prune_ratio: 0.2,
        block_prune_min_keep: 1,
        block_prune_max_keep: 0,
        enable_rl_planner: false,
        rl_policy_path: None,
        access_mode: AccessMode::Exclusive,
        node_id: None,
        tenant_id: None,
    };
    let db = EmbeddedProximaDB::new(config).expect("embedded db");
    (temp_dir, db)
}

fn record(
    oid: &str,
    vector: Vec<f32>,
    valid_to_ns: Option<i64>,
) -> proximadb_records::ProximaRecord {
    let ts = now_ns();
    proximadb_records::ProximaRecord {
        oid: oid.to_string(),
        embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
            "default", "vector", DIM as u32, vector,
        )],
        created_at_ns: ts,
        updated_at_ns: ts,
        record_version: 1,
        valid_to_ns,
        ..Default::default()
    }
}

/// Victim lives on its own axis; fillers are spread away from it so the
/// victim is the unambiguous top-1 for a query at its vector.
fn victim_vector() -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[0] = 1.0;
    v
}

fn filler_vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|j| {
            if j == 0 {
                0.05
            } else {
                (((i * 31 + j * 17) % 97) as f32 / 97.0) + 0.1
            }
        })
        .collect()
}

/// Query near the victim; `salt` perturbs it so each call is a distinct query.
fn query(salt: usize) -> Vec<f32> {
    let mut q = victim_vector();
    q[1] = 0.001 * (salt as f32 + 1.0);
    q
}

fn search(
    db: &EmbeddedProximaDB,
    collection: &str,
    salt: usize,
) -> Result<Vec<proximadb::core::search::results::OptimizedSearchRecord>, String> {
    db.runtime().block_on(async {
        db.shared_services()
            .vector_operations_service
            .unified_search_native(
                collection,
                query(salt),
                10,
                None,
                None,
                #[cfg(feature = "abac-policy")]
                &proximadb_abac::ReadContext::system(
                    proximadb_abac::SystemReadReason::Statistics,
                    "engine_ttl_conformance_test",
                ),
            )
            .await
            .map_err(|e| e.to_string())
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PathKind {
    Unflushed,
    Flushed,
}

impl PathKind {
    fn label(self) -> &'static str {
        match self {
            PathKind::Unflushed => "unflushed",
            PathKind::Flushed => "flushed",
        }
    }
}

fn run_ttl_conformance(engine: &str, path: PathKind) {
    let (_tmp, db) = create_test_db(engine);
    let collection = format!("ttl_conf_{engine}_{}", path.label());

    if let Err(e) = db.create_collection(&collection, DIM as u32, Some(engine)) {
        println!(
            "TTL_CONFORMANCE engine={engine} path={} verdict=n/a reason=\"create_collection failed: {e}\"",
            path.label()
        );
        // Not selectable for new collections (e.g. VIPER after ADR-093 S2):
        // the create-path matrix does not apply. Fail loudly only if the
        // error is something other than a rejection of the engine itself.
        let msg = e.to_string().to_lowercase();
        assert!(
            msg.contains("deprecated")
                || msg.contains("unknown storage engine")
                || msg.contains("not supported")
                || msg.contains("experimental")
                || msg.contains(engine),
            "[{engine}] unexpected create_collection failure: {e}"
        );
        return;
    }

    // Generous TTL for the flushed path: flush must complete before expiry
    // so the pre-expiry positive control is meaningful.
    let ttl = match path {
        PathKind::Unflushed => Duration::from_secs(3),
        PathKind::Flushed => Duration::from_secs(10),
    };
    let valid_to_ns = now_ns() + ttl.as_nanos() as i64;

    let mut records: Vec<_> = (0..FILLERS)
        .map(|i| record(&format!("live_{i:03}"), filler_vector(i), None))
        .collect();
    records.push(record(VICTIM, victim_vector(), Some(valid_to_ns)));
    let inserted = db
        .insert_proxima_records(&collection, records)
        .unwrap_or_else(|e| panic!("[{engine}] insert failed: {e}"));
    assert_eq!(inserted, FILLERS + 1, "[{engine}] insert count");

    if path == PathKind::Flushed {
        db.flush()
            .unwrap_or_else(|e| panic!("[{engine}] flush failed: {e}"));
    }

    // ---- Before expiry: victim must be found (positive control). ----
    let before = search(&db, &collection, 1)
        .unwrap_or_else(|e| panic!("[{engine}/{}] pre-expiry search failed: {e}", path.label()));
    let searched_at = now_ns();
    assert!(
        searched_at < valid_to_ns,
        "[{engine}/{}] test setup too slow: pre-expiry search ran after valid_to_ns \
         (flush took longer than the TTL); positive control is void",
        path.label()
    );
    let victim_hit = before.iter().find(|r| r.id == VICTIM);
    let pre_valid_to = victim_hit.map(|r| r.valid_to_ns);
    let pre_expires_at = victim_hit.map(|r| r.expires_at);
    println!(
        "[{engine}/{}] pre-expiry: {} hits, victim_found={}, victim.valid_to_ns={:?} \
         (inserted {valid_to_ns}), victim.expires_at={:?}",
        path.label(),
        before.len(),
        victim_hit.is_some(),
        pre_valid_to,
        pre_expires_at
    );
    assert!(
        victim_hit.is_some(),
        "[{engine}/{}] positive control failed: victim not found before expiry; ids={:?}",
        path.label(),
        before.iter().map(|r| r.id.as_str()).collect::<Vec<_>>()
    );
    let valid_to_populated = matches!(pre_valid_to, Some(Some(v)) if v == valid_to_ns);

    // ---- After expiry: victim must be hidden. ----
    let wait_ns = (valid_to_ns - now_ns()).max(0) as u64 + 750_000_000;
    std::thread::sleep(Duration::from_nanos(wait_ns));
    assert!(now_ns() > valid_to_ns);

    let after = search(&db, &collection, 2)
        .unwrap_or_else(|e| panic!("[{engine}/{}] post-expiry search failed: {e}", path.label()));
    let leaked = after.iter().any(|r| r.id == VICTIM);
    let live_after = after.iter().filter(|r| r.id.starts_with("live_")).count();

    println!(
        "TTL_CONFORMANCE engine={engine} path={} verdict={} valid_to_ns_on_hit={} \
         pre_valid_to_ns={:?} pre_expires_at={:?} post_hits={} post_live_hits={}",
        path.label(),
        if leaked { "LEAKS" } else { "hidden" },
        if valid_to_populated { "yes" } else { "no" },
        pre_valid_to,
        pre_expires_at,
        after.len(),
        live_after
    );

    assert!(
        live_after > 0,
        "[{engine}/{}] positive control failed: no live fillers returned after expiry",
        path.label()
    );
    assert!(
        !leaked,
        "[{engine}/{}] TTL LEAK: expired record '{VICTIM}' (valid_to_ns={valid_to_ns}) \
         still returned after expiry; hit={:?}",
        path.label(),
        after
            .iter()
            .find(|r| r.id == VICTIM)
            .map(|r| (r.score, r.valid_to_ns, r.expires_at))
    );

    db.close();
}

macro_rules! ttl_case {
    ($name:ident, $engine:literal, $path:expr) => {
        #[test]
        fn $name() {
            run_ttl_conformance($engine, $path);
        }
    };
    // Known failure, documented: run with `--run-ignored only` to reproduce.
    ($name:ident, $engine:literal, $path:expr, known = $why:literal) => {
        #[test]
        #[ignore = $why]
        fn $name() {
            run_ttl_conformance($engine, $path);
        }
    };
    ($name:ident, $engine:literal, $path:expr, cfg = $cfg:meta) => {
        #[cfg($cfg)]
        #[test]
        fn $name() {
            run_ttl_conformance($engine, $path);
        }
    };
    ($name:ident, $engine:literal, $path:expr, cfg = $cfg:meta, known = $why:literal) => {
        #[cfg($cfg)]
        #[test]
        #[ignore = $why]
        fn $name() {
            run_ttl_conformance($engine, $path);
        }
    };
}

ttl_case!(sst_unflushed_hides_expired, "sst", PathKind::Unflushed);
ttl_case!(sst_flushed_hides_expired, "sst", PathKind::Flushed);
ttl_case!(helix_unflushed_hides_expired, "helix", PathKind::Unflushed);
ttl_case!(
    helix_flushed_hides_expired,
    "helix",
    PathKind::Flushed,
    known = "TTL LEAK: HELIX segment search (readers::search_sstable -> proxima::search_helix_sstable) \
             drops valid_to_ns and never checks expiry"
);
ttl_case!(nova_unflushed_hides_expired, "nova", PathKind::Unflushed);
ttl_case!(
    nova_flushed_hides_expired,
    "nova",
    PathKind::Flushed,
    known = "TTL LEAK: NOVA search_standard reads via UnifiedParquetReader, which skips the \
             expires_at column, so valid_to_ns is lost and nothing filters expiry"
);
ttl_case!(viper_unflushed_hides_expired, "viper", PathKind::Unflushed);
ttl_case!(viper_flushed_hides_expired, "viper", PathKind::Flushed);
ttl_case!(tst_unflushed_hides_expired, "tst", PathKind::Unflushed);
ttl_case!(
    tst_flushed_hides_expired,
    "tst",
    PathKind::Flushed,
    known = "BLOCKED: TST do_flush ignores FlushParameters.vector_records and returns \
             FlushResult::default() (success=false), so the WAL flush never materializes"
);
ttl_case!(
    swift_unflushed_hides_expired,
    "swift",
    PathKind::Unflushed,
    cfg = feature = "experimental-engines"
);
ttl_case!(
    swift_flushed_hides_expired,
    "swift",
    PathKind::Flushed,
    cfg = feature = "experimental-engines",
    known = "TTL LEAK: SWIFT segment search builds hits without valid_to_ns (engine.rs / optimized_operations.rs) and does not filter expiry"
);
ttl_case!(
    raptor_unflushed_hides_expired,
    "raptor",
    PathKind::Unflushed,
    cfg = feature = "experimental-engines"
);
ttl_case!(
    raptor_flushed_hides_expired,
    "raptor",
    PathKind::Flushed,
    cfg = feature = "experimental-engines",
    known = "TTL LEAK: RAPTOR segment search builds hits without valid_to_ns (engine.rs / proximadb-raptor-engine lib.rs) and does not filter expiry"
);
