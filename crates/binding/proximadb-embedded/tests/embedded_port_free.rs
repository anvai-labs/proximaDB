//! Embedded co-design (Phase 1 / Pillar F): the fused `EmbeddedProximaDB` path is
//! **port-free** — it constructs no network listener and binds none of the
//! server's ports. This is the direct regression test for the "embedded server
//! port collision" the user hit: that collision was the *networked* `db.start()`
//! path (`multi_server::start_unified` binds 5678/5433/5680 + internal
//! 15678/15679), NOT the fused library. A fused in-process database deletes
//! Dimension 2 (network), so it must never open a socket (co-design tenets 1 & 3).
//!
//! The test does a real in-process create + insert + search (proving the fused
//! path is fully functional with zero ports), then asserts every well-known
//! ProximaDB server port that was free *before* init is *still* free after —
//! i.e. the embedded DB grabbed none of them. Ports held by an unrelated process
//! (e.g. a real server from a concurrent session) are skipped, so the test is
//! robust against external contention rather than flaky.

use proximadb_embedded::{EmbeddedConfig, EmbeddedProximaDB};
use std::net::TcpListener;
use tempfile::TempDir;

/// External REST/gRPC mux, pgwire, Arrow Flight, and the two `+10000` internal
/// listeners the networked server stands up in unified mode.
const SERVER_PORTS: &[u16] = &[5678, 5433, 5680, 15678, 15679];

fn bindable(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

#[test]
fn embedded_path_binds_no_server_ports() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Ports free in THIS environment before embedded init. Anything already held
    // (by a concurrent session's server, Docker, etc.) is not the fused core's
    // doing — skip it so the assertion only covers ports we can attribute.
    let free_before: Vec<u16> = SERVER_PORTS
        .iter()
        .copied()
        .filter(|p| bindable(*p))
        .collect();

    let mut config = EmbeddedConfig::for_low_memory(temp_dir.path().to_string_lossy().as_ref());
    config.enable_wal = false;
    let db = EmbeddedProximaDB::new(config).expect("create embedded db");

    // Exercise the full in-process path with zero network involvement.
    db.create_collection("port_free", 8, Some("tst"))
        .expect("create collection");
    let inserted = db
        .insert(
            "port_free",
            vec!["a".to_string(), "b".to_string()],
            vec![
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            None,
        )
        .expect("insert");
    assert_eq!(inserted, 2, "embedded insert should accept both vectors");
    let hits = db
        .search(
            "port_free",
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
            None,
        )
        .expect("search");
    assert!(!hits.is_empty(), "embedded search should return a hit");

    // The fused DB must not have bound any server port that was free before init.
    for p in &free_before {
        assert!(
            bindable(*p),
            "EmbeddedProximaDB bound server port {p} — the fused core must be port-free \
             (only the networked server's db.start() may open sockets)"
        );
    }
}
