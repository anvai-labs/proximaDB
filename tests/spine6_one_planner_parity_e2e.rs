//! TD-SPINE-6 (ADR-098 D5): one decision path for every surface — the
//! full-server legs. The in-process legs (canonical /sql handler, gRPC, MCP,
//! mutation probe) live in tests/integration/query_facade/
//! test_sql_transport_rendering_parity.rs; this file boots a REAL server and
//! proves the SAME SQL renders the SAME rows over:
//!
//!   pgwire             — the Postgres wire protocol (the scheduler's home)
//!   REST /api/v2/sql   — the api_handlers spine (the scheduler control leg)
//!   REST /api/v2/query — the unified query PORT with a plain-SQL body (the
//!                        leg TD-SPINE-6 converged onto the scheduler path)
//!
//! plus the Strong-freshness read-after-write ratchet with the OLAP result
//! cache enabled (mandate #16c: a Strong read through the port bypasses the
//! query cache, so it must observe every committed write).
//!
//! Env note: the second test sets PROXIMADB_QUERY_RESULT_CACHE=1 before the
//! server boots — env-gated tests keep their own FILE (process isolation
//! under nextest makes this deterministic; the registry row for the flag is
//! pre-existing).

use serde_json::json;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

fn cells_of(row: &serde_json::Value, columns: &[&str]) -> Vec<String> {
    columns
        .iter()
        .map(|column| {
            let cell = &row[*column];
            match cell {
                serde_json::Value::Null => String::new(),
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            }
        })
        .collect()
}

fn legs_agree(reference: &[Vec<String>], candidate: &[Vec<String>]) -> bool {
    reference.len() == candidate.len()
        && reference
            .iter()
            .zip(candidate.iter())
            .all(|(want, got)| want == got)
}

/// Same SQL over pgwire, /api/v2/sql and /api/v2/query on ONE live server.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn same_sql_one_planner_across_pgwire_rest_v2_and_unified_query_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");

    let table = server.unique_name("spine6");
    let ddl = format!("CREATE TABLE {table} (id TEXT, score DOUBLE, label TEXT)");
    let dml =
        format!("INSERT INTO {table} (id, score, label) VALUES ('a', 1.5, 'x'), ('b', 2.5, 'y')");
    let select = format!("SELECT id, score, label FROM {table} ORDER BY id");

    // Seed over pgwire: the write surfaces share one WAL/store, so the rows
    // are visible to every read surface (the cross-surface invariant).
    pg_server::rows(&client, &ddl).await;
    pg_server::rows(&client, &dml).await;

    let columns = ["id", "score", "label"];

    // pgwire leg (the scheduler's home surface).
    let pg_rows = pg_server::rows(&client, &select).await;
    assert_eq!(pg_rows.len(), 2, "pgwire leg: {pg_rows:?}");

    let (http, base) = server.rest_client().expect("rest client");

    // REST /api/v2/sql leg (api_handlers spine — the control).
    let sql_response = http
        .post(format!("{base}/api/v2/sql"))
        .json(&json!({ "query": select }))
        .send()
        .await
        .expect("POST /api/v2/sql");
    assert_eq!(
        sql_response.status(),
        reqwest::StatusCode::OK,
        "REST v2 /sql leg must return 200"
    );
    let sql_json: serde_json::Value = sql_response.json().await.expect("SqlResponse JSON");
    let rest_rows: Vec<Vec<String>> = sql_json["rows"]
        .as_array()
        .expect("REST rows")
        .iter()
        .map(|row| cells_of(row, &columns))
        .collect();

    // REST /api/v2/query leg (unified query port; UQL language with a plain
    // SQL body — pre-TD-SPINE-6 this body was refused by the federated core).
    let query_response = http
        .post(format!("{base}/api/v2/query"))
        .json(&json!({ "language": "uql", "query": select }))
        .send()
        .await
        .expect("POST /api/v2/query");
    assert_eq!(
        query_response.status(),
        reqwest::StatusCode::OK,
        "REST v2 /query (port) leg must return 200"
    );
    let port_json: serde_json::Value = query_response.json().await.expect("port JSON");
    let port_rows: Vec<Vec<String>> = port_json["records"]
        .as_array()
        .unwrap_or_else(|| panic!("port records array missing: {port_json}"))
        .iter()
        .map(|row| cells_of(row, &columns))
        .collect();

    // The one-planner acceptance: identical rendered rows on every surface.
    assert!(
        legs_agree(&pg_rows, &rest_rows),
        "pgwire vs REST /api/v2/sql divergence:\n  pgwire: {pg_rows:?}\n  rest:  {rest_rows:?}"
    );
    assert!(
        legs_agree(&pg_rows, &port_rows),
        "pgwire vs REST /api/v2/query (port) divergence:\n  pgwire: {pg_rows:?}\n  port:  {port_rows:?}"
    );
}

/// Strong freshness through the port with the OLAP result cache ON: every
/// committed write must be visible (mandate #16c — the Strong path bypasses
/// the query-cache lookup, so a stale cache entry can never be served).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strong_freshness_port_read_sees_every_write_with_result_cache_on_e2e() {
    // BEFORE boot: enable the (default-OFF) OLAP result cache. Read per call
    // (no OnceLock), but set it first so the initial SELECT can populate it.
    // Edition 2024: set_var is unsafe (single-threaded test prologue here).
    unsafe { std::env::set_var("PROXIMADB_QUERY_RESULT_CACHE", "1") };
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");

    let table = server.unique_name("spine6_fresh");
    pg_server::rows(&client, &format!("CREATE TABLE {table} (id TEXT)")).await;
    pg_server::rows(
        &client,
        &format!("INSERT INTO {table} (id) VALUES ('a'), ('b')"),
    )
    .await;
    let select = format!("SELECT id FROM {table} ORDER BY id");

    let (http, base) = server.rest_client().expect("rest client");

    // Prime the cache through the api_handlers surface (consults it under
    // Strong LSN-pinned eligibility) and the port surface (Strong ⇒ bypass).
    for _ in 0..2 {
        let status = http
            .post(format!("{base}/api/v2/sql"))
            .json(&json!({ "query": select }))
            .send()
            .await
            .expect("POST /api/v2/sql")
            .status();
        assert_eq!(status, reqwest::StatusCode::OK);
    }

    // A committed write AFTER the cached reads.
    pg_server::rows(&client, &format!("INSERT INTO {table} (id) VALUES ('c')")).await;

    // The port's Strong read must see all three rows — never a stale cached
    // two-row result (the 16c guarantee the bypass provides).
    let port_json: serde_json::Value = http
        .post(format!("{base}/api/v2/query"))
        .json(&json!({ "language": "uql", "query": select }))
        .send()
        .await
        .expect("POST /api/v2/query")
        .json()
        .await
        .expect("port JSON");
    let rows = port_json["records"]
        .as_array()
        .unwrap_or_else(|| panic!("port records array missing: {port_json}"));
    let ids: Vec<String> = rows
        .iter()
        .map(|row| row["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        ids,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "Strong-freshness port read must observe every committed write: {port_json}"
    );
}
