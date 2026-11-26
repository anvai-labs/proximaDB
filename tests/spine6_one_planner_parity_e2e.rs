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

/// POST `path` with the pgwire connection's tenant pinned via X-Tenant-ID:
/// the pgwire startup `database` doubles as the tenant/catalog (TD-064), and
/// TestPgServer connects with `dbname=proximadb` — without the header the
/// REST surfaces would act as the `default` tenant and not resolve the
/// pgwire-seeded table.
async fn post_tenant_json(
    http: &reqwest::Client,
    url: String,
    tenant: &str,
    body: serde_json::Value,
) -> Result<reqwest::Response, reqwest::Error> {
    http.post(url)
        .header("X-Tenant-ID", tenant)
        .json(&body)
        .send()
        .await
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
    let sql_response = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "proximadb",
        json!({ "query": select }),
    )
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
    let query_response = post_tenant_json(
        &http,
        format!("{base}/api/v2/query"),
        "proximadb",
        json!({ "language": "uql", "query": select }),
    )
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

    // MED-4 (round-1 review): a write on the port surface is a clean 4xx
    // naming the read-only policy — never a silent loss, never a mutation.
    let rejected = post_tenant_json(
        &http,
        format!("{base}/api/v2/query"),
        "proximadb",
        json!({ "language": "uql", "query": format!("DELETE FROM {table}") }),
    )
    .await
    .expect("DELETE probe request sends");
    let status = rejected.status();
    let body: serde_json::Value = rejected.json().await.expect("rejection body");
    assert_eq!(
        status,
        reqwest::StatusCode::BAD_REQUEST,
        "DELETE via /api/v2/query must be a 400, got {status}: {body}"
    );
    assert!(
        body.to_string().contains("read-only"),
        "the rejection must name the policy: {body}"
    );
    // …and the rows survive.
    let after = pg_server::rows(&client, &select).await;
    assert_eq!(after.len(), 2, "no row may be deleted: {after:?}");
}

/// MED-2 (round-1 review), reworked to have teeth. With the OLAP result
/// cache ON in this process (env set before boot; nextest runs each test in
/// its own process):
///
/// * Arm A — the cache IS consulted by the api_handlers path: a second
///   identical read over `/api/v2/sql` performs a LOOKUP (Strong-LSN-pinned)
///   — the lookup counter moves whether it hits or misses.
/// * Arm B — the PORT (Strong) BYPASSES the lookup entirely: the SAME SQL
///   over `/api/v2/query` leaves the lookup counter UNMOVED while returning
///   the identical rows. If the bypass were removed, this read would perform
///   a lookup (a miss if nothing was stamped, a hit if it was) and the
///   counter WOULD move — the test fails either way. (The gate's
///   pure-contract pin, default-env, lives in the unit test
///   `olap_cache_policy_strong_bypasses_the_result_cache`.)
/// * Read-after-write: a write committed after the cached reads is observed
///   by the port's Strong read (16c's user-visible guarantee).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strong_freshness_port_read_sees_every_write_with_result_cache_on_e2e() {
    // BEFORE boot: enable the (default-OFF) OLAP result cache. Read per call
    // (no OnceLock), but set it first so the initial SELECT can populate it.
    // Edition 2024: set_var is unsafe (single-threaded test prologue here).
    unsafe { std::env::set_var("PROXIMADB_QUERY_RESULT_CACHE", "1") };

    /// Total cache LOOKUPS (hits + misses) — the consultation signal. A
    /// lookup moves this even when it misses; the Strong bypass performs no
    /// lookup at all.
    fn cache_lookups() -> u64 {
        let stats = proximadb::network::postgres::relational_pipeline::olap_result_cache()
            .expect("cache flag is set in this process")
            .stats();
        stats.hits + stats.misses
    }

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
    // Strong LSN-pinned eligibility — a miss + insert on first read).
    let first: serde_json::Value = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "proximadb",
        json!({ "query": select }),
    )
    .await
    .expect("prime read sends")
    .json()
    .await
    .expect("prime read body");
    assert_eq!(
        first["rows"].as_array().map(Vec::len),
        Some(2),
        "prime read: {first}"
    );

    // Arm A: the SECOND identical api_handlers read CONSULTS the cache — its
    // lookup runs (the counter moves whether it hits or misses). Without a
    // consultation the Strong-bypass differential below proves nothing.
    let lookups_before = cache_lookups();
    let second: serde_json::Value = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "proximadb",
        json!({ "query": select }),
    )
    .await
    .expect("second read sends")
    .json()
    .await
    .expect("second read body");
    let lookups_after_control = cache_lookups();
    assert_eq!(
        second["rows"].as_array().map(Vec::len),
        Some(2),
        "second read: {second}"
    );
    assert!(
        lookups_after_control > lookups_before,
        "Arm A: the /api/v2/sql control read must CONSULT the cache ({lookups_before} -> {lookups_after_control}); \
         without a consultation the Strong-bypass differential below proves nothing"
    );

    // Arm B: the SAME SQL through the port (Strong ⇒ bypass) does NOT move
    // the hit counter — yet returns the identical rows.
    let port_json: serde_json::Value = post_tenant_json(
        &http,
        format!("{base}/api/v2/query"),
        "proximadb",
        json!({ "language": "uql", "query": select }),
    )
    .await
    .expect("port read sends")
    .json()
    .await
    .expect("port read body");
    let lookups_after_port = cache_lookups();
    let port_rows: Vec<Vec<String>> = port_json["records"]
        .as_array()
        .unwrap_or_else(|| panic!("port records array missing: {port_json}"))
        .iter()
        .map(|row| cells_of(row, &["id"]))
        .collect();
    let control_rows: Vec<Vec<String>> = second["rows"]
        .as_array()
        .expect("control rows")
        .iter()
        .map(|row| cells_of(row, &["id"]))
        .collect();
    assert!(
        legs_agree(&control_rows, &port_rows),
        "port and control must render identical rows: {control_rows:?} vs {port_rows:?}"
    );
    assert_eq!(
        lookups_after_port, lookups_after_control,
        "Arm B: the port's Strong read must NOT even LOOK UP the cache \
         ({lookups_after_control} -> {lookups_after_port}); if it did, the 16c bypass is gone"
    );

    // Read-after-write: a committed write after the cached reads is observed
    // by the port's Strong read (never a stale cached two-row result).
    pg_server::rows(&client, &format!("INSERT INTO {table} (id) VALUES ('c')")).await;
    let port_json: serde_json::Value = post_tenant_json(
        &http,
        format!("{base}/api/v2/query"),
        "proximadb",
        json!({ "language": "uql", "query": select }),
    )
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

/// HIGH-1 (round 2): `EXPLAIN (ANALYZE) <DML>` over the live `/api/v2/query`
/// surface is a 400 with STATE UNCHANGED (the parenthesized-ANALYZE bypass
/// executed the write pre-fix).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_analyze_dml_is_rejected_on_the_query_surface_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");
    let table = server.unique_name("spine6_r2");
    pg_server::rows(&client, &format!("CREATE TABLE {table} (id TEXT)")).await;
    pg_server::rows(
        &client,
        &format!("INSERT INTO {table} (id) VALUES ('keep')"),
    )
    .await;
    let (http, base) = server.rest_client().expect("rest client");

    for statement in [
        format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('rogue')"),
        format!("EXPLAIN (FORMAT JSON, ANALYZE) DELETE FROM {table}"),
        format!("EXPLAIN (ANALYZE, VERBOSE) UPDATE {table} SET id = 'x'"),
    ] {
        let response = post_tenant_json(
            &http,
            format!("{base}/api/v2/query"),
            "proximadb",
            json!({ "language": "uql", "query": statement }),
        )
        .await
        .expect("probe sends");
        let status = response.status();
        let body: serde_json::Value = response.json().await.expect("rejection body");
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "EXPLAIN Analyze DML must be a 400: {body}"
        );
        assert!(
            body.to_string().contains("EXPLAIN"),
            "the rejection must name EXPLAIN: {body}"
        );
    }

    // State unchanged (the tenant view over pgwire).
    let rows = pg_server::rows(&client, &format!("SELECT id FROM {table}")).await;
    assert_eq!(
        rows,
        vec![vec!["keep".to_string()]],
        "state must be unchanged: {rows:?}"
    );
}

/// MED-1 (round 2): on the LIVE server, a `<->` KNN read on `/api/v2/query`
/// is no longer rejected by the port's classifier (the GenericDialect
/// classifier answered a wrong "read-only" 400). Row parity against the
/// authority body over a real vector collection is pinned in-process
/// (`pgvector_knn_sql_on_every_port_leg_matches_the_authority_body`);
/// `/api/v2/sql` itself still rejects `<->` at its GenericDialect
/// single-statement validation (pre-existing, unchanged by this PR).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgvector_knn_read_is_not_rejected_by_the_port_classifier_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");
    let table = server.unique_name("spine6_knn");
    pg_server::rows(
        &client,
        &format!("CREATE TABLE {table} (id TEXT, embedding TEXT)"),
    )
    .await;
    let (http, base) = server.rest_client().expect("rest client");
    for knn in [
        format!("SELECT id FROM {table} WHERE embedding <-> '[1.0, 0.0]' < 1.5"),
        format!("SELECT id FROM {table} ORDER BY embedding <-> '[1.0, 0.0]' LIMIT 3"),
    ] {
        let port = post_tenant_json(
            &http,
            format!("{base}/api/v2/query"),
            "proximadb",
            json!({ "language": "uql", "query": knn }),
        )
        .await
        .expect("port leg sends");
        let status = port.status();
        let text = port.text().await.unwrap_or_default();
        assert!(
            !text.contains("read-only") && !text.contains("cannot parse"),
            "a <-> read must not be rejected by the port classifier ({status}): {knn} — {text}"
        );
    }
}

/// HIGH-2 (round 2): `EXPLAIN (ANALYZE) <DML>` executes within the CALLER's
/// tenant — tenant `acme`'s analyzed DELETE cannot touch tenant `default`'s
/// rows in the same table name (pre-fix the analyzed write ran unscoped).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_analyze_dml_is_tenant_scoped_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let table = server.unique_name("spine6_xt");
    let (http, base) = server.rest_client().expect("rest client");

    // The VICTIM table lives in the DEFAULT tenant (the namespace unscoped
    // resolution sees — exactly the pre-fix exploit's target); the ATTACKER
    // is tenant acme.
    // /api/v2/sql is single-statement: seed with two requests.
    for seed in [
        format!("CREATE TABLE {table} (id TEXT)"),
        format!("INSERT INTO {table} (id) VALUES ('victim-row')"),
    ] {
        let victim_seed = post_tenant_json(
            &http,
            format!("{base}/api/v2/sql"),
            "default",
            json!({ "query": seed }),
        )
        .await
        .expect("victim seed sends");
        let seed_status = victim_seed.status();
        assert_eq!(
            seed_status,
            reqwest::StatusCode::OK,
            "victim seed {seed}: {}",
            victim_seed.text().await.unwrap_or_default()
        );
    }

    // The ATTACK: tenant acme runs an ANALYZEd DELETE against the victim's
    // table. Pre-fix the analyzed write executed UNSCOPED and deleted the
    // victim's row; post-fix the execution is scoped to acme, which has no
    // such table — the write fails WITHOUT touching the victim's data. (The
    // EXPLAIN planning itself resolving the default-tenant table unscoped is
    // pre-existing schema-existence visibility, unchanged by this PR.)
    let attack = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "acme",
        json!({ "query": format!("EXPLAIN (ANALYZE, VERBOSE) DELETE FROM {table} WHERE id = 'victim-row'") }),
    )
    .await
    .expect("attack sends");
    let attack_status = attack.status();

    // The victim's row must SURVIVE the attack regardless of the attack's
    // own outcome.
    let victim_after = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "default",
        json!({ "query": format!("SELECT id FROM {table}") }),
    )
    .await
    .expect("victim read sends");
    let body: serde_json::Value = victim_after.json().await.expect("victim read body");
    let ids: Vec<String> = body["rows"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| row["id"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        ids,
        vec!["victim-row".to_string()],
        "acme's analyzed DELETE must not touch the default tenant's rows (attack status {attack_status}): {body}"
    );

    // Positive scoped arm: the DEFAULT tenant's own analyzed DELETE executes
    // (planning resolves for it) and deletes ITS OWN row — the write path is
    // alive, only the CROSS-tenant reach is closed.
    let own = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "default",
        json!({ "query": format!("EXPLAIN (ANALYZE) DELETE FROM {table} WHERE id = 'victim-row'") }),
    )
    .await
    .expect("own analyzed delete sends");
    assert_eq!(
        own.status(),
        reqwest::StatusCode::OK,
        "the tenant's own analyzed delete must execute: {}",
        own.text().await.unwrap_or_default()
    );
    let after = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "default",
        json!({ "query": format!("SELECT id FROM {table}") }),
    )
    .await
    .expect("final read sends");
    let after_body: serde_json::Value = after.json().await.expect("final read body");
    assert_eq!(
        after_body["rows"].as_array().map(Vec::len),
        Some(0),
        "the tenant's own row must be deleted: {after_body}"
    );
}

/// Connect a second pgwire client whose startup `database` (= tenant /
/// catalog, TD-064) is `tenant`.
async fn connect_as_tenant(server: &TestPgServer, tenant: &str) -> tokio_postgres::Client {
    let conn_str = server
        .conn_str()
        .replace("dbname=proximadb", &format!("dbname={tenant}"));
    let (client, connection) = tokio_postgres::connect(&conn_str, tokio_postgres::NoTls)
        .await
        .expect("tenant pgwire connect");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

/// The server SQLSTATE of a statement, or `"OK"` when it succeeded.
async fn sqlstate_or_ok(client: &tokio_postgres::Client, sql: &str) -> String {
    match client.simple_query(sql).await {
        Ok(_) => "OK".to_string(),
        Err(error) => error
            .as_db_error()
            .map(|db| db.code().code().to_string())
            .unwrap_or_else(|| format!("<no db error: {error}>")),
    }
}

/// The server SQLSTATE of a statement that must fail.
async fn sqlstate_of(client: &tokio_postgres::Client, sql: &str) -> String {
    match client.simple_query(sql).await {
        Ok(_) => panic!("must fail: {sql}"),
        Err(error) => error
            .as_db_error()
            .map(|db| db.code().code().to_string())
            .unwrap_or_else(|| format!("<no db error: {error}>")),
    }
}

/// TD-SPINE-6 round 3 (MED-1 / L1 / L2 / L7) over the LIVE pgwire surface:
/// * `EXPLAIN (ANALYZE off|false|0) INSERT` and `(VERBOSE, ANALYZE off,
///   COSTS) INSERT` PLAN only — pgwire's former substring parser executed
///   them;
/// * `(VERBOSE ANALYZE)` / an invalid ANALYZE value is a syntax error (42601),
///   never "execute";
/// * `(ANALYZE) INSERT` executes into the CONNECTION tenant's partition
///   (visible to that tenant over REST; HIGH-2);
/// * another tenant's connection cannot plan or execute against the table
///   (42P01 — no existence leak, L7 + the tenant-visibility gate, L2);
/// * inside a READ ONLY transaction the analyzed write is 25006 (L2).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_explain_analyze_options_gates_and_tenant_scoping_e2e() {
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");
    let table = server.unique_name("spine6_pgx");
    pg_server::rows(&client, &format!("CREATE TABLE {table} (id TEXT)")).await;
    pg_server::rows(
        &client,
        &format!("INSERT INTO {table} (id) VALUES ('keep')"),
    )
    .await;
    let select = format!("SELECT id FROM {table} ORDER BY id");
    let keep = vec![vec!["keep".to_string()]];

    // Plan-only shapes: the EXPLAIN succeeds and writes nothing.
    for plan_only in [
        format!("EXPLAIN (ANALYZE off) INSERT INTO {table} (id) VALUES ('off')"),
        format!("EXPLAIN (ANALYZE false) INSERT INTO {table} (id) VALUES ('false')"),
        format!("EXPLAIN (ANALYZE 0) INSERT INTO {table} (id) VALUES ('zero')"),
        format!("EXPLAIN (VERBOSE, ANALYZE off, COSTS) INSERT INTO {table} (id) VALUES ('mix')"),
    ] {
        let plan = pg_server::rows(&client, &plan_only).await;
        assert_eq!(plan.len(), 1, "a plan row for: {plan_only}");
        assert_eq!(
            pg_server::rows(&client, &select).await,
            keep,
            "plan-only EXPLAIN must not write: {plan_only}"
        );
    }

    // Malformed option lists are syntax errors, never executed.
    for malformed in [
        format!("EXPLAIN (VERBOSE ANALYZE) INSERT INTO {table} (id) VALUES ('bad1')"),
        format!("EXPLAIN (ANALYZE maybe) INSERT INTO {table} (id) VALUES ('bad2')"),
    ] {
        assert_eq!(
            sqlstate_of(&client, &malformed).await,
            "42601",
            "{malformed}"
        );
        assert_eq!(pg_server::rows(&client, &select).await, keep, "{malformed}");
    }

    // Transaction gates (L2, round 4 negative controls): a READ ONLY
    // transaction rejects the analyzed write with 25006, and any other open
    // transaction with 0A000 (it would execute immediately, outside the
    // transaction's buffered writes). Both outcomes are collected before
    // asserting so a gate regression reports BOTH arms.
    pg_server::rows(&client, "BEGIN READ ONLY").await;
    let read_only = sqlstate_or_ok(
        &client,
        &format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('ro')"),
    )
    .await;
    let _ = client.simple_query("ROLLBACK").await;
    pg_server::rows(&client, "BEGIN").await;
    let open_tx = sqlstate_or_ok(
        &client,
        &format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('tx')"),
    )
    .await;
    let _ = client.simple_query("ROLLBACK").await;
    assert_eq!(
        (read_only.as_str(), open_tx.as_str()),
        ("25006", "0A000"),
        "transaction gates for an analyzed write (READ ONLY, open transaction)"
    );
    assert_eq!(pg_server::rows(&client, &select).await, keep);

    // Another tenant's connection: the table does not exist for it — neither
    // the plan-only EXPLAIN nor the analyzed write reaches it.
    let other = connect_as_tenant(&server, "spine6_other_tenant").await;
    for cross in [
        format!("EXPLAIN INSERT INTO {table} (id) VALUES ('x-plan')"),
        format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('x-run')"),
    ] {
        assert_eq!(sqlstate_of(&other, &cross).await, "42P01", "{cross}");
    }
    assert_eq!(pg_server::rows(&client, &select).await, keep);

    // The executing form writes into the CONNECTION tenant's partition.
    let plan = pg_server::rows(
        &client,
        &format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('on')"),
    )
    .await;
    assert_eq!(plan.len(), 1, "EXPLAIN ANALYZE returns its plan row");
    assert_eq!(
        pg_server::rows(&client, &select).await,
        vec![vec!["keep".to_string()], vec!["on".to_string()]],
        "EXPLAIN (ANALYZE) INSERT executes"
    );
    // …and the tenant's REST view (X-Tenant-ID = the pgwire database) sees
    // it: the write landed in the connection tenant's partition, not the
    // unscoped one.
    let (http, base) = server.rest_client().expect("rest client");
    let response = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "proximadb",
        json!({ "query": select }),
    )
    .await
    .expect("rest read sends");
    let body: serde_json::Value = response.json().await.expect("rest body");
    let ids: Vec<String> = body["rows"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| row["id"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(ids, vec!["keep", "on"], "tenant REST view: {body}");
}

/// OLAP result-cache invalidations so far (the cache flag is set in-process).
fn olap_cache_invalidations() -> u64 {
    proximadb::network::postgres::relational_pipeline::olap_result_cache()
        .expect("cache flag is set in this process")
        .stats()
        .invalidations
}

/// Prime the OLAP result cache for `select` through `/api/v2/sql` (a miss +
/// insert, then a consulting re-read) and return the rows' ids.
async fn primed_rest_ids(http: &reqwest::Client, base: &str, select: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for _ in 0..2 {
        let body: serde_json::Value = post_tenant_json(
            http,
            format!("{base}/api/v2/sql"),
            "proximadb",
            json!({ "query": select }),
        )
        .await
        .expect("read sends")
        .json()
        .await
        .expect("read body");
        ids = body["rows"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .map(|row| row["id"].as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
    }
    ids
}

/// Round 4 (LOW-1), mandate #16b: a committed pgwire `EXPLAIN (ANALYZE)`
/// write invalidates the tenant's OLAP result-cache entries for the table
/// (the counter moves), and the next read sees the new row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_analyzed_write_invalidates_the_olap_result_cache_e2e() {
    // Edition 2024: set_var is unsafe (single-threaded test prologue).
    unsafe { std::env::set_var("PROXIMADB_QUERY_RESULT_CACHE", "1") };
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");
    let table = server.unique_name("spine6_inv_pg");
    pg_server::rows(&client, &format!("CREATE TABLE {table} (id TEXT)")).await;
    pg_server::rows(&client, &format!("INSERT INTO {table} (id) VALUES ('a')")).await;
    let select = format!("SELECT id FROM {table} ORDER BY id");
    let (http, base) = server.rest_client().expect("rest client");
    assert_eq!(primed_rest_ids(&http, &base, &select).await, vec!["a"]);

    let before = olap_cache_invalidations();
    pg_server::rows(
        &client,
        &format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('b')"),
    )
    .await;
    assert!(
        olap_cache_invalidations() > before,
        "the pgwire analyzed write must invalidate the cached read (mandate #16b)"
    );
    assert_eq!(primed_rest_ids(&http, &base, &select).await, vec!["a", "b"]);
}

/// Round 4 (LOW-1): the same for the SQL authority's analyzed write
/// (`/api/v2/sql` → `QueryFacadeAdapter`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authority_analyzed_write_invalidates_the_olap_result_cache_e2e() {
    // Edition 2024: set_var is unsafe (single-threaded test prologue).
    unsafe { std::env::set_var("PROXIMADB_QUERY_RESULT_CACHE", "1") };
    let server = TestPgServer::start().await.expect("server start");
    let client = server.connect().await.expect("pgwire connect");
    let table = server.unique_name("spine6_inv_rest");
    pg_server::rows(&client, &format!("CREATE TABLE {table} (id TEXT)")).await;
    pg_server::rows(&client, &format!("INSERT INTO {table} (id) VALUES ('a')")).await;
    let select = format!("SELECT id FROM {table} ORDER BY id");
    let (http, base) = server.rest_client().expect("rest client");
    assert_eq!(primed_rest_ids(&http, &base, &select).await, vec!["a"]);

    let before = olap_cache_invalidations();
    let response = post_tenant_json(
        &http,
        format!("{base}/api/v2/sql"),
        "proximadb",
        json!({ "query": format!("EXPLAIN (ANALYZE) INSERT INTO {table} (id) VALUES ('b')") }),
    )
    .await
    .expect("analyzed write sends");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert!(
        olap_cache_invalidations() > before,
        "the authority's analyzed write must invalidate the cached read (mandate #16b)"
    );
    assert_eq!(primed_rest_ids(&http, &base, &select).await, vec!["a", "b"]);
}
