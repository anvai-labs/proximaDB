// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Cross-modal table-function reachability over pgwire — e2e.
//!
//! Proves the routing fix: a `SELECT … FROM <materialized parquet table>
//! JOIN vector_search(…) / timeseries_range(…)` executes through the DataFusion
//! OLAP route over a real pgwire session, instead of declining to the legacy
//! single-table path (which reported "Column '…' does not exist").
//!
//! Two blockers were fixed: (1) `collect_from_table_factor` treated a
//! table-valued function `name(args)` as a catalog table, so catalog resolution
//! declined the query AND the parquet-backed route check failed; skipping
//! args-bearing factors routes it to DataFusion. (2) the shared relational
//! frontend now cleanly declines a table-function so the existing DataFusion
//! `ctx.sql` fallback (where the UDTFs are registered) resolves it.
//!
//! Gated on `datafusion-integration`: the UDTFs + the OLAP route only exist
//! there. Run:
//!   cargo test --features datafusion-integration --test pgwire_crossmodal_join_e2e -- --nocapture
#![cfg(feature = "datafusion-integration")]

use std::time::Duration;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
/// A connection string whose `dbname` is the tenant/catalog boundary
/// (`pgwire_resolve_read_tenant` → `catalog_tenant()` = the startup database). Used to read
/// under the SAME tenant a REST write used (TD-XMODAL-6).
fn conn_str_db(server: &TestPgServer, db: &str) -> String {
    format!(
        "host=127.0.0.1 port={} user=postgres dbname={db} sslmode=disable",
        server.pg_port
    )
}

async fn connect(server: &TestPgServer) -> tokio_postgres::Client {
    connect_db(server, "proximadb").await
}

async fn connect_db(server: &TestPgServer, db: &str) -> tokio_postgres::Client {
    let (client, conn) =
        tokio_postgres::connect(conn_str_db(server, db).as_str(), tokio_postgres::NoTls)
            .await
            .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// Run a query expected to SUCCEED (no error), returning the first cell as i64.
async fn count(client: &tokio_postgres::Client, sql: &str) -> i64 {
    let msgs = client.simple_query(sql).await.unwrap_or_else(|e| {
        let detail = e
            .as_db_error()
            .map(|d| format!("[{}] {}", d.code().code(), d.message()))
            .unwrap_or_else(|| e.to_string());
        panic!("cross-modal query declined (reachability regression): `{sql}` -> {detail}");
    });
    msgs.into_iter()
        .find_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                r.get(0).and_then(|s| s.parse::<i64>().ok())
            }
            _ => None,
        })
        .unwrap_or(i64::MIN)
}

/// A materialized (parquet-backed) table JOINed with the `vector_search` and
/// `timeseries_range` table functions PLANS + EXECUTES through the DataFusion
/// OLAP route over pgwire — it does not decline to the legacy path. (Row
/// visibility of live-written UDTF data is covered separately; here the fix is
/// that the query is *reachable* and returns a well-formed result rather than a
/// "column does not exist" decline.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crossmodal_udtf_join_reaches_datafusion_over_pgwire() {
    let server = TestPgServer::start().await.expect("server start");
    let client = connect(&server).await;

    // Unique per run — the native catalog persists table DDL outside the per-test
    // tempdir, so a fixed name collides across runs.
    let wh = format!("wh_{}", server.pg_port);
    client
        .simple_query(&format!("DROP TABLE IF EXISTS {wh}"))
        .await
        .ok();
    client
        .simple_query(&format!("CREATE TABLE {wh} (id TEXT, tier TEXT)"))
        .await
        .expect("create");
    client
        .simple_query(&format!(
            "INSERT INTO {wh} VALUES ('a','private'),('b','retail'),('z','private')"
        ))
        .await
        .expect("insert");
    client
        .simple_query(&format!("ALTER TABLE {wh} MATERIALIZE"))
        .await
        .expect("materialize");

    // Baseline: a plain aggregate on the materialized table succeeds (routes OLAP).
    assert_eq!(
        count(&client, &format!("SELECT count(*) FROM {wh}")).await,
        3
    );

    // The cross-modal join must REACH the DataFusion UDTF, not decline to the legacy
    // single-table path. Before the fix it declined with "Column 'score' does not exist
    // in table '{wh}'". After the fix the query routes to DataFusion, resolves the
    // `vector_search` UDTF, and invokes it — which for a missing collection surfaces a
    // UDTF-INTERNAL error ("vector search: Collection … not found"). That internal error
    // is itself the proof the UDTF was reached; only the "does not exist in table" decline
    // is a reachability regression.
    let vjoin = format!(
        "SELECT count(*) FROM {wh} d JOIN vector_search('sig','[0.1,0.2]',5) v ON d.id = v.id"
    );
    match client.simple_query(&vjoin).await {
        Ok(_) => {} // planned + executed (empty/rows) — reachable
        Err(e) => {
            let msg = e
                .as_db_error()
                .map(|d| d.message().to_string())
                .unwrap_or_else(|| e.to_string());
            assert!(
                !msg.contains("does not exist in table"),
                "vector_search declined to legacy (reachability regression): {msg}"
            );
            assert!(
                msg.contains("vector search") || msg.to_lowercase().contains("collection"),
                "expected a UDTF-internal error proving the UDTF was reached, got: {msg}"
            );
        }
    }

    // `timeseries_range` is graceful on a missing collection (empty), so the join executes
    // cleanly and returns a well-formed count — the end-to-end proof of the DataFusion
    // route + frontend decline → `ctx.sql` fallback → UDTF resolution.
    let tsrange = count(
        &client,
        &format!("SELECT count(*) FROM {wh} d CROSS JOIN timeseries_range('ts', 0, 9999999) t"),
    )
    .await;
    assert!(
        tsrange >= 0,
        "timeseries_range join must execute, got {tsrange}"
    );
}

/// TD-XMODAL-6 (row visibility): data written through the REST v2 timeseries API under a tenant
/// is READABLE by the `timeseries_range` UDTF over pgwire when the connection resolves to the
/// SAME tenant (`dbname` = the REST `X-Tenant-ID`). Proves the tenant is threaded pgwire →
/// `QueryExecutionContext` → `build_session_context` → the UDTF, which folds `{tenant}::{collection}`
/// exactly like the REST write path's `effective_collection`. Before the fix the UDTF used the raw
/// (unscoped) name and returned 0 rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crossmodal_timeseries_join_returns_rest_seeded_rows() {
    let server = TestPgServer::start().await.expect("server start");
    // Unique per run: the tenant (pgwire catalog + REST header) and the collection/table names,
    // so parallel runs and the persisted native catalog never collide.
    let tenant = format!("xmt{}", server.pg_port);
    let coll = format!("sensor_{}", server.pg_port);
    let wh = format!("wh_{}", server.pg_port);

    // ── Seed the timeseries collection + points via REST under an explicit tenant ──
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let create = http
        .post(format!("{base}/api/v2/timeseries/collections"))
        .header("X-Tenant-ID", &tenant)
        .json(&serde_json::json!({ "name": coll, "value_columns": [{ "name": "amount" }] }))
        .send()
        .await
        .expect("create req");
    assert!(
        create.status().is_success(),
        "create timeseries collection failed: {}",
        create.status()
    );
    let ingest = http
        .post(format!(
            "{base}/api/v2/timeseries/collections/{coll}/ingest"
        ))
        .header("X-Tenant-ID", &tenant)
        .json(&serde_json::json!({
            "points": [
                { "timestamp": 1_000, "values": { "amount": 100.0 } },
                { "timestamp": 2_000, "values": { "amount": 9_000.0 } },
            ]
        }))
        .send()
        .await
        .expect("ingest req");
    assert!(
        ingest.status().is_success(),
        "ingest timeseries failed: {}",
        ingest.status()
    );

    // ── Read it back over pgwire under the SAME tenant (dbname == X-Tenant-ID) ──
    let client = connect_db(&server, &tenant).await;
    client
        .simple_query(&format!("DROP TABLE IF EXISTS {wh}"))
        .await
        .ok();
    client
        .simple_query(&format!("CREATE TABLE {wh} (id TEXT)"))
        .await
        .expect("create wh");
    client
        .simple_query(&format!("INSERT INTO {wh} VALUES ('anchor')"))
        .await
        .expect("insert wh");
    client
        .simple_query(&format!("ALTER TABLE {wh} MATERIALIZE"))
        .await
        .expect("materialize wh");

    // The 1-row anchor CROSS JOIN the 2 seeded points ⇒ 2 rows. If the tenant fold were missing,
    // the UDTF would read the unscoped `sensor_*` partition (empty) ⇒ 0 rows.
    let joined = count(
        &client,
        &format!("SELECT count(*) FROM {wh} d CROSS JOIN timeseries_range('{coll}', 0, 9999999) t"),
    )
    .await;
    assert_eq!(
        joined, 2,
        "expected the 2 REST-seeded points visible via timeseries_range (tenant fold), got {joined}"
    );

    // Value correctness: only the 9000 point exceeds 5000 ⇒ exactly 1 row — the real values flow.
    let high = count(
        &client,
        &format!(
            "SELECT count(*) FROM {wh} d CROSS JOIN timeseries_range('{coll}', 0, 9999999) t WHERE t.value > 5000"
        ),
    )
    .await;
    assert_eq!(high, 1, "expected exactly the 9000-value point, got {high}");
}
