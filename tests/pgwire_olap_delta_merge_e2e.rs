// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-025 OLAP read-merge (relational cold path) — pgwire e2e.
//!
//! Proves the central correctness property: after `ALTER TABLE … MATERIALIZE`
//! freezes a cold Parquet snapshot, subsequent `DELETE` / `UPDATE` / `INSERT`
//! over pgwire are reflected in OLAP `SELECT`s that route to the DataFusion
//! engine — and that the behavior is genuinely gated (the
//! `PROXIMADB_OLAP_DELTA_MERGE=0` kill-switch serves the stale snapshot; the
//! default-ON merge reconciles it).
//!
//! Gated on `datafusion-integration`: the read-merge lives on the DataFusion
//! OLAP route, and the gate-OFF assertion (stale snapshot) is only meaningful
//! there — without the feature, parquet-backed tables route to the native
//! engine, which always reads live record state. Run with:
//!   cargo test --features datafusion-integration --test pgwire_olap_delta_merge_e2e -- --nocapture
#![cfg(feature = "datafusion-integration")]

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
/// Run a 2-column query and parse the rows into `(i64, i64)` pairs (pgwire simple
/// query renders every cell as text).
async fn pairs(client: &tokio_postgres::Client, sql: &str) -> Vec<(i64, i64)> {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some((
                r.get(0).unwrap_or_default().parse().unwrap_or(i64::MIN),
                r.get(1).unwrap_or_default().parse().unwrap_or(i64::MIN),
            )),
            _ => None,
        })
        .collect()
}

/// Run a single-cell aggregate query and parse the first column as `i64`.
async fn one_i64(client: &tokio_postgres::Client, sql: &str) -> i64 {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)))
        .into_iter()
        .find_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                r.get(0).and_then(|s| s.parse::<i64>().ok())
            }
            _ => None,
        })
        .unwrap_or(i64::MIN)
}

/// Connect a client scoped to a tenant (startup `dbname` = tenant/catalog).
/// Connection string scoped to a tenant (the startup `dbname` = tenant/catalog).
fn conn_str_for(server: &TestPgServer, dbname: &str) -> String {
    format!(
        "host=127.0.0.1 port={} user=postgres dbname={dbname} sslmode=disable",
        server.pg_port
    )
}

async fn connect(server: &TestPgServer, dbname: &str) -> tokio_postgres::Client {
    let (client, conn) =
        tokio_postgres::connect(conn_str_for(server, dbname).as_str(), tokio_postgres::NoTls)
            .await
            .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

/// A3 + A4: after MATERIALIZE, a `DELETE`/`UPDATE`/`INSERT` is invisible to the
/// DataFusion OLAP route with the merge OFF (stale snapshot), and fully reflected
/// with it ON — keyed by canonical oid, with `COUNT(*)` taking the merged
/// cardinality (not the Parquet footer count).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn olap_delta_merge_gate_off_stale_then_on_corrects() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // Schema: single-column INT PK (oid = PK text) + a value column.
    client.simple_query("DROP TABLE IF EXISTS dvm").await.ok();
    client
        .simple_query("CREATE TABLE dvm (id INT PRIMARY KEY, v INT)")
        .await
        .unwrap_or_else(|e| panic!("create: {}", pg_server::db_err(&e)));
    for sql in [
        "INSERT INTO dvm (id, v) VALUES (1, 10)",
        "INSERT INTO dvm (id, v) VALUES (2, 20)",
        "INSERT INTO dvm (id, v) VALUES (3, 30)",
    ] {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("insert `{sql}`: {}", pg_server::db_err(&e)));
    }

    // Freeze the cold Parquet snapshot at this LSN, then route SELECTs to OLAP.
    client
        .simple_query("ALTER TABLE dvm MATERIALIZE")
        .await
        .unwrap_or_else(|e| panic!("materialize: {}", pg_server::db_err(&e)));

    // Post-snapshot mutations: delete a row, update another, insert a new one.
    client
        .simple_query("DELETE FROM dvm WHERE id = 2")
        .await
        .unwrap_or_else(|e| panic!("delete: {}", pg_server::db_err(&e)));
    client
        .simple_query("UPDATE dvm SET v = 99 WHERE id = 3")
        .await
        .unwrap_or_else(|e| panic!("update: {}", pg_server::db_err(&e)));
    client
        .simple_query("INSERT INTO dvm (id, v) VALUES (4, 40)")
        .await
        .unwrap_or_else(|e| panic!("insert4: {}", pg_server::db_err(&e)));

    // GROUP BY / aggregate engages the relational engine → DataFusion over the
    // materialized Parquet base (never the legacy single-table path).
    let q_rows = "SELECT id, v FROM dvm GROUP BY id, v ORDER BY id";
    let q_count = "SELECT COUNT(*) AS c FROM dvm";

    // Gate OFF (kill-switch): the merge is default-ON, so an explicit falsy value
    // forces the legacy bare-Parquet read and the DataFusion route serves the
    // STALE snapshot — proving the kill-switch preserves legacy behavior.
    // SAFETY: toggled only between awaited queries — no request is in flight, so
    // no server thread reads this env var concurrently (Rust 2024 marks env
    // mutation unsafe for the general concurrent case). Tests run under nextest
    // process isolation (CLAUDE.md §11).
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "0") };
    assert_eq!(
        pairs(&client, q_rows).await,
        vec![(1, 10), (2, 20), (3, 30)],
        "kill-switch (=0) must serve the stale materialized snapshot"
    );

    // Gate ON: the read-merge reconciles delete(2) / update(3→99) / insert(4).
    // SAFETY: see the note above — no in-flight request, nextest-isolated.
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "1") };
    assert_eq!(
        pairs(&client, q_rows).await,
        vec![(1, 10), (3, 99), (4, 40)],
        "gate-on must reflect post-snapshot delete/update/insert keyed by oid"
    );
    assert_eq!(
        one_i64(&client, q_count).await,
        3,
        "COUNT(*) must be the merged cardinality, not the Parquet footer count"
    );

    // Default-ON (ADR-025 PR3, the headline of this flip): with the env var UNSET
    // the merge is on by default — removing it must STILL reconcile the snapshot,
    // not revert to the stale base. This also leaves the var cleared for cleanup.
    // SAFETY: toggled only between awaited queries — no request is in flight, so
    // no server thread reads this env var concurrently (Rust 2024 marks env
    // mutation unsafe for the general concurrent case). Tests run under nextest
    // process isolation (CLAUDE.md §11).
    unsafe { std::env::remove_var("PROXIMADB_OLAP_DELTA_MERGE") };
    assert_eq!(
        pairs(&client, q_rows).await,
        vec![(1, 10), (3, 99), (4, 40)],
        "default-ON (env unset) must reconcile post-snapshot delete/update/insert"
    );
}

/// Tenant isolation of the read-merge (and the empty-delta fast path). Two
/// tenants share a table name; tenant A mutates after MATERIALIZE, tenant B does
/// not. A's OLAP read reflects A's delete/update; B's is UNAFFECTED — B's feed is
/// tenant-scoped (empty), so the fast path serves B's pristine snapshot. A cross-
/// tenant feed leak would surface A's delete/update in B's result.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn olap_delta_merge_is_tenant_isolated() {
    // SAFETY: set before the server starts; never mutated while a request is in
    // flight. nextest process isolation (CLAUDE.md §11).
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "1") };
    let server = TestPgServer::start().await.expect("server start");
    let tenant_a = format!("tenant_a_{}", server.pg_port);
    let tenant_b = format!("tenant_b_{}", server.pg_port);
    let a = connect(&server, &tenant_a).await;
    let b = connect(&server, &tenant_b).await;

    // Both tenants: identical table name + data, each materialized to its own
    // tenant-isolated Parquet snapshot.
    for c in [&a, &b] {
        c.simple_query("CREATE TABLE shared (id INT PRIMARY KEY, v INT)")
            .await
            .unwrap_or_else(|e| panic!("create: {}", pg_server::db_err(&e)));
        for sql in [
            "INSERT INTO shared (id, v) VALUES (1, 10)",
            "INSERT INTO shared (id, v) VALUES (2, 20)",
            "INSERT INTO shared (id, v) VALUES (3, 30)",
        ] {
            c.simple_query(sql)
                .await
                .unwrap_or_else(|e| panic!("insert `{sql}`: {}", pg_server::db_err(&e)));
        }
        c.simple_query("ALTER TABLE shared MATERIALIZE")
            .await
            .unwrap_or_else(|e| panic!("materialize: {}", pg_server::db_err(&e)));
    }

    // Only tenant A mutates after its snapshot.
    a.simple_query("DELETE FROM shared WHERE id = 2")
        .await
        .unwrap_or_else(|e| panic!("A delete: {}", pg_server::db_err(&e)));
    a.simple_query("UPDATE shared SET v = 99 WHERE id = 3")
        .await
        .unwrap_or_else(|e| panic!("A update: {}", pg_server::db_err(&e)));

    let q = "SELECT id, v FROM shared GROUP BY id, v ORDER BY id";
    assert_eq!(
        pairs(&a, q).await,
        vec![(1, 10), (3, 99)],
        "tenant A must see its own post-snapshot delete(2)/update(3→99)"
    );
    assert_eq!(
        pairs(&b, q).await,
        vec![(1, 10), (2, 20), (3, 30)],
        "tenant B must NOT see tenant A's changes (feed is tenant-isolated)"
    );

    unsafe { std::env::remove_var("PROXIMADB_OLAP_DELTA_MERGE") };
}

/// ADR-031 O2 (end-to-end, mixed-read-safe): with the merge on, a delete written
/// while WAL object_id-keying is OFF (name-keyed) plus an update + insert written
/// while it is ON (object_id-keyed) must ALL be reconciled — the merge's dual-read
/// unions name-keyed and object_id-keyed entries. pgwire CREATE goes through the
/// native catalog, which allocates the table's object_id, so the update(3)/insert(4)
/// are genuinely object_id-keyed; if dual-read were broken they'd be missed and this
/// fails closed. SAFETY: env toggled only between awaited queries (nextest-isolated).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn olap_delta_merge_reconciles_mixed_name_and_object_id_keyed_wal() {
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "1") };
    unsafe { std::env::remove_var("PROXIMADB_WAL_OBJECT_ID_KEY") };
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .simple_query("DROP TABLE IF EXISTS dvm_oid")
        .await
        .ok();
    client
        .simple_query("CREATE TABLE dvm_oid (id INT PRIMARY KEY, v INT)")
        .await
        .unwrap_or_else(|e| panic!("create: {}", pg_server::db_err(&e)));
    for sql in [
        "INSERT INTO dvm_oid (id, v) VALUES (1, 10)",
        "INSERT INTO dvm_oid (id, v) VALUES (2, 20)",
        "INSERT INTO dvm_oid (id, v) VALUES (3, 30)",
    ] {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("insert `{sql}`: {}", pg_server::db_err(&e)));
    }
    client
        .simple_query("ALTER TABLE dvm_oid MATERIALIZE")
        .await
        .unwrap_or_else(|e| panic!("materialize: {}", pg_server::db_err(&e)));

    // Name-keyed delete (object_id WAL keying OFF).
    client
        .simple_query("DELETE FROM dvm_oid WHERE id = 2")
        .await
        .unwrap_or_else(|e| panic!("delete: {}", pg_server::db_err(&e)));

    // Flip to object_id-keyed WAL for the subsequent writes.
    unsafe { std::env::set_var("PROXIMADB_WAL_OBJECT_ID_KEY", "1") };
    client
        .simple_query("UPDATE dvm_oid SET v = 99 WHERE id = 3")
        .await
        .unwrap_or_else(|e| panic!("update: {}", pg_server::db_err(&e)));
    client
        .simple_query("INSERT INTO dvm_oid (id, v) VALUES (4, 40)")
        .await
        .unwrap_or_else(|e| panic!("insert4: {}", pg_server::db_err(&e)));

    assert_eq!(
        pairs(
            &client,
            "SELECT id, v FROM dvm_oid GROUP BY id, v ORDER BY id"
        )
        .await,
        vec![(1, 10), (3, 99), (4, 40)],
        "dual-read must union name-keyed delete(2) + object_id-keyed update(3)/insert(4)"
    );

    unsafe { std::env::remove_var("PROXIMADB_OLAP_DELTA_MERGE") };
    unsafe { std::env::remove_var("PROXIMADB_WAL_OBJECT_ID_KEY") };
}

/// Run a (text, i64) 2-column query into `(String, i64)` pairs — for a grouped
/// aggregate keyed by a string dimension.
async fn text_i64_pairs(client: &tokio_postgres::Client, sql: &str) -> Vec<(String, i64)> {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", pg_server::db_err(&e)))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some((
                r.get(0).unwrap_or_default().to_string(),
                r.get(1).unwrap_or_default().parse().unwrap_or(i64::MIN),
            )),
            _ => None,
        })
        .collect()
}

/// VALUE-ACCURACY on the analytical path: a multi-table JOIN + GROUP BY + SUM over
/// a fact table carrying a NON-EMPTY post-snapshot delta must return the *merged*
/// aggregate values — not the stale base, and without double-counting the appended
/// rows. The TPC-H/TPC-DS ratchets never mutate after MATERIALIZE, so they only
/// ever exercise the empty-delta fast path; this is the case they do not cover.
/// The merge reconciles rows into the scanned table BEFORE planning, so a correct
/// SUM here proves merged rows flow correctly through JOIN + aggregation (the
/// dimension `acct` is also materialized but unmutated → empty-delta fast path,
/// while `txn` is merged — exercising both registration paths in one query).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn olap_delta_merge_correct_aggregates_through_join() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // Idempotent setup: the catalog can survive across local re-runs, so drop any
    // leftovers first (matching the DROP-IF-EXISTS convention of the sibling tests).
    client.simple_query("DROP TABLE IF EXISTS txn").await.ok();
    client.simple_query("DROP TABLE IF EXISTS acct").await.ok();
    client
        .simple_query("CREATE TABLE acct (id INT PRIMARY KEY, region VARCHAR)")
        .await
        .unwrap_or_else(|e| panic!("create acct: {}", pg_server::db_err(&e)));
    client
        .simple_query("CREATE TABLE txn (id INT PRIMARY KEY, acct_id INT, amount INT)")
        .await
        .unwrap_or_else(|e| panic!("create txn: {}", pg_server::db_err(&e)));
    for sql in [
        "INSERT INTO acct (id, region) VALUES (1, 'US')",
        "INSERT INTO acct (id, region) VALUES (2, 'EU')",
        "INSERT INTO txn (id, acct_id, amount) VALUES (1, 1, 10)",
        "INSERT INTO txn (id, acct_id, amount) VALUES (2, 1, 20)",
        "INSERT INTO txn (id, acct_id, amount) VALUES (3, 2, 30)",
    ] {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("seed `{sql}`: {}", pg_server::db_err(&e)));
    }
    for t in ["acct", "txn"] {
        client
            .simple_query(&format!("ALTER TABLE {t} MATERIALIZE"))
            .await
            .unwrap_or_else(|e| panic!("materialize {t}: {}", pg_server::db_err(&e)));
    }

    // Post-snapshot mutations on the FACT only: delete(2) drops 20 from US,
    // update(3→99) lifts EU from 30 to 99, insert(4=40) adds 40 to US.
    client
        .simple_query("DELETE FROM txn WHERE id = 2")
        .await
        .unwrap_or_else(|e| panic!("delete: {}", pg_server::db_err(&e)));
    client
        .simple_query("UPDATE txn SET amount = 99 WHERE id = 3")
        .await
        .unwrap_or_else(|e| panic!("update: {}", pg_server::db_err(&e)));
    client
        .simple_query("INSERT INTO txn (id, acct_id, amount) VALUES (4, 1, 40)")
        .await
        .unwrap_or_else(|e| panic!("insert: {}", pg_server::db_err(&e)));

    let q = "SELECT a.region, SUM(t.amount) AS s FROM txn t \
             JOIN acct a ON t.acct_id = a.id GROUP BY a.region ORDER BY a.region";

    // Kill-switch OFF → stale base: US = 10 + 20 = 30, EU = 30.
    // SAFETY: env toggled only between awaited queries — no in-flight request;
    // nextest / `--test-threads=1` process/thread isolation (CLAUDE.md §11).
    unsafe { std::env::set_var("PROXIMADB_OLAP_DELTA_MERGE", "0") };
    assert_eq!(
        text_i64_pairs(&client, q).await,
        vec![("EU".to_string(), 30), ("US".to_string(), 30)],
        "kill-switch must serve stale aggregates through the join (US=30, EU=30)"
    );

    // Default-ON (env unset) → merged: US = 10 + 40 = 50 (id2 deleted), EU = 99.
    unsafe { std::env::remove_var("PROXIMADB_OLAP_DELTA_MERGE") };
    assert_eq!(
        text_i64_pairs(&client, q).await,
        vec![("EU".to_string(), 99), ("US".to_string(), 50)],
        "merge must reflect delete/update/insert in JOIN+GROUP BY+SUM (US=50, EU=99)"
    );
}
