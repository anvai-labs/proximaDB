// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-GRAPH-PGWIRE-1: `SELECT * FROM <graph collection>` over pgwire used to
//! fabricate a `(table_name, 0, 0)` row regardless of whether the collection
//! existed or had data. It now declines honestly (SQLSTATE `0A000`) instead
//! of returning a silent wrong answer — proven here for both a real,
//! successfully-created graph collection and a name that was never created.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
fn explain_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

#[test]
fn graph_select_over_pgwire_declines_honestly() {
    std::thread::Builder::new()
        .name("graph-select-pgwire-decline-e2e".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt")
                .block_on(body())
        })
        .expect("spawn")
        .join()
        .expect("panic");
}

async fn body() {
    let server = TestPgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // A real graph collection, created successfully…
    client
        .simple_query("CREATE TABLE graph_social (id TEXT) USING GRAPH")
        .await
        .expect("create graph table");

    // …a plain SELECT against it must decline (SQLSTATE 0A000), never return
    // the old fabricated (name, 0, 0) row.
    let err = client
        .simple_query("SELECT * FROM graph_social")
        .await
        .expect_err("SELECT over a graph collection must be declined, not served");
    let msg = explain_err(&err);
    assert!(msg.starts_with("[0A000]"), "unexpected error: {msg}");
    assert!(
        msg.contains("not supported over pgwire"),
        "decline message should explain WHY: {msg}"
    );
    assert!(
        msg.contains("ProximaGraphService") || msg.contains("/api/v2/graphs"),
        "decline message should point at the real graph surface: {msg}"
    );

    // A graph-shaped name that was NEVER created declines identically — the
    // SELECT-side classifier is syntactic (table-name prefix), so this must
    // NOT differ from the "exists" case, and must NOT silently create
    // anything (unlike a naive `GraphService::get_stats` call would).
    let err2 = client
        .simple_query("SELECT * FROM graph_never_created_xyz")
        .await
        .expect_err("SELECT over a nonexistent graph-shaped table must also decline");
    let msg2 = explain_err(&err2);
    assert!(msg2.starts_with("[0A000]"), "unexpected error: {msg2}");

    // Confirm no phantom collection was silently provisioned as a side
    // effect of the declined SELECT: DROP must fail "does not exist", not
    // succeed against something the SELECT itself created.
    let drop_err = client
        .simple_query("DROP TABLE graph_never_created_xyz")
        .await
        .expect_err("a declined SELECT must not have created a phantom collection");
    let drop_msg = explain_err(&drop_err);
    assert!(
        drop_msg.contains("does not exist") || drop_msg.contains("42P01"),
        "unexpected DROP outcome (phantom collection?): {drop_msg}"
    );
}
