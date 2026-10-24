// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-097 P3.1 PR 8: one query over a Parquet-backed table AND a native one.
//!
//! `dim` is materialized (`ALTER TABLE … MATERIALIZE` — a cold Parquet base
//! plus the ADR-025 post-snapshot delta) and `facts` stays native. A query
//! joining them used to fit neither DataFusion arm — the Parquet arm needs
//! every table Parquet-backed, the native arm none — and fell back to Volcano,
//! which declines a window function. With `PROXIMADB_NATIVE_TABLE_PROVIDER` on
//! the DataFusion floor registers both, and the join, the window and the
//! post-snapshot update to `dim` are all answered by value. With the gate off
//! the query still declines (the Volcano path, unchanged).
#![cfg(feature = "datafusion-integration")]

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
use tokio_postgres::SimpleQueryMessage;

fn rows(messages: &[SimpleQueryMessage]) -> Vec<Vec<String>> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).unwrap_or("NULL").trim().to_string())
                    .collect(),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_query_mixing_parquet_and_native_tables_runs_on_datafusion() {
    // SAFETY: this test is its own process; every env change happens between
    // awaited queries, with no request in flight.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };
    let server = TestPgServer::start().await.expect("server");
    let client = server.connect().await.expect("connect");
    let run = |sql: &'static str| {
        let client = &client;
        async move {
            client
                .simple_query(sql)
                .await
                .unwrap_or_else(|e| panic!("`{sql}`: {}", pg_server::db_err(&e)))
        }
    };

    run("CREATE TABLE dim (id INT PRIMARY KEY, label TEXT)").await;
    run("INSERT INTO dim (id, label) VALUES (1, 'a'), (2, 'b'), (3, 'c')").await;
    run("ALTER TABLE dim MATERIALIZE").await;
    // After the snapshot: the delta merge must show 'B', not the Parquet 'b'.
    run("UPDATE dim SET label = 'B' WHERE id = 2").await;

    run("CREATE TABLE facts (id INT PRIMARY KEY, dim_id INT, v INT)").await;
    run("INSERT INTO facts (id, dim_id, v) VALUES (1, 1, 10), (2, 2, 20), (3, 2, 30), (4, 3, 40), (5, 1, 50)")
        .await;

    let mixed = "select f.id, d.label, \
                 sum(f.v) over (partition by d.label order by f.id) as running \
                 from facts f join dim d on f.dim_id = d.id order by f.id";

    // Gate off: the mixed query is not DataFusion's, and the Volcano path
    // declines the window function rather than answer it wrongly.
    let err = client
        .simple_query(mixed)
        .await
        .expect_err("gate off: the window over a mixed join declines");
    let message = pg_server::db_err(&err);
    assert!(
        message.contains("0A000")
            && message.contains("window function sum")
            && message.contains("PROXIMADB_NATIVE_TABLE_PROVIDER"),
        "the window decline, naming the gate: {message}"
    );

    // SAFETY: see above.
    unsafe { std::env::set_var("PROXIMADB_NATIVE_TABLE_PROVIDER", "1") };
    let got =
        rows(&client.simple_query(mixed).await.unwrap_or_else(|e| {
            panic!("gate on: the mixed query serves: {}", pg_server::db_err(&e))
        }));
    let want: Vec<Vec<String>> = [
        ["1", "a", "10"],
        ["2", "B", "20"],
        ["3", "B", "50"],
        ["4", "c", "40"],
        ["5", "a", "60"],
    ]
    .iter()
    .map(|r| r.iter().map(|c| c.to_string()).collect())
    .collect();
    assert_eq!(
        got, want,
        "join + window over Parquet ⋈ native, delta merged"
    );

    // A plain aggregate join too.
    let aggregate = "select d.label, count(*) as n, sum(f.v) as s from facts f \
                     join dim d on f.dim_id = d.id group by d.label order by d.label";
    let aggregate_want = [["B", "2", "50"], ["a", "2", "60"], ["c", "1", "40"]];
    let got = rows(
        &client
            .simple_query(aggregate)
            .await
            .unwrap_or_else(|e| panic!("aggregate join: {}", pg_server::db_err(&e))),
    );
    assert_eq!(got, aggregate_want);

    // A native table over the row cap with no streaming provider (Layer A off)
    // cannot be registered: the mixed aggregate falls back to Volcano and is
    // still answered, never failed or truncated.
    // SAFETY: see above.
    unsafe {
        std::env::set_var("PROXIMADB_NATIVE_TABLE_LAYER_A", "0");
        std::env::set_var("PROXIMADB_NATIVE_TABLE_ROW_CAP", "2");
    }
    let got = rows(
        &client
            .simple_query(aggregate)
            .await
            .unwrap_or_else(|e| panic!("over the row cap: {}", pg_server::db_err(&e))),
    );
    assert_eq!(
        got, aggregate_want,
        "over the row cap the mixed query is still answered, exactly"
    );
    // And it is Volcano that answered: the window over the same join declines
    // there, while DataFusion would have served it.
    let err = client
        .simple_query(mixed)
        .await
        .expect_err("over the row cap: Volcano, which declines the window");
    assert!(
        pg_server::db_err(&err).contains("0A000"),
        "{}",
        pg_server::db_err(&err)
    );

    drop(client);
    server.shutdown().await;
    // SAFETY: see above.
    unsafe {
        std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER");
        std::env::remove_var("PROXIMADB_NATIVE_TABLE_LAYER_A");
        std::env::remove_var("PROXIMADB_NATIVE_TABLE_ROW_CAP");
    }
}
