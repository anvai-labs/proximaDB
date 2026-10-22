// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-097 P3.1: native SQL over a table whose rows live in Layer A spill
//! segments is served by the streaming provider — no row cap, correct values.
//!
//! The table is spilled for real (`PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT`),
//! updated and deleted across flushes, and is larger than
//! `PROXIMADB_NATIVE_TABLE_ROW_CAP`, so the materialized `MemTable` path would
//! decline it. Window and CTE answers are checked by value. The kill switch
//! (`PROXIMADB_NATIVE_TABLE_LAYER_A=0`) puts the capped path back, proving the
//! streaming provider is what served them.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
use tokio_postgres::SimpleQueryMessage;

const ROWS: i64 = 60;

#[test]
fn layer_a_serves_spilled_tables_without_a_row_cap() {
    std::thread::Builder::new()
        .name("layer-a-spill-sql".into())
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

fn cells(messages: &[SimpleQueryMessage], col: usize) -> Vec<Option<String>> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => Some(r.get(col).map(|s| s.trim().to_string())),
            _ => None,
        })
        .collect()
}

fn spill_segments_under(dir: &std::path::Path) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("spill-"))
            {
                n += 1;
            }
        }
    }
    n
}

/// Create and fill `spill_t` — inserts, then updates and deletes of rows already
/// in segments — returning the live `id -> val` it must hold.
async fn load(client: &tokio_postgres::Client) -> std::collections::BTreeMap<i64, i64> {
    client
        .simple_query("CREATE TABLE spill_t (id BIGINT PRIMARY KEY, val BIGINT)")
        .await
        .expect("create");
    // val = 2 * id, inserted a few rows per statement so flushes interleave.
    let mut expected: std::collections::BTreeMap<i64, i64> = std::collections::BTreeMap::new();
    for chunk in (1..=ROWS).collect::<Vec<_>>().chunks(5) {
        let values: Vec<String> = chunk.iter().map(|i| format!("({i}, {})", 2 * i)).collect();
        client
            .simple_query(&format!(
                "INSERT INTO spill_t (id, val) VALUES {}",
                values.join(", ")
            ))
            .await
            .expect("insert");
        for &i in chunk {
            expected.insert(i, 2 * i);
        }
    }
    // Updates and deletes of rows that are already in segments.
    for id in [3_i64, 17, 42] {
        client
            .simple_query(&format!("UPDATE spill_t SET val = 1000 WHERE id = {id}"))
            .await
            .expect("update");
        expected.insert(id, 1000);
    }
    for id in [5_i64, 6, 59] {
        client
            .simple_query(&format!("DELETE FROM spill_t WHERE id = {id}"))
            .await
            .expect("delete");
        expected.remove(&id);
    }
    expected
}

async fn body() {
    // SAFETY: this test is its own process, and every env change happens while
    // no server is running (servers start after, and are dropped before, each
    // change), so no other thread reads the environment concurrently.
    unsafe {
        std::env::set_var("PROXIMADB_RELATIONAL_SPILL_MAX_RESIDENT", "8");
        std::env::set_var("PROXIMADB_NATIVE_TABLE_PROVIDER", "1");
        std::env::set_var("PROXIMADB_NATIVE_TABLE_ROW_CAP", "10");
        std::env::remove_var("PROXIMADB_NATIVE_TABLE_LAYER_A");
    }

    let server = TestPgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let expected = load(&client).await;
    assert!(
        spill_segments_under(server.data_dir()) >= 2,
        "the fixture must actually spill to segments, or this test proves nothing"
    );
    assert!(expected.len() > 10, "the table must exceed the row cap");

    // A window function: the real previous-row delta, NULL first.
    let window = "select id, val - lag(val) over (order by id) as delta from spill_t order by id";
    let got = client.simple_query(window).await.unwrap_or_else(|e| {
        panic!(
            "window over a spilled native table must serve: {}",
            pg_server::db_err(&e)
        )
    });
    let want: Vec<Option<String>> = std::iter::once(None)
        .chain(
            expected
                .values()
                .zip(expected.values().skip(1))
                .map(|(prev, cur)| Some((cur - prev).to_string())),
        )
        .collect();
    assert_eq!(cells(&got, 1), want, "LAG deltas over the merged layers");
    let ids: Vec<Option<String>> = expected.keys().map(|k| Some(k.to_string())).collect();
    assert_eq!(cells(&got, 0), ids, "every live row exactly once");

    // A CTE with an aggregate.
    let cte = client
        .simple_query(
            "with big as (select id, val from spill_t where val >= 100) \
             select count(*) as n, sum(val) as s from big",
        )
        .await
        .unwrap_or_else(|e| panic!("cte: {}", pg_server::db_err(&e)));
    let big: Vec<i64> = expected.values().copied().filter(|v| *v >= 100).collect();
    assert_eq!(cells(&cte, 0), [Some(big.len().to_string())]);
    assert_eq!(cells(&cte, 1), [Some(big.iter().sum::<i64>().to_string())]);

    drop(client);
    server.shutdown().await;

    // Kill switch, on a fresh server: the capped MemTable path cannot register a
    // table past the cap, so the same window query must decline — never be
    // served truncated.
    // SAFETY: see above — no server is running.
    unsafe { std::env::set_var("PROXIMADB_NATIVE_TABLE_LAYER_A", "0") };
    let server = TestPgServer::start().await.expect("server 2");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect 2");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    load(&client).await;
    let err = client
        .simple_query(window)
        .await
        .expect_err("with the kill switch the over-cap table must decline");
    let message = pg_server::db_err(&err);
    assert!(
        message.contains("spill_t") && message.to_ascii_lowercase().contains("not found"),
        "the decline must be the unregistered over-cap table, got: {message}"
    );
    drop(client);
    server.shutdown().await;
    // SAFETY: see above — no server is running.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_LAYER_A") };
}
