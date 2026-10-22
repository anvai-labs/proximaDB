// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-USUB-2: SQL capability must not depend on physical storage format.
//!
//! Before this, `allow_engine_sql_fallback` (the `ctx.sql` escape hatch for every
//! shape the shared relational frontend declines) was reachable only on the
//! DataFusion route, which required `parquet_backed` — i.e. a manual
//! `ALTER TABLE … MATERIALIZE`. So CTEs, window functions, joins and subqueries
//! worked on materialized tables and failed on native ones, and the error text told
//! the user to go run DDL.
//!
//! This test pins the parity: the **same SQL** must behave the same on a native
//! (never-materialized) table as on a materialized one.
//!
//! The feature is default-OFF (mandate #8), so the test enables
//! `PROXIMADB_NATIVE_TABLE_PROVIDER` for the enabled half and asserts the
//! default-off half still declines — proving the gate actually gates — and that a
//! window function declines with SQLSTATE 0A000 rather than a silently wrong answer.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;
/// The shapes that were Parquet-only before TD-USUB-2. Each is declined by the
/// shared relational frontend and therefore reachable only through `ctx.sql`.
const CAPABILITY_SHAPES: &[(&str, &str)] = &[
    (
        "cte",
        "with hot as (select id, val from usub2_t where val > 10) select id, val from hot order by id",
    ),
    (
        "window_lag",
        "select id, val, val - lag(val) over (order by id) as delta from usub2_t order by id",
    ),
    (
        "window_rank",
        "select id, rank() over (order by val desc) as r from usub2_t order by r",
    ),
    (
        "scalar_subquery",
        "select id, val from usub2_t where val > (select avg(val) from usub2_t) order by id",
    ),
];

#[test]
fn native_table_capability_parity() {
    std::thread::Builder::new()
        .name("usub2-capability-parity".into())
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
    // --- Phase 1: gate OFF (the shipped default) --------------------------------
    // SAFETY: this integration test is its own process and single-threaded here; the
    // gate is read per-query via `std::env::var`, so toggling it between phases is
    // exactly how an operator would enable the feature.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };

    let server = TestPgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .simple_query("CREATE TABLE usub2_t (id BIGINT, val BIGINT)")
        .await
        .expect("create");
    client
        .simple_query("INSERT INTO usub2_t (id, val) VALUES (1, 5), (2, 20), (3, 12), (4, 40)")
        .await
        .expect("insert");

    // Baseline sanity: a shape the native path DOES serve must work with the gate off,
    // so a later failure is attributable to the gate and not to a broken fixture.
    let base = client
        .simple_query("SELECT id FROM usub2_t")
        .await
        .expect("plain select must work on a native table regardless of the gate");
    let base_rows = base
        .iter()
        .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
        .count();
    assert_eq!(base_rows, 4, "fixture should hold 4 rows");

    // With the gate OFF, at least one capability shape must still be Parquet-only —
    // otherwise this test proves nothing about the gate. We do NOT assert a specific
    // SQLSTATE: the proximate decline belongs to a shared guard this change does not own.
    let mut declined_with_gate_off = 0usize;
    for (_label, sql) in CAPABILITY_SHAPES {
        if client.simple_query(sql).await.is_err() {
            declined_with_gate_off += 1;
        }
    }
    assert!(
        declined_with_gate_off > 0,
        "at least one capability shape must be Parquet-only with the gate OFF — \
         otherwise this test proves nothing about the gate"
    );

    // Fail closed, never silently wrong (mandate #1).
    //
    // A bare window function over a NATIVE table with the gate off has no DataFusion
    // route, and the relational frontend declines `OVER (...)` (TD-187). It used to fall
    // through to the legacy single-table path, whose projection extractor has no
    // expression parser and re-read the bare `val` column — so `val - lag(val) OVER
    // (...)` came back as `val` for every row (the former KNOWN-BAD pin). The pipeline
    // now declines it with SQLSTATE 0A000 naming the function and the gate to enable.
    let err = client
        .simple_query(
            "select id, val - lag(val) over (order by id) as delta from usub2_t order by id",
        )
        .await
        .expect_err("a window function on a native table must decline with the gate OFF");
    let db = err
        .as_db_error()
        .unwrap_or_else(|| panic!("expected a server error, got {err}"));
    assert_eq!(
        db.code().code(),
        "0A000",
        "window decline must be feature_not_supported: {}",
        pg_server::db_err(&err)
    );
    assert!(
        db.message().contains("lag") && db.message().contains("PROXIMADB_NATIVE_TABLE_PROVIDER"),
        "decline must name the window function and the gate: {}",
        db.message()
    );

    drop(client);
    drop(server);

    // --- Phase 2: gate ON -------------------------------------------------------
    // SAFETY: see phase 1.
    unsafe { std::env::set_var("PROXIMADB_NATIVE_TABLE_PROVIDER", "1") };

    let server = TestPgServer::start().await.expect("server 2");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect 2");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    client
        .simple_query("CREATE TABLE usub2_t (id BIGINT, val BIGINT)")
        .await
        .expect("create 2");
    client
        .simple_query("INSERT INTO usub2_t (id, val) VALUES (1, 5), (2, 20), (3, 12), (4, 40)")
        .await
        .expect("insert 2");

    // NOTE: no `ALTER TABLE … MATERIALIZE` anywhere in this test. That is the point.
    for (label, sql) in CAPABILITY_SHAPES {
        let rows = client.simple_query(sql).await.unwrap_or_else(|e| {
            panic!(
                "`{label}` must serve on a NATIVE table: {}",
                pg_server::db_err(&e)
            )
        });
        let n = rows
            .iter()
            .filter(|m| matches!(m, tokio_postgres::SimpleQueryMessage::Row(_)))
            .count();
        assert!(n > 0, "`{label}` returned no rows on the native table");
    }

    // Value-correctness, not just "it executed" (ADR-040): LAG must produce the real
    // previous-row delta and NULL for the first row — the same bug class TD-187 fixed
    // on the materialized path. Rows are ordered by id: 5, 20, 12, 40.
    let msgs = client
        .simple_query(
            "select id, val - lag(val) over (order by id) as delta from usub2_t order by id",
        )
        .await
        .expect("lag delta");
    let deltas: Vec<Option<String>> = msgs
        .iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                Some(r.get(1).map(|s| s.trim().to_string()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(deltas.len(), 4, "expected 4 delta rows, got {deltas:?}");
    assert!(
        deltas[0].is_none() || deltas[0].as_deref() == Some(""),
        "first row has no predecessor → delta must be NULL, got {:?}",
        deltas[0]
    );
    let numeric: Vec<i64> = deltas[1..]
        .iter()
        .map(|d| {
            d.as_deref()
                .unwrap_or_default()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric delta {d:?}")) as i64
        })
        .collect();
    assert_eq!(
        numeric,
        vec![15, -8, 28],
        "LAG deltas must be the real previous-row differences (20-5, 12-20, 40-12)"
    );

    // SAFETY: restore the default so the process leaves no gate set.
    unsafe { std::env::remove_var("PROXIMADB_NATIVE_TABLE_PROVIDER") };
}
