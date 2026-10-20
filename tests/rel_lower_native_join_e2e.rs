// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-1 — comma-join / derived-table / explicit-JOIN SELECTs on the
//! NATIVE (pre-MATERIALIZE) route.
//!
//! The measured failure (perf-ledger native sweep, 2026-07-13): the relational
//! lowering declines these shapes, the query falls through to the legacy
//! single-table path, and its naive FROM tokenizer produces misleading errors
//! (`Table 'customer,' does not exist`, `Column 'l_extendedprice)' does not
//! exist in table 'orders'`). Per the full-ANSI-over-pgwire mandate, the
//! relational route must serve these shapes on native storage — this suite is
//! the regression gate.
//!
//!   RUST_LOG=proximadb=debug cargo test --test rel_lower_native_join_e2e -- --nocapture

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, db_err as db_error_detail};

/// Run a query and return its rows as sorted `col|col|…` strings.
async fn query_rows(client: &tokio_postgres::Client, sql: &str) -> Result<Vec<String>, String> {
    let msgs = client
        .simple_query(sql)
        .await
        .map_err(|e| db_error_detail(&e))?;
    let mut rows = Vec::new();
    for m in msgs {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            let cols: Vec<String> = (0..r.len())
                .map(|i| r.get(i).unwrap_or("").to_string())
                .collect();
            rows.push(cols.join("|"));
        }
    }
    rows.sort();
    Ok(rows)
}

/// `(sql, expected sorted rows)` — deterministic over the fixture data below.
/// Every shape here previously fell through to the legacy single-table parser
/// on the native route and errored.
fn join_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // Explicit JOIN + GROUP BY (the calibration-battery carve-out).
        (
            "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
            vec!["F|90", "O|80", "P|80"],
        ),
        // Comma-join (implicit cross join + filter) — the TPC-H form, 8 of 22.
        (
            "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o, lineitem l WHERE o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
            vec!["F|90", "O|80", "P|80"],
        ),
        // Unaliased comma-join with qualified names.
        (
            "SELECT count(*) FROM orders, lineitem WHERE orders.o_orderkey = lineitem.l_orderkey",
            vec!["6"],
        ),
        // Derived table (subquery in FROM).
        (
            "SELECT avg(t.s) FROM (SELECT o_custkey, sum(o_totalprice) AS s FROM orders GROUP BY o_custkey) t",
            vec!["266.6666666666667"],
        ),
    ]
}

/// TD-ROUTE-2 harness pin: run on a dedicated 16 MiB thread (dev-profile pgwire
/// lowering overflows the default test stack) — same as the sibling evals.
#[test]
fn native_route_serves_comma_join_and_derived_table_selects() {
    std::thread::Builder::new()
        .name("rel-lower-e2e-16m".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(eval_body())
        })
        .expect("spawn eval thread")
        .join()
        .expect("eval thread panicked");
}

async fn eval_body() {
    let server = TestPgServer::start().await.expect("server start");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    for ddl in [
        "DROP TABLE IF EXISTS lineitem",
        "DROP TABLE IF EXISTS orders",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_totalprice DOUBLE PRECISION, o_orderstatus VARCHAR)",
        "CREATE TABLE lineitem (l_orderkey INT, l_quantity DOUBLE PRECISION, l_extendedprice DOUBLE PRECISION)",
        "INSERT INTO orders VALUES (1,10,100.0,'O'),(2,20,200.0,'F'),(3,10,50.0,'O'),(4,30,300.0,'P'),(5,20,150.0,'F')",
        "INSERT INTO lineitem VALUES (1,5.0,50.0),(1,2.0,20.0),(2,3.0,60.0),(3,1.0,10.0),(4,4.0,80.0),(5,2.0,30.0)",
    ] {
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("{ddl}: {}", db_error_detail(&e)));
    }

    // NATIVE route: tables are NOT materialized — this is the seam where these
    // shapes used to fall through to the legacy single-table parser.
    let mut failures = Vec::new();
    for (sql, expected) in join_cases() {
        match query_rows(&client, sql).await {
            Ok(rows) => {
                let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
                if rows != expected {
                    failures.push(format!(
                        "WRONG ROWS on native route:\n  {sql}\n  expected {expected:?}\n  got      {rows:?}"
                    ));
                }
            }
            Err(e) => failures.push(format!("ERROR on native route:\n  {sql}\n  {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "TD-REL-LOWER-1 native-route failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
