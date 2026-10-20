// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "duckdb")]

//! ADR-059 rollout step 2 — DuckDB-Local as the PRIMARY engine for join/agg
//! parquet OLAP over the REAL pgwire route (default-OFF; this eval is the
//! enable gate, mandate #13).
//!
//! With `PROXIMADB_DUCKDB_ROUTE=1`, join/grouped SELECTs over materialized
//! (parquet-backed, delta-clean) tables are attempted on in-process DuckDB with
//! DataFusion as the correctness floor. This eval asserts, end-to-end:
//!
//! 1. **Result correctness** — the eligible battery returns exactly the known
//!    rows (the same fixtures/expectations the TD-REL-LOWER-1 and route-override
//!    evals verified on the DataFusion and native routes).
//! 2. **Attribution** — the cost model learns `DuckDbCompat` cells for the
//!    eligible classes (proof DuckDB actually served AND the io_trace re-stamp
//!    attributed the fold to the serving engine), and no `DuckDbCompat` cell
//!    appears for the scalar/metadata classes that must stay off DuckDB.
//!
//!   RUST_LOG=proximadb=debug cargo test --features duckdb --test duckdb_route_pgwire_e2e -- --nocapture

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, db_err as db_error_detail};
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

/// DuckDB-eligible battery (join-bearing / grouped over parquet), with the
/// exact expected rows over the fixture data.
fn eligible_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
            vec!["F|90", "O|80", "P|80"],
        ),
        (
            "SELECT o.o_orderstatus, sum(l.l_extendedprice) FROM orders o, lineitem l WHERE o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
            vec!["F|90", "O|80", "P|80"],
        ),
        (
            "SELECT o_orderstatus, count(*) FROM orders GROUP BY o_orderstatus ORDER BY o_orderstatus",
            vec!["F|2", "O|3", "P|1"],
        ),
        (
            "SELECT o.o_orderstatus, CAST(sum(l.l_extendedprice) AS DECIMAL(12,2)) FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey GROUP BY o.o_orderstatus ORDER BY o.o_orderstatus",
            vec!["F|90.00", "O|80.00", "P|80.00"],
        ),
        (
            "SELECT o.o_orderkey, l.l_extendedprice FROM orders o LEFT JOIN lineitem l ON o.o_orderkey = l.l_orderkey WHERE o.o_orderkey = 6",
            vec!["6|"],
        ),
    ]
}

/// TD-ROUTE-2 harness pin: 16 MiB thread (dev-profile pgwire lowering
/// overflows the default test stack) — same as the sibling evals.
#[test]
fn duckdb_primary_serves_join_parquet_with_result_parity() {
    // The route gate reads the env ONCE (OnceLock) — set it before any query.
    // nextest's process-per-test isolation makes this deterministic.
    unsafe { std::env::set_var("PROXIMADB_DUCKDB_ROUTE", "1") };
    std::thread::Builder::new()
        .name("duckdb-route-eval-16m".into())
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
        "INSERT INTO orders VALUES (1,10,100.0,'O'),(2,20,200.0,'F'),(3,10,50.0,'O'),(4,30,300.0,'P'),(5,20,150.0,'F'),(6,40,NULL,'O')",
        "INSERT INTO lineitem VALUES (1,5.0,50.0),(1,2.0,20.0),(2,3.0,60.0),(3,1.0,10.0),(4,4.0,80.0),(5,2.0,30.0)",
    ] {
        client
            .simple_query(ddl)
            .await
            .unwrap_or_else(|e| panic!("{ddl}: {}", db_error_detail(&e)));
    }
    // Materialize so the tables are parquet-backed (the only DuckDB-eligible
    // base). No post-MATERIALIZE writes ⇒ the delta is clean ⇒ eligible.
    for t in ["orders", "lineitem"] {
        client
            .simple_query(&format!("ALTER TABLE {t} MATERIALIZE"))
            .await
            .unwrap_or_else(|e| panic!("MATERIALIZE {t}: {e}"));
    }

    // 1. Result correctness on the DuckDB-eligible battery.
    for (sql, expected) in eligible_cases() {
        let rows = query_rows(&client, sql).await.unwrap_or_else(|e| {
            panic!("eligible query failed under the DuckDB route:\n  {sql}\n  {e}")
        });
        let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            rows, expected,
            "DuckDB route changed results (unsafe):\n  {sql}"
        );
    }
    // Control: scalar/metadata shape — must NOT be DuckDB-eligible.
    let rows = query_rows(&client, "SELECT count(*) FROM orders")
        .await
        .expect("scalar count");
    assert_eq!(rows, vec!["6".to_string()]);

    // 2. Attribution: DuckDbCompat cells learned for eligible classes only.
    let cells = proximadb::query::route_cost_model::GLOBAL_ROUTE_COST_MODEL.learned_cell_keys();
    let duck_cells: Vec<&(String, String)> = cells
        .iter()
        .filter(|(_, backend)| backend == "DuckDbCompat")
        .collect();
    assert!(
        !duck_cells.is_empty(),
        "no DuckDbCompat cost cells learned — DuckDB never served through the route \
         (gate/eligibility regression) or the io_trace re-stamp broke; cells: {cells:?}"
    );
    for (class, _) in &duck_cells {
        assert!(
            class.starts_with("olap/parquet"),
            "DuckDbCompat cell outside olap/parquet: {class}"
        );
        assert!(
            !class.contains("op=meta") && !class.contains("op=agg"),
            "DuckDbCompat served a scalar/metadata class that must stay on native/DataFusion: {class}"
        );
    }
    eprintln!("✓ DuckDbCompat cells: {duck_cells:?}");
}
