// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-4 — aggregates nested inside a projection expression
//! (`sum(x)/sum(y)`, `100.0*sum(case…)/sum(y)`) must lower on the native
//! route, not error `aggregate function in non-aggregate position`.
//!
//! These are the "aggregate ratio / market-share" shapes: TPC-H Q8
//! (`sum(case when nation=… then vol else 0 end)/sum(vol)`, grouped) and
//! Q14 (`100.0*sum(case when p_type like 'PROMO%' …)/sum(…)`, global).
//! The frontend previously handled only a *bare* top-level aggregate per
//! projection item; an aggregate nested under arithmetic/CASE was declined
//! ("Phase 3"). This drives the split-projection hoist: each nested
//! aggregate is extracted to an Aggregate slot and the surrounding
//! arithmetic becomes a post-aggregate Project expression.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_aggregate_arithmetic_projection() {
    std::thread::Builder::new()
        .name("rel-agg-arith-e2e-16m".into())
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
    for ddl in [
        "DROP TABLE IF EXISTS sales",
        "CREATE TABLE sales (id INT PRIMARY KEY, yr INT, nation VARCHAR, vol DOUBLE PRECISION)",
        // 1995: BR=100, total=400 → share 0.25 ; 1996: BR=50, total=100 → 0.5
        "INSERT INTO sales VALUES (1,1995,'BR',100),(2,1995,'US',300),(3,1996,'BR',50),(4,1996,'US',50)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q8 shape: grouped market-share ratio — an aggregate CASE over another
    // aggregate, in a division, per group, ordered by the group key.
    let got = rows(
        &client,
        "SELECT yr, sum(case when nation = 'BR' then vol else 0 end) / sum(vol) AS share \
         FROM sales GROUP BY yr ORDER BY yr",
    )
    .await;
    assert_eq!(got.len(), 2, "expected one row per year");
    assert_eq!(got[0][0], "1995");
    assert_eq!(got[1][0], "1996");
    let share95: f64 = got[0][1].parse().expect("f64");
    let share96: f64 = got[1][1].parse().expect("f64");
    assert!((share95 - 0.25).abs() < 1e-9, "1995 share = {share95}");
    assert!((share96 - 0.50).abs() < 1e-9, "1996 share = {share96}");

    // Q14 shape: global (no GROUP BY) scaled ratio with a literal factor.
    // 100.0 * (BR total 150) / (grand total 500) = 30.0
    let got2 = rows(
        &client,
        "SELECT 100.0 * sum(case when nation = 'BR' then vol else 0 end) / sum(vol) AS pct \
         FROM sales",
    )
    .await;
    assert_eq!(got2.len(), 1, "global aggregate is a single row");
    let pct: f64 = got2[0][0].parse().expect("f64");
    assert!((pct - 30.0).abs() < 1e-9, "pct = {pct}");

    // Plain difference of two aggregates (no CASE) must also lower.
    let got3 = rows(
        &client,
        "SELECT sum(vol) - sum(case when nation = 'US' then vol else 0 end) AS non_us FROM sales",
    )
    .await;
    let non_us: f64 = got3[0][0].parse().expect("f64");
    assert!((non_us - 150.0).abs() < 1e-9, "non_us = {non_us}");

    eprintln!("✓ TD-REL-LOWER-4 aggregate-arithmetic projections lower on the native route");
}
