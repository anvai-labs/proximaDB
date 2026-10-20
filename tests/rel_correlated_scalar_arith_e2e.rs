// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-8 (TPC-H Q17) — a correlated scalar aggregate wrapped in scalar
//! arithmetic (`qty < (SELECT 0.2 * avg(qty) FROM line WHERE line.lpk = part.pk)`)
//! must decorrelate and execute on the native route (was declined: only a BARE
//! correlated aggregate lowered). Proves the arithmetic wrapper is re-applied to
//! the decorrelated aggregate value, row-exact.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_correlated_scalar_aggregate_with_arithmetic_wrapper() {
    std::thread::Builder::new()
        .name("rel-corr-scalar-e2e-16m".into())
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
        "DROP TABLE IF EXISTS line",
        "DROP TABLE IF EXISTS part",
        "CREATE TABLE part (pk INT PRIMARY KEY, brand VARCHAR)",
        "CREATE TABLE line (lid INT PRIMARY KEY, lpk INT, qty INT)",
        "INSERT INTO part VALUES (1,'X'),(2,'X')",
        // pk=1 lines qty {2,100}: avg=51, 0.2*avg=10.2 → qty<10.2 ⇒ {2}
        // pk=2 lines qty {5,500}: avg=252.5, 0.2*avg=50.5 → qty<50.5 ⇒ {5}
        "INSERT INTO line VALUES (10,1,2),(11,1,100),(12,2,5),(13,2,500)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q17 shape: qty below 0.2× the per-part average — the correlated scalar
    // aggregate is wrapped in `0.2 *`. Qualifying: line(2) at pk=1, line(5) at
    // pk=2 → sum(qty) = 7. (Was declined: `correlated scalar subquery`.)
    let got = rows(
        &client,
        "SELECT sum(qty) AS s FROM line, part \
         WHERE pk = lpk AND brand = 'X' \
           AND qty < (SELECT 0.2 * avg(qty) FROM line WHERE lpk = pk)",
    )
    .await;
    assert_eq!(got.len(), 1, "global aggregate is a single row");
    let s: i64 = got[0][0].parse().expect("i64");
    assert_eq!(s, 7, "only qty below 0.2*avg(per-part) qualify: 2 + 5 = 7");

    eprintln!("✓ TD-REL-LOWER-8 correlated scalar aggregate w/ arithmetic wrapper on native route");
}
