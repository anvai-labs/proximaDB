// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 slice 3 (TPC-H Q20 root cause) — a correlated scalar aggregate
//! with TWO correlation keys (`WHERE lpk = pk AND lsk = sk`) must decorrelate and
//! execute on the native route. Previously declined at the single-key MVP guard
//! (`correlated scalar subquery`), which — masked upstream by the outer `IN`'s
//! fallback — was the true blocker behind Q20's "IN in expression position"
//! decline. Proves GROUP BY on both keys + a LEFT JOIN ON both keys, row-exact.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_two_key_correlated_scalar_aggregate() {
    std::thread::Builder::new()
        .name("rel-2key-corr-scalar-e2e-16m".into())
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
        "DROP TABLE IF EXISTS ps",
        "CREATE TABLE ps (psid INT PRIMARY KEY, pk INT, sk INT, availqty INT)",
        "CREATE TABLE line (lid INT PRIMARY KEY, lpk INT, lsk INT, qty INT)",
        "INSERT INTO ps VALUES (1,1,1,100),(2,1,2,10),(3,2,1,50)",
        // per (pk,sk): 0.5*sum(qty) threshold; keep ps rows with availqty > threshold
        //   (1,1): qty 40+40 = 80 → 0.5*80 = 40 ; availqty 100 > 40  ✓
        //   (1,2): qty 30      = 30 → 0.5*30 = 15 ; availqty 10  > 15 ✗
        //   (2,1): qty 10      = 10 → 0.5*10 = 5  ; availqty 50  > 5  ✓
        "INSERT INTO line VALUES (10,1,1,40),(11,1,1,40),(12,1,2,30),(13,2,1,10)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q20-shape inner: a correlated scalar aggregate correlated on TWO keys
    // (lpk = pk AND lsk = sk). The decorrelation groups line by (lpk, lsk) and
    // LEFT JOINs ps on both keys.
    let mut got = rows(
        &client,
        "SELECT pk, sk FROM ps \
         WHERE availqty > (SELECT 0.5 * sum(qty) FROM line WHERE lpk = pk AND lsk = sk) \
         ORDER BY pk, sk",
    )
    .await;
    let pairs: Vec<(i64, i64)> = got
        .drain(..)
        .map(|r| (r[0].parse().expect("pk"), r[1].parse().expect("sk")))
        .collect();
    assert_eq!(
        pairs,
        vec![(1, 1), (2, 1)],
        "only (pk,sk) whose availqty exceeds half its per-(pk,sk) qty total qualify"
    );

    eprintln!(
        "✓ TD-REL-LOWER-5 two-key correlated scalar aggregate (Q20 root cause) on native route"
    );
}
