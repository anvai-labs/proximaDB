// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 slice 1 (TPC-H Q2) — a correlated scalar aggregate whose body
//! is MULTI-TABLE (`min(cost) FROM ps, s WHERE ps_sk = sk AND region = 'EU' AND
//! ps_pk = p.pk`) must decorrelate and execute on the native route. Previously
//! declined (`correlated scalar subquery`): only a single-table subquery body
//! lowered. Proves the inner join conditions + inner-local filters fold into the
//! inner Filter while the one correlation equi becomes the group/join key,
//! row-exact.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_multi_table_correlated_scalar_aggregate() {
    std::thread::Builder::new()
        .name("rel-corr-scalar-mt-e2e-16m".into())
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
        "DROP TABLE IF EXISTS ps",
        "DROP TABLE IF EXISTS s",
        "DROP TABLE IF EXISTS p",
        "CREATE TABLE p (pk INT PRIMARY KEY, budget INT)",
        "CREATE TABLE s (sk INT PRIMARY KEY, region VARCHAR)",
        "CREATE TABLE ps (psid INT PRIMARY KEY, ps_pk INT, ps_sk INT, cost INT)",
        // p.budget is the target min-supplycost each part is compared against.
        "INSERT INTO p VALUES (1,10),(2,5),(3,99)",
        // suppliers: sk=1 in EU, sk=2 in US (US is filtered OUT of the subquery).
        "INSERT INTO s VALUES (1,'EU'),(2,'US')",
        // partsupp (ps_pk, ps_sk, cost):
        //   pk=1: (sk1,cost10 EU),(sk2,cost20 US)  → min over EU = 10
        //   pk=2: (sk1,cost5  EU),(sk2,cost3  US)  → min over EU = 5  (US 3 excluded)
        //   pk=3: (sk2,cost7  US)                  → no EU row → min = NULL
        "INSERT INTO ps VALUES (100,1,1,10),(101,1,2,20),(102,2,1,5),(103,2,2,3),(104,3,2,7)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q2 shape: keep parts whose budget equals the MIN cost across the joined
    // (ps ⋈ s, region='EU') for that part — correlated on ps_pk = p.pk. The
    // subquery body is MULTI-TABLE (ps, s); the inner join (ps_sk = sk) and
    // inner-local filter (region = 'EU') fold into the inner Filter, the one
    // correlation equi becomes the group/join key.
    //   pk=1: 10 == min{10} ✓   pk=2: 5 == min{5} ✓   pk=3: 99 == NULL ✗
    let mut got = rows(
        &client,
        "SELECT pk FROM p WHERE budget = \
         (SELECT min(cost) FROM ps, s WHERE ps_sk = sk AND region = 'EU' AND ps_pk = pk) \
         ORDER BY pk",
    )
    .await;
    let pks: Vec<i64> = got
        .drain(..)
        .map(|r| r[0].parse::<i64>().expect("i64"))
        .collect();
    assert_eq!(
        pks,
        vec![1, 2],
        "only parts whose budget equals the min EU supplycost qualify (pk=3's group is empty → NULL)"
    );

    eprintln!(
        "✓ TD-REL-LOWER-5 multi-table correlated scalar aggregate (Q2 shape) on native route"
    );
}
