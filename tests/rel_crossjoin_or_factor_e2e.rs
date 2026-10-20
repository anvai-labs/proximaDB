// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-6 — a comma-join whose equi-join key appears only inside a
//! top-level `OR` of range predicates (TPC-H Q19 shape) must normalize to an
//! equi-join on the native route, not decline with `0A000 … unnormalized cross
//! join`. The planner factors the shared `c.id = o.cid` out of the disjunction:
//! `(c.id=o.cid ∧ p1) ∨ (c.id=o.cid ∧ p2)` ≡ `c.id=o.cid ∧ (p1 ∨ p2)`.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_comma_join_disjunction_equi_key() {
    std::thread::Builder::new()
        .name("rel-crossjoin-or-e2e-16m".into())
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
        "DROP TABLE IF EXISTS cust",
        "DROP TABLE IF EXISTS ord",
        "CREATE TABLE cust (id INT PRIMARY KEY, tier VARCHAR)",
        "CREATE TABLE ord (oid INT PRIMARY KEY, cid INT, amt INT)",
        "INSERT INTO cust VALUES (1,'A'),(2,'B'),(3,'C')",
        // cid=1(tier A): amt 100,40 ; cid=2(tier B): amt 60,30 ; cid=3(tier C): 200
        "INSERT INTO ord VALUES (10,1,100),(11,1,40),(12,2,60),(13,2,30),(14,3,200)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q19 shape: the join key `cust.id = ord.cid` appears in BOTH branches of a
    // top-level OR of (tier, amount) range predicates. Was declined pre-fix with
    // `0A000 … unnormalized cross join`.
    //   branch A: tier='A' AND amt>=100 → (1,10) amt 100
    //   branch B: tier='B' AND amt>=50  → (2,12) amt 60
    //   sum = 160
    let got = rows(
        &client,
        "SELECT sum(ord.amt) AS rev FROM cust, ord \
         WHERE (cust.id = ord.cid AND cust.tier = 'A' AND ord.amt >= 100) \
            OR (cust.id = ord.cid AND cust.tier = 'B' AND ord.amt >= 50)",
    )
    .await;
    assert_eq!(got.len(), 1, "global aggregate is a single row");
    let rev: i64 = got[0][0].parse().expect("i64");
    assert_eq!(
        rev, 160,
        "OR-of-ANDs comma join must equi-join and sum correctly"
    );

    eprintln!("✓ TD-REL-LOWER-6 disjunction-shared equi-key normalizes on the native route");
}
