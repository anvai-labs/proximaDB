// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-7 — an N-way comma join whose FROM order places two tables with
//! no direct equi-key adjacent (TPC-H Q9: `part, supplier` joined only through
//! `lineitem`) must reorder by equi-connectivity so the left-deep chain has a
//! join key at every level, instead of declining `0A000 … unnormalized cross
//! join`. Here `pt` and `sp` connect only through the hub `li`.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_nway_comma_join_reorders_by_connectivity() {
    std::thread::Builder::new()
        .name("rel-join-reorder-e2e-16m".into())
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
        "DROP TABLE IF EXISTS pt",
        "DROP TABLE IF EXISTS sp",
        "DROP TABLE IF EXISTS li",
        "CREATE TABLE pt (pk INT PRIMARY KEY, pname VARCHAR)",
        "CREATE TABLE sp (sk INT PRIMARY KEY, snat VARCHAR)",
        "CREATE TABLE li (lid INT PRIMARY KEY, lpk INT, lsk INT)",
        "INSERT INTO pt VALUES (1,'A'),(2,'B')",
        "INSERT INTO sp VALUES (10,'X'),(20,'Y')",
        // li rows join pt on lpk=pk and sp on lsk=sk — all three match a pt+sp.
        "INSERT INTO li VALUES (100,1,10),(101,2,20),(102,1,20)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q9 shape: FROM order puts `pt, sp` adjacent, but they share NO direct
    // equi-key — both join only through the hub `li`. Was declined pre-fix with
    // `0A000 … unnormalized cross join`; the reorder makes it li-in-the-middle.
    let got = rows(
        &client,
        "SELECT count(*) AS n FROM pt, sp, li WHERE pt.pk = li.lpk AND sp.sk = li.lsk",
    )
    .await;
    assert_eq!(got.len(), 1);
    let n: i64 = got[0][0].parse().expect("i64");
    assert_eq!(n, 3, "each li row joins exactly one pt and one sp");

    // And with an extra filter, to exercise a residual over the reordered chain.
    let got2 = rows(
        &client,
        "SELECT count(*) AS n FROM pt, sp, li \
         WHERE pt.pk = li.lpk AND sp.sk = li.lsk AND pt.pname = 'A'",
    )
    .await;
    let n2: i64 = got2[0][0].parse().expect("i64");
    assert_eq!(n2, 2, "pt.pname='A' (pk=1) matches li rows 100 and 102");

    eprintln!(
        "✓ TD-REL-LOWER-7 N-way comma join reorders by equi-connectivity on the native route"
    );
}
