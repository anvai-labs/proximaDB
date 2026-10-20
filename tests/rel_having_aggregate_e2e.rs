// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 slice 2 (TPC-H Q11) — a `GROUP BY … HAVING <aggregate> >
//! (<uncorrelated scalar subquery>)` must lower and execute on the native route.
//! Previously declined (`HAVING with aggregate calls`). Proves the HAVING
//! aggregate is extracted into the Aggregate, the global scalar subquery is
//! hoisted as a LEFT JOIN, and the groups are filtered post-aggregate — row-exact.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_having_aggregate_over_scalar_subquery() {
    std::thread::Builder::new()
        .name("rel-having-agg-e2e-16m".into())
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
        "CREATE TABLE ps (psid INT PRIMARY KEY, pk INT, cost INT, qty INT)",
        // value(pk) = sum(cost*qty):
        //   pk=1: 10*2 + 5*4 = 40
        //   pk=2: 3*10       = 30
        //   pk=3: 100*1      = 100
        // global sum = 170; threshold = 170*0.2 = 34 → keep value > 34 ⇒ {1,3}.
        "INSERT INTO ps VALUES (1,1,10,2),(2,1,5,4),(3,2,3,10),(4,3,100,1)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q11 shape: groups whose aggregate exceeds a fraction of the GLOBAL aggregate
    // (an uncorrelated scalar subquery). HAVING references the group aggregate AND
    // the hoisted global scalar.
    let got = rows(
        &client,
        "SELECT pk, sum(cost*qty) AS value FROM ps GROUP BY pk \
         HAVING sum(cost*qty) > (SELECT sum(cost*qty) * 0.2 FROM ps) \
         ORDER BY pk",
    )
    .await;
    let pairs: Vec<(i64, i64)> = got
        .iter()
        .map(|r| (r[0].parse().expect("pk"), r[1].parse().expect("value")))
        .collect();
    assert_eq!(
        pairs,
        vec![(1, 40), (3, 100)],
        "only groups whose value exceeds 20% of the global total qualify (pk=2's 30 ≤ 34)"
    );

    eprintln!(
        "✓ TD-REL-LOWER-5 HAVING aggregate over uncorrelated scalar subquery (Q11) on native route"
    );
}
