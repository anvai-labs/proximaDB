// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-5 close-out — the two hardest TPC-H queries this arc unblocked,
//! **Q18** (uncorrelated `IN` over a `GROUP BY … HAVING sum(...)` body — rides the
//! Q11 HAVING-aggregate cascade) and **Q20** (nested `IN` + a TWO-key correlated
//! scalar — the N-key decorrelation slice), executed END-TO-END on the NATIVE
//! route over pgwire (non-materialized tables). Proves they don't just lower but
//! run and return the correct rows. Small deterministic dataset; the shared
//! `lineitem` rows are partitioned so Q18's order-aggregation and Q20's
//! part/supplier-correlated scalar never cross-contaminate.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, rows};
#[test]
fn native_tpch_q18_and_q20_execute_row_exact() {
    std::thread::Builder::new()
        .name("native-tpch-q18-q20-e2e-16m".into())
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
        "CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR)",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT, o_orderdate DATE, o_totalprice INT)",
        "CREATE TABLE nation (n_nationkey INT PRIMARY KEY, n_name VARCHAR)",
        "CREATE TABLE supplier (s_suppkey INT PRIMARY KEY, s_name VARCHAR, s_address VARCHAR, s_nationkey INT)",
        "CREATE TABLE part (p_partkey INT PRIMARY KEY, p_name VARCHAR)",
        "CREATE TABLE partsupp (psid INT PRIMARY KEY, ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost INT)",
        "CREATE TABLE lineitem (lid INT PRIMARY KEY, l_orderkey INT, l_partkey INT, l_suppkey INT, l_quantity INT, l_shipdate DATE)",
        // --- Q18 fixture ---
        "INSERT INTO customer VALUES (1,'Alice'),(2,'Bob')",
        "INSERT INTO orders VALUES (100,1,DATE '1995-01-01',999),(200,2,DATE '1995-02-01',500)",
        // order 100 lineitems sum(qty)=350 (>300 ✓); order 200 sum=150 (≤300 ✗).
        // Their part/supp/shipdate are set OUT of Q20's filter so they don't leak.
        "INSERT INTO lineitem VALUES \
         (1,100,999,999,200,DATE '1996-01-01'),(2,100,999,999,150,DATE '1996-01-01'),\
         (3,200,999,999,100,DATE '1996-01-01'),(4,200,999,999,50,DATE '1996-01-01'),\
         (5,999,100,10,100,DATE '1994-06-01')",
        // --- Q20 fixture --- (lineitem row 5 above is the Q20-relevant one:
        //   l_partkey=100,l_suppkey=10,shipdate in 1994, qty 100; l_orderkey=999
        //   has no matching order so it never enters Q18's IN set)
        "INSERT INTO nation VALUES (1,'CANADA'),(2,'USA')",
        "INSERT INTO supplier VALUES (10,'SupA','AddrA',1),(20,'SupB','AddrB',2)",
        "INSERT INTO part VALUES (100,'forest green'),(200,'blue metal')",
        "INSERT INTO partsupp VALUES (1,100,10,500,5),(2,200,10,500,5)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q18 — customers with an order whose total line quantity exceeds 300.
    // IN set = {order 100} (sum 350); result = Alice's order 100, sum 350.
    let q18 = rows(
        &client,
        "select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) \
         from customer, orders, lineitem \
         where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300) \
           and c_custkey = o_custkey and o_orderkey = l_orderkey \
         group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice \
         order by o_totalprice desc, o_orderdate",
    )
    .await;
    assert_eq!(
        q18.len(),
        1,
        "Q18 returns exactly one qualifying order, got {q18:?}"
    );
    assert_eq!(q18[0][0], "Alice", "Q18 customer");
    assert_eq!(q18[0][2], "100", "Q18 orderkey");
    assert_eq!(q18[0][5], "350", "Q18 sum(l_quantity)");

    // Q20 — CANADA suppliers of 'forest%' parts whose available qty exceeds half
    // the 1994 shipped quantity for that (part,supplier). Result = SupA / AddrA.
    let q20 = rows(
        &client,
        "select s_name, s_address from supplier, nation \
         where s_suppkey in (\
             select ps_suppkey from partsupp \
             where ps_partkey in (select p_partkey from part where p_name like 'forest%') \
               and ps_availqty > (select 0.5 * sum(l_quantity) from lineitem \
                                  where l_partkey = ps_partkey and l_suppkey = ps_suppkey \
                                    and l_shipdate >= DATE '1994-01-01' and l_shipdate < DATE '1995-01-01')) \
           and s_nationkey = n_nationkey and n_name = 'CANADA' \
         order by s_name",
    )
    .await;
    assert_eq!(
        q20.len(),
        1,
        "Q20 returns exactly one qualifying supplier, got {q20:?}"
    );
    assert_eq!(q20[0][0], "SupA", "Q20 supplier name");
    assert_eq!(q20[0][1], "AddrA", "Q20 supplier address");

    eprintln!(
        "✓ TPC-H Q18 + Q20 execute row-exact on the NATIVE route (ADR-064 / TD-REL-LOWER-5 close-out)"
    );
}
