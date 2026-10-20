// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-3 (ORDER-BY-alias case, q3/q52) — a GROUP BY key projected with
//! an alias and used as an ORDER BY key must execute on the native route, not
//! error `rebind_columns: column '<alias>' not in narrowed schema`.
//!
//! Root cause (fixed): the frontend put the projection ALIAS on the inner
//! `ColumnRef` of a grouped-key output instead of the group-key's source name,
//! so the planner's name-based rebind could not resolve it.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::{TestPgServer, pipe_rows as rows};
#[test]
fn native_orderby_alias_of_group_key() {
    std::thread::Builder::new()
        .name("rel-lower3-e2e-16m".into())
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
        "CREATE TABLE sales (id INT PRIMARY KEY, brand VARCHAR, amt INT)",
        "INSERT INTO sales VALUES (1,'A',10),(2,'A',20),(3,'B',5),(4,'C',30),(5,'C',30)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // GROUP-BY key aliased AND used as an ORDER BY key (q3/q52 shape). Ordered
    // by the alias, so the result order is deterministic.
    let got = rows(
        &client,
        "SELECT brand AS b, sum(amt) AS total FROM sales GROUP BY brand ORDER BY total DESC, b",
    )
    .await;
    // total DESC: C=60, A=30, B=5.
    assert_eq!(
        got,
        vec!["C|60".to_string(), "A|30".to_string(), "B|5".to_string()],
        "ORDER BY on a group-key alias returned wrong/erroring result"
    );

    // The pure alias-as-only-sort-key form.
    let got2 = rows(
        &client,
        "SELECT brand AS b, count(*) AS c FROM sales GROUP BY brand ORDER BY b",
    )
    .await;
    assert_eq!(
        got2,
        vec!["A|2".to_string(), "B|1".to_string(), "C|2".to_string()]
    );
    eprintln!("✓ TD-REL-LOWER-3 ORDER-BY-alias of a group key executes correctly");
}
