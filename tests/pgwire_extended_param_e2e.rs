//! TD-102 Slice A — pgwire extended (prepared-statement) protocol with bound
//! `$N` parameters on the vector-search path.
//!
//! mem0's PGVector provider parameterizes its queries
//! (`ORDER BY embedding <-> $1 LIMIT $2`) and drives them over the extended
//! query protocol (Parse/Bind/Execute), which `tokio_postgres::query(sql,
//! &params)` uses. Before this slice, the extended path sent a duplicate
//! `RowDescription` (an empty one from Describe(portal), then the real one
//! during Execute), so the client errored with `UnexpectedMessage`.
//!
//! This test exercises the extended protocol directly. After the fix it must
//! execute without a protocol error. (The simple-query path is covered by
//! `mem0_pgwire_filter_e2e`.)
//!
//! One ProximaDB boot per process (global WAL manifest is a set-once singleton).

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

use std::time::Duration;


#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_extended_protocol_vector_search_with_bound_params() {
    unsafe { std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true"); }
    let server = TestPgServer::start().await.expect("server start");

    let (client, connection) =
        tokio_postgres::connect(&server.conn_string(), tokio_postgres::NoTls)
            .await
            .expect("connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    let dim = 8usize;
    let vec_text = {
        let v: Vec<String> = (0..dim)
            .map(|j| format!("{:.4}", 0.10 + j as f32 * 0.01))
            .collect();
        format!("[{}]", v.join(","))
    };

    let table = server.table_name();
    // DDL + INSERT via simple_query (already supported).
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (id VARCHAR PRIMARY KEY, embedding VECTOR({dim}), payload JSONB)"
        ))
        .await
        .expect("pgwire CREATE TABLE");
    for i in 1..=3 {
        client
            .simple_query(&format!(
                "INSERT INTO {table} (id, embedding, payload) VALUES ('m{i}', '{vec_text}'::vector, '{{}}'::jsonb)"
            ))
            .await
            .unwrap_or_else(|error| panic!("pgwire INSERT m{i}: {error}"));
    }
    sleep(Duration::from_millis(150)).await;

    // The crux: extended protocol with bound params. `query()` issues
    // Parse/Bind/Describe/Execute and reads typed columns back.
    let limit: i64 = 5;
    let sql = format!("SELECT id FROM {table} ORDER BY embedding <-> $1 LIMIT $2");
    let rows = client
        .query(&sql, &[&vec_text, &limit])
        .await
        .expect("extended-protocol vector search with bound params");
    assert_eq!(rows.len(), 3, "Strong search must see all three WAL rows");
    assert!(
        rows.iter().all(|row| row.len() == 1),
        "Describe and Execute must both honor SELECT id projection"
    );
    let ids: std::collections::HashSet<String> =
        rows.iter().map(|row| row.get::<_, String>(0)).collect();
    assert_eq!(
        ids,
        ["m1", "m2", "m3"].into_iter().map(str::to_string).collect()
    );
}
