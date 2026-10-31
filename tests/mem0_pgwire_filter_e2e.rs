//! TD-100 Part B — end-to-end check that the **pgwire vector-search path
//! honours a WHERE metadata filter** (the shape mem0's PGVector provider
//! emits). Boots an in-process ProximaDB with a free pgwire port, connects via
//! a real `tokio-postgres` client, creates a mem0-style table, inserts rows of
//! two memory types, and issues:
//!
//! ```sql
//! SELECT id, payload FROM mem WHERE payload->>'type' = 'fact'
//!   ORDER BY embedding <-> '[...]' LIMIT k
//! ```
//!
//! The PostgreSQL AST lowerer must push the predicate into the canonical native
//! search and return exactly the matching IDs. Zero-row success is a failure:
//! setup, projection, and filter semantics are all part of this boundary.

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_vector_search_honours_where_metadata_filter() {
    unsafe {
        std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true");
    }
    let server = TestPgServer::start().await.expect("server start");

    let (client, connection) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    let dim = 8usize;
    let vec_lit = |seed: f32| -> String {
        let v: Vec<String> = (0..dim)
            .map(|j| format!("{:.4}", seed + j as f32 * 0.01))
            .collect();
        format!("[{}]", v.join(","))
    };

    let table = "mem_filter";
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (id VARCHAR PRIMARY KEY, embedding VECTOR({dim}), payload JSONB)"
        ))
        .await
        .expect("create mem0 table");

    // Two facts, one decision.
    for (id, mtype, seed) in [
        ("m1", "fact", 0.10f32),
        ("m2", "fact", 0.12f32),
        ("m3", "decision", 0.90f32),
    ] {
        let payload = serde_json::json!({"type": mtype, "content": id});
        client
            .simple_query(&format!(
                "INSERT INTO {table} (id, embedding, payload) VALUES ('{id}', '{}'::vector, '{}'::jsonb)",
                vec_lit(seed),
                payload
            ))
            .await
            .unwrap_or_else(|error| panic!("insert {id}: {error}"));
    }
    sleep(Duration::from_millis(200)).await;

    let q_vec = vec_lit(0.10);
    let filtered = format!(
        "SELECT id, payload FROM {table} WHERE payload->>'type' = 'fact' \
         ORDER BY embedding <-> '{q_vec}'::vector LIMIT 10"
    );
    let unfiltered = format!(
        "SELECT id, payload FROM {table} ORDER BY embedding <-> '{q_vec}'::vector LIMIT 10"
    );

    // Exercise the simple-query result encoder here; the extended/prepared
    // protocol has its own bound-parameter parity fixture.
    fn projected_rows(
        msgs: &[tokio_postgres::SimpleQueryMessage],
    ) -> std::collections::HashMap<String, serde_json::Value> {
        msgs.iter()
            .filter_map(|message| match message {
                tokio_postgres::SimpleQueryMessage::Row(row) => Some((
                    row.get("id").expect("projected id").to_string(),
                    serde_json::from_str(row.get("payload").expect("projected payload"))
                        .expect("payload is JSONB"),
                )),
                _ => None,
            })
            .collect()
    }

    let filtered_rows = projected_rows(
        &client
            .simple_query(&filtered)
            .await
            .expect("filtered vector search executes"),
    );
    let unfiltered_rows = projected_rows(
        &client
            .simple_query(&unfiltered)
            .await
            .expect("unfiltered vector search executes"),
    );

    assert_eq!(
        filtered_rows
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>(),
        ["m1", "m2"].into_iter().map(str::to_string).collect(),
        "metadata predicate must return exactly the fact records"
    );
    assert_eq!(
        unfiltered_rows
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>(),
        ["m1", "m2", "m3"].into_iter().map(str::to_string).collect(),
        "unfiltered Strong search must see all inserted records"
    );
    for (id, payload) in &filtered_rows {
        assert_eq!(payload["type"], "fact", "filtered payload for {id}");
        assert_eq!(
            payload["content"],
            id.as_str(),
            "projected payload for {id}"
        );
    }
    assert_eq!(unfiltered_rows["m3"]["type"], "decision");
}
