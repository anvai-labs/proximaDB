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

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct PgwireTestServer {
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl PgwireTestServer {
    async fn start() -> anyhow::Result<Self> {
        unsafe {
            std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true");
        }

        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        config.storage.metadata_url = format!("file://{}/metadata", tmp_data.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!(
                            "REST server didn't become ready on port {} within 15s",
                            rest_port
                        );
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn pg_connection_string(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }
}

impl Drop for PgwireTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_vector_search_honours_where_metadata_filter() {
    let server = PgwireTestServer::start().await.expect("server start");

    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
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
