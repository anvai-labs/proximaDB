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
    database: String,
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

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("REST not ready on {rest_port} within 15s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            database: format!("pgext_{pg_port}"),
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn conn_string(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname={} sslmode=disable",
            self.pg_port, self.database
        )
    }

    fn table_name(&self) -> String {
        format!("mem_ext_{}", self.pg_port)
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
async fn pgwire_extended_protocol_vector_search_with_bound_params() {
    let server = PgwireTestServer::start().await.expect("server start");

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
