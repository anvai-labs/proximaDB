// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-GRAPH-PGWIRE-1: `SELECT * FROM <graph collection>` over pgwire used to
//! fabricate a `(table_name, 0, 0)` row regardless of whether the collection
//! existed or had data. It now declines honestly (SQLSTATE `0A000`) instead
//! of returning a silent wrong answer — proven here for both a real,
//! successfully-created graph collection and a name that was never created.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

struct PgServer {
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl PgServer {
    async fn start() -> anyhow::Result<Self> {
        let pg_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp = TempDir::new()?;
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.unified_mode = false;
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());
        // TD-CONV-1: anchor the system catalog in this server's temp dir, not the
        // shared CWD-relative `file://./metadata` default (cross-run DDL replay).
        config.storage.metadata_url = format!("file://{}", tmp.path().join("metadata").display());
        let mut db = ProximaDB::new(config).await?;
        db.start().await?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(r) if r.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => anyhow::bail!("REST not ready"),
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        sleep(Duration::from_millis(200)).await;
        Ok(Self {
            pg_port,
            db: Some(db),
            _tmp: tmp,
        })
    }
    fn conn_str(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }
}
impl Drop for PgServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

fn explain_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

#[test]
fn graph_select_over_pgwire_declines_honestly() {
    std::thread::Builder::new()
        .name("graph-select-pgwire-decline-e2e".into())
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
    let server = PgServer::start().await.expect("server");
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // A real graph collection, created successfully…
    client
        .simple_query("CREATE TABLE graph_social (id TEXT) USING GRAPH")
        .await
        .expect("create graph table");

    // …a plain SELECT against it must decline (SQLSTATE 0A000), never return
    // the old fabricated (name, 0, 0) row.
    let err = client
        .simple_query("SELECT * FROM graph_social")
        .await
        .expect_err("SELECT over a graph collection must be declined, not served");
    let msg = explain_err(&err);
    assert!(msg.starts_with("[0A000]"), "unexpected error: {msg}");
    assert!(
        msg.contains("not supported over pgwire"),
        "decline message should explain WHY: {msg}"
    );
    assert!(
        msg.contains("ProximaGraphService") || msg.contains("/api/v2/graphs"),
        "decline message should point at the real graph surface: {msg}"
    );

    // A graph-shaped name that was NEVER created declines identically — the
    // SELECT-side classifier is syntactic (table-name prefix), so this must
    // NOT differ from the "exists" case, and must NOT silently create
    // anything (unlike a naive `GraphService::get_stats` call would).
    let err2 = client
        .simple_query("SELECT * FROM graph_never_created_xyz")
        .await
        .expect_err("SELECT over a nonexistent graph-shaped table must also decline");
    let msg2 = explain_err(&err2);
    assert!(msg2.starts_with("[0A000]"), "unexpected error: {msg2}");

    // Confirm no phantom collection was silently provisioned as a side
    // effect of the declined SELECT: DROP must fail "does not exist", not
    // succeed against something the SELECT itself created.
    let drop_err = client
        .simple_query("DROP TABLE graph_never_created_xyz")
        .await
        .expect_err("a declined SELECT must not have created a phantom collection");
    let drop_msg = explain_err(&drop_err);
    assert!(
        drop_msg.contains("does not exist") || drop_msg.contains("42P01"),
        "unexpected DROP outcome (phantom collection?): {drop_msg}"
    );
}
