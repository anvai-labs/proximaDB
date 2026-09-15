// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-RERANK-PGWIRE-1: documents the REAL `RERANK(...)` contract over pgwire.
//!
//! `RERANK()` is a gRPC/federated-only capability today (`QueryFacadeAdapter`,
//! constructed only from `src/network/grpc/*`) — a comment in
//! `src/query/facade/adapter.rs` previously claimed pgwire clients could
//! `SELECT * FROM RERANK(...)`, which was false: pgwire has zero RERANK
//! dispatch anywhere. This test pins the actual, current, SAFE behavior —
//! pgwire declines rather than silently returning wrong data — so a future
//! change to the legacy-path comma guard or the UDTF-reachability gap
//! (TD-XMODAL-10) can't regress this into a silent-wrong-answer path without
//! a test noticing.

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
fn rerank_over_pgwire_declines_not_silently_wrong() {
    std::thread::Builder::new()
        .name("rerank-pgwire-contract-e2e".into())
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

    // Neither shape must return a success with zero (or any) rows — that
    // would be the silent-wrong-answer failure mode. Both must be a real
    // error, i.e. the client can tell "not supported" from "no matches".
    for sql in [
        "SELECT * FROM RERANK('coll', '[1,2,3]', 10)",
        "SELECT * FROM rerank(candidates, 'model_x')",
    ] {
        let err = client
            .simple_query(sql)
            .await
            .expect_err("RERANK() over pgwire must not silently succeed");
        let msg = explain_err(&err);
        // SQLSTATE 0A000 (feature not supported) either way, whatever the
        // proximate guard — this test pins SAFETY (an honest error), not the
        // exact wording of a message that legitimately belongs to a shared,
        // separately-scoped guard (TD-XMODAL-10 / the stale TD-REL-LOWER-1
        // citation) this PR does not touch.
        assert!(
            msg.starts_with("[0A000]"),
            "RERANK query {sql:?} should decline with 0A000, got: {msg}"
        );
    }
}
