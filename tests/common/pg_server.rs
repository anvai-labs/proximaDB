// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Shared in-process pgwire e2e server harness.
//!
//! Every pgwire/relational e2e target used to hand-roll the same ~80 lines:
//! `free_port`, a `PgServer { pg_port, db, _tmp }` with `start()` wiring
//! `Config::default()` (bind address, rest/grpc/pg ports, temp data dir,
//! storage location, WAL buffer dir, the TD-CONV-1 `metadata_url` anchor),
//! a `/health` poll, `conn_str()`, and a `Drop` shutdown — plus a
//! `SimpleQuery` row-extraction helper. This module is the single copy;
//! include it per test target with:
//!
//! ```ignore
//! #[path = "common/pg_server.rs"]
//! mod pg_server;
//! ```
//!
//! and use `pg_server::TestPgServer::start()`, `pg_server::rows(...)`.

#![allow(dead_code, unused_imports, unused_variables)]

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

pub fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

/// One in-process ProximaDB server with the pgwire listener enabled.
/// Shutdown on drop is fire-and-forget (matches the harnesses this replaces).
pub struct TestPgServer {
    pub pg_port: u16,
    pub rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl TestPgServer {
    pub async fn start() -> anyhow::Result<Self> {
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
        // Anchor the system catalog to this test's temp dir: the CWD-relative
        // default shares one catalog across concurrently running nextest
        // processes, racing same-named DDL (TD-CONV-1).
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
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
            rest_port,
            db: Some(db),
            _tmp: tmp,
        })
    }

    pub fn conn_str(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }

    /// The server's temp data dir (also the single storage location's root).
    pub fn data_dir(&self) -> &std::path::Path {
        self._tmp.path()
    }

    /// The running server handle, for tests that inspect shared services.
    pub fn db(&self) -> &ProximaDB {
        self.db.as_ref().expect("db alive until drop")
    }

    /// Awaited shutdown, for harnesses that sequence server generations in one
    /// process (start → benchmark → shutdown → start) and need the shutdown to
    /// complete before proceeding. Consumes the server; the `Drop` impl then
    /// no-ops.
    pub async fn shutdown(mut self) {
        if let Some(mut db) = self.db.take() {
            let _ = db.shutdown().await;
        }
    }
}

impl Drop for TestPgServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// Render a `tokio_postgres` error with the server-side `DbError` code/message
/// when one is attached (the pgwire client otherwise surfaces a generic
/// "db error").
pub fn db_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

/// Run a simple query and return one `Vec<String>` of column values per row.
/// Panics with the SQL and the server `DbError` detail on failure.
pub async fn rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let msgs = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}\n  {}", db_err(&e)));
    msgs.iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).unwrap_or("").to_string())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect()
}

/// Same as [`rows`] but each row is rendered as one pipe-joined
/// `"a|b|c"` string.
pub async fn pipe_rows(client: &tokio_postgres::Client, sql: &str) -> Vec<String> {
    let msgs = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}\n  {}", db_err(&e)));
    msgs.iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).unwrap_or("").to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            _ => None,
        })
        .collect()
}
