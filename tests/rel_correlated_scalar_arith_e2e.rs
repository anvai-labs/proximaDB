// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-LOWER-8 (TPC-H Q17) — a correlated scalar aggregate wrapped in scalar
//! arithmetic (`qty < (SELECT 0.2 * avg(qty) FROM line WHERE line.lpk = part.pk)`)
//! must decorrelate and execute on the native route (was declined: only a BARE
//! correlated aggregate lowered). Proves the arithmetic wrapper is re-applied to
//! the decorrelated aggregate value, row-exact.

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
async fn rows(client: &tokio_postgres::Client, sql: &str) -> Vec<Vec<String>> {
    let msgs = client.simple_query(sql).await.unwrap_or_else(|e| {
        panic!(
            "{sql}\n  {}",
            e.as_db_error()
                .map(|d| format!("[{}] {}", d.code().code(), d.message()))
                .unwrap_or_else(|| e.to_string())
        )
    });
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

#[test]
fn native_correlated_scalar_aggregate_with_arithmetic_wrapper() {
    std::thread::Builder::new()
        .name("rel-corr-scalar-e2e-16m".into())
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
    for ddl in [
        "DROP TABLE IF EXISTS line",
        "DROP TABLE IF EXISTS part",
        "CREATE TABLE part (pk INT PRIMARY KEY, brand VARCHAR)",
        "CREATE TABLE line (lid INT PRIMARY KEY, lpk INT, qty INT)",
        "INSERT INTO part VALUES (1,'X'),(2,'X')",
        // pk=1 lines qty {2,100}: avg=51, 0.2*avg=10.2 → qty<10.2 ⇒ {2}
        // pk=2 lines qty {5,500}: avg=252.5, 0.2*avg=50.5 → qty<50.5 ⇒ {5}
        "INSERT INTO line VALUES (10,1,2),(11,1,100),(12,2,5),(13,2,500)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }

    // Q17 shape: qty below 0.2× the per-part average — the correlated scalar
    // aggregate is wrapped in `0.2 *`. Qualifying: line(2) at pk=1, line(5) at
    // pk=2 → sum(qty) = 7. (Was declined: `correlated scalar subquery`.)
    let got = rows(
        &client,
        "SELECT sum(qty) AS s FROM line, part \
         WHERE pk = lpk AND brand = 'X' \
           AND qty < (SELECT 0.2 * avg(qty) FROM line WHERE lpk = pk)",
    )
    .await;
    assert_eq!(got.len(), 1, "global aggregate is a single row");
    let s: i64 = got[0][0].parse().expect("i64");
    assert_eq!(s, 7, "only qty below 0.2*avg(per-part) qualify: 2 + 5 = 7");

    eprintln!("✓ TD-REL-LOWER-8 correlated scalar aggregate w/ arithmetic wrapper on native route");
}
