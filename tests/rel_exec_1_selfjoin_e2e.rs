// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-REL-EXEC-1 regression — the native route must not silently drop rows for
//! a residual predicate that references both aliases of a self-join.
//!
//! Root cause (fixed): `push_projections`' `rebind_columns` re-derived column
//! ordinals by NAME, and a self-join's output has DUPLICATE column names, so
//! every reference collapsed onto the first alias's ordinal — a silent wrong
//! result. The fix preserves the ordinal when it still names the column.
//!
//! This also covers the TD-REL-LOWER-3 derived-table-alias case (Q15 shape),
//! which the same fix closes; the ORDER-BY-alias case (q3/q52) is a separate
//! sub-mechanism and stays open under TD-REL-LOWER-3.

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

async fn scalar(client: &tokio_postgres::Client, sql: &str) -> String {
    let msgs = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}\n  {}", db_err(&e)));
    for m in msgs {
        if let tokio_postgres::SimpleQueryMessage::Row(r) = m {
            return r.get(0).unwrap_or("").to_string();
        }
    }
    String::new()
}
fn db_err(e: &tokio_postgres::Error) -> String {
    e.as_db_error()
        .map(|d| format!("[{}] {}", d.code().code(), d.message()))
        .unwrap_or_else(|| e.to_string())
}

/// 16 MiB stack — the TD-ROUTE-2 pgwire-lowering pin.
#[test]
fn native_selfjoin_residual_predicate_is_correct() {
    std::thread::Builder::new()
        .name("rel-exec1-e2e-16m".into())
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
        "DROP TABLE IF EXISTS nation",
        "DROP TABLE IF EXISTS edge",
        "CREATE TABLE nation (n_nationkey INT PRIMARY KEY, n_name VARCHAR)",
        "CREATE TABLE edge (e_from INT, e_to INT)",
        "INSERT INTO nation VALUES (1,'FRANCE'),(2,'GERMANY'),(3,'ITALY')",
        "INSERT INTO edge VALUES (1,2),(2,1),(1,3)",
    ] {
        client.simple_query(ddl).await.expect("ddl");
    }
    let base =
        "FROM edge, nation n1, nation n2 WHERE e_from=n1.n_nationkey AND e_to=n2.n_nationkey";

    // The core TD-REL-EXEC-1 case: an OR-of-ANDs residual over a self-join
    // (TPC-H Q7's shape). Was 0 (all rows silently dropped); must be 2.
    assert_eq!(
        scalar(&client, &format!(
            "SELECT count(*) {base} AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY') OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))"
        )).await,
        "2",
        "OR-of-ANDs residual over a self-join dropped rows (rebind name-collision)"
    );
    // A single-conjunct AND spanning both self-join aliases (can't push down) —
    // the minimal trigger. Was 0; must be 1.
    assert_eq!(
        scalar(&client, &format!(
            "SELECT count(*) {base} AND n1.n_name='FRANCE' AND n2.n_name='GERMANY' AND n1.n_nationkey<>n2.n_nationkey"
        )).await,
        "1"
    );
    // Guards that already worked — must stay correct (no regression).
    assert_eq!(
        scalar(&client, &format!("SELECT count(*) {base}")).await,
        "3"
    );
    assert_eq!(
        scalar(
            &client,
            &format!("SELECT count(*) {base} AND n1.n_name='FRANCE'")
        )
        .await,
        "2"
    );

    // TD-REL-LOWER-3 derived-table-alias case (Q15 shape) — the same fix closes
    // it: an outer predicate references a derived table's column alias.
    assert_eq!(
        scalar(
            &client,
            "SELECT count(*) FROM edge, (SELECT n_nationkey AS supplier_no FROM nation) r WHERE e_from = supplier_no",
        )
        .await,
        "3",
        "derived-table alias join dropped rows / errored (rebind)"
    );
    eprintln!("✓ TD-REL-EXEC-1 self-join residual + Q15 derived-alias correct");
}
