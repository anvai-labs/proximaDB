// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! pgwire transaction-control e2e (ADR-018 P2.D / TD-076).
//!
//! Drives the REAL driver path — `tokio_postgres::Client::transaction()`
//! sends BEGIN/COMMIT over the simple protocol and statements over the
//! extended protocol, exactly like production apps. Matrix:
//!
//! 1. commit-persists            — buffered INSERT replays at COMMIT.
//! 2. rollback-discards          — ROLLBACK drops the buffer.
//! 3. error-in-txn → 25P02       — failed-transaction semantics, then recover.
//! 4. simple batch BEGIN;INSERT;COMMIT — the historical silent-drop shape.
//! 5. DDL in txn rejected        — 0A000, buffer intact.
//! 6. read-only txn rejects DML  — 25006.
//! 7. drop-without-commit        — implicit ROLLBACK.
//! 8. savepoint rejected         — 0A000 with the ADR reason.
//!
//! P2.D semantics under test (documented divergences live in the ADR
//! amendment): writes buffer until COMMIT (store-only reads), connection
//! drop = rollback, mid-COMMIT replay failure fails the transaction.

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

struct TxnTestServer {
    pg_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Drop for TxnTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

impl TxnTestServer {
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

    fn conn_str(&self) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }
}

async fn connect(server: &TxnTestServer) -> tokio_postgres::Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = conn.await;
    });
    client
}

async fn setup_table(client: &tokio_postgres::Client, t: &str) {
    client
        .simple_query(&format!(
            "CREATE TABLE {t} (id BIGINT PRIMARY KEY, label VARCHAR)"
        ))
        .await
        .expect("create table");
}

async fn ids(client: &tokio_postgres::Client, t: &str) -> Vec<String> {
    let mut v: Vec<String> = client
        .simple_query(&format!("SELECT id FROM {t}"))
        .await
        .expect("select ids")
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                Some(r.get(0).unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect();
    v.sort();
    v
}

#[test]
fn pgwire_transaction_matrix() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("multi-thread test runtime")
        .block_on(pgwire_transaction_matrix_impl());
}

async fn pgwire_transaction_matrix_impl() {
    let server = TxnTestServer::start().await.expect("server start");
    let mut client = connect(&server).await;
    // Unique-per-run table name (same pattern as the other pgwire suites).
    let t = format!(
        "txn_t_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    setup_table(&client, &t).await;

    // ── 1. commit-persists (real driver path: BEGIN/COMMIT via simple query,
    //       the INSERT via the extended protocol). ──
    {
        let txn = client.transaction().await.expect("begin");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"1", &"committed"],
        )
        .await
        .expect("buffered insert");
        txn.commit().await.expect("commit");
    }
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1"],
        "COMMIT replays the buffer"
    );

    // ── 2. rollback-discards. ──
    {
        let txn = client.transaction().await.expect("begin 2");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"2", &"rolled-back"],
        )
        .await
        .expect("buffered insert 2");
        txn.rollback().await.expect("rollback");
    }
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1"],
        "ROLLBACK discards the buffer"
    );

    // ── 3. error-in-txn → 25P02 until ROLLBACK → then recover. ──
    {
        let txn = client.transaction().await.expect("begin 3");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"3", &"poison"],
        )
        .await
        .expect("buffered insert 3");
        // A statement error inside the txn (syntax) fails the transaction.
        let err = txn
            .execute(&format!("INSERT INTO {t} (id, label) VALUES ($1"), &[&"4"])
            .await
            .expect_err("malformed statement must error");
        let code = err
            .as_db_error()
            .map(|d| d.code().code().to_string())
            .unwrap_or_default();
        // The parse error itself is 42601; the transaction is now FAILED.
        assert_eq!(code, "42601");
        // The next statement in the failed txn → 25P02.
        let err = txn
            .execute(
                &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
                &[&"5", &"x"],
            )
            .await
            .expect_err("failed transaction must reject statements with 25P02");
        let code = err
            .as_db_error()
            .map(|d| d.code().code().to_string())
            .unwrap_or_default();
        assert_eq!(
            code, "25P02",
            "expected aborted-transaction error, got: {err}"
        );
        txn.rollback().await.expect("rollback recovers");
    }
    // New transaction works after recovery, and the poisoned INSERT never
    // landed.
    {
        let txn = client.transaction().await.expect("begin after recovery");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"6", &"after-recovery"],
        )
        .await
        .expect("insert after recovery");
        txn.commit().await.expect("commit after recovery");
    }
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1", "6"],
        "poisoned rows never landed; post-recovery commit did"
    );

    // ── 4. simple-protocol batch "BEGIN; INSERT; COMMIT" — the historical
    //       silent-INSERT-drop shape, now real. ──
    client
        .simple_query(&format!(
            "BEGIN; INSERT INTO {t} (id, label) VALUES (7, 'batched'); COMMIT;"
        ))
        .await
        .expect("batched transaction");
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1", "6", "7"],
        "the batched INSERT must land (the pre-P2.D bug silently dropped it)"
    );

    // ── 5. DDL in txn rejected; the txn stays usable. ──
    {
        let txn = client.transaction().await.expect("begin 5");
        let err = txn
            .execute("CREATE TABLE txn_ddl (id INT)", &[])
            .await
            .expect_err("DDL in a transaction must be rejected");
        let code = err
            .as_db_error()
            .map(|d| d.code().code().to_string())
            .unwrap_or_default();
        assert_eq!(code, "0A000", "expected 0A000 for DDL-in-txn, got: {err}");
        // PostgreSQL semantics (enforced by the send_error hook): the DDL
        // error FAILED the transaction — writes now 25P02 until ROLLBACK.
        let err = txn
            .execute(
                &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
                &[&"8", &"too-late"],
            )
            .await
            .expect_err("the DDL error must have failed the transaction");
        let code = err
            .as_db_error()
            .map(|d| d.code().code().to_string())
            .unwrap_or_default();
        assert_eq!(code, "25P02");
        txn.rollback().await.expect("rollback after DDL rejection");
        // A fresh transaction commits the write.
        let txn = client.transaction().await.expect("begin after ddl");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"8", &"after-ddl-reject"],
        )
        .await
        .expect("write after DDL rejection + rollback");
        txn.commit().await.expect("commit after DDL rejection");
    }
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1", "6", "7", "8"],
        "DDL rejection must not poison the transaction"
    );

    // ── 6. read-only txn rejects DML (25006). ──
    {
        client
            .simple_query("BEGIN READ ONLY")
            .await
            .expect("begin read only");
        let err = client
            .execute(
                &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
                &[&"9", &"read-only"],
            )
            .await
            .expect_err("read-only transaction must reject DML");
        let code = err
            .as_db_error()
            .map(|d| d.code().code().to_string())
            .unwrap_or_default();
        assert_eq!(code, "25006", "expected read-only violation, got: {err}");
        client
            .simple_query("ROLLBACK")
            .await
            .expect("rollback read-only");
    }

    // ── 7. drop-without-commit = implicit ROLLBACK. ──
    {
        let txn = client.transaction().await.expect("begin 7");
        txn.execute(
            &format!("INSERT INTO {t} (id, label) VALUES ($1, $2)"),
            &[&"10", &"dropped"],
        )
        .await
        .expect("buffered insert before drop");
        drop(txn); // tokio_postgres issues ROLLBACK on drop
    }
    assert_eq!(
        ids(&client, t.as_str()).await,
        vec!["1", "6", "7", "8"],
        "dropping a transaction without commit must roll back"
    );

    // ── 8. savepoint rejected with the ADR reason. ──
    let err = client
        .simple_query("SAVEPOINT sp1")
        .await
        .expect_err("savepoint must be rejected (P2.D defers savepoints)");
    let code = err
        .as_db_error()
        .map(|d| d.code().code().to_string())
        .unwrap_or_default();
    assert_eq!(code, "0A000", "expected 0A000 for SAVEPOINT, got: {err}");
}
