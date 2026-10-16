// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! pgwire atomic COMMIT e2e (ADR-095 D3 / TD-PGOLTP-1).
//!
//! The P2.D path replayed buffered statements one by one at COMMIT, so a failure
//! in statement N left statements 1..N-1 durable. With `[oltp] atomic_commit`
//! a transaction is staged, then made durable as ONE WAL group frame: it is
//! all-or-nothing. Through the real driver over pgwire:
//!
//! 1. a failure in statement 3 of 4 at COMMIT leaves NOTHING committed;
//! 2. the same across two tables;
//! 3. INSERT then UPDATE of the same row in one transaction commits the update;
//! 4. a fault injected before the durable append leaves nothing committed;
//! 5. INSERT … SELECT inside an atomic transaction is refused (0A000), because
//!    the table-write executor would bypass staging.
//!
//! Run with `PROXIMADB_OLTP_ATOMIC_COMMIT=0` to see 1–2 fail on the legacy
//! replay path (that is the defect this closes).

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;

const GATE: &str = "PROXIMADB_OLTP_ATOMIC_COMMIT";
const FAULT: &str = "PROXIMADB_TEST_TXN_COMMIT_FAULT";

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct Server {
    pg_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

impl Server {
    async fn start() -> anyhow::Result<Self> {
        // Arm the gate unless the runner pinned it (to demonstrate the legacy RED).
        if std::env::var(GATE).is_err() {
            // SAFETY: set before the server (and any other thread) starts;
            // nextest runs each test in its own process.
            unsafe { std::env::set_var(GATE, "1") };
        }
        let pg_port = free_port();
        let rest_port = free_port();
        let tmp = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = free_port();
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
            db: Some(db),
            _tmp: tmp,
        })
    }

    async fn connect(&self) -> tokio_postgres::Client {
        let conn_str = format!(
            "host=127.0.0.1 port={} user=anonymous dbname=proximadb sslmode=disable",
            self.pg_port
        );
        let (client, conn) = tokio_postgres::connect(&conn_str, tokio_postgres::NoTls)
            .await
            .expect("connect");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        client
    }
}

fn unique(prefix: &str) -> String {
    format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    )
}

async fn exec(client: &tokio_postgres::Client, sql: &str) {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e:?}"));
}

async fn column(client: &tokio_postgres::Client, sql: &str) -> Vec<String> {
    let mut values: Vec<String> = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e:?}"))
        .into_iter()
        .filter_map(|m| match m {
            tokio_postgres::SimpleQueryMessage::Row(r) => {
                Some(r.get(0).unwrap_or_default().to_string())
            }
            _ => None,
        })
        .collect();
    values.sort();
    values
}

/// SQLSTATE of a failed batch, or `None` when it succeeded.
async fn batch_sqlstate(client: &tokio_postgres::Client, sql: &str) -> Option<String> {
    match client.simple_query(sql).await {
        Ok(_) => None,
        Err(e) => Some(
            e.as_db_error()
                .map(|db| db.code().code().to_string())
                .unwrap_or_else(|| format!("non-db error: {e}")),
        ),
    }
}

fn run(test: impl std::future::Future<Output = ()>) {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(test);
}

#[test]
fn atomic_commit_is_all_or_nothing_on_a_mid_commit_failure() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("acct");
        exec(
            &client,
            &format!("CREATE TABLE {t} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {t} (id, label) VALUES (3, 'pre-existing')"),
        )
        .await;

        // Statement 3 collides with the committed row; buffering cannot see it,
        // so it fails AT COMMIT — after statements 1 and 2 were executed.
        let state = batch_sqlstate(
            &client,
            &format!(
                "BEGIN; \
                 INSERT INTO {t} (id, label) VALUES (1, 'a'); \
                 INSERT INTO {t} (id, label) VALUES (2, 'b'); \
                 INSERT INTO {t} (id, label) VALUES (3, 'dup'); \
                 INSERT INTO {t} (id, label) VALUES (4, 'd'); \
                 COMMIT"
            ),
        )
        .await;
        assert!(state.is_some(), "COMMIT must report the failed statement");
        exec(&client, "ROLLBACK").await;

        assert_eq!(
            column(&client, &format!("SELECT id FROM {t}")).await,
            vec!["3"],
            "a failed COMMIT must leave NOTHING durable (statements 1-2 leaked = the P2.D defect)"
        );
    });
}

#[test]
fn atomic_commit_spans_tables_and_sees_its_own_earlier_statements() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let a = unique("acct");
        let h = unique("hist");
        exec(
            &client,
            &format!("CREATE TABLE {a} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        exec(
            &client,
            &format!("CREATE TABLE {h} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {h} (id, label) VALUES (9, 'pre')"),
        )
        .await;

        // Two tables; the last statement fails at COMMIT → neither table changes.
        let state = batch_sqlstate(
            &client,
            &format!(
                "BEGIN; \
                 INSERT INTO {a} (id, label) VALUES (1, 'x'); \
                 INSERT INTO {h} (id, label) VALUES (9, 'dup'); \
                 COMMIT"
            ),
        )
        .await;
        assert!(state.is_some());
        exec(&client, "ROLLBACK").await;
        assert!(
            column(&client, &format!("SELECT id FROM {a}"))
                .await
                .is_empty()
        );
        assert_eq!(
            column(&client, &format!("SELECT label FROM {h}")).await,
            vec!["pre"]
        );

        // INSERT then UPDATE of the same row inside one transaction: the UPDATE
        // must find the row staged by the INSERT, and both tables commit.
        exec(
            &client,
            &format!(
                "BEGIN; \
                 INSERT INTO {a} (id, label) VALUES (10, 'first'); \
                 UPDATE {a} SET label = 'second' WHERE id = 10; \
                 INSERT INTO {h} (id, label) VALUES (10, 'moved'); \
                 COMMIT"
            ),
        )
        .await;
        assert_eq!(
            column(&client, &format!("SELECT label FROM {a} WHERE id = 10")).await,
            vec!["second"]
        );
        assert_eq!(
            column(&client, &format!("SELECT id FROM {h}")).await,
            vec!["10", "9"]
        );
    });
}

#[test]
fn atomic_commit_fault_before_append_and_insert_select_refusal() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("acct");
        exec(
            &client,
            &format!("CREATE TABLE {t} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;

        // SAFETY: single test per process under nextest; the fault is read per
        // append and cleared immediately after this COMMIT.
        unsafe { std::env::set_var(FAULT, "fail_before_append") };
        let state = batch_sqlstate(
            &client,
            &format!(
                "BEGIN; INSERT INTO {t} (id, label) VALUES (1, 'a'); \
                 INSERT INTO {t} (id, label) VALUES (2, 'b'); COMMIT"
            ),
        )
        .await;
        unsafe { std::env::remove_var(FAULT) };
        assert!(
            state.is_some(),
            "an injected append failure must fail COMMIT"
        );
        exec(&client, "ROLLBACK").await;
        assert!(
            column(&client, &format!("SELECT id FROM {t}"))
                .await
                .is_empty()
        );

        // The executor-backed INSERT … SELECT would bypass staging: refused.
        let src = unique("src");
        exec(
            &client,
            &format!("CREATE TABLE {src} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {src} (id, label) VALUES (5, 'e')"),
        )
        .await;
        let state = batch_sqlstate(
            &client,
            &format!("BEGIN; INSERT INTO {t} SELECT id, label FROM {src}; COMMIT"),
        )
        .await;
        if std::env::var(GATE).as_deref() == Ok("1") {
            assert_eq!(state.as_deref(), Some("0A000"));
        }
        exec(&client, "ROLLBACK").await;
        assert!(
            column(&client, &format!("SELECT id FROM {t}"))
                .await
                .is_empty()
        );
    });
}

/// TD-PGOLTP-1 S4: with the gate on, an AUTOCOMMIT statement is a one-statement
/// atomic transaction — a cascading DELETE writes child and parent rows as ONE
/// WAL group frame. An append failure must therefore leave parent AND children
/// intact (the legacy path wrote children, then the parent, as separate appends,
/// and bypassed the atomic append entirely).
#[test]
fn autocommit_cascade_delete_is_one_atomic_unit() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let parent = unique("par");
        let child = unique("chi");
        exec(
            &client,
            &format!("CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        exec(
            &client,
            &format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
                 FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE CASCADE)"
            ),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {parent} (id, label) VALUES ('p1', 'x')"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {child} (id, pid) VALUES ('c2', 'p1')"),
        )
        .await;

        // SAFETY: one test per process under nextest; cleared right after.
        unsafe { std::env::set_var(FAULT, "fail_before_append") };
        let state = batch_sqlstate(&client, &format!("DELETE FROM {parent} WHERE id = 'p1'")).await;
        unsafe { std::env::remove_var(FAULT) };
        assert!(
            state.is_some(),
            "an autocommit DELETE must go through the atomic append (and fail with it)"
        );
        assert_eq!(
            column(&client, &format!("SELECT id FROM {parent}")).await,
            vec!["p1"]
        );
        assert_eq!(
            column(&client, &format!("SELECT id FROM {child}")).await,
            vec!["c1", "c2"],
            "a failed cascading DELETE must not have removed any child"
        );

        // Without a fault the cascade commits whole.
        exec(&client, &format!("DELETE FROM {parent} WHERE id = 'p1'")).await;
        assert!(
            column(&client, &format!("SELECT id FROM {parent}"))
                .await
                .is_empty()
        );
        assert!(
            column(&client, &format!("SELECT id FROM {child}"))
                .await
                .is_empty()
        );
    });
}

/// Autocommit through the atomic path keeps the wire contract: command tags
/// report the rows each statement affected, and constraint errors keep their
/// SQLSTATE.
#[test]
fn autocommit_statements_keep_their_tags_and_sqlstates() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("acct");
        exec(
            &client,
            &format!("CREATE TABLE {t} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        let inserted = client
            .execute(&format!("INSERT INTO {t} (id, label) VALUES (1, 'a')"), &[])
            .await
            .expect("insert");
        assert_eq!(inserted, 1);
        let updated = client
            .execute(&format!("UPDATE {t} SET label = 'b' WHERE id = 1"), &[])
            .await
            .expect("update");
        assert_eq!(updated, 1);
        let dup = batch_sqlstate(
            &client,
            &format!("INSERT INTO {t} (id, label) VALUES (1, 'dup')"),
        )
        .await;
        assert!(dup.is_some(), "a duplicate primary key must fail");
        assert_eq!(
            column(&client, &format!("SELECT label FROM {t}")).await,
            vec!["b"]
        );
        let deleted = client
            .execute(&format!("DELETE FROM {t} WHERE id = 1"), &[])
            .await
            .expect("delete");
        assert_eq!(deleted, 1);
    });
}
