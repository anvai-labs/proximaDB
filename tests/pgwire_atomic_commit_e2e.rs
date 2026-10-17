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
    /// Start with the test-only commit fault armed BEFORE any server thread
    /// exists, so no env mutation races the server's env reads (review S4a F10).
    async fn start_with_fault(fault: &str) -> anyhow::Result<Self> {
        // SAFETY: called before the server (or any other thread) starts;
        // nextest runs each test in its own process.
        unsafe { std::env::set_var(FAULT, fault) };
        Self::start().await
    }

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
        // Anchor every storage root in this server's tempdir (TD-CONV-1
        // pattern). The defaults are CWD-relative (`file://./data`), so
        // vector-profile tables from concurrently running test servers shared
        // one directory and boot-time WAL recovery hit another server's
        // segments ("recovery segment collision").
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp.path().display()),
            ..Default::default()
        }];
        config.storage.metadata_url = format!("file://{}/metadata", tmp.path().display());
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp.path().display());

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

    /// Force the WAL → storage flush for one table's collection.
    async fn flush(&self, table: &str) {
        self.db
            .as_ref()
            .expect("server running")
            .force_flush_collection(table)
            .await
            .expect("force flush");
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

/// The fault seam is compiled only into debug builds; a release-profile run
/// has nothing to inject, so these tests have nothing to prove there.
fn fault_seam_available() -> bool {
    cfg!(debug_assertions)
}

async fn cascade_fixture(client: &tokio_postgres::Client) -> (String, String) {
    let parent = unique("par");
    let child = unique("chi");
    exec(
        client,
        &format!("CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, label VARCHAR)"),
    )
    .await;
    exec(
        client,
        &format!(
            "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
             FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE CASCADE)"
        ),
    )
    .await;
    exec(
        client,
        &format!("INSERT INTO {parent} (id, label) VALUES ('p1', 'x')"),
    )
    .await;
    exec(
        client,
        &format!("INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"),
    )
    .await;
    exec(
        client,
        &format!("INSERT INTO {child} (id, pid) VALUES ('c2', 'p1')"),
    )
    .await;
    (parent, child)
}

/// A fault injected before the durable append fails an explicit COMMIT and
/// leaves nothing; INSERT … SELECT inside an atomic transaction is refused.
#[test]
fn atomic_commit_fault_before_append_and_insert_select_refusal() {
    if !fault_seam_available() {
        return;
    }
    run(async {
        let server = Server::start_with_fault("fail_before_append")
            .await
            .expect("server");
        let client = server.connect().await;
        let t = unique("acct");
        // DDL and single-statement autocommit writes take the direct path, so
        // the armed fault does not touch them.
        exec(
            &client,
            &format!("CREATE TABLE {t} (id BIGINT PRIMARY KEY, label VARCHAR)"),
        )
        .await;
        let state = batch_sqlstate(
            &client,
            &format!(
                "BEGIN; INSERT INTO {t} (id, label) VALUES (1, 'a'); \
                 INSERT INTO {t} (id, label) VALUES (2, 'b'); COMMIT"
            ),
        )
        .await;
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

/// TD-PGOLTP-1 S4a: an autocommit cascading DELETE writes child and parent rows
/// as ONE atomic unit, so a failed append leaves parent AND children intact.
/// Red before S4a: the direct path bypassed the atomic append entirely.
#[test]
fn autocommit_cascade_delete_failure_leaves_parent_and_children() {
    if !fault_seam_available() {
        return;
    }
    run(async {
        let server = Server::start_with_fault("fail_before_append")
            .await
            .expect("server");
        let client = server.connect().await;
        let (parent, child) = cascade_fixture(&client).await;
        let state = batch_sqlstate(&client, &format!("DELETE FROM {parent} WHERE id = 'p1'")).await;
        assert!(
            state.is_some(),
            "an autocommit cascading DELETE must go through the atomic append (and fail with it)"
        );
        assert_ne!(
            state.as_deref(),
            Some("25P02"),
            "no transaction block exists to be aborted"
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
    });
}

/// Without a fault the cascade commits whole.
#[test]
fn autocommit_cascade_delete_commits_whole() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let (parent, child) = cascade_fixture(&client).await;
        let deleted = client
            .execute(&format!("DELETE FROM {parent} WHERE id = 'p1'"), &[])
            .await
            .expect("cascade delete");
        assert_eq!(deleted, 1);
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

/// Review S4a F2: when the durable outcome of an autocommit unit is unknown
/// (fsync failed → appender poisoned), the server sends FATAL 08007 and CLOSES
/// the connection over the simple-query protocol — it must not follow the
/// FATAL with an ERROR + ReadyForQuery and keep serving.
#[test]
fn autocommit_outcome_unknown_is_fatal_and_closes_the_connection() {
    if !fault_seam_available() {
        return;
    }
    run(async {
        let server = Server::start_with_fault("fail_sync").await.expect("server");
        let client = server.connect().await;
        let (parent, _child) = cascade_fixture(&client).await;
        let err = client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p1'"))
            .await
            .err()
            .expect("an indeterminate commit must fail");
        let code = err.as_db_error().map(|db| db.code().code().to_string());
        assert_eq!(code.as_deref(), Some("08007"), "{err:?}");
        assert!(
            client.simple_query("SELECT 1").await.is_err() || client.is_closed(),
            "the connection must be closed after FATAL"
        );
    });
}

/// Autocommit keeps the wire contract: command tags report affected rows and a
/// duplicate key reports 23505 (review S4a F10: assert the code, not just "some
/// error").
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
        assert_eq!(dup.as_deref(), Some("23505"), "duplicate primary key");
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

/// Review S4a F1: a vector-profile table routes to a store that cannot commit
/// atomically. Autocommit writes must keep working with the gate on (they take
/// the direct path); an explicit transaction fails closed with 0A000 and
/// writes nothing — never a partial commit.
#[test]
fn vector_tables_keep_working_with_the_gate_on() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("vec");
        // VECTOR(1): a pgwire `VECTOR(n)` table currently gets a 1-dimension
        // backing collection whatever `n` is (TD-PGWIRE-COPY-1, pre-existing).
        // Dimension 1 still exercises the vector route this test is about.
        exec(
            &client,
            &format!("CREATE TABLE {t} (id VARCHAR PRIMARY KEY, embedding VECTOR(1))"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {t} (id, embedding) VALUES ('v1', '[1]'::vector)"),
        )
        .await;
        assert_eq!(
            column(&client, &format!("SELECT id FROM {t}")).await,
            vec!["v1"]
        );
        let deleted = client
            .execute(&format!("DELETE FROM {t} WHERE id = 'v1'"), &[])
            .await
            .expect("delete");
        assert_eq!(
            deleted, 1,
            "the DELETE must find the row it is asked to delete"
        );
        assert!(
            column(&client, &format!("SELECT id FROM {t}"))
                .await
                .is_empty(),
            "the deleted row must not be returned (TD-PGWIRE-VECDEL-1)"
        );

        if std::env::var(GATE).as_deref() == Ok("1") {
            let state = batch_sqlstate(
                &client,
                &format!(
                    "BEGIN; INSERT INTO {t} (id, embedding) VALUES ('v2', '[4]'::vector); COMMIT"
                ),
            )
            .await;
            assert_eq!(state.as_deref(), Some("0A000"));
            exec(&client, "ROLLBACK").await;
            // Fail closed: the refused transaction wrote nothing.
            assert!(
                column(&client, &format!("SELECT id FROM {t}"))
                    .await
                    .is_empty(),
                "a refused atomic transaction must write nothing"
            );
        }
    });
}

/// TD-PGWIRE-VECDEL-1 regression: a vector-profile pgwire table is scanned
/// through `VectorOpsTableRecordStore`, whose scan read the RAW unflushed
/// delta (every physical version) and only dropped dead records. A DELETE's
/// tombstone was dropped as dead while the older live version survived, so
/// every scan-shaped read (`SELECT`, `COUNT(*)`, a non-key predicate) still
/// returned the row; only the by-key lookup honoured the delete.
#[test]
fn vector_table_delete_is_invisible_to_every_read_e2e() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("vecdel");
        exec(
            &client,
            &format!(
                "CREATE TABLE {t} (id VARCHAR PRIMARY KEY, label VARCHAR, embedding VECTOR(1))"
            ),
        )
        .await;
        for (id, label, v) in [("v1", "a", 1), ("v2", "b", 2)] {
            exec(
                &client,
                &format!(
                    "INSERT INTO {t} (id, label, embedding) VALUES ('{id}', '{label}', '[{v}]'::vector)"
                ),
            )
            .await;
        }
        // Each statement is a separate network round-trip, so the DELETE's batch
        // lands in a later millisecond than the INSERT's. The memtable's MVCC
        // winner rule depends on that today (TD-MEMTABLE-MVCC-1).
        exec(&client, &format!("DELETE FROM {t} WHERE id = 'v1'")).await;

        // DML target selection reads the same scan: a non-key UPDATE must not
        // match the deleted row, and so must not resurrect it.
        let resurrected = client
            .execute(
                &format!("UPDATE {t} SET label = 'x' WHERE label = 'a'"),
                &[],
            )
            .await
            .expect("non-key update");
        assert_eq!(
            resurrected, 0,
            "a non-key UPDATE must not match a deleted row"
        );

        let fresh = server.connect().await;
        for c in [&client, &fresh] {
            assert_eq!(
                column(c, &format!("SELECT id FROM {t}")).await,
                vec!["v2".to_string()],
                "full scan must not return the deleted row (mandate #16)"
            );
            assert_eq!(
                column(c, &format!("SELECT id FROM {t} WHERE id <> 'zz'")).await,
                vec!["v2".to_string()],
                "a non-key predicate scan must not return the deleted row"
            );
            assert!(
                column(c, &format!("SELECT id FROM {t} WHERE id = 'v1'"))
                    .await
                    .is_empty(),
                "the by-key read must not return the deleted row"
            );
            assert_eq!(
                column(c, &format!("SELECT COUNT(*) FROM {t}")).await,
                vec!["1".to_string()],
                "COUNT(*) must not count the deleted row"
            );
        }
    });
}

/// pgwire INSERT carried `updated_at_ns = 0` (only DELETE tombstones were
/// stamped), so any "newer stamp wins" rule lost to the tombstone. After
/// INSERT → flush → DELETE → re-INSERT → flush, the memtable keeps only the
/// tombstone (flush retains tombstone-only batches) and storage holds both
/// zero-stamped versions: the by-key read hid the live re-insert, or served the
/// deleted original. Writes are now stamped at the WAL write boundary.
#[test]
fn vector_table_reinsert_after_delete_survives_flush_e2e() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("vecreins");
        exec(
            &client,
            &format!(
                "CREATE TABLE {t} (id VARCHAR PRIMARY KEY, label VARCHAR, embedding VECTOR(1))"
            ),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {t} (id, label, embedding) VALUES ('v1', 'old', '[1]'::vector)"),
        )
        .await;
        server.flush(&t).await;
        exec(&client, &format!("DELETE FROM {t} WHERE id = 'v1'")).await;
        exec(
            &client,
            &format!("INSERT INTO {t} (id, label, embedding) VALUES ('v1', 'new', '[2]'::vector)"),
        )
        .await;
        server.flush(&t).await;

        assert_eq!(
            column(&client, &format!("SELECT label FROM {t} WHERE id = 'v1'")).await,
            vec!["new".to_string()],
            "the by-key read must return the re-inserted row, not hide it or serve the deleted one"
        );
    });
}

/// Same root cause as TD-PGWIRE-VECDEL-1: without per-row winner selection an
/// UPDATE on a vector-profile table left the superseded version visible next
/// to the new one, so a scan returned the row twice.
#[test]
fn vector_table_update_returns_only_the_current_version_e2e() {
    run(async {
        let server = Server::start().await.expect("server");
        let client = server.connect().await;
        let t = unique("vecupd");
        exec(
            &client,
            &format!(
                "CREATE TABLE {t} (id VARCHAR PRIMARY KEY, label VARCHAR, embedding VECTOR(1))"
            ),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {t} (id, label, embedding) VALUES ('v1', 'old', '[1]'::vector)"),
        )
        .await;
        exec(
            &client,
            &format!("UPDATE {t} SET label = 'new' WHERE id = 'v1'"),
        )
        .await;
        assert_eq!(
            column(&client, &format!("SELECT label FROM {t}")).await,
            vec!["new".to_string()],
            "a scan must return exactly the current version of an updated row"
        );
        assert_eq!(
            column(&client, &format!("SELECT COUNT(*) FROM {t}")).await,
            vec!["1".to_string()]
        );
    });
}
