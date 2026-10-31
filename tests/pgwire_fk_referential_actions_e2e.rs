//! TD-110: ON DELETE referential actions over pgwire (end-to-end).
//!
//! Foreign-key `ON DELETE` actions are catalogued from SQL and must be enforced
//! on the real DELETE path. This exercises the public pgwire surface: a parent
//! DELETE triggers RESTRICT (reject), CASCADE (delete children), or SET NULL
//! (clear the child FK) on child tables in the same tenant — proving the
//! enforcement that was lost in the `afafac3b6` merge is re-landed on the
//! tenant-scoped DML path.

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;
use tokio_postgres::SimpleQueryMessage;

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
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!(
                            "REST server didn't become ready on port {} within 20s",
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

/// First value of column `col` across the rows, as text (None for NULL).
fn scalar(messages: &[SimpleQueryMessage], col: &str) -> Option<String> {
    messages.iter().find_map(|msg| match msg {
        SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
        _ => None,
    })
}

/// `COUNT(*) AS n` from a simple-query result set.
fn count_star(messages: &[SimpleQueryMessage]) -> i64 {
    scalar(messages, "n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1)
}

/// Column-`col` values across all rows as a sorted set.
fn col_set(messages: &[SimpleQueryMessage], col: &str) -> BTreeSet<String> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_enforces_on_delete_referential_actions() {
    let server = PgwireTestServer::start().await.expect("server start");

    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // ---- RESTRICT / NO ACTION: deleting a referenced parent is rejected, and
    //      nothing is partially removed. ----
    {
        let parent = format!("par_restr_{suffix}");
        let child = format!("chl_restr_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
            ))
            .await
            .expect("CREATE parent (restrict)");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {parent}(id))"
            ))
            .await
            .expect("CREATE child (restrict)");
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p1', 'Alice')"
            ))
            .await
            .expect("INSERT parent");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"
            ))
            .await
            .expect("INSERT child");
        sleep(Duration::from_millis(300)).await;

        let err = client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p1'"))
            .await
            .expect_err("ON DELETE NO ACTION must reject deleting a referenced parent");
        // The pgwire client surfaces this as a generic "db error"; the real cause
        // is on the DbError. The RESTRICT contract is proven by the DELETE being
        // rejected (a referenced parent) plus the row-integrity asserts below — and
        // by contrast with the CASCADE / no-child cases later, which must succeed.
        if let Some(db_err) = err.as_db_error() {
            eprintln!("RESTRICT rejection DbError: {}", db_err.message());
        }

        // Nothing was removed: parent + child both still present.
        let parent_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {parent}"))
            .await
            .expect("COUNT parent");
        assert_eq!(
            count_star(&parent_rows),
            1,
            "parent row must survive a rejected DELETE"
        );
        let child_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {child}"))
            .await
            .expect("COUNT child");
        assert_eq!(
            count_star(&child_rows),
            1,
            "child row must survive a rejected DELETE"
        );
    }

    // ---- CASCADE: deleting the parent removes the referencing child row. ----
    {
        let parent = format!("par_casc_{suffix}");
        let child = format!("chl_casc_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
            ))
            .await
            .expect("CREATE parent (cascade)");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE child (cascade)");
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p1', 'Bob')"
            ))
            .await
            .expect("INSERT parent");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"
            ))
            .await
            .expect("INSERT child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p1'"))
            .await
            .expect("CASCADE delete of referenced parent must succeed");

        let child_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {child}"))
            .await
            .expect("COUNT child after cascade");
        assert_eq!(
            count_star(&child_rows),
            0,
            "ON DELETE CASCADE must remove the child row"
        );
    }

    // ---- SET NULL: deleting the parent keeps the child but nulls its FK. ----
    {
        let parent = format!("par_null_{suffix}");
        let child = format!("chl_null_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
            ))
            .await
            .expect("CREATE parent (set null)");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE SET NULL)"
            ))
            .await
            .expect("CREATE child (set null)");
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p1', 'Cara')"
            ))
            .await
            .expect("INSERT parent");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"
            ))
            .await
            .expect("INSERT child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p1'"))
            .await
            .expect("SET NULL delete of referenced parent must succeed");

        let child_rows = client
            .simple_query(&format!("SELECT id, pid FROM {child}"))
            .await
            .expect("SELECT child after set null");
        assert_eq!(
            col_set(&child_rows, "id"),
            ["c1".to_string()].into_iter().collect::<BTreeSet<_>>(),
            "ON DELETE SET NULL must keep the child row"
        );
        // SET NULL clears the FK so it no longer references the deleted parent.
        // The record-store value is genuinely ProximaValue::Null (covered by the
        // `delete_enforces_referential_actions` unit test) and now renders over
        // pgwire simple-query as a real SQL NULL (None), not an empty string.
        assert_eq!(
            scalar(&child_rows, "pid"),
            None,
            "ON DELETE SET NULL must render the child FK as SQL NULL"
        );
    }

    // ---- No referencing child: a parent DELETE succeeds (no false RESTRICT). ----
    {
        let parent = format!("par_none_{suffix}");
        let child = format!("chl_none_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
            ))
            .await
            .expect("CREATE parent (no-child)");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE child (no-child)");
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p1', 'Dave'), ('p2', 'Eve')"
            ))
            .await
            .expect("INSERT parents");
        // Child references only p1 — deleting p2 must succeed uneventfully.
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"
            ))
            .await
            .expect("INSERT child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p2'"))
            .await
            .expect("deleting an unreferenced parent must succeed");

        let remaining = client
            .simple_query(&format!("SELECT id FROM {parent}"))
            .await
            .expect("SELECT parents");
        assert_eq!(
            col_set(&remaining, "id"),
            ["p1".to_string()].into_iter().collect::<BTreeSet<_>>(),
            "only the unreferenced parent should be deleted"
        );
    }
}

/// TD-110 S1: recursive CASCADE, cyclic-FK rejection, the depth guard, and the
/// concurrency (no-orphan) invariant — all over pgwire against a real server.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgwire_cascade_recurses_and_rejects_cycles_depth_and_concurrency() {
    let server = PgwireTestServer::start().await.expect("server start");
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    // ── Section 1: 3-level CASCADE recursion (the shipped bug orphaned the
    //    grandchild). ──
    {
        let gp = format!("rc_gp_{suffix}");
        let mid = format!("rc_p_{suffix}");
        let leaf = format!("rc_c_{suffix}");
        client
            .simple_query(&format!("CREATE TABLE {gp} (id VARCHAR PRIMARY KEY)"))
            .await
            .expect("CREATE gp");
        client
            .simple_query(&format!(
                "CREATE TABLE {mid} (id VARCHAR PRIMARY KEY, gpid VARCHAR, FOREIGN KEY (gpid) REFERENCES {gp}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE mid");
        client
            .simple_query(&format!(
                "CREATE TABLE {leaf} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {mid}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE leaf");
        client
            .simple_query(&format!("INSERT INTO {gp} (id) VALUES ('g1')"))
            .await
            .expect("INSERT gp");
        client
            .simple_query(&format!("INSERT INTO {mid} (id, gpid) VALUES ('p1', 'g1')"))
            .await
            .expect("INSERT mid");
        client
            .simple_query(&format!("INSERT INTO {leaf} (id, pid) VALUES ('c1', 'p1')"))
            .await
            .expect("INSERT leaf");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {gp} WHERE id = 'g1'"))
            .await
            .expect("3-level CASCADE delete must succeed");
        assert_eq!(
            count_star(
                &client
                    .simple_query(&format!("SELECT COUNT(*) AS n FROM {mid}"))
                    .await
                    .expect("COUNT mid")
            ),
            0,
            "level-1 child must be cascade-deleted"
        );
        assert_eq!(
            count_star(
                &client
                    .simple_query(&format!("SELECT COUNT(*) AS n FROM {leaf}"))
                    .await
                    .expect("COUNT leaf")
            ),
            0,
            "grandchild must be cascade-deleted (was orphaned pre-S1)"
        );
    }

    // ── Section 2: cyclic CASCADE FK rejected (structural), no partial deletion. ──
    {
        let a = format!("rc_ca_{suffix}");
        let b = format!("rc_cb_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {a} (id VARCHAR PRIMARY KEY, bid VARCHAR, FOREIGN KEY (bid) REFERENCES {b}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE cyc_a");
        client
            .simple_query(&format!(
                "CREATE TABLE {b} (id VARCHAR PRIMARY KEY, aid VARCHAR, FOREIGN KEY (aid) REFERENCES {a}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE cyc_b");
        // Seed with NULL FKs (exempt from the insert-time FK check) — the cycle
        // is a *schema* cycle, detected from the FK graph with no data cycle.
        client
            .simple_query(&format!("INSERT INTO {a} (id) VALUES ('a1')"))
            .await
            .expect("INSERT cyc_a");
        client
            .simple_query(&format!("INSERT INTO {b} (id) VALUES ('b1')"))
            .await
            .expect("INSERT cyc_b");
        sleep(Duration::from_millis(300)).await;

        let err = client
            .simple_query(&format!("DELETE FROM {a} WHERE id = 'a1'"))
            .await
            .expect_err("cyclic CASCADE must be rejected");
        let msg = err
            .as_db_error()
            .map(|d| d.message().to_string())
            .unwrap_or_else(|| err.to_string());
        assert!(
            msg.contains("cycle"),
            "expected a cascade-cycle error, got: {msg}"
        );
        // No partial deletion: both rows survive (cycle detected pre-mutation).
        assert_eq!(
            count_star(
                &client
                    .simple_query(&format!("SELECT COUNT(*) AS n FROM {a}"))
                    .await
                    .expect("COUNT cyc_a")
            ),
            1,
            "cyclic-CASCADE rejection must not partially delete cyc_a"
        );
        assert_eq!(
            count_star(
                &client
                    .simple_query(&format!("SELECT COUNT(*) AS n FROM {b}"))
                    .await
                    .expect("COUNT cyc_b")
            ),
            1,
            "cyclic-CASCADE rejection must not partially delete cyc_b"
        );
    }

    // ── Section 3: bounded-depth guard trips on a 17-deep chain. ──
    {
        let root = format!("rc_d0_{suffix}");
        client
            .simple_query(&format!("CREATE TABLE {root} (id VARCHAR PRIMARY KEY)"))
            .await
            .expect("CREATE depth root");
        let mut chain: Vec<String> = vec![root.clone()];
        for i in 1u32..=16 {
            let t = format!("rc_d{i}_{suffix}");
            let prev = chain[(i - 1) as usize].clone();
            client
                .simple_query(&format!(
                    "CREATE TABLE {t} (id VARCHAR PRIMARY KEY, pid VARCHAR, FOREIGN KEY (pid) REFERENCES {prev}(id) ON DELETE CASCADE)"
                ))
                .await
                .expect("CREATE depth table");
            chain.push(t);
        }
        client
            .simple_query(&format!("INSERT INTO {root} (id) VALUES ('r0')"))
            .await
            .expect("INSERT depth root");
        for i in 1u32..=16 {
            let t = chain[i as usize].clone();
            client
                .simple_query(&format!(
                    "INSERT INTO {t} (id, pid) VALUES ('r{i}', 'r{}')",
                    i - 1
                ))
                .await
                .expect("INSERT depth row");
        }
        sleep(Duration::from_millis(300)).await;

        let err = client
            .simple_query(&format!("DELETE FROM {root} WHERE id = 'r0'"))
            .await
            .expect_err("over-deep CASCADE must trip the depth guard");
        let msg = err
            .as_db_error()
            .map(|d| d.message().to_string())
            .unwrap_or_else(|| err.to_string());
        assert!(
            msg.contains("depth"),
            "expected a depth-guard error, got: {msg}"
        );
    }

    // ── Section 4: concurrency — a child INSERT racing a parent DELETE
    //    (RESTRICT) can never orphan. The DELETE holds the non-reentrant
    //    in-process op-lock on the transitive child set for its whole critical
    //    section (cascade scan → parent tombstone), so the INSERT serializes
    //    behind it. The durable DML lock alone is pod-level + re-entrant and
    //    would NOT serialize two connections on this single embedded server;
    //    the op-lock (`TableOpLockRegistry`, shared via the CatalogManager) is
    //    what closes the intra-pod cross-table TOCTOU. ──
    {
        // A second connection gives true intra-pod concurrency (a single
        // tokio_postgres client serializes its own statements).
        let (client_b, connection_b) =
            tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
                .await
                .expect("connect client_b");
        tokio::spawn(async move {
            if let Err(e) = connection_b.await {
                eprintln!("pgwire connection_b error: {e}");
            }
        });

        const ITERS: usize = 8;
        for i in 0..ITERS {
            let parent = format!("cc_par_{suffix}_{i}");
            let child = format!("cc_chl_{suffix}_{i}");
            client
                .simple_query(&format!(
                    "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
                ))
                .await
                .expect("CREATE concurrency parent");
            client
                .simple_query(&format!(
                    "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
                     FOREIGN KEY (pid) REFERENCES {parent}(id))"
                ))
                .await
                .expect("CREATE concurrency child");
            client
                .simple_query(&format!(
                    "INSERT INTO {parent} (id, name) VALUES ('p1', 'root')"
                ))
                .await
                .expect("INSERT concurrency parent");
            sleep(Duration::from_millis(150)).await; // let p1 become visible

            // Race: A deletes the referenced parent; B inserts a child row
            // referencing it. Either may win — A wins ⇒ parent deleted, B's
            // INSERT then fails the FK check; B wins ⇒ child committed, A's
            // DELETE is RESTRICT-blocked — but the result can never be an orphan
            // (a child row whose parent was deleted underneath it). Both calls
            // tolerate an error; the invariant is asserted below.
            let del_sql = format!("DELETE FROM {parent} WHERE id = 'p1'");
            let ins_sql = format!("INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')");
            let a = client.simple_query(&del_sql);
            let b = client_b.simple_query(&ins_sql);
            let _ = tokio::join!(a, b);

            let parent_rows = client
                .simple_query(&format!("SELECT id FROM {parent}"))
                .await
                .expect("SELECT concurrency parent");
            let parent_has_p1 = col_set(&parent_rows, "id").contains("p1");
            let child_rows = client
                .simple_query(&format!("SELECT id FROM {child}"))
                .await
                .expect("SELECT concurrency child");
            let child_has_c1 = col_set(&child_rows, "id").contains("c1");

            // No orphan: if the child row committed, its parent must survive.
            assert!(
                !child_has_c1 || parent_has_p1,
                "orphaned child: {child}.c1 references {parent}.p1 which was deleted \
                 (iteration {i}); the cascade op-lock must serialize the INSERT behind \
                 the DELETE"
            );
        }
    }
}

/// TD-110 S2: composite (multi-column) FOREIGN KEY enforcement over pgwire —
/// RESTRICT / CASCADE / SET NULL on a composite-PK parent, plus MATCH-SIMPLE
/// NULL-exempt inserts and dangling-tuple rejection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_enforces_composite_fk_referential_actions() {
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

    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = format!("cmpar_{suffix}");
    // Composite-PK parent: PRIMARY KEY (region, pid).
    client
        .simple_query(&format!(
            "CREATE TABLE {parent} (region VARCHAR NOT NULL, pid VARCHAR NOT NULL, name VARCHAR, \
             PRIMARY KEY (region, pid))"
        ))
        .await
        .expect("CREATE composite parent");
    client
        .simple_query(&format!(
            "INSERT INTO {parent} (region, pid, name) VALUES \
             ('us', 'p1', 'Al'), ('eu', 'p2', 'Bo'), ('ap', 'p3', 'Cy')"
        ))
        .await
        .expect("INSERT composite parents");

    // ---- RESTRICT (default NO ACTION): deleting a referenced composite-PK row is rejected. ----
    {
        let child = format!("cmchl_restr_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, c_region VARCHAR, c_pid VARCHAR, \
                 FOREIGN KEY (c_region, c_pid) REFERENCES {parent} (region, pid))"
            ))
            .await
            .expect("CREATE restrict child");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, c_region, c_pid) VALUES ('r1', 'us', 'p1')"
            ))
            .await
            .expect("INSERT restrict child");
        sleep(Duration::from_millis(300)).await;

        let _ = client
            .simple_query(&format!(
                "DELETE FROM {parent} WHERE region = 'us' AND pid = 'p1'"
            ))
            .await
            .expect_err("ON DELETE NO ACTION must reject deleting a composite-PK parent");

        let parent_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {parent}"))
            .await
            .expect("COUNT composite parent (restrict)");
        assert_eq!(
            count_star(&parent_rows),
            3,
            "composite parent must survive a rejected DELETE"
        );
        let child_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {child}"))
            .await
            .expect("COUNT restrict child");
        assert_eq!(count_star(&child_rows), 1, "restrict child must survive");
    }

    // ---- CASCADE: deleting the parent removes the referencing composite-FK child. ----
    {
        let child = format!("cmchl_casc_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, c_region VARCHAR, c_pid VARCHAR, \
                 FOREIGN KEY (c_region, c_pid) REFERENCES {parent} (region, pid) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE cascade child");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, c_region, c_pid) VALUES ('c2', 'eu', 'p2')"
            ))
            .await
            .expect("INSERT cascade child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!(
                "DELETE FROM {parent} WHERE region = 'eu' AND pid = 'p2'"
            ))
            .await
            .expect("composite CASCADE delete must succeed");

        let child_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {child}"))
            .await
            .expect("COUNT cascade child");
        assert_eq!(
            count_star(&child_rows),
            0,
            "composite CASCADE must remove the child"
        );
    }

    // ---- SET NULL: deleting the parent nulls BOTH FK columns on the child. ----
    {
        let child = format!("cmchl_null_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, c_region VARCHAR, c_pid VARCHAR, \
                 FOREIGN KEY (c_region, c_pid) REFERENCES {parent} (region, pid) ON DELETE SET NULL)"
            ))
            .await
            .expect("CREATE set-null child");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, c_region, c_pid) VALUES ('s3', 'ap', 'p3')"
            ))
            .await
            .expect("INSERT set-null child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!(
                "DELETE FROM {parent} WHERE region = 'ap' AND pid = 'p3'"
            ))
            .await
            .expect("composite SET NULL delete must succeed");

        let child_rows = client
            .simple_query(&format!("SELECT id, c_region, c_pid FROM {child}"))
            .await
            .expect("SELECT set-null child");
        assert_eq!(
            col_set(&child_rows, "id"),
            ["s3".to_string()].into_iter().collect::<BTreeSet<_>>(),
            "composite SET NULL must keep the child row"
        );
        // Both FK columns cleared.
        assert_eq!(
            scalar(&child_rows, "c_region"),
            None,
            "c_region must be NULL after SET NULL"
        );
        assert_eq!(
            scalar(&child_rows, "c_pid"),
            None,
            "c_pid must be NULL after SET NULL"
        );
    }

    // ---- MATCH SIMPLE: a NULL FK column exempts the row (insert succeeds with no parent). ----
    {
        let child = format!("cmchl_nullfk_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, c_region VARCHAR, c_pid VARCHAR, \
                 FOREIGN KEY (c_region, c_pid) REFERENCES {parent} (region, pid))"
            ))
            .await
            .expect("CREATE match-simple child");
        // Either FK column NULL → exempt → no parent required.
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, c_region, c_pid) VALUES ('n1', NULL, 'p1')"
            ))
            .await
            .expect("MATCH SIMPLE: a NULL FK column exempts the row");
    }

    // ---- Dangling composite tuple: referencing a non-existent parent is rejected. ----
    {
        let child = format!("cmchl_dangling_{suffix}");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, c_region VARCHAR, c_pid VARCHAR, \
                 FOREIGN KEY (c_region, c_pid) REFERENCES {parent} (region, pid))"
            ))
            .await
            .expect("CREATE dangling child");
        let _ = client
            .simple_query(&format!(
                "INSERT INTO {child} (id, c_region, c_pid) VALUES ('d1', 'xx', 'zz')"
            ))
            .await
            .expect_err("a dangling composite FK tuple must be rejected");
    }
}

/// TD-110 S3: cross-namespace FOREIGN KEY over pgwire — a child in namespace B
/// whose FK references a parent in namespace A (`REFERENCES a.parent`) is found
/// by ON DELETE child discovery: CASCADE removes it, RESTRICT rejects the parent
/// delete, SET NULL nulls the FK.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_enforces_cross_namespace_fk_referential_actions() {
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

    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let ns_a = format!("xnsa_{suffix}");
    let ns_b = format!("xnsb_{suffix}");
    let parent = format!("{ns_a}.parent");
    client
        .simple_query(&format!("CREATE NAMESPACE {ns_a}"))
        .await
        .expect("CREATE NAMESPACE a");
    client
        .simple_query(&format!("CREATE NAMESPACE {ns_b}"))
        .await
        .expect("CREATE NAMESPACE b");
    client
        .simple_query(&format!(
            "CREATE TABLE {parent} (id VARCHAR PRIMARY KEY, name VARCHAR)"
        ))
        .await
        .expect("CREATE cross-ns parent");

    // ---- CASCADE: child in ns_b references the parent in ns_a. ----
    {
        let child = format!("{ns_b}.chl_casc");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
                 FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE CASCADE)"
            ))
            .await
            .expect("CREATE cross-ns cascade child");
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p1', 'Al')"
            ))
            .await
            .expect("INSERT cross-ns parent");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('c1', 'p1')"
            ))
            .await
            .expect("INSERT cross-ns cascade child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p1'"))
            .await
            .expect("cross-ns CASCADE delete must succeed");

        let rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {child}"))
            .await
            .expect("COUNT cross-ns cascade child");
        assert_eq!(
            count_star(&rows),
            0,
            "cross-ns CASCADE must remove the child in the other namespace"
        );
    }

    // ---- RESTRICT: a child in ns_b blocks deleting the parent in ns_a. ----
    {
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p2', 'Bo')"
            ))
            .await
            .expect("INSERT parent p2");
        let child = format!("{ns_b}.chl_restr");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
                 FOREIGN KEY (pid) REFERENCES {parent}(id))"
            ))
            .await
            .expect("CREATE cross-ns restrict child");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('r2', 'p2')"
            ))
            .await
            .expect("INSERT cross-ns restrict child");
        sleep(Duration::from_millis(300)).await;

        let _ = client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p2'"))
            .await
            .expect_err("cross-ns RESTRICT must reject the parent delete");

        let parent_rows = client
            .simple_query(&format!("SELECT COUNT(*) AS n FROM {parent}"))
            .await
            .expect("COUNT parent (restrict)");
        assert_eq!(
            count_star(&parent_rows),
            1,
            "parent must survive a cross-ns RESTRICT-rejected delete"
        );
    }

    // ---- SET NULL: deleting the parent nulls the FK on the ns_b child. ----
    {
        client
            .simple_query(&format!(
                "INSERT INTO {parent} (id, name) VALUES ('p3', 'Cy')"
            ))
            .await
            .expect("INSERT parent p3");
        let child = format!("{ns_b}.chl_null");
        client
            .simple_query(&format!(
                "CREATE TABLE {child} (id VARCHAR PRIMARY KEY, pid VARCHAR, \
                 FOREIGN KEY (pid) REFERENCES {parent}(id) ON DELETE SET NULL)"
            ))
            .await
            .expect("CREATE cross-ns set-null child");
        client
            .simple_query(&format!(
                "INSERT INTO {child} (id, pid) VALUES ('s3', 'p3')"
            ))
            .await
            .expect("INSERT cross-ns set-null child");
        sleep(Duration::from_millis(300)).await;

        client
            .simple_query(&format!("DELETE FROM {parent} WHERE id = 'p3'"))
            .await
            .expect("cross-ns SET NULL delete must succeed");

        let rows = client
            .simple_query(&format!("SELECT id, pid FROM {child}"))
            .await
            .expect("SELECT cross-ns set-null child");
        assert_eq!(
            col_set(&rows, "id"),
            ["s3".to_string()].into_iter().collect::<BTreeSet<_>>(),
            "cross-ns SET NULL must keep the child row"
        );
        assert_eq!(
            scalar(&rows, "pid"),
            None,
            "cross-ns SET NULL must clear the child FK"
        );
    }
}

// Risk-coverage scenarios (docs/10-quality/RISK_CONTRACT.toml :: referential-integrity):
//   scenario: fk-restrict       — ON DELETE NO ACTION/RESTRICT rejects + parent survives
//   scenario: fk-cascade        — ON DELETE CASCADE removes the child
//   scenario: fk-set-null       — ON DELETE SET NULL clears the FK columns
//   scenario: fk-null-exempt    — MATCH SIMPLE: a NULL FK column exempts the row
//   scenario: fk-dangling-reject — INSERT referencing a missing parent is rejected
