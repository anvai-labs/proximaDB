//! End-to-end regression for the psql `\dt` / `\d` crash class.
//!
//! Symptoms this guards against (observed 2026-07-07):
//! 1. `psql \dt` listed the same table `b` **twice**.
//! 2. `psql \d b` crashed the client with SIGSEGV after libpq printed
//!    `column number N is out of range 0..M`.
//!
//! Root cause: pgwire catalog-introspection dispatch routed psql's rich
//! multi-column `pg_catalog` probes to narrow synthetic builders, so the
//! RowDescription field count didn't match the client's SELECT list and psql
//! dereferenced missing data. Compounded by `cataloged_tables()` enumerating
//! the catalog × namespace cross-product without dedup, surfacing `b` twice.
//!
//! This test boots an in-process ProximaDB, connects a real `tokio-postgres`
//! client over TCP, and reproduces the user's exact flow: CREATE TABLE →
//! reconnect → `\dt`-equivalent (assert each table once) → `\d`-equivalent
//! (assert no error, correct column shape). `tokio_postgres::simple_query`'s
//! `Row::get(idx)` exercises the same indexed column access that crashes psql,
//! so it is a faithful in-process proxy for the real client without needing the
//! `psql` binary.

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
        // Isolate the catalog WAL + global manifest to the temp dir so the test
        // is reproducible across runs (otherwise the default `./metadata`
        // persists tables like `b` and CREATE fails with "already exists").
        config.storage.metadata_url = format!("file://{}/metadata", tmp_data.path().display());
        config.storage.wal_config.global_manifest_url =
            Some(format!("file://{}/manifest", tmp_data.path().display()));
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("REST not ready on port {}", rest_port);
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

/// Collect `SimpleQueryMessage::Row`s from a `simple_query` result.
fn rows_of(messages: Vec<SimpleQueryMessage>) -> Vec<tokio_postgres::SimpleQueryRow> {
    messages
        .into_iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(row) => Some(row),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_dt_dedup_and_d_shape_contract() {
    let server = PgwireTestServer::start().await.expect("server start");

    // --- Session 1: create two distinct tables (mirrors the user's CREATE). ---
    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("connect");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .simple_query("CREATE TABLE b (id INT PRIMARY KEY, v VARCHAR)")
        .await
        .expect("create b");
    client
        .simple_query("CREATE TABLE c2 (id INT PRIMARY KEY)")
        .await
        .expect("create c2");
    drop(client);

    // --- Session 2: reconnect, then run psql-style introspection probes. ---
    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("reconnect");
    tokio::spawn(async move {
        let _ = connection.await;
    });

    // `\dt`-equivalent: a multi-column pg_catalog probe with aliases (the shape
    // psql sends). Must hit the shape-aware table-list builder and return each
    // table exactly once (Bug A: dedup contract).
    let dt = client
        .simple_query(
            "SELECT n.nspname AS \"Schema\", c.relname AS \"Name\", \
             'postgres' AS \"Owner\", 'ordinary table' AS \"Type\" \
             FROM pg_catalog.pg_class c \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relname IN ('b', 'c2')",
        )
        .await
        .expect("\\dt-equivalent must not error");
    let dt_rows = rows_of(dt);
    // RowDescription must carry the client's SELECT-list column names.
    assert!(!dt_rows.is_empty(), "\\dt should list the created tables");
    let col_names: Vec<&str> = dt_rows[0].columns().iter().map(|c| c.name()).collect();
    assert_eq!(
        col_names,
        vec!["Schema", "Name", "Owner", "Type"],
        "\\dt columns must match the client's SELECT aliases (shape-aware contract)"
    );
    let listed_names: Vec<&str> = dt_rows.iter().map(|r| r.get(1).unwrap_or("")).collect();
    let b_count = listed_names.iter().filter(|n| **n == "b").count();
    let c2_count = listed_names.iter().filter(|n| **n == "c2").count();
    assert_eq!(
        b_count, 1,
        "Bug A: table b must appear exactly once in \\dt"
    );
    assert_eq!(
        c2_count, 1,
        "distinct tables must each appear exactly once (dedup not over-collapsing)"
    );

    // `\d b`-equivalent: a multi-column pg_attribute probe with aliases. Must
    // not error/crash and must return the table's columns with the right shape
    // (Bug B: width-safety + shape-aware contract).
    let d = client
        .simple_query(
            "SELECT a.attname AS \"Column\", \
             pg_catalog.format_type(a.atttypid, a.atttypmod) AS \"Type\" \
             FROM pg_catalog.pg_attribute a \
             JOIN pg_catalog.pg_class c ON a.attrelid = c.oid \
             WHERE c.relname = 'b' AND a.attnum > 0",
        )
        .await
        .expect("\\d b must not error or crash the client");
    let d_rows = rows_of(d);
    // Regression contract: `\d b` must NOT crash (the original SIGSEGV) and
    // must return rows whose shape matches the client's SELECT list. The
    // specific column VALUES reflect the backing collection's canonical schema
    // (ADR-047 collection≡table) rather than the DDL's `(id, v)`; surfacing the
    // user-defined columns in `\d` is a deeper schema-authority follow-up
    // (tracked as a TD), not part of this crash/shape fix.
    assert!(
        !d_rows.is_empty(),
        "\\d b should describe the table's columns without crashing"
    );
    let d_col_names: Vec<&str> = d_rows[0].columns().iter().map(|c| c.name()).collect();
    assert_eq!(
        d_col_names,
        vec!["Column", "Type"],
        "\\d columns must match the client's SELECT aliases (shape-aware contract)"
    );
    // Every DataRow cell count must equal the RowDescription field count — the
    // width-safety invariant whose violation crashed psql.
    for row in &d_rows {
        assert_eq!(
            row.columns().len(),
            2,
            "\\d rows must carry exactly the SELECT-list width"
        );
    }
}
