//! TD-RELIDEQ-1: `WHERE id = <value>` (or `record_id`) on a pgwire table with no
//! declared primary key returned no rows, and UPDATE / DELETE with the same
//! predicate reported 0 rows and changed nothing.
//!
//! Cause: with no declared key, `schema_primary_key_column` falls back to a
//! column named `id` or `record_id`, and the DML key fast path looks the value
//! up as the record `oid`. A keyless pgwire row's `oid` is generated, so every
//! lookup missed and the fast path returned the empty result without scanning.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use tempfile::TempDir;
use tokio::time::sleep;
use tokio_postgres::{Client, SimpleQueryMessage};

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

/// Render a tokio_postgres error including the real server-side DbError cause.
fn explain_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!("[{}] {}", db.code().code(), db.message())
    } else {
        e.to_string()
    }
}

async fn exec(client: &Client, sql: &str) {
    client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {}", explain_err(&e)));
}

/// The first column of every returned row, as text.
async fn column(client: &Client, sql: &str) -> Vec<String> {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)));
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => r.get(0).map(str::to_string),
            _ => None,
        })
        .collect()
}

/// Rows affected by a DML statement.
async fn affected(client: &Client, sql: &str) -> u64 {
    let messages = client
        .simple_query(sql)
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {}", explain_err(&e)));
    messages
        .iter()
        .find_map(|m| match m {
            SimpleQueryMessage::CommandComplete(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0)
}

async fn connect(server: &PgServer) -> Client {
    let (client, conn) = tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("pgwire connection error: {e}");
        }
    });
    client
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keyless_table_id_equality_reads_and_writes_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;

    for key in ["id", "record_id"] {
        let table = format!("keyless_{key}");
        exec(
            &client,
            &format!("CREATE TABLE {table} ({key} INT, name TEXT)"),
        )
        .await;
        exec(
            &client,
            &format!("INSERT INTO {table} VALUES (1, 'a'), (2, 'b'), (3, 'c')"),
        )
        .await;

        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 2")
            )
            .await,
            vec!["b".to_string()],
            "{table}: equality must return the row"
        );
        let mut names = column(
            &client,
            &format!("SELECT name FROM {table} WHERE {key} IN (1, 3)"),
        )
        .await;
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "c".to_string()], "{table}: IN");
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 1 AND name = 'a'")
            )
            .await,
            vec!["a".to_string()],
            "{table}: equality under AND"
        );
        assert!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 99")
            )
            .await
            .is_empty(),
            "{table}: a missing key still returns nothing"
        );

        assert_eq!(
            affected(
                &client,
                &format!("UPDATE {table} SET name = 'z' WHERE {key} = 2")
            )
            .await,
            1,
            "{table}: UPDATE must affect the row"
        );
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 2")
            )
            .await,
            vec!["z".to_string()],
            "{table}: the update is visible"
        );
        assert_eq!(
            affected(&client, &format!("DELETE FROM {table} WHERE {key} = 1")).await,
            1,
            "{table}: DELETE must affect the row"
        );
        assert_eq!(
            affected(
                &client,
                &format!("UPDATE {table} SET {key} = 30 WHERE {key} = 3")
            )
            .await,
            1,
            "{table}: a keyless table's `{key}` is an ordinary column and can be updated"
        );
        assert_eq!(
            column(
                &client,
                &format!("SELECT name FROM {table} WHERE {key} = 30")
            )
            .await,
            vec!["c".to_string()],
            "{table}: the row is found by its new value"
        );
        let mut left = column(
            &client,
            &format!("SELECT name FROM {table} WHERE {key} >= 0"),
        )
        .await;
        left.sort();
        assert_eq!(
            left,
            vec!["c".to_string(), "z".to_string()],
            "{table}: rows after DELETE"
        );
    }
}

/// Control: with a declared primary key the key fast path is correct and must
/// keep serving equality, IN, UPDATE and DELETE.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn declared_primary_key_equality_still_works_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    exec(
        &client,
        "CREATE TABLE keyed (id INT PRIMARY KEY, name TEXT)",
    )
    .await;
    exec(&client, "INSERT INTO keyed VALUES (1, 'a'), (2, 'b')").await;
    assert_eq!(
        column(&client, "SELECT name FROM keyed WHERE id = 2").await,
        vec!["b".to_string()]
    );
    assert_eq!(
        affected(&client, "UPDATE keyed SET name = 'z' WHERE id = 2").await,
        1
    );
    assert_eq!(affected(&client, "DELETE FROM keyed WHERE id = 1").await, 1);
    assert_eq!(
        column(&client, "SELECT name FROM keyed WHERE id = 2").await,
        vec!["z".to_string()]
    );
}
