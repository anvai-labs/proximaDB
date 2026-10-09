//! pgwire NULL rendering (end-to-end).
//!
//! SQL NULL must reach the client as a real NULL (pgwire `-1` length sentinel),
//! not an empty string. This exercises the simple-query text path: a NULL column
//! value renders as `None` to tokio-postgres, `IS NULL`/`IS NOT NULL` select on
//! it, and `ORDER BY` places NULLs last (ASC).

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

/// First value of `col`, in row order (None for SQL NULL).
fn scalar(messages: &[SimpleQueryMessage], col: &str) -> Option<String> {
    messages.iter().find_map(|msg| match msg {
        SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
        _ => None,
    })
}

/// All values of `col`, in row order (None entries for SQL NULL).
fn col_ordered(messages: &[SimpleQueryMessage], col: &str) -> Vec<Option<String>> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => Some(row.get(col).map(|s| s.to_string())),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_renders_sql_null_over_simple_query() {
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
    let t = format!("tnull_{suffix}");

    client
        .simple_query(&format!(
            "CREATE TABLE {t} (id BIGINT PRIMARY KEY, name VARCHAR)"
        ))
        .await
        .expect("CREATE");
    // Row 2 has an explicit NULL in `name`; rows 1 and 3 have values.
    client
        .simple_query(&format!(
            "INSERT INTO {t} (id, name) VALUES (1, 'alice'), (2, NULL), (3, 'bob')"
        ))
        .await
        .expect("INSERT");
    sleep(Duration::from_millis(400)).await;

    // (1) SELECTing a NULL column must yield SQL NULL (None), not "".
    let null_row = client
        .simple_query(&format!("SELECT name FROM {t} WHERE id = 2"))
        .await
        .expect("SELECT null row");
    assert_eq!(
        scalar(&null_row, "name"),
        None,
        "NULL column must render as SQL NULL (None), not an empty string"
    );
    // A non-NULL value still renders normally.
    let val_row = client
        .simple_query(&format!("SELECT name FROM {t} WHERE id = 1"))
        .await
        .expect("SELECT val row");
    assert_eq!(scalar(&val_row, "name").as_deref(), Some("alice"));

    // (2) WHERE col IS NULL matches the NULL row only.
    let is_null_rows = client
        .simple_query(&format!("SELECT id FROM {t} WHERE name IS NULL"))
        .await
        .expect("SELECT IS NULL");
    let is_null_ids: Vec<String> = col_ordered(&is_null_rows, "id")
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(
        is_null_ids,
        vec!["2".to_string()],
        "IS NULL must match only the NULL row"
    );

    // (3) WHERE col IS NOT NULL excludes the NULL row.
    let not_null_rows = client
        .simple_query(&format!(
            "SELECT id FROM {t} WHERE name IS NOT NULL ORDER BY id"
        ))
        .await
        .expect("SELECT IS NOT NULL");
    let not_null_ids: Vec<String> = col_ordered(&not_null_rows, "id")
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(
        not_null_ids,
        vec!["1".to_string(), "3".to_string()],
        "IS NOT NULL must exclude the NULL row"
    );

    // (4) ORDER BY over a column containing NULL must succeed and return every
    //     row. (Strict NULLS-LAST ordering is asserted on the protocol.rs
    //     in-memory sort path, which is already null-aware; the relational
    //     pipeline's own ORDER BY lowering is a separate path and out of scope
    //     for this NULL-*rendering* fix.)
    // TD-185b: the sort column must be in the projection — the legacy reader
    // orders only projected columns and now FAILS CLOSED on an unresolvable
    // ORDER BY instead of silently returning storage order.
    let ordered_rows = client
        .simple_query(&format!("SELECT id, name FROM {t} ORDER BY name ASC"))
        .await
        .expect("SELECT ORDER BY");
    let ordered_ids: Vec<String> = col_ordered(&ordered_rows, "id")
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(
        ordered_ids.len(),
        3,
        "ORDER BY over a NULL-containing column must return all rows"
    );
}
