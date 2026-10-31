//! pgwire relational SELECT — OR / mixed-AND-OR WHERE push-down, end-to-end.
//!
//! Proves the contract that the round-2 audit deferred (see
//! `pgwire_insert_props_e2e.rs:212` — "the sharper SQL-side assertion (SELECT
//! round-trip) is a follow-up"): a `SELECT … WHERE a OR b` sent over the
//! PostgreSQL wire returns the correct row UNION over data that was inserted
//! through the same pgwire path.
//!
//! This exercises the legacy relational SELECT path (`execute_relational_query`)
//! at the DEFAULT config: the new relational pipeline is enabled by default but
//! reads a separate, unpopulated in-memory engine, so for a pgwire-created table
//! it can't resolve the table and falls through to the legacy path. The legacy
//! path now converges its WHERE evaluation onto the same boolean predicate tree
//! UPDATE/DELETE use, so OR / mixed-AND-OR / grouped predicates push into the
//! record scan instead of degrading to a full scan with same-column-equality-only
//! `IN` folding.

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
        // Pin the relational SELECT to the LEGACY path (`execute_relational_query`)
        // that this slice modified. At default config the new pipeline already
        // falls through for pgwire-created tables, but forcing it off keeps the
        // test deterministic and immune to a future PATH B that reads real data.
        unsafe {
            std::env::set_var("PROXIMADB_PGWIRE_RELATIONAL_PIPELINE", "0");
        }

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

/// Collect the `id` column from a `simple_query` result into a sorted set.
fn ids(messages: &[SimpleQueryMessage]) -> BTreeSet<String> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get("id").map(|s| s.to_string()),
            _ => None,
        })
        .collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// Collect the `id` column preserving server row order (for ORDER BY asserts).
fn ids_ordered(messages: &[SimpleQueryMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|msg| match msg {
            SimpleQueryMessage::Row(row) => row.get("id").map(|s| s.to_string()),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_select_or_predicate_returns_union_over_real_data() {
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

    let table = format!(
        "pgw_or_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    client
        .simple_query(&format!(
            "CREATE TABLE {table} (id VARCHAR PRIMARY KEY, status VARCHAR, qty BIGINT)"
        ))
        .await
        .expect("pgwire CREATE TABLE");

    for (id, status, qty) in [
        ("i1", "active", 5),
        ("i2", "active", 15),
        ("i3", "idle", 25),
        ("i4", "idle", 35),
    ] {
        client
            .simple_query(&format!(
                "INSERT INTO {table} (id, status, qty) VALUES ('{id}', '{status}', {qty})"
            ))
            .await
            .expect("pgwire INSERT");
    }

    // Let the canonical write path flush through WAL + delta-merge.
    sleep(Duration::from_millis(500)).await;

    // (1) OR union: status='active' OR qty >= 30 → i1, i2 (active) + i4 (35).
    // Pre-change, this WHERE could not fold to a same-column-equality IN and so
    // ran a full scan; the result set is the discriminator either way, but it
    // now pushes the real boolean predicate into the scan.
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {table} WHERE status = 'active' OR qty >= 30"
        ))
        .await
        .expect("pgwire SELECT OR");
    assert_eq!(ids(&rows), set(&["i1", "i2", "i4"]), "OR union");

    // (2) PK leaf under OR must NOT shortcut to a point lookup that drops the
    // other branch: id='i2' OR status='idle' → i2, i3, i4.
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {table} WHERE id = 'i2' OR status = 'idle'"
        ))
        .await
        .expect("pgwire SELECT PK-under-OR");
    assert_eq!(
        ids(&rows),
        set(&["i2", "i3", "i4"]),
        "PK leaf under OR is not a point lookup"
    );

    // (3) Nested grouping must NOT flatten: status='idle' AND (qty < 30 OR
    // id='i1') → i3 only. Flattening to `idle AND qty<30 AND id='i1'` would
    // wrongly return zero rows.
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {table} WHERE status = 'idle' AND (qty < 30 OR id = 'i1')"
        ))
        .await
        .expect("pgwire SELECT nested");
    assert_eq!(ids(&rows), set(&["i3"]), "nested AND-of-OR not flattened");

    // (4) ORDER BY + LIMIT over an OR predicate must sort THEN truncate: the OR
    // matches {i1, i2, i4}; ordered by id ascending and capped at 2 → [i1, i2].
    // (If the limit were pushed before the sort, the first 2 scanned rows could
    // differ.)
    let rows = client
        .simple_query(&format!(
            "SELECT id FROM {table} WHERE status = 'active' OR qty >= 30 ORDER BY id LIMIT 2"
        ))
        .await
        .expect("pgwire SELECT OR + ORDER BY + LIMIT");
    assert_eq!(
        ids_ordered(&rows),
        vec!["i1".to_string(), "i2".to_string()],
        "ORDER BY id LIMIT 2 over OR → sort-then-truncate"
    );
}
