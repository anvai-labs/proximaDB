//! End-to-end fp16 collection validation through the **real pgwire
//! transport** — boots an in-process ProximaDB server with a free
//! pgwire port, connects via a real `tokio-postgres` client over TCP,
//! sends `CREATE TABLE ... WITH (canonical_embedding_precision='fp16')`,
//! and verifies the catalog row reflects fp16.
//!
//! Closes the transport gap that `services::ddl::tests` exercises at
//! the DDL-service level. This test proves:
//!
//! 1. The pgwire startup + auth handshake reaches the server
//! 2. The SQL parser extracts the `WITH (canonical_embedding_precision)`
//!    table-property
//! 3. The DDL service's `build_catalog_schema` reads the property and
//!    writes it to the catalog row
//! 4. Subsequent introspection (a follow-up SELECT or REST GET) reflects
//!    the fp16 precision
//!
//! Why a separate fixture instead of extending `fp16_network_e2e.rs`:
//! the pgwire listener needs its own port-allocator + tokio-postgres
//! dev-dep + careful await ordering for the asynchronous wire-protocol
//! handshake. Keeping it in its own file keeps the REST/gRPC fixture
//! focused.

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

/// In-process test server holding a `ProximaDB` instance bound to a
/// free pgwire port (plus a REST port so the existing `/health` probe
/// can report readiness — the pgwire listener doesn't have an
/// equivalent simple probe). Drop signals shutdown.
struct PgwireTestServer {
    pg_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl PgwireTestServer {
    async fn start() -> anyhow::Result<Self> {
        // SCHEMA_V2 enables the v2 WAL serializer that accepts non-fp32
        // records. For DDL-only tests this isn't strictly needed, but
        // we set it so a follow-up INSERT path (when added) doesn't
        // surprise the test with a v1 refuse-error.
        unsafe {
            std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true");
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
        // The new field — drives the pgwire bind override in database.rs.
        config.api.pg_port = Some(pg_port);
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        // Wait for REST /health to confirm boot, then assume pgwire is
        // up too (they share the same `start()` await; both listeners
        // bind before `start()` returns).
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
                        anyhow::bail!(
                            "REST server didn't become ready on port {} within 15s",
                            rest_port
                        );
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }

        // Give pgwire's TCP listener a brief moment after REST is up.
        // The startup path binds both before returning from start(),
        // but the OS may briefly have one bound and the other not
        // yet `accept`-ready.
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            rest_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn pg_connection_string(&self) -> String {
        // sslmode=disable: no TLS on the test fixture
        // user=postgres: the pgwire impl accepts any user in the
        // permissive default; we use the standard "postgres" name.
        format!(
            "host=127.0.0.1 port={} user=postgres dbname=proximadb sslmode=disable",
            self.pg_port
        )
    }

    fn rest_base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
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

/// Connect via a real Postgres wire-protocol client and verify that a
/// `CREATE TABLE ... WITH (canonical_embedding_precision = 'fp16')`
/// statement sent over TCP lands on the catalog with the right
/// precision. Closes the transport-level gap that the DDL-service
/// unit tests in `src/services/ddl/mod.rs` already cover at the
/// in-process API level.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_create_table_with_canonical_embedding_precision_fp16() {
    let server = PgwireTestServer::start().await.expect("server start");

    // Connect via tokio-postgres. `connect` returns the client and a
    // future that drives the connection — spawn it onto the runtime.
    let (client, connection) =
        tokio_postgres::connect(&server.pg_connection_string(), tokio_postgres::NoTls)
            .await
            .expect("tokio-postgres connect");

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });

    let table_name = format!(
        "pgwire_fp16_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // Send CREATE TABLE over the wire. `simple_query` runs the whole
    // statement in one round-trip (no prepare/bind/execute pipelining
    // — DDL doesn't need parameters).
    let create_sql = format!(
        "CREATE TABLE {} (id BIGINT PRIMARY KEY) WITH (canonical_embedding_precision = 'fp16')",
        table_name
    );
    client
        .simple_query(&create_sql)
        .await
        .expect("pgwire CREATE TABLE");

    // Verify via REST: the GET path reads canonical_embedding_precision
    // back from the catalog schema (manager.rs ~line 1950). The pgwire
    // path doesn't have a SQL introspection of this field today, so
    // we cross-protocol verify through REST. Both protocols hit the
    // same catalog row, so this is a meaningful end-to-end assertion
    // even though it spans two protocols on the read side.
    //
    // The pgwire DDL path materialises the catalog row in the "default"
    // namespace (services/collection/manager.rs::collection_table_identifier).
    // REST GET /api/v1/collections/<name> resolves the same identifier.
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build()
        .unwrap();
    let get_url = format!(
        "{}/api/v2/collections/{}",
        server.rest_base_url(),
        table_name
    );
    // Read as the SAME tenant the pgwire session wrote as: the startup
    // `database` (here `proximadb`) is the pgwire tenant boundary, and a REST
    // request with no header would act as the default tenant — a different
    // tenant, which reads this table as missing.
    let resp = http_client
        .get(&get_url)
        .header("X-Tenant-ID", "proximadb")
        .send()
        .await
        .expect("REST GET collection");
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    assert!(
        status.is_success(),
        "REST GET after pgwire CREATE TABLE failed: status={status}, body={body}"
    );

    // v2 GET surfaces `canonical_embedding_precision` as a top-level field;
    // accept the legacy nested `collection.config` / `config` shapes too.
    let precision = body
        .get("canonical_embedding_precision")
        .or_else(|| {
            body.get("collection")
                .and_then(|c| c.get("config"))
                .or_else(|| body.get("config"))
                .and_then(|cfg| cfg.get("canonical_embedding_precision"))
        })
        .unwrap_or_else(|| {
            panic!(
                "expected response to expose canonical_embedding_precision; \
                 actual body shape: {body}"
            )
        });

    // proto-serde renders the enum as either the SCREAMING string or
    // the numeric discriminant. Fp16 = 2.
    let matches_fp16 = match precision {
        serde_json::Value::String(s) => {
            s == "EMBEDDING_PRECISION_FP16" || s == "FP16" || s == "Fp16" || s == "fp16"
        }
        serde_json::Value::Number(n) => n.as_i64() == Some(2),
        _ => false,
    };
    assert!(
        matches_fp16,
        "pgwire CREATE TABLE WITH (canonical_embedding_precision='fp16') \
         must persist as Fp16 in the catalog row; REST GET returned: {precision:?}"
    );
}
