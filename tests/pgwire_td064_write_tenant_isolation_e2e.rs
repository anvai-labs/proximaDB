//! TD-064 — pgwire relational tenant isolation (write + read), end-to-end.
//!
//! Two pgwire connections with different startup `database` values are distinct
//! tenants (account=tenant=catalog). This proves the structural per-(tenant,
//! collection) isolation: each tenant's CREATE/INSERT/SELECT addresses its own
//! catalog schema row + record partition, the SAME table name and SAME primary
//! key coexist independently per tenant, and reads only ever return the
//! connecting tenant's rows. Backend-agnostic — the tenant shapes only the
//! logical partition/path, not any object-store scheme.

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
    rest_port: u16,
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
                        anyhow::bail!("REST server didn't become ready within 20s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(200)).await;

        Ok(Self {
            pg_port,
            rest_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    /// Connection string for a specific tenant (`database` = tenant/catalog).
    fn conn_string_for(&self, database: &str) -> String {
        format!(
            "host=127.0.0.1 port={} user=postgres dbname={} sslmode=disable",
            self.pg_port, database
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

/// Connect a client scoped to `database` (= tenant).
async fn connect(server: &PgwireTestServer, database: &str) -> tokio_postgres::Client {
    let (client, connection) =
        tokio_postgres::connect(&server.conn_string_for(database), tokio_postgres::NoTls)
            .await
            .expect("tokio-postgres connect");
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgwire connection error: {e}");
        }
    });
    client
}

/// First value of column `col` across the rows, as text.
fn scalar(messages: &[SimpleQueryMessage], col: &str) -> Option<String> {
    messages.iter().find_map(|msg| match msg {
        SimpleQueryMessage::Row(row) => row.get(col).map(|s| s.to_string()),
        _ => None,
    })
}

fn row_count(messages: &[SimpleQueryMessage]) -> usize {
    messages
        .iter()
        .filter(|m| matches!(m, SimpleQueryMessage::Row(_)))
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_relational_writes_and_reads_are_tenant_isolated() {
    let server = PgwireTestServer::start().await.expect("server start");
    let acme_tenant = format!("acmecorp_{}", server.pg_port);
    let globex_tenant = format!("globexco_{}", server.pg_port);
    let table = format!("orders_tbl_{}", server.pg_port);

    // Two tenants, distinguished only by the startup `database` (catalog) name.
    let acme = connect(&server, &acme_tenant).await;
    let globex = connect(&server, &globex_tenant).await;

    // Shared, identical table name across both tenants. Each CREATE addresses
    // its own tenant-prefixed catalog schema row (no cross-tenant collision).
    let ddl = format!("CREATE TABLE {table} (id TEXT NOT NULL, note TEXT, PRIMARY KEY (id));");
    acme.batch_execute(&ddl)
        .await
        .expect("acmecorp CREATE TABLE");
    globex
        .batch_execute(&ddl)
        .await
        .expect("globexco CREATE TABLE");

    // Same primary key value `1` inserted by BOTH tenants — must succeed
    // independently (distinct record partitions), not collide.
    acme.batch_execute(&format!(
        "INSERT INTO {table} (id, note) VALUES ('1', 'acmecorp-one'), ('2', 'acmecorp-two');"
    ))
    .await
    .expect("acmecorp INSERT");
    globex
        .batch_execute(&format!(
            "INSERT INTO {table} (id, note) VALUES ('1', 'globexco-one');"
        ))
        .await
        .expect("globexco INSERT (same PK as acme, different tenant)");

    // COUNT(*) (aggregate → relational pipeline) is tenant-scoped: each tenant
    // sees ONLY its own rows.
    let acme_count = acme
        .simple_query(&format!("SELECT COUNT(*) AS c FROM {table};"))
        .await
        .expect("acmecorp count");
    assert_eq!(
        scalar(&acme_count, "c").as_deref(),
        Some("2"),
        "acmecorp must see exactly its 2 rows"
    );
    let globex_count = globex
        .simple_query(&format!("SELECT COUNT(*) AS c FROM {table};"))
        .await
        .expect("globexco count");
    assert_eq!(
        scalar(&globex_count, "c").as_deref(),
        Some("1"),
        "globexco must see exactly its 1 row"
    );

    // Same PK `1` resolves to each tenant's OWN row (no cross-tenant bleed).
    let acme_one = acme
        .simple_query(&format!("SELECT note FROM {table} WHERE id = '1';"))
        .await
        .expect("acmecorp point read");
    assert_eq!(scalar(&acme_one, "note").as_deref(), Some("acmecorp-one"));
    let globex_one = globex
        .simple_query(&format!("SELECT note FROM {table} WHERE id = '1';"))
        .await
        .expect("globexco point read");
    assert_eq!(scalar(&globex_one, "note").as_deref(), Some("globexco-one"));

    // A full scan returns only the connecting tenant's rows.
    let acme_all = acme
        .simple_query(&format!("SELECT id FROM {table};"))
        .await
        .expect("acmecorp scan");
    assert_eq!(
        row_count(&acme_all),
        2,
        "acmecorp scan sees only acmecorp rows"
    );
    let globex_all = globex
        .simple_query(&format!("SELECT id FROM {table};"))
        .await
        .expect("globexco scan");
    assert_eq!(
        row_count(&globex_all),
        1,
        "globexco scan sees only globexco rows"
    );

    // DELETE is tenant-scoped: globex deleting its `1` must NOT affect acme's `1`.
    globex
        .batch_execute(&format!("DELETE FROM {table} WHERE id = '1';"))
        .await
        .expect("globexco DELETE");
    let acme_after = acme
        .simple_query(&format!("SELECT note FROM {table} WHERE id = '1';"))
        .await
        .expect("acmecorp read after globexco delete");
    assert_eq!(
        scalar(&acme_after, "note").as_deref(),
        Some("acmecorp-one"),
        "acme's row must survive globex's delete of the same PK"
    );
}

/// One backend message: (type byte, body).
async fn read_message(stream: &mut tokio::net::TcpStream) -> std::io::Result<(u8, Vec<u8>)> {
    use tokio::io::AsyncReadExt;
    let kind = stream.read_u8().await?;
    let len = stream.read_i32().await?;
    let mut body = vec![0u8; (len as usize).saturating_sub(4)];
    stream.read_exact(&mut body).await?;
    Ok((kind, body))
}

/// Drive `COPY … FROM STDIN` over the SIMPLE query protocol — what `psql \copy`
/// and most bulk loaders use, and the only COPY path this server implements
/// (tokio-postgres's `copy_in` goes through the extended protocol instead).
/// Returns the CommandComplete tag, or the server's error message.
async fn copy_csv_simple(
    server: &PgwireTestServer,
    database: &str,
    table: &str,
    rows: &str,
) -> Result<String, String> {
    use tokio::io::AsyncWriteExt;
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", server.pg_port))
        .await
        .map_err(|e| e.to_string())?;

    // StartupMessage (protocol 3.0) — trust auth on this test server.
    let mut startup = Vec::new();
    startup.extend_from_slice(&196_608_i32.to_be_bytes());
    for (key, value) in [("user", "postgres"), ("database", database)] {
        startup.extend_from_slice(key.as_bytes());
        startup.push(0);
        startup.extend_from_slice(value.as_bytes());
        startup.push(0);
    }
    startup.push(0);
    let mut frame = ((startup.len() + 4) as i32).to_be_bytes().to_vec();
    frame.extend_from_slice(&startup);
    stream.write_all(&frame).await.map_err(|e| e.to_string())?;
    loop {
        let (kind, body) = read_message(&mut stream).await.map_err(|e| e.to_string())?;
        match kind {
            b'Z' => break,
            b'E' => return Err(String::from_utf8_lossy(&body).into_owned()),
            _ => {}
        }
    }

    let query = format!("COPY {table} FROM STDIN WITH (FORMAT csv)\0");
    let mut frame = vec![b'Q'];
    frame.extend_from_slice(&((query.len() + 4) as i32).to_be_bytes());
    frame.extend_from_slice(query.as_bytes());
    stream.write_all(&frame).await.map_err(|e| e.to_string())?;
    let (kind, body) = read_message(&mut stream).await.map_err(|e| e.to_string())?;
    if kind != b'G' {
        return Err(format!(
            "expected CopyInResponse, got {:?}: {}",
            kind as char,
            String::from_utf8_lossy(&body)
        ));
    }

    let mut frame = vec![b'd'];
    frame.extend_from_slice(&((rows.len() + 4) as i32).to_be_bytes());
    frame.extend_from_slice(rows.as_bytes());
    frame.push(b'c');
    frame.extend_from_slice(&4_i32.to_be_bytes());
    stream.write_all(&frame).await.map_err(|e| e.to_string())?;

    let mut outcome = Err("connection closed before CommandComplete".to_string());
    loop {
        let (kind, body) = read_message(&mut stream).await.map_err(|e| e.to_string())?;
        match kind {
            b'C' => {
                outcome = Ok(String::from_utf8_lossy(&body)
                    .trim_end_matches('\0')
                    .to_string())
            }
            b'E' => outcome = Err(String::from_utf8_lossy(&body).into_owned()),
            b'Z' => break,
            _ => {}
        }
    }
    outcome
}

/// TD-PGWIRE-COPY-1: `COPY … FROM STDIN` is pgwire's bulk ingest for VECTOR
/// collections (rows are `id,<vector>`). It had no end-to-end coverage at all.
/// Now that it routes through the tenant-context insert and the primary-pod
/// gate, pin that an ordinary COPY still lands every parseable row (an
/// unparseable vector is skipped, as before) and that the rows are readable.
///
/// The collection is created over REST: a pgwire `CREATE TABLE … VECTOR(n)`
/// now parses as a RELATIONAL table whose backing collection is not an n-dim
/// vector collection, so COPY into it fails with a dimension mismatch — a
/// pre-existing gap recorded on TD-PGWIRE-COPY-1, not exercised here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_copy_from_stdin_ingests_vector_rows() {
    let server = PgwireTestServer::start().await.expect("server start");
    let collection = format!("embeds_coll_{}", server.pg_port);
    let rest = format!("http://127.0.0.1:{}", server.rest_port);
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("http client");

    let created = http
        .post(format!("{rest}/api/v2/collections"))
        .json(&serde_json::json!({
            "name": collection,
            "dimension": 3,
            "engine": "sst",
            "distance_metric": "cosine",
            "canonical_embedding_precision": "fp32",
            "enable_proxima_record": false
        }))
        .send()
        .await
        .expect("create collection");
    assert!(
        created.status().is_success(),
        "create collection: {}",
        created.text().await.unwrap_or_default()
    );

    // REST collections without a tenant header live in the default tenant;
    // COPY as that same tenant (startup database).
    let tag = copy_csv_simple(
        &server,
        proximadb_tenant::DEFAULT_TENANT,
        &collection,
        "v1,[0.1,0.2,0.3]\nv2,[0.4,0.5,0.6]\nbad,not-a-vector\n",
    )
    .await
    .expect("COPY must succeed");
    assert_eq!(
        tag, "COPY 2",
        "COPY must report the two parseable vector rows (the unparseable one is skipped)"
    );

    let read_vector = |id: &'static str| {
        let http = http.clone();
        let url = format!("{rest}/api/v2/collections/{collection}/records/{id}");
        async move {
            let read = http.get(url).send().await.expect("read record");
            if !read.status().is_success() {
                return None;
            }
            let body: serde_json::Value = read.json().await.expect("record json");
            Some(body["vector"].clone())
        }
    };
    let original = read_vector("v1")
        .await
        .expect("COPY row v1 must be readable");
    assert!(
        read_vector("v2").await.is_some(),
        "COPY row v2 must be readable"
    );

    // Insert-only, like INSERT: re-running a COPY that carries an existing id
    // must be REJECTED (an error, never a false `COPY n`) and must not
    // overwrite the stored row — an upsert path would silently replace v1.
    let rerun = copy_csv_simple(
        &server,
        proximadb_tenant::DEFAULT_TENANT,
        &collection,
        "v1,[0.9,0.9,0.9]\n",
    )
    .await;
    assert!(
        rerun
            .as_ref()
            .is_err_and(|error| error.contains("INSERT_CONFLICT")),
        "COPY of an existing id must be rejected as an insert conflict (not a dropped \
         connection), got {rerun:?}"
    );
    assert_eq!(
        read_vector("v1").await,
        Some(original),
        "a rejected COPY must leave the existing row unchanged"
    );

    // A duplicate id WITHIN one payload rejects the whole batch.
    let dup = copy_csv_simple(
        &server,
        proximadb_tenant::DEFAULT_TENANT,
        &collection,
        "v3,[0.1,0.1,0.1]\nv3,[0.2,0.2,0.2]\n",
    )
    .await;
    assert!(
        dup.as_ref()
            .is_err_and(|error| error.contains("INSERT_CONFLICT")),
        "in-payload duplicate ids must be rejected as an insert conflict, got {dup:?}"
    );
    assert!(
        read_vector("v3").await.is_none(),
        "a rejected batch must write none of its rows"
    );
}
