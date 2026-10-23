//! TD-PGEXT-1: the extended query protocol must Describe relational SELECTs
//! truthfully and accept bound parameters, so real drivers (JDBC, asyncpg,
//! psycopg 3, tokio-postgres) can read a table.
//!
//! Before this, `Describe(statement)` reported zero result columns for a
//! relational SELECT (only the pgvector shape was described) while Execute
//! suppressed its own RowDescription — so every driver that prepares a
//! statement saw a width-0 description and then N-column DataRows:
//! asyncpg "the number of columns in the result row (2) is different from
//! what was described (0)", psql \bind "unexpected field count in \"D\"
//! message". Every `$N` was also described as text, so typed parameters were
//! rejected client-side.
//!
//! Drives tokio-postgres `query()` / `prepare()` / `prepare_typed()`, and a
//! raw-wire Describe(portal) flow (what libpq — psql `\bind`, psycopg 3 — and
//! pgjdbc send), against relational and pgvector tables. The differential
//! test pins the core invariant: for every shape, the extended protocol's
//! Describe equals the simple protocol's RowDescription, or is empty.

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

async fn connect(server: &PgServer) -> tokio_postgres::Client {
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

fn explain_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!("[{}] {}", db.code().code(), db.message())
    } else {
        e.to_string()
    }
}

async fn seed(client: &tokio_postgres::Client) {
    for sql in [
        "CREATE TABLE menu_items (id INT PRIMARY KEY, name TEXT, price NUMERIC, ok BOOLEAN)",
        "INSERT INTO menu_items VALUES (1, 'a', 1.5, true), (2, 'b', 70, false)",
    ] {
        client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("seed `{sql}`: {}", explain_err(&e)));
    }
}

/// Column names of the simple protocol's RowDescription for `sql`, or `None`
/// when the simple protocol errors on it.
async fn simple_columns(client: &tokio_postgres::Client, sql: &str) -> Option<Vec<String>> {
    let messages = client.simple_query(sql).await.ok()?;
    messages.iter().find_map(|m| match m {
        tokio_postgres::SimpleQueryMessage::RowDescription(columns) => {
            Some(columns.iter().map(|c| c.name().to_string()).collect())
        }
        _ => None,
    })
}

/// Describe(statement) reports the real, typed columns — including `SELECT *`
/// — and binary-format rows decode with those types.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn extended_protocol_reads_relational_tables_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let stmt = client
        .prepare("SELECT id, name FROM menu_items ORDER BY id")
        .await
        .unwrap_or_else(|e| panic!("prepare: {}", explain_err(&e)));
    let cols: Vec<(String, tokio_postgres::types::Type)> = stmt
        .columns()
        .iter()
        .map(|c| (c.name().to_string(), c.type_().clone()))
        .collect();
    assert_eq!(
        cols,
        vec![
            ("id".to_string(), tokio_postgres::types::Type::INT4),
            ("name".to_string(), tokio_postgres::types::Type::TEXT),
        ]
    );
    let rows = client
        .query(&stmt, &[])
        .await
        .unwrap_or_else(|e| panic!("query: {}", explain_err(&e)));
    let ids: Vec<i32> = rows.iter().map(|r| r.get(0)).collect();
    assert_eq!(ids, vec![1, 2]);
    let names: Vec<String> = rows.iter().map(|r| r.get(1)).collect();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);

    // SELECT * is described from the table schema, in table order.
    let star = client
        .query("SELECT * FROM menu_items WHERE id = 1", &[])
        .await
        .unwrap_or_else(|e| panic!("select *: {}", explain_err(&e)));
    assert_eq!(star.len(), 1);
    let star_names: Vec<&str> = star[0].columns().iter().map(|c| c.name()).collect();
    assert_eq!(star_names, vec!["id", "name", "price", "ok"]);
    assert_eq!(star[0].get::<_, i32>(0), 1);
    assert_eq!(star[0].get::<_, String>(1), "a");
    assert!(star[0].get::<_, bool>(3));

    // count(*) routes through the relational pipeline and is int8.
    let n = client
        .query_one("SELECT count(*) FROM menu_items", &[])
        .await
        .unwrap_or_else(|e| panic!("count: {}", explain_err(&e)))
        .get::<_, i64>(0);
    assert_eq!(n, 2);
}

/// Clients that send parameter OIDs at Parse (JDBC, Npgsql, psycopg 3,
/// `prepare_typed`) get the same description, and bound parameters work.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn extended_protocol_bound_parameters_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let typed = client
        .prepare_typed(
            "SELECT id, name FROM menu_items WHERE name = $1",
            &[tokio_postgres::types::Type::TEXT],
        )
        .await
        .unwrap_or_else(|e| panic!("prepare_typed: {}", explain_err(&e)));
    assert_eq!(
        typed.columns().len(),
        2,
        "client-typed Parse is described too"
    );
    let row = client
        .query_one(&typed, &[&"b"])
        .await
        .unwrap_or_else(|e| panic!("typed param: {}", explain_err(&e)));
    assert_eq!(row.get::<_, i32>(0), 2);

    let name = client
        .query_one("SELECT name FROM menu_items WHERE name = $1", &[&"a"])
        .await
        .unwrap_or_else(|e| panic!("inferred param: {}", explain_err(&e)))
        .get::<_, String>(0);
    assert_eq!(name, "a");
}

/// The core invariant: Describe never advertises a shape Execute does not
/// stream. For each shape, extended Describe equals the simple protocol's
/// RowDescription, or is empty (unknown). Where it is non-empty, the rows
/// must decode in binary through the same statement.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn describe_matches_the_simple_protocol_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let shapes = [
        "SELECT id, name FROM menu_items",
        "SELECT name, id FROM menu_items",
        "SELECT * FROM menu_items",
        "SELECT id FROM menu_items WHERE name = 'a'",
        "SELECT id, name FROM menu_items ORDER BY id DESC",
        "SELECT id, name FROM menu_items LIMIT 1",
        "SELECT NAME FROM menu_items",
        "SELECT menu_items.name FROM menu_items",
        "SELECT name AS label FROM menu_items",
        "SELECT ok FROM menu_items",
        "SELECT price FROM menu_items",
        "SELECT count(*) FROM menu_items",
        "SELECT count(id) FROM menu_items",
        "SELECT count(*) AS n FROM menu_items",
        "SELECT name, count(*) FROM menu_items GROUP BY name",
        "SELECT sum(id) FROM menu_items",
        "SELECT max(id), count(*) FROM menu_items",
        "SELECT id FROM menu_items WHERE id IN (SELECT id FROM menu_items)",
        "SELECT DISTINCT name FROM menu_items",
        "SELECT count(*) FROM doc_users",
        "SELECT count(*) FROM log_events",
        "SELECT 1",
        "SELECT version()",
        "SELECT current_schema()",
        "SELECT table_name FROM information_schema.tables",
    ];
    let mut compared = Vec::new();
    for sql in shapes {
        let Some(simple) = simple_columns(&client, sql).await else {
            continue; // the simple protocol rejects it; nothing to compare
        };
        let stmt = match client.prepare(sql).await {
            Ok(stmt) => stmt,
            Err(e) => panic!("prepare `{sql}`: {}", explain_err(&e)),
        };
        let described: Vec<String> = stmt
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        if described.is_empty() {
            continue; // unknown: the pre-TD-PGEXT-1 behavior, allowed
        }
        assert_eq!(
            described, simple,
            "`{sql}`: extended Describe must match the simple RowDescription"
        );
        let rows = client
            .query(&stmt, &[])
            .await
            .unwrap_or_else(|e| panic!("query `{sql}`: {}", explain_err(&e)));
        for row in &rows {
            assert_eq!(row.len(), described.len(), "`{sql}`: DataRow width");
        }
        compared.push(sql);
    }
    eprintln!("compared {} shapes: {compared:?}", compared.len());
    // Guard against a vacuous pass: the shapes this fix describes must be
    // compared, not skipped as unknown.
    for must in [
        "SELECT id, name FROM menu_items",
        "SELECT * FROM menu_items",
        "SELECT NAME FROM menu_items",
        "SELECT name AS label FROM menu_items",
        "SELECT count(*) FROM menu_items",
        "SELECT name, count(*) FROM menu_items GROUP BY name",
    ] {
        assert!(
            compared.contains(&must),
            "`{must}` must be described and compared"
        );
    }
}

/// An Execute-time error is a SQL error; the connection survives it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn execute_error_keeps_the_connection_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let failed = client
        .query("SELECT no_such_column FROM menu_items", &[])
        .await;
    assert!(failed.is_err(), "an unknown column must fail");
    assert!(
        !client.is_closed(),
        "an Execute error must not close the connection"
    );
    let n = client
        .query_one("SELECT count(*) FROM menu_items", &[])
        .await
        .unwrap_or_else(|e| panic!("after error: {}", explain_err(&e)))
        .get::<_, i64>(0);
    assert_eq!(n, 2);
}

/// One frontend message: tag, length, body.
fn message(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    out.extend_from_slice(&((body.len() as i32 + 4).to_be_bytes()));
    out.extend_from_slice(body);
    out
}

async fn read_message(stream: &mut tokio::net::TcpStream) -> (u8, Vec<u8>) {
    use tokio::io::AsyncReadExt;
    let tag = stream.read_u8().await.expect("tag");
    let len = stream.read_i32().await.expect("len") as usize;
    let mut body = vec![0u8; len - 4];
    stream.read_exact(&mut body).await.expect("body");
    (tag, body)
}

/// A raw protocol-3.0 connection, past startup (ReadyForQuery received).
async fn raw_connect(server: &PgServer) -> tokio::net::TcpStream {
    use tokio::io::AsyncWriteExt;
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", server.pg_port))
        .await
        .expect("raw connect");
    let mut startup = Vec::new();
    startup.extend_from_slice(&196_608i32.to_be_bytes());
    for (key, value) in [("user", "postgres"), ("database", "proximadb")] {
        startup.extend_from_slice(key.as_bytes());
        startup.push(0);
        startup.extend_from_slice(value.as_bytes());
        startup.push(0);
    }
    startup.push(0);
    let mut framed = ((startup.len() as i32 + 4).to_be_bytes()).to_vec();
    framed.extend_from_slice(&startup);
    stream.write_all(&framed).await.expect("startup");
    while read_message(&mut stream).await.0 != b'Z' {}
    stream
}

/// Parse + Bind (no parameters, text results) + Execute on the unnamed
/// statement/portal.
fn parse_bind_execute(sql: &str) -> Vec<u8> {
    let mut parse = vec![0u8];
    parse.extend_from_slice(sql.as_bytes());
    parse.push(0);
    parse.extend_from_slice(&0i16.to_be_bytes());
    let mut bind = vec![0u8, 0u8];
    bind.extend_from_slice(&0i16.to_be_bytes());
    bind.extend_from_slice(&0i16.to_be_bytes());
    bind.extend_from_slice(&0i16.to_be_bytes());
    let mut out = message(b'P', &parse);
    out.extend(message(b'B', &bind));
    out.extend(message(b'E', &[0, 0, 0, 0, 0]));
    out
}

/// After an ErrorResponse, the server discards messages until Sync: a
/// pipelined batch (JDBC `executeBatch`, asyncpg `executemany`) must not run
/// the statements queued behind a failed one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn error_skips_pipelined_messages_until_sync_e2e() {
    use tokio::io::AsyncWriteExt;

    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let mut stream = raw_connect(&server).await;
    let mut out = parse_bind_execute("SELECT no_such_column FROM menu_items");
    out.extend(parse_bind_execute(
        "INSERT INTO menu_items VALUES (3, 'c', 1, true)",
    ));
    out.extend(message(b'S', &[]));
    stream.write_all(&out).await.expect("pipelined batch");

    let mut tags = Vec::new();
    loop {
        let (tag, _) = read_message(&mut stream).await;
        tags.push(tag as char);
        if tag == b'Z' {
            break;
        }
    }
    assert_eq!(
        tags.iter().filter(|t| **t == 'E').count(),
        1,
        "exactly one ErrorResponse, then nothing until ReadyForQuery: {tags:?}"
    );
    assert!(
        !tags.contains(&'C'),
        "the queued INSERT must not run: {tags:?}"
    );
    let n = client
        .query_one("SELECT count(*) FROM menu_items", &[])
        .await
        .unwrap_or_else(|e| panic!("count: {}", explain_err(&e)))
        .get::<_, i64>(0);
    assert_eq!(n, 2, "the INSERT behind the failed Execute was skipped");

    // The connection is usable again after Sync.
    stream
        .write_all(&{
            let mut b = parse_bind_execute("SELECT id FROM menu_items WHERE id = 1");
            b.extend(message(b'S', &[]));
            b
        })
        .await
        .expect("after sync");
    let mut rows = 0;
    loop {
        match read_message(&mut stream).await.0 {
            b'D' => rows += 1,
            b'E' => panic!("unexpected error after Sync"),
            b'Z' => break,
            _ => {}
        }
    }
    assert_eq!(rows, 1);
}

/// Execute must stream what Describe advertised, or fail cleanly (0A000)
/// before any row — never a different shape. Covers a cached statement after
/// DDL and a parameter value that changes the server's routing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn execute_never_streams_a_different_shape_than_described_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    // A parameter value the dispatcher used to route on (`CURRENT_USER`).
    let hijack = client
        .query(
            "SELECT id, name FROM menu_items WHERE name = $1",
            &[&"current_user"],
        )
        .await;
    match hijack {
        Ok(rows) => {
            for row in &rows {
                assert_eq!(row.len(), 2);
            }
        }
        Err(e) => assert_eq!(
            e.code().map(|c| c.code()),
            Some("0A000"),
            "a shape change must be 0A000: {}",
            explain_err(&e)
        ),
    }
    assert!(!client.is_closed());

    // A cached SELECT * after ALTER TABLE ADD COLUMN.
    let star = client
        .prepare("SELECT * FROM menu_items")
        .await
        .unwrap_or_else(|e| panic!("prepare: {}", explain_err(&e)));
    let width = star.columns().len();
    client
        .simple_query("ALTER TABLE menu_items ADD COLUMN extra TEXT")
        .await
        .unwrap_or_else(|e| panic!("alter: {}", explain_err(&e)));
    match client.query(&star, &[]).await {
        Ok(rows) => {
            for row in &rows {
                assert_eq!(row.len(), width, "rows must match the cached description");
            }
        }
        Err(e) => assert_eq!(
            e.code().map(|c| c.code()),
            Some("0A000"),
            "{}",
            explain_err(&e)
        ),
    }
    assert!(!client.is_closed());
    let n = client
        .query_one("SELECT count(*) FROM menu_items", &[])
        .await
        .unwrap_or_else(|e| panic!("after drift: {}", explain_err(&e)))
        .get::<_, i64>(0);
    assert_eq!(n, 2);
}

/// Leading whitespace (Python triple-quoted SQL) is still a SELECT.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn leading_whitespace_select_returns_rows_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;
    let rows = client
        .query("\n    SELECT id, name FROM menu_items ORDER BY id\n", &[])
        .await
        .unwrap_or_else(|e| panic!("query: {}", explain_err(&e)));
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get::<_, i32>(0), 1);
}

/// Raw bytes of any column type (to check a binary encoding directly).
struct RawCell(Vec<u8>);

impl<'a> tokio_postgres::types::FromSql<'a> for RawCell {
    fn from_sql(
        _: &tokio_postgres::types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(RawCell(raw.to_vec()))
    }

    fn accepts(_: &tokio_postgres::types::Type) -> bool {
        true
    }
}

/// pgvector shape over the extended protocol: the jsonb column is sent in
/// PostgreSQL's binary jsonb form (version byte 1 + JSON text), the distance
/// as float8 — previously the jsonb cell closed the connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pgvector_jsonb_and_distance_binary_e2e() {
    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    client
        .simple_query(
            "CREATE TABLE vecs (id VARCHAR PRIMARY KEY, embedding VECTOR(2), payload JSONB)",
        )
        .await
        .unwrap_or_else(|e| panic!("create: {}", explain_err(&e)));
    client
        .simple_query(
            "INSERT INTO vecs (id, embedding, payload) VALUES \
             ('a', '[1,0]'::vector, '{\"k\":\"a\"}'::jsonb)",
        )
        .await
        .unwrap_or_else(|e| panic!("insert: {}", explain_err(&e)));
    sleep(Duration::from_millis(200)).await;

    let rows = client
        .query(
            "SELECT id, payload FROM vecs ORDER BY embedding <-> $1 LIMIT $2",
            &[&"[1,0]", &1i64],
        )
        .await
        .unwrap_or_else(|e| panic!("pgvector query: {}", explain_err(&e)));
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, String>(0), "a");
    let RawCell(payload) = rows[0].get(1);
    assert_eq!(payload.first(), Some(&1u8), "jsonb binary version byte");
    let json: serde_json::Value =
        serde_json::from_slice(&payload[1..]).expect("jsonb body is JSON");
    assert_eq!(json["k"], "a");
    assert!(!client.is_closed());
}

/// Describe(portal) — what libpq (`PQsendQueryParams`: psql `\bind`,
/// psycopg 3), pgjdbc unnamed statements and Npgsql send — must report the
/// same columns as the DataRows that follow. Raw wire, so the exact message
/// sequence is under test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn describe_portal_matches_data_rows_e2e() {
    use tokio::io::AsyncWriteExt;

    let server = PgServer::start().await.expect("server start");
    let client = connect(&server).await;
    seed(&client).await;

    let mut stream = raw_connect(&server).await;

    // Parse(unnamed, $1) / Bind(unnamed, text param, text results) /
    // Describe(portal) / Execute / Sync — libpq's PQsendQueryParams.
    let sql = b"SELECT id, name FROM menu_items WHERE name = $1\0";
    let mut parse = vec![0u8];
    parse.extend_from_slice(sql);
    parse.extend_from_slice(&0i16.to_be_bytes());
    let mut bind = vec![0u8, 0u8]; // portal "", statement ""
    bind.extend_from_slice(&0i16.to_be_bytes()); // param formats: all text
    bind.extend_from_slice(&1i16.to_be_bytes()); // one parameter
    bind.extend_from_slice(&1i32.to_be_bytes());
    bind.push(b'a');
    bind.extend_from_slice(&0i16.to_be_bytes()); // result formats: all text
    let mut out = Vec::new();
    out.extend(message(b'P', &parse));
    out.extend(message(b'B', &bind));
    out.extend(message(b'D', b"P\0"));
    out.extend(message(b'E', &[0, 0, 0, 0, 0]));
    out.extend(message(b'S', &[]));
    stream.write_all(&out).await.expect("extended batch");

    let mut described: Option<i16> = None;
    let mut data_widths = Vec::new();
    loop {
        let (tag, body) = read_message(&mut stream).await;
        match tag {
            b'T' => described = Some(i16::from_be_bytes([body[0], body[1]])),
            b'D' => data_widths.push(i16::from_be_bytes([body[0], body[1]])),
            b'E' => panic!("server error: {}", String::from_utf8_lossy(&body)),
            b'Z' => break,
            _ => {}
        }
    }
    assert_eq!(described, Some(2), "Describe(portal) reports both columns");
    assert_eq!(data_widths, vec![2], "one 2-column DataRow for name = 'a'");
}
