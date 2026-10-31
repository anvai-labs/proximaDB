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

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

use std::time::Duration;



/// In-process test server holding a `ProximaDB` instance bound to a
/// free pgwire port (plus a REST port so the existing `/health` probe
/// can report readiness — the pgwire listener doesn't have an
/// equivalent simple probe). Drop signals shutdown.

/// Connect via a real Postgres wire-protocol client and verify that a
/// `CREATE TABLE ... WITH (canonical_embedding_precision = 'fp16')`
/// statement sent over TCP lands on the catalog with the right
/// precision. Closes the transport-level gap that the DDL-service
/// unit tests in `src/services/ddl/mod.rs` already cover at the
/// in-process API level.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pgwire_create_table_with_canonical_embedding_precision_fp16() {
    unsafe { std::env::set_var("PROXIMADB_EMBED_PRECISION_SCHEMA_V2", "true"); }
    let server = TestPgServer::start().await.expect("server start");

    // Connect via tokio-postgres. `connect` returns the client and a
    // future that drives the connection — spawn it onto the runtime.
    let (client, connection) =
        tokio_postgres::connect(&server.conn_str(), tokio_postgres::NoTls)
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
