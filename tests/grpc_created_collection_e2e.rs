//! TD-GRPCCREATE-1: a collection created over gRPC v2 must store records.
//!
//! Reproduced against the 0.4.1 image: `CreateCollection` over gRPC succeeded,
//! but every later `InsertRecords` failed ("Collection <id> not found"), and the
//! REST v2 listing omitted the collection. The Python SDK prefers gRPC, so every
//! Python quick start stored nothing. This drives the real server over both
//! transports: create over gRPC, then insert, search and list over gRPC and REST.

use std::net::TcpListener;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::proto::proximadb_v2::proxima_record_service_client::ProximaRecordServiceClient;
use proximadb::proto::proximadb_v2::{
    BatchWriteMode, GetSchemaRequest, ListSchemasRequest, ProximaRecord, ProximaRecordBatch,
    TypedSearchRequest, V2CollectionConfig, V2ListCollectionsRequest,
};
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct Server {
    rest_port: u16,
    grpc_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Server {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let tmp = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.unified_mode = false;
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.arrow_flight_port = free_port();
        config.api.pg_port = Some(free_port());
        config.api.enable_pgwire = false;
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
            .timeout(Duration::from_secs(5))
            .no_proxy()
            .build()?;
        let health = format!("http://127.0.0.1:{rest_port}/health");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            match http.get(&health).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ if std::time::Instant::now() > deadline => {
                    anyhow::bail!("server did not become ready in 20s")
                }
                _ => sleep(Duration::from_millis(100)).await,
            }
        }
        // Grace for the gRPC / Arrow IPC listeners to start accepting.
        sleep(Duration::from_millis(300)).await;

        Ok(Self {
            rest_port,
            grpc_port,
            db: Some(db),
            _tmp: tmp,
        })
    }
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

async fn connect_grpc(port: u16) -> ProximaRecordServiceClient<tonic::transport::Channel> {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        match ProximaRecordServiceClient::connect(format!("http://127.0.0.1:{port}")).await {
            Ok(client) => return client,
            Err(error) if std::time::Instant::now() > deadline => {
                panic!("gRPC server did not become ready: {error}")
            }
            Err(_) => sleep(Duration::from_millis(100)).await,
        }
    }
}

fn record(id: &str, vector: [f32; 3]) -> ProximaRecord {
    ProximaRecord {
        id: id.to_string(),
        vector: vector.to_vec(),
        ..Default::default()
    }
}

/// PUT the additive schema (embedding + lang) — the TD-GRPCSCHEMA-1/2 shape
/// both tests enable or evolve with. Returns (status, body).
async fn put_schema_embedding_lang(
    http: &reqwest::Client,
    base: &str,
    name: &str,
) -> (reqwest::StatusCode, String) {
    let response = http
        .put(format!("{base}/api/v2/collections/{name}/schema"))
        .json(&serde_json::json!({
            "columns": [
                {"name": "embedding", "data_type": "vector", "vector_dimension": 3},
                {"name": "lang", "data_type": "text", "nullable": true}
            ]
        }))
        .send()
        .await
        .expect("REST schema PUT");
    let status = response.status();
    let body = response.text().await.expect("body");
    (status, body)
}

/// GetSchema's `record_enabled` answer for `name` (the enforcement mode).
async fn grpc_record_enabled(
    grpc: &mut ProximaRecordServiceClient<tonic::transport::Channel>,
    name: &str,
) -> Option<bool> {
    grpc.get_schema(GetSchemaRequest {
        collection_id: name.to_string(),
        schema_id: None,
    })
    .await
    .expect("gRPC GetSchema")
    .into_inner()
    .record_enabled
}

/// REST GET /collections/{name}'s proxima_record_enabled value.
async fn rest_record_enabled(http: &reqwest::Client, base: &str, name: &str) -> serde_json::Value {
    http.get(format!("{base}/api/v2/collections/{name}"))
        .send()
        .await
        .expect("REST GET collection")
        .json()
        .await
        .expect("json")
}

/// A dimension-3 cosine SST collection created over gRPC, with record mode
/// explicitly on or off (TD-GRPCSCHEMA-2; canonical schema persisted either
/// way — TD-GRPCSCHEMA-1's shape).
async fn grpc_create_cosine_3d_with_record(
    grpc: &mut ProximaRecordServiceClient<tonic::transport::Channel>,
    name: &str,
    record: bool,
) {
    grpc.create_collection(V2CollectionConfig {
        name: name.to_string(),
        dimension: 3,
        distance_metric: "cosine".to_string(),
        storage_engine: "sst".to_string(),
        enable_proxima_record: Some(record),
        ..Default::default()
    })
    .await
    .expect("gRPC CreateCollection");
}

/// A dimension-3 cosine SST collection created over gRPC (TD-GRPCSCHEMA-1's
/// shape: no way to enable ProximaRecord, canonical schema persisted).
async fn grpc_create_cosine_3d(
    grpc: &mut ProximaRecordServiceClient<tonic::transport::Channel>,
    name: &str,
) {
    grpc.create_collection(V2CollectionConfig {
        name: name.to_string(),
        dimension: 3,
        distance_metric: "cosine".to_string(),
        storage_engine: "sst".to_string(),
        ..Default::default()
    })
    .await
    .expect("gRPC CreateCollection");
}

#[tokio::test]
async fn grpc_created_collection_stores_and_serves_records_e2e() {
    let server = Server::start().await.expect("server start");
    let mut grpc = connect_grpc(server.grpc_port).await;
    let name = "grpc_created";

    grpc_create_cosine_3d(&mut grpc, name).await;

    let inserted = grpc
        .insert_records(ProximaRecordBatch {
            collection_id: name.to_string(),
            records: vec![record("a", [0.1, 0.2, 0.3]), record("b", [0.3, 0.2, 0.1])],
            write_mode: BatchWriteMode::Insert as i32,
            validate_schema: true,
            return_ids: true,
            return_errors: true,
            ..Default::default()
        })
        .await
        .expect("gRPC InsertRecords transport")
        .into_inner();
    assert!(
        inserted.success && inserted.success_count == 2 && inserted.failed_count == 0,
        "gRPC insert into a gRPC-created collection must succeed: {inserted:?}"
    );

    let hits = grpc
        .search(TypedSearchRequest {
            collection_id: name.to_string(),
            query_vector: vec![0.1, 0.2, 0.3],
            top_k: 5,
            ..Default::default()
        })
        .await
        .expect("gRPC Search")
        .into_inner();
    assert_eq!(
        hits.results.len(),
        2,
        "gRPC search must see both records: {hits:?}"
    );

    let listed = grpc
        .list_collections(V2ListCollectionsRequest::default())
        .await
        .expect("gRPC ListCollections")
        .into_inner();
    assert!(
        listed
            .collections
            .iter()
            .any(|c| c.config.as_ref().is_some_and(|cfg| cfg.name == name)),
        "gRPC listing must include the collection"
    );

    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let rest_list: serde_json::Value = http
        .get(format!("{base}/api/v2/collections"))
        .send()
        .await
        .expect("REST list")
        .json()
        .await
        .expect("REST list json");
    assert!(
        rest_list.to_string().contains(name),
        "REST listing must include a gRPC-created collection: {rest_list}"
    );

    let rest_search: serde_json::Value = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({"vector": [0.1, 0.2, 0.3], "top_k": 5}))
        .send()
        .await
        .expect("REST search")
        .json()
        .await
        .expect("REST search json");
    assert_eq!(
        rest_search["results"].as_array().map(Vec::len),
        Some(2),
        "REST search must see records written over gRPC: {rest_search}"
    );
}

async fn rest_create(http: &reqwest::Client, base: &str, name: &str) {
    let resp = http
        .post(format!("{base}/api/v2/collections"))
        .json(
            &json!({"name": name, "dimension": 3, "engine": "sst", "enable_proxima_record": true}),
        )
        .send()
        .await
        .expect("REST create");
    assert!(resp.status().is_success(), "REST create: {}", resp.status());
}

async fn rest_insert(
    http: &reqwest::Client,
    base: &str,
    name: &str,
    id: &str,
) -> serde_json::Value {
    http.post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({"records": [{"id": id, "vector": [0.1, 0.2, 0.3], "props": {"category": "A"}}]}))
        .send()
        .await
        .expect("REST insert")
        .json()
        .await
        .expect("REST insert json")
}

async fn rest_search_ids(http: &reqwest::Client, base: &str, name: &str) -> Vec<String> {
    let body: serde_json::Value = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({"vector": [0.1, 0.2, 0.3], "top_k": 10}))
        .send()
        .await
        .expect("REST search")
        .json()
        .await
        .expect("REST search json");
    body["results"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|r| r["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// A schema update that names the vector column `embedding` (the natural name)
/// collided with the reserved `embedding` column: on 0.4.1 the PUT failed with
/// "Duplicate column name: embedding" AND left the collection with no table.
#[tokio::test]
async fn rest_schema_update_naming_embedding_keeps_collection_usable_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "schema_embedding";
    rest_create(&http, &base, name).await;

    let put = http
        .put(format!("{base}/api/v2/collections/{name}/schema"))
        .json(&json!({"columns": [
            {"name": "embedding", "data_type": "vector", "vector_dimension": 3},
            {"name": "category", "data_type": "text", "filterable": true}
        ]}))
        .send()
        .await
        .expect("REST schema PUT");
    let status = put.status();
    let body = put.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "schema PUT naming `embedding` must succeed: {status} {body}"
    );

    let inserted = rest_insert(&http, &base, name, "1").await;
    assert_eq!(
        inserted["inserted_count"], 1,
        "insert after schema update: {inserted}"
    );
    assert_eq!(
        rest_search_ids(&http, &base, name).await,
        vec!["1".to_string()]
    );
}

/// A rejected schema update must never cost the collection its table or data
/// (the invariant TD-GRPCCREATE-1 broke). This is a regression guard: on
/// `develop` this request is rejected before the catalog rewrite, so it passes
/// there too; the tests above are the ones that fail without the fix.
#[tokio::test]
async fn rejected_schema_update_does_not_destroy_collection_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "schema_rejected";
    rest_create(&http, &base, name).await;
    let before = rest_insert(&http, &base, name, "before").await;
    assert_eq!(before["inserted_count"], 1, "seed insert: {before}");

    let put = http
        .put(format!("{base}/api/v2/collections/{name}/schema"))
        // `force` skips schema-evolution checks, so this reaches the deepest
        // validation the server has; it must still be rejected without data loss.
        .json(&json!({"force": true, "columns": [
            {"name": "category", "data_type": "text"},
            {"name": "category", "data_type": "text"}
        ]}))
        .send()
        .await
        .expect("REST schema PUT");
    assert!(
        !put.status().is_success(),
        "a schema with duplicate columns must be rejected: {}",
        put.status()
    );

    let after = rest_insert(&http, &base, name, "after").await;
    assert_eq!(
        after["inserted_count"], 1,
        "insert after a rejected schema update: {after}"
    );
    let mut ids = rest_search_ids(&http, &base, name).await;
    ids.sort();
    assert_eq!(ids, vec!["after".to_string(), "before".to_string()]);
}

async fn rest_put_schema(
    http: &reqwest::Client,
    base: &str,
    name: &str,
    body: serde_json::Value,
) -> (reqwest::StatusCode, String) {
    let resp = http
        .put(format!("{base}/api/v2/collections/{name}/schema"))
        .json(&body)
        .send()
        .await
        .expect("REST schema PUT");
    let status = resp.status();
    (status, resp.text().await.unwrap_or_default())
}

async fn rest_schema_column_names(http: &reqwest::Client, base: &str, name: &str) -> Vec<String> {
    let body: serde_json::Value = http
        .get(format!("{base}/api/v2/collections/{name}/schema"))
        .send()
        .await
        .expect("REST schema GET")
        .json()
        .await
        .expect("REST schema GET json");
    let mut names: Vec<String> = body["schema"]["columns"]
        .as_array()
        .map(|cols| {
            cols.iter()
                .filter_map(|c| c["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

async fn rest_lists(http: &reqwest::Client, base: &str, name: &str) -> bool {
    let list: serde_json::Value = http
        .get(format!("{base}/api/v2/collections"))
        .send()
        .await
        .expect("REST list")
        .json()
        .await
        .expect("REST list json");
    list.to_string().contains(&format!("\"{name}\""))
}

/// Review repro: turning an existing canonical column filterable rebuilt the
/// reserved band with that name AND re-appended the preserved canonical column,
/// then dropped the table before `create_table` rejected the duplicate. The
/// collection vanished from the listing and its rows were unreachable.
#[tokio::test]
async fn making_a_column_filterable_keeps_the_collection_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "flip_filterable";
    rest_create(&http, &base, name).await;

    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [{"name": "rank", "data_type": "integer", "filterable": false}]}),
    )
    .await;
    assert!(status.is_success(), "first schema PUT: {status} {body}");
    let seeded = rest_insert(&http, &base, name, "before").await;
    assert_eq!(seeded["inserted_count"], 1, "seed insert: {seeded}");

    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [{"name": "rank", "data_type": "integer", "filterable": true}]}),
    )
    .await;
    assert!(
        status.is_success(),
        "making `rank` filterable must succeed: {status} {body}"
    );

    assert!(
        rest_lists(&http, &base, name).await,
        "collection must still be listed"
    );
    let after = rest_insert(&http, &base, name, "after").await;
    assert_eq!(after["inserted_count"], 1, "insert after the flip: {after}");
    let mut ids = rest_search_ids(&http, &base, name).await;
    ids.sort();
    assert_eq!(ids, vec!["after".to_string(), "before".to_string()]);
    assert_eq!(
        rest_schema_column_names(&http, &base, name).await,
        vec!["rank".to_string()]
    );
}

/// GetSchema must return every declared column, including ones that fold into
/// reserved columns of the same name (`embedding`, filterable columns).
#[tokio::test]
async fn schema_get_returns_every_declared_column_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "schema_complete";
    rest_create(&http, &base, name).await;

    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [
            {"name": "embedding", "data_type": "vector", "vector_dimension": 3},
            {"name": "rank", "data_type": "integer", "filterable": true},
            {"name": "notes", "data_type": "text"}
        ]}),
    )
    .await;
    assert!(status.is_success(), "schema PUT: {status} {body}");
    assert_eq!(
        rest_schema_column_names(&http, &base, name).await,
        vec![
            "embedding".to_string(),
            "notes".to_string(),
            "rank".to_string()
        ],
        "REST GetSchema must list every declared column"
    );
    // gRPC GetSchema is not checked here: a collection created over gRPC does
    // not enable ProximaRecord, so GetSchema answers NotFound (TD-GRPCSCHEMA-1).
}

/// A declared column that conflicts with a reserved one is rejected, and the
/// collection keeps working: it is not silently ignored.
#[tokio::test]
async fn conflicting_embedding_dimension_is_rejected_and_collection_survives_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "schema_conflict";
    rest_create(&http, &base, name).await;
    let seeded = rest_insert(&http, &base, name, "before").await;
    assert_eq!(seeded["inserted_count"], 1, "seed insert: {seeded}");

    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [{"name": "embedding", "data_type": "vector", "vector_dimension": 768}]}),
    )
    .await;
    // TD-RESTGET-1: the caller's mistake is a 400 naming it, not a 500.
    assert_eq!(
        status,
        reqwest::StatusCode::BAD_REQUEST,
        "a 768-dim embedding on a dim-3 collection is a client error: {body}"
    );
    assert!(
        body.contains("768") && body.contains("invalid_argument"),
        "the 400 must name the conflict: {body}"
    );
    assert!(
        rest_lists(&http, &base, name).await,
        "collection must still be listed"
    );
    let after = rest_insert(&http, &base, name, "after").await;
    assert_eq!(
        after["inserted_count"], 1,
        "insert after the rejected update: {after}"
    );
}

/// REST declares non-text columns filterable by default, and the stored
/// filterable column holds the narrowed type (`int32` → `Int64`). A declared
/// type that narrows to the stored one must be accepted, including on a second
/// identical update (which folds into the existing filterable columns).
#[tokio::test]
async fn narrowed_filterable_types_are_accepted_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "narrowed_types";
    rest_create(&http, &base, name).await;
    let body = json!({"columns": [
        {"name": "year", "data_type": "int32"},
        {"name": "score", "data_type": "float32"},
        {"name": "price", "data_type": "decimal", "precision": 10, "scale": 2}
    ]});
    for attempt in 1..=2 {
        let (status, text) = rest_put_schema(&http, &base, name, body.clone()).await;
        assert!(
            status.is_success(),
            "schema PUT #{attempt}: {status} {text}"
        );
    }
    assert_eq!(
        rest_schema_column_names(&http, &base, name).await,
        vec!["price".to_string(), "score".to_string(), "year".to_string()]
    );
    let after = rest_insert(&http, &base, name, "1").await;
    assert_eq!(
        after["inserted_count"], 1,
        "insert after the updates: {after}"
    );
}

/// TD-GRPCSCHEMA-1: a collection created over gRPC (which cannot enable
/// ProximaRecord) still has a persisted canonical schema — embedding plus any
/// filterable columns — and GetSchema must serve it instead of answering
/// NotFound. The REST GET answers the same catalog-authoritative view.
#[tokio::test]
async fn grpc_created_collection_serves_its_schema_e2e() {
    let server = Server::start().await.expect("server start");
    let mut grpc = connect_grpc(server.grpc_port).await;
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "grpc_schema_served";

    grpc_create_cosine_3d(&mut grpc, name).await;

    let schema = grpc
        .get_schema(GetSchemaRequest {
            collection_id: name.to_string(),
            schema_id: None,
        })
        .await
        .expect("gRPC GetSchema must serve the persisted schema")
        .into_inner();
    assert!(schema.success, "{schema:?}");
    let columns = schema.schema.expect("schema present").columns;
    let embedding = columns
        .iter()
        .find(|column| column.name == "embedding")
        .expect("the persisted schema carries the embedding column");
    assert_eq!(embedding.vector_dimension, Some(3));

    let rest = http
        .get(format!("{base}/api/v2/collections/{name}/schema"))
        .send()
        .await
        .expect("REST schema GET");
    assert_eq!(
        rest.status(),
        reqwest::StatusCode::OK,
        "REST must serve the same catalog-authoritative schema"
    );
    let body: serde_json::Value = rest.json().await.expect("json");
    let names: Vec<&str> = body["schema"]["columns"]
        .as_array()
        .map(|columns| {
            columns
                .iter()
                .map(|column| column["name"].as_str().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        names.contains(&"embedding"),
        "REST schema must carry the embedding column: {body}"
    );
}

/// TD-GRPCSCHEMA-1 (review round 1): the read and the evolution baseline must
/// agree — ListSchemas serves a flag-off collection's schema too, and a schema
/// PUT cannot silently drop a column GET advertises (the embedding column),
/// while an additive PUT still works.
#[tokio::test]
async fn schema_read_and_evolution_baseline_agree_e2e() {
    let server = Server::start().await.expect("server start");
    let mut grpc = connect_grpc(server.grpc_port).await;
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "grpc_schema_baseline";

    grpc_create_cosine_3d(&mut grpc, name).await;

    let listed = grpc
        .list_schemas(ListSchemasRequest {
            collection_id: name.to_string(),
            limit: None,
            offset: None,
        })
        .await
        .expect("gRPC ListSchemas")
        .into_inner();
    assert_eq!(
        listed.total_count, 1,
        "a flag-off collection lists its schema"
    );
    assert!(
        listed.schemas[0]
            .columns
            .iter()
            .any(|column| column.name == "embedding"),
        "the listed schema carries the embedding column"
    );

    // A PUT that drops the advertised embedding column is rejected.
    let rejected = http
        .put(format!("{base}/api/v2/collections/{name}/schema"))
        .json(&serde_json::json!({
            "columns": [
                {"name": "lang", "data_type": "text", "nullable": true}
            ]
        }))
        .send()
        .await
        .expect("REST schema PUT");
    let status = rejected.status();
    let body = rejected.text().await.expect("body");
    assert_eq!(
        status,
        reqwest::StatusCode::CONFLICT,
        "dropping the advertised embedding column must be a 409: {status} {body}"
    );
    assert!(
        body.contains("embedding"),
        "the rejection must name the column-removal rule for `embedding`: {body}"
    );

    // An additive PUT (embedding kept) succeeds and GET still serves it.
    let (status, body) = put_schema_embedding_lang(&http, &base, name).await;
    assert!(
        status.is_success(),
        "additive PUT must succeed: {status} {body}"
    );

    let schema = grpc
        .get_schema(GetSchemaRequest {
            collection_id: name.to_string(),
            schema_id: None,
        })
        .await
        .expect("gRPC GetSchema after the additive PUT")
        .into_inner();
    let schema = schema.schema.expect("schema present");
    let names: Vec<&str> = schema
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    assert!(
        names.contains(&"embedding") && names.contains(&"lang"),
        "{names:?}"
    );
}

/// TD-GRPCSCHEMA-2: record mode is settable at gRPC create (REST parity) and
/// enableable afterwards via a schema update; GetSchema surfaces the mode.
#[tokio::test]
async fn record_mode_is_settable_and_enableable_over_grpc_e2e() {
    let server = Server::start().await.expect("server start");
    let mut grpc = connect_grpc(server.grpc_port).await;
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    // Created with the flag ON: visible on GetSchema and REST immediately.
    let on = "grpc_record_on";
    grpc_create_cosine_3d_with_record(&mut grpc, on, true).await;
    let schema = grpc
        .get_schema(GetSchemaRequest {
            collection_id: on.to_string(),
            schema_id: None,
        })
        .await
        .expect("GetSchema")
        .into_inner();
    assert_eq!(schema.record_enabled, Some(true), "create-time mode is ON");
    let rest = rest_record_enabled(&http, &base, on).await;
    assert_eq!(rest["proxima_record_enabled"], true, "{rest}");

    // Created with the flag UNSET: OFF — REST create's default, and what every
    // pre-field gRPC client had (round 1 review: None must not mean ON).
    let unset = "grpc_record_unset";
    grpc.create_collection(V2CollectionConfig {
        name: unset.to_string(),
        dimension: 3,
        distance_metric: "cosine".to_string(),
        storage_engine: "sst".to_string(),
        ..Default::default()
    })
    .await
    .expect("gRPC CreateCollection (unset)");
    let rest = rest_record_enabled(&http, &base, unset).await;
    assert_eq!(
        rest["proxima_record_enabled"], false,
        "an UNSET create must land OFF: {rest}"
    );
    assert_eq!(
        grpc_record_enabled(&mut grpc, unset).await,
        Some(false),
        "record_enabled is the enforcement answer for an unset create"
    );

    // Created with the flag explicitly OFF: GetSchema agrees BEFORE any update
    // (round 1 review: it used to read ON from the persisted record_schema
    // block, contradicting REST).
    let later = "grpc_record_later";
    grpc_create_cosine_3d_with_record(&mut grpc, later, false).await;
    assert_eq!(
        grpc_record_enabled(&mut grpc, later).await,
        Some(false),
        "a Some(false) create must read OFF on GetSchema too"
    );
    let rest = rest_record_enabled(&http, &base, later).await;
    assert_eq!(
        rest["proxima_record_enabled"], false,
        "precondition: off after create"
    );

    let (status, body) = put_schema_embedding_lang(&http, &base, later).await;
    assert!(
        status.is_success(),
        "schema PUT must succeed: {status} {body}"
    );

    let rest = rest_record_enabled(&http, &base, later).await;
    assert_eq!(
        rest["proxima_record_enabled"], true,
        "a schema update must turn record mode ON (it used to be dropped by the merge)"
    );
    let schema = grpc
        .get_schema(GetSchemaRequest {
            collection_id: later.to_string(),
            schema_id: None,
        })
        .await
        .expect("GetSchema")
        .into_inner();
    assert_eq!(schema.record_enabled, Some(true));
}

// ---------------------------------------------------------------------------
// TD-RESTGET-1: the REST v2 create contract. A misspelt or unknown setting, or
// an invalid value, is a 400 naming it — never silently replaced by a default
// — and nothing is created.
// ---------------------------------------------------------------------------

async fn rest_post_create(
    http: &reqwest::Client,
    base: &str,
    body: serde_json::Value,
) -> (reqwest::StatusCode, String) {
    let resp = http
        .post(format!("{base}/api/v2/collections"))
        .json(&body)
        .send()
        .await
        .expect("REST create");
    let status = resp.status();
    (status, resp.text().await.unwrap_or_default())
}

async fn rest_get_status(http: &reqwest::Client, base: &str, name: &str) -> reqwest::StatusCode {
    http.get(format!("{base}/api/v2/collections/{name}"))
        .send()
        .await
        .expect("REST get")
        .status()
}

#[tokio::test]
async fn get_missing_collection_is_404_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let resp = http
        .get(format!("{base}/api/v2/collections/no_such_collection"))
        .send()
        .await
        .expect("REST get");
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let body = resp.text().await.unwrap_or_default();
    assert!(body.contains("collection_not_found"), "404 body: {body}");
}

#[tokio::test]
async fn create_rejects_unknown_and_misspelt_fields_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    let cases = [
        // top level
        (
            "unknown_top",
            json!({"name": "unknown_top", "dimension": 4, "replicas": 3}),
            "unknown field `replicas`",
        ),
        // nested: `hnsw` for `hnsw_config` used to create an index with no parameters
        (
            "misspelt_hnsw",
            json!({"name": "misspelt_hnsw", "dimension": 4,
                   "index_configs": [{"algorithm": "hnsw", "hnsw": {"m": 32}}]}),
            "unknown field `hnsw`",
        ),
        (
            "misspelt_quant",
            json!({"name": "misspelt_quant", "dimension": 4,
                   "quantization": {"enabled": true, "type": "scalar"}}),
            "unknown field `type`",
        ),
        (
            "misspelt_column",
            json!({"name": "misspelt_column", "dimension": 4, "enable_proxima_record": true,
                   "schema": {"columns": [{"name": "c", "data_type": "text", "filterabel": true}]}}),
            "unknown field `filterabel`",
        ),
        // invalid value: used to be replaced by smart_defaults
        (
            "bad_strategy",
            json!({"name": "bad_strategy", "dimension": 4,
                   "quantization": {"enabled": true, "strategy": "scalar"}}),
            "smart_defaults",
        ),
        // index_type other than the released SDKs' "hnsw" default was silently
        // not honoured; it now says which fields do express it
        (
            "index_type_ivf",
            json!({"name": "index_type_ivf", "dimension": 4, "index_type": "ivf"}),
            "index_configs",
        ),
    ];
    for (name, body, named) in cases {
        let (status, text) = rest_post_create(&http, &base, body).await;
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "{name}: must be a 400, got {status}: {text}"
        );
        assert!(
            text.contains("invalid_argument") && text.contains(named),
            "{name}: the 400 must be the API envelope naming `{named}`: {text}"
        );
        assert_eq!(
            rest_get_status(&http, &base, name).await,
            reqwest::StatusCode::NOT_FOUND,
            "{name}: a rejected create must not create the collection"
        );
    }
}

#[tokio::test]
async fn create_accepts_canonical_and_legacy_spellings_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    let accepted = [
        json!({"name": "valid_strategy", "dimension": 4,
               "quantization": {"enabled": true, "strategy": "custom_levels"}}),
        // what the released Rust / TypeScript SDKs send
        json!({"name": "legacy_sdk", "dimension": 4, "engine": "sst", "index_type": "hnsw",
               "distance_metric": "cosine", "description": "from the TS SDK"}),
        // aliases
        json!({"name": "aliased", "dimension": 4, "metric": "cosine", "storage_engine": "sst"}),
    ];
    for body in accepted {
        let name = body["name"].as_str().unwrap_or_default().to_string();
        let (status, text) = rest_post_create(&http, &base, body).await;
        assert!(
            status.is_success(),
            "{name}: must be accepted: {status} {text}"
        );
    }

    // `description` is persisted, not accepted-and-dropped.
    let mut grpc = connect_grpc(server.grpc_port).await;
    let legacy = grpc
        .get_collection(proximadb::proto::proximadb_v2::V2GetCollectionRequest {
            collection_id: "legacy_sdk".to_string(),
        })
        .await
        .expect("gRPC GetCollection")
        .into_inner();
    let legacy_config = legacy.config.expect("collection config");
    assert_eq!(
        legacy_config.description, "from the TS SDK",
        "the create description must be persisted"
    );
    // Round-1 review: the released SDKs' default `index_type: "hnsw"` must not
    // declare an index — that would opt every such collection into a
    // flush-time AXIS build and a different search route.
    assert!(
        legacy_config.index_specs.is_empty(),
        "index_type \"hnsw\" must not declare an index: {:?}",
        legacy_config.index_specs
    );

    // A body that is not JSON keeps axum's own status (here 415), not a 400.
    let resp = http
        .post(format!("{base}/api/v2/collections"))
        .body(r#"{"name": "no_content_type", "dimension": 4}"#)
        .send()
        .await
        .expect("REST create without content type");
    assert_eq!(resp.status(), reqwest::StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn schema_put_rejections_are_400_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "schema_put_rejects";
    rest_create(&http, &base, name).await;

    // A misspelt column field: the same strict body policy as create.
    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [{"name": "c", "data_type": "text", "filterabel": true}]}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.contains("invalid_argument") && body.contains("unknown field `filterabel`"),
        "{body}"
    );

    // A reserved system-metadata name: the caller's declaration, was a 500.
    let (status, body) = rest_put_schema(
        &http,
        &base,
        name,
        json!({"columns": [{"name": "__proxima_mine", "data_type": "text"}]}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("reserved"), "{body}");
}

#[tokio::test]
async fn grpc_schema_declaration_errors_are_invalid_argument_e2e() {
    use proximadb::proto::proximadb_v2::{
        ColumnDataType, CreateSchemaRequest, RecordSchema, TypedColumn,
    };

    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    rest_create(&http, &base, "grpc_sysdecl").await;

    let mut grpc = connect_grpc(server.grpc_port).await;
    let err = grpc
        .create_schema(CreateSchemaRequest {
            collection_id: "grpc_sysdecl".to_string(),
            schema: Some(RecordSchema {
                columns: vec![TypedColumn {
                    name: "tenant_id".to_string(),
                    data_type: ColumnDataType::Text as i32,
                    nullable: true,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("declaring a system column must be rejected");
    // TD-RESTGET-1: the caller's schema, not the server — was INTERNAL.
    assert_eq!(err.code(), tonic::Code::InvalidArgument, "{err:?}");
    assert!(err.message().contains("system column"), "{err:?}");
}

/// The exact create bodies the in-repo Rust and TypeScript SDKs emit for a
/// non-default index choice (TD-RESTGET-1 follow-up): accepted by the strict
/// server, and they do what they say — an IVF request declares an IVF index,
/// a Flat request is a routing policy and declares none.
#[tokio::test]
async fn sdk_index_shapes_are_accepted_e2e() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder().no_proxy().build().expect("http");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    let flat = json!({"name": "sdk_flat", "dimension": 4, "engine": "sst",
                      "distance_metric": "cosine", "index_policy": {"mode": "exact"}});
    let (status, text) = rest_post_create(&http, &base, flat).await;
    assert!(
        status.is_success(),
        "Flat (index_policy exact): {status} {text}"
    );
    for algorithm in ["ivf", "lsh", "pq", "annoy"] {
        let name = format!("sdk_{algorithm}");
        let body = json!({"name": name, "dimension": 4, "engine": "sst",
                          "distance_metric": "cosine",
                          "index_configs": [{"algorithm": algorithm, "is_primary": true}]});
        let (status, text) = rest_post_create(&http, &base, body).await;
        assert!(
            status.is_success(),
            "{algorithm} (index_configs): {status} {text}"
        );
    }

    let mut grpc = connect_grpc(server.grpc_port).await;
    // Every declared index reads back as itself — a server that accepted the
    // body but dropped or remapped the index would fail here.
    for algorithm in ["ivf", "lsh", "pq", "annoy"] {
        let specs = grpc_index_specs(&mut grpc, &format!("sdk_{algorithm}")).await;
        assert_eq!(specs.len(), 1, "{algorithm}: exactly one index: {specs:?}");
        assert!(
            specs[0].algorithm.eq_ignore_ascii_case(algorithm) && specs[0].is_primary,
            "{algorithm}: must read back as the primary {algorithm} index: {specs:?}"
        );
    }
    let flat = grpc_index_specs(&mut grpc, "sdk_flat").await;
    assert!(
        flat.is_empty(),
        "Flat is a routing policy, not an index: {flat:?}"
    );
}

async fn grpc_index_specs(
    grpc: &mut ProximaRecordServiceClient<tonic::transport::Channel>,
    name: &str,
) -> Vec<proximadb::proto::proximadb_v2::V2IndexSpec> {
    grpc.get_collection(proximadb::proto::proximadb_v2::V2GetCollectionRequest {
        collection_id: name.to_string(),
    })
    .await
    .expect("gRPC GetCollection")
    .into_inner()
    .config
    .expect("collection config")
    .index_specs
}
