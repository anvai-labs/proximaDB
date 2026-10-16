//! TD-SECSCAN-1 §5: every vector-search surface shares ONE `top_k` cap.
//!
//! Before this test, `PROXIMADB_SEARCH_MAX_TOP_K` was enforced only by the REST
//! v2 search handler. Arrow Flight (`DoGet` `vector_search` ticket and the
//! `bulk_search` DoExchange control frame) and gRPC v2 (`Search`,
//! `SearchStream`) passed `top_k` straight through to an eager
//! `BinaryHeap::with_capacity(top_k)` — a `top_k = u32::MAX` request asked the
//! server for a multi-hundred-GB allocation and aborted the process.
//!
//! The over-cap value used here is deliberately modest (1,000,000): it is far
//! above the default cap of 10,000 but small enough that the UNFIXED server
//! serves it without harm, so this test fails on the defect (the request
//! succeeds) rather than by crashing the test host.
//!
//! Boots one in-process server with separate REST / gRPC / Flight ports and
//! drives each surface with a real client.

use std::net::TcpListener;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_flight::Ticket;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{FlightData, FlightDescriptor};
use futures::{StreamExt, TryStreamExt, stream};
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::network::arrow_ipc::codec::ArrowProtoCodec;
use proximadb::proto::proximadb_v2::TypedSearchRequest;
use proximadb::proto::proximadb_v2::proxima_record_service_client::ProximaRecordServiceClient;
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

/// Far above the default cap (10,000), harmless to serve on an unfixed build.
const OVER_CAP_TOP_K: u32 = 1_000_000;
/// The documented default cap — exactly-at-cap must still be served.
const AT_CAP_TOP_K: u32 = 10_000;
const DIM: usize = 4;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct Server {
    rest_port: u16,
    grpc_port: u16,
    flight_port: u16,
    db: Option<ProximaDB>,
    _tmp: TempDir,
}

impl Server {
    async fn start() -> anyhow::Result<Self> {
        let rest_port = free_port();
        let grpc_port = free_port();
        let flight_port = free_port();
        let tmp = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp.path().to_path_buf();
        config.api.unified_mode = false;
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.arrow_flight_port = flight_port;
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
            flight_port,
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

fn query_vector() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    v
}

fn grpc_search_request(collection: &str, top_k: u32) -> TypedSearchRequest {
    TypedSearchRequest {
        collection_id: collection.to_string(),
        query_vector: query_vector(),
        top_k,
        ..Default::default()
    }
}

fn flight_ticket(collection: &str, top_k: u32) -> Ticket {
    Ticket {
        ticket: serde_json::to_vec(&json!({
            "type": "vector_search",
            "collection_id": collection,
            "query_vector": query_vector(),
            "top_k": top_k,
            "include_vector": false,
        }))
        .expect("ticket json")
        .into(),
    }
}

async fn flight_do_get(
    flight: &mut FlightServiceClient<tonic::transport::Channel>,
    collection: &str,
    top_k: u32,
) -> Result<Vec<RecordBatch>, tonic::Status> {
    let response = flight.do_get(flight_ticket(collection, top_k)).await?;
    let stream = FlightRecordBatchStream::new_from_flight_data(
        response
            .into_inner()
            .map_err(|s| FlightError::Tonic(Box::new(s))),
    );
    stream
        .try_collect()
        .await
        .map_err(|e| tonic::Status::internal(format!("decode: {e}")))
}

/// `bulk_search` DoExchange: descriptor/schema frame, a `{"top_k": k}`
/// control frame, then one query-vector batch (mirrors
/// `flight_bulk_search_e2e.rs`). Returns the number of response frames served.
async fn flight_bulk_search(
    flight: &mut FlightServiceClient<tonic::transport::Channel>,
    collection: &str,
    top_k: u32,
) -> Result<usize, tonic::Status> {
    let query = ProximaRecord {
        oid: "q0".to_string(),
        embeddings: vec![EmbeddingCell {
            model_id: "q".to_string(),
            modality: "dense_vector".to_string(),
            dim: DIM as u32,
            values: EmbeddingValues::Fp32(query_vector()),
            ..Default::default()
        }],
        ..ProximaRecord::default()
    };
    let batch = ArrowProtoCodec::vector_records_to_batch(vec![query], DIM).expect("query batch");
    let descriptor = FlightDescriptor::new_path(vec!["bulk_search".into(), collection.into()]);
    let mut frames: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .build(stream::iter(vec![Ok::<_, FlightError>(batch)]))
        .map(|r| r.expect("encode frame"))
        .collect()
        .await;
    let control = FlightData {
        flight_descriptor: None,
        data_header: Default::default(),
        app_metadata: serde_json::to_vec(&json!({ "top_k": top_k }))
            .expect("control json")
            .into(),
        data_body: Default::default(),
    };
    let data_frame = frames.remove(1);
    frames.push(control);
    frames.push(data_frame);

    let response = flight.do_exchange(stream::iter(frames)).await?;
    let served: Vec<FlightData> = response.into_inner().try_collect().await?;
    Ok(served.len())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_search_surface_rejects_over_cap_top_k() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "secscan1_topk_cap";

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name, "dimension": DIM, "engine": "sst",
            "distance_metric": "cosine", "enable_proxima_record": false,
        }))
        .send()
        .await
        .expect("create");
    assert!(create.status().is_success(), "create collection failed");
    let records: Vec<serde_json::Value> = (0..3)
        .map(|i| {
            let mut v = vec![0.0f32; DIM];
            v[i % DIM] = 1.0;
            json!({ "id": format!("rec-{i}"), "vector": v })
        })
        .collect();
    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("insert");
    assert!(insert.status().is_success(), "insert failed");
    sleep(Duration::from_millis(300)).await;

    // --- REST v2 (the pre-existing cap; regression guard for the shared helper).
    let rest_over = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({ "vector": query_vector(), "top_k": OVER_CAP_TOP_K }))
        .send()
        .await
        .expect("rest search");
    assert_eq!(
        rest_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "REST v2 must reject top_k above the cap"
    );
    let rest_at = http
        .post(format!("{base}/api/v2/collections/{name}/search"))
        .json(&json!({ "vector": query_vector(), "top_k": AT_CAP_TOP_K }))
        .send()
        .await
        .expect("rest search at cap");
    assert!(
        rest_at.status().is_success(),
        "REST v2 must serve top_k exactly at the cap, got {}",
        rest_at.status()
    );

    // --- Arrow Flight DoGet `vector_search` ticket.
    let channel =
        tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{}", server.flight_port))
            .expect("flight endpoint")
            .connect()
            .await
            .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);

    let at_cap = flight_do_get(&mut flight, name, AT_CAP_TOP_K).await;
    assert!(
        at_cap.is_ok(),
        "Flight DoGet must serve top_k exactly at the cap: {at_cap:?}"
    );
    // Over-cap violations are collected (not asserted one by one) so a single
    // run reports EVERY uncapped surface, not just the first.
    let mut violations: Vec<String> = Vec::new();
    let mut expect_invalid = |surface: &str, outcome: Result<String, tonic::Status>| match outcome {
        Ok(served) => violations.push(format!(
            "{surface}: accepted top_k={OVER_CAP_TOP_K} ({served}) — no cap on this surface"
        )),
        Err(status) if status.code() != tonic::Code::InvalidArgument => violations.push(format!(
            "{surface}: over-cap top_k must be InvalidArgument, got {status:?}"
        )),
        Err(_) => {}
    };

    expect_invalid(
        "Flight DoGet",
        flight_do_get(&mut flight, name, OVER_CAP_TOP_K)
            .await
            .map(|b| format!("{} batches", b.len())),
    );

    let at_cap = flight_bulk_search(&mut flight, name, AT_CAP_TOP_K).await;
    assert!(
        at_cap.is_ok(),
        "Flight bulk_search must serve top_k exactly at the cap: {at_cap:?}"
    );
    expect_invalid(
        "Flight bulk_search",
        flight_bulk_search(&mut flight, name, OVER_CAP_TOP_K)
            .await
            .map(|n| format!("{n} frames")),
    );

    // --- gRPC v2 Search + SearchStream.
    let mut grpc = connect_grpc(server.grpc_port).await;
    let at_cap = grpc.search(grpc_search_request(name, AT_CAP_TOP_K)).await;
    assert!(
        at_cap.is_ok(),
        "gRPC Search must serve top_k exactly at the cap: {at_cap:?}"
    );
    expect_invalid(
        "gRPC Search",
        grpc.search(grpc_search_request(name, OVER_CAP_TOP_K))
            .await
            .map(|r| format!("{} results", r.into_inner().results.len())),
    );
    expect_invalid(
        "gRPC SearchStream",
        grpc.search_stream(grpc_search_request(name, OVER_CAP_TOP_K))
            .await
            .map(|_| "stream opened".to_string()),
    );

    assert!(
        violations.is_empty(),
        "uncapped top_k surfaces:\n  {}",
        violations.join("\n  ")
    );

    // --- REST v1 progressive search (POST /api/v2/progressive/search/{id}).
    // v1 proto shape: {"queries":[{"vector":[...]}], "top_k": n}.
    let progressive_over = http
        .post(format!("{base}/api/v2/progressive/search/{name}"))
        .json(&json!({
            "collection_id": name,
            "queries": [{ "vector": query_vector() }],
            "top_k": OVER_CAP_TOP_K
        }))
        .send()
        .await
        .expect("progressive search");
    assert_eq!(
        progressive_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "progressive search must reject top_k above the cap: {}",
        progressive_over.status()
    );
    let progressive_at = http
        .post(format!("{base}/api/v2/progressive/search/{name}"))
        .json(&json!({
            "collection_id": name,
            "queries": [{ "vector": query_vector() }],
            "top_k": AT_CAP_TOP_K
        }))
        .send()
        .await
        .expect("progressive search at cap");
    assert!(
        progressive_at.status().is_success(),
        "progressive search must serve top_k exactly at the cap, got {}",
        progressive_at.status()
    );

    // --- Hybrid search (POST /api/v2/hybrid/search).
    let hybrid_over = http
        .post(format!("{base}/api/v2/hybrid/search"))
        .json(&json!({
            "collection": name,
            "vector": query_vector(),
            "top_k": OVER_CAP_TOP_K
        }))
        .send()
        .await
        .expect("hybrid search");
    assert_eq!(
        hybrid_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "hybrid search must reject top_k above the cap: {}",
        hybrid_over.status()
    );
    let hybrid_at = http
        .post(format!("{base}/api/v2/hybrid/search"))
        .json(&json!({
            "collection": name,
            "vector": query_vector(),
            "top_k": AT_CAP_TOP_K
        }))
        .send()
        .await
        .expect("hybrid search at cap");
    assert!(
        hybrid_at.status().is_success(),
        "hybrid search must serve top_k exactly at the cap, got {}: {}",
        hybrid_at.status(),
        hybrid_at.text().await.unwrap_or_default()
    );

    // --- Legacy REST entity search (POST /api/v1/collections/{id}/entities/search).
    // This handler hardcoded its own `top_k > 10000` bound and message; the
    // cap must come from the shared limit so PROXIMADB_SEARCH_MAX_TOP_K
    // governs it (the error text naming the env var is the discriminator).
    let legacy_over = http
        .post(format!("{base}/api/v1/collections/{name}/entities/search"))
        .json(&json!({ "query_vector": query_vector(), "top_k": OVER_CAP_TOP_K }))
        .send()
        .await
        .expect("legacy entity search");
    assert_eq!(
        legacy_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "legacy entity search must reject top_k above the cap: {}",
        legacy_over.status()
    );
    let legacy_body = legacy_over.text().await.unwrap_or_default();
    assert!(
        legacy_body.contains("PROXIMADB_SEARCH_MAX_TOP_K"),
        "legacy entity search rejection must come from the shared limit (named the env var), got: {legacy_body}"
    );
    let legacy_zero = http
        .post(format!("{base}/api/v1/collections/{name}/entities/search"))
        .json(&json!({ "query_vector": query_vector(), "top_k": 0 }))
        .send()
        .await
        .expect("legacy entity search zero");
    assert_eq!(
        legacy_zero.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "legacy entity search must keep rejecting top_k=0"
    );

    // --- Graph fusion search (POST /api/v2/graphs/{id}/fusion-search).
    // The limit cap is checked before any collection/graph lookup, so an
    // over-cap request is 400 even without a registered graph.
    let fusion_over = http
        .post(format!("{base}/api/v2/graphs/secscan1_graph/fusion-search"))
        .json(&json!({
            "vector_collection": name,
            "query_vector": query_vector(),
            "limit": OVER_CAP_TOP_K
        }))
        .send()
        .await
        .expect("fusion search");
    assert_eq!(
        fusion_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "fusion search must reject limit above the cap: {}",
        fusion_over.status()
    );
    let fusion_seeds_over = http
        .post(format!("{base}/api/v2/graphs/secscan1_graph/fusion-search"))
        .json(&json!({
            "vector_collection": name,
            "query_vector": query_vector(),
            "limit": 10,
            "max_seeds": OVER_CAP_TOP_K
        }))
        .send()
        .await
        .expect("fusion search max_seeds");
    assert_eq!(
        fusion_seeds_over.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "fusion search must reject max_seeds above the cap: {}",
        fusion_seeds_over.status()
    );
    let fusion_at = http
        .post(format!("{base}/api/v2/graphs/secscan1_graph/fusion-search"))
        .json(&json!({
            "vector_collection": name,
            "query_vector": query_vector(),
            "limit": AT_CAP_TOP_K
        }))
        .send()
        .await
        .expect("fusion search at cap");
    assert!(
        fusion_at.status().is_success(),
        "fusion search must serve limit exactly at the cap, got {}: {}",
        fusion_at.status(),
        fusion_at.text().await.unwrap_or_default()
    );
}

/// `POST /api/v2/aql/execute` body: `FROM <source> WHERE <predicate>`.
fn aql_body(source: &str, predicate: serde_json::Value) -> serde_json::Value {
    json!({
        "query": {
            "find": { "projections": [] },
            "from": { "Source": { "name": source, "alias": null } },
            "where_clause": { "predicate": predicate }
        }
    })
}

fn semantic_match(top_k: u32) -> serde_json::Value {
    json!({ "SemanticMatch": {
        "field": "embedding", "query": "", "threshold": 0.0, "top_k": top_k
    }})
}

/// Vector source: query vector + `SemanticMatch(top_k)`.
fn vector_predicate(top_k: u32) -> serde_json::Value {
    json!({ "And": {
        "lhs": { "Equals": { "field": "vector", "value": { "Vector": query_vector() } } },
        "rhs": semantic_match(top_k)
    }})
}

/// Agent-memory scope: tenant + query vector + `SemanticMatch(top_k)`.
fn memory_predicate(top_k: u32) -> serde_json::Value {
    json!({ "And": {
        "lhs": { "Equals": { "field": "tenant_id", "value": { "String": "t-secscan1" } } },
        "rhs": { "And": {
            "lhs": { "Equals": { "field": "vector", "value": { "Vector": query_vector() } } },
            "rhs": semantic_match(top_k)
        }}
    }})
}

/// TD-SECSCAN-1 §5 follow-up: AQL `SEMANTIC_MATCH` (the `vector` and `memory`
/// sources) reaches `VectorOperationsService::unified_search_v1`, which had no
/// cap. An over-cap `top_k` must be a 400 naming the env override; at-cap is
/// still served (the memory source's 5x recall oversample is clamped to the
/// cap rather than tripping it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aql_semantic_match_rejects_over_cap_top_k() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    // The AQL `vector` / `memory` sources search the collection named after
    // the source.
    for name in ["vector", "memory"] {
        let create = http
            .post(format!("{base}/api/v2/collections"))
            .json(&json!({
                "name": name, "dimension": DIM, "engine": "sst",
                "distance_metric": "cosine", "enable_proxima_record": false,
            }))
            .send()
            .await
            .expect("create");
        assert!(
            create.status().is_success(),
            "create collection {name} failed: {}",
            create.text().await.unwrap_or_default()
        );
        let records: Vec<serde_json::Value> = (0..3)
            .map(|i| {
                let mut v = vec![0.0f32; DIM];
                v[i % DIM] = 1.0;
                json!({
                    "id": format!("{name}-{i}"), "vector": v,
                    "props": { "tenant_id": "t-secscan1" }
                })
            })
            .collect();
        let insert = http
            .post(format!("{base}/api/v2/collections/{name}/records/batch"))
            .json(&json!({ "records": records }))
            .send()
            .await
            .expect("insert");
        assert!(insert.status().is_success(), "insert into {name} failed");
    }
    sleep(Duration::from_millis(300)).await;

    let aql = |body: serde_json::Value| {
        let http = http.clone();
        let url = format!("{base}/api/v2/aql/execute");
        async move {
            http.post(url)
                .json(&body)
                .send()
                .await
                .expect("aql execute")
        }
    };

    for (source, over, at) in [
        (
            "vector",
            vector_predicate(OVER_CAP_TOP_K),
            vector_predicate(AT_CAP_TOP_K),
        ),
        (
            "memory",
            memory_predicate(OVER_CAP_TOP_K),
            memory_predicate(AT_CAP_TOP_K),
        ),
    ] {
        let resp = aql(aql_body(source, over)).await;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "AQL {source} SEMANTIC_MATCH must reject top_k above the cap: {body}"
        );
        assert!(
            body.contains("PROXIMADB_SEARCH_MAX_TOP_K"),
            "AQL {source} rejection must name the env override, got: {body}"
        );
        assert!(
            body.contains("query_id"),
            "AQL {source} rejection must carry the audit query_id, got: {body}"
        );

        let resp = aql(aql_body(source, at)).await;
        let status = resp.status();
        let body: serde_json::Value = resp.json().await.unwrap_or_default();
        assert!(
            status.is_success(),
            "AQL {source} must serve top_k exactly at the cap, got {status}: {body}"
        );
        // At-cap runs to completion; for `memory` this also proves its 5x
        // recall pool is clamped to the cap rather than rejected.
        assert_eq!(
            body["audit_trail"]["outcome"],
            json!("Success"),
            "AQL {source} at-cap search must succeed: {body}"
        );
        // The query-vector predicate is consumed by the source; the executor's
        // WHERE post-filter must not drop every hit for lacking a `vector` field.
        let rows = body["result"].as_array().map_or(0, Vec::len);
        assert!(
            rows > 0,
            "AQL {source} correct-dimension search must return rows: {body}"
        );
    }

    // A vector-source query without a query vector is a client error (400),
    // not an empty vector reaching the distance kernel's dimension assert
    // (which panicked the handler and dropped the connection before the fix).
    let resp = aql(aql_body("vector", semantic_match(AT_CAP_TOP_K))).await;
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "AQL vector search without a query vector must be a 400: {}",
        resp.text().await.unwrap_or_default()
    );
}

/// AQL vector-source predicate with an explicit query vector.
fn vector_predicate_with(vector: Vec<f32>, top_k: u32) -> serde_json::Value {
    json!({ "And": {
        "lhs": { "Equals": { "field": "vector", "value": { "Vector": vector } } },
        "rhs": semantic_match(top_k)
    }})
}

/// Agent-memory predicate with an explicit (possibly absent) query vector.
fn memory_predicate_with(vector: Option<Vec<f32>>, top_k: u32) -> serde_json::Value {
    let tenant = json!({ "Equals": { "field": "tenant_id", "value": { "String": "t-secscan1" } } });
    match vector {
        Some(v) => json!({ "And": {
            "lhs": tenant,
            "rhs": { "And": {
                "lhs": { "Equals": { "field": "vector", "value": { "Vector": v } } },
                "rhs": semantic_match(top_k)
            }}
        }}),
        None => json!({ "And": { "lhs": tenant, "rhs": semantic_match(top_k) } }),
    }
}

/// A query vector whose dimension does not match the collection (or is
/// empty) used to reach `assert_eq!` in the distance kernel. Under the
/// `release-server` profile (`panic = "abort"`) that killed the server
/// process; under the test profile it panics the request task and drops the
/// connection. Every surface must instead answer with a client error, and the
/// server must keep serving afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_dimension_query_vectors_are_client_errors() {
    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);

    // `vector` / `memory` back the AQL sources; `secscan1_dim` the REST/gRPC cases.
    for name in ["vector", "memory", "secscan1_dim"] {
        let create = http
            .post(format!("{base}/api/v2/collections"))
            .json(&json!({
                "name": name, "dimension": DIM, "engine": "sst",
                "distance_metric": "cosine", "enable_proxima_record": false,
            }))
            .send()
            .await
            .expect("create");
        assert!(create.status().is_success(), "create {name} failed");
        let records: Vec<serde_json::Value> = (0..3)
            .map(|i| {
                let mut v = vec![0.0f32; DIM];
                v[i % DIM] = 1.0;
                json!({
                    "id": format!("{name}-{i}"), "vector": v,
                    "props": { "tenant_id": "t-secscan1" }
                })
            })
            .collect();
        let insert = http
            .post(format!("{base}/api/v2/collections/{name}/records/batch"))
            .json(&json!({ "records": records }))
            .send()
            .await
            .expect("insert");
        assert!(insert.status().is_success(), "insert into {name} failed");
    }
    sleep(Duration::from_millis(300)).await;

    let short = vec![1.0f32; DIM - 1];
    let mut failures: Vec<String> = Vec::new();

    // --- REST v2 collection search: wrong dimension.
    let outcome = http
        .post(format!("{base}/api/v2/collections/secscan1_dim/search"))
        .json(&json!({ "vector": short, "top_k": 5 }))
        .send()
        .await;
    match outcome {
        Ok(r) if r.status() == reqwest::StatusCode::BAD_REQUEST => {}
        Ok(r) => failures.push(format!("REST v2 wrong-dim: got {}", r.status())),
        Err(e) => failures.push(format!("REST v2 wrong-dim: connection failed ({e})")),
    }

    // --- AQL: vector wrong-dim, memory wrong-dim, memory without a vector.
    let aql_cases = [
        (
            "AQL vector wrong-dim",
            "vector",
            vector_predicate_with(short.clone(), 5),
        ),
        (
            "AQL memory wrong-dim",
            "memory",
            memory_predicate_with(Some(short.clone()), 5),
        ),
        (
            "AQL memory no vector",
            "memory",
            memory_predicate_with(None, 5),
        ),
    ];
    for (label, source, predicate) in aql_cases {
        let outcome = http
            .post(format!("{base}/api/v2/aql/execute"))
            .json(&aql_body(source, predicate))
            .send()
            .await;
        match outcome {
            Ok(r) if r.status() == reqwest::StatusCode::BAD_REQUEST => {}
            Ok(r) => {
                let status = r.status();
                failures.push(format!(
                    "{label}: got {status}: {}",
                    r.text().await.unwrap_or_default()
                ))
            }
            Err(e) => failures.push(format!("{label}: connection failed ({e})")),
        }
    }

    // --- gRPC v2 Search: wrong dimension and empty vector.
    let mut grpc = connect_grpc(server.grpc_port).await;
    for (label, vector) in [
        ("gRPC wrong-dim", short.clone()),
        ("gRPC empty", Vec::new()),
    ] {
        let request = TypedSearchRequest {
            collection_id: "secscan1_dim".to_string(),
            query_vector: vector,
            top_k: 5,
            ..Default::default()
        };
        match grpc.search(request).await {
            Err(status) if status.code() == tonic::Code::InvalidArgument => {}
            Err(status) => failures.push(format!("{label}: got {status:?}")),
            Ok(r) => failures.push(format!(
                "{label}: accepted ({} results)",
                r.into_inner().results.len()
            )),
        }
    }

    // The server is still alive and serving correct-dimension queries.
    let alive = http
        .post(format!("{base}/api/v2/collections/secscan1_dim/search"))
        .json(&json!({ "vector": query_vector(), "top_k": 5 }))
        .send()
        .await;
    match alive {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => failures.push(format!("follow-up REST search: got {}", r.status())),
        Err(e) => failures.push(format!("follow-up REST search: server gone ({e})")),
    }
    let alive = grpc.search(grpc_search_request("secscan1_dim", 5)).await;
    if let Err(status) = alive {
        failures.push(format!("follow-up gRPC search: {status:?}"));
    }

    assert!(
        failures.is_empty(),
        "wrong-dimension / empty query vectors must be client errors:\n  {}",
        failures.join("\n  ")
    );
}

/// True when a client-facing error names a collection by its internal
/// numeric object id (`collection '16'`) instead of the name the client used.
fn names_collection_by_internal_id(message: &str) -> bool {
    message.match_indices("collection '").any(|(i, m)| {
        message[i + m.len()..]
            .split('\'')
            .next()
            .is_some_and(|quoted| !quoted.is_empty() && quoted.bytes().all(|b| b.is_ascii_digit()))
    })
}

/// TD-SECSCAN-1 §5 residuals: a wrong-dimension query vector is the client's
/// error on EVERY search route — HTTP 400 / gRPC `InvalidArgument`, never 500
/// / `Internal` — and the message carries the cause (the dimension) and the
/// collection name the client used, never the internal numeric object id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_dimension_is_a_client_error_on_every_search_route() {
    use proximadb::proto::proximadb_v2 as pv2;
    use pv2::proxima_entity_service_client::ProximaEntityServiceClient;
    use pv2::proxima_fusion_service_client::ProximaFusionServiceClient;

    let server = Server::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .expect("http client");
    let base = format!("http://127.0.0.1:{}", server.rest_port);
    let name = "secscan1_route";

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name, "dimension": DIM, "engine": "sst",
            "distance_metric": "cosine", "enable_proxima_record": false,
        }))
        .send()
        .await
        .expect("create");
    assert!(create.status().is_success(), "create {name} failed");
    // The canonical object id the server assigned (create returns it).
    let created: serde_json::Value = create.json().await.expect("create response json");
    let object_id = created["collection_id"]
        .as_str()
        .filter(|id| !id.is_empty() && *id != name)
        .map(str::to_string)
        .unwrap_or_else(|| panic!("create must return the object id: {created}"));
    let records: Vec<serde_json::Value> = (0..3)
        .map(|i| {
            let mut v = vec![0.0f32; DIM];
            v[i % DIM] = 1.0;
            let topic = if i % 2 == 0 { "alpha" } else { "beta" };
            json!({ "id": format!("{name}-{i}"), "vector": v, "props": { "topic": topic } })
        })
        .collect();
    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("insert");
    assert!(insert.status().is_success(), "insert into {name} failed");
    sleep(Duration::from_millis(300)).await;

    let short = vec![1.0f32; DIM - 1];
    let mut failures: Vec<String> = Vec::new();

    let rest_cases = [
        (
            "REST v2 collection search",
            format!("{base}/api/v2/collections/{name}/search"),
            json!({ "vector": short, "top_k": 5 }),
        ),
        (
            "REST v1 progressive search",
            format!("{base}/api/v2/progressive/search/{name}"),
            json!({ "collection_id": name, "queries": [{ "vector": short }], "top_k": 5 }),
        ),
        (
            "REST hybrid search",
            format!("{base}/api/v2/hybrid/search"),
            json!({ "collection": name, "vector": short, "top_k": 5 }),
        ),
        (
            "REST v1 entity search",
            format!("{base}/api/v1/collections/{name}/entities/search"),
            json!({ "query_vector": short, "top_k": 5 }),
        ),
        (
            "REST v2 entity search",
            format!("{base}/api/v2/collections/{name}/entities/search"),
            json!({ "query_vector": short, "top_k": 5 }),
        ),
        (
            "REST graph fusion-search",
            format!("{base}/api/v2/graphs/{name}/fusion-search"),
            json!({ "vector_collection": name, "query_vector": short, "limit": 5 }),
        ),
        (
            "REST rank search",
            format!("{base}/api/v2/rank/search"),
            json!({ "collection": name, "query_vector": short, "k": 5 }),
        ),
    ];
    for (label, url, body) in rest_cases {
        match http.post(url).json(&body).send().await {
            Ok(r) => {
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                if label == "REST rank search" && status == reqwest::StatusCode::NOT_IMPLEMENTED {
                    // RankServices not injected in this server mode.
                    eprintln!("REST rank search: not wired here (501), skipped");
                    continue;
                }
                if status != reqwest::StatusCode::BAD_REQUEST {
                    failures.push(format!("{label}: got {status}: {text}"));
                    continue;
                }
                if !text.contains("dimension") {
                    failures.push(format!("{label}: 400 hides the dimension cause: {text}"));
                }
                if !text.contains(name) {
                    failures.push(format!(
                        "{label}: message does not name the collection '{name}': {text}"
                    ));
                }
                if names_collection_by_internal_id(&text) {
                    failures.push(format!(
                        "{label}: message names the internal object id: {text}"
                    ));
                }
            }
            Err(e) => failures.push(format!("{label}: connection failed ({e})")),
        }
    }

    // A request that addresses the collection by its object id reaches it and
    // gets the dimension error, built only from what the request sent: the
    // collection's name, which the request did not contain, must not appear.
    let by_id = [
        (
            "REST v1 progressive search by object id",
            format!("{base}/api/v2/progressive/search/{object_id}"),
            json!({ "collection_id": object_id, "queries": [{ "vector": short }], "top_k": 5 }),
        ),
        (
            "REST v2 collection search by object id",
            format!("{base}/api/v2/collections/{object_id}/search"),
            json!({ "vector": short, "top_k": 5 }),
        ),
    ];
    for (label, url, body) in by_id {
        match http.post(url).json(&body).send().await {
            Ok(r) => {
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                if status != reqwest::StatusCode::BAD_REQUEST || !text.contains("dimension") {
                    failures.push(format!(
                        "{label}: expected 400 with the dimension cause, got {status}: {text}"
                    ));
                }
                if text.contains(name) {
                    failures.push(format!(
                        "{label}: echoed a name the request did not contain: {text}"
                    ));
                }
            }
            Err(e) => failures.push(format!("{label}: request failed ({e})")),
        }
    }

    // Arrow Flight DoGet `vector_search` ticket.
    let channel =
        tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{}", server.flight_port))
            .expect("flight endpoint")
            .connect()
            .await
            .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);
    let ticket = Ticket {
        ticket: serde_json::to_vec(&json!({
            "type": "vector_search",
            "collection_id": name,
            "query_vector": short,
            "top_k": 5,
            "include_vector": false,
        }))
        .expect("ticket json")
        .into(),
    };
    match flight.do_get(ticket).await {
        Err(status)
            if status.code() == tonic::Code::InvalidArgument
                && status.message().contains("dimension")
                && status.message().contains(name) => {}
        Err(status) => failures.push(format!("Flight DoGet: got {status:?}")),
        Ok(_) => failures.push("Flight DoGet: accepted a wrong-dim vector".to_string()),
    }

    // Analytics entanglement loads records through `unified_search_native`
    // with an EMPTY query vector by design; the query-vector gate must not
    // break it (it may still 400 for data reasons, never 500 / "is empty").
    match http
        .get(format!(
            "{base}/api/v2/analytics/collections/{name}/entanglement?topic_field=topic&limit=10"
        ))
        .send()
        .await
    {
        Ok(r) => {
            let status = r.status();
            let text = r.text().await.unwrap_or_default();
            eprintln!("analytics entanglement: {status}: {text}");
            if status.is_server_error() || text.contains("is empty") {
                failures.push(format!("analytics entanglement: got {status}: {text}"));
            }
        }
        Err(e) => failures.push(format!("analytics entanglement: connection failed ({e})")),
    }

    let grpc_endpoint = format!("http://127.0.0.1:{}", server.grpc_port);
    let mut fusion = ProximaFusionServiceClient::connect(grpc_endpoint.clone())
        .await
        .expect("fusion gRPC connect");
    let outcome = fusion
        .fusion_search(pv2::FusionSearchRequest {
            graph_id: name.to_string(),
            vector_collection: name.to_string(),
            query_vector: short.clone(),
            limit: 5,
            ..Default::default()
        })
        .await;
    match outcome {
        Err(status)
            if status.code() == tonic::Code::InvalidArgument
                && status.message().contains("dimension")
                && !names_collection_by_internal_id(status.message()) => {}
        Err(status) => failures.push(format!("gRPC FusionSearch: got {status:?}")),
        Ok(_) => failures.push("gRPC FusionSearch: accepted a wrong-dim vector".to_string()),
    }

    let mut entities = ProximaEntityServiceClient::connect(grpc_endpoint)
        .await
        .expect("entity gRPC connect");
    let outcome = entities
        .search_entities(pv2::SearchEntitiesRequest {
            collection_id: name.to_string(),
            similar: Some(pv2::SimilarQuery {
                query: Some(pv2::similar_query::Query::Vector(pv2::VectorData {
                    values: short.clone(),
                })),
                ..Default::default()
            }),
            top_k: 5,
            ..Default::default()
        })
        .await;
    match outcome {
        Err(status)
            if status.code() == tonic::Code::InvalidArgument
                && status.message().contains("dimension") => {}
        Err(status) => failures.push(format!("gRPC SearchEntities: got {status:?}")),
        Ok(_) => failures.push("gRPC SearchEntities: accepted a wrong-dim vector".to_string()),
    }

    // Still alive and serving a correct-dimension query on the same routes.
    let alive = http
        .post(format!("{base}/api/v2/progressive/search/{name}"))
        .json(&json!({ "collection_id": name, "queries": [{ "vector": query_vector() }], "top_k": 5 }))
        .send()
        .await;
    match alive {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => failures.push(format!("follow-up progressive search: got {}", r.status())),
        Err(e) => failures.push(format!("follow-up progressive search: server gone ({e})")),
    }
    let alive = http
        .post(format!("{base}/api/v2/graphs/{name}/fusion-search"))
        .json(&json!({ "vector_collection": name, "query_vector": query_vector(), "limit": 5 }))
        .send()
        .await;
    match alive {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => failures.push(format!("follow-up fusion search: got {}", r.status())),
        Err(e) => failures.push(format!("follow-up fusion search: server gone ({e})")),
    }

    assert!(
        failures.is_empty(),
        "wrong-dimension query vectors must be client errors on every route:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn internal_id_detector() {
    assert!(names_collection_by_internal_id(
        "Query vector has dimension 3 but collection '16' expects dimension 4"
    ));
    assert!(!names_collection_by_internal_id(
        "Query vector has dimension 3 but the collection expects dimension 4"
    ));
    assert!(!names_collection_by_internal_id(
        "Query vector has dimension 3 but collection 'secscan1_route' expects dimension 4"
    ));
}
