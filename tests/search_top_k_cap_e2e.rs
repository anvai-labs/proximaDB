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
