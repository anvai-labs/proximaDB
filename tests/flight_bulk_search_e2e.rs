//! End-to-end test for the Arrow Flight `bulk_search` DoExchange on the
//! canonical v2 path (TD-FLIGHT-1 acceptance #2).
//!
//! Boots an in-process ProximaDB, inserts records over REST v2, then issues a
//! batched `bulk_search` DoExchange over a real Flight socket (query vectors as
//! a ProximaRecord Arrow batch + a top_k control frame) and asserts:
//!   - the exchange completes with the correct `query_count`, and
//!   - per-query result blocks arrive in **submission order** (the guarantee
//!     #1351 added — the old path dropped/​mis-ordered results and swallowed
//!     per-query errors).

use std::net::TcpListener;
use std::time::Duration;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{FlightData, FlightDescriptor};
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::network::arrow_ipc::codec::ArrowProtoCodec;
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct FlightServer {
    flight_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl FlightServer {
    async fn start() -> anyhow::Result<Self> {
        let flight_port = free_port();
        let rest_port = free_port();
        let tmp_data = TempDir::new()?;
        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = free_port();
        config.api.arrow_flight_port = flight_port;
        config.api.pg_port = Some(free_port());
        config.api.unified_mode = false;
        config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
            url: format!("file://{}", tmp_data.path().display()),
            ..Default::default()
        }];
        config.storage.wal_config.write_buffer_directory =
            format!("file://{}/wal", tmp_data.path().display());

        let mut db = ProximaDB::new(config).await?;
        db.start().await?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("server didn't become ready in 15s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        sleep(Duration::from_millis(300)).await;
        Ok(Self {
            flight_port,
            rest_port,
            db: Some(db),
            _tmp_data: tmp_data,
        })
    }

    fn flight_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.flight_port)
    }
    fn rest_base(&self) -> String {
        format!("http://127.0.0.1:{}", self.rest_port)
    }
}

impl Drop for FlightServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

/// Unit-norm vector whose cosine to `e_0` is exactly `first` (distinct first
/// components ⇒ a strict, deterministic cosine ordering).
fn unit_with_first(first: f32, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = first;
    v[1] = (1.0 - first * first).sqrt();
    v
}

fn query_record(oid: &str, vec: Vec<f32>) -> ProximaRecord {
    ProximaRecord {
        oid: oid.to_string(),
        embeddings: vec![EmbeddingCell::new_fp32(
            "q",
            "dense_vector",
            vec.len() as u32,
            vec,
        )],
        ..ProximaRecord::default()
    }
}

/// Encode a bulk_search DoExchange frame stream: descriptor on the first frame,
/// a top_k control frame, then the query-vector batch. Mirrors the graph
/// DoExchange encode (`graph_flight_socket_e2e.rs`) with the control-frame
/// injection TD-FLIGHT-1 added.
async fn encode_bulk_search_frames(
    query_batch: RecordBatch,
    collection: &str,
    top_k: u32,
) -> Vec<FlightData> {
    let descriptor = FlightDescriptor::new_path(vec!["bulk_search".into(), collection.into()]);
    let input = stream::iter(vec![Ok::<_, FlightError>(query_batch)]);
    let mut enc: Vec<FlightData> = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .build(input)
        .map(|r| r.expect("encode frame"))
        .collect()
        .await;
    // enc = [descriptor/schema frame, query-batch data frame]. Inject the
    // top_k control frame between them so the server applies it before decoding
    // the query batch.
    let control = FlightData {
        flight_descriptor: None,
        data_header: Default::default(),
        app_metadata: serde_json::to_vec(&json!({ "top_k": top_k }))
            .unwrap()
            .into(),
        data_body: Default::default(),
    };
    if enc.len() >= 2 {
        let data_frame = enc.remove(1);
        enc.push(control);
        enc.push(data_frame);
    } else {
        enc.push(control);
    }
    enc
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_bulk_search_preserves_order() {
    let server = FlightServer::start().await.expect("server start");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap();
    let base = server.rest_base();
    let name = format!(
        "flight_bulk_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dim: usize = 4;

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({
            "name": name, "dimension": dim, "engine": "sst",
            "distance_metric": "cosine", "enable_proxima_record": false,
        }))
        .send()
        .await
        .expect("create");
    assert!(create.status().is_success(), "create failed");

    // rec-i has cosine (1 - 0.1*i) to e_0 ⇒ rec-0..rec-3 are strictly ordered.
    let records: Vec<serde_json::Value> = (0..4)
        .map(|i| {
            let first = 1.0 - 0.1 * i as f32;
            json!({ "id": format!("rec-{i}"), "vector": unit_with_first(first, dim) })
        })
        .collect();
    let insert = http
        .post(format!("{base}/api/v2/collections/{name}/records/batch"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .expect("insert");
    assert!(insert.status().is_success(), "insert failed");
    sleep(Duration::from_millis(500)).await;

    // Submit 3 queries in a scrambled order; each query is one record's own
    // vector so its deterministic top-1 (top_k=1) is that record.
    // Submission order [rec-2, rec-0, rec-3] ⇒ expected result order the same.
    let query_records = vec![
        query_record("q0", unit_with_first(0.8, dim)), // → rec-2
        query_record("q1", unit_with_first(1.0, dim)), // → rec-0 (== e_0)
        query_record("q2", unit_with_first(0.7, dim)), // → rec-3
    ];
    let query_batch =
        ArrowProtoCodec::vector_records_to_batch(query_records, dim).expect("encode query batch");

    let channel = tonic::transport::Endpoint::from_shared(server.flight_url())
        .expect("endpoint")
        .connect()
        .await
        .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);

    let frames = encode_bulk_search_frames(query_batch, &name, 1).await;
    let resp = flight
        .do_exchange(stream::iter(frames))
        .await
        .expect("do_exchange")
        .into_inner();
    let all: Vec<FlightData> = resp
        .map_err(|e| FlightError::Tonic(Box::new(e)))
        .try_collect()
        .await
        .expect("collect response frames");

    // Split control/result frames; decode the result batches in stream order.
    let mut query_count: Option<u64> = None;
    let mut result_frames: Vec<FlightData> = Vec::new();
    for fd in all {
        if !fd.app_metadata.is_empty()
            && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&fd.app_metadata)
        {
            match v.get("type").and_then(|t| t.as_str()) {
                Some("complete") => {
                    query_count = v.get("query_count").and_then(|c| c.as_u64());
                    continue;
                }
                Some("error") => panic!("bulk_search returned a per-query error frame: {v}"),
                _ => {}
            }
        }
        result_frames.push(fd);
    }

    // The result stream carries one Schema+data block per query (bulk_search
    // encodes each query's result independently). FlightRecordBatchStream
    // rejects multiple Schema messages, so decode each block separately — a new
    // block starts at every schema frame (non-empty data_header, empty body).
    let mut blocks: Vec<Vec<FlightData>> = Vec::new();
    for fd in result_frames {
        let is_schema_start = !fd.data_header.is_empty() && fd.data_body.is_empty();
        if is_schema_start || blocks.is_empty() {
            blocks.push(vec![fd]);
        } else if let Some(last) = blocks.last_mut() {
            last.push(fd);
        }
    }

    let mut ids: Vec<String> = Vec::new();
    for block in blocks {
        let decoded = FlightRecordBatchStream::new_from_flight_data(stream::iter(
            block.into_iter().map(Ok::<_, FlightError>),
        ));
        let batches: Vec<RecordBatch> = decoded.try_collect().await.expect("decode result block");
        for b in &batches {
            let col = b
                .column_by_name("id")
                .expect("id column")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("id is StringArray");
            for i in 0..col.len() {
                ids.push(col.value(i).to_string());
            }
        }
    }

    assert_eq!(
        query_count,
        Some(3),
        "completion frame should report 3 queries"
    );
    // Submission order is preserved: each query's top-1 appears in order.
    assert_eq!(
        ids,
        vec![
            "rec-2".to_string(),
            "rec-0".to_string(),
            "rec-3".to_string()
        ],
        "bulk_search must preserve per-query submission order (TD-FLIGHT-1 #2)"
    );
}
