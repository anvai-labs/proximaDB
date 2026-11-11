//! End-to-end test of the batched columnar graph path over the **real Arrow
//! Flight transport** — boots an in-process ProximaDB server, creates a graph
//! via REST, then drives `graph_nodes` `DoExchange` (bulk columnar ingest) and
//! `DoGet` (paginated columnar export) across a real `FlightServiceClient` over
//! TCP, and verifies the nodes round-trip and are queryable.
//!
//! This closes the transport gap above the service-level coverage in
//! `tests/graph_flight_columnar_test.rs` (codec + graph batch APIs) and the unit
//! tests in `graph_codec` — neither exercises the descriptor parsing, FlightData
//! streaming, or the `FlightDataEncoder` across a socket.

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{FlightData, FlightDescriptor, Ticket};
use futures::{StreamExt, TryStreamExt, stream};
use proximadb::core::Config;
use proximadb::database::ProximaDB;
use proximadb::graph::model::{EmbeddingVersion, Node, PropertyValue, property_value::Value};
use proximadb::network::arrow_ipc::graph_codec;
use tempfile::TempDir;
use tokio::time::sleep;

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind port 0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct FlightTestServer {
    flight_port: u16,
    rest_port: u16,
    db: Option<ProximaDB>,
    _tmp_data: TempDir,
}

impl FlightTestServer {
    async fn start() -> anyhow::Result<Self> {
        let flight_port = free_port();
        let rest_port = free_port();
        let grpc_port = free_port();
        let pg_port = free_port();
        let tmp_data = TempDir::new()?;

        let mut config = Config::default();
        config.server.bind_address = "127.0.0.1".to_string();
        config.server.port = rest_port;
        config.server.data_dir = tmp_data.path().to_path_buf();
        config.api.rest_port = rest_port;
        config.api.grpc_port = grpc_port;
        config.api.arrow_flight_port = flight_port;
        config.api.pg_port = Some(pg_port);
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
            .timeout(Duration::from_secs(2))
            .no_proxy()
            .build()?;
        let health_url = format!("http://127.0.0.1:{}/health", rest_port);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match http.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => {
                    if std::time::Instant::now() > deadline {
                        anyhow::bail!("REST server not ready on {rest_port} within 15s");
                    }
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        // Grace for the Arrow Flight listener.
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

impl Drop for FlightTestServer {
    fn drop(&mut self) {
        if let Some(mut db) = self.db.take() {
            tokio::spawn(async move {
                let _ = db.shutdown().await;
            });
        }
    }
}

fn sample_node(id: &str, dim: usize) -> Node {
    let mut properties = HashMap::new();
    properties.insert(
        "name".to_string(),
        PropertyValue {
            value: Some(Value::StringValue(format!("name-{id}"))),
        },
    );
    Node {
        id: id.to_string(),
        labels: vec!["Person".to_string()],
        properties,
        embedding: (dim > 0).then(|| EmbeddingVersion {
            model_id: "bge".to_string(),
            model_version: "v1".to_string(),
            vector: (0..dim).map(|i| i as f32 * 0.1).collect(),
            dimension: dim as u32,
            created_at_ms: 0,
            model_params: Default::default(),
            modality: 0,
        }),
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

/// Encode a node batch into Flight `do_exchange` frames with the descriptor on
/// the first frame (the `graph_nodes` ingest route).
async fn encode_exchange_frames(batch: RecordBatch, graph_id: &str) -> Vec<FlightData> {
    let descriptor = FlightDescriptor::new_path(vec!["graph_nodes".into(), graph_id.into()]);
    let input = stream::iter(vec![Ok::<_, FlightError>(batch)]);
    FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor))
        .build(input)
        .map(|r| r.expect("encode frame"))
        .collect()
        .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_nodes_exchange_and_get_over_socket() {
    let server = FlightTestServer::start().await.expect("server start");
    let graph_id = format!(
        "g_socket_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // 1. Create the graph collection via REST (the Flight routes assume it exists).
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let resp = http
        .post(format!("{}/api/v2/graphs", server.rest_base()))
        .json(&serde_json::json!({ "graph_id": graph_id, "name": graph_id }))
        .send()
        .await
        .expect("create graph request");
    assert!(
        resp.status().is_success(),
        "graph create failed: {}",
        resp.status()
    );

    // 2. Connect a real Flight client.
    let channel = tonic::transport::Endpoint::from_shared(server.flight_url())
        .expect("endpoint")
        .connect()
        .await
        .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);

    // 3. Bulk columnar ingest via DoExchange (graph_nodes).
    let originals = vec![
        sample_node("n1", 4),
        sample_node("n2", 4),
        sample_node("n3", 4),
    ];
    let batch = graph_codec::nodes_to_batch(&originals).expect("encode batch");
    let frames = encode_exchange_frames(batch, &graph_id).await;
    let mut ack = flight
        .do_exchange(stream::iter(frames))
        .await
        .expect("do_exchange")
        .into_inner();
    let mut got_final_ack = false;
    while let Some(msg) = ack.next().await {
        let fd = msg.expect("exchange ack frame");
        if !fd.app_metadata.is_empty()
            && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&fd.app_metadata)
            && v.get("complete").and_then(|c| c.as_bool()) == Some(true)
        {
            assert_eq!(v["success"], true, "ingest reported failure: {v}");
            assert_eq!(v["total_rows_written"], 3);
            got_final_ack = true;
        }
    }
    assert!(
        got_final_ack,
        "no completion ack from graph_nodes DoExchange"
    );

    // 4. Columnar export via DoGet (graph_nodes ticket) — decode the streamed
    //    RecordBatches back into nodes.
    let ticket = Ticket {
        ticket: serde_json::to_vec(&serde_json::json!({
            "model": "graph_nodes",
            "graph_id": graph_id,
        }))
        .unwrap()
        .into(),
    };
    let response = flight.do_get(ticket).await.expect("do_get").into_inner();
    let record_stream = FlightRecordBatchStream::new_from_flight_data(
        response.map_err(|s| FlightError::Tonic(Box::new(s))),
    );
    let batches: Vec<RecordBatch> = record_stream.try_collect().await.expect("collect batches");

    let mut nodes: Vec<Node> = Vec::new();
    for b in &batches {
        nodes.extend(graph_codec::batch_to_nodes(b).expect("decode batch"));
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));

    assert_eq!(nodes.len(), 3, "all three ingested nodes export back");
    for (orig, got) in originals.iter().zip(nodes.iter()) {
        assert_eq!(orig.id, got.id);
        assert_eq!(orig.labels, got.labels);
        assert_eq!(orig.properties, got.properties);
        assert_eq!(
            orig.embedding.as_ref().map(|e| &e.vector),
            got.embedding.as_ref().map(|e| &e.vector),
            "embedding vector survives the Flight round trip for {}",
            orig.id
        );
    }
}

/// A DoGet on a graph that was never created is the caller's NOT_FOUND, on
/// both the node and the edge export paths, over a real socket (#2511 review:
/// the node path's FIRST page used to map every error, including a missing
/// graph, to INTERNAL). It must also not provision the graph as a side effect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_doget_on_missing_graph_is_not_found() {
    let server = FlightTestServer::start().await.expect("server start");
    let channel = tonic::transport::Endpoint::from_shared(server.flight_url())
        .expect("endpoint")
        .connect()
        .await
        .expect("flight channel");
    let mut flight = FlightServiceClient::new(channel);

    for model in ["graph_nodes", "graph_edges"] {
        let ticket = Ticket {
            ticket: serde_json::to_vec(&serde_json::json!({
                "model": model,
                "graph_id": "never_created_graph",
            }))
            .unwrap()
            .into(),
        };
        let status = match flight.do_get(ticket).await {
            Err(status) => status,
            // A server may accept the call and fail on the first message.
            Ok(response) => match response.into_inner().next().await {
                Some(Err(status)) => status,
                other => panic!("{model}: DoGet on a missing graph must fail, got {other:?}"),
            },
        };
        assert_eq!(
            status.code(),
            tonic::Code::NotFound,
            "{model}: expected NOT_FOUND, got {:?}: {}",
            status.code(),
            status.message()
        );
    }

    // The failed reads must not have provisioned the graph.
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let listed = http
        .get(format!("{}/api/v2/graphs", server.rest_base()))
        .send()
        .await
        .expect("list graphs")
        .text()
        .await
        .expect("list body");
    assert!(
        !listed.contains("never_created_graph"),
        "a DoGet must not provision the graph: {listed}"
    );
}
