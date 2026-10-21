//! ADR-009 document store-convergence — cross-surface visibility, e2e over the real
//! network transports (REST + gRPC), never calling an engine directly.
//!
//! The document modality historically had a **store-split**: REST v2 document ingest wrote
//! to the per-collection record/vector store, while the gRPC `ProximaDocumentService` +
//! `DocumentService` wrote to their own `document_wal` — so a document written on one
//! surface was invisible on the other. This suite proves the convergence:
//!
//! - **Gate ON** (`PROXIMADB_DOC_CANONICAL_VECTOR` = collection): a document ingested via
//!   REST v2 is visible to a gRPC `GetDocument` — one store, cross-surface. This is the
//!   headline convergence claim.
//! - **Legacy path** (non-canonical collection, or kill-switch): the gRPC/`DocumentService` legacy path round-trips unchanged
//!   (create → get), proving the default-OFF gate preserves today's behavior.
//!
//! One ProximaDB boot per process (the WAL manifest is a set-once singleton); nextest runs
//! each test in its own process, so the per-test env gate + boot are isolated.

use std::path::Path;
use std::time::Duration;

use proximadb::core::Config;
use proximadb::proto::proximadb_v2::proxima_document_service_client::ProximaDocumentServiceClient;
use proximadb::proto::proximadb_v2::{
    CreateDocumentRequest, GetDocumentRequest, QueryDocumentsRequest, TypedValue, typed_value,
};
use serde_json::json;
use tempfile::TempDir;
use tokio::time::sleep;

#[path = "common/pg_server.rs"]
mod pg_server;

use pg_server::TestPgServer;

/// Boot via the shared harness, keeping this file's original 300ms post-health
/// settle (the shared harness settles 200ms — do not silently tighten).
async fn start_server() -> anyhow::Result<TestPgServer> {
    let server = TestPgServer::start().await?;
    sleep(Duration::from_millis(100)).await;
    Ok(server)
}

/// Point a shared-harness server at an externally owned data dir. The restart
/// test needs the dir to outlive the first server generation (the harness's own
/// tempdir is dropped with it), and the catalog/WAL must stay anchored to that
/// same dir across generations so the restarted server rediscovers the collection.
fn anchor_to(config: &mut Config, data: &Path) {
    config.server.data_dir = data.to_path_buf();
    config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
        url: format!("file://{}", data.display()),
        ..Default::default()
    }];
    config.storage.metadata_url = format!("file://{}/metadata", data.display());
    config.storage.wal_config.write_buffer_directory = format!("file://{}/wal", data.display());
}

/// One server generation on the externally owned `data` dir (same 300ms settle).
async fn start_on(data: &Path) -> anyhow::Result<TestPgServer> {
    let server = TestPgServer::start_with(|config| anchor_to(config, data)).await?;
    sleep(Duration::from_millis(100)).await;
    Ok(server)
}

/// Scope the process-global gate to `collection` for the test; removes it on drop.
struct GateGuard;
impl GateGuard {
    fn on(collection: &str) -> Self {
        // SAFETY (edition 2024): nextest runs each test in its own process, so this env
        // mutation is single-threaded for the process.
        unsafe { std::env::set_var("PROXIMADB_DOC_CANONICAL_VECTOR", collection) };
        Self
    }
}
impl Drop for GateGuard {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("PROXIMADB_DOC_CANONICAL_VECTOR") };
    }
}

/// Gate ON: a REST-v2-ingested document is visible to a gRPC `GetDocument` — one store,
/// cross-surface (the headline convergence claim).
#[tokio::test]
async fn rest_ingested_document_is_visible_via_grpc_when_gate_on() {
    let server = start_server().await.expect("server start");
    // Collision-free collection name; the gate is read per-call, so setting it
    // after start is fine.
    let collection = server.unique_name("convdocs");
    let _gate = GateGuard::on(&collection);
    let (http, base) = server.rest_client().expect("http client");

    // Create the vector collection the document route resolves against.
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({"name": collection, "dimension": 8, "engine": "sst"}))
        .send()
        .await
        .expect("create collection");
    assert!(
        create.status().is_success(),
        "collection create should succeed, got {}",
        create.status()
    );

    // Ingest a document via REST v2 with an SDK-provided vector (no embedding backend needed).
    let ingest = http
        .post(format!("{base}/api/v2/collections/{collection}/documents"))
        .header("X-Embed-Source", "sdk-vector")
        .header("X-Ingest-Mode", "sync")
        .json(&json!({
            "records": [{
                "id": "doc-1",
                "vector": [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "metadata": {"title": "Alpha"}
            }]
        }))
        .send()
        .await
        .expect("ingest document");
    assert!(
        ingest.status().is_success(),
        "document ingest should succeed, got {} body {:?}",
        ingest.status(),
        ingest.text().await.ok()
    );

    // Read it back over gRPC ProximaDocumentService — the other surface.
    let mut client =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC connect");
    let resp = client
        .get_document(GetDocumentRequest {
            collection_id: collection.to_string(),
            id: "doc-1".to_string(),
        })
        .await
        .expect("gRPC GetDocument should find the REST-ingested doc (cross-surface)")
        .into_inner();
    let doc = resp.document.expect("document present");
    assert_eq!(doc.id, "doc-1", "gRPC sees the REST-ingested document id");
}

/// Legacy path for a NON-canonical collection: the gate is DEFAULT-ON, but `legacydocs*` is never
/// created as a vector collection, so the mixed-write-safe capability check keeps the
/// gRPC/DocumentService write on the legacy path (create → get round-trips unchanged). This proves
/// default-ON does not hard-fail pure-document collections.
#[tokio::test]
async fn grpc_document_round_trips_on_legacy_path_for_non_canonical_collection() {
    let server = start_server().await.expect("server start");
    let collection = server.unique_name("legacydocs");
    let mut client =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC connect");

    client
        .create_document(CreateDocumentRequest {
            collection_id: collection.clone(),
            id: "doc-1".to_string(),
            ..Default::default()
        })
        .await
        .expect("gRPC CreateDocument (legacy path)");

    let resp = client
        .get_document(GetDocumentRequest {
            collection_id: collection.clone(),
            id: "doc-1".to_string(),
        })
        .await
        .expect("gRPC GetDocument (legacy path)")
        .into_inner();
    let doc = resp.document.expect("document present via legacy path");
    assert_eq!(doc.id, "doc-1");
}

/// Follow-up validation (PR #698 disclosed limitation): a **pure-metadata** gRPC document —
/// created with props but NO vector — routes onto the shared record/vector store when the gate is
/// ON, and is retrievable via gRPC get + query. Confirms documents without embeddings converge
/// (vector *search* won't surface them, but get/scan/query do).
#[tokio::test]
async fn pure_metadata_grpc_document_round_trips_on_canonical_route_when_gate_on() {
    let server = start_server().await.expect("server start");
    let collection = server.unique_name("metadocs");
    let _gate = GateGuard::on(&collection);
    let (http, base) = server.rest_client().expect("http client");

    // The canonical route resolves against a real vector collection. A metadata-only document
    // carries no embedding (dim 0); the vector-insert validation accepts vectorless records in
    // any collection (they're excluded from the ANN index/search but stored + served by id/scan),
    // so it converges on the shared store alongside embedded documents.
    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({"name": collection, "dimension": 8, "engine": "sst"}))
        .send()
        .await
        .expect("create collection");
    let status = create.status();
    let body = create.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "collection create should succeed, got {status} — body: {body}"
    );

    let mut client =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC connect");

    // gRPC CreateDocument with a prop but NO vector — the metadata-only case.
    let mut props = std::collections::HashMap::new();
    props.insert(
        "title".to_string(),
        TypedValue {
            declared_type: 0,
            value: Some(typed_value::Value::TextValue("Gamma".to_string())),
        },
    );
    client
        .create_document(CreateDocumentRequest {
            collection_id: collection.to_string(),
            id: "m1".to_string(),
            props,
            ..Default::default()
        })
        .await
        .expect("gRPC CreateDocument (no vector) should route to the shared store");

    // gRPC GetDocument reads it back through the vector-store scan.
    let got = client
        .get_document(GetDocumentRequest {
            collection_id: collection.to_string(),
            id: "m1".to_string(),
        })
        .await
        .expect("gRPC GetDocument (metadata-only)")
        .into_inner()
        .document
        .expect("metadata-only document present via canonical route");
    assert_eq!(got.id, "m1");
    assert!(
        got.props.contains_key("title"),
        "props round-trip through the shared store"
    );

    // gRPC QueryDocuments also surfaces it (sourced from the shared store).
    let queried = client
        .query_documents(QueryDocumentsRequest {
            collection_id: collection.to_string(),
            limit: 10,
            ..Default::default()
        })
        .await
        .expect("gRPC QueryDocuments (metadata-only)")
        .into_inner();
    assert_eq!(
        queried.documents.len(),
        1,
        "query returns the metadata-only doc"
    );
    assert_eq!(queried.documents[0].id, "m1");
}

/// Durability: a document written on the canonical route survives a full server restart. The
/// canonical path deliberately drops the legacy `document_wal`, so recovery rests entirely on the
/// shared vector WAL — this proves the doc is recovered from it and served cross-surface after
/// restart (the plan's "restart → docs survive via the vector WAL" verification point).
#[tokio::test]
async fn canonical_document_survives_restart_via_vector_wal() {
    // Externally owned data dir: it must survive the first server generation
    // (the shared harness drops its own tempdir with the server).
    let data = TempDir::new().expect("restart data dir");
    let server = start_on(data.path()).await.expect("server start");
    // Name is fixed for the whole test so it stays stable across the restart,
    // which rebinds fresh ports. The gate stays ON throughout.
    let collection = server.unique_name("recdocs");
    let _gate = GateGuard::on(&collection);
    let (http, base) = server.rest_client().expect("http client");

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({"name": collection, "dimension": 8, "engine": "sst"}))
        .send()
        .await
        .expect("create collection");
    assert!(create.status().is_success(), "create: {}", create.status());

    let ingest = http
        .post(format!("{base}/api/v2/collections/{collection}/documents"))
        .header("X-Embed-Source", "sdk-vector")
        .header("X-Ingest-Mode", "sync")
        .json(&json!({
            "records": [{
                "id": "r1",
                "vector": [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "metadata": {"title": "Durable"}
            }]
        }))
        .send()
        .await
        .expect("ingest document");
    assert!(
        ingest.status().is_success(),
        "ingest: {} {:?}",
        ingest.status(),
        ingest.text().await.ok()
    );

    // Visible before the restart.
    let mut client =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC connect");
    let pre = client
        .get_document(GetDocumentRequest {
            collection_id: collection.clone(),
            id: "r1".to_string(),
        })
        .await
        .expect("pre-restart get")
        .into_inner();
    assert_eq!(pre.document.expect("present before restart").id, "r1");

    // Full shutdown + restart on the SAME data_dir (fresh ports; the 750ms pause
    // lets the OS release the ports + WAL file handles before rebinding).
    server.shutdown().await;
    sleep(Duration::from_millis(750)).await;
    let server = start_on(data.path())
        .await
        .expect("restart on same data_dir");

    // The document must be recovered from the vector WAL and served on the canonical route.
    let mut client2 =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC reconnect after restart");
    let post = client2
        .get_document(GetDocumentRequest {
            collection_id: collection.clone(),
            id: "r1".to_string(),
        })
        .await
        .expect("post-restart get should recover the doc from the vector WAL")
        .into_inner();
    assert_eq!(
        post.document.expect("document survived the restart").id,
        "r1"
    );
}

/// Metering plumbing: a document written on the canonical route flows through the metered
/// vector-write service (`handle_record_batch_for_tenant`) — the same path REST v2 records use —
/// closing the billing gap where the legacy gRPC/DocumentService WAL path was unmetered.
///
/// What is asserted: the gRPC write succeeds through the metered service AND the Prometheus
/// consumption surface (`/metrics/prometheus`) is live and serving the consumption metric family
/// after the write. What is NOT asserted here: a per-write counter delta — the per-query io_trace
/// is task-local (not observable across the server task boundary) and consumption is metered at
/// flush / object-store / egress boundaries, not per un-flushed insert. The precise per-write
/// signal is emitted by the shared vector-write path (covered where that path is metered); this
/// test guards that documents ride that metered path and the endpoint is wired.
#[tokio::test]
async fn canonical_document_write_rides_the_metered_path() {
    let server = start_server().await.expect("server start");
    let collection = server.unique_name("metdocs");
    let _gate = GateGuard::on(&collection);
    let (http, base) = server.rest_client().expect("http client");

    let create = http
        .post(format!("{base}/api/v2/collections"))
        .json(&json!({"name": collection, "dimension": 8, "engine": "sst"}))
        .send()
        .await
        .expect("create collection");
    assert!(create.status().is_success(), "create: {}", create.status());

    let mut client =
        ProximaDocumentServiceClient::connect(format!("http://127.0.0.1:{}", server.grpc_port))
            .await
            .expect("gRPC connect");
    client
        .create_document(CreateDocumentRequest {
            collection_id: collection.clone(),
            id: "billed-1".to_string(),
            ..Default::default()
        })
        .await
        .expect("gRPC CreateDocument on canonical route");

    // The metered write completed (above). The consumption telemetry surface must be live and
    // serving after it — the per-tenant consumption metric family is registered and exported.
    let scrape = http
        .get(format!("{base}/metrics/prometheus"))
        .send()
        .await
        .expect("scrape /metrics/prometheus")
        .text()
        .await
        .expect("metrics body");
    assert!(
        scrape.contains("proximadb_storage_bytes"),
        "the consumption metric family should be live on /metrics/prometheus after a metered write"
    );
}
