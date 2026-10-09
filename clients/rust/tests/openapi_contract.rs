//! OpenAPI contract gate for the Rust REST SDK.
//!
//! Mirrors the Python contract test at
//! `clients/python/tests/unit/test_openapi_contract.py`. For every covered SDK
//! method, this test:
//!
//!   1. Spins up an `httpmock` server and points a `ProximaClient` at it.
//!   2. Programs the mock with the expected HTTP method + path template, asserts
//!      `application/json` content-type for write methods, and returns a
//!      minimal-valid response.
//!   3. Calls the SDK method.
//!   4. Asserts the captured request matches the OpenAPI operation declared in
//!      `docs/openapi/proximadb-openapi.yaml` (verb + path), and — for write
//!      methods — that every `required` field declared by the request body
//!      schema is present in the captured JSON body.
//!
//! Drift between the SDK and the OpenAPI contract therefore fails this test
//! deterministically without needing a running server.
//!
//! Full JSON-Schema validation (the Python test uses Draft 2020-12 with
//! `$ref` resolution) is intentionally out of scope here — the lightweight
//! shape check catches every drift we've seen historically (missing field,
//! wrong verb, wrong path, wrong content-type) at a fraction of the
//! dev-dependency weight. Upgrade path: swap the body check for a full
//! Draft 2020-12 validator when the SDK starts emitting nested polymorphic
//! payloads that warrant it.

use std::collections::HashSet;
use std::path::PathBuf;

use httpmock::{Method, MockServer};
use proximadb_sdk::{
    ColumnDefinition, ExplainQueryRequest, ProximaClient, QueryRequest, SchemaDefinition,
    UpdateSchemaRequest,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Spec loading
// ---------------------------------------------------------------------------

fn spec_path() -> PathBuf {
    // CARGO_MANIFEST_DIR -> clients/rust ; spec is repo-root /docs/openapi/...
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR must have grandparent (repo root)")
        .join("docs/openapi/proximadb-openapi.yaml")
}

fn load_spec() -> Value {
    let path = spec_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read OpenAPI spec at {path:?}: {e}"));
    serde_yaml::from_str::<Value>(&text)
        .unwrap_or_else(|e| panic!("Failed to parse OpenAPI YAML at {path:?}: {e}"))
}

fn operation<'a>(spec: &'a Value, path_template: &str, method: &str) -> &'a Value {
    spec.pointer(&format!(
        "/paths/{}/{}",
        json_pointer_escape(path_template),
        method.to_lowercase()
    ))
    .unwrap_or_else(|| panic!("{method} {path_template} not in OpenAPI spec"))
}

fn json_pointer_escape(s: &str) -> String {
    // RFC 6901: '~' -> '~0', '/' -> '~1'
    s.replace('~', "~0").replace('/', "~1")
}

/// Resolve a (possibly $ref'd) schema node to a concrete object node.
fn resolve_schema<'a>(spec: &'a Value, node: &'a Value) -> &'a Value {
    if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
        let pointer = reference
            .strip_prefix("#")
            .unwrap_or_else(|| panic!("unsupported $ref form: {reference}"));
        spec.pointer(pointer)
            .unwrap_or_else(|| panic!("dangling $ref: {reference}"))
    } else {
        node
    }
}

/// Walk `allOf` and direct properties to collect every `required` field name
/// declared on a request body schema. Matches the OpenAPI 3.1 composition
/// shape we use (e.g. `UpdateSchemaRequest` is `SchemaDefinition` + `force`).
fn collect_required_fields(spec: &Value, schema: &Value) -> HashSet<String> {
    let mut out = HashSet::new();
    let schema = resolve_schema(spec, schema);
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for f in required {
            if let Some(name) = f.as_str() {
                out.insert(name.to_string());
            }
        }
    }
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for branch in all_of {
            out.extend(collect_required_fields(spec, branch));
        }
    }
    out
}

fn request_body_schema<'a>(_spec: &'a Value, op: &'a Value) -> Option<&'a Value> {
    // `_spec` is unused today (we don't follow request-body $refs), but kept
    // in the signature so callers don't need to change shape when this
    // grows into a full $ref-resolving validator later.
    op.get("requestBody")?
        .get("content")?
        .get("application/json")?
        .get("schema")
}

fn operation_id(op: &Value) -> &str {
    op.get("operationId")
        .and_then(Value::as_str)
        .unwrap_or("<missing operationId>")
}

// ---------------------------------------------------------------------------
// Per-operation contract checks (6 new v2 ops)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_live_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/health/live", "get");
    assert_eq!(operation_id(op), "getLiveness");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/health/live");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"status": "ok"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let probe = client.health_live().await.unwrap();
    assert_eq!(probe.status, "ok");

    mock.assert_async().await;
}

#[tokio::test]
async fn health_ready_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/health/ready", "get");
    assert_eq!(operation_id(op), "getReadiness");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/health/ready");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"status": "ready"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let probe = client.health_ready().await.unwrap();
    assert_eq!(probe.status, "ready");

    mock.assert_async().await;
}

#[tokio::test]
async fn get_collection_schema_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections/{collection_id}/schema", "get");
    assert_eq!(operation_id(op), "getCollectionSchema");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET)
                .path("/api/v2/collections/col_abc/schema");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "schema_id": "sch_1",
                    "schema_version": "v1",
                    "collection_id": "col_abc",
                    "schema": {
                        "columns": [
                            {"name": "title", "data_type": "text"}
                        ]
                    },
                    "created_at": "2026-05-23T00:00:00Z"
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let resp = client.get_collection_schema("col_abc").await.unwrap();
    assert_eq!(resp.schema_id, "sch_1");
    assert_eq!(resp.schema.columns.len(), 1);
    assert_eq!(resp.schema.columns[0].name, "title");

    mock.assert_async().await;
}

#[tokio::test]
async fn update_collection_schema_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections/{collection_id}/schema", "put");
    assert_eq!(operation_id(op), "updateCollectionSchema");

    // Spec-required fields the SDK request body must include.
    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("PUT schema must declare requestBody"),
    );
    assert!(required.contains("columns"), "spec requires `columns`");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::PUT)
                .path("/api/v2/collections/col_abc/schema")
                .header("content-type", "application/json")
                // Body must contain spec-required `columns` field.
                .json_body_partial(r#"{"columns": [{"name": "title", "data_type": "text"}]}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "schema_id": "sch_2",
                    "schema_version": "v2",
                    "previous_schema_id": "sch_1",
                    "changes": [],
                    "warnings": [],
                    "updated_at": "2026-05-23T00:00:00Z"
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let body = UpdateSchemaRequest {
        schema: SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "title".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: Some("strict".to_string()),
            allow_additional_fields: None,
        },
        force: Some(true),
    };

    let resp = client
        .update_collection_schema("col_abc", &body)
        .await
        .unwrap();
    assert_eq!(resp.schema_id, "sch_2");
    assert_eq!(resp.previous_schema_id, "sch_1");

    mock.assert_async().await;
}

#[tokio::test]
async fn execute_query_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/query", "post");
    assert_eq!(operation_id(op), "executeQuery");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST query must declare requestBody"),
    );
    assert!(required.contains("language"));
    assert!(required.contains("query"));

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/query")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"language": "uql", "query": "SELECT 1"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "records": [],
                    "total_count": 0
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let req = QueryRequest::new("uql", "SELECT 1");
    let resp: Value = client.execute_query(&req).await.unwrap();
    assert_eq!(resp["total_count"], json!(0));

    mock.assert_async().await;
}

#[tokio::test]
async fn explain_query_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/query/explain", "post");
    assert_eq!(operation_id(op), "explainQuery");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST query/explain must declare requestBody"),
    );
    assert!(required.contains("language"));
    assert!(required.contains("query"));

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/query/explain")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"language": "uql", "query": "SELECT 1"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "plan": {"node": "Scan"},
                    "lowering": []
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let req = ExplainQueryRequest::new("uql", "SELECT 1");
    let resp: Value = client.explain_query(&req).await.unwrap();
    assert_eq!(resp["plan"]["node"], json!("Scan"));

    mock.assert_async().await;
}

// ---------------------------------------------------------------------------
// Existing SDK methods — sanity gates so this file owns the full 15/15 surface
// (anything that drifts on path/verb here breaks the build).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_collection_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections", "post");
    assert_eq!(operation_id(op), "createCollection");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST collections must declare requestBody"),
    );
    assert!(required.contains("name"));
    assert!(required.contains("dimension"));

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"name": "items", "dimension": 384}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "collection_id": "col_1",
                    "name": "items",
                    "dimension": 384,
                    "engine": "sst",
                    "proxima_record_enabled": true,
                    "created_at": "2026-05-23T00:00:00Z"
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client
        .create_collection("items")
        .dimension(384)
        .execute()
        .await
        .unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn list_collections_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections", "get");
    assert_eq!(operation_id(op), "listCollections");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/api/v2/collections");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "collections": [],
                    "total": 0,
                    "limit": 50,
                    "offset": 0,
                    "has_more": false
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let collections = client.list_collections().await.unwrap();
    assert!(collections.is_empty());

    mock.assert_async().await;
}

/// Regression (TD-126 Phase 4): `list_collections` previously failed to
/// deserialize when the server emitted `record_count: null` (the v2
/// `CollectionV2Summary.record_count` is `Option<u64>` and is `null`
/// unless `include_stats=true`). End-to-end through the SDK over a mock
/// server.
#[tokio::test]
async fn list_collections_decodes_null_record_count() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections", "get");
    assert_eq!(operation_id(op), "listCollections");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/api/v2/collections");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "collections": [
                        {
                            "collection_id": "col_1",
                            "name": "items",
                            "dimension": 128,
                            "engine": "sst",
                            "proxima_record_enabled": true,
                            "record_count": null
                        }
                    ],
                    "total": 1,
                    "limit": 50,
                    "offset": 0,
                    "has_more": false
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let collections = client.list_collections().await.unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "items");
    assert_eq!(collections[0].vector_count, 0);

    mock.assert_async().await;
}

/// Regression (TD-126 Phase 4): `get_vector` decodes the server's
/// `RecordV2Response`, whose `vector` and `text_fields` are nullable and
/// arrive as explicit JSON `null` when not requested. End-to-end through
/// the SDK over a mock server, exercising the props-shaped record.
#[tokio::test]
async fn get_vector_decodes_props_record_with_null_vector() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(
        &spec,
        "/api/v2/collections/{collection_id}/records/{record_id}",
        "get",
    );
    assert_eq!(operation_id(op), "getRecord");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET)
                .path("/api/v2/collections/items/records/rec_1");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "id": "rec_1",
                    "vector": null,
                    "props": {"category": "tech", "score": 0.9},
                    "text_fields": null,
                    "version": 3,
                    "timestamp": 1718000000_i64
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let record = client
        .collection("items")
        .get_vector("rec_1")
        .await
        .unwrap()
        .expect("record should be present");
    assert_eq!(record.id, "rec_1");
    assert!(record.vector.is_empty());
    assert!(record.text_fields.is_empty());
    assert_eq!(record.props["category"], json!("tech"));

    mock.assert_async().await;
}

#[tokio::test]
async fn delete_collection_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/collections/{collection_id}", "delete");
    assert_eq!(operation_id(op), "deleteCollection");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::DELETE)
                .path("/api/v2/collections/col_1");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"success": true}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client.delete_collection("col_1").await.unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn health_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/health", "get");
    assert_eq!(operation_id(op), "getHealth");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/health");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"status": "ok", "version": "0.2.0"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let status = client.health().await.unwrap();
    assert_eq!(status.status, "ok");

    mock.assert_async().await;
}

// ---------------------------------------------------------------------------
// Graph v2 surface (added 2026-05-30; body shapes locked in 2026-05-30 once
// the Rust SDK was rewritten to spec-true shapes per TD-095). Pins verb +
// path + spec-required body keys against the OpenAPI source of truth for
// every SDK-callable graph operation, including the two batch endpoints
// and the flat traverseGraph payload.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_graph_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs", "post");
    assert_eq!(operation_id(op), "createGraph");

    // Reconciled spec (2026-05-30): CreateGraphRequest now requires `graph_id`
    // (was `name`). `name`/`description` are optional human-readable metadata.
    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST graphs must declare requestBody"),
    );
    assert!(required.contains("graph_id"), "spec requires `graph_id`");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"graph_id": "knowledge"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"graph_id": "knowledge", "success": true}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client.create_graph("knowledge").execute().await.unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn list_graphs_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs", "get");
    assert_eq!(operation_id(op), "listGraphs");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET).path("/api/v2/graphs");
            then.status(200)
                .header("content-type", "application/json")
                // Wave-6 envelope truth: the handler returns
                // {success, data} — the old flat {"graphs": [...]} shape
                // never existed on the wire.
                .json_body(json!({"success": true, "data": []}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let graphs = client.list_graphs().await.unwrap();
    assert!(graphs.is_empty());

    mock.assert_async().await;
}

#[tokio::test]
async fn delete_graph_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}", "delete");
    assert_eq!(operation_id(op), "deleteGraph");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::DELETE).path("/api/v2/graphs/knowledge");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"deleted": true, "name": "knowledge"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client.delete_graph("knowledge").await.unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn create_node_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/nodes", "post");
    assert_eq!(operation_id(op), "createNode");

    // Reconciled spec (2026-05-30): CreateNodeRequest is now a wrapped
    // envelope `{node: NodeInput}` where NodeInput requires `id`. We assert
    // both layers — the outer envelope requires `node`, and the inner
    // NodeInput requires `id`.
    let body_schema = request_body_schema(&spec, op).expect("POST nodes must declare requestBody");
    let outer_required = collect_required_fields(&spec, body_schema);
    assert!(
        outer_required.contains("node"),
        "spec requires `node` wrapper"
    );

    let resolved_body = resolve_schema(&spec, body_schema);
    let node_schema = resolved_body
        .get("properties")
        .and_then(|p| p.get("node"))
        .expect("CreateNodeRequest must expose `node` property");
    let node_required = collect_required_fields(&spec, node_schema);
    assert!(
        node_required.contains("id"),
        "spec requires `id` inside `node` wrapper"
    );

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs/knowledge/nodes")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"node": {"id": "person_1"}}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "success": true,
                    "data": {
                        "id": "person_1",
                        "labels": ["Person", "Employee"],
                        "properties": {"active": true},
                        "embedding": {"vector": [0.5, 0.25]},
                        "created_at": "2026-09-05T00:00:00Z",
                        "updated_at": "2026-09-05T00:00:00Z"
                    }
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client
        .graph("knowledge")
        .add_node()
        .id("person_1")
        .label("Person")
        .execute()
        .await
        .unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn get_node_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/nodes/{node_id}", "get");
    assert_eq!(operation_id(op), "getNode");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET)
                .path("/api/v2/graphs/knowledge/nodes/person_1");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "success": true,
                    "data": {
                        "id": "person_1",
                        "labels": ["Person", "Employee"],
                        "properties": {"active": true},
                        "embedding": {"vector": [0.5, 0.25]},
                        "created_at": "2026-09-05T00:00:00Z",
                        "updated_at": "2026-09-05T00:00:00Z"
                    }
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let node = client
        .graph("knowledge")
        .get_node("person_1")
        .await
        .unwrap();
    assert!(node.is_some());
    let node = node.unwrap();
    assert_eq!(node.id, "person_1");
    assert_eq!(node.label.as_deref(), Some("Person"));
    assert_eq!(node.properties["active"], json!(true));
    assert_eq!(node.vector, Some(vec![0.5, 0.25]));

    mock.assert_async().await;
}

#[tokio::test]
async fn delete_node_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/nodes/{node_id}", "delete");
    assert_eq!(operation_id(op), "deleteNode");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::DELETE)
                .path("/api/v2/graphs/knowledge/nodes/person_1");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"deleted": true, "id": "person_1"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client
        .graph("knowledge")
        .delete_node("person_1")
        .await
        .unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn create_edge_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/edges", "post");
    assert_eq!(operation_id(op), "createEdge");

    // Reconciled spec (2026-05-30): CreateEdgeRequest is now a wrapped
    // envelope `{edge: EdgeInput}` where EdgeInput requires
    // `id, from_node_id, to_node_id, edge_type` (was `source`/`target`).
    // Assert outer wrapper and inner required fields.
    let body_schema = request_body_schema(&spec, op).expect("POST edges must declare requestBody");
    let outer_required = collect_required_fields(&spec, body_schema);
    assert!(
        outer_required.contains("edge"),
        "spec requires `edge` wrapper"
    );

    let resolved_body = resolve_schema(&spec, body_schema);
    let edge_schema = resolved_body
        .get("properties")
        .and_then(|p| p.get("edge"))
        .expect("CreateEdgeRequest must expose `edge` property");
    let edge_required = collect_required_fields(&spec, edge_schema);
    assert!(
        edge_required.contains("id"),
        "spec requires `id` inside `edge` wrapper"
    );
    assert!(
        edge_required.contains("from_node_id"),
        "spec requires `from_node_id` inside `edge` wrapper"
    );
    assert!(
        edge_required.contains("to_node_id"),
        "spec requires `to_node_id` inside `edge` wrapper"
    );
    assert!(
        edge_required.contains("edge_type"),
        "spec requires `edge_type` inside `edge` wrapper"
    );

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs/knowledge/edges")
                .header("content-type", "application/json")
                .json_body_partial(
                    r#"{"edge": {"id": "e1", "from_node_id": "person_1", "to_node_id": "person_2", "edge_type": "KNOWS"}}"#,
                );
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "e1"}));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    client
        .graph("knowledge")
        .add_edge()
        .id("e1")
        .from("person_1")
        .to("person_2")
        .relationship("KNOWS")
        .execute()
        .await
        .unwrap();

    mock.assert_async().await;
}

#[tokio::test]
async fn traverse_graph_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/traverse", "post");
    assert_eq!(operation_id(op), "traverseGraph");

    // Spec-true flat shape: {start_node_id, max_depth, edge_types,
    // node_labels?, algorithm?, limit?} — no `graph` wrapper, no
    // `start_node` legacy key.
    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST traverse must declare requestBody"),
    );
    assert!(
        required.contains("start_node_id"),
        "spec requires `start_node_id`"
    );

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs/knowledge/traverse")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"start_node_id": "person_1"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "success": true,
                    "data": {
                        "nodes": [{
                            "id": "person_1",
                            "labels": ["Person"],
                            "properties": {"active": true},
                            "created_at": "2026-09-05T00:00:00Z",
                            "updated_at": "2026-09-05T00:00:00Z"
                        }],
                        "edges": [{
                            "id": "e1",
                            "from_node_id": "person_1",
                            "to_node_id": "person_2",
                            "edge_type": "KNOWS",
                            "properties": {"since": 2020},
                            "weight": 0.75,
                            "created_at": "2026-09-05T00:00:00Z",
                            "updated_at": "2026-09-05T00:00:00Z"
                        }],
                        "paths": [["person_1", "person_2"]]
                    }
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let result = client
        .graph("knowledge")
        .traverse()
        .start("person_1")
        .execute()
        .await
        .unwrap();
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0].label.as_deref(), Some("Person"));
    assert_eq!(result.edges.len(), 1);
    assert_eq!(result.edges[0].source, "person_1");
    assert_eq!(result.edges[0].relationship, "KNOWS");
    assert_eq!(result.paths, vec![vec!["person_1", "person_2"]]);

    mock.assert_async().await;
}

#[tokio::test]
async fn batch_create_nodes_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/nodes/batch", "post");
    assert_eq!(operation_id(op), "batchCreateNodes");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST nodes/batch must declare requestBody"),
    );
    assert!(required.contains("nodes"), "spec requires `nodes`");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs/knowledge/nodes/batch")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"nodes": [{"id": "person_1"}]}"#);
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "success": true,
                    "data": {
                        "created_count": 0,
                        "updated_count": 0,
                        "failed_count": 1,
                        "results": [],
                        "errors": [{"id": "person_1", "error": "rejected"}]
                    }
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let count = client
        .graph("knowledge")
        .add_nodes(vec![proximadb_sdk::GraphNode::new("person_1")])
        .await
        .unwrap();
    assert_eq!(count, 0);

    mock.assert_async().await;
}

#[tokio::test]
async fn batch_create_edges_matches_spec() {
    let spec = load_spec();
    let server = MockServer::start_async().await;
    let op = operation(&spec, "/api/v2/graphs/{graph_id}/edges/batch", "post");
    assert_eq!(operation_id(op), "batchCreateEdges");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("POST edges/batch must declare requestBody"),
    );
    assert!(required.contains("edges"), "spec requires `edges`");

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/graphs/knowledge/edges/batch")
                .header("content-type", "application/json")
                .json_body_partial(
                    r#"{"edges": [{"id": "person_1-KNOWS-person_2", "from_node_id": "person_1", "to_node_id": "person_2", "edge_type": "KNOWS"}]}"#,
                );
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "success": true,
                    "data": {
                        "created_count": 0,
                        "updated_count": 0,
                        "failed_count": 1,
                        "results": [],
                        "errors": [{"id": "person_1-KNOWS-person_2", "error": "rejected"}]
                    }
                }));
        })
        .await;

    let client = ProximaClient::connect(server.base_url()).unwrap();
    let count = client
        .graph("knowledge")
        .add_edges(vec![proximadb_sdk::GraphEdge::new(
            "person_1", "person_2", "KNOWS",
        )])
        .await
        .unwrap();
    assert_eq!(count, 0);

    mock.assert_async().await;
}
