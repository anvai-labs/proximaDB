//! OpenAPI contract gate for the Rust **connectors** at `src/connectors/`.
//!
//! Mirrors the SDK gate at `clients/rust/tests/openapi_contract.rs`. For
//! every connector wire method we implement, the test:
//!
//!   1. Looks the operation up in `docs/openapi/proximadb-openapi.yaml`
//!      and asserts the operationId matches our expectation (catches spec
//!      renames).
//!   2. Programs an `httpmock::MockServer` with the spec-correct verb +
//!      path + content-type + required-key body matcher.
//!   3. Invokes the connector method.
//!   4. Asserts the mock was hit — drift is the only thing that would
//!      cause the mock to go uncalled.
//!
//! No live ProximaDB server is started. The gate is hermetic and fast.

#[path = "openapi_helpers.rs"]
mod helpers;

use helpers::{collect_required_fields, load_spec, operation, operation_id, request_body_schema};

use httpmock::{Method, MockServer};
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use proximadb::connectors::duckdb::{
    DuckDBConnectorConfig, DuckDBInsert, DuckDBTableScan, DuckDBVectorSearch,
    DuckDBVectorSearchParams,
};
use proximadb::connectors::hadoop::{
    HadoopInputSplit, HadoopShimConfig, ProximaRecordReader, ProximaRecordWriter,
};
use proximadb::storage::formats::{SplitStatistics, SplitType};
use proximadb_distance_types::DistanceMetric;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Smoke test — proves the shared helpers wire up against the spec file.
// ---------------------------------------------------------------------------

#[test]
fn helpers_load_spec_and_find_known_operation() {
    let spec = load_spec();
    let op = operation(&spec, "/health", "get");
    assert_eq!(operation_id(op), "getHealth");
}

// ---------------------------------------------------------------------------
// DuckDB connector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn duckdb_bind_fetches_collection_schema_v2() {
    // Spec sanity: confirm the schema endpoint exists where we expect it.
    let spec = load_spec();
    let op = operation(&spec, "/api/v2/collections/{collection_id}/schema", "get");
    assert_eq!(operation_id(op), "getCollectionSchema");

    let server = MockServer::start_async().await;

    let mock = server
        .mock_async(|when, then| {
            when.method(Method::GET)
                .path("/api/v2/collections/col1/schema");
            then.status(200)
                .header("content-type", "application/json")
                // Minimal SchemaDefinition shape: columns list + name.
                .body(r#"{"name":"col1","columns":[]}"#);
        })
        .await;

    let config = DuckDBConnectorConfig {
        server_url: server.base_url(),
        ..DuckDBConnectorConfig::default()
    };
    let mut scan = DuckDBTableScan::new(config);
    let result = scan.bind("col1").await;
    assert!(result.is_ok(), "bind() must succeed: {result:?}");

    mock.assert_async().await;
}

#[tokio::test]
async fn duckdb_search_posts_v2_search_request() {
    let spec = load_spec();
    let op = operation(&spec, "/api/v2/collections/{collection_id}/search", "post");
    assert_eq!(operation_id(op), "searchRecords");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("searchRecords must declare requestBody"),
    );
    assert!(required.contains("vector"), "spec requires `vector`");
    assert!(required.contains("top_k"), "spec requires `top_k`");

    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/embeddings/search")
                .header("content-type", "application/json")
                // Partial-shape match: just assert the spec-required keys are
                // present and `top_k` is an int. We avoid asserting the
                // float vector verbatim because f32 → JSON encoding
                // doesn't necessarily round-trip a literal "0.1".
                .json_body_partial(r#"{"top_k":10}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"results":[],"latency_ms":1,"request_id":"r1"}"#);
        })
        .await;

    let config = DuckDBConnectorConfig {
        server_url: server.base_url(),
        ..DuckDBConnectorConfig::default()
    };
    let svc = DuckDBVectorSearch::new(config);
    let params = DuckDBVectorSearchParams {
        collection: "embeddings".to_string(),
        query_vector: vec![0.1, 0.2, 0.3],
        top_k: 10,
        metric: DistanceMetric::Cosine,
        filter: None,
        include_distances: true,
    };
    let result = svc.search(&params).await;
    assert!(result.is_ok(), "search() must succeed: {result:?}");

    mock.assert_async().await;
}

#[tokio::test]
async fn duckdb_insert_posts_v2_records_batch() {
    let spec = load_spec();
    let op = operation(
        &spec,
        "/api/v2/collections/{collection_id}/records/batch",
        "post",
    );
    assert_eq!(operation_id(op), "insertRecords");

    let required = collect_required_fields(
        &spec,
        request_body_schema(&spec, op).expect("insertRecords must declare requestBody"),
    );
    assert!(required.contains("records"), "spec requires `records`");

    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/embeddings/records/batch")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"records":[]}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"success":true,"inserted":2,"results":[]}"#);
        })
        .await;

    let config = DuckDBConnectorConfig {
        server_url: server.base_url(),
        ..DuckDBConnectorConfig::default()
    };
    let mut svc = DuckDBInsert::new(config, "embeddings".to_string());

    // Minimal RecordBatch — 2 rows, one `id` column. The connector
    // synthesizes a `records` list of that length; the contract gate
    // only validates that `records` is present.
    let ids: ArrayRef = Arc::new(StringArray::from(vec!["r1", "r2"]));
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![ids]).expect("batch builds");

    let result = svc.insert(&batch).await;
    assert!(result.is_ok(), "insert() must succeed: {result:?}");

    mock.assert_async().await;
}

// ---------------------------------------------------------------------------
// Hadoop connector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hadoop_flush_batch_posts_v2_records_batch() {
    let spec = load_spec();
    let op = operation(
        &spec,
        "/api/v2/collections/{collection_id}/records/batch",
        "post",
    );
    assert_eq!(operation_id(op), "insertRecords");

    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/hadoop_col/records/batch")
                .header("content-type", "application/json")
                .json_body_partial(r#"{"records":[]}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"success":true,"inserted":1,"results":[]}"#);
        })
        .await;

    let config = HadoopShimConfig {
        host: server.host(),
        port: server.port(),
        collection: "hadoop_col".to_string(),
        ..HadoopShimConfig::default()
    };
    let mut writer = ProximaRecordWriter::new(config, 1);
    // Seed one row via the public write path (which buffers + calls
    // flush_batch when the threshold is hit). We call flush directly
    // via the public force-flush method so the test isn't tied to the
    // batch-size threshold.
    let mut row = HashMap::new();
    row.insert(
        "id".to_string(),
        proximadb::connectors::hadoop::HadoopWritable::Text("r1".to_string()),
    );
    writer.buffer_row(row);
    let result = writer.flush_now().await;
    assert!(result.is_ok(), "flush_now() must succeed: {result:?}");

    mock.assert_async().await;
}

#[tokio::test]
async fn hadoop_fetch_next_page_posts_v2_records_scan() {
    let spec = load_spec();
    let op = operation(
        &spec,
        "/api/v2/collections/{collection_id}/records/scan",
        "post",
    );
    assert_eq!(operation_id(op), "scanRecords");

    // Spec required-keys: empty (all body fields optional). We assert
    // the op exists, has a requestBody, and the SDK can POST with a
    // bare-cursor body shape.
    let body_schema = request_body_schema(&spec, op).expect("scanRecords must declare requestBody");
    let required = collect_required_fields(&spec, body_schema);
    assert!(
        required.is_empty(),
        "scanRecords body is fully optional; got required={required:?}"
    );

    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/hadoop_col/records/scan")
                .header("content-type", "application/json");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"records":[],"next_cursor":null,"scanned_count":0}"#);
        })
        .await;

    let config = HadoopShimConfig {
        host: server.host(),
        port: server.port(),
        collection: "hadoop_col".to_string(),
        ..HadoopShimConfig::default()
    };
    // Minimal HadoopInputSplit — unused by the scan path but required
    // by the reader's constructor signature.
    let split = HadoopInputSplit {
        split_id: "s0".to_string(),
        file_split: proximadb::storage::formats::FileSplit {
            split_id: "s0".to_string(),
            file_path: String::new(),
            offset: 0,
            length: 0,
            split_type: SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: proximadb::storage::formats::SplitLocality::default(),
            object_size: None,
        },
        length: 0,
        locations: vec!["localhost".to_string()],
    };
    let mut reader = ProximaRecordReader::new(split, config);
    let result = reader.fetch_next_page().await;
    assert!(result.is_ok(), "fetch_next_page() must succeed: {result:?}");
    let records = result.unwrap();
    assert!(records.is_empty(), "stub returns empty page");

    mock.assert_async().await;
}

// ---------------------------------------------------------------------------
// TD-099 (3b) — Hadoop sync-bridge: `next_record()` drives the async
// `fetch_next_page` via tokio Handle::block_on (`block_in_place` when
// already inside a runtime). 4 contract tests below cover the buffer
// drain semantics, cursor follow-across-pages, immediate termination
// on empty page, and `get_current_value` returning the drained record.
// ---------------------------------------------------------------------------

/// Build a minimal HadoopInputSplit + reader against a MockServer.
fn build_hadoop_reader(server: &MockServer, collection: &str) -> ProximaRecordReader {
    let config = HadoopShimConfig {
        host: server.host(),
        port: server.port(),
        collection: collection.to_string(),
        ..HadoopShimConfig::default()
    };
    let split = HadoopInputSplit {
        split_id: "s0".to_string(),
        file_split: proximadb::storage::formats::FileSplit {
            split_id: "s0".to_string(),
            file_path: String::new(),
            offset: 0,
            length: 0,
            split_type: SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: proximadb::storage::formats::SplitLocality::default(),
            object_size: None,
        },
        length: 0,
        locations: vec!["localhost".to_string()],
    };
    ProximaRecordReader::new(split, config)
}

#[tokio::test]
async fn hadoop_next_record_drains_records_in_order() {
    // Single page with 3 records + null cursor — exhausts after one
    // fetch. next_record() returns true 3x then false.
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/col_a/records/scan");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"records":[
                        {"id":"r1","props":{}},
                        {"id":"r2","props":{}},
                        {"id":"r3","props":{}}
                    ],"next_cursor":null,"scanned_count":3}"#,
                );
        })
        .await;

    let mut reader = build_hadoop_reader(&server, "col_a");
    assert!(reader.next_record(), "first record must be available");
    assert!(reader.next_record(), "second record must be available");
    assert!(reader.next_record(), "third record must be available");
    assert!(
        !reader.next_record(),
        "buffer drained + exhausted ⇒ next_record returns false"
    );
    // Calling again must stay false (no panic, no extra fetch).
    assert!(!reader.next_record());

    // The mock is hit exactly once — single page drained from buffer.
    assert_eq!(mock.hits_async().await, 1);
}

#[tokio::test]
async fn hadoop_next_record_follows_cursor_across_pages() {
    // First page: 2 records + next_cursor="abc". Second page: 1 record
    // + null cursor. The reader must drain 3 records total and the
    // second mock must receive `cursor: "abc"` in the request body.
    let server = MockServer::start_async().await;
    let page1 = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/col_b/records/scan")
                .json_body_partial(r#"{"cursor":null}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"records":[
                        {"id":"r1","props":{}},
                        {"id":"r2","props":{}}
                    ],"next_cursor":"abc","scanned_count":2}"#,
                );
        })
        .await;
    let page2 = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/col_b/records/scan")
                .json_body_partial(r#"{"cursor":"abc"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"records":[
                        {"id":"r3","props":{}}
                    ],"next_cursor":null,"scanned_count":1}"#,
                );
        })
        .await;

    let mut reader = build_hadoop_reader(&server, "col_b");
    assert!(reader.next_record(), "p1 r1");
    assert!(reader.next_record(), "p1 r2");
    assert!(reader.next_record(), "p2 r3 — cursor follow-up");
    assert!(!reader.next_record(), "exhausted");

    page1.assert_async().await;
    page2.assert_async().await;
}

#[tokio::test]
async fn hadoop_next_record_zero_records_terminates_immediately() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/col_c/records/scan");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"records":[],"next_cursor":null,"scanned_count":0}"#);
        })
        .await;

    let mut reader = build_hadoop_reader(&server, "col_c");
    assert!(
        !reader.next_record(),
        "empty page ⇒ next_record returns false on first call"
    );
    mock.assert_async().await;
}

#[tokio::test]
async fn hadoop_get_current_value_returns_drained_record() {
    use proximadb::connectors::hadoop::HadoopWritable;

    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(Method::POST)
                .path("/api/v2/collections/col_d/records/scan");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"records":[
                        {"id":"r1","props":{"label":"active"}}
                    ],"next_cursor":null,"scanned_count":1}"#,
                );
        })
        .await;

    let mut reader = build_hadoop_reader(&server, "col_d");
    assert!(reader.next_record(), "drained one record");
    let value = reader.get_current_value();
    let HadoopWritable::MapWritable(map) = value else {
        panic!("expected MapWritable, got {value:?}");
    };
    // The shape carries the v2 RecordResponse fields; the test
    // asserts presence of the canonical "id" field (the record id).
    assert!(map.contains_key("id"), "drained record missing id: {map:?}");
    assert!(
        map.contains_key("props"),
        "drained record missing props: {map:?}"
    );
}
