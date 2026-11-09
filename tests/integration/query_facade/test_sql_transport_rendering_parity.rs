// Same-SQL TRANSPORT rendering parity — the strongest variant (2026-11-09).
//
// The rendering-parity file next door (test_rest_grpc_rendering_parity.rs)
// exercises two different transport stacks over ONE facade but reaches it via
// DIFFERENT adapter methods (federated_query vs graph_query) — because its
// exploration recorded the REST /api/v2/sql handler as "needs full AppState,
// not reasonably buildable". That blocker was FALSE: an in-crate test helper
// (canonical/handlers.rs build_test_app_state) builds a complete AppState via
// SharedServices::new(ServiceProfile::Embedded), and the OpenAPI-resolution
// test drives create_v2_router with tower oneshot. This file does the same:
//
//   REST leg : the REAL v2 router (axum + oneshot) → v2/sql.rs execute_sql
//              → state.api_handlers.execute_sql → v2_row_render (JSON rows)
//   gRPC leg : the REAL v2 record service (direct tonic call)
//              → the SAME api_handlers.execute_sql → V2QueryResponse
//              (rows[].values HashMap<String, TypedValue>)
//
// Both legs converge on ONE SharedServices' Arc<dyn ApiHandlersPort> — same
// adapter, same DmlService, same tempdir-backed engine stack. A divergence
// between the two RENDERERS of the identical execution result fails here.
//
// A dedicated test below PINS UUID-cell rendering agreement: both legs
// render the dashed canonical string (the exploration's raw-bytes-on-gRPC
// claim was refuted empirically — the pin now guards the agreement).
/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 */

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::json;
use tonic::Request as TonicRequest;
use tower::ServiceExt;

use proximadb::network::grpc::v2::record_service::ProximaRecordServiceImpl;
use proximadb::network::multi_server::{ServiceProfile, SharedServices};
use proximadb::network::rest::canonical::handlers::{AppState, RestCoreServices};
use proximadb::network::rest::v2::create_v2_router;
use proximadb::proto::proximadb_v2::proxima_record_service_server::ProximaRecordService as _;
use proximadb::proto::proximadb_v2::{V2QueryRequest, V2QueryResponse};

fn file_url(path: &std::path::Path) -> String {
    format!("file://{}", path.display())
}

/// Build the full (real, tempdir-backed) service stack — mirroring the
/// in-crate build_test_app_state helper, minus the mock-port overrides
/// (we want the REAL api_handlers spine on both legs).
async fn build_stack() -> (AppState, Arc<SharedServices>, tempfile::TempDir) {
    let _ = proximadb_hardware::hardware_capabilities(); // OnceLock init

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let storage_path = temp_dir.path().join("storage");
    let metadata_path = temp_dir.path().join("metadata");
    let data_dir = temp_dir.path().join("server_data");
    std::fs::create_dir_all(&storage_path).expect("storage dir");
    std::fs::create_dir_all(&metadata_path).expect("metadata dir");
    std::fs::create_dir_all(&data_dir).expect("data dir");

    let mut config = proximadb::core::config::Config::default();
    config.server.data_dir = data_dir.clone();
    config.storage.metadata_url = file_url(&metadata_path);
    config.storage.storage_locations = vec![proximadb::core::config::StorageLocation {
        url: file_url(&storage_path),
        weight: 1,
        tags: vec!["test".to_string()],
        io_budget: None,
    }];

    let (shared_services, _) = SharedServices::new(
        None,
        &config.storage,
        None,
        Some(&config),
        ServiceProfile::Embedded,
    )
    .await
    .expect("shared services");

    let state = AppState::new(
        RestCoreServices::from_shared_services(&shared_services),
        shared_services.graph_execution_service.clone(),
        None,
        data_dir,
        None,
        None,
    );
    (state, Arc::new(shared_services), temp_dir)
}

fn default_identity() -> proximadb_tenant::ResolvedRequestIdentity {
    proximadb_tenant::ResolvedRequestIdentity {
        tenant: "default".to_string(),
        subject: None,
        auth_class: proximadb_tenant::AuthClass::Anonymous,
        tenant_stable_id: None,
    }
}

/// REST leg: real v2 router, tower oneshot POST /sql.
async fn rest_sql(state: &AppState, sql: &str) -> serde_json::Value {
    let router = create_v2_router()
        .with_state(state.clone())
        .layer(axum::Extension(default_identity()));

    let body = json!({ "query": sql }).to_string();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sql")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .expect("request builds"),
        )
        .await
        .expect("oneshot");

    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert_eq!(
        status,
        StatusCode::OK,
        "REST leg must return 200, got {status}: {}",
        String::from_utf8_lossy(&bytes)
    );
    serde_json::from_slice(&bytes).expect("SqlResponse JSON")
}

/// gRPC leg: real v2 record service, direct tonic call.
async fn grpc_sql(services: &SharedServices, sql: &str) -> V2QueryResponse {
    let svc =
        ProximaRecordServiceImpl::new(services.api_handlers.clone(), services.record_ops.clone());
    let response = svc
        .execute_query(TonicRequest::new(V2QueryRequest {
            query: sql.to_string(),
            collection_id: String::new(),
            limit: None,
            offset: None,
        }))
        .await
        .expect("gRPC execute_query succeeds");
    response.into_inner()
}

/// Normalize a gRPC TypedValue cell to a serde_json Value for comparison.
fn typed_to_json(tv: &proximadb::proto::proximadb_v2::TypedValue) -> serde_json::Value {
    use proximadb::proto::proximadb_v2::typed_value::Value as TV;
    match &tv.value {
        Some(TV::TextValue(s)) => json!(s),
        Some(TV::IntegerValue(i)) => json!(i),
        Some(TV::FloatValue(d)) => json!(d),
        Some(TV::BooleanValue(b)) => json!(b),
        None => serde_json::Value::Null,
        other => json!(format!("{other:?}")), // non-scalar: FAILS LOUDLY by design — extend this matcher before seeding non-scalar types
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_sql_renders_identically_across_rest_and_grpc() {
    let (state, services, _temp) = build_stack().await;

    // Seed through the SHARED spine (either leg works; use REST for DDL+DML).
    // ANSI TABLE syntax (the frontend parses standard SQL — not the
    // vector-era COLLECTION keyword).
    let ddl = "CREATE TABLE parity_sql (id TEXT, score DOUBLE, label TEXT)";
    let dml = "INSERT INTO parity_sql (id, score, label) VALUES ('a', 1.5, 'x'), ('b', 2.5, 'y')";
    // Failures here already surface via the 200-status assert with body text.
    rest_sql(&state, ddl).await;
    rest_sql(&state, dml).await;

    let select = "SELECT id, score, label FROM parity_sql ORDER BY id";

    let rest = rest_sql(&state, select).await;
    let grpc = grpc_sql(&services, select).await;

    // Column parity (order-sensitive).
    let rest_columns: Vec<String> = rest["columns"]
        .as_array()
        .expect("REST columns array")
        .iter()
        .map(|v| v.as_str().expect("column name").to_string())
        .collect();
    assert_eq!(
        rest_columns,
        vec!["id", "score", "label"],
        "REST renders expected columns in order"
    );
    assert_eq!(
        grpc.columns, rest_columns,
        "gRPC renders the same columns in the same order"
    );

    // Row-count parity.
    let rest_rows = rest["rows"].as_array().expect("REST rows array");
    assert_eq!(rest_rows.len(), 2, "REST returns both seeded rows");
    assert_eq!(grpc.rows.len(), 2, "gRPC returns both seeded rows");

    // Cell parity, row by row, by column name.
    for (ri, (rest_row, grpc_row)) in rest_rows.iter().zip(grpc.rows.iter()).enumerate() {
        let rest_obj = rest_row.as_object().expect("REST row object");
        for col in &rest_columns {
            let rest_cell = &rest_obj[col];
            let grpc_cell = grpc_row
                .values
                .get(col)
                .unwrap_or_else(|| panic!("gRPC row {ri} missing column {col}"));
            let grpc_json = typed_to_json(grpc_cell);
            // Numeric compare with tolerance (JSON f64 vs proto double).
            if let (Some(r), Some(g)) = (rest_cell.as_f64(), grpc_json.as_f64()) {
                assert!(
                    (r - g).abs() < 1e-9,
                    "row {ri} col {col}: REST {r} != gRPC {g}"
                );
            } else {
                assert_eq!(
                    rest_cell, &grpc_json,
                    "row {ri} col {col}: REST/gRPC cell divergence"
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uuid_cells_render_identically_across_transports() {
    // UUID rendering pin. The #2485 exploration claimed an intentional
    // asymmetry (REST dashes, gRPC carries raw bytes) — EMPIRICALLY REFUTED
    // on this path: the gRPC TypedValue for a SQL-selected UUID is
    // TextValue(<dashed string>), byte-identical to REST's v2_row_render
    // output. This pin asserts the AGREEMENT; if either side changes
    // (e.g. gRPC starts carrying UuidValue bytes), it fails and forces a
    // conscious contract decision instead of silent drift.
    let (state, services, _temp) = build_stack().await;

    rest_sql(&state, "CREATE TABLE parity_uuid (uid UUID)").await;
    rest_sql(
        &state,
        "INSERT INTO parity_uuid (uid) VALUES ('123e4567-e89b-42d3-a456-426614174000')",
    )
    .await;
    let select = "SELECT uid FROM parity_uuid";

    let rest = rest_sql(&state, select).await;
    let grpc = grpc_sql(&services, select).await;

    let expected = "123e4567-e89b-42d3-a456-426614174000";

    let rest_uid = rest["rows"][0]["uid"]
        .as_str()
        .expect("REST uid is a string");
    assert_eq!(rest_uid, expected, "REST renders the dashed UUID form");

    use proximadb::proto::proximadb_v2::typed_value::Value as TV;
    let grpc_cell = grpc.rows[0]
        .values
        .get("uid")
        .expect("gRPC uid cell present");
    match &grpc_cell.value {
        Some(TV::TextValue(text)) => {
            assert_eq!(
                text, expected,
                "gRPC renders the SAME dashed string (the exploration's raw-bytes claim was refuted empirically)"
            );
        }
        other => panic!(
            "gRPC UUID cell changed shape ({other:?}) — a renderer divergence: decide the contract consciously"
        ),
    }
}
