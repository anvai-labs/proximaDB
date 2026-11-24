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

use axum::Json;
use axum::body::{Body, to_bytes};
use axum::extract::{Extension, State};
use axum::http::{Request, StatusCode};
use serde_json::json;
use tonic::Request as TonicRequest;
use tower::ServiceExt;

use proximadb::network::grpc::v2::record_service::ProximaRecordServiceImpl;
use proximadb::network::mcp::EngineBackend;
use proximadb::network::multi_server::{ServiceProfile, SharedServices};
use proximadb::network::rest::canonical::handlers::{
    AppState, RestCoreServices, SqlQueryRequest, execute_sql as canonical_execute_sql,
};
use proximadb::network::rest::v2::create_v2_router;
use proximadb::proto::proximadb_v2::proxima_record_service_server::ProximaRecordService as _;
use proximadb::proto::proximadb_v2::{V2QueryRequest, V2QueryResponse, V2QueryRow};
use proximadb_mcp::tools::ReferenceBackend;

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

// ════════════════════════════════════════════════════════════════════════════
// TD-SPINE-6 (ADR-098 D5): one decision path for every surface.
//
// The legs above prove REST-v2/sql ↔ gRPC over the api_handlers spine. The
// tests below extend the SAME-SQL parity to the surfaces that reach the
// engine through `UnifiedQueryPort` — the canonical /sql handler, and MCP —
// which NOW delegate plain SQL to the same `QueryFacadeAdapter`
// SQL-authority body (ComputeScheduler::route_select decision path) instead
// of the federated core that used to refuse plain SQL. The pgwire +
// REST-v2/query legs live in tests/spine6_one_planner_parity_e2e.rs (they
// need a full server boot).
// ════════════════════════════════════════════════════════════════════════════

/// Same stack as `build_stack`, plus the unified query port wired onto the
/// AppState exactly the way the live server wires it (server.rs) — the port
/// impl wraps the SAME adapter the api_handlers spine uses.
async fn build_port_stack() -> (AppState, Arc<SharedServices>, tempfile::TempDir) {
    let (mut state, services, temp) = build_stack().await;
    let port = Arc::new(
        proximadb::query::UnifiedQueryPortImpl::new(services.query_adapter.clone())
            .with_catalog_manager(services.catalog_manager.clone()),
    ) as Arc<dyn proximadb_runtime::UnifiedQueryPort>;
    state = state.with_unified_query_port(port.clone());
    (state, services, temp)
}

/// Canonical /sql leg: the REAL canonical handler (handlers::execute_sql)
/// called in-process — the route the canonical REST surface uses. Runs the
/// caller's tenant through the port context (identity as data).
async fn canonical_port_sql(state: &AppState, sql: &str) -> serde_json::Value {
    let tenant = proximadb::network::middleware::tenant::TenantContext::new(
        "default",
        proximadb::network::middleware::tenant::TenantIdSource::Default,
    );
    let Json(body) = canonical_execute_sql(
        State(state.clone()),
        Some(Extension(tenant)),
        Json(SqlQueryRequest {
            query: sql.to_string(),
            parameters: None,
            collection: None,
            timeout_ms: None,
            seeding: None,
        }),
    )
    .await
    .unwrap_or_else(|error| panic!("canonical /sql leg failed for {sql}: {error:?}"));
    body
}

/// MCP leg: the REAL engine backend's `search` tool over the SAME port. The
/// backend's other dependencies (event log, vector ops) are not on the SQL
/// search path and stay unwired, exactly like the minimal MCP mount.
async fn mcp_search(
    api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    port: Arc<dyn proximadb_runtime::UnifiedQueryPort>,
    query: &str,
) -> serde_json::Value {
    let backend = EngineBackend::new(
        api_handlers,
        Some(port),
        None,
        None,
        Some("default".to_string()),
    );
    backend
        .search(&json!({ "collection_id": "spine6", "query": query }))
        .await
        .expect("MCP search must succeed")
}

/// Render one JSON row object into ordered cells for the named columns.
fn cells_of(row: &serde_json::Value, columns: &[&str]) -> Vec<String> {
    columns
        .iter()
        .map(|column| {
            let cell = &row[*column];
            match cell {
                serde_json::Value::Null => String::new(),
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            }
        })
        .collect()
}

/// The one-planner row comparator: legs agree on the ordered, rendered cells
/// of every row. Exposed so the mutation probe can prove it detects drift.
fn legs_agree(reference: &[Vec<String>], candidate: &[Vec<String>]) -> bool {
    reference.len() == candidate.len()
        && reference
            .iter()
            .zip(candidate.iter())
            .all(|(want, got)| want == got)
}

#[track_caller]
fn assert_leg(reference: &[Vec<String>], candidate: &[Vec<String>], leg: &str) {
    assert!(
        legs_agree(reference, candidate),
        "one-planner parity broken on the {leg} leg:\n  reference: {reference:?}\n  candidate: {candidate:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_sql_runs_through_the_scheduler_on_every_in_process_surface() {
    let (state, services, _temp) = build_port_stack().await;

    // Seed THROUGH THE PORT (canonical /sql): DDL and DML through the unified
    // port exercise the shared write dispatch — pre-TD-SPINE-6 these were
    // refused by the federated core ("Standard relational SQL execution is
    // not configured").
    let ddl = "CREATE TABLE spine6_port (id TEXT, score DOUBLE, label TEXT)";
    let dml = "INSERT INTO spine6_port (id, score, label) VALUES ('a', 1.5, 'x'), ('b', 2.5, 'y')";
    let select = "SELECT id, score, label FROM spine6_port ORDER BY id";
    canonical_port_sql(&state, ddl).await;
    canonical_port_sql(&state, dml).await;

    let columns = ["id", "score", "label"];

    // Reference: the api_handlers spine (REST v2 /sql over the REAL router).
    let rest = rest_sql(&state, select).await;
    let reference: Vec<Vec<String>> = rest["rows"]
        .as_array()
        .expect("REST rows")
        .iter()
        .map(|row| cells_of(row, &columns))
        .collect();
    assert_eq!(
        reference.len(),
        2,
        "seeded rows must be visible: {reference:?}"
    );

    // Canonical /sql (unified query port → scheduler decision path).
    let canonical = canonical_port_sql(&state, select).await;
    let canonical_rows: Vec<Vec<String>> = canonical["rows"]
        .as_array()
        .expect("canonical /sql rows")
        .iter()
        .map(|row| cells_of(row, &columns))
        .collect();
    assert_leg(&reference, &canonical_rows, "canonical /sql (port)");

    // gRPC (api_handlers spine — already scheduler-routed; must stay equal).
    let grpc = grpc_sql(&services, select).await;
    let grpc_rows: Vec<Vec<String>> = grpc
        .rows
        .iter()
        .map(|row| cells_of(&grpc_row_to_json(row), &columns))
        .collect();
    assert_leg(&reference, &grpc_rows, "gRPC");

    // MCP (unified query port → scheduler decision path).
    let port = state
        .unified_query_port
        .clone()
        .expect("port wired by build_port_stack");
    let mcp = mcp_search(services.api_handlers.clone(), port, select).await;
    let mcp_records = mcp["records"]
        .as_array()
        .unwrap_or_else(|| panic!("MCP records array missing: {mcp}"));
    let mcp_rows: Vec<Vec<String>> = mcp_records
        .iter()
        .map(|row| cells_of(row, &columns))
        .collect();
    assert_leg(&reference, &mcp_rows, "MCP search (port)");
}

/// Render a gRPC row (HashMap<String, TypedValue>) as a JSON object the
/// shared cell renderer can read.
fn grpc_row_to_json(row: &proximadb::proto::proximadb_v2::V2QueryRow) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (column, cell) in &row.values {
        object.insert(column.clone(), typed_to_json(cell));
    }
    serde_json::Value::Object(object)
}

/// TD-SPINE-6 mutation probe: the parity comparator must FAIL when any leg's
/// rendering drifts (a cell altered, a row dropped, a row added) — proving
/// the one-planner comparisons above are not tautological.
#[test]
fn parity_comparator_detects_mutated_legs() {
    let reference = vec![
        vec!["a".to_string(), "1.5".to_string()],
        vec!["b".to_string(), "2.5".to_string()],
    ];
    // A mutated cell.
    let mutated_cell = vec![
        vec!["a".to_string(), "9.9".to_string()],
        vec!["b".to_string(), "2.5".to_string()],
    ];
    assert!(
        !legs_agree(&reference, &mutated_cell),
        "a mutated cell must break parity"
    );

    // A dropped row and an added row.
    let dropped_row = vec![vec!["a".to_string(), "1.5".to_string()]];
    assert!(
        !legs_agree(&reference, &dropped_row),
        "a dropped row must break parity"
    );
    let added_row = vec![
        vec!["a".to_string(), "1.5".to_string()],
        vec!["b".to_string(), "2.5".to_string()],
        vec!["c".to_string(), "3.5".to_string()],
    ];
    assert!(
        !legs_agree(&reference, &added_row),
        "an added row must break parity"
    );

    // And the identity check: an unmutated leg agrees.
    let same = reference.clone();
    assert!(legs_agree(&reference, &same), "an unmutated leg must agree");
}

/// TD-SPINE-6 fail-closed acceptance: with neither an explicit identity
/// tenant nor an in-force request scope, plain SQL through the port errors
/// instead of executing unscoped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_sql_through_the_port_fails_closed_without_identity_or_scope() {
    let (state, _services, _temp) = build_port_stack().await;
    let port = state
        .unified_query_port
        .clone()
        .expect("port wired by build_port_stack");

    // Seed a real table through the canonical leg (it wraps a scope).
    canonical_port_sql(&state, "CREATE TABLE spine6_gate (id TEXT)").await;
    canonical_port_sql(&state, "INSERT INTO spine6_gate (id) VALUES ('a')").await;
    let select = "SELECT id FROM spine6_gate";

    // No task-local scope on this thread (nothing wrapped the call), and an
    // anonymous context carries no tenant: the port must refuse.
    let error = port
        .execute_unified_query(
            select.to_string(),
            None,
            None,
            None,
            proximadb_runtime::UnifiedQueryPortContext::for_identity(
                proximadb_runtime::PortIdentity::anonymous(),
            ),
        )
        .await
        .expect_err("anonymous, scope-less plain SQL must fail closed");
    assert!(
        error.to_string().contains("no request identity"),
        "unexpected error class: {error}"
    );

    // The same call WITH an explicit identity tenant succeeds through the
    // scheduler path (the positive control — the gate is not a blanket deny).
    let ok = port
        .execute_unified_query(
            select.to_string(),
            None,
            None,
            None,
            proximadb_runtime::UnifiedQueryPortContext::for_identity(
                proximadb_runtime::PortIdentity::for_tenant("default"),
            ),
        )
        .await
        .expect("identity-carrying plain SQL must execute via the scheduler");
    assert!(
        ok["records"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "the seeded row must be visible: {ok}"
    );
}
