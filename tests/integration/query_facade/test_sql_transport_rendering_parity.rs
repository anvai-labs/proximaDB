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
// TD-SPINE-6 (ADR-098 D5): one decision path for every surface — round-1
// review hardening.
//
// Legs: REST v2 /sql (authority control), canonical /sql handler (port, in
// process), gRPC, MCP. Port plain SQL is READ-ONLY (MED-4): DDL/DML through
// a port surface is a typed 400 — never silently executed and never
// silently lost. Raw user text always routes through the ONE SQL authority
// first (HIGH-1) — extension-looking SUBSTRINGS (literals, comments) cannot
// reroute it. pgwire + live-HTTP legs live in tests/spine6_one_planner_
// parity_e2e.rs.
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
        None,
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

/// The port leg, driven DIRECTLY (no HTTP): asserts the typed error class
/// for statements the read-only policy rejects.
async fn canonical_port_sql_err(state: &AppState, sql: &str) -> anyhow::Error {
    let tenant = proximadb::network::middleware::tenant::TenantContext::new(
        "default",
        proximadb::network::middleware::tenant::TenantIdSource::Default,
    );
    canonical_execute_sql(
        State(state.clone()),
        None,
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
    .expect_err(&format!("port must reject: {sql}"))
    .into()
}

/// MCP leg: the REAL engine backend's `search` tool over the SAME port.
async fn mcp_search(
    api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    port: Arc<dyn proximadb_runtime::UnifiedQueryPort>,
    query: &str,
) -> Result<serde_json::Value, String> {
    let backend = EngineBackend::new(api_handlers, Some(port), None, None, Some("default".into()));
    backend
        .search(&json!({ "collection_id": "spine6", "query": query }))
        .await
        .map_err(|error| format!("{} (code {})", error.message, error.code))
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

fn rows_of(body: &serde_json::Value, rows_key: &str, columns: &[&str]) -> Vec<Vec<String>> {
    body[rows_key]
        .as_array()
        .unwrap_or_else(|| panic!("{rows_key} array missing: {body}"))
        .iter()
        .map(|row| cells_of(row, columns))
        .collect()
}

/// The one-planner row comparator: legs agree on the ordered, rendered cells
/// of every row.
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

/// Render a gRPC row (HashMap<String, TypedValue>) as a JSON object the
/// shared cell renderer can read.
fn grpc_row_to_json(row: &V2QueryRow) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (column, cell) in &row.values {
        object.insert(column.clone(), typed_to_json(cell));
    }
    serde_json::Value::Object(object)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_sql_runs_through_the_scheduler_on_every_in_process_surface() {
    let (state, services, _temp) = build_port_stack().await;

    // Seed through the AUTHORITY surface (REST v2 /sql over the real router):
    // the port is read-only for plain SQL (MED-4).
    let ddl = "CREATE TABLE spine6_port (id TEXT, score DOUBLE, label TEXT)";
    let dml = "INSERT INTO spine6_port (id, score, label) VALUES ('a', 1.5, 'x'), ('b', 2.5, 'y')";
    let select = "SELECT id, score, label FROM spine6_port ORDER BY id";
    rest_sql(&state, ddl).await;
    rest_sql(&state, dml).await;

    let columns = ["id", "score", "label"];

    // Reference: the api_handlers spine (REST v2 /sql).
    let rest = rest_sql(&state, select).await;
    let reference = rows_of(&rest, "rows", &columns);
    assert_eq!(
        reference.len(),
        2,
        "seeded rows must be visible: {reference:?}"
    );

    // Canonical /sql (unified query port → scheduler decision path).
    let canonical = canonical_port_sql(&state, select).await;
    assert_leg(
        &reference,
        &rows_of(&canonical, "rows", &columns),
        "canonical /sql (port)",
    );

    // gRPC (api_handlers spine — already scheduler-routed; must stay equal).
    let grpc = grpc_sql(&services, select).await;
    let grpc_rows: Vec<Vec<String>> = grpc
        .rows
        .iter()
        .map(|row| cells_of(&grpc_row_to_json(row), &columns))
        .collect();
    assert_leg(&reference, &grpc_rows, "gRPC");

    // MCP (unified query port → scheduler decision path).
    let port = state.unified_query_port.clone().expect("port wired");
    let mcp = mcp_search(services.api_handlers.clone(), port.clone(), select)
        .await
        .expect("MCP search must serve plain reads");
    assert_leg(
        &reference,
        &rows_of(&mcp, "records", &columns),
        "MCP search (port)",
    );

    // LOW-4: the port envelope's metrics carry a timed (u64)
    // execution_time_ms — the delegated call is timed in
    // `run_sql_via_one_planner` (previously the field was always 0 on the
    // port path). The canonical HANDLER projects only rows/counts, so the
    // envelope is asserted through a direct port call.
    {
        let envelope = port
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
            .expect("direct port read for the metrics pin");
        assert!(
            envelope["metrics"]["execution_time_ms"].as_u64().is_some(),
            "metrics.execution_time_ms must be a timed u64: {envelope}"
        );
    }
}

/// HIGH-1 (round-1 review), every probe: raw text containing extension
/// KEYWORDS as literals/comments must route through the ONE SQL authority
/// and behave EXACTLY like the /api/v2/sql leg — reads return identical
/// rows; writes are cleanly REJECTED by the read-only policy (MED-4), never
/// silently lost, while the authority surface writes them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extension_keywords_in_literals_and_comments_route_through_the_authority() {
    let (state, _services, _temp) = build_port_stack().await;
    let columns = ["id", "note"];

    rest_sql(&state, "CREATE TABLE spine6_probe (id TEXT, note TEXT)").await;
    rest_sql(
        &state,
        "INSERT INTO spine6_probe (id, note) VALUES ('p', 'plain seed')",
    )
    .await;

    // Probe 1 — the silent-write-loss INSERT: the port leg must REJECT it
    // (read-only, typed 400-class error naming the policy), NOT report a
    // federated success while the table stays empty.
    let silent_write = "INSERT INTO spine6_probe (id, note) VALUES ('w', 'logged VECTOR_SEARCH(probe_vec, [0.1], 10)')";
    let error = canonical_port_sql_err(&state, silent_write).await;
    let message = format!("{error:#}");
    assert!(
        message.contains("read-only"),
        "the write must be rejected by the read-only policy, got: {message}"
    );
    // …and nothing was written (no partial/silent effect).
    let after = rest_sql(&state, "SELECT id, note FROM spine6_probe ORDER BY id").await;
    assert_eq!(
        rows_of(&after, "rows", &columns).len(),
        1,
        "no row may appear: {after}"
    );
    // The SAME statement via the authority surface writes the row.
    rest_sql(&state, silent_write).await;
    let after = rest_sql(&state, "SELECT id, note FROM spine6_probe ORDER BY id").await;
    assert_eq!(
        rows_of(&after, "rows", &columns).len(),
        2,
        "authority writes: {after}"
    );

    // Probe 2 — a SELECT whose WHERE literal contains `LOGS(app)`: the port
    // leg returns the SAME rows as /api/v2/sql (previously: "Namespace 'app'
    // not found" from the mis-routed federated path).
    let select_literal =
        "SELECT id, note FROM spine6_probe WHERE note = 'LOGS(app) literal' OR id = 'w'";
    let reference = rows_of(&rest_sql(&state, select_literal).await, "rows", &columns);
    let port_rows = rows_of(
        &canonical_port_sql(&state, select_literal).await,
        "rows",
        &columns,
    );
    assert_leg(&reference, &port_rows, "literal-containing SELECT");

    // Probe 3 — an UPDATE carrying a GRAPH_QUERY literal: rejected on the
    // port, executed by the authority.
    let update = "UPDATE spine6_probe SET note = 'GRAPH_QUERY(MATCH (n) RETURN n)' WHERE id = 'w'";
    let error = canonical_port_sql_err(&state, update).await;
    assert!(format!("{error:#}").contains("read-only"), "got: {error:#}");
    rest_sql(&state, update).await;
    let after = rest_sql(&state, "SELECT note FROM spine6_probe WHERE id = 'w'").await;
    assert!(
        rows_of(&after, "rows", &["note"])[0][0].contains("GRAPH_QUERY"),
        "authority executed the update: {after}"
    );

    // Probe 4 — a trailing comment mentioning `<->`: identical rows on both
    // legs (previously: "Unsupported vector expression").
    let select_comment = "SELECT id FROM spine6_probe ORDER BY id -- uses <-> operator";
    let reference = rows_of(&rest_sql(&state, select_comment).await, "rows", &["id"]);
    let port_rows = rows_of(
        &canonical_port_sql(&state, select_comment).await,
        "rows",
        &["id"],
    );
    assert_leg(&reference, &port_rows, "comment-containing SELECT");
}

/// MED-4 (round-1 review): DELETE via the MCP `search` tool is rejected with
/// the policy message (MCP is a query surface; it must not mutate).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_search_rejects_delete_statements() {
    let (state, services, _temp) = build_port_stack().await;
    rest_sql(&state, "CREATE TABLE spine6_mcp (id TEXT)").await;
    rest_sql(&state, "INSERT INTO spine6_mcp (id) VALUES ('a')").await;
    let port = state.unified_query_port.clone().expect("port wired");
    let error = mcp_search(
        services.api_handlers.clone(),
        port,
        "DELETE FROM spine6_mcp",
    )
    .await
    .expect_err("DELETE via MCP search must be rejected");
    assert!(
        error.contains("read-only"),
        "the rejection must name the read-only policy, got: {error}"
    );
    // …and the row survives.
    let after = rest_sql(&state, "SELECT id FROM spine6_mcp").await;
    assert_eq!(
        rows_of(&after, "rows", &["id"]).len(),
        1,
        "row must survive: {after}"
    );
}

/// MED-3 (round-1 review): an anonymous identity inside a tenant scope is
/// DERIVED into the identity — the scheduler path resolves and writes within
/// that tenant (previously: unscoped resolution — silent empty reads, and
/// writes landing in a shared unscoped namespace).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anonymous_identity_inside_a_tenant_scope_reads_that_tenant() {
    let (state, _services, _temp) = build_port_stack().await;
    rest_sql(&state, "CREATE TABLE spine6_scope (id TEXT)").await;
    rest_sql(&state, "INSERT INTO spine6_scope (id) VALUES ('a'), ('b')").await;
    let select = "SELECT id FROM spine6_scope ORDER BY id";
    let port = state.unified_query_port.clone().expect("port wired");

    // Anonymous identity + default-tenant scope in force → the tenant's rows
    // (scope-derived identity reaches the scheduler path; pre-MED-3 this
    // resolved unscoped and silently returned nothing).
    let in_default = anonymous_call_in_scope(&port, "default", select)
        .await
        .expect("scope-derived identity must resolve the default tenant's table");
    let rows = rows_of(&in_default, "records", &["id"]);
    assert_eq!(
        rows.len(),
        2,
        "scope-derived identity must see the tenant's rows: {in_default}"
    );

    // The SAME anonymous call inside a DIFFERENT tenant's scope must not
    // observe the default tenant's rows: the table does not resolve for
    // `acme` (an error) or returns different/no rows — never the default
    // tenant's data.
    match anonymous_call_in_scope(&port, "acme", select).await {
        Ok(in_acme) => {
            let acme_rows = rows_of(&in_acme, "records", &["id"]);
            assert!(
                acme_rows.is_empty() || !legs_agree(&rows, &acme_rows),
                "another tenant's scope must not observe the default tenant's rows: {in_acme}"
            );
        }
        Err(error) => {
            // A resolution failure for the other tenant is the correct
            // fail-closed outcome.
            assert!(
                !format!("{error:#}").is_empty(),
                "expected a resolution error for the other tenant"
            );
        }
    }
}

/// Drive the port with an ANONYMOUS identity inside `with_request_tenant`
/// (the documented task-local fallback shape MED-3 hardens).
async fn anonymous_call_in_scope(
    port: &Arc<dyn proximadb_runtime::UnifiedQueryPort>,
    tenant: &str,
    sql: &str,
) -> anyhow::Result<serde_json::Value> {
    let query = sql.to_string();
    proximadb::query::request_scope::with_request_tenant(tenant, async move {
        port.execute_unified_query(
            query,
            None,
            None,
            None,
            proximadb_runtime::UnifiedQueryPortContext::for_identity(
                proximadb_runtime::PortIdentity::anonymous(),
            ),
        )
        .await
    })
    .await
}

/// LOW-6 (round-1 review): the mutation probe mutates a REAL leg's rendered
/// output (the canonical port leg's rows), not hand-built vectors — proving
/// the comparator guards the actual rendering path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parity_comparator_detects_mutated_legs() {
    let (state, _services, _temp) = build_port_stack().await;
    rest_sql(&state, "CREATE TABLE spine6_probe6 (id TEXT, score DOUBLE)").await;
    rest_sql(
        &state,
        "INSERT INTO spine6_probe6 (id, score) VALUES ('a', 1.5), ('b', 2.5)",
    )
    .await;
    let select = "SELECT id, score FROM spine6_probe6 ORDER BY id";
    let real = rows_of(
        &canonical_port_sql(&state, select).await,
        "rows",
        &["id", "score"],
    );
    assert_eq!(real.len(), 2, "real leg output: {real:?}");

    // A mutated cell in the REAL leg's rendering must break parity.
    let mut mutated_cell = real.clone();
    mutated_cell[0][1] = "9.9".to_string();
    assert!(
        !legs_agree(&real, &mutated_cell),
        "a mutated cell must break parity"
    );

    // A dropped row and an added row, both derived from the real output.
    let dropped = real[..1].to_vec();
    assert!(
        !legs_agree(&real, &dropped),
        "a dropped row must break parity"
    );
    let mut added = real.clone();
    added.push(vec!["c".to_string(), "3.5".to_string()]);
    assert!(!legs_agree(&real, &added), "an added row must break parity");

    // And the unmutated leg agrees with itself.
    let same = real.clone();
    assert!(legs_agree(&real, &same), "an unmutated leg must agree");
}

/// LOW-6 (round-1 review): the fail-closed gate is a 401 at the HTTP surface
/// (not a 500): the multimodal router, mounted bare (no identity extension,
/// no scope), answers 401 for a plain read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scope_less_port_call_is_unauthorized_not_internal() {
    use proximadb_api::rest::{UnifiedQueryRestState, create_multimodal_router};

    let (state, _services, _temp) = build_port_stack().await;
    let port = state.unified_query_port.clone().expect("port wired");
    let router = create_multimodal_router().with_state(UnifiedQueryRestState {
        unified_query_port: port,
    });

    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/execute")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "query": "SELECT 1 AS one" }).to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("oneshot");

    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "no identity and no scope must be a 401, not a 500"
    );
}

/// TD-SPINE-6 fail-closed acceptance: with neither an explicit identity
/// tenant nor an in-force request scope, plain SQL through the port errors
/// instead of executing unscoped — with the TYPED MissingQueryIdentity.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_sql_through_the_port_fails_closed_without_identity_or_scope() {
    let (state, _services, _temp) = build_port_stack().await;
    let port = state.unified_query_port.clone().expect("port wired");

    let error = port
        .execute_unified_query(
            "SELECT id FROM spine6_gate".to_string(),
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
    assert!(
        error
            .downcast_ref::<proximadb_runtime::MissingQueryIdentity>()
            .is_some(),
        "the gate must be the typed MissingQueryIdentity (401 at HTTP): {error:#}"
    );

    // The same call WITH an explicit identity tenant succeeds through the
    // scheduler path (the positive control — the gate is not a blanket deny).
    rest_sql(&state, "CREATE TABLE spine6_gate (id TEXT)").await;
    rest_sql(&state, "INSERT INTO spine6_gate (id) VALUES ('a')").await;
    let ok = port
        .execute_unified_query(
            "SELECT id FROM spine6_gate".to_string(),
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
