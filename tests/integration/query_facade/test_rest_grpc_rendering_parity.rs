// Dual-TRANSPORT rendering parity over one query facade (2026-11-08).
//
// The re-enabled facade-consistency suite (test_rest_grpc_parity.rs) calls the
// SAME adapter method twice — idempotence, not parity (probe-proven in its
// review). THIS file exercises two genuinely different transport stacks:
//
//   REST leg : the real api-crate multimodal router (axum, tower oneshot)
//              → UnifiedQueryPortImpl → adapter.federated_query
//              → renderer: `result_to_json` (records[].{id,score,rank})
//   gRPC leg : the real v2 graph service (direct tonic method call)
//              → adapter.graph_query
//              → renderer: rows[].columns["data"] (stringified JSON)
//
// Both legs share ONE `QueryFacadeAdapter` over one `UnifiedQueryFacade` whose
// mock strategies return mirrored fixtures. A rendering regression on either
// stack (field renames, type coercions, truncation, ordering loss) fails here
// even though each leg renders through entirely different code.
//
// HONEST SCOPE: the legs reach the facade via different methods
// (`federated_query` vs `graph_query`) because no transport handler calls the
// facade's native vector entry today (verified 2026-11-08: the only
// `adapter.vector_search` caller is the embedded binding), and the same-SQL
// pairing (REST /api/v2/sql vs gRPC ProximaRecordService.ExecuteQuery) needs
// a full AppState + concrete services — recorded as the stronger follow-up.
// What this file proves is TRANSPORT RENDERING parity for mirrored results,
// not same-method equivalence.
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

use async_trait::async_trait;
use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde_json::json;
use tonic::Request as TonicRequest;

use proximadb::graph::service::GraphOperationsService;
use proximadb::network::grpc::v2::graph_service::ProximaGraphServiceImpl;
use proximadb::proto::proximadb_v2::proxima_graph_service_server::ProximaGraphService as _;
use proximadb::proto::proximadb_v2::{ExecuteGraphQueryRequest, GraphQueryLanguage};
use proximadb::query::{
    FacadeConfig, QueryContext, QueryFacadeAdapter, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy, QueryType, ScoredRecord, UnifiedQueryFacade, UnifiedQueryPortImpl,
};
use proximadb_api::rest::{UnifiedQueryRestState, create_multimodal_router};
use proximadb_records::ProximaRecord;
use proximadb_runtime::UnifiedQueryPort;

use tower::ServiceExt;

/// Mirrored fixture: (id, score) pairs both legs must render identically
/// (ordered, exact ids; scores compared with float tolerance).
const FIXTURE: &[(&str, f32)] = &[("vec_001", 0.95), ("vec_002", 0.90), ("vec_003", 0.85)];

fn scored_fixture() -> Vec<ScoredRecord> {
    FIXTURE
        .iter()
        .enumerate()
        .map(|(i, (id, score))| ScoredRecord {
            record: ProximaRecord {
                oid: id.to_string(),
                ..Default::default()
            },
            score: *score,
            rank: (i + 1) as u32,
        })
        .collect()
}

// =============================================================================
// Mock strategies (mirrored fixtures per query type)
// =============================================================================

struct FederatedEchoStrategy;
struct GraphEchoStrategy;

#[async_trait]
impl QueryStrategy for FederatedEchoStrategy {
    fn name(&self) -> &str {
        "federated_echo"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(request.query_type, QueryType::Federated)
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(
        &self,
        _request: QueryRequest,
        _ctx: &QueryContext,
    ) -> anyhow::Result<QueryResult> {
        Ok(QueryResult {
            data: QueryResultData::VectorResults(scored_fixture()),
            metrics: None,
        })
    }
}

#[async_trait]
impl QueryStrategy for GraphEchoStrategy {
    fn name(&self) -> &str {
        "graph_echo"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(request.query_type, QueryType::Graph)
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(
        &self,
        _request: QueryRequest,
        _ctx: &QueryContext,
    ) -> anyhow::Result<QueryResult> {
        // Rows of plain JSON objects — the graph service surfaces Rows items
        // through the same `data` string column as Graph nodes.
        let rows = FIXTURE
            .iter()
            .map(|(id, score)| json!({ "id": id, "score": score }))
            .collect();
        Ok(QueryResult {
            data: QueryResultData::Rows(rows),
            metrics: None,
        })
    }
}

fn shared_adapter() -> Arc<QueryFacadeAdapter> {
    let strategies: Vec<Arc<dyn QueryStrategy>> =
        vec![Arc::new(FederatedEchoStrategy), Arc::new(GraphEchoStrategy)];
    let facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));
    Arc::new(QueryFacadeAdapter::without_validation(facade))
}

// =============================================================================
// Legs
// =============================================================================

/// REST leg: the real multimodal router over the real port impl.
/// Returns the rendered `records[].{id, score}` list.
async fn rest_leg(adapter: Arc<QueryFacadeAdapter>) -> Vec<serde_json::Value> {
    let port: Arc<dyn UnifiedQueryPort> = Arc::new(UnifiedQueryPortImpl::new(adapter));
    let router = create_multimodal_router().with_state(UnifiedQueryRestState {
        unified_query_port: port,
    });

    let body = json!({
        "query": "SELECT * FROM VECTOR_SEARCH('parity_collection', '[0.1, 0.2]', 3)",
    });
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/federated")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("oneshot completes");

    assert_eq!(
        response.status(),
        axum::http::StatusCode::OK,
        "REST leg must return 200 (got body: {:?})",
        to_bytes(response.into_body(), usize::MAX).await
    );
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON body");
    json["records"]
        .as_array()
        .expect("records array in response")
        .clone()
}

/// gRPC leg: the real v2 graph service over the same adapter.
/// Returns the parsed `data`-column JSON objects.
async fn grpc_leg(adapter: Arc<QueryFacadeAdapter>) -> Vec<serde_json::Value> {
    let svc = ProximaGraphServiceImpl::new(Arc::new(GraphOperationsService::new()), Some(adapter));
    let request = TonicRequest::new(ExecuteGraphQueryRequest {
        graph_id: String::new(), // empty → adapter target None (tenant-scoped default)
        language: GraphQueryLanguage::Cypher as i32,
        query: "MATCH (n) RETURN n".to_string(),
        timeout_ms: None,
    });

    let response = svc
        .execute_query(request)
        .await
        .expect("gRPC execute_query succeeds");
    let inner = response.into_inner();
    assert!(
        inner.error_message.is_none(),
        "gRPC leg errored: {:?}",
        inner.error_message
    );

    inner
        .rows
        .into_iter()
        .map(|row| {
            let data = row
                .columns
                .get("data")
                .and_then(|v| v.value.as_ref())
                .expect("data column present");
            let text = match data {
                proximadb::proto::proximadb_v2::graph_property_value::Value::StringValue(s) => s,
                other => panic!("expected stringified JSON data column, got {other:?}"),
            };
            serde_json::from_str(text).expect("data column is valid JSON")
        })
        .collect()
}

// =============================================================================
// Parity assertions
// =============================================================================

#[tokio::test]
async fn mirrored_results_render_identically_across_transports() {
    let adapter = shared_adapter();
    let (rest, grpc) = tokio::join!(rest_leg(adapter.clone()), grpc_leg(adapter));

    // Row counts equal (no truncation on either renderer).
    assert_eq!(rest.len(), FIXTURE.len(), "REST renders all fixture rows");
    assert_eq!(grpc.len(), FIXTURE.len(), "gRPC renders all fixture rows");

    // Ordered ids equal across transports (ordering preserved by both).
    let rest_ids: Vec<&str> = rest
        .iter()
        .map(|r| r["id"].as_str().expect("REST id is a string"))
        .collect();
    let grpc_ids: Vec<&str> = grpc
        .iter()
        .map(|r| r["id"].as_str().expect("gRPC id is a string"))
        .collect();
    let fixture_ids: Vec<&str> = FIXTURE.iter().map(|(id, _)| *id).collect();
    assert_eq!(rest_ids, fixture_ids, "REST preserves fixture ordering");
    assert_eq!(grpc_ids, fixture_ids, "gRPC preserves fixture ordering");

    // Scores compared EXACTLY (review round-1: both legs widen the identical
    // f32 through serde_json's f32→f64 conversion, so cross-leg scores are
    // bit-identical today — exact equality catches sub-1e-6 coercion drift a
    // tolerance would hide).
    for (i, ((rest_row, grpc_row), fixture)) in
        rest.iter().zip(grpc.iter()).zip(FIXTURE.iter()).enumerate()
    {
        let rest_score = rest_row["score"].as_f64().expect("REST score numeric");
        let grpc_score = grpc_row["score"].as_f64().expect("gRPC score numeric");
        assert_eq!(
            rest_score, grpc_score,
            "row {i}: REST/gRPC score divergence (fixture {fixture:?})"
        );
        assert_eq!(
            rest_score, fixture.1 as f64,
            "row {i}: REST score drifted from fixture"
        );
    }
}

/// Shape pin for the REST renderer: records are `{id, score, rank}` objects
/// (`result_to_json`). If the shape drifts, the parity test above and this
/// pin both fail — the pin gives the sharper diagnostic. It also guards WHICH
/// result arm the REST leg renders: if the federated mock returned Rows
/// instead of VectorResults, the parity comparison would pass vacuously but
/// this pin fails (the two-test pairing closes that hole).
#[tokio::test]
async fn rest_leg_renders_the_port_json_shape() {
    let adapter = shared_adapter();
    let rest = rest_leg(adapter).await;
    let first = &rest[0];
    assert!(first.get("id").is_some(), "REST record has id");
    assert!(first.get("score").is_some(), "REST record has score");
    assert_eq!(first["rank"].as_u64(), Some(1), "REST record carries rank");
}
