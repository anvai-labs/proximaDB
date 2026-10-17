//! # Graph REST Handlers
//!
//! Graph database endpoints migrated to `proximadb-api` using `GraphPort`.
//!
//! ## Endpoint Overview
//!
//! ```text
//! POST   /api/v2/graphs/{graph_id}/nodes              - Create node
//! GET    /api/v2/graphs/{graph_id}/nodes/{id}         - Get node
//! PUT    /api/v2/graphs/{graph_id}/nodes/{id}         - Update node
//! DELETE /api/v2/graphs/{graph_id}/nodes/{id}         - Delete node
//! GET    /api/v2/graphs/{graph_id}/nodes/{id}/neighbors - Get neighbors
//! POST   /api/v2/graphs/{graph_id}/edges              - Create edge
//! GET    /api/v2/graphs/{graph_id}/edges/{id}         - Get edge
//! PUT    /api/v2/graphs/{graph_id}/edges/{id}         - Update edge
//! DELETE /api/v2/graphs/{graph_id}/edges/{id}         - Delete edge
//! POST   /api/v2/graphs/{graph_id}/traverse           - Graph traversal
//! POST   /api/v2/graphs/{graph_id}/walk               - BFS walk (agentic)
//! POST   /api/v2/graphs/{graph_id}/step               - Single-step navigation
//! POST   /api/v2/graphs/{graph_id}/shortest_path      - Shortest path
//! POST   /api/v2/graphs/{graph_id}/query/nodes        - Query nodes
//! POST   /api/v2/graphs/{graph_id}/query/edges        - Query edges
//! POST   /api/v2/graphs/{graph_id}/query              - Declarative query
//! GET    /api/v2/graphs/{graph_id}/stats              - Graph statistics
//! POST   /api/v2/graphs/{graph_id}/nodes/batch        - Batch create nodes
//! POST   /api/v2/graphs/{graph_id}/edges/batch        - Batch create edges
//! GET    /api/v2/graphs/{graph_id}/components         - Connected components
//! GET    /api/v2/graphs/{graph_id}/cycles             - Cycle detection
//! POST   /api/v2/graphs/{graph_id}/constraints/unique - Add unique constraint
//! DELETE /api/v2/graphs/{graph_id}/constraints/unique - Remove unique constraint
//! ```
//!
//! Graph collection management and RAG endpoints return `501 Not Implemented`
//! until they are exposed through a platform port.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Router,
    extract::Path,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post, put},
};
use proximadb_proto::v1::PropertyValue;
use proximadb_proto::v1::{
    BatchEdgeRequest, BatchNodeRequest, CreateEdgeRequest, CreateGraphRequest, CreateNodeRequest,
    DeleteEdgeRequest, DeleteNodeRequest, Edge, EdgeQuery, EmbeddingVersion, GetEdgeRequest,
    GetNeighborsRequest, GetNodeRequest, GetStatsRequest, GraphSchema, Node, NodeQuery,
    PropertyFilter, PropertyFilterOperator, ShortestPathAlgorithm, TraversalAlgorithm,
    TraversalRequest, UniqueConstraintRequest, UpdateEdgeRequest, UpdateNodeRequest,
    property_value,
};
use proximadb_runtime::GraphPort;
use serde::{Deserialize, Serialize};

use super::graph_tenant_scope::TenantGraph;
use tracing::{debug, error, info, warn};

// ── State ────────────────────────────────────────────────────────────────────

/// Axum state for graph REST handlers.
#[derive(Clone)]
pub struct GraphRestState {
    pub graph_port: Arc<dyn GraphPort>,
}

// ── Handler marker structs ────────────────────────────────────────────────────

pub struct GraphHandler;
pub struct GraphTraversalHandler;

impl GraphHandler {
    pub fn new() -> Self {
        Self
    }
}
impl Default for GraphHandler {
    fn default() -> Self {
        Self::new()
    }
}
impl GraphTraversalHandler {
    pub fn new() -> Self {
        Self
    }
}
impl Default for GraphTraversalHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Canonical response types (inlined — no root-crate dep) ───────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GraphError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ResponseMetadata>,
}

impl<T> GraphResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            metadata: None,
        }
    }
    pub fn error(error: GraphError) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            metadata: None,
        }
    }
    pub fn from_error(code: ErrorCode, message: impl Into<String>) -> Self {
        Self::error(GraphError::new(code, message))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphError {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl GraphError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }
    pub fn not_found(entity_type: &str, id: &str) -> Self {
        Self {
            code: ErrorCode::NotFound,
            message: format!("{entity_type} '{id}' not found"),
            details: Some(serde_json::json!({ "entity_type": entity_type, "entity_id": id })),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InternalError, message)
    }
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    NotFound,
    AlreadyExists,
    InvalidArgument,
    ConstraintViolation,
    InternalError,
    Timeout,
    PermissionDenied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalNode {
    pub id: String,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<CanonicalEmbedding>,
    pub created_at: String,
    pub updated_at: String,
}

impl CanonicalNode {
    fn from_proto(node: &Node) -> Self {
        Self {
            id: node.id.clone(),
            labels: node.labels.clone(),
            properties: props_to_json(&node.properties),
            embedding: node.embedding.as_ref().map(CanonicalEmbedding::from_proto),
            created_at: fmt_ts(node.created_at_ms),
            updated_at: fmt_ts(node.updated_at_ms),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalEmbedding {
    pub model_id: String,
    pub model_version: String,
    pub vector: Vec<f32>,
    pub dimension: u32,
}

impl CanonicalEmbedding {
    fn from_proto(e: &EmbeddingVersion) -> Self {
        Self {
            model_id: e.model_id.clone(),
            model_version: e.model_version.clone(),
            vector: e.vector.clone(),
            dimension: e.dimension,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalEdge {
    pub id: String,
    pub from_node_id: String,
    pub to_node_id: String,
    pub edge_type: String,
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

impl CanonicalEdge {
    fn from_proto(edge: &Edge) -> Self {
        Self {
            id: edge.id.clone(),
            from_node_id: edge.from_node_id.clone(),
            to_node_id: edge.to_node_id.clone(),
            edge_type: edge.edge_type.clone(),
            properties: props_to_json(&edge.properties),
            weight: edge.weight,
            created_at: fmt_ts(edge.created_at_ms),
            updated_at: fmt_ts(edge.updated_at_ms),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults<T> {
    pub items: Vec<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_count: Option<u64>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

impl<T> QueryResults<T> {
    fn new(items: Vec<T>, has_more: bool) -> Self {
        Self {
            items,
            total_count: None,
            has_more,
            next_token: None,
        }
    }
    fn with_next_token(mut self, token: impl Into<String>) -> Self {
        self.next_token = Some(token.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResults<T> {
    pub created_count: usize,
    pub updated_count: usize,
    pub failed_count: usize,
    pub results: Vec<T>,
    pub errors: Vec<serde_json::Value>,
}

impl<T> BatchResults<T> {
    fn new(results: Vec<T>) -> Self {
        Self::with_errors(results, Vec::new())
    }

    /// Batch outcome with per-item rejections: `failed_count`/`errors[]` carry
    /// what was skipped while `results` carries what landed (the shape this
    /// response always promised).
    fn with_errors(results: Vec<T>, errors: Vec<serde_json::Value>) -> Self {
        let created_count = results.len();
        Self {
            created_count,
            updated_count: 0,
            failed_count: errors.len(),
            results,
            errors,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResults {
    pub nodes: Vec<CanonicalNode>,
    pub edges: Vec<CanonicalEdge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paths: Option<Vec<Vec<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<RestTraversalStats>,
}

/// REST-layer traversal statistics for graph endpoints.
///
/// Naming note: this type used to be called `TraversalStats` and collided
/// with the proto `proximadb_v1::TraversalStats` wire form
/// (execution_time_microseconds: u64) and with the proximadb-query
/// `HybridTraverseStats` operator-level stats. Renamed to match the
/// existing pattern in `src/network/rest/canonical/graph.rs::RestTraversalStats`
/// (legacy root-crate REST DTO with the same field set) so the Phase 9
/// migration deletes-rather-than-renames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestTraversalStats {
    pub nodes_visited: u64,
    pub edges_traversed: u64,
    pub max_depth_reached: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathResult {
    pub path: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_weight: Option<f64>,
    pub found: bool,
}

impl ShortestPathResult {
    fn found(path: Vec<String>, total_weight: f64) -> Self {
        Self {
            path,
            total_weight: Some(total_weight),
            found: true,
        }
    }
    fn not_found() -> Self {
        Self {
            path: vec![],
            total_weight: None,
            found: false,
        }
    }
}

// ── REST input types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RestNodeInput {
    id: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    embedding: Option<RestEmbeddingInput>,
}

#[derive(Debug, Deserialize)]
struct RestEmbeddingInput {
    vector: Vec<f32>,
    #[serde(default)]
    version: String,
    #[serde(default)]
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct RestEdgeInput {
    id: String,
    from_node_id: String,
    to_node_id: String,
    edge_type: String,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    weight: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CreateNodeBody {
    node: RestNodeInput,
}

#[derive(Debug, Deserialize)]
struct CreateEdgeBody {
    edge: RestEdgeInput,
}

#[derive(Debug, Deserialize)]
struct RestTraversalRequest {
    start_node_id: String,
    #[serde(default = "default_max_depth")]
    max_depth: u32,
    #[serde(default)]
    edge_types: Vec<String>,
    #[serde(default)]
    node_labels: Vec<String>,
    #[serde(default = "default_algorithm")]
    algorithm: String,
    limit: Option<u32>,
}

fn default_max_depth() -> u32 {
    5
}
fn default_algorithm() -> String {
    "bfs".to_string()
}

#[derive(Debug, Deserialize)]
struct WalkRequest {
    start_node_id: String,
    #[serde(default = "default_walk_depth")]
    max_depth: u32,
    #[serde(default = "default_walk_limit")]
    limit: u32,
}

fn default_walk_depth() -> u32 {
    2
}
fn default_walk_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
struct WalkStepRequest {
    node_id: String,
    edge_type: Option<String>,
    #[serde(default = "default_step_limit")]
    #[allow(dead_code)]
    limit: u32,
}

fn default_step_limit() -> u32 {
    50
}

#[derive(Debug, Deserialize, Clone)]
struct RestNodeQuery {
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    #[serde(default = "default_query_limit")]
    limit: u32,
    offset: Option<u32>,
    continuation_token: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct RestEdgeQuery {
    #[serde(default)]
    edge_type: String,
    from_node_id: Option<String>,
    to_node_id: Option<String>,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    #[serde(default = "default_query_limit")]
    limit: u32,
    offset: Option<u32>,
    continuation_token: Option<String>,
}

fn default_query_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
struct RestShortestPathRequest {
    start_node_id: String,
    target_node_id: String,
    max_depth: Option<u32>,
    #[serde(default)]
    edge_types: Vec<String>,
    algorithm: Option<String>,
    k: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RestUniqueConstraintRequest {
    label: String,
    property: String,
}

#[derive(Debug, Deserialize)]
struct BatchCreateNodesRequest {
    nodes: Vec<RestNodeInput>,
}

#[derive(Debug, Deserialize)]
struct BatchCreateEdgesRequest {
    edges: Vec<RestEdgeInput>,
}

#[derive(Debug, Deserialize)]
struct RestGraphQueryRequest {
    query: String,
    // Accepted for wire compatibility (clients send `language: "cypher"`), but the
    // query type is inferred by the shared query adapter from the query text; and
    // `timeout_ms` is reserved for future per-query deadline plumbing (TD-GRAPH-2).
    #[serde(default = "default_query_language")]
    #[allow(dead_code)]
    language: String,
    #[allow(dead_code)]
    timeout_ms: Option<u32>,
}

fn default_query_language() -> String {
    "native".to_string()
}

// ── Property conversion helpers ───────────────────────────────────────────────

fn fmt_ts(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms).map_or_else(
        || "1970-01-01T00:00:00.000Z".to_string(),
        |dt: chrono::DateTime<chrono::Utc>| dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
    )
}

fn props_to_json(props: &HashMap<String, PropertyValue>) -> HashMap<String, serde_json::Value> {
    props
        .iter()
        .filter_map(|(k, v)| pv_to_json(v).map(|jv| (k.clone(), jv)))
        .collect()
}

fn pv_to_json(pv: &PropertyValue) -> Option<serde_json::Value> {
    use property_value::Value;
    pv.value.as_ref().map(|v| match v {
        Value::StringValue(s) => serde_json::Value::String(s.clone()),
        Value::IntValue(i) => serde_json::json!(*i),
        Value::DoubleValue(d) => serde_json::json!(*d),
        Value::BoolValue(b) => serde_json::json!(*b),
        Value::BytesValue(bytes) => {
            // Encode bytes as lowercase hex
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            serde_json::Value::String(hex)
        }
        Value::ArrayValue(arr) => {
            let items: Vec<serde_json::Value> = arr.values.iter().filter_map(pv_to_json).collect();
            serde_json::Value::Array(items)
        }
        Value::ObjectValue(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .fields
                .iter()
                .filter_map(|(k, v)| pv_to_json(v).map(|jv| (k.clone(), jv)))
                .collect();
            serde_json::Value::Object(map)
        }
        Value::VectorValue(vec) => serde_json::json!(vec.values),
    })
}

fn json_to_props(props: HashMap<String, serde_json::Value>) -> HashMap<String, PropertyValue> {
    props
        .into_iter()
        .filter_map(|(k, v)| json_to_pv(v).map(|pv| (k, pv)))
        .collect()
}

fn json_to_pv(v: serde_json::Value) -> Option<PropertyValue> {
    use property_value::Value;
    use proximadb_proto::v1::{PropertyArray, PropertyObject, VectorData};

    let value = match v {
        serde_json::Value::Null => return None,
        serde_json::Value::Bool(b) => Value::BoolValue(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::IntValue(i)
            } else {
                Value::DoubleValue(n.as_f64()?)
            }
        }
        serde_json::Value::String(s) => Value::StringValue(s),
        serde_json::Value::Array(arr) => {
            if arr.iter().all(|v| v.is_number()) {
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                Value::VectorValue(VectorData { values: floats })
            } else {
                let values: Vec<PropertyValue> = arr.into_iter().filter_map(json_to_pv).collect();
                Value::ArrayValue(PropertyArray { values })
            }
        }
        serde_json::Value::Object(map) => {
            let fields: HashMap<String, PropertyValue> = map
                .into_iter()
                .filter_map(|(k, v)| json_to_pv(v).map(|pv| (k, pv)))
                .collect();
            Value::ObjectValue(PropertyObject { fields })
        }
    };
    Some(PropertyValue { value: Some(value) })
}

fn rest_node_to_proto(input: RestNodeInput) -> Node {
    Node {
        id: input.id,
        labels: input.labels,
        properties: json_to_props(input.properties),
        embedding: input.embedding.map(|e| {
            let dimension = e.vector.len() as u32;
            EmbeddingVersion {
                vector: e.vector,
                model_version: e.version,
                model_id: e.model_id,
                dimension,
                created_at_ms: 0,
                model_params: HashMap::new(),
                modality: 0,
            }
        }),
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

fn rest_edge_to_proto(input: RestEdgeInput) -> Edge {
    Edge {
        id: input.id,
        from_node_id: input.from_node_id,
        to_node_id: input.to_node_id,
        edge_type: input.edge_type,
        properties: json_to_props(input.properties),
        weight: input.weight,
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

fn parse_traversal_algorithm(s: &str) -> i32 {
    match s.to_ascii_lowercase().as_str() {
        "bfs" => TraversalAlgorithm::Bfs as i32,
        "dfs" => TraversalAlgorithm::Dfs as i32,
        "parallel_bfs" | "pbfs" | "parallel" => TraversalAlgorithm::ParallelBfs as i32,
        _ => TraversalAlgorithm::Unspecified as i32,
    }
}

fn parse_sp_algorithm(s: Option<&str>) -> Option<i32> {
    match s.unwrap_or("dijkstra").to_ascii_lowercase().as_str() {
        "astar" => Some(ShortestPathAlgorithm::Astar as i32),
        "dijkstra" => Some(ShortestPathAlgorithm::Dijkstra as i32),
        _ => None,
    }
}

fn props_filter_from_map(map: HashMap<String, serde_json::Value>) -> Vec<PropertyFilter> {
    map.into_iter()
        .filter_map(|(k, v)| {
            json_to_pv(v).map(|pv| PropertyFilter {
                key: k,
                operator: PropertyFilterOperator::Equals as i32,
                value: Some(pv),
            })
        })
        .collect()
}

fn is_not_found(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("not found") || msg.contains("does not exist") || msg.contains("no such")
}

fn err_response<T: Serialize>(err: anyhow::Error) -> Response {
    if is_not_found(&err) {
        let graph_error = GraphError::new(ErrorCode::NotFound, err.to_string());
        (
            StatusCode::NOT_FOUND,
            Json(GraphResponse::<T>::error(graph_error)),
        )
            .into_response()
    } else {
        let graph_error = GraphError::internal(err.to_string());
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GraphResponse::<T>::error(graph_error)),
        )
            .into_response()
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn create_graph_router() -> Router<GraphRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            // Graph collection management
            .route("/graphs", post(create_graph_collection))
            .route("/graphs", get(list_graph_collections))
            .route("/graphs/{graph_id}", get(get_graph_collection))
            .route("/graphs/{graph_id}", delete(delete_graph_collection))
            .route("/graphs/{graph_id}/schema", put(update_graph_schema))
            // Node CRUD
            .route("/graphs/{graph_id}/nodes", post(create_node))
            .route("/graphs/{graph_id}/nodes/{id}", get(get_node))
            .route("/graphs/{graph_id}/nodes/{id}", put(update_node))
            .route("/graphs/{graph_id}/nodes/{id}", delete(delete_node))
            .route(
                "/graphs/{graph_id}/nodes/{id}/neighbors",
                get(get_node_neighbors),
            )
            // Edge CRUD
            .route("/graphs/{graph_id}/edges", post(create_edge))
            .route("/graphs/{graph_id}/edges/{id}", get(get_edge))
            .route("/graphs/{graph_id}/edges/{id}", put(update_edge))
            .route("/graphs/{graph_id}/edges/{id}", delete(delete_edge))
            // Traversal + agentic navigation
            .route("/graphs/{graph_id}/traverse", post(traverse_graph))
            .route("/graphs/{graph_id}/walk", post(walk_graph))
            .route("/graphs/{graph_id}/step", post(step_graph))
            // Shortest path
            .route("/graphs/{graph_id}/shortest-path", post(shortest_path))
            // Queries
            .route("/graphs/{graph_id}/query/nodes", post(query_nodes))
            .route("/graphs/{graph_id}/query/edges", post(query_edges))
            .route("/graphs/{graph_id}/query", post(execute_graph_query))
            // RAG (root-crate concrete dep → 501)
            .route("/graphs/{graph_id}/rag", post(not_implemented_handler))
            // Batch
            .route("/graphs/{graph_id}/nodes/batch", post(batch_create_nodes))
            .route("/graphs/{graph_id}/edges/batch", post(batch_create_edges))
            // Statistics + analysis
            .route("/graphs/{graph_id}/stats", get(get_graph_stats))
            .route(
                "/graphs/{graph_id}/components",
                get(get_connected_components),
            )
            .route("/graphs/{graph_id}/cycles", get(check_cycles))
            // Constraints DDL
            .route(
                "/graphs/{graph_id}/constraints/unique",
                post(add_unique_constraint),
            )
            .route(
                "/graphs/{graph_id}/constraints/unique",
                delete(remove_unique_constraint),
            )
            // Engine selection is ORION-only; retired engine-specific endpoints are not exposed.
            // Legacy redirects (self-contained, no port needed)
            .route("/nodes", post(create_node_legacy))
            .route("/nodes/{id}", get(get_node_legacy))
            .route("/edges", post(create_edge_legacy))
            .route("/stats", get(get_graph_stats_legacy)),
    )
}

// ── Not-implemented stub ──────────────────────────────────────────────────────

async fn not_implemented_handler() -> impl IntoResponse {
    let graph_error = GraphError::new(
        ErrorCode::InvalidArgument,
        "This endpoint is not yet available in the platform API. \
         Use the root-crate server until the relevant port trait is extracted.",
    );
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(GraphResponse::<()>::error(graph_error)),
    )
        .into_response()
}

// ── Node handlers ─────────────────────────────────────────────────────────────

async fn create_node(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(body): Json<CreateNodeBody>,
) -> impl IntoResponse {
    debug!("Creating node {} in graph {}", body.node.id, graph_id);
    let proto_node = rest_node_to_proto(body.node);
    match s
        .graph_port
        .create_node(CreateNodeRequest {
            graph_id,
            node: Some(proto_node),
        })
        .await
    {
        Ok(node) => {
            info!("Created node {}", node.id);
            (
                StatusCode::CREATED,
                Json(GraphResponse::success(CanonicalNode::from_proto(&node))),
            )
                .into_response()
        }
        Err(e) => {
            error!("Failed to create node: {e}");
            err_response::<CanonicalNode>(e)
        }
    }
}

async fn get_node(
    TenantGraph(s): TenantGraph,
    Path((graph_id, node_id)): Path<(String, String)>,
) -> impl IntoResponse {
    debug!("Getting node {} from graph {}", node_id, graph_id);
    match s
        .graph_port
        .get_node(GetNodeRequest {
            graph_id,
            node_id: node_id.clone(),
        })
        .await
    {
        Ok(node) => Json(GraphResponse::success(CanonicalNode::from_proto(&node))).into_response(),
        Err(e) => {
            if is_not_found(&e) {
                warn!("Node not found: {node_id}");
                let graph_error = GraphError::not_found("Node", &node_id);
                (
                    StatusCode::NOT_FOUND,
                    Json(GraphResponse::<CanonicalNode>::error(graph_error)),
                )
                    .into_response()
            } else {
                error!("Failed to get node {node_id}: {e}");
                err_response::<CanonicalNode>(e)
            }
        }
    }
}

async fn update_node(
    TenantGraph(s): TenantGraph,
    Path((graph_id, node_id)): Path<(String, String)>,
    Json(mut input): Json<RestNodeInput>,
) -> impl IntoResponse {
    debug!("Updating node {} in graph {}", node_id, graph_id);
    input.id = node_id;
    let proto_node = rest_node_to_proto(input);
    match s
        .graph_port
        .update_node(UpdateNodeRequest {
            graph_id,
            node: Some(proto_node),
        })
        .await
    {
        Ok(node) => Json(GraphResponse::success(CanonicalNode::from_proto(&node))).into_response(),
        Err(e) => {
            error!("Failed to update node: {e}");
            err_response::<CanonicalNode>(e)
        }
    }
}

async fn delete_node(
    TenantGraph(s): TenantGraph,
    Path((graph_id, node_id)): Path<(String, String)>,
) -> impl IntoResponse {
    debug!("Deleting node {} from graph {}", node_id, graph_id);
    match s
        .graph_port
        .delete_node(DeleteNodeRequest {
            graph_id,
            node_id: node_id.clone(),
        })
        .await
    {
        Ok(node) => Json(GraphResponse::success(CanonicalNode::from_proto(&node))).into_response(),
        Err(e) => {
            if is_not_found(&e) {
                warn!("Node not found for deletion: {node_id}");
                let graph_error = GraphError::not_found("Node", &node_id);
                (
                    StatusCode::NOT_FOUND,
                    Json(GraphResponse::<CanonicalNode>::error(graph_error)),
                )
                    .into_response()
            } else {
                error!("Failed to delete node {node_id}: {e}");
                err_response::<CanonicalNode>(e)
            }
        }
    }
}

async fn get_node_neighbors(
    TenantGraph(s): TenantGraph,
    Path((graph_id, node_id)): Path<(String, String)>,
) -> impl IntoResponse {
    debug!(
        "Getting neighbors for node {} in graph {}",
        node_id, graph_id
    );
    match s
        .graph_port
        .get_neighbors(GetNeighborsRequest {
            graph_id,
            node_id: node_id.clone(),
            edge_type: None,
        })
        .await
    {
        Ok(batch) => {
            let canonical: Vec<CanonicalNode> =
                batch.nodes.iter().map(CanonicalNode::from_proto).collect();
            Json(GraphResponse::success(canonical)).into_response()
        }
        Err(e) => {
            error!("Failed to get neighbors for {node_id}: {e}");
            err_response::<Vec<CanonicalNode>>(e)
        }
    }
}

// ── Edge handlers ─────────────────────────────────────────────────────────────

async fn create_edge(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(body): Json<CreateEdgeBody>,
) -> impl IntoResponse {
    debug!("Creating edge {} in graph {}", body.edge.id, graph_id);
    let proto_edge = rest_edge_to_proto(body.edge);
    match s
        .graph_port
        .create_edge(CreateEdgeRequest {
            graph_id,
            edge: Some(proto_edge),
        })
        .await
    {
        Ok(edge) => {
            info!("Created edge {}", edge.id);
            (
                StatusCode::CREATED,
                Json(GraphResponse::success(CanonicalEdge::from_proto(&edge))),
            )
                .into_response()
        }
        Err(e) => {
            error!("Failed to create edge: {e}");
            err_response::<CanonicalEdge>(e)
        }
    }
}

async fn get_edge(
    TenantGraph(s): TenantGraph,
    Path((graph_id, edge_id)): Path<(String, String)>,
) -> impl IntoResponse {
    debug!("Getting edge {} from graph {}", edge_id, graph_id);
    match s
        .graph_port
        .get_edge(GetEdgeRequest {
            graph_id,
            edge_id: edge_id.clone(),
        })
        .await
    {
        Ok(edge) => Json(GraphResponse::success(CanonicalEdge::from_proto(&edge))).into_response(),
        Err(e) => {
            if is_not_found(&e) {
                warn!("Edge not found: {edge_id}");
                let graph_error = GraphError::not_found("Edge", &edge_id);
                (
                    StatusCode::NOT_FOUND,
                    Json(GraphResponse::<CanonicalEdge>::error(graph_error)),
                )
                    .into_response()
            } else {
                error!("Failed to get edge {edge_id}: {e}");
                err_response::<CanonicalEdge>(e)
            }
        }
    }
}

async fn update_edge(
    TenantGraph(s): TenantGraph,
    Path((graph_id, edge_id)): Path<(String, String)>,
    Json(mut input): Json<RestEdgeInput>,
) -> impl IntoResponse {
    debug!("Updating edge {} in graph {}", edge_id, graph_id);
    input.id = edge_id;
    let proto_edge = rest_edge_to_proto(input);
    match s
        .graph_port
        .update_edge(UpdateEdgeRequest {
            graph_id,
            edge: Some(proto_edge),
        })
        .await
    {
        Ok(edge) => Json(GraphResponse::success(CanonicalEdge::from_proto(&edge))).into_response(),
        Err(e) => {
            error!("Failed to update edge: {e}");
            err_response::<CanonicalEdge>(e)
        }
    }
}

async fn delete_edge(
    TenantGraph(s): TenantGraph,
    Path((graph_id, edge_id)): Path<(String, String)>,
) -> impl IntoResponse {
    debug!("Deleting edge {} from graph {}", edge_id, graph_id);
    match s
        .graph_port
        .delete_edge(DeleteEdgeRequest {
            graph_id,
            edge_id: edge_id.clone(),
        })
        .await
    {
        Ok(edge) => Json(GraphResponse::success(CanonicalEdge::from_proto(&edge))).into_response(),
        Err(e) => {
            if is_not_found(&e) {
                warn!("Edge not found for deletion: {edge_id}");
                let graph_error = GraphError::not_found("Edge", &edge_id);
                (
                    StatusCode::NOT_FOUND,
                    Json(GraphResponse::<CanonicalEdge>::error(graph_error)),
                )
                    .into_response()
            } else {
                error!("Failed to delete edge {edge_id}: {e}");
                err_response::<CanonicalEdge>(e)
            }
        }
    }
}

// ── Query handlers ────────────────────────────────────────────────────────────

async fn query_nodes(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(mut q): Json<RestNodeQuery>,
) -> impl IntoResponse {
    // Decode continuation token as "offset:<n>"
    if q.offset.is_none()
        && let Some(token) = &q.continuation_token
        && let Some(rest) = token.strip_prefix("offset:")
        && let Ok(n) = rest.parse::<u32>()
    {
        q.offset = Some(n);
    }

    let proto_query = NodeQuery {
        graph_id,
        labels: q.labels.clone(),
        filters: props_filter_from_map(q.properties.clone()),
        limit: Some(q.limit),
        offset: q.offset,
        continuation_token: q.continuation_token.clone(),
    };

    match s.graph_port.query_nodes(proto_query).await {
        Ok(batch) => {
            let lim = q.limit;
            let has_more = (batch.nodes.len() as u32) == lim;
            let canonical: Vec<CanonicalNode> =
                batch.nodes.iter().map(CanonicalNode::from_proto).collect();
            let mut results = QueryResults::new(canonical, has_more);
            if has_more {
                let next_off = q.offset.unwrap_or(0).saturating_add(lim);
                results = results.with_next_token(format!("offset:{next_off}"));
            }
            Json(GraphResponse::success(results)).into_response()
        }
        Err(e) => {
            error!("Failed to query nodes: {e}");
            err_response::<QueryResults<CanonicalNode>>(e)
        }
    }
}

async fn query_edges(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(mut q): Json<RestEdgeQuery>,
) -> impl IntoResponse {
    if q.offset.is_none()
        && let Some(token) = &q.continuation_token
        && let Some(rest) = token.strip_prefix("offset:")
        && let Ok(n) = rest.parse::<u32>()
    {
        q.offset = Some(n);
    }

    let proto_query = EdgeQuery {
        graph_id,
        from_node_id: q.from_node_id.clone(),
        to_node_id: q.to_node_id.clone(),
        edge_types: if q.edge_type.is_empty() {
            vec![]
        } else {
            vec![q.edge_type.clone()]
        },
        filters: props_filter_from_map(q.properties.clone()),
        limit: Some(q.limit),
        offset: q.offset,
        continuation_token: q.continuation_token.clone(),
    };

    match s.graph_port.query_edges(proto_query).await {
        Ok(batch) => {
            let lim = q.limit;
            let has_more = (batch.edges.len() as u32) == lim;
            let canonical: Vec<CanonicalEdge> =
                batch.edges.iter().map(CanonicalEdge::from_proto).collect();
            let mut results = QueryResults::new(canonical, has_more);
            if has_more {
                let next_off = q.offset.unwrap_or(0).saturating_add(lim);
                results = results.with_next_token(format!("offset:{next_off}"));
            }
            Json(GraphResponse::success(results)).into_response()
        }
        Err(e) => {
            error!("Failed to query edges: {e}");
            err_response::<QueryResults<CanonicalEdge>>(e)
        }
    }
}

async fn execute_graph_query(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<RestGraphQueryRequest>,
) -> impl IntoResponse {
    // v2-era path (TD-GRAPH-2): the engine yields result rows as `serde_json::Value`
    // directly, so this REST surface bypasses the legacy v1 `QueryValue` /
    // `GraphQueryResponse` wrapping entirely. The declarative language (cypher) is
    // interpreted by the shared query adapter from the query text.
    match s.graph_port.execute_query_json(&graph_id, &req.query).await {
        Ok(rows) => {
            #[derive(Serialize)]
            struct QueryResult {
                rows: Vec<serde_json::Value>,
                row_count: u64,
            }
            let result = QueryResult {
                row_count: rows.len() as u64,
                rows,
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("Graph query failed: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

// ── Traversal handlers ────────────────────────────────────────────────────────

async fn traverse_graph(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<RestTraversalRequest>,
) -> impl IntoResponse {
    debug!("Traversal from {} in graph {}", req.start_node_id, graph_id);

    let proto_req = TraversalRequest {
        graph_id,
        start_node_id: req.start_node_id,
        max_depth: req.max_depth,
        edge_types: req.edge_types,
        node_labels: req.node_labels,
        filters: vec![],
        algorithm: parse_traversal_algorithm(&req.algorithm),
        limit: req.limit,
        timeout_ms: None,
        max_frontier: None,
    };

    match s.graph_port.traverse_graph(proto_req).await {
        Ok(resp) => {
            let canonical_nodes: Vec<CanonicalNode> =
                resp.nodes.iter().map(CanonicalNode::from_proto).collect();
            let canonical_edges: Vec<CanonicalEdge> =
                resp.edges.iter().map(CanonicalEdge::from_proto).collect();
            let paths: Option<Vec<Vec<String>>> = if resp.paths.is_empty() {
                None
            } else {
                Some(
                    resp.paths
                        .iter()
                        .map(|p| p.entities.iter().map(|e| e.id.clone()).collect())
                        .collect(),
                )
            };
            let stats = resp.stats.as_ref().map(|st| RestTraversalStats {
                nodes_visited: st.nodes_visited as u64,
                edges_traversed: st.edges_traversed as u64,
                max_depth_reached: st.max_depth_reached,
                execution_time_ms: Some(st.execution_time_microseconds / 1000),
            });
            let result = TraversalResults {
                nodes: canonical_nodes,
                edges: canonical_edges,
                paths,
                stats,
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("Graph traversal failed: {e}");
            err_response::<TraversalResults>(e)
        }
    }
}

/// BFS walk up to `limit` nodes within `max_depth` hops — implements `walk_graph`
/// via `GraphPort::traverse_graph` with BFS algorithm.
async fn walk_graph(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<WalkRequest>,
) -> impl IntoResponse {
    debug!(
        "GraphWalk: graph={} start={} max_depth={} limit={}",
        graph_id, req.start_node_id, req.max_depth, req.limit
    );

    let proto_req = TraversalRequest {
        graph_id,
        start_node_id: req.start_node_id,
        max_depth: req.max_depth,
        edge_types: vec![],
        node_labels: vec![],
        filters: vec![],
        algorithm: TraversalAlgorithm::Bfs as i32,
        limit: Some(req.limit),
        timeout_ms: None,
        max_frontier: None,
    };

    match s.graph_port.traverse_graph(proto_req).await {
        Ok(resp) => {
            let canonical_nodes: Vec<CanonicalNode> =
                resp.nodes.iter().map(CanonicalNode::from_proto).collect();
            let canonical_edges: Vec<CanonicalEdge> =
                resp.edges.iter().map(CanonicalEdge::from_proto).collect();
            let result = TraversalResults {
                nodes: canonical_nodes,
                edges: canonical_edges,
                paths: None,
                stats: None,
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("GraphWalk failed for graph: {e}");
            err_response::<TraversalResults>(e)
        }
    }
}

/// Single-step navigation — returns the neighbors of one node via
/// `GraphPort::get_neighbors`. Implements `step_graph`.
async fn step_graph(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<WalkStepRequest>,
) -> impl IntoResponse {
    debug!(
        "GraphStep: graph={} node={} edge_type={:?}",
        graph_id, req.node_id, req.edge_type
    );

    match s
        .graph_port
        .get_neighbors(GetNeighborsRequest {
            graph_id,
            node_id: req.node_id.clone(),
            edge_type: req.edge_type.clone(),
        })
        .await
    {
        Ok(batch) => {
            let canonical_nodes: Vec<CanonicalNode> =
                batch.nodes.iter().map(CanonicalNode::from_proto).collect();
            let result = TraversalResults {
                nodes: canonical_nodes,
                edges: vec![],
                paths: None,
                stats: None,
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("GraphStep failed for node {}: {e}", req.node_id);
            err_response::<TraversalResults>(e)
        }
    }
}

// ── Analytics handlers ────────────────────────────────────────────────────────

async fn get_graph_stats(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
) -> impl IntoResponse {
    debug!("Getting graph stats for {}", graph_id);
    match s
        .graph_port
        .get_graph_stats(GetStatsRequest { graph_id })
        .await
    {
        Ok(stats) => {
            #[derive(Serialize)]
            struct GraphStatsJson {
                total_nodes: u64,
                total_edges: u64,
                label_stats: Vec<serde_json::Value>,
                edge_type_stats: Vec<serde_json::Value>,
                total_properties: u64,
                memory_usage_bytes: u64,
                average_degree: f64,
                max_degree: u32,
                connected_components: u32,
            }
            let label_stats: Vec<serde_json::Value> = stats
                .label_stats
                .iter()
                .map(|ls| serde_json::json!({ "label": ls.label, "count": ls.count }))
                .collect();
            let edge_type_stats: Vec<serde_json::Value> = stats
                .edge_type_stats
                .iter()
                .map(|es| serde_json::json!({ "edge_type": es.edge_type, "count": es.count }))
                .collect();
            let result = GraphStatsJson {
                total_nodes: stats.total_nodes,
                total_edges: stats.total_edges,
                label_stats,
                edge_type_stats,
                total_properties: stats.total_properties,
                memory_usage_bytes: stats.memory_usage_bytes,
                average_degree: stats.average_degree,
                max_degree: stats.max_degree,
                connected_components: stats.connected_components,
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("Failed to get graph stats: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

async fn shortest_path(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    _headers: HeaderMap,
    Json(req): Json<RestShortestPathRequest>,
) -> impl IntoResponse {
    use proximadb_proto::v1::ShortestPathRequest as ProtoSpRequest;

    let algorithm = parse_sp_algorithm(req.algorithm.as_deref());
    let proto_req = ProtoSpRequest {
        graph_id,
        start_node_id: req.start_node_id,
        target_node_id: req.target_node_id,
        max_depth: req.max_depth,
        edge_types: req.edge_types,
        algorithm,
        k: req.k,
    };

    match s.graph_port.shortest_path(proto_req).await {
        Ok(resp) => {
            let result = if resp.node_ids.is_empty() {
                ShortestPathResult::not_found()
            } else {
                ShortestPathResult::found(resp.node_ids, resp.total_weight.unwrap_or(0.0))
            };
            Json(GraphResponse::success(result)).into_response()
        }
        Err(e) => {
            error!("Shortest path failed: {e}");
            err_response::<ShortestPathResult>(e)
        }
    }
}

async fn get_connected_components(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
) -> impl IntoResponse {
    match s
        .graph_port
        .get_connected_components(GetStatsRequest { graph_id })
        .await
    {
        Ok(resp) => {
            let components: Vec<Vec<String>> =
                resp.components.into_iter().map(|c| c.node_ids).collect();
            #[derive(Serialize)]
            struct ComponentsData {
                components: Vec<Vec<String>>,
            }
            Json(GraphResponse::success(ComponentsData { components })).into_response()
        }
        Err(e) => {
            error!("Failed to get connected components: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

async fn check_cycles(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
) -> impl IntoResponse {
    match s.graph_port.has_cycle(GetStatsRequest { graph_id }).await {
        Ok(resp) => {
            #[derive(Serialize)]
            struct CycleData {
                has_cycle: bool,
            }
            Json(GraphResponse::success(CycleData {
                has_cycle: resp.has_cycle,
            }))
            .into_response()
        }
        Err(e) => {
            error!("Failed to check cycles: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

// ── Constraint handlers ───────────────────────────────────────────────────────

async fn add_unique_constraint(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<RestUniqueConstraintRequest>,
) -> impl IntoResponse {
    match s
        .graph_port
        .add_unique_constraint(UniqueConstraintRequest {
            graph_id,
            label: req.label,
            property: req.property,
        })
        .await
    {
        Ok(resp) => {
            #[derive(Serialize)]
            struct DdlResult {
                success: bool,
            }
            Json(GraphResponse::success(DdlResult {
                success: resp.success,
            }))
            .into_response()
        }
        Err(e) => {
            error!("Failed to add unique constraint: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

async fn remove_unique_constraint(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<RestUniqueConstraintRequest>,
) -> impl IntoResponse {
    match s
        .graph_port
        .remove_unique_constraint(UniqueConstraintRequest {
            graph_id,
            label: req.label,
            property: req.property,
        })
        .await
    {
        Ok(resp) => {
            #[derive(Serialize)]
            struct DdlResult {
                success: bool,
            }
            Json(GraphResponse::success(DdlResult {
                success: resp.success,
            }))
            .into_response()
        }
        Err(e) => {
            error!("Failed to remove unique constraint: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

// ── Batch handlers ────────────────────────────────────────────────────────────

async fn batch_create_nodes(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<BatchCreateNodesRequest>,
) -> impl IntoResponse {
    debug!(
        "Batch creating {} nodes in graph {}",
        req.nodes.len(),
        graph_id
    );
    let nodes: Vec<Node> = req.nodes.into_iter().map(rest_node_to_proto).collect();
    match s
        .graph_port
        .batch_create_nodes(BatchNodeRequest { graph_id, nodes })
        .await
    {
        Ok(batch) => {
            let canonical: Vec<CanonicalNode> =
                batch.nodes.iter().map(CanonicalNode::from_proto).collect();
            info!("Batch created {} nodes", canonical.len());
            Json(GraphResponse::success(BatchResults::new(canonical))).into_response()
        }
        Err(e) => {
            error!("Batch create nodes failed: {e}");
            err_response::<BatchResults<CanonicalNode>>(e)
        }
    }
}

async fn batch_create_edges(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(req): Json<BatchCreateEdgesRequest>,
) -> impl IntoResponse {
    debug!(
        "Batch creating {} edges in graph {}",
        req.edges.len(),
        graph_id
    );
    let edges: Vec<Edge> = req.edges.into_iter().map(rest_edge_to_proto).collect();
    match s
        .graph_port
        .batch_create_edges(BatchEdgeRequest { graph_id, edges })
        .await
    {
        Ok(batch) => {
            let canonical: Vec<CanonicalEdge> =
                batch.edges.iter().map(CanonicalEdge::from_proto).collect();
            let errors: Vec<serde_json::Value> = batch
                .errors
                .iter()
                .map(|e| serde_json::json!({"id": e.id, "error": e.message}))
                .collect();
            info!(
                "Batch created {} edges ({} rejected)",
                canonical.len(),
                errors.len()
            );
            Json(GraphResponse::success(BatchResults::with_errors(
                canonical, errors,
            )))
            .into_response()
        }
        Err(e) => {
            error!("Batch create edges failed: {e}");
            err_response::<BatchResults<CanonicalEdge>>(e)
        }
    }
}

// ── Graph collection management handlers ─────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CreateGraphCollectionBody {
    graph_id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateGraphSchemaBody {
    schema: serde_json::Value,
}

async fn create_graph_collection(
    TenantGraph(s): TenantGraph,
    Json(body): Json<CreateGraphCollectionBody>,
) -> impl IntoResponse {
    info!("Creating graph collection: {}", body.graph_id);
    let request = CreateGraphRequest {
        graph_id: body.graph_id,
        name: body.name,
        description: body.description,
        schema: None,
        storage_config: None,
        engine_config: None,
        access_control: None,
    };
    match s.graph_port.create_graph_collection(request).await {
        Ok(col) => {
            let data = serde_json::to_value(&col).unwrap_or_default();
            (StatusCode::CREATED, Json(GraphResponse::success(data))).into_response()
        }
        Err(e) => {
            error!("Failed to create graph collection: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

async fn list_graph_collections(TenantGraph(s): TenantGraph) -> impl IntoResponse {
    debug!("Listing graph collections");
    match s.graph_port.list_graph_collections().await {
        Ok(cols) => {
            let data: Vec<serde_json::Value> = cols
                .iter()
                .map(|c| serde_json::to_value(c).unwrap_or_default())
                .collect();
            Json(GraphResponse::success(data)).into_response()
        }
        Err(e) => {
            error!("Failed to list graph collections: {e}");
            err_response::<Vec<serde_json::Value>>(e)
        }
    }
}

async fn get_graph_collection(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
) -> impl IntoResponse {
    debug!("Getting graph collection: {graph_id}");
    match s.graph_port.get_graph_collection(graph_id.clone()).await {
        Ok(Some(col)) => {
            let data = serde_json::to_value(&col).unwrap_or_default();
            Json(GraphResponse::success(data)).into_response()
        }
        Ok(None) => {
            let err_resp = GraphResponse::<serde_json::Value>::from_error(
                ErrorCode::NotFound,
                format!("Graph collection '{graph_id}' not found"),
            );
            (StatusCode::NOT_FOUND, Json(err_resp)).into_response()
        }
        Err(e) => {
            error!("Failed to get graph collection {graph_id}: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

async fn delete_graph_collection(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
) -> impl IntoResponse {
    info!("Deleting graph collection: {graph_id}");
    match s.graph_port.delete_graph_collection(graph_id.clone()).await {
        Ok(true) => (StatusCode::NO_CONTENT, Json(GraphResponse::success(()))).into_response(),
        Ok(false) => {
            let err_resp = GraphResponse::<()>::from_error(
                ErrorCode::NotFound,
                format!("Graph collection '{graph_id}' not found"),
            );
            (StatusCode::NOT_FOUND, Json(err_resp)).into_response()
        }
        Err(e) => {
            error!("Failed to delete graph collection {graph_id}: {e}");
            err_response::<()>(e)
        }
    }
}

async fn update_graph_schema(
    TenantGraph(s): TenantGraph,
    Path(graph_id): Path<String>,
    Json(body): Json<UpdateGraphSchemaBody>,
) -> impl IntoResponse {
    info!("Updating schema for graph collection: {graph_id}");
    let schema: GraphSchema = match serde_json::from_value(body.schema) {
        Ok(s) => s,
        Err(e) => {
            let err_resp = GraphResponse::<serde_json::Value>::from_error(
                ErrorCode::InvalidArgument,
                format!("Invalid schema JSON: {e}"),
            );
            return (StatusCode::BAD_REQUEST, Json(err_resp)).into_response();
        }
    };
    match s
        .graph_port
        .update_graph_schema(graph_id.clone(), schema)
        .await
    {
        Ok(col) => {
            let data = serde_json::to_value(&col).unwrap_or_default();
            Json(GraphResponse::success(data)).into_response()
        }
        Err(e) if is_not_found(&e) => {
            let err_resp = GraphResponse::<serde_json::Value>::from_error(
                ErrorCode::NotFound,
                format!("Graph collection '{graph_id}' not found"),
            );
            (StatusCode::NOT_FOUND, Json(err_resp)).into_response()
        }
        Err(e) => {
            error!("Failed to update schema for {graph_id}: {e}");
            err_response::<serde_json::Value>(e)
        }
    }
}

// ── Legacy redirect handlers ──────────────────────────────────────────────────

const DEFAULT_GRAPH_ID: &str = "default";
const LEGACY_GRAPH_SUNSET_DATE: &str = "2026-06-30";

/// Legacy compatibility routes (`/api/v1/graph/nodes`, `/api/v1/graph/edges`, etc.)
/// return `308 Permanent Redirect` to their canonical multi-graph equivalents,
/// carrying `Deprecation` and `Sunset` metadata. Sunset date: `2026-06-30`.
///
/// This contract is drift-guarded by `scripts/validate_capability_matrix.py`
/// (capability `graph_legacy_route_deprecation`). The guard previously watched an
/// unmounted duplicate of this module under `src/`; it now watches the
/// implementation that is actually served.
fn legacy_redirect(canonical_path: impl Into<String>) -> Response {
    let canonical_path = canonical_path.into();
    warn!(
        canonical_route = %canonical_path,
        sunset = LEGACY_GRAPH_SUNSET_DATE,
        "Legacy graph endpoint deprecated; redirecting"
    );

    let mut response = StatusCode::PERMANENT_REDIRECT.into_response();
    if let Ok(loc) = HeaderValue::from_str(&canonical_path) {
        response.headers_mut().insert(header::LOCATION, loc.clone());
        response.headers_mut().insert(
            header::HeaderName::from_static("x-proximadb-canonical-route"),
            loc,
        );
    }
    response.headers_mut().insert(
        header::HeaderName::from_static("deprecation"),
        HeaderValue::from_static("true"),
    );
    response.headers_mut().insert(
        header::HeaderName::from_static("sunset"),
        HeaderValue::from_static(LEGACY_GRAPH_SUNSET_DATE),
    );
    response
}

async fn create_node_legacy() -> impl IntoResponse {
    legacy_redirect(format!("/api/v2/graphs/{DEFAULT_GRAPH_ID}/nodes"))
}

async fn get_node_legacy(Path(node_id): Path<String>) -> impl IntoResponse {
    legacy_redirect(format!("/api/v2/graphs/{DEFAULT_GRAPH_ID}/nodes/{node_id}"))
}

async fn create_edge_legacy() -> impl IntoResponse {
    legacy_redirect(format!("/api/v2/graphs/{DEFAULT_GRAPH_ID}/edges"))
}

async fn get_graph_stats_legacy() -> impl IntoResponse {
    legacy_redirect(format!("/api/v2/graphs/{DEFAULT_GRAPH_ID}/stats"))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use async_trait::async_trait;
    use proximadb_proto::v1 as proto;

    #[derive(Clone, Copy)]
    enum MockMode {
        Ok,
        NotFound,
        Internal,
    }

    struct MockGraphPort {
        mode: MockMode,
    }

    impl MockGraphPort {
        fn state(mode: MockMode) -> TenantGraph {
            TenantGraph::for_tenant(
                &GraphRestState {
                    graph_port: Arc::new(Self { mode }),
                },
                proximadb_tenant::DEFAULT_TENANT,
            )
        }

        fn ok_or_fail<T>(&self, value: T) -> Result<T> {
            match self.mode {
                MockMode::Ok => Ok(value),
                MockMode::NotFound => Err(anyhow!("graph item not found")),
                MockMode::Internal => Err(anyhow!("graph backend failed")),
            }
        }
    }

    fn sample_node(id: &str) -> Node {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            PropertyValue {
                value: Some(property_value::Value::StringValue("Alice".to_string())),
            },
        );
        Node {
            id: id.to_string(),
            labels: vec!["Person".to_string()],
            properties,
            embedding: Some(EmbeddingVersion {
                model_id: "text-embedding".to_string(),
                model_version: "v1".to_string(),
                vector: vec![0.1, 0.2],
                dimension: 2,
                created_at_ms: 0,
                model_params: HashMap::new(),
                modality: 0,
            }),
            created_at_ms: 0,
            updated_at_ms: 1_000,
        }
    }

    fn sample_edge(id: &str) -> Edge {
        let mut properties = HashMap::new();
        properties.insert(
            "since".to_string(),
            PropertyValue {
                value: Some(property_value::Value::IntValue(2024)),
            },
        );
        Edge {
            id: id.to_string(),
            from_node_id: "n1".to_string(),
            to_node_id: "n2".to_string(),
            edge_type: "KNOWS".to_string(),
            properties,
            weight: Some(0.7),
            created_at_ms: 0,
            updated_at_ms: 1_000,
        }
    }

    fn batch_response() -> proto::BatchResponse {
        proto::BatchResponse {
            success: true,
            nodes: vec![sample_node("n1")],
            edges: vec![sample_edge("e1")],
            error_message: None,
            next_token: Some("offset:2".to_string()),
            created_count: Some(1),
            updated_count: Some(0),
            failed_count: Some(0),
            failed_ids: Vec::new(),
            error_messages: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn sample_graph_collection(graph_id: &str) -> proto::GraphCollection {
        proto::GraphCollection {
            graph_id: graph_id.to_string(),
            name: "Knowledge Graph".to_string(),
            description: "test graph".to_string(),
            schema: Some(GraphSchema::default()),
            storage_config: None,
            engine_config: None,
            access_control: None,
            stats: None,
            created_at: 0,
            updated_at: 1_000,
        }
    }

    #[async_trait]
    impl GraphPort for MockGraphPort {
        async fn create_node(&self, request: CreateNodeRequest) -> Result<Node> {
            self.ok_or_fail(request.node.unwrap_or_else(|| sample_node("n1")))
        }

        async fn get_node(&self, _request: GetNodeRequest) -> Result<Node> {
            self.ok_or_fail(sample_node("n1"))
        }

        async fn update_node(&self, request: UpdateNodeRequest) -> Result<Node> {
            self.ok_or_fail(request.node.unwrap_or_else(|| sample_node("n1")))
        }

        async fn delete_node(&self, _request: DeleteNodeRequest) -> Result<Node> {
            self.ok_or_fail(sample_node("n1"))
        }

        async fn create_edge(&self, request: CreateEdgeRequest) -> Result<Edge> {
            self.ok_or_fail(request.edge.unwrap_or_else(|| sample_edge("e1")))
        }

        async fn get_edge(&self, _request: GetEdgeRequest) -> Result<Edge> {
            self.ok_or_fail(sample_edge("e1"))
        }

        async fn update_edge(&self, request: UpdateEdgeRequest) -> Result<Edge> {
            self.ok_or_fail(request.edge.unwrap_or_else(|| sample_edge("e1")))
        }

        async fn delete_edge(&self, _request: DeleteEdgeRequest) -> Result<Edge> {
            self.ok_or_fail(sample_edge("e1"))
        }

        async fn query_nodes(&self, _request: NodeQuery) -> Result<proto::BatchResponse> {
            self.ok_or_fail(batch_response())
        }

        async fn query_edges(&self, _request: EdgeQuery) -> Result<proto::BatchResponse> {
            self.ok_or_fail(batch_response())
        }

        async fn execute_query(
            &self,
            _request: proto::GraphQueryRequest,
        ) -> Result<proto::GraphQueryResponse> {
            let mut columns = HashMap::new();
            columns.insert("n".to_string(), proto::QueryValue::default());
            self.ok_or_fail(proto::GraphQueryResponse {
                rows: vec![proto::ResultRow { columns }],
                stats: None,
                query_plan: None,
                error_message: None,
            })
        }
        async fn execute_query_json(
            &self,
            _graph_id: &str,
            _query: &str,
        ) -> Result<Vec<serde_json::Value>> {
            self.ok_or_fail(vec![serde_json::json!({"n": {"id": "mock"}})])
        }

        async fn get_neighbors(
            &self,
            _request: GetNeighborsRequest,
        ) -> Result<proto::BatchResponse> {
            self.ok_or_fail(batch_response())
        }

        async fn traverse_graph(
            &self,
            _request: TraversalRequest,
        ) -> Result<proto::TraversalResponse> {
            self.ok_or_fail(proto::TraversalResponse {
                nodes: vec![sample_node("n1"), sample_node("n2")],
                edges: vec![sample_edge("e1")],
                paths: vec![proto::GraphPath {
                    entities: vec![
                        proto::Entity {
                            id: "n1".to_string(),
                            ..proto::Entity::default()
                        },
                        proto::Entity {
                            id: "n2".to_string(),
                            ..proto::Entity::default()
                        },
                    ],
                    relations: Vec::new(),
                }],
                stats: Some(proto::TraversalStats {
                    nodes_visited: 2,
                    edges_traversed: 1,
                    max_depth_reached: 1,
                    execution_time_microseconds: 4_000,
                }),
            })
        }

        async fn stream_traverse(
            &self,
            _request: TraversalRequest,
        ) -> Result<Vec<proto::TraversalChunk>> {
            self.ok_or_fail(Vec::new())
        }

        async fn get_graph_stats(&self, _request: GetStatsRequest) -> Result<proto::GraphStats> {
            self.ok_or_fail(proto::GraphStats {
                total_nodes: 2,
                total_edges: 1,
                label_stats: vec![proto::LabelStats {
                    label: "Person".to_string(),
                    count: 2,
                }],
                edge_type_stats: vec![proto::EdgeTypeStats {
                    edge_type: "KNOWS".to_string(),
                    count: 1,
                }],
                total_properties: 3,
                memory_usage_bytes: 1024,
                average_degree: 1.0,
                max_degree: 2,
                connected_components: 1,
            })
        }

        async fn shortest_path(
            &self,
            _request: proto::ShortestPathRequest,
        ) -> Result<proto::ShortestPathResponse> {
            self.ok_or_fail(proto::ShortestPathResponse {
                node_ids: vec!["n1".to_string(), "n2".to_string()],
                total_weight: Some(0.7),
            })
        }

        async fn get_connected_components(
            &self,
            _request: GetStatsRequest,
        ) -> Result<proto::ConnectedComponentsResponse> {
            self.ok_or_fail(proto::ConnectedComponentsResponse {
                components: vec![proto::Component {
                    node_ids: vec!["n1".to_string(), "n2".to_string()],
                }],
            })
        }

        async fn has_cycle(&self, _request: GetStatsRequest) -> Result<proto::CycleCheckResponse> {
            self.ok_or_fail(proto::CycleCheckResponse { has_cycle: true })
        }

        async fn add_unique_constraint(
            &self,
            _request: UniqueConstraintRequest,
        ) -> Result<proto::UniqueConstraintResponse> {
            self.ok_or_fail(proto::UniqueConstraintResponse {
                success: true,
                error_message: None,
            })
        }

        async fn remove_unique_constraint(
            &self,
            _request: UniqueConstraintRequest,
        ) -> Result<proto::UniqueConstraintResponse> {
            self.ok_or_fail(proto::UniqueConstraintResponse {
                success: true,
                error_message: None,
            })
        }

        async fn batch_create_nodes(
            &self,
            _request: BatchNodeRequest,
        ) -> Result<proto::BatchResponse> {
            self.ok_or_fail(batch_response())
        }

        async fn batch_create_edges(
            &self,
            _request: BatchEdgeRequest,
        ) -> Result<proto::BatchResponse> {
            self.ok_or_fail(batch_response())
        }

        async fn execute_hybrid_query(
            &self,
            _request: proto::HybridSearchRequest,
        ) -> Result<proto::HybridSearchResponse> {
            self.ok_or_fail(proto::HybridSearchResponse::default())
        }

        async fn create_graph_collection(
            &self,
            request: CreateGraphRequest,
        ) -> Result<proto::GraphCollection> {
            self.ok_or_fail(sample_graph_collection(&request.graph_id))
        }

        async fn get_graph_collection(
            &self,
            graph_id: String,
        ) -> Result<Option<proto::GraphCollection>> {
            match self.mode {
                MockMode::Ok => Ok(Some(sample_graph_collection(&graph_id))),
                MockMode::NotFound => Ok(None),
                MockMode::Internal => Err(anyhow!("graph backend failed")),
            }
        }

        async fn delete_graph_collection(&self, _graph_id: String) -> Result<bool> {
            match self.mode {
                MockMode::Ok => Ok(true),
                MockMode::NotFound => Ok(false),
                MockMode::Internal => Err(anyhow!("graph backend failed")),
            }
        }

        async fn list_graph_collections(&self) -> Result<Vec<proto::GraphCollection>> {
            self.ok_or_fail(vec![sample_graph_collection("g1")])
        }

        async fn update_graph_schema(
            &self,
            graph_id: String,
            schema: GraphSchema,
        ) -> Result<proto::GraphCollection> {
            let mut collection = sample_graph_collection(&graph_id);
            collection.schema = Some(schema);
            self.ok_or_fail(collection)
        }
    }

    #[test]
    fn test_graph_response_success() {
        let r = GraphResponse::success("hello");
        assert!(r.success);
        assert_eq!(r.data, Some("hello"));
        assert!(r.error.is_none());
    }

    #[test]
    fn test_graph_response_error() {
        let r = GraphResponse::<String>::from_error(ErrorCode::NotFound, "not found");
        assert!(!r.success);
        assert!(r.data.is_none());
        assert_eq!(r.error.as_ref().unwrap().code, ErrorCode::NotFound);
    }

    #[test]
    fn test_is_not_found() {
        let e = anyhow::anyhow!("Node 'abc' not found in graph");
        assert!(is_not_found(&e));
        let e2 = anyhow::anyhow!("internal panic");
        assert!(!is_not_found(&e2));
    }

    #[test]
    fn test_json_to_props_roundtrip() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), serde_json::json!("Alice"));
        map.insert("age".to_string(), serde_json::json!(30));
        map.insert("active".to_string(), serde_json::json!(true));
        let proto = json_to_props(map);
        let back = props_to_json(&proto);
        assert_eq!(back["name"], serde_json::json!("Alice"));
        assert_eq!(back["age"], serde_json::json!(30));
        assert_eq!(back["active"], serde_json::json!(true));
    }

    #[test]
    fn test_parse_traversal_algorithm() {
        assert_eq!(
            parse_traversal_algorithm("bfs"),
            TraversalAlgorithm::Bfs as i32
        );
        assert_eq!(
            parse_traversal_algorithm("dfs"),
            TraversalAlgorithm::Dfs as i32
        );
        assert_eq!(
            parse_traversal_algorithm("parallel_bfs"),
            TraversalAlgorithm::ParallelBfs as i32
        );
        assert_eq!(
            parse_traversal_algorithm("unknown"),
            TraversalAlgorithm::Unspecified as i32
        );
    }

    #[test]
    fn property_and_input_conversion_helpers_cover_nested_shapes() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), serde_json::json!("Alice"));
        props.insert("age".to_string(), serde_json::json!(30));
        props.insert("score".to_string(), serde_json::json!(1.5));
        props.insert("active".to_string(), serde_json::json!(true));
        props.insert("bytes".to_string(), serde_json::json!([1, 2, 3]));
        props.insert("nested".to_string(), serde_json::json!({"k": "v"}));
        props.insert("drop_null".to_string(), serde_json::Value::Null);

        let converted = json_to_props(props);
        let restored = props_to_json(&converted);

        assert_eq!(restored["name"], "Alice");
        assert_eq!(restored["age"], 30);
        assert_eq!(restored["score"], 1.5);
        assert_eq!(restored["active"], true);
        assert_eq!(restored["bytes"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(restored["nested"]["k"], "v");
        assert!(!restored.contains_key("drop_null"));

        assert_eq!(fmt_ts(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            parse_sp_algorithm(Some("astar")),
            Some(ShortestPathAlgorithm::Astar as i32)
        );
        assert_eq!(parse_sp_algorithm(Some("bad")), None);
        assert_eq!(default_max_depth(), 5);
        assert_eq!(default_algorithm(), "bfs");
        assert_eq!(default_walk_depth(), 2);
        assert_eq!(default_walk_limit(), 100);
        assert_eq!(default_step_limit(), 50);
        assert_eq!(default_query_limit(), 100);
        assert_eq!(default_query_language(), "native");

        let proto_node = rest_node_to_proto(RestNodeInput {
            id: "n1".to_string(),
            labels: vec!["Person".to_string()],
            properties: HashMap::new(),
            embedding: Some(RestEmbeddingInput {
                vector: vec![0.1, 0.2, 0.3],
                version: "v1".to_string(),
                model_id: "m".to_string(),
            }),
        });
        assert_eq!(proto_node.embedding.unwrap().dimension, 3);

        let proto_edge = rest_edge_to_proto(RestEdgeInput {
            id: "e1".to_string(),
            from_node_id: "n1".to_string(),
            to_node_id: "n2".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: HashMap::new(),
            weight: Some(1.0),
        });
        assert_eq!(proto_edge.edge_type, "KNOWS");
    }

    #[tokio::test]
    async fn graph_node_and_edge_crud_handlers_route_through_graph_port() {
        let node_body = CreateNodeBody {
            node: RestNodeInput {
                id: "n1".to_string(),
                labels: vec!["Person".to_string()],
                properties: HashMap::new(),
                embedding: None,
            },
        };
        assert_eq!(
            create_node(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(node_body),
            )
            .await
            .into_response()
            .status(),
            StatusCode::CREATED
        );
        assert_eq!(
            get_node(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "n1".to_string())),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            update_node(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "n1".to_string())),
                Json(RestNodeInput {
                    id: "ignored".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: HashMap::new(),
                    embedding: None,
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_node(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "n1".to_string())),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );

        let edge_body = CreateEdgeBody {
            edge: RestEdgeInput {
                id: "e1".to_string(),
                from_node_id: "n1".to_string(),
                to_node_id: "n2".to_string(),
                edge_type: "KNOWS".to_string(),
                properties: HashMap::new(),
                weight: Some(0.5),
            },
        };
        assert_eq!(
            create_edge(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(edge_body),
            )
            .await
            .into_response()
            .status(),
            StatusCode::CREATED
        );
        assert_eq!(
            get_edge(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "e1".to_string())),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            update_edge(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "e1".to_string())),
                Json(RestEdgeInput {
                    id: "ignored".to_string(),
                    from_node_id: "n1".to_string(),
                    to_node_id: "n2".to_string(),
                    edge_type: "KNOWS".to_string(),
                    properties: HashMap::new(),
                    weight: None,
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_edge(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "e1".to_string())),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn graph_query_traversal_batch_analytics_and_constraint_handlers_succeed() {
        for response in [
            get_node_neighbors(
                MockGraphPort::state(MockMode::Ok),
                Path(("g1".to_string(), "n1".to_string())),
            )
            .await
            .into_response(),
            query_nodes(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestNodeQuery {
                    labels: vec!["Person".to_string()],
                    properties: HashMap::new(),
                    limit: 1,
                    offset: None,
                    continuation_token: Some("offset:2".to_string()),
                }),
            )
            .await
            .into_response(),
            query_edges(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestEdgeQuery {
                    edge_type: "KNOWS".to_string(),
                    from_node_id: Some("n1".to_string()),
                    to_node_id: Some("n2".to_string()),
                    properties: HashMap::new(),
                    limit: 1,
                    offset: None,
                    continuation_token: Some("offset:2".to_string()),
                }),
            )
            .await
            .into_response(),
            execute_graph_query(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestGraphQueryRequest {
                    query: "MATCH (n) RETURN n".to_string(),
                    language: "cypher".to_string(),
                    timeout_ms: Some(100),
                }),
            )
            .await
            .into_response(),
            traverse_graph(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestTraversalRequest {
                    start_node_id: "n1".to_string(),
                    max_depth: 2,
                    edge_types: vec!["KNOWS".to_string()],
                    node_labels: vec!["Person".to_string()],
                    algorithm: "parallel".to_string(),
                    limit: Some(10),
                }),
            )
            .await
            .into_response(),
            walk_graph(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(WalkRequest {
                    start_node_id: "n1".to_string(),
                    max_depth: 2,
                    limit: 10,
                }),
            )
            .await
            .into_response(),
            step_graph(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(WalkStepRequest {
                    node_id: "n1".to_string(),
                    edge_type: Some("KNOWS".to_string()),
                    limit: 5,
                }),
            )
            .await
            .into_response(),
            get_graph_stats(MockGraphPort::state(MockMode::Ok), Path("g1".to_string()))
                .await
                .into_response(),
            shortest_path(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                HeaderMap::new(),
                Json(RestShortestPathRequest {
                    start_node_id: "n1".to_string(),
                    target_node_id: "n2".to_string(),
                    max_depth: Some(3),
                    edge_types: vec!["KNOWS".to_string()],
                    algorithm: Some("dijkstra".to_string()),
                    k: Some(1),
                }),
            )
            .await
            .into_response(),
            get_connected_components(MockGraphPort::state(MockMode::Ok), Path("g1".to_string()))
                .await
                .into_response(),
            check_cycles(MockGraphPort::state(MockMode::Ok), Path("g1".to_string()))
                .await
                .into_response(),
            add_unique_constraint(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestUniqueConstraintRequest {
                    label: "Person".to_string(),
                    property: "email".to_string(),
                }),
            )
            .await
            .into_response(),
            remove_unique_constraint(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(RestUniqueConstraintRequest {
                    label: "Person".to_string(),
                    property: "email".to_string(),
                }),
            )
            .await
            .into_response(),
            batch_create_nodes(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(BatchCreateNodesRequest {
                    nodes: vec![RestNodeInput {
                        id: "n1".to_string(),
                        labels: Vec::new(),
                        properties: HashMap::new(),
                        embedding: None,
                    }],
                }),
            )
            .await
            .into_response(),
            batch_create_edges(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(BatchCreateEdgesRequest {
                    edges: vec![RestEdgeInput {
                        id: "e1".to_string(),
                        from_node_id: "n1".to_string(),
                        to_node_id: "n2".to_string(),
                        edge_type: "KNOWS".to_string(),
                        properties: HashMap::new(),
                        weight: Some(1.0),
                    }],
                }),
            )
            .await
            .into_response(),
        ] {
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn graph_collection_management_redirects_and_error_paths_are_explicit() {
        assert_eq!(
            create_graph_collection(
                MockGraphPort::state(MockMode::Ok),
                Json(CreateGraphCollectionBody {
                    graph_id: "g1".to_string(),
                    name: Some("Knowledge".to_string()),
                    description: Some("docs".to_string()),
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::CREATED
        );
        assert_eq!(
            list_graph_collections(MockGraphPort::state(MockMode::Ok))
                .await
                .into_response()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            get_graph_collection(MockGraphPort::state(MockMode::Ok), Path("g1".to_string()))
                .await
                .into_response()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            delete_graph_collection(MockGraphPort::state(MockMode::Ok), Path("g1".to_string()))
                .await
                .into_response()
                .status(),
            StatusCode::NO_CONTENT
        );
        assert_eq!(
            update_graph_schema(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(UpdateGraphSchemaBody {
                    schema: serde_json::to_value(GraphSchema::default()).unwrap(),
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            update_graph_schema(
                MockGraphPort::state(MockMode::Ok),
                Path("g1".to_string()),
                Json(UpdateGraphSchemaBody {
                    schema: serde_json::json!("not a schema"),
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            get_graph_collection(
                MockGraphPort::state(MockMode::NotFound),
                Path("g1".to_string())
            )
            .await
            .into_response()
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            delete_graph_collection(
                MockGraphPort::state(MockMode::NotFound),
                Path("g1".to_string())
            )
            .await
            .into_response()
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            get_node(
                MockGraphPort::state(MockMode::NotFound),
                Path(("g1".to_string(), "missing".to_string())),
            )
            .await
            .into_response()
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            create_node(
                MockGraphPort::state(MockMode::Internal),
                Path("g1".to_string()),
                Json(CreateNodeBody {
                    node: RestNodeInput {
                        id: "n1".to_string(),
                        labels: Vec::new(),
                        properties: HashMap::new(),
                        embedding: None,
                    },
                }),
            )
            .await
            .into_response()
            .status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );

        let not_impl = not_implemented_handler().await.into_response();
        assert_eq!(not_impl.status(), StatusCode::NOT_IMPLEMENTED);

        let legacy = get_node_legacy(Path("n1".to_string()))
            .await
            .into_response();
        assert_eq!(legacy.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(
            legacy.headers().get(header::LOCATION).unwrap(),
            "/api/v2/graphs/default/nodes/n1"
        );
        assert_eq!(
            create_node_legacy().await.into_response().status(),
            StatusCode::PERMANENT_REDIRECT
        );
        assert_eq!(
            create_edge_legacy().await.into_response().status(),
            StatusCode::PERMANENT_REDIRECT
        );
        assert_eq!(
            get_graph_stats_legacy().await.into_response().status(),
            StatusCode::PERMANENT_REDIRECT
        );

        let _router = create_graph_router();
        let _graph_handler = GraphHandler;
        let _traversal_handler = GraphTraversalHandler::new();
    }
}
