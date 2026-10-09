//! Graph operations for ProximaDB
//!
//! Provides a fluent API for graph database operations including
//! node and edge management, traversal queries, and graph analytics.
//!
//! # Examples
//!
//! ```rust,ignore
//! use proximadb_sdk::{ProximaClient, GraphBuilder};
//!
//! let client = ProximaClient::connect("http://localhost:5678")?;
//!
//! // Create a graph
//! client.create_graph("knowledge")
//!     .execute()
//!     .await?;
//!
//! // Add nodes with fluent API
//! client.graph("knowledge")
//!     .add_node()
//!     .id("person_1")
//!     .label("Person")
//!     .property("name", "Alice")
//!     .property("age", 30)
//!     .execute()
//!     .await?;
//!
//! // Add edges
//! client.graph("knowledge")
//!     .add_edge()
//!     .id("edge_1")
//!     .from("person_1")
//!     .to("person_2")
//!     .relationship("KNOWS")
//!     .property("since", "2020")
//!     .execute()
//!     .await?;
//!
//! // Traverse the graph
//! let results = client.graph("knowledge")
//!     .traverse()
//!     .start("person_1")
//!     .relationship("KNOWS")
//!     .max_depth(3)
//!     .execute()
//!     .await?;
//! ```

use crate::error::{ProximaError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,
    /// Node label (type)
    #[serde(default)]
    pub label: Option<String>,
    /// Node properties
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional embedding vector for semantic operations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            properties: HashMap::new(),
            vector: None,
        }
    }

    /// Set the node label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a property
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the embedding vector
    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }
}

/// A graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Relationship type
    pub relationship: String,
    /// Edge properties
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional weight
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        relationship: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relationship: relationship.into(),
            properties: HashMap::new(),
            weight: None,
        }
    }

    /// Add a property
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the edge weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }
}

/// Traversal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TraversalDirection {
    /// Traverse outgoing edges
    #[default]
    Outgoing,
    /// Traverse incoming edges
    Incoming,
    /// Traverse both directions
    Both,
}

/// Graph traversal result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    /// Nodes in the traversal path
    pub nodes: Vec<GraphNode>,
    /// Edges in the traversal path
    pub edges: Vec<GraphEdge>,
    /// Path from start to each node
    #[serde(default)]
    pub paths: Vec<Vec<String>>,
}

/// Handle to a graph for fluent operations
pub struct GraphHandle<'a> {
    #[cfg(feature = "client")]
    client: &'a crate::client::ProximaClient,
    name: String,
}

impl<'a> GraphHandle<'a> {
    /// Create a new graph handle
    #[cfg(feature = "client")]
    pub fn new(client: &'a crate::client::ProximaClient, name: &str) -> Self {
        Self {
            client,
            name: name.to_string(),
        }
    }

    /// Get the graph name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start building a node addition
    #[cfg(feature = "client")]
    pub fn add_node(&'a self) -> NodeBuilder<'a> {
        NodeBuilder::new(self)
    }

    /// Start building an edge addition
    #[cfg(feature = "client")]
    pub fn add_edge(&'a self) -> EdgeBuilder<'a> {
        EdgeBuilder::new(self)
    }

    /// Start building a traversal query
    #[cfg(feature = "client")]
    pub fn traverse(&'a self) -> TraversalBuilder<'a> {
        TraversalBuilder::new(self)
    }

    /// Add a batch of nodes. Posts to the spec-defined batch endpoint
    /// `POST /api/v2/graphs/{id}/nodes/batch` (the generated
    /// `batch_create_nodes` operation's path/verb) with body
    /// `{nodes: [NodeInput, ...]}`. Each `GraphNode` is lowered to a
    /// server-true `NodeInput` (label → labels, vector → embedding).
    ///
    /// Kept on the facade's permissive `BatchNodesResponse` DTO rather than
    /// the typed `batch_create_nodes().send()`: the facade tolerates both the
    /// `GraphResponse` envelope (`data.created_count`) and the legacy flat
    /// `added_count` shape, whereas the generated `BatchNodesResponse` models
    /// only the envelope. The wire connection still flows through the shared
    /// transport via `self.client.post`.
    #[cfg(feature = "client")]
    pub async fn add_nodes(&self, nodes: Vec<GraphNode>) -> Result<usize> {
        let inputs: Vec<NodeInput> = nodes.into_iter().map(NodeInput::from_graph_node).collect();
        let request = BatchNodesRequest { nodes: inputs };
        let url = format!(
            "{}/api/v2/graphs/{}/nodes/batch",
            self.client.url(),
            self.name
        );
        let response: BatchNodesResponse = self.client.post(&url, &request).await?;
        Ok(response.added_count())
    }

    /// Add a batch of edges. Posts to the spec-defined batch endpoint
    /// `POST /api/v2/graphs/{id}/edges/batch` (the generated
    /// `batch_create_edges` operation's path/verb) with body
    /// `{edges: [EdgeInput, ...]}`. Each `GraphEdge` is lowered to a
    /// server-true `EdgeInput`. The `id` defaults to
    /// `"{source}-{relationship}-{target}"` if not provided — callers
    /// who need deterministic edge ids should pass them via the
    /// per-edge builder instead.
    ///
    /// Kept on the facade's permissive `BatchEdgesResponse` DTO for the same
    /// reason as [`add_nodes`](Self::add_nodes); the wire connection flows
    /// through the shared transport via `self.client.post`.
    #[cfg(feature = "client")]
    pub async fn add_edges(&self, edges: Vec<GraphEdge>) -> Result<usize> {
        let inputs: Vec<EdgeInput> = edges.into_iter().map(EdgeInput::from_graph_edge).collect();
        let request = BatchEdgesRequest { edges: inputs };
        let url = format!(
            "{}/api/v2/graphs/{}/edges/batch",
            self.client.url(),
            self.name
        );
        let response: BatchEdgesResponse = self.client.post(&url, &request).await?;
        Ok(response.added_count())
    }

    /// Get a node by ID (`GET /api/v2/graphs/{graph_id}/nodes/{node_id}`,
    /// the generated `get_node` operation's path/verb).
    ///
    /// Kept on the facade's permissive `GraphNode` DTO + hand-formatted path
    /// rather than the typed `get_node().send()`: the generated response wraps
    /// a canonical node with `labels` and `embedding`, while the facade exposes
    /// the first `label` and the embedding `vector` directly. `404` must also
    /// surface as `Ok(None)` (the typed builder folds it into
    /// `Error::ErrorResponse`). The wire connection, auth, and error mapping
    /// still flow through the shared transport via `self.client.get`.
    #[cfg(feature = "client")]
    pub async fn get_node(&self, id: &str) -> Result<Option<GraphNode>> {
        let url = format!(
            "{}/api/v2/graphs/{}/nodes/{}",
            self.client.url(),
            self.name,
            id
        );
        match self.client.get::<GraphResponse<CanonicalNode>>(&url).await {
            Ok(response) => Ok(Some(response.data.into())),
            Err(ProximaError::Network(crate::error::NetworkError::HttpError {
                status: 404,
                ..
            })) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Delete a node by ID via the generated `delete_node` operation's
    /// path/verb (`DELETE /api/v2/graphs/{graph_id}/nodes/{node_id}`),
    /// through the shared generated transport connection (see
    /// `GraphBuilder::execute`).
    #[cfg(feature = "client")]
    pub async fn delete_node(&self, id: &str) -> Result<()> {
        let url = format!(
            "{}/api/v2/graphs/{}/nodes/{}",
            self.client.url(),
            self.name,
            id
        );
        self.client.delete::<serde_json::Value>(&url).await?;
        Ok(())
    }

    /// Delete an edge.
    ///
    /// Hand-built (NOT routed through the generated client): the generated
    /// `genrest` module exposes no operation for this 3-segment path. The
    /// spec only declares `DELETE /api/v2/graphs/{id}/edges/{edge_id}` with a
    /// single `edge_id`, whereas this SDK signature addresses an edge by its
    /// `(source, target, relationship)` triple
    /// (`/edges/{source}/{target}/{relationship}`). The endpoint-shape
    /// mismatch is tracked separately (TD-126 Phase-1 annotation follow-up);
    /// until the spec/server gains the triple shape (or this SDK adopts the
    /// `{edge_id}` shape), the request URL stays hand-formatted — but the
    /// wire connection, auth, and error mapping still go through the shared
    /// transport via `self.client.delete`.
    #[cfg(feature = "client")]
    pub async fn delete_edge(&self, source: &str, target: &str, relationship: &str) -> Result<()> {
        let url = format!(
            "{}/api/v2/graphs/{}/edges/{}/{}/{}",
            self.client.url(),
            self.name,
            source,
            target,
            relationship
        );
        self.client.delete::<serde_json::Value>(&url).await?;
        Ok(())
    }
}

/// Builder for adding nodes
pub struct NodeBuilder<'a> {
    handle: &'a GraphHandle<'a>,
    id: Option<String>,
    label: Option<String>,
    properties: HashMap<String, serde_json::Value>,
    vector: Option<Vec<f32>>,
}

impl<'a> NodeBuilder<'a> {
    fn new(handle: &'a GraphHandle<'a>) -> Self {
        Self {
            handle,
            id: None,
            label: None,
            properties: HashMap::new(),
            vector: None,
        }
    }

    /// Set the node ID
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the node label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a property
    pub fn property(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the embedding vector
    pub fn vector(mut self, vector: &[f32]) -> Self {
        self.vector = Some(vector.to_vec());
        self
    }

    /// Execute the node addition. Posts the spec-true wrapped envelope
    /// `{node: NodeInput}` to the generated `create_node` operation's
    /// path/verb (`POST /api/v2/graphs/{graph_id}/nodes`). The SDK's `label`
    /// becomes `labels: [label]`, and `vector` is wrapped in an
    /// `EmbeddingInput` so the body matches the OpenAPI contract. Goes
    /// through the shared generated transport connection with the facade's
    /// 2xx-tolerant error mapping (see `GraphBuilder::execute` for why the
    /// fully-typed `create_node().send()` is not used).
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<()> {
        let id = self
            .id
            .ok_or_else(|| ProximaError::Internal("Node ID is required".to_string()))?;

        let labels = self.label.map(|l| vec![l]);
        let embedding = self.vector.map(|v| EmbeddingInput {
            vector: v,
            model_id: None,
            modality: None,
        });

        let node = NodeInput {
            id,
            labels,
            properties: if self.properties.is_empty() {
                None
            } else {
                Some(self.properties)
            },
            embedding,
        };

        let request = CreateNodeRequest { node };

        let url = format!(
            "{}/api/v2/graphs/{}/nodes",
            self.handle.client.url(),
            self.handle.name
        );
        let _response: serde_json::Value = self.handle.client.post(&url, &request).await?;
        Ok(())
    }
}

/// Builder for adding edges
pub struct EdgeBuilder<'a> {
    handle: &'a GraphHandle<'a>,
    id: Option<String>,
    source: Option<String>,
    target: Option<String>,
    relationship: Option<String>,
    properties: HashMap<String, serde_json::Value>,
    weight: Option<f64>,
}

impl<'a> EdgeBuilder<'a> {
    fn new(handle: &'a GraphHandle<'a>) -> Self {
        Self {
            handle,
            id: None,
            source: None,
            target: None,
            relationship: None,
            properties: HashMap::new(),
            weight: None,
        }
    }

    /// Set an explicit edge ID. If not set, `execute()` synthesises one
    /// from `{source}-{relationship}-{target}` to satisfy the spec's
    /// `EdgeInput.id` requirement.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the source node ID
    pub fn from(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the target node ID
    pub fn to(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set the relationship type
    pub fn relationship(mut self, rel: impl Into<String>) -> Self {
        self.relationship = Some(rel.into());
        self
    }

    /// Add a property
    pub fn property(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the edge weight
    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Execute the edge addition. Posts the spec-true wrapped envelope
    /// `{edge: EdgeInput}` with `id`, `from_node_id`, `to_node_id`, and
    /// `edge_type` fields (server-true names) to the generated `create_edge`
    /// operation's path/verb (`POST /api/v2/graphs/{graph_id}/edges`),
    /// through the shared generated transport connection (see
    /// `GraphBuilder::execute`).
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<()> {
        let source = self
            .source
            .ok_or_else(|| ProximaError::Internal("Source node ID is required".to_string()))?;
        let target = self
            .target
            .ok_or_else(|| ProximaError::Internal("Target node ID is required".to_string()))?;
        let relationship = self
            .relationship
            .ok_or_else(|| ProximaError::Internal("Relationship type is required".to_string()))?;
        let id = self
            .id
            .unwrap_or_else(|| format!("{source}-{relationship}-{target}"));

        let edge = EdgeInput {
            id,
            from_node_id: source,
            to_node_id: target,
            edge_type: relationship,
            properties: if self.properties.is_empty() {
                None
            } else {
                Some(self.properties)
            },
            weight: self.weight,
        };

        let request = CreateEdgeRequest { edge };

        let url = format!(
            "{}/api/v2/graphs/{}/edges",
            self.handle.client.url(),
            self.handle.name
        );
        let _response: serde_json::Value = self.handle.client.post(&url, &request).await?;
        Ok(())
    }
}

/// Builder for graph traversal queries
pub struct TraversalBuilder<'a> {
    handle: &'a GraphHandle<'a>,
    start_node: Option<String>,
    relationships: Vec<String>,
    direction: TraversalDirection,
    max_depth: usize,
    limit: usize,
    filter: Option<String>,
    node_labels: Vec<String>,
    algorithm: Option<String>,
}

impl<'a> TraversalBuilder<'a> {
    fn new(handle: &'a GraphHandle<'a>) -> Self {
        Self {
            handle,
            start_node: None,
            relationships: Vec::new(),
            direction: TraversalDirection::default(),
            max_depth: 3,
            limit: 100,
            filter: None,
            node_labels: Vec::new(),
            algorithm: None,
        }
    }

    /// Set the starting node
    pub fn start(mut self, node_id: impl Into<String>) -> Self {
        self.start_node = Some(node_id.into());
        self
    }

    /// Add a relationship type to follow
    pub fn relationship(mut self, rel: impl Into<String>) -> Self {
        self.relationships.push(rel.into());
        self
    }

    /// Add multiple relationship types
    pub fn relationships(mut self, rels: Vec<String>) -> Self {
        self.relationships.extend(rels);
        self
    }

    /// Set the traversal direction. Stored locally for fluent
    /// configuration, but **not** transmitted on the wire — the server's
    /// `TraverseRequest` (per OpenAPI spec) does not accept a
    /// `direction` field today. Tracked separately.
    pub fn direction(mut self, dir: TraversalDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Traverse outgoing edges only
    pub fn outgoing(mut self) -> Self {
        self.direction = TraversalDirection::Outgoing;
        self
    }

    /// Traverse incoming edges only
    pub fn incoming(mut self) -> Self {
        self.direction = TraversalDirection::Incoming;
        self
    }

    /// Traverse both directions
    pub fn both(mut self) -> Self {
        self.direction = TraversalDirection::Both;
        self
    }

    /// Set the maximum traversal depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the maximum number of results
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Add a filter expression for nodes. Kept for backwards source
    /// compatibility, but not on the wire — see `direction()`.
    pub fn filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Add a node label filter (server-true `node_labels`).
    pub fn node_label(mut self, label: impl Into<String>) -> Self {
        self.node_labels.push(label.into());
        self
    }

    /// Add multiple node labels at once.
    pub fn node_labels(mut self, labels: Vec<String>) -> Self {
        self.node_labels.extend(labels);
        self
    }

    /// Select the traversal algorithm (bfs | dfs | shortest_path). Maps
    /// to the spec's optional `algorithm` field.
    pub fn algorithm(mut self, alg: impl Into<String>) -> Self {
        self.algorithm = Some(alg.into());
        self
    }

    /// Execute the traversal. Posts the spec-true flat shape
    /// `{start_node_id, max_depth, edge_types, node_labels?, algorithm?,
    /// limit?}` to `POST /api/v2/graphs/{id}/traverse` (the generated
    /// `traverse_graph` operation's path/verb) — no `graph` wrapper,
    /// server-true field names.
    ///
    /// Kept on the facade's permissive `TraversalResult` DTO rather than the
    /// typed `traverse_graph().send()`: its canonical nodes/edges require an
    /// explicit lowering (`labels`/`embedding` and canonical edge field names)
    /// into the facade `GraphNode`/`GraphEdge`. The wire connection flows
    /// through the shared transport via `self.handle.client.post`.
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<TraversalResult> {
        let start_node_id = self.start_node.ok_or_else(|| {
            ProximaError::Internal("Start node is required for traversal".to_string())
        })?;

        let request = TraversalRequest {
            start_node_id,
            max_depth: self.max_depth,
            edge_types: self.relationships,
            node_labels: if self.node_labels.is_empty() {
                None
            } else {
                Some(self.node_labels)
            },
            algorithm: self.algorithm,
            limit: Some(self.limit),
        };

        let url = format!(
            "{}/api/v2/graphs/{}/traverse",
            self.handle.client.url(),
            self.handle.name
        );
        let response: GraphResponse<CanonicalTraversalResult> =
            self.handle.client.post(&url, &request).await?;
        Ok(response.data.into())
    }
}

/// Builder for creating graphs
pub struct GraphBuilder<'a> {
    #[cfg(feature = "client")]
    client: &'a crate::client::ProximaClient,
    graph_id: String,
    name: Option<String>,
    description: Option<String>,
}

impl<'a> GraphBuilder<'a> {
    /// Create a new graph builder. The supplied identifier is used as
    /// the spec-required `graph_id`; call `.name(...)` to set an
    /// optional human-readable name.
    #[cfg(feature = "client")]
    pub fn new(client: &'a crate::client::ProximaClient, graph_id: &str) -> Self {
        Self {
            client,
            graph_id: graph_id.to_string(),
            name: None,
            description: None,
        }
    }

    /// Set the optional human-readable graph name (server defaults to
    /// `graph_id` when omitted).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the graph description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Execute the graph creation. Posts the spec-true body
    /// `{graph_id, name?, description?}` to the generated `create_graph`
    /// operation's path/verb (`POST /api/v2/graphs`).
    ///
    /// The request goes through the shared generated transport connection
    /// (`self.client.post` → `transport.client()`) with the facade's auth,
    /// permissive DTOs, and 2xx-tolerant error mapping. NB: routing the
    /// fully-typed `transport.create_graph().send()` was tried and reverted
    /// — the generated success matcher accepts only the spec-declared `200`,
    /// but the live server returns `201 Created`, which the facade's
    /// `is_success()` path tolerates. Sourcing path/verb from the generated
    /// client while keeping the facade's 2xx semantics is the documented
    /// fallback (TD-126).
    #[cfg(feature = "client")]
    pub async fn execute(self) -> Result<()> {
        let request = CreateGraphRequest {
            graph_id: self.graph_id,
            name: self.name,
            description: self.description,
        };

        // Path/verb mirror the generated `create_graph` operation
        // (`POST /api/v2/graphs`); see genrest::Client::create_graph.
        let url = format!("{}/api/v2/graphs", self.client.url());
        let _response: serde_json::Value = self.client.post(&url, &request).await?;
        Ok(())
    }
}

// Request/Response types for HTTP API — mirror the OpenAPI v2 graph
// schemas (`docs/openapi/proximadb-openapi.yaml`).

#[derive(Debug, Serialize)]
struct CreateGraphRequest {
    graph_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

/// Spec-true node payload nested inside `CreateNodeRequest.node`.
#[derive(Debug, Serialize)]
struct NodeInput {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<EmbeddingInput>,
}

impl NodeInput {
    fn from_graph_node(node: GraphNode) -> Self {
        let labels = node.label.map(|l| vec![l]);
        let embedding = node.vector.map(|v| EmbeddingInput {
            vector: v,
            model_id: None,
            modality: None,
        });
        Self {
            id: node.id,
            labels,
            properties: if node.properties.is_empty() {
                None
            } else {
                Some(node.properties)
            },
            embedding,
        }
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingInput {
    vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modality: Option<String>,
}

#[derive(Debug, Serialize)]
struct CreateNodeRequest {
    node: NodeInput,
}

/// Spec-true edge payload nested inside `CreateEdgeRequest.edge`.
#[derive(Debug, Serialize)]
struct EdgeInput {
    id: String,
    from_node_id: String,
    to_node_id: String,
    edge_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight: Option<f64>,
}

impl EdgeInput {
    fn from_graph_edge(edge: GraphEdge) -> Self {
        let id = format!("{}-{}-{}", edge.source, edge.relationship, edge.target);
        Self {
            id,
            from_node_id: edge.source,
            to_node_id: edge.target,
            edge_type: edge.relationship,
            properties: if edge.properties.is_empty() {
                None
            } else {
                Some(edge.properties)
            },
            weight: edge.weight,
        }
    }
}

#[derive(Debug, Serialize)]
struct CreateEdgeRequest {
    edge: EdgeInput,
}

#[derive(Debug, Serialize)]
struct BatchNodesRequest {
    nodes: Vec<NodeInput>,
}

#[derive(Debug, Deserialize)]
struct BatchNodesResponse {
    /// Server returns a `GraphResponse<BatchResults<Node>>` envelope.
    /// `data.created_count` is the canonical added count; `added_count` is a
    /// flatter legacy shape we still accept for back-compat with older
    /// mocks.
    #[serde(default)]
    data: Option<BatchData>,
    #[serde(default)]
    added_count: Option<usize>,
}

impl BatchNodesResponse {
    fn added_count(&self) -> usize {
        self.data
            .as_ref()
            .and_then(|d| d.created_count.or(d.count))
            .or(self.added_count)
            .unwrap_or(0)
    }
}

#[derive(Debug, Serialize)]
struct BatchEdgesRequest {
    edges: Vec<EdgeInput>,
}

#[derive(Debug, Deserialize)]
struct BatchEdgesResponse {
    #[serde(default)]
    data: Option<BatchData>,
    #[serde(default)]
    added_count: Option<usize>,
}

impl BatchEdgesResponse {
    fn added_count(&self) -> usize {
        self.data
            .as_ref()
            .and_then(|d| d.created_count.or(d.count))
            .or(self.added_count)
            .unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
struct BatchData {
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    created_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GraphResponse<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct CanonicalEmbedding {
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CanonicalNode {
    id: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    embedding: Option<CanonicalEmbedding>,
}

impl From<CanonicalNode> for GraphNode {
    fn from(node: CanonicalNode) -> Self {
        Self {
            id: node.id,
            label: node.labels.into_iter().next(),
            properties: node.properties,
            vector: node.embedding.map(|embedding| embedding.vector),
        }
    }
}

#[derive(Debug, Deserialize)]
struct CanonicalEdge {
    from_node_id: String,
    to_node_id: String,
    edge_type: String,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    weight: Option<f64>,
}

impl From<CanonicalEdge> for GraphEdge {
    fn from(edge: CanonicalEdge) -> Self {
        Self {
            source: edge.from_node_id,
            target: edge.to_node_id,
            relationship: edge.edge_type,
            properties: edge.properties,
            weight: edge.weight,
        }
    }
}

#[derive(Debug, Deserialize)]
struct CanonicalTraversalResult {
    #[serde(default)]
    nodes: Vec<CanonicalNode>,
    #[serde(default)]
    edges: Vec<CanonicalEdge>,
    #[serde(default)]
    paths: Vec<Vec<String>>,
}

impl From<CanonicalTraversalResult> for TraversalResult {
    fn from(result: CanonicalTraversalResult) -> Self {
        Self {
            nodes: result.nodes.into_iter().map(GraphNode::from).collect(),
            edges: result.edges.into_iter().map(GraphEdge::from).collect(),
            paths: result.paths,
        }
    }
}

#[derive(Debug, Serialize)]
struct TraversalRequest {
    start_node_id: String,
    max_depth: usize,
    edge_types: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    node_labels: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProximaClient;
    use serde_json::json;

    #[test]
    fn test_graph_node_builder() {
        let node = GraphNode::new("node_1")
            .with_label("Person")
            .with_property("name", "Alice")
            .with_property("age", 30);

        assert_eq!(node.id, "node_1");
        assert_eq!(node.label, Some("Person".to_string()));
        assert_eq!(
            node.properties.get("name"),
            Some(&serde_json::json!("Alice"))
        );
        assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));
    }

    #[test]
    fn test_graph_edge_builder() {
        let edge = GraphEdge::new("person_1", "person_2", "KNOWS")
            .with_property("since", "2020")
            .with_weight(0.9);

        assert_eq!(edge.source, "person_1");
        assert_eq!(edge.target, "person_2");
        assert_eq!(edge.relationship, "KNOWS");
        assert_eq!(edge.weight, Some(0.9));
    }

    #[test]
    fn test_traversal_direction() {
        let dir = TraversalDirection::default();
        assert_eq!(dir, TraversalDirection::Outgoing);
    }

    #[test]
    fn graph_node_serialization_skips_absent_vector_and_defaults_fields() {
        let node: GraphNode = serde_json::from_value(json!({"id": "n1"})).unwrap();
        assert_eq!(node.id, "n1");
        assert_eq!(node.label, None);
        assert!(node.properties.is_empty());
        assert_eq!(node.vector, None);

        let serialized = serde_json::to_value(GraphNode::new("n2")).unwrap();
        assert_eq!(
            serialized,
            json!({"id": "n2", "label": null, "properties": {}})
        );

        let with_vector = GraphNode::new("n3").with_vector(vec![0.1, 0.2]);
        assert_eq!(with_vector.vector, Some(vec![0.1, 0.2]));
    }

    #[test]
    fn graph_edge_serialization_skips_absent_weight_and_defaults_properties() {
        let edge: GraphEdge = serde_json::from_value(json!({
            "source": "a",
            "target": "b",
            "relationship": "KNOWS"
        }))
        .unwrap();

        assert_eq!(edge.source, "a");
        assert_eq!(edge.target, "b");
        assert_eq!(edge.relationship, "KNOWS");
        assert!(edge.properties.is_empty());
        assert_eq!(edge.weight, None);

        let serialized = serde_json::to_value(edge).unwrap();
        assert_eq!(
            serialized,
            json!({"source": "a", "target": "b", "relationship": "KNOWS", "properties": {}})
        );
    }

    #[test]
    fn traversal_direction_serializes_as_lowercase_api_value() {
        assert_eq!(
            serde_json::to_value(TraversalDirection::Outgoing).unwrap(),
            json!("outgoing")
        );
        assert_eq!(
            serde_json::to_value(TraversalDirection::Incoming).unwrap(),
            json!("incoming")
        );
        assert_eq!(
            serde_json::to_value(TraversalDirection::Both).unwrap(),
            json!("both")
        );
    }

    #[test]
    fn graph_handle_and_builders_record_fluent_state() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let handle = GraphHandle::new(&client, "knowledge");

        assert_eq!(handle.name(), "knowledge");

        let node = handle
            .add_node()
            .id("person_1")
            .label("Person")
            .property("name", "Alice")
            .vector(&[0.1, 0.2]);
        assert_eq!(node.id.as_deref(), Some("person_1"));
        assert_eq!(node.label.as_deref(), Some("Person"));
        assert_eq!(node.properties["name"], json!("Alice"));
        assert_eq!(node.vector, Some(vec![0.1, 0.2]));

        let edge = handle
            .add_edge()
            .from("person_1")
            .to("person_2")
            .relationship("KNOWS")
            .property("since", 2020)
            .weight(0.5);
        assert_eq!(edge.source.as_deref(), Some("person_1"));
        assert_eq!(edge.target.as_deref(), Some("person_2"));
        assert_eq!(edge.relationship.as_deref(), Some("KNOWS"));
        assert_eq!(edge.properties["since"], json!(2020));
        assert_eq!(edge.weight, Some(0.5));

        let traversal = handle
            .traverse()
            .start("person_1")
            .relationship("KNOWS")
            .relationships(vec!["LIKES".to_string()])
            .incoming()
            .outgoing()
            .both()
            .direction(TraversalDirection::Incoming)
            .max_depth(4)
            .limit(25)
            .filter("age > 30")
            .node_label("Person")
            .node_labels(vec!["Org".to_string()])
            .algorithm("bfs");
        assert_eq!(traversal.start_node.as_deref(), Some("person_1"));
        assert_eq!(
            traversal.relationships,
            vec!["KNOWS".to_string(), "LIKES".to_string()]
        );
        assert_eq!(traversal.direction, TraversalDirection::Incoming);
        assert_eq!(traversal.max_depth, 4);
        assert_eq!(traversal.limit, 25);
        assert_eq!(traversal.filter.as_deref(), Some("age > 30"));
        assert_eq!(
            traversal.node_labels,
            vec!["Person".to_string(), "Org".to_string()]
        );
        assert_eq!(traversal.algorithm.as_deref(), Some("bfs"));
    }

    #[tokio::test]
    async fn node_edge_and_traversal_builders_validate_required_fields_before_network() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let handle = GraphHandle::new(&client, "knowledge");

        let node_error = handle.add_node().execute().await.unwrap_err();
        assert!(
            matches!(node_error, ProximaError::Internal(message) if message == "Node ID is required")
        );

        let edge_error = handle.add_edge().execute().await.unwrap_err();
        assert!(
            matches!(edge_error, ProximaError::Internal(message) if message == "Source node ID is required")
        );

        let edge_error = handle.add_edge().from("a").execute().await.unwrap_err();
        assert!(
            matches!(edge_error, ProximaError::Internal(message) if message == "Target node ID is required")
        );

        let edge_error = handle
            .add_edge()
            .from("a")
            .to("b")
            .execute()
            .await
            .unwrap_err();
        assert!(
            matches!(edge_error, ProximaError::Internal(message) if message == "Relationship type is required")
        );

        let traversal_error = handle.traverse().execute().await.unwrap_err();
        assert!(
            matches!(traversal_error, ProximaError::Internal(message) if message == "Start node is required for traversal")
        );
    }

    #[test]
    fn graph_request_dtos_serialize_spec_true_shapes() {
        // CreateGraphRequest: {graph_id, name?, description?}
        let create = CreateGraphRequest {
            graph_id: "knowledge".to_string(),
            name: Some("Knowledge Graph".to_string()),
            description: Some("domain graph".to_string()),
        };
        assert_eq!(
            serde_json::to_value(create).unwrap(),
            json!({
                "graph_id": "knowledge",
                "name": "Knowledge Graph",
                "description": "domain graph"
            })
        );

        // Minimal CreateGraphRequest omits optional fields entirely.
        let minimal = CreateGraphRequest {
            graph_id: "k".to_string(),
            name: None,
            description: None,
        };
        assert_eq!(
            serde_json::to_value(minimal).unwrap(),
            json!({"graph_id": "k"})
        );

        // CreateNodeRequest: {node: NodeInput}
        let node_request = CreateNodeRequest {
            node: NodeInput {
                id: "n1".to_string(),
                labels: Some(vec!["Person".to_string()]),
                properties: None,
                embedding: Some(EmbeddingInput {
                    vector: vec![0.5, 0.25],
                    model_id: None,
                    modality: None,
                }),
            },
        };
        assert_eq!(
            serde_json::to_value(node_request).unwrap(),
            json!({"node": {
                "id": "n1",
                "labels": ["Person"],
                "embedding": {"vector": [0.5, 0.25]}
            }})
        );

        // CreateEdgeRequest: {edge: EdgeInput}
        let edge_request = CreateEdgeRequest {
            edge: EdgeInput {
                id: "e1".to_string(),
                from_node_id: "a".to_string(),
                to_node_id: "b".to_string(),
                edge_type: "KNOWS".to_string(),
                properties: None,
                weight: Some(0.9),
            },
        };
        assert_eq!(
            serde_json::to_value(edge_request).unwrap(),
            json!({"edge": {
                "id": "e1",
                "from_node_id": "a",
                "to_node_id": "b",
                "edge_type": "KNOWS",
                "weight": 0.9
            }})
        );

        // BatchCreateNodesRequest: {nodes: [NodeInput, ...]}
        let batch_nodes = BatchNodesRequest {
            nodes: vec![NodeInput {
                id: "n1".to_string(),
                labels: None,
                properties: None,
                embedding: None,
            }],
        };
        assert_eq!(
            serde_json::to_value(batch_nodes).unwrap(),
            json!({"nodes": [{"id": "n1"}]})
        );

        // BatchCreateEdgesRequest: {edges: [EdgeInput, ...]}
        let batch_edges = BatchEdgesRequest {
            edges: vec![EdgeInput {
                id: "e1".to_string(),
                from_node_id: "a".to_string(),
                to_node_id: "b".to_string(),
                edge_type: "KNOWS".to_string(),
                properties: None,
                weight: None,
            }],
        };
        assert_eq!(
            serde_json::to_value(batch_edges).unwrap(),
            json!({"edges": [{
                "id": "e1",
                "from_node_id": "a",
                "to_node_id": "b",
                "edge_type": "KNOWS"
            }]})
        );

        // TraversalRequest: flat shape with start_node_id, no graph wrapper
        let traversal = TraversalRequest {
            start_node_id: "n1".to_string(),
            max_depth: 3,
            edge_types: vec!["KNOWS".to_string()],
            node_labels: Some(vec!["Person".to_string()]),
            algorithm: Some("bfs".to_string()),
            limit: Some(10),
        };
        assert_eq!(
            serde_json::to_value(traversal).unwrap(),
            json!({
                "start_node_id": "n1",
                "max_depth": 3,
                "edge_types": ["KNOWS"],
                "node_labels": ["Person"],
                "algorithm": "bfs",
                "limit": 10
            })
        );
    }

    #[test]
    fn batch_responses_accept_graph_response_envelope_and_legacy_added_count() {
        // Spec envelope: the success count is explicit and may be lower than
        // the request/result length when individual items are rejected.
        let nodes: BatchNodesResponse = serde_json::from_value(json!({
            "success": true,
            "data": {
                "created_count": 1,
                "updated_count": 0,
                "failed_count": 1,
                "results": [{"id": "n1"}],
                "errors": [{"id": "n2", "error": "rejected"}]
            }
        }))
        .unwrap();
        assert_eq!(nodes.added_count(), 1);

        // Legacy flat shape for back-compat.
        let nodes_legacy: BatchNodesResponse =
            serde_json::from_value(json!({"added_count": 3})).unwrap();
        assert_eq!(nodes_legacy.added_count(), 3);

        let edges: BatchEdgesResponse = serde_json::from_value(json!({
            "success": true,
            "data": {
                "created_count": 0,
                "updated_count": 0,
                "failed_count": 1,
                "results": [],
                "errors": [{"id": "e1", "error": "rejected"}]
            }
        }))
        .unwrap();
        assert_eq!(edges.added_count(), 0);
    }

    #[test]
    fn traversal_result_deserializes_default_paths() {
        let result: TraversalResult = serde_json::from_value(json!({
            "nodes": [{"id": "n1"}],
            "edges": []
        }))
        .unwrap();

        assert_eq!(result.nodes.len(), 1);
        assert!(result.edges.is_empty());
        assert!(result.paths.is_empty());
    }

    #[test]
    fn graph_builder_records_graph_id_name_and_description() {
        let client = ProximaClient::for_tests("http://localhost:5678");
        let builder = GraphBuilder::new(&client, "knowledge")
            .name("Knowledge Graph")
            .description("domain graph");

        assert_eq!(builder.graph_id, "knowledge");
        assert_eq!(builder.name.as_deref(), Some("Knowledge Graph"));
        assert_eq!(builder.description.as_deref(), Some("domain graph"));
    }

    #[test]
    fn node_input_lowering_maps_label_to_labels_and_vector_to_embedding() {
        let node = GraphNode::new("n1")
            .with_label("Person")
            .with_property("name", "Alice")
            .with_vector(vec![0.5, 0.25]);

        let input = NodeInput::from_graph_node(node);
        assert_eq!(input.id, "n1");
        assert_eq!(input.labels.as_deref(), Some(&["Person".to_string()][..]));
        assert_eq!(input.properties.as_ref().unwrap()["name"], json!("Alice"));
        assert_eq!(input.embedding.as_ref().unwrap().vector, vec![0.5, 0.25]);
    }

    #[test]
    fn edge_input_lowering_synthesises_deterministic_id_from_triple() {
        let edge = GraphEdge::new("a", "b", "KNOWS").with_weight(0.42);
        let input = EdgeInput::from_graph_edge(edge);
        assert_eq!(input.id, "a-KNOWS-b");
        assert_eq!(input.from_node_id, "a");
        assert_eq!(input.to_node_id, "b");
        assert_eq!(input.edge_type, "KNOWS");
        assert_eq!(input.weight, Some(0.42));
    }
}
