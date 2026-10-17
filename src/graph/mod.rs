/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # ProximaDB Native Graph Database Engine
//!
//! This module implements ProximaDB's native graph database capabilities over
//! canonical `ProximaRecord` node and edge records. Protocol-specific graph
//! types are compatibility edges; durable graph truth is the shared record
//! envelope defined by the convergence design.
//!
//! ## Design Principles
//!
//! - **Record-First**: Nodes and edges map to canonical `ProximaRecord`
//! - **Arc-Based Sharing**: Zero-copy memory sharing between vector and graph engines
//! - **CSR Projection**: Compressed Sparse Row is a rebuildable topology projection
//! - **ORION Runtime**: Graph projection over canonical records; distributed
//!   coordination and tiering are relational/storage substrate concerns
//!
//! ## Performance Characteristics
//!
//! - **Traversal**: 1M+ edges/second
//! - **Node Lookup**: < 1μs
//! - **Memory Overhead**: < 100 bytes/node
//! - **Arc Clone**: ~8 bytes (pointer copy)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │            GraphService             │
//! │        (Business Logic Layer)       │
//! ├─────────────────────────────────────┤
//! │          ORION Graph Runtime        │
//! │  ┌───────────────────────────────┐  │
//! │  │ CSR projection + graph planner│  │
//! │  └───────────────────────────────┘  │
//! │   Relational routing/tiering below  │
//! ├─────────────────────────────────────┤
//! │           Arc Memory Pool           │
//! │    ┌────────────┬─────────────┐     │
//! │    │ Nodes/Edges│ Adj/CSR     │     │
//! │    │ Records    │ Projections │     │
//! │    └────────────┴─────────────┘     │
//! └─────────────────────────────────────┘
//! ```

pub mod adjacency_projection;
pub mod canonical;
/// Catapult shortcut table (LLD 6.3, arXiv 2603.02164).
pub use proximadb_catapult::catapult;
/// Cold graph-payload record store (TD-168 Phase 2): durable, Cool-tiered object
/// storage backing for node/edge payloads so a graph larger than RAM is servable.
pub mod cold_payload_store;
/// Segment-batched cold graph-payload store (TD-168 #3, Phase 1): many records per
/// object to cut object-store op count, with an oid→byte-range index for ranged
/// point-gets. Capability only — not yet wired into production.
pub mod cold_segment_store;
pub mod engines;
pub mod merge;
pub mod model;
pub mod proto_convert;
pub mod rag;
// Generic, engine-agnostic traversal utilities
pub use engines::generic_traversal;
// Default graph WAL factory (unified) for composition-root injection into the
// ORION engine constructors (ORION cascade PR 5).
pub use crate::storage::persistence::write_ahead_log::wal_operations::unified_wal_factory;
pub mod hybrid;
pub mod monitoring;
pub mod query;
pub mod service;
pub mod service_algorithms;

// Re-export public types
pub use cold_payload_store::ColdGraphRecordStore;
pub use cold_segment_store::ColdGraphSegmentStore;
pub use engines::orion::OrionGraphEngine;
pub use engines::{
    EmbeddingMode, EngineCapabilities, GraphEngineConfig, GraphEngineFactory, GraphEngineType,
};
pub use hybrid::HybridQueryEngine;
pub use monitoring::GraphMonitor;
pub use query::{PatternMatcher, QueryPlanner};
pub use service::GraphOperationsService;
// Backward compatibility alias
pub use service::GraphOperationsService as GraphService;
// Transaction support types
pub use service::{
    IsolationLevel, TransactionHandle, TransactionId, TransactionManager, TransactionState,
    UnitOfWork,
};

// Algorithm types for high-level API
pub use service_algorithms::{
    CentralityAlgorithm, CentralityConfig, CentralityResult, CommunityAlgorithm, CommunityConfig,
    CommunityResult,
};

// Canonical types for REST/gRPC parity
pub use canonical::{
    BatchError, BatchResults, CanonicalEdge, CanonicalEmbedding, CanonicalNode, CanonicalPath,
    ErrorCode, GraphError, GraphResponse, QueryResults, ResponseMetadata, ShortestPathResult,
    TraversalResults, TraversalStats as CanonicalTraversalStats,
};

// Neutral, transport-agnostic graph domain types (TD-123 Step 1). The engine and
// services speak these; wire adapters convert proto <-> these at the boundary.
pub use model::{
    Edge, EdgeQuery, EdgeTypeStats, EmbeddingVersion, GraphPath, GraphStats, ImpactDirection,
    LabelStats, Node, NodeQuery, PropertyArray, PropertyFilter, PropertyFilterOperator,
    PropertyObject, PropertyValue, TraversalAlgorithm, TraversalRequest, TraversalResponse,
    TraversalStats, property_value, property_value::Value,
};

use proximadb_kernel::error::ProximaDBError;
#[cfg(test)]
use proximadb_orion_engine::property_value_to_string;
use std::sync::Arc;
type Result<T> = std::result::Result<T, ProximaDBError>;

/// Node ID type alias for clarity
pub type NodeId = String;

/// Edge ID type alias for clarity
pub type EdgeId = String;

/// The ONE structural composition point for graph tenant scoping. The network layer
/// passes a tenant-CLEAN `graph_id`; the service composes the physical scope key here.
/// Path-style `{tenant}/{graph_id}` (never `{tenant}::{graph_id}`). Validates the tenant
/// as a path segment (fail-closed) before it becomes an engine-registry / oid / constraint key.
///
/// Isolation is structural, not a per-query name predicate: every read/write path routes
/// through this one composition so create and read agree, and no network handler bakes the
/// tenant into a user-visible name. Preserves the prior isolation mechanism (distinct engine
/// per (tenant, graph)); only the composition location (service, not handler) and separator
/// (`/` not `::`) change.
pub fn scoped_graph_id(tenant: &str, graph_id: &str) -> anyhow::Result<String> {
    proximadb_tenant::validate_request_tenant(tenant)
        .map_err(|e| anyhow::anyhow!("invalid tenant '{tenant}': {e}"))?;
    // A `/` in the graph id would address another tenant's `{tenant}/{graph_id}` key.
    proximadb_tenant::validate_scoped_resource_name(graph_id)
        .map_err(|e| anyhow::anyhow!("invalid graph id: {e}"))?;
    // The default tenant stays UNSCOPED (bare `graph_id`). Graph node/edge ops resolve the
    // backing engine through `get_or_create_graph_engine`, which requires the collection to
    // already exist in `collection_service` — and graph collection CREATION is not yet
    // tenant-scoped (created bare via the REST/pgwire/internal create paths). Prefixing the
    // default tenant would key reads at `default/{graph_id}` while creation stays bare, so every
    // default-tenant create→use would fail "collection does not exist". Named tenants DO scope
    // (`{tenant}/{graph_id}`) as the structural foundation — but that path is only end-to-end
    // functional once collection creation is scoped in lockstep (TD-GRAPH-TENANT-1, the same
    // collection-lifecycle work that gates the document/entity slices). Until then a named-tenant
    // graph must be provisioned under the scoped key explicitly (as the isolation tests do).
    if tenant == proximadb_tenant::DEFAULT_TENANT {
        return Ok(graph_id.to_string());
    }
    Ok(format!("{tenant}/{graph_id}"))
}

/// A tenant-scoped view over [`GraphOperationsService`].
///
/// The network layer obtains one per request via [`GraphOperationsService::for_tenant`],
/// passes tenant-CLEAN `graph_id`s, and this handle composes the structural scope
/// (`{tenant}/{graph_id}`, via [`scoped_graph_id`]) exactly ONCE before delegating to the
/// untouched raw methods. Isolation is therefore structural and composed at the boundary —
/// never a per-handler `{tenant}::{graph_id}` name predicate baked into user-visible names.
///
/// This is additive: the raw `GraphOperationsService` methods keep taking a bare `graph_id`,
/// so the ~50 internal/embedded/transaction callers (which have no request tenant) are
/// unaffected. Only request-scoped network surfaces route through this handle. The isolation
/// mechanism is preserved verbatim (a distinct engine per `(tenant, graph)` via the engine
/// registry); only the composition location (here, not each handler) and the separator
/// (`/` not `::`) change.
pub struct TenantGraphOps<'a> {
    inner: &'a GraphOperationsService,
    tenant: String,
}

impl GraphOperationsService {
    /// Obtain a [`TenantGraphOps`] that scopes every operation to `tenant`. Network handlers
    /// call `self.graph.for_tenant(&tenant).create_node(clean_graph_id, node)` etc.
    pub fn for_tenant(&self, tenant: &str) -> TenantGraphOps<'_> {
        TenantGraphOps {
            inner: self,
            tenant: tenant.to_string(),
        }
    }
}

impl TenantGraphOps<'_> {
    /// The single structural composition point for this handle: validate the tenant and
    /// render `{tenant}/{graph_id}`. Fails closed (invalid tenant → `InvalidInput`).
    fn scope(&self, graph_id: &str) -> Result<String> {
        scoped_graph_id(&self.tenant, graph_id)
            .map_err(|e| ProximaDBError::InvalidInput(e.to_string()))
    }

    // ── Node ops ────────────────────────────────────────────────────────────
    pub async fn create_node(&self, graph_id: &str, node: Node) -> Result<Arc<Node>> {
        self.inner.create_node(&self.scope(graph_id)?, node).await
    }
    pub async fn get_node(&self, graph_id: &str, id: &NodeId) -> Result<Option<Arc<Node>>> {
        self.inner.get_node(&self.scope(graph_id)?, id).await
    }
    pub async fn update_node(&self, graph_id: &str, node: Node) -> Result<Arc<Node>> {
        self.inner.update_node(&self.scope(graph_id)?, node).await
    }
    pub async fn delete_node(&self, graph_id: &str, id: &NodeId) -> Result<Option<Arc<Node>>> {
        self.inner.delete_node(&self.scope(graph_id)?, id).await
    }
    pub async fn get_neighbors(&self, graph_id: &str, node_id: &NodeId) -> Result<Vec<Arc<Node>>> {
        self.inner
            .get_neighbors(&self.scope(graph_id)?, node_id)
            .await
    }
    pub async fn query_nodes(&self, graph_id: &str, query: NodeQuery) -> Result<Vec<Arc<Node>>> {
        self.inner.query_nodes(&self.scope(graph_id)?, query).await
    }
    pub async fn get_nodes(
        &self,
        graph_id: &str,
        ids: &[String],
    ) -> Result<Vec<Option<Arc<Node>>>> {
        self.inner.get_nodes(&self.scope(graph_id)?, ids).await
    }

    // ── Edge ops ────────────────────────────────────────────────────────────
    pub async fn create_edge(&self, graph_id: &str, edge: Edge) -> Result<Arc<Edge>> {
        self.inner.create_edge(&self.scope(graph_id)?, edge).await
    }
    pub async fn update_edge(&self, graph_id: &str, edge: Edge) -> Result<Arc<Edge>> {
        self.inner.update_edge(&self.scope(graph_id)?, edge).await
    }
    pub async fn delete_edge(&self, graph_id: &str, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        self.inner.delete_edge(&self.scope(graph_id)?, id).await
    }
    pub async fn get_edge(&self, graph_id: &str, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        self.inner.get_edge(&self.scope(graph_id)?, id).await
    }
    pub async fn query_edges(&self, graph_id: &str, query: EdgeQuery) -> Result<Vec<Arc<Edge>>> {
        self.inner.query_edges(&self.scope(graph_id)?, query).await
    }

    // ── Traversal / algorithms ──────────────────────────────────────────────
    pub async fn traverse(
        &self,
        graph_id: &str,
        request: TraversalRequest,
    ) -> Result<TraversalResponse> {
        self.inner.traverse(&self.scope(graph_id)?, request).await
    }
    // NOTE: `shortest_path` is intentionally NOT forwarded here. Its raw signature names a v1
    // proto type (`ShortestPathAlgorithm`), and duplicating that path in a forwarder adds a
    // net-new legacy-v1-proto reference that trips the TD-123 ratchet. Its single caller (the v2
    // gRPC handler) instead composes the scope via [`scoped_graph_id`] and calls the raw method
    // directly — same structural key, no extra legacy-proto reference.
    pub async fn connected_components(&self, graph_id: &str) -> Result<Vec<Vec<NodeId>>> {
        self.inner
            .connected_components(&self.scope(graph_id)?)
            .await
    }
    pub async fn has_cycle(&self, graph_id: &str) -> Result<bool> {
        self.inner.has_cycle(&self.scope(graph_id)?).await
    }
    pub async fn get_stats(&self, graph_id: &str) -> Result<GraphStats> {
        self.inner.get_stats(&self.scope(graph_id)?).await
    }

    // ── Batch ───────────────────────────────────────────────────────────────
    pub async fn batch_create_nodes(
        &self,
        graph_id: &str,
        nodes: Vec<Node>,
    ) -> Result<Vec<Arc<Node>>> {
        self.inner
            .batch_create_nodes(&self.scope(graph_id)?, nodes)
            .await
    }
    pub async fn batch_create_nodes_with_strategy(
        &self,
        graph_id: &str,
        nodes: Vec<Node>,
        if_exists: &str,
    ) -> Result<Vec<Arc<Node>>> {
        self.inner
            .batch_create_nodes_with_strategy(&self.scope(graph_id)?, nodes, if_exists)
            .await
    }
    pub async fn batch_create_edges(
        &self,
        graph_id: &str,
        edges: Vec<Edge>,
    ) -> Result<crate::graph::service::BatchCreateEdgesResult> {
        self.inner
            .batch_create_edges(&self.scope(graph_id)?, edges)
            .await
    }

    // ── Constraints ─────────────────────────────────────────────────────────
    pub async fn add_unique_constraint(
        &self,
        graph_id: &str,
        label: &str,
        property: &str,
    ) -> Result<()> {
        self.inner
            .add_unique_constraint(&self.scope(graph_id)?, label, property)
            .await
    }
    pub async fn remove_unique_constraint(
        &self,
        graph_id: &str,
        label: &str,
        property: &str,
    ) -> Result<()> {
        self.inner
            .remove_unique_constraint(&self.scope(graph_id)?, label, property)
            .await
    }
}

/// Graph operation mode for flexible deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationMode {
    /// Graph operations only
    GraphOnly,
    /// Vector operations only (graph operations return errors)
    VectorOnly,
    /// Both graph and vector operations available
    Unified,
}

pub use proximadb_orion_engine::GraphMemoryPool;

#[cfg(test)]
mod tests {
    use super::*;
    // PropertyValue is now a struct, not enum - use direct field access;

    #[test]
    fn test_memory_pool_creation() {
        let pool = GraphMemoryPool::new();
        assert_eq!(pool.node_count(), 0);
        assert_eq!(pool.edge_count(), 0);
    }

    #[test]
    fn test_node_operations() {
        let pool = GraphMemoryPool::new();

        // Create a test node
        let node = Node {
            id: "node1".to_string(),
            labels: vec!["Person".to_string()],
            properties: std::collections::HashMap::from([(
                "name".to_string(),
                PropertyValue {
                    value: Some(Value::StringValue("Alice".to_string())),
                },
            )]),
            embedding: None,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        };

        // Insert node
        let node_arc = pool.insert_node(node);
        assert_eq!(pool.node_count(), 1);

        // Get node
        let retrieved = pool.get_node(&"node1".to_string()).unwrap();
        assert_eq!(retrieved.id, "node1");
        assert_eq!(retrieved.labels[0], "Person");

        // Verify Arc sharing (same pointer)
        assert!(Arc::ptr_eq(&node_arc, &retrieved));

        // Remove node
        let removed = pool.remove_node(&"node1".to_string()).unwrap();
        assert_eq!(removed.id, "node1");
        assert_eq!(pool.node_count(), 0);
    }

    #[test]
    fn test_property_value_to_string() {
        let string_val = PropertyValue {
            value: Some(Value::StringValue("test".to_string())),
        };
        assert_eq!(property_value_to_string(&string_val), "test");

        let int_val = PropertyValue {
            value: Some(Value::IntValue(42)),
        };
        assert_eq!(property_value_to_string(&int_val), "42");

        let bool_val = PropertyValue {
            value: Some(Value::BoolValue(true)),
        };
        assert_eq!(property_value_to_string(&bool_val), "true");
    }
}
