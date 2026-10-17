//! Tenant scoping for the REST graph surface.
//!
//! Every REST graph handler reaches the graph subsystem through a
//! [`TenantScopedGraphPort`], obtained per request by the [`TenantGraph`]
//! extractor from the request tenant. The decorator composes the tenant-scoped
//! engine key (`{tenant}/{graph_id}`, default tenant bare) for every request, the
//! same composition the gRPC graph surface uses, so a REST request reaches only
//! its own tenant's graphs. A graph id carrying a `/` is rejected before it
//! reaches the graph subsystem. A request without a tenant is refused.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::{extract::FromRequestParts, http::request::Parts};
use proximadb_proto::v1::{
    BatchEdgeRequest, BatchNodeRequest, BatchResponse, ConnectedComponentsResponse,
    CreateEdgeRequest, CreateGraphRequest, CreateNodeRequest, CycleCheckResponse,
    DeleteEdgeRequest, DeleteNodeRequest, Edge, EdgeQuery, GetEdgeRequest, GetNeighborsRequest,
    GetNodeRequest, GetStatsRequest, GraphCollection, GraphQueryRequest, GraphQueryResponse,
    GraphSchema, GraphStats, HybridSearchRequest, HybridSearchResponse, Node, NodeQuery,
    ShortestPathRequest, ShortestPathResponse, TraversalChunk, TraversalRequest, TraversalResponse,
    UniqueConstraintRequest, UniqueConstraintResponse, UpdateEdgeRequest, UpdateNodeRequest,
};
use proximadb_runtime::GraphPort;

use super::graph::GraphRestState;
use crate::rest::errors::RestError;
use crate::rest::state::TenantContext;

/// A [`GraphPort`] that addresses only `tenant`'s graphs.
pub struct TenantScopedGraphPort {
    inner: Arc<dyn GraphPort>,
    tenant: String,
}

impl TenantScopedGraphPort {
    pub fn new(inner: Arc<dyn GraphPort>, tenant: impl Into<String>) -> Self {
        Self {
            inner,
            tenant: tenant.into(),
        }
    }

    /// The engine key for a tenant-clean graph id.
    fn key(&self, graph_id: &str) -> Result<String> {
        proximadb_tenant::scoped_resource_key(&self.tenant, graph_id)
            .map_err(|e| anyhow!("invalid graph id: {e}"))
    }

    /// Rewrite a request's graph id to the tenant's engine key in place.
    fn scope(&self, graph_id: &mut String) -> Result<()> {
        *graph_id = self.key(graph_id)?;
        Ok(())
    }

    /// Present a collection under its tenant-clean id; `None` when it is not
    /// `tenant`'s.
    fn unscope(&self, mut collection: GraphCollection) -> Option<GraphCollection> {
        let clean = proximadb_tenant::unscoped_resource_name(&self.tenant, &collection.graph_id)?
            .to_string();
        // A name defaulted from the engine key shows the clean id too.
        if collection.name == collection.graph_id {
            collection.name = clean.clone();
        }
        collection.graph_id = clean;
        Some(collection)
    }

    fn unscope_owned(&self, collection: GraphCollection) -> Result<GraphCollection> {
        self.unscope(collection)
            .ok_or_else(|| anyhow!("graph collection not found"))
    }
}

#[async_trait]
impl GraphPort for TenantScopedGraphPort {
    async fn create_node(&self, mut request: CreateNodeRequest) -> Result<Node> {
        self.scope(&mut request.graph_id)?;
        self.inner.create_node(request).await
    }
    async fn get_node(&self, mut request: GetNodeRequest) -> Result<Node> {
        self.scope(&mut request.graph_id)?;
        self.inner.get_node(request).await
    }
    async fn update_node(&self, mut request: UpdateNodeRequest) -> Result<Node> {
        self.scope(&mut request.graph_id)?;
        self.inner.update_node(request).await
    }
    async fn delete_node(&self, mut request: DeleteNodeRequest) -> Result<Node> {
        self.scope(&mut request.graph_id)?;
        self.inner.delete_node(request).await
    }
    async fn create_edge(&self, mut request: CreateEdgeRequest) -> Result<Edge> {
        self.scope(&mut request.graph_id)?;
        self.inner.create_edge(request).await
    }
    async fn get_edge(&self, mut request: GetEdgeRequest) -> Result<Edge> {
        self.scope(&mut request.graph_id)?;
        self.inner.get_edge(request).await
    }
    async fn update_edge(&self, mut request: UpdateEdgeRequest) -> Result<Edge> {
        self.scope(&mut request.graph_id)?;
        self.inner.update_edge(request).await
    }
    async fn delete_edge(&self, mut request: DeleteEdgeRequest) -> Result<Edge> {
        self.scope(&mut request.graph_id)?;
        self.inner.delete_edge(request).await
    }
    async fn query_nodes(&self, mut request: NodeQuery) -> Result<BatchResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.query_nodes(request).await
    }
    async fn query_edges(&self, mut request: EdgeQuery) -> Result<BatchResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.query_edges(request).await
    }
    async fn execute_query(&self, mut request: GraphQueryRequest) -> Result<GraphQueryResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.execute_query(request).await
    }
    async fn execute_query_json(
        &self,
        graph_id: &str,
        query: &str,
    ) -> Result<Vec<serde_json::Value>> {
        let key = self.key(graph_id)?;
        self.inner.execute_query_json(&key, query).await
    }
    async fn get_neighbors(&self, mut request: GetNeighborsRequest) -> Result<BatchResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.get_neighbors(request).await
    }
    async fn traverse_graph(&self, mut request: TraversalRequest) -> Result<TraversalResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.traverse_graph(request).await
    }
    async fn stream_traverse(&self, mut request: TraversalRequest) -> Result<Vec<TraversalChunk>> {
        self.scope(&mut request.graph_id)?;
        self.inner.stream_traverse(request).await
    }
    async fn get_graph_stats(&self, mut request: GetStatsRequest) -> Result<GraphStats> {
        self.scope(&mut request.graph_id)?;
        self.inner.get_graph_stats(request).await
    }
    async fn shortest_path(
        &self,
        mut request: ShortestPathRequest,
    ) -> Result<ShortestPathResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.shortest_path(request).await
    }
    async fn get_connected_components(
        &self,
        mut request: GetStatsRequest,
    ) -> Result<ConnectedComponentsResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.get_connected_components(request).await
    }
    async fn has_cycle(&self, mut request: GetStatsRequest) -> Result<CycleCheckResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.has_cycle(request).await
    }
    async fn add_unique_constraint(
        &self,
        mut request: UniqueConstraintRequest,
    ) -> Result<UniqueConstraintResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.add_unique_constraint(request).await
    }
    async fn remove_unique_constraint(
        &self,
        mut request: UniqueConstraintRequest,
    ) -> Result<UniqueConstraintResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.remove_unique_constraint(request).await
    }
    async fn batch_create_nodes(&self, mut request: BatchNodeRequest) -> Result<BatchResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.batch_create_nodes(request).await
    }
    async fn batch_create_edges(&self, mut request: BatchEdgeRequest) -> Result<BatchResponse> {
        self.scope(&mut request.graph_id)?;
        self.inner.batch_create_edges(request).await
    }
    /// The deprecated v1 hybrid query is not tenant-scoped; it is not served
    /// through a tenant's view.
    async fn execute_hybrid_query(
        &self,
        _request: HybridSearchRequest,
    ) -> Result<HybridSearchResponse> {
        Err(anyhow!(
            "v1 hybrid graph query is not available on this surface; use v2 FusionSearch"
        ))
    }
    async fn create_graph_collection(
        &self,
        mut request: CreateGraphRequest,
    ) -> Result<GraphCollection> {
        self.scope(&mut request.graph_id)?;
        let created = self.inner.create_graph_collection(request).await?;
        self.unscope_owned(created)
    }
    async fn get_graph_collection(&self, graph_id: String) -> Result<Option<GraphCollection>> {
        let key = self.key(&graph_id)?;
        Ok(self
            .inner
            .get_graph_collection(key)
            .await?
            .and_then(|c| self.unscope(c)))
    }
    async fn delete_graph_collection(&self, graph_id: String) -> Result<bool> {
        let key = self.key(&graph_id)?;
        self.inner.delete_graph_collection(key).await
    }
    async fn list_graph_collections(&self) -> Result<Vec<GraphCollection>> {
        Ok(self
            .inner
            .list_graph_collections()
            .await?
            .into_iter()
            .filter_map(|c| self.unscope(c))
            .collect())
    }
    async fn update_graph_schema(
        &self,
        graph_id: String,
        schema: GraphSchema,
    ) -> Result<GraphCollection> {
        let key = self.key(&graph_id)?;
        let updated = self.inner.update_graph_schema(key, schema).await?;
        self.unscope_owned(updated)
    }
}

/// Extractor giving a handler the graph state for the request tenant only.
///
/// Reads the tenant the tenant middleware resolved; a request without one is
/// refused rather than served unscoped.
pub struct TenantGraph(pub GraphRestState);

impl TenantGraph {
    /// The graph state as `tenant` sees it.
    pub fn for_tenant(state: &GraphRestState, tenant: &str) -> Self {
        Self(GraphRestState {
            graph_port: Arc::new(TenantScopedGraphPort::new(state.graph_port.clone(), tenant)),
        })
    }
}

impl FromRequestParts<GraphRestState> for TenantGraph {
    type Rejection = RestError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &GraphRestState,
    ) -> Result<Self, Self::Rejection> {
        let tenant = parts
            .extensions
            .get::<TenantContext>()
            .map(|t| t.tenant_id.clone())
            .filter(|t| !t.is_empty())
            .ok_or_else(|| RestError::Unauthorized("request tenant is required".to_string()))?;
        Ok(Self::for_tenant(state, &tenant))
    }
}
