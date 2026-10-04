//! Fusion search port (ADR-094) — the shared retrieval seam for graph-modality
//! fusion, owned by the runtime so REST/gRPC handlers stop constructing the
//! concrete `FusionService` params themselves (4 duplicate construction sites).
//!
//! DTOs are runtime-owned and mirror `GraphFusionParams` minus the identity
//! fields: `PortIdentity` carries tenant/principal/stable-id (1:1 with
//! `GraphFusionParams::{tenant, principal, tenant_stable_id}`). Policy/stats
//! types re-export from the std-only `proximadb-cross-modal-fusion` crate.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

pub use proximadb_cross_modal_fusion::cross_modal_fusion::{FusedItem, FusionPolicy, FusionStats};

use crate::service_ports::PortIdentity;

/// How source oids map to the fusion key (TD-142 / TD-146 scope B).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FusionOidKey {
    /// Vector + graph share the canonical oid `graph/{graph_id}/node/{node_id}`. Identity keying.
    #[default]
    Canonical,
    /// Entity keying: vector oids are `{node_id}/{model_id}`, graph oids are
    /// `graph/{graph_id}/node/{node_id}`. Both normalize to the entity `node_id`.
    EntityNode,
}

impl FusionOidKey {
    /// Fusion key for a source oid (the value the Fuser merges by).
    pub fn fusion_key(&self, oid: &str) -> String {
        match self {
            Self::Canonical => oid.to_string(),
            Self::EntityNode => entity_node_id_from_oid(oid).to_string(),
        }
    }

    /// Graph `node_id` to seed traversal, recovered from a vector-hit oid.
    pub fn seed_node_id(&self, graph_id: &str, hit_id: &str) -> String {
        match self {
            Self::Canonical => {
                let prefix = format!("graph/{graph_id}/node/");
                hit_id
                    .strip_prefix(&prefix)
                    .map(str::to_string)
                    .unwrap_or_else(|| hit_id.to_string())
            }
            Self::EntityNode => entity_node_id_from_oid(hit_id).to_string(),
        }
    }
}

/// Recover the entity `node_id` from either oid form: the canonical graph oid
/// `graph/{graph_id}/node/{node_id}` or the auxiliary vector oid `{node_id}/{model_id}`.
fn entity_node_id_from_oid(oid: &str) -> &str {
    if let Some(rest) = oid.strip_prefix("graph/")
        && let Some((_gid, node_id)) = rest.split_once("/node/")
    {
        return node_id;
    }
    oid.rsplit_once('/')
        .map(|(node_id, _)| node_id)
        .unwrap_or(oid)
}

/// Whether graph expansion contributes node candidates, edge (relationship)
/// candidates, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphGrain {
    #[default]
    Nodes,
    Edges,
    Both,
}

/// Optional document-modality contribution to a fusion query (TD-138). When
/// present, the implementation runs BM25/full-text search over the collection's
/// in-memory index and emits an oid-keyed document source merged by shared oid.
#[derive(Debug, Clone)]
pub struct DocumentFusionSpec {
    /// The BM25/full-text query. Its presence is what enables the document source.
    pub text_query: String,
    /// Collection whose full-text index to search. `None` ⇒ the vector collection.
    pub collection: Option<String>,
    /// Document modality weight (mirrors `vector_weight` / `graph_weight`).
    pub weight: f32,
    /// Top-k documents to take from the index. `None` ⇒ reuse the query `limit`.
    pub k: Option<usize>,
}

/// Runtime-owned cost-routing policy (TD-141): budget each modality by weight
/// and drop negligible ones. `None` in the request ⇒ unbounded fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionRoutePolicy {
    pub min_weight_fraction: f32,
    pub total_budget: usize,
}

/// One graph-modality fusion request (ADR-094 port DTO). Identity travels via
/// `PortIdentity`: `tenant_id` is the STRUCTURAL isolation boundary threading
/// into both legs; `subject` is the within-tenant RBAC principal;
/// `tenant_stable_id` is the ABAC binding-filter key.
#[derive(Debug, Clone)]
pub struct GraphFusionRequest {
    pub graph_id: String,
    pub vector_collection: String,
    pub query_vector: Vec<f32>,
    pub max_depth: u32,
    pub edge_types: Vec<String>,
    /// How many of the top vector seeds to expand from (bounded — D8).
    pub max_seeds: usize,
    pub limit: usize,
    pub vector_weight: f32,
    pub graph_weight: f32,
    /// Node / edge / both grain for the graph contribution (D8).
    pub grain: GraphGrain,
    pub policy: FusionPolicy,
    /// How source oids map to the fusion key. Defaults to Canonical.
    pub oid_key: FusionOidKey,
    /// Optional document-modality contribution. `None` ⇒ vector+graph only.
    pub document: Option<DocumentFusionSpec>,
    /// Optional cost-routing policy. `None` ⇒ unbounded (each source keeps its
    /// full candidate pool).
    pub route_policy: Option<FusionRoutePolicy>,
}

/// Result of one fusion search.
#[derive(Debug, Clone)]
pub struct FusionSearchResult {
    pub items: Vec<FusedItem>,
    pub stats: FusionStats,
    /// Per-entity labels emitted by `graph_fusion_search_with_labels` (empty on
    /// the plain variant).
    pub labels: HashMap<String, Vec<String>>,
}

/// The shared retrieval seam every fusion surface delegates to
/// (`SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`): one retrieval engine, no
/// per-handler construction. Implemented by root `FusionService`.
#[async_trait]
pub trait FusionSearchPort: Send + Sync {
    async fn fusion_search(
        &self,
        request: GraphFusionRequest,
        identity: PortIdentity<'_>,
    ) -> Result<FusionSearchResult>;
}
