// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! gRPC V2 fusion service — a thin facade over the shared `FusionService` port.
//!
//! This is the canonical cross-modal retrieval surface over gRPC
//! (`SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`): `seed → expand → calibrate →
//! fuse-by-oid → rank`. It mirrors the REST `POST /api/v2/graphs/{id}/fusion-search`
//! and owns **no** retrieval/ranking logic — it delegates to `FusionService`,
//! the single internal fusion port (vector seed + graph expansion + PIT-calibrated
//! fuse-by-`oid`). Tenant isolation is structural: the backing collections are
//! tenant-namespaced via `x-tenant-id`, never a per-query predicate.

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::debug;

use crate::core::search::cross_modal_fusion::FusionPolicy;
use crate::network::grpc::auth as grpc_auth;
use crate::proto::proximadb_v2 as pv2;
use crate::proto::proximadb_v2::proxima_fusion_service_server::{
    ProximaFusionService, ProximaFusionServiceServer,
};
use crate::services::fusion_service::{
    DocumentFusionSpec, FusionOidKey, FusionService, GraphFusionParams, GraphGrain,
};

/// Defaults shared with the REST handler (`src/network/rest/v2/graphs.rs`).
const DEFAULT_LIMIT: usize = 10;
const DEFAULT_DEPTH: u32 = 1;
const DEFAULT_MAX_SEEDS: usize = 5;

/// gRPC V2 fusion service — thin delegating facade over `FusionService`.
pub struct ProximaFusionServiceImpl {
    fusion: Arc<FusionService>,
}

impl ProximaFusionServiceImpl {
    /// Build from the concrete vector + graph backing services plus the shared
    /// full-text index map (TD-138: powers the document modality). The
    /// `FusionService` port is constructed once here and shared across requests —
    /// never per-RPC.
    pub fn new(
        vector: Arc<crate::services::VectorOperationsService>,
        graph: Arc<crate::graph::GraphOperationsService>,
        fulltext_indexes: crate::network::hybrid_search::HybridFullTextIndexMap,
    ) -> Self {
        Self {
            fusion: Arc::new(FusionService::new(vector, graph, fulltext_indexes)),
        }
    }

    /// Convert to a tonic server.
    pub fn into_server(self) -> ProximaFusionServiceServer<Self> {
        ProximaFusionServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl ProximaFusionService for ProximaFusionServiceImpl {
    async fn fusion_search(
        &self,
        request: Request<pv2::FusionSearchRequest>,
    ) -> Result<Response<pv2::FusionSearchResponse>, Status> {
        // Tenant isolation is structural: resolve the deployment-aware identity
        // before either backing collection can be reached.
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let principal = grpc_auth::user_id(&request);
        // TD-TENANT-1 PR2: the tenant's stable u64 id (stamped by the gRPC auth
        // layer) for ABAC per-record enforcement.
        let tenant_stable_id = grpc_auth::tenant_stable_id(&request);
        let req = request.into_inner();

        if req.query_vector.is_empty() {
            return Err(Status::invalid_argument("query_vector must not be empty"));
        }
        if req.graph_id.trim().is_empty() {
            return Err(Status::invalid_argument("graph_id is required"));
        }
        if req.vector_collection.trim().is_empty() {
            return Err(Status::invalid_argument("vector_collection is required"));
        }

        // Shared server top_k cap (TD-SECSCAN-1 §5): same bound on the fusion
        // result limit and the graph-expansion seed count.
        proximadb_search_types::top_k_limit::check_top_k(u64::from(req.limit))
            .map_err(|e| Status::invalid_argument(format!("limit: {e}")))?;
        proximadb_search_types::top_k_limit::check_top_k(u64::from(req.max_seeds))
            .map_err(|e| Status::invalid_argument(format!("max_seeds: {e}")))?;

        debug!(
            tenant_id = ?tenant_id,
            graph_id = %req.graph_id,
            vector_collection = %req.vector_collection,
            "v2 gRPC FusionSearch"
        );

        // Fusion policy: PIT-calibrated linear (default) or rank-based RRF fallback.
        let mut policy = if req.rrf {
            FusionPolicy::rrf()
        } else {
            FusionPolicy::default()
        };
        if let Some(beta) = req.consensus_beta {
            policy.consensus_beta = beta;
        }

        let grain = match req.grain {
            2 => GraphGrain::Edges,
            3 => GraphGrain::Both,
            _ => GraphGrain::Nodes, // UNSPECIFIED (0) and NODES (1)
        };

        let params = GraphFusionParams {
            route_policy: None,
            graph_id: req.graph_id.clone(),
            // Structural tenant boundary — scopes both fusion legs (TD-ENTITY-TENANT-1).
            tenant: Some(tenant_id.clone()),
            // TD-TENANT-1 PR2: tenant stable id from the gRPC auth context
            // (stamped by the auth layer via the wired TenantStableIdResolver).
            tenant_stable_id,
            vector_collection: req.vector_collection.clone(),
            query_vector: req.query_vector.clone(),
            max_depth: if req.max_depth == 0 {
                DEFAULT_DEPTH
            } else {
                req.max_depth
            },
            edge_types: req.edge_types.clone(),
            max_seeds: if req.max_seeds == 0 {
                DEFAULT_MAX_SEEDS
            } else {
                req.max_seeds as usize
            },
            limit: if req.limit == 0 {
                DEFAULT_LIMIT
            } else {
                req.limit as usize
            },
            vector_weight: req.vector_weight.unwrap_or(1.0),
            graph_weight: req.graph_weight.unwrap_or(1.0),
            grain,
            // Within-tenant `permitted_principals` RBAC (TD-134): thread the caller's principal
            // so the seam's RBAC gate filters records. `None` (unauthenticated) ⇒ fails open
            // (structural isolation preserved).
            principal,
            policy,
            oid_key: FusionOidKey::Canonical,
            // TD-138 document modality (gRPC parity with REST #508): enable iff a non-empty
            // `text_query` was supplied. The shared service's DocumentExpander runs the BM25
            // search; absent/empty ⇒ vector+graph only (unchanged).
            document: req
                .text_query
                .as_ref()
                .filter(|t| !t.trim().is_empty())
                .map(|t| DocumentFusionSpec {
                    text_query: t.clone(),
                    collection: req.document_collection.clone(),
                    weight: req.document_weight.unwrap_or(1.0),
                    k: None,
                }),
        };

        let (items, stats) = self
            .fusion
            .graph_fusion_search(params)
            .await
            .map_err(|e| Status::internal(format!("fusion search failed: {e}")))?;

        let results = items
            .into_iter()
            .map(|item| pv2::FusionHit {
                oid: item.oid,
                score: item.score,
                source_count: item.source_count as u32,
            })
            .collect();

        Ok(Response::new(pv2::FusionSearchResponse {
            results,
            stats: Some(pv2::FusionStats {
                sources_fused: stats.sources_fused as u32,
                sources_skipped: stats.sources_skipped as u32,
                candidates_in: stats.candidates_in as u32,
                items_out: stats.items_out as u32,
            }),
        }))
    }
}
