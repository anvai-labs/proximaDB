// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! gRPC V2 native entity service implementation.
//!
//! **Orchestration layer over existing services — NOT a separate storage path.**
//!
//! The EntityService orchestrates across three existing primitives:
//! - **GraphOperationsService** — entity-as-node with relations (topology)
//! - **VectorOperationsService** — embeddings (ANN search)
//! - **DocumentService** — provenance chunks (metadata)
//!
//! This avoids redundant storage: an "entity" is just a graph node with associated
//! vectors and optional document chunks. Tenant isolation is structural — the
//! request tenant (`x-tenant-id`, resolved via [`grpc_auth::resolved_tenant_id`]) is
//! applied per-leg with a tenant-CLEAN collection name: graph via `for_tenant`,
//! vector via `TenantContext`, provenance document via the scoped collection key.
//! Never a `{tenant}::collection` name fold, never a per-query predicate.

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, error, warn};

use crate::core::search::cross_modal_fusion::FusionPolicy;
use crate::graph::{
    Edge, GraphOperationsService, Node, NodeId, NodeQuery, PropertyFilter, PropertyValue,
    property_value::Value as GraphValue,
};
use crate::network::grpc::auth as grpc_auth;
use crate::proto::proximadb_v2 as pv2;
use crate::proto::proximadb_v2::proxima_entity_service_server::{
    ProximaEntityService, ProximaEntityServiceServer,
};
use crate::services::fusion_service::{FusionOidKey, FusionService, GraphFusionParams, GraphGrain};
use crate::services::operations::vectors::VectorOperationsService;
use crate::storage::document::{DocumentRecord, DocumentService};
use proximadb_records::{
    EmbeddingCell, EmbeddingScalarType, EmbeddingValues, ProximaRecord, ProximaTree,
    ProximaTreeNode, ProximaValue,
};

/// gRPC V2 native entity service — orchestration layer over graph + vector + document.
pub struct ProximaEntityServiceImpl {
    /// Graph service for entity-as-node and relations
    graph_service: Arc<GraphOperationsService>,
    /// Vector service for embeddings
    vector_service: Arc<VectorOperationsService>,
    /// Cross-modal fusion port — the single retrieval engine
    /// (`SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`). SearchEntities delegates its
    /// vector (`similar`) mode here rather than reimplementing retrieval.
    fusion_service: Arc<FusionService>,
    /// Document service for provenance/evidence chunks
    document_service: Arc<DocumentService>,
}

impl ProximaEntityServiceImpl {
    /// Create a new service from its concrete backing services (production wiring).
    /// The fusion port is built once here from the vector + graph services
    /// (mirrors the REST AppState pattern, PR #282).
    pub fn new(
        graph: Arc<GraphOperationsService>,
        vector: Arc<VectorOperationsService>,
        document: Arc<DocumentService>,
    ) -> Self {
        Self {
            graph_service: graph.clone(),
            vector_service: vector.clone(),
            fusion_service: Arc::new(FusionService::new(
                vector,
                graph,
                crate::network::hybrid_search::HybridFullTextIndexMap::default(),
            )),
            document_service: document,
        }
    }

    /// Convert to a tonic server.
    pub fn into_server(self) -> ProximaEntityServiceServer<Self> {
        ProximaEntityServiceServer::new(self)
    }

    /// Generate a unique node ID for this entity in the graph.
    fn entity_node_id(collection_id: &str, entity_id: &str) -> NodeId {
        format!("entity:{collection_id}:{entity_id}")
    }

    /// Generate a unique auxiliary ID (embedding vector / provenance document)
    /// for an entity. The entity **node id** is the recoverable prefix (split on
    /// the last `/`), so fusion results — which carry only the vector `oid` —
    /// project back to their entity node without a re-fetch. The `/` delimiter
    /// matches the codebase's canonical-oid convention (`graph/{id}/node/{id}`).
    fn auxiliary_id(collection_id: &str, entity_id: &str, model_id: &str) -> String {
        format!(
            "{}/{model_id}",
            Self::entity_node_id(collection_id, entity_id)
        )
    }

    /// Recover the entity node id from an auxiliary (vector/provenance) oid.
    /// Inverse of [`Self::auxiliary_id`].
    fn node_id_from_auxiliary_oid(oid: &str) -> &str {
        oid.rsplit_once('/')
            .map(|(node_id, _)| node_id)
            .unwrap_or(oid)
    }

    /// Server-side embedding hop (TD-146 residual): embed a text query via the global
    /// [`proximadb_embedding::EmbeddingService`] so `similar.text` / `similar.raw_data`
    /// can drive the fusion path without the client pre-embedding. Routes through the
    /// deployment's configured embedding model (tenant route).
    ///
    /// Correctness caveat: results are meaningful when the entity's stored embeddings
    /// use the same model — the same multi-model caveat the `similar.vector` path
    /// already carries for multi-model collections.
    async fn embed_query(text: &str, tenant_id: &str) -> Result<Vec<f32>, Status> {
        let service = proximadb_embedding::EmbeddingService::try_global().ok_or_else(|| {
            Status::unavailable(
                "embedding service is not configured; cannot embed similar.text/raw_data \
                 (use similar.vector with a client-side embedding instead)",
            )
        })?;
        let result = service
            .embed_sync(proximadb_embedding::EmbedBatch {
                records: vec![proximadb_embedding::EmbedRecord {
                    id: "entity-search-query".to_string(),
                    text: text.to_string(),
                    tenant_id: tenant_id.to_string(),
                }],
                mode: proximadb_embedding::IngestMode::Sync,
            })
            .await
            .map_err(|e| Status::internal(format!("embedding hop failed: {e}")))?;
        result
            .vectors
            .into_iter()
            .next()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| Status::internal("embedding service returned an empty vector"))
    }
}

/// Map an internal error to a gRPC status.
fn entity_status(operation: &str, err: impl std::fmt::Display) -> Status {
    let message = err.to_string();
    let lower = message.to_lowercase();
    if lower.contains("not found") {
        Status::not_found(message)
    } else if lower.contains("already exists") {
        Status::already_exists(message)
    } else if lower.contains("invalid") || lower.contains("required") {
        Status::invalid_argument(message)
    } else {
        Status::internal(format!("{operation}: {message}"))
    }
}

#[tonic::async_trait]
impl ProximaEntityService for ProximaEntityServiceImpl {
    async fn upsert_entity(
        &self,
        request: Request<pv2::UpsertEntityRequest>,
    ) -> Result<Response<pv2::UpsertEntityResponse>, Status> {
        // Tenant-CLEAN collection name; each leg is scoped structurally by the resolved tenant.
        let collection = request.get_ref().collection_id.clone();
        let tenant = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();
        let entity = req
            .entity
            .ok_or_else(|| Status::invalid_argument("upsert_entity: `entity` is required"))?;
        let entity_id = if entity.id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            entity.id.clone()
        };

        debug!("v2 gRPC UpsertEntity collection={collection} entity_id={entity_id}");

        // Step 1: Create/update the graph node (authoritative entity record).
        let node_id = Self::entity_node_id(&collection, &entity_id);
        let mut node_properties = HashMap::new();
        for (k, v) in &entity.flexible_metadata {
            let pv = typed_value_to_property_value(v).map_err(|reason| {
                Status::invalid_argument(format!(
                    "upsert_entity: invalid flexible_metadata[{k}]: {reason}"
                ))
            })?;
            node_properties.insert(k.clone(), pv);
        }
        node_properties.insert("_entity_type".to_string(), str_property("entity"));
        node_properties.insert("_collection_id".to_string(), str_property(&collection));

        let now_ms = chrono::Utc::now().timestamp_millis();
        let node = Node {
            id: node_id.clone(),
            labels: vec!["entity".to_string()],
            properties: node_properties,
            embedding: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        // Upsert semantics: try create, fall back to update if the node exists.
        match self
            .graph_service
            .for_tenant(&tenant)
            .create_node(&collection, node.clone())
            .await
        {
            Ok(_) => debug!("Created graph node {node_id}"),
            Err(create_err) => {
                if let Err(update_err) = self
                    .graph_service
                    .for_tenant(&tenant)
                    .update_node(&collection, node)
                    .await
                {
                    error!("Entity node upsert failed: create={create_err}; update={update_err}");
                    return Err(entity_status("upsert entity", update_err));
                }
                debug!("Updated graph node {node_id}");
            }
        }

        // Step 2: Upsert embedding vectors (best-effort; non-fatal on failure).
        for embedding in &entity.embeddings {
            let vector_id = Self::auxiliary_id(&collection, &entity_id, &embedding.model_id);
            let mut props = ProximaTree::new();
            props.insert(
                "_entity_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(entity_id.clone())),
            );
            props.insert(
                "_entity_collection".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(collection.clone())),
            );
            props.insert(
                "_embedding_model_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(embedding.model_id.clone())),
            );

            let record = ProximaRecord {
                oid: vector_id.clone(),
                tenant_id: tenant.clone(),
                local_id: Some(format!("{entity_id}:{}", embedding.model_id)),
                embeddings: vec![EmbeddingCell {
                    model_id: embedding.model_id.clone(),
                    modality: modality_to_string(embedding.modality),
                    values: EmbeddingValues::Fp32(embedding.vector.clone()),
                    dim: embedding.dimension,
                    precision: EmbeddingScalarType::Fp32,
                    precision_epoch: None,
                }],
                props,
                ..Default::default()
            };

            debug!(
                "Upsert embedding model={} dim={} vector_id={vector_id}",
                embedding.model_id,
                embedding.vector.len()
            );

            match self
                .vector_service
                .insert_batch_with_tenant_context(
                    &collection,
                    vec![record],
                    Some(&crate::storage::tenant::context::TenantContext::for_tenant_id(&tenant)),
                )
                .await
            {
                Ok(result) => {
                    if !result.success {
                        warn!(
                            "Embedding insert non-success for {vector_id}: {:?}",
                            result.errors
                        );
                    }
                }
                Err(e) => warn!("Embedding insert failed (non-fatal) for {vector_id}: {e}"),
            }
        }

        // Step 3: Store provenance as a document chunk if present (best-effort).
        if let Some(provenance) = entity.provenance.as_ref() {
            let doc_id = Self::auxiliary_id(&collection, &entity_id, "provenance");
            let mut tree = ProximaTree::new();
            tree.insert(
                "source_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(provenance.source_id.clone())),
            );
            tree.insert(
                "chunk_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(provenance.chunk_id.clone())),
            );
            tree.insert(
                "chunk_position".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int32(provenance.chunk_position as i32)),
            );
            if !provenance.extraction_method.is_empty() {
                tree.insert(
                    "extraction_method".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(
                        provenance.extraction_method.clone(),
                    )),
                );
            }
            for (k, v) in &provenance.metadata {
                tree.insert(
                    k.clone(),
                    ProximaTreeNode::Value(ProximaValue::String(v.clone())),
                );
            }
            tree.insert(
                "_entity_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(entity_id.clone())),
            );
            tree.insert(
                "_entity_collection".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(collection.clone())),
            );

            // Scope the provenance doc collection to the tenant (both storage key and
            // record.collection_id key the same OID), matching the document fix. The tenant is
            // already validated (the graph create above would have failed otherwise).
            let scoped_doc_collection =
                crate::storage::document::service::scoped_document_collection(&tenant, &collection)
                    .map_err(|e| entity_status("scope provenance collection", e))?;
            let doc_record = DocumentRecord::from_tree(
                doc_id.clone(),
                tree,
                scoped_doc_collection.clone(),
                None,
                Some("entity_provenance".to_string()),
            );

            debug!("Insert provenance document doc_id={doc_id}");
            match self
                .document_service
                .insert_document_record(&scoped_doc_collection, doc_record)
                .await
            {
                Ok(_) => debug!("Provenance document inserted"),
                Err(e) => warn!("Provenance insert failed (non-fatal) for {doc_id}: {e}"),
            }
        }

        // Step 4: Create relation edges (best-effort; non-fatal on failure).
        for relation in &entity.relations {
            let source = Self::entity_node_id(&collection, &relation.source_entity_id);
            let target = Self::entity_node_id(&collection, &relation.target_entity_id);
            let edge_id = format!("{source}:{}->{target}", relation.relation_type);

            let mut edge_properties = HashMap::new();
            for (k, v) in &relation.properties {
                edge_properties.insert(k.clone(), str_property(v));
            }
            edge_properties.insert("_entity_relation".to_string(), str_property("true"));

            let edge = Edge {
                id: edge_id.clone(),
                from_node_id: source.clone(),
                to_node_id: target.clone(),
                edge_type: relation.relation_type.clone(),
                properties: edge_properties,
                weight: Some(relation.weight as f64),
                created_at_ms: now_ms,
                updated_at_ms: now_ms,
            };

            match self
                .graph_service
                .for_tenant(&tenant)
                .create_edge(&collection, edge)
                .await
            {
                Ok(_) => debug!("Created edge {edge_id}"),
                Err(e) => warn!("Edge create failed (non-fatal) for {edge_id}: {e}"),
            }
        }

        Ok(Response::new(pv2::UpsertEntityResponse {
            success: true,
            entity_id,
            message: "Entity upserted successfully".to_string(),
        }))
    }

    async fn get_entity(
        &self,
        request: Request<pv2::GetEntityRequest>,
    ) -> Result<Response<pv2::GetEntityResponse>, Status> {
        let collection = request.get_ref().collection_id.clone();
        let tenant = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();

        debug!(
            "v2 gRPC GetEntity collection={collection} entity_id={}",
            req.entity_id
        );

        let node_id = Self::entity_node_id(&collection, &req.entity_id);
        let node = self
            .graph_service
            .for_tenant(&tenant)
            .get_node(&collection, &node_id)
            .await
            .map_err(|e| entity_status("get entity", e))?
            .ok_or_else(|| Status::not_found(format!("Entity '{}' not found", req.entity_id)))?;

        Ok(Response::new(pv2::GetEntityResponse {
            entity: Some(node_to_entity(&node, &collection)),
        }))
    }

    async fn delete_entity(
        &self,
        request: Request<pv2::DeleteEntityRequest>,
    ) -> Result<Response<pv2::DeleteEntityResponse>, Status> {
        let collection = request.get_ref().collection_id.clone();
        let tenant = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();

        debug!(
            "v2 gRPC DeleteEntity collection={collection} entity_id={}",
            req.entity_id
        );

        let node_id = Self::entity_node_id(&collection, &req.entity_id);
        let deleted = self
            .graph_service
            .for_tenant(&tenant)
            .delete_node(&collection, &node_id)
            .await
            .map_err(|e| entity_status("delete entity", e))?
            .is_some();

        // NOTE: associated embedding vectors and provenance documents are not
        // automatically cascaded. They can be removed separately by the caller
        // using the auxiliary-id convention (`entity:{collection}:{entity_id}/{model}`).
        Ok(Response::new(pv2::DeleteEntityResponse {
            success: deleted,
            message: if deleted {
                "Entity deleted successfully".to_string()
            } else {
                "Entity not found".to_string()
            },
        }))
    }

    async fn search_entities(
        &self,
        request: Request<pv2::SearchEntitiesRequest>,
    ) -> Result<Response<pv2::SearchEntitiesResponse>, Status> {
        let collection = request.get_ref().collection_id.clone();
        let tenant = grpc_auth::resolved_tenant_id(&request)?;
        let principal = grpc_auth::user_id(&request);
        let req = request.into_inner();

        debug!(
            "v2 gRPC SearchEntities collection={collection} top_k={} has_vector={} has_filters={}",
            req.top_k,
            req.similar.is_some(),
            req.filters.is_some()
        );

        // Case 1: vector (`similar`) search — delegate to the fusion seam
        // (`SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`, TD-146). Graph-augmented fusion
        // is enabled (TD-146 scope B): the seam normalizes the auxiliary vector oid and
        // the canonical graph oid to the entity `node_id` (TD-142) so the vector and
        // graph sources co-rank. `similar.text`/`raw_data` are embedded server-side
        // (TD-146 residual) via the global EmbeddingService, then run the same path.
        if let Some(similar) = req.similar.as_ref() {
            let query_vector = match similar.query.as_ref() {
                Some(pv2::similar_query::Query::Vector(v)) => v.values.clone(),
                Some(pv2::similar_query::Query::Text(t)) => Self::embed_query(t, &tenant).await?,
                Some(pv2::similar_query::Query::RawData(bytes)) => {
                    // Treat raw_data as UTF-8 text to embed.
                    let text = String::from_utf8(bytes.clone()).map_err(|_| {
                        Status::invalid_argument(
                            "similar.raw_data must be valid UTF-8 text to embed",
                        )
                    })?;
                    Self::embed_query(&text, &tenant).await?
                }
                None => {
                    return Err(Status::invalid_argument(
                        "similar.query is required (vector | text | raw_data)",
                    ));
                }
            };
            if query_vector.is_empty() {
                return Err(Status::invalid_argument(
                    "similar.vector.values must not be empty",
                ));
            }
            // NOTE: combining `similar` with `filters` is not yet supported (needs
            // the index metadata-predicate mask, TD-139). Vector mode takes precedence.
            let limit = if req.top_k == 0 {
                10
            } else {
                req.top_k as usize
            };
            let params = GraphFusionParams {
                route_policy: None,
                graph_id: collection.clone(),
                vector_collection: collection.clone(),
                // Structural tenant boundary — scopes BOTH fusion legs (clean names in, TD-ENTITY-TENANT-1).
                tenant: Some(tenant.clone()),
                tenant_stable_id: None, // FA-2 PR-D3: gRPC resolver wiring deferred
                query_vector,
                // TD-146 scope B: graph-augmented entity fusion. Entity vectors live under the
                // auxiliary `{node_id}/{model_id}` oid; `EntityNode` keying normalizes both the
                // vector and graph sources to the entity `node_id` so they co-rank (TD-142).
                max_depth: 1, // expand direct entity neighbours
                edge_types: Vec::new(),
                max_seeds: limit,
                limit,
                vector_weight: 1.0,
                graph_weight: 0.3, // modest graph boost; vector similarity leads (tunable)
                grain: GraphGrain::Nodes,
                // Within-tenant `permitted_principals` RBAC (TD-134): thread the caller's
                // principal so the seam's RBAC gate filters records by `permitted_principals`.
                // `None` (unauthenticated request) ⇒ structural isolation only.
                principal,
                policy: FusionPolicy::default(),
                oid_key: FusionOidKey::EntityNode,
                // TD-138 document modality is REST-only for now; gRPC entity fusion is deferred.
                document: None,
            };

            let (items, _stats) = self
                .fusion_service
                .graph_fusion_search(params)
                .await
                .map_err(|e| entity_status("search entities (fusion)", e))?;

            // Project fused vector oids → entity nodes. The seam late-materializes
            // (oid + score only), so fetch the nodes for their metadata. Batched
            // (`get_nodes`) so the M result lookups collapse to ~1 round-trip of
            // latency for the cold misses (ADR-034 depth-collapse) instead of one
            // serial GET per result.
            let node_ids: Vec<String> = items
                .iter()
                .map(|item| Self::node_id_from_auxiliary_oid(&item.oid).to_owned())
                .collect();
            let nodes = self
                .graph_service
                .for_tenant(&tenant)
                .get_nodes(&collection, &node_ids)
                .await
                .map_err(|e| entity_status("materialize search entities", e))?;
            let mut results = Vec::with_capacity(items.len());
            for (item, node) in items.iter().zip(nodes) {
                if let Some(node) = node {
                    results.push(pv2::EntityResult {
                        entity: Some(node_to_entity(&node, &collection)),
                        score: item.score,
                        debug_info: HashMap::new(),
                    });
                }
            }
            let total = results.len() as u32;
            return Ok(Response::new(pv2::SearchEntitiesResponse {
                results,
                total,
                page_info: None,
                progress: None,
            }));
        }

        // Cases 2 & 3: metadata-filtered or unfiltered node scan.
        let limit = if req.top_k == 0 { 50 } else { req.top_k };
        let mut filters = Vec::new();
        if let Some(meta_filter) = req.filters.as_ref() {
            for clause in &meta_filter.clauses {
                if let Some(value) = filter_clause_to_property_value(clause) {
                    filters.push(PropertyFilter {
                        key: clause.field.clone(),
                        operator: entity_op_to_graph_op(clause.op),
                        value: Some(value),
                    });
                }
            }
        }

        let query = NodeQuery {
            graph_id: collection.clone(),
            labels: vec!["entity".to_string()],
            filters,
            limit: Some(limit),
            offset: None,
            continuation_token: None,
        };

        let nodes = self
            .graph_service
            .for_tenant(&tenant)
            .query_nodes(&collection, query)
            .await
            .map_err(|e| entity_status("search entities", e))?;

        let prefix = format!("entity:{collection}:");
        let results = nodes
            .into_iter()
            .filter(|n| n.id.starts_with(&prefix))
            .map(|n| pv2::EntityResult {
                entity: Some(node_to_entity(&n, &collection)),
                score: 0.0,
                debug_info: HashMap::new(),
            })
            .collect::<Vec<_>>();
        let total = results.len() as u32;

        Ok(Response::new(pv2::SearchEntitiesResponse {
            results,
            total,
            page_info: None,
            progress: None,
        }))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Build an entity-shaped v2 [`pv2::Entity`] from a graph node, stripping the
/// internal `_`-prefixed bookkeeping properties.
fn node_to_entity(node: &Node, collection: &str) -> pv2::Entity {
    let mut flexible_metadata = HashMap::new();
    for (k, v) in &node.properties {
        if !k.starts_with('_') {
            flexible_metadata.insert(k.clone(), property_value_to_typed_value(v));
        }
    }

    let prefix = format!("entity:{collection}:");
    let id = node
        .id
        .strip_prefix(&prefix)
        .unwrap_or(&node.id)
        .to_string();

    pv2::Entity {
        id,
        collection_id: collection.to_string(),
        flexible_metadata,
        embeddings: vec![],
        typed_metadata: None,
        provenance: None,
        relations: vec![],
        temporal: None,
    }
}

/// Create a string graph [`PropertyValue`].
fn str_property(s: impl Into<String>) -> PropertyValue {
    PropertyValue {
        value: Some(GraphValue::StringValue(s.into())),
    }
}

/// Map an [`pv2::EntityModality`] discriminant to a storage modality string.
fn modality_to_string(modality: i32) -> String {
    match modality {
        2 => "image",
        3 => "audio",
        4 => "video",
        5 => "multimodal",
        // UNSPECIFIED (0) and TEXT (1) both default to text.
        _ => "text",
    }
    .to_string()
}

/// Map an [`pv2::EntityComparisonOp`] discriminant to a graph
/// [`crate::graph::PropertyFilterOperator`] discriminant.
fn entity_op_to_graph_op(entity_op: i32) -> i32 {
    use crate::graph::PropertyFilterOperator as Op;
    let mapped = match entity_op {
        1 => Op::Equals,       // EQ
        2 => Op::NotEquals,    // NE
        3 => Op::GreaterThan,  // GT
        4 => Op::GreaterEqual, // GTE
        5 => Op::LessThan,     // LT
        6 => Op::LessEqual,    // LTE
        9 => Op::Contains,     // CONTAINS
        // IN / NOT_IN / UNSPECIFIED default to equals.
        _ => Op::Equals,
    };
    mapped as i32
}

/// Convert v2 `TypedValue` to graph `PropertyValue`.
fn typed_value_to_property_value(
    tv: &pv2::TypedValue,
) -> std::result::Result<PropertyValue, String> {
    let canonical = proximadb_records::proto_v2::typed_value_to_proxima(tv)
        .map_err(|error| error.to_string())?;
    Ok(crate::graph::adjacency_projection::proxima_value_to_property_value(&canonical))
}

/// Convert graph `PropertyValue` to v2 `TypedValue`.
fn property_value_to_typed_value(pv: &PropertyValue) -> pv2::TypedValue {
    let canonical = crate::graph::adjacency_projection::property_value_to_proxima(pv);
    proximadb_records::proto_v2::proxima_value_to_typed_value(&canonical)
}

/// Convert a v2 search `FilterClause` value to a graph `PropertyValue`.
fn filter_clause_to_property_value(clause: &pv2::FilterClause) -> Option<PropertyValue> {
    use pv2::filter_clause::Value as ClauseValue;

    let value = match clause.value.as_ref()? {
        ClauseValue::StringValue(s) => GraphValue::StringValue(s.clone()),
        ClauseValue::IntValue(i) => GraphValue::IntValue(*i),
        ClauseValue::DoubleValue(f) => GraphValue::DoubleValue(*f),
        ClauseValue::BoolValue(b) => GraphValue::BoolValue(*b),
    };

    Some(PropertyValue { value: Some(value) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_value_property_round_trip() {
        use pv2::typed_value::Value;

        let tv = pv2::TypedValue {
            declared_type: 0,
            value: Some(Value::TextValue("test".to_string())),
        };

        let pv = typed_value_to_property_value(&tv).unwrap();
        let back = property_value_to_typed_value(&pv);

        match back.value {
            Some(Value::TextValue(s)) => assert_eq!(s, "test"),
            other => panic!("expected TextValue, got {other:?}"),
        }

        let structured = ProximaValue::Map(HashMap::from([(
            "items".to_string(),
            ProximaValue::Array(vec![ProximaValue::Int64(7), ProximaValue::Null]),
        )]));
        let typed = proximadb_records::proto_v2::proxima_value_to_typed_value(&structured);
        let property = typed_value_to_property_value(&typed).unwrap();
        let round_tripped = property_value_to_typed_value(&property);
        let round_tripped =
            proximadb_records::proto_v2::typed_value_to_proxima(&round_tripped).unwrap();
        assert_eq!(
            proximadb_records::conversions::proxima_to_json(&round_tripped),
            proximadb_records::conversions::proxima_to_json(&structured),
            "v2 structured metadata and null children survive the graph boundary"
        );
    }

    #[test]
    fn json_null_property_is_parsed_and_invalid_json_is_rejected() {
        use pv2::typed_value::Value;

        let null = pv2::TypedValue {
            declared_type: 0,
            value: Some(Value::JsonValue(" \n null \t".to_string())),
        };
        assert_eq!(
            typed_value_to_property_value(&null),
            Ok(PropertyValue { value: None })
        );

        let invalid = pv2::TypedValue {
            declared_type: 0,
            value: Some(Value::JsonValue("not-json".to_string())),
        };
        assert!(
            typed_value_to_property_value(&invalid).is_err(),
            "malformed v2 JSON must reject the request"
        );
    }

    #[test]
    fn entity_op_mapping_is_lossless_for_common_ops() {
        use crate::graph::PropertyFilterOperator as Op;
        assert_eq!(entity_op_to_graph_op(1), Op::Equals as i32);
        assert_eq!(entity_op_to_graph_op(2), Op::NotEquals as i32);
        assert_eq!(entity_op_to_graph_op(4), Op::GreaterEqual as i32);
        assert_eq!(entity_op_to_graph_op(9), Op::Contains as i32);
        // Unknown / IN / NOT_IN default to Equals.
        assert_eq!(entity_op_to_graph_op(7), Op::Equals as i32);
    }

    #[test]
    fn graph_boundary_degradation_table_is_pinned() {
        // The graph property model is neutral (string/int/double/bool/bytes/
        // array/object/vector) — richer v2 variants cannot carry their type
        // marker through it without a graph-format change. This table PINS
        // the exact read-back degradation so any future movement (in either
        // direction) is a conscious contract change, not silent drift.
        // Value bytes survive; the declared type does not (TD-PROTO-2).
        use pv2::typed_value::Value;

        let cases: Vec<(Value, Value)> = vec![
            // Identity survivors.
            (
                Value::TextValue("hello".to_string()),
                Value::TextValue("hello".to_string()),
            ),
            (Value::IntegerValue(42), Value::IntegerValue(42)),
            (Value::FloatValue(1.5), Value::FloatValue(1.5)),
            (Value::BooleanValue(true), Value::BooleanValue(true)),
            (
                Value::BinaryValue(vec![1, 2, 3, 4]),
                Value::BinaryValue(vec![1, 2, 3, 4]),
            ),
            // Degradations: the graph model has no native slot.
            (
                Value::UuidValue(vec![7u8; 16]),
                Value::BinaryValue(vec![7u8; 16]),
            ),
            (
                Value::UlidValue(vec![8u8; 16]),
                Value::BinaryValue(vec![8u8; 16]),
            ),
            (
                Value::TimestampValue(1234567890),
                Value::IntegerValue(1234567890),
            ),
            (
                Value::TimestampTzValue(pv2::TimestampTzValue {
                    timestamp_us: 1234567890,
                    timezone: "America/Chicago".to_string(),
                }),
                Value::IntegerValue(1234567890),
            ),
            (Value::DateValue(20260), Value::IntegerValue(20260)),
            (Value::TimeValue(86399), Value::IntegerValue(86399)),
            (
                Value::DecimalValue(pv2::DecimalValue {
                    value: b"3.14159".to_vec(),
                    precision: 38,
                    scale: 18,
                }),
                Value::TextValue("3.14159".to_string()),
            ),
            (
                Value::SymbolValue("status".to_string()),
                Value::TextValue("status".to_string()),
            ),
            (
                Value::SparseVectorValue(pv2::SparseVector {
                    indices: vec![4, 1],
                    values: vec![0.25, 0.75],
                    dimension: 8,
                }),
                Value::TextValue(r#"{"indices":[4,1],"values":[0.25,0.75]}"#.to_string()),
            ),
            (Value::IsNull(true), Value::IsNull(true)),
        ];

        for (wire, expected_back) in cases {
            let tv_in = pv2::TypedValue {
                declared_type: 0,
                value: Some(wire),
            };
            let pv = typed_value_to_property_value(&tv_in)
                .unwrap_or_else(|e| panic!("write side rejected {tv_in:?}: {e}"));
            let back = property_value_to_typed_value(&pv);
            assert_eq!(
                back.value,
                Some(expected_back),
                "graph-boundary degradation drifted for input {tv_in:?}"
            );
        }
    }

    #[test]
    fn graph_boundary_rejects_invalid_typed_values() {
        use pv2::typed_value::Value;

        let invalid = [
            Value::UuidValue(vec![0; 15]),
            Value::UlidValue(vec![0; 17]),
            Value::Int8Value(i32::from(i8::MAX) + 1),
            Value::Uint8Value(u32::from(u8::MAX) + 1),
            Value::JsonValue("not-json".to_string()),
            Value::JsonbValue(vec![0xc1]),
            Value::IsNull(false),
        ];

        for wire in invalid {
            let typed = pv2::TypedValue {
                declared_type: 0,
                value: Some(wire),
            };
            assert!(
                typed_value_to_property_value(&typed).is_err(),
                "invalid typed value must fail before graph mutation: {typed:?}"
            );
        }
    }
}
