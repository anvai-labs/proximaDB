// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Wire-agnostic entity orchestration — the single place that turns an "entity"
//! (a graph node + optional embedding vectors + optional provenance chunk +
//! optional relation edges) into storage writes, and runs entity retrieval.
//!
//! Both transports are thin adapters over this:
//! - gRPC `ProximaEntityServiceImpl` (`src/network/grpc/v2/entity_service.rs`)
//!   converts proto ↔ these neutral types.
//! - REST `entities.rs` (`src/network/rest/v2/entities.rs`) converts JSON ↔ these.
//!
//! Per `SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`: retrieval delegates to the
//! fusion seam (`FusionService`) — this orchestrator owns no ranking. Tenant
//! isolation is structural: the caller passes a tenant-CLEAN `collection` + the
//! request `tenant`, and each leg is scoped here — graph via `for_tenant`, vector
//! via `TenantContext`, provenance document via the scoped collection key — never a
//! `{tenant}::collection` name fold.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{debug, error, warn};

use crate::core::search::cross_modal_fusion::FusionPolicy;
use crate::graph::{
    Edge, GraphOperationsService, Node, NodeId, NodeQuery, PropertyFilter, PropertyValue,
    property_value::Value as GraphValue,
};
use crate::services::fusion_service::{FusionOidKey, FusionService, GraphFusionParams, GraphGrain};
use crate::services::operations::vectors::VectorOperationsService;
use crate::storage::document::{DocumentRecord, DocumentService};
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTree, ProximaTreeNode, ProximaValue};

/// A neutral embedding input (wire-agnostic).
pub struct EntityEmbedding {
    pub model_id: String,
    pub modality: String,
    pub vector: Vec<f32>,
    pub dimension: u32,
}

/// A neutral relation input (wire-agnostic).
pub struct EntityRelation {
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub properties: HashMap<String, String>,
}

/// A neutral provenance input (wire-agnostic).
pub struct EntityProvenance {
    pub source_id: String,
    pub chunk_id: String,
    pub chunk_position: u32,
    pub extraction_method: String,
    pub metadata: HashMap<String, String>,
}

/// Neutral upsert input — what every transport facade produces from its wire type.
pub struct EntityUpsert {
    /// Empty ⇒ the orchestrator generates a UUID.
    pub entity_id: String,
    pub metadata: HashMap<String, PropertyValue>,
    pub embeddings: Vec<EntityEmbedding>,
    pub provenance: Option<EntityProvenance>,
    pub relations: Vec<EntityRelation>,
}

/// A single search hit: the entity graph node + its (fusion) score.
pub struct EntitySearchHit {
    pub node: Arc<Node>,
    pub score: f32,
}

/// One retrieval engine over graph + vector + document; owned by both transports.
pub struct EntityOrchestrator {
    graph: Arc<GraphOperationsService>,
    vector: Arc<VectorOperationsService>,
    fusion: Arc<FusionService>,
    document: Arc<DocumentService>,
}

impl EntityOrchestrator {
    /// Build from the four backing services.
    pub fn new(
        graph: Arc<GraphOperationsService>,
        vector: Arc<VectorOperationsService>,
        fusion: Arc<FusionService>,
        document: Arc<DocumentService>,
    ) -> Self {
        Self {
            graph,
            vector,
            fusion,
            document,
        }
    }

    /// Entity node id: `entity:{collection}:{entity_id}`.
    pub fn entity_node_id(collection_id: &str, entity_id: &str) -> NodeId {
        format!("entity:{collection_id}:{entity_id}")
    }

    /// Auxiliary id (embedding vector / provenance document). The entity **node
    /// id** is the recoverable prefix (split on the last `/`), so fusion results
    /// — which carry only the vector `oid` — project back to the entity node with
    /// no re-fetch.
    pub fn auxiliary_id(collection_id: &str, entity_id: &str, model_id: &str) -> String {
        format!(
            "{}/{model_id}",
            Self::entity_node_id(collection_id, entity_id)
        )
    }

    /// Recover the entity node id from an auxiliary oid. Inverse of [`Self::auxiliary_id`].
    pub fn node_id_from_auxiliary_oid(oid: &str) -> &str {
        oid.rsplit_once('/')
            .map(|(node_id, _)| node_id)
            .unwrap_or(oid)
    }

    fn str_property(s: impl Into<String>) -> PropertyValue {
        PropertyValue {
            value: Some(GraphValue::StringValue(s.into())),
        }
    }

    /// Create/update the graph node + best-effort embeddings, provenance, edges.
    /// Returns the resolved entity id (generated if the input was empty).
    pub async fn upsert(
        &self,
        collection: &str,
        tenant_id: &str,
        input: EntityUpsert,
    ) -> Result<String> {
        let entity_id = if input.entity_id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            input.entity_id
        };
        let node_id = Self::entity_node_id(collection, &entity_id);

        // Step 1: graph node (authoritative).
        let mut node_properties = input.metadata;
        node_properties.insert("_entity_type".to_string(), Self::str_property("entity"));
        node_properties.insert("_collection_id".to_string(), Self::str_property(collection));
        let now_ms = chrono::Utc::now().timestamp_millis();
        let node = Node {
            id: node_id.clone(),
            labels: vec!["entity".to_string()],
            properties: node_properties,
            embedding: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };
        // Route graph ops through the tenant-scoped handle (structural isolation composed once);
        // the collection name stays tenant-clean.
        match self
            .graph
            .for_tenant(tenant_id)
            .create_node(collection, node.clone())
            .await
        {
            Ok(_) => debug!("Created graph node {node_id}"),
            Err(create_err) => {
                if let Err(update_err) = self
                    .graph
                    .for_tenant(tenant_id)
                    .update_node(collection, node)
                    .await
                {
                    error!("Entity node upsert failed: create={create_err}; update={update_err}");
                    return Err(update_err).context("entity node upsert");
                }
                debug!("Updated graph node {node_id}");
            }
        }

        // Step 2: embeddings (best-effort).
        for emb in input.embeddings {
            let vector_id = Self::auxiliary_id(collection, &entity_id, &emb.model_id);
            let mut props = ProximaTree::new();
            props.insert(
                "_entity_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(entity_id.clone())),
            );
            props.insert(
                "_entity_collection".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(collection.to_string())),
            );
            props.insert(
                "_embedding_model_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(emb.model_id.clone())),
            );
            let record = ProximaRecord {
                oid: vector_id.clone(),
                tenant_id: tenant_id.to_string(),
                local_id: Some(format!("{}:{}", entity_id, emb.model_id)),
                embeddings: vec![EmbeddingCell::new_fp32(
                    emb.model_id.clone(),
                    emb.modality,
                    emb.dimension,
                    emb.vector,
                )],
                props,
                ..Default::default()
            };
            // Tenant-context write: validates access + stamps tenant_id on the record so the vector
            // leg isolates by TenantContext (clean collection name), matching the fusion read path.
            match self
                .vector
                .insert_batch_with_tenant_context(
                    collection,
                    vec![record],
                    Some(&crate::storage::tenant::context::TenantContext::for_tenant_id(tenant_id)),
                )
                .await
            {
                Ok(r) => {
                    if !r.success {
                        warn!(
                            "Embedding insert non-success for {vector_id}: {:?}",
                            r.errors
                        );
                    }
                }
                Err(e) => warn!("Embedding insert failed (non-fatal) for {vector_id}: {e}"),
            }
        }

        // Step 3: provenance document (best-effort).
        if let Some(prov) = input.provenance {
            let doc_id = Self::auxiliary_id(collection, &entity_id, "provenance");
            let mut tree = ProximaTree::new();
            tree.insert(
                "source_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(prov.source_id)),
            );
            tree.insert(
                "chunk_id".to_string(),
                ProximaTreeNode::Value(ProximaValue::String(prov.chunk_id)),
            );
            tree.insert(
                "chunk_position".to_string(),
                ProximaTreeNode::Value(ProximaValue::Int32(prov.chunk_position as i32)),
            );
            if !prov.extraction_method.is_empty() {
                tree.insert(
                    "extraction_method".to_string(),
                    ProximaTreeNode::Value(ProximaValue::String(prov.extraction_method)),
                );
            }
            for (k, v) in &prov.metadata {
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
                ProximaTreeNode::Value(ProximaValue::String(collection.to_string())),
            );
            // Scope the provenance doc collection to the tenant (both the storage key and the
            // record's collection_id — they key the same OID), matching the document fix. The
            // tenant is already validated (graph create above would have failed otherwise).
            let scoped_doc_collection =
                crate::storage::document::service::scoped_document_collection(
                    tenant_id, collection,
                )
                .context("scope provenance document collection")?;
            let doc = DocumentRecord::from_tree(
                doc_id.clone(),
                tree,
                scoped_doc_collection.clone(),
                None,
                Some("entity_provenance".to_string()),
            );
            match self
                .document
                .insert_document_record(&scoped_doc_collection, doc)
                .await
            {
                Ok(_) => debug!("Provenance document inserted doc_id={doc_id}"),
                Err(e) => warn!("Provenance insert failed (non-fatal) for {doc_id}: {e}"),
            }
        }

        // Step 4: relation edges (best-effort).
        for rel in input.relations {
            let source = Self::entity_node_id(collection, &rel.source_entity_id);
            let target = Self::entity_node_id(collection, &rel.target_entity_id);
            let edge_id = format!("{source}:{}->{target}", rel.relation_type);
            let mut edge_properties = HashMap::new();
            for (k, v) in &rel.properties {
                edge_properties.insert(k.clone(), Self::str_property(v));
            }
            edge_properties.insert("_entity_relation".to_string(), Self::str_property("true"));
            let edge = Edge {
                id: edge_id.clone(),
                from_node_id: source,
                to_node_id: target,
                edge_type: rel.relation_type,
                properties: edge_properties,
                weight: Some(rel.weight as f64),
                created_at_ms: now_ms,
                updated_at_ms: now_ms,
            };
            match self
                .graph
                .for_tenant(tenant_id)
                .create_edge(collection, edge)
                .await
            {
                Ok(_) => debug!("Created edge {edge_id}"),
                Err(e) => warn!("Edge create failed (non-fatal) for {edge_id}: {e}"),
            }
        }

        Ok(entity_id)
    }

    /// Fetch the entity graph node, if present. `tenant` scopes the graph read structurally.
    pub async fn get(
        &self,
        collection: &str,
        tenant: &str,
        entity_id: &str,
    ) -> Result<Option<Arc<Node>>> {
        let node_id = Self::entity_node_id(collection, entity_id);
        self.graph
            .for_tenant(tenant)
            .get_node(collection, &node_id)
            .await
            .context("get entity node")
    }

    /// Delete the entity graph node. Returns whether it existed. (Associated
    /// vectors/provenance are not cascaded — callers remove them via the
    /// auxiliary-id convention.) `tenant` scopes the graph delete structurally.
    pub async fn delete(&self, collection: &str, tenant: &str, entity_id: &str) -> Result<bool> {
        let node_id = Self::entity_node_id(collection, entity_id);
        Ok(self
            .graph
            .for_tenant(tenant)
            .delete_node(collection, &node_id)
            .await
            .context("delete entity node")?
            .is_some())
    }

    /// Search entities. With a `query_vector`: graph-augmented vector fusion via the
    /// fusion seam (TD-146 scope B — the seam normalizes the auxiliary vector oid and
    /// the canonical graph oid to the entity `node_id`, TD-142). Otherwise:
    /// metadata-filtered / unfiltered node scan. Returns hits with scores.
    pub async fn search(
        &self,
        collection: &str,
        tenant: &str,
        query_vector: Option<Vec<f32>>,
        filters: Vec<PropertyFilter>,
        top_k: usize,
    ) -> Result<Vec<EntitySearchHit>> {
        if let Some(query_vector) = query_vector {
            // Vector mode → fusion seam (one retrieval engine; no ranking here).
            let limit = if top_k == 0 { 10 } else { top_k };
            let params = GraphFusionParams {
                route_policy: None,
                graph_id: collection.to_string(),
                vector_collection: collection.to_string(),
                // Structural tenant boundary — scopes BOTH fusion legs (clean names in, TD-ENTITY-TENANT-1).
                tenant: Some(tenant.to_string()),
                tenant_stable_id: None, // FA-2 PR-D3: internal path — no client subject
                query_vector,
                // TD-146 scope B: graph-augmented entity fusion. `EntityNode` keying normalizes
                // the auxiliary vector oid + canonical graph oid to the entity `node_id` so the
                // sources co-rank (TD-142).
                max_depth: 1, // expand direct entity neighbours
                edge_types: Vec::new(),
                max_seeds: limit,
                limit,
                vector_weight: 1.0,
                graph_weight: 0.3, // modest graph boost; vector similarity leads (tunable)
                grain: GraphGrain::Nodes,
                principal: None,
                policy: FusionPolicy::default(),
                oid_key: FusionOidKey::EntityNode,
                // TD-138 document modality is REST-only for now; orchestrator passes `None`.
                document: None,
            };
            let (items, _stats) = self
                .fusion
                .graph_fusion_search(params)
                .await
                .context("entity fusion search")?;
            let mut hits = Vec::with_capacity(items.len());
            for item in items {
                let node_id = Self::node_id_from_auxiliary_oid(&item.oid).to_owned();
                if let Ok(Some(node)) = self
                    .graph
                    .for_tenant(tenant)
                    .get_node(collection, &node_id)
                    .await
                {
                    hits.push(EntitySearchHit {
                        node,
                        score: item.score,
                    });
                }
            }
            return Ok(hits);
        }

        // Metadata-filtered / unfiltered node scan.
        let limit = if top_k == 0 { 50 } else { top_k };
        let query = NodeQuery {
            graph_id: collection.to_string(),
            labels: vec!["entity".to_string()],
            filters,
            limit: Some(limit as u32),
            offset: None,
            continuation_token: None,
        };
        let nodes = self
            .graph
            .for_tenant(tenant)
            .query_nodes(collection, query)
            .await
            .context("entity node scan")?;
        let prefix = format!("entity:{collection}:");
        Ok(nodes
            .into_iter()
            .filter(|n| n.id.starts_with(&prefix))
            .map(|node| EntitySearchHit { node, score: 0.0 })
            .collect())
    }
}
