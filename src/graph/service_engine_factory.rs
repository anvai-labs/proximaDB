//! # Graph Engine Factory
//!
//! Centralizes graph engine creation, selection, and first-time initialization.
//!
//! ## Engine Selection
//!
//! | Engine | Status | Recommended |
//! |--------|--------|-------------|
//! | ORION | Production | Yes (default) |
//!
//! ## Default Engine
//!
//! **ORION is the default and recommended graph engine for all workloads.**
//!
//! When no engine type is specified in collection metadata, ORION is automatically
//! selected. ORION provides:
//! - Production-grade reliability with WAL persistence
//! - High performance (1M+ edges/sec traversal, <1us node lookup)
//! - Full feature support (graph algorithms, label indexes, concurrent access)
//!
//! ## Responsibilities
//!
//! - Resolve engine type from collection metadata (ORION only; retired names rejected)
//! - Normalize storage root URL and pass through to persistence layer
//! - Initialize schema-derived unique/multi-unique indexes on first load
//! - Keep distributed/tiered graph behavior behind relational/storage substrate policy

use super::Result;
use crate::graph::engines::GraphEngine;
use crate::proto::proximadb_v1::GraphCollection;
use proximadb_kernel::error::ProximaDBError;
use std::sync::Arc;

impl super::GraphOperationsService {
    /// Get or create a graph engine for the specified graph ID
    /// Engine for a CREATING write (create/batch-create, transactions, adding
    /// a schema constraint): provisions the graph collection on first use.
    ///
    /// Lazily provisioning under this (possibly tenant-scoped) key means a
    /// tenant-scoped first write (`for_tenant` composes `{tenant}/{graph_id}`)
    /// auto-registers its collection instead of failing "does not exist",
    /// making named-tenant graph create->use work without an out-of-band bare
    /// create (TD-GRAPH-TENANT-1). The default tenant stays bare, so its
    /// explicitly-created collections are found, not recreated.
    ///
    /// Reads and mutations of existing data must use
    /// [`Self::get_existing_graph_engine`] instead: provisioning on a read
    /// made a lookup in a mistyped graph silently create and register it.
    pub async fn get_or_create_graph_engine(
        &self,
        graph_id: &str,
    ) -> Result<Arc<crate::graph::engines::GraphEngineImpl>> {
        if let Some(engine) = self.graphs.get(graph_id) {
            return Ok(Arc::clone(&engine));
        }
        let collection = self
            .collection_service
            .ensure_or_create_graph(graph_id)
            .await?;
        self.open_graph_engine(graph_id, &collection).await
    }

    /// Engine for a READ or a mutation of existing data. Never provisions:
    /// a graph that was never created is an error ("Graph collection '...'
    /// not found", surfaced as NOT_FOUND), not an empty graph created as a
    /// side effect of the lookup.
    pub async fn get_existing_graph_engine(
        &self,
        graph_id: &str,
    ) -> Result<Arc<crate::graph::engines::GraphEngineImpl>> {
        if let Some(engine) = self.graphs.get(graph_id) {
            return Ok(Arc::clone(&engine));
        }
        let Some(collection) = self.collection_service.get_graph(graph_id).await? else {
            // Same wording as `update_graph_schema` (service.rs): the message-based
            // gRPC / entity mappers check "not found" before "invalid", so this
            // surfaces as NOT_FOUND on the wire.
            return Err(ProximaDBError::InvalidInput(format!(
                "Graph collection '{graph_id}' not found"
            )));
        };
        self.open_graph_engine(graph_id, &collection).await
    }

    /// Open (and cache) the engine for an existing graph collection.
    async fn open_graph_engine(
        &self,
        graph_id: &str,
        collection: &GraphCollection,
    ) -> Result<Arc<crate::graph::engines::GraphEngineImpl>> {
        // Determine engine type and storage ROOT URL
        let engine_type_str = collection
            .engine_config
            .as_ref()
            .map(|cfg| cfg.engine_type.clone())
            .or_else(|| {
                collection
                    .storage_config
                    .as_ref()
                    .map(|cfg| cfg.engine_type.clone())
            })
            .unwrap_or_else(|| "ORION".to_string());

        let storage_root_url = collection
            .storage_config
            .as_ref()
            .map_or_else(|| self.base_storage_url.clone(), |cfg| cfg.base_url.clone());

        tracing::info!(
            "Creating new graph engine for '{}' type={} storage_root={}",
            graph_id,
            engine_type_str,
            storage_root_url
        );

        // Create engine based on type (default ORION - production-ready).
        let engine_impl = match engine_type_str.to_ascii_uppercase().as_str() {
            "PULSAR" | "QUASAR" => {
                return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                    format!(
                        "{engine_type_str} graph engine has been retired; use ORION. \
                         Distributed placement and projection tiering are provided by the \
                         relational/storage substrate."
                    ),
                ));
            }
            "ORION" | "" => {
                // TD-066 (c) Part 1: thread the shared canonical WAL
                // path into ORION's persistence layer so recovery can
                // scan and log the latest checkpoint LSN per graph.
                // The path is None for callers that haven't wired
                // SharedServices.canonical_wal_appender yet — preserves
                // today's recovery behavior.
                let canonical_wal_path = self.canonical_wal_path.clone();
                let orion =
                    crate::graph::OrionGraphEngine::with_persistence_for_graph_and_canonical_wal(
                        graph_id.to_string(),
                        storage_root_url,
                        true,
                        canonical_wal_path,
                        crate::graph::unified_wal_factory(),
                    )
                    .await?;
                crate::graph::engines::GraphEngineImpl::Orion(orion)
            }
            other => {
                return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                    format!("Unknown graph engine type '{other}'. Valid option: ORION"),
                ));
            }
        };

        let engine = Arc::new(engine_impl);
        self.graphs
            .insert(graph_id.to_string(), Arc::clone(&engine));

        // Initialize schema-derived constraints (unique, multi-unique) if present
        if let Some(coll) = self.collection_service.get_graph(graph_id).await?
            && let Some(schema) = &coll.schema
        {
            self.initialize_schema_constraints(graph_id, engine.as_ref(), schema)
                .await?;
        }

        Ok(engine)
    }

    /// Initialize constraint registries based on graph schema (unique constraints)
    pub(super) async fn initialize_schema_constraints(
        &self,
        graph_id: &str,
        engine: &crate::graph::engines::GraphEngineImpl,
        schema: &crate::proto::proximadb_v1::GraphSchema,
    ) -> Result<()> {
        use dashmap::DashMap;
        // Build composite (multi-property) unique constraint indexes
        for uc in &schema.unique_constraints {
            if uc.properties.is_empty() {
                continue;
            }
            let labels_key = Self::normalize_list(&uc.node_labels);
            let props_key = Self::normalize_list(&uc.properties);
            // Always use a normalized (sorted) property list to build composite keys
            let props_sorted: Vec<String> = props_key.split('|').map(|s| s.to_string()).collect();
            let map: DashMap<String, String> = DashMap::new();
            let existing_nodes = engine.get_all_nodes()?;
            for node in existing_nodes {
                if !Self::node_has_all_labels(&node, &uc.node_labels) {
                    continue;
                }
                if let Some(composite) = Self::composite_key_for_node(&node, &props_sorted) {
                    if let Some(existing) = map.get(&composite)
                        && existing.value() != &node.id
                    {
                        return Err(ProximaDBError::InvalidInput(format!(
                            "Existing duplicate composite value '{}' for unique {:?}",
                            composite, uc.properties
                        )));
                    }
                    map.insert(composite, node.id.clone());
                }
            }
            self.memory_pool
                .unique_constraints_multi
                .insert((graph_id.to_string(), labels_key, props_key), map);
        }
        Ok(())
    }
}
