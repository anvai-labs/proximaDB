//! Edge Operations API (extracted from service.rs)
//!
//! Provides edge CRUD, single-edge retrieval, and property/type-based edge
//! querying, keeping the main service lean and focused.

use super::Result;
use crate::graph::EdgeQuery;
use crate::graph::adjacency_projection::edge_to_canonical_record;
use crate::graph::engines::GraphEngine;
use crate::graph::{Edge, EdgeId};
use proximadb_graph::projection::GraphTopologyProjection;
use proximadb_graph::record::GraphNodeKey;
use std::sync::Arc;

impl super::GraphOperationsService {
    /// Create a new edge
    pub async fn create_edge(&self, graph_id: &str, edge: Edge) -> Result<Arc<Edge>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_or_create_graph_engine(graph_id).await?;

        // Enforce composite (from,to,type) uniqueness using in-memory index
        if self
            .memory_pool
            .edge_composite_index
            .get(&(
                edge.from_node_id.clone(),
                edge.to_node_id.clone(),
                edge.edge_type.clone(),
            ))
            .is_some()
        {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                format!(
                    "Composite edge already exists: (from='{}', to='{}', type='{}')",
                    edge.from_node_id, edge.to_node_id, edge.edge_type
                ),
            ));
        }

        // Schema validation for edge using endpoint labels if schema defines constraints
        if let (Some(from), Some(to)) = (
            engine.get_node(&edge.from_node_id)?,
            engine.get_node(&edge.to_node_id)?,
        ) {
            self.enforce_schema_on_edge(graph_id, &edge, &from.labels, &to.labels)
                .await?;
            self.enforce_cardinality_on_edge(graph_id, &edge, engine.as_ref())
                .await?;
        }

        self.upsert_canonical_edge_record(graph_id, &edge).await?;
        let edge_arc = engine.insert_edge(edge).await?;
        let edge_record = edge_to_canonical_record(graph_id, &edge_arc);
        self.adjacency_projection(graph_id)
            .apply_edge(&edge_record)
            .await?;
        self.advance_edge_epoch(graph_id);
        // Update edge stats
        self.stats_edges
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.edge_type_counts
            .entry(edge_arc.edge_type.clone())
            .or_insert_with(|| std::sync::atomic::AtomicU64::new(0))
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(edge_arc)
    }

    /// Update an edge
    pub async fn update_edge(&self, graph_id: &str, edge: Edge) -> Result<Arc<Edge>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let engine = self.get_or_create_graph_engine(graph_id).await?;
        // If we can load endpoints, enforce schema
        if let (Some(from), Some(to)) = (
            engine.get_node(&edge.from_node_id)?,
            engine.get_node(&edge.to_node_id)?,
        ) {
            self.enforce_schema_on_edge(graph_id, &edge, &from.labels, &to.labels)
                .await?;
            self.enforce_cardinality_on_edge(graph_id, &edge, engine.as_ref())
                .await?;
        }
        self.upsert_canonical_edge_record(graph_id, &edge).await?;
        let edge_arc = engine.update_edge(edge).await?;
        let edge_record = edge_to_canonical_record(graph_id, &edge_arc);
        self.adjacency_projection(graph_id)
            .apply_edge(&edge_record)
            .await?;
        self.advance_edge_epoch(graph_id);
        Ok(edge_arc)
    }

    /// Delete an edge
    pub async fn delete_edge(&self, graph_id: &str, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_or_create_graph_engine(graph_id).await?;
        let deleted = crate::graph::engines::GraphEngine::delete_edge(&*engine, id).await?;
        if let Some(ref edge) = deleted {
            let edge_record = edge_to_canonical_record(graph_id, edge);
            self.adjacency_projection(graph_id)
                .remove_edge(&edge_record)
                .await?;
            self.delete_canonical_edge_record(graph_id, &edge.id)
                .await?;
            self.advance_edge_epoch(graph_id);
            self.stats_edges
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(v) = self.edge_type_counts.get(&edge.edge_type) {
                v.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(deleted)
    }
    /// Get an edge by ID
    pub async fn get_edge(&self, graph_id: &str, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_or_create_graph_engine(graph_id).await?;
        if let Some(edge) = engine.get_edge(id)? {
            return Ok(Some(edge));
        }
        // TD-168 cold-payload tier (gated default-OFF): on a hot miss, serve from
        // the byte-budgeted cache over the canonical record store. The gate is
        // checked inside `cold_fetch_edge` (returns today's `None` when off).
        self.cold_fetch_edge(graph_id, id).await
    }

    /// Query edges by endpoints, properties and types.
    ///
    /// When the query is endpoint-bound (has `from_node_id` or `to_node_id`),
    /// reads are served from the adjacency projection rather than scanning the
    /// underlying engine, consistent with the convergence mandate in
    /// `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`
    /// Phase 3.  Engine traversal is only used for full-graph edge scans.
    pub async fn query_edges(&self, graph_id: &str, query: EdgeQuery) -> Result<Vec<Arc<Edge>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_or_create_graph_engine(graph_id).await?;
        let edge_prefix = format!("graph/{graph_id}/edge/");

        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        let is_endpoint_bound = query.from_node_id.is_some() || query.to_node_id.is_some();

        if is_endpoint_bound
            && self.has_fresh_orion_csr(graph_id)
            && let Some(csr_results) = self.query_edges_from_fresh_csr(engine.as_ref(), &query)?
        {
            results = csr_results;
        }

        if is_endpoint_bound && results.is_empty() {
            let projection = self.adjacency_projection(graph_id);

            if let Some(from) = &query.from_node_id {
                let node_oid = GraphNodeKey::new(graph_id, from.as_str()).canonical_oid();
                if let Ok(entries) = projection.edges_by_src(&node_oid) {
                    for entry in entries {
                        let edge_oid = &entry.key.edge_oid;
                        if seen.contains(edge_oid) {
                            continue;
                        }
                        if let Some(edge_id) = edge_oid.strip_prefix(&edge_prefix)
                            && let Ok(Some(edge)) = engine.get_edge(&edge_id.to_string())
                        {
                            seen.insert(edge_oid.clone());
                            results.push(edge);
                        }
                    }
                }
            }

            if let Some(to) = &query.to_node_id {
                let node_oid = GraphNodeKey::new(graph_id, to.as_str()).canonical_oid();
                if let Ok(entries) = projection.edges_by_dst(&node_oid) {
                    for entry in entries {
                        let edge_oid = &entry.key.edge_oid;
                        if seen.contains(edge_oid) {
                            continue;
                        }
                        if let Some(edge_id) = edge_oid.strip_prefix(&edge_prefix)
                            && let Ok(Some(edge)) = engine.get_edge(&edge_id.to_string())
                        {
                            seen.insert(edge_oid.clone());
                            results.push(edge);
                        }
                    }
                }
            }
        }
        // Non-endpoint-bound queries return an empty result; a full-graph edge
        // scan is not supported without explicit endpoint or type filters.

        if let Some(from) = &query.from_node_id {
            results.retain(|edge| edge.from_node_id == *from);
        }
        if let Some(to) = &query.to_node_id {
            results.retain(|edge| edge.to_node_id == *to);
        }
        if !query.edge_types.is_empty() {
            results.retain(|edge| query.edge_types.contains(&edge.edge_type));
        }

        // Property filters (simple, if provided)
        if !query.filters.is_empty() {
            results.retain(|edge| {
                for filter in &query.filters {
                    use crate::graph::PropertyFilterOperator as Op;
                    let prop_val_opt = edge.properties.get(&filter.key);
                    let pass = match Op::try_from(filter.operator).unwrap_or(Op::Unspecified) {
                        Op::Equals => match prop_val_opt {
                            Some(v) => {
                                let filter_val = match filter.value.as_ref() {
                                    Some(val) => val,
                                    None => return false,
                                };
                                v.value == filter_val.value
                            }
                            None => false,
                        },
                        Op::NotEquals => match prop_val_opt {
                            Some(v) => {
                                let filter_val = match filter.value.as_ref() {
                                    Some(val) => val,
                                    None => return true,
                                };
                                v.value != filter_val.value
                            }
                            None => true,
                        },
                        _ => true,
                    };
                    if !pass {
                        return false;
                    }
                }
                true
            });
        }
        Ok(results)
    }

    /// Enumerate every edge in a graph.
    ///
    /// The engine has no full edge scan — edges live in per-source adjacency, so
    /// [`query_edges`](Self::query_edges) deliberately returns empty without an
    /// endpoint. But every edge has exactly one source node, so iterating each
    /// node's *outgoing* edges yields every edge exactly once (no dedup). Used by
    /// the columnar Flight export to dump a whole graph's edges (a whole-graph
    /// Arrow dump needs all edges, not just one node's adjacency).
    ///
    /// Materializes the node set and walks adjacency — an O(N+E) scan intended
    /// for export/ETL, not the hot traversal path.
    pub async fn all_edges(&self, graph_id: &str) -> Result<Vec<Arc<Edge>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_or_create_graph_engine(graph_id).await?;
        let nodes = engine.get_all_nodes()?;
        let mut edges = Vec::new();
        for node in nodes {
            edges.extend(engine.get_outgoing_edges(&node.id, None)?);
        }
        Ok(edges)
    }

    fn query_edges_from_fresh_csr(
        &self,
        engine: &crate::graph::engines::GraphEngineImpl,
        query: &EdgeQuery,
    ) -> Result<Option<Vec<Arc<Edge>>>> {
        let Some(endpoint) = query.from_node_id.as_ref().or(query.to_node_id.as_ref()) else {
            return Ok(None);
        };

        let edge_type = if query.edge_types.len() == 1 {
            Some(query.edge_types[0].as_str())
        } else {
            None
        };

        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        if let Some(from) = &query.from_node_id {
            for edge in engine.get_outgoing_edges(from, edge_type)? {
                if seen.insert(edge.id.clone()) {
                    results.push(edge);
                }
            }
        }

        if query.from_node_id.is_none() {
            for edge in engine.get_incoming_edges(endpoint, edge_type)? {
                if seen.insert(edge.id.clone()) {
                    results.push(edge);
                }
            }
        }

        Ok(Some(results))
    }
}

#[cfg(test)]
mod tests {
    use super::super::GraphOperationsService;
    use crate::graph::{Edge, Node};
    use crate::proto::proximadb_v1::CreateGraphRequest;
    use async_trait::async_trait;
    use proximadb_records::{ProximaRecord, RecordKey, RecordStore, RecordStoreResult};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    /// Node fixture: single `label`, no properties, no embedding. Only the id,
    /// label, and timestamps vary across these tests — pass each test's exact
    /// values ((1,1) seed loops, (0,0) CSR rebuild, (7,9) cold-tier payloads).
    fn mk_node(id: &str, label: &str, created_at_ms: i64, updated_at_ms: i64) -> Node {
        Node {
            id: id.to_string(),
            labels: vec![label.to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms,
            updated_at_ms,
        }
    }

    /// Edge fixture: no properties, unweighted. Only the endpoints, type, and
    /// timestamps vary — pass each test's exact values.
    fn mk_edge(
        id: &str,
        from: &str,
        to: &str,
        edge_type: &str,
        created_at_ms: i64,
        updated_at_ms: i64,
    ) -> Edge {
        Edge {
            id: id.to_string(),
            from_node_id: from.to_string(),
            to_node_id: to.to_string(),
            edge_type: edge_type.to_string(),
            properties: HashMap::new(),
            weight: None,
            created_at_ms,
            updated_at_ms,
        }
    }

    /// Create the graph collection under the unique per-test id
    /// `{prefix}_{pid}` (the process suffix keeps concurrent runs isolated).
    async fn seed_graph(service: &GraphOperationsService, prefix: &str) -> String {
        let graph_id = format!("{prefix}_{}", std::process::id());
        service
            .create_graph_collection(CreateGraphRequest {
                graph_id: graph_id.clone(),
                name: None,
                description: None,
                schema: None,
                storage_config: None,
                engine_config: None,
                access_control: None,
            })
            .await
            .expect("create graph");
        graph_id
    }

    #[derive(Default)]
    struct MemoryRecordStore {
        records: RwLock<HashMap<String, ProximaRecord>>,
    }

    #[async_trait]
    impl RecordStore for MemoryRecordStore {
        async fn upsert_record(&self, record: ProximaRecord) -> RecordStoreResult<ProximaRecord> {
            self.records
                .write()
                .expect("record store write lock")
                .insert(record.oid.clone(), record.clone());
            Ok(record)
        }

        async fn get_record(&self, key: &RecordKey) -> RecordStoreResult<Option<ProximaRecord>> {
            Ok(self
                .records
                .read()
                .expect("record store read lock")
                .get(&key.oid)
                .cloned())
        }

        async fn delete_record(&self, key: &RecordKey) -> RecordStoreResult<bool> {
            Ok(self
                .records
                .write()
                .expect("record store write lock")
                .remove(&key.oid)
                .is_some())
        }
    }

    #[tokio::test]
    async fn edge_crud_maintains_adjacency_projection() {
        let service = GraphOperationsService::new();
        let graph_id = seed_graph(&service, "adjacency_service_test").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3"] {
            service
                .create_node(graph_id, mk_node(node_id, "Person", 1, 1))
                .await
                .expect("create node");
        }

        service
            .create_edge(graph_id, mk_edge("e1", "n1", "n2", "KNOWS", 1, 1))
            .await
            .expect("create edge");
        assert_eq!(
            service
                .adjacency_projection_edge_count(graph_id)
                .expect("edge count"),
            1
        );

        service
            .update_edge(graph_id, mk_edge("e1", "n1", "n3", "KNOWS", 1, 2))
            .await
            .expect("update edge");
        assert_eq!(
            service
                .adjacency_projection(graph_id)
                .edges_by_dst(&format!("graph/{graph_id}/node/n3"))
                .expect("incoming")
                .len(),
            1
        );
        assert!(
            service
                .adjacency_projection(graph_id)
                .edges_by_dst(&format!("graph/{graph_id}/node/n2"))
                .expect("old incoming")
                .is_empty()
        );

        service
            .delete_edge(graph_id, &"e1".to_string())
            .await
            .expect("delete edge");
        assert_eq!(
            service
                .adjacency_projection_edge_count(graph_id)
                .expect("edge count"),
            0
        );

        service
            .batch_create_edges(
                graph_id,
                vec![
                    mk_edge("e2", "n1", "n2", "KNOWS", 3, 3),
                    mk_edge("e3", "n1", "n3", "KNOWS", 3, 3),
                ],
            )
            .await
            .expect("batch create edges");
        assert_eq!(
            service
                .adjacency_projection_edge_count(graph_id)
                .expect("edge count"),
            2
        );
    }

    /// #1524 follow-up, measured on the Victor repo-scale corpus: ONE edge
    /// referencing a nonexistent node must reject ITSELF — not abort the batch
    /// and throw away the other valid edges (the old behavior dropped 292 of
    /// 293). The rejection is reported per-edge; the valid rest lands in the
    /// engine, the adjacency projection, and the counters.
    #[tokio::test]
    async fn batch_create_edges_skips_bad_edges_and_lands_the_rest() {
        let service = GraphOperationsService::new();
        let graph_id = seed_graph(&service, "batch_partial_test").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3"] {
            service
                .create_node(graph_id, mk_node(node_id, "Sym", 1, 1))
                .await
                .expect("create node");
        }

        let outcome = service
            .batch_create_edges(
                graph_id,
                vec![
                    mk_edge("e1", "n1", "n2", "CALLS", 1, 1),
                    // Dangling source — must reject only itself.
                    mk_edge("e_bad", "ghost", "n2", "CALLS", 1, 1),
                    mk_edge("e2", "n2", "n3", "CALLS", 1, 1),
                    // Dangling target — must reject only itself.
                    mk_edge("e_bad2", "n1", "ghost", "CALLS", 1, 1),
                    // Duplicate composite of e1 within the batch.
                    mk_edge("e_dup", "n1", "n2", "CALLS", 1, 1),
                ],
            )
            .await
            .expect("partial batch must not error");

        let created_ids: Vec<&str> = outcome.created.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(created_ids, vec!["e1", "e2"], "valid edges land");
        assert_eq!(outcome.rejected.len(), 3, "each bad edge rejects itself");
        let rejected_ids: Vec<&str> = outcome
            .rejected
            .iter()
            .map(|r| r.edge_id.as_str())
            .collect();
        assert_eq!(rejected_ids, vec!["e_bad", "e_bad2", "e_dup"]);
        assert!(
            outcome.rejected[0].reason.contains("ghost"),
            "reason names the missing node: {}",
            outcome.rejected[0].reason
        );

        // The landed edges are fully wired: projection + engine reads see them.
        assert_eq!(
            service
                .adjacency_projection_edge_count(graph_id)
                .expect("projection count"),
            2
        );
        let stats = service.get_stats(graph_id).await.expect("stats");
        assert_eq!(stats.total_edges, 2, "counters reflect only landed edges");
    }

    #[tokio::test]
    async fn endpoint_bound_query_served_from_adjacency_projection() {
        use crate::graph::EdgeQuery;

        let service = GraphOperationsService::new();
        let graph_id = seed_graph(&service, "adj_query_test").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3"] {
            service
                .create_node(graph_id, mk_node(node_id, "P", 1, 1))
                .await
                .expect("create node");
        }

        service
            .create_edge(graph_id, mk_edge("e1", "n1", "n2", "KNOWS", 1, 1))
            .await
            .expect("create e1");
        service
            .create_edge(graph_id, mk_edge("e2", "n1", "n3", "KNOWS", 1, 1))
            .await
            .expect("create e2");

        // Outgoing from n1 via adjacency projection
        let outgoing = service
            .query_edges(
                graph_id,
                EdgeQuery {
                    graph_id: graph_id.to_string(),
                    from_node_id: Some("n1".to_string()),
                    to_node_id: None,
                    edge_types: vec![],
                    filters: vec![],
                    offset: None,
                    limit: None,
                    continuation_token: None,
                },
            )
            .await
            .expect("query outgoing");
        assert_eq!(outgoing.len(), 2, "n1 should have 2 outgoing edges");

        // Incoming to n2 via adjacency projection
        let incoming = service
            .query_edges(
                graph_id,
                EdgeQuery {
                    graph_id: graph_id.to_string(),
                    from_node_id: None,
                    to_node_id: Some("n2".to_string()),
                    edge_types: vec![],
                    filters: vec![],
                    offset: None,
                    limit: None,
                    continuation_token: None,
                },
            )
            .await
            .expect("query incoming");
        assert_eq!(incoming.len(), 1, "n2 should have 1 incoming edge");
        assert_eq!(incoming[0].id, "e1");
    }

    #[tokio::test]
    async fn endpoint_bound_query_uses_fresh_csr_and_falls_back_when_stale() {
        use crate::graph::EdgeQuery;

        let service = GraphOperationsService::new();
        let graph_id = seed_graph(&service, "csr_query_test").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3", "n4"] {
            service
                .create_node(graph_id, mk_node(node_id, "P", 1, 1))
                .await
                .expect("create node");
        }

        for (id, to) in [("e1", "n2"), ("e2", "n3")] {
            service
                .create_edge(graph_id, mk_edge(id, "n1", to, "KNOWS", 1, 1))
                .await
                .expect("create edge");
        }

        assert!(!service.has_fresh_orion_csr(graph_id));
        service
            .rebuild_orion_csr_from_adjacency_projection(graph_id)
            .await
            .expect("rebuild csr");
        assert_eq!(
            service.csr_rebuild_epoch(graph_id),
            Some(service.edge_epoch(graph_id))
        );
        assert!(service.has_fresh_orion_csr(graph_id));

        let query = EdgeQuery {
            graph_id: graph_id.to_string(),
            from_node_id: Some("n1".to_string()),
            to_node_id: None,
            edge_types: vec!["KNOWS".to_string()],
            filters: vec![],
            offset: None,
            limit: None,
            continuation_token: None,
        };

        let engine_arc = service
            .graphs
            .get(graph_id)
            .expect("engine present")
            .value()
            .clone();
        let mut csr_only_ids: Vec<_> = service
            .query_edges_from_fresh_csr(engine_arc.as_ref(), &query)
            .expect("csr candidates")
            .expect("csr endpoint query")
            .into_iter()
            .map(|edge| edge.id.clone())
            .collect();
        csr_only_ids.sort();
        assert_eq!(csr_only_ids, vec!["e1".to_string(), "e2".to_string()]);

        let mut query_ids: Vec<_> = service
            .query_edges(graph_id, query.clone())
            .await
            .expect("fresh csr query")
            .into_iter()
            .map(|edge| edge.id.clone())
            .collect();
        query_ids.sort();
        assert_eq!(query_ids, vec!["e1".to_string(), "e2".to_string()]);

        service
            .create_edge(graph_id, mk_edge("e3", "n1", "n4", "KNOWS", 2, 2))
            .await
            .expect("create stale edge");

        assert!(!service.has_fresh_orion_csr(graph_id));
        let mut stale_query_ids: Vec<_> = service
            .query_edges(graph_id, query)
            .await
            .expect("stale csr falls back to adjacency")
            .into_iter()
            .map(|edge| edge.id.clone())
            .collect();
        stale_query_ids.sort();
        assert_eq!(
            stale_query_ids,
            vec!["e1".to_string(), "e2".to_string(), "e3".to_string()]
        );
    }

    #[tokio::test]
    async fn canonical_record_store_receives_graph_writes() {
        let record_store = Arc::new(MemoryRecordStore::default());
        let service =
            GraphOperationsService::new().with_canonical_record_store(record_store.clone());
        let graph_id = seed_graph(&service, "canonical_graph_store_test").await;
        let graph_id = graph_id.as_str();

        service
            .create_node(graph_id, mk_node("n1", "Person", 1, 1))
            .await
            .expect("create n1");
        service
            .create_node(graph_id, mk_node("n2", "Person", 1, 1))
            .await
            .expect("create n2");
        service
            .create_edge(graph_id, mk_edge("e1", "n1", "n2", "KNOWS", 1, 1))
            .await
            .expect("create edge");

        assert!(
            record_store
                .get_record(&RecordKey::new(format!("graph/{graph_id}/node/n1")))
                .await
                .expect("get node record")
                .is_some()
        );
        assert!(
            record_store
                .get_record(&RecordKey::new(format!("graph/{graph_id}/edge/e1")))
                .await
                .expect("get edge record")
                .is_some()
        );

        service
            .delete_edge(graph_id, &"e1".to_string())
            .await
            .expect("delete edge");
        assert!(
            record_store
                .get_record(&RecordKey::new(format!("graph/{graph_id}/edge/e1")))
                .await
                .expect("get deleted edge")
                .is_none()
        );

        service
            .delete_node(graph_id, &"n1".to_string())
            .await
            .expect("delete node");
        assert!(
            record_store
                .get_record(&RecordKey::new(format!("graph/{graph_id}/node/n1")))
                .await
                .expect("get deleted node")
                .is_none()
        );
    }

    /// TD-066 Part 2 / #52: graph recovery replays the engine WAL into the in-memory
    /// engine only, leaving a *buffered* canonical/cold store empty after a crash.
    /// `repopulate_canonical_store` re-drives the recovered nodes + edges through the
    /// canonical store so the cold tier is rebuilt from the authoritative recovered
    /// engine state — the data-loss fix that unblocks wiring `ColdGraphSegmentStore`.
    #[tokio::test]
    async fn recovery_repopulates_canonical_store_from_engine() {
        let record_store = Arc::new(MemoryRecordStore::default());
        let service =
            GraphOperationsService::new().with_canonical_record_store(record_store.clone());
        let graph_id = seed_graph(&service, "canonical_recovery_test").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3"] {
            service
                .create_node(graph_id, mk_node(node_id, "Person", 1, 1))
                .await
                .expect("create node");
        }
        service
            .create_edge(graph_id, mk_edge("e1", "n1", "n2", "KNOWS", 1, 1))
            .await
            .expect("create edge");

        // Simulate a crash that lost the buffered cold store while the engine —
        // rebuilt from its own WAL on restart — retains the authoritative state.
        record_store.records.write().expect("clear store").clear();
        assert!(
            record_store.records.read().expect("read store").is_empty(),
            "cold store emptied (buffer lost on crash)"
        );

        // Recovery re-population rebuilds the canonical store from the engine.
        let driven = service
            .repopulate_canonical_store(graph_id)
            .await
            .expect("repopulate canonical store");
        assert_eq!(
            driven, 4,
            "3 nodes + 1 edge re-driven into the canonical store"
        );

        for node_id in ["n1", "n2", "n3"] {
            assert!(
                record_store
                    .get_record(&RecordKey::new(format!("graph/{graph_id}/node/{node_id}")))
                    .await
                    .expect("get recovered node")
                    .is_some(),
                "node {node_id} re-populated after recovery"
            );
        }
        assert!(
            record_store
                .get_record(&RecordKey::new(format!("graph/{graph_id}/edge/e1")))
                .await
                .expect("get recovered edge")
                .is_some(),
            "edge re-populated after recovery"
        );
    }

    /// TD-168 cold-payload tier: a node/edge whose payload is durable in the
    /// canonical record store but NOT resident in the engine is servable when the
    /// gate is ON, and yields `None` (today's behavior) when OFF. Process-isolated
    /// under nextest, so the env gate does not leak across tests.
    #[tokio::test]
    async fn cold_payload_tier_serves_node_and_edge_on_engine_miss() {
        let record_store = Arc::new(MemoryRecordStore::default());
        let service =
            GraphOperationsService::new().with_canonical_record_store(record_store.clone());
        let graph_id = seed_graph(&service, "cold_payload_test").await;
        let graph_id = graph_id.as_str();

        // Seed canonical records directly — the engine never sees these, so a hit
        // can only come from the cold record-store path.
        let node = mk_node("cold_n1", "Person", 7, 9);
        let edge = mk_edge("cold_e1", "cold_n1", "cold_n2", "KNOWS", 7, 9);
        record_store
            .upsert_record(
                crate::graph::adjacency_projection::node_to_canonical_record(graph_id, &node),
            )
            .await
            .expect("seed node record");
        record_store
            .upsert_record(
                crate::graph::adjacency_projection::edge_to_canonical_record(graph_id, &edge),
            )
            .await
            .expect("seed edge record");

        // Gate OFF (default): engine miss → None, no cold-fetch.
        unsafe { std::env::remove_var("PROXIMADB_GRAPH_COLD_PAYLOADS") };
        assert!(
            service
                .get_node(graph_id, &"cold_n1".to_string())
                .await
                .expect("get_node off")
                .is_none(),
            "gate OFF must not cold-fetch nodes"
        );
        assert!(
            service
                .get_edge(graph_id, &"cold_e1".to_string())
                .await
                .expect("get_edge off")
                .is_none(),
            "gate OFF must not cold-fetch edges"
        );

        // Gate ON: engine miss → cold-fetch from the record store.
        unsafe { std::env::set_var("PROXIMADB_GRAPH_COLD_PAYLOADS", "1") };
        let got_node = service
            .get_node(graph_id, &"cold_n1".to_string())
            .await
            .expect("get_node on")
            .expect("cold-served node");
        assert_eq!(got_node.id, "cold_n1");
        assert_eq!(got_node.labels, vec!["Person".to_string()]);

        let got_edge = service
            .get_edge(graph_id, &"cold_e1".to_string())
            .await
            .expect("get_edge on")
            .expect("cold-served edge");
        assert_eq!(got_edge.id, "cold_e1");
        assert_eq!(got_edge.from_node_id, "cold_n1");
        assert_eq!(got_edge.to_node_id, "cold_n2");
        assert_eq!(got_edge.edge_type, "KNOWS");

        // A missing id still yields None even with the gate on.
        assert!(
            service
                .get_node(graph_id, &"nonexistent".to_string())
                .await
                .expect("get_node missing")
                .is_none()
        );

        unsafe { std::env::remove_var("PROXIMADB_GRAPH_COLD_PAYLOADS") };
    }

    /// TD-168 Phase 2 end-to-end: the *production* cold store
    /// (`ColdGraphRecordStore` over object storage) — not the in-memory mock —
    /// backs the canonical record store. A node/edge seeded into it (durable via
    /// `put_with_tier`, here on `memory://` which is untiered so the tier degrades
    /// to a plain write) is served through the cold-fetch path on an engine miss,
    /// proving the dedicated store satisfies the `RecordStore` contract AND that
    /// graph records survive the `ProximaRecordV2` bincode round-trip intact.
    #[tokio::test]
    async fn cold_graph_record_store_serves_payloads_end_to_end() {
        let record_store = Arc::new(
            crate::graph::ColdGraphRecordStore::from_storage_root(
                "memory://",
                proximadb_storage_filesystem_types::ObjectAccessTier::Cool,
            )
            .expect("open memory cold store"),
        );
        let service =
            GraphOperationsService::new().with_canonical_record_store(record_store.clone());
        let graph_id = seed_graph(&service, "cold_store_e2e").await;
        let graph_id = graph_id.as_str();

        // Seed the cold store directly — the engine never sees these, so a hit can
        // only come from `ColdGraphRecordStore` via the cold-fetch path.
        let node = mk_node("cold_n1", "Person", 7, 9);
        let edge = mk_edge("cold_e1", "cold_n1", "cold_n2", "KNOWS", 7, 9);
        record_store
            .upsert_record(
                crate::graph::adjacency_projection::node_to_canonical_record(graph_id, &node),
            )
            .await
            .expect("seed node into cold store");
        record_store
            .upsert_record(
                crate::graph::adjacency_projection::edge_to_canonical_record(graph_id, &edge),
            )
            .await
            .expect("seed edge into cold store");

        unsafe { std::env::set_var("PROXIMADB_GRAPH_COLD_PAYLOADS", "1") };
        let got_node = service
            .get_node(graph_id, &"cold_n1".to_string())
            .await
            .expect("get_node on")
            .expect("cold-served node from ColdGraphRecordStore");
        assert_eq!(got_node.id, "cold_n1");
        assert_eq!(got_node.labels, vec!["Person".to_string()]);

        let got_edge = service
            .get_edge(graph_id, &"cold_e1".to_string())
            .await
            .expect("get_edge on")
            .expect("cold-served edge from ColdGraphRecordStore");
        assert_eq!(got_edge.id, "cold_e1");
        assert_eq!(got_edge.from_node_id, "cold_n1");
        assert_eq!(got_edge.to_node_id, "cold_n2");
        assert_eq!(got_edge.edge_type, "KNOWS");

        unsafe { std::env::remove_var("PROXIMADB_GRAPH_COLD_PAYLOADS") };
    }

    /// Depth-collapse: `get_nodes` batches the cold misses through
    /// `RecordStore::get_records` and returns one slot per id, in order
    /// (present/absent/present), served via the batched cold-fetch path.
    #[tokio::test]
    async fn get_nodes_batches_cold_misses_in_order() {
        let record_store = Arc::new(
            crate::graph::ColdGraphRecordStore::from_storage_root(
                "memory://",
                proximadb_storage_filesystem_types::ObjectAccessTier::Cool,
            )
            .expect("open memory cold store"),
        );
        let service =
            GraphOperationsService::new().with_canonical_record_store(record_store.clone());
        let graph_id = seed_graph(&service, "cold_batch").await;
        let graph_id = graph_id.as_str();

        // Seed n1 and n3 directly (engine never sees them); n2 is absent.
        for id in ["n1", "n3"] {
            let node = mk_node(id, "Person", 1, 1);
            record_store
                .upsert_record(
                    crate::graph::adjacency_projection::node_to_canonical_record(graph_id, &node),
                )
                .await
                .expect("seed node");
        }

        unsafe { std::env::set_var("PROXIMADB_GRAPH_COLD_PAYLOADS", "1") };
        let ids = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        let got = service.get_nodes(graph_id, &ids).await.expect("get_nodes");
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].as_ref().map(|n| n.id.as_str()), Some("n1"));
        assert!(got[1].is_none(), "absent node → None slot");
        assert_eq!(got[2].as_ref().map(|n| n.id.as_str()), Some("n3"));
        unsafe { std::env::remove_var("PROXIMADB_GRAPH_COLD_PAYLOADS") };
    }

    /// Verify that ORION CSR can be rebuilt from the adjacency projection.
    ///
    /// Creates 2 edges via the service (which updates the adjacency projection),
    /// then calls `rebuild_orion_csr_from_adjacency_projection`, and checks that
    /// ORION's CSR-backed `get_outgoing_targets` returns the expected neighbours.
    #[tokio::test]
    async fn orion_csr_rebuild_from_adjacency_projection() {
        use crate::graph::engines::GraphEngineImpl;

        let service = GraphOperationsService::new();
        let graph_id = seed_graph(&service, "csr_rebuild_test").await;
        let graph_id = graph_id.as_str();

        for nid in ["a", "b", "c"] {
            service
                .create_node(graph_id, mk_node(nid, "V", 0, 0))
                .await
                .expect("create node");
        }

        service
            .create_edge(graph_id, mk_edge("ea", "a", "b", "E", 0, 0))
            .await
            .expect("create ea");
        service
            .create_edge(graph_id, mk_edge("eb", "a", "c", "E", 0, 0))
            .await
            .expect("create eb");

        // Rebuild CSR from the adjacency projection.
        service
            .rebuild_orion_csr_from_adjacency_projection(graph_id)
            .await
            .expect("CSR rebuild");

        // Read back via ORION's CSR get_outgoing_targets.
        let engine_arc = service
            .graphs
            .get(graph_id)
            .expect("engine present")
            .value()
            .clone();
        let GraphEngineImpl::Orion(orion) = engine_arc.as_ref();
        let mut outgoing = orion
            .get_outgoing_targets(&"a".to_string())
            .await
            .expect("outgoing targets");
        outgoing.sort();
        assert_eq!(
            outgoing,
            vec!["b".to_string(), "c".to_string()],
            "CSR should have both outgoing neighbours for node a"
        );
    }

    /// #52 / TD-066 Part 2: `flush_wal` (the checkpoint AND graceful-shutdown path)
    /// must flush a BUFFERED canonical store unconditionally — in BOTH the scoped and
    /// non-scoped engine-WAL configs. This is the regression guard for the flush
    /// placement: if it were buried inside the `if scoped` block, the non-scoped run
    /// would silently leave the buffer unflushed (lost on crash/stop). Verified by
    /// reopening a fresh store over the same object backing and finding the record.
    #[tokio::test]
    async fn flush_wal_flushes_buffered_cold_store_in_both_gate_states() {
        use crate::graph::ColdGraphSegmentStore;
        use proximadb_object_store::ProximaObjectStore;
        use proximadb_storage_filesystem_types::ObjectAccessTier;

        for scoped in [None, Some("1")] {
            match scoped {
                Some(v) => unsafe {
                    std::env::set_var("PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE", v)
                },
                None => unsafe { std::env::remove_var("PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE") },
            }

            let backing = ProximaObjectStore::from_url("memory://").expect("mem store");
            // High thresholds ⇒ the write stays buffered until flush_wal flushes it.
            let cold = ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool)
                .with_flush_thresholds(u64::MAX, usize::MAX);
            let graph_id = format!("flush_wal_cold_{}_{scoped:?}", std::process::id());
            let graph_id = graph_id.as_str();
            let service = GraphOperationsService::new().with_canonical_record_store(Arc::new(cold));

            service
                .create_graph_collection(CreateGraphRequest {
                    graph_id: graph_id.to_string(),
                    name: None,
                    description: None,
                    schema: None,
                    storage_config: None,
                    engine_config: None,
                    access_control: None,
                })
                .await
                .expect("create graph");
            service
                .create_node(
                    graph_id,
                    Node {
                        id: "n1".to_string(),
                        labels: vec!["Person".to_string()],
                        properties: HashMap::new(),
                        embedding: None,
                        created_at_ms: 1,
                        updated_at_ms: 1,
                    },
                )
                .await
                .expect("create node");

            // The checkpoint/shutdown path must flush the cold buffer regardless of
            // the scoped gate.
            service.flush_wal(graph_id).await.expect("flush_wal");

            // Reopen over the same backing: the buffered node is now durable.
            let reopened = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
            reopened.load_index().await.expect("load index");
            assert!(
                reopened
                    .get_record(&RecordKey::new(format!("graph/{graph_id}/node/n1")))
                    .await
                    .expect("get")
                    .is_some(),
                "flush_wal flushed the cold buffer (scoped={scoped:?})"
            );

            unsafe { std::env::remove_var("PROXIMADB_GRAPH_CANONICAL_REPLAY_SCOPE") };
        }
    }

    /// #52 end-to-end: a graph backed by the BUFFERED `ColdGraphSegmentStore` that
    /// crashes before the buffer flushes loses the cold copy — but recovery
    /// re-population rebuilds it from the (full-resident) engine, and a flush makes it
    /// durable again. This is the safety property that lets the segment store be wired.
    #[tokio::test]
    async fn segment_store_crash_without_flush_recovers_via_repopulate() {
        use crate::graph::ColdGraphSegmentStore;
        use proximadb_object_store::ProximaObjectStore;
        use proximadb_storage_filesystem_types::ObjectAccessTier;

        let backing = ProximaObjectStore::from_url("memory://").expect("mem store");
        // High thresholds ⇒ writes stay buffered (the pre-flush crash window).
        let cold = Arc::new(
            ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool)
                .with_flush_thresholds(u64::MAX, usize::MAX),
        );
        let service = GraphOperationsService::new().with_canonical_record_store(cold.clone());
        let graph_id = seed_graph(&service, "seg_crash_recover").await;
        let graph_id = graph_id.as_str();

        for node_id in ["n1", "n2", "n3"] {
            service
                .create_node(graph_id, mk_node(node_id, "Person", 1, 1))
                .await
                .expect("create node");
        }
        service
            .create_edge(graph_id, mk_edge("e1", "n1", "n2", "KNOWS", 1, 1))
            .await
            .expect("create edge");

        // CRASH: the buffer was never flushed. A fresh store over the same backing
        // has an empty buffer and `load_index` finds nothing — the records are lost.
        let crashed = ColdGraphSegmentStore::new(backing.clone(), ObjectAccessTier::Cool);
        crashed.load_index().await.expect("load index");
        assert!(
            crashed
                .get_record(&RecordKey::new(format!("graph/{graph_id}/node/n1")))
                .await
                .expect("get")
                .is_none(),
            "buffered records are lost on crash without flush"
        );

        // RECOVERY: re-drive the recovered engine state back into the cold store, then
        // flush to make it durable (in production: `recover_graph` repopulates,
        // `flush_wal` flushes).
        let driven = service
            .repopulate_canonical_store(graph_id)
            .await
            .expect("repopulate");
        assert_eq!(driven, 4, "3 nodes + 1 edge re-driven");
        cold.flush().await.expect("flush");

        // A fresh reopen now finds every node + edge durably.
        let recovered = ColdGraphSegmentStore::new(backing, ObjectAccessTier::Cool);
        recovered.load_index().await.expect("load index");
        for node_id in ["n1", "n2", "n3"] {
            assert!(
                recovered
                    .get_record(&RecordKey::new(format!("graph/{graph_id}/node/{node_id}")))
                    .await
                    .expect("get")
                    .is_some(),
                "node {node_id} durable after recovery + flush"
            );
        }
        assert!(
            recovered
                .get_record(&RecordKey::new(format!("graph/{graph_id}/edge/e1")))
                .await
                .expect("get")
                .is_some(),
            "edge durable after recovery + flush"
        );
    }
}
