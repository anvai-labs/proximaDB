//! Node Operations API (extracted from service.rs)
//!
//! Provides node CRUD, neighbor retrieval, node querying, and maintenance of
//! single-field and multi-field unique constraints for per-graph scopes.

use super::Result;
use crate::graph::NodeQuery;
use crate::graph::adjacency_projection::edge_to_canonical_record;
use crate::graph::engines::GraphEngine;
use crate::graph::{Node, NodeId};
use proximadb_graph::projection::GraphTopologyProjection;
use std::collections::HashSet;
use std::sync::Arc;

impl super::GraphOperationsService {
    /// Get neighbors of a node
    pub async fn get_neighbors(&self, graph_id: &str, node_id: &NodeId) -> Result<Vec<Arc<Node>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        engine.get_neighbors(node_id, None)
    }

    /// Query nodes by a single label (convenience method)
    ///
    /// Returns all nodes that have the specified label in the graph.
    pub async fn query_nodes_by_label(
        &self,
        graph_id: &str,
        label: &str,
    ) -> Result<Vec<Arc<Node>>> {
        let query = NodeQuery {
            graph_id: graph_id.to_string(),
            labels: vec![label.to_string()],
            filters: vec![],
            offset: None,
            limit: None,
            continuation_token: None,
        };
        self.query_nodes(graph_id, query).await
    }

    /// Query nodes by properties with optional label filter (convenience method)
    ///
    /// # Arguments
    /// * `graph_id` - The graph to query
    /// * `label` - Optional label to filter by
    /// * `property_filters` - List of (property_name, value) pairs for equality matching
    pub async fn query_nodes_by_properties(
        &self,
        graph_id: &str,
        label: Option<&str>,
        property_filters: &[(String, String)],
    ) -> Result<Vec<Arc<Node>>> {
        use crate::graph::{PropertyFilter, PropertyFilterOperator, PropertyValue};

        // Build property filters
        let filters: Vec<PropertyFilter> = property_filters
            .iter()
            .map(|(key, value)| PropertyFilter {
                key: key.clone(),
                operator: PropertyFilterOperator::Equals as i32,
                value: Some(PropertyValue {
                    value: Some(crate::graph::model::property_value::Value::StringValue(
                        value.clone(),
                    )),
                }),
            })
            .collect();

        let query = NodeQuery {
            graph_id: graph_id.to_string(),
            labels: label.map(|l| vec![l.to_string()]).unwrap_or_default(),
            filters,
            offset: None,
            limit: None,
            continuation_token: None,
        };

        self.query_nodes(graph_id, query).await
    }

    /// Query nodes by labels and properties
    pub async fn query_nodes(&self, graph_id: &str, query: NodeQuery) -> Result<Vec<Arc<Node>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let engine = self.get_existing_graph_engine(graph_id).await?;

        // Initial candidate set from labels or all nodes
        let mut candidates: HashSet<NodeId> = if !query.labels.is_empty() {
            let mut set = HashSet::new();
            for label in &query.labels {
                if let Ok(nodes) = engine.get_nodes_by_label(label) {
                    for n in nodes {
                        set.insert(n.id.clone());
                    }
                }
            }
            set
        } else {
            engine
                .get_all_nodes()?
                .into_iter()
                .map(|n| n.id.clone())
                .collect()
        };

        // Use property indexes / ordered indexes for prefiltering
        for filter in &query.filters {
            use crate::graph::PropertyFilterOperator as Op;
            match Op::try_from(filter.operator).unwrap_or(Op::Unspecified) {
                Op::Equals => {
                    // Look up index for this property
                    if let Some(index_map) = self.memory_pool.node_property_indexes.get(&filter.key)
                    {
                        // Safely handle missing filter value
                        let Some(filter_val) = filter.value.as_ref() else {
                            continue; // Skip filter if value is missing
                        };
                        let key = super::index_key_for_value(filter_val);
                        if let Some(ids_vec) = index_map.get(&key) {
                            let id_set: HashSet<NodeId> = ids_vec.iter().cloned().collect();
                            candidates.retain(|id| id_set.contains(id));
                        } else {
                            // No matches for this property value; result is empty
                            candidates.clear();
                            break;
                        }
                    } else {
                        // No index for this property; will verify via scan later
                        continue;
                    }
                }
                Op::StartsWith => {
                    // Safely handle missing filter value
                    let Some(filter_val) = filter.value.as_ref() else {
                        continue; // Skip filter if value is missing
                    };
                    if let Some(prefix) = super::extract_string_from_value(filter_val)
                        && let Some(map_lock) =
                            self.memory_pool.node_property_str_ordered.get(&filter.key)
                    {
                        let map = map_lock.read();
                        let mut matched = HashSet::new();
                        for (_k, ids) in map
                            .range(prefix.to_string()..)
                            .take_while(|(k, _)| k.starts_with(prefix))
                        {
                            matched.extend(ids.iter().cloned());
                        }
                        candidates.retain(|id| matched.contains(id));
                    }
                }
                Op::GreaterThan | Op::GreaterEqual | Op::LessThan | Op::LessEqual => {
                    // Safely handle missing filter value
                    let Some(filter_val) = filter.value.as_ref() else {
                        continue; // Skip filter if value is missing
                    };
                    // Prefer numeric range if value numeric, else fallback to string ordered
                    if let Some(num) = super::extract_number_from_value(filter_val)
                        && let Some(map_lock) =
                            self.memory_pool.node_property_num_indexes.get(&filter.key)
                    {
                        let map = map_lock.read();
                        let mut matched = HashSet::new();
                        match Op::try_from(filter.operator).unwrap_or(Op::Unspecified) {
                            Op::GreaterThan => {
                                for (k, ids) in map.iter() {
                                    if (*k as f64) > num {
                                        matched.extend(ids.iter().cloned());
                                    }
                                }
                            }
                            Op::GreaterEqual => {
                                for (k, ids) in map.iter() {
                                    if (*k as f64) >= num {
                                        matched.extend(ids.iter().cloned());
                                    }
                                }
                            }
                            Op::LessThan => {
                                for (k, ids) in map.iter() {
                                    if (*k as f64) < num {
                                        matched.extend(ids.iter().cloned());
                                    }
                                }
                            }
                            Op::LessEqual => {
                                for (k, ids) in map.iter() {
                                    if (*k as f64) <= num {
                                        matched.extend(ids.iter().cloned());
                                    }
                                }
                            }
                            _ => {}
                        }
                        candidates.retain(|id| matched.contains(id));
                    } else if let Some(map_lock) =
                        self.memory_pool.node_property_str_ordered.get(&filter.key)
                    {
                        let map = map_lock.read();
                        let mut matched = HashSet::new();
                        // filter_val was already validated at the start of this Op branch
                        let s = super::extract_string_from_value(filter_val).unwrap_or("");
                        match Op::try_from(filter.operator).unwrap_or(Op::Unspecified) {
                            Op::GreaterThan => {
                                for (_k, ids) in map.range((
                                    std::ops::Bound::Excluded(s.to_string()),
                                    std::ops::Bound::Unbounded,
                                )) {
                                    matched.extend(ids.iter().cloned());
                                }
                            }
                            Op::GreaterEqual => {
                                for (_k, ids) in map.range((
                                    std::ops::Bound::Included(s.to_string()),
                                    std::ops::Bound::Unbounded,
                                )) {
                                    matched.extend(ids.iter().cloned());
                                }
                            }
                            Op::LessThan => {
                                for (_k, ids) in map.range((
                                    std::ops::Bound::Unbounded,
                                    std::ops::Bound::Excluded(s.to_string()),
                                )) {
                                    matched.extend(ids.iter().cloned());
                                }
                            }
                            Op::LessEqual => {
                                for (_k, ids) in map.range((
                                    std::ops::Bound::Unbounded,
                                    std::ops::Bound::Included(s.to_string()),
                                )) {
                                    matched.extend(ids.iter().cloned());
                                }
                            }
                            _ => {}
                        }
                        candidates.retain(|id| matched.contains(id));
                    }
                }
                _ => {}
            }
        }

        // Final verification scan over candidate set
        let mut results: Vec<Arc<Node>> = Vec::new();
        for id in candidates.into_iter() {
            if let Ok(Some(node)) = engine.get_node(&id) {
                // Additional property filters
                if !query.filters.is_empty() {
                    let mut pass_all = true;
                    for filter in &query.filters {
                        use crate::graph::PropertyFilterOperator as Op;
                        let prop_val_opt = node.properties.get(&filter.key);

                        // Safely get filter value - if missing, filter fails
                        let Some(filter_val) = filter.value.as_ref() else {
                            pass_all = false;
                            break;
                        };

                        let pass = match Op::try_from(filter.operator).unwrap_or(Op::Unspecified) {
                            Op::Equals => match prop_val_opt {
                                Some(v) => v.value == filter_val.value,
                                None => false,
                            },
                            Op::NotEquals => match prop_val_opt {
                                Some(v) => v.value != filter_val.value,
                                None => true,
                            },
                            Op::GreaterThan => super::cmp_prop_gt(prop_val_opt, filter_val),
                            Op::GreaterEqual => super::cmp_prop_ge(prop_val_opt, filter_val),
                            Op::LessThan => super::cmp_prop_lt(prop_val_opt, filter_val),
                            Op::LessEqual => super::cmp_prop_le(prop_val_opt, filter_val),
                            Op::StartsWith => super::prop_starts_with(prop_val_opt, filter_val),
                            Op::Contains => super::prop_contains(prop_val_opt, filter_val),
                            _ => false,
                        };
                        if !pass {
                            pass_all = false;
                            break;
                        }
                    }
                    if !pass_all {
                        continue;
                    }
                }
                results.push(node);
            }
        }

        // Stable total order by node id before paging. The candidate set is
        // gathered from hash maps (non-deterministic iteration), so without this
        // sort `offset`/`limit` would shift between calls and paginated reads
        // (e.g. the Flight columnar export) would duplicate or skip rows.
        results.sort_by(|a, b| a.id.cmp(&b.id));

        // Apply offset/limit for pagination
        let offset = query.offset.unwrap_or(0) as usize;
        let limit = query.limit.unwrap_or(results.len() as u32) as usize;
        if offset >= results.len() {
            return Ok(Vec::new());
        }
        let end = (offset + limit).min(results.len());
        Ok(results.drain(offset..end).collect())
    }
    /// Create a new node
    pub async fn create_node(&self, graph_id: &str, node: Node) -> Result<Arc<Node>> {
        tracing::debug!(
            "GraphOperationsService::create_node called for graph: {}, node: {}",
            graph_id,
            node.id
        );

        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let engine = self.get_or_create_graph_engine(graph_id).await?;
        tracing::debug!("Got engine, enforcing schema for node: {}", node.id);
        self.enforce_schema_on_node(graph_id, &node).await?;
        self.enforce_unique_constraints_on_node(graph_id, &node)?;
        self.enforce_multi_unique_constraints_on_node(graph_id, &node)?;
        self.upsert_canonical_node_record(graph_id, &node).await?;

        tracing::debug!("Calling engine.insert_node for node: {}", node.id);
        let node_arc = engine.insert_node(node).await?;
        tracing::debug!("engine.insert_node returned successfully");
        self.register_node_in_unique_constraints(graph_id, &node_arc);
        self.register_node_in_multi_unique_constraints(graph_id, &node_arc);
        Ok(node_arc)
    }

    /// Get a node by ID
    pub async fn get_node(&self, graph_id: &str, id: &NodeId) -> Result<Option<Arc<Node>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        if let Some(node) = engine.get_node(id)? {
            return Ok(Some(node));
        }
        // TD-168 cold-payload tier (gated default-OFF): on a hot (in-RAM) miss,
        // serve the payload from the byte-budgeted cache over the canonical record
        // store, so a graph whose payloads exceed RAM is still servable. The gate is
        // checked inside `cold_fetch_node`, which returns `None` (today's behavior)
        // when the tier is off.
        self.cold_fetch_node(graph_id, id).await
    }

    /// Update a node, applying ADR-012 merge semantics for label sets.
    pub async fn update_node(&self, graph_id: &str, mut node: Node) -> Result<Arc<Node>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;

        // ADR-012 Add-Wins label union: preserve labels from the existing node.
        if let Ok(Some(existing)) = engine.get_node(&node.id) {
            node.labels = crate::graph::merge::merge_labels(&existing.labels, &node.labels);
        }

        self.enforce_schema_on_node(graph_id, &node).await?;
        self.enforce_unique_constraints_on_node(graph_id, &node)?;
        self.enforce_multi_unique_constraints_on_node(graph_id, &node)?;
        self.upsert_canonical_node_record(graph_id, &node).await?;
        let node_arc = engine.update_node(node).await?;
        self.register_node_in_unique_constraints(graph_id, &node_arc);
        self.register_node_in_multi_unique_constraints(graph_id, &node_arc);
        Ok(node_arc)
    }

    /// Delete a node
    pub async fn delete_node(&self, graph_id: &str, id: &NodeId) -> Result<Option<Arc<Node>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        if let Some(node) = engine.get_node(id)? {
            self.unregister_node_from_unique_constraints(graph_id, &node);
            self.unregister_node_from_multi_unique_constraints(graph_id, &node);
        }
        let deleted = crate::graph::engines::GraphEngine::delete_node(&*engine, id).await?;
        if deleted.is_some() {
            self.delete_canonical_node_record(graph_id, id).await?;
        }
        Ok(deleted)
    }

    /// Delete a node and detach all incident edges (DETACH mode)
    pub async fn delete_node_detach(
        &self,
        graph_id: &str,
        id: &NodeId,
    ) -> Result<Option<Arc<Node>>> {
        if !self.graph_enabled() {
            return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        let mut edge_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for e in engine.get_outgoing_edges(id, None)? {
            edge_ids.insert(e.id.clone());
        }
        for e in engine.get_incoming_edges(id, None)? {
            edge_ids.insert(e.id.clone());
        }
        for eid in edge_ids.into_iter() {
            if let Some(edge) =
                crate::graph::engines::GraphEngine::delete_edge(&*engine, &eid).await?
            {
                let edge_record = edge_to_canonical_record(graph_id, &edge);
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
        }
        if let Some(node) = engine.get_node(id)? {
            self.unregister_node_from_unique_constraints(graph_id, &node);
            self.unregister_node_from_multi_unique_constraints(graph_id, &node);
        }
        let deleted = crate::graph::engines::GraphEngine::delete_node(&*engine, id).await?;
        if deleted.is_some() {
            self.delete_canonical_node_record(graph_id, id).await?;
        }
        Ok(deleted)
    }

    // ===== Unique constraints (single-field) =====
    pub(super) fn enforce_unique_constraints_on_node(
        &self,
        graph_id: &str,
        node: &Node,
    ) -> Result<()> {
        for label in &node.labels {
            for entry in self.memory_pool.unique_constraints.iter() {
                let (cgraph, clabel, cprop) = entry.key();
                let map = entry.value();
                if cgraph == graph_id
                    && clabel == label
                    && let Some(val) = node.properties.get(cprop)
                {
                    let k = super::index_key_for_value(val);
                    if let Some(existing) = map.get(&k)
                        && existing.value() != &node.id
                    {
                        return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                            format!(
                                "Unique constraint violation on (label='{}', property='{}') for value '{}'",
                                clabel, cprop, k
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn register_node_in_unique_constraints(&self, graph_id: &str, node: &Arc<Node>) {
        for label in &node.labels {
            let label = label.clone();
            for entry in self.memory_pool.unique_constraints.iter() {
                let (cgraph, clabel, cprop) = entry.key();
                let map = entry.value();
                if cgraph == graph_id
                    && *clabel == label
                    && let Some(val) = node.properties.get(cprop)
                {
                    let k = super::index_key_for_value(val);
                    map.insert(k, node.id.clone());
                }
            }
        }
    }

    pub(super) fn unregister_node_from_unique_constraints(&self, graph_id: &str, node: &Arc<Node>) {
        for label in &node.labels {
            let label = label.clone();
            for entry in self.memory_pool.unique_constraints.iter() {
                let (cgraph, clabel, cprop) = entry.key();
                let map = entry.value();
                if cgraph == graph_id
                    && *clabel == label
                    && let Some(val) = node.properties.get(cprop)
                {
                    let k = super::index_key_for_value(val);
                    if let Some(existing) = map.get(&k)
                        && existing.value() == &node.id
                    {
                        map.remove(&k);
                    }
                }
            }
        }
    }

    // ===== Unique constraints (multi-field) =====
    pub(super) fn enforce_multi_unique_constraints_on_node(
        &self,
        graph_id: &str,
        node: &Node,
    ) -> Result<()> {
        for entry in self.memory_pool.unique_constraints_multi.iter() {
            let (cgraph, labels_key, props_key) = entry.key();
            if cgraph != graph_id {
                continue;
            }
            let props: Vec<String> = props_key.split('|').map(|s| s.to_string()).collect();
            let labels: Vec<String> = labels_key.split('|').map(|s| s.to_string()).collect();
            if !super::GraphOperationsService::node_has_all_labels(node, &labels) {
                continue;
            }
            if let Some(comp) = super::GraphOperationsService::composite_key_for_node(node, &props)
            {
                let map = entry.value();
                if let Some(existing) = map.get(&comp)
                    && existing.value() != &node.id
                {
                    return Err(proximadb_kernel::error::ProximaDBError::InvalidInput(
                        format!("Duplicate composite key for unique ({:?})", props),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn register_node_in_multi_unique_constraints(
        &self,
        graph_id: &str,
        node: &Arc<Node>,
    ) {
        for entry in self.memory_pool.unique_constraints_multi.iter_mut() {
            let (cgraph, labels_key, props_key) = entry.key().clone();
            if cgraph != graph_id {
                continue;
            }
            let props: Vec<String> = props_key.split('|').map(|s| s.to_string()).collect();
            let labels: Vec<String> = labels_key.split('|').map(|s| s.to_string()).collect();
            if !super::GraphOperationsService::node_has_all_labels(node, &labels) {
                continue;
            }
            if let Some(comp) = super::GraphOperationsService::composite_key_for_node(node, &props)
            {
                entry.value().insert(comp, node.id.clone());
            }
        }
    }

    pub(super) fn unregister_node_from_multi_unique_constraints(
        &self,
        graph_id: &str,
        node: &Arc<Node>,
    ) {
        for entry in self.memory_pool.unique_constraints_multi.iter_mut() {
            let (cgraph, labels_key, props_key) = entry.key().clone();
            if cgraph != graph_id {
                continue;
            }
            let props: Vec<String> = props_key.split('|').map(|s| s.to_string()).collect();
            let labels: Vec<String> = labels_key.split('|').map(|s| s.to_string()).collect();
            if !super::GraphOperationsService::node_has_all_labels(node, &labels) {
                continue;
            }
            if let Some(comp) = super::GraphOperationsService::composite_key_for_node(node, &props)
            {
                entry.value().remove(&comp);
            }
        }
    }
}
