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

//! Rebuildable graph adjacency projections over canonical edge records.
//!
//! This module is the root-runtime bridge for Phase 3 of
//! `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.
//! It indexes canonical graph edge `ProximaRecord`s into `edges_by_src` and
//! `edges_by_dst` tables for write-heavy graph paths. The projection is not
//! durable authority; it can be dropped and rebuilt from canonical records.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::RwLock;

use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_graph::projection::{
    AdjacencyIndexKind, AdjacencyProjectionEntry, GraphTopologyDescriptor, GraphTopologyProjection,
    ProjectionResult, TopologyApplyResult, TopologyEpoch, TopologyFormat,
    adjacency_entries_for_edge,
};
use proximadb_graph::record::{
    CanonicalEdge, CanonicalNode, GraphNodeKey, canonical_edge_from_record,
    canonical_node_from_record,
};
use proximadb_records::{ProximaRecord, ProximaTree, ProximaTreeNode};

use crate::graph::{Edge, Node};
use crate::graph::{PropertyArray, PropertyObject, PropertyValue, property_value};

/// In-memory adjacency table projection for one graph dataset.
#[derive(Debug)]
pub struct InMemoryGraphAdjacencyProjection {
    descriptor: GraphTopologyDescriptor,
    state: RwLock<AdjacencyState>,
}

#[derive(Debug, Default)]
struct AdjacencyState {
    epoch: TopologyEpoch,
    edges_by_src: BTreeMap<String, BTreeMap<String, AdjacencyProjectionEntry>>,
    edges_by_dst: BTreeMap<String, BTreeMap<String, AdjacencyProjectionEntry>>,
    /// Reverse index `edge_oid -> (src_node_oid, dst_node_oid)`, maintained by
    /// `apply_entries` / `remove_edge_oid` (the only two state mutators, which
    /// every apply/remove/rebuild path funnels through). It exists so removal
    /// can address exactly two buckets instead of scanning every node key:
    /// the previous `remove_edge_oid` walked BOTH outer maps per edge, which
    /// made bulk ingest O(E*V) — measured at 3.5-7 s per 1,000-edge batch on a
    /// ~90k-node graph, ~1,300 s over a repo-scale index, and the dominant
    /// term in ProximaDB losing 3.6x to client-side SQLite on the same corpus.
    edge_endpoints: HashMap<String, (String, String)>,
}

impl InMemoryGraphAdjacencyProjection {
    pub fn new(graph_id: impl Into<String>) -> Self {
        let graph_id = graph_id.into();
        Self {
            descriptor: GraphTopologyDescriptor::new(
                format!("adjacency-{graph_id}"),
                graph_id,
                TopologyFormat::AdjacencyTable,
            ),
            state: RwLock::new(AdjacencyState::default()),
        }
    }

    /// Return outgoing adjacency entries for a canonical source node oid.
    pub fn edges_by_src(&self, node_oid: &str) -> ProjectionResult<Vec<AdjacencyProjectionEntry>> {
        let state = self.state.read().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection read lock poisoned".to_string(),
            )
        })?;
        Ok(state
            .edges_by_src
            .get(node_oid)
            .map(|edges| edges.values().cloned().collect())
            .unwrap_or_default())
    }

    /// Return incoming adjacency entries for a canonical destination node oid.
    pub fn edges_by_dst(&self, node_oid: &str) -> ProjectionResult<Vec<AdjacencyProjectionEntry>> {
        let state = self.state.read().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection read lock poisoned".to_string(),
            )
        })?;
        Ok(state
            .edges_by_dst
            .get(node_oid)
            .map(|edges| edges.values().cloned().collect())
            .unwrap_or_default())
    }

    pub fn edge_count(&self) -> ProjectionResult<usize> {
        let state = self.state.read().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection read lock poisoned".to_string(),
            )
        })?;
        Ok(state
            .edges_by_src
            .values()
            .map(std::collections::BTreeMap::len)
            .sum())
    }

    /// Snapshot all unique edges as `(from_node_oid, to_node_oid, edge_oid)` tuples.
    ///
    /// Only iterates `edges_by_src` (each edge appears exactly once there) so
    /// callers receive one tuple per edge without duplicates.  Used by
    /// `GraphOperationsService::rebuild_orion_csr_from_adjacency_projection`.
    pub fn snapshot_edge_endpoints(&self) -> ProjectionResult<Vec<(String, String, String)>> {
        let state = self.state.read().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection read lock poisoned".to_string(),
            )
        })?;
        let mut result = Vec::new();
        for (src_node_oid, edges) in &state.edges_by_src {
            for (edge_oid, entry) in edges {
                result.push((
                    src_node_oid.clone(),
                    entry.opposite_node_oid.clone(),
                    edge_oid.clone(),
                ));
            }
        }
        Ok(result)
    }

    fn apply_entries(state: &mut AdjacencyState, entries: [AdjacencyProjectionEntry; 2]) -> usize {
        // Record the endpoints before the entries move: removal must be able to
        // name its two buckets without scanning (see `edge_endpoints`).
        let edge_oid_for_index = entries[0].key.edge_oid.clone();
        let mut src_oid = None;
        let mut dst_oid = None;
        for entry in &entries {
            match entry.key.kind {
                AdjacencyIndexKind::EdgesBySrc => src_oid = Some(entry.key.node_oid.clone()),
                AdjacencyIndexKind::EdgesByDst => dst_oid = Some(entry.key.node_oid.clone()),
            }
        }
        if let (Some(src), Some(dst)) = (src_oid, dst_oid) {
            state.edge_endpoints.insert(edge_oid_for_index, (src, dst));
        }

        let mut written = 0;
        for entry in entries {
            let edge_oid = entry.key.edge_oid.clone();
            match entry.key.kind {
                AdjacencyIndexKind::EdgesBySrc => {
                    let replaced = state
                        .edges_by_src
                        .entry(entry.key.node_oid.clone())
                        .or_default()
                        .insert(edge_oid, entry)
                        .is_some();
                    if !replaced {
                        written += 1;
                    }
                }
                AdjacencyIndexKind::EdgesByDst => {
                    let replaced = state
                        .edges_by_dst
                        .entry(entry.key.node_oid.clone())
                        .or_default()
                        .insert(edge_oid, entry)
                        .is_some();
                    if !replaced {
                        written += 1;
                    }
                }
            }
        }
        written
    }

    fn remove_edge_oid(state: &mut AdjacencyState, edge_oid: &str) -> bool {
        // The reverse index names the exact (src, dst) buckets, so this is two
        // O(log n) map operations. It used to be a full scan of BOTH outer maps
        // per edge — O(V) per call, O(E*V) for a bulk apply, and the measured
        // dominant cost of repo-scale graph ingest (see `edge_endpoints`).
        // An oid absent from the index is absent from the maps by construction:
        // every insert goes through `apply_entries`, which records it. No
        // fallback scan — one would silently reintroduce the O(V) behaviour.
        let Some((src, dst)) = state.edge_endpoints.remove(edge_oid) else {
            return false;
        };
        let mut removed = false;
        for (by_node, node_oid) in [
            (&mut state.edges_by_src, &src),
            (&mut state.edges_by_dst, &dst),
        ] {
            if let Some(edges) = by_node.get_mut(node_oid) {
                removed |= edges.remove(edge_oid).is_some();
                if edges.is_empty() {
                    by_node.remove(node_oid);
                }
            }
        }
        removed
    }

    /// TD-130: apply a batch of canonical edge records under a SINGLE write-lock
    /// acquisition and a SINGLE epoch bump, instead of one lock-cycle + epoch
    /// bump per edge (`apply_edge`). This removes the per-edge serialization that
    /// dominated bulk edge ingest — the projection lock, not the O(1) composite
    /// probe, was the measured per-edge cost on the batch path (e.g. ~96.5K
    /// edges/repo at initial code-graph index). Entries are computed outside the
    /// lock; upsert semantics match `apply_edge` (each edge's prior entries are
    /// removed first). Returns the total adjacency entries written.
    pub fn apply_edges(&self, edge_records: &[ProximaRecord]) -> ProjectionResult<usize> {
        // Compute all entries before taking the lock; fail closed on any record
        // that is not a canonical graph edge (same contract as `apply_edge`).
        let mut prepared = Vec::with_capacity(edge_records.len());
        for edge_record in edge_records {
            let entries = adjacency_entries_for_edge(edge_record).ok_or_else(|| {
                proximadb_kernel::error::ProximaDBError::InvalidInput(
                    "record is not a canonical graph edge".to_string(),
                )
            })?;
            prepared.push((edge_record.oid.as_str(), entries));
        }
        if prepared.is_empty() {
            return Ok(0);
        }

        let mut state = self.state.write().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection write lock poisoned".to_string(),
            )
        })?;
        let mut entries_written = 0;
        for (edge_oid, entries) in prepared {
            Self::remove_edge_oid(&mut state, edge_oid);
            entries_written += Self::apply_entries(&mut state, entries);
        }
        // One epoch transition for the whole batch, so readers observe the batch
        // atomically rather than N intermediate epochs.
        state.epoch = state.epoch.next();
        Ok(entries_written)
    }
}

/// Convert the root graph edge compatibility shape into a canonical edge
/// record for projection maintenance.
pub fn edge_to_canonical_record(graph_id: &str, edge: &Edge) -> ProximaRecord {
    let mut canonical = CanonicalEdge::new(
        graph_id,
        edge.id.clone(),
        GraphNodeKey::new(graph_id, edge.from_node_id.clone()).canonical_oid(),
        GraphNodeKey::new(graph_id, edge.to_node_id.clone()).canonical_oid(),
        edge.edge_type.clone(),
        property_map_to_tree(&edge.properties),
    );

    canonical.created_at_ns = edge.created_at_ms.saturating_mul(1_000_000);
    canonical.updated_at_ns = edge.updated_at_ms.saturating_mul(1_000_000);
    canonical.into_proxima_record()
}

/// Convert the root graph node compatibility shape into a canonical node
/// record for durable record-store writes.
pub fn node_to_canonical_record(graph_id: &str, node: &Node) -> ProximaRecord {
    let node_label = node.labels.first().cloned().unwrap_or_default();
    let mut canonical = CanonicalNode::new(
        graph_id,
        node.id.clone(),
        node_label,
        property_map_to_tree(&node.properties),
    );

    canonical.created_at_ns = node.created_at_ms.saturating_mul(1_000_000);
    canonical.updated_at_ns = node.updated_at_ms.saturating_mul(1_000_000);
    canonical.into_proxima_record()
}

/// Reconstruct the root graph node compatibility shape from a canonical record —
/// the inverse of [`node_to_canonical_record`]. Returns `None` if the record is
/// not a graph-node record. Used by the cold-payload tier (TD-168) to materialize
/// a `Node` fetched from the canonical record store on a cache miss.
///
/// Fields the forward projection does not persist are reconstructed as defaults:
/// `embedding` is `None`, and `labels` carries the single stored label (the
/// canonical record keeps one label, not the full multi-label set).
pub fn node_from_canonical_record(record: &ProximaRecord) -> Option<Node> {
    let canonical = canonical_node_from_record(record)?;
    let labels = if canonical.node_label.is_empty() {
        Vec::new()
    } else {
        vec![canonical.node_label]
    };
    Some(Node {
        id: canonical.key.node_id,
        labels,
        properties: tree_to_property_map(&canonical.properties),
        embedding: None,
        created_at_ms: ns_to_ms(canonical.created_at_ns),
        updated_at_ms: ns_to_ms(canonical.updated_at_ns),
    })
}

/// Reconstruct the root graph edge compatibility shape from a canonical record —
/// the inverse of [`edge_to_canonical_record`]. Endpoints are recovered by
/// reversing the canonical node oid (`graph/{graph_id}/node/{node_id}`). `weight`
/// is not persisted by the forward projection and is reconstructed as `None`.
pub fn edge_from_canonical_record(record: &ProximaRecord) -> Option<Edge> {
    let canonical = canonical_edge_from_record(record)?;
    let graph_id = canonical.key.graph_id.clone();
    Some(Edge {
        id: canonical.key.edge_id,
        from_node_id: node_id_from_oid(&canonical.src_node_oid, &graph_id),
        to_node_id: node_id_from_oid(&canonical.dst_node_oid, &graph_id),
        edge_type: canonical.edge_label,
        properties: tree_to_property_map(&canonical.properties),
        weight: None,
        created_at_ms: ns_to_ms(canonical.created_at_ns),
        updated_at_ms: ns_to_ms(canonical.updated_at_ns),
    })
}

/// Recover the original node id from a canonical node oid
/// (`graph/{graph_id}/node/{node_id}`). Falls back to the whole oid if the prefix
/// does not match — never panics.
fn node_id_from_oid(oid: &str, graph_id: &str) -> String {
    oid.strip_prefix(&format!("graph/{graph_id}/node/"))
        .unwrap_or(oid)
        .to_string()
}

/// Canonical record clock is nanoseconds; the graph model carries milliseconds.
fn ns_to_ms(ns: i64) -> i64 {
    ns / 1_000_000
}

fn tree_to_property_map(tree: &ProximaTree) -> std::collections::HashMap<String, PropertyValue> {
    tree.iter()
        .map(|(key, node)| (key.clone(), tree_node_to_property_value(node)))
        .collect()
}

fn tree_node_to_property_value(node: &ProximaTreeNode) -> PropertyValue {
    match node {
        ProximaTreeNode::Object(fields) => PropertyValue {
            value: Some(property_value::Value::ObjectValue(PropertyObject {
                fields: fields
                    .iter()
                    .map(|(key, child)| (key.clone(), tree_node_to_property_value(child)))
                    .collect(),
            })),
        },
        ProximaTreeNode::Value(value) => proxima_value_to_property_value(value),
    }
}

pub(crate) fn proxima_value_to_property_value(value: &ProximaValue) -> PropertyValue {
    use property_value::Value;
    // The neutral graph object type does not distinguish ProximaValue::Map from
    // ProximaValue::Struct. Both retain their fields and lower to ObjectValue.
    let inner = match value {
        ProximaValue::String(v) | ProximaValue::Symbol(v) | ProximaValue::Decimal(v) => {
            Some(Value::StringValue(v.clone()))
        }
        ProximaValue::Int8(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::Int16(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::Int32(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::Int64(v) => Some(Value::IntValue(*v)),
        ProximaValue::UInt8(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::UInt16(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::UInt32(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::UInt64(v) => match i64::try_from(*v) {
            Ok(value) => Some(Value::IntValue(value)),
            Err(_) => Some(Value::StringValue(v.to_string())),
        },
        ProximaValue::Float16(v) | ProximaValue::Float32(v) => {
            Some(Value::DoubleValue(f64::from(*v)))
        }
        ProximaValue::Float64(v) => Some(Value::DoubleValue(*v)),
        ProximaValue::Boolean(v) => Some(Value::BoolValue(*v)),
        ProximaValue::Binary(v) | ProximaValue::BinaryVector(v) => {
            Some(Value::BytesValue(v.clone()))
        }
        ProximaValue::Date(v) => Some(Value::IntValue(i64::from(*v))),
        ProximaValue::Time(v, _)
        | ProximaValue::Timestamp(v, _)
        | ProximaValue::TimestampTz(v, _) => Some(Value::IntValue(*v)),
        ProximaValue::Uuid(v) | ProximaValue::ULID(v) => Some(Value::BytesValue(v.to_vec())),
        ProximaValue::Array(items) => Some(Value::ArrayValue(PropertyArray {
            values: items.iter().map(proxima_value_to_property_value).collect(),
        })),
        ProximaValue::Struct(fields) | ProximaValue::Map(fields) => {
            Some(Value::ObjectValue(PropertyObject {
                fields: fields
                    .iter()
                    .map(|(key, child)| (key.clone(), proxima_value_to_property_value(child)))
                    .collect(),
            }))
        }
        ProximaValue::DenseVector(v) => Some(Value::VectorValue(v.clone())),
        // Round 8: JSON(B) maps to canonical JSON text — the wildcard silently
        // dropped it at this live projection seam (the round-7 fix landed in
        // the dead sql_value_to_property_value twin).
        ProximaValue::Json(v) | ProximaValue::Jsonb(v) => match v {
            // A JSON null stays the property model's null form — rendering it
            // as the string "null" would flip null-equality to string
            // equality at graph filters (round 12).
            serde_json::Value::Null => None,
            // Sorted-key canonical text — unsorted varies with
            // preserve_order/insertion order across write seams (round 18).
            other => Some(Value::StringValue(
                crate::storage::entity_store::graph_schema::canonical_json_string(other),
            )),
        },
        ProximaValue::SparseVector { .. } => Some(Value::StringValue(
            // Same canonical-text contract as the JSON(B) arm (round 18):
            // unsorted keys vary with preserve_order across write seams.
            crate::storage::entity_store::graph_schema::canonical_json_string(
                &proximadb_records::conversions::proxima_to_json(value),
            ),
        )),
        ProximaValue::Null => None,
    };
    PropertyValue { value: inner }
}

fn property_map_to_tree(
    properties: &std::collections::HashMap<String, PropertyValue>,
) -> ProximaTree {
    properties
        .iter()
        .map(|(key, value)| (key.clone(), property_value_to_tree_node(value)))
        .collect()
}

fn property_value_to_tree_node(value: &PropertyValue) -> ProximaTreeNode {
    match &value.value {
        Some(property_value::Value::ObjectValue(object)) => ProximaTreeNode::Object(
            object
                .fields
                .iter()
                .map(|(key, value)| (key.clone(), property_value_to_tree_node(value)))
                .collect(),
        ),
        _ => ProximaTreeNode::Value(property_value_to_proxima(value)),
    }
}

pub(crate) fn property_value_to_proxima(value: &PropertyValue) -> ProximaValue {
    match &value.value {
        Some(property_value::Value::StringValue(value)) => ProximaValue::String(value.clone()),
        Some(property_value::Value::IntValue(value)) => ProximaValue::Int64(*value),
        Some(property_value::Value::DoubleValue(value)) => ProximaValue::Float64(*value),
        Some(property_value::Value::BoolValue(value)) => ProximaValue::Boolean(*value),
        Some(property_value::Value::BytesValue(value)) => ProximaValue::Binary(value.clone()),
        Some(property_value::Value::ArrayValue(array)) => {
            ProximaValue::Array(array.values.iter().map(property_value_to_proxima).collect())
        }
        Some(property_value::Value::ObjectValue(object)) => ProximaValue::Struct(
            object
                .fields
                .iter()
                .map(|(key, value)| (key.clone(), property_value_to_proxima(value)))
                .collect(),
        ),
        Some(property_value::Value::VectorValue(vector)) => {
            ProximaValue::DenseVector(vector.clone())
        }
        None => ProximaValue::Null,
    }
}

#[async_trait]
impl GraphTopologyProjection for InMemoryGraphAdjacencyProjection {
    fn descriptor(&self) -> &GraphTopologyDescriptor {
        &self.descriptor
    }

    fn applied_epoch(&self) -> TopologyEpoch {
        self.state
            .read()
            .map(|state| state.epoch)
            .unwrap_or_else(|_| TopologyEpoch::initial())
    }

    async fn apply_edge(
        &self,
        edge_record: &ProximaRecord,
    ) -> ProjectionResult<TopologyApplyResult> {
        let entries = adjacency_entries_for_edge(edge_record).ok_or_else(|| {
            proximadb_kernel::error::ProximaDBError::InvalidInput(
                "record is not a canonical graph edge".to_string(),
            )
        })?;

        let mut state = self.state.write().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection write lock poisoned".to_string(),
            )
        })?;

        Self::remove_edge_oid(&mut state, &edge_record.oid);
        let entries_written = Self::apply_entries(&mut state, entries);
        state.epoch = state.epoch.next();

        Ok(TopologyApplyResult {
            edge_oid: edge_record.oid.clone(),
            entries_written,
            applied_epoch: state.epoch,
        })
    }

    async fn remove_edge(&self, edge_record: &ProximaRecord) -> ProjectionResult<bool> {
        let mut state = self.state.write().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection write lock poisoned".to_string(),
            )
        })?;
        let removed = Self::remove_edge_oid(&mut state, &edge_record.oid);
        if removed {
            state.epoch = state.epoch.next();
        }
        Ok(removed)
    }

    async fn rebuild_from_records(
        &self,
        _node_records: &[ProximaRecord],
        edge_records: &[ProximaRecord],
    ) -> ProjectionResult<usize> {
        let mut next = AdjacencyState::default();
        let mut edge_oids = BTreeSet::new();

        for edge_record in edge_records {
            if let Some(entries) = adjacency_entries_for_edge(edge_record) {
                Self::apply_entries(&mut next, entries);
                edge_oids.insert(edge_record.oid.clone());
            }
        }

        next.epoch = self.applied_epoch().next();
        let edge_count = edge_oids.len();

        let mut state = self.state.write().map_err(|_| {
            proximadb_kernel::error::ProximaDBError::Internal(
                "graph adjacency projection write lock poisoned".to_string(),
            )
        })?;
        *state = next;

        Ok(edge_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_graph::record::{CanonicalEdge, GraphNodeKey};
    use proximadb_records::ProximaTree;
    use std::collections::HashMap;

    fn edge_record(graph_id: &str, edge_id: &str, src: &str, dst: &str) -> ProximaRecord {
        CanonicalEdge::new(
            graph_id,
            edge_id,
            GraphNodeKey::new(graph_id, src).canonical_oid(),
            GraphNodeKey::new(graph_id, dst).canonical_oid(),
            "KNOWS",
            ProximaTree::new(),
        )
        .into_proxima_record()
    }

    #[test]
    fn root_edge_maps_to_canonical_edge_record() {
        let mut properties = HashMap::new();
        properties.insert(
            "rank".to_string(),
            PropertyValue {
                value: Some(property_value::Value::IntValue(7)),
            },
        );
        let edge = Edge {
            id: "e1".to_string(),
            from_node_id: "n1".to_string(),
            to_node_id: "n2".to_string(),
            edge_type: "KNOWS".to_string(),
            properties,
            weight: Some(0.5),
            created_at_ms: 10,
            updated_at_ms: 20,
        };

        let record = edge_to_canonical_record("g1", &edge);

        assert_eq!(record.oid, "graph/g1/edge/e1");
        assert_eq!(record.created_at_ns, 10_000_000);
        assert_eq!(record.updated_at_ns, 20_000_000);
        assert_eq!(
            record.edge.as_ref().expect("edge shape").source_id,
            "graph/g1/node/n1"
        );
        assert_eq!(
            record.props.get("rank"),
            Some(&ProximaTreeNode::Value(ProximaValue::Int64(7)))
        );
    }

    #[test]
    fn root_node_maps_to_canonical_node_record() {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            PropertyValue {
                value: Some(property_value::Value::StringValue("Ada".to_string())),
            },
        );
        let node = Node {
            id: "n1".to_string(),
            labels: vec!["Person".to_string()],
            properties,
            embedding: None,
            created_at_ms: 10,
            updated_at_ms: 20,
        };

        let record = node_to_canonical_record("g1", &node);

        assert_eq!(record.oid, "graph/g1/node/n1");
        assert_eq!(record.created_at_ns, 10_000_000);
        assert_eq!(record.updated_at_ns, 20_000_000);
        assert_eq!(
            record.props.get("name"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "Ada".to_string()
            )))
        );
    }

    fn prop_str(s: &str) -> PropertyValue {
        PropertyValue {
            value: Some(property_value::Value::StringValue(s.to_string())),
        }
    }
    fn prop_int(i: i64) -> PropertyValue {
        PropertyValue {
            value: Some(property_value::Value::IntValue(i)),
        }
    }

    #[test]
    fn node_round_trips_through_canonical_record() {
        // TD-168: a node fetched from the cold record store must reconstruct
        // identically to the original (over the fields the canonical record
        // persists — single label, no embedding).
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), prop_str("Ada"));
        properties.insert("rank".to_string(), prop_int(7));
        properties.insert(
            "tags".to_string(),
            PropertyValue {
                value: Some(property_value::Value::ArrayValue(PropertyArray {
                    values: vec![prop_str("a"), prop_str("b")],
                })),
            },
        );
        properties.insert(
            "meta".to_string(),
            PropertyValue {
                value: Some(property_value::Value::ObjectValue(PropertyObject {
                    fields: {
                        let mut m = HashMap::new();
                        m.insert("k".to_string(), prop_int(1));
                        m
                    },
                })),
            },
        );
        let node = Node {
            id: "n1".to_string(),
            labels: vec!["Person".to_string()],
            properties,
            embedding: None,
            created_at_ms: 10,
            updated_at_ms: 20,
        };

        let record = node_to_canonical_record("g1", &node);
        let restored = node_from_canonical_record(&record).expect("node record");
        assert_eq!(restored, node);
    }

    #[test]
    fn canonical_map_and_json_properties_project_without_loss() {
        let document = serde_json::json!({"memory": {"type": "fact"}, "rank": 7});
        let properties = ProximaTree::from([
            (
                "profile".to_string(),
                ProximaTreeNode::Value(ProximaValue::Jsonb(document.clone())),
            ),
            (
                "deleted_value".to_string(),
                ProximaTreeNode::Value(ProximaValue::Jsonb(serde_json::Value::Null)),
            ),
            (
                "attributes".to_string(),
                ProximaTreeNode::Value(ProximaValue::Map(HashMap::from([(
                    "rank".to_string(),
                    ProximaValue::Int64(7),
                )]))),
            ),
        ]);
        let record = CanonicalNode::new("g1", "n1", "Person", properties).into_proxima_record();

        let restored = node_from_canonical_record(&record).expect("canonical node record");

        assert_eq!(
            restored.properties.get("profile"),
            Some(&PropertyValue {
                value: Some(property_value::Value::StringValue(document.to_string())),
            })
        );
        assert_eq!(
            restored.properties.get("attributes"),
            Some(&PropertyValue {
                value: Some(property_value::Value::ObjectValue(PropertyObject {
                    fields: HashMap::from([("rank".to_string(), prop_int(7))]),
                })),
            })
        );
        assert_eq!(
            restored.properties.get("deleted_value"),
            Some(&PropertyValue { value: None }),
            "JSON null must remain the graph property model's null form"
        );
    }

    #[test]
    fn edge_round_trips_through_canonical_record() {
        // TD-168: endpoints are recovered by reversing the canonical node oid;
        // `weight` is not persisted by the projection, so the original must omit
        // it for an identity round-trip.
        let mut properties = HashMap::new();
        properties.insert("since".to_string(), prop_int(2020));
        let edge = Edge {
            id: "e1".to_string(),
            from_node_id: "n1".to_string(),
            to_node_id: "n2".to_string(),
            edge_type: "KNOWS".to_string(),
            properties,
            weight: None,
            created_at_ms: 10,
            updated_at_ms: 20,
        };

        let record = edge_to_canonical_record("g1", &edge);
        let restored = edge_from_canonical_record(&record).expect("edge record");
        assert_eq!(restored, edge);
    }

    #[test]
    fn from_record_rejects_wrong_element_kind() {
        // A node record is not an edge and vice-versa.
        let node_rec = node_to_canonical_record(
            "g1",
            &Node {
                id: "n1".to_string(),
                labels: vec!["Person".to_string()],
                ..Default::default()
            },
        );
        assert!(edge_from_canonical_record(&node_rec).is_none());
        assert!(node_from_canonical_record(&node_rec).is_some());
    }

    #[tokio::test]
    async fn indexes_edges_by_src_and_dst() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let edge = edge_record("g1", "e1", "n1", "n2");

        let result = projection.apply_edge(&edge).await.expect("apply edge");
        assert_eq!(result.entries_written, 2);
        assert_eq!(result.applied_epoch, TopologyEpoch(1));

        let outgoing = projection
            .edges_by_src("graph/g1/node/n1")
            .expect("outgoing entries");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].opposite_node_oid, "graph/g1/node/n2");

        let incoming = projection
            .edges_by_dst("graph/g1/node/n2")
            .expect("incoming entries");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].opposite_node_oid, "graph/g1/node/n1");
    }

    #[tokio::test]
    async fn update_replaces_existing_edge_entries() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let first = edge_record("g1", "e1", "n1", "n2");
        let second = edge_record("g1", "e1", "n1", "n3");

        projection.apply_edge(&first).await.expect("apply first");
        projection.apply_edge(&second).await.expect("apply second");

        assert_eq!(projection.edge_count().expect("edge count"), 1);
        assert!(
            projection
                .edges_by_dst("graph/g1/node/n2")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            projection.edges_by_dst("graph/g1/node/n3").unwrap()[0]
                .key
                .edge_oid,
            "graph/g1/edge/e1"
        );
        assert_eq!(projection.applied_epoch(), TopologyEpoch(2));
    }

    #[test]
    fn apply_edges_batches_under_one_epoch() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let records = vec![
            edge_record("g1", "e1", "n1", "n2"),
            edge_record("g1", "e2", "n1", "n3"),
            edge_record("g1", "e3", "n2", "n3"),
        ];

        // Whole batch applied: 2 adjacency entries per edge (src + dst).
        let written = projection.apply_edges(&records).expect("apply batch");
        assert_eq!(written, 6);

        // The key property (TD-130): ONE epoch transition for the batch, not one
        // per edge — three `apply_edge` calls would land at TopologyEpoch(3).
        assert_eq!(projection.applied_epoch(), TopologyEpoch(1));

        // Topology is identical to the per-edge path.
        assert_eq!(projection.edge_count().expect("edge count"), 3);
        assert_eq!(
            projection
                .edges_by_src("graph/g1/node/n1")
                .expect("n1 outgoing")
                .len(),
            2
        );
        assert_eq!(
            projection
                .edges_by_dst("graph/g1/node/n3")
                .expect("n3 incoming")
                .len(),
            2
        );
    }

    #[test]
    fn apply_edges_upserts_and_rejects_non_edges() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        // Seed an edge, then re-point it via a later batch — upsert must drop the
        // old destination, matching `apply_edge`'s replace semantics.
        projection
            .apply_edges(&[edge_record("g1", "e1", "n1", "n2")])
            .expect("seed");
        projection
            .apply_edges(&[edge_record("g1", "e1", "n1", "n3")])
            .expect("repoint");
        assert_eq!(projection.edge_count().expect("edge count"), 1);
        assert!(
            projection
                .edges_by_dst("graph/g1/node/n2")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            projection.edges_by_dst("graph/g1/node/n3").unwrap()[0]
                .key
                .edge_oid,
            "graph/g1/edge/e1"
        );

        // A non-edge record fails closed before any state is mutated.
        let node = node_to_canonical_record(
            "g1",
            &Node {
                id: "n9".to_string(),
                labels: vec!["Person".to_string()],
                properties: HashMap::new(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            },
        );
        assert!(projection.apply_edges(&[node]).is_err());
    }

    #[tokio::test]
    async fn batch_upsert_relocates_entries_when_endpoints_change() {
        // The batch path must preserve the same upsert semantics as
        // `apply_edge`: re-applying an edge whose endpoints CHANGED removes the
        // stale entries at the old buckets. This is the one case the old
        // remove-first full scan existed for; after the reverse-index change,
        // removal addresses exactly the recorded buckets — and this test fails
        // if the index ever drifts from the maps.
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let first = edge_record("g1", "e1", "n1", "n2");
        projection
            .apply_edges(std::slice::from_ref(&first))
            .expect("apply first batch");

        let moved = edge_record("g1", "e1", "n3", "n4");
        projection
            .apply_edges(std::slice::from_ref(&moved))
            .expect("apply moved batch");

        assert_eq!(projection.edge_count().expect("edge count"), 1);
        assert!(
            projection
                .edges_by_src("graph/g1/node/n1")
                .unwrap()
                .is_empty(),
            "stale src bucket must be dropped"
        );
        assert!(
            projection
                .edges_by_dst("graph/g1/node/n2")
                .unwrap()
                .is_empty(),
            "stale dst bucket must be dropped"
        );
        assert_eq!(
            projection.edges_by_src("graph/g1/node/n3").unwrap()[0]
                .key
                .edge_oid,
            "graph/g1/edge/e1"
        );
        assert_eq!(
            projection.edges_by_dst("graph/g1/node/n4").unwrap()[0]
                .key
                .edge_oid,
            "graph/g1/edge/e1"
        );
    }

    #[tokio::test]
    async fn removes_edge_from_both_directions() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let edge = edge_record("g1", "e1", "n1", "n2");

        projection.apply_edge(&edge).await.expect("apply edge");
        assert!(projection.remove_edge(&edge).await.expect("remove edge"));

        assert!(
            projection
                .edges_by_src("graph/g1/node/n1")
                .unwrap()
                .is_empty()
        );
        assert!(
            projection
                .edges_by_dst("graph/g1/node/n2")
                .unwrap()
                .is_empty()
        );
        assert_eq!(projection.applied_epoch(), TopologyEpoch(2));
    }

    #[tokio::test]
    async fn rebuilds_from_canonical_edge_records() {
        let projection = InMemoryGraphAdjacencyProjection::new("g1");
        let edges = vec![
            edge_record("g1", "e1", "n1", "n2"),
            edge_record("g1", "e2", "n1", "n3"),
        ];

        let rebuilt = projection
            .rebuild_from_records(&[], &edges)
            .await
            .expect("rebuild");

        assert_eq!(rebuilt, 2);
        assert_eq!(
            projection.edges_by_src("graph/g1/node/n1").unwrap().len(),
            2
        );
        assert_eq!(projection.edge_count().expect("edge count"), 2);
        assert!(projection.freshness(TopologyEpoch(1)).is_fresh());
    }
}
