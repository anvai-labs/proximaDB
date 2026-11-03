//! Canonical graph record contracts.
//!
//! Graph nodes and edges are facades over `ProximaRecord`, not a separate
//! durable envelope. This module is the graph modality's low-level contract for
//! mapping property-graph concepts onto the shared record spine described in
//! `docs/12-design/RELATIONAL_DOCUMENT_GRAPH_CONVERGENCE_2026_05_14.adoc`.
//!
//! Durable authority for nodes and edges flows through canonical record
//! storage. Adjacency tables, CSR/COO topology structures, and HNSW graph
//! indexes are rebuildable projections consumed by query/traversal engines.

use async_trait::async_trait;
use proximadb_data_model::ProximaValue;
use proximadb_kernel::error::ProximaDBError;
use proximadb_records::{
    EdgeShape, LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode, RecordKey, RecordStore,
};

/// Stable label on canonical records that originated from the graph node facade.
pub const GRAPH_NODE_LABEL: &str = "graph_node";

/// Stable label on canonical records that originated from the graph edge facade.
pub const GRAPH_EDGE_LABEL: &str = "graph_edge";

/// Reserved property used to retain graph id in canonical records.
pub const GRAPH_ID_PROP: &str = "_graph_id";

/// Reserved property used to retain the user-visible node/edge label.
pub const GRAPH_ELEMENT_LABEL_PROP: &str = "_graph_label";

/// Reserved property for edge source node oid.
pub const GRAPH_EDGE_SRC_PROP: &str = "_graph_edge_src";

/// Reserved property for edge target node oid.
pub const GRAPH_EDGE_DST_PROP: &str = "_graph_edge_dst";

/// Canonical identity for a graph node record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNodeKey {
    /// Graph namespace or dataset id.
    pub graph_id: String,
    /// User-visible node id.
    pub node_id: String,
}

impl GraphNodeKey {
    pub fn new(graph_id: impl Into<String>, node_id: impl Into<String>) -> Self {
        Self {
            graph_id: graph_id.into(),
            node_id: node_id.into(),
        }
    }

    /// Deterministic canonical record oid.
    pub fn canonical_oid(&self) -> String {
        format!("graph/{}/node/{}", self.graph_id, self.node_id)
    }
}

/// Canonical identity for a graph edge record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeKey {
    /// Graph namespace or dataset id.
    pub graph_id: String,
    /// User-visible edge id.
    pub edge_id: String,
}

impl GraphEdgeKey {
    pub fn new(graph_id: impl Into<String>, edge_id: impl Into<String>) -> Self {
        Self {
            graph_id: graph_id.into(),
            edge_id: edge_id.into(),
        }
    }

    /// Deterministic canonical record oid.
    pub fn canonical_oid(&self) -> String {
        format!("graph/{}/edge/{}", self.graph_id, self.edge_id)
    }
}

/// Graph node facade shape before it is written as a canonical record.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalNode {
    pub key: GraphNodeKey,
    /// User-visible node label (type/kind in the property graph model).
    pub node_label: String,
    /// Node properties as NF2 property tree.
    pub properties: ProximaTree,
    /// Owning tenant.
    pub tenant_id: String,
    /// Principals allowed to read this record.
    pub permitted_principals: Vec<String>,
    /// Engine-level RLS policy id.
    pub rls_policy_id: Option<String>,
    /// Optimistic-concurrency version.
    pub version: u64,
    /// Creation time in the canonical record clock (nanoseconds since Unix epoch).
    pub created_at_ns: i64,
    /// Last-update time in the canonical record clock (nanoseconds since Unix epoch).
    pub updated_at_ns: i64,
    /// Optional valid-from time for bitemporal graph facts.
    pub valid_from_ns: Option<i64>,
    /// Optional valid-to time for bitemporal graph facts.
    pub valid_to_ns: Option<i64>,
    /// Source system or connector that produced this node.
    pub origin: Option<String>,
    /// Principal that authored this node.
    pub actor: Option<String>,
    /// Ingestion method, e.g. api/cdc/migration.
    pub method: Option<String>,
}

impl CanonicalNode {
    pub fn new(
        graph_id: impl Into<String>,
        node_id: impl Into<String>,
        node_label: impl Into<String>,
        properties: ProximaTree,
    ) -> Self {
        let record_defaults = ProximaRecord::default();
        Self {
            key: GraphNodeKey::new(graph_id, node_id),
            node_label: node_label.into(),
            properties,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            version: 0,
            created_at_ns: record_defaults.created_at_ns,
            updated_at_ns: record_defaults.updated_at_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
        }
    }

    /// Convert this graph node facade into the durable `ProximaRecord` envelope.
    pub fn into_proxima_record(self) -> ProximaRecord {
        let mut labels = LabelSet::new();
        labels.insert(GRAPH_NODE_LABEL);

        let mut props = self.properties;
        props.insert(
            GRAPH_ID_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.key.graph_id.clone())),
        );
        props.insert(
            GRAPH_ELEMENT_LABEL_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.node_label)),
        );

        ProximaRecord {
            oid: self.key.canonical_oid(),
            local_id: Some(self.key.node_id),
            record_version: self.version,
            tenant_id: self.tenant_id,
            permitted_principals: self.permitted_principals,
            rls_policy_id: self.rls_policy_id,
            created_at_ns: self.created_at_ns,
            updated_at_ns: self.updated_at_ns,
            valid_from_ns: self.valid_from_ns,
            valid_to_ns: self.valid_to_ns,
            origin: self.origin,
            actor: self.actor,
            method: self.method,
            props,
            labels,
            ..ProximaRecord::default()
        }
    }
}

/// Graph edge facade shape before it is written as a canonical record.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalEdge {
    pub key: GraphEdgeKey,
    /// Source node canonical oid.
    pub src_node_oid: String,
    /// Target node canonical oid.
    pub dst_node_oid: String,
    /// User-visible edge label (relationship type).
    pub edge_label: String,
    /// Edge properties as NF2 property tree.
    pub properties: ProximaTree,
    /// Owning tenant.
    pub tenant_id: String,
    /// Principals allowed to read this record.
    pub permitted_principals: Vec<String>,
    /// Engine-level RLS policy id.
    pub rls_policy_id: Option<String>,
    /// Optimistic-concurrency version.
    pub version: u64,
    /// Creation time in the canonical record clock (nanoseconds since Unix epoch).
    pub created_at_ns: i64,
    /// Last-update time in the canonical record clock (nanoseconds since Unix epoch).
    pub updated_at_ns: i64,
    /// Optional valid-from time for bitemporal graph facts.
    pub valid_from_ns: Option<i64>,
    /// Optional valid-to time for bitemporal graph facts.
    pub valid_to_ns: Option<i64>,
    /// Source system or connector that produced this edge.
    pub origin: Option<String>,
    /// Principal that authored this edge.
    pub actor: Option<String>,
    /// Ingestion method, e.g. api/cdc/migration.
    pub method: Option<String>,
}

impl CanonicalEdge {
    pub fn new(
        graph_id: impl Into<String>,
        edge_id: impl Into<String>,
        src_node_oid: impl Into<String>,
        dst_node_oid: impl Into<String>,
        edge_label: impl Into<String>,
        properties: ProximaTree,
    ) -> Self {
        let record_defaults = ProximaRecord::default();
        Self {
            key: GraphEdgeKey::new(graph_id, edge_id),
            src_node_oid: src_node_oid.into(),
            dst_node_oid: dst_node_oid.into(),
            edge_label: edge_label.into(),
            properties,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            version: 0,
            created_at_ns: record_defaults.created_at_ns,
            updated_at_ns: record_defaults.updated_at_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
        }
    }

    /// Convert this graph edge facade into the durable `ProximaRecord` envelope.
    ///
    /// Edge endpoints are encoded as reference fields so topology projections
    /// (adjacency tables, CSR) can be built as rebuildable indexes over the
    /// canonical record set without needing a separate edge-specific WAL.
    pub fn into_proxima_record(self) -> ProximaRecord {
        let mut labels = LabelSet::new();
        labels.insert(GRAPH_EDGE_LABEL);

        let mut props = self.properties;
        props.insert(
            GRAPH_ID_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.key.graph_id.clone())),
        );
        props.insert(
            GRAPH_ELEMENT_LABEL_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.edge_label.clone())),
        );
        props.insert(
            GRAPH_EDGE_SRC_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.src_node_oid.clone())),
        );
        props.insert(
            GRAPH_EDGE_DST_PROP.to_string(),
            ProximaTreeNode::Value(ProximaValue::String(self.dst_node_oid.clone())),
        );

        let edge_shape = EdgeShape {
            source_id: self.src_node_oid,
            target_id: self.dst_node_oid,
            edge_type: self.edge_label.clone(),
            weight: None,
        };

        ProximaRecord {
            oid: self.key.canonical_oid(),
            local_id: Some(self.key.edge_id),
            record_version: self.version,
            tenant_id: self.tenant_id,
            permitted_principals: self.permitted_principals,
            rls_policy_id: self.rls_policy_id,
            created_at_ns: self.created_at_ns,
            updated_at_ns: self.updated_at_ns,
            valid_from_ns: self.valid_from_ns,
            valid_to_ns: self.valid_to_ns,
            origin: self.origin,
            actor: self.actor,
            method: self.method,
            edge: Some(edge_shape),
            props,
            labels,
            ..ProximaRecord::default()
        }
    }
}

/// Rebuild a graph node from a canonical record.
pub fn canonical_node_from_record(record: &ProximaRecord) -> Option<CanonicalNode> {
    if !record.labels.contains(GRAPH_NODE_LABEL) {
        return None;
    }

    let graph_id = match record.props.get(GRAPH_ID_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(graph_id))) => graph_id.clone(),
        _ => return None,
    };

    let node_label = match record.props.get(GRAPH_ELEMENT_LABEL_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(label))) => label.clone(),
        _ => String::new(),
    };

    let node_id = record
        .local_id
        .clone()
        .unwrap_or_else(|| record.oid.clone());

    let mut properties = record.props.clone();
    properties.remove(GRAPH_ID_PROP);
    properties.remove(GRAPH_ELEMENT_LABEL_PROP);

    Some(CanonicalNode {
        key: GraphNodeKey::new(graph_id, node_id),
        node_label,
        properties,
        tenant_id: record.tenant_id.clone(),
        permitted_principals: record.permitted_principals.clone(),
        rls_policy_id: record.rls_policy_id.clone(),
        version: record.record_version,
        created_at_ns: record.created_at_ns,
        updated_at_ns: record.updated_at_ns,
        valid_from_ns: record.valid_from_ns,
        valid_to_ns: record.valid_to_ns,
        origin: record.origin.clone(),
        actor: record.actor.clone(),
        method: record.method.clone(),
    })
}

/// Rebuild a graph edge from a canonical record.
pub fn canonical_edge_from_record(record: &ProximaRecord) -> Option<CanonicalEdge> {
    if !record.labels.contains(GRAPH_EDGE_LABEL) {
        return None;
    }

    let graph_id = match record.props.get(GRAPH_ID_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(graph_id))) => graph_id.clone(),
        _ => return None,
    };

    let edge_label = match record.props.get(GRAPH_ELEMENT_LABEL_PROP) {
        Some(ProximaTreeNode::Value(ProximaValue::String(label))) => label.clone(),
        _ => String::new(),
    };

    let edge_shape = record.edge.as_ref()?;
    let src_node_oid = edge_shape.source_id.clone();
    let dst_node_oid = edge_shape.target_id.clone();

    let edge_id = record
        .local_id
        .clone()
        .unwrap_or_else(|| record.oid.clone());

    let mut properties = record.props.clone();
    properties.remove(GRAPH_ID_PROP);
    properties.remove(GRAPH_ELEMENT_LABEL_PROP);
    properties.remove(GRAPH_EDGE_SRC_PROP);
    properties.remove(GRAPH_EDGE_DST_PROP);

    Some(CanonicalEdge {
        key: GraphEdgeKey::new(graph_id, edge_id),
        src_node_oid,
        dst_node_oid,
        edge_label,
        properties,
        tenant_id: record.tenant_id.clone(),
        permitted_principals: record.permitted_principals.clone(),
        rls_policy_id: record.rls_policy_id.clone(),
        version: record.record_version,
        created_at_ns: record.created_at_ns,
        updated_at_ns: record.updated_at_ns,
        valid_from_ns: record.valid_from_ns,
        valid_to_ns: record.valid_to_ns,
        origin: record.origin.clone(),
        actor: record.actor.clone(),
        method: record.method.clone(),
    })
}

/// Canonical graph node storage contract.
///
/// Implementations write/read `ProximaRecord` as the durable truth.
/// Adjacency tables, CSR, and HNSW graph indexes are projection consumers.
#[async_trait]
pub trait CanonicalNodeStore: Send + Sync {
    async fn upsert_node(&self, node: CanonicalNode) -> Result<ProximaRecord, ProximaDBError>;

    async fn get_node(&self, key: &GraphNodeKey) -> Result<Option<ProximaRecord>, ProximaDBError>;

    async fn delete_node(&self, key: &GraphNodeKey) -> Result<bool, ProximaDBError>;
}

/// Canonical graph edge storage contract.
#[async_trait]
pub trait CanonicalEdgeStore: Send + Sync {
    async fn upsert_edge(&self, edge: CanonicalEdge) -> Result<ProximaRecord, ProximaDBError>;

    async fn get_edge(&self, key: &GraphEdgeKey) -> Result<Option<ProximaRecord>, ProximaDBError>;

    async fn delete_edge(&self, key: &GraphEdgeKey) -> Result<bool, ProximaDBError>;
}

#[async_trait]
impl<T> CanonicalNodeStore for T
where
    T: RecordStore + Send + Sync,
{
    async fn upsert_node(&self, node: CanonicalNode) -> Result<ProximaRecord, ProximaDBError> {
        self.upsert_record(node.into_proxima_record())
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn get_node(&self, key: &GraphNodeKey) -> Result<Option<ProximaRecord>, ProximaDBError> {
        self.get_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn delete_node(&self, key: &GraphNodeKey) -> Result<bool, ProximaDBError> {
        self.delete_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }
}

#[async_trait]
impl<T> CanonicalEdgeStore for T
where
    T: RecordStore + Send + Sync,
{
    async fn upsert_edge(&self, edge: CanonicalEdge) -> Result<ProximaRecord, ProximaDBError> {
        self.upsert_record(edge.into_proxima_record())
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn get_edge(&self, key: &GraphEdgeKey) -> Result<Option<ProximaRecord>, ProximaDBError> {
        self.get_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }

    async fn delete_edge(&self, key: &GraphEdgeKey) -> Result<bool, ProximaDBError> {
        self.delete_record(&RecordKey::new(key.canonical_oid()))
            .await
            .map_err(|error| ProximaDBError::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::test_support::MemoryRecordStore;

    #[test]
    fn node_key_builds_stable_canonical_oid() {
        let key = GraphNodeKey::new("social", "user-1");
        assert_eq!(key.canonical_oid(), "graph/social/node/user-1");
    }

    #[test]
    fn edge_key_builds_stable_canonical_oid() {
        let key = GraphEdgeKey::new("social", "edge-42");
        assert_eq!(key.canonical_oid(), "graph/social/edge/edge-42");
    }

    #[test]
    fn canonical_node_maps_to_proxima_record() {
        let mut props = ProximaTree::new();
        props.insert(
            "name".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("Alice".to_string())),
        );
        let node = CanonicalNode {
            tenant_id: "t1".to_string(),
            version: 3,
            ..CanonicalNode::new("social", "user-1", "Person", props)
        };

        let record = node.into_proxima_record();

        assert_eq!(record.oid, "graph/social/node/user-1");
        assert_eq!(record.local_id.as_deref(), Some("user-1"));
        assert_eq!(record.record_version, 3);
        assert_eq!(record.tenant_id, "t1");
        assert!(record.labels.contains(GRAPH_NODE_LABEL));
        assert!(record.props.contains_key("name"));
        assert_eq!(
            record.props.get(GRAPH_ELEMENT_LABEL_PROP),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "Person".to_string()
            )))
        );
    }

    #[test]
    fn canonical_edge_encodes_endpoints_in_record() {
        let node_key = GraphNodeKey::new("social", "user-1");
        let src_oid = node_key.canonical_oid();
        let dst_oid = GraphNodeKey::new("social", "user-2").canonical_oid();

        let edge = CanonicalEdge::new(
            "social",
            "edge-1",
            src_oid.clone(),
            dst_oid.clone(),
            "KNOWS",
            ProximaTree::new(),
        );

        let record = edge.into_proxima_record();

        assert_eq!(record.oid, "graph/social/edge/edge-1");
        assert!(record.labels.contains(GRAPH_EDGE_LABEL));
        assert_eq!(
            record.props.get(GRAPH_EDGE_SRC_PROP),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                src_oid.clone()
            )))
        );
        assert_eq!(
            record.props.get(GRAPH_EDGE_DST_PROP),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                dst_oid.clone()
            )))
        );
        let edge_shape = record.edge.as_ref().expect("edge shape");
        assert_eq!(edge_shape.source_id, src_oid);
        assert_eq!(edge_shape.target_id, dst_oid);
        assert_eq!(edge_shape.edge_type, "KNOWS");
    }

    #[test]
    fn canonical_node_round_trips_through_record() {
        let mut props = ProximaTree::new();
        props.insert(
            "age".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(30)),
        );
        let node = CanonicalNode::new("g1", "n1", "User", props);
        let record = node.into_proxima_record();
        let rebuilt = canonical_node_from_record(&record).expect("node record");

        assert_eq!(rebuilt.key.graph_id, "g1");
        assert_eq!(rebuilt.key.node_id, "n1");
        assert_eq!(rebuilt.node_label, "User");
        assert!(rebuilt.properties.contains_key("age"));
        assert!(!rebuilt.properties.contains_key(GRAPH_ID_PROP));
    }

    #[test]
    fn canonical_edge_round_trips_through_record() {
        let edge = CanonicalEdge::new(
            "g1",
            "e1",
            "graph/g1/node/n1",
            "graph/g1/node/n2",
            "FOLLOWS",
            ProximaTree::new(),
        );
        let record = edge.into_proxima_record();
        let rebuilt = canonical_edge_from_record(&record).expect("edge record");

        assert_eq!(rebuilt.key.graph_id, "g1");
        assert_eq!(rebuilt.edge_label, "FOLLOWS");
        assert_eq!(rebuilt.src_node_oid, "graph/g1/node/n1");
        assert_eq!(rebuilt.dst_node_oid, "graph/g1/node/n2");
    }

    #[test]
    fn non_node_record_is_not_rebuilt_as_node() {
        assert!(canonical_node_from_record(&ProximaRecord::default()).is_none());
    }

    #[test]
    fn non_edge_record_is_not_rebuilt_as_edge() {
        assert!(canonical_edge_from_record(&ProximaRecord::default()).is_none());
    }

    #[tokio::test]
    async fn node_store_adapts_to_record_store() {
        let store = MemoryRecordStore::default();
        let node = CanonicalNode::new("g1", "n1", "Person", ProximaTree::new());
        let key = node.key.clone();

        let written = store.upsert_node(node).await.expect("upsert node");
        assert_eq!(written.oid, "graph/g1/node/n1");

        let fetched = store
            .get_node(&key)
            .await
            .expect("get node")
            .expect("node exists");
        assert!(canonical_node_from_record(&fetched).is_some());

        assert!(store.delete_node(&key).await.expect("delete node"));
    }

    #[tokio::test]
    async fn edge_store_adapts_to_record_store() {
        let store = MemoryRecordStore::default();
        let edge = CanonicalEdge::new(
            "g1",
            "e1",
            "graph/g1/node/n1",
            "graph/g1/node/n2",
            "KNOWS",
            ProximaTree::new(),
        );
        let key = edge.key.clone();

        let written = store.upsert_edge(edge).await.expect("upsert edge");
        assert_eq!(written.oid, "graph/g1/edge/e1");

        let fetched = store
            .get_edge(&key)
            .await
            .expect("get edge")
            .expect("edge exists");
        assert!(canonical_edge_from_record(&fetched).is_some());

        assert!(store.delete_edge(&key).await.expect("delete edge"));
    }
}
