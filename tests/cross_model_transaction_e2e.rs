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

//! End-to-end tests for the cross-model transaction coordinator (TD-133).
//!
//! Verifies atomic multi-modal writes across the graph engine (nodes/edges) and
//! record storage (embeddings), the flag-gated disabled path, rollback
//! compensation on failure, and idempotent/concurrent behavior.
//!
//! These tests use the real ORION graph engine plus an in-memory record-store
//! mock, and a controllable (edge-failing) graph-engine wrapper to drive the
//! phase-3 rollback path so the coordinator must compensate an already-committed
//! node + embedding.

use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::Utc;
use proximadb::catalog::CatalogTableSchema;
use proximadb::graph::engines::GraphEngine;
use proximadb::graph::engines::orion::OrionGraphEngine;
use proximadb::graph::model::{Edge, Node, PropertyValue, property_value::Value};
use proximadb::services::record_store::{
    TableRecordGetRequest, TableRecordGetResponse, TableRecordMutation, TableRecordMutationKind,
    TableRecordStore, TableRecordWriteResult,
};
use proximadb::services::transaction::{CrossModelTransactionCoordinator, TransactionOutcome};
use proximadb::storage::tenant::context::StorageTenantContext;
use proximadb_kernel::error::ProximaDBError;
use proximadb_records::ProximaRecord;

/// Environment variable flag gating cross-model transaction support.
const FLAG: &str = "PROXIMADB_CROSS_MODEL_TX_ENABLED";

fn enable_flag() {
    unsafe { env::set_var(FLAG, "true") };
}

fn clear_flag() {
    unsafe { env::remove_var(FLAG) };
}

/// Build a node with a couple of scalar properties.
fn create_test_node(id: &str, label: &str) -> Node {
    let mut properties = HashMap::new();
    properties.insert(
        "name".to_string(),
        PropertyValue {
            value: Some(Value::StringValue(format!("Node {}", id))),
        },
    );
    properties.insert(
        "created_at".to_string(),
        PropertyValue {
            value: Some(Value::StringValue(Utc::now().to_rfc3339())),
        },
    );

    Node {
        id: id.to_string(),
        labels: vec![label.to_string()],
        properties,
        embedding: None,
        created_at_ms: Utc::now().timestamp_millis(),
        updated_at_ms: Utc::now().timestamp_millis(),
    }
}

/// Build a directed edge between two node IDs.
fn create_test_edge(id: &str, from: &str, to: &str, edge_type: &str) -> Edge {
    Edge {
        id: id.to_string(),
        from_node_id: from.to_string(),
        to_node_id: to.to_string(),
        edge_type: edge_type.to_string(),
        properties: HashMap::new(),
        weight: None,
        created_at_ms: Utc::now().timestamp_millis(),
        updated_at_ms: Utc::now().timestamp_millis(),
    }
}

fn create_test_tenant() -> StorageTenantContext {
    StorageTenantContext::for_tenant_id("test_tenant_e2e")
}

/// In-memory record store that tracks written records by OID. Used both as the
/// embedding persistence sink and to assert what was committed vs. compensated.
struct MockTableRecordStore {
    records: Mutex<HashMap<String, ProximaRecord>>,
}

impl MockTableRecordStore {
    fn new() -> Self {
        Self {
            records: Mutex::new(HashMap::new()),
        }
    }

    fn contains_oid(&self, oid: &str) -> bool {
        self.records.lock().unwrap().contains_key(oid)
    }
}

#[async_trait::async_trait]
impl TableRecordStore for MockTableRecordStore {
    async fn write_mutations(
        &self,
        _table_schema: &CatalogTableSchema,
        mutations: Vec<TableRecordMutation>,
        _tenant_context: Option<&StorageTenantContext>,
    ) -> Result<TableRecordWriteResult> {
        let mut map = self.records.lock().unwrap();
        let mut record_ids = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            let oid = if mutation.record.oid.is_empty() {
                mutation.record.local_id.clone().unwrap_or_default()
            } else {
                mutation.record.oid.clone()
            };
            match mutation.kind {
                TableRecordMutationKind::Delete => {
                    map.remove(&oid);
                }
                _ => {
                    map.insert(oid.clone(), mutation.record);
                }
            }
            record_ids.push(oid);
        }
        Ok(TableRecordWriteResult {
            max_lsn: None,
            success: true,
            record_ids,
            metrics: Default::default(),
            errors: Vec::new(),
            error_code: None,
        })
    }

    async fn get_by_key(
        &self,
        _table_schema: &CatalogTableSchema,
        _request: TableRecordGetRequest,
        _tenant_context: Option<&StorageTenantContext>,
        #[cfg(feature = "abac-policy")] _read_context: &proximadb_abac::ReadContext,
    ) -> Result<TableRecordGetResponse> {
        Ok(None)
    }
}

/// Graph engine wrapper that delegates every operation to a real ORION engine,
/// but can be armed to fail `insert_edge`. This lets a test commit the node
/// (phase 1) and embedding (phase 2), then force a phase-3 (edge) failure so the
/// coordinator must roll back both already-committed writes.
struct FailingGraphEngine {
    inner: Arc<OrionGraphEngine>,
    fail_edges: AtomicBool,
}

impl FailingGraphEngine {
    fn new() -> Self {
        Self {
            inner: Arc::new(OrionGraphEngine::new()),
            fail_edges: AtomicBool::new(false),
        }
    }

    fn arm_edge_failure(&self) {
        self.fail_edges.store(true, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl GraphEngine for FailingGraphEngine {
    async fn insert_node(&self, node: Node) -> std::result::Result<Arc<Node>, ProximaDBError> {
        self.inner.insert_node(node).await
    }

    fn get_node(&self, id: &String) -> std::result::Result<Option<Arc<Node>>, ProximaDBError> {
        self.inner.get_node(id)
    }

    async fn update_node(&self, node: Node) -> std::result::Result<Arc<Node>, ProximaDBError> {
        self.inner.update_node(node).await
    }

    async fn delete_node(
        &self,
        id: &String,
    ) -> std::result::Result<Option<Arc<Node>>, ProximaDBError> {
        self.inner.delete_node(id).await
    }

    async fn insert_edge(&self, edge: Edge) -> std::result::Result<Arc<Edge>, ProximaDBError> {
        if self.fail_edges.load(Ordering::SeqCst) {
            return Err(ProximaDBError::Internal(
                "simulated edge write failure".to_string(),
            ));
        }
        self.inner.insert_edge(edge).await
    }

    fn get_edge(&self, id: &String) -> std::result::Result<Option<Arc<Edge>>, ProximaDBError> {
        self.inner.get_edge(id)
    }

    async fn update_edge(&self, edge: Edge) -> std::result::Result<Arc<Edge>, ProximaDBError> {
        self.inner.update_edge(edge).await
    }

    async fn delete_edge(
        &self,
        id: &String,
    ) -> std::result::Result<Option<Arc<Edge>>, ProximaDBError> {
        self.inner.delete_edge(id).await
    }

    fn get_outgoing_edges(
        &self,
        node_id: &String,
        edge_type: Option<&str>,
    ) -> std::result::Result<Vec<Arc<Edge>>, ProximaDBError> {
        self.inner.get_outgoing_edges(node_id, edge_type)
    }

    fn get_incoming_edges(
        &self,
        node_id: &String,
        edge_type: Option<&str>,
    ) -> std::result::Result<Vec<Arc<Edge>>, ProximaDBError> {
        self.inner.get_incoming_edges(node_id, edge_type)
    }

    fn get_neighbors(
        &self,
        node_id: &String,
        edge_type: Option<&str>,
    ) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
        self.inner.get_neighbors(node_id, edge_type)
    }

    fn get_nodes_by_label(
        &self,
        label: &str,
    ) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
        self.inner.get_nodes_by_label(label)
    }

    fn node_count(&self) -> std::result::Result<usize, ProximaDBError> {
        self.inner.node_count()
    }

    fn edge_count(&self) -> std::result::Result<usize, ProximaDBError> {
        self.inner.edge_count()
    }

    fn get_all_nodes(&self) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
        self.inner.get_all_nodes()
    }
}

/// Build a coordinator over a real ORION engine, returning the record-store
/// handle so tests can assert persistence/compensation.
fn real_coordinator_with_store() -> (CrossModelTransactionCoordinator, Arc<MockTableRecordStore>) {
    let engine = Arc::new(OrionGraphEngine::new());
    let record_store = Arc::new(MockTableRecordStore::new());
    let coordinator = CrossModelTransactionCoordinator::new(
        engine,
        record_store.clone(),
        "test_embeddings".to_string(),
    );
    (coordinator, record_store)
}

// ---------------------------------------------------------------------------
// Flag gate
// ---------------------------------------------------------------------------

/// Coordinator is disabled by default (flag unset).
#[test]
fn test_coordinator_disabled_by_default() {
    clear_flag();
    let (coordinator, _store) = real_coordinator_with_store();
    assert!(!coordinator.is_enabled());
}

/// Coordinator enables when the flag is set.
#[test]
fn test_coordinator_enabled_with_flag() {
    enable_flag();
    let (coordinator, _store) = real_coordinator_with_store();
    assert!(coordinator.is_enabled());
    clear_flag();
}

/// With the flag off, writes short-circuit to `Disabled` (caller falls back).
#[tokio::test]
async fn test_returns_disabled_when_flag_off() {
    clear_flag();
    let (coordinator, _store) = real_coordinator_with_store();

    let node = create_test_node("n1", "Person");
    let result = coordinator
        .write_symbol_atomically(node, vec![0.1, 0.2, 0.3], vec![], &create_test_tenant())
        .await
        .unwrap();

    assert_eq!(result, TransactionOutcome::Disabled);
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

/// A full node + embedding + edges transaction commits atomically and all three
/// domains are durably observable through their canonical read paths.
#[tokio::test]
async fn test_successful_atomic_commit_persists_all() {
    enable_flag();
    let (coordinator, record_store) = real_coordinator_with_store();

    let node = create_test_node("symbol1", "Symbol");
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let edges = vec![
        create_test_edge("edge1", "symbol1", "other1", "CONNECTS_TO"),
        create_test_edge("edge2", "symbol1", "other2", "REFERENCES"),
    ];

    // Pre-insert the edge endpoints so ORION's referential-integrity check
    // (both endpoints must exist) passes when phase 3 inserts the edges.
    coordinator
        .graph_engine()
        .insert_node(create_test_node("other1", "Symbol"))
        .await
        .unwrap();
    coordinator
        .graph_engine()
        .insert_node(create_test_node("other2", "Symbol"))
        .await
        .unwrap();

    let result = coordinator
        .write_symbol_atomically(
            node.clone(),
            embedding.clone(),
            edges.clone(),
            &create_test_tenant(),
        )
        .await
        .unwrap();

    match result {
        TransactionOutcome::Committed { node_oid } => assert_eq!(node_oid, "symbol1"),
        other => panic!("expected Committed, got {other:?}"),
    }

    let graph = coordinator.graph_engine();

    // Node persisted.
    let stored_node = graph.get_node(&node.id).unwrap().unwrap();
    assert_eq!(stored_node.id, "symbol1");
    assert!(stored_node.labels.iter().any(|l| l == "Symbol"));

    // Edges persisted.
    for edge in &edges {
        let stored = graph.get_edge(&edge.id).unwrap().unwrap();
        assert_eq!(stored.id, edge.id);
        assert_eq!(stored.from_node_id, "symbol1");
    }

    // Embedding persisted to record storage under the canonical OID.
    let embed_oid = format!("embed_{}", node.id);
    assert!(
        record_store.contains_oid(&embed_oid),
        "embedding record must be persisted"
    );

    clear_flag();
}

/// Empty edge list is a valid transaction (node + embedding only).
#[tokio::test]
async fn test_handles_empty_edges() {
    enable_flag();
    let (coordinator, record_store) = real_coordinator_with_store();

    let node = create_test_node("symbol_no_edges", "Symbol");
    let result = coordinator
        .write_symbol_atomically(
            node.clone(),
            vec![0.1, 0.2, 0.3],
            vec![],
            &create_test_tenant(),
        )
        .await
        .unwrap();

    match result {
        TransactionOutcome::Committed { node_oid } => assert_eq!(node_oid, "symbol_no_edges"),
        other => panic!("expected Committed, got {other:?}"),
    }

    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_some()
    );
    assert!(record_store.contains_oid(&format!("embed_{}", node.id)));

    clear_flag();
}

/// A 1536-dim (OpenAI-sized) embedding commits correctly.
#[tokio::test]
async fn test_large_embedding_vector() {
    enable_flag();
    let (coordinator, _store) = real_coordinator_with_store();

    let node = create_test_node("symbol_large", "Symbol");
    let large_embedding: Vec<f32> = (0..1536).map(|i| i as f32 / 1536.0).collect();

    let result = coordinator
        .write_symbol_atomically(node.clone(), large_embedding, vec![], &create_test_tenant())
        .await
        .unwrap();

    match result {
        TransactionOutcome::Committed { node_oid } => assert_eq!(node_oid, "symbol_large"),
        other => panic!("expected Committed, got {other:?}"),
    }

    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_some()
    );

    clear_flag();
}

// ---------------------------------------------------------------------------
// Rollback / compensation
// ---------------------------------------------------------------------------

/// A non-finite (NaN) embedding fails phase-2 validation, so the already-written
/// node is rolled back and no embedding is persisted.
#[tokio::test]
async fn test_rollback_on_embedding_validation_failure() {
    enable_flag();
    let (coordinator, record_store) = real_coordinator_with_store();

    let node = create_test_node("symbol_nan", "Symbol");
    let invalid_embedding = vec![0.1, f32::NAN, 0.3];

    let result = coordinator
        .write_symbol_atomically(
            node.clone(),
            invalid_embedding,
            vec![],
            &create_test_tenant(),
        )
        .await
        .unwrap();

    match result {
        TransactionOutcome::RolledBack { reason } => {
            assert!(
                reason.contains("Embedding write failed"),
                "unexpected reason: {reason}"
            );
        }
        other => panic!("expected RolledBack, got {other:?}"),
    }

    // Node compensated.
    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_none(),
        "node must be rolled back"
    );
    // Embedding never written.
    assert!(
        !record_store.contains_oid(&format!("embed_{}", node.id)),
        "embedding must not be persisted on validation failure"
    );

    clear_flag();
}

/// Inf values are likewise rejected with the same compensation.
#[tokio::test]
async fn test_embedding_validation_rejects_inf() {
    enable_flag();
    let (coordinator, _store) = real_coordinator_with_store();

    let node = create_test_node("symbol_inf", "Symbol");
    let inf_embedding = vec![0.1, f32::INFINITY, 0.3];

    let result = coordinator
        .write_symbol_atomically(node.clone(), inf_embedding, vec![], &create_test_tenant())
        .await
        .unwrap();

    match result {
        TransactionOutcome::RolledBack { reason } => {
            assert!(reason.contains("non-finite"), "unexpected reason: {reason}")
        }
        other => panic!("expected RolledBack, got {other:?}"),
    }

    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_none(),
        "node must be rolled back"
    );

    clear_flag();
}

/// A phase-3 (edge) failure after a committed node + embedding triggers full
/// cross-model compensation: both the node and the embedding are removed.
#[tokio::test]
async fn test_rollback_on_edge_failure_compensates_node_and_embedding() {
    enable_flag();

    let engine = Arc::new(FailingGraphEngine::new());
    let record_store = Arc::new(MockTableRecordStore::new());
    let coordinator = CrossModelTransactionCoordinator::new(
        engine.clone(),
        record_store.clone(),
        "test_embeddings".to_string(),
    );

    // Node + embedding will commit (phases 1-2); the edge write (phase 3) fails.
    engine.arm_edge_failure();

    let node = create_test_node("symbol_edge_fail", "Symbol");
    let embedding = vec![0.1, 0.2, 0.3];
    let edges = vec![create_test_edge(
        "edge_fail_1",
        "symbol_edge_fail",
        "other1",
        "LINK",
    )];

    let result = coordinator
        .write_symbol_atomically(node.clone(), embedding, edges, &create_test_tenant())
        .await
        .unwrap();

    match result {
        TransactionOutcome::RolledBack { reason } => {
            assert!(
                reason.contains("Edge write failed"),
                "unexpected reason: {reason}"
            );
        }
        other => panic!("expected RolledBack, got {other:?}"),
    }

    // Node compensated.
    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_none(),
        "node must be compensated on edge failure"
    );
    // Embedding compensated (delete mutation applied during rollback).
    assert!(
        !record_store.contains_oid(&format!("embed_{}", node.id)),
        "embedding must be compensated on edge failure"
    );

    clear_flag();
}

// ---------------------------------------------------------------------------
// Idempotency / concurrency
// ---------------------------------------------------------------------------

/// Re-running the same transaction is safe (upsert semantics). Node + embedding
/// writes are idempotent (insert_node upserts by id; the embedding record upserts
/// by OID), so a client retry after a transient failure does not corrupt state.
#[tokio::test]
async fn test_idempotent_retry_upsert() {
    enable_flag();
    let (coordinator, record_store) = real_coordinator_with_store();

    let node = create_test_node("symbol_upsert", "Symbol");
    let embedding = vec![0.1, 0.2, 0.3];

    let result1 = coordinator
        .write_symbol_atomically(
            node.clone(),
            embedding.clone(),
            vec![],
            &create_test_tenant(),
        )
        .await
        .unwrap();
    assert!(matches!(result1, TransactionOutcome::Committed { .. }));

    // Retry the exact same logical write.
    let result2 = coordinator
        .write_symbol_atomically(node.clone(), embedding, vec![], &create_test_tenant())
        .await
        .unwrap();
    assert!(matches!(result2, TransactionOutcome::Committed { .. }));

    assert!(
        coordinator
            .graph_engine()
            .get_node(&node.id)
            .unwrap()
            .is_some()
    );
    // Embedding upserted (present, exactly one record under the OID).
    assert!(record_store.contains_oid(&format!("embed_{}", node.id)));

    clear_flag();
}

/// Concurrent transactions on distinct symbols do not interfere.
#[tokio::test]
async fn test_concurrent_transactions() {
    enable_flag();
    let coordinator = Arc::new(real_coordinator_with_store().0);

    let mut handles = Vec::with_capacity(10);
    for i in 0..10 {
        let coordinator = Arc::clone(&coordinator);
        handles.push(tokio::spawn(async move {
            let node = create_test_node(&format!("symbol_concurrent_{i}"), "Symbol");
            coordinator
                .write_symbol_atomically(
                    node.clone(),
                    vec![0.1; 128],
                    vec![],
                    &create_test_tenant(),
                )
                .await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap().unwrap())
        .collect();

    assert_eq!(results.len(), 10);
    for result in results {
        assert!(matches!(result, TransactionOutcome::Committed { .. }));
    }

    let graph = coordinator.graph_engine();
    for i in 0..10 {
        let node_id = format!("symbol_concurrent_{i}");
        assert!(graph.get_node(&node_id).unwrap().is_some());
    }

    clear_flag();
}
