// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! End-to-end graph-RAG **fusion** over the REAL vector-search retriever.
//!
//! `graph_rag_integration_test.rs` exercises the `RagPipeline` with a MOCK
//! retriever (fixed seeds) — it proves composition, not the fusion. This test
//! drives the *real* match-then-traverse path (ADR-061 D4 / TD-045 Modular Graph
//! RAG, the `POST /graphs/{id}/rag` engine) against a live
//! `VectorOperationsService` built by the canonical
//! [`UnifiedTestEnvironment::vector_operations_service`] harness:
//!
//!   real vector search (`VectorNodeRetriever`) → record `oid` seeds
//!     → real ORION k-hop BFS (`KHopSubgraphBuilder`) → fused subgraph
//!
//! Load-bearing invariant (TD-ORION-1): the graph `NodeId` **is** the vector
//! record `oid` (the SDK mints ONE id for both — `record_symbol_id`, ADR-044),
//! so a vector hit correlates to a graph node with no translation layer. The
//! discrimination case (a different query vector reaches a different, isolated
//! node) is something a mock retriever cannot fake.

#[path = "common/integration_test_helpers.rs"]
mod harness;

use harness::UnifiedTestEnvironment;
use proximadb::graph::engines::GraphEngine;
use proximadb::graph::engines::orion::OrionGraphEngine;
use proximadb::graph::rag::engine_impls::{KHopSubgraphBuilder, VectorNodeRetriever};
use proximadb::graph::rag::{IdentityFilter, RagBudget, RagPipeline, RagQuery};
use proximadb::graph::{Edge, Node};
use proximadb::services::vector_operations_service::VectorOperationsService;
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};
use std::collections::HashMap;
use std::sync::Arc;

const COLLECTION: &str = "code_symbols_rag";
const DIM: u32 = 4;

// Symbol `oid`s — the ONE id the SDK uses for both the vector record and the
// graph node (`record_symbol_id`, ADR-044).
const OID_RUST: &str = "graph/repo/node/rust";
const OID_VECTOR: &str = "graph/repo/node/vector";
const OID_PROXIMADB: &str = "graph/repo/node/proximadb";
const OID_PYTHON: &str = "graph/repo/node/python"; // isolated (no edges)

fn node(id: &str) -> Node {
    Node {
        id: id.to_string(),
        labels: vec!["Symbol".to_string()],
        properties: HashMap::new(),
        embedding: None,
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

fn edge(id: &str, from: &str, to: &str) -> Edge {
    Edge {
        id: id.to_string(),
        from_node_id: from.to_string(),
        to_node_id: to.to_string(),
        edge_type: "calls".to_string(),
        properties: HashMap::new(),
        weight: None,
        created_at_ms: 0,
        updated_at_ms: 0,
    }
}

async fn insert_symbol(svc: &VectorOperationsService, oid: &str, vector: Vec<f32>) {
    let record = ProximaRecord {
        oid: oid.to_string(),
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32("test", "dense_vector", DIM, vector)],
        ..Default::default()
    };
    svc.insert_vectors_direct(COLLECTION, Arc::new(vec![record]))
        .await
        .expect("insert symbol vector");
}

/// Real ORION graph `rust -> vector -> proximadb`; `python` isolated. Keyed by
/// the SAME `oid`s the vectors carry.
async fn real_graph() -> Arc<OrionGraphEngine> {
    let engine = Arc::new(OrionGraphEngine::new());
    for n in [OID_RUST, OID_VECTOR, OID_PROXIMADB, OID_PYTHON] {
        engine.insert_node(node(n)).await.expect("insert node");
    }
    engine
        .insert_edge(edge("e1", OID_RUST, OID_VECTOR))
        .await
        .expect("edge e1");
    engine
        .insert_edge(edge("e2", OID_VECTOR, OID_PROXIMADB))
        .await
        .expect("edge e2");
    engine
}

/// Build the real vector service (canonical harness) with the four symbol
/// vectors + the ORION graph. Returns the env (owns the tempdir — keep alive),
/// the vector service, and the graph.
async fn setup() -> (
    UnifiedTestEnvironment,
    Arc<VectorOperationsService>,
    Arc<OrionGraphEngine>,
) {
    let env = UnifiedTestEnvironment::new().await.expect("test env");
    let (vector_service, collections) = env
        .vector_operations_service()
        .await
        .expect("vector service");
    UnifiedTestEnvironment::create_vector_collection(&collections, COLLECTION, DIM)
        .await
        .expect("create collection");

    // Axis-aligned so cosine nearest-neighbour is unambiguous.
    insert_symbol(&vector_service, OID_RUST, vec![1.0, 0.0, 0.0, 0.0]).await;
    insert_symbol(&vector_service, OID_VECTOR, vec![0.0, 1.0, 0.0, 0.0]).await;
    insert_symbol(&vector_service, OID_PROXIMADB, vec![0.0, 0.0, 1.0, 0.0]).await;
    insert_symbol(&vector_service, OID_PYTHON, vec![0.0, 0.0, 0.0, 1.0]).await;

    let graph = real_graph().await;
    (env, vector_service, graph)
}

fn pipeline(
    vector_service: Arc<VectorOperationsService>,
    engine: Arc<OrionGraphEngine>,
    hops: u32,
    max_seeds: usize,
) -> RagPipeline<VectorNodeRetriever, KHopSubgraphBuilder, IdentityFilter> {
    let retriever = VectorNodeRetriever::new(vector_service, COLLECTION.to_string(), max_seeds);
    let builder = KHopSubgraphBuilder::new(engine as Arc<dyn GraphEngine>, hops, None);
    RagPipeline::without_filter(
        retriever,
        builder,
        RagBudget {
            max_seeds,
            max_subgraph_nodes: 32,
        },
    )
}

/// A query near `rust` retrieves `rust` via REAL vector search, then ORION
/// 2-hop BFS fuses in its real neighbours `vector` and `proximadb`.
#[tokio::test]
async fn real_vector_search_seeds_two_hop_graph_expansion() {
    let (_env, vector_service, graph) = setup().await;
    let rag = pipeline(vector_service, graph, 2, 1);

    let query = RagQuery {
        query: "the rust symbol".to_string(),
        query_vector: Some(vec![0.92, 0.08, 0.0, 0.0]), // nearest to rust
        allowed_labels: vec![],
    };
    let subgraph = rag.run(&query).await.expect("pipeline run");

    assert!(
        subgraph.nodes.contains(&OID_RUST.to_string()),
        "real search must seed the rust oid; got {:?}",
        subgraph.nodes
    );
    assert!(
        subgraph.nodes.contains(&OID_VECTOR.to_string())
            && subgraph.nodes.contains(&OID_PROXIMADB.to_string()),
        "2-hop traverse must reach vector + proximadb; got {:?}",
        subgraph.nodes
    );
    assert!(
        !subgraph.nodes.contains(&OID_PYTHON.to_string()),
        "python is unreachable from rust; got {:?}",
        subgraph.nodes
    );
    assert_eq!(subgraph.edges.len(), 2, "rust->vector->proximadb = 2 edges");
}

/// Discrimination a mock can't fake: a query near the ISOLATED `python` symbol
/// retrieves `python` (not `rust`) and expands to nothing.
#[tokio::test]
async fn real_search_discriminates_to_isolated_node() {
    let (_env, vector_service, graph) = setup().await;
    let rag = pipeline(vector_service, graph, 2, 1);

    let query = RagQuery {
        query: "the python symbol".to_string(),
        query_vector: Some(vec![0.05, 0.0, 0.0, 0.95]), // nearest to python
        allowed_labels: vec![],
    };
    let subgraph = rag.run(&query).await.expect("pipeline run");

    assert_eq!(
        subgraph.nodes,
        vec![OID_PYTHON.to_string()],
        "real search must seed python alone; got {:?}",
        subgraph.nodes
    );
    assert!(subgraph.edges.is_empty(), "python is isolated: no edges");
}
