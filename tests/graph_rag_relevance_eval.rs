// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Graph-RAG **relevance eval** (mandate #13 — evals for ranked/generated surfaces).
//!
//! `graph_rag_fusion_e2e.rs` is a *test*: axis-aligned vectors → exact assertions
//! on the deterministic wiring. This is an *eval*: it measures the fusion's
//! **relevance quality** on a controlled-but-realistic code-symbol corpus and
//! gates it on a **rubric threshold** (a ratchet — the number only goes up),
//! covering both:
//!   - **output** — does `/rag` (VectorNodeRetriever → KHopSubgraphBuilder →
//!     RagPipeline, TD-045) return the right symbols? (recall@k / precision vs a
//!     ground-truth relevant set), and
//!   - **trajectory** — was the route sound? (the top vector-search seed lands in
//!     the query's own topic cluster, i.e. the retriever found the right
//!     neighbourhood before graph expansion).
//!
//! Corpus is fully deterministic (controlled cluster embeddings, no ONNX/model) so
//! it is reproducible in CI, yet non-trivial: 4 topic clusters + a distractor
//! cluster + a code-call graph. A regression in the retriever (wrong cluster) or
//! the expander (missed neighbours / leaked distractors) drops the metric below
//! the ratchet and fails the eval.

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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const COLLECTION: &str = "code_symbols_rag_eval";
const DIM: u32 = 8; // dims 0..3 = topic-cluster signal; dims 4..7 = sub-topic signal
const CLUSTERS: [&str; 4] = ["auth", "storage", "network", "parse"];
const PER_CLUSTER: usize = 6;
const SEED_K: usize = 3; // top-k vector seeds
const HOPS: u32 = 2; // graph expansion depth

// --- Rubric thresholds (RATCHET — raise as the corpus/fusion improve, never lower
// without recording the measured regression + rationale here). The corpus + fusion
// are fully deterministic (verified byte-identical across repeated runs), so the
// floors sit a hair below the measured baseline: a regression fails, an improvement
// raises the floor.
//
// Measured baseline 2026-07-17 (develop, real RagPipeline over the corpus below):
//   mean recall = 0.845, mean precision = 0.929, trajectory rate = 1.000
//   (per-query recall 0.833–0.857 — hop-depth misses a far ring symbol; precision
//    0.857–1.000 — 2-hop expansion leaks one cross-cluster node for auth/network).
const RECALL_FLOOR: f32 = 0.84;
const PRECISION_FLOOR: f32 = 0.92;
const TRAJECTORY_FLOOR: f32 = 1.00; // every query must seed its own cluster

fn sym(cluster: &str, i: usize) -> String {
    // Graph NodeId == vector oid (the SDK's one id, record_symbol_id / ADR-044).
    format!("graph/repo/node/{cluster}_{i}")
}

/// Deterministic embedding: a dominant one-hot topic-cluster signal (dims 0..3)
/// plus a weak sub-topic signal (dims 4..7) for intra-cluster variety. Cosine
/// within a cluster ≫ across clusters.
fn topic_embedding(cluster_idx: usize, sym_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[cluster_idx] = 1.0;
    // Distinct sub-topic signature per symbol so the ranking is a deterministic
    // total order (no identical embeddings → no tie-break flakiness in CI). The
    // cluster dim still dominates, so clusters stay well separated.
    v[4 + (sym_idx % 4)] = 0.10 + 0.02 * sym_idx as f32;
    v
}

/// Distractor: sub-topic signal only, no cluster signal → orthogonal to every
/// topic query, and no graph edges. It must NOT be retrieved/expanded; if it is,
/// precision drops (the eval catches retrieval/expansion leakage).
fn distractor_embedding(idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[4 + (idx % 4)] = 1.0;
    v
}

fn query_vector(cluster_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    v[cluster_idx] = 1.0;
    v
}

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

async fn insert_vec(svc: &VectorOperationsService, oid: &str, vector: Vec<f32>) {
    let record = ProximaRecord {
        oid: oid.to_string(),
        record_version: 1,
        embeddings: vec![EmbeddingCell::new_fp32("eval", "dense_vector", DIM, vector)],
        ..Default::default()
    };
    svc.insert_vectors_direct(COLLECTION, Arc::new(vec![record]))
        .await
        .expect("insert eval vector");
}

/// The code-call graph edges (also the ground-truth structure the relevant set is
/// derived from): an intra-cluster ring per topic cluster + a couple of realistic
/// cross-cluster calls (auth→storage, network→parse). Distractors have no edges.
fn graph_edges() -> Vec<(String, String)> {
    let mut edges = Vec::new();
    for c in CLUSTERS {
        for i in 0..PER_CLUSTER {
            edges.push((sym(c, i), sym(c, (i + 1) % PER_CLUSTER)));
        }
    }
    edges.push((sym("auth", 0), sym("storage", 0)));
    edges.push((sym("network", 0), sym("parse", 0)));
    edges
}

/// Ground-truth relevant set for a topic query: the cluster's own symbols plus the
/// 1-hop out-neighbours of each (which pulls in the cross-cluster call targets).
fn relevant_set(cluster: &str, edges: &[(String, String)]) -> HashSet<String> {
    let mut relevant: HashSet<String> = (0..PER_CLUSTER).map(|i| sym(cluster, i)).collect();
    let owned: HashSet<String> = (0..PER_CLUSTER).map(|i| sym(cluster, i)).collect();
    for (from, to) in edges {
        if owned.contains(from) {
            relevant.insert(to.clone());
        }
    }
    relevant
}

async fn build_corpus() -> (
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

    // Topic-cluster symbols.
    for (ci, c) in CLUSTERS.iter().enumerate() {
        for i in 0..PER_CLUSTER {
            insert_vec(&vector_service, &sym(c, i), topic_embedding(ci, i)).await;
        }
    }
    // Distractor cluster (no edges, orthogonal to topic queries).
    for i in 0..PER_CLUSTER {
        insert_vec(&vector_service, &sym("misc", i), distractor_embedding(i)).await;
    }

    let engine = Arc::new(OrionGraphEngine::new());
    // Nodes for topics + distractors (distractors are isolated).
    for c in CLUSTERS.iter().chain(std::iter::once(&"misc")) {
        for i in 0..PER_CLUSTER {
            engine
                .insert_node(node(&sym(c, i)))
                .await
                .expect("insert node");
        }
    }
    for (n, (from, to)) in graph_edges().into_iter().enumerate() {
        engine
            .insert_edge(edge(&format!("e{n}"), &from, &to))
            .await
            .expect("insert edge");
    }
    (env, vector_service, engine)
}

fn pipeline(
    vector_service: Arc<VectorOperationsService>,
    engine: Arc<OrionGraphEngine>,
) -> RagPipeline<VectorNodeRetriever, KHopSubgraphBuilder, IdentityFilter> {
    let retriever = VectorNodeRetriever::new(vector_service, COLLECTION.to_string(), SEED_K);
    let builder = KHopSubgraphBuilder::new(engine as Arc<dyn GraphEngine>, HOPS, None);
    RagPipeline::without_filter(
        retriever,
        builder,
        RagBudget {
            max_seeds: SEED_K,
            max_subgraph_nodes: 64,
        },
    )
}

/// Run the real graph-RAG fusion over the corpus for all four topic queries and
/// gate mean recall@k / precision / trajectory on the rubric thresholds.
#[tokio::test]
async fn graph_rag_relevance_meets_rubric() {
    let (_env, vector_service, engine) = build_corpus().await;
    let rag = pipeline(vector_service, engine);
    let edges = graph_edges();

    let mut recalls = Vec::new();
    let mut precisions = Vec::new();
    let mut trajectory_hits = 0usize;

    for (ci, cluster) in CLUSTERS.iter().enumerate() {
        let relevant = relevant_set(cluster, &edges);
        let query = RagQuery {
            query: format!("symbols related to {cluster}"),
            query_vector: Some(query_vector(ci)),
            allowed_labels: vec![],
        };
        let subgraph = rag.run(&query).await.expect("pipeline run");
        let returned: HashSet<String> = subgraph.nodes.iter().cloned().collect();
        let hits = returned.intersection(&relevant).count();
        let recall = hits as f32 / relevant.len() as f32;
        let precision = hits as f32 / returned.len().max(1) as f32;
        // Trajectory: the top vector seed (first node — KHopSubgraphBuilder pushes
        // seeds before expansion) is in the query's own cluster.
        let prefix = format!("graph/repo/node/{cluster}_");
        let trajectory_ok = subgraph
            .nodes
            .first()
            .is_some_and(|s| s.starts_with(&prefix));
        if trajectory_ok {
            trajectory_hits += 1;
        }
        println!(
            "eval[{cluster}]: recall={recall:.3} precision={precision:.3} \
             trajectory_ok={trajectory_ok} (hits={hits}/{} returned={})",
            relevant.len(),
            returned.len()
        );
        recalls.push(recall);
        precisions.push(precision);
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let mean_recall = mean(&recalls);
    let mean_precision = mean(&precisions);
    let trajectory_rate = trajectory_hits as f32 / CLUSTERS.len() as f32;
    println!(
        "eval SUMMARY: mean_recall={mean_recall:.3} mean_precision={mean_precision:.3} \
         trajectory_rate={trajectory_rate:.3} (floors: recall≥{RECALL_FLOOR} \
         precision≥{PRECISION_FLOOR} trajectory≥{TRAJECTORY_FLOOR})"
    );

    assert!(
        mean_recall >= RECALL_FLOOR,
        "graph-RAG mean recall regressed: {mean_recall:.3} < {RECALL_FLOOR}"
    );
    assert!(
        mean_precision >= PRECISION_FLOOR,
        "graph-RAG mean precision regressed: {mean_precision:.3} < {PRECISION_FLOOR}"
    );
    assert!(
        trajectory_rate >= TRAJECTORY_FLOOR,
        "graph-RAG trajectory regressed: {trajectory_rate:.3} < {TRAJECTORY_FLOOR} \
         (a query seeded outside its own cluster)"
    );
}
