//! Code-graph embedded-parity gate (ADR-0017 §F, gate #3).
//!
//! Indexes a small synthesized code-graph fixture via the embedded `EmbeddedProximaDB` (one graph +
//! one co-indexed vector collection) and asserts `impact_analysis(forward/backward)` and the hybrid
//! seed→expand (`fusion_search`) return deterministic, correct results.
//!
//! ## Parity rationale
//! The embedded `fusion_search`/`impact_analysis` (TD-131) delegate to the SAME `FusionService` and
//! `GraphOperationsService::impact_analysis` that the REST `POST /api/v2/graphs/{id}/fusion-search`
//! and `…/impact-analysis` endpoints use (PR #308). The REST handler and `EmbeddedProximaDB` both
//! construct `FusionService::new(vector, graph)` identically, so the embedded result IS the server
//! result by construction. PR #308's `graph_impact_analysis_integration_test` + `fusion_service`
//! unit tests verify the server-side core; this test verifies the embedded path delegates to that
//! same core correctly and is deterministic — together they clear the "parity-tested embedded path"
//! gate. (A cross-process embedded-vs-networked-server comparison is deferred: the data-loading
//! asymmetry makes it flaky-prone, and structural parity + dual correctness is the robust proof.)
//!
//! ## Test isolation
//! `GraphCollectionService` persists graph metadata to a process-shared path, so multiple
//! `EmbeddedProximaDB` instances in one test process share graph state. Each test therefore builds
//! its DB under a UNIQUE graph id (threaded through `build_db`/`fixture`/`canonical_oid`) so the
//! three `#[test]`s — which share one process under `--test-threads=1` — never collide on a graph
//! that "already exists".

use std::collections::HashSet;

use proximadb_embedded::{EmbeddedConfig, EmbeddedGraphEdge, EmbeddedGraphNode, EmbeddedProximaDB};

const VEC_COLLECTION: &str = "code_vecs";

fn canonical_oid(graph_id: &str, node_id: &str) -> String {
    format!("graph/{graph_id}/node/{node_id}")
}

/// main --CALLS--> parse --CALLS--> validate
/// main --CALLS--> io
/// parse --IMPORTS--> util
struct Fixture {
    node_ids: &'static [&'static str],
    /// (from, to, "CALLS"|"IMPORTS")
    edges: &'static [(&'static str, &'static str, &'static str)],
    /// deterministic 4-d embedding per node; `main` is the query target
    embeddings: Vec<(String, Vec<f32>)>,
}

fn fixture(graph_id: &str) -> Fixture {
    let emb = |id: &str, v: Vec<f32>| (canonical_oid(graph_id, id), v);
    Fixture {
        node_ids: &["main", "parse", "validate", "io", "util"],
        edges: &[
            ("main", "parse", "CALLS"),
            ("parse", "validate", "CALLS"),
            ("main", "io", "CALLS"),
            ("parse", "util", "IMPORTS"),
        ],
        embeddings: vec![
            emb("main", vec![1.0, 0.0, 0.0, 0.0]),
            emb("parse", vec![0.0, 1.0, 0.0, 0.0]),
            emb("validate", vec![0.0, 0.0, 1.0, 0.0]),
            emb("io", vec![0.0, 0.0, 0.0, 1.0]),
            emb("util", vec![0.1, 0.1, 0.1, 0.1]),
        ],
    }
}

/// Build an embedded DB seeded with the code-graph fixture under `graph_id`.
///
/// `graph_id` MUST be unique per test (see the module-level isolation note) —
/// `GraphCollectionService` shares graph metadata across in-process instances.
fn build_db(graph_id: &str) -> EmbeddedProximaDB {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_path = dir.path().join("data");
    std::fs::create_dir_all(&data_path).expect("create data dir");
    // Keep the tempdir alive for the DB's lifetime by leaking it (test process is short-lived).
    std::mem::forget(dir);

    let mut config = EmbeddedConfig::for_low_memory(data_path.to_string_lossy().to_string());
    config.enable_wal = true;
    let db = EmbeddedProximaDB::new(config).expect("create embedded db");

    let fx = fixture(graph_id);

    // Co-indexed vector collection: record id == canonical graph node oid.
    db.create_collection(VEC_COLLECTION, 4, Some("sst"))
        .expect("create vector collection");
    let ids: Vec<String> = fx.embeddings.iter().map(|(id, _)| id.clone()).collect();
    let vectors: Vec<Vec<f32>> = fx.embeddings.iter().map(|(_, v)| v.clone()).collect();
    db.insert(VEC_COLLECTION, ids, vectors, None)
        .expect("insert co-indexed vectors");

    // Graph: nodes + edges.
    db.create_graph(graph_id, None).expect("create graph");
    let nodes: Vec<EmbeddedGraphNode> = fx
        .node_ids
        .iter()
        .map(|id| EmbeddedGraphNode::new(*id).with_label("Symbol"))
        .collect();
    db.create_nodes(graph_id, nodes).expect("create nodes");
    let edges: Vec<EmbeddedGraphEdge> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (from, to, et))| EmbeddedGraphEdge::new(*from, *to, *et).with_id(format!("e{i}")))
        .collect();
    db.create_edges(graph_id, edges).expect("create edges");

    db
}

/// Hybrid seed→expand: query vector = `main`'s embedding (top seed), 1-hop CALLS expand reaches
/// parse + io. Fused oids = {main, parse, io}.
#[test]
fn embedded_fusion_search_seeds_and_expands() {
    let graph_id = "codegraph_fusion";
    let db = build_db(graph_id);

    let (items, stats) = db
        .fusion_search(
            graph_id,
            VEC_COLLECTION,
            vec![1.0, 0.0, 0.0, 0.0], // == main's embedding → top seed
            1,                        // max_depth
            vec!["CALLS".to_string()],
            1,     // max_seeds
            10,    // limit
            1.0,   // vector_weight
            1.0,   // graph_weight
            false, // calibrated (not rrf)
            "nodes",
            None, // text_query — no document modality (vector+graph only)
            None, // document_collection
            None, // document_weight
        )
        .expect("embedded fusion_search");

    let oids: HashSet<String> = items.into_iter().map(|i| i.oid).collect();
    assert!(stats.items_out > 0, "fusion must return candidates");
    assert!(
        oids.contains(&canonical_oid(graph_id, "main")),
        "seed oid present: {oids:?}"
    );
    assert!(
        oids.contains(&canonical_oid(graph_id, "parse"))
            && oids.contains(&canonical_oid(graph_id, "io")),
        "1-hop expand reaches parse + io: {oids:?}"
    );
}

/// TD-138 document modality over embedded fusion (parity with REST #508): supplying a
/// `text_query` makes the document source contribute. A document-ONLY oid (not in the vector or
/// graph) surfacing in the results proves the document modality is live end-to-end; a document
/// co-indexed with the top vector seed merges by shared `oid` (`source_count >= 2`).
#[test]
fn embedded_fusion_search_document_modality_contributes() {
    use proximadb::storage::engines::core::formats::columnar::fulltext_index::{
        FullTextIndex, TokenizerConfig,
    };
    let graph_id = "codegraph_fusion_doc";
    let db = build_db(graph_id);

    // Populate the co-indexed collection's full-text index with two documents whose doc_ids are
    // oids: (a) `main`'s canonical oid (co-indexed → merges with the vector seed), and (b) a
    // document-ONLY oid vector+graph alone could never surface.
    let main_oid = canonical_oid(graph_id, "main");
    let doc_solo = "doc_solo".to_string();
    let mut index = FullTextIndex::new(TokenizerConfig::for_keyword_search());
    index
        .add_document(main_oid.as_str(), "machine learning model")
        .expect("add main doc");
    index
        .add_document(doc_solo.as_str(), "machine learning algorithm")
        .expect("add solo doc");
    db.shared_services()
        .fulltext_indexes
        .write()
        .expect("fulltext lock")
        .insert(VEC_COLLECTION.to_string(), index);

    let (items, _stats) = db
        .fusion_search(
            graph_id,
            VEC_COLLECTION,
            vec![1.0, 0.0, 0.0, 0.0], // == main's embedding → top vector seed
            1,
            vec!["CALLS".to_string()],
            1,
            10,
            1.0,
            1.0,
            false,
            "nodes",
            Some("machine learning".to_string()), // text_query → document modality ON
            None,
            None,
        )
        .expect("embedded fusion_search with document modality");

    let oids: HashSet<String> = items.iter().map(|i| i.oid.clone()).collect();
    // The document-ONLY oid proves the document source contributed.
    assert!(
        oids.contains(&doc_solo),
        "document-only oid must surface when text_query is set: {oids:?}"
    );
    // The co-indexed document merged with the vector seed by shared oid (source_count >= 2).
    let main = items
        .iter()
        .find(|i| i.oid == main_oid)
        .expect("main oid fused");
    assert!(
        main.source_count >= 2,
        "vector + document merge on the shared oid: {main:?}"
    );
}

/// Forward blast radius from main (CALLS, depth 2): parse + io (d1), validate (d2).
#[test]
fn embedded_impact_analysis_forward() {
    let graph_id = "codegraph_fwd";
    let db = build_db(graph_id);
    let result = db
        .impact_analysis(
            graph_id,
            "main",
            "forward",
            Some(vec!["CALLS".to_string()]),
            2,
            100,
        )
        .expect("forward impact analysis");

    let ids: HashSet<String> = result.nodes.into_iter().map(|n| n.id).collect();
    for expected in ["parse", "io", "validate"] {
        assert!(
            ids.contains(expected),
            "forward from main should reach {expected}: {ids:?}"
        );
    }
}

/// Backward blast radius to validate (CALLS, depth 2): parse (d1), main (d2). util/io must NOT appear.
#[test]
fn embedded_impact_analysis_backward() {
    let graph_id = "codegraph_bwd";
    let db = build_db(graph_id);
    let result = db
        .impact_analysis(
            graph_id,
            "validate",
            "backward",
            Some(vec!["CALLS".to_string()]),
            2,
            100,
        )
        .expect("backward impact analysis");

    let ids: HashSet<String> = result.nodes.into_iter().map(|n| n.id).collect();
    assert!(
        ids.contains("parse") && ids.contains("main"),
        "backward to validate reaches parse + main: {ids:?}"
    );
    assert!(
        !ids.contains("util") && !ids.contains("io"),
        "util/io are not callers of validate: {ids:?}"
    );
}
