//! Graph Engine Benchmark Report Generator (CSV)
//!
//! Generates CSV reports for graph engine operations:
//! - CRUD throughput (nodes/edges)
//! - Traversal latency (BFS/DFS)
//! - Shortest path (Dijkstra/A*)
//! - Connectivity (components) and cycle detection
//!
//! Run with:
//!   cargo bench --bench graph_engine_reporter

use anyhow::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use proximadb::graph::{
    Edge, Node, PropertyValue, TraversalRequest, property_value::Value,
    service::GraphOperationsService,
};
use proximadb::proto::proximadb_v1::{CreateGraphRequest, GraphEngineConfig, TraversalAlgorithm};
use proximadb::services::graph_collection::GraphCollectionService;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// Ensure we only write CSV once per benchmark process
static CSV_WRITTEN: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
struct GraphBenchConfig {
    engine: String,
    nodes: usize,
    edges: usize,
}

#[derive(Debug)]
struct GraphBenchResult {
    engine: String,
    nodes: usize,
    edges: usize,
    // CRUD
    node_insert_ms: u64,
    edge_insert_ms: u64,
    // Traversal
    bfs_ms: u64,
    dfs_ms: u64,
    // Shortest path
    dijkstra_ms: u64,
    astar_ms: u64,
    // Graph structure
    components_ms: u64,
    has_cycle_ms: u64,
}

async fn ensure_graph(collection_service: &GraphCollectionService, engine: &str, graph_id: &str) {
    if collection_service
        .list_graphs()
        .await
        .unwrap_or_default()
        .iter()
        .any(|g| g.graph_id == graph_id)
    {
        return;
    }
    let cfg = GraphEngineConfig {
        engine_type: engine.to_string(),
        memory_pool_size_mb: 0,
        csr_cache_size_mb: 0,
        enable_parallel_operations: true,
        max_traversal_depth: 10,
        advanced_config: HashMap::new(),
    };
    let req = CreateGraphRequest {
        graph_id: graph_id.to_string(),
        name: Some(graph_id.to_string()),
        description: None,
        schema: None,
        storage_config: None,
        engine_config: Some(cfg),
        access_control: None,
    };
    let _ = collection_service.create_graph(req).await;
}

async fn run_config(config: &GraphBenchConfig) -> Result<GraphBenchResult> {
    let graph_id = format!(
        "report_{}_{}n_{}e",
        config.engine.to_lowercase(),
        config.nodes,
        config.edges
    );
    let cs = Arc::new(GraphCollectionService::new());
    let service = Arc::new(GraphOperationsService::new_with_collection_service(
        cs.clone(),
    ));
    ensure_graph(&cs, &config.engine, &graph_id).await;

    // Node creation
    let t0 = Instant::now();
    for i in 0..config.nodes {
        let node = Node {
            id: format!("r_n{}", i),
            labels: vec!["R".to_string()],
            properties: HashMap::from([(
                "i".to_string(),
                PropertyValue {
                    value: Some(Value::IntValue(i as i64)),
                },
            )]),
            embedding: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };
        let _ = service.create_node(&graph_id, node).await.ok();
    }
    let node_insert_ms = t0.elapsed().as_millis() as u64;

    // Edge creation (pseudo-random connections)
    let t1 = Instant::now();
    for i in 0..config.edges {
        let from = i % config.nodes.max(1);
        let to = (i * 17 + 3) % config.nodes.max(1);
        if from == to {
            continue;
        }
        let edge = Edge {
            id: format!("r_e{}", i),
            from_node_id: format!("r_n{}", from),
            to_node_id: format!("r_n{}", to),
            edge_type: "R_E".to_string(),
            properties: HashMap::new(),
            weight: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        };
        let _ = service.create_edge(&graph_id, edge).await.ok();
    }
    let edge_insert_ms = t1.elapsed().as_millis() as u64;

    // Traversal BFS
    let t2 = Instant::now();
    let _ = service
        .traverse(
            &graph_id,
            TraversalRequest {
                graph_id: graph_id.clone(),
                start_node_id: "r_n0".to_string(),
                algorithm: TraversalAlgorithm::Bfs as i32,
                max_depth: 3,
                edge_types: vec![],
                node_labels: vec![],
                filters: vec![],
                limit: Some(256),
                timeout_ms: None,
                max_frontier: None,
            },
        )
        .await?;
    let bfs_ms = t2.elapsed().as_millis() as u64;

    // Traversal DFS
    let t3 = Instant::now();
    let _ = service
        .traverse(
            &graph_id,
            TraversalRequest {
                graph_id: graph_id.clone(),
                start_node_id: "r_n0".to_string(),
                algorithm: TraversalAlgorithm::Dfs as i32,
                max_depth: 4,
                edge_types: vec![],
                node_labels: vec![],
                filters: vec![],
                limit: Some(256),
                timeout_ms: None,
                max_frontier: None,
            },
        )
        .await?;
    let dfs_ms = t3.elapsed().as_millis() as u64;

    // Dijkstra
    let t4 = Instant::now();
    let _ = service
        .shortest_path(
            &graph_id,
            &"r_n0".to_string(),
            &format!("r_n{}", (config.nodes / 2).max(1)),
            Some(64),
            None,
            None,
            None,
            None,
            None,
        )
        .await?;
    let dijkstra_ms = t4.elapsed().as_millis() as u64;

    // A*
    let t5 = Instant::now();
    let _ = service
        .shortest_path(
            &graph_id,
            &"r_n0".to_string(),
            &format!("r_n{}", (config.nodes / 2).max(1)),
            Some(64),
            None,
            Some(proximadb::proto::proximadb_v1::ShortestPathAlgorithm::Astar),
            None,
            None,
            None,
        )
        .await?;
    let astar_ms = t5.elapsed().as_millis() as u64;

    // Components
    let t6 = Instant::now();
    let _ = service.connected_components(&graph_id).await?;
    let components_ms = t6.elapsed().as_millis() as u64;

    // Cycle detection
    let t7 = Instant::now();
    let _ = service.has_cycle(&graph_id).await?;
    let has_cycle_ms = t7.elapsed().as_millis() as u64;

    Ok(GraphBenchResult {
        engine: config.engine.clone(),
        nodes: config.nodes,
        edges: config.edges,
        node_insert_ms,
        edge_insert_ms,
        bfs_ms,
        dfs_ms,
        dijkstra_ms,
        astar_ms,
        components_ms,
        has_cycle_ms,
    })
}

fn write_csv(results: &[GraphBenchResult], filename: &str) -> Result<()> {
    let mut f = File::create(filename)?;
    writeln!(
        f,
        "Engine,Nodes,Edges,NodeInsertMs,EdgeInsertMs,NodeInsertKPerSec,EdgeInsertKPerSec,\
        BfsMs,DfsMs,DijkstraMs,AstarMs,ComponentsMs,HasCycleMs"
    )?;
    for r in results {
        // K elem/s = elems / ms
        let node_kps = if r.node_insert_ms > 0 {
            (r.nodes as f64) / (r.node_insert_ms as f64)
        } else {
            0.0
        };
        let edge_kps = if r.edge_insert_ms > 0 {
            (r.edges as f64) / (r.edge_insert_ms as f64)
        } else {
            0.0
        };
        writeln!(
            f,
            "{},{},{},{},{},{:.2},{:.2},{},{},{},{},{},{}",
            r.engine,
            r.nodes,
            r.edges,
            r.node_insert_ms,
            r.edge_insert_ms,
            node_kps,
            edge_kps,
            r.bfs_ms,
            r.dfs_ms,
            r.dijkstra_ms,
            r.astar_ms,
            r.components_ms,
            r.has_cycle_ms
        )?;
    }
    println!("✅ Graph report written to {}", filename);
    Ok(())
}

fn bench_graph_engine_reporter(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    c.bench_function("graph_engine_reporter_generate", |b| {
        b.iter(|| {
            rt.block_on(async {
                let engines = ["ORION"];
                let scales = vec![(1000, 3000), (5000, 15000)];

                let mut results = Vec::new();
                for e in engines {
                    for (n, m) in &scales {
                        let cfg = GraphBenchConfig {
                            engine: e.to_string(),
                            nodes: *n,
                            edges: *m,
                        };
                        if let Ok(r) = run_config(&cfg).await {
                            results.push(r);
                        }
                    }
                }

                // Write CSV exactly once per benchmark process
                if !CSV_WRITTEN.swap(true, Ordering::SeqCst) {
                    let target_dir =
                        std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
                    let _ = std::fs::create_dir_all(&target_dir);
                    let latest = format!("{}/graph_benchmark_report_latest.csv", target_dir);
                    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
                    let stamped = format!("{}/graph_benchmark_report_{}.csv", target_dir, ts);
                    let _ = write_csv(&results, &latest);
                    let _ = write_csv(&results, &stamped);
                }
            })
        })
    });
}

criterion_group!(benches, bench_graph_engine_reporter);
criterion_main!(benches);
