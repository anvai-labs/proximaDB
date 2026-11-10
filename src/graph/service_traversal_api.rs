//! Traversal and Pathfinding API (extracted from service.rs)
//!
//! Provides BFS/DFS traversal, connected components, cycle detection,
//! and shortest path (Dijkstra/A*) operations. This keeps the main
//! GraphOperationsService lean by separating traversal concerns.

use super::Result;
use crate::graph::NodeId;
use crate::graph::engines::GraphEngine;
use proximadb_kernel::error::ProximaDBError;
use tracing::debug;

impl super::GraphOperationsService {
    /// Perform graph traversal (basic implementation)
    pub async fn traverse(
        &self,
        graph_id: &str,
        request: crate::graph::TraversalRequest,
    ) -> Result<crate::graph::TraversalResponse> {
        use std::time::Instant;
        let _t0 = Instant::now();
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let config = crate::graph::engines::orion::traversal::TraversalConfig {
            max_depth: if request.max_depth > 0 {
                Some(request.max_depth)
            } else {
                None
            },
            max_nodes: None,
            edge_types: if request.edge_types.is_empty() {
                None
            } else {
                Some(request.edge_types.clone())
            },
            node_filter: None,
            early_stop: None,
            track_paths: true,
            parallel_processing: true,
            timeout_ms: request.timeout_ms.map(|t| t as u64),
            max_frontier: None,
            enable_prefetch: true,
            prefetch_budget: 8,
            astar_heuristic:
                crate::graph::engines::orion::traversal::AStarHeuristic::EuclideanEmbedding,
        };

        let engine = self.get_existing_graph_engine(graph_id).await?;
        let traversal_result = match &*engine {
            crate::graph::engines::GraphEngineImpl::Orion(e) => {
                crate::graph::engines::orion::traversal::breadth_first_search(
                    e,
                    &request.start_node_id,
                    config,
                )
                .await?
            }
        };

        // Materialize the result node payloads COLD-AWARE: the engine returns the
        // visited `node_ids` (topology), but under the cold-payload tier (TD-168)
        // its `nodes` is only the RAM-resident subset (empty after a topology-only
        // load). Fetch the full set from `node_ids` via the cold-aware batch —
        // engine-hot nodes stay in RAM, cold misses collapse to ~1 round-trip
        // (#487). In the all-RAM path (gate OFF) this resolves entirely from the
        // engine, so the result set is unchanged.
        let proto_nodes: Vec<crate::graph::Node> = self
            .get_nodes(graph_id, &traversal_result.node_ids)
            .await?
            .into_iter()
            .flatten()
            .map(|n| (*n).clone())
            .collect();
        let proto_edges: Vec<crate::graph::Edge> = traversal_result
            .edges
            .iter()
            .map(|e| (**e).clone())
            .collect();

        let proto_paths: Vec<crate::graph::GraphPath> = traversal_result
            .paths
            .iter()
            .map(|node_id_path| crate::graph::GraphPath {
                node_ids: node_id_path.to_vec(),
                edge_ids: vec![],
            })
            .collect();

        let proto_stats = Some(crate::graph::TraversalStats {
            nodes_visited: traversal_result.stats.nodes_visited as u32,
            edges_traversed: traversal_result.stats.edges_traversed as u32,
            max_depth_reached: traversal_result.stats.max_depth_reached,
            execution_time_microseconds: traversal_result.stats.execution_time_microseconds,
        });

        Ok(crate::graph::TraversalResponse {
            nodes: proto_nodes,
            edges: proto_edges,
            paths: proto_paths,
            stats: proto_stats,
        })
    }

    /// Perform graph traversal with per-call override hints (prefetch settings)
    pub async fn traverse_with_overrides(
        &self,
        graph_id: &str,
        request: crate::graph::TraversalRequest,
        _override_enable_prefetch: Option<bool>,
        _override_prefetch_budget: Option<usize>,
    ) -> Result<crate::graph::TraversalResponse> {
        use crate::graph::engines::orion::traversal::TraversalConfig;
        let traversal_config = TraversalConfig {
            enable_prefetch: _override_enable_prefetch.unwrap_or(true),
            prefetch_budget: _override_prefetch_budget.unwrap_or(1000),
            max_depth: if request.max_depth > 0 {
                Some(request.max_depth)
            } else {
                None
            },
            max_nodes: None,
            edge_types: None,
            node_filter: None,
            early_stop: None,
            track_paths: false,
            parallel_processing: true,
            timeout_ms: None,
            max_frontier: None,
            astar_heuristic:
                crate::graph::engines::orion::traversal::AStarHeuristic::EuclideanEmbedding,
        };
        self.traverse_with_config(graph_id, request, traversal_config)
            .await
    }

    /// Execute traversal with specific configuration
    async fn traverse_with_config(
        &self,
        graph_id: &str,
        request: crate::graph::TraversalRequest,
        config: crate::graph::engines::orion::traversal::TraversalConfig,
    ) -> Result<crate::graph::TraversalResponse> {
        let mut response = self.traverse(graph_id, request).await?;
        if config.enable_prefetch && config.prefetch_budget > 0 {
            debug!(
                "Traversal executed with prefetch budget: {}",
                config.prefetch_budget
            );
        }
        if let Some(stats) = &mut response.stats
            && stats.max_depth_reached > config.max_depth.unwrap_or(u32::MAX)
        {
            debug!("Traversal limited by max_depth: {:?}", config.max_depth);
        }
        Ok(response)
    }

    /// Get connected components (basic implementation)
    pub async fn connected_components(
        &self,
        graph_id: &str,
    ) -> Result<Vec<Vec<crate::graph::NodeId>>> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        let crate::graph::engines::GraphEngineImpl::Orion(e) = &*engine;
        crate::graph::engines::orion::traversal::connected_components(e).await
    }

    /// Check for cycles (basic implementation)
    pub async fn has_cycle(&self, graph_id: &str) -> Result<bool> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }
        let engine = self.get_existing_graph_engine(graph_id).await?;
        let crate::graph::engines::GraphEngineImpl::Orion(e) = &*engine;
        crate::graph::engines::orion::traversal::has_cycle(e).await
    }

    /// Compute shortest path (Dijkstra/A*) with optional k-shortest and overrides
    pub async fn shortest_path(
        &self,
        graph_id: &str,
        start_node_id: &NodeId,
        target_node_id: &NodeId,
        max_depth: Option<u32>,
        edge_types: Option<Vec<String>>,
        algorithm: Option<crate::proto::proximadb_v1::ShortestPathAlgorithm>,
        k: Option<u32>,
        override_enable_prefetch: Option<bool>,
        override_prefetch_budget: Option<usize>,
    ) -> Result<Option<(Vec<NodeId>, f64)>> {
        use crate::graph::engines::orion::traversal::{
            TraversalConfig, astar_shortest_path, dijkstra_shortest_path, k_shortest_paths,
        };
        let config = TraversalConfig {
            max_depth,
            max_nodes: None,
            edge_types,
            node_filter: None,
            early_stop: None,
            track_paths: true,
            parallel_processing: false,
            timeout_ms: Some(500),
            max_frontier: Some(100_000),
            enable_prefetch: override_enable_prefetch
                .unwrap_or(self.graph_settings.enable_prefetch),
            prefetch_budget: override_prefetch_budget
                .unwrap_or(self.graph_settings.prefetch_budget),
            astar_heuristic:
                crate::graph::engines::orion::traversal::AStarHeuristic::EuclideanEmbedding,
        };
        let engine = self.get_existing_graph_engine(graph_id).await?;
        let crate::graph::engines::GraphEngineImpl::Orion(orion_engine) = &*engine;

        if let Some(kk) = k
            && kk > 1
        {
            let paths = k_shortest_paths(
                orion_engine,
                start_node_id,
                target_node_id,
                kk as usize,
                config,
            )
            .await?;
            return Ok(paths.first().cloned());
        }

        let result = match algorithm
            .unwrap_or(crate::proto::proximadb_v1::ShortestPathAlgorithm::Dijkstra)
        {
            crate::proto::proximadb_v1::ShortestPathAlgorithm::Astar => {
                astar_shortest_path(orion_engine, start_node_id, target_node_id, config).await
            }
            _ => dijkstra_shortest_path(orion_engine, start_node_id, target_node_id, config).await,
        }?;

        if let Some(updater) = &self.metrics_updater {
            let _ = updater
                .record_operation(
                    "graph",
                    crate::metrics::updater::OperationMetricsUpdate {
                        operation_type: "graph.shortest_path".into(),
                        latency_us: 0.0,
                        success: result.is_some(),
                        bytes_processed: 0,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as i64,
                    },
                )
                .await;
        }
        Ok(result)
    }

    /// Single-step graph navigation for agentic tool-calling.
    ///
    /// Returns the immediate neighbors of `node_id` (optionally filtered by
    /// `edge_type`), capped at `limit`. This is the primitive that maps most
    /// directly to GraphWalk's "move + look" tool surface (arXiv:2604.01610):
    /// the agent picks one neighbor and calls again, so the database is never
    /// asked to materialize a subgraph that won't fit in the agent's context
    /// window.
    ///
    /// Use `graph_walk` when you want a bounded BFS expansion in one call.
    /// Use `graph_step` when you want the agent to drive traversal step by step.
    pub async fn graph_step(
        &self,
        graph_id: &str,
        node_id: &str,
        edge_type: Option<&str>,
        limit: usize,
    ) -> Result<crate::graph::canonical::TraversalResults> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let engine = self.get_existing_graph_engine(graph_id).await?;
        let node_id_owned = node_id.to_string();

        // Current node, then neighbors -- include the start so the agent has
        // its own properties available without an extra round trip.
        let current = engine
            .get_node(&node_id_owned)?
            .ok_or_else(|| ProximaDBError::InvalidInput(format!("node '{}' not found", node_id)))?;

        let neighbors = engine.get_neighbors(&node_id_owned, edge_type)?;
        let cap = if limit == 0 {
            neighbors.len()
        } else {
            limit.min(neighbors.len())
        };

        let mut canonical_nodes = Vec::with_capacity(cap + 1);
        canonical_nodes.push(crate::graph::canonical::CanonicalNode::from_proto(
            &(*current).clone().into(),
        ));
        for n in neighbors.iter().take(cap) {
            canonical_nodes.push(crate::graph::canonical::CanonicalNode::from_proto(
                &(**n).clone().into(),
            ));
        }

        Ok(crate::graph::canonical::TraversalResults {
            nodes: canonical_nodes,
            edges: Vec::new(),
            paths: None,
            stats: Some(crate::graph::canonical::TraversalStats {
                nodes_visited: cap as u64 + 1,
                edges_traversed: cap as u64,
                max_depth_reached: 1,
                execution_time_ms: None,
            }),
        })
    }

    /// Perform an iterative GraphWalk optimized for agentic tool-calling.
    ///
    /// This method provides a breadth-first exploration of the graph from a starting
    /// node, specifically designed for LLM agents to iteratively discover
    /// information without overwhelming their context window.
    pub async fn graph_walk(
        &self,
        graph_id: &str,
        start_node_id: &str,
        max_depth: u32,
        limit: usize,
    ) -> Result<crate::graph::canonical::TraversalResults> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let engine = self.get_existing_graph_engine(graph_id).await?;

        let start_node_id_string = start_node_id.to_string();

        // Use generic BFS as it returns the structured results we need
        let gtr = crate::graph::engines::generic_traversal::bfs_generic(
            engine.as_ref(),
            &start_node_id_string,
            None, // No specific edge types
            if max_depth > 0 { Some(max_depth) } else { None },
            Some(limit),
        )?;

        Ok(crate::graph::canonical::TraversalResults {
            nodes: gtr
                .nodes
                .iter()
                .map(|n| crate::graph::canonical::CanonicalNode::from_proto(&(**n).clone().into()))
                .collect(),
            edges: gtr
                .edges
                .iter()
                .map(|e| crate::graph::canonical::CanonicalEdge::from_proto(&(**e).clone().into()))
                .collect(),
            paths: None,
            stats: Some(crate::graph::canonical::TraversalStats {
                nodes_visited: gtr.nodes.len() as u64,
                edges_traversed: gtr.edges.len() as u64,
                max_depth_reached: max_depth,
                execution_time_ms: None,
            }),
        })
    }

    /// Impact analysis (TD-131): forward blast radius (OUTGOING edges — "what does X impact") or
    /// backward (INCOMING edges — "what impacts X"). Forward reuses [`Self::traverse`] (BFS already
    /// follows outgoing edges). Backward does a level-by-level BFS over incoming edges via
    /// [`Self::query_edges`] with `to_node_id`, since the engine exposes only outgoing adjacency —
    /// the same mechanism as the embedded `get_incoming_edges`. `nodes` is capped at `limit`.
    pub async fn impact_analysis(
        &self,
        graph_id: &str,
        start_node_id: &str,
        direction: crate::graph::model::ImpactDirection,
        edge_types: Vec<String>,
        max_depth: u32,
        limit: usize,
    ) -> Result<crate::graph::TraversalResponse> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        // Forward = standard BFS over outgoing edges.
        if matches!(direction, crate::graph::model::ImpactDirection::Forward) {
            let request = crate::graph::TraversalRequest {
                graph_id: graph_id.to_string(),
                start_node_id: start_node_id.to_string(),
                max_depth,
                edge_types,
                node_labels: Vec::new(),
                filters: Vec::new(),
                algorithm: 1, // TraversalAlgorithm::Bfs
                limit: Some(limit as u32),
                timeout_ms: None,
                max_frontier: None,
            };
            return self.traverse(graph_id, request).await;
        }

        // Backward = BFS over incoming edges (predecessors), level by level.
        // Materialization goes through the cold-aware service methods
        // (`get_node`/`get_nodes`), so a graph larger than RAM is served here too
        // (TD-168); the engine handle is no longer needed directly.
        use std::collections::HashSet;

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_node_id.to_string());
        let mut frontier: Vec<String> = vec![start_node_id.to_string()];

        let mut nodes: Vec<crate::graph::Node> = Vec::new();
        let start_id = start_node_id.to_string();
        if let Some(start) = self.get_node(graph_id, &start_id).await? {
            nodes.push((*start).clone());
        }
        let mut edges: Vec<crate::graph::Edge> = Vec::new();
        let mut depth_reached = 0u32;

        for depth in 0..max_depth {
            if frontier.is_empty() || nodes.len() >= limit {
                break;
            }
            depth_reached = depth + 1;
            let mut next_frontier: Vec<String> = Vec::new();
            for node_id in &frontier {
                let query = crate::graph::EdgeQuery {
                    graph_id: graph_id.to_string(),
                    from_node_id: None,
                    to_node_id: Some(node_id.clone()),
                    edge_types: edge_types.clone(),
                    filters: Vec::new(),
                    limit: None,
                    offset: None,
                    continuation_token: None,
                };
                let incoming = self.query_edges(graph_id, query).await?;
                for edge in incoming {
                    let predecessor = edge.from_node_id.clone();
                    edges.push((*edge).clone());
                    if visited.insert(predecessor.clone()) {
                        next_frontier.push(predecessor);
                    }
                }
            }
            // Batched, cold-aware materialization of the next frontier: engine-hot
            // nodes stay in RAM, cold misses collapse to ~1 round-trip (#487)
            // instead of one serial GET per node.
            for n in self
                .get_nodes(graph_id, &next_frontier)
                .await?
                .into_iter()
                .flatten()
            {
                if nodes.len() >= limit {
                    break;
                }
                nodes.push((*n).clone());
            }
            frontier = next_frontier;
        }

        let edges_traversed = edges.len() as u32;
        nodes.truncate(limit);
        Ok(crate::graph::TraversalResponse {
            nodes,
            edges,
            paths: Vec::new(),
            stats: Some(crate::graph::TraversalStats {
                nodes_visited: 0,
                edges_traversed,
                max_depth_reached: depth_reached,
                execution_time_microseconds: 0,
            }),
        })
    }
}
