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

//! Graph algorithm service methods
//!
//! This module provides high-level service methods for running graph algorithms:
//! - **Centrality**: PageRank, closeness, harmonic, betweenness
//! - **Community Detection**: Louvain, label propagation
//! - **Pathfinding**: Shortest path (already in traversal API)
//!
//! These methods are exposed via GraphOperationsService and can be called from
//! REST, gRPC, or embedded APIs.

use crate::graph::service::GraphOperationsService;
use proximadb_kernel::error::ProximaDBError;
use std::collections::HashMap;
use std::sync::Arc;

type Result<T> = std::result::Result<T, ProximaDBError>;

/// Centrality algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralityAlgorithm {
    /// PageRank - measures node importance based on incoming links
    PageRank,
    /// Closeness - measures how close a node is to all others
    Closeness,
    /// Harmonic - variant of closeness that handles disconnected graphs
    Harmonic,
    /// Betweenness - measures how often a node lies on shortest paths
    Betweenness,
}

/// Community detection algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain - greedy modularity optimization
    Louvain,
    /// Label propagation - fast semi-supervised clustering
    LabelPropagation,
}

/// Centrality algorithm configuration
#[derive(Debug, Clone)]
pub struct CentralityConfig {
    /// Damping factor for PageRank (default: 0.85)
    pub damping_factor: f64,
    /// Maximum iterations (default: 100)
    pub max_iterations: usize,
    /// Convergence tolerance (default: 1e-6)
    pub tolerance: f64,
    /// Whether to normalize scores (default: true)
    pub normalized: bool,
}

impl Default for CentralityConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            normalized: true,
        }
    }
}

/// Community detection configuration
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Resolution parameter for Louvain (default: 1.0)
    pub resolution: f64,
    /// Maximum iterations (default: 100)
    pub max_iterations: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
        }
    }
}

/// Result of a centrality algorithm
#[derive(Debug, Clone)]
pub struct CentralityResult {
    /// Node ID to centrality score mapping
    pub scores: HashMap<String, f64>,
    /// Algorithm used
    pub algorithm: String,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of nodes processed
    pub node_count: usize,
}

/// Result of community detection
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Node ID to community ID mapping
    pub communities: HashMap<String, usize>,
    /// Number of communities detected
    pub community_count: usize,
    /// Modularity score (if applicable)
    pub modularity: Option<f64>,
    /// Algorithm used
    pub algorithm: String,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

impl GraphOperationsService {
    /// Run a centrality algorithm on a graph
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    /// * `algorithm` - The centrality algorithm to use
    /// * `config` - Optional configuration (uses defaults if not provided)
    ///
    /// # Returns
    ///
    /// Centrality scores for each node in the graph
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = service.run_centrality(
    ///     "my_graph",
    ///     CentralityAlgorithm::PageRank,
    ///     None
    /// ).await?;
    ///
    /// for (node_id, score) in result.scores.iter() {
    ///     println!("{}: {:.4}", node_id, score);
    /// }
    /// ```
    pub async fn run_centrality(
        &self,
        graph_id: &str,
        algorithm: CentralityAlgorithm,
        config: Option<CentralityConfig>,
    ) -> Result<CentralityResult> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        let config = config.unwrap_or_default();

        let engine = self.get_existing_graph_engine(graph_id).await?;

        let crate::graph::engines::GraphEngineImpl::Orion(orion_engine) = engine.as_ref();

        let scores = match algorithm {
            CentralityAlgorithm::PageRank => {
                // Use existing page_rank function from traversal.rs
                crate::graph::engines::orion::traversal::page_rank(
                    orion_engine,
                    config.damping_factor,
                    config.max_iterations,
                    config.tolerance,
                )
                .await?
            }
            CentralityAlgorithm::Closeness => {
                // Use ClosenessCentrality from algorithms module
                use crate::graph::engines::orion::algorithms::centrality::ClosenessCentrality;
                use crate::graph::engines::orion::algorithms::traits::{GraphAlgorithm, NoInput};

                let closeness =
                    ClosenessCentrality::new(Arc::new(orion_engine.clone()), config.normalized);
                closeness.execute(NoInput)?
            }
            CentralityAlgorithm::Harmonic => {
                // Use HarmonicCentrality from algorithms module
                use crate::graph::engines::orion::algorithms::centrality::HarmonicCentrality;
                use crate::graph::engines::orion::algorithms::traits::{GraphAlgorithm, NoInput};

                let harmonic =
                    HarmonicCentrality::new(Arc::new(orion_engine.clone()), config.normalized);
                harmonic.execute(NoInput)?
            }
            CentralityAlgorithm::Betweenness => {
                // Betweenness not yet implemented - return error
                return Err(ProximaDBError::NotImplemented(
                    "Betweenness centrality not yet implemented".to_string(),
                ));
            }
        };

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let node_count = scores.len();

        Ok(CentralityResult {
            scores,
            algorithm: format!("{:?}", algorithm),
            execution_time_ms,
            node_count,
        })
    }

    /// Run community detection on a graph
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    /// * `algorithm` - The community detection algorithm to use
    /// * `config` - Optional configuration (uses defaults if not provided)
    ///
    /// # Returns
    ///
    /// Community assignments for each node
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = service.run_community_detection(
    ///     "my_graph",
    ///     CommunityAlgorithm::Louvain,
    ///     None
    /// ).await?;
    ///
    /// println!("Found {} communities", result.community_count);
    /// ```
    pub async fn run_community_detection(
        &self,
        graph_id: &str,
        algorithm: CommunityAlgorithm,
        config: Option<CommunityConfig>,
    ) -> Result<CommunityResult> {
        if !self.graph_enabled() {
            return Err(ProximaDBError::InvalidInput(
                "Graph operations disabled in current mode".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        let config = config.unwrap_or_default();

        let engine = self.get_existing_graph_engine(graph_id).await?;

        let crate::graph::engines::GraphEngineImpl::Orion(orion_engine) = engine.as_ref();

        let (communities, modularity) = match algorithm {
            CommunityAlgorithm::Louvain => {
                use crate::graph::engines::orion::algorithms::community::LouvainCommunityDetection;
                use crate::graph::engines::orion::algorithms::traits::{GraphAlgorithm, NoInput};

                // Get CSR storage from ORION engine
                let csr_guard = orion_engine.csr_outgoing.read().map_err(|_| {
                    ProximaDBError::Internal("Failed to acquire CSR read lock".to_string())
                })?;

                // Clone CSR for algorithm (algorithms need Arc<CsrStorage>)
                let csr = Arc::new(csr_guard.clone());
                drop(csr_guard);

                let louvain = LouvainCommunityDetection::new(
                    csr.clone(),
                    config.resolution,
                    config.max_iterations,
                );

                let communities = louvain.execute(NoInput)?;

                // Calculate modularity
                let communities_usize: HashMap<usize, usize> = communities
                    .iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, *v)))
                    .collect();

                let modularity = louvain.compute_modularity(&communities_usize);

                (communities, Some(modularity))
            }
            CommunityAlgorithm::LabelPropagation => {
                // Label propagation not yet implemented
                return Err(ProximaDBError::NotImplemented(
                    "Label propagation not yet implemented".to_string(),
                ));
            }
        };

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Count unique communities
        let community_count = communities
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(CommunityResult {
            communities,
            community_count,
            modularity,
            algorithm: format!("{:?}", algorithm),
            execution_time_ms,
        })
    }

    /// Convenience method to run PageRank with default settings
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    ///
    /// # Returns
    ///
    /// PageRank scores for each node
    pub async fn pagerank(&self, graph_id: &str) -> Result<HashMap<String, f64>> {
        let result = self
            .run_centrality(graph_id, CentralityAlgorithm::PageRank, None)
            .await?;
        Ok(result.scores)
    }

    /// Convenience method to run PageRank with custom parameters
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    /// * `damping_factor` - Damping factor (typically 0.85)
    /// * `max_iterations` - Maximum iterations
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// PageRank scores for each node
    pub async fn pagerank_with_params(
        &self,
        graph_id: &str,
        damping_factor: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<HashMap<String, f64>> {
        let config = CentralityConfig {
            damping_factor,
            max_iterations,
            tolerance,
            normalized: true,
        };
        let result = self
            .run_centrality(graph_id, CentralityAlgorithm::PageRank, Some(config))
            .await?;
        Ok(result.scores)
    }

    /// Convenience method to run Louvain community detection
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    ///
    /// # Returns
    ///
    /// Community assignments for each node
    pub async fn detect_communities(&self, graph_id: &str) -> Result<HashMap<String, usize>> {
        let result = self
            .run_community_detection(graph_id, CommunityAlgorithm::Louvain, None)
            .await?;
        Ok(result.communities)
    }

    /// Get centrality scores for specific nodes
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    /// * `algorithm` - The centrality algorithm to use
    /// * `node_ids` - List of node IDs to get scores for
    ///
    /// # Returns
    ///
    /// Centrality scores for requested nodes (nodes not found are omitted)
    pub async fn get_node_centrality(
        &self,
        graph_id: &str,
        algorithm: CentralityAlgorithm,
        node_ids: &[String],
    ) -> Result<HashMap<String, f64>> {
        let result = self.run_centrality(graph_id, algorithm, None).await?;

        let filtered: HashMap<String, f64> = node_ids
            .iter()
            .filter_map(|id| result.scores.get(id).map(|score| (id.clone(), *score)))
            .collect();

        Ok(filtered)
    }

    /// Get top N nodes by centrality
    ///
    /// # Arguments
    ///
    /// * `graph_id` - The graph to analyze
    /// * `algorithm` - The centrality algorithm to use
    /// * `n` - Number of top nodes to return
    ///
    /// # Returns
    ///
    /// Top N nodes sorted by centrality score (descending)
    pub async fn top_central_nodes(
        &self,
        graph_id: &str,
        algorithm: CentralityAlgorithm,
        n: usize,
    ) -> Result<Vec<(String, f64)>> {
        let result = self.run_centrality(graph_id, algorithm, None).await?;

        let mut scores_vec: Vec<(String, f64)> = result.scores.into_iter().collect();
        scores_vec.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or({
                // If either score is NaN, treat them as equal
                std::cmp::Ordering::Equal
            })
        });
        scores_vec.truncate(n);

        Ok(scores_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::service::GraphOperationsService;
    use crate::graph::{Edge as ProtoEdge, Node as ProtoNode};
    use crate::proto::proximadb_v1::CreateGraphRequest;
    use std::collections::HashMap;

    async fn create_test_graph(service: &GraphOperationsService) -> String {
        let graph_id = format!("test_algo_graph_{}", uuid::Uuid::new_v4());

        // Create graph
        let req = CreateGraphRequest {
            graph_id: graph_id.clone(),
            name: Some(graph_id.clone()),
            description: None,
            schema: None,
            storage_config: None,
            engine_config: None,
            access_control: None,
        };
        service
            .create_graph_collection(req)
            .await
            .expect("Failed to create test graph collection");

        // Create nodes
        for i in 0..5 {
            let node = ProtoNode {
                id: format!("n{}", i),
                labels: vec!["Node".to_string()],
                properties: HashMap::new(),
                embedding: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            };
            service
                .create_node(&graph_id, node)
                .await
                .expect("Failed to create test node");
        }

        // Create edges (star topology with n0 at center)
        // n0 -> n1, n0 -> n2, n0 -> n3, n0 -> n4
        for i in 1..5 {
            let edge = ProtoEdge {
                id: format!("e0{}", i),
                from_node_id: "n0".to_string(),
                to_node_id: format!("n{}", i),
                edge_type: "CONNECTS".to_string(),
                properties: HashMap::new(),
                weight: None,
                created_at_ms: 0,
                updated_at_ms: 0,
            };
            service
                .create_edge(&graph_id, edge)
                .await
                .expect("Failed to create test edge");
        }

        graph_id
    }

    #[tokio::test]
    async fn test_pagerank() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let scores = service
            .pagerank(&graph_id)
            .await
            .expect("PageRank should succeed");

        // All nodes should have scores
        assert_eq!(scores.len(), 5);

        // Center node (n0) should have highest score in this star topology
        // because it has the most outgoing links
        for score in scores.values() {
            assert!(*score > 0.0, "All scores should be positive");
        }
    }

    #[tokio::test]
    async fn test_closeness_centrality() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let result = service
            .run_centrality(&graph_id, CentralityAlgorithm::Closeness, None)
            .await
            .expect("Closeness centrality should succeed");

        assert_eq!(result.node_count, 5);
        assert!(result.execution_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_harmonic_centrality() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let result = service
            .run_centrality(&graph_id, CentralityAlgorithm::Harmonic, None)
            .await
            .expect("Harmonic centrality should succeed");

        assert_eq!(result.node_count, 5);
    }

    #[tokio::test]
    async fn test_louvain_community_detection() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let result = service
            .run_community_detection(&graph_id, CommunityAlgorithm::Louvain, None)
            .await
            .expect("Louvain community detection should succeed");

        // All nodes should have community assignments
        assert_eq!(result.communities.len(), 5);

        // Should have at least 1 community
        assert!(result.community_count >= 1);

        // Modularity should be finite
        let modularity = match result.modularity {
            Some(value) => value,
            None => panic!("Modularity should be present for Louvain"),
        };
        assert!(modularity.is_finite());
    }

    #[tokio::test]
    async fn test_top_central_nodes() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let top_3 = service
            .top_central_nodes(&graph_id, CentralityAlgorithm::PageRank, 3)
            .await
            .expect("top_central_nodes should succeed");

        assert_eq!(top_3.len(), 3);

        // Scores should be sorted descending
        for window in top_3.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    #[tokio::test]
    async fn test_get_node_centrality() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let node_ids = vec![
            "n0".to_string(),
            "n1".to_string(),
            "nonexistent".to_string(),
        ];
        let scores = service
            .get_node_centrality(&graph_id, CentralityAlgorithm::PageRank, &node_ids)
            .await
            .expect("get_node_centrality should succeed");

        // Should only return scores for existing nodes
        assert!(scores.contains_key("n0"));
        assert!(scores.contains_key("n1"));
        assert!(!scores.contains_key("nonexistent"));
    }

    #[tokio::test]
    async fn test_betweenness_not_implemented() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let result = service
            .run_centrality(&graph_id, CentralityAlgorithm::Betweenness, None)
            .await;

        assert!(matches!(result, Err(ProximaDBError::NotImplemented(_))));
    }

    #[tokio::test]
    async fn test_label_propagation_not_implemented() {
        let service = GraphOperationsService::new();
        let graph_id = create_test_graph(&service).await;

        let result = service
            .run_community_detection(&graph_id, CommunityAlgorithm::LabelPropagation, None)
            .await;

        assert!(matches!(result, Err(ProximaDBError::NotImplemented(_))));
    }
}
