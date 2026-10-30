/*
 * Copyright 2025 ProximaDB
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

//! Centrality algorithms
//!
//! Implements various centrality measures for identifying important nodes:
//! - **Closeness Centrality**: Measures how close a node is to all other nodes
//! - **Harmonic Centrality**: Harmonic mean of distances (handles disconnected graphs)
//! - **Betweenness Centrality**: Measures how often a node lies on shortest paths
//! - **PageRank**: Google's ranking algorithm (existing in traversal.rs)
//!
//! # Design Principles
//!
//! 1. **Reuse BFS**: All distance-based centrality measures reuse existing BFS from traversal.rs
//! 2. **Parallel Execution**: Leverage Rayon for parallel BFS from multiple sources
//! 3. **CSR Access**: Direct access to CSR storage for O(degree) neighbor queries
//! 4. **Incremental Updates**: Support for dynamic graphs via IncrementalAlgorithm trait
//!
//! # Example
//!
//! ```ignore
//! use proximadb::graph::engines::orion::algorithms::centrality::ClosenessCentrality;
//! use proximadb::graph::engines::orion::algorithms::traits::GraphAlgorithm;
//!
//! let closeness = ClosenessCentrality::new(engine, true);
//! let scores = closeness.execute(())?;
//!
//! // scores = HashMap<NodeId, f64>
//! // Higher score = more central node
//! ```

use super::traits::{
    AlgorithmComplexity, CentralityScores, GraphAlgorithm, NoInput, ParallelAlgorithm,
};
use crate::OrionGraphEngine;
use proximadb_kernel::error::ProximaDBError;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Single-source shortest-path distances (unit edge weights) via BFS over the
/// outgoing CSR — the shared distance primitive behind the closeness and
/// harmonic centrality passes.
fn bfs_distances(
    engine: &OrionGraphEngine,
    source_idx: usize,
) -> Result<HashMap<usize, usize>, ProximaDBError> {
    let mut distances: HashMap<usize, usize> = HashMap::new();
    let mut queue = VecDeque::new();

    distances.insert(source_idx, 0);
    queue.push_back(source_idx);

    // BFS using CSR storage
    let csr_out = engine
        .csr_outgoing
        .read()
        .map_err(|_| ProximaDBError::Internal("Failed to acquire CSR read lock".to_string()))?;

    while let Some(current_idx) = queue.pop_front() {
        let current_distance = distances[&current_idx];

        // Get neighbors from CSR (O(degree) access)
        let neighbors = csr_out.get_neighbors(current_idx).unwrap_or(&[]);

        for &neighbor_idx in neighbors {
            if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor_idx) {
                e.insert(current_distance + 1);
                queue.push_back(neighbor_idx);
            }
        }
    }

    Ok(distances)
}

/// The graph shape every centrality pass starts from: the node count, the
/// index-to-node-id mapping, and the CSR read guard (held or dropped by the
/// caller, since the sequential betweenness pass needs the CSR for its BFS).
/// `Ok(None)` means an empty graph — every caller short-circuits to empty
/// scores.
fn centrality_shape(
    engine: &OrionGraphEngine,
) -> Result<
    Option<(
        usize,
        std::sync::RwLockReadGuard<'_, Vec<String>>,
        std::sync::RwLockReadGuard<'_, crate::storage::CsrStorage>,
    )>,
    ProximaDBError,
> {
    let csr_out = engine
        .csr_outgoing
        .read()
        .map_err(|_| ProximaDBError::Internal("Failed to acquire CSR read lock".to_string()))?;
    let node_count = csr_out.node_count();
    if node_count == 0 {
        return Ok(None);
    }
    let index_to_node = engine.index_to_node.read().map_err(|_| {
        ProximaDBError::Internal("Failed to acquire index mapping lock".to_string())
    })?;
    Ok(Some((node_count, index_to_node, csr_out)))
}

/// Node id for a CSR index, falling back to the index's decimal string when
/// the mapping has no entry.
fn node_id_for(index_to_node: &[String], idx: usize) -> String {
    index_to_node
        .get(idx)
        .cloned()
        .unwrap_or_else(|| idx.to_string())
}

/// Closeness of one node from its BFS distances — shared by the sequential
/// and parallel passes. If disconnected, only reachable nodes count; an
/// isolated node scores 0.
fn closeness_score(distances: &HashMap<usize, usize>, node_count: usize, normalized: bool) -> f64 {
    // Sum of distances to all reachable nodes
    let total_distance: usize = distances.values().sum();

    // Closeness = (n-1) / sum of distances
    // If disconnected, only count reachable nodes
    let reachable_count = distances.len() - 1; // Exclude source itself

    if total_distance > 0 {
        if normalized {
            (reachable_count as f64) / (total_distance as f64)
        } else {
            // Non-normalized: just reciprocal of average distance
            (reachable_count as f64) / (total_distance as f64) * ((node_count - 1) as f64)
        }
    } else {
        0.0 // Isolated node
    }
}

/// Harmonic centrality of one node from its BFS distances — shared by the
/// sequential and parallel passes (unreachable nodes contribute 0).
fn harmonic_score(
    distances: &HashMap<usize, usize>,
    node_idx: usize,
    node_count: usize,
    normalized: bool,
) -> f64 {
    // Sum of reciprocals of distances
    let harmonic_sum: f64 = distances
        .iter()
        .filter(|(idx, _)| **idx != node_idx) // Exclude source itself
        .map(|(_, dist)| {
            if *dist > 0 {
                1.0 / (*dist as f64)
            } else {
                0.0 // Source to itself
            }
        })
        .sum();

    if normalized {
        harmonic_sum / ((node_count - 1) as f64)
    } else {
        harmonic_sum
    }
}

/// One Brandes source pass over the CSR: BFS counting shortest paths, then
/// dependency back-propagation. Returns the per-node dependency contribution
/// of `s`; callers accumulate it while skipping the source itself.
fn brandes_source(csr: &crate::storage::CsrStorage, s: usize, node_count: usize) -> Vec<f64> {
    // BFS from source s
    let mut stack: Vec<usize> = Vec::new();
    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; node_count];
    let mut sigma = vec![0.0f64; node_count]; // number of shortest paths
    sigma[s] = 1.0;
    let mut dist: Vec<i64> = vec![-1; node_count];
    dist[s] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(s);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        let neighbors = csr.get_neighbors(v).unwrap_or(&[]);
        for &w in neighbors {
            // w found for the first time?
            if dist[w] < 0 {
                dist[w] = dist[v] + 1;
                queue.push_back(w);
            }
            // Is this a shortest path to w via v?
            if dist[w] == dist[v] + 1 {
                sigma[w] += sigma[v];
                predecessors[w].push(v);
            }
        }
    }

    // Back-propagation of dependencies
    let mut delta = vec![0.0f64; node_count];
    while let Some(w) = stack.pop() {
        for &v in &predecessors[w] {
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
    }
    delta
}

/// Shared betweenness tail: optional (n-1)(n-2) normalization, then the
/// index-to-node-id projection into the score map.
fn betweenness_scores(
    mut betweenness: Vec<f64>,
    index_to_node: &[String],
    normalized: bool,
) -> HashMap<String, f64> {
    let node_count = betweenness.len();

    // Normalize for undirected graphs: divide by 2
    // For directed graphs the formula is already correct
    // Optionally normalize to [0,1] by dividing by (n-1)(n-2)
    if normalized && node_count > 2 {
        let norm = ((node_count - 1) * (node_count - 2)) as f64;
        for b in &mut betweenness {
            *b /= norm;
        }
    }

    let mut scores = HashMap::new();
    for (idx, &b) in betweenness.iter().enumerate() {
        scores.insert(node_id_for(index_to_node, idx), b);
    }
    scores
}

/// Closeness centrality algorithm
///
/// Closeness centrality measures how close a node is to all other nodes in the graph.
/// It's defined as the reciprocal of the sum of distances to all other nodes.
///
/// Formula:
/// C(v) = (n-1) / Σ d(v, u) for all u ≠ v
///
/// Where:
/// - n = number of nodes
/// - d(v, u) = shortest path distance from v to u
///
/// # Algorithm Complexity
///
/// - Time: O(n * (n + m)) where n = nodes, m = edges (BFS from each node)
/// - Space: O(n) for distance arrays
///
/// # Design
///
/// - **Reuses BFS**: Leverages existing breadth_first_search from traversal.rs
/// - **Parallel**: Computes BFS from all nodes in parallel using Rayon
/// - **Normalized**: Option to normalize scores to [0, 1] range
///
/// # References
///
/// Freeman, L. C. "Centrality in social networks: Conceptual clarification."
/// Social Networks 1.3 (1978): 215-239.
pub struct ClosenessCentrality {
    /// ORION graph engine (reused for BFS traversal)
    engine: Arc<OrionGraphEngine>,

    /// Whether to normalize scores (divide by n-1)
    normalized: bool,
}

impl ClosenessCentrality {
    /// Create a new closeness centrality algorithm
    ///
    /// # Arguments
    ///
    /// * `engine` - ORION graph engine to operate on
    /// * `normalized` - Whether to normalize scores to [0, 1] range
    ///
    /// # Returns
    ///
    /// New closeness centrality instance
    pub fn new(engine: Arc<OrionGraphEngine>, normalized: bool) -> Self {
        Self { engine, normalized }
    }
}

impl GraphAlgorithm for ClosenessCentrality {
    type Input = NoInput;
    type Output = CentralityScores;

    fn execute(&self, _input: NoInput) -> Result<CentralityScores, ProximaDBError> {
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };
        drop(csr_out); // Release lock before parallel BFS

        let mut scores: CentralityScores = HashMap::new();

        // Compute closeness for each node
        for node_idx in 0..node_count {
            let distances = bfs_distances(&self.engine, node_idx)?;
            let closeness = closeness_score(&distances, node_count, self.normalized);
            // Map index to node ID
            scores.insert(node_id_for(&index_to_node, node_idx), closeness);
        }

        Ok(scores)
    }

    fn estimated_complexity(&self) -> AlgorithmComplexity {
        // O(n * (n + m)) = O(n²) for sparse graphs
        AlgorithmComplexity::QuadraticVertices
    }

    fn name(&self) -> &'static str {
        "ClosenessCentrality"
    }
}

impl ParallelAlgorithm for ClosenessCentrality {
    fn execute_parallel(
        &self,
        _input: NoInput,
        _thread_pool: &rayon::ThreadPool,
    ) -> Result<CentralityScores, ProximaDBError> {
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };
        drop(csr_out);

        // Parallel BFS from all nodes
        let scores_vec: Vec<(String, f64)> = (0..node_count)
            .into_par_iter()
            .filter_map(|node_idx| {
                let distances = bfs_distances(&self.engine, node_idx).ok()?;
                let closeness = closeness_score(&distances, node_count, self.normalized);
                Some((node_id_for(&index_to_node, node_idx), closeness))
            })
            .collect();

        Ok(scores_vec.into_iter().collect())
    }

    fn min_graph_size_for_parallel(&self) -> usize {
        // Closeness benefits from parallelism for graphs with > 100 nodes
        100
    }
}

/// Harmonic centrality algorithm
///
/// Harmonic centrality is a variant of closeness centrality that handles disconnected graphs better.
/// Instead of summing distances, it sums the reciprocals of distances.
///
/// Formula:
/// H(v) = Σ 1/d(v, u) for all u ≠ v
///
/// Where:
/// - d(v, u) = shortest path distance from v to u
/// - If u is unreachable, 1/d(v, u) = 0
///
/// # Algorithm Complexity
///
/// - Time: O(n * (n + m)) where n = nodes, m = edges
/// - Space: O(n) for distance arrays
///
/// # Design
///
/// - **Handles Disconnected Graphs**: Gracefully handles infinite distances (unreachable nodes)
/// - **Reuses BFS**: Same BFS infrastructure as closeness centrality
/// - **Parallel**: Computes BFS from all nodes in parallel
///
/// # References
///
/// Boldi, P., & Vigna, S. "Axioms for centrality." Internet Mathematics 10.3-4 (2014): 222-262.
pub struct HarmonicCentrality {
    /// ORION graph engine (reused for BFS traversal)
    engine: Arc<OrionGraphEngine>,

    /// Whether to normalize scores by (n-1)
    normalized: bool,
}

impl HarmonicCentrality {
    /// Create a new harmonic centrality algorithm
    ///
    /// # Arguments
    ///
    /// * `engine` - ORION graph engine to operate on
    /// * `normalized` - Whether to normalize scores by (n-1)
    ///
    /// # Returns
    ///
    /// New harmonic centrality instance
    pub fn new(engine: Arc<OrionGraphEngine>, normalized: bool) -> Self {
        Self { engine, normalized }
    }
}

impl GraphAlgorithm for HarmonicCentrality {
    type Input = NoInput;
    type Output = CentralityScores;

    fn execute(&self, _input: NoInput) -> Result<CentralityScores, ProximaDBError> {
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };
        drop(csr_out);

        let mut scores: CentralityScores = HashMap::new();

        // Compute harmonic centrality for each node
        for node_idx in 0..node_count {
            let distances = bfs_distances(&self.engine, node_idx)?;
            let harmonic_centrality =
                harmonic_score(&distances, node_idx, node_count, self.normalized);
            scores.insert(node_id_for(&index_to_node, node_idx), harmonic_centrality);
        }

        Ok(scores)
    }

    fn estimated_complexity(&self) -> AlgorithmComplexity {
        // O(n * (n + m)) = O(n²) for sparse graphs
        AlgorithmComplexity::QuadraticVertices
    }

    fn name(&self) -> &'static str {
        "HarmonicCentrality"
    }
}

impl ParallelAlgorithm for HarmonicCentrality {
    fn execute_parallel(
        &self,
        _input: NoInput,
        _thread_pool: &rayon::ThreadPool,
    ) -> Result<CentralityScores, ProximaDBError> {
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };
        drop(csr_out);

        // Parallel BFS from all nodes
        let scores_vec: Vec<(String, f64)> = (0..node_count)
            .into_par_iter()
            .filter_map(|node_idx| {
                let distances = bfs_distances(&self.engine, node_idx).ok()?;
                let harmonic_centrality =
                    harmonic_score(&distances, node_idx, node_count, self.normalized);
                Some((node_id_for(&index_to_node, node_idx), harmonic_centrality))
            })
            .collect();

        Ok(scores_vec.into_iter().collect())
    }

    fn min_graph_size_for_parallel(&self) -> usize {
        // Harmonic benefits from parallelism for graphs with > 100 nodes
        100
    }
}

/// Betweenness centrality algorithm (Brandes)
///
/// Measures how often a node lies on shortest paths between other node pairs.
/// Uses Brandes' O(VE) algorithm with BFS-based dependency accumulation.
///
/// Formula:
/// B(v) = Σ σ_st(v) / σ_st  for all s ≠ v ≠ t
///
/// Where:
/// - σ_st = number of shortest paths from s to t
/// - σ_st(v) = number of shortest paths from s to t that pass through v
pub struct BetweennessCentrality {
    engine: Arc<OrionGraphEngine>,
    normalized: bool,
}

impl BetweennessCentrality {
    /// Create a new betweenness centrality calculator for the given graph engine.
    pub fn new(engine: Arc<OrionGraphEngine>, normalized: bool) -> Self {
        Self { engine, normalized }
    }
}

impl GraphAlgorithm for BetweennessCentrality {
    type Input = NoInput;
    type Output = CentralityScores;

    fn execute(&self, _input: NoInput) -> Result<CentralityScores, ProximaDBError> {
        // The CSR read guard is held for the whole pass: the Brandes BFS below
        // reads it once per source.
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };

        // Brandes algorithm: accumulate dependency from each source
        let mut betweenness = vec![0.0f64; node_count];

        for s in 0..node_count {
            let delta = brandes_source(&csr_out, s, node_count);
            for (w, &d) in delta.iter().enumerate() {
                if w != s {
                    betweenness[w] += d;
                }
            }
        }

        Ok(betweenness_scores(
            betweenness,
            &index_to_node,
            self.normalized,
        ))
    }

    fn estimated_complexity(&self) -> AlgorithmComplexity {
        // Brandes: O(VE) for unweighted graphs
        AlgorithmComplexity::QuadraticVertices
    }

    fn name(&self) -> &'static str {
        "BetweennessCentrality"
    }
}

impl ParallelAlgorithm for BetweennessCentrality {
    fn execute_parallel(
        &self,
        _input: NoInput,
        _thread_pool: &rayon::ThreadPool,
    ) -> Result<CentralityScores, ProximaDBError> {
        let Some((node_count, index_to_node, csr_out)) = centrality_shape(&self.engine)? else {
            return Ok(HashMap::new());
        };
        drop(csr_out);

        // Parallel BFS from each source, then reduce
        let partial: Vec<Vec<f64>> = (0..node_count)
            .into_par_iter()
            .map(|s| {
                let csr = self
                    .engine
                    .csr_outgoing
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                let n = csr.node_count();
                let delta = brandes_source(&csr, s, n);
                let mut local_b = vec![0.0f64; n];
                for (w, &d) in delta.iter().enumerate() {
                    if w != s {
                        local_b[w] += d;
                    }
                }
                local_b
            })
            .collect();

        // Reduce partial results
        let mut betweenness = vec![0.0f64; node_count];
        for p in &partial {
            for (i, v) in p.iter().enumerate() {
                betweenness[i] += v;
            }
        }

        Ok(betweenness_scores(
            betweenness,
            &index_to_node,
            self.normalized,
        ))
    }

    fn min_graph_size_for_parallel(&self) -> usize {
        50
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closeness_centrality_empty_graph() {
        let engine = Arc::new(OrionGraphEngine::new());
        let closeness = ClosenessCentrality::new(engine, true);

        let scores = closeness.execute(NoInput).unwrap();

        // Empty graph should return empty scores
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_harmonic_centrality_empty_graph() {
        let engine = Arc::new(OrionGraphEngine::new());
        let harmonic = HarmonicCentrality::new(engine, true);

        let scores = harmonic.execute(NoInput).unwrap();

        // Empty graph should return empty scores
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_closeness_algorithm_complexity() {
        let engine = Arc::new(OrionGraphEngine::new());
        let closeness = ClosenessCentrality::new(engine, true);

        assert_eq!(
            closeness.estimated_complexity(),
            AlgorithmComplexity::QuadraticVertices
        );
        assert_eq!(closeness.name(), "ClosenessCentrality");
    }

    #[test]
    fn test_harmonic_algorithm_complexity() {
        let engine = Arc::new(OrionGraphEngine::new());
        let harmonic = HarmonicCentrality::new(engine, true);

        assert_eq!(
            harmonic.estimated_complexity(),
            AlgorithmComplexity::QuadraticVertices
        );
        assert_eq!(harmonic.name(), "HarmonicCentrality");
    }

    #[test]
    fn test_parallel_execution_threshold() {
        let engine = Arc::new(OrionGraphEngine::new());
        let closeness = ClosenessCentrality::new(engine.clone(), true);
        let harmonic = HarmonicCentrality::new(engine.clone(), true);
        let betweenness = BetweennessCentrality::new(engine, true);

        assert_eq!(closeness.min_graph_size_for_parallel(), 100);
        assert_eq!(harmonic.min_graph_size_for_parallel(), 100);
        assert_eq!(betweenness.min_graph_size_for_parallel(), 50);
    }

    #[test]
    fn test_betweenness_centrality_empty() {
        let engine = Arc::new(OrionGraphEngine::new());
        let bc = BetweennessCentrality::new(engine, true);
        let scores = bc.execute(NoInput).unwrap();
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_betweenness_centrality_name() {
        let engine = Arc::new(OrionGraphEngine::new());
        let bc = BetweennessCentrality::new(engine, true);
        assert_eq!(bc.name(), "BetweennessCentrality");
        assert_eq!(
            bc.estimated_complexity(),
            AlgorithmComplexity::QuadraticVertices
        );
    }
}
