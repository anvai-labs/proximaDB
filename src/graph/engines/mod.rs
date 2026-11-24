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

//! # Graph Storage Engines
//!
//! ProximaDB uses ORION as the graph runtime. ORION consumes canonical
//! `ProximaRecord` node/edge records and maintains rebuildable topology
//! projections rather than owning a separate durable graph record model.
//! Distributed placement, coordination, fanout/fanin, and storage tiering are
//! delegated to the relational/storage substrate and surfaced to ORION as
//! cataloged execution and projection policy.
//!
//! ## Embedding Storage Modes
//!
//! Graph engines support three embedding storage modes:
//!
//! - **None** (DEFAULT): No embeddings stored - pure graph, best performance
//! - **Cold**: Embeddings in vector engine (SST/HELIX) - SKS with large graphs
//! - **Memory**: Embeddings cached in memory - SKS-heavy workloads, consumer override
//!
//! CSR (Compressed Sparse Row) format NEVER contains embedding data and is not
//! the durable authority for node or edge facts. Embeddings are optionally
//! stored in separate vector storage engines.

/// Generic graph traversal algorithms (BFS, DFS, Dijkstra, A*) usable across all engines.
pub mod generic_traversal;
/// ORION in-memory CSR projection engine for real-time traversal at 1M+ edges/sec.
pub mod orion;
#[cfg(test)]
mod orion_recovery_tests;

use proximadb_kernel::error::ProximaDBError;
type Result<T> = std::result::Result<T, ProximaDBError>;
use crate::graph::{Edge, EdgeId, Node, NodeId};
use std::sync::Arc;

/// Embedding storage mode for graph engines
///
/// Controls how node embeddings are stored relative to the graph structure.
/// CSR (Compressed Sparse Row) format NEVER contains embedding data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingMode {
    /// No embeddings stored - pure graph workloads (DEFAULT, best performance)
    ///
    /// Use this for:
    /// - Pure graph traversal/analytics
    /// - Pattern matching
    /// - Path finding
    /// - When embeddings are not needed
    #[default]
    None,

    /// Embeddings stored in cold tier vector engine (SST/HELIX)
    ///
    /// Use this for:
    /// - SKS (Semantic Knowledge Search) with large graphs
    /// - When graph + embeddings don't fit in memory
    /// - Cost-optimized production deployments
    Cold,

    /// Embeddings cached in memory (consumer override)
    ///
    /// Use this for:
    /// - SKS-heavy workloads with smaller graphs
    /// - When latency is critical
    /// - Development/testing
    Memory,
}

impl EmbeddingMode {
    /// Parse embedding mode from config string
    pub fn parse_from_config(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cold" => EmbeddingMode::Cold,
            "memory" => EmbeddingMode::Memory,
            _ => EmbeddingMode::None, // Default for "none" or invalid
        }
    }

    /// Check if embeddings are stored at all
    pub fn stores_embeddings(&self) -> bool {
        !matches!(self, EmbeddingMode::None)
    }

    /// Check if embeddings are in cold tier
    pub fn is_cold(&self) -> bool {
        matches!(self, EmbeddingMode::Cold)
    }

    /// Check if embeddings are in memory
    pub fn is_memory(&self) -> bool {
        matches!(self, EmbeddingMode::Memory)
    }
}

#[cfg(test)]
mod embedding_mode_tests {
    use super::EmbeddingMode;

    #[test]
    fn test_embedding_mode_default() {
        // Default mode should be None (pure graph, best performance)
        let mode = EmbeddingMode::default();
        assert_eq!(mode, EmbeddingMode::None);
        assert!(!mode.stores_embeddings());
    }

    #[test]
    fn test_embedding_mode_parse_from_config() {
        // Test parsing from config strings
        assert_eq!(
            EmbeddingMode::parse_from_config("none"),
            EmbeddingMode::None
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("None"),
            EmbeddingMode::None
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("NONE"),
            EmbeddingMode::None
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("cold"),
            EmbeddingMode::Cold
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("Cold"),
            EmbeddingMode::Cold
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("COLD"),
            EmbeddingMode::Cold
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("memory"),
            EmbeddingMode::Memory
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("Memory"),
            EmbeddingMode::Memory
        );
        assert_eq!(
            EmbeddingMode::parse_from_config("MEMORY"),
            EmbeddingMode::Memory
        );

        // Invalid strings should default to None
        assert_eq!(
            EmbeddingMode::parse_from_config("invalid"),
            EmbeddingMode::None
        );
        assert_eq!(EmbeddingMode::parse_from_config(""), EmbeddingMode::None);
    }

    #[test]
    fn test_embedding_mode_stores_embeddings() {
        assert!(!EmbeddingMode::None.stores_embeddings());
        assert!(EmbeddingMode::Cold.stores_embeddings());
        assert!(EmbeddingMode::Memory.stores_embeddings());
    }

    #[test]
    fn test_embedding_mode_is_cold() {
        assert!(!EmbeddingMode::None.is_cold());
        assert!(EmbeddingMode::Cold.is_cold());
        assert!(!EmbeddingMode::Memory.is_cold());
    }

    #[test]
    fn test_embedding_mode_is_memory() {
        assert!(!EmbeddingMode::None.is_memory());
        assert!(!EmbeddingMode::Cold.is_memory());
        assert!(EmbeddingMode::Memory.is_memory());
    }
}

/// Backwards-compat alias for [`GraphEngineStats`].
pub type EngineStats = GraphEngineStats;

pub use proximadb_graph_engine_traits::{
    GraphEngine, GraphEngineStats, GraphExportFormat, MemoryUsage, PersistenceConfig,
};

/// Engine type enumeration for factory creation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphEngineType {
    /// ORION: In-memory CSR format engine (Production-ready).
    Orion,
}

/// Enum wrapper for different graph engine implementations
#[derive(Debug)]
pub enum GraphEngineImpl {
    /// ORION in-memory CSR engine instance.
    Orion(orion::OrionGraphEngine),
}

impl GraphEngineImpl {
    /// True when an edge with this exact `(from, to, type)` composite already
    /// exists in the engine's memory pool.
    ///
    /// Batch admission must probe THIS index — the engine's — not a
    /// service-owned pool. `GraphOperationsService` carries its own
    /// `GraphMemoryPool` for constraint/property indexes, but nothing ever
    /// writes edges into it, so probing it for composite uniqueness always
    /// answered "absent" and cross-batch duplicates were silently admitted.
    pub fn has_composite_edge(&self, key: &(NodeId, NodeId, String)) -> bool {
        match self {
            GraphEngineImpl::Orion(engine) => {
                engine.memory_pool.edge_composite_index.contains_key(key)
            }
        }
    }

    /// Create a new graph engine based on type and configuration
    pub fn new(engine_type: GraphEngineType, _config: GraphEngineConfig) -> Result<Self> {
        match engine_type {
            GraphEngineType::Orion => {
                let engine = orion::OrionGraphEngine::new();
                Ok(GraphEngineImpl::Orion(engine))
            }
        }
    }
}

// Implement GraphEngine for GraphEngineImpl by delegating to the underlying engine
#[async_trait::async_trait]
impl GraphEngine for GraphEngineImpl {
    async fn insert_node(&self, node: Node) -> Result<Arc<Node>> {
        tracing::debug!("GraphEngineImpl::insert_node called for node: {}", node.id);
        match self {
            GraphEngineImpl::Orion(engine) => {
                tracing::debug!("Delegating to Orion engine");
                engine.insert_node(node).await
            }
        }
    }

    fn get_node(&self, id: &NodeId) -> Result<Option<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_node(id),
        }
    }

    async fn update_node(&self, node: Node) -> Result<Arc<Node>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.update_node(node).await,
        }
    }

    async fn delete_node(&self, id: &NodeId) -> Result<Option<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => GraphEngine::delete_node(engine, id).await,
        }
    }

    async fn insert_edge(&self, edge: Edge) -> Result<Arc<Edge>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.insert_edge(edge).await,
        }
    }

    fn get_edge(&self, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_edge(id),
        }
    }

    async fn update_edge(&self, edge: Edge) -> Result<Arc<Edge>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.update_edge(edge).await,
        }
    }

    async fn delete_edge(&self, id: &EdgeId) -> Result<Option<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => GraphEngine::delete_edge(engine, id).await,
        }
    }

    fn get_outgoing_edges(
        &self,
        node_id: &NodeId,
        edge_type: Option<&str>,
    ) -> Result<Vec<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_outgoing_edges(node_id, edge_type),
        }
    }

    fn get_incoming_edges(
        &self,
        node_id: &NodeId,
        edge_type: Option<&str>,
    ) -> Result<Vec<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_incoming_edges(node_id, edge_type),
        }
    }

    fn get_neighbors(&self, node_id: &NodeId, edge_type: Option<&str>) -> Result<Vec<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_neighbors(node_id, edge_type),
        }
    }

    fn get_nodes_by_label(&self, label: &str) -> Result<Vec<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_nodes_by_label(label),
        }
    }

    fn node_count(&self) -> Result<usize> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.node_count(),
        }
    }

    fn edge_count(&self) -> Result<usize> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.edge_count(),
        }
    }

    fn get_all_nodes(&self) -> Result<Vec<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_all_nodes(),
        }
    }

    fn get_all_edges(&self) -> Result<Vec<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_all_edges(),
        }
    }

    // Delegate other methods with default implementations
    fn get_engine_stats(&self) -> Result<GraphEngineStats> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_engine_stats(),
        }
    }

    fn get_memory_usage(&self) -> Result<MemoryUsage> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.get_memory_usage(),
        }
    }

    // ===== Bulk Operations - Critical for Performance =====
    // These MUST be delegated to underlying engines to avoid O(n) per-edge overhead

    async fn bulk_insert_nodes(&self, nodes: Vec<Node>) -> Result<Vec<Arc<Node>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.bulk_insert_nodes(nodes).await,
        }
    }

    async fn bulk_insert_edges(&self, edges: Vec<Edge>) -> Result<Vec<Arc<Edge>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.bulk_insert_edges(edges).await,
        }
    }

    async fn bulk_delete_nodes(&self, node_ids: Vec<NodeId>) -> Result<Vec<Option<Arc<Node>>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.bulk_delete_nodes(node_ids).await,
        }
    }

    async fn bulk_delete_edges(&self, edge_ids: Vec<EdgeId>) -> Result<Vec<Option<Arc<Edge>>>> {
        match self {
            GraphEngineImpl::Orion(engine) => engine.bulk_delete_edges(edge_ids).await,
        }
    }
}

/// Graph engine factory for creating different engine types
pub struct GraphEngineFactory;

impl GraphEngineFactory {
    /// Create a graph engine based on type and configuration
    pub fn create_engine(
        engine_type: GraphEngineType,
        config: GraphEngineConfig,
    ) -> Result<GraphEngineImpl> {
        GraphEngineImpl::new(engine_type, config)
    }

    /// Get available engine types
    pub fn available_engines() -> Vec<GraphEngineType> {
        vec![GraphEngineType::Orion]
    }

    /// Get engine type from string
    pub fn engine_type_from_string(name: &str) -> Option<GraphEngineType> {
        match name.to_lowercase().as_str() {
            "orion" => Some(GraphEngineType::Orion),
            _ => None,
        }
    }
}

/// Configuration for graph engine creation
#[derive(Debug, Clone, Default)]
pub struct GraphEngineConfig;

/// Engine capabilities description
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Engine name, currently "ORION".
    pub name: String,
    /// Human-readable description of the engine.
    pub description: String,
    /// List of supported features (e.g. "CSR storage", "DashMap concurrent access").
    pub features: Vec<String>,
    /// Recommended use cases for this engine.
    pub use_cases: Vec<String>,
    /// Performance characteristics (e.g. "1M+ edges/second traversal").
    pub performance_characteristics: Vec<String>,
}

impl GraphEngineFactory {
    /// Get capabilities for each engine type
    pub fn get_engine_capabilities(engine_type: GraphEngineType) -> EngineCapabilities {
        match engine_type {
            GraphEngineType::Orion => EngineCapabilities {
                name: "ORION".to_string(),
                description: "Canonical graph runtime with rebuildable CSR projections".to_string(),
                features: vec![
                    "CSR (Compressed Sparse Row) storage".to_string(),
                    "Arc-based zero-copy sharing".to_string(),
                    "DashMap concurrent access".to_string(),
                    "Label and property indexes".to_string(),
                    "Graph-aware hints for relational distributed planning".to_string(),
                    "Projection tiering policy delegated to catalog/storage layers".to_string(),
                ],
                use_cases: vec![
                    "Real-time graph traversal".to_string(),
                    "Interactive graph queries".to_string(),
                    "Graph projections over canonical records".to_string(),
                ],
                performance_characteristics: vec![
                    "1M+ edges/second traversal".to_string(),
                    "<1μs node lookup".to_string(),
                    "<100 bytes/node memory overhead".to_string(),
                ],
            },
        }
    }
}
