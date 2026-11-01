//! # ProximaDB Graph Modality
//!
//! This crate contains native graph database capabilities including:
//!
//! - **Core types** - Nodes, edges, properties, graph schemas
//! - **Traversal** - Graph traversal algorithms (BFS, DFS, shortest path)
//! - **Query** - Graph query execution and Cypher/GQL support
//! - **Storage** - ORION graph runtime and canonical graph projections
//! - **RAG** - Graph RAG (Retrieval-Augmented Generation) capabilities
//!
//! ## Architecture
//!
//! The graph modality provides:
//! - CSR (Compressed Sparse Row) format for efficient adjacency
//! - Zero-copy Arc-based node/edge references
//! - Multi-edge support between nodes
//! - Property graph model with typed attributes
//!
//! ## Dependencies
//!
//! - `proximadb-kernel` - Core error types and foundational contracts
//! - `proximadb-records` - Canonical `ProximaRecord` envelope
//! - `proximadb-data-model` - Canonical `ProximaValue` rich type system
//! - `proximadb-graph-query` - Graph query contracts

pub mod algorithms;
pub mod compaction;
pub mod core;
pub mod cypher_functions;
pub mod cypher_parser;
pub mod graphwalk;
pub mod parser;
pub mod projection;
pub mod query;
pub mod query_ast;
pub mod record;
pub mod storage;
pub mod topology_codec;
pub mod traversal;

// TD-GRAPH-4 §3: the Cypher corpus pin (test-only — see the module docs).
#[cfg(test)]
mod cypher_corpus_pin;

// Re-export core types
pub use core::{DirectedGraph, Edge, EdgeId, Graph, GraphId, Node, NodeId, UndirectedGraph};
pub use graphwalk::{GraphWalkRequest, GraphWalkResult, GraphWalkStep, GraphWalkTool};

pub use traversal::{
    BreadthFirst, DepthFirst, ShortestPath, Traversal, TraversalOrder, TraversalResult,
};

pub use algorithms::{
    AlgorithmComplexity, AllPairsShortestPaths, ApproximateAlgorithm, CentralityScores,
    CommunityAssignment, GraphAlgorithm, GraphChange, IncrementalAlgorithm, NoInput, NodePairInput,
    ParallelAlgorithm, ShortestPath as AlgoShortestPath, SingleNodeInput, SubgraphInput,
};
pub use compaction::{CompactionConfig, CompactionManager, CompactionStats};
pub use cypher_functions::{CypherFunction, CypherFunctionRegistry};
pub use parser::{FilterExpression, GraphQuery, ParseResult, Pattern, Projection, parse_query};
pub use query_ast::{
    CompiledPattern, CreateClause, CreateEdgeSpec, CreateNodeSpec, CypherClause, CypherParser,
    CypherStatement, DeleteClause, EdgeDirection as QueryEdgeDirection, FoundPath, GraphNode,
    MatchClause, MatchResult, MergeClause, PatternEdge, PatternNode, PropertyConstraint,
    PropertyProjection, ReadingClause, RemoveClause, RemoveItem, ReturnClause, SetClause, SetItem,
    UpdatingClause, WhereClause, WithClause,
};
pub use storage::{GraphStorage, MemoryGraphStorage};

/// Parser trait for graph query languages.
///
/// This mirrors the root parser trait so modality-owned parsers can compile and
/// be reused without depending upward on the root query module.
pub trait QueryParser {
    /// The type of AST produced by this parser.
    type Output;

    /// Parse a query string into an AST.
    fn parse(&self, input: &str) -> anyhow::Result<Self::Output>;
}

/// Graph direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Graph statistics
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f32,
    pub diameter: Option<usize>,
    pub is_connected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_module_imports() {
        let _id = GraphId::new();
        let _node = Node::new("test");
        let _edge = Edge::new(NodeId::new(), NodeId::new(), "connected");
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = DirectedGraph::new();
        let n1 = graph.add_node("A");
        let n2 = graph.add_node("B");
        graph.add_edge(n1, n2, "knows");

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_traversal() {
        let mut graph = DirectedGraph::new();
        let n1 = graph.add_node("A");
        let n2 = graph.add_node("B");
        let n3 = graph.add_node("C");
        graph.add_edge(n1, n2, "knows");
        graph.add_edge(n2, n3, "knows");

        let result = BreadthFirst::from(&graph).start(n1).execute();
        assert_eq!(result.visited_nodes.len(), 3);
    }
}
