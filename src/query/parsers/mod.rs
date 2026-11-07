//! # Query Parsers Module
//!
//! This module provides parsers for various query languages used in ProximaDB.
//!
//! ## Available Parsers
//!
//! - **MongoDB Parser**: Parse MongoDB-style queries, projections, and aggregation pipelines
//! - **Cypher Parser**: Parse Cypher Query Language for graph database operations
//!
//! ## Design Principles
//!
//! The parsers follow SOLID principles:
//! - **Single Responsibility**: Each parser handles one query language
//! - **Open/Closed**: Parser trait allows extension without modification
//! - **Liskov Substitution**: All parsers implement the same trait interface
//! - **Interface Segregation**: Separate traits for parsing and AST conversion
//! - **Dependency Inversion**: High-level modules depend on abstractions

pub mod cypher;
pub mod mongodb;

// Re-export MongoDB types
pub use mongodb::{
    // AST types
    MongoDBExpression,
    MongoDBParseResult,
    // Parser types
    MongoDBParser,
    MongoDBPipelineStage,
    MongoDBProjection,
    MongoDBQuery,
    // Visitor pattern
    MongoDBVisitor,
    // Conversion
    ToDocumentFilter,
};

// Re-export Cypher types
pub use cypher::{
    // Function types
    CypherFunction,
    CypherLexer,
    // Parser types
    CypherParser,
    // Visitor pattern
    CypherVisitor,
    // Statement opcode (renamed ExecutableGraphQueryType 2026-11-07; the
    // GraphQueryType alias re-exports the deprecated upstream alias)
    ExecutableGraphQueryType,
    // GraphQuery conversion (renamed ExecutableGraphQuery upstream
    // 2026-11-07; aliased back at the cypher.rs re-export site)
    GraphQuery,
    GraphQueryType,
    LocatedToken,
    QueryValidator as CypherQueryValidator,
    // Token types
    Token as CypherToken,
    cypher_to_graph_query,
};

use anyhow::Result;

/// Parser trait for extensibility
///
/// All query language parsers implement this trait to provide
/// a consistent interface for parsing different query formats.
pub trait QueryParser {
    /// The type of AST produced by this parser
    type Output;

    /// Parse a query string into an AST
    fn parse(&self, input: &str) -> Result<Self::Output>;
}

/// AST visitor trait for traversal
///
/// Implements the visitor pattern for AST traversal,
/// allowing different operations on the same AST structure.
pub trait AstVisitor<T> {
    /// The output type of the visitor
    type Output;

    /// Visit an AST node
    fn visit(&mut self, node: &T) -> Self::Output;
}

/// Trait for converting parsed AST to DocumentFilter
pub trait ToFilter {
    /// Convert this AST node to a DocumentFilter
    fn to_filter(&self) -> Result<crate::proto::proximadb_v1::DocumentFilter>;
}

/// Trait for converting parsed AST to GraphQuery
pub trait ToGraphQuery {
    /// Convert this AST to an executable GraphQuery
    fn to_graph_query(&self, graph_id: &str) -> Result<cypher::GraphQuery>;
}
