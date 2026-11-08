//! Compatibility shim for the unified Cypher parser entry points.
//!
//! The canonical parser implementation is being migrated to the `proximadb-graph`
//! workspace crate. This module preserves the historical root import surface.

use super::ast::QueryResult;
use super::cypher_parser::{CypherParser, CypherStatement};

/// Parse a Cypher query string
pub fn parse_cypher(query: &str) -> QueryResult<CypherStatement> {
    let parser = CypherParser::new();
    parser.parse(query)
}

/// Parse a Cypher query string with context
pub fn parse_cypher_with_context(
    query: &str,
    _context: &GraphParserQueryContext,
) -> QueryResult<CypherStatement> {
    parse_cypher(query)
}

/// Backwards-compat alias for [`GraphParserQueryContext`].
pub type QueryContext = GraphParserQueryContext;

/// Query context for parsing
#[derive(Debug, Clone, Default)]
pub struct GraphParserQueryContext {
    // TODO: Add context fields
}
