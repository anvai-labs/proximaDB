//! Graph pattern query parser STUBS (renamed 2026-11-07, R4 wave).
//!
//! These are legacy compatibility stubs — `parse_query` returns an
//! ALWAYS-EMPTY query. They were originally named `GraphQuery`/
//! `ParseResult`, shadowing the two functional same-named types in this
//! crate (the `query::GraphQuery` traversal IR and the
//! `cypher_parser::GraphQuery` executable envelope) — and the lib re-export
//! promoted the STUB as the crate's public face while zero consumers used
//! it (verified by census). Renamed `Stub*` so the shadowing is visible;
//! deprecated aliases below keep source compatibility.

/// Parse result
#[derive(Debug, Clone)]
pub struct StubParseResult {
    pub query: StubGraphQuery,
}

/// Graph query (parsed representation)
#[derive(Debug, Clone)]
pub struct StubGraphQuery {
    pub patterns: Vec<Pattern>,
    pub filters: Vec<FilterExpression>,
    pub projections: Vec<Projection>,
}

/// Pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    pub subject: String,
    pub predicate: Option<String>,
    pub object: String,
}

/// Filter expression
#[derive(Debug, Clone)]
pub enum FilterExpression {
    Property {
        variable: String,
        property: String,
        operator: String,
        value: String,
    },
}

/// Projection
#[derive(Debug, Clone)]
pub struct Projection {
    pub variable: String,
    pub property: Option<String>,
    pub alias: Option<String>,
}

/// Parse a graph query string
pub fn parse_query(_input: &str) -> Result<StubParseResult, String> {
    Ok(StubParseResult {
        query: StubGraphQuery {
            patterns: vec![],
            filters: vec![],
            projections: vec![],
        },
    })
}

#[deprecated(
    since = "0.5.1",
    note = "stub type renamed StubGraphQuery; the crate's public GraphQuery re-export pointed at this STUB while nothing consumed it — use the canonical parsers instead"
)]
pub type GraphQuery = StubGraphQuery;

#[deprecated(since = "0.5.1", note = "stub type renamed StubParseResult")]
pub type ParseResult = StubParseResult;
