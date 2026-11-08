//! Cypher/GQL query AST types for the graph modality.
//!
//! Merged from root-crate src/graph/query/{ast,cypher_ast,cypher_parser}.rs to
//! break the circular super:: reference chain while keeping all types visible
//! under a single module path.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Primitive edge direction ─────────────────────────────────────────────────

/// Edge direction in pattern matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
    Bidirectional,
}

// ── WHERE clause types ───────────────────────────────────────────────────────

/// Property constraint for WHERE clauses — the SEAM's canonical enum
/// (TD-GRAPH-4 step 2: the twin is gone; the seam's is a superset adding
/// NotIn and the GreaterThanOrEqual/LessThanOrEqual alias variants).
pub use proximadb_graph_query::ast::PropertyConstraint;

/// Property projection for RETURN clauses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyProjection {
    Variable(String),
    Property { variable: String, property: String },
    Count,
    Sum { variable: String, property: String },
    Avg { variable: String, property: String },
    Min { variable: String, property: String },
    Max { variable: String, property: String },
}

/// WHERE clause conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhereClause {
    Property {
        variable: String,
        property: String,
        constraint: PropertyConstraint,
    },
    And(Box<WhereClause>, Box<WhereClause>),
    Or(Box<WhereClause>, Box<WhereClause>),
    Not(Box<WhereClause>),
    Exists(String),
}

// ── Pattern types ────────────────────────────────────────────────────────────

/// Pattern node in a graph pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    pub id: String,
    pub variable: String,
    pub labels: Vec<String>,
    pub label: Option<String>,
    pub properties: HashMap<String, serde_json::Value>,
    pub optional: bool,
}

/// Pattern edge in a graph pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    pub variable: Option<String>,
    pub from_variable: String,
    pub to_variable: String,
    pub from: String,
    pub to: String,
    pub label: Option<String>,
    pub edge_types: Vec<String>,
    pub direction: EdgeDirection,
    pub properties: HashMap<String, serde_json::Value>,
    pub optional: bool,
}

/// Compiled graph pattern (result of parsing a MATCH pattern)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPattern {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
    pub paths: Vec<String>,
    pub where_clause: Option<WhereClause>,
    pub where_clauses: Vec<WhereClause>,
    pub with_clauses: Vec<String>,
}

/// Graph node returned in a match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
}

/// Path found during traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundPath {
    pub nodes: Vec<String>,
    pub edges: Vec<String>,
}

/// Result of pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub bindings: HashMap<String, GraphNode>,
    pub paths: Vec<FoundPath>,
}

/// Result type for query operations
pub type QueryResult<T> = Result<T, String>;

// ── DML clause types ─────────────────────────────────────────────────────────

/// CREATE node specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNodeSpec {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// CREATE edge specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEdgeSpec {
    pub variable: Option<String>,
    pub from_variable: Option<String>,
    pub to_variable: Option<String>,
    pub edge_type: Option<String>,
    pub properties: HashMap<String, serde_json::Value>,
    pub direction: EdgeDirection,
}

/// CREATE clause containing nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateClause {
    pub nodes: Vec<CreateNodeSpec>,
    pub edges: Vec<CreateEdgeSpec>,
}

/// DELETE clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteClause {
    pub variables: Vec<String>,
    pub detach: bool,
}

/// SET clause item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SetItem {
    Property {
        variable: String,
        property: String,
        value: serde_json::Value,
    },
    AddLabel {
        variable: String,
        label: String,
    },
    MergeProperties {
        variable: String,
        properties: HashMap<String, serde_json::Value>,
    },
    AllProperties {
        variable: String,
        properties: HashMap<String, serde_json::Value>,
    },
}

/// SET clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

/// REMOVE clause item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoveItem {
    Property { variable: String, property: String },
    Label { variable: String, label: String },
}

/// REMOVE clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveClause {
    pub items: Vec<RemoveItem>,
}

/// MERGE clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeClause {
    pub pattern: CompiledPattern,
    pub on_create: Option<SetClause>,
    pub on_match: Option<SetClause>,
}

/// WITH clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithClause {
    pub projections: Vec<PropertyProjection>,
    pub distinct: bool,
    pub where_clause: Option<WhereClause>,
    pub order_by: Vec<String>,
    pub limit: Option<usize>,
    pub skip: Option<usize>,
}

// ── Cypher statement types ───────────────────────────────────────────────────

/// MATCH clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchClause {
    pub patterns: Vec<CompiledPattern>,
}

/// RETURN clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnClause {
    pub items: Vec<String>,
    pub variables: Option<Vec<String>>,
    pub projections: Option<Vec<PropertyProjection>>,
    pub distinct: bool,
    pub order_by: Vec<String>,
    pub limit: Option<usize>,
    pub skip: Option<usize>,
}

/// Reading clause (MATCH, OPTIONAL MATCH, WITH)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadingClause {
    Match {
        pattern: CompiledPattern,
        optional: bool,
    },
    With(WithClause),
}

/// Updating clause as defined by Cypher write semantics.
///
/// `Merge(MergeClause)` is the largest variant (combines a CREATE
/// clause's pattern bag with an ON CREATE / ON MATCH branch); Boxing
/// would touch every match site in the executor. Intentionally
/// deferred per `clippy::large_enum_variant`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdatingClause {
    Create(CreateClause),
    Delete(DeleteClause),
    Set(SetClause),
    Remove(RemoveClause),
    Merge(MergeClause),
}

/// Cypher clause (MATCH, RETURN, WHERE, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CypherClause {
    Match(MatchClause),
    Return(ReturnClause),
}

/// Cypher statement AST node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CypherStatement {
    pub clauses: Vec<CypherClause>,
    pub reading_clauses: Vec<ReadingClause>,
    pub updating_clauses: Vec<UpdatingClause>,
    pub with_clauses: Vec<WithClause>,
    pub union_clauses: Vec<CypherStatement>,
    pub return_spec: Option<ReturnClause>,
}

impl CypherStatement {
    pub fn is_read_only(&self) -> bool {
        self.updating_clauses.is_empty()
            && self.union_clauses.iter().all(CypherStatement::is_read_only)
    }

    pub fn has_create(&self) -> bool {
        self.updating_clauses
            .iter()
            .any(|clause| matches!(clause, UpdatingClause::Create(_)))
            || self.union_clauses.iter().any(CypherStatement::has_create)
    }

    pub fn has_delete(&self) -> bool {
        self.updating_clauses
            .iter()
            .any(|clause| matches!(clause, UpdatingClause::Delete(_)))
            || self.union_clauses.iter().any(CypherStatement::has_delete)
    }

    pub fn has_merge(&self) -> bool {
        self.updating_clauses
            .iter()
            .any(|clause| matches!(clause, UpdatingClause::Merge(_)))
            || self.union_clauses.iter().any(CypherStatement::has_merge)
    }

    pub fn has_set(&self) -> bool {
        self.updating_clauses
            .iter()
            .any(|clause| matches!(clause, UpdatingClause::Set(_)))
            || self.union_clauses.iter().any(CypherStatement::has_set)
    }

    pub fn has_remove(&self) -> bool {
        self.updating_clauses
            .iter()
            .any(|clause| matches!(clause, UpdatingClause::Remove(_)))
            || self.union_clauses.iter().any(CypherStatement::has_remove)
    }
}

// ── Cypher parser ────────────────────────────────────────────────────────────

/// Cypher query parser (stub — full recursive-descent parser lives in query layer)
#[derive(Debug, Clone)]
pub struct CypherParser;

impl CypherParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, _query: &str) -> Result<CypherStatement, String> {
        Ok(CypherStatement::default())
    }
}

impl Default for CypherParser {
    fn default() -> Self {
        Self::new()
    }
}
