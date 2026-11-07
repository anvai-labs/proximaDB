//! # Graph Query Module
//!
//! Graph query execution and planning.

use super::core::{Graph, NodeId};
use serde::{Deserialize, Serialize};

/// Graph query expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalQuery {
    /// Query type
    pub query_type: GraphQueryType,
    /// Starting node(s) for traversal
    pub start_nodes: Vec<QueryNode>,
    /// Traversal pattern
    pub pattern: TraversalPattern,
    /// Filters to apply
    pub filters: Vec<GraphFilter>,
    /// Projection (what to return)
    pub projection: QueryProjection,
}

impl TraversalQuery {
    pub fn new(query_type: GraphQueryType) -> Self {
        Self {
            query_type,
            start_nodes: Vec::new(),
            pattern: TraversalPattern::default(),
            filters: Vec::new(),
            projection: QueryProjection::default(),
        }
    }

    pub fn with_start_node(mut self, node: QueryNode) -> Self {
        self.start_nodes.push(node);
        self
    }

    pub fn with_pattern(mut self, pattern: TraversalPattern) -> Self {
        self.pattern = pattern;
        self
    }

    pub fn with_filter(mut self, filter: GraphFilter) -> Self {
        self.filters.push(filter);
        self
    }
}

impl Default for TraversalQuery {
    fn default() -> Self {
        Self::new(GraphQueryType::Traversal)
    }
}

/// Graph query types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphQueryType {
    /// Basic graph traversal
    Traversal,
    /// Shortest path query
    ShortestPath,
    /// Pattern matching
    PatternMatch,
    /// Aggregation (count, sum, etc.)
    Aggregation,
}

/// Query node specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryNode {
    pub id: Option<NodeId>,
    pub label: Option<String>,
    pub properties: Vec<PropertyFilter>,
}

impl QueryNode {
    pub fn by_id(id: NodeId) -> Self {
        Self {
            id: Some(id),
            label: None,
            properties: Vec::new(),
        }
    }

    pub fn by_label(label: impl Into<String>) -> Self {
        Self {
            id: None,
            label: Some(label.into()),
            properties: Vec::new(),
        }
    }
}

/// Traversal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPattern {
    pub direction: EdgeDirection,
    pub min_depth: usize,
    pub max_depth: usize,
    pub edge_labels: Vec<String>,
}

impl Default for TraversalPattern {
    fn default() -> Self {
        Self {
            direction: EdgeDirection::Outgoing,
            min_depth: 1,
            max_depth: 1,
            edge_labels: Vec::new(),
        }
    }
}

/// Edge direction for traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Graph filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphFilter {
    NodeLabel(String),
    EdgeLabel(String),
    Property {
        key: String,
        value: FilterValue,
    },
    Degree {
        min: Option<usize>,
        max: Option<usize>,
    },
}

/// Filter value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
}

/// Query projection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum QueryProjection {
    #[default]
    Nodes,
    Edges,
    Paths,
    Count,
    Properties(Vec<String>),
}

/// Property filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFilter {
    pub key: String,
    pub value: FilterValue,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResult {
    Nodes(Vec<NodeData>),
    Paths(Vec<Vec<NodeId>>),
    Count(usize),
}

/// Node data for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: NodeId,
    pub label: String,
    pub properties: Vec<(String, serde_json::Value)>,
}

/// Graph query executor
pub struct QueryExecutor;

impl QueryExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn execute(
        &self,
        _graph: &dyn Graph,
        _query: &TraversalQuery,
    ) -> Result<QueryResult, String> {
        // Placeholder implementation
        // In production, this would parse and execute the query
        Ok(QueryResult::Count(0))
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_construction() {
        let query = TraversalQuery::new(GraphQueryType::Traversal)
            .with_start_node(QueryNode::by_label("Person"))
            .with_pattern(TraversalPattern {
                direction: EdgeDirection::Outgoing,
                min_depth: 1,
                max_depth: 3,
                edge_labels: vec!["knows".to_string()],
            });

        assert_eq!(query.start_nodes.len(), 1);
        assert_eq!(query.pattern.min_depth, 1);
    }
}
