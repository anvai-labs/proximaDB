// Per-model source executors for federated SQL extensions.
//
// Provides a trait-based abstraction for executing individual SQL extension
// functions (VECTOR_SEARCH, GRAPH_QUERY, DOCUMENT_QUERY, LOGS, METRICS)
// against their respective storage backends.
//
// These supplement the monolithic `FederatedExecutor` by providing a clean
// interface for testing and future extension.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::query::graph_lowering::lower_supported_graph_query_expr;

/// Result value from a federated source.
#[derive(Debug, Clone)]
pub enum FederatedValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Vector(Vec<f32>),
    Json(String),
}

impl std::fmt::Display for FederatedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FederatedValue::String(s) => write!(f, "{}", s),
            FederatedValue::Integer(i) => write!(f, "{}", i),
            FederatedValue::Float(v) => write!(f, "{}", v),
            FederatedValue::Boolean(b) => write!(f, "{}", b),
            FederatedValue::Null => write!(f, "NULL"),
            FederatedValue::Vector(v) => write!(f, "{:?}", v),
            FederatedValue::Json(j) => write!(f, "{}", j),
        }
    }
}

/// Parameters for a federated source query.
#[derive(Debug, Clone)]
pub struct SourceParams {
    /// Source-specific parameters as key-value pairs.
    pub params: HashMap<String, String>,
}

impl SourceParams {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn with(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.params.get(key).and_then(|v| v.parse().ok())
    }
}

impl Default for SourceParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Tabular result from a federated source execution.
#[derive(Debug, Clone)]
pub struct FederatedSourceResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<FederatedValue>>,
}

impl FederatedSourceResult {
    pub fn empty(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: vec![],
        }
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// Trait for per-model source executors.
///
/// Each implementation handles a specific SQL extension function
/// (e.g., VECTOR_SEARCH, GRAPH_QUERY) and translates parameters
/// into backend calls.
#[async_trait]
pub trait FederatedSourceExecutor: Send + Sync {
    /// Execute the source query with the given parameters.
    async fn execute(&self, params: &SourceParams) -> Result<FederatedSourceResult>;
    /// Identifier for this source type (e.g., "vector", "graph", "document").
    fn source_type(&self) -> &str;
}

/// Vector search executor for `VECTOR_SEARCH(collection, query_vector, top_k)`.
pub struct VectorSearchSourceExecutor;

impl VectorSearchSourceExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Parse a vector from a string representation like "[1.0, 2.0, 3.0]".
    pub fn parse_vector(s: &str) -> Result<Vec<f32>> {
        let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
        if trimmed.is_empty() {
            // The ONE empty-vector policy (every other boundary rejects —
            // dispatching a 0-dimension query is a kernel error or garbage).
            return Err(anyhow!("query vector must be non-empty"));
        }
        trimmed
            .split(',')
            .map(|v| {
                let f = v
                    .trim()
                    .parse::<f32>()
                    .map_err(|e| anyhow!("invalid vector component '{}': {}", v.trim(), e))?;
                // The ONE non-finite policy (inf/NaN reaches the kernels).
                if !f.is_finite() {
                    return Err(anyhow!("query vector components must be finite"));
                }
                Ok(f)
            })
            .collect()
    }
}

impl Default for VectorSearchSourceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FederatedSourceExecutor for VectorSearchSourceExecutor {
    async fn execute(&self, params: &SourceParams) -> Result<FederatedSourceResult> {
        let _collection = params
            .get("collection")
            .ok_or_else(|| anyhow!("VECTOR_SEARCH requires 'collection' parameter"))?;
        let _query_vector_str = params
            .get("query_vector")
            .ok_or_else(|| anyhow!("VECTOR_SEARCH requires 'query_vector' parameter"))?;
        let _top_k = params.get_usize("top_k").unwrap_or(10);

        // Actual execution is delegated to FederatedExecutor.execute_vector_search()
        // which has the storage backend reference. This executor validates parameters
        // and provides the trait interface for testing and extensibility.
        Ok(FederatedSourceResult::empty(vec![
            "id".to_string(),
            "score".to_string(),
        ]))
    }

    fn source_type(&self) -> &str {
        "vector"
    }
}

/// Graph query executor for `GRAPH_QUERY('cypher_query')`.
pub struct GraphQuerySourceExecutor;

impl GraphQuerySourceExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphQuerySourceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FederatedSourceExecutor for GraphQuerySourceExecutor {
    async fn execute(&self, params: &SourceParams) -> Result<FederatedSourceResult> {
        let cypher = params
            .get("cypher")
            .ok_or_else(|| anyhow!("GRAPH_QUERY requires 'cypher' parameter"))?;
        let graph_query = lower_supported_graph_query_expr(cypher, None, Some("default"))?;

        Ok(FederatedSourceResult::empty(graph_query.output_columns))
    }

    fn source_type(&self) -> &str {
        "graph"
    }
}

/// Document query executor for `DOCUMENT_QUERY(collection, filter)`.
pub struct DocumentQuerySourceExecutor;

impl DocumentQuerySourceExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DocumentQuerySourceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FederatedSourceExecutor for DocumentQuerySourceExecutor {
    async fn execute(&self, params: &SourceParams) -> Result<FederatedSourceResult> {
        let _collection = params
            .get("collection")
            .ok_or_else(|| anyhow!("DOCUMENT_QUERY requires 'collection' parameter"))?;

        Ok(FederatedSourceResult::empty(vec![
            "id".to_string(),
            "document".to_string(),
        ]))
    }

    fn source_type(&self) -> &str {
        "document"
    }
}

/// Observability executor for `LOGS(namespace)` and `METRICS(namespace)`.
pub struct ObservabilitySourceExecutor;

impl ObservabilitySourceExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ObservabilitySourceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FederatedSourceExecutor for ObservabilitySourceExecutor {
    async fn execute(&self, params: &SourceParams) -> Result<FederatedSourceResult> {
        let _namespace = params
            .get("namespace")
            .ok_or_else(|| anyhow!("LOGS/METRICS requires 'namespace' parameter"))?;
        let query_type = params.get("type").unwrap_or("logs");

        let columns = match query_type {
            "metrics" => vec![
                "timestamp".to_string(),
                "name".to_string(),
                "value".to_string(),
                "labels".to_string(),
            ],
            _ => vec![
                "timestamp".to_string(),
                "severity".to_string(),
                "message".to_string(),
                "source".to_string(),
            ],
        };

        Ok(FederatedSourceResult::empty(columns))
    }

    fn source_type(&self) -> &str {
        "observability"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_value_display() {
        assert_eq!(FederatedValue::String("hello".into()).to_string(), "hello");
        assert_eq!(FederatedValue::Integer(42).to_string(), "42");
        assert_eq!(FederatedValue::Float(3.5).to_string(), "3.5");
        assert_eq!(FederatedValue::Boolean(true).to_string(), "true");
        assert_eq!(FederatedValue::Null.to_string(), "NULL");
    }

    #[test]
    fn test_source_params() {
        let params = SourceParams::new()
            .with("collection", "products")
            .with("top_k", "10");

        assert_eq!(params.get("collection"), Some("products"));
        assert_eq!(params.get_usize("top_k"), Some(10));
        assert_eq!(params.get("missing"), None);
    }

    #[test]
    fn test_parse_vector() {
        let v = VectorSearchSourceExecutor::parse_vector("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);

        // Empty and non-finite literals reject (the ONE policy).
        assert!(VectorSearchSourceExecutor::parse_vector("[]").is_err());
        assert!(VectorSearchSourceExecutor::parse_vector("[NaN]").is_err());

        assert!(VectorSearchSourceExecutor::parse_vector("[a, b]").is_err());
    }

    #[tokio::test]
    async fn test_vector_executor_validates_params() {
        let executor = VectorSearchSourceExecutor::new();
        assert_eq!(executor.source_type(), "vector");

        // Missing collection should error
        let result = executor.execute(&SourceParams::new()).await;
        assert!(result.is_err());

        // With collection but missing vector should error
        let result = executor
            .execute(&SourceParams::new().with("collection", "products"))
            .await;
        assert!(result.is_err());

        // With all params should succeed
        let result = executor
            .execute(
                &SourceParams::new()
                    .with("collection", "products")
                    .with("query_vector", "[1.0, 2.0]")
                    .with("top_k", "5"),
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_graph_executor_validates_params() {
        let executor = GraphQuerySourceExecutor::new();
        assert_eq!(executor.source_type(), "graph");

        let result = executor.execute(&SourceParams::new()).await;
        assert!(result.is_err());

        let result = executor
            .execute(&SourceParams::new().with("cypher", "MATCH (n) RETURN n"))
            .await;
        assert!(result.is_ok());

        let projected = executor
            .execute(
                &SourceParams::new()
                    .with("cypher", "MATCH (n:Person) RETURN n.name AS person_name"),
            )
            .await
            .expect("projected graph query should validate");
        assert_eq!(projected.columns, vec!["person_name"]);
    }

    #[tokio::test]
    async fn test_document_executor_validates_params() {
        let executor = DocumentQuerySourceExecutor::new();
        assert_eq!(executor.source_type(), "document");

        let result = executor.execute(&SourceParams::new()).await;
        assert!(result.is_err());

        let result = executor
            .execute(&SourceParams::new().with("collection", "orders"))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().columns, vec!["id", "document"]);
    }

    #[tokio::test]
    async fn test_observability_executor() {
        let executor = ObservabilitySourceExecutor::new();
        assert_eq!(executor.source_type(), "observability");

        let result = executor
            .execute(
                &SourceParams::new()
                    .with("namespace", "production")
                    .with("type", "metrics"),
            )
            .await
            .unwrap();
        assert!(result.columns.contains(&"value".to_string()));

        let log_result = executor
            .execute(&SourceParams::new().with("namespace", "prod"))
            .await
            .unwrap();
        assert!(log_result.columns.contains(&"message".to_string()));
    }
}
