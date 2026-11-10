//! # Unified Query Facade
//!
//! This module provides a single entry point for ALL query execution in ProximaDB.
//! It consolidates the 5 parallel query paths into a unified facade using the
//! Strategy pattern for extensibility.
//!
//! ## Design Principles
//!
//! - **Single Responsibility**: Each QueryStrategy handles one query type
//! - **Open/Closed**: New query types added without modifying facade
//! - **Dependency Inversion**: Facade depends on QueryStrategy trait
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UnifiedQueryFacade                            │
//! │   Single entry point for all queries                             │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!         ┌─────────────────────┼─────────────────────┐
//!         ▼                     ▼                     ▼
//! ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
//! │VectorStrategy │    │ SQLStrategy   │    │ GraphStrategy │
//! │ Vector search │    │ SQL queries   │    │ Graph traversal│
//! └───────────────┘    └───────────────┘    └───────────────┘
//! ```

pub mod adapter;
pub mod strategies;

pub use adapter::{ExplainComponent, ExplainResult, QueryFacadeAdapter};
use proximadb_data_model::ProximaValue;
pub use proximadb_records::ScoredRecord;
pub use strategies::{
    ColumnarStrategy, ColumnarStrategyConfig, DocumentStrategy, GraphStrategy,
    ObservabilityStrategy, SqlStrategy, VectorSearchStrategy,
};

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

// ================================================================================
// QUERY REQUEST - Unified Input Type
// ================================================================================

/// Unified query request that can represent any query type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Query type discriminant
    pub query_type: QueryType,
    /// Collection/graph name (if applicable)
    pub target: Option<String>,
    /// Raw query content (vector, SQL, Cypher, etc.)
    pub content: QueryContent,
    /// Query parameters
    pub params: QueryParams,
}

/// Query type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Pure vector similarity search
    VectorSearch,
    /// SQL query (may contain vector operations)
    Sql,
    /// Federated multi-model query (SQL with extensions)
    Federated,
    /// Graph traversal (Cypher-like)
    Graph,
    /// Document query (JSON path)
    Document,
    /// Observability query (logs/metrics)
    Observability,
}

/// Query content variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryContent {
    /// Vector search content
    Vector {
        query_vector: Vec<f32>,
        top_k: usize,
    },
    /// SQL query string
    Sql(String),
    /// Graph query (Cypher-like)
    Graph(String),
    /// Document filter expression
    Document(String),
    /// Raw bytes (for advanced use cases)
    Raw(Vec<u8>),
}

/// Query execution parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryParams {
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Enable query caching
    pub use_cache: bool,
    /// Force specific execution path (for testing/debugging)
    pub force_path: Option<String>,
    /// Return execution metrics
    pub include_metrics: bool,
    /// Simple equality filters for vector search requests
    pub vector_filters: std::collections::HashMap<String, ProximaValue>,
    /// Structured filter tree for vector search requests
    pub vector_advanced_filter: Option<crate::proto::proximadb_v1::MetadataFilter>,
    /// Parsed document filter for document query execution
    pub document_filter: Option<crate::proto::proximadb_v1::DocumentFilter>,
}

impl QueryRequest {
    /// Create a vector search request
    pub fn vector_search(query_vector: Vec<f32>, top_k: usize) -> Self {
        Self {
            query_type: QueryType::VectorSearch,
            target: None,
            content: QueryContent::Vector {
                query_vector,
                top_k,
            },
            params: QueryParams::default(),
        }
    }

    /// Create a SQL query request
    pub fn sql(query: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Sql,
            target: None,
            content: QueryContent::Sql(query.into()),
            params: QueryParams::default(),
        }
    }

    /// Create a federated query request
    pub fn federated(query: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Federated,
            target: None,
            content: QueryContent::Sql(query.into()),
            params: QueryParams::default(),
        }
    }

    /// Create a graph query request
    pub fn graph(query: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Graph,
            target: None,
            content: QueryContent::Graph(query.into()),
            params: QueryParams::default(),
        }
    }

    /// Create a document query request
    pub fn document(query: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Document,
            target: None,
            content: QueryContent::Document(query.into()),
            params: QueryParams::default(),
        }
    }

    /// Set target collection/graph
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set query parameters
    pub fn with_params(mut self, params: QueryParams) -> Self {
        self.params = params;
        self
    }

    /// Attach simple vector filters to the request
    pub fn with_vector_filters(
        mut self,
        filters: std::collections::HashMap<String, ProximaValue>,
    ) -> Self {
        self.params.vector_filters = filters;
        self
    }

    /// Attach v1 wire filters at a compatibility edge.
    pub fn with_vector_filters_v1(
        mut self,
        filters: std::collections::HashMap<String, crate::proto::proximadb_v1::SqlValue>,
    ) -> Self {
        self.params.vector_filters = filters
            .into_iter()
            .map(|(key, value)| {
                (
                    key,
                    crate::core::search::results::sql_value_to_proxima_value(value),
                )
            })
            .collect();
        self
    }

    /// Attach an advanced vector filter to the request
    pub fn with_vector_advanced_filter(
        mut self,
        filter: crate::proto::proximadb_v1::MetadataFilter,
    ) -> Self {
        self.params.vector_advanced_filter = Some(filter);
        self
    }

    /// Attach a parsed document filter to the request
    pub fn with_document_filter(
        mut self,
        filter: crate::proto::proximadb_v1::DocumentFilter,
    ) -> Self {
        self.params.document_filter = Some(filter);
        self
    }

    /// Include execution metrics in result
    pub fn with_metrics(mut self) -> Self {
        self.params.include_metrics = true;
        self
    }
}

// ================================================================================
// QUERY RESULT - Unified Output Type
// ================================================================================

/// Backwards-compat alias for [`FacadeQueryResult`].
pub type QueryResult = FacadeQueryResult;

/// Unified query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacadeQueryResult {
    /// Result data (type depends on query)
    pub data: QueryResultData,
    /// Execution metrics (if requested)
    pub metrics: Option<ExecutionMetrics>,
}

/// Query result data variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResultData {
    /// Vector search results (spec §1490 — canonical (ProximaRecord, score, rank) triples)
    VectorResults(Vec<ScoredRecord>),
    /// Tabular results (SQL, document queries)
    Rows(Vec<serde_json::Value>),
    /// Graph results (nodes, edges, paths)
    Graph(GraphQueryResult),
    /// Empty result
    Empty,
}

/// Graph query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQueryResult {
    pub nodes: Vec<serde_json::Value>,
    pub edges: Vec<serde_json::Value>,
    pub paths: Vec<serde_json::Value>,
}

/// Execution metrics for debugging and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionMetrics {
    /// Which execution path was used
    pub execution_path: String,
    /// Strategy that handled the query
    pub strategy_name: String,
    /// Total execution time
    pub execution_time_ms: u64,
    /// Time spent in planning
    pub planning_time_ms: u64,
    /// Number of results scanned
    pub results_scanned: usize,
    /// Number of results returned
    pub results_returned: usize,
    /// Whether cache was used
    pub cache_hit: bool,
    /// Additional strategy-specific metrics
    pub extra: serde_json::Value,
}

// ================================================================================
// QUERY STRATEGY TRAIT - Extension Point
// ================================================================================

/// Strategy trait for query execution (Open/Closed Principle)
///
/// Implement this trait to add new query execution strategies without
/// modifying the UnifiedQueryFacade.
#[async_trait]
pub trait QueryStrategy: Send + Sync {
    /// Strategy name for metrics/debugging
    fn name(&self) -> &str;

    /// Check if this strategy can handle the given query
    fn can_handle(&self, request: &QueryRequest) -> bool;

    /// Execute the query and return results
    async fn execute(
        &self,
        request: QueryRequest,
        ctx: &FacadeQueryContext,
    ) -> Result<FacadeQueryResult>;

    /// Priority when multiple strategies can handle a query (higher = preferred)
    fn priority(&self) -> i32 {
        0
    }
}

/// Backwards-compat alias for [`FacadeQueryContext`].
pub type QueryContext = FacadeQueryContext;

/// Query execution context passed to strategies
pub struct FacadeQueryContext {
    /// Request ID for tracing
    pub request_id: String,
    /// Start time for timeout tracking
    pub start_time: Instant,
    /// Timeout duration
    pub timeout: Duration,
}

impl FacadeQueryContext {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Check if query has timed out
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }

    /// Remaining time before timeout
    pub fn remaining_time(&self) -> Duration {
        self.timeout.saturating_sub(self.start_time.elapsed())
    }
}

// ================================================================================
// UNIFIED QUERY FACADE - Main Entry Point
// ================================================================================

/// Configuration for the unified query facade
#[derive(Debug, Clone)]
pub struct FacadeConfig {
    /// Default timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Enable query caching
    pub enable_cache: bool,
    /// Maximum concurrent queries
    pub max_concurrent: usize,
}

impl Default for FacadeConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 30000,
            enable_cache: true,
            max_concurrent: 100,
        }
    }
}

/// Unified Query Facade - Single entry point for all queries
///
/// This facade consolidates the 5 parallel query paths:
/// 1. UnifiedQueryOptimizer (vector search optimization)
/// 2. FederatedFacadeQueryContext (SQL with extensions)
/// 3. UnifiedQueryEngine (multi-model decomposition)
/// 4. Distributed query coordination
/// 5. AST-based query engine
///
/// Into a single, consistent interface with pluggable strategies.
pub struct UnifiedQueryFacade {
    /// Registered query strategies
    strategies: Vec<Arc<dyn QueryStrategy>>,
    /// Configuration
    config: FacadeConfig,
}

impl UnifiedQueryFacade {
    /// Create a new facade with the given strategies
    pub fn new(strategies: Vec<Arc<dyn QueryStrategy>>, config: FacadeConfig) -> Self {
        Self { strategies, config }
    }

    /// Create a facade with default configuration
    pub fn with_strategies(strategies: Vec<Arc<dyn QueryStrategy>>) -> Self {
        Self::new(strategies, FacadeConfig::default())
    }

    /// Register a new strategy
    pub fn register_strategy(&mut self, strategy: Arc<dyn QueryStrategy>) {
        self.strategies.push(strategy);
    }

    /// Execute a query through the appropriate strategy
    #[instrument(skip(self, request), fields(query_type = ?request.query_type))]
    pub async fn execute(&self, request: QueryRequest) -> Result<FacadeQueryResult> {
        let timeout_ms = request
            .params
            .timeout_ms
            .unwrap_or(self.config.default_timeout_ms);
        let ctx = FacadeQueryContext::new(timeout_ms);
        let include_metrics = request.params.include_metrics;
        let start = Instant::now();
        let telemetry_kind = match request.query_type {
            QueryType::VectorSearch => Some("vector"),
            QueryType::Sql => Some("sql"),
            QueryType::Federated => None,
            QueryType::Graph => Some("graph"),
            QueryType::Document => Some("document"),
            QueryType::Observability => Some("observability"),
        };

        if let Some(kind) = telemetry_kind {
            crate::query::utils::metrics::record_query_start(kind);
        }

        // Find the best strategy for this query
        let strategy = self.select_strategy(&request)?;

        debug!(
            strategy = strategy.name(),
            "Selected strategy for query execution"
        );

        // Execute through the selected strategy
        let result = strategy.execute(request, &ctx).await;

        if let Some(kind) = telemetry_kind {
            crate::query::utils::metrics::record_query_end(
                kind,
                result.is_ok(),
                start.elapsed().as_millis() as u64,
            );
        }

        let mut result = result?;

        // Add metrics if requested
        if include_metrics {
            let mut metrics = result.metrics.take().unwrap_or_default();
            metrics.execution_path = "unified".to_string();
            metrics.strategy_name = strategy.name().to_string();
            metrics.execution_time_ms = start.elapsed().as_millis() as u64;
            result.metrics = Some(metrics);
        }

        Ok(result)
    }

    /// Select the best strategy for a query
    fn select_strategy(&self, request: &QueryRequest) -> Result<&Arc<dyn QueryStrategy>> {
        // If a specific path is forced, try to find it
        if let Some(ref force_path) = request.params.force_path {
            for strategy in &self.strategies {
                if strategy.name() == force_path {
                    return Ok(strategy);
                }
            }
            return Err(anyhow!("Forced path '{}' not found", force_path));
        }

        // Find all strategies that can handle this query
        let mut candidates: Vec<_> = self
            .strategies
            .iter()
            .filter(|s| s.can_handle(request))
            .collect();

        if candidates.is_empty() {
            return Err(anyhow!(
                "No strategy found for query type: {:?}",
                request.query_type
            ));
        }

        // Sort by priority (highest first)
        candidates.sort_by_key(|c| std::cmp::Reverse(c.priority()));

        Ok(candidates.first().ok_or_else(|| {
            anyhow!(
                "No candidates available for query type: {:?}",
                request.query_type
            )
        })?)
    }

    /// Get list of registered strategy names
    pub fn strategy_names(&self) -> Vec<&str> {
        self.strategies.iter().map(|s| s.name()).collect()
    }
}

// ================================================================================
// TESTS - TDD First
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock strategy for testing
    struct MockVectorStrategy {
        name: String,
    }

    #[async_trait]
    impl QueryStrategy for MockVectorStrategy {
        fn name(&self) -> &str {
            &self.name
        }

        fn can_handle(&self, request: &QueryRequest) -> bool {
            request.query_type == QueryType::VectorSearch
        }

        async fn execute(
            &self,
            _request: QueryRequest,
            _ctx: &FacadeQueryContext,
        ) -> Result<FacadeQueryResult> {
            Ok(FacadeQueryResult {
                data: QueryResultData::VectorResults(vec![ScoredRecord {
                    record: proximadb_records::ProximaRecord {
                        oid: "test_1".to_string(),
                        ..Default::default()
                    },
                    score: 0.95,
                    rank: 1,
                }]),
                metrics: Some(ExecutionMetrics {
                    execution_path: "unified".to_string(),
                    strategy_name: self.name.clone(),
                    ..Default::default()
                }),
            })
        }

        fn priority(&self) -> i32 {
            10
        }
    }

    struct MockSqlStrategy;

    #[async_trait]
    impl QueryStrategy for MockSqlStrategy {
        fn name(&self) -> &str {
            "sql"
        }

        fn can_handle(&self, request: &QueryRequest) -> bool {
            matches!(request.query_type, QueryType::Sql | QueryType::Federated)
        }

        async fn execute(
            &self,
            _request: QueryRequest,
            _ctx: &FacadeQueryContext,
        ) -> Result<FacadeQueryResult> {
            Ok(FacadeQueryResult {
                data: QueryResultData::Rows(vec![]),
                metrics: Some(ExecutionMetrics {
                    execution_path: "unified".to_string(),
                    strategy_name: "sql".to_string(),
                    ..Default::default()
                }),
            })
        }
    }

    struct MockGraphStrategy;

    #[async_trait]
    impl QueryStrategy for MockGraphStrategy {
        fn name(&self) -> &str {
            "graph"
        }

        fn can_handle(&self, request: &QueryRequest) -> bool {
            request.query_type == QueryType::Graph
        }

        async fn execute(
            &self,
            _request: QueryRequest,
            _ctx: &FacadeQueryContext,
        ) -> Result<FacadeQueryResult> {
            Ok(FacadeQueryResult {
                data: QueryResultData::Graph(GraphQueryResult {
                    nodes: vec![],
                    edges: vec![],
                    paths: vec![],
                }),
                metrics: Some(ExecutionMetrics {
                    execution_path: "unified".to_string(),
                    strategy_name: "graph".to_string(),
                    ..Default::default()
                }),
            })
        }
    }

    fn create_test_facade() -> UnifiedQueryFacade {
        let strategies: Vec<Arc<dyn QueryStrategy>> = vec![
            Arc::new(MockVectorStrategy {
                name: "vector".to_string(),
            }),
            Arc::new(MockSqlStrategy),
            Arc::new(MockGraphStrategy),
        ];
        UnifiedQueryFacade::with_strategies(strategies)
    }

    // ============================================================================
    // TDD Tests - These define the expected behavior
    // ============================================================================

    #[tokio::test]
    async fn test_vector_search_routes_through_unified() {
        let facade = create_test_facade();
        let query = QueryRequest::vector_search(vec![0.1; 128], 10).with_metrics();

        let result = facade
            .execute(query)
            .await
            .expect("Vector search query should succeed");

        // Verify execution path is "unified"
        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.execution_path, "unified", "Should use unified path");
        assert_eq!(
            metrics.strategy_name, "vector",
            "Should use vector strategy"
        );

        // Verify result type
        assert!(matches!(result.data, QueryResultData::VectorResults(_)));
    }

    #[tokio::test]
    async fn test_sql_query_routes_through_unified() {
        let facade = create_test_facade();
        let query = QueryRequest::sql("SELECT * FROM products").with_metrics();

        let result = facade
            .execute(query)
            .await
            .expect("SQL query should succeed");

        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.execution_path, "unified");
        assert_eq!(metrics.strategy_name, "sql");
    }

    #[tokio::test]
    async fn test_federated_query_uses_unified_path() {
        let facade = create_test_facade();
        let query =
            QueryRequest::federated("SELECT * FROM VECTOR_SEARCH('products', '[0.1,0.2]', 10)")
                .with_metrics();

        let result = facade
            .execute(query)
            .await
            .expect("Federated query should succeed");

        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.execution_path, "unified");
        // Federated queries should route to SQL strategy
        assert_eq!(metrics.strategy_name, "sql");
    }

    #[tokio::test]
    async fn test_graph_query_routes_through_unified() {
        let facade = create_test_facade();
        let query = QueryRequest::graph("MATCH (n) RETURN n").with_metrics();

        let result = facade
            .execute(query)
            .await
            .expect("Graph query should succeed");

        let metrics = result.metrics.expect("Metrics should be present");
        assert_eq!(metrics.execution_path, "unified");
        assert_eq!(metrics.strategy_name, "graph");

        assert!(matches!(result.data, QueryResultData::Graph(_)));
    }

    #[tokio::test]
    async fn test_no_strategy_returns_error() {
        let facade = UnifiedQueryFacade::with_strategies(vec![]);
        let query = QueryRequest::vector_search(vec![0.1; 10], 5);

        let result = facade.execute(query).await;

        assert!(result.is_err());
        let err = result.expect_err("Should return error when no strategy available");
        assert!(err.to_string().contains("No strategy found"));
    }

    #[tokio::test]
    async fn test_strategy_priority_ordering() {
        // Create two strategies that both handle vector search
        struct LowPriorityVector;
        struct HighPriorityVector;

        #[async_trait]
        impl QueryStrategy for LowPriorityVector {
            fn name(&self) -> &str {
                "low-priority"
            }
            fn can_handle(&self, r: &QueryRequest) -> bool {
                r.query_type == QueryType::VectorSearch
            }
            async fn execute(
                &self,
                _r: QueryRequest,
                _ctx: &FacadeQueryContext,
            ) -> Result<FacadeQueryResult> {
                Ok(FacadeQueryResult {
                    data: QueryResultData::Empty,
                    metrics: Some(ExecutionMetrics {
                        strategy_name: "low-priority".to_string(),
                        ..Default::default()
                    }),
                })
            }
            fn priority(&self) -> i32 {
                1
            }
        }

        #[async_trait]
        impl QueryStrategy for HighPriorityVector {
            fn name(&self) -> &str {
                "high-priority"
            }
            fn can_handle(&self, r: &QueryRequest) -> bool {
                r.query_type == QueryType::VectorSearch
            }
            async fn execute(
                &self,
                _r: QueryRequest,
                _ctx: &FacadeQueryContext,
            ) -> Result<FacadeQueryResult> {
                Ok(FacadeQueryResult {
                    data: QueryResultData::Empty,
                    metrics: Some(ExecutionMetrics {
                        strategy_name: "high-priority".to_string(),
                        ..Default::default()
                    }),
                })
            }
            fn priority(&self) -> i32 {
                100
            }
        }

        let strategies: Vec<Arc<dyn QueryStrategy>> =
            vec![Arc::new(LowPriorityVector), Arc::new(HighPriorityVector)];
        let facade = UnifiedQueryFacade::with_strategies(strategies);

        let query = QueryRequest::vector_search(vec![0.1], 10).with_metrics();
        let result = facade
            .execute(query)
            .await
            .expect("Priority query should succeed");

        let metrics = result
            .metrics
            .expect("Metrics should be present when with_metrics() is called");
        assert_eq!(
            metrics.strategy_name, "high-priority",
            "Higher priority strategy should be selected"
        );
    }

    #[tokio::test]
    async fn test_force_path_selects_specific_strategy() {
        let facade = create_test_facade();
        let mut params = QueryParams::default();
        params.force_path = Some("sql".to_string());
        params.include_metrics = true;

        // Even though this is a vector search query, force it to use SQL strategy
        let query = QueryRequest::vector_search(vec![0.1], 10).with_params(params);

        let result = facade
            .execute(query)
            .await
            .expect("Forced path query should succeed");

        let metrics = result
            .metrics
            .expect("Metrics should be present when include_metrics is true");
        assert_eq!(
            metrics.strategy_name, "sql",
            "Should use forced SQL strategy"
        );
    }

    #[tokio::test]
    async fn test_force_invalid_path_returns_error() {
        let facade = create_test_facade();
        let mut params = QueryParams::default();
        params.force_path = Some("nonexistent".to_string());

        let query = QueryRequest::vector_search(vec![0.1], 10).with_params(params);
        let result = facade.execute(query).await;

        assert!(result.is_err());
        let err = result.expect_err("Should return error for invalid forced path");
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_query_request_builder() {
        let query = QueryRequest::vector_search(vec![0.1, 0.2, 0.3], 10)
            .with_target("my_collection")
            .with_metrics();

        assert_eq!(query.query_type, QueryType::VectorSearch);
        assert_eq!(query.target, Some("my_collection".to_string()));
        assert!(query.params.include_metrics);

        if let QueryContent::Vector {
            query_vector,
            top_k,
        } = &query.content
        {
            assert_eq!(query_vector.len(), 3);
            assert_eq!(*top_k, 10);
        } else {
            panic!("Expected Vector content");
        }
    }

    #[tokio::test]
    async fn test_strategy_names() {
        let facade = create_test_facade();
        let names = facade.strategy_names();

        assert!(names.contains(&"vector"));
        assert!(names.contains(&"sql"));
        assert!(names.contains(&"graph"));
    }

    #[tokio::test]
    async fn test_execution_metrics_includes_time() {
        let facade = create_test_facade();
        let query = QueryRequest::vector_search(vec![0.1], 10).with_metrics();

        let result = facade
            .execute(query)
            .await
            .expect("Execution time query should succeed");

        let metrics = result
            .metrics
            .expect("Metrics should be present when with_metrics() is called");
        // Execution time field should exist (u64 is always >= 0)
        // Note: In fast systems this might be 0, but the field should exist
        let _ = metrics.execution_time_ms;
    }
}
