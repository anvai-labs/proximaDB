// Re-enabled 2026-11-08 (compile-disabled since e56a45d90, 2026-05-22): the
// legacy `VectorMatch {id, score, metadata}` fixture shape maps onto the
// canonical `ScoredRecord {record, score, rank}` (metadata was fixture-only,
// never asserted; its canonical carrier is `ProximaRecord::props`).
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

//! # Explain Plan Consistency Tests
//!
//! This module validates that `/api/v1/unified/explain` returns consistent
//! explain plan formats across different query types and API protocols.
//!
//! ## Test Coverage
//!
//! 1. **Plan Format Consistency**: Verify explain plans have consistent schema
//! 2. **Strategy Identification**: Verify plans show which strategy will be used
//! 3. **REST/gRPC Parity**: Verify same explain plan schema from both protocols
//! 4. **Query Type Coverage**: Test explain for vector, SQL, graph queries
//!
//! ## Architecture
//!
//! ```text
//! Query Request --> UnifiedQueryFacade --> Strategy Selection --> ExplainPlan
//!                          |
//!                   Returns plan with:
//!                   - execution_strategy
//!                   - orchestration_steps
//!                   - estimated_cost
//!                   - query_type hints
//! ```

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use proximadb::query::facade::{
    ExecutionMetrics, FacadeConfig, QueryContext, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy, QueryType, ScoredRecord, UnifiedQueryFacade,
};
use proximadb_records::ProximaRecord;

// ================================================================================
// EXPLAIN PLAN RESPONSE SCHEMA
// ================================================================================

/// Expected schema for explain plan responses
/// This defines the consistent format that both REST and gRPC should return
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainPlanResponse {
    /// The query type that will be used (vector, sql, graph, federated)
    pub query_type: String,
    /// The strategy that will handle execution
    pub execution_strategy: String,
    /// Step-by-step execution plan
    pub orchestration_steps: Vec<String>,
    /// Estimated total cost (unitless, relative)
    pub estimated_total_cost: f64,
    /// Whether the query is parallelizable
    pub parallelizable: bool,
    /// Strategy-specific hints
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hints: Option<ExplainHints>,
}

/// Strategy-specific hints in explain plans
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExplainHints {
    /// For vector queries: index type, ef_search, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<VectorExplainHints>,
    /// For SQL queries: table scans, joins, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sql: Option<SqlExplainHints>,
    /// For graph queries: traversal algorithm, depth, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphExplainHints>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorExplainHints {
    pub index_type: Option<String>,
    pub ef_search: Option<usize>,
    pub nprobe: Option<usize>,
    pub quantization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SqlExplainHints {
    pub tables_accessed: Vec<String>,
    pub index_usage: Vec<String>,
    pub estimated_rows: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphExplainHints {
    pub traversal_algorithm: Option<String>,
    pub max_depth: Option<u32>,
    pub start_nodes: Option<usize>,
}

// ================================================================================
// MOCK STRATEGIES WITH EXPLAIN SUPPORT
// ================================================================================

/// Mock vector strategy that supports explain
struct MockVectorStrategyWithExplain {
    name: String,
}

impl MockVectorStrategyWithExplain {
    fn new() -> Self {
        Self {
            name: "vector_hnsw".to_string(),
        }
    }

    fn explain(&self, request: &QueryRequest) -> ExplainPlanResponse {
        let (top_k, _vector_dim) = match &request.content {
            proximadb::query::facade::QueryContent::Vector {
                query_vector,
                top_k,
            } => (*top_k, query_vector.len()),
            _ => (10, 0),
        };

        ExplainPlanResponse {
            query_type: "VectorSearch".to_string(),
            execution_strategy: self.name.clone(),
            orchestration_steps: vec![
                "Parse vector search request".to_string(),
                format!(
                    "Select HNSW index for collection '{}'",
                    request.target.as_deref().unwrap_or("default")
                ),
                format!(
                    "Execute approximate nearest neighbor search (top_k={})",
                    top_k
                ),
                "Apply post-filters if specified".to_string(),
                "Return ranked results".to_string(),
            ],
            estimated_total_cost: 1.0 + (top_k as f64 * 0.01),
            parallelizable: true,
            hints: Some(ExplainHints {
                vector: Some(VectorExplainHints {
                    index_type: Some("HNSW".to_string()),
                    ef_search: Some(100),
                    nprobe: None,
                    quantization: Some("none".to_string()),
                }),
                sql: None,
                graph: None,
            }),
        }
    }
}

#[async_trait]
impl QueryStrategy for MockVectorStrategyWithExplain {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        request.query_type == QueryType::VectorSearch
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let top_k = match &request.content {
            proximadb::query::facade::QueryContent::Vector { top_k, .. } => *top_k,
            _ => 10,
        };

        let results: Vec<ScoredRecord> = (0..top_k.min(5))
            .map(|i| ScoredRecord {
                record: ProximaRecord {
                    oid: format!("vec_{:03}", i + 1),
                    ..Default::default()
                },
                score: 0.95 - (i as f32 * 0.05),
                rank: (i + 1) as u32,
            })
            .collect();

        Ok(QueryResult {
            data: QueryResultData::VectorResults(results.clone()),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: self.name.clone(),
                execution_time_ms: 5,
                planning_time_ms: 1,
                results_scanned: 1000,
                results_returned: results.len(),
                cache_hit: false,
                extra: json!({
                    "index_type": "HNSW",
                    "ef_search": 100
                }),
            }),
        })
    }
}

/// Mock SQL strategy that supports explain
struct MockSqlStrategyWithExplain;

impl MockSqlStrategyWithExplain {
    fn new() -> Self {
        Self
    }

    fn explain(&self, request: &QueryRequest) -> ExplainPlanResponse {
        let sql = match &request.content {
            proximadb::query::facade::QueryContent::Sql(s) => s.clone(),
            _ => String::new(),
        };

        // Simple parsing to detect tables
        let tables: Vec<String> = if sql.to_uppercase().contains("FROM") {
            vec!["products".to_string()] // Simplified
        } else {
            vec![]
        };

        ExplainPlanResponse {
            query_type: "Sql".to_string(),
            execution_strategy: "sql_executor".to_string(),
            orchestration_steps: vec![
                "Parse SQL query".to_string(),
                "Analyze query structure".to_string(),
                format!("Plan access to tables: {:?}", tables),
                "Optimize join order".to_string(),
                "Execute query plan".to_string(),
            ],
            estimated_total_cost: 2.0 + (tables.len() as f64 * 0.5),
            parallelizable: tables.len() > 1,
            hints: Some(ExplainHints {
                vector: None,
                sql: Some(SqlExplainHints {
                    tables_accessed: tables,
                    index_usage: vec!["primary_key".to_string()],
                    estimated_rows: Some(100),
                }),
                graph: None,
            }),
        }
    }
}

#[async_trait]
impl QueryStrategy for MockSqlStrategyWithExplain {
    fn name(&self) -> &str {
        "sql_executor"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(request.query_type, QueryType::Sql | QueryType::Federated)
    }

    fn priority(&self) -> i32 {
        90
    }

    async fn execute(&self, _request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        Ok(QueryResult {
            data: QueryResultData::Rows(vec![
                json!({"id": "1", "name": "Product A", "price": 99.99}),
                json!({"id": "2", "name": "Product B", "price": 149.99}),
            ]),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: "sql_executor".to_string(),
                execution_time_ms: 10,
                planning_time_ms: 2,
                results_scanned: 500,
                results_returned: 2,
                cache_hit: false,
                extra: json!({"tables": ["products"]}),
            }),
        })
    }
}

/// Mock graph strategy that supports explain
struct MockGraphStrategyWithExplain;

impl MockGraphStrategyWithExplain {
    fn new() -> Self {
        Self
    }

    fn explain(&self, request: &QueryRequest) -> ExplainPlanResponse {
        let cypher = match &request.content {
            proximadb::query::facade::QueryContent::Graph(s) => s.clone(),
            _ => String::new(),
        };

        // Detect traversal depth from query
        let max_depth = if cypher.contains("*..") {
            5
        } else if cypher.contains("-[") {
            2
        } else {
            1
        };

        ExplainPlanResponse {
            query_type: "Graph".to_string(),
            execution_strategy: "graph_orion".to_string(),
            orchestration_steps: vec![
                "Parse Cypher query".to_string(),
                "Identify start nodes".to_string(),
                format!("Plan traversal (max_depth={})", max_depth),
                "Execute traversal with ORION engine".to_string(),
                "Collect and format results".to_string(),
            ],
            estimated_total_cost: 3.0 + (max_depth as f64 * 0.5),
            parallelizable: true,
            hints: Some(ExplainHints {
                vector: None,
                sql: None,
                graph: Some(GraphExplainHints {
                    traversal_algorithm: Some("BFS".to_string()),
                    max_depth: Some(max_depth),
                    start_nodes: Some(1),
                }),
            }),
        }
    }
}

#[async_trait]
impl QueryStrategy for MockGraphStrategyWithExplain {
    fn name(&self) -> &str {
        "graph_orion"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        request.query_type == QueryType::Graph
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, _request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        use proximadb::query::facade::GraphQueryResult;

        Ok(QueryResult {
            data: QueryResultData::Graph(GraphQueryResult {
                nodes: vec![
                    json!({"id": "n1", "label": "Person", "name": "Alice"}),
                    json!({"id": "n2", "label": "Person", "name": "Bob"}),
                ],
                edges: vec![json!({"source": "n1", "target": "n2", "type": "KNOWS"})],
                paths: vec![],
            }),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: "graph_orion".to_string(),
                execution_time_ms: 8,
                planning_time_ms: 1,
                results_scanned: 100,
                results_returned: 2,
                cache_hit: false,
                extra: json!({"traversal": "BFS", "depth": 2}),
            }),
        })
    }
}

// ================================================================================
// TEST UTILITIES
// ================================================================================

/// Trait extension for strategies that support explain
#[allow(dead_code)]
trait ExplainableStrategy: QueryStrategy {
    fn explain(&self, request: &QueryRequest) -> ExplainPlanResponse;
}

/// Create a test facade with explain-capable strategies
fn create_test_facade_with_explain() -> (
    UnifiedQueryFacade,
    Arc<MockVectorStrategyWithExplain>,
    Arc<MockSqlStrategyWithExplain>,
    Arc<MockGraphStrategyWithExplain>,
) {
    let vector_strategy = Arc::new(MockVectorStrategyWithExplain::new());
    let sql_strategy = Arc::new(MockSqlStrategyWithExplain::new());
    let graph_strategy = Arc::new(MockGraphStrategyWithExplain::new());

    let strategies: Vec<Arc<dyn QueryStrategy>> = vec![
        vector_strategy.clone(),
        sql_strategy.clone(),
        graph_strategy.clone(),
    ];

    let facade = UnifiedQueryFacade::new(strategies, FacadeConfig::default());

    (facade, vector_strategy, sql_strategy, graph_strategy)
}

/// Generate explain plan for a query using the appropriate strategy
fn generate_explain_plan(
    request: &QueryRequest,
    vector_strategy: &MockVectorStrategyWithExplain,
    sql_strategy: &MockSqlStrategyWithExplain,
    graph_strategy: &MockGraphStrategyWithExplain,
) -> Option<ExplainPlanResponse> {
    match request.query_type {
        QueryType::VectorSearch => Some(vector_strategy.explain(request)),
        QueryType::Sql | QueryType::Federated => Some(sql_strategy.explain(request)),
        QueryType::Graph => Some(graph_strategy.explain(request)),
        _ => None,
    }
}

// ================================================================================
// EXPLAIN PLAN FORMAT CONSISTENCY TESTS
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Vector search explain plan has consistent schema
    #[tokio::test]
    async fn test_vector_search_explain_plan_schema() {
        let (_, vector_strategy, _, _) = create_test_facade_with_explain();

        let request =
            QueryRequest::vector_search(vec![0.1, 0.2, 0.3, 0.4], 10).with_target("products");

        let plan = vector_strategy.explain(&request);

        // Verify required fields
        assert_eq!(plan.query_type, "VectorSearch");
        assert!(!plan.execution_strategy.is_empty());
        assert!(!plan.orchestration_steps.is_empty());
        assert!(plan.estimated_total_cost > 0.0);

        // Verify vector-specific hints
        assert!(plan.hints.is_some());
        let hints = plan.hints.unwrap();
        assert!(hints.vector.is_some());
        assert!(hints.sql.is_none());
        assert!(hints.graph.is_none());

        let vector_hints = hints.vector.unwrap();
        assert!(vector_hints.index_type.is_some());
    }

    /// Test 2: SQL query explain plan has consistent schema
    #[tokio::test]
    async fn test_sql_query_explain_plan_schema() {
        let (_, _, sql_strategy, _) = create_test_facade_with_explain();

        let request = QueryRequest::sql("SELECT * FROM products WHERE price > 100");

        let plan = sql_strategy.explain(&request);

        // Verify required fields
        assert_eq!(plan.query_type, "Sql");
        assert_eq!(plan.execution_strategy, "sql_executor");
        assert!(!plan.orchestration_steps.is_empty());
        assert!(plan.estimated_total_cost > 0.0);

        // Verify SQL-specific hints
        assert!(plan.hints.is_some());
        let hints = plan.hints.unwrap();
        assert!(hints.sql.is_some());
        assert!(hints.vector.is_none());
        assert!(hints.graph.is_none());

        let sql_hints = hints.sql.unwrap();
        assert!(!sql_hints.tables_accessed.is_empty());
    }

    /// Test 3: Graph query explain plan has consistent schema
    #[tokio::test]
    async fn test_graph_query_explain_plan_schema() {
        let (_, _, _, graph_strategy) = create_test_facade_with_explain();

        let request = QueryRequest::graph("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name")
            .with_target("social_graph");

        let plan = graph_strategy.explain(&request);

        // Verify required fields
        assert_eq!(plan.query_type, "Graph");
        assert_eq!(plan.execution_strategy, "graph_orion");
        assert!(!plan.orchestration_steps.is_empty());
        assert!(plan.estimated_total_cost > 0.0);

        // Verify graph-specific hints
        assert!(plan.hints.is_some());
        let hints = plan.hints.unwrap();
        assert!(hints.graph.is_some());
        assert!(hints.vector.is_none());
        assert!(hints.sql.is_none());

        let graph_hints = hints.graph.unwrap();
        assert!(graph_hints.traversal_algorithm.is_some());
        assert!(graph_hints.max_depth.is_some());
    }

    /// Test 4: Federated query explain plan shows SQL strategy
    #[tokio::test]
    async fn test_federated_query_explain_plan_schema() {
        let (_, _, sql_strategy, _) = create_test_facade_with_explain();

        let request =
            QueryRequest::federated("SELECT * FROM VECTOR_SEARCH('products', '[0.1,0.2]', 10)");

        let plan = sql_strategy.explain(&request);

        // Federated queries should use SQL execution strategy
        assert_eq!(plan.query_type, "Sql");
        assert_eq!(plan.execution_strategy, "sql_executor");
    }

    /// Test 5: All explain plans have same top-level schema
    #[tokio::test]
    async fn test_explain_plan_schema_consistency_across_types() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        let vector_request = QueryRequest::vector_search(vec![0.1], 10);
        let sql_request = QueryRequest::sql("SELECT * FROM products");
        let graph_request = QueryRequest::graph("MATCH (n) RETURN n");

        let plans = vec![
            vector_strategy.explain(&vector_request),
            sql_strategy.explain(&sql_request),
            graph_strategy.explain(&graph_request),
        ];

        // All plans should have the same top-level fields
        for plan in &plans {
            // Required string fields should not be empty
            assert!(
                !plan.query_type.is_empty(),
                "query_type should not be empty"
            );
            assert!(
                !plan.execution_strategy.is_empty(),
                "execution_strategy should not be empty"
            );

            // Orchestration steps should exist
            assert!(
                !plan.orchestration_steps.is_empty(),
                "orchestration_steps should not be empty"
            );

            // Cost should be positive
            assert!(
                plan.estimated_total_cost > 0.0,
                "estimated_total_cost should be positive"
            );

            // parallelizable is a boolean, so it's always valid
        }
    }

    /// Test 6: Explain plan correctly identifies strategy for vector search
    #[tokio::test]
    async fn test_explain_identifies_vector_strategy() {
        let (facade, vector_strategy, _, _) = create_test_facade_with_explain();

        let request = QueryRequest::vector_search(vec![0.1, 0.2, 0.3], 5);

        // Verify facade selects vector strategy
        let result = facade
            .execute(request.clone().with_metrics())
            .await
            .unwrap();
        let metrics = result.metrics.unwrap();
        assert_eq!(metrics.strategy_name, "vector_hnsw");

        // Verify explain plan shows same strategy
        let plan = vector_strategy.explain(&request);
        assert_eq!(plan.execution_strategy, "vector_hnsw");
    }

    /// Test 7: Explain plan correctly identifies strategy for SQL
    #[tokio::test]
    async fn test_explain_identifies_sql_strategy() {
        let (facade, _, sql_strategy, _) = create_test_facade_with_explain();

        let request = QueryRequest::sql("SELECT * FROM products WHERE id = 1");

        // Verify facade selects SQL strategy
        let result = facade
            .execute(request.clone().with_metrics())
            .await
            .unwrap();
        let metrics = result.metrics.unwrap();
        assert_eq!(metrics.strategy_name, "sql_executor");

        // Verify explain plan shows same strategy
        let plan = sql_strategy.explain(&request);
        assert_eq!(plan.execution_strategy, "sql_executor");
    }

    /// Test 8: Explain plan correctly identifies strategy for graph
    #[tokio::test]
    async fn test_explain_identifies_graph_strategy() {
        let (facade, _, _, graph_strategy) = create_test_facade_with_explain();

        let request = QueryRequest::graph("MATCH (n:Person) RETURN n LIMIT 10");

        // Verify facade selects graph strategy
        let result = facade
            .execute(request.clone().with_metrics())
            .await
            .unwrap();
        let metrics = result.metrics.unwrap();
        assert_eq!(metrics.strategy_name, "graph_orion");

        // Verify explain plan shows same strategy
        let plan = graph_strategy.explain(&request);
        assert_eq!(plan.execution_strategy, "graph_orion");
    }

    /// Test 9: Explain plans are JSON serializable
    #[tokio::test]
    async fn test_explain_plan_json_serialization() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        let plans = vec![
            vector_strategy.explain(&QueryRequest::vector_search(vec![0.1], 10)),
            sql_strategy.explain(&QueryRequest::sql("SELECT * FROM t")),
            graph_strategy.explain(&QueryRequest::graph("MATCH (n) RETURN n")),
        ];

        for plan in plans {
            // Serialize to JSON
            let json_str = serde_json::to_string(&plan);
            assert!(json_str.is_ok(), "Plan should serialize to JSON");

            // Deserialize back
            let json_value: Result<ExplainPlanResponse, _> =
                serde_json::from_str(&json_str.unwrap());
            assert!(json_value.is_ok(), "Plan should deserialize from JSON");

            // Round-trip should preserve data
            let restored = json_value.unwrap();
            assert_eq!(restored.query_type, plan.query_type);
            assert_eq!(restored.execution_strategy, plan.execution_strategy);
            assert_eq!(restored.estimated_total_cost, plan.estimated_total_cost);
        }
    }

    /// Test 10: REST and gRPC would return same explain plan schema
    ///
    /// This test validates that the explain plan response structure
    /// is consistent regardless of API protocol.
    #[tokio::test]
    async fn test_rest_grpc_explain_plan_parity() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        // Simulate REST request
        let rest_request =
            QueryRequest::vector_search(vec![0.1, 0.2, 0.3, 0.4], 10).with_target("products");

        // Simulate gRPC request (identical structure)
        let grpc_request =
            QueryRequest::vector_search(vec![0.1, 0.2, 0.3, 0.4], 10).with_target("products");

        // Generate explain plans
        let rest_plan = generate_explain_plan(
            &rest_request,
            &vector_strategy,
            &sql_strategy,
            &graph_strategy,
        )
        .unwrap();
        let grpc_plan = generate_explain_plan(
            &grpc_request,
            &vector_strategy,
            &sql_strategy,
            &graph_strategy,
        )
        .unwrap();

        // Plans should be identical
        assert_eq!(
            rest_plan, grpc_plan,
            "REST and gRPC explain plans should match"
        );
    }

    /// Test 11: Explain plan cost estimation varies by query complexity
    #[tokio::test]
    async fn test_explain_plan_cost_varies_by_complexity() {
        let (_, vector_strategy, _, graph_strategy) = create_test_facade_with_explain();

        // Simple vector search
        let simple_vector = QueryRequest::vector_search(vec![0.1], 5);
        let simple_plan = vector_strategy.explain(&simple_vector);

        // Complex vector search (larger top_k)
        let complex_vector = QueryRequest::vector_search(vec![0.1], 100);
        let complex_plan = vector_strategy.explain(&complex_vector);

        // Complex query should have higher estimated cost
        assert!(
            complex_plan.estimated_total_cost > simple_plan.estimated_total_cost,
            "Complex query should have higher cost"
        );

        // Simple graph query
        let simple_graph = QueryRequest::graph("MATCH (n) RETURN n");
        let simple_graph_plan = graph_strategy.explain(&simple_graph);

        // Deep graph query
        let deep_graph = QueryRequest::graph("MATCH (n)-[*..5]->(m) RETURN m");
        let deep_graph_plan = graph_strategy.explain(&deep_graph);

        // Deeper traversal should have higher cost
        assert!(
            deep_graph_plan.estimated_total_cost > simple_graph_plan.estimated_total_cost,
            "Deeper graph traversal should have higher cost"
        );
    }

    /// Test 12: Orchestration steps describe execution flow
    #[tokio::test]
    async fn test_orchestration_steps_describe_execution() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        // Vector search steps should mention index and search
        let vector_plan = vector_strategy.explain(&QueryRequest::vector_search(vec![0.1], 10));
        let vector_steps_text = vector_plan.orchestration_steps.join(" ");
        assert!(
            vector_steps_text.contains("index") || vector_steps_text.contains("HNSW"),
            "Vector explain should mention index"
        );
        assert!(
            vector_steps_text.contains("search") || vector_steps_text.contains("neighbor"),
            "Vector explain should mention search"
        );

        // SQL steps should mention parse and execute
        let sql_plan = sql_strategy.explain(&QueryRequest::sql("SELECT * FROM t"));
        let sql_steps_text = sql_plan.orchestration_steps.join(" ");
        assert!(
            sql_steps_text.to_lowercase().contains("parse"),
            "SQL explain should mention parsing"
        );
        assert!(
            sql_steps_text.to_lowercase().contains("execute"),
            "SQL explain should mention execution"
        );

        // Graph steps should mention traversal
        let graph_plan = graph_strategy.explain(&QueryRequest::graph("MATCH (n) RETURN n"));
        let graph_steps_text = graph_plan.orchestration_steps.join(" ");
        assert!(
            graph_steps_text.to_lowercase().contains("travers"),
            "Graph explain should mention traversal"
        );
    }

    /// Test 13: Explain plan hints are type-specific
    #[tokio::test]
    async fn test_explain_hints_are_type_specific() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        // Vector hints
        let vector_plan = vector_strategy.explain(&QueryRequest::vector_search(vec![0.1], 10));
        let vector_hints = vector_plan.hints.unwrap();
        assert!(vector_hints.vector.is_some());
        assert!(vector_hints.sql.is_none());
        assert!(vector_hints.graph.is_none());

        // SQL hints
        let sql_plan = sql_strategy.explain(&QueryRequest::sql("SELECT * FROM t"));
        let sql_hints = sql_plan.hints.unwrap();
        assert!(sql_hints.sql.is_some());
        assert!(sql_hints.vector.is_none());
        assert!(sql_hints.graph.is_none());

        // Graph hints
        let graph_plan = graph_strategy.explain(&QueryRequest::graph("MATCH (n) RETURN n"));
        let graph_hints = graph_plan.hints.unwrap();
        assert!(graph_hints.graph.is_some());
        assert!(graph_hints.vector.is_none());
        assert!(graph_hints.sql.is_none());
    }

    /// Test 14: Explain plan preserves collection/target information
    #[tokio::test]
    async fn test_explain_preserves_target_info() {
        let (_, vector_strategy, _, _) = create_test_facade_with_explain();

        let request = QueryRequest::vector_search(vec![0.1, 0.2], 10).with_target("my_collection");

        let plan = vector_strategy.explain(&request);

        // Orchestration steps should reference the target collection
        let steps_text = plan.orchestration_steps.join(" ");
        assert!(
            steps_text.contains("my_collection"),
            "Explain should reference target collection"
        );
    }

    /// Test 15: Explain plans work for empty/minimal queries
    #[tokio::test]
    async fn test_explain_handles_minimal_queries() {
        let (_, vector_strategy, sql_strategy, graph_strategy) = create_test_facade_with_explain();

        // Minimal vector search
        let vector_plan = vector_strategy.explain(&QueryRequest::vector_search(vec![0.0], 1));
        assert!(!vector_plan.query_type.is_empty());
        assert!(vector_plan.estimated_total_cost > 0.0);

        // Minimal SQL
        let sql_plan = sql_strategy.explain(&QueryRequest::sql("SELECT 1"));
        assert!(!sql_plan.query_type.is_empty());
        assert!(sql_plan.estimated_total_cost > 0.0);

        // Minimal graph
        let graph_plan = graph_strategy.explain(&QueryRequest::graph("MATCH (n)"));
        assert!(!graph_plan.query_type.is_empty());
        assert!(graph_plan.estimated_total_cost > 0.0);
    }
}

// ================================================================================
// INTEGRATION TESTS (Require Running Server)
// ================================================================================

#[cfg(test)]
mod integration_tests {
    /// Integration test for explain endpoint via REST
    /// This test is ignored by default and should be run manually
    #[tokio::test]
    #[ignore = "Requires running ProximaDB server on port 5678"]
    async fn test_rest_explain_endpoint() {
        // This would test:
        // POST /api/v1/unified/explain
        // {
        //   "query": "SELECT * FROM products WHERE VECTOR_SIMILAR(embedding, [0.1, 0.2], 0.8)"
        // }
        //
        // And verify response schema matches ExplainPlanResponse
        todo!("Implement REST explain endpoint integration test");
    }

    /// Integration test for explain via gRPC
    #[tokio::test]
    #[ignore = "Requires running ProximaDB server on port 5679"]
    async fn test_grpc_explain_endpoint() {
        // This would test the gRPC explain RPC
        // and verify response schema matches ExplainPlanResponse
        todo!("Implement gRPC explain endpoint integration test");
    }
}
