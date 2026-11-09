// Re-enabled 2026-11-08: this suite was compile-disabled (`#![cfg(any())]`)
// since the 2026-05-22 nomenclature migration (e56a45d90). The rewrite maps
// the legacy `VectorMatch {id, score, metadata}` fixture shape onto the
// canonical `ScoredRecord {record, score, rank}` (metadata was fixture-only,
// never asserted; its canonical carrier is `ProximaRecord::props`), and
// `ExecuteSqlResponse` onto its WS7 rename `ExecuteQueryResponse`.
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

//! REST/gRPC Facade Consistency Tests
//!
//! **Honest scope (2026-11-08 re-enable review):** these tests call the SAME
//! in-process `QueryFacadeAdapter` method twice and assert identical results —
//! they verify facade determinism/idempotence and response-schema shape, NOT
//! REST-vs-gRPC wire behavior (no HTTP, no gRPC, no protocol label is
//! exercised). This was true before the 2026-05-22 disable as well. A mutation
//! probe corrupting mock fixtures leaves the comparisons green — their teeth
//! are the schema/error assertions, not the comparison. Real wire parity lives
//! in `tests/rest_grpc_parity_test.rs` (live-server, manual-only); the
//! automated dual-transport RENDERING parity test now exists next door
//! (`test_rest_grpc_rendering_parity.rs`).
//!
//! ## Test Coverage
//!
//! 1. **Facade consistency**: the same request through the adapter produces
//!    identical results on repeated invocation
//! 2. **Response schema shape**: result structures match the expected envelopes
//! 3. **Error paths**: invalid requests are rejected by the adapter
//!
//! ## Architecture
//!
//! ```text
//! request --> QueryFacadeAdapter --> UnifiedQueryFacade (mock strategies) --> Result
//! (twice; assert equal)
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

// Proto types for request/response structures
use proximadb::proto::proximadb_v1::{
    self, ExecuteQueryResponse, SearchQuery, SqlValue, VectorOperationResponse, VectorSearchRequest,
};

// Query facade types
use proximadb::query::{
    FacadeConfig, GraphQueryResult, QueryContext, QueryFacadeAdapter, QueryRequest, QueryResult,
    QueryResultData, QueryStrategy, UnifiedQueryFacade,
};
use proximadb_records::ProximaRecord;

// =============================================================================
// Test Helpers
// =============================================================================

/// Represents a normalized search result for comparison
#[derive(Debug, Clone, PartialEq)]
struct NormalizedSearchResult {
    id: String,
    score: f64,
}

/// Represents a normalized SQL result row for comparison
#[derive(Debug, Clone, PartialEq)]
struct NormalizedSqlRow {
    columns: HashMap<String, serde_json::Value>,
}

/// Extract normalized results from VectorOperationResponse
fn normalize_vector_response(response: &VectorOperationResponse) -> Vec<NormalizedSearchResult> {
    response
        .results
        .as_ref()
        .map(|r| {
            r.results
                .iter()
                .map(|record| NormalizedSearchResult {
                    id: record.id.clone(),
                    score: record.score,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Extract normalized results from ExecuteQueryResponse (gRPC; renamed from
/// ExecuteSqlResponse in WS7 — 279b1e5c3)
fn normalize_grpc_sql_response(response: &ExecuteQueryResponse) -> Vec<NormalizedSqlRow> {
    response
        .rows
        .iter()
        .map(|row| NormalizedSqlRow {
            columns: row
                .fields
                .iter()
                .map(|field| {
                    let value = field
                        .value
                        .as_ref()
                        .map(sql_value_to_json)
                        .unwrap_or(serde_json::Value::Null);
                    (field.key.clone(), value)
                })
                .collect(),
        })
        .collect()
}

// SqlValue → JSON via the canonical records conversion. This file's private
// copy (which predated the JsonbValue variant and drifted on BytesValue /
// non-finite-float rendering) was deleted at the 2026-11-08 re-enable.
// Note: the only current consumer is the retained `normalize_grpc_sql_response`
// helper (kept for future live-server parity tests — see the module doc);
// no executed test path renders SqlValue today.
use proximadb_records::conversions::sql_value_to_json;

/// Compare two sets of normalized results with tolerance for floating point scores
fn compare_search_results(
    rest_results: &[NormalizedSearchResult],
    grpc_results: &[NormalizedSearchResult],
    score_tolerance: f64,
) -> bool {
    if rest_results.len() != grpc_results.len() {
        return false;
    }

    for (rest, grpc) in rest_results.iter().zip(grpc_results.iter()) {
        if rest.id != grpc.id {
            return false;
        }
        if (rest.score - grpc.score).abs() > score_tolerance {
            return false;
        }
    }

    true
}

/// Compare two sets of SQL result rows
fn compare_sql_results(
    rest_results: &[NormalizedSqlRow],
    grpc_results: &[NormalizedSqlRow],
) -> bool {
    if rest_results.len() != grpc_results.len() {
        return false;
    }

    for (rest, grpc) in rest_results.iter().zip(grpc_results.iter()) {
        if rest.columns.len() != grpc.columns.len() {
            return false;
        }
        for (key, rest_val) in &rest.columns {
            match grpc.columns.get(key) {
                Some(grpc_val) => {
                    if !json_values_equal(rest_val, grpc_val) {
                        return false;
                    }
                }
                None => return false,
            }
        }
    }

    true
}

/// Compare JSON values with tolerance for floating point
fn json_values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;

    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64().unwrap_or(0.0);
            let fb = nb.as_f64().unwrap_or(0.0);
            (fa - fb).abs() < 1e-6
        }
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Bool(ba), Value::Bool(bb)) => ba == bb,
        (Value::Null, Value::Null) => true,
        (Value::Array(aa), Value::Array(ab)) => {
            aa.len() == ab.len()
                && aa
                    .iter()
                    .zip(ab.iter())
                    .all(|(a, b)| json_values_equal(a, b))
        }
        (Value::Object(oa), Value::Object(ob)) => {
            oa.len() == ob.len()
                && oa.iter().all(|(k, v)| {
                    ob.get(k)
                        .map(|ov| json_values_equal(v, ov))
                        .unwrap_or(false)
                })
        }
        _ => false,
    }
}

// =============================================================================
// Mock Strategies for Testing
// =============================================================================

/// Mock vector search strategy for testing parity
struct MockVectorStrategy {
    results: Vec<(String, f64)>,
}

impl MockVectorStrategy {
    fn new() -> Self {
        Self {
            results: vec![
                ("vec_001".to_string(), 0.95),
                ("vec_002".to_string(), 0.87),
                ("vec_003".to_string(), 0.82),
                ("vec_004".to_string(), 0.76),
                ("vec_005".to_string(), 0.71),
            ],
        }
    }
}

#[async_trait::async_trait]
impl QueryStrategy for MockVectorStrategy {
    fn name(&self) -> &str {
        "mock_vector"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(
            request.query_type,
            proximadb::query::QueryType::VectorSearch
        )
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, _request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let matches: Vec<proximadb::query::ScoredRecord> = self
            .results
            .iter()
            .enumerate()
            .map(|(rank, (id, score))| proximadb::query::ScoredRecord {
                record: ProximaRecord {
                    oid: id.clone(),
                    ..Default::default()
                },
                score: *score as f32,
                rank: (rank + 1) as u32,
            })
            .collect();

        Ok(QueryResult {
            data: QueryResultData::VectorResults(matches),
            metrics: None,
        })
    }
}

/// Mock SQL strategy for testing parity
struct MockSqlStrategy {
    results: Vec<HashMap<String, serde_json::Value>>,
}

impl MockSqlStrategy {
    fn new() -> Self {
        Self {
            results: vec![
                {
                    let mut row = HashMap::new();
                    row.insert("id".to_string(), json!("row_001"));
                    row.insert("name".to_string(), json!("Product A"));
                    row.insert("price".to_string(), json!(99.99));
                    row
                },
                {
                    let mut row = HashMap::new();
                    row.insert("id".to_string(), json!("row_002"));
                    row.insert("name".to_string(), json!("Product B"));
                    row.insert("price".to_string(), json!(149.99));
                    row
                },
            ],
        }
    }
}

#[async_trait::async_trait]
impl QueryStrategy for MockSqlStrategy {
    fn name(&self) -> &str {
        "mock_sql"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(request.query_type, proximadb::query::QueryType::Sql)
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, _request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let rows: Vec<serde_json::Value> = self
            .results
            .iter()
            .map(|row| {
                serde_json::Value::Object(row.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            })
            .collect();

        Ok(QueryResult {
            data: QueryResultData::Rows(rows),
            metrics: None,
        })
    }
}

/// Mock graph strategy for testing parity
struct MockGraphStrategy {
    nodes: Vec<String>,
}

impl MockGraphStrategy {
    fn new() -> Self {
        Self {
            nodes: vec![
                "node_001".to_string(),
                "node_002".to_string(),
                "node_003".to_string(),
            ],
        }
    }
}

#[async_trait::async_trait]
impl QueryStrategy for MockGraphStrategy {
    fn name(&self) -> &str {
        "mock_graph"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        matches!(request.query_type, proximadb::query::QueryType::Graph)
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, _request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        Ok(QueryResult {
            data: QueryResultData::Graph(GraphQueryResult {
                nodes: self
                    .nodes
                    .iter()
                    .map(|id| json!({ "id": id, "label": "TestNode" }))
                    .collect(),
                edges: vec![],
                paths: vec![],
            }),
            metrics: None,
        })
    }
}

/// Create a test adapter with mock strategies
fn create_test_adapter() -> QueryFacadeAdapter {
    let strategies: Vec<Arc<dyn QueryStrategy>> = vec![
        Arc::new(MockVectorStrategy::new()),
        Arc::new(MockSqlStrategy::new()),
        Arc::new(MockGraphStrategy::new()),
    ];

    let facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));
    QueryFacadeAdapter::new(facade)
}

// =============================================================================
// Parity Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Vector search via REST produces same results as via gRPC
    ///
    /// Both protocols use QueryFacadeAdapter.vector_search() which routes
    /// through UnifiedQueryFacade, so results should be identical.
    #[tokio::test]
    async fn test_vector_search_parity() {
        let adapter = create_test_adapter();

        // Simulate REST request
        let rest_request = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 5,
            queries: vec![SearchQuery {
                vector: vec![0.1, 0.2, 0.3, 0.4],
                filters: HashMap::new(),
                advanced_filter: None,
            }],
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        // Simulate gRPC request (identical structure)
        let grpc_request = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 5,
            queries: vec![SearchQuery {
                vector: vec![0.1, 0.2, 0.3, 0.4],
                filters: HashMap::new(),
                advanced_filter: None,
            }],
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        // Execute both through the adapter
        let rest_response = adapter
            .vector_search(rest_request)
            .await
            .expect("REST search failed");
        let grpc_response = adapter
            .vector_search(grpc_request)
            .await
            .expect("gRPC search failed");

        // Normalize and compare results
        let rest_results = normalize_vector_response(&rest_response);
        let grpc_results = normalize_vector_response(&grpc_response);

        assert!(
            compare_search_results(&rest_results, &grpc_results, 1e-6),
            "Vector search results should be identical between REST and gRPC\nREST: {:?}\ngRPC: {:?}",
            rest_results,
            grpc_results
        );

        // Verify response schema consistency
        assert_eq!(
            rest_response.success, grpc_response.success,
            "Success status should match"
        );
        assert_eq!(
            rest_response.operation, grpc_response.operation,
            "Operation type should match"
        );
    }

    /// Test 2: SQL queries via REST produce same results as via gRPC
    ///
    /// Both protocols use QueryFacadeAdapter.sql_query() which routes
    /// through UnifiedQueryFacade, so results should be identical.
    #[tokio::test]
    async fn test_sql_query_parity() {
        let adapter = create_test_adapter();

        let sql = "SELECT * FROM products WHERE category = 'electronics' LIMIT 10";

        // Execute through adapter (same path for both REST and gRPC when facade routing is enabled)
        let rest_result = adapter.sql_query(sql).await.expect("REST SQL query failed");
        let grpc_result = adapter.sql_query(sql).await.expect("gRPC SQL query failed");

        // Compare QueryResult data
        match (&rest_result.data, &grpc_result.data) {
            (QueryResultData::Rows(rest_rows), QueryResultData::Rows(grpc_rows)) => {
                assert_eq!(
                    rest_rows.len(),
                    grpc_rows.len(),
                    "SQL query should return same number of rows"
                );

                for (rest_row, grpc_row) in rest_rows.iter().zip(grpc_rows.iter()) {
                    assert!(
                        json_values_equal(rest_row, grpc_row),
                        "SQL row data should match\nREST: {:?}\ngRPC: {:?}",
                        rest_row,
                        grpc_row
                    );
                }
            }
            _ => panic!("Expected Rows result type for SQL query"),
        }
    }

    /// Test 3: Graph queries via REST produce same results as via gRPC
    ///
    /// Both protocols use QueryFacadeAdapter.graph_query() which routes
    /// through UnifiedQueryFacade, so results should be identical.
    #[tokio::test]
    async fn test_graph_query_parity() {
        let adapter = create_test_adapter();

        let cypher = "MATCH (n:Person) RETURN n LIMIT 10";
        let graph_name = Some("test_graph");

        // Execute through adapter
        let rest_result = adapter
            .graph_query(cypher, graph_name)
            .await
            .expect("REST graph query failed");
        let grpc_result = adapter
            .graph_query(cypher, graph_name)
            .await
            .expect("gRPC graph query failed");

        // Compare QueryResult data
        match (&rest_result.data, &grpc_result.data) {
            (QueryResultData::Graph(rest_graph), QueryResultData::Graph(grpc_graph)) => {
                assert_eq!(
                    rest_graph.nodes.len(),
                    grpc_graph.nodes.len(),
                    "Graph query should return same number of nodes"
                );
                assert_eq!(
                    rest_graph.edges.len(),
                    grpc_graph.edges.len(),
                    "Graph query should return same number of edges"
                );

                // Compare node data
                for (rest_node, grpc_node) in rest_graph.nodes.iter().zip(grpc_graph.nodes.iter()) {
                    assert!(
                        json_values_equal(rest_node, grpc_node),
                        "Graph node data should match\nREST: {:?}\ngRPC: {:?}",
                        rest_node,
                        grpc_node
                    );
                }
            }
            _ => panic!("Expected Graph result type for graph query"),
        }
    }

    /// Test 4: Response schemas are consistent between protocols
    ///
    /// This verifies that the response structures produced by both protocols
    /// have consistent field names and types.
    #[tokio::test]
    async fn test_response_schema_consistency() {
        let adapter = create_test_adapter();

        // Test vector search response schema
        let search_request = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 3,
            queries: vec![SearchQuery {
                vector: vec![0.1, 0.2, 0.3],
                filters: HashMap::new(),
                advanced_filter: None,
            }],
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        let response = adapter
            .vector_search(search_request)
            .await
            .expect("Vector search failed");

        // Verify required fields are present
        assert!(
            response.results.is_some(),
            "Results field should be present"
        );

        let results = response.results.unwrap();
        assert!(!results.results.is_empty(), "Results should not be empty");

        // Verify each result has required fields
        for result in &results.results {
            assert!(!result.id.is_empty(), "Result ID should not be empty");
            // Score should be a valid number
            assert!(result.score.is_finite(), "Score should be a finite number");
        }
    }

    /// Test 5: Empty results are handled consistently
    #[tokio::test]
    async fn test_empty_results_parity() {
        // Create adapter with strategies that return empty results
        struct EmptyVectorStrategy;

        #[async_trait::async_trait]
        impl QueryStrategy for EmptyVectorStrategy {
            fn name(&self) -> &str {
                "empty_vector"
            }

            fn can_handle(&self, request: &QueryRequest) -> bool {
                matches!(
                    request.query_type,
                    proximadb::query::QueryType::VectorSearch
                )
            }

            fn priority(&self) -> i32 {
                100
            }

            async fn execute(
                &self,
                _request: QueryRequest,
                _ctx: &QueryContext,
            ) -> Result<QueryResult> {
                Ok(QueryResult {
                    data: QueryResultData::VectorResults(vec![]),
                    metrics: None,
                })
            }
        }

        let strategies: Vec<Arc<dyn QueryStrategy>> = vec![Arc::new(EmptyVectorStrategy)];
        let facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));
        let adapter = QueryFacadeAdapter::new(facade);

        let request = VectorSearchRequest {
            collection_id: "empty_collection".to_string(),
            top_k: 10,
            queries: vec![SearchQuery {
                vector: vec![0.1, 0.2],
                filters: HashMap::new(),
                advanced_filter: None,
            }],
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        let response = adapter.vector_search(request).await.expect("Search failed");

        // Verify empty results are handled correctly
        assert!(
            response.success,
            "Success should be true even for empty results"
        );
        assert!(
            response.results.is_some(),
            "Results field should be present"
        );
        assert!(
            response.results.as_ref().unwrap().results.is_empty(),
            "Results should be empty"
        );
    }

    /// Test 6: Error handling is consistent between protocols
    #[tokio::test]
    async fn test_error_handling_parity() {
        let adapter = create_test_adapter();

        // Request with empty vector should produce consistent error
        let request = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 10,
            queries: vec![], // No query vectors
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        let result = adapter.vector_search(request).await;

        // Should produce an error for missing query vector
        assert!(
            result.is_err(),
            "Empty query vector should produce an error"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("No query vector")
                || err.to_string().contains("empty")
                || err.to_string().contains("required"),
            "Error message should indicate missing query vector: {}",
            err
        );
    }

    /// Test 7: Federated query parity
    ///
    /// Federated queries (SQL with multi-model extensions) should produce
    /// identical results when routed through the facade.
    #[tokio::test]
    async fn test_federated_query_parity() {
        let adapter = create_test_adapter();

        let sql = "SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2, 0.3]', 10)";

        // Execute through federated_query path
        let result = adapter.federated_query(sql).await;

        // The result type depends on the registered strategies
        match result {
            Ok(query_result) => {
                match query_result.data {
                    QueryResultData::Rows(_)
                    | QueryResultData::VectorResults(_)
                    | QueryResultData::Empty => {
                        // Valid result types for federated queries
                    }
                    _ => {
                        // Other types are also acceptable depending on strategy
                    }
                }
            }
            Err(e) => {
                // It's acceptable if no strategy handles federated queries
                assert!(
                    e.to_string().contains("No strategy found")
                        || e.to_string().contains("not implemented")
                        || e.to_string().contains("unsupported"),
                    "Unexpected error: {}",
                    e
                );
            }
        }
    }

    /// Test 8: Multiple concurrent requests produce consistent results
    #[tokio::test]
    async fn test_concurrent_parity() {
        let adapter = Arc::new(create_test_adapter());

        let mut handles = Vec::new();

        for i in 0..10 {
            let adapter_clone = adapter.clone();
            let handle = tokio::spawn(async move {
                let request = VectorSearchRequest {
                    collection_id: format!("collection_{}", i),
                    top_k: 5,
                    queries: vec![SearchQuery {
                        vector: vec![0.1 * (i as f32), 0.2, 0.3],
                        filters: HashMap::new(),
                        advanced_filter: None,
                    }],
                    include_fields: None,
                    search_params: None,
                    distance_metric_override: None,
                    search_optimization: None,
                };

                adapter_clone.vector_search(request).await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        for handle in handles {
            let result = handle.await.expect("Task panicked");
            // All requests should succeed (mock strategy always returns same results)
            assert!(
                result.is_ok(),
                "Concurrent request failed: {:?}",
                result.err()
            );
        }
    }

    /// Test 9: Request metadata propagation is consistent
    #[tokio::test]
    async fn test_metadata_propagation_parity() {
        let adapter = create_test_adapter();

        // Create requests with different search parameters
        let request_with_params = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 5,
            queries: vec![SearchQuery {
                vector: vec![0.1, 0.2, 0.3],
                filters: {
                    let mut filters = HashMap::new();
                    filters.insert(
                        "category".to_string(),
                        SqlValue {
                            value: Some(proximadb_v1::sql_value::Value::StringValue(
                                "electronics".to_string(),
                            )),
                        },
                    );
                    filters
                },
                advanced_filter: None,
            }],
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        let response = adapter
            .vector_search(request_with_params)
            .await
            .expect("Search with filters failed");

        // Verify the search completed successfully
        assert!(response.success, "Search with filters should succeed");
    }
}

// =============================================================================
// Integration Tests (require running server)
// =============================================================================

#[cfg(test)]
mod integration_tests {
    /// Integration test that requires a running ProximaDB server
    /// This test is ignored by default and should be run manually
    #[tokio::test]
    #[ignore = "Requires running ProximaDB server on ports 5678 (REST) and 5679 (gRPC)"]
    async fn test_live_rest_grpc_parity() {
        // This test would:
        // 1. Create a collection via REST
        // 2. Insert test vectors
        // 3. Execute vector search via REST
        // 4. Execute same search via gRPC
        // 5. Compare results

        // Implementation left as placeholder since it requires running server
        todo!("Implement live server parity test");
    }
}

// Suppress unused warnings for helper functions that may be used in future tests
#[allow(dead_code)]
fn _unused_helpers() {
    let _ = normalize_grpc_sql_response;
    let _ = compare_sql_results;
}
