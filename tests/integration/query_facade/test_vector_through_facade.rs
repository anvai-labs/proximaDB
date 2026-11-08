// Re-enabled 2026-11-08 (compile-disabled since e56a45d90, 2026-05-22): the
// legacy `VectorMatch {id, score, metadata}` fixture shape maps onto the
// canonical `ScoredRecord {record, score, rank}` — the "careful redesign"
// turned out to be mechanical: metadata was fixture-only (never asserted;
// its canonical carrier is `ProximaRecord::props`), and score/rank map 1:1
// with rank assigned by fixture position.
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

//! # Vector Search Through Facade Integration Tests
//!
//! Tests that validate vector search queries route correctly through the
//! `UnifiedQueryFacade` when the `unified-facade-routing` feature is enabled.
//!
//! ## Test Coverage
//!
//! - Vector search request conversion
//! - Strategy selection for vector queries
//! - Response format validation
//! - Error handling for invalid requests
//! - Metrics inclusion in responses

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use proximadb::proto::proximadb_v1::{SearchQuery, VectorSearchRequest};
use proximadb::query::facade::{
    ExecutionMetrics, FacadeConfig, QueryContext, QueryFacadeAdapter, QueryRequest, QueryResult,
    QueryResultData, QueryStrategy, QueryType, ScoredRecord, UnifiedQueryFacade,
};
use proximadb_records::ProximaRecord;

// ================================================================================
// MOCK STRATEGIES FOR TESTING
// ================================================================================

/// Mock vector search strategy that returns predictable results
struct MockVectorSearchStrategy {
    /// Results to return
    results: Vec<ScoredRecord>,
    /// Whether to simulate an error
    should_error: bool,
}

impl MockVectorSearchStrategy {
    fn new() -> Self {
        Self {
            results: vec![
                ScoredRecord {
                    record: ProximaRecord {
                        oid: "vec_001".to_string(),
                        ..Default::default()
                    },
                    score: 0.98,
                    rank: 1,
                },
                ScoredRecord {
                    record: ProximaRecord {
                        oid: "vec_002".to_string(),
                        ..Default::default()
                    },
                    score: 0.92,
                    rank: 2,
                },
                ScoredRecord {
                    record: ProximaRecord {
                        oid: "vec_003".to_string(),
                        ..Default::default()
                    },
                    score: 0.85,
                    rank: 3,
                },
            ],
            should_error: false,
        }
    }

    fn with_results(results: Vec<ScoredRecord>) -> Self {
        Self {
            results,
            should_error: false,
        }
    }

    fn with_error() -> Self {
        Self {
            results: vec![],
            should_error: true,
        }
    }
}

#[async_trait]
impl QueryStrategy for MockVectorSearchStrategy {
    fn name(&self) -> &str {
        "mock_vector"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        request.query_type == QueryType::VectorSearch
    }

    fn priority(&self) -> i32 {
        100
    }

    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        if self.should_error {
            return Err(anyhow::anyhow!("Mock vector search error"));
        }

        // Validate request content
        let (_query_vector, top_k) = match &request.content {
            proximadb::query::facade::QueryContent::Vector {
                query_vector,
                top_k,
            } => (query_vector, *top_k),
            _ => return Err(anyhow::anyhow!("Expected vector content")),
        };

        // Return limited results based on top_k
        let limited_results: Vec<ScoredRecord> = self.results.iter().take(top_k).cloned().collect();

        Ok(QueryResult {
            data: QueryResultData::VectorResults(limited_results.clone()),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: "mock_vector".to_string(),
                execution_time_ms: 5,
                planning_time_ms: 1,
                results_scanned: self.results.len(),
                results_returned: limited_results.len(),
                cache_hit: false,
                extra: serde_json::json!({
                    "mock": true,
                    "engine": "MockVectorSearch"
                }),
            }),
        })
    }
}

// ================================================================================
// TEST UTILITIES
// ================================================================================

/// Create a test facade with mock vector strategy
fn create_test_facade() -> UnifiedQueryFacade {
    let strategies: Vec<Arc<dyn QueryStrategy>> = vec![Arc::new(MockVectorSearchStrategy::new())];
    UnifiedQueryFacade::new(strategies, FacadeConfig::default())
}

/// Create a test facade adapter
fn create_test_adapter() -> QueryFacadeAdapter {
    let facade = Arc::new(create_test_facade());
    QueryFacadeAdapter::new(facade)
}

/// Create a facade with custom results
fn create_facade_with_results(results: Vec<ScoredRecord>) -> QueryFacadeAdapter {
    let strategies: Vec<Arc<dyn QueryStrategy>> =
        vec![Arc::new(MockVectorSearchStrategy::with_results(results))];
    let facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));
    QueryFacadeAdapter::new(facade)
}

/// Create a facade that simulates errors
fn create_error_facade() -> QueryFacadeAdapter {
    let strategies: Vec<Arc<dyn QueryStrategy>> =
        vec![Arc::new(MockVectorSearchStrategy::with_error())];
    let facade = Arc::new(UnifiedQueryFacade::new(strategies, FacadeConfig::default()));
    QueryFacadeAdapter::new(facade)
}

// ================================================================================
// VECTOR SEARCH ROUTING TESTS
// ================================================================================

/// Test that vector search routes through facade with correct conversion
#[tokio::test]
async fn test_vector_search_routes_through_facade() {
    let adapter = create_test_adapter();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let response = adapter.vector_search(request).await.unwrap();

    // Validate response structure
    assert!(response.success, "Response should indicate success");
    assert!(response.results.is_some(), "Should have results");

    let results = response.results.unwrap();
    assert_eq!(results.results.len(), 3, "Should return 3 mock results");

    // Validate result ordering (by score descending)
    assert_eq!(results.results[0].id, "vec_001");
    assert!((results.results[0].score - 0.98).abs() < 0.001);
}

/// Test that facade correctly respects top_k limit
#[tokio::test]
async fn test_vector_search_respects_top_k() {
    let adapter = create_test_adapter();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 2, // Limit to 2 results
        queries: vec![SearchQuery {
            vector: vec![0.1, 0.2, 0.3],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let response = adapter.vector_search(request).await.unwrap();

    assert!(response.success);
    let results = response.results.unwrap();
    assert_eq!(results.results.len(), 2, "Should limit to top_k=2");
}

/// Test empty query vector returns error
#[tokio::test]
async fn test_vector_search_empty_vector_error() {
    let adapter = create_test_adapter();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 10,
        queries: vec![], // No queries
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let result = adapter.vector_search(request).await;

    assert!(result.is_err(), "Should return error for empty queries");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("No query vector"),
        "Error should mention missing vector"
    );
}

/// Test empty vector in query returns error
#[tokio::test]
async fn test_vector_search_empty_vector_in_query_error() {
    let adapter = create_test_adapter();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![], // Empty vector
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let result = adapter.vector_search(request).await;

    assert!(result.is_err(), "Should return error for empty vector");
}

/// Test strategy error propagates correctly
#[tokio::test]
async fn test_vector_search_strategy_error_propagation() {
    let adapter = create_error_facade();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![0.1, 0.2, 0.3],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let result = adapter.vector_search(request).await;

    assert!(result.is_err(), "Should propagate strategy error");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Mock vector search error"),
        "Should contain original error message"
    );
}

// ================================================================================
// RESPONSE FORMAT TESTS
// ================================================================================

/// Test response includes correct score format
#[tokio::test]
async fn test_vector_search_response_format() {
    let adapter = create_test_adapter();

    let request = VectorSearchRequest {
        collection_id: "products".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![0.5; 128], // 128-dim vector
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let response = adapter.vector_search(request).await.unwrap();

    assert!(response.success);
    let results = response.results.unwrap();

    // Check each result has proper format
    for result in &results.results {
        assert!(!result.id.is_empty(), "Result ID should not be empty");
        assert!(result.score >= 0.0, "Score should be non-negative");
        assert!(result.score <= 1.0, "Score should be <= 1.0");

        // Similarity should match score
        if let Some(similarity) = result.similarity {
            assert!((similarity as f64 - result.score).abs() < 0.001);
        }
    }

    // Check total_found
    assert_eq!(results.total_found, results.results.len() as i64);
}

/// Test response with no results
#[tokio::test]
async fn test_vector_search_empty_results() {
    let adapter = create_facade_with_results(vec![]);

    let request = VectorSearchRequest {
        collection_id: "empty_collection".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![0.1, 0.2, 0.3],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let response = adapter.vector_search(request).await.unwrap();

    assert!(response.success);
    let results = response.results.unwrap();
    assert_eq!(results.results.len(), 0, "Should return empty results");
    assert_eq!(results.total_found, 0);
}

// ================================================================================
// UNIFIED FACADE DIRECT TESTS
// ================================================================================

/// Test facade directly with QueryRequest
#[tokio::test]
async fn test_facade_vector_search_directly() {
    let facade = create_test_facade();

    let request = QueryRequest::vector_search(vec![0.1, 0.2, 0.3], 5)
        .with_target("test_collection")
        .with_metrics();

    let result = facade.execute(request).await.unwrap();

    // Verify execution path
    let metrics = result.metrics.expect("Should have metrics");
    assert_eq!(metrics.execution_path, "unified");
    assert_eq!(metrics.strategy_name, "mock_vector");

    // Verify result data
    assert!(matches!(result.data, QueryResultData::VectorResults(_)));
    if let QueryResultData::VectorResults(matches) = result.data {
        assert!(matches.len() <= 5, "Should respect top_k limit");
    }
}

/// Test facade strategy selection
#[tokio::test]
async fn test_facade_selects_vector_strategy() {
    let facade = create_test_facade();

    // Vector search should select mock_vector strategy
    let request = QueryRequest::vector_search(vec![0.1], 10).with_metrics();
    let result = facade.execute(request).await.unwrap();

    let metrics = result.metrics.unwrap();
    assert_eq!(metrics.strategy_name, "mock_vector");
}

/// Test facade with no matching strategy
#[tokio::test]
async fn test_facade_no_matching_strategy() {
    // Create facade with only vector strategy
    let facade = create_test_facade();

    // SQL query should fail - no SQL strategy registered
    let request = QueryRequest::sql("SELECT * FROM products");
    let result = facade.execute(request).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("No strategy found"),
        "Should indicate no strategy found"
    );
}

// ================================================================================
// ADAPTER LIFECYCLE TESTS
// ================================================================================

/// Test adapter can be cloned and shared
#[tokio::test]
async fn test_adapter_clone_and_share() {
    let adapter = create_test_adapter();
    let cloned = adapter.clone();

    // Both should reference the same facade
    assert!(Arc::ptr_eq(adapter.facade(), cloned.facade()));

    // Both should work independently
    let request1 = VectorSearchRequest {
        collection_id: "col1".to_string(),
        top_k: 5,
        queries: vec![SearchQuery {
            vector: vec![0.1],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let request2 = VectorSearchRequest {
        collection_id: "col2".to_string(),
        top_k: 3,
        queries: vec![SearchQuery {
            vector: vec![0.2],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    let (result1, result2) = tokio::join!(
        adapter.vector_search(request1),
        cloned.vector_search(request2)
    );

    assert!(result1.is_ok());
    assert!(result2.is_ok());
}

/// Test adapter facade reference counting
#[test]
fn test_adapter_facade_reference() {
    let adapter = create_test_adapter();

    // Check we can get facade reference
    let facade = adapter.facade();
    assert!(Arc::strong_count(facade) >= 1);

    // Clone increases reference count
    let cloned = adapter.clone();
    assert!(Arc::strong_count(cloned.facade()) >= 2);
}

// ================================================================================
// CONCURRENT REQUEST TESTS
// ================================================================================

/// Test multiple concurrent vector searches
#[tokio::test]
async fn test_concurrent_vector_searches() {
    let adapter = create_test_adapter();

    // Create 10 concurrent search requests
    let mut handles = Vec::new();
    for i in 0..10 {
        let adapter_clone = adapter.clone();
        let handle = tokio::spawn(async move {
            let request = VectorSearchRequest {
                collection_id: format!("collection_{}", i),
                top_k: 5,
                queries: vec![SearchQuery {
                    vector: vec![i as f32 / 10.0; 128],
                    filters: Default::default(),
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

    // All requests should succeed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent request should succeed");
    }
}

// ================================================================================
// FEATURE FLAG VALIDATION
// ================================================================================

/// Test that validates the unified-facade-routing concept
/// This test documents the expected behavior when the feature is enabled
#[tokio::test]
async fn test_unified_facade_routing_concept() {
    // When unified-facade-routing is enabled:
    // 1. All vector search requests should go through QueryFacadeAdapter
    // 2. Adapter converts proto request to QueryRequest
    // 3. Facade selects appropriate strategy
    // 4. Strategy executes and returns QueryResult
    // 5. Adapter converts back to proto response

    let adapter = create_test_adapter();

    // Simulate REST/gRPC handler receiving a request
    let proto_request = VectorSearchRequest {
        collection_id: "my_vectors".to_string(),
        top_k: 10,
        queries: vec![SearchQuery {
            vector: vec![0.1, 0.2, 0.3, 0.4],
            filters: Default::default(),
            advanced_filter: None,
        }],
        include_fields: None,
        search_params: None,
        distance_metric_override: None,
        search_optimization: None,
    };

    // Handler routes through adapter (this is what happens with feature enabled)
    let proto_response = adapter.vector_search(proto_request).await.unwrap();

    // Verify the full round-trip
    assert!(proto_response.success);
    assert!(proto_response.results.is_some());

    // Verify operation type is search
    assert_eq!(proto_response.operation, 1); // 1 = Search
}
