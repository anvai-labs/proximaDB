//! # Unified Multi-Model Query API
//!
//! REST API for cross-model queries combining all ProximaDB data models.
//!
//! ## Available Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/api/v1/unified/query` | POST | Structured multi-model query |
//! | `/api/v1/unified/federated` | POST | SQL with extensions (VECTOR_SEARCH, GRAPH_QUERY, etc.) |
//!
//! ## Query Paths
//!
//! This module provides two query paths:
//!
//! ### 1. Structured Unified Queries (`/query`)
//!
//! Uses `QueryDecomposer` to break down a structured query into sub-queries for each model,
//! execute them in parallel, and fuse the results.
//!
//! ```json
//! {
//!   "vector": { "collection": "embeddings", "query": [0.1, 0.2], "top_k": 10 },
//!   "graph": { "cypher": "MATCH (n)-[:RELATED]->(m) RETURN m" },
//!   "document": { "collection": "docs", "filter": "type = 'article'" },
//!   "fusion_strategy": "rrf"
//! }
//! ```
//!
//! ### 2. Federated SQL Queries (`/federated`)
//!
//! Uses `FederatedQueryContext` to parse and execute SQL with multi-model extensions.
//! Independent function-backed sources are executable; correlated/LATERAL joins are
//! still reported as unsupported on the live path.
//!
//! ```json
//! {
//!   "query": "SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2]', 10) v JOIN GRAPH_QUERY('MATCH (p)-[:CATEGORY]->(c) RETURN c.name') g ON v.id = g.product_id"
//! }
//! ```
//!
//! ## SQL Extensions
//!
//! - `VECTOR_SEARCH(collection, query_vector, top_k)` - Vector similarity search
//! - `GRAPH_QUERY('cypher')` - Graph traversal via Cypher
//! - `DOCUMENT_QUERY(collection, filter)` - Document queries
//! - `LOGS(namespace)` / `METRICS(namespace)` - Observability queries
//! - `<->` operator - pgvector-compatible distance operator
//!
//! ## Example cURL
//!
//! ```bash
//! # Federated SQL query
//! curl -X POST http://localhost:5678/api/v1/unified/federated \
//!   -H "Content-Type: application/json" \
//!   -d '{"query": "SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1, 0.2]', 10)"}'
//!
//! # Structured query
//! curl -X POST http://localhost:5678/api/v1/unified/query \
//!   -H "Content-Type: application/json" \
//!   -d '{"vector": {"collection": "embeddings", "query": [0.1, 0.2], "top_k": 10}}'
//! ```

use axum::{
    Router,
    extract::{Json, Path, State},
    response::Json as JsonResponse,
    routing::{delete, post},
};
use proximadb_graph_query::service::GraphQueryService;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::errors::{ApiError, ApiResult};
use crate::observability::ObservabilityService;
use crate::query::QueryFacadeAdapter;
use crate::query::authority_context::{AuthoritySource, resolve_catalog_authority_context};
use crate::query::explain::StorageAuthorityExplanation;
use crate::query::federated::{
    FederatedParser, FederatedQueryContext, QueryType as FederatedQueryType,
};
use crate::query::multimodal::plan::PlanContext;
use crate::query::prepared::{ParameterValue, PreparedStatementCache, PreparedStatementConfig};
use crate::query::unified::executor::ParallelExecutor;
use crate::query::unified::{
    DataModel, FusionStrategy, QueryDecomposer, ResultFuser, UnifiedQueryConfig,
};
use crate::security::rbac_service::ConsolidatedRBACManager;
use crate::services::VectorOperationsService;
use crate::storage::document::DocumentService;
use crate::storage::traits::UnifiedStorageFormat;
use proximadb_vector_query::VectorQueryService;

/// Unified Query API state with all services for cross-model queries
///
/// # Architecture Modes
///
/// This state supports two execution modes:
///
/// ## 1. Adapter Mode (Recommended when `unified-facade-routing` is enabled)
///
/// When `query_adapter` is set, all queries route through `QueryFacadeAdapter`:
/// - Simpler configuration: only `query_adapter` is needed
/// - Consistent execution across REST, gRPC, and internal paths
/// - Use `new_with_adapter()` constructor
///
/// ## 2. Legacy Mode (Fallback)
///
/// When `query_adapter` is None, queries use the internal pipeline:
/// - Requires: `decomposer`, `executor`, `fuser`, `config`
/// - Services: `vector_ops`, `document_service`, `graph_service`, `observability_service`
/// - Optional: `federated_context` for SQL with multi-model extensions
///
/// The legacy mode will be deprecated once adapter mode is proven stable.
#[derive(Clone)]
pub struct UnifiedQueryApiState {
    // =========================================================================
    // ADAPTER MODE (preferred) - Only query_adapter is needed
    // =========================================================================
    /// Query facade adapter for unified query execution
    ///
    /// When set, all queries route through this adapter, making the legacy
    /// fields below unnecessary. This is the preferred mode.
    pub query_adapter: Option<Arc<QueryFacadeAdapter>>,

    /// Optional catalog manager for EXPLAIN storage-authority metadata.
    pub catalog_manager: Option<Arc<crate::catalog::CatalogManager>>,

    // =========================================================================
    // SECURITY - RBAC Manager for permission validation
    // =========================================================================
    /// RBAC manager for validating query permissions
    pub rbac_manager: Option<Arc<ConsolidatedRBACManager>>,

    // =========================================================================
    // PREPARED STATEMENTS - Thread-safe statement cache
    // =========================================================================
    /// Prepared statement cache for parse-once-execute-many pattern
    ///
    /// Provides significant performance improvement for agentic AI workloads
    /// with repetitive query patterns.
    pub prepared_statement_cache: Arc<PreparedStatementCache>,

    // =========================================================================
    // LEGACY MODE - Used when query_adapter is None
    // =========================================================================
    /// Document service for document queries (legacy mode)
    pub document_service: Arc<DocumentService>,
    /// Storage engine for vector queries (legacy mode)
    pub storage_engine: Arc<dyn UnifiedStorageFormat>,
    /// Vector operations service for vector searches (legacy mode)
    pub vector_ops: Option<Arc<VectorOperationsService>>,
    /// Vector query service trait object (Phase 2.3 - preferred interface)
    pub vector_query_service: Option<Arc<dyn VectorQueryService>>,
    /// Graph query/traversal service for graph requests (legacy mode)
    pub graph_service: Option<Arc<dyn GraphQueryService>>,
    /// Observability service for logs/metrics (legacy mode)
    pub observability_service: Option<Arc<ObservabilityService>>,
    /// Query decomposer (legacy mode)
    pub decomposer: Arc<QueryDecomposer>,
    /// Parallel executor for multi-model queries (legacy mode)
    pub executor: Arc<ParallelExecutor>,
    /// Result fuser (legacy mode)
    pub fuser: Arc<ResultFuser>,
    /// Configuration (legacy mode)
    pub config: UnifiedQueryConfig,
    /// Federated query context for SQL with multi-model extensions (legacy mode)
    pub federated_context: Option<Arc<FederatedQueryContext>>,
    /// Federated parser for detecting multi-model query patterns (legacy mode)
    #[allow(dead_code)]
    federated_parser: Arc<FederatedParser>,
}

impl UnifiedQueryApiState {
    /// Create a new state with only the query adapter (preferred mode)
    ///
    /// When using this constructor, all queries route through the adapter.
    /// The legacy fields are initialized with placeholder values that won't be used.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let adapter = Arc::new(QueryFacadeAdapter::new(facade));
    /// let state = UnifiedQueryApiState::new_with_adapter(adapter, document_service, storage_engine);
    /// ```
    pub fn new_with_adapter(
        adapter: Arc<QueryFacadeAdapter>,
        document_service: Arc<DocumentService>,
        storage_engine: Arc<dyn UnifiedStorageFormat>,
    ) -> Self {
        let config = UnifiedQueryConfig::default();
        Self {
            query_adapter: Some(adapter),
            catalog_manager: None,
            rbac_manager: None, // RBAC manager can be added later
            // Prepared statement cache with default configuration
            prepared_statement_cache: Arc::new(PreparedStatementCache::new(
                PreparedStatementConfig::default(),
            )),
            // Legacy fields (not used when adapter is present, but required for struct)
            document_service,
            storage_engine,
            vector_ops: None,
            vector_query_service: None,
            graph_service: None,
            observability_service: None,
            decomposer: Arc::new(QueryDecomposer::new()),
            executor: Arc::new(ParallelExecutor::new(1)), // Minimal, not used
            fuser: Arc::new(ResultFuser::new(config.default_fusion.clone())),
            config,
            federated_context: None,
            federated_parser: Arc::new(FederatedParser::new()),
        }
    }

    /// Create a new state with custom prepared statement cache configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let prepared_config = PreparedStatementConfig {
    ///     max_statements: 5000,
    ///     default_ttl: Duration::from_secs(7200), // 2 hours
    ///     ..Default::default()
    /// };
    /// let state = UnifiedQueryApiState::new_with_adapter_and_prepared_config(
    ///     adapter, document_service, storage_engine, prepared_config
    /// );
    /// ```
    pub fn new_with_adapter_and_prepared_config(
        adapter: Arc<QueryFacadeAdapter>,
        document_service: Arc<DocumentService>,
        storage_engine: Arc<dyn UnifiedStorageFormat>,
        prepared_config: PreparedStatementConfig,
    ) -> Self {
        let config = UnifiedQueryConfig::default();
        Self {
            query_adapter: Some(adapter),
            catalog_manager: None,
            rbac_manager: None,
            prepared_statement_cache: Arc::new(PreparedStatementCache::new(prepared_config)),
            document_service,
            storage_engine,
            vector_ops: None,
            vector_query_service: None,
            graph_service: None,
            observability_service: None,
            decomposer: Arc::new(QueryDecomposer::new()),
            executor: Arc::new(ParallelExecutor::new(1)),
            fuser: Arc::new(ResultFuser::new(config.default_fusion.clone())),
            config,
            federated_context: None,
            federated_parser: Arc::new(FederatedParser::new()),
        }
    }

    /// Set RBAC manager for permission validation
    pub fn with_rbac_manager(mut self, rbac_manager: Arc<ConsolidatedRBACManager>) -> Self {
        self.rbac_manager = Some(rbac_manager);
        self
    }

    /// Attach the extracted vector query service contract for legacy REST state.
    ///
    /// Adapter mode should still be preferred for new query routes. This hook is
    /// for protocol wiring that needs vector search capability without depending
    /// on concrete vector operations beyond the composition root.
    pub fn with_vector_query_service(
        mut self,
        vector_query_service: Arc<dyn VectorQueryService>,
    ) -> Self {
        self.vector_query_service = Some(vector_query_service);
        self
    }

    /// Attach catalog metadata resolution for EXPLAIN storage-authority output.
    pub fn with_catalog_manager(
        mut self,
        catalog_manager: Arc<crate::catalog::CatalogManager>,
    ) -> Self {
        self.catalog_manager = Some(catalog_manager);
        self
    }
}

/// Execute unified query request
#[derive(Debug, Deserialize)]
pub struct ExecuteQueryRequest {
    /// SQL-like query string (e.g., "SELECT * FROM products WHERE $.category = 'electronics' AND VECTOR_SIMILAR(embedding, ?, 0.8)")
    pub query: String,
    /// Query vector (for VECTOR_SIMILAR clauses)
    #[serde(default)]
    pub query_vector: Option<Vec<f32>>,
    /// Fusion strategy override
    #[serde(default)]
    pub fusion_strategy: Option<String>,
    /// Maximum results
    #[serde(default = "default_limit")]
    pub limit: Option<u32>,
}

fn default_limit() -> Option<u32> {
    Some(100)
}

/// Multi-model query request (programmatic API)
#[derive(Debug, Deserialize)]
pub struct MultiModelQueryRequest {
    /// Components of the query
    pub components: Vec<QueryComponentRequest>,
    /// Fusion strategy
    #[serde(default = "default_fusion")]
    pub fusion_strategy: String,
    /// Maximum results
    #[serde(default = "default_limit")]
    pub limit: Option<u32>,
}

fn default_fusion() -> String {
    "intersection".to_string()
}

/// Single query component
#[derive(Debug, Deserialize)]
pub struct QueryComponentRequest {
    /// Component type: "vector", "document", "graph", "log", "metric"
    pub component_type: String,
    /// Component-specific configuration
    pub config: serde_json::Value,
}

/// Query result response
#[derive(Debug, Serialize)]
pub struct QueryResultResponse {
    /// Result records
    pub records: Vec<UnifiedRecordResponse>,
    /// Total count (if available)
    pub total_count: Option<u64>,
    /// Number of records returned in this response
    pub records_returned: u64,
    /// Execution metrics
    pub metrics: QueryMetricsResponse,
}

/// Single unified record
#[derive(Debug, Serialize)]
pub struct UnifiedRecordResponse {
    /// Record ID
    pub id: String,
    /// Source model (vector, document, graph, observability)
    pub source_model: String,
    /// Record data
    pub data: serde_json::Value,
    /// Relevance score (if applicable)
    pub score: Option<f64>,
    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Query execution metrics
#[derive(Debug, Serialize)]
pub struct QueryMetricsResponse {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Time per sub-query
    pub sub_query_times: Vec<SubQueryTimeResponse>,
    /// Number of records scanned
    pub records_scanned: u64,
    /// Number of records returned
    pub records_returned: u64,
}

/// Sub-query timing info
#[derive(Debug, Serialize)]
pub struct SubQueryTimeResponse {
    /// Model type
    pub model: String,
    /// Execution time in milliseconds
    pub time_ms: f64,
}

/// Query explain response
#[derive(Debug, Serialize)]
pub struct ExplainResponse {
    /// Component plans
    pub components: Vec<ComponentPlanResponse>,
    /// Fusion strategy
    pub fusion_strategy: String,
    /// Estimated total cost
    pub estimated_total_cost: f64,
    /// Cataloged storage authority metadata for referenced targets, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_authority: Option<Vec<StorageAuthorityExplanation>>,
}

/// Component plan info
#[derive(Debug, Serialize)]
pub struct ComponentPlanResponse {
    /// Data model
    pub model: String,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Whether parallelizable
    pub parallelizable: bool,
}

/// Create router for unified query endpoints
pub fn create_router() -> Router<UnifiedQueryApiState> {
    Router::new()
        .route("/execute", post(execute_query))
        .route("/multi-model", post(execute_multi_model_query))
        .route("/federated", post(execute_federated_query))
        .route("/distributed", post(execute_distributed_query))
        .route("/explain", post(explain_query))
        // Prepared statement endpoints
        .route("/prepare", post(prepare_statement))
        .route("/execute/{statement_id}", post(execute_prepared_statement))
        .route(
            "/prepared/{statement_id}",
            delete(delete_prepared_statement),
        )
        .route("/prepared/stats", post(get_prepared_stats))
}

/// Execute a unified SQL-like query
///
/// POST /api/v1/unified/execute
///
/// This endpoint intelligently routes queries based on content:
/// - When `unified-facade-routing` feature is enabled and adapter is available,
///   all queries route through `QueryFacadeAdapter.federated_query()`
/// - Otherwise, queries with multi-model extensions (VECTOR_SEARCH, GRAPH_QUERY, etc.)
///   are routed to FederatedQueryContext
/// - Standard SQL queries use the existing QueryDecomposer
///
/// Request body:
/// ```json
/// {
///   "query": "SELECT * FROM products WHERE $.category = 'electronics' AND VECTOR_SIMILAR(embedding, ?, 0.8)",
///   "query_vector": [0.1, 0.2, ...],
///   "fusion_strategy": "intersection",
///   "limit": 100
/// }
/// ```
///
/// ## Multi-Model SQL Extensions (routed to federated engine)
///
/// - `VECTOR_SEARCH('collection', vector, top_k)` - Vector similarity search
/// - `GRAPH_QUERY('MATCH (a)-[:KNOWS]->(b) RETURN b')` - Graph traversal via Cypher
/// - `DOCUMENT_QUERY('collection', filter)` - Document queries
/// - `LOGS('namespace')`, `METRICS('namespace')` - Observability queries
/// - `embedding <-> '[0.1,0.2]'::vector` - pgvector-compatible distance operator
async fn execute_query(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<ExecuteQueryRequest>,
) -> ApiResult<JsonResponse<QueryResultResponse>> {
    use std::time::Instant;

    info!("Executing unified query: {}", request.query);
    let start = Instant::now();

    // All queries route through QueryFacadeAdapter
    let adapter = state.query_adapter.as_ref().ok_or_else(|| {
        ApiError::Internal(
            "QueryFacadeAdapter not configured. Use UnifiedQueryApiState::new_with_adapter()"
                .to_string(),
        )
    })?;

    debug!("Routing query through QueryFacadeAdapter");
    execute_query_via_adapter(adapter, &request, start).await
}

/// Execute query through the unified QueryFacadeAdapter
///
/// This function routes the query through the facade for consistent execution
/// across all query paths (REST, gRPC, internal).
async fn execute_query_via_adapter(
    adapter: &QueryFacadeAdapter,
    request: &ExecuteQueryRequest,
    start: std::time::Instant,
) -> ApiResult<JsonResponse<QueryResultResponse>> {
    let result = adapter
        .federated_query(&request.query)
        .await
        .map_err(|e| ApiError::Internal(format!("Query execution failed: {}", e)))?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert QueryResult to QueryResultResponse
    let response = transform_query_result_to_response(result, elapsed_ms, request.limit);

    info!(
        "Query via adapter executed in {:.2}ms, returned {} records",
        elapsed_ms,
        response.records.len()
    );

    Ok(JsonResponse(response))
}

/// Transform QueryResult from facade to QueryResultResponse for REST API
///
/// This function handles backward-compatible conversion from the unified
/// QueryResult type to the REST API response format.
fn transform_query_result_to_response(
    result: crate::query::facade::QueryResult,
    elapsed_ms: f64,
    limit: Option<u32>,
) -> QueryResultResponse {
    use crate::query::facade::QueryResultData;

    let limit = limit.unwrap_or(100) as usize;

    let records: Vec<UnifiedRecordResponse> = match result.data {
        QueryResultData::Rows(rows) => rows
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, row)| UnifiedRecordResponse {
                id: row
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map_or_else(|| format!("row_{}", i), |s| s.to_string()),
                source_model: "unified".to_string(),
                data: row,
                score: None,
                metadata: HashMap::new(),
            })
            .collect(),
        QueryResultData::VectorResults(matches) => matches
            .into_iter()
            .take(limit)
            .map(|m| {
                let metadata = m
                    .record
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), format!("{v:?}")))
                    .collect();
                UnifiedRecordResponse {
                    id: m.record.oid,
                    source_model: "vector".to_string(),
                    data: serde_json::json!({ "score": m.score }),
                    score: Some(m.score as f64),
                    metadata,
                }
            })
            .collect(),
        QueryResultData::Graph(graph_result) => graph_result
            .nodes
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, node)| UnifiedRecordResponse {
                id: format!("node_{}", i),
                source_model: "graph".to_string(),
                data: node,
                score: None,
                metadata: HashMap::new(),
            })
            .collect(),
        QueryResultData::Empty => vec![],
    };

    let metrics_info = result.metrics.unwrap_or_default();

    QueryResultResponse {
        total_count: Some(records.len() as u64),
        records_returned: records.len() as u64,
        records,
        metrics: QueryMetricsResponse {
            total_time_ms: elapsed_ms,
            sub_query_times: vec![],
            records_scanned: metrics_info.results_scanned as u64,
            records_returned: metrics_info.results_returned as u64,
        },
    }
}

/// Execute a programmatic multi-model query
///
/// POST /api/v1/unified/multi-model
///
/// This endpoint converts programmatic multi-model queries to federated SQL
/// and routes through the unified adapter for consistent execution.
///
/// Request body:
/// ```json
/// {
///   "components": [
///     {
///       "component_type": "vector",
///       "config": {
///         "collection": "products",
///         "query_vector": [0.1, 0.2, ...],
///         "top_k": 10,
///         "threshold": 0.8
///       }
///     },
///     {
///       "component_type": "document",
///       "config": {
///         "collection": "products",
///         "filter": "$.category = 'electronics'"
///       }
///     }
///   ],
///   "fusion_strategy": "intersection",
///   "limit": 100
/// }
/// ```
async fn execute_multi_model_query(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<MultiModelQueryRequest>,
) -> ApiResult<JsonResponse<QueryResultResponse>> {
    use std::time::Instant;

    info!(
        "Executing multi-model query with {} components",
        request.components.len()
    );
    let start = Instant::now();

    // All queries route through QueryFacadeAdapter
    let adapter = state.query_adapter.as_ref().ok_or_else(|| {
        ApiError::Internal(
            "QueryFacadeAdapter not configured. Use UnifiedQueryApiState::new_with_adapter()"
                .to_string(),
        )
    })?;

    debug!("Routing multi-model query through QueryFacadeAdapter");
    execute_multi_model_via_adapter(adapter, &request, start).await
}

/// Execute multi-model query through the unified QueryFacadeAdapter
///
/// Converts the programmatic multi-model query to a federated SQL query
/// and routes through the adapter for consistent execution.
async fn execute_multi_model_via_adapter(
    adapter: &QueryFacadeAdapter,
    request: &MultiModelQueryRequest,
    start: std::time::Instant,
) -> ApiResult<JsonResponse<QueryResultResponse>> {
    // Convert multi-model request to federated SQL
    let sql = convert_multi_model_to_sql(request)?;
    debug!("Converted multi-model query to SQL: {}", sql);

    let result = adapter
        .federated_query(&sql)
        .await
        .map_err(|e| ApiError::Internal(format!("Multi-model query failed: {}", e)))?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert QueryResult to QueryResultResponse
    let response = transform_query_result_to_response(result, elapsed_ms, request.limit);

    info!(
        "Multi-model query via adapter executed in {:.2}ms, returned {} records",
        elapsed_ms,
        response.records.len()
    );

    Ok(JsonResponse(response))
}

/// Convert a programmatic multi-model query to federated SQL
///
/// Generates SQL with multi-model extensions (VECTOR_SEARCH, GRAPH_QUERY, etc.)
/// that can be executed through the federated query engine.
fn inject_graph_target_into_cypher(graph: &str, cypher: &str) -> String {
    let graph = graph.trim();
    let cypher = cypher.trim().trim_end_matches(';').trim();

    if graph.is_empty() || graph == "default" {
        return cypher.to_string();
    }

    let upper = cypher.to_uppercase();
    if upper.contains(" FROM ") {
        return cypher.to_string();
    }

    let insertion_index = [" WHERE ", " RETURN ", " ORDER BY ", " LIMIT ", " SKIP "]
        .iter()
        .filter_map(|needle| upper.find(needle))
        .min();

    if let Some(index) = insertion_index {
        format!("{} FROM {}{}", &cypher[..index], graph, &cypher[index..])
    } else {
        format!("{} FROM {}", cypher, graph)
    }
}

fn convert_multi_model_to_sql(request: &MultiModelQueryRequest) -> ApiResult<String> {
    let mut sql_parts = Vec::new();

    for component in &request.components {
        let sql_part = match component.component_type.as_str() {
            "vector" => {
                let collection = component
                    .config
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let query_vector = component
                    .config
                    .get("query_vector")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64())
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default();
                let top_k = component
                    .config
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10);

                format!(
                    "SELECT * FROM VECTOR_SEARCH('{}', '[{}]', {})",
                    collection.replace('\'', "''"),
                    query_vector,
                    top_k
                )
            }
            "document" => {
                let collection = component
                    .config
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let filter = component
                    .config
                    .get("filter")
                    .and_then(|v| v.as_str())
                    .unwrap_or("true");

                format!(
                    "SELECT * FROM DOCUMENT_QUERY('{}', '{}')",
                    collection.replace('\'', "''"),
                    filter.replace('\'', "''")
                )
            }
            "graph" => {
                let graph = component
                    .config
                    .get("graph")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let cypher = component
                    .config
                    .get("cypher")
                    .and_then(|v| v.as_str())
                    .unwrap_or("MATCH (n) RETURN n");
                let cypher = inject_graph_target_into_cypher(graph, cypher);

                format!(
                    "SELECT * FROM GRAPH_QUERY('{}')",
                    cypher.replace('\'', "''")
                )
            }
            "log" => {
                let namespace = component
                    .config
                    .get("namespace")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");

                format!("SELECT * FROM LOGS('{}')", namespace.replace('\'', "''"))
            }
            "metric" => {
                let namespace = component
                    .config
                    .get("namespace")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");

                format!("SELECT * FROM METRICS('{}')", namespace.replace('\'', "''"))
            }
            unknown => {
                return Err(ApiError::InvalidArgument(format!(
                    "Unknown component type: {}",
                    unknown
                )));
            }
        };
        sql_parts.push(sql_part);
    }

    // Combine with UNION based on fusion strategy
    let combined_sql = if sql_parts.len() == 1 {
        sql_parts.into_iter().next().unwrap_or_default()
    } else {
        // For intersection/union strategies, use UNION (federated engine handles fusion)
        sql_parts.join(" UNION ALL ")
    };

    // Add LIMIT if specified
    let final_sql = if let Some(limit) = request.limit {
        format!("{} LIMIT {}", combined_sql, limit)
    } else {
        combined_sql
    };

    Ok(final_sql)
}

/// Explain a query's execution plan
///
/// POST /api/v1/unified/explain
///
/// Routes through the adapter's explain method for consistent behavior
/// across all query paths.
///
/// Request body:
/// ```json
/// {
///   "query": "SELECT * FROM products WHERE VECTOR_SIMILAR(embedding, ?, 0.8)"
/// }
/// ```
async fn explain_query(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<ExecuteQueryRequest>,
) -> ApiResult<JsonResponse<ExplainResponse>> {
    info!("Explaining query: {}", request.query);

    // All queries route through QueryFacadeAdapter
    let adapter = state.query_adapter.as_ref().ok_or_else(|| {
        ApiError::Internal(
            "QueryFacadeAdapter not configured. Use UnifiedQueryApiState::new_with_adapter()"
                .to_string(),
        )
    })?;

    debug!("Explaining query through QueryFacadeAdapter");

    let explain_result = adapter
        .explain(&request.query)
        .map_err(|e| ApiError::Internal(format!("Explain failed: {}", e)))?;
    let storage_authority =
        explain_storage_authorities(state.catalog_manager.as_ref(), &request.query).await;

    let response = ExplainResponse {
        components: explain_result
            .components
            .into_iter()
            .map(|c| ComponentPlanResponse {
                model: c.model,
                estimated_cost: c.estimated_cost,
                parallelizable: c.parallelizable,
            })
            .collect(),
        fusion_strategy: explain_result.fusion_strategy,
        estimated_total_cost: explain_result.estimated_total_cost,
        storage_authority: if storage_authority.is_empty() {
            None
        } else {
            Some(storage_authority)
        },
    };

    Ok(JsonResponse(response))
}

async fn explain_storage_authorities(
    catalog_manager: Option<&Arc<crate::catalog::CatalogManager>>,
    sql: &str,
) -> Vec<StorageAuthorityExplanation> {
    let Some(catalog_manager) = catalog_manager else {
        return Vec::new();
    };

    let mut authorities = Vec::new();
    for target in explain_catalog_targets(sql) {
        let mut context = PlanContext::default();
        match resolve_catalog_authority_context(
            catalog_manager,
            AuthoritySource::new(target.clone(), "relational"),
        )
        .await
        {
            Ok(resolved) => context.resolved_objects.push(resolved),
            Err(error) => {
                debug!(
                    table = %target,
                    error = %error,
                    "Unified EXPLAIN storage authority metadata unavailable during catalog resolution/lookup"
                );
                continue;
            }
        }

        if let Some(authority) = StorageAuthorityExplanation::from_plan_context(&context) {
            authorities.push(authority);
        }
    }

    authorities
}

fn explain_catalog_targets(sql: &str) -> Vec<String> {
    let mut targets = Vec::new();

    for function_name in ["VECTOR_SEARCH", "DOCUMENT_QUERY", "LOGS", "METRICS"] {
        collect_quoted_first_args(sql, function_name, &mut targets);
    }

    collect_from_targets(sql, &mut targets);
    targets.dedup();
    targets
}

fn collect_quoted_first_args(sql: &str, function_name: &str, targets: &mut Vec<String>) {
    let upper = sql.to_uppercase();
    let mut search_start = 0;

    while let Some(relative_pos) = upper[search_start..].find(function_name) {
        let name_start = search_start + relative_pos;
        let after_name = name_start + function_name.len();
        let Some(open_relative) = sql[after_name..].find('(') else {
            break;
        };
        let mut arg_start = after_name + open_relative + 1;
        while let Some(ch) = sql[arg_start..].chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            arg_start += ch.len_utf8();
        }
        if sql[arg_start..].starts_with('\'') {
            let value_start = arg_start + 1;
            if let Some(close_relative) = sql[value_start..].find('\'') {
                let value = sql[value_start..value_start + close_relative].trim();
                if !value.is_empty() {
                    push_unique_target(targets, value);
                }
                search_start = value_start + close_relative + 1;
                continue;
            }
        }
        search_start = after_name;
    }
}

fn collect_from_targets(sql: &str, targets: &mut Vec<String>) {
    let mut previous_was_from = false;
    for token in sql.split_whitespace() {
        if previous_was_from {
            let candidate =
                token.trim_matches(|ch: char| matches!(ch, ',' | ';' | '"' | '`' | '[' | ']'));
            if !candidate.is_empty()
                && !candidate.contains('(')
                && !candidate.eq_ignore_ascii_case("SELECT")
            {
                push_unique_target(targets, candidate);
            }
            previous_was_from = false;
            continue;
        }

        previous_was_from = token.eq_ignore_ascii_case("FROM")
    }
}

fn push_unique_target(targets: &mut Vec<String>, value: &str) {
    if !targets.iter().any(|existing| existing == value) {
        targets.push(value.to_string());
    }
}

/// Parse fusion strategy from string
#[allow(dead_code)]
fn parse_fusion_strategy(s: &str) -> FusionStrategy {
    match s.to_lowercase().as_str() {
        "intersection" | "and" => FusionStrategy::Intersection,
        "union" | "or" => FusionStrategy::Union,
        "rrf" | "reciprocal_rank_fusion" => FusionStrategy::ReciprocalRankFusion { k: 60 },
        "ranked" | "weighted" => FusionStrategy::RankedFusion {
            weights: HashMap::new(),
            normalize: true,
        },
        _ => FusionStrategy::Intersection,
    }
}

/// Estimate cost for a data model
#[allow(dead_code)]
fn estimate_cost(model: &DataModel) -> f64 {
    match model {
        DataModel::Vector => 1.0,
        DataModel::Document => 2.0,
        DataModel::Graph => 3.0,
        DataModel::Observability | DataModel::TimeSeries => 2.5,
        DataModel::Relational => 1.5,
        DataModel::Event => 2.0,
    }
}

/// Execute a federated SQL query with multi-model extensions
///
/// POST /api/v1/unified/federated
///
/// This endpoint explicitly routes queries through the FederatedQueryContext,
/// supporting PostgreSQL-compatible SQL with multi-model extensions.
///
/// ## Request body:
/// ```json
/// {
///   "query": "SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1,0.2]', 10)",
///   "query_vector": [0.1, 0.2, ...],
///   "limit": 100
/// }
/// ```
///
/// ## Supported SQL Extensions:
///
/// ### Vector Search (pgvector compatible)
/// ```sql
/// SELECT * FROM products ORDER BY embedding <-> '[0.1,0.2,...]'::vector LIMIT 10;
/// SELECT * FROM VECTOR_SEARCH('embeddings', query_vector, 10);
/// ```
///
/// ### Graph Traversal (embedded Cypher)
/// ```sql
/// SELECT * FROM GRAPH_QUERY('MATCH (a)-[:KNOWS]->(b) RETURN b.name');
/// ```
///
/// ### Document Query
/// ```sql
/// SELECT * FROM DOCUMENT_QUERY('products', '$.price > 100');
/// ```
///
/// ### Cross-Model Joins
/// ```sql
/// SELECT u.*, v.similar_products
/// FROM users u
/// JOIN LATERAL VECTOR_SEARCH('embeddings', u.preference_vector, 10) v ON true;
/// ```
///
/// ### Observability Queries
/// ```sql
/// SELECT * FROM LOGS('production') WHERE timestamp > now() - interval '1h';
/// SELECT * FROM METRICS('cpu') WHERE value > 90;
/// ```
async fn execute_federated_query(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<ExecuteQueryRequest>,
) -> ApiResult<JsonResponse<FederatedQueryResponse>> {
    use std::time::Instant;

    info!("Executing federated query: {}", request.query);
    let start = Instant::now();

    // All queries route through QueryFacadeAdapter
    let adapter = state.query_adapter.as_ref().ok_or_else(|| {
        ApiError::Internal(
            "QueryFacadeAdapter not configured. Use UnifiedQueryApiState::new_with_adapter()"
                .to_string(),
        )
    })?;

    debug!("Routing federated query through QueryFacadeAdapter");
    execute_federated_via_adapter(adapter, &request, start).await
}

/// Convert Arrow RecordBatches to JSON records for the federated response
#[allow(dead_code)]
fn convert_arrow_to_records(
    result: &crate::query::federated::ExecutionResult,
    limit: usize,
) -> Vec<FederatedRecordResponse> {
    let mut records = Vec::new();
    let mut count = 0;

    for batch in &result.batches {
        if count >= limit {
            break;
        }

        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            if count >= limit {
                break;
            }

            let mut data = serde_json::Map::new();
            let mut id = format!("row_{}", count);
            let mut score: Option<f64> = None;

            for (col_idx, field) in schema.fields().iter().enumerate() {
                let column = batch.column(col_idx);
                let value = extract_value_from_array(column.as_ref(), row_idx);

                // Extract special fields
                if field.name() == "id" {
                    if let serde_json::Value::String(s) = &value {
                        id = s.clone();
                    }
                } else if (field.name() == "score" || field.name() == "distance")
                    && let serde_json::Value::Number(n) = &value
                {
                    score = n.as_f64();
                }

                data.insert(field.name().clone(), value);
            }

            records.push(FederatedRecordResponse {
                id,
                source_model: detect_source_model(&schema),
                data: serde_json::Value::Object(data),
                score,
                metadata: HashMap::new(),
            });

            count += 1;
        }
    }

    records
}

/// Convert Arrow RecordBatches to unified record format
#[allow(dead_code)]
fn convert_arrow_to_unified_records(
    result: &crate::query::federated::ExecutionResult,
    _query_type: &FederatedQueryType,
    limit: usize,
) -> Vec<UnifiedRecordResponse> {
    let federated_records = convert_arrow_to_records(result, limit);

    federated_records
        .into_iter()
        .map(|r| UnifiedRecordResponse {
            id: r.id,
            source_model: r.source_model,
            data: r.data,
            score: r.score,
            metadata: r.metadata,
        })
        .collect()
}

/// Extract a JSON value from an Arrow array at the given row index
#[allow(dead_code)]
fn extract_value_from_array(array: &dyn arrow::array::Array, row_idx: usize) -> serde_json::Value {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };
    use arrow::datatypes::DataType;

    if array.is_null(row_idx) {
        return serde_json::Value::Null;
    }

    match array.data_type() {
        DataType::Utf8 => array.as_any().downcast_ref::<StringArray>().map_or_else(
            || serde_json::Value::String("<invalid UTF8 array>".to_string()),
            |arr| serde_json::Value::String(arr.value(row_idx).to_string()),
        ),
        DataType::Float32 => array.as_any().downcast_ref::<Float32Array>().map_or_else(
            || serde_json::Value::String("<invalid Float32 array>".to_string()),
            |arr| serde_json::json!(arr.value(row_idx)),
        ),
        DataType::Float64 => array.as_any().downcast_ref::<Float64Array>().map_or_else(
            || serde_json::Value::String("<invalid Float64 array>".to_string()),
            |arr| serde_json::json!(arr.value(row_idx)),
        ),
        DataType::Int32 => array.as_any().downcast_ref::<Int32Array>().map_or_else(
            || serde_json::Value::String("<invalid Int32 array>".to_string()),
            |arr| serde_json::json!(arr.value(row_idx)),
        ),
        DataType::Int64 => array.as_any().downcast_ref::<Int64Array>().map_or_else(
            || serde_json::Value::String("<invalid Int64 array>".to_string()),
            |arr| serde_json::json!(arr.value(row_idx)),
        ),
        DataType::Boolean => array.as_any().downcast_ref::<BooleanArray>().map_or_else(
            || serde_json::Value::String("<invalid Boolean array>".to_string()),
            |arr| serde_json::json!(arr.value(row_idx)),
        ),
        _ => serde_json::Value::String(format!("<unsupported type: {:?}>", array.data_type())),
    }
}

/// Detect the source model from the schema
fn detect_source_model(schema: &arrow::datatypes::Schema) -> String {
    let field_names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    if field_names.contains(&"score")
        || field_names.contains(&"embedding")
        || field_names.contains(&"vector")
    {
        "Vector".to_string()
    } else if field_names.contains(&"node_id")
        || field_names.contains(&"edge_id")
        || field_names.contains(&"label")
    {
        "Graph".to_string()
    } else if field_names.contains(&"document") || field_names.contains(&"doc") {
        "Document".to_string()
    } else if field_names.contains(&"timestamp")
        && (field_names.contains(&"level") || field_names.contains(&"metric_name"))
    {
        "Observability".to_string()
    } else {
        "Relational".to_string()
    }
}

/// Response for federated query execution
#[derive(Debug, Serialize)]
pub struct FederatedQueryResponse {
    /// Result records
    pub records: Vec<FederatedRecordResponse>,
    /// Total count (if available)
    pub total_count: Option<u64>,
    /// Number of records returned in this response
    pub records_returned: u64,
    /// Query type detected
    pub query_type: String,
    /// Models involved in the query
    pub involved_models: Vec<String>,
    /// Execution plan details
    pub execution_plan: Option<FederatedPlanResponse>,
    /// Execution metrics
    pub metrics: FederatedMetricsResponse,
}

/// Single federated record
#[derive(Debug, Serialize)]
pub struct FederatedRecordResponse {
    /// Record ID
    pub id: String,
    /// Source model (vector, document, graph, observability, relational)
    pub source_model: String,
    /// Record data
    pub data: serde_json::Value,
    /// Relevance score (if applicable)
    pub score: Option<f64>,
    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Federated execution plan response
#[derive(Debug, Serialize)]
pub struct FederatedPlanResponse {
    /// Whether this is a cross-model join
    pub is_cross_model: bool,
    /// SQL extensions used
    pub extensions_used: Vec<String>,
}

/// Federated execution metrics
#[derive(Debug, Serialize)]
pub struct FederatedMetricsResponse {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Parse time in milliseconds
    pub parse_time_ms: f64,
    /// Optimization time in milliseconds
    pub optimize_time_ms: f64,
    /// Execution time in milliseconds
    pub execute_time_ms: f64,
    /// Time per sub-query
    pub sub_query_times: Vec<SubQueryTimeResponse>,
    /// Rows scanned
    pub rows_scanned: u64,
}

/// Execute federated query through the unified QueryFacadeAdapter
///
/// This function routes the query through the facade for consistent execution
/// across all query paths (REST, gRPC, internal).
async fn execute_federated_via_adapter(
    adapter: &QueryFacadeAdapter,
    request: &ExecuteQueryRequest,
    start: std::time::Instant,
) -> ApiResult<JsonResponse<FederatedQueryResponse>> {
    use crate::query::facade::QueryResultData;

    let result = adapter
        .federated_query(&request.query)
        .await
        .map_err(|e| ApiError::Internal(format!("Federated query failed: {}", e)))?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert QueryResult to FederatedQueryResponse
    let records = match result.data {
        QueryResultData::Rows(rows) => rows
            .into_iter()
            .take(request.limit.unwrap_or(100) as usize)
            .enumerate()
            .map(|(i, row)| FederatedRecordResponse {
                id: format!("row_{}", i),
                source_model: "unified".to_string(),
                data: row,
                score: None,
                metadata: HashMap::new(),
            })
            .collect(),
        QueryResultData::VectorResults(matches) => {
            matches
                .into_iter()
                .take(request.limit.unwrap_or(100) as usize)
                .map(|m| {
                    // Convert metadata from Value to HashMap<String, String>
                    let metadata = m
                        .record
                        .props
                        .iter()
                        .map(|(k, v)| (k.clone(), format!("{v:?}")))
                        .collect();
                    FederatedRecordResponse {
                        id: m.record.oid,
                        source_model: "vector".to_string(),
                        data: serde_json::json!({ "score": m.score }),
                        score: Some(m.score as f64),
                        metadata,
                    }
                })
                .collect()
        }
        QueryResultData::Graph(graph_result) => graph_result
            .nodes
            .into_iter()
            .take(request.limit.unwrap_or(100) as usize)
            .enumerate()
            .map(|(i, node)| FederatedRecordResponse {
                id: format!("node_{}", i),
                source_model: "graph".to_string(),
                data: node,
                score: None,
                metadata: HashMap::new(),
            })
            .collect(),
        QueryResultData::Empty => vec![],
    };

    let metrics_info = result.metrics.unwrap_or_default();

    let response = FederatedQueryResponse {
        records_returned: records.len() as u64,
        total_count: Some(records.len() as u64),
        records,
        query_type: "unified".to_string(),
        involved_models: vec!["unified".to_string()],
        execution_plan: Some(FederatedPlanResponse {
            is_cross_model: false,
            extensions_used: vec![],
        }),
        metrics: FederatedMetricsResponse {
            total_time_ms: elapsed_ms,
            parse_time_ms: metrics_info.planning_time_ms as f64,
            optimize_time_ms: 0.0,
            execute_time_ms: metrics_info.execution_time_ms as f64,
            sub_query_times: vec![],
            rows_scanned: 0,
        },
    };

    info!(
        "Federated query via adapter in {:.2}ms, returned {} records",
        elapsed_ms, response.records_returned
    );

    Ok(JsonResponse(response))
}

// =============================================================================
// DISTRIBUTED QUERY API
// =============================================================================

/// Execute distributed query across the cluster
///
/// POST /api/v1/unified/distributed
///
/// This endpoint executes queries through the DistributedQueryCoordinator for
/// cluster-aware query execution that can span multiple nodes in a ProximaDB cluster.
///
/// ## Features
///
/// - **Automatic query distribution**: Decomposes queries into subqueries for each node
/// - **Result aggregation**: Merges results from multiple nodes
/// - **Shard-aware routing**: Routes subqueries to appropriate shards
/// - **Result caching**: Caches query results for improved performance
/// - **Cross-shard joins**: Supports shuffle exchange for complex joins
///
/// Request body:
/// ```json
/// {
///   "query": "SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2]', 10)",
///   "strategy": "auto",
///   "timeout_ms": 30000,
///   "min_nodes": 1
/// }
/// ```
///
/// ## Execution Strategies
///
/// - `localOnly`: Execute on local node only (ignore cluster)
/// - `distributed`: Distribute across cluster
/// - `broadcast`: Send to all nodes
/// - `auto`: Let coordinator decide (default)
async fn execute_distributed_query(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<DistributedQueryRequest>,
) -> ApiResult<JsonResponse<DistributedQueryResponse>> {
    use std::time::Instant;

    info!(
        "Executing distributed query with strategy: {:?}",
        request.strategy
    );
    let start = Instant::now();

    // All queries route through QueryFacadeAdapter
    let adapter = state.query_adapter.as_ref().ok_or_else(|| {
        ApiError::Internal(
            "QueryFacadeAdapter not configured. Use UnifiedQueryApiState::new_with_adapter()"
                .to_string(),
        )
    })?;

    debug!("Routing distributed query through QueryFacadeAdapter");
    execute_distributed_via_adapter(adapter, &request, start).await
}

/// Execute distributed query through the unified QueryFacadeAdapter
///
/// This function routes the query through the distributed execution path
/// for cluster-aware query execution.
async fn execute_distributed_via_adapter(
    adapter: &QueryFacadeAdapter,
    request: &DistributedQueryRequest,
    start: std::time::Instant,
) -> ApiResult<JsonResponse<DistributedQueryResponse>> {
    // Apply strategy hint if provided
    let query_with_strategy = match request.strategy {
        ExecutionStrategy::LocalOnly => {
            format!("/* LOCAL_ONLY */ {}", request.query)
        }
        ExecutionStrategy::Distributed => {
            format!("/* DISTRIBUTED */ {}", request.query)
        }
        ExecutionStrategy::Broadcast => {
            format!("/* BROADCAST */ {}", request.query)
        }
        ExecutionStrategy::Auto => request.query.clone(),
    };

    let result = adapter
        .distributed_query(&query_with_strategy)
        .await
        .map_err(|e| ApiError::Internal(format!("Distributed query failed: {}", e)))?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Extract metrics if available
    let metrics_info = result.metrics.unwrap_or_default();
    let distributed_extra = &metrics_info.extra;
    let nodes_involved = distributed_extra
        .get("nodes_involved")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(0);
    let cache_hits = distributed_extra
        .get("cache_hits")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);

    // Convert QueryResult to DistributedQueryResponse
    let records = match result.data {
        crate::query::facade::QueryResultData::Rows(rows) => rows
            .into_iter()
            .take(request.limit.unwrap_or(100) as usize)
            .collect(),
        _ => vec![],
    };

    let records_returned = records.len() as u64;

    let response = DistributedQueryResponse {
        records,
        total_count: None,
        records_returned,
        execution_plan: DistributedPlanResponse {
            strategy: request.strategy,
            nodes_involved,
            execution_time_ms: elapsed_ms,
        },
        metrics: DistributedMetricsResponse {
            total_time_ms: elapsed_ms,
            planning_time_ms: metrics_info.planning_time_ms as f64,
            execution_time_ms: metrics_info.execution_time_ms as f64,
            cache_hits,
        },
    };

    info!(
        "Distributed query executed in {:.2}ms, returned {} records",
        elapsed_ms, response.records_returned
    );

    Ok(JsonResponse(response))
}

/// Request for distributed query execution
#[derive(Debug, Deserialize)]
pub struct DistributedQueryRequest {
    /// SQL query to execute
    pub query: String,
    /// Execution strategy hint
    #[serde(default)]
    pub strategy: ExecutionStrategy,
    /// Maximum results to return
    pub limit: Option<u64>,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Minimum nodes required for execution
    pub min_nodes: Option<usize>,
}

/// Execution strategy for distributed queries
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub enum ExecutionStrategy {
    /// Execute locally only
    LocalOnly,
    /// Distribute across cluster
    Distributed,
    /// Broadcast to all nodes
    Broadcast,
    /// Let coordinator decide
    #[serde(alias = "auto")]
    #[default]
    Auto,
}

/// Response for distributed query execution
#[derive(Debug, Serialize)]
pub struct DistributedQueryResponse {
    /// Result records
    pub records: Vec<serde_json::Value>,
    /// Total count (if available)
    pub total_count: Option<u64>,
    /// Number of records returned
    pub records_returned: u64,
    /// Execution plan details
    pub execution_plan: DistributedPlanResponse,
    /// Execution metrics
    pub metrics: DistributedMetricsResponse,
}

/// Distributed execution plan information
#[derive(Debug, Serialize)]
pub struct DistributedPlanResponse {
    /// Strategy used for execution
    pub strategy: ExecutionStrategy,
    /// Number of nodes involved
    pub nodes_involved: usize,
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Distributed query execution metrics
#[derive(Debug, Serialize)]
pub struct DistributedMetricsResponse {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Query planning time in milliseconds
    pub planning_time_ms: f64,
    /// Query execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of cache hits
    pub cache_hits: u64,
}

// =============================================================================
// PREPARED STATEMENTS API
// =============================================================================

/// Request to prepare a SQL statement
#[derive(Debug, Deserialize)]
pub struct PrepareStatementRequest {
    /// SQL query with parameter placeholders ($1, $2, etc.)
    pub sql: String,
    /// Optional TTL in seconds (default: 3600 = 1 hour)
    pub ttl_seconds: Option<u64>,
}

/// Response from prepare statement
#[derive(Debug, Serialize)]
pub struct PrepareStatementResponse {
    /// Unique statement ID
    pub statement_id: String,
    /// Number of parameters expected
    pub parameter_count: usize,
    /// TTL in seconds
    pub ttl_seconds: u64,
    /// Success flag
    pub success: bool,
    /// Error message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request to execute a prepared statement
#[derive(Debug, Deserialize)]
pub struct ExecutePreparedRequest {
    /// Parameter values to bind
    pub params: Vec<serde_json::Value>,
    /// Optional limit override
    pub limit: Option<u32>,
}

/// Response from execute prepared
#[derive(Debug, Serialize)]
pub struct ExecutePreparedResponse {
    /// Query results
    pub records: Vec<UnifiedRecordResponse>,
    /// Total count
    pub total_count: Option<u64>,
    /// Number of records returned
    pub records_returned: u64,
    /// Execution metrics
    pub metrics: QueryMetricsResponse,
    /// Statement execution count (lifetime)
    pub statement_execution_count: u64,
}

/// Response from delete prepared statement
#[derive(Debug, Serialize)]
pub struct DeletePreparedResponse {
    /// Success flag
    pub success: bool,
    /// Error message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Prepared statement cache statistics
#[derive(Debug, Serialize)]
pub struct PreparedStatsResponse {
    /// Number of cached statements
    pub cached_statements: usize,
    /// Maximum allowed statements
    pub max_statements: usize,
    /// Total executions across all statements
    pub total_executions: u64,
    /// Total access count
    pub total_access_count: u64,
    /// Oldest statement age in seconds
    pub oldest_statement_age_secs: u64,
}

/// Prepare a SQL statement for repeated execution
///
/// POST /api/v1/unified/prepare
///
/// This endpoint parses and optimizes a SQL query once, caching the result
/// for efficient repeated execution with different parameters.
///
/// ## Request body:
/// ```json
/// {
///   "sql": "SELECT * FROM VECTOR_SEARCH($1, $2, 10) WHERE category = $3",
///   "ttl_seconds": 3600
/// }
/// ```
///
/// ## Response:
/// ```json
/// {
///   "statement_id": "stmt_0000000000000001",
///   "parameter_count": 3,
///   "ttl_seconds": 3600,
///   "success": true
/// }
/// ```
async fn prepare_statement(
    State(state): State<UnifiedQueryApiState>,
    Json(request): Json<PrepareStatementRequest>,
) -> ApiResult<JsonResponse<PrepareStatementResponse>> {
    use std::time::Duration;

    info!("Preparing statement: {}", request.sql);

    let ttl = Duration::from_secs(request.ttl_seconds.unwrap_or(3600));

    match state
        .prepared_statement_cache
        .prepare_with_ttl(&request.sql, ttl)
    {
        Ok(statement_id) => {
            let statement = state
                .prepared_statement_cache
                .get(&statement_id)
                .map_err(|e| {
                    ApiError::Internal(format!("Failed to retrieve prepared statement: {}", e))
                })?;

            info!(
                statement_id = %statement_id,
                parameter_count = statement.parameter_count(),
                "Statement prepared successfully"
            );

            Ok(JsonResponse(PrepareStatementResponse {
                statement_id,
                parameter_count: statement.parameter_count(),
                ttl_seconds: ttl.as_secs(),
                success: true,
                error: None,
            }))
        }
        Err(e) => {
            warn!("Failed to prepare statement: {}", e);
            Ok(JsonResponse(PrepareStatementResponse {
                statement_id: String::new(),
                parameter_count: 0,
                ttl_seconds: 0,
                success: false,
                error: Some(e.to_string()),
            }))
        }
    }
}

/// Execute a prepared statement with parameters
///
/// POST /api/v1/unified/execute/{statement_id}
///
/// Executes a previously prepared statement by substituting the provided
/// parameters and running the query.
///
/// ## Request body:
/// ```json
/// {
///   "params": ["embeddings", "[0.1, 0.2, 0.3]", "electronics"],
///   "limit": 100
/// }
/// ```
async fn execute_prepared_statement(
    State(state): State<UnifiedQueryApiState>,
    Path(statement_id): Path<String>,
    Json(request): Json<ExecutePreparedRequest>,
) -> ApiResult<JsonResponse<ExecutePreparedResponse>> {
    use std::time::Instant;

    info!("Executing prepared statement: {}", statement_id);
    let start = Instant::now();

    // Convert JSON values to ParameterValues
    let params: Vec<ParameterValue> = request.params.iter().map(json_to_parameter_value).collect();

    // Get the substituted SQL
    let sql = state
        .prepared_statement_cache
        .execute_sql(&statement_id, &params)
        .map_err(|e| match e {
            crate::query::prepared::PreparedStatementError::NotFound(_) => {
                ApiError::NotFound(format!("Prepared statement not found: {}", statement_id))
            }
            crate::query::prepared::PreparedStatementError::Expired(_) => {
                ApiError::Gone(format!("Prepared statement expired: {}", statement_id))
            }
            crate::query::prepared::PreparedStatementError::ParameterCountMismatch {
                expected,
                actual,
            } => ApiError::InvalidArgument(format!(
                "Parameter count mismatch: expected {}, got {}",
                expected, actual
            )),
            other => ApiError::Internal(format!("Prepared statement error: {}", other)),
        })?;

    debug!("Substituted SQL: {}", sql);

    // Execute the query through the adapter
    let adapter = state
        .query_adapter
        .as_ref()
        .ok_or_else(|| ApiError::Internal("QueryFacadeAdapter not configured".to_string()))?;

    let result = adapter
        .federated_query(&sql)
        .await
        .map_err(|e| ApiError::Internal(format!("Query execution failed: {}", e)))?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Get execution count from the cached statement
    let execution_count = state
        .prepared_statement_cache
        .get(&statement_id)
        .map(|s| s.execution_count)
        .unwrap_or(0);

    // Convert result to response
    let query_response = transform_query_result_to_response(result, elapsed_ms, request.limit);

    info!(
        statement_id = %statement_id,
        records = query_response.records.len(),
        elapsed_ms = elapsed_ms,
        "Prepared statement executed"
    );

    Ok(JsonResponse(ExecutePreparedResponse {
        records: query_response.records,
        total_count: query_response.total_count,
        records_returned: query_response.records_returned,
        metrics: query_response.metrics,
        statement_execution_count: execution_count,
    }))
}

/// Delete a prepared statement
///
/// DELETE /api/v1/unified/prepared/{statement_id}
///
/// Removes a prepared statement from the cache, freeing resources.
async fn delete_prepared_statement(
    State(state): State<UnifiedQueryApiState>,
    Path(statement_id): Path<String>,
) -> ApiResult<JsonResponse<DeletePreparedResponse>> {
    info!("Deleting prepared statement: {}", statement_id);

    match state.prepared_statement_cache.drop_statement(&statement_id) {
        Ok(()) => {
            info!(statement_id = %statement_id, "Prepared statement deleted");
            Ok(JsonResponse(DeletePreparedResponse {
                success: true,
                error: None,
            }))
        }
        Err(e) => {
            warn!("Failed to delete prepared statement: {}", e);
            Ok(JsonResponse(DeletePreparedResponse {
                success: false,
                error: Some(e.to_string()),
            }))
        }
    }
}

/// Get prepared statement cache statistics
///
/// POST /api/v1/unified/prepared/stats
///
/// Returns statistics about the prepared statement cache, useful for
/// monitoring and debugging.
async fn get_prepared_stats(
    State(state): State<UnifiedQueryApiState>,
) -> ApiResult<JsonResponse<PreparedStatsResponse>> {
    let stats = state.prepared_statement_cache.stats();

    Ok(JsonResponse(PreparedStatsResponse {
        cached_statements: stats.cached_statements,
        max_statements: stats.max_statements,
        total_executions: stats.total_executions,
        total_access_count: stats.total_access_count,
        oldest_statement_age_secs: stats.oldest_statement_age_secs,
    }))
}

/// Convert a JSON value to a ParameterValue
fn json_to_parameter_value(v: &serde_json::Value) -> ParameterValue {
    match v {
        serde_json::Value::String(s) => ParameterValue::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ParameterValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                ParameterValue::Float(f)
            } else {
                ParameterValue::String(n.to_string())
            }
        }
        serde_json::Value::Bool(b) => ParameterValue::Bool(*b),
        serde_json::Value::Null => ParameterValue::Null,
        serde_json::Value::Array(arr) => {
            // Try to parse as vector of f32
            let floats: Vec<f32> = arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            if floats.len() == arr.len() {
                ParameterValue::Vector(floats)
            } else {
                ParameterValue::Json(v.clone())
            }
        }
        serde_json::Value::Object(_) => ParameterValue::Json(v.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fusion_strategy() {
        assert!(matches!(
            parse_fusion_strategy("intersection"),
            FusionStrategy::Intersection
        ));
        assert!(matches!(
            parse_fusion_strategy("union"),
            FusionStrategy::Union
        ));
        assert!(matches!(
            parse_fusion_strategy("rrf"),
            FusionStrategy::ReciprocalRankFusion { .. }
        ));
        assert!(matches!(
            parse_fusion_strategy("ranked"),
            FusionStrategy::RankedFusion { .. }
        ));
        assert!(matches!(
            parse_fusion_strategy("unknown"),
            FusionStrategy::Intersection
        ));
    }

    #[test]
    fn test_estimate_cost() {
        assert_eq!(estimate_cost(&DataModel::Vector), 1.0);
        assert_eq!(estimate_cost(&DataModel::Document), 2.0);
        assert_eq!(estimate_cost(&DataModel::Graph), 3.0);
    }

    #[test]
    fn test_should_use_federated_detection() {
        // Test the federated detection logic directly using a helper function
        // This avoids the need for complex mock setup

        // VECTOR_SEARCH should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1,0.2]', 10)"
        ));

        // pgvector distance operator should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM products ORDER BY embedding <-> '[0.1,0.2]' LIMIT 10"
        ));

        // pgvector type cast should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM products ORDER BY embedding <-> '[0.1]'::vector LIMIT 10"
        ));

        // GRAPH_QUERY should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM GRAPH_QUERY('MATCH (a)-[:KNOWS]->(b) RETURN b.name')"
        ));

        // DOCUMENT_QUERY should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM DOCUMENT_QUERY('products', '$.price > 100')"
        ));

        // LOGS should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM LOGS('production') WHERE timestamp > now() - interval '1h'"
        ));

        // METRICS should use federated
        assert!(should_use_federated_pattern(
            "SELECT * FROM METRICS('cpu') WHERE value > 90"
        ));

        // Standard SQL should not use federated
        assert!(!should_use_federated_pattern(
            "SELECT * FROM users WHERE id = 1"
        ));
        assert!(!should_use_federated_pattern(
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        ));
        assert!(!should_use_federated_pattern(
            "INSERT INTO users (name) VALUES ('test')"
        ));
    }

    /// Helper function to test federated detection without needing full state
    fn should_use_federated_pattern(query: &str) -> bool {
        let query_upper = query.to_uppercase();
        query_upper.contains("VECTOR_SEARCH")
            || query_upper.contains("GRAPH_QUERY")
            || query_upper.contains("DOCUMENT_QUERY")
            || query_upper.contains("LOGS(")
            || query_upper.contains("METRICS(")
            || query.contains("<->")  // pgvector distance operator
            || query.contains("::vector") // pgvector type cast
    }

    #[test]
    fn test_detect_source_model() {
        use arrow::datatypes::{DataType, Field, Schema};

        // Vector schema
        let vector_schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("score", DataType::Float32, false),
        ]);
        assert_eq!(detect_source_model(&vector_schema), "Vector");

        // Graph schema
        let graph_schema = Schema::new(vec![
            Field::new("node_id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, true),
        ]);
        assert_eq!(detect_source_model(&graph_schema), "Graph");

        // Document schema
        let doc_schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, false),
        ]);
        assert_eq!(detect_source_model(&doc_schema), "Document");

        // Observability schema (logs)
        let obs_schema = Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("level", DataType::Utf8, false),
            Field::new("message", DataType::Utf8, false),
        ]);
        assert_eq!(detect_source_model(&obs_schema), "Observability");

        // Relational schema (default)
        let rel_schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, true),
        ]);
        assert_eq!(detect_source_model(&rel_schema), "Relational");
    }

    #[test]
    fn test_federated_parser_creation() {
        let parser = FederatedParser::new();
        let extensions = parser.supported_extensions();
        assert!(extensions.contains(&"VECTOR_SEARCH"));
        assert!(extensions.contains(&"GRAPH_QUERY"));
        assert!(extensions.contains(&"LOGS"));
    }

    // =========================================================================
    // Request / Response serialization tests
    // =========================================================================

    #[test]
    fn test_unified_query_request_parsing() {
        // Verify ExecuteQueryRequest deserializes correctly from JSON
        let json = serde_json::json!({
            "query": "SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2]', 10)",
            "query_vector": [0.1, 0.2, 0.3],
            "fusion_strategy": "rrf",
            "limit": 50
        });

        let request: ExecuteQueryRequest =
            serde_json::from_value(json).expect("should deserialize");

        assert_eq!(
            request.query,
            "SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2]', 10)"
        );
        let qv = request
            .query_vector
            .expect("query_vector should be present");
        assert_eq!(qv.len(), 3);
        assert!((qv[0] - 0.1).abs() < f32::EPSILON);
        assert_eq!(request.fusion_strategy.as_deref(), Some("rrf"));
        assert_eq!(request.limit, Some(50));

        // Minimal request (only required field)
        let minimal = serde_json::json!({ "query": "SELECT 1" });
        let req: ExecuteQueryRequest =
            serde_json::from_value(minimal).expect("minimal request should parse");
        assert_eq!(req.query, "SELECT 1");
        assert!(req.query_vector.is_none());
        assert!(req.fusion_strategy.is_none());
        // default_limit() returns Some(100)
        assert_eq!(req.limit, Some(100));
    }

    #[test]
    fn test_unified_query_response_serialization() {
        // Verify QueryResultResponse serializes to the expected JSON shape
        let response = QueryResultResponse {
            records: vec![
                UnifiedRecordResponse {
                    id: "vec_001".into(),
                    source_model: "vector".into(),
                    data: serde_json::json!({"score": 0.95}),
                    score: Some(0.95),
                    metadata: HashMap::new(),
                },
                UnifiedRecordResponse {
                    id: "doc_042".into(),
                    source_model: "document".into(),
                    data: serde_json::json!({"title": "Rust programming"}),
                    score: None,
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert("collection".into(), "articles".into());
                        m
                    },
                },
            ],
            total_count: Some(2),
            records_returned: 2,
            metrics: QueryMetricsResponse {
                total_time_ms: 12.5,
                sub_query_times: vec![
                    SubQueryTimeResponse {
                        model: "vector".into(),
                        time_ms: 5.0,
                    },
                    SubQueryTimeResponse {
                        model: "document".into(),
                        time_ms: 7.5,
                    },
                ],
                records_scanned: 1000,
                records_returned: 2,
            },
        };

        let json = serde_json::to_value(&response).expect("should serialize");
        assert_eq!(json["total_count"], 2);
        assert_eq!(json["records_returned"], 2);
        assert_eq!(json["records"].as_array().expect("records array").len(), 2);
        assert_eq!(json["records"][0]["id"], "vec_001");
        assert_eq!(json["records"][0]["source_model"], "vector");
        assert!((json["records"][0]["score"].as_f64().unwrap() - 0.95).abs() < f64::EPSILON);
        // metadata with entries should be present; empty metadata should be absent
        assert!(json["records"][1]["metadata"].is_object());
        assert!(
            json["records"][0].get("metadata").is_none()
                || json["records"][0]["metadata"]
                    .as_object()
                    .is_some_and(|m| m.is_empty()),
            "empty metadata should be skipped or empty"
        );
        assert!((json["metrics"]["total_time_ms"].as_f64().unwrap() - 12.5).abs() < f64::EPSILON);
        assert_eq!(
            json["metrics"]["sub_query_times"]
                .as_array()
                .expect("sub_query_times")
                .len(),
            2
        );
    }

    #[test]
    fn test_federated_query_request() {
        // Verify MultiModelQueryRequest (cross-model programmatic API) parses correctly
        let json = serde_json::json!({
            "components": [
                {
                    "component_type": "vector",
                    "config": {
                        "collection": "embeddings",
                        "query_vector": [0.1, 0.2],
                        "top_k": 10
                    }
                },
                {
                    "component_type": "document",
                    "config": {
                        "collection": "articles",
                        "filter": "$.category = 'science'"
                    }
                },
                {
                    "component_type": "graph",
                    "config": {
                        "graph": "social",
                        "cypher": "MATCH (a)-[:KNOWS]->(b) RETURN b.name"
                    }
                }
            ],
            "fusion_strategy": "rrf",
            "limit": 25
        });

        let request: MultiModelQueryRequest =
            serde_json::from_value(json).expect("should deserialize cross-model request");

        assert_eq!(request.components.len(), 3);
        assert_eq!(request.components[0].component_type, "vector");
        assert_eq!(request.components[1].component_type, "document");
        assert_eq!(request.components[2].component_type, "graph");
        assert_eq!(request.fusion_strategy, "rrf");
        assert_eq!(request.limit, Some(25));

        // Verify nested config values are accessible
        let vec_config = &request.components[0].config;
        assert_eq!(vec_config["collection"], "embeddings");
        assert_eq!(vec_config["top_k"], 10);

        // Default fusion strategy should be "intersection"
        let minimal = serde_json::json!({
            "components": []
        });
        let req: MultiModelQueryRequest =
            serde_json::from_value(minimal).expect("minimal request should parse");
        assert_eq!(req.fusion_strategy, "intersection");
        assert_eq!(req.limit, Some(100));
    }

    #[test]
    fn test_convert_multi_model_to_sql_uses_supported_graph_target_lowering() {
        let request: MultiModelQueryRequest = serde_json::from_value(serde_json::json!({
            "components": [
                {
                    "component_type": "graph",
                    "config": {
                        "graph": "social",
                        "cypher": "MATCH (a)-[:KNOWS]->(b) RETURN b.name"
                    }
                }
            ]
        }))
        .expect("request should parse");

        let sql = convert_multi_model_to_sql(&request).expect("sql conversion should succeed");
        assert_eq!(
            sql,
            "SELECT * FROM GRAPH_QUERY('MATCH (a)-[:KNOWS]->(b) FROM social RETURN b.name') LIMIT 100"
        );
    }

    #[test]
    fn test_inject_graph_target_into_cypher_does_not_duplicate_from_clause() {
        let cypher = "MATCH (n:Person) FROM social RETURN n.name";
        assert_eq!(
            inject_graph_target_into_cypher("other", cypher),
            cypher.to_string()
        );
    }

    #[test]
    fn test_query_explain_request() {
        // ExplainResponse is the output of the /explain endpoint.
        // Verify it serializes with the expected structure.
        let explain = ExplainResponse {
            components: vec![
                ComponentPlanResponse {
                    model: "vector".into(),
                    estimated_cost: 1.0,
                    parallelizable: true,
                },
                ComponentPlanResponse {
                    model: "document".into(),
                    estimated_cost: 2.0,
                    parallelizable: true,
                },
                ComponentPlanResponse {
                    model: "graph".into(),
                    estimated_cost: 3.0,
                    parallelizable: false,
                },
            ],
            fusion_strategy: "rrf".into(),
            estimated_total_cost: 6.0,
            storage_authority: None,
        };

        let json = serde_json::to_value(&explain).expect("should serialize");
        assert_eq!(json["fusion_strategy"], "rrf");
        assert!((json["estimated_total_cost"].as_f64().unwrap() - 6.0).abs() < f64::EPSILON);

        let components = json["components"].as_array().expect("components array");
        assert_eq!(components.len(), 3);
        assert_eq!(components[0]["model"], "vector");
        assert!(components[0]["parallelizable"].as_bool().unwrap());
        assert!(!components[2]["parallelizable"].as_bool().unwrap());
        assert!((components[1]["estimated_cost"].as_f64().unwrap() - 2.0).abs() < f64::EPSILON);

        // Also verify the explain endpoint uses the same ExecuteQueryRequest format
        let explain_input = serde_json::json!({
            "query": "SELECT * FROM VECTOR_SEARCH('products', '[0.5]', 5) JOIN GRAPH_QUERY('MATCH (n) RETURN n')"
        });
        let req: ExecuteQueryRequest = serde_json::from_value(explain_input)
            .expect("explain input should parse as ExecuteQueryRequest");
        assert!(req.query.contains("VECTOR_SEARCH"));
        assert!(req.query.contains("GRAPH_QUERY"));
    }

    #[test]
    fn test_explain_catalog_targets_from_multimodel_sql() {
        let targets = explain_catalog_targets(
            "SELECT * FROM VECTOR_SEARCH('products', '[0.5]', 5) \
             UNION ALL SELECT * FROM DOCUMENT_QUERY('docs', '$.kind = article') \
             UNION ALL SELECT * FROM events",
        );

        assert_eq!(
            targets,
            vec![
                "products".to_string(),
                "docs".to_string(),
                "events".to_string()
            ]
        );
    }
}
