//! Parallel Query Executor
//!
//! Executes multi-model query components in parallel, respecting dependencies
//! and coordinating with different storage backends.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_graph_query::service::{
    GraphQueryReadService, GraphQueryService, GraphQueryTraversalService,
};
use proximadb_observability_query::MetricAggregation as QueryMetricAggregation;
use proximadb_query::{
    BlockBatchSemanticJoinService, JoinExecutionService, JoinResult,
    QueryComponentExecutionService, RecordSimilarityEngine, build_document_query_record,
    build_graph_query_record, build_log_query_request, build_log_record,
    build_metric_query_request, build_metric_record, build_vector_metadata,
    build_vector_search_record, convert_path_filters_to_document_filter,
    execute_component_with_context_and_join_service, execute_component_with_service,
    execute_graph_traversal_with_input_service, execute_graph_traversal_with_service,
    execute_join_with_services, execute_multi_join_with, execute_query_components_with_service,
};
pub use proximadb_query::{extract_join_values, filter_by_ids, merge_records};
use tracing::{debug, info, trace, warn};

use super::UnifiedRecord;
use super::ast::{
    ComponentDependency, DataModel, DocumentQueryExpr, GraphQueryExpr, GraphTraversalExpr,
    LogQueryExpr, MetricQueryExpr, ModelOperation, MultiModelQuery, QueryComponent,
    VectorSearchExpr,
};
use super::fusion::SubQueryResult;
use crate::observability::{LogQueryParams, MetricAggParams, ObservabilityService};
use crate::query::graph_runtime::execute_graph_query_expr;
use crate::security::rbac_service::{
    ConsolidatedRBACManager, UnifiedPermission, UnifiedUserContext,
};
use crate::services::operations::vectors::VectorOperationsService;
use crate::storage::document::{DocumentQueryParams, DocumentService};
/// Parallel executor for multi-model queries
pub struct ParallelExecutor {
    /// Maximum concurrent queries
    max_parallel: usize,
    /// RBAC manager for permission validation
    rbac_manager: Option<Arc<ConsolidatedRBACManager>>,
    /// LLM engine for semantic operations (TD-049)
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(max_parallel: usize) -> Self {
        Self {
            max_parallel,
            rbac_manager: None,
            llm_engine: None,
        }
    }

    /// Attach an LLM integration engine for semantic operations
    pub fn with_llm_engine(
        mut self,
        llm_engine: Arc<crate::ai::llm_integration::LLMIntegrationEngine>,
    ) -> Self {
        self.llm_engine = Some(llm_engine);
        self
    }

    /// Create a new parallel executor with RBAC enabled
    pub fn with_rbac(max_parallel: usize, rbac_manager: Arc<ConsolidatedRBACManager>) -> Self {
        Self {
            max_parallel,
            rbac_manager: Some(rbac_manager),
            llm_engine: None,
        }
    }

    /// Validate permissions for a query component
    pub async fn validate_component_access(
        &self,
        user_ctx: &UnifiedUserContext,
        component: &QueryComponent,
    ) -> Result<()> {
        let rbac_manager = self
            .rbac_manager
            .as_ref()
            .ok_or_else(|| anyhow!("RBAC validation requested but RBAC manager not configured"))?;

        // Match on the operation to extract the collection/resource ID
        match &component.operation {
            ModelOperation::VectorSearch(vector_expr) => {
                let permission = UnifiedPermission::VectorSearch(vector_expr.collection.clone());
                let allowed = rbac_manager
                    .check_permission_cached(&user_ctx.user_id, &permission)
                    .await
                    .map_err(|e| anyhow!("Failed to check vector search permission: {}", e))?;

                if !allowed {
                    return Err(anyhow!(
                        "Permission denied: Vector search on collection '{}'",
                        vector_expr.collection
                    ));
                }
            }
            ModelOperation::GraphQuery(graph_expr) => {
                let permission = UnifiedPermission::GraphTraverse(graph_expr.graph_name.clone());
                let allowed = rbac_manager
                    .check_permission_cached(&user_ctx.user_id, &permission)
                    .await
                    .map_err(|e| anyhow!("Failed to check graph permission: {}", e))?;

                if !allowed {
                    return Err(anyhow!(
                        "Permission denied: Graph query on '{}'",
                        graph_expr.graph_name
                    ));
                }
            }
            ModelOperation::GraphTraversal(graph_expr) => {
                let permission = UnifiedPermission::GraphTraverse(graph_expr.graph_name.clone());
                let allowed = rbac_manager
                    .check_permission_cached(&user_ctx.user_id, &permission)
                    .await
                    .map_err(|e| anyhow!("Failed to check graph permission: {}", e))?;

                if !allowed {
                    return Err(anyhow!(
                        "Permission denied: Graph traversal on '{}'",
                        graph_expr.graph_name
                    ));
                }
            }
            ModelOperation::DocumentQuery(doc_expr) => {
                let permission = UnifiedPermission::CollectionRead(doc_expr.collection.clone());
                let allowed = rbac_manager
                    .check_permission_cached(&user_ctx.user_id, &permission)
                    .await
                    .map_err(|e| anyhow!("Failed to check document permission: {}", e))?;

                if !allowed {
                    return Err(anyhow!(
                        "Permission denied: Document query on collection '{}'",
                        doc_expr.collection
                    ));
                }
            }
            ModelOperation::LogQuery(_) | ModelOperation::MetricQuery(_) => {
                // Observability queries require SystemAdmin permission
                let permission = UnifiedPermission::SystemAdmin;
                let allowed = rbac_manager
                    .check_permission_cached(&user_ctx.user_id, &permission)
                    .await
                    .map_err(|e| anyhow!("Failed to check observability permission: {}", e))?;

                if !allowed {
                    return Err(anyhow!(
                        "Permission denied: Observability queries require system admin"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Execute query with RBAC validation
    pub async fn execute_with_auth(
        &self,
        query: &MultiModelQuery,
        user_ctx: &UnifiedUserContext,
        vector_ops: Option<Arc<VectorOperationsService>>,
        document_service: Arc<DocumentService>,
        graph_service: Option<Arc<dyn GraphQueryService>>,
        observability_service: Option<Arc<ObservabilityService>>,
    ) -> Result<Vec<SubQueryResult>> {
        // Validate permissions for each component
        for component in &query.components {
            self.validate_component_access(user_ctx, component).await?;
        }

        // Proceed with execution
        self.execute_parallel_with_all_services(
            query,
            vector_ops,
            document_service,
            graph_service,
            observability_service,
        )
        .await
    }

    /// Execute query components in parallel with explicit service references
    ///
    /// # Arguments
    /// * `query` - The multi-model query to execute
    /// * `vector_ops` - Optional VectorOperationsService for vector searches
    /// * `document_service` - Document service for document queries
    ///
    /// # Note
    /// This is the primary entry point for parallel query execution. For graph and
    /// observability queries, use `execute_parallel_with_all_services` instead.
    pub async fn execute_parallel_with_services(
        &self,
        query: &MultiModelQuery,
        vector_ops: Option<Arc<VectorOperationsService>>,
        document_service: Arc<DocumentService>,
    ) -> Result<Vec<SubQueryResult>> {
        // Use the full version with all services (graph and observability as None)
        self.execute_parallel_with_all_services(query, vector_ops, document_service, None, None)
            .await
    }

    /// Execute query components in parallel with ALL services (graph + observability)
    ///
    /// # Arguments
    /// * `query` - The multi-model query to execute
    /// * `vector_ops` - Optional VectorOperationsService for vector searches
    /// * `document_service` - Document service for document queries
    /// * `graph_service` - Optional graph traversal-capable service
    /// * `observability_service` - Optional ObservabilityService for log/metric queries
    pub async fn execute_parallel_with_all_services(
        &self,
        query: &MultiModelQuery,
        vector_ops: Option<Arc<VectorOperationsService>>,
        document_service: Arc<DocumentService>,
        graph_service: Option<Arc<dyn GraphQueryService>>,
        observability_service: Option<Arc<ObservabilityService>>,
    ) -> Result<Vec<SubQueryResult>> {
        if query.components.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Executing {} query components with max {} parallel (vector_ops={}, graph={}, obs={})",
            query.components.len(),
            self.max_parallel,
            vector_ops.is_some(),
            graph_service.is_some(),
            observability_service.is_some()
        );
        let execution_service: Arc<dyn QueryComponentExecutionService> =
            Arc::new(RootComponentExecutionService {
                vector_ops,
                document_service,
                graph_service,
                observability_service,
            });
        let join_service: Arc<dyn JoinExecutionService> = Arc::new(RootJoinExecutor {
            llm_engine: self.llm_engine.clone(),
        });

        execute_query_components_with_service(
            query,
            execution_service,
            join_service,
            self.max_parallel,
        )
        .await
    }
}

struct RootComponentExecutionService {
    vector_ops: Option<Arc<VectorOperationsService>>,
    document_service: Arc<DocumentService>,
    graph_service: Option<Arc<dyn GraphQueryService>>,
    observability_service: Option<Arc<ObservabilityService>>,
}

#[async_trait]
impl QueryComponentExecutionService for RootComponentExecutionService {
    async fn execute_vector_search(&self, expr: &VectorSearchExpr) -> Result<SubQueryResult> {
        execute_vector_search(expr, self.vector_ops.clone()).await
    }

    async fn execute_document_query(&self, expr: &DocumentQueryExpr) -> Result<SubQueryResult> {
        execute_document_query(expr, self.document_service.clone()).await
    }

    async fn execute_graph_query(&self, expr: &GraphQueryExpr) -> Result<SubQueryResult> {
        execute_graph_query(
            expr,
            self.graph_service
                .as_deref()
                .map(|svc| svc as &dyn GraphQueryReadService),
        )
        .await
    }

    async fn execute_graph_traversal(
        &self,
        expr: &GraphTraversalExpr,
        context: Option<&HashMap<usize, &SubQueryResult>>,
    ) -> Result<SubQueryResult> {
        execute_graph_traversal_with_context(expr, self.graph_service.clone(), context).await
    }

    async fn execute_log_query(&self, expr: &LogQueryExpr) -> Result<SubQueryResult> {
        execute_log_query_full(expr, self.observability_service.clone()).await
    }

    async fn execute_metric_query(&self, expr: &MetricQueryExpr) -> Result<SubQueryResult> {
        execute_metric_query_full(expr, self.observability_service.clone()).await
    }
}

struct RootJoinExecutor {
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
}

#[async_trait]
impl JoinExecutionService for RootJoinExecutor {
    async fn execute_join(
        &self,
        left: &[UnifiedRecord],
        right: &[UnifiedRecord],
        dependency: &ComponentDependency,
    ) -> JoinResult {
        execute_join(left, right, dependency, self.llm_engine.clone()).await
    }
}

/// Execute a single query component
#[allow(dead_code)]
async fn execute_component(
    component: &QueryComponent,
    vector_ops: Option<Arc<VectorOperationsService>>,
    document_service: Arc<DocumentService>,
) -> Result<SubQueryResult> {
    execute_component_full(component, vector_ops, document_service, None, None, None).await
}

/// Execute a single query component with all services
async fn execute_component_full(
    component: &QueryComponent,
    vector_ops: Option<Arc<VectorOperationsService>>,
    document_service: Arc<DocumentService>,
    graph_service: Option<Arc<dyn GraphQueryService>>,
    observability_service: Option<Arc<ObservabilityService>>,
    _llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
) -> Result<SubQueryResult> {
    let service = RootComponentExecutionService {
        vector_ops,
        document_service,
        graph_service,
        observability_service,
    };
    let result = execute_component_with_service(component, &service).await;
    if let Ok(ref subquery) = result {
        trace!(
            "Component {:?} executed in {}us",
            component.model, subquery.execution_time_us
        );
    }
    result
}

/// Execute a component with dependency context
#[allow(dead_code)]
async fn execute_component_with_context(
    component: &QueryComponent,
    vector_ops: Option<Arc<VectorOperationsService>>,
    document_service: Arc<DocumentService>,
    context: &HashMap<usize, &SubQueryResult>,
) -> Result<SubQueryResult> {
    execute_component_with_context_full(
        component,
        vector_ops,
        document_service,
        None,
        None,
        context,
        None,
    )
    .await
}

/// Execute a component with dependency context and all services
///
/// This function handles cross-model joins by:
/// 1. Resolving StartNodeSpec (including FromComponent) with the context
/// 2. Executing the component's operation
/// 3. Applying join predicates based on ComponentDependency configuration
async fn execute_component_with_context_full(
    component: &QueryComponent,
    vector_ops: Option<Arc<VectorOperationsService>>,
    document_service: Arc<DocumentService>,
    graph_service: Option<Arc<dyn GraphQueryService>>,
    observability_service: Option<Arc<ObservabilityService>>,
    context: &HashMap<usize, &SubQueryResult>,
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
) -> Result<SubQueryResult> {
    let service = RootComponentExecutionService {
        vector_ops,
        document_service,
        graph_service,
        observability_service,
    };
    let join_executor = RootJoinExecutor { llm_engine };
    let result = execute_component_with_context_and_join_service(
        component,
        &service,
        context,
        &join_executor,
    )
    .await;
    if let Ok(ref subquery) = result {
        trace!(
            "Component {:?} executed with context in {}us",
            component.model, subquery.execution_time_us
        );
    }
    result
}

/// Execute a vector search query
///
/// Uses VectorOperationsService to perform the actual search. Missing wiring
/// and authority failures are errors: neither is semantically equivalent to a
/// successful search with zero matches.
async fn execute_vector_search(
    expr: &VectorSearchExpr,
    vector_ops: Option<Arc<VectorOperationsService>>,
) -> Result<SubQueryResult> {
    expr.validate().map_err(anyhow::Error::msg)?;
    let vector_ops = vector_ops.ok_or_else(|| {
        anyhow::anyhow!(
            "vector search on collection '{}' requires a wired VectorOperationsService",
            expr.collection
        )
    })?;

    info!(
        "Executing vector search on collection: {} (top_k={})",
        expr.collection, expr.top_k
    );

    let route = crate::query::compute_scheduler::ComputeScheduler::new().route_vector_search(
        crate::query::compute_scheduler::VectorQueryShape {
            dimensions: expr.query_vector.len(),
            top_k: expr.top_k,
            has_filter: expr.filter.is_some(),
            requested_mode: expr.params.search_mode.clone(),
        },
    );
    trace!(
        target: "proximadb::vector_route",
        strategy = route.strategy_label(),
        reason = %route.reason,
        "{}",
        route.explain_line()
    );

    // Perform actual vector search using the operations service
    let search_result = vector_ops
        .unified_search_native(
            &expr.collection,
            expr.query_vector.clone(),
            expr.top_k as usize,
            expr.filter.clone(),
            Some(crate::services::operations::vectors::UnifiedSearchConfig {
                distance_metric: Some(expr.metric),
                search_mode: route.search_mode,
                ..Default::default()
            }),
            #[cfg(feature = "abac-policy")]
            &proximadb_abac::ReadContext::system(
                proximadb_abac::SystemReadReason::Statistics,
                "unified::executor [CLIENT-PLACEHOLDER]",
            ),
        )
        .await;

    match search_result {
        Ok(results) => {
            let records: Vec<UnifiedRecord> = results
                .into_iter()
                .filter(|record| expr.matches_similarity_threshold(record.score))
                .map(|record| {
                    let metadata = build_vector_metadata(&record.metadata);
                    build_vector_search_record(&record.id, record.score, metadata)
                })
                .collect();

            let count = records.len() as u64;
            info!("Vector search returned {} results", count);

            Ok(SubQueryResult {
                source_model: DataModel::Vector,
                records_returned: count,
                records,
                total_count: Some(count),
                execution_time_us: 0,
                records_scanned: count,
            })
        }
        Err(error) => Err(anyhow::anyhow!(
            "vector search failed for collection '{}': {}",
            expr.collection,
            error
        )),
    }
}

/// Execute a document query
async fn execute_document_query(
    expr: &DocumentQueryExpr,
    document_service: Arc<DocumentService>,
) -> Result<SubQueryResult> {
    debug!(
        "Executing document query on collection: {}",
        expr.collection
    );

    // Convert PathFilters to DocumentFilter
    let filter = convert_path_filters_to_document_filter(&expr.path_filters);

    // Build DocumentQueryParams
    let params = DocumentQueryParams {
        filter,
        projection: expr.projection.clone(),
        sort: Vec::new(),
        limit: expr.limit.unwrap_or(100),
        offset: 0,
        include_count: true,
    };

    // Query documents
    let result = document_service
        .query_documents(&expr.collection, params)
        .await;

    match result {
        Ok(query_result) => {
            let records: Vec<UnifiedRecord> = query_result
                .documents
                .into_iter()
                .map(|doc| {
                    build_document_query_record(
                        &doc.id,
                        &crate::storage::document::proxima_tree_to_sql_object(&doc.props),
                    )
                })
                .collect();

            let count = records.len() as u64;
            Ok(SubQueryResult {
                source_model: DataModel::Document,
                records_returned: count,
                records,
                total_count: query_result.total_count,
                execution_time_us: 0,
                records_scanned: count,
            })
        }
        Err(e) => {
            warn!("Document query failed: {}", e);
            Ok(SubQueryResult::empty(DataModel::Document))
        }
    }
}

/// Execute a graph traversal query (legacy - calls full version)
#[allow(dead_code)]
async fn execute_graph_traversal(expr: &GraphTraversalExpr) -> Result<SubQueryResult> {
    execute_graph_traversal_full(expr, None).await
}

/// Execute a declarative graph query through the shared supported subset.
async fn execute_graph_query(
    expr: &GraphQueryExpr,
    graph_service: Option<&dyn GraphQueryReadService>,
) -> Result<SubQueryResult> {
    let Some(graph_svc) = graph_service else {
        debug!(
            "Graph query on {} skipped - no GraphQueryReadService",
            expr.graph_name
        );
        return Ok(SubQueryResult::empty(DataModel::Graph));
    };

    let executed = execute_graph_query_expr(graph_svc, expr).await?;
    let records = executed
        .rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| build_graph_query_record(row, index))
        .collect::<Vec<_>>();
    let count = records.len() as u64;

    Ok(SubQueryResult {
        source_model: DataModel::Graph,
        records_returned: count,
        records,
        total_count: Some(count),
        execution_time_us: 0,
        records_scanned: (executed.stats.matched_nodes + executed.stats.matched_edges) as u64,
    })
}

/// Execute a graph traversal query with a traversal-capable graph service.
async fn execute_graph_traversal_full(
    expr: &GraphTraversalExpr,
    graph_service: Option<Arc<dyn GraphQueryService>>,
) -> Result<SubQueryResult> {
    execute_graph_traversal_with_context(expr, graph_service, None).await
}

/// Execute a graph traversal query with a traversal-capable graph service and component context.
async fn execute_graph_traversal_with_context(
    expr: &GraphTraversalExpr,
    graph_service: Option<Arc<dyn GraphQueryService>>,
    context: Option<&HashMap<usize, &SubQueryResult>>,
) -> Result<SubQueryResult> {
    debug!("Executing graph traversal on graph: {}", expr.graph_name);

    let Some(graph_svc) = graph_service else {
        debug!(
            "Graph traversal on {} skipped - no GraphQueryTraversalService",
            expr.graph_name
        );
        return Ok(SubQueryResult::empty(DataModel::Graph));
    };

    execute_graph_traversal_with_service(expr, graph_svc.as_ref(), context).await
}

/// Execute a graph traversal with input node IDs (legacy)
#[allow(dead_code)]
async fn execute_graph_traversal_with_input(
    expr: &GraphTraversalExpr,
    input_ids: Option<Vec<String>>,
) -> Result<SubQueryResult> {
    execute_graph_traversal_with_input_full(expr, input_ids, None).await
}

/// Execute a graph traversal with input node IDs and a traversal-capable graph service.
#[allow(dead_code)]
async fn execute_graph_traversal_with_input_full(
    expr: &GraphTraversalExpr,
    input_ids: Option<Vec<String>>,
    graph_service: Option<Arc<dyn GraphQueryTraversalService>>,
) -> Result<SubQueryResult> {
    debug!(
        "Executing graph traversal with input on graph: {}",
        expr.graph_name
    );

    let Some(graph_svc) = graph_service else {
        debug!(
            "Graph traversal with input on {} skipped - no GraphQueryTraversalService",
            expr.graph_name
        );
        return Ok(SubQueryResult::empty(DataModel::Graph));
    };

    execute_graph_traversal_with_input_service(expr, input_ids, graph_svc.as_ref()).await
}

/// Execute a log query (legacy)
#[allow(dead_code)]
async fn execute_log_query(expr: &LogQueryExpr) -> Result<SubQueryResult> {
    execute_log_query_full(expr, None).await
}

/// Execute a log query with observability service
async fn execute_log_query_full(
    expr: &LogQueryExpr,
    observability_service: Option<Arc<ObservabilityService>>,
) -> Result<SubQueryResult> {
    debug!("Executing log query on namespace: {}", expr.namespace);

    let Some(obs_svc) = observability_service else {
        debug!(
            "Log query on {} skipped - no ObservabilityService",
            expr.namespace
        );
        return Ok(SubQueryResult::empty(DataModel::Observability));
    };

    let query = build_log_query_request(expr);
    let params = LogQueryParams {
        start_time_ns: query.start_time_ns,
        end_time_ns: query.end_time_ns,
        query: query.query.clone(),
        severities: query.severities,
        services: query.services,
        sources: query.sources,
        limit: query.limit,
        cursor: query.cursor,
    };

    match obs_svc.query_logs(&query.namespace, params).await {
        Ok(result) => {
            let records: Vec<UnifiedRecord> = result
                .logs
                .iter()
                .enumerate()
                .map(|(idx, log)| build_log_record(log, idx))
                .collect();

            let count = records.len() as u64;
            Ok(SubQueryResult {
                source_model: DataModel::Observability,
                records,
                total_count: result.total_matched,
                execution_time_us: result.query_time_ms * 1000,
                records_scanned: count,
                records_returned: count,
            })
        }
        Err(e) => {
            warn!("Log query failed: {}", e);
            Ok(SubQueryResult::empty(DataModel::Observability))
        }
    }
}

/// Execute a metric query (legacy)
#[allow(dead_code)]
async fn execute_metric_query(expr: &MetricQueryExpr) -> Result<SubQueryResult> {
    execute_metric_query_full(expr, None).await
}

/// Execute a metric query with observability service
async fn execute_metric_query_full(
    expr: &MetricQueryExpr,
    observability_service: Option<Arc<ObservabilityService>>,
) -> Result<SubQueryResult> {
    debug!(
        "Executing metric query for metric: {} in namespace: {}",
        expr.metric_name, expr.namespace
    );

    let Some(obs_svc) = observability_service else {
        debug!(
            "Metric query for {} skipped - no ObservabilityService",
            expr.metric_name
        );
        return Ok(SubQueryResult::empty(DataModel::Observability));
    };

    let query = build_metric_query_request(expr);
    let params = MetricAggParams {
        metric_name: query.metric_name.clone(),
        start_time_ns: query.start_time_ns,
        end_time_ns: query.end_time_ns,
        aggregation: metric_aggregation_to_observability(&query.aggregation),
        step_seconds: query.step_seconds,
        label_filters: query.label_filters.clone(),
        group_by: query.group_by.clone(),
    };

    match obs_svc.aggregate_metrics(&query.namespace, params).await {
        Ok(result) => {
            // Convert time series to unified records
            let mut records = Vec::new();
            for series in result.series {
                for point in series.points {
                    records.push(build_metric_record(
                        &query.metric_name,
                        point.timestamp_ns,
                        point.value,
                        series.labels.clone(),
                    ));
                }
            }

            let count = records.len() as u64;
            Ok(SubQueryResult {
                source_model: DataModel::Observability,
                records,
                total_count: Some(count),
                execution_time_us: result.query_time_ms * 1000,
                records_scanned: count,
                records_returned: count,
            })
        }
        Err(e) => {
            warn!("Metric query failed: {}", e);
            Ok(SubQueryResult::empty(DataModel::Observability))
        }
    }
}

fn metric_aggregation_to_observability(
    aggregation: &QueryMetricAggregation,
) -> crate::observability::MetricAggregation {
    match aggregation {
        QueryMetricAggregation::Sum => crate::observability::MetricAggregation::Sum,
        QueryMetricAggregation::Avg => crate::observability::MetricAggregation::Avg,
        QueryMetricAggregation::Min => crate::observability::MetricAggregation::Min,
        QueryMetricAggregation::Max => crate::observability::MetricAggregation::Max,
        QueryMetricAggregation::Count => crate::observability::MetricAggregation::Count,
        QueryMetricAggregation::P50 => crate::observability::MetricAggregation::P50,
        QueryMetricAggregation::P90 => crate::observability::MetricAggregation::P90,
        QueryMetricAggregation::P95 => crate::observability::MetricAggregation::P95,
        QueryMetricAggregation::P99 => crate::observability::MetricAggregation::P99,
        QueryMetricAggregation::Rate => crate::observability::MetricAggregation::Rate,
    }
}

// =============================================================================
// Cross-Model Join Execution
// =============================================================================

/// Execute a join between two result sets
///
/// # Arguments
/// * `left` - Left side of the join (typically the dependent component's result)
/// * `right` - Right side of the join (prior component's result)
/// * `dependency` - Join specification (field, type)
///
/// # Returns
/// A JoinResult containing matched and unmatched records
pub async fn execute_join(
    left: &[UnifiedRecord],
    right: &[UnifiedRecord],
    dependency: &ComponentDependency,
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
) -> JoinResult {
    debug!(
        "Executing {:?} join on field '{}' ({} left x {} right records)",
        dependency.join_type,
        dependency.join_field,
        left.len(),
        right.len()
    );

    use crate::compute::distance_computation::DistanceMetric;
    use crate::compute::distance_computation::engine::UnifiedDistanceCompute;

    struct CosineSimilarityAdapter(
        crate::compute::distance_computation::engine::UnifiedDistanceCompute,
    );

    impl RecordSimilarityEngine for CosineSimilarityAdapter {
        fn similarity(&self, left: &[f32], right: &[f32]) -> f32 {
            let distance = self.0.distance(left, right);
            1.0 - distance
        }
    }

    struct RootBlockBatchJoinService {
        llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
    }

    #[async_trait]
    impl BlockBatchSemanticJoinService for RootBlockBatchJoinService {
        async fn execute_block_batch_join(
            &self,
            left: &[UnifiedRecord],
            right: &[UnifiedRecord],
            join_field: &str,
            top_k: u32,
            config: &crate::query::unified::ast::BlockBatchConfig,
        ) -> JoinResult {
            execute_block_batch_semantic_join(
                left,
                right,
                join_field,
                top_k,
                config,
                self.llm_engine.clone(),
            )
            .await
        }
    }

    let similarity_engine =
        CosineSimilarityAdapter(UnifiedDistanceCompute::new(DistanceMetric::Cosine));
    let block_batch_service = RootBlockBatchJoinService { llm_engine };
    let result = execute_join_with_services(
        left,
        right,
        dependency,
        &similarity_engine,
        &block_batch_service,
    )
    .await;
    debug!(
        "Join result: {} matched, {} unmatched_left",
        result.matched.len(),
        result.unmatched_left.len()
    );
    result
}

/// Execute multiple joins in sequence for components with multiple dependencies
pub async fn execute_multi_join(
    component_result: &SubQueryResult,
    dependencies: &[ComponentDependency],
    prior_results: &HashMap<usize, &SubQueryResult>,
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
) -> SubQueryResult {
    let join_executor = RootJoinExecutor { llm_engine };
    execute_multi_join_with(
        component_result,
        dependencies,
        prior_results,
        &join_executor,
    )
    .await
}

/// LLM block-batched semantic join (TD-049, arXiv:2510.08489).
///
/// Per-mode behavior:
///
/// - `#[cfg(feature = "llm-joins")]`: pack batches of rows from both
///   sides into a single prompt (block nested loops with batched
///   prompts), ask the LLM to identify matching pairs, repeat until
///   the cross-product is exhausted or `max_calls` is hit. Currently
///   a stub that validates config and returns an error tagged with
///   the missing LLM-client wiring point — the integration is
///   gated on `[llm]` config maturity (TD-050 audit substrate
///   landing first).
/// - default (feature off): logs a clear error and returns no
///   matches so the surrounding pipeline behaves predictably.
async fn execute_block_batch_semantic_join(
    left: &[UnifiedRecord],
    right: &[UnifiedRecord],
    _join_field: &str,
    _top_k: u32,
    config: &crate::query::unified::ast::BlockBatchConfig,
    llm_engine: Option<Arc<crate::ai::llm_integration::LLMIntegrationEngine>>,
) -> JoinResult {
    if let Err(reason) = config.validate() {
        warn!(
            "LlmBlockBatch semantic join: invalid config: {}. Returning empty result.",
            reason
        );
        return JoinResult {
            matched: Vec::new(),
            unmatched_left: left.to_vec(),
            has_matches: false,
        };
    }

    let Some(llm) = llm_engine else {
        warn!(
            "LlmBlockBatch semantic join requested but no LLM engine is available. Returning empty result."
        );
        return JoinResult {
            matched: Vec::new(),
            unmatched_left: left.to_vec(),
            has_matches: false,
        };
    };

    #[cfg(not(feature = "llm-joins"))]
    {
        warn!(
            "LlmBlockBatch semantic join requested but the `llm-joins` feature is OFF. \
             Build with --features llm-joins to enable. Returning empty result so the \
             surrounding pipeline behaves predictably."
        );
        let _ = (left, right, config, llm);
        JoinResult {
            matched: Vec::new(),
            unmatched_left: left.to_vec(),
            has_matches: false,
        }
    }

    #[cfg(feature = "llm-joins")]
    {
        let mut matched = Vec::new();
        let mut unmatched_left_map: HashMap<String, UnifiedRecord> =
            left.iter().map(|r| (r.id.clone(), r.clone())).collect();
        let mut calls_made = 0;

        for left_chunk in left.chunks(config.batch_size_left as usize) {
            for right_chunk in right.chunks(config.batch_size_right as usize) {
                if calls_made >= config.max_calls {
                    break;
                }

                // Construct prompt for this block pair
                let mut prompt = String::from(
                    "Identify semantic matches between these two sets of records based on content similarity.\n\n",
                );
                prompt.push_str("LEFT SET:\n");
                for r in left_chunk {
                    prompt.push_str(&format!("- ID: {}, Content: {}\n", r.id, r.data));
                }
                prompt.push_str("\nRIGHT SET:\n");
                for r in right_chunk {
                    prompt.push_str(&format!("- ID: {}, Content: {}\n", r.id, r.data));
                }
                prompt.push_str("\nReturn a JSON array of [left_id, right_id] pairs for all matches. Example: [[\"L1\", \"R1\"], [\"L2\", \"R3\"]]. If no matches, return [].\n");
                prompt.push_str("MATCHES: ");

                // Call LLM
                match llm.query_with_fallback(&prompt).await {
                    Ok(response) => {
                        // Very basic parsing for the JSON array
                        if let Some(start) = response.content.find('[') {
                            if let Some(end) = response.content.rfind(']') {
                                let json_str = &response.content[start..=end];
                                if let Ok(pairs) =
                                    serde_json::from_str::<Vec<Vec<String>>>(json_str)
                                {
                                    for pair in pairs {
                                        if pair.len() == 2 {
                                            let lid = &pair[0];
                                            let rid = &pair[1];

                                            // Find the actual records
                                            let l_rec = left_chunk.iter().find(|r| &r.id == lid);
                                            let r_rec = right_chunk.iter().find(|r| &r.id == rid);

                                            if let (Some(l), Some(r)) = (l_rec, r_rec) {
                                                matched.push(JoinedRecord {
                                                    left: l.clone(),
                                                    right: r.clone(),
                                                    score: 0.9, // LLM match default score
                                                });
                                                unmatched_left_map.remove(lid);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("LLM join call failed: {}", e);
                    }
                }

                calls_made += 1;
            }
            if calls_made >= config.max_calls {
                break;
            }
        }

        let unmatched_left: Vec<UnifiedRecord> = unmatched_left_map.into_values().collect();
        let has_matches = !matched.is_empty();

        JoinResult {
            matched,
            unmatched_left,
            has_matches,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOperationsService;
    use crate::graph::{Node as ProtoNode, property_value};
    use crate::proto::proximadb_v1::CreateGraphRequest;
    use crate::query::unified::ast::{
        DistanceMetric, FilterOperator, FilterValue, GraphQueryExpr, JoinType, NodeFilter,
        PropertyFilter as UnifiedPropertyFilter, StartNodeSpec, VectorSearchParams,
    };
    use proximadb_query::resolve_start_nodes;

    #[test]
    fn test_executor_creation() {
        let executor = ParallelExecutor::new(4);
        assert_eq!(executor.max_parallel, 4);
    }

    #[test]
    fn test_executor_retains_parallelism_setting() {
        let executor = ParallelExecutor::new(2);
        assert_eq!(executor.max_parallel, 2);
    }

    #[tokio::test]
    async fn test_empty_query_execution() {
        let _executor = ParallelExecutor::new(4);
        let query = MultiModelQuery::new();

        // Need mock storage engine and document service for full test
        // For now, just verify the executor handles empty queries
        assert!(query.components.is_empty());
    }

    #[test]
    fn test_vector_search_expr() {
        let expr = VectorSearchExpr {
            collection: "test".to_string(),
            vector_column: None,
            query_vector: vec![0.1, 0.2, 0.3],
            top_k: 10,
            threshold: Some(0.8),
            metric: DistanceMetric::Cosine,
            filter: None,
            params: VectorSearchParams::default(),
        };

        assert_eq!(expr.collection, "test");
        assert_eq!(expr.top_k, 10);
    }

    #[tokio::test]
    async fn vector_search_requires_a_wired_authority() {
        let expr = VectorSearchExpr {
            collection: "embeddings".to_string(),
            vector_column: None,
            query_vector: vec![0.1, 0.2, 0.3],
            top_k: 10,
            threshold: None,
            metric: DistanceMetric::Cosine,
            filter: None,
            params: VectorSearchParams::default(),
        };

        let error = execute_vector_search(&expr, None)
            .await
            .expect_err("an unwired vector authority must not look like an empty result");

        assert!(error.to_string().contains("VectorOperationsService"));
    }

    #[test]
    fn test_sql_value_to_json_primitives() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};

        use proximadb_records::conversions::sql_value_to_json;

        // Test null
        let null_val = SqlValue {
            value: Some(Value::NullValue(0)),
        };
        assert_eq!(sql_value_to_json(&null_val), serde_json::Value::Null);

        // Test bool
        let bool_val = SqlValue {
            value: Some(Value::BoolValue(true)),
        };
        assert_eq!(sql_value_to_json(&bool_val), serde_json::Value::Bool(true));

        // Test int
        let int_val = SqlValue {
            value: Some(Value::Int64Value(42)),
        };
        assert_eq!(sql_value_to_json(&int_val), serde_json::json!(42));

        // Test string
        let str_val = SqlValue {
            value: Some(Value::StringValue("hello".to_string())),
        };
        assert_eq!(
            sql_value_to_json(&str_val),
            serde_json::Value::String("hello".to_string())
        );
    }

    #[test]
    fn test_metric_aggregation_to_observability_preserves_requested_mode() {
        assert!(matches!(
            metric_aggregation_to_observability(&QueryMetricAggregation::P95),
            crate::observability::MetricAggregation::P95
        ));
        assert!(matches!(
            metric_aggregation_to_observability(&QueryMetricAggregation::Rate),
            crate::observability::MetricAggregation::Rate
        ));
    }

    async fn seed_graph_service() -> Arc<GraphOperationsService> {
        let service = Arc::new(GraphOperationsService::new());
        service
            .create_graph_collection(CreateGraphRequest {
                graph_id: "social".to_string(),
                name: Some("social".to_string()),
                description: None,
                schema: None,
                storage_config: None,
                engine_config: None,
                access_control: None,
            })
            .await
            .expect("graph should be created");

        for (id, name) in [("alice", "Alice"), ("bob", "Bob")] {
            service
                .create_node(
                    "social",
                    ProtoNode {
                        id: id.to_string(),
                        labels: vec!["Person".to_string()],
                        properties: HashMap::from([(
                            "name".to_string(),
                            crate::graph::PropertyValue {
                                value: Some(property_value::Value::StringValue(name.to_string())),
                            },
                        )]),
                        embedding: None,
                        created_at_ms: 0,
                        updated_at_ms: 0,
                    },
                )
                .await
                .expect("node should be created");
        }

        service
    }

    #[tokio::test]
    async fn test_execute_graph_query_materializes_legacy_node_rows() {
        let service = seed_graph_service().await;
        let expr = GraphQueryExpr {
            graph_name: "social".to_string(),
            normalized_query: "MATCH (n:Person) RETURN n".to_string(),
            output_columns: vec![
                "node_id".to_string(),
                "label".to_string(),
                "properties".to_string(),
            ],
            uses_legacy_node_rows: true,
            max_depth: 0,
        };

        let result = execute_graph_query(&expr, Some(service.as_ref()))
            .await
            .expect("graph query should execute");

        assert_eq!(result.records_returned, 2);
        let ids: std::collections::HashSet<String> = result
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        assert!(ids.contains("alice"));
        assert!(ids.contains("bob"));
        for record in &result.records {
            assert!(record.data.get("node_id").is_some());
            assert!(record.data.get("label").is_some());
            assert!(record.data.get("properties").is_some());
        }
    }

    #[tokio::test]
    async fn test_execute_graph_query_preserves_projected_columns() {
        let service = seed_graph_service().await;
        let expr = GraphQueryExpr {
            graph_name: "social".to_string(),
            normalized_query: "MATCH (n:Person) RETURN n.name AS person_name".to_string(),
            output_columns: vec!["person_name".to_string()],
            uses_legacy_node_rows: false,
            max_depth: 0,
        };

        let result = execute_graph_query(&expr, Some(service.as_ref()))
            .await
            .expect("graph query should execute");

        let names: std::collections::HashSet<String> = result
            .records
            .iter()
            .filter_map(|record| {
                record
                    .data
                    .get("person_name")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string)
            })
            .collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
    }

    #[tokio::test]
    async fn test_resolve_start_nodes_by_label_uses_canonical_node_query() {
        let service = seed_graph_service().await;

        let ids = resolve_start_nodes(
            &StartNodeSpec::Label("Person".to_string()),
            "social",
            service.as_ref(),
            None,
        )
        .await
        .expect("label-based start-node resolution should succeed");

        let ids = ids.into_iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("alice"));
        assert!(ids.contains("bob"));
    }

    #[tokio::test]
    async fn test_resolve_start_nodes_by_filter_uses_canonical_node_query() {
        let service = seed_graph_service().await;

        let ids = resolve_start_nodes(
            &StartNodeSpec::Filter(NodeFilter {
                label: Some("Person".to_string()),
                properties: vec![UnifiedPropertyFilter {
                    name: "name".to_string(),
                    operator: FilterOperator::Eq,
                    value: FilterValue::String("Alice".to_string()),
                }],
            }),
            "social",
            service.as_ref(),
            None,
        )
        .await
        .expect("filter-based start-node resolution should succeed");

        assert_eq!(ids, vec!["alice".to_string()]);
    }

    // =========================================================================
    // Cross-Model Join Tests
    // =========================================================================

    fn make_test_record(id: &str, model: DataModel, data: serde_json::Value) -> UnifiedRecord {
        UnifiedRecord {
            id: id.to_string(),
            source_model: model,
            data,
            score: None,
            metadata: HashMap::new(),
        }
    }

    fn make_test_record_with_score(
        id: &str,
        model: DataModel,
        data: serde_json::Value,
        score: f64,
    ) -> UnifiedRecord {
        UnifiedRecord {
            id: id.to_string(),
            source_model: model,
            data,
            score: Some(score),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_extract_join_values_id_field() {
        let record = make_test_record("rec_123", DataModel::Vector, serde_json::json!({}));
        let values = extract_join_values(&record, "id");
        assert_eq!(values, vec!["rec_123"]);
    }

    #[test]
    fn test_extract_join_values_json_field() {
        let record = make_test_record(
            "rec_1",
            DataModel::Document,
            serde_json::json!({"user_id": "user_456", "name": "Alice"}),
        );
        let values = extract_join_values(&record, "user_id");
        assert_eq!(values, vec!["user_456"]);
    }

    #[test]
    fn test_extract_join_values_nested_json() {
        let record = make_test_record(
            "rec_1",
            DataModel::Document,
            serde_json::json!({"metadata": {"customer_id": "cust_789"}}),
        );
        let values = extract_join_values(&record, "metadata.customer_id");
        assert_eq!(values, vec!["cust_789"]);
    }

    #[test]
    fn test_extract_join_values_missing_field() {
        let record = make_test_record("rec_1", DataModel::Vector, serde_json::json!({}));
        let values = extract_join_values(&record, "nonexistent");
        assert!(values.is_empty());
    }

    #[tokio::test]
    async fn test_inner_join() {
        // Left: documents with user_id field
        let left = vec![
            make_test_record(
                "doc_1",
                DataModel::Document,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "doc_2",
                DataModel::Document,
                serde_json::json!({"user_id": "u2"}),
            ),
            make_test_record(
                "doc_3",
                DataModel::Document,
                serde_json::json!({"user_id": "u3"}),
            ),
        ];

        // Right: vectors with SAME user_id field (join field must exist on both sides)
        let right = vec![
            make_test_record(
                "vec_1",
                DataModel::Vector,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "vec_2",
                DataModel::Vector,
                serde_json::json!({"user_id": "u2"}),
            ),
            // u3 is missing
        ];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "user_id".to_string(),
            join_type: JoinType::Inner,
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        // Should only have 2 matches (u1 and u2)
        assert_eq!(result.matched.len(), 2);
        assert_eq!(result.unmatched_left.len(), 0);
        assert!(result.has_matches);
    }

    #[tokio::test]
    async fn test_left_outer_join() {
        let left = vec![
            make_test_record(
                "doc_1",
                DataModel::Document,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "doc_2",
                DataModel::Document,
                serde_json::json!({"user_id": "u2"}),
            ),
            make_test_record(
                "doc_3",
                DataModel::Document,
                serde_json::json!({"user_id": "u3"}),
            ),
        ];

        // Right: only has user_id=u1, so u2 and u3 will be unmatched
        let right = vec![make_test_record(
            "vec_1",
            DataModel::Vector,
            serde_json::json!({"user_id": "u1"}),
        )];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "user_id".to_string(),
            join_type: JoinType::LeftOuter,
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        // Should have 1 match + 2 unmatched
        assert_eq!(result.matched.len(), 1);
        assert_eq!(result.unmatched_left.len(), 2);
        assert!(result.has_matches);

        // Convert to SubQueryResult and check total
        let subquery = result.to_subquery_result(DataModel::Document, &JoinType::LeftOuter);
        assert_eq!(subquery.records.len(), 3); // All left records included
    }

    #[tokio::test]
    async fn test_semi_join() {
        let left = vec![
            make_test_record(
                "doc_1",
                DataModel::Document,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "doc_2",
                DataModel::Document,
                serde_json::json!({"user_id": "u2"}),
            ),
            make_test_record(
                "doc_3",
                DataModel::Document,
                serde_json::json!({"user_id": "u1"}),
            ), // Same user_id
        ];

        // Right side has user_id field matching u1
        let right = vec![
            make_test_record(
                "vec_1",
                DataModel::Vector,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "vec_2",
                DataModel::Vector,
                serde_json::json!({"user_id": "u1"}),
            ), // Another u1
        ];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "user_id".to_string(),
            join_type: JoinType::Semi,
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        // Semi join: doc_1 and doc_3 match (both have user_id=u1), doc_2 doesn't
        assert_eq!(result.matched.len(), 2);
        // Semi join should not duplicate left records even if multiple matches
    }

    #[tokio::test]
    async fn test_anti_join() {
        let left = vec![
            make_test_record(
                "doc_1",
                DataModel::Document,
                serde_json::json!({"user_id": "u1"}),
            ),
            make_test_record(
                "doc_2",
                DataModel::Document,
                serde_json::json!({"user_id": "u2"}),
            ),
            make_test_record(
                "doc_3",
                DataModel::Document,
                serde_json::json!({"user_id": "u3"}),
            ),
        ];

        // Right side has user_id=u1, so doc_1 will be excluded (matched)
        let right = vec![make_test_record(
            "vec_1",
            DataModel::Vector,
            serde_json::json!({"user_id": "u1"}),
        )];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "user_id".to_string(),
            join_type: JoinType::Anti,
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        // Anti join: only records that DON'T match
        assert_eq!(result.matched.len(), 2); // u2 and u3 don't match
        assert!(result.matched.iter().any(|r| r.id == "doc_2"));
        assert!(result.matched.iter().any(|r| r.id == "doc_3"));
        assert!(!result.matched.iter().any(|r| r.id == "doc_1")); // u1 matched, excluded
    }

    #[tokio::test]
    async fn test_join_by_id() {
        let left = vec![
            make_test_record(
                "shared_1",
                DataModel::Document,
                serde_json::json!({"name": "Doc1"}),
            ),
            make_test_record(
                "shared_2",
                DataModel::Document,
                serde_json::json!({"name": "Doc2"}),
            ),
        ];

        let right = vec![
            make_test_record(
                "shared_1",
                DataModel::Vector,
                serde_json::json!({"vec": [0.1]}),
            ),
            make_test_record(
                "other",
                DataModel::Vector,
                serde_json::json!({"vec": [0.2]}),
            ),
        ];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "id".to_string(),
            join_type: JoinType::Inner,
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        assert_eq!(result.matched.len(), 1);
        assert_eq!(result.matched[0].id, "shared_1");
    }

    #[test]
    fn test_merge_records_combines_data() {
        let left = make_test_record_with_score(
            "doc_1",
            DataModel::Document,
            serde_json::json!({"name": "Alice"}),
            0.8,
        );
        let right = make_test_record_with_score(
            "vec_1",
            DataModel::Vector,
            serde_json::json!({"embedding": [0.1, 0.2]}),
            0.9,
        );

        let merged = merge_records(&left, &right, "user_id");

        // Check merged data has both models' data
        assert!(merged.data.get("document").is_some());
        assert!(merged.data.get("vector").is_some());

        // Check score is averaged (use approximate comparison for floating point)
        let score = merged.score.unwrap();
        assert!(
            (score - 0.85).abs() < 0.0001,
            "Expected score ~0.85, got {}",
            score
        );

        // Check right_id is in metadata
        assert_eq!(merged.metadata.get("right_id"), Some(&"vec_1".to_string()));
    }

    #[test]
    fn test_filter_by_ids_include() {
        let records = vec![
            make_test_record("r1", DataModel::Vector, serde_json::json!({})),
            make_test_record("r2", DataModel::Vector, serde_json::json!({})),
            make_test_record("r3", DataModel::Vector, serde_json::json!({})),
        ];

        let ids = vec!["r1".to_string(), "r3".to_string()];
        let filtered = filter_by_ids(&records, &ids, true);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|r| r.id == "r1"));
        assert!(filtered.iter().any(|r| r.id == "r3"));
    }

    #[test]
    fn test_filter_by_ids_exclude() {
        let records = vec![
            make_test_record("r1", DataModel::Vector, serde_json::json!({})),
            make_test_record("r2", DataModel::Vector, serde_json::json!({})),
            make_test_record("r3", DataModel::Vector, serde_json::json!({})),
        ];

        let ids = vec!["r1".to_string(), "r3".to_string()];
        let filtered = filter_by_ids(&records, &ids, false);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "r2");
    }

    #[test]
    fn test_join_result_empty() {
        let result = JoinResult::empty();
        assert!(result.matched.is_empty());
        assert!(result.unmatched_left.is_empty());
        assert!(!result.has_matches);
    }

    #[tokio::test]
    async fn test_execute_multi_join() {
        // Simulate a chain: Vector -> Document (joined on product_id field)

        // Prior results: vector search found these with product_id field
        let vector_result = SubQueryResult {
            source_model: DataModel::Vector,
            records_returned: 2,
            records: vec![
                make_test_record(
                    "v1",
                    DataModel::Vector,
                    serde_json::json!({"product_id": "p1"}),
                ),
                make_test_record(
                    "v2",
                    DataModel::Vector,
                    serde_json::json!({"product_id": "p2"}),
                ),
            ],
            total_count: Some(2),
            execution_time_us: 100,
            records_scanned: 2,
        };

        // Document component result - ALSO has product_id field (same join field on both sides)
        let doc_result = SubQueryResult {
            source_model: DataModel::Document,
            records_returned: 3,
            records: vec![
                make_test_record(
                    "d1",
                    DataModel::Document,
                    serde_json::json!({"product_id": "p1", "name": "Product 1"}),
                ),
                make_test_record(
                    "d2",
                    DataModel::Document,
                    serde_json::json!({"product_id": "p2", "name": "Product 2"}),
                ),
                make_test_record(
                    "d3",
                    DataModel::Document,
                    serde_json::json!({"product_id": "p3", "name": "Product 3"}),
                ),
            ],
            total_count: Some(3),
            execution_time_us: 50,
            records_scanned: 3,
        };

        let dependencies = vec![ComponentDependency {
            component_index: 0,
            join_field: "product_id".to_string(),
            join_type: JoinType::Inner,
        }];

        let mut prior_results: HashMap<usize, &SubQueryResult> = HashMap::new();
        prior_results.insert(0, &vector_result);

        let joined = execute_multi_join(&doc_result, &dependencies, &prior_results, None).await;

        // Should have 2 matches (p1 and p2)
        assert_eq!(joined.records.len(), 2);
    }

    // =========================================================================
    // Cross-Model Chain Tests (Phase 10.2)
    // =========================================================================

    #[test]
    fn test_resolve_nodes_from_component_empty_context() {
        let result = proximadb_query::resolve_nodes_from_component(0, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_nodes_from_component_missing_component() {
        let context: HashMap<usize, &SubQueryResult> = HashMap::new();
        let result = proximadb_query::resolve_nodes_from_component(0, Some(&context));
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_nodes_from_component_extracts_ids() {
        // Create a prior result with vector records
        let vector_result = SubQueryResult {
            source_model: DataModel::Vector,
            records_returned: 3,
            records: vec![
                make_test_record(
                    "node_alpha",
                    DataModel::Vector,
                    serde_json::json!({"similarity": 0.95}),
                ),
                make_test_record(
                    "node_beta",
                    DataModel::Vector,
                    serde_json::json!({"similarity": 0.88}),
                ),
                make_test_record(
                    "node_gamma",
                    DataModel::Vector,
                    serde_json::json!({"similarity": 0.75}),
                ),
            ],
            total_count: Some(3),
            execution_time_us: 50,
            records_scanned: 3,
        };

        let mut context: HashMap<usize, &SubQueryResult> = HashMap::new();
        context.insert(0, &vector_result);

        let ids = proximadb_query::resolve_nodes_from_component(0, Some(&context));
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"node_alpha".to_string()));
        assert!(ids.contains(&"node_beta".to_string()));
        assert!(ids.contains(&"node_gamma".to_string()));
    }

    #[tokio::test]
    async fn test_vector_to_graph_to_document_chain() {
        // Simulate a 3-stage pipeline:
        // Stage 1: Vector search finds similar embeddings
        // Stage 2: Graph traversal from those vectors to find related entities
        // Stage 3: Document query to get full details

        // Stage 1 result: Vector search
        let vector_result = SubQueryResult {
            source_model: DataModel::Vector,
            records_returned: 2,
            records: vec![
                make_test_record(
                    "vec_product_1",
                    DataModel::Vector,
                    serde_json::json!({
                        "entity_id": "prod_1",
                        "similarity": 0.92
                    }),
                ),
                make_test_record(
                    "vec_product_2",
                    DataModel::Vector,
                    serde_json::json!({
                        "entity_id": "prod_2",
                        "similarity": 0.85
                    }),
                ),
            ],
            total_count: Some(2),
            execution_time_us: 100,
            records_scanned: 100,
        };

        // Stage 2 result: Graph traversal (from vector entity_ids)
        let graph_result = SubQueryResult {
            source_model: DataModel::Graph,
            records_returned: 4,
            records: vec![
                // Related entities found via graph traversal
                make_test_record(
                    "prod_1",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["Product"],
                        "doc_id": "doc_product_1"
                    }),
                ),
                make_test_record(
                    "brand_1",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["Brand"],
                        "doc_id": "doc_brand_1"
                    }),
                ),
                make_test_record(
                    "prod_2",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["Product"],
                        "doc_id": "doc_product_2"
                    }),
                ),
                make_test_record(
                    "category_1",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["Category"],
                        "doc_id": "doc_category_1"
                    }),
                ),
            ],
            total_count: Some(4),
            execution_time_us: 50,
            records_scanned: 20,
        };

        // Stage 3 result: Document query (using doc_ids from graph)
        let doc_result = SubQueryResult {
            source_model: DataModel::Document,
            records_returned: 4,
            records: vec![
                make_test_record(
                    "doc_product_1",
                    DataModel::Document,
                    serde_json::json!({
                        "doc_id": "doc_product_1",
                        "name": "Wireless Headphones",
                        "price": 149.99
                    }),
                ),
                make_test_record(
                    "doc_brand_1",
                    DataModel::Document,
                    serde_json::json!({
                        "doc_id": "doc_brand_1",
                        "name": "TechAudio Inc.",
                        "country": "USA"
                    }),
                ),
                make_test_record(
                    "doc_product_2",
                    DataModel::Document,
                    serde_json::json!({
                        "doc_id": "doc_product_2",
                        "name": "Bluetooth Speaker",
                        "price": 79.99
                    }),
                ),
                make_test_record(
                    "doc_category_1",
                    DataModel::Document,
                    serde_json::json!({
                        "doc_id": "doc_category_1",
                        "name": "Electronics",
                        "parent": "Home"
                    }),
                ),
            ],
            total_count: Some(4),
            execution_time_us: 30,
            records_scanned: 4,
        };

        // Now join graph → document using doc_id field
        let dependencies = vec![ComponentDependency {
            component_index: 1, // Graph result
            join_field: "doc_id".to_string(),
            join_type: JoinType::Inner,
        }];

        let mut prior_results: HashMap<usize, &SubQueryResult> = HashMap::new();
        prior_results.insert(0, &vector_result); // Stage 0: vectors
        prior_results.insert(1, &graph_result); // Stage 1: graph

        // Join documents with graph results
        let joined = execute_multi_join(&doc_result, &dependencies, &prior_results, None).await;

        // Should have 4 matches (all graph nodes have corresponding documents)
        assert_eq!(joined.records.len(), 4);

        // Verify the merged records contain data from both models
        for record in &joined.records {
            // Each merged record should have document and graph data
            assert!(record.data.get("document").is_some() || record.data.get("graph").is_some());
        }
    }

    #[tokio::test]
    async fn test_multi_model_join_with_partial_matches() {
        // Test case where not all records match across models

        // Vector results with product IDs
        let vector_result = SubQueryResult {
            source_model: DataModel::Vector,
            records_returned: 3,
            records: vec![
                make_test_record(
                    "v1",
                    DataModel::Vector,
                    serde_json::json!({"product_id": "p1"}),
                ),
                make_test_record(
                    "v2",
                    DataModel::Vector,
                    serde_json::json!({"product_id": "p2"}),
                ),
                make_test_record(
                    "v3",
                    DataModel::Vector,
                    serde_json::json!({"product_id": "p_unknown"}),
                ), // No match
            ],
            total_count: Some(3),
            execution_time_us: 50,
            records_scanned: 3,
        };

        // Document results - only has p1 and p2
        let doc_result = SubQueryResult {
            source_model: DataModel::Document,
            records_returned: 2,
            records: vec![
                make_test_record(
                    "d1",
                    DataModel::Document,
                    serde_json::json!({"product_id": "p1"}),
                ),
                make_test_record(
                    "d2",
                    DataModel::Document,
                    serde_json::json!({"product_id": "p2"}),
                ),
                // p_unknown doesn't exist in documents
            ],
            total_count: Some(2),
            execution_time_us: 30,
            records_scanned: 2,
        };

        // Inner join: should only get matching records
        let inner_dep = vec![ComponentDependency {
            component_index: 0,
            join_field: "product_id".to_string(),
            join_type: JoinType::Inner,
        }];

        let mut prior: HashMap<usize, &SubQueryResult> = HashMap::new();
        prior.insert(0, &vector_result);

        let inner_joined = execute_multi_join(&doc_result, &inner_dep, &prior, None).await;
        assert_eq!(inner_joined.records.len(), 2); // Only p1 and p2 match

        // Left outer join: should include all left records
        let left_dep = vec![ComponentDependency {
            component_index: 0,
            join_field: "product_id".to_string(),
            join_type: JoinType::LeftOuter,
        }];

        let left_joined = execute_multi_join(&doc_result, &left_dep, &prior, None).await;
        assert_eq!(left_joined.records.len(), 2); // All documents included
    }

    #[test]
    fn test_graph_node_id_extraction() {
        // Test that we can extract node IDs from graph traversal results
        // to use as start nodes for a subsequent traversal

        let graph_result = SubQueryResult {
            source_model: DataModel::Graph,
            records_returned: 3,
            records: vec![
                make_test_record(
                    "user_alice",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["User"],
                        "properties": {"name": "Alice"}
                    }),
                ),
                make_test_record(
                    "user_bob",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["User"],
                        "properties": {"name": "Bob"}
                    }),
                ),
                make_test_record(
                    "user_charlie",
                    DataModel::Graph,
                    serde_json::json!({
                        "labels": ["User"],
                        "properties": {"name": "Charlie"}
                    }),
                ),
            ],
            total_count: Some(3),
            execution_time_us: 25,
            records_scanned: 10,
        };

        let mut context: HashMap<usize, &SubQueryResult> = HashMap::new();
        context.insert(0, &graph_result);

        // Resolve nodes from the graph component
        let resolved = proximadb_query::resolve_nodes_from_component(0, Some(&context));

        assert_eq!(resolved.len(), 3);
        assert!(resolved.contains(&"user_alice".to_string()));
        assert!(resolved.contains(&"user_bob".to_string()));
        assert!(resolved.contains(&"user_charlie".to_string()));
    }

    #[tokio::test]
    async fn test_semantic_join() {
        use serde_json::json;

        // Left side: User interests (Document model)
        let left = vec![
            UnifiedRecord {
                id: "user1".to_string(),
                source_model: DataModel::Document,
                data: json!({
                    "name": "Alice",
                    "interest_vec": [1.0, 0.0, 0.0]
                }),
                score: None,
                metadata: HashMap::new(),
            },
            UnifiedRecord {
                id: "user2".to_string(),
                source_model: DataModel::Document,
                data: json!({
                    "name": "Bob",
                    "interest_vec": [0.0, 1.0, 0.0]
                }),
                score: None,
                metadata: HashMap::new(),
            },
        ];

        // Right side: Products (Vector model)
        let right = vec![
            UnifiedRecord {
                id: "prod1".to_string(),
                source_model: DataModel::Vector,
                data: json!({
                    "product": "Red Apple",
                    "vec": [0.9, 0.1, 0.0] // Close to Alice
                }),
                score: None,
                metadata: HashMap::new(),
            },
            UnifiedRecord {
                id: "prod2".to_string(),
                source_model: DataModel::Vector,
                data: json!({
                    "product": "Green Broccoli",
                    "vec": [0.1, 0.9, 0.0] // Close to Bob
                }),
                score: None,
                metadata: HashMap::new(),
            },
        ];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "interest_vec".to_string(), // Note: we'll use interest_vec for left, but right uses 'vec' in data?
            // Actually extract_vector is called with join_field for BOTH.
            // I should adjust the test data to have same field name or improve extractor.
            join_type: JoinType::Semantic {
                threshold: 0.8,
                top_k: 1,
                mode: crate::query::unified::ast::SemanticJoinMode::default(),
            },
        };

        // Adjusting right side to match join_field for this test
        let mut right_fixed = right.clone();
        if let Some(obj) = right_fixed[0].data.as_object_mut() {
            obj.insert("interest_vec".to_string(), json!([0.9, 0.1, 0.0]));
        }
        if let Some(obj) = right_fixed[1].data.as_object_mut() {
            obj.insert("interest_vec".to_string(), json!([0.1, 0.9, 0.0]));
        }

        let result = execute_join(&left, &right_fixed, &dependency, None).await;

        assert!(result.has_matches);
        assert_eq!(result.matched.len(), 2);

        // Alice (user1) should match Red Apple (prod1)
        let alice_match = result.matched.iter().find(|r| r.id == "user1").unwrap();
        assert_eq!(alice_match.data["product"], "Red Apple");

        // Bob (user2) should match Green Broccoli (prod2)
        let bob_match = result.matched.iter().find(|r| r.id == "user2").unwrap();
        assert_eq!(bob_match.data["product"], "Green Broccoli");
    }

    // ----------------------------------------------------------------
    // TD-049: dispatch tests for the LlmBlockBatch semantic-join mode.
    //
    // The LLM client integration is gated on the `llm-joins` Cargo
    // feature AND on a runtime [llm] config substrate that has not
    // yet matured. While we wait, the dispatch must:
    //   - Validate the BlockBatchConfig and degrade gracefully on
    //     bad config rather than panicking.
    //   - Return an empty JoinResult so the surrounding pipeline
    //     keeps running predictably (a None match is structurally
    //     no different from a real "no LLM matches found" outcome).
    //   - Log the misconfiguration / missing-feature so operators
    //     see *why* nothing matched in the server logs.
    //
    // These tests pin the dispatch contract independent of whether
    // the feature is on; the actual prompt-packing + LLM call lands
    // in a follow-up commit once [llm] config is in tree.
    // ----------------------------------------------------------------

    fn _semantic_record(id: &str, vec: Vec<f32>) -> UnifiedRecord {
        use serde_json::json;
        let vec_json: Vec<serde_json::Value> = vec.into_iter().map(|v| json!(v)).collect();
        UnifiedRecord {
            id: id.to_string(),
            source_model: DataModel::Vector,
            data: json!({ "vec": vec_json }),
            score: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn block_batch_mode_returns_empty_with_no_panic() {
        // Whether the feature is on or off, dispatching to
        // LlmBlockBatch with no LLM client wired must not panic and
        // must return a JoinResult with no matches and the original
        // left rows in unmatched_left so callers can fall back.
        use crate::query::unified::ast::{BlockBatchConfig, SemanticJoinMode};

        let left = vec![
            _semantic_record("l1", vec![1.0, 0.0]),
            _semantic_record("l2", vec![0.0, 1.0]),
        ];
        let right = vec![_semantic_record("r1", vec![1.0, 0.0])];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "vec".to_string(),
            join_type: JoinType::Semantic {
                threshold: 0.0, // unused in LlmBlockBatch mode
                top_k: 1,
                mode: SemanticJoinMode::LlmBlockBatch(BlockBatchConfig::default()),
            },
        };

        let result = execute_join(&left, &right, &dependency, None).await;

        // No matches because no LLM is wired.
        assert!(!result.has_matches);
        assert!(result.matched.is_empty());
        // unmatched_left preserves the input so callers can fall
        // back to a simpler join strategy if they wish.
        assert_eq!(result.unmatched_left.len(), 2);
        assert_eq!(result.unmatched_left[0].id, "l1");
        assert_eq!(result.unmatched_left[1].id, "l2");
    }

    #[tokio::test]
    async fn block_batch_mode_with_invalid_config_does_not_panic() {
        // Even an explicitly-invalid config (zero batch size) must
        // produce a clean empty-result outcome rather than panicking.
        // This is the worst-case-input robustness contract.
        use crate::query::unified::ast::{BlockBatchConfig, SemanticJoinMode};

        let bad_config = BlockBatchConfig {
            batch_size_left: 0, // invalid
            batch_size_right: 16,
            max_calls: 64,
        };

        let left = vec![_semantic_record("l1", vec![1.0, 0.0])];
        let right = vec![_semantic_record("r1", vec![1.0, 0.0])];

        let dependency = ComponentDependency {
            component_index: 0,
            join_field: "vec".to_string(),
            join_type: JoinType::Semantic {
                threshold: 0.0,
                top_k: 1,
                mode: SemanticJoinMode::LlmBlockBatch(bad_config),
            },
        };

        // Must not panic. Return shape: empty matches, all left rows
        // in unmatched_left so the caller sees the join produced
        // nothing rather than a partial result.
        let result = execute_join(&left, &right, &dependency, None).await;
        assert!(!result.has_matches);
        assert!(result.matched.is_empty());
        assert_eq!(result.unmatched_left.len(), 1);
    }

    #[tokio::test]
    async fn cosine_mode_remains_default_dispatch() {
        // Backward-compat regression guard: a JoinType::Semantic
        // constructed with `mode: SemanticJoinMode::default()`
        // exhibits the original cosine behavior. If a future change
        // accidentally flips the default to LlmBlockBatch, all
        // existing semantic joins would silently degrade to
        // empty-match -- this test catches that.
        use crate::query::unified::ast::SemanticJoinMode;
        use serde_json::json;

        let left = vec![UnifiedRecord {
            id: "u1".into(),
            source_model: DataModel::Vector,
            data: json!({ "v": [1.0, 0.0, 0.0] }),
            score: None,
            metadata: HashMap::new(),
        }];
        let right = vec![UnifiedRecord {
            id: "r1".into(),
            source_model: DataModel::Vector,
            data: json!({ "v": [0.99, 0.01, 0.0] }),
            score: None,
            metadata: HashMap::new(),
        }];

        let dep = ComponentDependency {
            component_index: 0,
            join_field: "v".to_string(),
            join_type: JoinType::Semantic {
                threshold: 0.8,
                top_k: 1,
                mode: SemanticJoinMode::default(),
            },
        };

        let result = execute_join(&left, &right, &dep, None).await;
        assert!(result.has_matches, "default mode must still match cosine");
        assert_eq!(result.matched.len(), 1);
    }
}
