/*
 * Copyright 2025 ProximaDB
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

//! # Distributed Query Strategy
//!
//! This strategy wraps the DistributedQueryCoordinator to enable cluster-aware
//! query execution that spans multiple nodes in a ProximaDB cluster.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use proximadb_graph_query::service::GraphQueryService;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

use crate::cluster::ClusterManager;
use crate::observability::ObservabilityService;
use crate::query::distributed::DistributedQueryConfig;
use crate::query::distributed::DistributedQueryCoordinator;
use crate::query::facade::{
    ExecutionMetrics, QueryContent, QueryContext, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy,
};
use crate::query::federated::parser::{SqlExtension, VectorQuery};
use crate::query::federated::{FederatedParser, QueryType as FederatedQueryType};
use crate::query::graph_lowering::lower_supported_graph_query_component;
use crate::query::unified::ast::{
    DataModel, DistanceMetric, DocumentQueryExpr, FilterOperator, FilterValue, LogQueryExpr,
    MetricAggregation, MetricQueryExpr, ModelOperation, MultiModelQuery, PathFilter,
    QueryComponent, VectorSearchExpr, VectorSearchParams,
};
use crate::query::unified::fusion::SubQueryResult;
use crate::services::operations::vectors::VectorOperationsService;
use crate::storage::document::DocumentService;

/// Configuration for distributed query strategy
#[derive(Debug, Clone)]
pub struct DistributedStrategyConfig {
    /// Maximum concurrent remote queries
    pub max_concurrent_remote_queries: usize,
    /// Remote query timeout in seconds
    pub remote_query_timeout_secs: u64,
    /// Enable result caching
    pub enable_result_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Prefer local execution when possible
    pub prefer_local_execution: bool,
    /// Enable shuffle exchange for cross-shard joins
    pub enable_shuffle: bool,
}

impl Default for DistributedStrategyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_remote_queries: 10,
            remote_query_timeout_secs: 30,
            enable_result_cache: true,
            cache_ttl_secs: 60,
            prefer_local_execution: true,
            enable_shuffle: true,
        }
    }
}

/// Distributed query strategy
///
/// Wraps the DistributedQueryCoordinator to provide cluster-aware query
/// execution through the unified query facade.
pub struct DistributedQueryStrategy {
    /// Distributed query coordinator
    coordinator: DistributedQueryCoordinator,
    /// Local node ID
    local_node_id: String,
    /// Strategy configuration
    #[allow(dead_code)]
    config: DistributedStrategyConfig,
}

impl DistributedQueryStrategy {
    /// Create a new distributed query strategy
    pub fn new(local_node_id: String, config: DistributedStrategyConfig) -> Self {
        let dist_config = DistributedQueryConfig {
            max_concurrent_remote_queries: config.max_concurrent_remote_queries,
            remote_query_timeout: Duration::from_secs(config.remote_query_timeout_secs),
            enable_result_cache: config.enable_result_cache,
            cache_ttl_seconds: config.cache_ttl_secs,
            prefer_local_execution: config.prefer_local_execution,
            retry_failed_queries: true,
            max_retries: 3,
            parallel_remote_execution: true,
            enable_shuffle: config.enable_shuffle,
            shuffle_batch_size: 1000,
        };

        let coordinator = DistributedQueryCoordinator::new(dist_config, local_node_id.clone());

        Self {
            coordinator,
            local_node_id,
            config,
        }
    }

    /// Set cluster manager for distributed execution
    pub fn with_cluster(mut self, cluster_manager: Arc<ClusterManager>) -> Self {
        self.coordinator = self.coordinator.with_cluster(cluster_manager);
        self
    }

    /// Wire vector operations into local distributed execution.
    pub fn with_vector_ops(mut self, vector_ops: Arc<VectorOperationsService>) -> Self {
        self.coordinator = self.coordinator.with_vector_ops(vector_ops);
        self
    }

    /// Wire document service into local distributed execution.
    pub fn with_document_service(mut self, document_service: Arc<DocumentService>) -> Self {
        self.coordinator = self.coordinator.with_document_service(document_service);
        self
    }

    /// Wire graph query/traversal service into local distributed execution.
    pub fn with_graph_service<G>(mut self, graph_service: Arc<G>) -> Self
    where
        G: GraphQueryService + 'static,
    {
        self.coordinator = self.coordinator.with_graph_service(graph_service);
        self
    }

    /// Wire observability service into local distributed execution.
    pub fn with_observability_service(
        mut self,
        observability_service: Arc<ObservabilityService>,
    ) -> Self {
        self.coordinator = self
            .coordinator
            .with_observability_service(observability_service);
        self
    }

    /// Get local node ID
    pub fn local_node_id(&self) -> &str {
        &self.local_node_id
    }

    /// Convert SubQueryResults to QueryResultData
    fn convert_results(&self, results: Vec<SubQueryResult>) -> QueryResultData {
        // Convert SubQueryResults to JSON format
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .flat_map(|r| {
                r.records.iter().map(|record| {
                    // Extract the JSON data from UnifiedRecord instead of serializing the whole record
                    serde_json::json!({
                        "source_model": format!("{:?}", r.source_model),
                        "execution_time_us": r.execution_time_us,
                        "records_returned": r.records_returned,
                        "id": record.id,
                        "score": record.score,
                        "metadata": record.metadata,
                        "data": record.data,
                    })
                })
            })
            .collect();

        QueryResultData::Rows(json_results)
    }

    fn extract_sql<'a>(&self, request: &'a QueryRequest) -> Result<&'a str> {
        match &request.content {
            QueryContent::Sql(sql) => Ok(sql.as_str()),
            _ => Err(anyhow!(
                "Distributed strategy requires SQL/federated query content"
            )),
        }
    }

    fn strip_strategy_comments(&self, sql: &str) -> Result<String> {
        let comment_re = Regex::new(r"(?s)/\*.*?\*/")
            .map_err(|error| anyhow!("Failed to compile distributed comment regex: {error}"))?;
        Ok(comment_re.replace_all(sql, " ").trim().to_string())
    }

    fn parse_limit(&self, sql: &str) -> Result<Option<u32>> {
        let limit_re = Regex::new(r"(?i)\bLIMIT\s+(\d+)\b")
            .map_err(|error| anyhow!("Failed to compile distributed LIMIT regex: {error}"))?;
        Ok(limit_re
            .captures(sql)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse::<u32>().ok()))
    }

    fn normalize_document_path(&self, field: &str) -> String {
        let trimmed = field.trim();
        if trimmed.starts_with("$.") {
            trimmed.to_string()
        } else {
            format!(
                "$.{}",
                trimmed.trim_start_matches('$').trim_start_matches('.')
            )
        }
    }

    fn parse_filter_value(&self, raw: &str) -> FilterValue {
        let trimmed = raw.trim().trim_matches('\'').trim_matches('"');
        if trimmed.eq_ignore_ascii_case("true") {
            FilterValue::Bool(true)
        } else if trimmed.eq_ignore_ascii_case("false") {
            FilterValue::Bool(false)
        } else if let Ok(number) = trimmed.parse::<f64>() {
            FilterValue::Number(number)
        } else {
            FilterValue::String(trimmed.to_string())
        }
    }

    fn parse_document_path_filters(&self, filter: Option<&str>) -> Result<Vec<PathFilter>> {
        let Some(filter) = filter.map(str::trim).filter(|filter| !filter.is_empty()) else {
            return Ok(Vec::new());
        };

        let expr_re = Regex::new(r#"^\s*([$A-Za-z0-9_.]+)\s*(=|!=|>=|<=|>|<)\s*(.+?)\s*$"#)
            .map_err(|error| anyhow!("Failed to compile document filter regex: {error}"))?;

        let caps = expr_re.captures(filter).ok_or_else(|| {
            anyhow!(
                "Distributed document query only supports simple comparisons today; got '{}'",
                filter
            )
        })?;

        let field = caps
            .get(1)
            .map(|m| m.as_str())
            .ok_or_else(|| anyhow!("Document filter field is missing"))?;
        let operator = caps
            .get(2)
            .map(|m| m.as_str())
            .ok_or_else(|| anyhow!("Document filter operator is missing"))?;
        let value = caps
            .get(3)
            .map(|m| m.as_str())
            .ok_or_else(|| anyhow!("Document filter value is missing"))?;

        let operator = match operator {
            "=" => FilterOperator::Eq,
            "!=" => FilterOperator::Ne,
            ">" => FilterOperator::Gt,
            ">=" => FilterOperator::Gte,
            "<" => FilterOperator::Lt,
            "<=" => FilterOperator::Lte,
            other => {
                return Err(anyhow!(
                    "Unsupported distributed document filter operator '{}'",
                    other
                ));
            }
        };

        Ok(vec![PathFilter {
            path: self.normalize_document_path(field),
            operator,
            value: self.parse_filter_value(value),
        }])
    }

    fn parse_vector_literal(&self, raw: &str) -> Result<Vec<f32>> {
        let normalized = Self::strip_pgvector_cast(raw.trim());
        let trimmed = normalized
            .trim_matches('\'')
            .trim_matches('"')
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']');
        if trimmed.is_empty() {
            return Err(anyhow!("query vector must be non-empty"));
        }

        trimmed
            .split(',')
            .map(|value| {
                let f = value.trim().parse::<f32>().map_err(|error| {
                    anyhow!(
                        "Failed to parse vector literal value '{}' for distributed query: {}",
                        value.trim(),
                        error
                    )
                })?;
                // The ONE non-finite policy (an inf/NaN reaches the
                // distance kernels).
                if !f.is_finite() {
                    return Err(anyhow!("query vector components must be finite"));
                }
                Ok(f)
            })
            .collect()
    }

    fn strip_pgvector_cast(value: &str) -> &str {
        let lower = value.to_ascii_lowercase();
        let Some(cast_start) = lower.rfind("::vector") else {
            return value;
        };
        let suffix = lower[cast_start + "::vector".len()..].trim();
        if suffix.is_empty()
            || (suffix.starts_with('(')
                && suffix.ends_with(')')
                && suffix[1..suffix.len() - 1]
                    .chars()
                    .all(|ch| ch.is_ascii_digit()))
        {
            value[..cast_start].trim()
        } else {
            value
        }
    }

    fn query_to_multimodal(&self, request: &QueryRequest, sql: &str) -> Result<MultiModelQuery> {
        let parser = FederatedParser::new();
        let federated = parser.parse(sql)?;

        if federated.extensions.is_empty() {
            return Err(anyhow!(
                "Distributed execution currently supports federated extension queries only"
            ));
        }

        if federated.query_type == FederatedQueryType::Sql {
            return Err(anyhow!(
                "Distributed execution is not configured for generic relational SQL scans yet"
            ));
        }

        let mut multi = MultiModelQuery::new();
        multi.limit = self.parse_limit(sql)?;

        for extension in &federated.extensions {
            let component = self.extension_to_component(request, &federated, extension)?;
            multi.components.push(component);
        }

        if multi.components.is_empty() {
            return Err(anyhow!(
                "Distributed query did not produce any executable multi-model components"
            ));
        }

        Ok(multi)
    }

    fn extension_to_component(
        &self,
        request: &QueryRequest,
        federated: &crate::query::federated::FederatedQuery,
        extension: &SqlExtension,
    ) -> Result<QueryComponent> {
        match extension {
            SqlExtension::VectorSearch {
                collection,
                query_vector,
                top_k,
            } => {
                let query_vector = match query_vector {
                    VectorQuery::Literal(values) => values.clone(),
                    VectorQuery::Expression(expr) => {
                        return Err(anyhow!(
                            "Distributed vector execution does not support expression-based query vectors yet: '{}'",
                            expr
                        ));
                    }
                };

                Ok(QueryComponent {
                    model: DataModel::Vector,
                    operation: ModelOperation::VectorSearch(VectorSearchExpr {
                        collection: collection.clone(),
                        vector_column: None,
                        query_vector,
                        top_k: *top_k as u32,
                        threshold: None,
                        metric: DistanceMetric::Cosine,
                        filter: None,
                        params: VectorSearchParams::default(),
                    }),
                    filters: Vec::new(),
                    dependencies: Vec::new(),
                })
            }
            SqlExtension::VectorDistance {
                left_column: _,
                right_literal,
            } => {
                let collection = federated
                    .targets
                    .first()
                    .map(|target| target.name.clone())
                    .or_else(|| request.target.clone())
                    .unwrap_or_else(|| "default".to_string());

                Ok(QueryComponent {
                    model: DataModel::Vector,
                    operation: ModelOperation::VectorSearch(VectorSearchExpr {
                        collection,
                        vector_column: None,
                        query_vector: self.parse_vector_literal(right_literal)?,
                        top_k: self.parse_limit(&federated.sql)?.unwrap_or(10),
                        threshold: None,
                        metric: DistanceMetric::Cosine,
                        filter: None,
                        params: VectorSearchParams::default(),
                    }),
                    filters: Vec::new(),
                    dependencies: Vec::new(),
                })
            }
            SqlExtension::DocumentQuery { collection, filter } => Ok(QueryComponent {
                model: DataModel::Document,
                operation: ModelOperation::DocumentQuery(DocumentQueryExpr {
                    collection: collection.clone(),
                    path_filters: self.parse_document_path_filters(filter.as_deref())?,
                    text_search: None,
                    projection: Vec::new(),
                    sort: None,
                    limit: self.parse_limit(&federated.sql)?,
                }),
                filters: Vec::new(),
                dependencies: Vec::new(),
            }),
            SqlExtension::GraphQuery { cypher } => lower_supported_graph_query_component(
                cypher,
                request.target.as_deref(),
                Some("default"),
            ),
            SqlExtension::Logs { namespace } => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64;

                Ok(QueryComponent {
                    model: DataModel::Observability,
                    operation: ModelOperation::LogQuery(LogQueryExpr {
                        namespace: namespace.clone(),
                        start_time_ns: now - 3_600_000_000_000,
                        end_time_ns: now,
                        query: None,
                        severities: Vec::new(),
                        services: Vec::new(),
                        limit: self.parse_limit(&federated.sql)?.unwrap_or(100),
                    }),
                    filters: Vec::new(),
                    dependencies: Vec::new(),
                })
            }
            SqlExtension::Metrics { namespace } => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64;

                Ok(QueryComponent {
                    model: DataModel::Observability,
                    operation: ModelOperation::MetricQuery(MetricQueryExpr {
                        namespace: namespace.clone(),
                        metric_name: "*".to_string(),
                        start_time_ns: now - 3_600_000_000_000,
                        end_time_ns: now,
                        aggregation: MetricAggregation::Avg,
                        group_by: Vec::new(),
                        label_filters: std::collections::HashMap::new(),
                    }),
                    filters: Vec::new(),
                    dependencies: Vec::new(),
                })
            }
            SqlExtension::Traces { namespace } => Err(anyhow!(
                "Distributed execution does not support TRACES('{}') yet; use the federated observability executor",
                namespace
            )),
            SqlExtension::RerankSearch { .. } => Err(anyhow!(
                "Distributed execution does not support RerankSearch yet; \
                 route through the local rank pipeline"
            )),
        }
    }
}

#[async_trait]
impl QueryStrategy for DistributedQueryStrategy {
    /// Strategy name for metrics/debugging
    fn name(&self) -> &str {
        "distributed"
    }

    /// Check if this strategy can handle the given query
    fn can_handle(&self, request: &QueryRequest) -> bool {
        // Handle distributed query type requests
        request.query_type == crate::query::facade::QueryType::Federated
            && request.params.force_path.as_deref() == Some("distributed")
    }

    /// Execute the query and return results
    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        debug!("Executing distributed query: {:?}", request.query_type);

        let sql = self.extract_sql(&request)?;
        let normalized_sql = self.strip_strategy_comments(sql)?;
        let query = self.query_to_multimodal(&request, &normalized_sql)?;

        let stats_before = self.coordinator.get_stats().await;

        // Execute via distributed coordinator
        let execution = self
            .coordinator
            .execute_with_metadata(&query)
            .await
            .map_err(|e| anyhow!("Distributed query failed: {}", e))?;
        let stats_after = self.coordinator.get_stats().await;

        let nodes_involved = execution.plan.as_ref().map(|plan| {
            plan.local_subqueries
                .iter()
                .chain(plan.remote_subqueries.iter())
                .map(|subquery| subquery.target_node.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        });
        let local_subqueries = execution
            .plan
            .as_ref()
            .map(|plan| plan.local_subqueries.len())
            .unwrap_or_default();
        let remote_subqueries = execution
            .plan
            .as_ref()
            .map(|plan| plan.remote_subqueries.len())
            .unwrap_or_default();
        let records_returned = execution
            .results
            .iter()
            .map(|result| result.records.len())
            .sum::<usize>();

        // Convert results
        let data = self.convert_results(execution.results);

        // Create execution metrics
        let extra = serde_json::json!({
            "query_type": "distributed",
            "num_results": records_returned,
            "num_components": query.components.len(),
            "local_node_id": self.local_node_id,
            "nodes_involved": nodes_involved,
            "local_subqueries": local_subqueries,
            "remote_subqueries": remote_subqueries,
            "cache_hits": stats_after.cache_hits.saturating_sub(stats_before.cache_hits),
            "total_queries_delta": stats_after.total_queries.saturating_sub(stats_before.total_queries),
            "local_only_queries_delta": stats_after
                .local_only_queries
                .saturating_sub(stats_before.local_only_queries),
            "distributed_queries_delta": stats_after
                .distributed_queries
                .saturating_sub(stats_before.distributed_queries),
            "failed_remote_subqueries_delta": stats_after
                .failed_remote_subqueries
                .saturating_sub(stats_before.failed_remote_subqueries),
            "shuffle_count_delta": stats_after.shuffle_count.saturating_sub(stats_before.shuffle_count),
        });

        Ok(QueryResult {
            data,
            metrics: Some(ExecutionMetrics {
                results_returned: records_returned,
                cache_hit: execution.cache_hit,
                extra,
                ..Default::default()
            }),
        })
    }
}

/// Statistics for distributed query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedQueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// Queries executed locally only
    pub local_only_queries: u64,
    /// Queries requiring remote execution
    pub distributed_queries: u64,
    /// Total remote subqueries
    pub remote_subqueries: u64,
    /// Failed remote subqueries
    pub failed_remote_subqueries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Number of shuffle operations executed
    pub shuffle_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::{DocumentCollectionConfig, SqlObject, SqlValue, sql_value};
    use crate::query::facade::QueryContext;
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::{
        CompactionParameters, CompactionResult, FlushParameters, FlushResult,
        StorageFormatStrategy, UnifiedStorageFormat,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    #[test]
    fn test_distributed_strategy_config_default() {
        let config = DistributedStrategyConfig::default();
        assert_eq!(config.max_concurrent_remote_queries, 10);
        assert_eq!(config.remote_query_timeout_secs, 30);
        assert!(config.enable_result_cache);
        assert!(config.prefer_local_execution);
    }

    #[test]
    fn test_distributed_query_stats_default() {
        let stats = DistributedQueryStats {
            total_queries: 0,
            local_only_queries: 0,
            distributed_queries: 0,
            remote_subqueries: 0,
            failed_remote_subqueries: 0,
            cache_hits: 0,
            shuffle_count: 0,
        };
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_query_to_multimodal_translates_document_filter() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let request = QueryRequest::federated(
            "SELECT * FROM DOCUMENT_QUERY('users', 'status = active') LIMIT 5",
        );

        let query = strategy
            .query_to_multimodal(
                &request,
                "SELECT * FROM DOCUMENT_QUERY('users', 'status = active') LIMIT 5",
            )
            .expect("document distributed query should translate");

        assert_eq!(query.components.len(), 1);
        assert_eq!(query.limit, Some(5));
        match &query.components[0].operation {
            ModelOperation::DocumentQuery(expr) => {
                assert_eq!(expr.collection, "users");
                assert_eq!(expr.path_filters.len(), 1);
                assert_eq!(expr.path_filters[0].path, "$.status");
                assert!(matches!(expr.path_filters[0].operator, FilterOperator::Eq));
            }
            other => panic!("expected document query component, got {:?}", other),
        }
    }

    #[test]
    fn test_query_to_multimodal_rejects_expression_vector() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let request =
            QueryRequest::federated("SELECT * FROM VECTOR_SEARCH('users', u.embedding, 10)");

        let error = strategy
            .query_to_multimodal(
                &request,
                "SELECT * FROM VECTOR_SEARCH('users', u.embedding, 10)",
            )
            .expect_err("expression vectors should remain unsupported for distributed execution");

        assert!(error.to_string().contains("expression-based query vectors"));
    }

    #[test]
    fn test_query_to_multimodal_lowers_pgvector_distance_cast() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let sql =
            "SELECT id FROM memories ORDER BY embedding <-> '[0.1, 0.2, -0.3]'::vector(3) LIMIT 5";
        let request = QueryRequest::federated(sql);

        let query = strategy
            .query_to_multimodal(&request, sql)
            .expect("pgvector distance query should lower");

        assert_eq!(query.components.len(), 1);
        match &query.components[0].operation {
            ModelOperation::VectorSearch(expr) => {
                assert_eq!(expr.collection, "memories");
                assert_eq!(expr.query_vector, vec![0.1, 0.2, -0.3]);
                assert_eq!(expr.top_k, 5);
            }
            other => panic!("expected vector search component, got {:?}", other),
        }
    }

    #[test]
    fn test_query_to_multimodal_lowers_cross_modal_components() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let sql = "\
            SELECT *
            FROM DOCUMENT_QUERY('profiles', 'tier = gold') p
            JOIN LATERAL VECTOR_SEARCH('memories', '[0.1, 0.2]'::vector(2), 3) v ON true
            JOIN LATERAL GRAPH_QUERY('MATCH (n:Agent) FROM agent_graph RETURN n') g ON true
            LIMIT 3";
        let request = QueryRequest::federated(sql);

        let query = strategy
            .query_to_multimodal(&request, sql)
            .expect("cross-modal query should lower through unified query components");

        assert_eq!(query.components.len(), 3);
        assert_eq!(query.limit, Some(3));
        assert!(matches!(
            query.components[0].operation,
            ModelOperation::DocumentQuery(_)
        ));
        assert!(matches!(
            query.components[1].operation,
            ModelOperation::VectorSearch(_)
        ));
        assert!(matches!(
            query.components[2].operation,
            ModelOperation::GraphQuery(_)
        ));
    }

    #[test]
    fn test_query_to_multimodal_uses_graph_target_from_supported_subset() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let request = QueryRequest::federated(
            "SELECT * FROM GRAPH_QUERY('MATCH (n:Person) FROM social RETURN n')",
        );

        let query = strategy
            .query_to_multimodal(
                &request,
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Person) FROM social RETURN n')",
            )
            .expect("graph distributed query should translate");

        assert_eq!(query.components.len(), 1);
        match &query.components[0].operation {
            ModelOperation::GraphQuery(expr) => {
                assert_eq!(expr.graph_name, "social");
                assert_eq!(expr.normalized_query, "MATCH (n:Person) RETURN n");
                assert_eq!(
                    expr.output_columns,
                    vec![
                        "node_id".to_string(),
                        "label".to_string(),
                        "properties".to_string()
                    ]
                );
                assert!(expr.uses_legacy_node_rows);
                assert_eq!(expr.max_depth, 0);
            }
            other => panic!("expected graph query component, got {:?}", other),
        }
    }

    #[test]
    fn test_query_to_multimodal_rejects_conflicting_graph_target_and_from_clause() {
        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        );
        let request =
            QueryRequest::federated("SELECT * FROM GRAPH_QUERY('MATCH (n) FROM social RETURN n')")
                .with_target("api_graph");

        let error = strategy
            .query_to_multimodal(
                &request,
                "SELECT * FROM GRAPH_QUERY('MATCH (n) FROM social RETURN n')",
            )
            .expect_err("conflicting graph targets should be rejected");

        assert!(error.to_string().contains("target conflict"));
    }

    struct MockStorageEngine {
        filesystem_factory: FilesystemFactory,
    }

    impl MockStorageEngine {
        async fn new() -> Self {
            let filesystem_factory = FilesystemFactory::create(FilesystemConfig::default())
                .await
                .expect("mock filesystem factory should initialize");
            Self { filesystem_factory }
        }
    }

    #[async_trait::async_trait]
    impl UnifiedStorageFormat for MockStorageEngine {
        fn engine_name(&self) -> &'static str {
            "mock"
        }

        fn engine_version(&self) -> &'static str {
            "1.0.0"
        }

        fn strategy(&self) -> StorageFormatStrategy {
            StorageFormatStrategy::Viper
        }

        async fn do_flush(&self, _params: &FlushParameters) -> Result<FlushResult> {
            Ok(FlushResult {
                success: true,
                collections_affected: Vec::new(),
                entries_flushed: Some(0),
                bytes_written: Some(0),
                files_created: Some(0),
                file_paths: Vec::new(),
                duration_ms: Some(0),
                completed_at: Utc::now(),
                engine_metrics: HashMap::new(),
                compaction_triggered: false,
                compaction_error: None,
                flushed_batch_ids: Vec::new(),
            })
        }

        async fn do_compact(&self, _params: &CompactionParameters) -> Result<CompactionResult> {
            Ok(CompactionResult {
                success: true,
                collections_affected: Vec::new(),
                entries_processed: Some(0),
                entries_removed: Some(0),
                bytes_read: Some(0),
                bytes_written: Some(0),
                input_files: Some(0),
                output_files: Some(0),
                duration_ms: Some(0),
                completed_at: Utc::now(),
                engine_metrics: HashMap::new(),
            })
        }

        async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
            Ok(HashMap::new())
        }

        async fn vector_by_id(
            &self,
            _collection_id: &str,
            _base_path: &str,
            _vector_id: &str,
        ) -> Result<Option<proximadb_records::ProximaRecord>> {
            Ok(None)
        }

        async fn search_vectors_unified(
            &self,
            _ctx: &crate::storage::traits::StorageQueryContext,
        ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn test_distributed_strategy_executes_document_query_locally() {
        let storage_engine: Arc<dyn UnifiedStorageFormat> =
            Arc::new(MockStorageEngine::new().await);
        let document_service = Arc::new(DocumentService::new(storage_engine));
        document_service
            .create_collection("users", DocumentCollectionConfig::default())
            .await
            .expect("document collection should be created");
        document_service
            .insert_document(
                "users",
                Some("user-1"),
                SqlObject {
                    fields: HashMap::from([(
                        "status".to_string(),
                        SqlValue {
                            value: Some(sql_value::Value::StringValue("active".to_string())),
                        },
                    )]),
                },
            )
            .await
            .expect("document should be inserted");

        let strategy = DistributedQueryStrategy::new(
            "test-node".to_string(),
            DistributedStrategyConfig::default(),
        )
        .with_document_service(document_service);

        let result = strategy
            .execute(
                QueryRequest::federated("SELECT * FROM DOCUMENT_QUERY('users', 'status = active')"),
                &QueryContext::new(5_000),
            )
            .await
            .expect("distributed local document query should execute");

        match result.data {
            QueryResultData::Rows(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0]["id"], "user-1");
                assert_eq!(rows[0]["source_model"], "Document");
            }
            other => panic!("expected row results, got {:?}", other),
        }
    }
}
