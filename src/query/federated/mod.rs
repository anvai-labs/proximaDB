//! # Federated Query Engine
//!
//! Provides federated execution for ProximaDB SQL extensions across vector, document, graph,
//! and observability sources.
//!
//! The live server path is currently strongest for function-backed sources such as
//! `VECTOR_SEARCH(...)`, `GRAPH_QUERY(...)`, `DOCUMENT_QUERY(...)`, `LOGS(...)`, and
//! `METRICS(...)`. Limited correlated `LATERAL` execution is now available when the outer
//! source is also function-backed. Generic relational scans still require additional execution
//! work and are reported explicitly when not supported.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    FEDERATED QUERY ENGINE                        │
//! │  PostgreSQL Wire Protocol + SQL Extensions + Federated Execution │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!         ┌─────────────────────┼─────────────────────┐
//!         ▼                     ▼                     ▼
//! ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
//! │ FederatedParser│    │CrossModelOpt  │    │FederatedExec  │
//! │ SQL+Extensions │    │ Cost Model    │    │ Arrow Batches │
//! └───────────────┘    └───────────────┘    └───────────────┘
//! ```
//!
//! ## SQL Extensions
//!
//! ProximaDB extends standard SQL with special functions for each data model:
//!
//! ### VECTOR_SEARCH(collection, query_vector, top_k)
//!
//! Performs vector similarity search using the configured distance metric.
//!
//! ```sql
//! -- Find 10 most similar products to a query vector
//! SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2, 0.3, ...]', 10);
//! ```
//!
//! ### GRAPH_QUERY('cypher_query')
//!
//! Executes a Cypher-like graph traversal query.
//!
//! ```sql
//! -- Find all friends of friends
//! SELECT * FROM GRAPH_QUERY('MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.name');
//!
//! -- Get connections with specific relationship types
//! SELECT * FROM GRAPH_QUERY('MATCH (p:Person {name: "Alice"})-[:WORKS_AT]->(c:Company) RETURN c');
//! ```
//!
//! ### DOCUMENT_QUERY(collection, filter)
//!
//! Queries document collections with optional filter expressions.
//!
//! ```sql
//! -- Find documents matching a filter
//! SELECT * FROM DOCUMENT_QUERY('orders', 'status = "pending"');
//! ```
//!
//! ### LOGS(namespace) / METRICS(namespace)
//!
//! Queries observability data (logs, metrics, and traces).
//!
//! ```sql
//! -- Query recent logs
//! SELECT * FROM LOGS('production') WHERE timestamp > now() - interval '1h';
//!
//! -- Query metrics
//! SELECT * FROM METRICS('system') WHERE metric_name = 'cpu_usage';
//!
//! -- Query traces
//! SELECT * FROM TRACES('production');
//! ```
//!
//! ### Vector Distance Operator (<->)
//!
//! pgvector-compatible distance operator for ORDER BY clauses.
//!
//! ```sql
//! -- Sort by vector distance (pgvector syntax)
//! SELECT * FROM products ORDER BY embedding <-> '[0.1,0.2,...]'::vector LIMIT 10;
//! ```
//!
//! ## Cross-Model Queries
//!
//! The engine is designed to combine multiple data models in a single query:
//!
//! ```sql
//! -- Find similar products for a user, then get related reviews
//! -- Function-backed sources are executable today:
//! SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2, 0.3]', 10);
//!
//! -- Function-backed correlated LATERAL joins execute today:
//! -- SELECT *
//! -- FROM DOCUMENT_QUERY('profiles') p
//! -- JOIN LATERAL VECTOR_SEARCH('products', p.document.embedding, 10) v ON true;
//!
//! -- Generic relational outer scans still require a relational execution backend:
//! -- SELECT u.name, v.score
//! -- FROM users u
//! -- JOIN LATERAL VECTOR_SEARCH('products', u.preference_vector, 10) v ON true;
//! ```
//!
//! ## API Access
//!
//! Federated queries are available via:
//!
//! - **REST**: `POST /api/v1/unified/federated` with `{ "query": "SELECT ..." }`
//! - **gRPC**: `QueryService.ExecuteQuery` with federated SQL
//! - **PostgreSQL Wire Protocol**: Connect with psql and run queries directly
//!
//! ## Example Usage (REST)
//!
//! ```bash
//! curl -X POST http://localhost:5678/api/v1/unified/federated \
//!   -H "Content-Type: application/json" \
//!   -d '{"query": "SELECT * FROM VECTOR_SEARCH('embeddings', '[0.1, 0.2]', 10)"}'
//! ```

pub mod execution;
pub mod optimizer;
pub use proximadb_query_fusion::parser;

// Re-exports
pub use execution::{ExecutionResult, FederatedExecutor};
pub use optimizer::{CrossModelOptimizer, FederatedQueryPlan, PlanNode};
pub use parser::{FederatedParser, FederatedQuery, QueryType};

use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

use super::cache::{CacheInvalidator, FederatedQueryResultCache, QueryKey};
use crate::catalog::CatalogManager;
use crate::storage::MultiModelStorageFacade;

/// Federated query context containing all necessary components
pub struct FederatedQueryContext {
    /// Multi-model storage facade
    pub storage: Arc<MultiModelStorageFacade>,
    /// Parser for SQL with extensions
    pub parser: FederatedParser,
    /// Cross-model optimizer
    pub optimizer: CrossModelOptimizer,
    /// Federated executor
    pub executor: FederatedExecutor,
    /// Query result cache for repetitive queries
    pub cache: Option<Arc<FederatedQueryResultCache>>,
    /// Cache invalidator for real-time invalidation
    pub invalidator: Option<Arc<CacheInvalidator>>,
    /// Catalog manager for external table resolution
    pub catalog_manager: Option<Arc<CatalogManager>>,
}

impl FederatedQueryContext {
    /// Set a statistics provider for cost-based optimization
    ///
    /// When set, `execute()` will use `optimize_with_statistics()` instead of
    /// `optimize()`, enabling cardinality-aware cost estimation from real
    /// collection stats provided by the storage engine layer.
    pub fn with_statistics_provider(
        mut self,
        provider: std::sync::Arc<dyn crate::query::federated::optimizer::StatisticsProvider>,
    ) -> Self {
        self.optimizer.set_statistics_provider(provider);
        self
    }

    /// Reuse the live collection metadata service so federated vector queries
    /// inherit storage assignments, engines, and canonical collection IDs.
    pub fn with_collection_port(
        mut self,
        collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
    ) -> Self {
        self.executor = self.executor.with_collection_port(collection_port);
        self
    }

    /// Reuse the existing vector operations service so federated vector SQL
    /// follows the same search path as REST/gRPC/embedded direct search.
    pub fn with_vector_operations(
        mut self,
        vector_operations_service: Arc<
            crate::services::operations::vectors::VectorOperationsService,
        >,
    ) -> Self {
        self.executor = self
            .executor
            .with_vector_operations(vector_operations_service);
        self
    }

    /// Wire the rank-pipeline singleton so SQL RERANK(...) shares the
    /// REST profile registry + candidate provider + scorer wiring
    /// (R-7c.4d). Without this, the federated executor falls back to
    /// the first-phase-only stub for any rank profile.
    pub fn with_rank_services(
        mut self,
        rank_services: Arc<crate::network::rest::canonical::rank::RankServices>,
    ) -> Self {
        self.executor = self.executor.with_rank_services(rank_services);
        self
    }

    /// Create a new federated query context
    pub fn new(storage: Arc<MultiModelStorageFacade>) -> Self {
        Self {
            storage: storage.clone(),
            parser: FederatedParser::new(),
            optimizer: CrossModelOptimizer::new(),
            executor: FederatedExecutor::new(storage),
            cache: None,
            invalidator: None,
            catalog_manager: None,
        }
    }

    /// Create a new federated query context with caching enabled
    pub fn with_cache(
        storage: Arc<MultiModelStorageFacade>,
        cache: Arc<FederatedQueryResultCache>,
    ) -> Self {
        let invalidator = Arc::new(CacheInvalidator::new(cache.clone()));
        Self {
            storage: storage.clone(),
            parser: FederatedParser::new(),
            optimizer: CrossModelOptimizer::new(),
            executor: FederatedExecutor::new(storage),
            cache: Some(cache),
            invalidator: Some(invalidator),
            catalog_manager: None,
        }
    }

    /// Create with external catalog manager
    pub fn with_catalog_manager(
        storage: Arc<MultiModelStorageFacade>,
        catalog_manager: Arc<CatalogManager>,
    ) -> Self {
        Self {
            storage: storage.clone(),
            parser: FederatedParser::new(),
            optimizer: CrossModelOptimizer::new(),
            executor: FederatedExecutor::new(storage),
            cache: None,
            invalidator: None,
            catalog_manager: Some(catalog_manager),
        }
    }

    /// Enable caching on an existing context
    pub fn enable_cache(&mut self, cache: Arc<FederatedQueryResultCache>) {
        let invalidator = Arc::new(CacheInvalidator::new(cache.clone()));
        self.cache = Some(cache);
        self.invalidator = Some(invalidator);
    }

    /// Get the cache if enabled
    pub fn get_cache(&self) -> Option<&Arc<FederatedQueryResultCache>> {
        self.cache.as_ref()
    }

    /// Get the invalidator if caching is enabled
    pub fn get_invalidator(&self) -> Option<&Arc<CacheInvalidator>> {
        self.invalidator.as_ref()
    }

    /// Get the catalog manager if external catalogs are configured
    pub fn get_catalog_manager(&self) -> Option<&Arc<CatalogManager>> {
        self.catalog_manager.as_ref()
    }

    /// Execute a federated query with optional caching
    pub async fn execute(&self, sql: &str) -> Result<ExecutionResult> {
        // Fail closed: a federated query runs only within a request scope
        // (a tenant, or explicitly internal) — see `request_scope`.
        if crate::query::request_scope::current_scope().is_none() {
            return Err(anyhow::anyhow!(
                "federated query has no request scope; the caller must act as a tenant"
            ));
        }
        crate::query::utils::metrics::record_query_start("federated");
        let start = std::time::Instant::now();

        // Check cache first if enabled
        if let Some(ref cache) = self.cache {
            let key = federated_cache_key(sql);
            if let Some(cached) = cache.get(&key) {
                debug!(
                    query_fingerprint = key.fingerprint,
                    "Cache hit for federated query"
                );
                // Return a clone of the cached result
                // Note: ExecutionResult contains Arc<Schema> and Vec<RecordBatch>,
                // which are relatively cheap to clone
                return Ok(ExecutionResult {
                    batches: cached.result.batches.clone(),
                    schema: cached.result.schema.clone(),
                    stats: cached.result.stats.clone(),
                });
            }
        }

        let result = async {
            // 1. Parse the query
            let federated_query = self.parser.parse(sql)?;

            if federated_query.query_type == QueryType::Sql {
                return Err(anyhow::anyhow!(
                    "Standard relational SQL execution is not configured in FederatedQueryContext; use columnar providers or SQL extensions such as VECTOR_SEARCH, GRAPH_QUERY, DOCUMENT_QUERY, LOGS, METRICS, or TRACES"
                ));
            }

            // 2. Optimize the query plan
            let plan = self.optimizer.optimize(&federated_query)?;

            // Extract dependencies for cache invalidation
            let dependencies: Vec<String> = plan
                .metadata
                .involved_models
                .iter()
                .map(|m| format!("{:?}", m).to_lowercase())
                .collect();

            // 3. Execute the plan
            let result = self.executor.execute(plan).await?;

            // Cache the result if caching is enabled
            if let Some(ref cache) = self.cache {
                let key = federated_cache_key(sql);
                // Clone the result for caching
                let cached_result = ExecutionResult {
                    batches: result.batches.clone(),
                    schema: result.schema.clone(),
                    stats: result.stats.clone(),
                };
                if let Err(e) = cache.insert(key, cached_result, dependencies) {
                    debug!("Failed to cache query result: {:?}", e);
                }
            }

            Ok(result)
        }
        .await;

        crate::query::utils::metrics::record_query_end(
            "federated",
            result.is_ok(),
            start.elapsed().as_millis() as u64,
        );

        result
    }

    /// Execute a query without caching (bypass cache)
    pub async fn execute_uncached(&self, sql: &str) -> Result<ExecutionResult> {
        crate::query::utils::metrics::record_query_start("federated_uncached");
        let start = std::time::Instant::now();

        // 1. Parse the query
        let result = async {
            let federated_query = self.parser.parse(sql)?;
            if federated_query.query_type == QueryType::Sql {
                return Err(anyhow::anyhow!(
                    "Standard relational SQL execution is not configured in FederatedQueryContext; use columnar providers or SQL extensions such as VECTOR_SEARCH, GRAPH_QUERY, DOCUMENT_QUERY, LOGS, METRICS, or TRACES"
                ));
            }
            let plan = self.optimizer.optimize(&federated_query)?;
            self.executor.execute(plan).await
        }
        .await;

        crate::query::utils::metrics::record_query_end(
            "federated_uncached",
            result.is_ok(),
            start.elapsed().as_millis() as u64,
        );

        result
    }

    /// Invalidate cache entries for a collection
    ///
    /// Call this when data in a collection is modified.
    pub fn invalidate_collection(&self, collection: &str) -> usize {
        if let Some(ref invalidator) = self.invalidator {
            invalidator.invalidate_collection(collection)
        } else {
            0
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<super::cache::QueryCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::search::results::OptimizedSearchRecord;
    use crate::graph::engines::GraphEngine;
    use crate::graph::{Edge, EdgeId, Node, NodeId};
    use crate::graph::{PropertyValue, property_value};
    use crate::proto::proximadb_v1::{
        DocumentCollectionConfig, DocumentFilter, DocumentUpdate, LogEntry, LogFilter,
        MetricAggregation, MetricSample, ObservabilityNamespaceConfig, RetentionConfig, Severity,
        SqlArray, SqlObject, SqlValue, TraceData, sql_value,
    };
    use crate::query::federated::{FederatedQueryContext, FederatedQueryResultCache};
    use crate::storage::MultiModelStorageFacade;
    use crate::storage::multimodel::stores::{
        DocumentStore, DocumentStoreConfig, GraphStore, GraphStoreConfig, ObservabilityStore,
        ObservabilityStoreConfig, VectorStore, VectorStoreConfig,
    };
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::{
        CompactionParameters, DataPointValue, DocumentCollectionInfo, DocumentRecord,
        DocumentStorageOperations, FlushParameters, IngestResult, LogQueryResult,
        MetricAggregationParams, MetricAggregationResult, NamespaceInfo,
        ObservabilityStorageOperations, TimeSeriesData, UnifiedStorageFormat,
    };
    use anyhow::Result;
    use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
    use async_trait::async_trait;
    use proximadb_kernel::error::ProximaDBError;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    struct MockVectorEngine {
        filesystem_factory: FilesystemFactory,
        results: Vec<OptimizedSearchRecord>,
        query_vectors: Mutex<Vec<Vec<f32>>>,
        storage_urls: Mutex<Vec<String>>,
    }

    impl MockVectorEngine {
        async fn new(results: Vec<OptimizedSearchRecord>) -> Self {
            let filesystem_factory = FilesystemFactory::create(FilesystemConfig::default())
                .await
                .expect("mock vector engine should create filesystem factory");
            Self {
                filesystem_factory,
                results,
                query_vectors: Mutex::new(Vec::new()),
                storage_urls: Mutex::new(Vec::new()),
            }
        }

        fn recorded_queries(&self) -> Vec<Vec<f32>> {
            self.query_vectors
                .lock()
                .expect("mock vector engine query tracking lock should not be poisoned")
                .clone()
        }

        fn recorded_storage_urls(&self) -> Vec<String> {
            self.storage_urls
                .lock()
                .expect("mock vector engine storage tracking lock should not be poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl UnifiedStorageFormat for MockVectorEngine {
        fn engine_name(&self) -> &'static str {
            "mock-vector"
        }

        fn engine_version(&self) -> &'static str {
            "0"
        }

        fn strategy(&self) -> crate::storage::traits::StorageFormatStrategy {
            crate::storage::traits::StorageFormatStrategy::Sst
        }

        async fn do_flush(
            &self,
            _params: &FlushParameters,
        ) -> Result<crate::storage::traits::FlushResult> {
            Ok(Default::default())
        }

        async fn do_compact(
            &self,
            _params: &CompactionParameters,
        ) -> Result<crate::storage::traits::CompactionResult> {
            Ok(Default::default())
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
            ctx: &crate::storage::traits::StorageQueryContext,
        ) -> Result<Vec<OptimizedSearchRecord>> {
            if let Some(vector) = ctx.query_vector() {
                self.query_vectors
                    .lock()
                    .expect("mock vector engine query tracking lock should not be poisoned")
                    .push(vector.to_vec());
            }
            if let Some(storage_url) = ctx.storage_url() {
                self.storage_urls
                    .lock()
                    .expect("mock vector engine storage tracking lock should not be poisoned")
                    .push(storage_url.to_string());
            }
            Ok(self.results.clone())
        }
    }

    struct MockGraphEngine {
        nodes: Vec<Arc<Node>>,
    }

    impl MockGraphEngine {
        fn new(nodes: Vec<Node>) -> Self {
            Self {
                nodes: nodes.into_iter().map(Arc::new).collect(),
            }
        }

        fn unsupported_graph_operation() -> ProximaDBError {
            ProximaDBError::NotImplemented("mock graph mutation not implemented".to_string())
        }
    }

    #[async_trait]
    impl GraphEngine for MockGraphEngine {
        async fn insert_node(&self, _node: Node) -> std::result::Result<Arc<Node>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        fn get_node(&self, id: &NodeId) -> std::result::Result<Option<Arc<Node>>, ProximaDBError> {
            Ok(self.nodes.iter().find(|node| node.id == *id).cloned())
        }

        async fn update_node(&self, _node: Node) -> std::result::Result<Arc<Node>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        async fn delete_node(
            &self,
            _id: &NodeId,
        ) -> std::result::Result<Option<Arc<Node>>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        async fn insert_edge(&self, _edge: Edge) -> std::result::Result<Arc<Edge>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        fn get_edge(&self, _id: &EdgeId) -> std::result::Result<Option<Arc<Edge>>, ProximaDBError> {
            Ok(None)
        }

        async fn update_edge(&self, _edge: Edge) -> std::result::Result<Arc<Edge>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        async fn delete_edge(
            &self,
            _id: &EdgeId,
        ) -> std::result::Result<Option<Arc<Edge>>, ProximaDBError> {
            Err(Self::unsupported_graph_operation())
        }

        fn get_outgoing_edges(
            &self,
            _node_id: &NodeId,
            _edge_type: Option<&str>,
        ) -> std::result::Result<Vec<Arc<Edge>>, ProximaDBError> {
            Ok(Vec::new())
        }

        fn get_incoming_edges(
            &self,
            _node_id: &NodeId,
            _edge_type: Option<&str>,
        ) -> std::result::Result<Vec<Arc<Edge>>, ProximaDBError> {
            Ok(Vec::new())
        }

        fn get_neighbors(
            &self,
            _node_id: &NodeId,
            _edge_type: Option<&str>,
        ) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
            Ok(Vec::new())
        }

        fn get_nodes_by_label(
            &self,
            label: &str,
        ) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
            Ok(self
                .nodes
                .iter()
                .filter(|node| node.labels.iter().any(|node_label| node_label == label))
                .cloned()
                .collect())
        }

        fn node_count(&self) -> std::result::Result<usize, ProximaDBError> {
            Ok(self.nodes.len())
        }

        fn edge_count(&self) -> std::result::Result<usize, ProximaDBError> {
            Ok(0)
        }

        fn get_all_nodes(&self) -> std::result::Result<Vec<Arc<Node>>, ProximaDBError> {
            Ok(self.nodes.clone())
        }
    }

    struct MockDocumentService {
        collections: HashMap<String, Vec<DocumentRecord>>,
    }

    impl MockDocumentService {
        fn new(collections: HashMap<String, Vec<DocumentRecord>>) -> Self {
            Self { collections }
        }

        fn matches_filter(document: &DocumentRecord, filter: &DocumentFilter) -> bool {
            filter.conditions.iter().all(|condition| {
                let operator =
                    crate::proto::proximadb_v1::DocFilterOperator::try_from(condition.operator)
                        .unwrap_or(crate::proto::proximadb_v1::DocFilterOperator::Unspecified);
                let path = condition.path.strip_prefix("$.").unwrap_or(&condition.path);
                let actual = document.document.fields.get(path);
                match operator {
                    crate::proto::proximadb_v1::DocFilterOperator::Eq => {
                        Self::sql_values_equal(actual, condition.value.as_ref())
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Ne => {
                        !Self::sql_values_equal(actual, condition.value.as_ref())
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Gt => {
                        Self::compare_sql_values(actual, condition.value.as_ref(), |left, right| {
                            left > right
                        })
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Gte => {
                        Self::compare_sql_values(actual, condition.value.as_ref(), |left, right| {
                            left >= right
                        })
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Lt => {
                        Self::compare_sql_values(actual, condition.value.as_ref(), |left, right| {
                            left < right
                        })
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Lte => {
                        Self::compare_sql_values(actual, condition.value.as_ref(), |left, right| {
                            left <= right
                        })
                    }
                    crate::proto::proximadb_v1::DocFilterOperator::Contains => {
                        let actual = actual.and_then(Self::sql_value_string);
                        let expected = condition.value.as_ref().and_then(Self::sql_value_string);
                        actual
                            .zip(expected)
                            .is_some_and(|(left, right)| left.contains(&right))
                    }
                    _ => false,
                }
            })
        }

        fn sql_values_equal(left: Option<&SqlValue>, right: Option<&SqlValue>) -> bool {
            match (
                left.and_then(Self::sql_value_string),
                right.and_then(Self::sql_value_string),
            ) {
                (Some(left), Some(right)) => left == right,
                _ => match (
                    left.and_then(Self::sql_value_number),
                    right.and_then(Self::sql_value_number),
                ) {
                    (Some(left), Some(right)) => (left - right).abs() < f64::EPSILON,
                    _ => false,
                },
            }
        }

        fn compare_sql_values<F>(
            left: Option<&SqlValue>,
            right: Option<&SqlValue>,
            predicate: F,
        ) -> bool
        where
            F: FnOnce(f64, f64) -> bool,
        {
            left.and_then(Self::sql_value_number)
                .zip(right.and_then(Self::sql_value_number))
                .is_some_and(|(left, right)| predicate(left, right))
        }

        fn sql_value_string(value: &SqlValue) -> Option<String> {
            match value.value.as_ref()? {
                sql_value::Value::StringValue(value) => Some(value.clone()),
                sql_value::Value::BoolValue(value) => Some(value.to_string()),
                sql_value::Value::Int64Value(value) => Some(value.to_string()),
                sql_value::Value::NumberValue(value) => Some(value.to_string()),
                _ => None,
            }
        }

        fn sql_value_number(value: &SqlValue) -> Option<f64> {
            match value.value.as_ref()? {
                sql_value::Value::Int64Value(value) => Some(*value as f64),
                sql_value::Value::NumberValue(value) => Some(*value),
                sql_value::Value::StringValue(value) => value.parse().ok(),
                _ => None,
            }
        }
    }

    #[async_trait]
    impl DocumentStorageOperations for MockDocumentService {
        async fn insert_document(
            &self,
            _collection: &str,
            _id: &str,
            _document: SqlObject,
            _indexed_paths: Vec<String>,
        ) -> Result<DocumentRecord> {
            Err(anyhow::anyhow!("mock insert not implemented"))
        }

        async fn get_document(&self, collection: &str, id: &str) -> Result<Option<DocumentRecord>> {
            Ok(self
                .collections
                .get(collection)
                .and_then(|documents| documents.iter().find(|document| document.id == id))
                .cloned())
        }

        async fn query_documents(
            &self,
            collection: &str,
            filter: Option<DocumentFilter>,
            limit: usize,
            offset: usize,
        ) -> Result<Vec<DocumentRecord>> {
            let mut documents = self
                .collections
                .get(collection)
                .cloned()
                .unwrap_or_default();

            if let Some(filter) = filter.as_ref() {
                documents.retain(|document| Self::matches_filter(document, filter));
            }

            Ok(documents.into_iter().skip(offset).take(limit).collect())
        }

        async fn update_document(
            &self,
            _collection: &str,
            _id: &str,
            _updates: Vec<DocumentUpdate>,
        ) -> Result<DocumentRecord> {
            Err(anyhow::anyhow!("mock update not implemented"))
        }

        async fn delete_document(&self, _collection: &str, _id: &str) -> Result<bool> {
            Err(anyhow::anyhow!("mock delete not implemented"))
        }

        async fn create_document_collection(
            &self,
            config: DocumentCollectionConfig,
        ) -> Result<String> {
            Ok(config.name)
        }

        async fn list_document_collections(&self) -> Result<Vec<DocumentCollectionInfo>> {
            Ok(self
                .collections
                .iter()
                .map(|(name, documents)| DocumentCollectionInfo {
                    name: name.clone(),
                    document_count: documents.len() as u64,
                    storage_size_bytes: 0,
                    indexes: vec![],
                })
                .collect())
        }
    }

    struct MockObservabilityService {
        logs: HashMap<String, Vec<LogEntry>>,
        metrics: HashMap<String, MetricAggregationResult>,
        traces: HashMap<String, Vec<TraceData>>,
        metric_params: Mutex<Vec<(String, MetricAggregationParams)>>,
    }

    impl MockObservabilityService {
        fn new(
            logs: HashMap<String, Vec<LogEntry>>,
            metrics: HashMap<String, MetricAggregationResult>,
        ) -> Self {
            Self {
                logs,
                metrics,
                traces: HashMap::new(),
                metric_params: Mutex::new(Vec::new()),
            }
        }

        fn with_traces(mut self, traces: HashMap<String, Vec<TraceData>>) -> Self {
            self.traces = traces;
            self
        }

        fn recorded_metric_params(&self) -> Vec<(String, MetricAggregationParams)> {
            self.metric_params
                .lock()
                .expect("metric params lock should not be poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl ObservabilityStorageOperations for MockObservabilityService {
        async fn ingest_logs(&self, _namespace: &str, logs: Vec<LogEntry>) -> Result<IngestResult> {
            Ok(IngestResult {
                ingested: logs.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn ingest_metrics(
            &self,
            _namespace: &str,
            metrics: Vec<MetricSample>,
        ) -> Result<IngestResult> {
            Ok(IngestResult {
                ingested: metrics.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn ingest_traces(
            &self,
            _namespace: &str,
            traces: Vec<TraceData>,
        ) -> Result<IngestResult> {
            Ok(IngestResult {
                ingested: traces.len() as u64,
                failed: 0,
                errors: vec![],
                processing_time_ms: 0,
            })
        }

        async fn query_logs(
            &self,
            namespace: &str,
            _start_time_ns: i64,
            _end_time_ns: i64,
            _filter: Option<LogFilter>,
            limit: u32,
        ) -> Result<LogQueryResult> {
            let logs = self
                .logs
                .get(namespace)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(limit as usize)
                .collect::<Vec<_>>();
            Ok(LogQueryResult {
                total_matched: logs.len() as u64,
                logs,
                next_cursor: None,
                query_time_ms: 0,
            })
        }

        async fn aggregate_metrics(
            &self,
            namespace: &str,
            params: MetricAggregationParams,
        ) -> Result<MetricAggregationResult> {
            self.metric_params
                .lock()
                .expect("metric params lock should not be poisoned")
                .push((namespace.to_string(), params));
            Ok(self
                .metrics
                .get(namespace)
                .cloned()
                .unwrap_or(MetricAggregationResult {
                    series: vec![],
                    query_time_ms: 0,
                }))
        }

        async fn query_traces(
            &self,
            namespace: &str,
            _start_time_ns: i64,
            _end_time_ns: i64,
            _trace_id: Option<String>,
            _service: Option<String>,
            limit: u32,
        ) -> Result<Vec<TraceData>> {
            Ok(self
                .traces
                .get(namespace)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(limit as usize)
                .collect())
        }

        async fn create_namespace(&self, config: ObservabilityNamespaceConfig) -> Result<String> {
            Ok(config.name)
        }

        async fn list_namespaces(&self) -> Result<Vec<NamespaceInfo>> {
            Ok(self
                .logs
                .iter()
                .map(|(name, logs)| NamespaceInfo {
                    name: name.clone(),
                    log_count: logs.len() as u64,
                    metric_count: 0,
                    trace_count: 0,
                    retention_config: Some(RetentionConfig {
                        hot_retention_hours: 24,
                        warm_retention_days: 7,
                        cold_retention_days: 30,
                        archive_retention_days: 0,
                    }),
                })
                .collect())
        }
    }

    fn string_value(value: &str) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::StringValue(value.to_string())),
        }
    }

    fn int_value(value: i64) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::Int64Value(value)),
        }
    }

    fn array_value(values: &[f64]) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::ArrayValue(SqlArray {
                values: values
                    .iter()
                    .map(|value| SqlValue {
                        value: Some(sql_value::Value::NumberValue(*value)),
                    })
                    .collect(),
            })),
        }
    }

    fn object_value(fields: Vec<(&str, SqlValue)>) -> SqlValue {
        SqlValue {
            value: Some(sql_value::Value::ObjectValue(SqlObject {
                fields: fields
                    .into_iter()
                    .map(|(key, value)| (key.to_string(), value))
                    .collect(),
            })),
        }
    }

    fn sql_object(fields: &[(&str, &str)]) -> SqlObject {
        SqlObject {
            fields: fields
                .iter()
                .map(|(key, value)| ((*key).to_string(), string_value(value)))
                .collect(),
        }
    }

    /// A federated query outside any request scope fails closed instead of
    /// running unscoped; an explicitly internal scope runs it.
    #[tokio::test]
    async fn federated_query_without_a_request_scope_fails_closed() {
        let ctx = FederatedQueryContext::new(Arc::new(MultiModelStorageFacade::new()));
        let denied = ctx.execute("SELECT 1").await;
        assert!(
            denied
                .as_ref()
                .err()
                .is_some_and(|e| e.to_string().contains("no request scope")),
            "no scope must fail closed"
        );
        let internal =
            crate::query::request_scope::with_internal_scope("test", ctx.execute("SELECT 1")).await;
        assert!(
            internal
                .as_ref()
                .err()
                .is_none_or(|e| !e.to_string().contains("no request scope")),
            "an internal scope is not denied"
        );
    }

    #[test]
    fn test_federated_context_creation() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);
        assert!(!ctx.parser.supported_extensions().is_empty());
    }

    #[test]
    fn test_federated_context_with_cache() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let cache = Arc::new(FederatedQueryResultCache::with_defaults());
        let ctx = FederatedQueryContext::with_cache(storage, cache);

        assert!(ctx.cache.is_some());
        assert!(ctx.invalidator.is_some());
        assert!(ctx.get_cache().is_some());
        assert!(ctx.get_invalidator().is_some());
    }

    #[tokio::test]
    async fn test_vector_search_uses_collection_service_storage_assignment() {
        use crate::catalog::CatalogManager;
        use crate::core::config::{StorageConfig, StorageLocation};
        use crate::proto::proximadb_v1::{CollectionConfig, StorageEngine};
        use crate::services::collection::manager::CollectionService;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("temp dir should be created");
        let base_url = format!("file://{}", temp_dir.path().display());
        let metadata_url = format!("{}/metadata", base_url);

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &metadata_url)
            .await
            .expect("xCatalog should be created");

        let mut storage_config = StorageConfig::default();
        storage_config.storage_locations = vec![StorageLocation {
            url: base_url.clone(),
            weight: 1,
            tags: vec!["local".to_string()],
            io_budget: None,
        }];
        storage_config.metadata_url = metadata_url;

        let collection_service = Arc::new(
            CollectionService::new(storage_config)
                .await
                .expect("collection service should be created")
                .with_catalog_manager(catalog_manager.clone()),
        );

        let create_result = collection_service
            .create_collection(&CollectionConfig {
                name: "productsaa".to_string(),
                dimension: 2,
                storage_engine: Some(StorageEngine::Sst as i32),
                ..Default::default()
            })
            .await
            .expect("collection creation should succeed");

        assert!(create_result.success, "collection creation should succeed");

        let collection = create_result
            .collection
            .expect("created collection should be returned");
        let expected_storage = collection
            .storage_assignment
            .as_ref()
            .map(|assignment| assignment.base_location.clone())
            .expect("created collection should have storage assignment");

        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.99)])
                .await,
        );
        let vector_store = Arc::new(VectorStore::with_engine(vector_engine.clone()));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage).with_collection_port(
            collection_service.clone() as Arc<dyn proximadb_runtime::CollectionPort>,
        );

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached(
                "SELECT id, score FROM VECTOR_SEARCH('productsaa', '[0.1, 0.2]', 1)",
            ),
        )
        .await
        .expect("vector search should execute");

        assert_eq!(result.row_count(), 1);
        assert_eq!(
            vector_engine.recorded_storage_urls(),
            vec![expected_storage],
            "federated vector search should reuse the collection storage assignment",
        );
    }

    #[tokio::test]
    async fn test_vector_search_uses_normalized_similarity_for_score_column() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord {
                    id: "doc-1".to_string(),
                    vector_id: Some("doc-1".to_string()),
                    score: 2.0,
                    similarity: Some(1.0),
                    vector: None,
                    metadata: HashMap::new(),
                    debug_info: None,
                    version: None,
                    timestamp: None,
                    updated_at: None,
                    expires_at: None,
                    source: None,
                    expanded_context: Vec::new(),
                    semantic_similarity: None,
                    quantization_info: None,
                    engine_stats: None,
                    index_path: None,
                    ..Default::default()
                },
                OptimizedSearchRecord {
                    id: "doc-2".to_string(),
                    vector_id: Some("doc-2".to_string()),
                    score: 1.0,
                    similarity: Some(0.8535534),
                    vector: None,
                    metadata: HashMap::new(),
                    debug_info: None,
                    version: None,
                    timestamp: None,
                    updated_at: None,
                    expires_at: None,
                    source: None,
                    expanded_context: Vec::new(),
                    semantic_similarity: None,
                    quantization_info: None,
                    engine_stats: None,
                    index_path: None,
                    ..Default::default()
                },
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT id, score FROM VECTOR_SEARCH('products', '[0.1]', 2)"),
        )
        .await
        .expect("vector search should execute");

        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let scores = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score column should be Float32");

        assert!((scores.value(0) - 1.0).abs() < f32::EPSILON);
        assert!((scores.value(1) - 0.8535534).abs() < 1e-6);
    }

    #[test]
    fn test_federated_context_enable_cache() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let mut ctx = FederatedQueryContext::new(storage);

        assert!(ctx.cache.is_none());

        let cache = Arc::new(FederatedQueryResultCache::with_defaults());
        ctx.enable_cache(cache);

        assert!(ctx.cache.is_some());
        assert!(ctx.invalidator.is_some());
    }

    #[test]
    fn test_cache_stats() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let cache = Arc::new(FederatedQueryResultCache::with_defaults());
        let ctx = FederatedQueryContext::with_cache(storage, cache);

        let stats = ctx.cache_stats();
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_invalidate_collection() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let cache = Arc::new(FederatedQueryResultCache::with_defaults());
        let ctx = FederatedQueryContext::with_cache(storage, cache);

        // With no cached entries, invalidation returns 0
        let invalidated = ctx.invalidate_collection("products");
        assert_eq!(invalidated, 0);
    }

    #[test]
    fn test_context_without_cache_stats() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);

        assert!(ctx.cache_stats().is_none());
    }

    #[test]
    fn test_context_without_cache_invalidation() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);

        // Without cache, invalidation is a no-op
        let invalidated = ctx.invalidate_collection("products");
        assert_eq!(invalidated, 0);
    }

    #[tokio::test]
    async fn test_plain_sql_requires_non_federated_backend() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM users"),
        )
        .await
        .expect_err("plain SQL should not execute through federated context");

        assert!(
            error
                .to_string()
                .contains("Standard relational SQL execution is not configured")
        );
    }

    #[tokio::test]
    async fn test_vector_search_requires_configured_store() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM VECTOR_SEARCH('products', '[0.1, 0.2]', 2)"),
        )
        .await
        .expect_err("vector search should fail without a configured vector store");

        assert!(error.to_string().contains("Vector store is not configured"));
    }

    #[tokio::test]
    async fn test_multi_source_join_executes_on_function_backed_sources() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-1".to_string(), 0.99),
                OptimizedSearchRecord::new("doc-2".to_string(), 0.87),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "docs".to_string(),
            vec![
                DocumentRecord {
                    id: "doc-1".to_string(),
                    document: sql_object(&[("status", "active"), ("title", "Alpha")]),
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "doc-2".to_string(),
                    document: sql_object(&[("status", "active"), ("title", "Beta")]),
                    version: 1,
                    created_at_ns: 2,
                    updated_at_ns: 2,
                },
                DocumentRecord {
                    id: "doc-3".to_string(),
                    document: sql_object(&[("status", "inactive"), ("title", "Gamma")]),
                    version: 1,
                    created_at_ns: 3,
                    updated_at_ns: 3,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM VECTOR_SEARCH('products', '[0.1]', 1), DOCUMENT_QUERY('docs', 'status = \"active\"')",
            ))
            .await
            .expect("cross-source join should execute for function-backed sources");

        assert_eq!(result.row_count(), 2);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["id", "score", "right_id", "document"]);
    }

    #[tokio::test]
    async fn test_vector_search_supports_projection_filter_and_limit() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-1".to_string(), 0.99),
                OptimizedSearchRecord::new("doc-2".to_string(), 0.87),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached(
                "SELECT id FROM VECTOR_SEARCH('products', '[0.1]', 2) WHERE id = 'doc-2' LIMIT 1",
            ),
        )
        .await
        .expect("projection/filter/limit should execute for vector search");

        assert_eq!(result.row_count(), 1);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["id"]);
        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("projected id column should be Utf8");
        assert_eq!(ids.value(0), "doc-2");
    }

    #[tokio::test]
    async fn test_vector_search_supports_order_by_and_limit() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-2".to_string(), 0.12),
                OptimizedSearchRecord::new("doc-1".to_string(), 0.91),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT id, score FROM VECTOR_SEARCH('products', '[0.1]', 2) ORDER BY score DESC LIMIT 1",
            ))
            .await
            .expect("order by should execute for vector search");

        assert_eq!(result.row_count(), 1);
        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id column should be Utf8");
        let scores = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score column should be Float32");
        assert_eq!(ids.value(0), "doc-1");
        assert!((scores.value(0) - 0.91).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_vector_search_supports_global_aggregates() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-1".to_string(), 0.91),
                OptimizedSearchRecord::new("doc-2".to_string(), 0.12),
                OptimizedSearchRecord::new("doc-3".to_string(), 0.47),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT COUNT(*) AS total, MAX(score) AS best_score, AVG(score) AS avg_score FROM VECTOR_SEARCH('products', '[0.1]', 3) WHERE score > 0.2",
            ))
            .await
            .expect("global aggregates should execute for vector search");

        assert_eq!(result.row_count(), 1);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["total", "best_score", "avg_score"]);

        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let totals = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column should be Int64");
        let best_scores = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("max column should be Float32");
        let avg_scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("avg column should be Float64");

        assert_eq!(totals.value(0), 2);
        assert!((best_scores.value(0) - 0.91).abs() < f32::EPSILON);
        assert!((avg_scores.value(0) - 0.69).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_group_by_executes_for_vector_search() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-1".to_string(), 0.91),
                OptimizedSearchRecord::new("doc-1".to_string(), 0.87),
                OptimizedSearchRecord::new("doc-2".to_string(), 0.12),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT id AS doc_id, COUNT(*) AS matches FROM VECTOR_SEARCH('products', '[0.1]', 3) GROUP BY id ORDER BY matches DESC LIMIT 1",
            ))
            .await
            .expect("group by should execute for grouped federated aggregates");

        assert_eq!(result.row_count(), 1);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["doc_id", "matches"]);

        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("grouped id column should be Utf8");
        let counts = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("grouped count column should be Int64");

        assert_eq!(ids.value(0), "doc-1");
        assert_eq!(counts.value(0), 2);
    }

    #[tokio::test]
    async fn test_select_distinct_executes_for_vector_search() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-1".to_string(), 0.91),
                OptimizedSearchRecord::new("doc-1".to_string(), 0.87),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT DISTINCT id FROM VECTOR_SEARCH('products', '[0.1]', 2)"),
        )
        .await
        .expect("distinct should execute for simple federated projections");

        assert_eq!(result.row_count(), 1);
        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("distinct id column should be Utf8");
        assert_eq!(ids.value(0), "doc-1");
    }

    #[tokio::test]
    async fn test_projection_alias_executes_for_vector_search() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT id AS doc_id, score AS similarity FROM VECTOR_SEARCH('products', '[0.1]', 1)",
            ))
            .await
            .expect("projection aliases should execute for simple federated projections");

        assert_eq!(result.row_count(), 1);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["doc_id", "similarity"]);
    }

    #[tokio::test]
    async fn test_order_by_projection_alias_executes() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![
                OptimizedSearchRecord::new("doc-2".to_string(), 0.12),
                OptimizedSearchRecord::new("doc-1".to_string(), 0.91),
            ])
            .await,
        ) as Arc<dyn UnifiedStorageFormat>;
        let vector_store =
            Arc::new(VectorStore::new(VectorStoreConfig::default()).with_sst_engine(vector_engine));
        let storage = Arc::new(MultiModelStorageFacade::new().with_vector_store(vector_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT id AS doc_id, score AS similarity FROM VECTOR_SEARCH('products', '[0.1]', 2) ORDER BY similarity DESC LIMIT 1",
            ))
            .await
            .expect("order by projection alias should execute");

        assert_eq!(result.row_count(), 1);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["doc_id", "similarity"]);

        let batch = result
            .batches
            .first()
            .expect("result should contain a batch");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("alias id column should be Utf8");
        assert_eq!(ids.value(0), "doc-1");
    }

    #[tokio::test]
    async fn test_lateral_join_executes_for_function_backed_correlated_vector_search() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "profiles".to_string(),
            vec![
                DocumentRecord {
                    id: "profile-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[0.1, 0.2]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "profile-2".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[0.3, 0.4]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 2,
                    updated_at_ns: 2,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('profiles') p JOIN LATERAL VECTOR_SEARCH('products', p.document.embedding, 1) v ON true",
            ))
            .await
            .expect("function-backed lateral join should execute");

        assert_eq!(result.row_count(), 2);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        // TD-032: "embedding" column now appears as a native Arrow FixedSizeList<Float32>
        // column instead of being embedded inside the "document" JSON string
        assert_eq!(
            field_names,
            vec!["id", "document", "embedding", "right_id", "score"]
        );
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_executes_for_nested_document_vector_path() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "profiles".to_string(),
            vec![
                DocumentRecord {
                    id: "profile-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "profile".to_string(),
                            object_value(vec![("embedding", array_value(&[0.1, 0.2]))]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "profile-2".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "profile".to_string(),
                            object_value(vec![("embedding", array_value(&[0.3, 0.4]))]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 2,
                    updated_at_ns: 2,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('profiles') p JOIN LATERAL VECTOR_SEARCH('products', p.document.profile.embedding, 1) v ON true",
            ))
            .await
            .expect("nested function-backed lateral join should execute");

        assert_eq!(result.row_count(), 2);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(
            field_names,
            vec![
                "id",
                "document",
                "document.profile.embedding",
                "right_id",
                "score"
            ]
        );
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_executes_for_graph_vector_property_path() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "node-1".to_string(),
                    labels: vec!["Entity".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.9, 0.1])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "node-2".to_string(),
                    labels: vec!["Entity".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.2, 0.8])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Entity) RETURN n') g JOIN LATERAL VECTOR_SEARCH('products', g.properties.embedding, 1) v ON true",
            ))
            .await
            .expect("graph-backed correlated vector search should execute");

        assert_eq!(result.row_count(), 2);
        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(
            field_names,
            vec!["node_id", "label", "properties", "embedding", "id", "score"]
        );
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.9, 0.1], vec![0.2, 0.8]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_skips_document_rows_without_correlated_vector() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "profiles".to_string(),
            vec![
                DocumentRecord {
                    id: "profile-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[0.1, 0.2]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "profile-2".to_string(),
                    document: sql_object(&[("title", "No embedding")]),
                    version: 1,
                    created_at_ns: 2,
                    updated_at_ns: 2,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('profiles') p JOIN LATERAL VECTOR_SEARCH('products', p.document.embedding, 1) v ON true",
            ))
            .await
            .expect("lateral join should skip rows without a correlated vector");

        assert_eq!(result.row_count(), 1);
        assert_eq!(vector_engine.recorded_queries(), vec![vec![0.1, 0.2]]);
    }

    #[tokio::test]
    async fn test_lateral_join_skips_graph_rows_without_correlated_vector_property() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "node-1".to_string(),
                    labels: vec!["Entity".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.9, 0.1])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "node-2".to_string(),
                    labels: vec!["Entity".to_string()],
                    properties: HashMap::from([(
                        "title".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::StringValue(
                                "No embedding".to_string(),
                            )),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Entity) RETURN n') g JOIN LATERAL VECTOR_SEARCH('products', g.properties.embedding, 1) v ON true",
            ))
            .await
            .expect("lateral join should skip graph rows without a correlated vector property");

        assert_eq!(result.row_count(), 1);
        assert_eq!(vector_engine.recorded_queries(), vec![vec![0.9, 0.1]]);
    }

    #[tokio::test]
    async fn test_lateral_join_rejects_malformed_document_correlated_vector() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "profiles".to_string(),
            vec![DocumentRecord {
                id: "profile-1".to_string(),
                document: sql_object(&[("embedding", "not-a-vector")]),
                version: 1,
                created_at_ns: 1,
                updated_at_ns: 1,
            }],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('profiles') p JOIN LATERAL VECTOR_SEARCH('products', p.document.embedding, 1) v ON true",
            ))
            .await
            .expect_err("malformed correlated vectors should still fail");

        let error_text = error.to_string();
        assert!(error_text.contains("Failed to resolve lateral join correlations"));
        assert!(error_text.contains("did not contain a vector literal"));
    }

    #[tokio::test]
    async fn test_lateral_join_uses_right_document_alias_in_multi_document_outer_join() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([
            (
                "left_profiles".to_string(),
                vec![DocumentRecord {
                    id: "left-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[9.0, 8.0]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                }],
            ),
            (
                "right_profiles".to_string(),
                vec![
                    DocumentRecord {
                        id: "right-1".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.1, 0.2]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 2,
                        updated_at_ns: 2,
                    },
                    DocumentRecord {
                        id: "right-2".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.3, 0.4]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 3,
                        updated_at_ns: 3,
                    },
                ],
            ),
        ]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') p, DOCUMENT_QUERY('right_profiles') q JOIN LATERAL VECTOR_SEARCH('products', q.document.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-document lateral join should execute");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_treats_document_aliases_case_insensitively() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([
            (
                "left_profiles".to_string(),
                vec![DocumentRecord {
                    id: "left-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[9.0, 8.0]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                }],
            ),
            (
                "right_profiles".to_string(),
                vec![
                    DocumentRecord {
                        id: "right-1".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.1, 0.2]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 2,
                        updated_at_ns: 2,
                    },
                    DocumentRecord {
                        id: "right-2".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.3, 0.4]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 3,
                        updated_at_ns: 3,
                    },
                ],
            ),
        ]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') p, DOCUMENT_QUERY('right_profiles') q JOIN LATERAL VECTOR_SEARCH('products', Q.document.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-document lateral join should treat aliases case-insensitively");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_supports_quoted_document_aliases() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([
            (
                "left_profiles".to_string(),
                vec![DocumentRecord {
                    id: "left-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[9.0, 8.0]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                }],
            ),
            (
                "right_profiles".to_string(),
                vec![
                    DocumentRecord {
                        id: "right-1".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.1, 0.2]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 2,
                        updated_at_ns: 2,
                    },
                    DocumentRecord {
                        id: "right-2".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.3, 0.4]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 3,
                        updated_at_ns: 3,
                    },
                ],
            ),
        ]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') \"LeftAlias\", DOCUMENT_QUERY('right_profiles') \"RightAlias\" JOIN LATERAL VECTOR_SEARCH('products', \"RightAlias\".document.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-document lateral join should support quoted aliases");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_supports_quoted_document_aliases_with_dots() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([
            (
                "left_profiles".to_string(),
                vec![DocumentRecord {
                    id: "left-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[9.0, 8.0]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                }],
            ),
            (
                "right_profiles".to_string(),
                vec![
                    DocumentRecord {
                        id: "right-1".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.1, 0.2]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 2,
                        updated_at_ns: 2,
                    },
                    DocumentRecord {
                        id: "right-2".to_string(),
                        document: SqlObject {
                            fields: HashMap::from([(
                                "embedding".to_string(),
                                array_value(&[0.3, 0.4]),
                            )]),
                        },
                        version: 1,
                        created_at_ns: 3,
                        updated_at_ns: 3,
                    },
                ],
            ),
        ]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') \"Left.Alias\", DOCUMENT_QUERY('right_profiles') \"Right.Alias\" JOIN LATERAL VECTOR_SEARCH('products', \"Right.Alias\".document.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-document lateral join should support quoted dotted aliases");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_rejects_mismatched_quoted_document_alias_case() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );

        let document_service = Arc::new(MockDocumentService::new(HashMap::from([
            (
                "left_profiles".to_string(),
                vec![DocumentRecord {
                    id: "left-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[9.0, 8.0]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                }],
            ),
            (
                "right_profiles".to_string(),
                vec![DocumentRecord {
                    id: "right-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([(
                            "embedding".to_string(),
                            array_value(&[0.1, 0.2]),
                        )]),
                    },
                    version: 1,
                    created_at_ns: 2,
                    updated_at_ns: 2,
                }],
            ),
        ]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_document_store(document_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('left_profiles') \"LeftAlias\", DOCUMENT_QUERY('right_profiles') \"RightAlias\" JOIN LATERAL VECTOR_SEARCH('products', \"RIGHTALIAS\".document.embedding, 1) v ON true",
            ))
            .await
            .expect_err("quoted document alias case mismatch should not silently bind another source");

        let error_text = error.to_string();
        assert!(error_text.contains("did not match any outer source alias"));
        assert!(error_text.contains("\"RIGHTALIAS\".document.embedding"));
    }

    #[tokio::test]
    async fn test_lateral_join_uses_right_graph_alias_in_multi_graph_outer_join() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "left-1".to_string(),
                    labels: vec!["Left".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![9.0, 8.0])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "right-1".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.1, 0.2])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
                Node {
                    id: "right-2".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.3, 0.4])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 3,
                    updated_at_ms: 3,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Left) RETURN n') p, GRAPH_QUERY('MATCH (n:Right) RETURN n') q JOIN LATERAL VECTOR_SEARCH('products', q.properties.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-graph lateral join should execute");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_supports_quoted_graph_aliases() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "left-1".to_string(),
                    labels: vec!["Left".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![9.0, 8.0])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "right-1".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.1, 0.2])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
                Node {
                    id: "right-2".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.3, 0.4])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 3,
                    updated_at_ms: 3,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Left) RETURN n') \"LeftAlias\", GRAPH_QUERY('MATCH (n:Right) RETURN n') \"RightAlias\" JOIN LATERAL VECTOR_SEARCH('products', \"RightAlias\".properties.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-graph lateral join should support quoted aliases");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_supports_quoted_graph_aliases_with_dots() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "left-1".to_string(),
                    labels: vec!["Left".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![9.0, 8.0])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "right-1".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.1, 0.2])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
                Node {
                    id: "right-2".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.3, 0.4])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 3,
                    updated_at_ms: 3,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Left) RETURN n') \"Left.Alias\", GRAPH_QUERY('MATCH (n:Right) RETURN n') \"Right.Alias\" JOIN LATERAL VECTOR_SEARCH('products', \"Right.Alias\".properties.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-graph lateral join should support quoted dotted aliases");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_lateral_join_rejects_mismatched_quoted_graph_alias_case() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "left-1".to_string(),
                    labels: vec!["Left".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![9.0, 8.0])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "right-1".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.1, 0.2])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Left) RETURN n') \"LeftAlias\", GRAPH_QUERY('MATCH (n:Right) RETURN n') \"RightAlias\" JOIN LATERAL VECTOR_SEARCH('products', \"RIGHTALIAS\".properties.embedding, 1) v ON true",
            ))
            .await
            .expect_err("quoted graph alias case mismatch should not silently bind another source");

        let error_text = error.to_string();
        assert!(error_text.contains("did not match any outer source alias"));
        assert!(error_text.contains("\"RIGHTALIAS\".properties.embedding"));
    }

    #[tokio::test]
    async fn test_lateral_join_treats_graph_aliases_case_insensitively() {
        let vector_engine = Arc::new(
            MockVectorEngine::new(vec![OptimizedSearchRecord::new("doc-1".to_string(), 0.91)])
                .await,
        );
        let vector_store = Arc::new(
            VectorStore::new(VectorStoreConfig::default())
                .with_sst_engine(vector_engine.clone() as Arc<dyn UnifiedStorageFormat>),
        );
        let graph_store = Arc::new(GraphStore::new(GraphStoreConfig::default()).with_engine(
            Arc::new(MockGraphEngine::new(vec![
                Node {
                    id: "left-1".to_string(),
                    labels: vec!["Left".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![9.0, 8.0])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 1,
                    updated_at_ms: 1,
                },
                Node {
                    id: "right-1".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.1, 0.2])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 2,
                    updated_at_ms: 2,
                },
                Node {
                    id: "right-2".to_string(),
                    labels: vec!["Right".to_string()],
                    properties: HashMap::from([(
                        "embedding".to_string(),
                        PropertyValue {
                            value: Some(property_value::Value::VectorValue(vec![0.3, 0.4])),
                        },
                    )]),
                    embedding: None,
                    created_at_ms: 3,
                    updated_at_ms: 3,
                },
            ])) as Arc<dyn GraphEngine>,
        ));

        let storage = Arc::new(
            MultiModelStorageFacade::new()
                .with_vector_store(vector_store)
                .with_graph_store(graph_store),
        );
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM GRAPH_QUERY('MATCH (n:Left) RETURN n') p, GRAPH_QUERY('MATCH (n:Right) RETURN n') q JOIN LATERAL VECTOR_SEARCH('products', Q.properties.embedding, 1) v ON true",
            ))
            .await
            .expect("multi-graph lateral join should treat aliases case-insensitively");

        assert_eq!(result.row_count(), 2);
        assert_eq!(
            vector_engine.recorded_queries(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]]
        );
    }

    #[tokio::test]
    async fn test_document_query_root_hides_internal_native_vector_columns() {
        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "profiles".to_string(),
            vec![DocumentRecord {
                id: "profile-1".to_string(),
                document: SqlObject {
                    fields: HashMap::from([("embedding".to_string(), array_value(&[0.1, 0.2]))]),
                },
                version: 1,
                created_at_ns: 1,
                updated_at_ns: 1,
            }],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(MultiModelStorageFacade::new().with_document_store(document_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM DOCUMENT_QUERY('profiles')"),
        )
        .await
        .expect("plain document query should execute");

        let field_names: Vec<String> = result
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert_eq!(field_names, vec!["id", "document"]);
    }

    #[tokio::test]
    async fn test_document_query_filter_supports_comparison_and_and_clauses() {
        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "orders".to_string(),
            vec![
                DocumentRecord {
                    id: "order-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([
                            ("status".to_string(), string_value("pending")),
                            ("price".to_string(), int_value(125)),
                        ]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "order-2".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([
                            ("status".to_string(), string_value("pending")),
                            ("price".to_string(), int_value(25)),
                        ]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "order-3".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([
                            ("status".to_string(), string_value("shipped")),
                            ("price".to_string(), int_value(225)),
                        ]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(MultiModelStorageFacade::new().with_document_store(document_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached(
                "SELECT id FROM DOCUMENT_QUERY('orders', 'status = \"pending\" AND price >= 100')",
            ),
        )
        .await
        .expect("document comparison filters should execute");

        assert_eq!(result.row_count(), 1);
        let ids = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id column should be utf8");
        assert_eq!(ids.value(0), "order-1");
    }

    #[tokio::test]
    async fn test_document_query_filter_keywords_are_case_insensitive() {
        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "orders".to_string(),
            vec![
                DocumentRecord {
                    id: "order-1".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([
                            ("status".to_string(), string_value("pending")),
                            ("price".to_string(), int_value(125)),
                        ]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
                DocumentRecord {
                    id: "order-2".to_string(),
                    document: SqlObject {
                        fields: HashMap::from([
                            ("status".to_string(), string_value("closed")),
                            ("price".to_string(), int_value(125)),
                        ]),
                    },
                    version: 1,
                    created_at_ns: 1,
                    updated_at_ns: 1,
                },
            ],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(MultiModelStorageFacade::new().with_document_store(document_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT id FROM DOCUMENT_QUERY('orders', 'status contains \"pend\" and price >= 100')",
            ))
            .await
            .expect("document filter keywords should be case-insensitive");

        assert_eq!(result.row_count(), 1);
        let ids = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id column should be utf8");
        assert_eq!(ids.value(0), "order-1");
    }

    #[tokio::test]
    async fn test_document_query_filter_rejects_unsupported_or_clause() {
        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "orders".to_string(),
            vec![DocumentRecord {
                id: "order-1".to_string(),
                document: SqlObject {
                    fields: HashMap::from([("status".to_string(), string_value("pending"))]),
                },
                version: 1,
                created_at_ns: 1,
                updated_at_ns: 1,
            }],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(MultiModelStorageFacade::new().with_document_store(document_store));
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('orders', 'status = \"pending\" OR status = \"shipped\"')",
            ))
            .await
            .expect_err("unsupported OR filters should fail explicitly");

        assert!(
            error
                .to_string()
                .contains("Unsupported DOCUMENT_QUERY filter clause")
        );
    }

    #[tokio::test]
    async fn test_document_query_filter_rejects_lowercase_or_clause() {
        let document_service = Arc::new(MockDocumentService::new(HashMap::from([(
            "orders".to_string(),
            vec![DocumentRecord {
                id: "order-1".to_string(),
                document: SqlObject {
                    fields: HashMap::from([("status".to_string(), string_value("pending"))]),
                },
                version: 1,
                created_at_ns: 1,
                updated_at_ns: 1,
            }],
        )]))) as Arc<dyn DocumentStorageOperations>;
        let document_store = Arc::new(
            DocumentStore::new(DocumentStoreConfig::default()).with_service(document_service),
        );

        let storage = Arc::new(MultiModelStorageFacade::new().with_document_store(document_store));
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM DOCUMENT_QUERY('orders', 'status = \"pending\" or status = \"shipped\"')",
            ))
            .await
            .expect_err("lowercase OR filters should fail explicitly");

        assert!(
            error
                .to_string()
                .contains("OR filters are not yet supported")
        );
    }

    #[tokio::test]
    async fn test_observability_logs_query_executes_against_store() {
        let observability_service = Arc::new(MockObservabilityService::new(
            HashMap::from([(
                "production".to_string(),
                vec![LogEntry {
                    timestamp_ns: 42,
                    severity: Severity::Error as i32,
                    message: "disk full".to_string(),
                    fields: HashMap::new(),
                    source: Some("node-1".to_string()),
                    service: Some("storage".to_string()),
                }],
            )]),
            HashMap::new(),
        )) as Arc<dyn ObservabilityStorageOperations>;
        let observability_store = Arc::new(
            ObservabilityStore::new(ObservabilityStoreConfig::default())
                .with_service(observability_service),
        );

        let storage =
            Arc::new(MultiModelStorageFacade::new().with_observability_store(observability_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM LOGS('production')"),
        )
        .await
        .expect("logs query should execute against the observability store");

        assert_eq!(result.row_count(), 1);
        let timestamp = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("timestamp column should be int64");
        let level = result.batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("level column should be utf8");
        let message = result.batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("message column should be utf8");

        assert_eq!(timestamp.value(0), 42);
        assert_eq!(level.value(0), "ERROR");
        assert_eq!(message.value(0), "disk full");
    }

    #[tokio::test]
    async fn test_observability_metrics_query_executes_against_store() {
        let observability_service = Arc::new(MockObservabilityService::new(
            HashMap::new(),
            HashMap::from([(
                "production".to_string(),
                MetricAggregationResult {
                    series: vec![TimeSeriesData {
                        labels: HashMap::from([("__name__".to_string(), "cpu_usage".to_string())]),
                        points: vec![DataPointValue {
                            timestamp_ns: 99,
                            value: 0.75,
                        }],
                    }],
                    query_time_ms: 0,
                },
            )]),
        ));
        let recorded_service = observability_service.clone();
        let observability_store = Arc::new(
            ObservabilityStore::new(ObservabilityStoreConfig::default())
                .with_service(observability_service as Arc<dyn ObservabilityStorageOperations>),
        );

        let storage =
            Arc::new(MultiModelStorageFacade::new().with_observability_store(observability_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM METRICS('production')"),
        )
        .await
        .expect("metrics query should execute against the observability store");

        assert_eq!(result.row_count(), 1);
        let timestamp = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("timestamp column should be int64");
        let metric_name = result.batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("metric_name column should be utf8");
        let value = result.batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("value column should be float32");

        assert_eq!(timestamp.value(0), 99);
        assert_eq!(metric_name.value(0), "cpu_usage");
        assert!((value.value(0) - 0.75).abs() < f32::EPSILON);

        let recorded_params = recorded_service.recorded_metric_params();
        assert_eq!(recorded_params.len(), 1);
        assert_eq!(recorded_params[0].0, "production");
        assert_eq!(recorded_params[0].1.metric_name, "*");
        assert_eq!(recorded_params[0].1.aggregation, MetricAggregation::Avg);
        assert_eq!(recorded_params[0].1.step_seconds, 60);
    }

    #[tokio::test]
    async fn test_observability_metrics_query_applies_sql_filter() {
        let observability_service = Arc::new(MockObservabilityService::new(
            HashMap::new(),
            HashMap::from([(
                "production".to_string(),
                MetricAggregationResult {
                    series: vec![
                        TimeSeriesData {
                            labels: HashMap::from([(
                                "__name__".to_string(),
                                "cpu_usage".to_string(),
                            )]),
                            points: vec![DataPointValue {
                                timestamp_ns: 100,
                                value: 0.75,
                            }],
                        },
                        TimeSeriesData {
                            labels: HashMap::from([(
                                "__name__".to_string(),
                                "memory_usage".to_string(),
                            )]),
                            points: vec![DataPointValue {
                                timestamp_ns: 101,
                                value: 0.55,
                            }],
                        },
                    ],
                    query_time_ms: 0,
                },
            )]),
        )) as Arc<dyn ObservabilityStorageOperations>;
        let observability_store = Arc::new(
            ObservabilityStore::new(ObservabilityStoreConfig::default())
                .with_service(observability_service),
        );

        let storage =
            Arc::new(MultiModelStorageFacade::new().with_observability_store(observability_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached(
                "SELECT * FROM METRICS('production') WHERE metric_name = 'cpu_usage'",
            ),
        )
        .await
        .expect("metrics query should apply SQL filters after store execution");

        assert_eq!(result.row_count(), 1);
        let metric_name = result.batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("metric_name column should be utf8");
        assert_eq!(metric_name.value(0), "cpu_usage");
    }

    #[tokio::test]
    async fn test_observability_traces_query_executes_against_store() {
        let observability_service = Arc::new(
            MockObservabilityService::new(HashMap::new(), HashMap::new()).with_traces(
                HashMap::from([(
                    "production".to_string(),
                    vec![TraceData {
                        trace_id: "trace-1".to_string(),
                        span_id: "span-1".to_string(),
                        parent_span_id: None,
                        name: "flush_segment".to_string(),
                        kind: 0,
                        start_time_ns: 1_000,
                        end_time_ns: 1_750,
                        status: None,
                        attributes: HashMap::new(),
                        events: vec![],
                        links: vec![],
                    }],
                )]),
            ),
        ) as Arc<dyn ObservabilityStorageOperations>;
        let observability_store = Arc::new(
            ObservabilityStore::new(ObservabilityStoreConfig::default())
                .with_service(observability_service),
        );

        let storage =
            Arc::new(MultiModelStorageFacade::new().with_observability_store(observability_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM TRACES('production')"),
        )
        .await
        .expect("traces query should execute against the observability store");

        assert_eq!(result.row_count(), 1);
        let trace_id = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("trace_id column should be utf8");
        let span_id = result.batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("span_id column should be utf8");
        let operation = result.batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("operation column should be utf8");
        let duration = result.batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("duration_ns column should be int64");

        assert_eq!(trace_id.value(0), "trace-1");
        assert_eq!(span_id.value(0), "span-1");
        assert_eq!(operation.value(0), "flush_segment");
        assert_eq!(duration.value(0), 750);
    }

    #[tokio::test]
    async fn test_observability_traces_query_applies_duration_filter() {
        let observability_service = Arc::new(
            MockObservabilityService::new(HashMap::new(), HashMap::new()).with_traces(
                HashMap::from([(
                    "production".to_string(),
                    vec![
                        TraceData {
                            trace_id: "trace-fast".to_string(),
                            span_id: "span-fast".to_string(),
                            parent_span_id: None,
                            name: "fast_path".to_string(),
                            kind: 0,
                            start_time_ns: 1_000,
                            end_time_ns: 1_100,
                            status: None,
                            attributes: HashMap::new(),
                            events: vec![],
                            links: vec![],
                        },
                        TraceData {
                            trace_id: "trace-slow".to_string(),
                            span_id: "span-slow".to_string(),
                            parent_span_id: None,
                            name: "slow_path".to_string(),
                            kind: 0,
                            start_time_ns: 2_000,
                            end_time_ns: 3_500,
                            status: None,
                            attributes: HashMap::new(),
                            events: vec![],
                            links: vec![],
                        },
                    ],
                )]),
            ),
        ) as Arc<dyn ObservabilityStorageOperations>;
        let observability_store = Arc::new(
            ObservabilityStore::new(ObservabilityStoreConfig::default())
                .with_service(observability_service),
        );

        let storage =
            Arc::new(MultiModelStorageFacade::new().with_observability_store(observability_store));
        let ctx = FederatedQueryContext::new(storage);

        let result = crate::query::request_scope::with_internal_scope(
            "test",
            ctx.execute_uncached("SELECT * FROM TRACES('production') WHERE duration_ns >= 1000"),
        )
        .await
        .expect("trace query should apply SQL duration filters after store execution");

        assert_eq!(result.row_count(), 1);
        let trace_id = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("trace_id column should be utf8");
        let duration = result.batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("duration_ns column should be int64");
        assert_eq!(trace_id.value(0), "trace-slow");
        assert_eq!(duration.value(0), 1_500);
    }

    #[tokio::test]
    async fn test_lateral_join_still_requires_relational_scan_backend_for_tables() {
        let storage = Arc::new(MultiModelStorageFacade::new());
        let ctx = FederatedQueryContext::new(storage);

        let error = crate::query::request_scope::with_internal_scope("test", ctx
            .execute_uncached(
                "SELECT * FROM users u JOIN LATERAL VECTOR_SEARCH('products', u.embedding, 1) v ON true",
            ))
            .await
            .expect_err("generic relational outer scans should still fail explicitly");

        let msg = error.to_string();
        assert!(
            msg.contains("Scan execution is not") && msg.contains("users"),
            "Expected scan error for 'users' but got: {}",
            msg
        );
    }
}

/// Result-cache key for `sql` in the current request scope: a tenant's
/// results are keyed by that tenant, so one tenant never reads another's
/// cached rows.
fn federated_cache_key(sql: &str) -> QueryKey {
    match crate::query::request_scope::current_scope() {
        Some(crate::query::request_scope::RequestScope::Tenant(tenant)) => {
            QueryKey::from_sql_with_params(sql, &["tenant", tenant.as_str()])
        }
        _ => QueryKey::from_sql(sql),
    }
}
