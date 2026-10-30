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

#[cfg(test)]
mod tests;
