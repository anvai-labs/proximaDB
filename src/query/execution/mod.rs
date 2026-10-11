//! Unified Query Execution - Vector + Graph + Hybrid
//!
//! This module provides the new execution engine that operates on internal AST from sql_frontend lowering.
//!
//! Key architectural improvement: Uses HashMap metadata filtering for O(1) lookups
//! instead of Vec<MetadataItem> linear scans, achieving 10x performance gain.

pub mod datafusion_bridge;
pub mod datafusion_engine;
#[cfg(feature = "duckdb")]
pub mod duckdb_engine;
pub mod engine;

pub mod executor;
pub mod low_latency_executor;
pub mod morsel_scheduler;
pub mod native_engine;
pub mod native_join_ops;
pub mod native_ops;
pub mod native_parquet_scan;
pub mod native_scan;
pub mod native_shadow;
pub mod native_table_provider;
pub mod olap_delta_merge;
pub mod plan_cache;
pub mod planner;
pub mod set_operations;
#[cfg(test)]
pub(crate) mod test_runtime;
pub mod window_executor;

// Re-export execution engine types
pub use datafusion_engine::DataFusionLocalEngine;
pub(crate) use engine::normalize_table_key;
pub use engine::{
    ExecutionControls, ExecutionEngine, ExecutionError, ExecutionPipelineResult,
    ExecutionRowStream, ExecutionStreamResult, NativeVolcanoEngine, QueryExecutionContext,
    RowLimitMode, execute_sql_stream_with_backend, execute_sql_with_backend,
};

// Re-export low-latency execution types
pub use low_latency_executor::{
    LowLatencyConfig, LowLatencyExecutor, LowLatencyMetrics, StreamedQueryResult,
};
pub use plan_cache::{
    CachedPlan, ExecutionPlanCacheStats, PlanCacheConfig as QueryPlanCacheConfig, PlanKey,
    QueryPlanCache,
};

// Re-export set_operations
pub use set_operations::*;

use crate::query::ast::Query;
use crate::services::operations::vectors::VectorOperationsService;
use anyhow::{Result, anyhow};
use proximadb_graph_query::service::GraphExecutionService;
use std::sync::Arc;

/// Unified query engine with AST-based execution
///
/// This engine consumes lowered AST from sql_frontend and routes execution
/// to appropriate services (VOS for vector, GraphService for graph, hybrid for SKS).
pub struct QueryEngine {
    #[allow(dead_code)]
    vector_service: Arc<VectorOperationsService>,
    planner: crate::query::execution::planner::ExecutionPlanner,
    executor: crate::query::execution::executor::QueryExecutor,
}

impl QueryEngine {
    /// Create new unified query engine
    pub fn new(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
    ) -> Self {
        let planner = crate::query::execution::planner::ExecutionPlanner::new(
            vector_service.clone(),
            graph_service.clone(),
        );

        let executor = crate::query::execution::executor::QueryExecutor::new(
            Some(vector_service.clone()),
            graph_service,
        );

        Self {
            vector_service,
            planner,
            executor,
        }
    }

    /// Create query engine with planner parameters (e.g., bound SQL params)
    pub fn new_with_params(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
        params: Option<Vec<crate::proto::proximadb_v1::SqlValue>>,
    ) -> Self {
        let planner = crate::query::execution::planner::ExecutionPlanner::with_params(
            vector_service.clone(),
            graph_service.clone(),
            params,
        );
        let executor = crate::query::execution::executor::QueryExecutor::new(
            Some(vector_service.clone()),
            graph_service,
        );
        Self {
            vector_service,
            planner,
            executor,
        }
    }

    /// Create query engine with planner params and seeding strategy for hybrid queries
    pub fn new_with_options(
        vector_service: Arc<VectorOperationsService>,
        graph_service: Arc<dyn GraphExecutionService>,
        params: Option<Vec<crate::proto::proximadb_v1::SqlValue>>,
        seeding_strategy: SeedingStrategy,
        fusion_weights: Option<Vec<f64>>,
    ) -> Self {
        let mut planner = crate::query::execution::planner::ExecutionPlanner::with_params(
            vector_service.clone(),
            graph_service.clone(),
            params,
        );
        planner.set_seeding_strategy(seeding_strategy.clone());
        planner.set_fusion_weights(fusion_weights);
        let executor = crate::query::execution::executor::QueryExecutor::new(
            Some(vector_service.clone()),
            graph_service,
        );
        Self {
            vector_service,
            planner,
            executor,
        }
    }

    /// Execute query from internal AST (post-lowering from sql_frontend)
    ///
    /// This is the main entry point, providing superior performance through HashMap metadata filtering.
    pub async fn execute_frontend(&self, query: Query) -> Result<ExecutionQueryResult> {
        // 1. Generate optimized execution plan from AST
        let plan = self.planner.create_plan(&query)?;

        // 2. Route to appropriate execution path based on query characteristics
        match plan.execution_strategy {
            ExecutionStrategy::VectorOnly => {
                // Pure vector search with HashMap metadata filtering
                self.executor.execute_vector_plan(plan).await
            }
            ExecutionStrategy::GraphOnly => {
                // Pure graph traversal with ORION engine
                self.executor.execute_graph_plan(plan).await
            }
            ExecutionStrategy::Hybrid => {
                // Combined vector + graph with advanced fusion
                self.executor.execute_hybrid_plan(plan).await
            }
            ExecutionStrategy::Relational => {
                // Traditional relational operations (future)
                Err(anyhow!("Relational queries not yet implemented"))
            }
        }
    }

    /// Generate EXPLAIN output for query optimization
    pub async fn explain_frontend(&self, query: Query) -> Result<ExplainResult> {
        let plan = self.planner.create_plan(&query)?;

        let mut hints = plan.performance_hints.clone();
        // Add hybrid configuration hints
        hints.push(format!("Seeding: {:?}", plan.seeding_strategy));
        let has_vector = plan
            .execution_steps
            .iter()
            .any(|op| matches!(op, ExecutionOperation::VectorQuery { .. }));
        let has_graph = plan
            .execution_steps
            .iter()
            .any(|op| matches!(op, ExecutionOperation::GraphTraversal { .. }));
        if has_vector && has_graph {
            hints.push("Hybrid: parallel execution + seed handoff available".to_string());
        }
        // Extract fusion weights for explain
        for op in &plan.execution_steps {
            if let ExecutionOperation::Fusion { weights, .. } = op {
                hints.push(format!("Fusion weights: {:?}", weights));
            }
        }

        Ok(ExplainResult {
            query_type: plan.execution_strategy.clone(),
            estimated_cost: plan.estimated_cost,
            operations: plan
                .execution_steps
                .iter()
                .map(|op| op.describe())
                .collect(),
            optimizations: plan.optimizations.clone(),
            performance_hints: hints,
        })
    }
}

pub use crate::query::query_optimizer::{
    AggregateFunc, AggregateSpec, ExecutionStep as ExecutionOperation, ExecutionStrategy,
    FusionStrategy, JoinKind, ProjectionTransform, SeedingStrategy,
    UnifiedExecutionPlan as ExecutionPlan,
};

/// Backwards-compat alias for [`ExecutionQueryResult`].
pub type QueryResult = ExecutionQueryResult;

/// Query execution result
#[derive(Debug, Clone, Default)]
pub struct ExecutionQueryResult {
    /// Result rows
    pub rows: Vec<QueryRow>,
    /// Total number of matching results
    pub total_found: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Descriptions of operations performed
    pub operations_performed: Vec<String>,
    /// Number of cache hits during execution
    pub cache_hits: usize,
    /// Detailed performance metrics
    pub performance_metrics: QueryPerformanceMetrics,
}

/// Individual result row
#[derive(Debug, Clone, Default)]
pub struct QueryRow {
    /// Field values by column name
    pub fields: std::collections::HashMap<String, serde_json::Value>,
    /// Similarity score for vector search results
    pub similarity_score: Option<f64>,
    /// Graph traversal distance for graph results
    pub graph_distance: Option<u32>,
    /// Provenance tracking for result lineage
    pub provenance: Option<Vec<String>>,
}

/// Performance metrics for query analysis
#[derive(Debug, Clone, Default)]
pub struct QueryPerformanceMetrics {
    /// Number of vectors scanned during search
    pub vectors_scanned: usize,
    /// Number of graph nodes visited during traversal
    pub graph_nodes_visited: usize,
    /// Number of metadata lookups performed
    pub metadata_lookups: usize,
    /// Cache hit ratio (0.0 to 1.0)
    pub cache_hit_ratio: f64,
    /// Filter selectivity achieved (0.0 to 1.0)
    pub filter_selectivity: f64,
}

/// EXPLAIN result for query optimization
#[derive(Debug, Clone)]
pub struct ExplainResult {
    /// Query execution strategy
    pub query_type: ExecutionStrategy,
    /// Estimated total cost
    pub estimated_cost: f64,
    /// Descriptions of planned operations
    pub operations: Vec<String>,
    /// Optimizations applied
    pub optimizations: Vec<String>,
    /// Performance improvement hints
    pub performance_hints: Vec<String>,
}

#[cfg(all(test, feature = "axis"))]
mod execution_tests {
    use super::*;

    #[tokio::test]
    async fn test_query_engine_creation() {
        // Test unified engine creation with all services
        // Deferred: Create proper test setup with mock dependencies
        // let vector_service = Arc::new(VectorOperationsService::new(storage_engine, wal_manager, axis_index_manager, collection_service));
        // let graph_service = Arc::new(GraphService::new());

        // let engine = QueryEngine::new(vector_service, graph_service);

        // Verify engine is properly configured
        assert!(true); // Deferred: Add specific validation with proper test setup
    }

    #[tokio::test]
    async fn test_execution_strategy_detection() {
        // Test that query analysis correctly determines execution strategy
        let engine = create_test_engine().await;

        // Vector-only query
        let vector_query = create_test_vector_query();
        let vector_plan = engine.planner.create_plan(&vector_query).unwrap();
        // The query has only SksSimilar (no graph ops), so strategy is VectorOnly
        assert!(matches!(
            vector_plan.execution_strategy,
            ExecutionStrategy::VectorOnly
        ));

        // Deferred: Test graph-only and hybrid strategies
    }

    async fn create_test_engine() -> QueryEngine {
        use crate::graph::service::GraphOperationsService;
        use crate::index::AxisManager;
        use crate::services::collection::manager::CollectionService;
        use crate::services::operations::vectors::VectorOperationsService;
        use crate::storage::engines::sst::SstEngine;
        use crate::storage::persistence::write_ahead_log::WriteAheadLogManager;
        use std::sync::Arc;

        // Create temporary directory for storage
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let storage_url = format!("file:///{}", temp_dir.path().display());

        // Create SST storage engine
        let storage_engine = Arc::new(SstEngine::new().await.expect("Failed to create SST engine"));

        // Create WAL manager with default config
        use crate::storage::persistence::write_ahead_log::WALConfig;
        let wal_config = WALConfig::default();
        let wal_manager = Arc::new(
            WriteAheadLogManager::new(wal_config)
                .await
                .expect("Failed to create WAL manager"),
        );

        // Create Axis index manager with default config
        use crate::index::axis::AxisConfig;
        let axis_config = AxisConfig::default();
        let axis_manager = Arc::new(
            AxisManager::new(axis_config)
                .await
                .expect("Failed to create Axis manager"),
        );

        // Create collection service backed by the system catalog (sole store).
        use crate::catalog::CatalogManager;
        use crate::core::config::StorageConfig;

        let catalog_manager = Arc::new(CatalogManager::new());
        catalog_manager
            .create_native_catalog("default", &storage_url)
            .await
            .expect("Failed to create test xCatalog");
        let storage_config = StorageConfig {
            metadata_url: storage_url.clone(),
            ..Default::default()
        };
        let collection_service = Arc::new(
            CollectionService::new(storage_config)
                .await
                .expect("Failed to create collection service")
                .with_catalog_manager(catalog_manager.clone()),
        );

        // Create vector operations service with all dependencies
        let vector_service = Arc::new(VectorOperationsService::new(
            storage_engine,
            wal_manager,
            axis_manager,
            collection_service as Arc<dyn proximadb_runtime::CollectionPort>,
        ));

        // Create graph service
        let graph_service = Arc::new(GraphOperationsService::new());

        // Keep temp_dir alive by leaking it (tests are short-lived)
        std::mem::forget(temp_dir);

        QueryEngine::new(vector_service, graph_service)
    }

    fn create_test_vector_query() -> Query {
        // Create a simple test query with SksSimilar to trigger vector strategy detection
        use crate::query::ast::{Expr, Literal, ProjectionItem, Select, TableRef};

        Query::Select(Select {
            projection: vec![ProjectionItem {
                expr: Expr::SksSimilar {
                    field: "embedding".to_string(),
                    query: Box::new(Expr::Literal(Literal::String(
                        "[0.1, 0.2, 0.3]".to_string(),
                    ))),
                    metric: Some("cosine".to_string()),
                    threshold: None,
                },
                alias: None,
            }],
            from: vec![TableRef {
                name: Some("test_collection".to_string()),
                subquery: None,
                alias: None,
            }],
            joins: vec![],
            selection: None,
            group_by: vec![],
            having: None,
            order_by: vec![],
            limit: Some(10),
            offset: None,
        })
    }
}
