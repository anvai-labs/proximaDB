//! # Query Module - SQL and Vector Search Engine
//!
//! This module provides ProximaDB's sophisticated query processing engine with support
//! for both SQL queries and native vector searches. It implements intelligent query
//! optimization, predicate pushdown, and adaptive execution strategies.
//!
//! ## Role in ProximaDB Architecture
//!
//! The query engine orchestrates search execution:
//! ```text
//! Query Request (SQL or Vector)
//!           ↓
//!    Query Parser & Analyzer (sql_frontend)
//!           ↓
//!    Query Optimizer (Cost-Based)
//!           ↓
//! ┌─────────────────────────────────┐
//! │    Execution Plan Selection      │
//! ├─────────────────────────────────┤
//! │ Index │ Storage │ Compute │ Cache│
//! └─────────────────────────────────┘
//!           ↓
//!    Parallel Execution
//!           ↓
//!    Result Aggregation
//! ```
//!
//! ## Key Features
//!
//! ### 1. **SQL Support**
//! Full SQL interface for vector operations:
//! ```sql
//! -- Create collection with schema
//! CREATE COLLECTION products (
//!     id TEXT PRIMARY KEY,
//!     embedding VECTOR(384),
//!     category TEXT,
//!     price FLOAT
//! );
//!
//! -- Vector similarity search with filters
//! SELECT id, category, COSINE_DISTANCE(embedding, [0.1, 0.2, ...]) as score
//! FROM products
//! WHERE category = 'electronics' AND price < 1000
//! ORDER BY score DESC
//! LIMIT 10;
//! ```
//!
//! ### 2. **Query Optimization**
//! Cost-based optimizer with multiple strategies:
//! - **Predicate Pushdown**: Filter at storage level
//! - **Index Selection**: Choose optimal index (HNSW, IVF, etc.)
//! - **Join Reordering**: Optimize multi-collection queries
//! - **Parallel Execution**: Distribute across cores
//!
//! ### 3. **Hybrid Search**
//! Combine vector and metadata filtering:
//! - Pre-filtering: Apply metadata filters before vector search
//! - Post-filtering: Apply filters after similarity computation
//! - Adaptive: Choose strategy based on selectivity
//!
//! ### 4. **Execution Strategies**
//! Multiple execution paths for different workloads:
//! - **Index-First**: Use AXIS index for fast approximate search
//! - **Storage-First**: Scan storage with filters
//! - **Hybrid**: Combine index and storage scans
//! - **Cached**: Use cached results when available
//!
//! ## Performance Characteristics
//!
//! - **SQL Parsing**: < 1ms for typical queries
//! - **Optimization Time**: 1-5ms for plan generation
//! - **Vector Search**: < 10ms for 1M vectors (indexed)
//! - **Metadata Filtering**: 100K+ filters/sec
//! - **Result Aggregation**: Near-zero overhead with streaming
//!
//! ## Module Organization
//!
//! - **`sql_frontend/`**: SQL parsing and lowering to internal AST.
//! - **`execution/`**: Unified execution engine for executing lowered AST.
//! - **`vector_search/`**: Native vector search
//!   - `query.rs`: Search query structures
//!   - `executor.rs`: Search execution
//!   - `algorithms.rs`: Search algorithms
//!
//! - **`unified_query_optimizer/`**: Query optimization
//!   - `cost_model.rs`: Cost estimation
//!   - `statistics.rs`: Collection statistics
//!   - `rules.rs`: Optimization rules
//!   - `plan.rs`: Execution plan structures
//!
//! ## SQL Extensions for Vectors
//!
//! ProximaDB extends SQL with vector operations:
//!
//! | Function | Description | Example |
//! |----------|-------------|---------|
//! | `VECTOR(n)` | Vector column type | `embedding VECTOR(384)` |
//! | `COSINE_DISTANCE` | Cosine similarity | `COSINE_DISTANCE(v1, v2)` |
//! | `EUCLIDEAN_DISTANCE` | L2 distance | `EUCLIDEAN_DISTANCE(v1, v2)` |
//! | `DOT_PRODUCT` | Inner product | `DOT_PRODUCT(v1, v2)` |
//! | `VECTOR_DIMS` | Get dimensions | `VECTOR_DIMS(embedding)` |
//! | `VECTOR_NORM` | L2 norm | `VECTOR_NORM(embedding)` |
//!
//! ## Query Optimization Rules
//!
//! The optimizer applies these rules:
//! 1. **Filter Selectivity**: Push selective filters down
//! 2. **Index Availability**: Use indexes when beneficial
//! 3. **Data Locality**: Minimize data movement
//! 4. **Parallelism**: Distribute work across cores
//! 5. **Memory Budget**: Stay within memory limits
//!

pub mod aql; // TD-050: Agentic Query Language (RUBICON)
pub mod arrow_graph_bridge; // TD-035: Arrow bridge for graph query results
pub mod ast;
pub mod authority_context; // Shared xCatalog authority conversion for planner/EXPLAIN context
pub mod cache; // C2: Query result caching for agentic AI workloads with repetitive queries
pub mod capability; // Capability registry for query validation and API parity
pub mod columnar; // M2: Dual Columnar Execution - ColumnarReadProvider abstraction
pub mod compute_scheduler; // Course-correction §5 P0: read-side multi-engine route selection
pub mod ddl_dml; // DDL/DML execution (CREATE TABLE, INSERT, UPDATE, DELETE)
pub mod distributed; // Distributed query coordination across cluster nodes
pub mod execution; // New unified execution engine
pub mod explain;
pub mod explain_schema; // Explain schema for API parity (Issue #47, SB-17)
pub mod facade; // Unified query facade - single entry point for all queries (consolidates 5 parallel paths)
pub mod federated; // Federated multi-model query engine (cross-model joins, SQL extensions)
pub mod graph_lowering; // Shared lowering from supported graph queries into multimodal IR
pub mod graph_runtime; // Shared runtime for lowered graph-query execution and canonical row shaping
pub mod graph_subset; // Shared graph query subset for facade and federated SQL extensions
pub mod materialized_view; // A1: Materialized views for complex dashboard queries
pub mod multimodal; // MultiModalPlan v1 - Unified cross-model query execution
pub mod multimodal_executor; // Multi-model SQL executor - SqlPlan lowering + dispatch
pub mod multimodal_router; // Multi-model SQL router - DataModel detection + result envelope
pub mod multimodel_executor; // Canonical multi-model SQL executor implementation
pub mod multimodel_router; // Canonical multi-model SQL router implementation
pub mod nl; // AV-SQL (TD-048) — 3-Agent Decomposition
pub mod parsers; // Query language parsers (MongoDB, etc.)
pub mod prepared; // Prepared statements for parse-once-execute-many pattern
pub mod query_optimizer;
pub mod query_router; // Query routing (Issue #46, SB-16)
pub mod read_route; // Read-side route/explain contract for DataFusion/Ballista execution
pub mod request_scope; // The request tenant carried through the facade + federated executor
pub mod route_cost_model; // Co-design C4: trace-driven route cost model (feeds the scheduler)
pub mod unified_query_port_impl; // Root-crate implementation of UnifiedQueryPort
pub use unified_query_port_impl::UnifiedQueryPortImpl;
pub mod repair; // LLD §9 — repair controller primitives (SURE-RAG aggregator + decision)
pub mod rl_planner; // RL-based adaptive query planner
pub mod semantic_analysis;
pub mod sql_frontend;
pub mod table_write_executor;
pub mod table_write_plan;
pub mod unified; // Multi-model query engine (vector, document, graph, observability)
pub mod utils;
pub mod validator; // Plan validation for capability checking
pub mod vector_search;

// Re-export main types
pub use query_optimizer::{
    UnifiedCostWeights, UnifiedExecutionPlan as QueryPlan, UnifiedMetadataFilter as MetadataFilter,
    UnifiedOptimizerConfig, UnifiedQueryOptimizer as QueryOptimizer,
};
pub use vector_search::{SearchParameters, VectorSearchQuery, VectorSearchResult};

// Re-export AQL types - TD-050 RUBICON
pub use aql::{
    AqlFind, AqlFrom, AqlPredicate, AqlProjection, AqlQuery, AqlResult, AqlScope, AqlSource,
    AqlValue, AqlWhere, AuditContext, AuditFrame, AuditOp, AuditOutcome, AuditTrail,
    DataModel as AqlDataModel, JoinType as AqlJoinType,
};

// Re-export capability registry types
pub use capability::{Capability, CapabilityCheckError, CapabilityRegistry, CapabilitySet};
pub use validator::{PlanValidator, ValidationResult};

// Re-export federated query types
pub use federated::{
    CrossModelOptimizer, ExecutionResult as FederatedExecutionResult, FederatedExecutor,
    FederatedParser, FederatedQuery, FederatedQueryContext, FederatedQueryPlan, PlanNode,
    QueryType as FederatedQueryType,
};

// Re-export unified facade types - PREFERRED ENTRY POINT
pub use facade::{
    ExecutionMetrics,
    FacadeConfig,
    GraphQueryResult,
    GraphStrategy,
    QueryContent,
    QueryContext,
    // Protocol adapter for REST/gRPC handlers
    QueryFacadeAdapter,
    QueryParams,
    QueryRequest,
    QueryResult,
    QueryResultData,
    QueryStrategy,
    QueryType,
    ScoredRecord,
    SqlStrategy,
    UnifiedQueryFacade,
    // Real strategy implementations
    VectorSearchStrategy,
};

// Re-export columnar types - M2 Dual Columnar Execution
pub use columnar::{
    ArrowInMemoryProvider, ColumnarAccessStats, ColumnarBatchStream, ColumnarCapabilities,
    ColumnarRange, ColumnarReadProvider, ParquetRangePrunedProvider, PredicatePushdownConfig,
};

// Re-export prepared statement types - C3 Prepared Statements
pub use prepared::{
    CachedStatement, ParameterBinding, ParameterValue, PreparedStatement, PreparedStatementCache,
    PreparedStatementConfig, PreparedStatementError, PreparedStatementId,
};

// Re-export query cache types - C2 Query Result Caching
pub use cache::{
    BroadcastInvalidator, CacheInvalidator, CachedResult, ChangeOperation, InvalidationConfig,
    InvalidationEvent, InvalidationListener, InvalidationStats, InvalidationStatsSnapshot,
    QueryCacheError, QueryCacheKey, QueryCacheResult, QueryCacheStats, QueryKey, QueryResultCache,
    QueryResultCacheConfig,
};

// Re-export materialized view types - A1 Materialized Views
pub use materialized_view::{
    ColumnDef, MaterializedView, MaterializedViewConfig, MaterializedViewDefinition,
    MaterializedViewError, MaterializedViewId, MaterializedViewParser, MaterializedViewResult,
    MaterializedViewState, MaterializedViewStatement, MaterializedViewStats, RefreshContext,
    RefreshEvent, RefreshEventType, RefreshResult, RefreshScheduler, RefreshStrategy,
};

// Re-export parser types - MongoDB query language support
pub use parsers::{
    AstVisitor, MongoDBExpression, MongoDBParseResult, MongoDBParser, MongoDBPipelineStage,
    MongoDBProjection, MongoDBQuery, MongoDBVisitor, QueryParser, ToDocumentFilter, ToFilter,
};

// Re-export Cypher parser types - Graph query language support
pub use parsers::{
    CypherFunction, CypherLexer, CypherParser, CypherQueryValidator, CypherToken, CypherVisitor,
    GraphQuery, GraphQueryType, LocatedToken, ToGraphQuery, cypher_to_graph_query,
};
