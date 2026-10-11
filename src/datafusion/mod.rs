//! # DataFusion Integration Module
//!
//! This module provides DataFusion TableProvider implementations for ProximaDB,
//! enabling SQL queries and compute engine compatibility.
//!
//! ## Architecture
//!
//! ```text
//! SQL Query / DataFrame API
//!           ↓
//! ┌─────────────────────────────────────┐
//! │      DataFusion SessionContext      │
//! │  ┌─────────────────────────────────┐│
//! │  │  ProximaDBTableProvider         ││
//! │  │  - schema inference             ││
//! │  │  - statistics                   ││
//! │  │  - predicate pushdown           ││
//! │  └─────────────────────────────────┘│
//! └─────────────────────────────────────┘
//!           ↓
//! ┌─────────────────────────────────────┐
//! │      ProximaDBScanExec              │
//! │  - Parallel partition scan          │
//! │  - Filter pushdown to storage       │
//! │  - Projection pushdown              │
//! └─────────────────────────────────────┘
//!           ↓
//! ┌─────────────────────────────────────┐
//! │      Storage Format Layer           │
//! │  - InternalFormat::read_batches()   │
//! │  - Arrow RecordBatch output         │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::datafusion::{ProximaDBTableProvider, create_session_context};
//!
//! // Create a session context with ProximaDB collections
//! let ctx = create_session_context(collection_service, format_registry).await?;
//!
//! // Register a collection as a table
//! ctx.register_table("my_vectors", ProximaDBTableProvider::new(collection)?)?;
//!
//! // Query with SQL
//! let df = ctx.sql("SELECT * FROM my_vectors WHERE category = 'science'").await?;
//! let batches = df.collect().await?;
//! ```

pub mod predicate_pushdown;
pub mod scan_executor;
pub mod schema_inference;
pub mod table_provider;

// TD-OLAP-3: resolved PhysicalExpr → split-pruning predicate translation
// (consumes DataFusion 54's join runtime filters at stream-open).
pub(crate) mod physical_filter_translate;

// New split-based table provider and execution plan modules
pub mod proxima_scan_exec;
pub mod proxima_table_provider;

// Engine-specific TableProvider adapters
pub mod engine_adapters;

// Custom scalar UDFs (e.g. Monte Carlo option pricing)
pub mod udf;

// P4: shared logical lowering — relational LogicalNode -> DataFusion LogicalPlan
pub mod logical_lowering;

// Track B (§8 moat): cross-modal source bridge — vector-search results as a
// DataFusion-joinable table so one SQL plan joins vector similarity with relational data.
pub mod cross_modal;

// TD-DOC-PUSHDOWN-1 (ADR-055 P-Pushdown): correct, storage-inclusive `documents(collection)` PAX
// pushdown provider — flushed PAX-pruned scan + unflushed WAL delta + dead-filter/tombstone merge.
pub mod document_pax_provider;

// F2: registry -> DataFusion scalar-UDF adapter. Binds engine-neutral ProximaFunctionRegistry
// kernels into DataFusion as ScalarUDFs (native builtins stay the fast path).
pub mod registry_udf;

// Re-exports for convenience
pub use table_provider::{
    // Original format-based provider
    ProximaDBTableProvider,
    ProximaDBTableProviderConfig,
    // New split-based provider
    ProximaDataFusionTable,
    ProximaDataFusionTableConfig,
    create_nova_table_provider,
    // Factory functions
    create_viper_table_provider,
};

pub use scan_executor::{PartitionInfo, ProximaDBScanExec};

pub use predicate_pushdown::{FilterPushdownResult, PushdownCapability, convert_expr_to_filter};

pub use schema_inference::{extract_statistics, infer_schema_from_collection};

// Re-export new split-based types
pub use proxima_table_provider::{
    CollectionInfo,
    // Types
    EngineType,
    // Null implementation for testing
    NullProximaTableProvider,
    // Trait
    ProximaTableProvider,
    PruningStatistics,
};

pub use proxima_scan_exec::{
    EmptyRecordBatchStream,
    // Null implementations for testing
    NullSplitReader,
    // Core execution plan
    ProximaScanExec,
    ProximaScanExecBuilder,
    // Split reader trait
    SplitReader,
};

// Re-export FileSplit from storage::formats
pub use crate::storage::formats::FileSplit;

// Re-export engine-specific adapters
pub use engine_adapters::{
    // Parquet-over-FileSystem adapter (local file:// and s3:// via the canonical trait)
    FilesystemParquetSplitReader,
    FilesystemParquetTable,
    HelixSplitReader,
    // HELIX engine adapter
    HelixTableProvider,
    ObjectStoreParquetSplitReader,
    ObjectStoreParquetTable,
    // PAX-native OLAP scan (TD-OLAP-1 slice 1, default-off; bridges PaxBlockReader → Arrow)
    PaxSplitReader,
    SstSplitReader,
    // SST engine adapter
    SstTableProvider,
    ViperSplitReader,
    // VIPER engine adapter
    ViperTableProvider,
    register_object_store_parquet_location,
    register_parquet_path,
};

// Re-export custom scalar UDFs
pub use udf::mc_price_udf;

use datafusion::prelude::*;

/// Create a DataFusion SessionContext with ProximaDB integration.
///
/// This context is pre-configured with:
/// - ProximaDB catalog provider for automatic table discovery
/// - Custom vector distance functions (cosine, euclidean, dot_product)
/// - Optimizer rules for predicate pushdown
pub fn create_session_context() -> datafusion::error::Result<SessionContext> {
    build_session_context(None, None, proximadb_runtime::OwnedPortIdentity::default())
}

/// Like [`create_session_context`] but also registers the live `vector_search` table function
/// (F4), backed by `vector_ops` — so a cross-modal
/// `... JOIN vector_search('coll', '[..]', k) v ON d.id = v.id` is expressible directly over
/// the DataFusion path. Used by the pgwire OLAP route, which owns the vector service.
pub fn create_session_context_with_vector_ops(
    vector_ops: std::sync::Arc<dyn proximadb_runtime::VectorOpsPort>,
) -> datafusion::error::Result<SessionContext> {
    build_session_context(
        Some(vector_ops),
        None,
        proximadb_runtime::OwnedPortIdentity::default(),
    )
}

/// Like [`create_session_context_with_vector_ops`] but also registers the live `graph_traverse`
/// table function (F6), backed by a graph read service — so a cross-modal
/// `... JOIN graph_traverse('g','start','edge',k) g ON d.id = g.node_id` is expressible over the
/// DataFusion path. Either port may be `None`; the pgwire OLAP route supplies both.
pub fn create_session_context_with_ports(
    vector_ops: Option<std::sync::Arc<dyn proximadb_runtime::VectorOpsPort>>,
    graph_ops: Option<std::sync::Arc<dyn proximadb_graph_query::service::GraphQueryReadService>>,
    identity: proximadb_runtime::OwnedPortIdentity,
) -> datafusion::error::Result<SessionContext> {
    build_session_context(vector_ops, graph_ops, identity)
}

/// `identity` is the canonical pgwire request carrier. `vector_search` retains
/// every field for ABAC; the other cross-modal UDTFs derive their structural
/// tenant scope from the same carrier (TD-XMODAL-6 / ADR-087).
fn build_session_context(
    vector_ops: Option<std::sync::Arc<dyn proximadb_runtime::VectorOpsPort>>,
    graph_ops: Option<std::sync::Arc<dyn proximadb_graph_query::service::GraphQueryReadService>>,
    identity: proximadb_runtime::OwnedPortIdentity,
) -> datafusion::error::Result<SessionContext> {
    let tenant = identity.tenant_id.clone();
    let mut config = SessionConfig::new()
        .with_batch_size(8192)
        .with_target_partitions(num_cpus::get());
    // TD-185a: recursive CTEs are the graph-in-SQL reachability surface
    // (graph-LDBC g10, TPC-DS `cte`). The DF default is already `true`;
    // pinning it here keeps the WITH→DataFusion-floor route explicit and
    // immune to an upstream default flip.
    config.options_mut().execution.enable_recursive_ctes = true;

    let ctx = SessionContext::new_with_config(config);

    // Register ProximaDB catalog
    // Deferred: Implement ProximaDBCatalogProvider
    // ctx.register_catalog("proximadb", Arc::new(ProximaDBCatalogProvider::new(...)));

    // Register custom scalar UDFs.
    // - mc_price: Monte Carlo European option pricing (financial compute benchmark).
    // Deferred: cosine_distance, euclidean_distance, etc.
    ctx.register_udf(udf::mc_price_udf());

    // JSON extraction UDFs — the analytical (DataFusion) counterpart of the pgwire
    // JSON operator translation, so GROUP BY / aggregate queries over JSON document
    // columns (`doc->'a'->>'b'`) work on the OLAP route, not just the legacy path.
    ctx.register_udf(udf::json_extract_udf());
    ctx.register_udf(udf::json_extract_text_udf());
    ctx.register_udf(udf::json_extract_path_text_udf());

    // F2: bind every NON-native registry scalar as a DataFusion ScalarUDF (the consolidated
    // engine-neutral functions defined once in proximadb-functions). DataFusion's own
    // vectorized builtins (UPPER/ABS/…) stay the fast path; this covers registry/custom
    // functions DataFusion lacks.
    registry_udf::register_proxima_scalars(&ctx, proximadb_functions::builtins());
    // F3b: bind every registry aggregate as a DataFusion AggregateUDF (e.g. `product`).
    registry_udf::register_proxima_aggregates(&ctx, proximadb_functions::builtins());

    // F4: the cross-modal moat — `vector_search(collection, query, k)` as a joinable table,
    // backed by the live vector service (registered only when one is supplied).
    if let Some(ops) = vector_ops {
        ctx.register_udtf(
            "vector_search",
            std::sync::Arc::new(cross_modal::VectorSearchTableFunction::with_identity(
                ops,
                identity.clone(),
            )),
        );
    }

    // F5: the cross-modal moat, timeseries slice — `timeseries_range(collection, start_ms,
    // end_ms)` as a joinable table, backed by the process time-series service (registered only
    // when it is initialised). Lets the graph/relational ⋈ timeseries join the vertical copilots
    // do client-side (fan out N series, intersect in the client) execute in one data-local
    // DataFusion plan instead.
    if let Some(ts) = crate::services::timeseries_service::timeseries_service() {
        ctx.register_udtf(
            "timeseries_range",
            std::sync::Arc::new(
                cross_modal::TimeseriesRangeTableFunction::from_service_with_tenant(
                    ts,
                    tenant.clone(),
                ),
            ),
        );
    }

    // F6: the cross-modal moat, graph slice — `graph_traverse(graph_id, start_id, edge_type,
    // max_depth)` as a joinable `(node_id, depth)` table, backed by the live graph read service
    // (registered only when one is supplied). Completes the four-way relational ⋈ vector ⋈
    // timeseries ⋈ graph cross-modal join over one DataFusion plan.
    if let Some(graph) = graph_ops {
        ctx.register_udtf(
            "graph_traverse",
            std::sync::Arc::new(cross_modal::GraphTraverseTableFunction::with_tenant(
                graph,
                tenant.clone(),
            )),
        );
    }

    // F7: the cross-modal moat, document slice — `documents(collection)` as a joinable `(id, props)`
    // table over the converged document store (ADR-055 P-DFSource), backed by the process document
    // service (registered only when it is initialised). Lets documents ⋈ vector ⋈ timeseries ⋈
    // graph ⋈ relational execute in one data-local DataFusion plan instead of client-side glue.
    if let Some(doc) = crate::services::document_service::document_service() {
        ctx.register_udtf(
            "documents",
            std::sync::Arc::new(
                cross_modal::DocumentsTableFunction::from_service_with_tenant(doc, tenant.clone()),
            ),
        );
    }

    Ok(ctx)
}

/// Create a DataFusion SessionContext with custom configuration.
pub fn create_session_context_with_config(
    config: &DataFusionConfig,
) -> datafusion::error::Result<SessionContext> {
    let session_config = SessionConfig::new()
        .with_batch_size(config.batch_size)
        .with_target_partitions(config.target_partitions);

    let ctx = SessionContext::new_with_config(session_config);

    Ok(ctx)
}

/// Timing breakdown for the Monte Carlo option-pricing benchmark.
#[derive(Debug, Clone)]
pub struct McBenchTiming {
    /// Number of priced rows returned.
    pub rows: usize,
    /// Physical-plan partition count actually executed (proves intra-node parallelism).
    pub partitions: usize,
    /// Table registration time — reads the Parquet footer/metadata through the
    /// `FileSystem` trait (the I/O-bound phase).
    pub register: std::time::Duration,
    /// Query execution time — the compute-bound phase (the `mc_price` UDF), with all
    /// partitions driven concurrently.
    pub compute: std::time::Duration,
}

/// Benchmark helper: register a Parquet options table (read through `fs`) and run the
/// `mc_price` UDF over every row at the given parallelism, returning row count plus an
/// I/O-vs-compute timing split.
///
/// Keeps DataFusion types out of downstream benchmark binaries — callers pass a filesystem
/// handle + URL and receive timings. The SQL the DataFrame API lowers to is identical.
pub async fn benchmark_mc_price_over_parquet(
    fs: std::sync::Arc<dyn crate::storage::persistence::filesystem::FileSystem>,
    url: &str,
    n_paths: usize,
    target_partitions: usize,
) -> datafusion::error::Result<McBenchTiming> {
    use std::time::Instant;

    let session_config = SessionConfig::new()
        .with_batch_size(8192)
        .with_target_partitions(target_partitions.max(1));
    let ctx = SessionContext::new_with_config(session_config);
    ctx.register_udf(udf::mc_price_udf());

    let t_io = Instant::now();
    let _table = engine_adapters::register_parquet_path(&ctx, fs, "options", url).await?;
    let register = t_io.elapsed();

    let sql = format!(
        "SELECT id, mc_price(spot, strike, vol, rate, t, is_call, {n_paths}) AS price FROM options"
    );
    // Build the physical plan and drive every partition concurrently via
    // `collect_partitioned` (each partition runs as its own task), so intra-node
    // parallelism is real and measured — `collect()` would coalesce to one stream first.
    let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
    let partitions = plan.properties().output_partitioning().partition_count();

    // Drive each partition on its OWN spawned task so the CPU-heavy `mc_price` projection
    // runs on a separate worker thread per partition — true intra-node parallelism. This is
    // the coordinator-drives-workers pattern; `collect_partitioned` would instead poll every
    // partition on one task, serializing synchronous CPU work.
    use datafusion::error::DataFusionError;
    use futures::StreamExt;

    let task_ctx = ctx.task_ctx();
    let t_compute = Instant::now();
    let mut handles = Vec::with_capacity(partitions);
    for p in 0..partitions {
        let mut stream = plan.execute(p, task_ctx.clone())?;
        handles.push(tokio::spawn(async move {
            let mut rows = 0usize;
            while let Some(batch) = stream.next().await {
                rows += batch?.num_rows();
            }
            Ok::<usize, DataFusionError>(rows)
        }));
    }
    let mut rows = 0usize;
    for h in handles {
        rows += h
            .await
            .map_err(|e| DataFusionError::Execution(format!("partition task join: {e}")))??;
    }
    let compute = t_compute.elapsed();

    Ok(McBenchTiming {
        rows,
        partitions,
        register,
        compute,
    })
}

/// Configuration for DataFusion integration.
#[derive(Debug, Clone)]
pub struct DataFusionConfig {
    /// Batch size for record batch processing
    pub batch_size: usize,
    /// Number of parallel partitions
    pub target_partitions: usize,
    /// Enable predicate pushdown
    pub enable_predicate_pushdown: bool,
    /// Enable projection pushdown
    pub enable_projection_pushdown: bool,
    /// Cache statistics for this duration (seconds)
    pub statistics_cache_ttl_seconds: u64,
}

impl Default for DataFusionConfig {
    fn default() -> Self {
        Self {
            batch_size: 8192,
            target_partitions: num_cpus::get(),
            enable_predicate_pushdown: true,
            enable_projection_pushdown: true,
            statistics_cache_ttl_seconds: 60,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the DataFusion integration module compiles and
    /// basic session context creation works with DataFusion 51.x / Arrow 57.x.
    #[test]
    fn test_create_session_context() {
        let ctx = create_session_context();
        assert!(ctx.is_ok(), "SessionContext creation should succeed");
    }

    #[test]
    fn test_create_session_context_with_custom_config() {
        let config = DataFusionConfig {
            batch_size: 4096,
            target_partitions: 2,
            enable_predicate_pushdown: true,
            enable_projection_pushdown: false,
            statistics_cache_ttl_seconds: 30,
        };
        let ctx = create_session_context_with_config(&config);
        assert!(ctx.is_ok(), "Custom SessionContext creation should succeed");
    }

    #[test]
    fn test_datafusion_config_default() {
        let config = DataFusionConfig::default();
        assert_eq!(config.batch_size, 8192);
        assert!(config.enable_predicate_pushdown);
        assert!(config.enable_projection_pushdown);
        assert_eq!(config.statistics_cache_ttl_seconds, 60);
    }
}

// TDD tests for PAX-Native OLAP Scan (TD-OLAP-1)
#[cfg(test)]
mod pax_olap_tests;
