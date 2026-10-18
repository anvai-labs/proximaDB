//! # ProximaDB Execution Plan for DataFusion
//!
//! Implements a custom DataFusion ExecutionPlan that reads from ProximaDB splits.
//! This plan supports parallel partition scanning with filter and projection pushdown.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         PROXIMA SCAN EXEC                                   │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  ProximaScanExec                                                      │  │
//! │  │  - schema: SchemaRef                                                  │  │
//! │  │  - splits: Vec<FileSplit>                                             │  │
//! │  │  - projection: Option<Vec<usize>>                                     │  │
//! │  │  - filters: Vec<Expr>                                                 │  │
//! │  │  - limit: Option<usize>                                               │  │
//! │  │  - reader: Arc<dyn SplitReader>                                       │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                      │                                       │
//! │                                      ▼                                       │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │  SplitReader Implementations                                          │  │
//! │  │  - SstSplitReader                                                     │  │
//! │  │  - HelixSplitReader                                                   │  │
//! │  │  - ViperSplitReader (Parquet)                                         │  │
//! │  │  - NovaSplitReader                                                    │  │
//! │  │  - SwiftSplitReader                                                   │  │
//! │  │  - RaptorSplitReader                                                  │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! let scan_exec = ProximaScanExec::builder()
//!     .schema(schema)
//!     .splits(splits)
//!     .projection(Some(vec![0, 1, 2]))
//!     .reader(Arc::new(SstSplitReader::new()))
//!     .build()?;
//!
//! // Execute partition 0
//! let stream = scan_exec.execute(0, context)?;
//! ```

use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::Statistics;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;

use super::physical_filter_translate::pruning_predicates;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{Stream, StreamExt, TryStreamExt};
use tracing::debug;

use super::proxima_table_provider::EngineType;
use crate::storage::formats::FileSplit;
use proximadb_storage_common::format_splits::ScalarPredicate;

// ============================================================================
// Split Reader Trait
// ============================================================================

/// Trait for reading data from splits.
///
/// Each storage engine implements this trait to provide engine-specific
/// split reading logic with support for projection and batch sizing.
#[async_trait]
pub trait SplitReader: Send + Sync + Debug {
    /// Read a split and return a record batch stream.
    ///
    /// # Arguments
    /// * `split` - The split to read
    /// * `projection` - Optional column indices to read (None = all columns)
    /// * `batch_size` - Target number of rows per batch
    ///
    /// # Returns
    /// * A stream of RecordBatches
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream>;

    /// Read a split under additional conjunctive pruning predicates resolved at
    /// execute time (join runtime filters). Readers with sub-split pruning
    /// (Parquet page index) narrow the decode to rows that can match; the
    /// default ignores the hint. Exact filtering stays above the scan.
    async fn read_split_with_predicates(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
        runtime: &[(String, ScalarPredicate)],
    ) -> DFResult<SendableRecordBatchStream> {
        let _ = runtime;
        self.read_split(split, projection, batch_size).await
    }

    /// Get the schema for this reader.
    fn schema(&self) -> SchemaRef;

    /// Get the engine type for this reader.
    fn engine_type(&self) -> EngineType;

    /// Check if this reader supports filter pushdown.
    fn supports_filter_pushdown(&self) -> bool {
        false
    }

    /// Check if this reader supports projection pushdown.
    fn supports_projection_pushdown(&self) -> bool {
        true
    }
}

// ============================================================================
// Proxima Scan Execution Plan
// ============================================================================

/// DataFusion ExecutionPlan for scanning ProximaDB collections.
///
/// This plan divides work into partitions based on splits, allowing
/// parallel execution across multiple threads or nodes.
pub struct ProximaScanExec {
    /// Output schema (after projection)
    schema: SchemaRef,
    /// Original schema (before projection)
    #[allow(dead_code)] // reserved field for planned engine read-path wiring (unused in --lib)
    original_schema: SchemaRef,
    /// Splits to read, organized by partition
    partitions: Vec<Vec<FileSplit>>,
    /// Column projection (indices)
    projection: Option<Vec<usize>>,
    /// Pushed-down filters (as DataFusion expressions)
    filters: Vec<datafusion::logical_expr::Expr>,
    /// Row limit (if any)
    limit: Option<usize>,
    /// Engine-specific split reader
    reader: Arc<dyn SplitReader>,
    /// Batch size for reading
    batch_size: usize,
    /// Collection name for logging
    collection_name: String,
    /// Plan properties
    properties: Arc<PlanProperties>,
    /// Cached statistics
    statistics: Option<Statistics>,
    /// Absorbed join runtime filters (TD-OLAP-3): DataFusion 54's HashJoin
    /// pushes a `DynamicFilterPhysicalExpr` over the probe keys in the
    /// Post-phase filter pushdown; `execute()` snapshots it at stream-open
    /// (after the build side completed) and prunes splits before fetch.
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    /// Per-query I/O trace captured at **build** time (physical planning, in
    /// scope). `execute()` runs on DataFusion-spawned partition tasks where the
    /// `IO_TRACE` task-local is absent, so the split-pruning counters
    /// (`record_splits`) must record through this captured handle to attribute
    /// to the right query (TD-OLAP-3). `None` when built outside a scope.
    trace: Option<Arc<crate::observability::io_trace::IoTrace>>,
}

/// TD-OLAP-3 slice B: should this split be skipped under the CURRENT state of
/// the absorbed join runtime filters? `snapshot()` re-resolves each
/// `DynamicFilterPhysicalExpr` to the per-column bounds / in-list DataFusion
/// built from the HashJoin build keys; translation is conservative
/// (unrecognized shapes prune nothing) and the join above re-applies exact
/// semantics regardless.
///
/// Evaluated **per split, immediately before its read** — NOT once per stream:
/// in partitioned hash-join mode DataFusion polls build and probe
/// concurrently, so early probe splits race the build (the filter is still
/// the `lit(true)` placeholder) while later splits see the resolved filter.
/// Per-split late evaluation is exactly how `DataSourceExec` consumes dynamic
/// filters (per file open); a split evaluated too early is read, never
/// wrongly skipped.
fn split_pruned_by_runtime_filters(
    split: &FileSplit,
    dynamic_filters: &[Arc<dyn PhysicalExpr>],
) -> bool {
    for dynamic in dynamic_filters {
        if let Ok(Some(resolved)) = dynamic.snapshot() {
            for (column, predicate) in pruning_predicates(&resolved) {
                if split.can_prune_scalar(&column, &predicate) {
                    return true;
                }
            }
        }
    }
    false
}

/// The current conjunctive `(column, predicate)` view of the absorbed runtime
/// filters (unresolved placeholders contribute nothing) — handed to the reader
/// for page-level pruning inside each surviving split.
fn resolved_runtime_predicates(
    dynamic_filters: &[Arc<dyn PhysicalExpr>],
) -> Vec<(String, ScalarPredicate)> {
    let mut out = Vec::new();
    for dynamic in dynamic_filters {
        if let Ok(Some(resolved)) = dynamic.snapshot() {
            out.extend(pruning_predicates(&resolved));
        }
    }
    out
}

/// Gate for absorbing DataFusion's join runtime filters into the scan.
///
/// **Promoted to default-ON** (2026-07-06, TD-OLAP-3 evidence v6, executing the
/// promotion v5/#696 found evidence-supported). On the four TPC-DS star-join
/// queries (q3/q42/q52/q55) the runtime filter cuts the same query's `bytes_read`
/// 30–57% (fact scan collapses to the row groups the `date_dim` build side
/// selects) — measured stable across scale 0.01/0.05; the other conformance
/// queries are byte-identical and all 38 return identical result sets on both
/// gates. Correctness-neutral: the hash join still applies exact semantics, so
/// absorbing the dynamic filter is a scan-pruning hint. Opt out with
/// `PROXIMADB_DF_RUNTIME_FILTER_PRUNE=0`.
pub fn runtime_filter_prune_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PROXIMADB_DF_RUNTIME_FILTER_PRUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    })
}

/// The probe scan's wait budget for the build-side runtime filter's
/// `wait_complete()` rendezvous (ADR-056 AQE-S11). Default **1500 ms** —
/// Impala's effective 1000 ms + margin; on object-storage the wait is *more*
/// favorable than HDFS (each skipped ranged GET = round-trip + egress $, the
/// dominant term). Tunable per workload; the arrived-vs-timed-out outcome is
/// recorded into `IoTrace` so the route cost model learns whether the budget
/// pays. Set `PROXIMADB_DF_RUNTIME_FILTER_WAIT_MS` to override.
pub fn runtime_filter_wait_ms() -> u64 {
    static WAIT_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WAIT_MS.get_or_init(|| {
        std::env::var("PROXIMADB_DF_RUNTIME_FILTER_WAIT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .unwrap_or(1500)
    })
}

impl ProximaScanExec {
    /// Create a new ProximaScanExec using the builder pattern.
    pub fn builder() -> ProximaScanExecBuilder {
        ProximaScanExecBuilder::default()
    }

    /// Create a simple scan exec with minimal configuration.
    // build() cannot fail here: the builder has every required field set, so the
    // Result is infallible by construction (keeping `new` panic-on-bug, infallible API).
    #[allow(clippy::expect_used)]
    pub fn new(schema: SchemaRef, splits: Vec<FileSplit>, reader: Arc<dyn SplitReader>) -> Self {
        Self::builder()
            .schema(schema)
            .splits(splits)
            .reader(reader)
            .build()
            .expect("Failed to build ProximaScanExec")
    }

    /// Get the number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Get splits for a specific partition.
    pub fn partition_splits(&self, partition: usize) -> Option<&[FileSplit]> {
        self.partitions.get(partition).map(|v| v.as_slice())
    }

    /// Get total split count across all partitions.
    pub fn total_split_count(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    /// Get the engine type.
    pub fn engine_type(&self) -> EngineType {
        self.reader.engine_type()
    }

    /// Get the collection name.
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the projection.
    pub fn projection(&self) -> Option<&[usize]> {
        self.projection.as_deref()
    }

    /// Get the limit.
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Copy of this scan with absorbed runtime filters attached (TD-OLAP-3).
    fn with_dynamic_filters(&self, dynamic_filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self {
            schema: self.schema.clone(),
            original_schema: self.original_schema.clone(),
            partitions: self.partitions.clone(),
            projection: self.projection.clone(),
            filters: self.filters.clone(),
            limit: self.limit,
            reader: self.reader.clone(),
            batch_size: self.batch_size,
            collection_name: self.collection_name.clone(),
            properties: self.properties.clone(),
            statistics: self.statistics.clone(),
            dynamic_filters,
            trace: self.trace.clone(),
        }
    }

    /// Apply schema projection.
    fn project_schema(schema: &SchemaRef, projection: &Option<Vec<usize>>) -> SchemaRef {
        if let Some(proj) = projection {
            let fields: Vec<_> = proj
                .iter()
                .filter_map(|&i| schema.fields().get(i))
                .map(|f| f.as_ref().clone())
                .collect();
            Arc::new(arrow_schema::Schema::new(fields))
        } else {
            schema.clone()
        }
    }
}

impl Debug for ProximaScanExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProximaScanExec")
            .field("collection", &self.collection_name)
            .field("partitions", &self.partitions.len())
            .field("total_splits", &self.total_split_count())
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .field("engine", &self.reader.engine_type())
            .finish()
    }
}

impl DisplayAs for ProximaScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ProximaScanExec: collection={}, engine={}, partitions={}, splits={}, projection={:?}, filters={}, runtime_filters={}, limit={:?}",
                    self.collection_name,
                    self.reader.engine_type(),
                    self.partitions.len(),
                    self.total_split_count(),
                    self.projection,
                    self.filters.len(),
                    self.dynamic_filters.len(),
                    self.limit
                )
            }
        }
    }
}

impl ExecutionPlan for ProximaScanExec {
    fn name(&self) -> &str {
        "ProximaScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![] // Leaf node - no children
    }

    // DataFusion 55 split this out of `children()`: report this node's
    // physical expression roots. The absorbed TD-OLAP-3 join runtime filters
    // (`dynamic_filters`) MUST be reported: `HashJoinExec::execute` detects
    // consumers via `plan_contains_expression_id`, which walks these roots —
    // omitting them makes the join skip the dynamic filter entirely (no
    // bounds published, never marked complete) and this scan then waits out
    // its full runtime-filter budget per partition while pruning nothing.
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        datafusion::physical_plan::apply_expression_roots(&self.dynamic_filters, f)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Leaf node - no children to replace
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        debug!(
            "Executing ProximaScanExec partition {} of {} for collection '{}' ({} splits)",
            partition,
            self.partitions.len(),
            self.collection_name,
            self.partitions.get(partition).map(|p| p.len()).unwrap_or(0)
        );

        let splits = self
            .partitions
            .get(partition)
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Invalid partition {} for collection '{}' (max: {})",
                    partition,
                    self.collection_name,
                    self.partitions.len()
                ))
            })?
            .clone();

        let reader = self.reader.clone();
        let projection = self.projection.clone();
        let batch_size = self.batch_size;
        let limit = self.limit;
        let schema = self.schema.clone();
        let dynamic_filters = self.dynamic_filters.clone();
        let collection_name = self.collection_name.clone();
        let trace = self.trace.clone();

        // Drive each split's async `read_split` lazily and flatten the per-split batch
        // streams into one. Each split's future owns its `FileSplit` + an `Arc` clone of
        // the reader, so the resulting stream is `'static` and `Send`.
        //
        // Runtime-filter protocol (TD-OLAP-3, race PROVEN by evidence): the
        // probe scan is drained eagerly (RepartitionExec producer tasks), so
        // without synchronization every split races the HashJoin build and
        // sees the `lit(true)` placeholder. The stream head therefore AWAITS
        // each absorbed filter's completion signal (`wait_complete`, fired by
        // the join's `mark_complete` after the build) before any split is
        // evaluated — bounded by a timeout so no plan shape can stall a scan
        // (on timeout, splits are read unpruned: conservative). Per-split
        // evaluation below stays as defense-in-depth.
        //
        // ADR-056 AQE-S11: the budget is a configurable dial
        // (`PROXIMADB_DF_RUNTIME_FILTER_WAIT_MS`, default 1500 ms) and the
        // arrived-vs-timed-out outcome is recorded into `IoTrace` so the route
        // cost model learns per-workload whether the wait pays.
        let wait_filters = dynamic_filters.clone();
        let wait_trace = self.trace.clone();
        let wait_budget_ms = runtime_filter_wait_ms();
        let head = futures::stream::once(async move {
            for dynamic in &wait_filters {
                if let Some(df) = dynamic.downcast_ref::<DynamicFilterPhysicalExpr>() {
                    let started = std::time::Instant::now();
                    let arrived = tokio::time::timeout(
                        std::time::Duration::from_millis(wait_budget_ms),
                        df.wait_complete(),
                    )
                    .await
                    .is_ok();
                    let waited_ms = started.elapsed().as_millis() as u64;
                    match &wait_trace {
                        Some(t) => t.record_runtime_filter_wait(arrived, waited_ms),
                        None => crate::observability::io_trace::record_runtime_filter_wait(
                            arrived, waited_ms,
                        ),
                    }
                }
            }
            futures::stream::empty::<DFResult<RecordBatch>>()
        })
        .flatten();
        let empty_schema = schema.clone();
        let batches = futures::stream::iter(splits)
            .then(move |split| {
                let reader = reader.clone();
                let projection = projection.clone();
                let dynamic_filters = dynamic_filters.clone();
                let collection_name = collection_name.clone();
                let empty_schema = empty_schema.clone();
                let trace = trace.clone();
                async move {
                    if !dynamic_filters.is_empty()
                        && split_pruned_by_runtime_filters(&split, &dynamic_filters)
                    {
                        // Attribute through the captured handle: this closure runs
                        // on a spawned partition task where the task-local is absent
                        // (TD-OLAP-3). Record ONLY the dynamic skip into
                        // `splits_pruned` — every split reaching `execute` was
                        // already counted in `splits_total` by the provider
                        // `scan()` (candidates + static skips). Re-adding to the
                        // total here double-counts survivors and understates the
                        // runtime-filter skip ratio the promotion gate reads.
                        match &trace {
                            Some(t) => t.record_splits(0, 1),
                            None => crate::observability::io_trace::record_splits(0, 1),
                        }
                        debug!(
                            "Runtime filter skipped split {} of '{}' (partition {})",
                            split.split_id, collection_name, partition
                        );
                        let empty: SendableRecordBatchStream = Box::pin(
                            RecordBatchStreamAdapter::new(empty_schema, futures::stream::empty()),
                        );
                        return Ok(empty);
                    }
                    // TD-CACHE-10 / attribution closeout: rebind the captured
                    // query trace around the split read. This closure runs on
                    // a DataFusion-spawned partition task where the io_trace
                    // task-local is absent; without the rebind, every physical
                    // read inside `read_split` records NOWHERE (ambient
                    // snapshots saw 0 bytes — the measured attribution gap).
                    // `scope_with_handle` is the documented rebind seam.
                    let runtime = resolved_runtime_predicates(&dynamic_filters);
                    let read_fut = reader.read_split_with_predicates(
                        &split,
                        projection.as_deref(),
                        batch_size,
                        &runtime,
                    );
                    match &trace {
                        Some(t) => {
                            crate::observability::io_trace::scope_with_handle(t.clone(), read_fut)
                                .await
                        }
                        None => read_fut.await,
                    }
                }
            })
            .try_flatten();
        let batches = head.chain(batches);

        // Honor the row limit without over-reading: stop once `limit` rows are emitted.
        let limited = batches.scan(0usize, move |emitted, item| {
            let out = match (item, limit) {
                (Ok(batch), Some(lim)) => {
                    if *emitted >= lim {
                        None
                    } else {
                        let remaining = lim - *emitted;
                        let batch = if batch.num_rows() > remaining {
                            batch.slice(0, remaining)
                        } else {
                            batch
                        };
                        *emitted += batch.num_rows();
                        Some(Ok(batch))
                    }
                }
                (Ok(batch), None) => Some(Ok(batch)),
                (Err(e), _) => Some(Err(e)),
            };
            futures::future::ready(out)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, limited)))
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        // TD-OLAP-3 slice B: absorb ONLY runtime (dynamic) join filters, and
        // only in the Post phase where HashJoin pushes them. Static predicates
        // must stay in the FilterExec above — split pruning is inexact, so
        // claiming support for them would drop rows. Absorbing a dynamic
        // filter is safe: it is purely an optimization hint (the join still
        // applies exact semantics).
        if !matches!(phase, FilterPushdownPhase::Post) || !runtime_filter_prune_enabled() {
            return Ok(FilterPushdownPropagation::if_all(child_pushdown_result));
        }
        let mut absorbed: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
        let statuses: Vec<PushedDown> = child_pushdown_result
            .parent_filters
            .iter()
            .map(|parent| {
                if parent
                    .filter
                    .downcast_ref::<DynamicFilterPhysicalExpr>()
                    .is_some()
                {
                    absorbed.push(Arc::clone(&parent.filter));
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect();
        if absorbed.is_empty() {
            return Ok(FilterPushdownPropagation::if_all(child_pushdown_result));
        }
        let mut dynamic_filters = self.dynamic_filters.clone();
        dynamic_filters.extend(absorbed);
        let updated: Arc<dyn ExecutionPlan> = Arc::new(self.with_dynamic_filters(dynamic_filters));
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(statuses)
                .with_updated_node(updated),
        )
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        // DataFusion 54 indexes `column_statistics[col.index()]` when a parent
        // (e.g. AggregateExec) derives its own statistics. A bare
        // `Statistics::default()` carries an EMPTY `column_statistics`, so any
        // aggregate planned over this scan panics with "index out of bounds".
        // Always hand back a schema-sized Statistics: use the provided stats only
        // when their column count matches the schema, else an unknown-but-sized one.
        let schema = self.schema();
        let stats = match &self.statistics {
            Some(s) if s.column_statistics.len() == schema.fields().len() => s.clone(),
            _ => Statistics::new_unknown(&schema),
        };
        Ok(Arc::new(stats))
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for ProximaScanExec.
#[derive(Default)]
pub struct ProximaScanExecBuilder {
    schema: Option<SchemaRef>,
    splits: Option<Vec<FileSplit>>,
    projection: Option<Vec<usize>>,
    filters: Vec<datafusion::logical_expr::Expr>,
    limit: Option<usize>,
    reader: Option<Arc<dyn SplitReader>>,
    batch_size: usize,
    collection_name: String,
    target_partitions: usize,
    statistics: Option<Statistics>,
}

impl ProximaScanExecBuilder {
    /// Set the output schema.
    pub fn schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the splits to read.
    pub fn splits(mut self, splits: Vec<FileSplit>) -> Self {
        self.splits = Some(splits);
        self
    }

    /// Set column projection.
    pub fn projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Set pushed-down filters.
    pub fn filters(mut self, filters: Vec<datafusion::logical_expr::Expr>) -> Self {
        self.filters = filters;
        self
    }

    /// Set row limit.
    pub fn limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Set the split reader.
    pub fn reader(mut self, reader: Arc<dyn SplitReader>) -> Self {
        self.reader = Some(reader);
        self
    }

    /// Set batch size for reading.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set collection name for logging.
    pub fn collection_name(mut self, name: String) -> Self {
        self.collection_name = name;
        self
    }

    /// Set target number of partitions.
    pub fn target_partitions(mut self, partitions: usize) -> Self {
        self.target_partitions = partitions;
        self
    }

    /// Set statistics.
    pub fn statistics(mut self, stats: Statistics) -> Self {
        self.statistics = Some(stats);
        self
    }

    /// Build the ProximaScanExec.
    pub fn build(self) -> DFResult<ProximaScanExec> {
        let schema = self.schema.ok_or_else(|| {
            DataFusionError::Plan("Schema is required for ProximaScanExec".to_string())
        })?;

        let reader = self.reader.ok_or_else(|| {
            DataFusionError::Plan("Reader is required for ProximaScanExec".to_string())
        })?;

        let splits = self.splits.unwrap_or_default();
        let batch_size = if self.batch_size > 0 {
            self.batch_size
        } else {
            8192
        };
        let target_partitions = if self.target_partitions > 0 {
            self.target_partitions
        } else {
            num_cpus::get()
        };

        // Partition splits for parallel execution
        let partitions = partition_splits(splits, target_partitions);

        // Apply projection to schema
        let projected_schema = ProximaScanExec::project_schema(&schema, &self.projection);

        // Create plan properties
        let partitioning = Partitioning::UnknownPartitioning(partitions.len().max(1));
        let eq_properties = EquivalenceProperties::new(projected_schema.clone());
        let properties = PlanProperties::new(
            eq_properties,
            partitioning,
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );

        Ok(ProximaScanExec {
            schema: projected_schema,
            original_schema: schema,
            partitions,
            projection: self.projection,
            filters: self.filters,
            limit: self.limit,
            reader,
            batch_size,
            collection_name: self.collection_name,
            properties: Arc::new(properties),
            statistics: self.statistics,
            dynamic_filters: Vec::new(),
            // Capture the active query trace now (physical planning runs in the
            // query scope) so `execute()` — driven on spawned partition tasks —
            // can still attribute its split-pruning counters (TD-OLAP-3).
            trace: crate::observability::io_trace::current_handle(),
        })
    }
}

/// Partition splits for parallel execution using a greedy algorithm.
fn partition_splits(splits: Vec<FileSplit>, target_partitions: usize) -> Vec<Vec<FileSplit>> {
    if splits.is_empty() {
        return vec![vec![]];
    }

    if target_partitions <= 1 {
        return vec![splits];
    }

    let mut partitions: Vec<Vec<FileSplit>> = vec![vec![]; target_partitions];
    let mut partition_costs: Vec<u64> = vec![0; target_partitions];

    // Sort splits by cost (descending) for better load balancing
    let mut sorted_splits = splits;
    sorted_splits.sort_by_key(|s| std::cmp::Reverse(s.estimated_cost()));

    // Greedy assignment to partition with lowest cost
    for split in sorted_splits {
        let cost = split.estimated_cost();
        let min_idx = partition_costs
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        partitions[min_idx].push(split);
        partition_costs[min_idx] += cost;
    }

    // Remove empty partitions
    partitions.retain(|p| !p.is_empty());

    // Ensure at least one partition
    if partitions.is_empty() {
        partitions.push(vec![]);
    }

    partitions
}

// ============================================================================
// Null Split Reader for Testing
// ============================================================================

/// Null implementation of SplitReader for testing.
#[derive(Debug)]
pub struct NullSplitReader {
    schema: SchemaRef,
    engine_type: EngineType,
}

impl NullSplitReader {
    /// Create a new null reader for testing.
    pub fn new(schema: SchemaRef, engine_type: EngineType) -> Self {
        Self {
            schema,
            engine_type,
        }
    }
}

#[async_trait]
impl SplitReader for NullSplitReader {
    async fn read_split(
        &self,
        _split: &FileSplit,
        _projection: Option<&[usize]>,
        _batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        // Return an empty stream for testing
        Ok(Box::pin(EmptyRecordBatchStream::new(self.schema.clone())))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn engine_type(&self) -> EngineType {
        self.engine_type
    }
}

/// Empty RecordBatchStream for testing.
pub struct EmptyRecordBatchStream {
    schema: SchemaRef,
}

impl EmptyRecordBatchStream {
    /// Create an empty record-batch stream that yields no rows for `schema`.
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
}

impl Stream for EmptyRecordBatchStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

impl RecordBatchStream for EmptyRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeBinary(512), false),
            Field::new("metadata", DataType::Utf8, true),
        ]))
    }

    #[test]
    fn test_partition_splits_empty() {
        let partitions = partition_splits(vec![], 4);
        assert_eq!(partitions.len(), 1);
        assert!(partitions[0].is_empty());
    }

    #[test]
    fn test_partition_splits_single() {
        let splits = vec![FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100)];
        let partitions = partition_splits(splits, 1);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].len(), 1);
    }

    #[test]
    fn test_partition_splits_balanced() {
        let splits = vec![
            FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f1.sst".to_string(), 1, 1000, 1000, 100),
            FileSplit::new_block("/f2.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f2.sst".to_string(), 1, 1000, 1000, 100),
        ];
        let partitions = partition_splits(splits, 2);
        assert_eq!(partitions.len(), 2);
        // Greedy algorithm should balance approximately
        assert!(!partitions[0].is_empty() && !partitions[1].is_empty());
    }

    #[test]
    fn test_proxima_scan_exec_builder() {
        let schema = test_schema();
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Sst));
        let splits = vec![FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100)];

        let exec = ProximaScanExec::builder()
            .schema(schema)
            .splits(splits)
            .reader(reader)
            .collection_name("test".to_string())
            .batch_size(1024)
            .target_partitions(2)
            .build()
            .unwrap();

        assert_eq!(exec.collection_name(), "test");
        assert_eq!(exec.batch_size(), 1024);
        assert_eq!(exec.engine_type(), EngineType::Sst);
        assert!(exec.partition_count() >= 1);
    }

    #[test]
    fn test_proxima_scan_exec_new() {
        let schema = test_schema();
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Viper));
        let splits = vec![FileSplit::new_row_group(
            "/f1.parquet".to_string(),
            0,
            0,
            65536,
            10000,
        )];

        let exec = ProximaScanExec::new(schema, splits, reader);

        assert_eq!(exec.engine_type(), EngineType::Viper);
        assert_eq!(exec.total_split_count(), 1);
    }

    #[test]
    fn test_proxima_scan_exec_projection() {
        let schema = test_schema();
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Nova));

        let exec = ProximaScanExec::builder()
            .schema(schema)
            .splits(vec![])
            .reader(reader)
            .projection(Some(vec![0, 2])) // Select id and metadata
            .build()
            .unwrap();

        // Projected schema should have 2 fields
        assert_eq!(exec.schema.fields().len(), 2);
        assert_eq!(exec.projection(), Some(&[0, 2][..]));
    }

    #[test]
    fn test_proxima_scan_exec_limit() {
        let schema = test_schema();
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Swift));

        let exec = ProximaScanExec::builder()
            .schema(schema)
            .splits(vec![])
            .reader(reader)
            .limit(Some(100))
            .build()
            .unwrap();

        assert_eq!(exec.limit(), Some(100));
    }

    #[test]
    fn test_proxima_scan_exec_display() {
        let schema = test_schema();
        let reader = Arc::new(NullSplitReader::new(schema.clone(), EngineType::Raptor));
        let splits = vec![
            FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f1.sst".to_string(), 1, 1000, 1000, 100),
        ];

        let exec = ProximaScanExec::builder()
            .schema(schema)
            .splits(splits)
            .reader(reader)
            .collection_name("test_collection".to_string())
            .build()
            .unwrap();

        let display = format!("{:?}", exec);
        assert!(display.contains("test_collection"));
        assert!(display.contains("ProximaScanExec"));
    }

    #[test]
    fn test_null_split_reader() {
        let schema = test_schema();
        let reader = NullSplitReader::new(schema.clone(), EngineType::Helix);

        assert_eq!(reader.engine_type(), EngineType::Helix);
        assert_eq!(reader.schema().fields().len(), 3);
        assert!(!reader.supports_filter_pushdown());
        assert!(reader.supports_projection_pushdown());
    }

    #[tokio::test]
    async fn test_null_split_reader_read() {
        let schema = test_schema();
        let reader = NullSplitReader::new(schema.clone(), EngineType::Sst);
        let split = FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100);

        let stream = reader.read_split(&split, None, 1024).await.unwrap();

        // Should return empty stream
        let schema = stream.schema();
        assert_eq!(schema.fields().len(), 3);
    }

    #[test]
    fn test_empty_record_batch_stream() {
        let schema = test_schema();
        let stream = EmptyRecordBatchStream::new(schema.clone());
        assert_eq!(stream.schema().fields().len(), 3);
    }

    /// TD-OLAP-3 slice B: an absorbed `DynamicFilterPhysicalExpr`, once the
    /// build side resolves it to key bounds, prunes non-overlapping splits at
    /// stream-open — the whole snapshot → translate → `can_prune_scalar` chain.
    #[test]
    fn runtime_filter_prunes_splits_at_execute() {
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
        use proximadb_storage_common::format_splits::{ColumnBounds, SplitStatistics};

        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));

        let split_with_bounds = |file: &str, min: i64, max: i64| {
            let mut split = FileSplit::new_row_group(file.to_string(), 0, 0, 1000, 100);
            let mut stats = SplitStatistics {
                row_count: Some(100),
                byte_size: Some(1000),
                ..Default::default()
            };
            stats.column_stats.insert(
                "x".to_string(),
                ColumnBounds {
                    min: Some(serde_json::json!(min)),
                    max: Some(serde_json::json!(max)),
                    null_count: 0,
                    distinct_count: None,
                },
            );
            split.statistics = stats;
            split
        };

        let exec = ProximaScanExec::builder()
            .schema(schema.clone())
            .splits(vec![])
            .reader(Arc::new(NullSplitReader::new(
                schema.clone(),
                EngineType::Viper,
            )))
            .collection_name("t".to_string())
            .target_partitions(1)
            .build()
            .unwrap();

        // The join build side resolved to keys in [50, 300].
        let dynamic = DynamicFilterPhysicalExpr::new(vec![col("x", &schema).unwrap()], lit(true));
        dynamic
            .update(Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    col("x", &schema).unwrap(),
                    Operator::GtEq,
                    lit(50i64),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(
                    col("x", &schema).unwrap(),
                    Operator::LtEq,
                    lit(300i64),
                )),
            )))
            .unwrap();
        let filters: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(dynamic)];
        // The exec carries the filters (absorption path); pruning itself is
        // the free fn the deferred stream invokes at first poll.
        let _exec = exec.with_dynamic_filters(filters.clone());

        let splits = vec![
            split_with_bounds("a.parquet", 1, 10),    // disjoint → pruned
            split_with_bounds("b.parquet", 100, 200), // overlaps → kept
            split_with_bounds("c.parquet", 250, 400), // overlaps → kept
        ];
        let survivors: Vec<_> = splits
            .iter()
            .filter(|s| !split_pruned_by_runtime_filters(s, &filters))
            .collect();
        assert_eq!(survivors.len(), 2, "disjoint split pruned before fetch");
        assert!(survivors.iter().all(|s| s.file_path != "a.parquet"));

        // Unresolved placeholder (`lit(true)`) prunes nothing.
        let placeholder: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(
            DynamicFilterPhysicalExpr::new(vec![col("x", &schema).unwrap()], lit(true)),
        )];
        let split = split_with_bounds("a.parquet", 1, 10);
        assert!(!split_pruned_by_runtime_filters(&split, &placeholder));
    }
}
