use crate::query::table_write_plan::ComputeBackend;
use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use proximadb_relational_executor::{
    ExecMetrics, ExecutionContext as VolcanoExecutionContext, NodeMetric, ReaderFactory,
    build_executor, collect,
};
use proximadb_relational_planner::PhysicalPlan;
use proximadb_relational_types::{RelationalRow, RelationalSchema};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Result of a successful query execution.
///
/// `Clone` is required so the pgwire OLAP result cache
/// (`QueryResultCache<ExecutionPipelineResult>`) can retain + serve a copy of
/// a result without re-executing the query.
#[derive(Debug, Clone)]
pub struct ExecutionPipelineResult {
    /// Ordered result schema.
    pub schema: RelationalSchema,
    /// Fully materialized rows.
    pub rows: Vec<RelationalRow>,
}

impl crate::query::cache::query_result_cache::CacheableResult for ExecutionPipelineResult {
    fn estimated_size_bytes(&self) -> usize {
        // Same heuristic the cache historically used for ExecutionResult:
        // ~100 bytes per field per row + per-field schema overhead.
        let row_count = self.rows.len();
        let field_count = self.schema.columns.len();
        row_count * field_count * 100 + field_count * 64 + 256
    }
}

impl ExecutionPipelineResult {
    /// Return an error when the materialized result exceeds the configured row
    /// limit. Engines call this before returning results to avoid silent
    /// truncation at protocol boundaries.
    pub fn enforce_row_limit(self, max_rows: Option<usize>) -> Result<Self, ExecutionError> {
        if let Some(limit) = max_rows
            && self.rows.len() > limit
        {
            return Err(ExecutionError::RowLimitExceeded {
                limit,
                actual: self.rows.len(),
            });
        }
        Ok(self)
    }

    /// Convert a materialized result into the streaming contract.
    ///
    /// This preserves schema while exposing rows through the same API shape that
    /// future engine-native streams will use. It is not a memory-saving path.
    pub fn into_stream(self) -> ExecutionStreamResult {
        ExecutionStreamResult {
            schema: self.schema,
            rows: stream::iter(self.rows.into_iter().map(Ok)).boxed(),
        }
    }
}

/// Row stream returned by execution engines.
pub type ExecutionRowStream = BoxStream<'static, Result<RelationalRow, ExecutionError>>;

/// Schema-bearing stream result for execution engines.
pub struct ExecutionStreamResult {
    /// Ordered result schema.
    pub schema: RelationalSchema,
    /// Stream of rows that match `schema`.
    pub rows: ExecutionRowStream,
}

/// Typed execution error for physical query engines.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("unsupported execution backend: {0:?}")]
    UnsupportedBackend(ComputeBackend),
    #[error("engine feature is disabled: {0}")]
    FeatureDisabled(&'static str),
    #[error("execution context error: {0}")]
    Context(String),
    #[error("query planning error: {0}")]
    Planning(String),
    #[error("query execution error: {0}")]
    Execution(String),
    #[error("result conversion error: {0}")]
    ResultConversion(String),
    #[error("query execution was cancelled")]
    Cancelled,
    #[error("query result exceeded row limit {limit} with {actual} rows")]
    RowLimitExceeded { limit: usize, actual: usize },
}

/// Execution engine for one physical backend.
#[async_trait]
pub trait ExecutionEngine: Send + Sync {
    /// Execute a SQL query and return the complete result set.
    ///
    /// `sql`: The SQL query string.
    /// `context`: Engine-specific execution context (e.g. table registrations).
    async fn execute_sql(
        &self,
        sql: &str,
        context: QueryExecutionContext,
    ) -> Result<ExecutionPipelineResult, ExecutionError>;

    /// Execute a SQL query as a row stream.
    ///
    /// The default implementation materializes via [`ExecutionEngine::execute_sql`]
    /// and wraps the rows in a stream. Engines should override this when they can
    /// produce rows incrementally with real backpressure and cancellation.
    async fn execute_sql_stream(
        &self,
        sql: &str,
        context: QueryExecutionContext,
    ) -> Result<ExecutionStreamResult, ExecutionError> {
        self.execute_sql(sql, context)
            .await
            .map(|result| result.into_stream())
    }
}

/// How a physical engine reacts when a result reaches `max_rows`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowLimitMode {
    /// Fail loud with [`ExecutionError::RowLimitExceeded`] when the result would
    /// exceed `max_rows`. This is the safety-cap behavior used by internal
    /// callers that treat an oversized result as an error.
    #[default]
    Error,
    /// Silently stop after `max_rows` rows and return them. This is the
    /// PostgreSQL extended-query portal behavior (`Execute` with a row cap),
    /// where a partial result is expected and resumable.
    Truncate,
}

/// Request-scoped execution controls shared by physical engines.
#[derive(Clone, Default)]
pub struct ExecutionControls {
    /// Optional maximum number of materialized rows the caller will accept.
    pub max_rows: Option<usize>,
    /// How to react when `max_rows` is reached (error vs. truncate).
    pub row_limit_mode: RowLimitMode,
    /// Optional cooperative cancellation flag checked by engines at stable
    /// boundaries.
    pub cancellation_flag: Option<Arc<AtomicBool>>,
}

impl ExecutionControls {
    /// Fail fast when the request has been cancelled.
    pub fn check_cancelled(&self) -> Result<(), ExecutionError> {
        if self
            .cancellation_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(ExecutionError::Cancelled);
        }
        Ok(())
    }
}

/// Execution context carrying table registrations and session-scoped state.
#[derive(Default, Clone)]
pub struct QueryExecutionContext {
    /// Tables backed by Parquet files (name -> location).
    pub parquet_tables: Vec<(String, String)>,
    /// ADR-058 D5/§9.A: per-table footer-stats trust, keyed by table name
    /// (parallel to `parquet_tables`). Drives whether the DataFusion adapter may
    /// answer COUNT/MIN/MAX from the parquet FOOTER (`Trusted`) or must scan
    /// (`Untrusted` — external/federated writer's footer is not trusted). Empty
    /// ⇒ default `Trusted` per-table at registration.
    pub parquet_table_trust: std::collections::HashMap<String, proximadb_data_model::StatsTrust>,
    /// Optional vector operations service for cross-modal search.
    pub vector_ops: Option<Arc<dyn proximadb_runtime::VectorOpsPort>>,
    /// Optional graph read service for the cross-modal `graph_traverse` table function.
    pub graph_ops: Option<Arc<dyn proximadb_graph_query::service::GraphQueryReadService>>,
    /// Canonical request identity for cross-modal reads. Consumers that need only
    /// tenant scope derive it from this carrier; vector/ABAC boundaries retain the
    /// subject, stable tenant key, and authentication class too.
    pub identity: proximadb_runtime::OwnedPortIdentity,
    /// ADR-025 OLAP read-merge (relational cold path): when set, parquet-backed
    /// tables listed here are reconciled at scan time against the authoritative
    /// post-snapshot WAL delta (deletes/updates/inserts that landed after the
    /// `MATERIALIZE` snapshot), keyed by canonical `oid`. `None` ⇒ legacy bare
    /// Parquet reads (the merge is default-ON; the kill-switch
    /// `PROXIMADB_OLAP_DELTA_MERGE=0` forces this). See [`super::olap_delta_merge`].
    pub olap_delta: Option<super::olap_delta_merge::OlapDeltaConfig>,
    /// TD-USUB-2: native (non-Parquet) relational tables to register with the
    /// DataFusion destination, so SQL capability stops depending on storage format.
    /// `None` ⇒ only Parquet-backed tables are registered (the pre-TD-USUB-2
    /// behaviour). Default-OFF behind `PROXIMADB_NATIVE_TABLE_PROVIDER`.
    /// See [`super::native_table_provider`].
    pub native_tables: Option<super::native_table_provider::NativeTableConfig>,
    /// Escape hatch for SQL dialect gaps while the shared relational lowering
    /// catches up. Keep false for production routes that require one logical
    /// plane across Volcano and DataFusion.
    pub allow_engine_sql_fallback: bool,
    /// Request-scoped execution guardrails.
    pub controls: ExecutionControls,
}

/// Normalize a SQL table reference to the catalog/table lookup key used by the
/// relational execution layer: last dotted segment, unquoted, lowercased.
pub(crate) fn normalize_table_key(name: &str) -> String {
    name.rsplit('.')
        .next()
        .unwrap_or(name)
        .trim_matches('"')
        .to_ascii_lowercase()
}

/// Dispatch a SQL query to a physical backend from the query layer. Protocol
/// surfaces should call this instead of constructing concrete engines directly.
pub async fn execute_sql_with_backend(
    backend: ComputeBackend,
    sql: &str,
    context: QueryExecutionContext,
) -> Result<ExecutionPipelineResult, ExecutionError> {
    match backend {
        ComputeBackend::DataFusionLocal => {
            let engine = super::datafusion_engine::DataFusionLocalEngine;
            engine.execute_sql(sql, context).await
        }
        #[cfg(feature = "duckdb")]
        ComputeBackend::DuckDbCompat => {
            let engine = super::duckdb_engine::DuckDbLocalEngine;
            engine.execute_sql(sql, context).await
        }
        other => Err(ExecutionError::UnsupportedBackend(other)),
    }
}

/// Dispatch a SQL query to a physical backend and return rows through the
/// schema-bearing stream contract.
pub async fn execute_sql_stream_with_backend(
    backend: ComputeBackend,
    sql: &str,
    context: QueryExecutionContext,
) -> Result<ExecutionStreamResult, ExecutionError> {
    match backend {
        ComputeBackend::DataFusionLocal => {
            let engine = super::datafusion_engine::DataFusionLocalEngine;
            engine.execute_sql_stream(sql, context).await
        }
        #[cfg(feature = "duckdb")]
        ComputeBackend::DuckDbCompat => {
            let engine = super::duckdb_engine::DuckDbLocalEngine;
            engine.execute_sql_stream(sql, context).await
        }
        other => Err(ExecutionError::UnsupportedBackend(other)),
    }
}

/// Push a request-scoped row cap into the physical plan as a `Limit`, so the
/// executor stops early instead of draining the whole result and then rejecting
/// it. Returns the plan unchanged when no cap is set.
///
/// In [`RowLimitMode::Truncate`] the cap is exact (`Limit n`). In
/// [`RowLimitMode::Error`] one extra row is allowed through (`Limit n+1`) so the
/// engine can observe the overflow and raise [`ExecutionError::RowLimitExceeded`]
/// rather than silently truncating.
fn cap_plan(plan: PhysicalPlan, controls: &ExecutionControls) -> PhysicalPlan {
    let Some(max_rows) = controls.max_rows else {
        return plan;
    };
    let limit = match controls.row_limit_mode {
        RowLimitMode::Truncate => max_rows as u64,
        RowLimitMode::Error => (max_rows as u64).saturating_add(1),
    };
    PhysicalPlan::Limit {
        input: Box::new(plan),
        limit: Some(limit),
        offset: 0,
    }
}

/// Apply the row-limit mode to a fully materialized result. `cap_plan` has
/// already bounded the row count, so this only needs to turn an `Error`-mode
/// overflow into [`ExecutionError::RowLimitExceeded`]; `Truncate` passes through.
fn finalize_row_limit(
    result: ExecutionPipelineResult,
    controls: &ExecutionControls,
) -> Result<ExecutionPipelineResult, ExecutionError> {
    match controls.row_limit_mode {
        RowLimitMode::Error => result.enforce_row_limit(controls.max_rows),
        RowLimitMode::Truncate => Ok(result),
    }
}

/// Native Volcano execution engine for already-planned physical plans.
pub struct NativeVolcanoEngine;

impl NativeVolcanoEngine {
    /// Build, open, and drain a physical plan through the Rust-native Volcano
    /// executor.
    pub async fn execute_physical<F: ReaderFactory>(
        physical: PhysicalPlan,
        factory: &F,
        controls: ExecutionControls,
    ) -> Result<ExecutionPipelineResult, ExecutionError> {
        controls.check_cancelled()?;
        let physical = cap_plan(physical, &controls);
        // ADR-054 Phase 2 (TD-OLAP-10): try the native vectorized path before
        // the Volcano. Runs after `cap_plan` so a request-scoped row cap (added
        // as a trailing `Limit`) is honored by the vectorized lowering too.
        // Default-off; any decline or failure falls back to the Volcano inside
        // `try_vectorized` (the experimental path never fails a query).
        // Production ScanCtx wire (ADR-054 Phase 2.5): build the scan context
        // from the plan + the lazy-global FilesystemFactory so real PAX-backed
        // queries route through the native engine. On ANY failure → None → Volcano.
        let scan_ctx = super::native_engine::build_scan_ctx(&physical).await;
        if let Some(result) =
            super::native_engine::try_vectorized(&physical, scan_ctx.as_ref()).await?
        {
            // Shadow mode (ADR-054 §7 Phase 0.5, TD-OLAP-11 §Gate): run the
            // Volcano reference too, compare result multisets, and on divergence
            // auto-demote the shape + fail safe to the reference. Default-off.
            if super::native_shadow::native_join_shadow_enabled() {
                let mut ref_exec = build_executor(
                    physical.clone(),
                    factory,
                    &VolcanoExecutionContext::default()
                        .with_cancellation_flag(controls.cancellation_flag.clone()),
                )
                .map_err(|e| ExecutionError::Execution(format!("shadow build_executor: {e}")))?;
                ref_exec
                    .open()
                    .await
                    .map_err(|e| ExecutionError::Execution(format!("shadow open: {e}")))?;
                let ref_schema = ref_exec.schema().clone();
                let ref_rows = collect(&mut *ref_exec)
                    .await
                    .map_err(|e| ExecutionError::Execution(format!("shadow scan: {e}")))?;
                let reference = ExecutionPipelineResult {
                    schema: ref_schema,
                    rows: ref_rows,
                };
                if super::native_shadow::shadow_compare_and_record(&physical, &result, &reference)
                    .diverged
                {
                    return finalize_row_limit(reference, &controls);
                }
            }
            return finalize_row_limit(result, &controls);
        }
        // TD-EXEC-2 Slice 1 (observe-only): measure the build recursion's stack
        // high-water mark alongside the plan-lowering probe at the plan seam.
        let volcano_context = VolcanoExecutionContext::default()
            .with_cancellation_flag(controls.cancellation_flag.clone());
        let (built, build_hwm) = proximadb_relational_planner::stack_probe::probe(|| {
            build_executor(physical, factory, &volcano_context)
        });
        if build_hwm > 0 {
            crate::observability::io_trace::record_stack_hwm(build_hwm);
        }
        let mut exec =
            built.map_err(|e| ExecutionError::Execution(format!("build_executor: {e}")))?;
        controls.check_cancelled()?;
        exec.open()
            .await
            .map_err(|e| ExecutionError::Execution(format!("open: {e}")))?;
        controls.check_cancelled()?;
        let schema = exec.schema().clone();
        let rows = collect(&mut *exec)
            .await
            .map_err(|e| ExecutionError::Execution(format!("scan: {e}")))?;
        controls.check_cancelled()?;
        finalize_row_limit(ExecutionPipelineResult { schema, rows }, &controls)
    }

    /// Build and open a physical plan, then return its rows through the
    /// schema-bearing stream contract instead of materializing them up front.
    ///
    /// The executor tree returned by `build_executor` is `'static + Send` and
    /// owns its readers, so it can be moved directly into a row stream. Rows are
    /// pulled lazily via `ExecNode::next_row`, keeping resident memory bounded by
    /// one row rather than the whole result set. Cooperative cancellation is
    /// re-checked before every pull so a cancelled request stops mid-stream.
    ///
    /// `controls.max_rows` is enforced defensively here as an error on overflow,
    /// preserving the materialized path's semantics; plan-level limit pushdown and
    /// truncate-vs-error modes are layered on in a later phase.
    pub async fn execute_physical_stream<F: ReaderFactory>(
        physical: PhysicalPlan,
        factory: &F,
        controls: ExecutionControls,
    ) -> Result<ExecutionStreamResult, ExecutionError> {
        controls.check_cancelled()?;
        let physical = cap_plan(physical, &controls);
        // TD-EXEC-2 Slice 1 (observe-only): measure the build recursion's stack
        // high-water mark alongside the plan-lowering probe at the plan seam.
        let volcano_context = VolcanoExecutionContext::default()
            .with_cancellation_flag(controls.cancellation_flag.clone());
        let (built, build_hwm) = proximadb_relational_planner::stack_probe::probe(|| {
            build_executor(physical, factory, &volcano_context)
        });
        if build_hwm > 0 {
            crate::observability::io_trace::record_stack_hwm(build_hwm);
        }
        let mut exec =
            built.map_err(|e| ExecutionError::Execution(format!("build_executor: {e}")))?;
        controls.check_cancelled()?;
        exec.open()
            .await
            .map_err(|e| ExecutionError::Execution(format!("open: {e}")))?;
        let schema = exec.schema().clone();

        // State threaded through the stream: the owned executor, the request
        // controls, and the count of rows already emitted (for the limit guard).
        let rows = stream::try_unfold(
            (exec, controls, 0usize),
            |(mut exec, controls, emitted)| async move {
                controls.check_cancelled()?;
                match exec
                    .next_row()
                    .await
                    .map_err(|e| ExecutionError::Execution(format!("scan: {e}")))?
                {
                    Some(row) => {
                        // `cap_plan` pushed `Limit n` (Truncate) or `Limit n+1`
                        // (Error) into the plan. In Truncate mode the stream simply
                        // ends at n; in Error mode the (n+1)th row that the plan let
                        // through is surfaced as an overflow error here.
                        if controls.row_limit_mode == RowLimitMode::Error
                            && let Some(limit) = controls.max_rows
                            && emitted >= limit
                        {
                            return Err(ExecutionError::RowLimitExceeded {
                                limit,
                                actual: emitted + 1,
                            });
                        }
                        Ok(Some((row, (exec, controls, emitted + 1))))
                    }
                    None => Ok(None),
                }
            },
        )
        .boxed();

        Ok(ExecutionStreamResult { schema, rows })
    }

    /// Build, meter, open, and drain a physical plan through the Rust-native
    /// Volcano executor.
    ///
    /// This is used by EXPLAIN ANALYZE paths that need per-operator actuals
    /// while keeping executor construction outside protocol facades.
    pub async fn execute_physical_metered<F: ReaderFactory>(
        physical: PhysicalPlan,
        factory: &F,
        controls: ExecutionControls,
    ) -> Result<(ExecutionPipelineResult, Vec<NodeMetric>), ExecutionError> {
        let metrics = Arc::new(ExecMetrics::new());
        let result = {
            controls.check_cancelled()?;
            let mut exec = build_executor(
                physical,
                factory,
                &VolcanoExecutionContext::with_metrics(metrics.clone())
                    .with_cancellation_flag(controls.cancellation_flag.clone()),
            )
            .map_err(|e| ExecutionError::Execution(format!("build_executor: {e}")))?;
            controls.check_cancelled()?;
            exec.open()
                .await
                .map_err(|e| ExecutionError::Execution(format!("open: {e}")))?;
            controls.check_cancelled()?;
            let schema = exec.schema().clone();
            let rows = collect(&mut *exec)
                .await
                .map_err(|e| ExecutionError::Execution(format!("scan: {e}")))?;
            controls.check_cancelled()?;
            ExecutionPipelineResult { schema, rows }.enforce_row_limit(controls.max_rows)?
            // `exec` dropped here, flushing MeteredExec counters into metrics.
        };
        Ok((result, metrics.snapshot()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::TryStreamExt;
    use proximadb_data_model::{ProximaType, ProximaValue};
    use proximadb_relational_algebra::TableId;
    use proximadb_relational_executor::ExecError;
    use proximadb_relational_planner::PhysicalPlan;
    use proximadb_relational_reader::RelationalReader;
    use proximadb_relational_types::{ColumnInfo, Expr};

    struct EmptyReaderFactory;

    impl ReaderFactory for EmptyReaderFactory {
        fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError> {
            Err(ExecError::Internal(format!(
                "unexpected scan in execution adapter test: {}",
                table.name
            )))
        }
    }

    fn values_plan() -> PhysicalPlan {
        PhysicalPlan::Values {
            rows: vec![
                vec![Expr::literal(ProximaValue::Int64(1))],
                vec![Expr::literal(ProximaValue::Int64(2))],
            ],
            output_schema: RelationalSchema::new(vec![ColumnInfo::new(
                "id",
                ProximaType::Int64,
                false,
            )]),
        }
    }

    struct StaticEngine;

    #[async_trait]
    impl ExecutionEngine for StaticEngine {
        async fn execute_sql(
            &self,
            _sql: &str,
            _context: QueryExecutionContext,
        ) -> Result<ExecutionPipelineResult, ExecutionError> {
            Ok(ExecutionPipelineResult {
                schema: RelationalSchema::new(vec![ColumnInfo::new(
                    "id",
                    ProximaType::Int64,
                    false,
                )]),
                rows: vec![vec![ProximaValue::Int64(7)], vec![ProximaValue::Int64(8)]],
            })
        }
    }

    #[test]
    fn normalize_table_key_strips_qualifier_and_quotes() {
        assert_eq!(normalize_table_key("Inv"), "inv");
        assert_eq!(normalize_table_key("public.Orders"), "orders");
        assert_eq!(normalize_table_key("\"Mixed\""), "mixed");
    }

    // Watchdog-wrapped (`run_sync`): a bare `#[tokio::test]` here has no 30s bound,
    // so the multi-threaded-runtime hang (CLAUDE.md #11) rides to nextest's 120s
    // slow-timeout. These Volcano tests are pure compute (no real `.await` I/O).
    #[test]
    fn dispatcher_rejects_unimplemented_backend_with_typed_error() {
        crate::query::execution::test_runtime::run_sync(async {
            let err = execute_sql_with_backend(
                ComputeBackend::DataFusionDistributed,
                "SELECT 1",
                QueryExecutionContext::default(),
            )
            .await
            .expect_err("distributed execution is not implemented");

            assert!(matches!(
                err,
                ExecutionError::UnsupportedBackend(ComputeBackend::DataFusionDistributed)
            ));
        });
    }

    #[test]
    fn execution_controls_report_cancellation() {
        let flag = Arc::new(AtomicBool::new(true));
        let controls = ExecutionControls {
            cancellation_flag: Some(flag),
            ..Default::default()
        };

        assert!(matches!(
            controls.check_cancelled(),
            Err(ExecutionError::Cancelled)
        ));
    }

    #[test]
    fn materialized_results_fail_when_row_limit_is_exceeded() {
        let result = ExecutionPipelineResult {
            schema: RelationalSchema::new(vec![ColumnInfo::new("id", ProximaType::Int64, false)]),
            rows: vec![vec![ProximaValue::Int64(1)], vec![ProximaValue::Int64(2)]],
        };

        let err = result
            .enforce_row_limit(Some(1))
            .expect_err("row limit should reject oversized results");

        assert!(matches!(
            err,
            ExecutionError::RowLimitExceeded {
                limit: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn native_volcano_adapter_executes_values_plan() {
        crate::query::execution::test_runtime::run_sync(async {
            let result = NativeVolcanoEngine::execute_physical(
                values_plan(),
                &EmptyReaderFactory,
                ExecutionControls::default(),
            )
            .await
            .expect("values plan should execute");

            assert_eq!(result.schema.columns[0].name, "id");
            assert_eq!(
                result.rows,
                vec![vec![ProximaValue::Int64(1)], vec![ProximaValue::Int64(2)]]
            );
        });
    }

    #[test]
    fn native_volcano_metered_adapter_returns_rows_and_metrics() {
        crate::query::execution::test_runtime::run_sync(async {
            let plan = PhysicalPlan::Limit {
                input: Box::new(values_plan()),
                limit: Some(1),
                offset: 0,
            };
            let (result, metrics) = NativeVolcanoEngine::execute_physical_metered(
                plan,
                &EmptyReaderFactory,
                ExecutionControls::default(),
            )
            .await
            .expect("metered values plan should execute");

            assert_eq!(result.rows, vec![vec![ProximaValue::Int64(1)]]);
            let labels: Vec<&str> = metrics.iter().map(|m| m.label.as_str()).collect();
            assert_eq!(labels, vec!["Limit", "Values"]);
            assert_eq!(
                metrics.iter().map(|m| m.arity).collect::<Vec<_>>(),
                vec![1, 0]
            );
            assert_eq!(metrics[0].rows, 1);
        });
    }

    #[test]
    fn execution_engine_stream_default_preserves_schema_and_rows() {
        crate::query::execution::test_runtime::run_sync(async {
            let stream_result = StaticEngine
                .execute_sql_stream("SELECT id FROM t", QueryExecutionContext::default())
                .await
                .expect("default stream wrapper should execute");

            assert_eq!(stream_result.schema.columns[0].name, "id");
            let rows = stream_result
                .rows
                .try_collect::<Vec<RelationalRow>>()
                .await
                .expect("stream rows should be ok");
            assert_eq!(
                rows,
                vec![vec![ProximaValue::Int64(7)], vec![ProximaValue::Int64(8)]]
            );
        });
    }

    // Root-cause fix for CLAUDE.md #11: use `test_runtime::run_sync` (block_on)
    // instead of `#[tokio::test]` — see `test_runtime.rs` for the SOLID design.
    #[test]
    fn native_volcano_stream_yields_rows_incrementally() {
        crate::query::execution::test_runtime::run_sync(async {
            let result = NativeVolcanoEngine::execute_physical_stream(
                values_plan(),
                &EmptyReaderFactory,
                ExecutionControls::default(),
            )
            .await
            .expect("values plan should open for streaming");

            // Schema is available before draining any row.
            assert_eq!(result.schema.columns[0].name, "id");

            let mut rows = result.rows;
            let first = rows.next().await.expect("first row").expect("row ok");
            assert_eq!(first, vec![ProximaValue::Int64(1)]);
            let second = rows.next().await.expect("second row").expect("row ok");
            assert_eq!(second, vec![ProximaValue::Int64(2)]);
            assert!(rows.next().await.is_none());
        });
    }

    #[test]
    fn native_volcano_stream_honors_mid_stream_cancellation() {
        crate::query::execution::test_runtime::run_sync(async {
            let flag = Arc::new(AtomicBool::new(false));
            let controls = ExecutionControls {
                cancellation_flag: Some(flag.clone()),
                ..Default::default()
            };
            let result = NativeVolcanoEngine::execute_physical_stream(
                values_plan(),
                &EmptyReaderFactory,
                controls,
            )
            .await
            .expect("values plan should open for streaming");

            let mut rows = result.rows;
            let first = rows.next().await.expect("first row").expect("row ok");
            assert_eq!(first, vec![ProximaValue::Int64(1)]);

            // Cancel after the first row; the next pull must surface Cancelled rather
            // than continuing to drain the plan.
            flag.store(true, Ordering::Relaxed);
            let err = rows
                .next()
                .await
                .expect("a stream item")
                .expect_err("cancelled mid-stream");
            assert!(matches!(err, ExecutionError::Cancelled));
        });
    }

    #[test]
    fn native_volcano_stream_enforces_row_limit() {
        crate::query::execution::test_runtime::run_sync(async {
            let controls = ExecutionControls {
                max_rows: Some(1),
                ..Default::default()
            };
            let result = NativeVolcanoEngine::execute_physical_stream(
                values_plan(),
                &EmptyReaderFactory,
                controls,
            )
            .await
            .expect("values plan should open for streaming");

            let mut rows = result.rows;
            let first = rows.next().await.expect("first row").expect("row ok");
            assert_eq!(first, vec![ProximaValue::Int64(1)]);
            let err = rows
                .next()
                .await
                .expect("a stream item")
                .expect_err("row limit should reject the overflow row");
            assert!(matches!(
                err,
                ExecutionError::RowLimitExceeded {
                    limit: 1,
                    actual: 2
                }
            ));
        });
    }

    #[test]
    fn native_volcano_stream_truncates_without_error() {
        crate::query::execution::test_runtime::run_sync(async {
            let controls = ExecutionControls {
                max_rows: Some(1),
                row_limit_mode: RowLimitMode::Truncate,
                ..Default::default()
            };
            let result = NativeVolcanoEngine::execute_physical_stream(
                values_plan(),
                &EmptyReaderFactory,
                controls,
            )
            .await
            .expect("values plan should open for streaming");

            let mut rows = result.rows;
            let first = rows.next().await.expect("first row").expect("row ok");
            assert_eq!(first, vec![ProximaValue::Int64(1)]);
            // Truncate mode stops at the cap with no overflow error: the second row
            // of the values plan is never produced.
            assert!(rows.next().await.is_none());
        });
    }

    #[test]
    fn native_volcano_materialized_truncates_without_error() {
        crate::query::execution::test_runtime::run_sync(async {
            let controls = ExecutionControls {
                max_rows: Some(1),
                row_limit_mode: RowLimitMode::Truncate,
                ..Default::default()
            };
            let result =
                NativeVolcanoEngine::execute_physical(values_plan(), &EmptyReaderFactory, controls)
                    .await
                    .expect("truncate mode returns the capped result, not an error");

            assert_eq!(result.rows, vec![vec![ProximaValue::Int64(1)]]);
        });
    }

    #[test]
    fn native_volcano_materialized_errors_on_overflow() {
        crate::query::execution::test_runtime::run_sync(async {
            let controls = ExecutionControls {
                max_rows: Some(1),
                row_limit_mode: RowLimitMode::Error,
                ..Default::default()
            };
            let err =
                NativeVolcanoEngine::execute_physical(values_plan(), &EmptyReaderFactory, controls)
                    .await
                    .expect_err("error mode rejects an oversized result");

            assert!(matches!(
                err,
                ExecutionError::RowLimitExceeded {
                    limit: 1,
                    actual: 2
                }
            ));
        });
    }
}
