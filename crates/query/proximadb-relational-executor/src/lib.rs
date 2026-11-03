//! Volcano-style executor for ProximaDB's relational query path
//! (ADR-019 L4 / S3). Consumes a [`PhysicalPlan`] from the planner
//! and produces a tree of [`ExecNode`]s, each pulling rows from its
//! children with an `open` → `next_row` → `close` lifecycle.
//!
//! Highlights:
//!
//! - Async throughout — every reader call may hit storage.
//! - Pure pull model. Each operator pulls from its child; the root
//!   is `next_row`'d by the caller.
//! - Eager build, streaming probe for hash join / hash aggregate.
//!   Scans, filters, projects, limits, union-all are pure pipeline.
//! - Three-valued logic in filters. NULL predicate excludes the row
//!   (Postgres semantics).
//! - PK lookup access is honoured: the planner produces
//!   `Scan { access: PkLookup }` and the executor dispatches to
//!   [`RelationalReader::lookup_pk`] instead of opening a scan.
//! - Aggregate / distinct / hash join group keys go through a
//!   wrapper that bit-packs floats and explicitly errors on
//!   nested types (`Json`, `Vector`, `Map`, `Struct`) — MVP-only.
//!   Phase 3 widens to a richer key encoding.
//!
//! Reader sourcing is done via [`ReaderFactory`]. In tests we use
//! [`VecReaderFactory`]; production callers wire it to the engine
//! registry.

use async_trait::async_trait;
use proximadb_data_model::{ProximaType, ProximaValue, TimeUnit};
use proximadb_functions::builtins;
use proximadb_relational_algebra::{
    AggregateExpr, JoinKind, JoinStrategy, NamedAggregate, NamedExpr, SetOpKind, SortKey, TableId,
};
use proximadb_relational_planner::{
    AggregateStrategy, DistinctStrategy, PhysicalPlan, ScanAccess, SortStrategy,
};
use proximadb_relational_reader::{ReadSnapshot, ReaderError, RelationalReader, ScanContext};
use proximadb_relational_types::{BinaryOp, Expr, ExprError, RelationalRow, RelationalSchema};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;

// =========================================================================
// Errors
// =========================================================================

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("reader error: {0}")]
    Reader(#[from] ReaderError),

    #[error("expression error: {0}")]
    Expr(#[from] ExprError),

    #[error("operator not yet implemented: {0}")]
    Unimplemented(&'static str),

    #[error("type mismatch during execution: {0}")]
    TypeMismatch(String),

    #[error("group-by/distinct key type not supported: {0:?}")]
    UnsupportedGroupKey(ProximaType),

    #[error("aggregate function not supported by MVP executor: {0}")]
    UnsupportedAggregate(String),

    #[error("scalar subquery returned more than one row")]
    ScalarSubqueryCardinality,

    #[error("query execution was cancelled")]
    Cancelled,

    #[error("internal executor error: {0}")]
    Internal(String),
}

// =========================================================================
// Execution context + reader factory
// =========================================================================

/// What the executor needs at runtime. Carried by every operator
/// at construction time; the snapshot is propagated unchanged to
/// every Scan in the tree.
#[derive(Clone)]
pub struct ExecutionContext {
    pub snapshot: ReadSnapshot,
    /// When set (EXPLAIN ANALYZE), `build_executor` wraps every operator in a
    /// [`MeteredExec`] that records actual rows + inclusive time. `None` on the
    /// normal execution path → no wrapping, zero overhead.
    pub metrics: Option<Arc<ExecMetrics>>,
    /// Cooperative request cancellation, checked by executor wrappers at a
    /// bounded stride inside operator pull loops.
    pub cancellation_flag: Option<Arc<AtomicBool>>,
}

impl ExecutionContext {
    pub fn new(snapshot: ReadSnapshot) -> Self {
        Self {
            snapshot,
            metrics: None,
            cancellation_flag: None,
        }
    }

    /// Context that meters every operator into `metrics` (EXPLAIN ANALYZE).
    pub fn with_metrics(metrics: Arc<ExecMetrics>) -> Self {
        Self {
            snapshot: ReadSnapshot::latest(),
            metrics: Some(metrics),
            cancellation_flag: None,
        }
    }

    pub fn with_cancellation_flag(mut self, flag: Option<Arc<AtomicBool>>) -> Self {
        self.cancellation_flag = flag;
        self
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new(ReadSnapshot::latest())
    }
}

// =========================================================================
// Per-operator execution metrics (EXPLAIN ANALYZE)
// =========================================================================

/// Actuals recorded for one physical-plan operator during EXPLAIN ANALYZE.
#[derive(Debug, Clone)]
pub struct NodeMetric {
    /// Operator keyword (matches the planner's `explain_physical` line order).
    pub label: String,
    /// Number of DIRECT child operators. With the pre-order ordering this lets a
    /// consumer reconstruct the tree (and derive self-time = inclusive − children).
    pub arity: usize,
    /// Rows this operator emitted.
    pub rows: u64,
    /// Wall-clock nanoseconds spent in this operator's `next_row`, INCLUSIVE of
    /// children (pull model) — matches Postgres "actual time".
    pub elapsed_ns: u128,
}

/// Collector shared by every [`MeteredExec`] in a metered run. Operators are
/// allocated a pre-order id at build time (`alloc`, parent before children) so the
/// metric vector aligns index-for-index with `explain_physical`'s line order.
#[derive(Default)]
pub struct ExecMetrics {
    nodes: Mutex<Vec<NodeMetric>>,
}

impl ExecMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an operator and return its pre-order id (its slot index).
    fn alloc(&self, label: &str, arity: usize) -> usize {
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.push(NodeMetric {
            label: label.to_string(),
            arity,
            rows: 0,
            elapsed_ns: 0,
        });
        nodes.len() - 1
    }

    /// Record the final counters for operator `idx` (called from `MeteredExec::drop`).
    fn record(&self, idx: usize, rows: u64, elapsed_ns: u128) {
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = nodes.get_mut(idx) {
            slot.rows = rows;
            slot.elapsed_ns = elapsed_ns;
        }
    }

    /// Per-operator metrics in pre-order. Call after the executor tree is dropped
    /// (each [`MeteredExec`] flushes its counters on `Drop`).
    pub fn snapshot(&self) -> Vec<NodeMetric> {
        self.nodes.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Self (exclusive) time per operator = inclusive − sum of direct children's
/// inclusive time. Reconstructs the tree from the pre-order `metrics` + each node's
/// [`NodeMetric::arity`] (a node's `arity` subtrees follow it in pre-order), so the
/// result is index-aligned with `metrics` (and with `explain_physical`'s lines).
/// `saturating_sub` guards against sub-nanosecond timing noise. This is the headline
/// "which operator is actually slow" — children's time is attributed to them, not the
/// parent. Returns inclusive time unchanged for malformed input (no panic).
pub fn self_times(metrics: &[NodeMetric]) -> Vec<u128> {
    let mut out: Vec<u128> = metrics.iter().map(|m| m.elapsed_ns).collect();
    // Returns the index just past the subtree rooted at `pos`.
    fn walk(metrics: &[NodeMetric], pos: usize, out: &mut [u128]) -> usize {
        let mut child = pos + 1;
        let mut children_ns = 0u128;
        for _ in 0..metrics[pos].arity {
            if child >= metrics.len() {
                break; // malformed arity → leave inclusive time, don't index OOB
            }
            children_ns += metrics[child].elapsed_ns;
            child = walk(metrics, child, out);
        }
        out[pos] = metrics[pos].elapsed_ns.saturating_sub(children_ns);
        child
    }
    if !metrics.is_empty() {
        walk(metrics, 0, &mut out);
    }
    out
}

/// Direct-children input rows per operator = the sum of the operator's DIRECT
/// children's emitted rows (their `rows_out`), index-aligned with `metrics`
/// (pre-order). A leaf (arity 0) has `rows_in = 0`; a single-child operator's
/// `rows_in` equals that child's `rows_out`. Reconstructs the tree from each node's
/// [`NodeMetric::arity`] exactly like [`self_times`] (a node's `arity` subtrees
/// follow it in pre-order). `saturating_add` guards overflow; malformed arity leaves
/// the affected node at 0 (no panic, no OOB). This is the `rows_in` half of the
/// TD-TRACE-1 per-operator `exec[]` vector (rows_out is `NodeMetric::rows`).
pub fn child_input_rows(metrics: &[NodeMetric]) -> Vec<u64> {
    let mut out: Vec<u64> = vec![0; metrics.len()];
    // Returns the index just past the subtree rooted at `pos`.
    fn walk(metrics: &[NodeMetric], pos: usize, out: &mut [u64]) -> usize {
        let mut child = pos + 1;
        let mut children_rows = 0u64;
        for _ in 0..metrics[pos].arity {
            if child >= metrics.len() {
                break; // malformed arity → don't index OOB
            }
            children_rows = children_rows.saturating_add(metrics[child].rows);
            child = walk(metrics, child, out);
        }
        out[pos] = children_rows;
        child
    }
    if !metrics.is_empty() {
        walk(metrics, 0, &mut out);
    }
    out
}

/// Source for relational readers. The executor calls
/// [`open_reader`] once per `Scan` in the plan; the returned
/// reader is owned by the resulting [`ScanExec`].
pub trait ReaderFactory: Send + Sync {
    fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError>;
}

// =========================================================================
// ExecNode trait
// =========================================================================

/// Operator interface. Lifecycle: `open` → repeated `next_row` →
/// `close`. Schema is constant after construction.
#[async_trait]
pub trait ExecNode: Send {
    fn schema(&self) -> &RelationalSchema;
    async fn open(&mut self) -> Result<(), ExecError>;
    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError>;
    async fn close(&mut self) -> Result<(), ExecError>;
}

/// Helper to drain an opened executor to a `Vec`. Convenient in
/// tests and at the top of a query handler.
pub async fn collect<E: ExecNode + ?Sized>(node: &mut E) -> Result<Vec<RelationalRow>, ExecError> {
    let mut out = Vec::new();
    while let Some(row) = node.next_row().await? {
        out.push(row);
    }
    Ok(out)
}

// =========================================================================
// build_executor: PhysicalPlan → ExecNode tree
// =========================================================================

/// Build an executor tree from a physical plan. Construction is
/// sync; reader I/O happens later inside `open`.
/// Stack red-zone / growth for the executor-build recursion (TD-EXEC-2 §1.a); it
/// mirrors the plan depth exactly like the planner's lowering. Generous pending
/// Slice-1 calibration — the check is ~ns, growth fires only when near the limit.
const BUILD_RED_ZONE: usize = 256 * 1024;
const BUILD_STACK_GROW: usize = 4 * 1024 * 1024;

/// Build an executor tree from a physical plan.
///
/// Guards every recursion level with [`stacker::maybe_grow`] so a deeply nested
/// plan grows the stack on demand rather than overflowing the worker thread
/// (TD-EXEC-2 §1.a). The recursive descent goes back through this public entry, so
/// each level checks its own headroom.
pub fn build_executor<F: ReaderFactory>(
    plan: PhysicalPlan,
    factory: &F,
    ctx: &ExecutionContext,
) -> Result<Box<dyn ExecNode>, ExecError> {
    stacker::maybe_grow(BUILD_RED_ZONE, BUILD_STACK_GROW, || {
        // TD-EXEC-2 Slice 1: sample the recursion's stack low-water mark when a
        // probe is armed (a TLS read + branch otherwise — same order as the
        // maybe_grow check above).
        proximadb_relational_planner::stack_probe::note();
        build_executor_inner(plan, factory, ctx)
    })
}

fn build_executor_inner<F: ReaderFactory>(
    plan: PhysicalPlan,
    factory: &F,
    ctx: &ExecutionContext,
) -> Result<Box<dyn ExecNode>, ExecError> {
    // Allocate this operator's metric id BEFORE the match recurses into children, so
    // ids are assigned parent-first / left-to-right — exactly the pre-order the
    // planner's `explain_physical` renders, keeping metric[i] aligned with line[i].
    let metric = ctx
        .metrics
        .as_ref()
        .map(|m| (m.clone(), m.alloc(plan_label(&plan), plan_arity(&plan))));
    let node: Box<dyn ExecNode> = match plan {
        PhysicalPlan::Scan {
            table,
            output_schema,
            projection,
            predicate,
            limit,
            access,
        } => Box::new(ScanExec::new(
            factory.open_reader(&table)?,
            output_schema,
            projection,
            predicate,
            limit,
            access,
            ctx.snapshot,
        )),
        PhysicalPlan::Filter { input, predicate } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(FilterExec::new(child, predicate))
        }
        PhysicalPlan::Project { input, outputs } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(ProjectExec::new(child, outputs))
        }
        PhysicalPlan::Join {
            left,
            right,
            kind,
            on,
            strategy,
        } => {
            let l = build_executor(*left, factory, ctx)?;
            let r = build_executor(*right, factory, ctx)?;
            match strategy {
                JoinStrategy::Hash { .. } => Box::new(HashJoinExec::new(l, r, kind, on)?),
                _ => Box::new(NestedLoopJoinExec::new(l, r, kind, on)),
            }
        }
        PhysicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
            having,
            strategy,
        } => {
            let child = build_executor(*input, factory, ctx)?;
            match strategy {
                AggregateStrategy::Streaming => {
                    Box::new(StreamingAggregateExec::new(child, aggregates, having))
                }
                AggregateStrategy::Hash | AggregateStrategy::Sorted => {
                    Box::new(HashAggregateExec::new(child, group_by, aggregates, having))
                }
            }
        }
        PhysicalPlan::Sort {
            input,
            keys,
            strategy: _,
        } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(SortExec::new(child, keys))
        }
        PhysicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(LimitExec::new(child, limit, offset))
        }
        PhysicalPlan::Distinct { input, strategy: _ } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(DistinctExec::new(child))
        }
        PhysicalPlan::AssertMaxOneRow { input } => {
            let child = build_executor(*input, factory, ctx)?;
            Box::new(AssertMaxOneRowExec::new(child))
        }
        PhysicalPlan::Union { inputs, all } => {
            let mut built = Vec::with_capacity(inputs.len());
            for i in inputs {
                built.push(build_executor(i, factory, ctx)?);
            }
            Box::new(UnionExec::new(built, all))
        }
        PhysicalPlan::Values {
            rows,
            output_schema,
        } => Box::new(ValuesExec::new(rows, output_schema)),
        PhysicalPlan::SetOp {
            op,
            left,
            right,
            all,
        } => {
            let l = build_executor(*left, factory, ctx)?;
            let r = build_executor(*right, factory, ctx)?;
            Box::new(SetOpExec::new(l, r, op, all))
        }
    };
    let _ = (DistinctStrategy::Hash, SortStrategy::InMemory); // silence "imported but only matched in path"
    let node: Box<dyn ExecNode> = match metric {
        Some((metrics, idx)) => Box::new(MeteredExec::new(node, idx, metrics)),
        None => node,
    };
    Ok(match &ctx.cancellation_flag {
        Some(flag) => Box::new(CancellationExec::new(node, flag.clone())),
        None => node,
    })
}

/// Runtime-agnostic single yield: returns `Pending` once (waking itself so it
/// is re-polled immediately) then `Ready`, handing control back to the async
/// scheduler exactly like `tokio::task::yield_now` — without coupling this leaf
/// executor crate to a specific runtime (tokio is only a dev-dependency here).
struct YieldOnce(bool);

impl std::future::Future for YieldOnce {
    type Output = ();
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

/// Cooperative cancellation wrapper installed around every operator. The
/// stride bounds atomic overhead while ensuring loops inside a parent operator
/// re-enter a wrapped child and observe cancellation promptly.
struct CancellationExec {
    child: Box<dyn ExecNode>,
    flag: Arc<AtomicBool>,
    pulls: u32,
}

impl CancellationExec {
    const CHECK_MASK: u32 = 4095;

    fn new(child: Box<dyn ExecNode>, flag: Arc<AtomicBool>) -> Self {
        Self {
            child,
            flag,
            pulls: 0,
        }
    }

    /// Poll the cancellation flag on a stride boundary. Returns `Ok(true)` when
    /// this call landed on the stride (the caller then yields — see `next_row`),
    /// `Ok(false)` off-stride, `Err(Cancelled)` when the flag is set on-stride.
    fn check(&mut self) -> Result<bool, ExecError> {
        let on_stride = self.pulls & Self::CHECK_MASK == 0;
        self.pulls = self.pulls.wrapping_add(1);
        if on_stride && self.flag.load(Ordering::Relaxed) {
            Err(ExecError::Cancelled)
        } else {
            Ok(on_stride)
        }
    }
}

#[async_trait]
impl ExecNode for CancellationExec {
    fn schema(&self) -> &RelationalSchema {
        self.child.schema()
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.pulls = 0;
        self.check()?;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        if self.check()? {
            // On the stride: yield so the scheduler can run the task that SETS
            // the flag. The native operators are synchronous CPU with no real
            // `.await` points, so a grinding query holds its worker thread and
            // the out-of-band cancel handler never runs — then the poll above
            // could never observe a cancellation. Yielding makes long native
            // scans cooperative; without it, cancellation is inert end-to-end
            // (measured: an equi-join blow-up is never cancelled). This is the
            // load-bearing half of TD-EXEC-CANCEL-1.
            YieldOnce(false).await;
        }
        self.child.next_row().await
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

/// Operator keyword for a physical-plan node — matches the planner's
/// `explain_physical` line keywords (and is allocated in the same pre-order), so
/// per-operator metrics line up with the rendered plan.
fn plan_label(plan: &PhysicalPlan) -> &'static str {
    match plan {
        PhysicalPlan::Scan { .. } => "Scan",
        PhysicalPlan::Filter { .. } => "Filter",
        PhysicalPlan::Project { .. } => "Project",
        PhysicalPlan::Join { .. } => "Join",
        PhysicalPlan::Aggregate { .. } => "Aggregate",
        PhysicalPlan::Sort { .. } => "Sort",
        PhysicalPlan::Limit { .. } => "Limit",
        PhysicalPlan::Distinct { .. } => "Distinct",
        PhysicalPlan::AssertMaxOneRow { .. } => "AssertMaxOneRow",
        PhysicalPlan::Union { .. } => "Union",
        PhysicalPlan::Values { .. } => "Values",
        PhysicalPlan::SetOp { .. } => "SetOp",
    }
}

/// Number of direct child operators — must match `plan_label`'s recursion (and
/// `explain_physical`'s child order) so the pre-order metric vector reconstructs
/// the same tree.
fn plan_arity(plan: &PhysicalPlan) -> usize {
    match plan {
        PhysicalPlan::Scan { .. } | PhysicalPlan::Values { .. } => 0,
        PhysicalPlan::Filter { .. }
        | PhysicalPlan::Project { .. }
        | PhysicalPlan::Aggregate { .. }
        | PhysicalPlan::Sort { .. }
        | PhysicalPlan::Limit { .. }
        | PhysicalPlan::Distinct { .. }
        | PhysicalPlan::AssertMaxOneRow { .. } => 1,
        PhysicalPlan::Join { .. } => 2,
        PhysicalPlan::Union { inputs, .. } => inputs.len(),
        PhysicalPlan::SetOp { .. } => 2,
    }
}

// =========================================================================
// MeteredExec — transparent per-operator metrics wrapper (EXPLAIN ANALYZE)
// =========================================================================

/// Wraps a child operator, recording rows emitted and inclusive `next_row` time.
/// Counters are flushed to the shared [`ExecMetrics`] on `Drop` — the pipeline
/// drains via [`collect`] and never calls `close`, so `Drop` is the reliable point.
struct MeteredExec {
    inner: Box<dyn ExecNode>,
    idx: usize,
    metrics: Arc<ExecMetrics>,
    rows: u64,
    elapsed_ns: u128,
}

impl MeteredExec {
    fn new(inner: Box<dyn ExecNode>, idx: usize, metrics: Arc<ExecMetrics>) -> Self {
        Self {
            inner,
            idx,
            metrics,
            rows: 0,
            elapsed_ns: 0,
        }
    }
}

#[async_trait]
impl ExecNode for MeteredExec {
    fn schema(&self) -> &RelationalSchema {
        self.inner.schema()
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.inner.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        let started = Instant::now();
        let result = self.inner.next_row().await;
        self.elapsed_ns += started.elapsed().as_nanos();
        if matches!(result, Ok(Some(_))) {
            self.rows += 1;
        }
        result
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.inner.close().await
    }
}

impl Drop for MeteredExec {
    fn drop(&mut self) {
        self.metrics.record(self.idx, self.rows, self.elapsed_ns);
    }
}

// =========================================================================
// ScanExec
// =========================================================================

/// Reads a table. Dispatches between `full scan` (open + next_row)
/// and `pk lookup` (lookup_pk → at most one row).
pub struct ScanExec {
    reader: Box<dyn RelationalReader>,
    output_schema: RelationalSchema,
    projection: Option<Vec<String>>,
    predicate: Option<Expr>,
    limit: Option<u64>,
    access: ScanAccess,
    snapshot: ReadSnapshot,
    /// For PkLookup: have we already emitted the (at most) one row?
    pk_emitted: bool,
    /// PkLookup row, computed at `open`.
    pk_row: Option<RelationalRow>,
    /// For PkLookupBatch / SecondaryLookup (candidate path): the FULL
    /// rows resolved at `open`, streamed (with projection narrowing)
    /// by `next_row`. (TD-127 / TD-128.)
    buffered_rows: Vec<RelationalRow>,
    buffer_cursor: usize,
    /// For SecondaryLookup: the reader had no built index (`Ok(None)`)
    /// so `open` fell back to a full scan; `next_row` streams the
    /// reader directly (it applies projection internally) instead of
    /// the buffer. (TD-127.)
    secondary_fell_back: bool,
}

impl ScanExec {
    fn new(
        reader: Box<dyn RelationalReader>,
        output_schema: RelationalSchema,
        projection: Option<Vec<String>>,
        predicate: Option<Expr>,
        limit: Option<u64>,
        access: ScanAccess,
        snapshot: ReadSnapshot,
    ) -> Self {
        Self {
            reader,
            output_schema,
            projection,
            predicate,
            limit,
            access,
            snapshot,
            pk_emitted: false,
            pk_row: None,
            buffered_rows: Vec::new(),
            buffer_cursor: 0,
            secondary_fell_back: false,
        }
    }

    /// Narrow a FULL (unprojected) row to the scan's projection,
    /// resolving names against the reader's full schema (the reader
    /// is not `open`ed on the lookup paths, so `schema()` is the full
    /// table schema). Shared by the PkLookup / PkLookupBatch /
    /// SecondaryLookup candidate paths.
    fn project_full_row(&self, row: RelationalRow) -> Result<RelationalRow, ExecError> {
        let Some(names) = &self.projection else {
            return Ok(row);
        };
        let full = self.reader.schema();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let (idx, _) = full.column_by_name(name).ok_or_else(|| {
                ExecError::Internal(format!("projection column `{}` not in reader schema", name))
            })?;
            out.push(row[idx].clone());
        }
        Ok(out)
    }

    /// Stream the next buffered (lookup-path) row, narrowing it to the
    /// projection. `Ok(None)` at buffer EOF.
    fn next_buffered_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        if self.buffer_cursor >= self.buffered_rows.len() {
            return Ok(None);
        }
        let row = self.buffered_rows[self.buffer_cursor].clone();
        self.buffer_cursor += 1;
        Ok(Some(self.project_full_row(row)?))
    }
}

#[async_trait]
impl ExecNode for ScanExec {
    fn schema(&self) -> &RelationalSchema {
        &self.output_schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        match &self.access {
            ScanAccess::FullScan => {
                let ctx = ScanContext {
                    projection: self.projection.clone(),
                    predicate: self.predicate.clone(),
                    limit: self.limit,
                    snapshot: self.snapshot,
                };
                self.reader.open(&ctx).await?;
            }
            ScanAccess::PkLookup { key } => {
                let key_values = evaluate_key(key)?;
                self.pk_row = self.reader.lookup_pk(&key_values).await?;
                self.pk_emitted = false;
            }
            ScanAccess::PkLookupBatch { keys } => {
                // Evaluate each constant key tuple, then probe in one batch.
                let mut key_values = Vec::with_capacity(keys.len());
                for k in keys {
                    key_values.push(evaluate_key(k)?);
                }
                self.buffered_rows = self.reader.lookup_pk_batch(&key_values).await?;
                self.buffer_cursor = 0;
            }
            ScanAccess::SecondaryLookup { column, values } => {
                let probe_values = evaluate_key(values)?;
                match self.reader.lookup_secondary(column, &probe_values).await? {
                    Some(rows) => {
                        // Index answered — stream candidates (the residual
                        // Filter the planner kept re-checks each).
                        self.buffered_rows = rows;
                        self.buffer_cursor = 0;
                        self.secondary_fell_back = false;
                    }
                    None => {
                        // No built index — fall back to a full scan. The
                        // reader applies projection/predicate internally; the
                        // residual Filter above still re-checks for correctness.
                        let ctx = ScanContext {
                            projection: self.projection.clone(),
                            predicate: self.predicate.clone(),
                            limit: self.limit,
                            snapshot: self.snapshot,
                        };
                        self.reader.open(&ctx).await?;
                        self.secondary_fell_back = true;
                    }
                }
            }
        }
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        match &self.access {
            ScanAccess::FullScan => Ok(self.reader.next_row().await?),
            ScanAccess::PkLookup { .. } => {
                if self.pk_emitted {
                    return Ok(None);
                }
                self.pk_emitted = true;
                // `lookup_pk` returns the FULL row (the reader is
                // not "opened" on this path, so it has no chance to
                // apply projection internally). Narrow here if the
                // plan declares one.
                let row = match self.pk_row.take() {
                    None => return Ok(None),
                    Some(r) => r,
                };
                Ok(Some(self.project_full_row(row)?))
            }
            // TD-128: discrete batch — stream the buffered FULL rows,
            // narrowing each to the projection.
            ScanAccess::PkLookupBatch { .. } => self.next_buffered_row(),
            // TD-127: secondary lookup — either the buffered candidate
            // rows (narrowed) or, on the no-index fallback, the reader's
            // own (already-projected) full-scan stream.
            ScanAccess::SecondaryLookup { .. } => {
                if self.secondary_fell_back {
                    Ok(self.reader.next_row().await?)
                } else {
                    self.next_buffered_row()
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        Ok(self.reader.close().await?)
    }
}

/// Evaluate a vector of PK-lookup key expressions to concrete
/// values. The key expressions must not reference columns (the
/// planner guarantees this); we pass an empty row.
fn evaluate_key(keys: &[Expr]) -> Result<Vec<ProximaValue>, ExecError> {
    let empty_row: RelationalRow = Vec::new();
    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        out.push(k.eval(&empty_row, builtins())?);
    }
    Ok(out)
}

// =========================================================================
// FilterExec
// =========================================================================

pub struct FilterExec {
    child: Box<dyn ExecNode>,
    predicate: Expr,
    schema: RelationalSchema,
}

impl FilterExec {
    fn new(child: Box<dyn ExecNode>, predicate: Expr) -> Self {
        let schema = child.schema().clone();
        Self {
            child,
            predicate,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for FilterExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        loop {
            match self.child.next_row().await? {
                None => return Ok(None),
                Some(row) => {
                    let v = self.predicate.eval(&row, builtins())?;
                    // Three-valued logic: NULL and false both
                    // exclude the row; only `true` admits.
                    if matches!(v, ProximaValue::Boolean(true)) {
                        return Ok(Some(row));
                    }
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// ProjectExec
// =========================================================================

pub struct ProjectExec {
    child: Box<dyn ExecNode>,
    outputs: Vec<NamedExpr>,
    schema: RelationalSchema,
}

impl ProjectExec {
    fn new(child: Box<dyn ExecNode>, outputs: Vec<NamedExpr>) -> Self {
        let schema = RelationalSchema::new(
            outputs
                .iter()
                .map(|o| proximadb_relational_types::ColumnInfo {
                    name: o.name.clone(),
                    ty: o.expr.result_type(),
                    nullable: true,
                })
                .collect(),
        );
        Self {
            child,
            outputs,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for ProjectExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        let Some(row) = self.child.next_row().await? else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(self.outputs.len());
        for o in &self.outputs {
            out.push(o.expr.eval(&row, builtins())?);
        }
        Ok(Some(out))
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// LimitExec
// =========================================================================

pub struct LimitExec {
    child: Box<dyn ExecNode>,
    limit: Option<u64>,
    offset: u64,
    skipped: u64,
    emitted: u64,
    schema: RelationalSchema,
}

impl LimitExec {
    fn new(child: Box<dyn ExecNode>, limit: Option<u64>, offset: u64) -> Self {
        let schema = child.schema().clone();
        Self {
            child,
            limit,
            offset,
            skipped: 0,
            emitted: 0,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for LimitExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.skipped = 0;
        self.emitted = 0;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        // Skip OFFSET rows first.
        while self.skipped < self.offset {
            match self.child.next_row().await? {
                Some(_) => self.skipped += 1,
                None => return Ok(None),
            }
        }
        if let Some(l) = self.limit
            && self.emitted >= l
        {
            return Ok(None);
        }
        match self.child.next_row().await? {
            Some(row) => {
                self.emitted += 1;
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// DistinctExec (hash-based)
// =========================================================================

pub struct DistinctExec {
    child: Box<dyn ExecNode>,
    seen: HashSet<GroupKey>,
    schema: RelationalSchema,
}

impl DistinctExec {
    fn new(child: Box<dyn ExecNode>) -> Self {
        let schema = child.schema().clone();
        Self {
            child,
            seen: HashSet::new(),
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for DistinctExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.seen.clear();
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        loop {
            match self.child.next_row().await? {
                None => return Ok(None),
                Some(row) => {
                    let key = GroupKey::from_row(&row)?;
                    if self.seen.insert(key) {
                        return Ok(Some(row));
                    }
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// AssertMaxOneRowExec — scalar-subquery cardinality guard
// =========================================================================

/// Passes its child through unchanged, but errors if the child yields MORE
/// than one row. Used to enforce SQL's "a scalar subquery returns at most one
/// row" rule: the subquery plan is wrapped in this node and `LEFT JOIN`ed (ON
/// TRUE) onto the outer relation, so zero rows → NULL, one row → its value,
/// two+ rows → [`ExecError::ScalarSubqueryCardinality`].
pub struct AssertMaxOneRowExec {
    child: Box<dyn ExecNode>,
    seen: bool,
    schema: RelationalSchema,
}

impl AssertMaxOneRowExec {
    fn new(child: Box<dyn ExecNode>) -> Self {
        let schema = child.schema().clone();
        Self {
            child,
            seen: false,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for AssertMaxOneRowExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.seen = false;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        match self.child.next_row().await? {
            None => Ok(None),
            Some(row) => {
                if self.seen {
                    return Err(ExecError::ScalarSubqueryCardinality);
                }
                self.seen = true;
                Ok(Some(row))
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// UnionExec
// =========================================================================

pub struct UnionExec {
    children: Vec<Box<dyn ExecNode>>,
    cursor: usize,
    all: bool,
    seen: HashSet<GroupKey>,
    opened: Vec<bool>,
    schema: RelationalSchema,
}

impl UnionExec {
    fn new(children: Vec<Box<dyn ExecNode>>, all: bool) -> Self {
        let schema = children
            .first()
            .map(|c| c.schema().clone())
            .unwrap_or_default();
        let opened = vec![false; children.len()];
        Self {
            children,
            cursor: 0,
            all,
            seen: HashSet::new(),
            opened,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for UnionExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        // Open the first child eagerly; defer the rest until we
        // need them (matches the volcano contract).
        self.cursor = 0;
        self.seen.clear();
        if let Some(first) = self.children.first_mut() {
            first.open().await?;
            self.opened[0] = true;
        }
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        loop {
            if self.cursor >= self.children.len() {
                return Ok(None);
            }
            // Lazily open the current child.
            if !self.opened[self.cursor] {
                self.children[self.cursor].open().await?;
                self.opened[self.cursor] = true;
            }
            match self.children[self.cursor].next_row().await? {
                Some(row) => {
                    if self.all {
                        return Ok(Some(row));
                    }
                    let key = GroupKey::from_row(&row)?;
                    if self.seen.insert(key) {
                        return Ok(Some(row));
                    }
                    // Already seen — loop.
                }
                None => {
                    // Close current child and advance.
                    self.children[self.cursor].close().await?;
                    self.opened[self.cursor] = false;
                    self.cursor += 1;
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        for (i, child) in self.children.iter_mut().enumerate() {
            if self.opened[i] {
                let _ = child.close().await;
                self.opened[i] = false;
            }
        }
        Ok(())
    }
}

// =========================================================================
// SetOpExec (INTERSECT / EXCEPT)
// =========================================================================

/// Binary set operation. Materialises the RIGHT input as a per-row-key count
/// map at `open`, then streams the LEFT input deciding emission per row. Output
/// schema = left's. Set membership uses the same full-row [`GroupKey`] as
/// `DISTINCT`/`UNION`, so the same nested-type limitation applies.
///
/// Semantics:
/// - `INTERSECT`      → distinct left rows whose key is present on the right.
/// - `INTERSECT ALL`  → multiset: `min(left_count, right_count)` per key.
/// - `EXCEPT`         → distinct left rows whose key is absent on the right.
/// - `EXCEPT ALL`     → multiset: `max(0, left_count − right_count)` per key.
pub struct SetOpExec {
    left: Box<dyn ExecNode>,
    right: Box<dyn ExecNode>,
    op: SetOpKind,
    all: bool,
    schema: RelationalSchema,
    /// Per-key occurrence count on the right (built at `open`).
    right_counts: HashMap<GroupKey, u64>,
    /// Distinct cases: keys already emitted (emit each at most once).
    emitted: HashSet<GroupKey>,
    /// ALL (multiset) cases: how many left rows of each key we've seen so far.
    left_seen: HashMap<GroupKey, u64>,
}

impl SetOpExec {
    fn new(left: Box<dyn ExecNode>, right: Box<dyn ExecNode>, op: SetOpKind, all: bool) -> Self {
        let schema = left.schema().clone();
        Self {
            left,
            right,
            op,
            all,
            schema,
            right_counts: HashMap::new(),
            emitted: HashSet::new(),
            left_seen: HashMap::new(),
        }
    }
}

#[async_trait]
impl ExecNode for SetOpExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.right_counts.clear();
        self.emitted.clear();
        self.left_seen.clear();
        // Drain the RIGHT input into a per-key count map, then close it.
        self.right.open().await?;
        while let Some(row) = self.right.next_row().await? {
            let key = GroupKey::from_row(&row)?;
            *self.right_counts.entry(key).or_insert(0) += 1;
        }
        let _ = self.right.close().await;
        // Stream the LEFT input.
        self.left.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        loop {
            let Some(row) = self.left.next_row().await? else {
                return Ok(None);
            };
            let key = GroupKey::from_row(&row)?;
            let right_count = self.right_counts.get(&key).copied().unwrap_or(0);
            let emit = match (self.op, self.all) {
                // Distinct: emit each qualifying key at most once.
                (SetOpKind::Intersect, false) => right_count > 0 && self.emitted.insert(key),
                (SetOpKind::Except, false) => right_count == 0 && self.emitted.insert(key),
                // Multiset: count left occurrences and compare against the right count.
                (SetOpKind::Intersect, true) => {
                    let c = self.left_seen.entry(key).or_insert(0);
                    *c += 1;
                    *c <= right_count // emit up to min(left, right)
                }
                (SetOpKind::Except, true) => {
                    let c = self.left_seen.entry(key).or_insert(0);
                    *c += 1;
                    *c > right_count // suppress the first `right_count`, emit the rest
                }
            };
            if emit {
                return Ok(Some(row));
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        let _ = self.left.close().await;
        Ok(())
    }
}

// =========================================================================
// ValuesExec
// =========================================================================

pub struct ValuesExec {
    rows: Vec<Vec<Expr>>,
    cursor: usize,
    schema: RelationalSchema,
}

impl ValuesExec {
    fn new(rows: Vec<Vec<Expr>>, schema: RelationalSchema) -> Self {
        Self {
            rows,
            cursor: 0,
            schema,
        }
    }
}

#[async_trait]
impl ExecNode for ValuesExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.cursor = 0;
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        if self.cursor >= self.rows.len() {
            return Ok(None);
        }
        let exprs = &self.rows[self.cursor];
        self.cursor += 1;
        let empty_row: RelationalRow = Vec::new();
        let mut row = Vec::with_capacity(exprs.len());
        for e in exprs {
            row.push(e.eval(&empty_row, builtins())?);
        }
        Ok(Some(row))
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        Ok(())
    }
}

// =========================================================================
// SortExec (in-memory)
// =========================================================================

pub struct SortExec {
    child: Box<dyn ExecNode>,
    keys: Vec<SortKey>,
    buffered: Option<Vec<RelationalRow>>,
    cursor: usize,
    schema: RelationalSchema,
}

impl SortExec {
    fn new(child: Box<dyn ExecNode>, keys: Vec<SortKey>) -> Self {
        let schema = child.schema().clone();
        Self {
            child,
            keys,
            buffered: None,
            cursor: 0,
            schema,
        }
    }

    async fn ensure_buffered(&mut self) -> Result<(), ExecError> {
        if self.buffered.is_some() {
            return Ok(());
        }
        let mut rows = Vec::new();
        while let Some(row) = self.child.next_row().await? {
            rows.push(row);
        }
        // Decorate-sort-undecorate: precompute each row's sort
        // key materially. Avoids re-evaluating expressions during
        // the O(n log n) compares.
        let mut decorated: Vec<(Vec<ProximaValue>, RelationalRow)> = rows
            .into_iter()
            .map(|r| -> Result<_, ExecError> {
                let mut keys = Vec::with_capacity(self.keys.len());
                for k in &self.keys {
                    keys.push(k.expr.eval(&r, builtins())?);
                }
                Ok((keys, r))
            })
            .collect::<Result<_, _>>()?;
        let key_dirs: Vec<(bool, bool)> = self
            .keys
            .iter()
            .map(|k| (k.descending, k.nulls_first))
            .collect();
        decorated.sort_by(|a, b| compare_keys(&a.0, &b.0, &key_dirs));
        self.buffered = Some(decorated.into_iter().map(|(_, r)| r).collect());
        Ok(())
    }
}

#[async_trait]
impl ExecNode for SortExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.buffered = None;
        self.cursor = 0;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        self.ensure_buffered().await?;
        let buf = self.buffered.as_ref().expect("buffered after ensure");
        if self.cursor >= buf.len() {
            return Ok(None);
        }
        let row = buf[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(row))
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.buffered = None;
        self.child.close().await
    }
}

fn compare_keys(
    a: &[ProximaValue],
    b: &[ProximaValue],
    dirs: &[(bool, bool)],
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (i, (desc, nulls_first)) in dirs.iter().copied().enumerate() {
        let av = &a[i];
        let bv = &b[i];
        // NULL handling.
        let a_null = matches!(av, ProximaValue::Null);
        let b_null = matches!(bv, ProximaValue::Null);
        let ord = match (a_null, b_null) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, false) => compare_values(av, bv),
        };
        let ord = if desc { ord.reverse() } else { ord };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_values(a: &ProximaValue, b: &ProximaValue) -> std::cmp::Ordering {
    use ProximaValue as V;
    use std::cmp::Ordering;
    match (a, b) {
        (V::Boolean(x), V::Boolean(y)) => x.cmp(y),
        (V::Int8(x), V::Int8(y)) => x.cmp(y),
        (V::Int16(x), V::Int16(y)) => x.cmp(y),
        (V::Int32(x), V::Int32(y)) => x.cmp(y),
        (V::Int64(x), V::Int64(y)) => x.cmp(y),
        (V::UInt8(x), V::UInt8(y)) => x.cmp(y),
        (V::UInt16(x), V::UInt16(y)) => x.cmp(y),
        (V::UInt32(x), V::UInt32(y)) => x.cmp(y),
        (V::UInt64(x), V::UInt64(y)) => x.cmp(y),
        (V::Float32(x), V::Float32(y)) | (V::Float16(x), V::Float16(y)) => {
            x.partial_cmp(y).unwrap_or(Ordering::Equal)
        }
        (V::Float64(x), V::Float64(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (V::Decimal(x), V::Decimal(y)) => x.cmp(y),
        (V::String(x), V::String(y)) => x.cmp(y),
        (V::Symbol(x), V::Symbol(y)) => x.cmp(y),
        (V::Binary(x), V::Binary(y)) => x.cmp(y),
        (V::Date(x), V::Date(y)) => x.cmp(y),
        (V::Time(x, _), V::Time(y, _)) => x.cmp(y),
        (V::Timestamp(x, _), V::Timestamp(y, _)) => x.cmp(y),
        (V::TimestampTz(x, _), V::TimestampTz(y, _)) => x.cmp(y),
        (V::Uuid(x), V::Uuid(y)) => x.cmp(y),
        (V::ULID(x), V::ULID(y)) => x.cmp(y),
        // Mixed types or unsupported: treat as equal so we degrade
        // gracefully rather than panicking. Phase 3 enforces type
        // homogeneity at plan-time.
        _ => Ordering::Equal,
    }
}

// =========================================================================
// NestedLoopJoinExec
// =========================================================================

pub struct NestedLoopJoinExec {
    left: Box<dyn ExecNode>,
    right: Box<dyn ExecNode>,
    kind: JoinKind,
    on: Option<Expr>,
    schema: RelationalSchema,
    // Eager-buffer the right (build) side.
    right_buf: Option<Vec<RelationalRow>>,
    // Current left row (drives outer loop).
    current_left: Option<RelationalRow>,
    // Cursor into right_buf for the current left row.
    right_cursor: usize,
    // For LEFT/FULL outer joins: track whether any right row
    // matched the current left row, so we can emit a null-padded
    // row when nothing matched.
    current_left_matched: bool,
    // For FULL outer joins: track which right rows have been
    // matched by at least one left row.
    right_matched: Vec<bool>,
    // For FULL outer's right-only-emit phase: position into right_buf.
    right_only_cursor: usize,
    right_only_phase: bool,
}

impl NestedLoopJoinExec {
    fn new(
        left: Box<dyn ExecNode>,
        right: Box<dyn ExecNode>,
        kind: JoinKind,
        on: Option<Expr>,
    ) -> Self {
        let schema = build_join_schema(left.schema(), right.schema(), kind);
        Self {
            left,
            right,
            kind,
            on,
            schema,
            right_buf: None,
            current_left: None,
            right_cursor: 0,
            current_left_matched: false,
            right_matched: Vec::new(),
            right_only_cursor: 0,
            right_only_phase: false,
        }
    }

    async fn ensure_right_buf(&mut self) -> Result<(), ExecError> {
        if self.right_buf.is_some() {
            return Ok(());
        }
        let mut rows = Vec::new();
        while let Some(r) = self.right.next_row().await? {
            rows.push(r);
        }
        self.right_matched = vec![false; rows.len()];
        self.right_buf = Some(rows);
        Ok(())
    }

    fn predicate_passes(
        &self,
        left: &RelationalRow,
        right: &RelationalRow,
    ) -> Result<bool, ExecError> {
        let Some(pred) = &self.on else {
            return Ok(true);
        };
        let mut combined = Vec::with_capacity(left.len() + right.len());
        combined.extend_from_slice(left);
        combined.extend_from_slice(right);
        let v = pred.eval(&combined, builtins())?;
        Ok(matches!(v, ProximaValue::Boolean(true)))
    }

    /// Three-valued ON evaluation for AntiNullAware (`NOT IN`): `Some(true)` /
    /// `Some(false)` for a Boolean result, `None` for NULL/UNKNOWN (or a
    /// non-boolean result). A NULL probe key or build key makes `x = c` NULL here.
    fn predicate_eval3(
        &self,
        left: &RelationalRow,
        right: &RelationalRow,
    ) -> Result<Option<bool>, ExecError> {
        let Some(pred) = &self.on else {
            return Ok(Some(true));
        };
        let mut combined = Vec::with_capacity(left.len() + right.len());
        combined.extend_from_slice(left);
        combined.extend_from_slice(right);
        Ok(match pred.eval(&combined, builtins())? {
            ProximaValue::Boolean(b) => Some(b),
            _ => None,
        })
    }
}

#[async_trait]
impl ExecNode for NestedLoopJoinExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.left.open().await?;
        self.right.open().await?;
        self.right_buf = None;
        self.current_left = None;
        self.right_cursor = 0;
        self.current_left_matched = false;
        self.right_only_cursor = 0;
        self.right_only_phase = false;
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        self.ensure_right_buf().await?;
        let left_width = self.left.schema().len();
        let right_width = self.right.schema().len();
        // FULL outer right-only emit phase.
        if self.right_only_phase {
            let buf = self.right_buf.as_ref().expect("buf");
            while self.right_only_cursor < buf.len() {
                let idx = self.right_only_cursor;
                self.right_only_cursor += 1;
                if !self.right_matched[idx] {
                    let mut out = vec![ProximaValue::Null; left_width];
                    out.extend(buf[idx].clone());
                    return Ok(Some(out));
                }
            }
            return Ok(None);
        }
        loop {
            // Ensure we have a current left row.
            if self.current_left.is_none() {
                let Some(l) = self.left.next_row().await? else {
                    // Left exhausted. Enter the right-only phase to emit
                    // unmatched right rows null-padded on the left — for FULL
                    // AND RIGHT outer (RIGHT preserves the right side).
                    if matches!(self.kind, JoinKind::Full | JoinKind::Right) {
                        self.right_only_phase = true;
                        self.right_only_cursor = 0;
                        return self.next_row().await;
                    }
                    return Ok(None);
                };
                self.current_left = Some(l);
                self.right_cursor = 0;
                self.current_left_matched = false;
            }
            let left = self.current_left.as_ref().unwrap().clone();
            // AntiNullAware (NOT IN): SQL three-valued semantics. Emit the left row
            // iff the ON is FALSE for EVERY right row; any TRUE (match) or
            // NULL/UNKNOWN (probe key NULL, or a build key NULL → `x = c` is NULL)
            // excludes it. So `x NOT IN (S)` is empty when S has any NULL, and a NULL
            // `x` is never emitted.
            if matches!(self.kind, JoinKind::AntiNullAware) {
                let buf = self.right_buf.as_ref().expect("buf");
                let mut excluded = false;
                for right in buf {
                    match self.predicate_eval3(&left, right)? {
                        Some(true) | None => {
                            excluded = true;
                            break;
                        }
                        Some(false) => {}
                    }
                }
                self.current_left = None;
                if !excluded {
                    return Ok(Some(left));
                }
                continue;
            }
            // Inner loop over right side.
            let buf = self.right_buf.as_ref().expect("buf");
            while self.right_cursor < buf.len() {
                let idx = self.right_cursor;
                self.right_cursor += 1;
                let right = &buf[idx];
                let matched = self.predicate_passes(&left, right)?;
                if matched {
                    self.current_left_matched = true;
                    self.right_matched[idx] = true;
                    match self.kind {
                        JoinKind::Inner
                        | JoinKind::Left
                        | JoinKind::Right
                        | JoinKind::Full
                        | JoinKind::Cross => {
                            let mut out = Vec::with_capacity(left.len() + right.len());
                            out.extend_from_slice(&left);
                            out.extend_from_slice(right);
                            return Ok(Some(out));
                        }
                        JoinKind::Semi => {
                            // Emit left row once; advance to next left.
                            self.current_left = None;
                            return Ok(Some(left));
                        }
                        // AntiNullAware is fully handled by the dedicated branch
                        // above and never reaches this inner loop; grouped here only
                        // for exhaustiveness.
                        JoinKind::Anti | JoinKind::AntiNullAware => {
                            // Match found ⇒ anti excludes this left row;
                            // fast-forward to next left.
                            self.current_left = None;
                            self.right_cursor = 0;
                            break;
                        }
                    }
                }
            }
            // Right side exhausted for this left row.
            let was_matched = self.current_left_matched;
            self.current_left = None;
            match self.kind {
                JoinKind::Left | JoinKind::Full if !was_matched => {
                    let mut out = Vec::with_capacity(left.len() + right_width);
                    out.extend_from_slice(&left);
                    out.extend(std::iter::repeat_n(ProximaValue::Null, right_width));
                    return Ok(Some(out));
                }
                JoinKind::Right => {
                    // RIGHT does NOT null-extend unmatched LEFT rows (the right
                    // side is preserved). Unmatched RIGHT rows are emitted in the
                    // right-only phase after the left input is exhausted.
                }
                JoinKind::Anti if !was_matched => {
                    return Ok(Some(left));
                }
                _ => {}
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.right_buf = None;
        self.right_matched.clear();
        let _ = self.left.close().await;
        let _ = self.right.close().await;
        Ok(())
    }
}

fn build_join_schema(
    left: &RelationalSchema,
    right: &RelationalSchema,
    kind: JoinKind,
) -> RelationalSchema {
    if matches!(
        kind,
        JoinKind::Semi | JoinKind::Anti | JoinKind::AntiNullAware
    ) {
        return left.clone();
    }
    let mut cols = left.columns.clone();
    cols.extend(right.columns.clone());
    RelationalSchema::new(cols)
}

// =========================================================================
// HashJoinExec
// =========================================================================

/// Build hash side from the right input; probe from the left.
/// Only equi-join predicates that decompose into
/// `left_col = right_col` AND-chains are supported on the hash
/// path; anything else degrades to nested-loop semantics in the
/// build_executor factory (the planner already only picks Hash
/// when [`is_equi_join_predicate`] holds).
pub struct HashJoinExec {
    left: Box<dyn ExecNode>,
    right: Box<dyn ExecNode>,
    kind: JoinKind,
    on: Option<Expr>,
    schema: RelationalSchema,
    /// Index into the right (build) row, paired with the same
    /// ordinal on the left (probe) row.
    eq_pairs: Vec<(usize, usize)>,
    /// Build-side hash table: GroupKey → rows.
    build_table: Option<HashMap<GroupKey, Vec<RelationalRow>>>,
    /// Probe state: current left row + matches against build.
    probe_left: Option<RelationalRow>,
    probe_matches: Vec<RelationalRow>,
    probe_cursor: usize,
    /// RIGHT/FULL build-side drain. Hash only runs on pure-equi `on`, so a probe
    /// key-hit matches the whole build bucket → track matched-ness per GroupKey.
    /// After the probe (left) input is exhausted, unmatched build (right) rows are
    /// emitted null-padded on the left.
    matched_keys: HashSet<GroupKey>,
    drain_rows: Vec<RelationalRow>,
    drain_cursor: usize,
    drain_started: bool,
}

impl HashJoinExec {
    fn new(
        left: Box<dyn ExecNode>,
        right: Box<dyn ExecNode>,
        kind: JoinKind,
        on: Option<Expr>,
    ) -> Result<Self, ExecError> {
        // NAAJ requires SQL three-valued ON evaluation the hash-equi path can't
        // express; the planner always routes it to NestedLoop, so reaching Hash is a
        // planner bug — fail loudly rather than silently mis-handle NULLs.
        if matches!(kind, JoinKind::AntiNullAware) {
            return Err(ExecError::Internal(
                "AntiNullAware join must use NestedLoop, not Hash".into(),
            ));
        }
        let schema = build_join_schema(left.schema(), right.schema(), kind);
        let left_width = left.schema().len();
        let eq_pairs = match &on {
            Some(p) => decompose_equi_join(p, left_width).ok_or_else(|| {
                ExecError::Internal("HashJoinExec received non-equi predicate".into())
            })?,
            None => Vec::new(),
        };
        Ok(Self {
            left,
            right,
            kind,
            on,
            schema,
            eq_pairs,
            build_table: None,
            probe_left: None,
            probe_matches: Vec::new(),
            probe_cursor: 0,
            matched_keys: HashSet::new(),
            drain_rows: Vec::new(),
            drain_cursor: 0,
            drain_started: false,
        })
    }

    async fn ensure_build(&mut self) -> Result<(), ExecError> {
        if self.build_table.is_some() {
            return Ok(());
        }
        let mut table: HashMap<GroupKey, Vec<RelationalRow>> = HashMap::new();
        while let Some(r) = self.right.next_row().await? {
            // Right side keys: take the RIGHT column ordinals
            // (.1 in each pair).
            let key_vals: Vec<ProximaValue> = self
                .eq_pairs
                .iter()
                .map(|(_l, r_idx)| r[*r_idx].clone())
                .collect();
            let key = GroupKey::from_values(&key_vals)?;
            table.entry(key).or_default().push(r);
        }
        self.build_table = Some(table);
        Ok(())
    }
}

#[async_trait]
impl ExecNode for HashJoinExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.left.open().await?;
        self.right.open().await?;
        self.build_table = None;
        self.probe_left = None;
        self.probe_matches.clear();
        self.probe_cursor = 0;
        self.matched_keys.clear();
        self.drain_rows.clear();
        self.drain_cursor = 0;
        self.drain_started = false;
        Ok(())
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        self.ensure_build().await?;
        let right_width = self.right.schema().len();
        let left_width = self.left.schema().len();
        loop {
            // Drain matches for current probe row, if any.
            if self.probe_left.is_some() && self.probe_cursor < self.probe_matches.len() {
                let l = self.probe_left.as_ref().unwrap().clone();
                let r = self.probe_matches[self.probe_cursor].clone();
                self.probe_cursor += 1;
                match self.kind {
                    JoinKind::Semi => {
                        // Emit left once; advance to next probe.
                        self.probe_left = None;
                        self.probe_matches.clear();
                        self.probe_cursor = 0;
                        return Ok(Some(l));
                    }
                    JoinKind::Anti => {
                        // Should never have entered this branch
                        // for Anti — Anti emits ONLY when there
                        // were zero matches. Skip.
                        continue;
                    }
                    _ => {
                        let mut out = Vec::with_capacity(l.len() + r.len());
                        out.extend(l);
                        out.extend(r);
                        return Ok(Some(out));
                    }
                }
            }
            // Need a fresh probe row.
            let Some(l) = self.left.next_row().await? else {
                // Left (probe) exhausted. For RIGHT/FULL, drain unmatched build
                // (right) rows null-padded on the left. Hash runs only on pure-equi
                // `on`, so a matched GroupKey means its whole bucket matched.
                if matches!(self.kind, JoinKind::Right | JoinKind::Full) {
                    if !self.drain_started {
                        self.drain_started = true;
                        if let Some(table) = &self.build_table {
                            for (key, rows) in table {
                                if !self.matched_keys.contains(key) {
                                    self.drain_rows.extend(rows.iter().cloned());
                                }
                            }
                        }
                    }
                    if self.drain_cursor < self.drain_rows.len() {
                        let r = self.drain_rows[self.drain_cursor].clone();
                        self.drain_cursor += 1;
                        let mut out = vec![ProximaValue::Null; left_width];
                        out.extend(r);
                        return Ok(Some(out));
                    }
                }
                return Ok(None);
            };
            // Compute probe key from left row.
            let key_vals: Vec<ProximaValue> = self
                .eq_pairs
                .iter()
                .map(|(l_idx, _r)| l[*l_idx].clone())
                .collect();
            let key = GroupKey::from_values(&key_vals)?;
            let matches = self
                .build_table
                .as_ref()
                .and_then(|t| t.get(&key))
                .cloned()
                .unwrap_or_default();
            // For non-equi extra condition: re-evaluate the
            // full `on` predicate for each candidate to filter.
            // (Phase 3 — for MVP the planner only picks Hash on
            // pure equi predicates, so `on` matches the eq_pairs
            // exactly.)
            let _ = &self.on;
            match self.kind {
                JoinKind::Inner | JoinKind::Cross => {
                    if !matches.is_empty() {
                        self.probe_left = Some(l);
                        self.probe_matches = matches;
                        self.probe_cursor = 0;
                    }
                    // else loop and try next left row
                }
                JoinKind::Left => {
                    if matches.is_empty() {
                        let mut out = Vec::with_capacity(l.len() + right_width);
                        out.extend(l);
                        out.extend(std::iter::repeat_n(ProximaValue::Null, right_width));
                        return Ok(Some(out));
                    }
                    self.probe_left = Some(l);
                    self.probe_matches = matches;
                    self.probe_cursor = 0;
                }
                JoinKind::Right => {
                    // Mark the build bucket matched (equi → the whole key bucket
                    // matched); unmatched buckets are drained null-left after the
                    // probe input is exhausted. Unmatched LEFT rows are NOT emitted
                    // (the right side is preserved). Matched pairs emit as for INNER.
                    if !matches.is_empty() {
                        self.matched_keys.insert(key);
                        self.probe_left = Some(l);
                        self.probe_matches = matches;
                        self.probe_cursor = 0;
                    }
                }
                JoinKind::Full => {
                    // FULL = LEFT (emit unmatched probe rows null-right) + the
                    // build-side drain (emit unmatched build rows null-left) +
                    // matched-bucket tracking.
                    if matches.is_empty() {
                        let mut out = Vec::with_capacity(l.len() + right_width);
                        out.extend(l);
                        out.extend(std::iter::repeat_n(ProximaValue::Null, right_width));
                        return Ok(Some(out));
                    }
                    self.matched_keys.insert(key);
                    self.probe_left = Some(l);
                    self.probe_matches = matches;
                    self.probe_cursor = 0;
                }
                JoinKind::Semi => {
                    if !matches.is_empty() {
                        return Ok(Some(l));
                    }
                }
                // AntiNullAware never reaches HashJoin (the planner routes it to
                // NestedLoop and `new` rejects it); grouped for exhaustiveness only.
                JoinKind::Anti | JoinKind::AntiNullAware => {
                    if matches.is_empty() {
                        return Ok(Some(l));
                    }
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.build_table = None;
        let _ = self.left.close().await;
        let _ = self.right.close().await;
        Ok(())
    }
}

/// Decompose an AND-chain of `left.col = right.col` equalities
/// into `(left_ordinal, right_ordinal)` pairs. Returns `None`
/// if any conjunct isn't a column-to-column equality, or both
/// sides of an equality reference the same input.
fn decompose_equi_join(expr: &Expr, left_width: usize) -> Option<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    decompose_equi_into(expr, left_width, &mut out).then_some(out)
}

fn decompose_equi_into(expr: &Expr, left_width: usize, out: &mut Vec<(usize, usize)>) -> bool {
    match expr {
        Expr::BinaryOp {
            op: BinaryOp::And,
            left,
            right,
        } => {
            decompose_equi_into(left, left_width, out)
                && decompose_equi_into(right, left_width, out)
        }
        Expr::BinaryOp {
            op: BinaryOp::Eq,
            left,
            right,
        } => {
            let (l, r) = match (left.as_ref(), right.as_ref()) {
                (Expr::Column(a), Expr::Column(b)) => (a, b),
                _ => return false,
            };
            let l_side = l.ordinal < left_width;
            let r_side = r.ordinal < left_width;
            match (l_side, r_side) {
                (true, false) => {
                    out.push((l.ordinal, r.ordinal - left_width));
                    true
                }
                (false, true) => {
                    out.push((r.ordinal, l.ordinal - left_width));
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

// =========================================================================
// StreamingAggregateExec (no GROUP BY)
// =========================================================================

pub struct StreamingAggregateExec {
    child: Box<dyn ExecNode>,
    aggregates: Vec<NamedAggregate>,
    having: Option<Expr>,
    schema: RelationalSchema,
    emitted: bool,
}

impl StreamingAggregateExec {
    fn new(
        child: Box<dyn ExecNode>,
        aggregates: Vec<NamedAggregate>,
        having: Option<Expr>,
    ) -> Self {
        let cols: Vec<proximadb_relational_types::ColumnInfo> = aggregates
            .iter()
            .map(|a| proximadb_relational_types::ColumnInfo {
                name: a.name.clone(),
                ty: a.agg.result_type(),
                nullable: !matches!(a.agg, AggregateExpr::Count { .. }),
            })
            .collect();
        Self {
            child,
            aggregates,
            having,
            schema: RelationalSchema::new(cols),
            emitted: false,
        }
    }
}

#[async_trait]
impl ExecNode for StreamingAggregateExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.emitted = false;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        let mut accs: Vec<Accumulator> = self
            .aggregates
            .iter()
            .map(|a| Accumulator::new(&a.agg))
            .collect();
        while let Some(row) = self.child.next_row().await? {
            for (acc, named) in accs.iter_mut().zip(self.aggregates.iter()) {
                acc.accumulate(&named.agg, &row)?;
            }
        }
        let result: RelationalRow = accs.into_iter().map(|a| a.finalize()).collect();
        if let Some(having) = &self.having {
            let v = having.eval(&result, builtins())?;
            if !matches!(v, ProximaValue::Boolean(true)) {
                return Ok(None);
            }
        }
        Ok(Some(result))
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.child.close().await
    }
}

// =========================================================================
// HashAggregateExec (GROUP BY)
// =========================================================================

pub struct HashAggregateExec {
    child: Box<dyn ExecNode>,
    group_by: Vec<NamedExpr>,
    aggregates: Vec<NamedAggregate>,
    having: Option<Expr>,
    schema: RelationalSchema,
    /// Map group key → (group_by_values, accumulators).
    /// Computed eagerly on first next_row.
    groups: Option<Vec<(Vec<ProximaValue>, Vec<Accumulator>)>>,
    cursor: usize,
}

impl HashAggregateExec {
    fn new(
        child: Box<dyn ExecNode>,
        group_by: Vec<NamedExpr>,
        aggregates: Vec<NamedAggregate>,
        having: Option<Expr>,
    ) -> Self {
        let mut cols: Vec<proximadb_relational_types::ColumnInfo> = group_by
            .iter()
            .map(|g| proximadb_relational_types::ColumnInfo {
                name: g.name.clone(),
                ty: g.expr.result_type(),
                nullable: true,
            })
            .collect();
        for a in &aggregates {
            cols.push(proximadb_relational_types::ColumnInfo {
                name: a.name.clone(),
                ty: a.agg.result_type(),
                nullable: !matches!(a.agg, AggregateExpr::Count { .. }),
            });
        }
        Self {
            child,
            group_by,
            aggregates,
            having,
            schema: RelationalSchema::new(cols),
            groups: None,
            cursor: 0,
        }
    }

    async fn build_groups(&mut self) -> Result<(), ExecError> {
        if self.groups.is_some() {
            return Ok(());
        }
        let mut table: HashMap<GroupKey, (Vec<ProximaValue>, Vec<Accumulator>)> = HashMap::new();
        // Stable iteration order for output is not strictly
        // required by SQL but is nice to keep tests
        // deterministic. We rebuild the entries Vec from
        // insertion order via a parallel Vec.
        let mut insertion_order: Vec<GroupKey> = Vec::new();
        while let Some(row) = self.child.next_row().await? {
            let mut group_vals = Vec::with_capacity(self.group_by.len());
            for g in &self.group_by {
                group_vals.push(g.expr.eval(&row, builtins())?);
            }
            let key = GroupKey::from_values(&group_vals)?;
            let entry = match table.get_mut(&key) {
                Some(e) => e,
                None => {
                    let accs: Vec<Accumulator> = self
                        .aggregates
                        .iter()
                        .map(|a| Accumulator::new(&a.agg))
                        .collect();
                    insertion_order.push(key.clone());
                    table
                        .entry(key.clone())
                        .or_insert((group_vals.clone(), accs))
                }
            };
            for (acc, named) in entry.1.iter_mut().zip(self.aggregates.iter()) {
                acc.accumulate(&named.agg, &row)?;
            }
        }
        let mut out = Vec::with_capacity(insertion_order.len());
        for key in insertion_order {
            if let Some(entry) = table.remove(&key) {
                out.push(entry);
            }
        }
        self.groups = Some(out);
        Ok(())
    }
}

#[async_trait]
impl ExecNode for HashAggregateExec {
    fn schema(&self) -> &RelationalSchema {
        &self.schema
    }

    async fn open(&mut self) -> Result<(), ExecError> {
        self.groups = None;
        self.cursor = 0;
        self.child.open().await
    }

    async fn next_row(&mut self) -> Result<Option<RelationalRow>, ExecError> {
        self.build_groups().await?;
        let groups = self.groups.as_ref().expect("built");
        while self.cursor < groups.len() {
            let (gvals, accs) = &groups[self.cursor];
            self.cursor += 1;
            let mut row: RelationalRow = gvals.clone();
            row.extend(accs.iter().map(|a| a.finalize()));
            if let Some(having) = &self.having {
                let v = having.eval(&row, builtins())?;
                if !matches!(v, ProximaValue::Boolean(true)) {
                    continue;
                }
            }
            return Ok(Some(row));
        }
        Ok(None)
    }

    async fn close(&mut self) -> Result<(), ExecError> {
        self.groups = None;
        self.child.close().await
    }
}

// =========================================================================
// Aggregate accumulators
// =========================================================================

/// Per-aggregate per-group accumulator state. NULLs are skipped
/// per SQL semantics; an aggregate over zero rows yields NULL
/// (except COUNT, which yields 0).
enum Accumulator {
    Count {
        skip_null: bool,
        n: i64,
        distinct: Option<HashSet<GroupKey>>,
    },
    Sum {
        running: Option<f64>,
        distinct: Option<HashSet<GroupKey>>,
    },
    Avg {
        running_sum: f64,
        n: i64,
        distinct: Option<HashSet<GroupKey>>,
    },
    Min {
        current: Option<ProximaValue>,
    },
    Max {
        current: Option<ProximaValue>,
    },
    /// Registry-supplied aggregate (`AggregateExpr::Custom`). `acc` is `None` when the name
    /// did not resolve in the function registry — accumulation then errors clearly.
    Custom {
        acc: Option<Box<dyn proximadb_functions::AggregateAccumulator>>,
        distinct: Option<HashSet<GroupKey>>,
    },
}

impl Accumulator {
    fn new(agg: &AggregateExpr) -> Self {
        match agg {
            AggregateExpr::Count { arg, distinct } => Accumulator::Count {
                skip_null: arg.is_some(),
                n: 0,
                distinct: if *distinct {
                    Some(HashSet::new())
                } else {
                    None
                },
            },
            AggregateExpr::Sum { distinct, .. } => Accumulator::Sum {
                running: None,
                distinct: if *distinct {
                    Some(HashSet::new())
                } else {
                    None
                },
            },
            AggregateExpr::Avg { distinct, .. } => Accumulator::Avg {
                running_sum: 0.0,
                n: 0,
                distinct: if *distinct {
                    Some(HashSet::new())
                } else {
                    None
                },
            },
            AggregateExpr::Min { .. } => Accumulator::Min { current: None },
            AggregateExpr::Max { .. } => Accumulator::Max { current: None },
            AggregateExpr::StringAgg { .. } => Accumulator::Min { current: None }, // placeholder
            AggregateExpr::Custom { name, distinct, .. } => Accumulator::Custom {
                acc: builtins()
                    .lookup_aggregate(name)
                    .map(|def| def.kernel.new_accumulator()),
                distinct: if *distinct {
                    Some(HashSet::new())
                } else {
                    None
                },
            },
        }
    }

    fn accumulate(&mut self, agg: &AggregateExpr, row: &RelationalRow) -> Result<(), ExecError> {
        match (self, agg) {
            (
                Accumulator::Count {
                    skip_null,
                    n,
                    distinct,
                },
                AggregateExpr::Count { arg, .. },
            ) => {
                let v = match arg {
                    None => ProximaValue::Boolean(true),
                    Some(e) => e.eval(row, builtins())?,
                };
                if *skip_null && matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                if let Some(seen) = distinct {
                    let key = GroupKey::from_values(std::slice::from_ref(&v))?;
                    if seen.insert(key) {
                        *n += 1;
                    }
                } else {
                    *n += 1;
                }
                Ok(())
            }
            (Accumulator::Sum { running, distinct }, AggregateExpr::Sum { arg, .. }) => {
                let v = arg.eval(row, builtins())?;
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                if let Some(seen) = distinct {
                    let key = GroupKey::from_values(std::slice::from_ref(&v))?;
                    if !seen.insert(key) {
                        return Ok(());
                    }
                }
                let n = numeric_to_f64(&v)?;
                *running = Some(running.unwrap_or(0.0) + n);
                Ok(())
            }
            (
                Accumulator::Avg {
                    running_sum,
                    n,
                    distinct,
                },
                AggregateExpr::Avg { arg, .. },
            ) => {
                let v = arg.eval(row, builtins())?;
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                if let Some(seen) = distinct {
                    let key = GroupKey::from_values(std::slice::from_ref(&v))?;
                    if !seen.insert(key) {
                        return Ok(());
                    }
                }
                let f = numeric_to_f64(&v)?;
                *running_sum += f;
                *n += 1;
                Ok(())
            }
            (Accumulator::Min { current }, AggregateExpr::Min { arg }) => {
                let v = arg.eval(row, builtins())?;
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                match current {
                    None => *current = Some(v),
                    Some(c) => {
                        if compare_values(&v, c) == std::cmp::Ordering::Less {
                            *current = Some(v);
                        }
                    }
                }
                Ok(())
            }
            (Accumulator::Max { current }, AggregateExpr::Max { arg }) => {
                let v = arg.eval(row, builtins())?;
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                match current {
                    None => *current = Some(v),
                    Some(c) => {
                        if compare_values(&v, c) == std::cmp::Ordering::Greater {
                            *current = Some(v);
                        }
                    }
                }
                Ok(())
            }
            (_, AggregateExpr::StringAgg { .. }) => {
                Err(ExecError::UnsupportedAggregate("STRING_AGG".into()))
            }
            (Accumulator::Custom { acc, distinct }, AggregateExpr::Custom { name, args, .. }) => {
                let acc = acc
                    .as_mut()
                    .ok_or_else(|| ExecError::UnsupportedAggregate(name.clone()))?;
                let vals = args
                    .iter()
                    .map(|e| e.eval(row, builtins()))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(seen) = distinct {
                    let key = GroupKey::from_values(&vals)?;
                    if !seen.insert(key) {
                        return Ok(());
                    }
                }
                acc.update(&vals)?;
                Ok(())
            }
            // Mismatch (shouldn't happen — accumulators are paired
            // with their aggregate at construction).
            _ => Err(ExecError::Internal("accumulator/aggregate mismatch".into())),
        }
    }

    fn finalize(&self) -> ProximaValue {
        match self {
            Accumulator::Count { n, .. } => ProximaValue::Int64(*n),
            Accumulator::Sum { running, .. } => match running {
                Some(f) => ProximaValue::Float64(*f),
                None => ProximaValue::Null,
            },
            Accumulator::Avg { running_sum, n, .. } => {
                if *n == 0 {
                    ProximaValue::Null
                } else {
                    ProximaValue::Float64(*running_sum / *n as f64)
                }
            }
            Accumulator::Min { current } | Accumulator::Max { current } => {
                current.clone().unwrap_or(ProximaValue::Null)
            }
            Accumulator::Custom { acc, .. } => match acc {
                Some(a) => a.finalize(),
                None => ProximaValue::Null,
            },
        }
    }
}

fn numeric_to_f64(v: &ProximaValue) -> Result<f64, ExecError> {
    v.as_numeric_f64()
        .ok_or_else(|| ExecError::TypeMismatch(format!("expected numeric, got {v:?}")))
}

// =========================================================================
// GroupKey: hashable wrapper around a Vec<ProximaValue>
// =========================================================================

/// Hashable canonicalization of a row of ProximaValues. Used for
/// distinct, group-by, and hash-join keys. Floats are stored by
/// bit pattern so NaN == NaN (acceptable for grouping); nested
/// types (Json, Array, Map, Struct, vectors) error out — MVP
/// declines to define group equality for them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<KeyComponent>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyComponent {
    Null,
    Boolean(bool),
    Int(i128), // covers all integer types up to i64/u64
    Float(u64),
    Decimal(String),
    String(String),
    Symbol(String),
    Binary(Vec<u8>),
    Date(i32),
    Time(i64, u8),
    Timestamp(i64, u8),
    TimestampTz(i64, u8),
    Uuid([u8; 16]),
    // Matches the variant name on `ProximaValue::ULID` in the
    // foundation `proximadb-data-model` crate. Keeping the same
    // casing here avoids a cross-crate rename.
    #[allow(clippy::upper_case_acronyms)]
    ULID([u8; 16]),
}

impl GroupKey {
    fn from_row(row: &RelationalRow) -> Result<Self, ExecError> {
        Self::from_values(row.as_slice())
    }
    fn from_values(values: &[ProximaValue]) -> Result<Self, ExecError> {
        let mut out = Vec::with_capacity(values.len());
        for v in values {
            out.push(value_to_key(v)?);
        }
        Ok(GroupKey(out))
    }
}

fn value_to_key(v: &ProximaValue) -> Result<KeyComponent, ExecError> {
    use ProximaValue as V;
    Ok(match v {
        V::Null => KeyComponent::Null,
        V::Boolean(b) => KeyComponent::Boolean(*b),
        V::Int8(x) => KeyComponent::Int(*x as i128),
        V::Int16(x) => KeyComponent::Int(*x as i128),
        V::Int32(x) => KeyComponent::Int(*x as i128),
        V::Int64(x) => KeyComponent::Int(*x as i128),
        V::UInt8(x) => KeyComponent::Int(*x as i128),
        V::UInt16(x) => KeyComponent::Int(*x as i128),
        V::UInt32(x) => KeyComponent::Int(*x as i128),
        V::UInt64(x) => KeyComponent::Int(*x as i128),
        V::Float16(x) | V::Float32(x) => KeyComponent::Float((*x as f64).to_bits()),
        V::Float64(x) => KeyComponent::Float(x.to_bits()),
        V::Decimal(s) => KeyComponent::Decimal(s.clone()),
        V::String(s) => KeyComponent::String(s.clone()),
        V::Symbol(s) => KeyComponent::Symbol(s.clone()),
        V::Binary(b) => KeyComponent::Binary(b.clone()),
        V::Date(d) => KeyComponent::Date(*d),
        V::Time(x, unit) => KeyComponent::Time(*x, time_unit_byte(*unit)),
        V::Timestamp(x, unit) => KeyComponent::Timestamp(*x, time_unit_byte(*unit)),
        V::TimestampTz(x, unit) => KeyComponent::TimestampTz(*x, time_unit_byte(*unit)),
        V::Uuid(u) => KeyComponent::Uuid(*u),
        V::ULID(u) => KeyComponent::ULID(*u),
        // Reject nested / vector types — defining grouping
        // equality for these is a Phase 3 design.
        other => {
            return Err(ExecError::UnsupportedGroupKey(infer_value_type(other)));
        }
    })
}

fn time_unit_byte(u: TimeUnit) -> u8 {
    match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

fn infer_value_type(v: &ProximaValue) -> ProximaType {
    use ProximaValue as V;
    // Used only inside [`ExecError::UnsupportedGroupKey`] for
    // diagnostics. Map JSON/array/map/struct/vector types onto
    // the closest concrete scalar so the error is human-readable
    // without us having to fabricate struct-variant payloads.
    match v {
        V::Boolean(_) => ProximaType::Boolean,
        V::Int8(_) => ProximaType::Int8,
        V::Int16(_) => ProximaType::Int16,
        V::Int32(_) => ProximaType::Int32,
        V::Int64(_) => ProximaType::Int64,
        V::UInt8(_) => ProximaType::UInt8,
        V::UInt16(_) => ProximaType::UInt16,
        V::UInt32(_) => ProximaType::UInt32,
        V::UInt64(_) => ProximaType::UInt64,
        V::Float16(_) => ProximaType::Float16,
        V::Float32(_) => ProximaType::Float32,
        V::Float64(_) => ProximaType::Float64,
        V::String(_) | V::Symbol(_) => ProximaType::String,
        V::Binary(_) => ProximaType::Binary,
        V::Date(_) => ProximaType::Date,
        V::Json(_) => ProximaType::Json,
        V::Jsonb(_) => ProximaType::Jsonb,
        _ => ProximaType::String,
    }
}

// =========================================================================
// Test helpers + tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_relational_algebra::{AggregateExpr as AggExpr, NamedAggregate, NamedExpr};
    use proximadb_relational_planner::{
        AggregateStrategy, DistinctStrategy, PhysicalPlan, ScanAccess, SortStrategy,
    };
    use proximadb_relational_reader::VecReader;
    use proximadb_relational_types::{BinaryOp, ColumnInfo, ColumnRef, Expr};
    use std::sync::Mutex;

    #[test]
    fn aggregate_numeric_conversion_keeps_float16_and_type_errors() {
        assert_eq!(numeric_to_f64(&ProximaValue::Float16(1.5)).unwrap(), 1.5);
        for value in [
            ProximaValue::Decimal("1.5".into()),
            ProximaValue::String("1.5".into()),
            ProximaValue::Null,
        ] {
            let result = numeric_to_f64(&value);
            assert!(matches!(result, Err(ExecError::TypeMismatch(ref msg))
                if msg == &format!("expected numeric, got {value:?}")));
        }
    }

    type ReaderSource = (RelationalSchema, Vec<RelationalRow>, Vec<usize>);

    /// Test reader factory backed by a name-keyed registry. Each
    /// table returns a clone of a stored `VecReader` source.
    struct VecReaderFactory {
        sources: Mutex<HashMap<String, ReaderSource>>,
        /// Optional non-PK secondary-index ordinals per table (TD-127).
        secondary: Mutex<HashMap<String, Vec<usize>>>,
    }

    impl VecReaderFactory {
        fn new() -> Self {
            Self {
                sources: Mutex::new(HashMap::new()),
                secondary: Mutex::new(HashMap::new()),
            }
        }
        fn register(
            &self,
            name: &str,
            schema: RelationalSchema,
            rows: Vec<RelationalRow>,
            pk_columns: Vec<usize>,
        ) {
            self.sources
                .lock()
                .unwrap()
                .insert(name.to_string(), (schema, rows, pk_columns));
        }
        /// Register a non-PK secondary index on the given ordinals (TD-127),
        /// so the table's reader answers `lookup_secondary` for them.
        fn register_secondary(&self, name: &str, columns: Vec<usize>) {
            self.secondary
                .lock()
                .unwrap()
                .insert(name.to_string(), columns);
        }
    }

    impl ReaderFactory for VecReaderFactory {
        fn open_reader(&self, table: &TableId) -> Result<Box<dyn RelationalReader>, ExecError> {
            let lock = self.sources.lock().unwrap();
            let (schema, rows, pk) = lock
                .get(&table.name)
                .ok_or_else(|| ExecError::Internal(format!("no table {}", table.name)))?
                .clone();
            let secondary = self
                .secondary
                .lock()
                .unwrap()
                .get(&table.name)
                .cloned()
                .unwrap_or_default();
            Ok(Box::new(
                VecReader::new(schema, rows, pk).with_secondary_index(secondary),
            ))
        }
    }

    fn users_schema() -> RelationalSchema {
        RelationalSchema::new(vec![
            ColumnInfo::new("id", ProximaType::Int64, false),
            ColumnInfo::new("name", ProximaType::String, true),
            ColumnInfo::new("age", ProximaType::Int32, true),
        ])
    }

    fn users_rows() -> Vec<RelationalRow> {
        vec![
            vec![
                ProximaValue::Int64(1),
                ProximaValue::String("alice".into()),
                ProximaValue::Int32(30),
            ],
            vec![
                ProximaValue::Int64(2),
                ProximaValue::String("bob".into()),
                ProximaValue::Int32(25),
            ],
            vec![
                ProximaValue::Int64(3),
                ProximaValue::String("carol".into()),
                ProximaValue::Int32(40),
            ],
        ]
    }

    fn factory_with_users() -> VecReaderFactory {
        let f = VecReaderFactory::new();
        f.register("users", users_schema(), users_rows(), vec![0]);
        f
    }

    fn scan_users() -> PhysicalPlan {
        PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::FullScan,
        }
    }

    // ----- Scan + Filter + Project -------------------------------------

    #[tokio::test]
    async fn scan_emits_all_rows() {
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(scan_users(), &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 3);
    }

    /// Executor-build stack high-water for a `depth`-level filter chain, measured
    /// on a dedicated large-stack thread so `maybe_grow` never switches segments
    /// and the probe reading is exact (see the planner's `stack_probe` docs).
    fn build_stack_hwm(depth: usize) -> u64 {
        use proximadb_relational_planner::stack_probe;
        std::thread::Builder::new()
            // Large enough that even TD-EXEC-1's 200 KB/frame upper guess for
            // ~1.1k frames fits without segment growth.
            .stack_size(256 * 1024 * 1024)
            .spawn(move || {
                let mut plan = scan_users();
                for _ in 0..depth {
                    plan = PhysicalPlan::Filter {
                        input: Box::new(plan),
                        predicate: Expr::literal(ProximaValue::Boolean(true)),
                    };
                }
                let f = factory_with_users();
                let ctx = ExecutionContext::default();
                let (exec, hwm) = stack_probe::probe(|| build_executor(plan, &f, &ctx));
                // Avoid a deep recursive Drop of the executor tree; only the
                // measurement matters here.
                std::mem::forget(exec.expect("deep build must succeed"));
                hwm
            })
            .expect("spawn large-stack calibration thread")
            .join()
            .expect("calibration build must not fail")
    }

    /// TD-EXEC-2 Slice 1 calibration: measure the real per-frame stack cost of
    /// the executor-build recursion — the second of the three depth-proportional
    /// recursions (planner lowering, executor build, native walk) — resolving
    /// the ~100× `frame_bytes` unknown for this crate. The delta between two
    /// chain depths divides out fixed overhead; the upper bound is a regression
    /// ratchet against frame bloat.
    #[test]
    fn calibrate_build_frame_cost() {
        let shallow = build_stack_hwm(100);
        let deep = build_stack_hwm(1100);
        if shallow == 0 && deep == 0 {
            // Platform cannot report remaining stack; nothing to calibrate.
            return;
        }
        assert!(
            deep > shallow,
            "1100-deep build must consume more stack than 100-deep: \
             shallow={shallow} deep={deep}"
        );
        let per_frame = (deep - shallow) / 1000;
        eprintln!(
            "TD-EXEC-2 Slice-1 calibration: executor-build frame cost ≈ {per_frame} B/frame \
             (hwm {shallow} B @ depth 100 → {deep} B @ depth 1100)"
        );
        assert!(
            per_frame <= 32 * 1024,
            "executor-build frame cost regressed past 32 KiB/frame: measured {per_frame} B/frame"
        );
    }

    #[tokio::test]
    async fn filter_uses_three_valued_logic() {
        // age > 26 — bob(25) excluded; alice(30), carol(40) kept.
        let age = users_schema().resolve_column("age").unwrap();
        let plan = PhysicalPlan::Filter {
            input: Box::new(scan_users()),
            predicate: Expr::bin(
                BinaryOp::Gt,
                Expr::column(age),
                Expr::literal(ProximaValue::Int32(26)),
            ),
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn project_evaluates_expressions() {
        let id = users_schema().resolve_column("id").unwrap();
        let plan = PhysicalPlan::Project {
            input: Box::new(scan_users()),
            outputs: vec![NamedExpr::new(
                "double_id",
                Expr::bin(
                    BinaryOp::Mul,
                    Expr::column(id),
                    Expr::literal(ProximaValue::Int64(2)),
                ),
            )],
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 3);
        // Each row has one column; alice/bob/carol → 2/4/6.
        let vals: Vec<i64> = rows
            .iter()
            .map(|r| match r[0] {
                ProximaValue::Int64(x) => x,
                _ => panic!(),
            })
            .collect();
        assert_eq!(vals, vec![2, 4, 6]);
    }

    // ----- Limit + Offset ----------------------------------------------

    #[tokio::test]
    async fn limit_caps_emitted_rows() {
        let plan = PhysicalPlan::Limit {
            input: Box::new(scan_users()),
            limit: Some(2),
            offset: 0,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn offset_skips_first_rows() {
        let plan = PhysicalPlan::Limit {
            input: Box::new(scan_users()),
            limit: None,
            offset: 1,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2); // skipped row 0
        // First emitted row is now bob (id=2).
        assert_eq!(rows[0][0], ProximaValue::Int64(2));
    }

    // ----- Distinct + Union --------------------------------------------

    #[tokio::test]
    async fn distinct_deduplicates() {
        let f = VecReaderFactory::new();
        let rows = vec![
            vec![ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(2)],
        ];
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        f.register("t", schema.clone(), rows, vec![]);
        let plan = PhysicalPlan::Distinct {
            input: Box::new(PhysicalPlan::Scan {
                table: TableId::new("t"),
                output_schema: schema,
                projection: None,
                predicate: None,
                limit: None,
                access: ScanAccess::FullScan,
            }),
            strategy: DistinctStrategy::Hash,
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
    }

    // ----- AssertMaxOneRow (scalar-subquery cardinality guard) ---------

    fn one_col_scan(
        f: &VecReaderFactory,
        name: &str,
        rows: Vec<Vec<ProximaValue>>,
    ) -> PhysicalPlan {
        let schema = RelationalSchema::new(vec![ColumnInfo::new("v", ProximaType::Int64, true)]);
        f.register(name, schema.clone(), rows, vec![]);
        PhysicalPlan::Scan {
            table: TableId::new(name),
            output_schema: schema,
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::FullScan,
        }
    }

    #[tokio::test]
    async fn assert_max_one_row_passes_single_row() {
        let f = VecReaderFactory::new();
        let plan = PhysicalPlan::AssertMaxOneRow {
            input: Box::new(one_col_scan(&f, "sq", vec![vec![ProximaValue::Int64(7)]])),
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows, vec![vec![ProximaValue::Int64(7)]]);
    }

    #[tokio::test]
    async fn assert_max_one_row_errors_on_second_row() {
        let f = VecReaderFactory::new();
        let plan = PhysicalPlan::AssertMaxOneRow {
            input: Box::new(one_col_scan(
                &f,
                "sq",
                vec![vec![ProximaValue::Int64(1)], vec![ProximaValue::Int64(2)]],
            )),
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let err = collect(&mut *exec).await.unwrap_err();
        assert!(
            matches!(err, ExecError::ScalarSubqueryCardinality),
            "expected ScalarSubqueryCardinality, got {err:?}"
        );
    }

    #[tokio::test]
    async fn left_join_on_true_over_empty_scalar_yields_null() {
        // The full scalar-subquery shape: base LEFT JOIN ON TRUE an empty
        // AssertMaxOneRow → base rows survive, null-padded on the right.
        let f = VecReaderFactory::new();
        f.register("users", users_schema(), users_rows(), vec![0]);
        let right = PhysicalPlan::AssertMaxOneRow {
            input: Box::new(one_col_scan(&f, "sq", vec![])), // empty subquery
        };
        let plan = PhysicalPlan::Join {
            left: Box::new(scan_users()),
            right: Box::new(right),
            kind: JoinKind::Left,
            on: None,
            strategy: JoinStrategy::NestedLoop,
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        // Every user row survives; the appended scalar column (last) is NULL.
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(*r.last().unwrap(), ProximaValue::Null);
        }
    }

    // ----- Sort --------------------------------------------------------

    #[tokio::test]
    async fn sort_orders_rows() {
        // SORT BY age DESC.
        let age = users_schema().resolve_column("age").unwrap();
        let plan = PhysicalPlan::Sort {
            input: Box::new(scan_users()),
            keys: vec![SortKey::desc(Expr::column(age))],
            strategy: SortStrategy::InMemory,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        // carol(40), alice(30), bob(25)
        assert_eq!(rows[0][1], ProximaValue::String("carol".into()));
        assert_eq!(rows[1][1], ProximaValue::String("alice".into()));
        assert_eq!(rows[2][1], ProximaValue::String("bob".into()));
    }

    // ----- Aggregate ---------------------------------------------------

    #[tokio::test]
    async fn streaming_aggregate_count_no_group_by() {
        let plan = PhysicalPlan::Aggregate {
            input: Box::new(scan_users()),
            group_by: vec![],
            aggregates: vec![NamedAggregate::new(
                "n",
                AggExpr::Count {
                    arg: None,
                    distinct: false,
                },
            )],
            having: None,
            strategy: AggregateStrategy::Streaming,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::Int64(3));
    }

    #[tokio::test]
    async fn streaming_aggregate_custom_product() {
        // F3: AggregateExpr::Custom dispatches through the shared function registry (the
        // `product` builtin accumulator): product(age) over [30, 25, 40] = 30000.
        let age = users_schema().resolve_column("age").unwrap();
        let plan = PhysicalPlan::Aggregate {
            input: Box::new(scan_users()),
            group_by: vec![],
            aggregates: vec![NamedAggregate::new(
                "p",
                AggExpr::Custom {
                    name: "product".into(),
                    args: vec![Expr::column(age)],
                    distinct: false,
                    return_ty: ProximaType::Float64,
                },
            )],
            having: None,
            strategy: AggregateStrategy::Streaming,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::Float64(30000.0));
    }

    #[tokio::test]
    async fn hash_aggregate_groups_by_column() {
        // Add a duplicate-age row so we have a real group.
        let f = VecReaderFactory::new();
        let mut rows = users_rows();
        rows.push(vec![
            ProximaValue::Int64(4),
            ProximaValue::String("dave".into()),
            ProximaValue::Int32(30), // same age as alice
        ]);
        f.register("users", users_schema(), rows, vec![0]);
        let age = users_schema().resolve_column("age").unwrap();
        let plan = PhysicalPlan::Aggregate {
            input: Box::new(scan_users()),
            group_by: vec![NamedExpr::new("age", Expr::column(age))],
            aggregates: vec![NamedAggregate::new(
                "n",
                AggExpr::Count {
                    arg: None,
                    distinct: false,
                },
            )],
            having: None,
            strategy: AggregateStrategy::Hash,
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        // Three groups: 30 → 2, 25 → 1, 40 → 1.
        assert_eq!(rows.len(), 3);
        let mut groups: HashMap<i32, i64> = HashMap::new();
        for r in rows {
            let age = match r[0] {
                ProximaValue::Int32(x) => x,
                _ => panic!(),
            };
            let n = match r[1] {
                ProximaValue::Int64(x) => x,
                _ => panic!(),
            };
            groups.insert(age, n);
        }
        assert_eq!(groups.get(&30), Some(&2));
        assert_eq!(groups.get(&25), Some(&1));
        assert_eq!(groups.get(&40), Some(&1));
    }

    // ----- Nested-loop join --------------------------------------------

    #[tokio::test]
    async fn nested_loop_inner_join_combines_rows() {
        let f = VecReaderFactory::new();
        let orders_schema = RelationalSchema::new(vec![
            ColumnInfo::new("oid", ProximaType::Int64, false),
            ColumnInfo::new("uid", ProximaType::Int64, false),
        ]);
        let orders_rows = vec![
            vec![ProximaValue::Int64(100), ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(101), ProximaValue::Int64(2)],
            vec![ProximaValue::Int64(102), ProximaValue::Int64(1)],
        ];
        f.register("users", users_schema(), users_rows(), vec![0]);
        f.register("orders", orders_schema.clone(), orders_rows, vec![0]);
        // ON users.id = orders.uid
        let combined_schema_len = users_schema().len();
        let plan = PhysicalPlan::Join {
            left: Box::new(scan_users()),
            right: Box::new(PhysicalPlan::Scan {
                table: TableId::new("orders"),
                output_schema: orders_schema,
                projection: None,
                predicate: None,
                limit: None,
                access: ScanAccess::FullScan,
            }),
            kind: JoinKind::Inner,
            on: Some(Expr::bin(
                BinaryOp::Eq,
                Expr::column(ColumnRef {
                    name: "id".into(),
                    ordinal: 0,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
                Expr::column(ColumnRef {
                    name: "uid".into(),
                    ordinal: combined_schema_len + 1,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
            )),
            strategy: JoinStrategy::NestedLoop,
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        // alice has orders 100 & 102; bob has 101; carol none.
        assert_eq!(rows.len(), 3);
    }

    /// users(id@0,name@1,age@2) ⋈ orders(oid@3,uid@4) on users.id = orders.uid.
    /// `orders_rows` lets each test supply (un)matched rows; `strategy` picks the
    /// physical operator (NestedLoop / Hash).
    fn users_orders_join(
        kind: JoinKind,
        orders_rows: Vec<RelationalRow>,
        strategy: JoinStrategy,
        f: &VecReaderFactory,
    ) -> PhysicalPlan {
        let orders_schema = RelationalSchema::new(vec![
            ColumnInfo::new("oid", ProximaType::Int64, false),
            ColumnInfo::new("uid", ProximaType::Int64, false),
        ]);
        f.register("users", users_schema(), users_rows(), vec![0]);
        f.register("orders", orders_schema.clone(), orders_rows, vec![0]);
        let combined = users_schema().len();
        PhysicalPlan::Join {
            left: Box::new(scan_users()),
            right: Box::new(PhysicalPlan::Scan {
                table: TableId::new("orders"),
                output_schema: orders_schema,
                projection: None,
                predicate: None,
                limit: None,
                access: ScanAccess::FullScan,
            }),
            kind,
            on: Some(Expr::bin(
                BinaryOp::Eq,
                Expr::column(ColumnRef {
                    name: "id".into(),
                    ordinal: 0,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
                Expr::column(ColumnRef {
                    name: "uid".into(),
                    ordinal: combined + 1,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
            )),
            strategy,
        }
    }

    // orders: 100→u1, 101→u2, 102→u1, 103→u99 (no such user, unmatched right).
    fn orders_with_unmatched() -> Vec<RelationalRow> {
        vec![
            vec![ProximaValue::Int64(100), ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(101), ProximaValue::Int64(2)],
            vec![ProximaValue::Int64(102), ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(103), ProximaValue::Int64(99)],
        ]
    }

    /// Shared assertions for a RIGHT join over `orders_with_unmatched`:
    /// 3 matched + order 103 null-padded on the left; no unmatched-left rows.
    fn assert_right_join_rows(rows: &[RelationalRow]) {
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter().filter(|r| r[0] == ProximaValue::Null).count(),
            1,
            "unmatched right (order 103) emitted with NULL user columns"
        );
        assert!(
            rows.iter().all(|r| r[3] != ProximaValue::Null),
            "RIGHT must NOT emit unmatched left rows (carol)"
        );
    }

    /// Shared assertions for a FULL join: 3 matched + carol (NULL right) +
    /// order 103 (NULL left).
    fn assert_full_join_rows(rows: &[RelationalRow]) {
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows.iter().filter(|r| r[0] == ProximaValue::Null).count(),
            1,
            "unmatched right (order 103) → NULL left"
        );
        assert_eq!(
            rows.iter().filter(|r| r[3] == ProximaValue::Null).count(),
            1,
            "unmatched left (carol) → NULL right"
        );
    }

    fn nm(label: &str, arity: usize, elapsed_ns: u128) -> NodeMetric {
        NodeMetric {
            label: label.into(),
            arity,
            rows: 0,
            elapsed_ns,
        }
    }

    #[test]
    fn self_times_subtracts_direct_children() {
        // Filter(100, incl) over Join(90) over Scan(30), Scan(20):
        //   self(Filter) = 100 - 90 = 10
        //   self(Join)   = 90 - (30 + 20) = 40
        //   self(Scan*)  = leaves unchanged
        let metrics = vec![
            nm("Filter", 1, 100),
            nm("Join", 2, 90),
            nm("Scan", 0, 30),
            nm("Scan", 0, 20),
        ];
        assert_eq!(self_times(&metrics), vec![10, 40, 30, 20]);
    }

    #[test]
    fn self_times_saturates_on_noise_and_tolerates_empty() {
        // Child inclusive > parent inclusive (timing noise) → self saturates to 0.
        let noisy = vec![nm("Filter", 1, 5), nm("Scan", 0, 9)];
        assert_eq!(self_times(&noisy), vec![0, 9]);
        assert_eq!(self_times(&[]), Vec::<u128>::new());
    }

    fn nm_rows(label: &str, arity: usize, rows: u64) -> NodeMetric {
        NodeMetric {
            label: label.into(),
            arity,
            rows,
            elapsed_ns: 0,
        }
    }

    #[test]
    fn child_input_rows_sums_direct_children() {
        // Filter(1) over Join(2) over Scan(30), Scan(20):
        //   rows_in(Filter) = rows_out(Join)          = 45
        //   rows_in(Join)   = 30 + 20                  = 50
        //   rows_in(Scan*)  = leaves → 0
        let metrics = vec![
            nm_rows("Filter", 1, 12),
            nm_rows("Join", 2, 45),
            nm_rows("Scan", 0, 30),
            nm_rows("Scan", 0, 20),
        ];
        assert_eq!(child_input_rows(&metrics), vec![45, 50, 0, 0]);
    }

    #[test]
    fn child_input_rows_tolerates_malformed_arity_and_empty() {
        // A node claims 2 children but only 1 subtree follows → no panic / no OOB;
        // it sums what is actually present.
        let malformed = vec![nm_rows("Join", 2, 7), nm_rows("Scan", 0, 4)];
        assert_eq!(child_input_rows(&malformed), vec![4, 0]);
        assert_eq!(child_input_rows(&[]), Vec::<u64>::new());
    }

    async fn run_setop(op: SetOpKind, all: bool, left: &[i64], right: &[i64]) -> Vec<i64> {
        let int_schema =
            || RelationalSchema::new(vec![ColumnInfo::new("v", ProximaType::Int64, false)]);
        let int_rows = |vals: &[i64]| vals.iter().map(|v| vec![ProximaValue::Int64(*v)]).collect();
        let f = VecReaderFactory::new();
        f.register("lft", int_schema(), int_rows(left), vec![0]);
        f.register("rgt", int_schema(), int_rows(right), vec![0]);
        let scan = |t: &str| PhysicalPlan::Scan {
            table: TableId::new(t),
            output_schema: int_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::FullScan,
        };
        let plan = PhysicalPlan::SetOp {
            op,
            left: Box::new(scan("lft")),
            right: Box::new(scan("rgt")),
            all,
        };
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        let mut out: Vec<i64> = rows
            .iter()
            .map(|r| match &r[0] {
                ProximaValue::Int64(v) => *v,
                other => panic!("expected Int64, got {other:?}"),
            })
            .collect();
        out.sort(); // set-op output order is unspecified
        out
    }

    async fn run_not_in(left: &[Option<i64>], right: &[Option<i64>]) -> Vec<i64> {
        let int_schema =
            || RelationalSchema::new(vec![ColumnInfo::new("v", ProximaType::Int64, true)]);
        let to_rows = |vals: &[Option<i64>]| -> Vec<RelationalRow> {
            vals.iter()
                .map(|v| vec![v.map_or(ProximaValue::Null, ProximaValue::Int64)])
                .collect()
        };
        let f = VecReaderFactory::new();
        f.register("lft", int_schema(), to_rows(left), vec![0]);
        f.register("rgt", int_schema(), to_rows(right), vec![0]);
        let scan = |t: &str| PhysicalPlan::Scan {
            table: TableId::new(t),
            output_schema: int_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::FullScan,
        };
        let col = |ordinal| {
            Expr::column(ColumnRef {
                name: "v".into(),
                ordinal,
                ty: ProximaType::Int64,
                nullable: true,
            })
        };
        let plan = PhysicalPlan::Join {
            left: Box::new(scan("lft")),
            right: Box::new(scan("rgt")),
            kind: JoinKind::AntiNullAware,
            on: Some(Expr::bin(BinaryOp::Eq, col(0), col(1))),
            strategy: JoinStrategy::NestedLoop,
        };
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        let mut out: Vec<i64> = rows
            .iter()
            .map(|r| match &r[0] {
                ProximaValue::Int64(v) => *v,
                other => panic!("expected Int64, got {other:?}"),
            })
            .collect();
        out.sort();
        out
    }

    #[tokio::test]
    async fn not_in_null_aware_anti_join_semantics() {
        // x NOT IN (S) — SQL three-valued semantics.
        // basic: {1,2,3} NOT IN {2} → {1,3}
        assert_eq!(
            run_not_in(&[Some(1), Some(2), Some(3)], &[Some(2)]).await,
            vec![1, 3]
        );
        // S contains a NULL → the predicate is UNKNOWN for every x → EMPTY.
        assert_eq!(
            run_not_in(&[Some(1), Some(2)], &[None]).await,
            Vec::<i64>::new()
        );
        // a NULL x is never emitted; the non-null rows behave normally.
        // {NULL,1,3} NOT IN {3} → {1}
        assert_eq!(
            run_not_in(&[None, Some(1), Some(3)], &[Some(3)]).await,
            vec![1]
        );
        // NOT IN an empty set → all (non-null) left rows. {1,2} NOT IN {} → {1,2}
        assert_eq!(run_not_in(&[Some(1), Some(2)], &[]).await, vec![1, 2]);
    }

    #[tokio::test]
    async fn setop_intersect_except_distinct_and_all() {
        // left = {1, 2, 2, 3}, right = {2, 3, 3, 4}
        let l = [1, 2, 2, 3];
        let r = [2, 3, 3, 4];
        // INTERSECT (distinct): keys in both → {2, 3}.
        assert_eq!(
            run_setop(SetOpKind::Intersect, false, &l, &r).await,
            vec![2, 3]
        );
        // INTERSECT ALL: min(count_l, count_r) → 2×min(2,1)=1, 3×min(1,2)=1 → [2,3].
        assert_eq!(
            run_setop(SetOpKind::Intersect, true, &l, &r).await,
            vec![2, 3]
        );
        // EXCEPT (distinct): left keys absent on right → {1}.
        assert_eq!(run_setop(SetOpKind::Except, false, &l, &r).await, vec![1]);
        // EXCEPT ALL: max(0, count_l − count_r) → 1×1, 2×(2−1)=1, 3×0 → [1,2].
        assert_eq!(run_setop(SetOpKind::Except, true, &l, &r).await, vec![1, 2]);
    }

    #[tokio::test]
    async fn metered_exec_aligns_metrics_pre_order_with_plan() {
        // Filter(Join(Scan users, Scan orders)) → pre-order labels must be
        // [Filter, Join, Scan, Scan], matching how `explain_physical` renders the
        // tree. The root (Filter) row count equals the drained row count.
        let f = VecReaderFactory::new();
        let plan = PhysicalPlan::Filter {
            input: Box::new(users_orders_join(
                JoinKind::Inner,
                orders_with_unmatched(),
                JoinStrategy::NestedLoop,
                &f,
            )),
            predicate: Expr::literal(ProximaValue::Boolean(true)),
        };
        let metrics = std::sync::Arc::new(ExecMetrics::new());
        let ctx = ExecutionContext::with_metrics(metrics.clone());
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        drop(exec); // flush MeteredExec counters
        let nodes = metrics.snapshot();
        let labels: Vec<&str> = nodes.iter().map(|n| n.label.as_str()).collect();
        assert_eq!(
            labels,
            ["Filter", "Join", "Scan", "Scan"],
            "metrics align pre-order with the plan tree"
        );
        let arities: Vec<usize> = nodes.iter().map(|n| n.arity).collect();
        assert_eq!(
            arities,
            [1, 2, 0, 0],
            "arity lets the tree be reconstructed from the pre-order vector"
        );
        assert_eq!(
            nodes[0].rows as usize,
            rows.len(),
            "root (Filter) rows == drained rows"
        );
    }

    #[tokio::test]
    async fn nested_loop_right_join_emits_unmatched_right_rows() {
        let f = VecReaderFactory::new();
        let plan = users_orders_join(
            JoinKind::Right,
            orders_with_unmatched(),
            JoinStrategy::NestedLoop,
            &f,
        );
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        assert_right_join_rows(&collect(&mut *exec).await.unwrap());
    }

    #[tokio::test]
    async fn nested_loop_full_join_emits_both_unmatched_sides() {
        let f = VecReaderFactory::new();
        let plan = users_orders_join(
            JoinKind::Full,
            orders_with_unmatched(),
            JoinStrategy::NestedLoop,
            &f,
        );
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        assert_full_join_rows(&collect(&mut *exec).await.unwrap());
    }

    // ----- Hash join ---------------------------------------------------

    #[tokio::test]
    async fn hash_right_join_drains_unmatched_build_rows() {
        let f = VecReaderFactory::new();
        let plan = users_orders_join(
            JoinKind::Right,
            orders_with_unmatched(),
            JoinStrategy::Hash {
                build_side: proximadb_relational_algebra::JoinSide::Right,
            },
            &f,
        );
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        assert_right_join_rows(&collect(&mut *exec).await.unwrap());
    }

    #[tokio::test]
    async fn hash_full_join_drains_unmatched_build_and_emits_unmatched_probe() {
        let f = VecReaderFactory::new();
        let plan = users_orders_join(
            JoinKind::Full,
            orders_with_unmatched(),
            JoinStrategy::Hash {
                build_side: proximadb_relational_algebra::JoinSide::Right,
            },
            &f,
        );
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        assert_full_join_rows(&collect(&mut *exec).await.unwrap());
    }

    #[tokio::test]
    async fn hash_inner_join_combines_rows() {
        let f = VecReaderFactory::new();
        let orders_schema = RelationalSchema::new(vec![
            ColumnInfo::new("oid", ProximaType::Int64, false),
            ColumnInfo::new("uid", ProximaType::Int64, false),
        ]);
        let orders_rows = vec![
            vec![ProximaValue::Int64(100), ProximaValue::Int64(1)],
            vec![ProximaValue::Int64(101), ProximaValue::Int64(2)],
            vec![ProximaValue::Int64(102), ProximaValue::Int64(1)],
        ];
        f.register("users", users_schema(), users_rows(), vec![0]);
        f.register("orders", orders_schema.clone(), orders_rows, vec![0]);
        let combined_schema_len = users_schema().len();
        let plan = PhysicalPlan::Join {
            left: Box::new(scan_users()),
            right: Box::new(PhysicalPlan::Scan {
                table: TableId::new("orders"),
                output_schema: orders_schema,
                projection: None,
                predicate: None,
                limit: None,
                access: ScanAccess::FullScan,
            }),
            kind: JoinKind::Inner,
            on: Some(Expr::bin(
                BinaryOp::Eq,
                Expr::column(ColumnRef {
                    name: "id".into(),
                    ordinal: 0,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
                Expr::column(ColumnRef {
                    name: "uid".into(),
                    ordinal: combined_schema_len + 1,
                    ty: ProximaType::Int64,
                    nullable: false,
                }),
            )),
            strategy: JoinStrategy::Hash {
                build_side: proximadb_relational_algebra::JoinSide::Right,
            },
        };
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 3);
    }

    // ----- PK lookup ---------------------------------------------------

    #[tokio::test]
    async fn scan_pk_lookup_returns_single_row() {
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::PkLookup {
                key: vec![Expr::literal(ProximaValue::Int64(2))],
            },
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1], ProximaValue::String("bob".into()));
    }

    #[tokio::test]
    async fn scan_pk_lookup_missing_returns_empty() {
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::PkLookup {
                key: vec![Expr::literal(ProximaValue::Int64(999))],
            },
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[tokio::test]
    async fn scan_pk_lookup_applies_projection() {
        // PkLookup with projection=["name"] must emit a single-column
        // row containing just the name, not the full (id, name, age)
        // row that lookup_pk returns. The narrowed output_schema mirrors
        // what the planner would produce after pushdown.
        let narrowed =
            RelationalSchema::new(vec![ColumnInfo::new("name", ProximaType::String, true)]);
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: narrowed,
            projection: Some(vec!["name".into()]),
            predicate: None,
            limit: None,
            access: ScanAccess::PkLookup {
                key: vec![Expr::literal(ProximaValue::Int64(2))],
            },
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 1, "row should be narrowed to one column");
        assert_eq!(rows[0][0], ProximaValue::String("bob".into()));
    }

    // ----- PK batch lookup (TD-128) ------------------------------------

    #[tokio::test]
    async fn scan_pk_lookup_batch_returns_matching_rows() {
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::PkLookupBatch {
                keys: vec![
                    vec![Expr::literal(ProximaValue::Int64(1))],
                    vec![Expr::literal(ProximaValue::Int64(3))],
                    vec![Expr::literal(ProximaValue::Int64(999))], // miss
                ],
            },
        };
        let f = factory_with_users();
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2, "two hits, miss omitted");
        let names: Vec<&str> = rows
            .iter()
            .map(|r| match &r[1] {
                ProximaValue::String(s) => s.as_str(),
                _ => unreachable!(),
            })
            .collect();
        assert!(names.contains(&"alice"));
        assert!(names.contains(&"carol"));
    }

    #[tokio::test]
    async fn scan_pk_lookup_batch_applies_projection() {
        let narrowed =
            RelationalSchema::new(vec![ColumnInfo::new("name", ProximaType::String, true)]);
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: narrowed,
            projection: Some(vec!["name".into()]),
            predicate: None,
            limit: None,
            access: ScanAccess::PkLookupBatch {
                keys: vec![
                    vec![Expr::literal(ProximaValue::Int64(1))],
                    vec![Expr::literal(ProximaValue::Int64(2))],
                ],
            },
        };
        let f = factory_with_users();
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.len() == 1), "narrowed to one column");
    }

    // ----- Secondary lookup (TD-127) -----------------------------------

    #[tokio::test]
    async fn scan_secondary_lookup_returns_candidates() {
        // Secondary index on `name` (ordinal 1).
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::SecondaryLookup {
                column: "name".into(),
                values: vec![Expr::literal(ProximaValue::String("bob".into()))],
            },
        };
        let f = factory_with_users();
        f.register_secondary("users", vec![1]);
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::Int64(2));
    }

    #[tokio::test]
    async fn scan_secondary_lookup_falls_back_to_full_scan_without_index() {
        // No secondary index registered → reader answers Ok(None) → the
        // executor falls back to a full scan (all rows). The residual
        // Filter the planner keeps would re-narrow in a full plan; here we
        // assert the fallback streams every row rather than erroring.
        let plan = PhysicalPlan::Scan {
            table: TableId::new("users"),
            output_schema: users_schema(),
            projection: None,
            predicate: None,
            limit: None,
            access: ScanAccess::SecondaryLookup {
                column: "name".into(),
                values: vec![Expr::literal(ProximaValue::String("bob".into()))],
            },
        };
        let f = factory_with_users(); // NO register_secondary
        let mut exec = build_executor(plan, &f, &ExecutionContext::default()).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 3, "fallback full scan emits all rows");
    }

    // ----- Values ------------------------------------------------------

    #[tokio::test]
    async fn values_yields_inline_rows() {
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let plan = PhysicalPlan::Values {
            rows: vec![
                vec![Expr::literal(ProximaValue::Int64(1))],
                vec![Expr::literal(ProximaValue::Int64(2))],
            ],
            output_schema: schema,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], ProximaValue::Int64(1));
        assert_eq!(rows[1][0], ProximaValue::Int64(2));
    }

    // ----- Scalar functions (F1b) -------------------------------------

    #[tokio::test]
    async fn project_dispatches_scalar_function_through_builtins() {
        // F1b: a projected `Expr::FuncCall` now dispatches through the shared builtin
        // registry. Before wiring, the executor passed `NoFunctions`, so every scalar
        // call raised `UnknownFunction`. `upper('ab')`→"AB"; `concat('x','y')`→"xy".
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let values = PhysicalPlan::Values {
            rows: vec![vec![Expr::literal(ProximaValue::Int64(0))]],
            output_schema: schema,
        };
        let plan = PhysicalPlan::Project {
            input: Box::new(values),
            outputs: vec![
                NamedExpr::new(
                    "u",
                    Expr::FuncCall {
                        name: "upper".into(),
                        args: vec![Expr::literal(ProximaValue::String("ab".into()))],
                        return_ty: ProximaType::String,
                    },
                ),
                NamedExpr::new(
                    "c",
                    Expr::FuncCall {
                        name: "concat".into(),
                        args: vec![
                            Expr::literal(ProximaValue::String("x".into())),
                            Expr::literal(ProximaValue::String("y".into())),
                        ],
                        return_ty: ProximaType::String,
                    },
                ),
            ],
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], ProximaValue::String("AB".into()));
        assert_eq!(rows[0][1], ProximaValue::String("xy".into()));
    }

    #[tokio::test]
    async fn unknown_scalar_function_errors_at_execution() {
        // Unknown functions lower permissively in the frontend; the Volcano path raises a
        // clear error at execution (not silently null).
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let plan = PhysicalPlan::Project {
            input: Box::new(PhysicalPlan::Values {
                rows: vec![vec![Expr::literal(ProximaValue::Int64(0))]],
                output_schema: schema,
            }),
            outputs: vec![NamedExpr::new(
                "z",
                Expr::FuncCall {
                    name: "no_such_fn".into(),
                    args: vec![],
                    return_ty: ProximaType::String,
                },
            )],
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let err = collect(&mut *exec).await.unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("no_such_fn") || msg.to_lowercase().contains("unknown"),
            "expected UnknownFunction-style error, got {msg}"
        );
    }

    // ----- Union -------------------------------------------------------

    #[tokio::test]
    async fn union_all_concatenates_children() {
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let plan = PhysicalPlan::Union {
            inputs: vec![
                PhysicalPlan::Values {
                    rows: vec![vec![Expr::literal(ProximaValue::Int64(1))]],
                    output_schema: schema.clone(),
                },
                PhysicalPlan::Values {
                    rows: vec![vec![Expr::literal(ProximaValue::Int64(1))]],
                    output_schema: schema.clone(),
                },
            ],
            all: true,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn union_distinct_dedupes_across_children() {
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let plan = PhysicalPlan::Union {
            inputs: vec![
                PhysicalPlan::Values {
                    rows: vec![vec![Expr::literal(ProximaValue::Int64(1))]],
                    output_schema: schema.clone(),
                },
                PhysicalPlan::Values {
                    rows: vec![vec![Expr::literal(ProximaValue::Int64(1))]],
                    output_schema: schema.clone(),
                },
            ],
            all: false,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default();
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        let rows = collect(&mut *exec).await.unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn executor_wrapper_observes_mid_scan_cancellation() {
        let flag = Arc::new(AtomicBool::new(false));
        let schema = RelationalSchema::new(vec![ColumnInfo::new("x", ProximaType::Int64, false)]);
        let rows = (0..5000)
            .map(|value| vec![Expr::literal(ProximaValue::Int64(value))])
            .collect();
        let plan = PhysicalPlan::Values {
            rows,
            output_schema: schema,
        };
        let f = factory_with_users();
        let ctx = ExecutionContext::default().with_cancellation_flag(Some(flag.clone()));
        let mut exec = build_executor(plan, &f, &ctx).unwrap();
        exec.open().await.unwrap();
        assert!(exec.next_row().await.unwrap().is_some());

        flag.store(true, Ordering::Relaxed);
        let mut cancelled = false;
        for _ in 0..5000 {
            match exec.next_row().await {
                Err(ExecError::Cancelled) => {
                    cancelled = true;
                    break;
                }
                Ok(Some(_)) => {}
                other => panic!("expected cancellation before exhaustion, got {other:?}"),
            }
        }
        assert!(
            cancelled,
            "cancellation must be observed within the pull stride"
        );
    }

    // ----- Aggregate accumulators (direct unit) -----------------------

    #[test]
    fn count_skips_null_for_count_expr() {
        let agg = AggExpr::Count {
            arg: Some(Expr::Column(ColumnRef {
                name: "x".into(),
                ordinal: 0,
                ty: ProximaType::Int64,
                nullable: true,
            })),
            distinct: false,
        };
        let mut acc = Accumulator::new(&agg);
        acc.accumulate(&agg, &vec![ProximaValue::Int64(1)]).unwrap();
        acc.accumulate(&agg, &vec![ProximaValue::Null]).unwrap();
        acc.accumulate(&agg, &vec![ProximaValue::Int64(2)]).unwrap();
        // COUNT(x) ignores NULL → 2.
        assert_eq!(acc.finalize(), ProximaValue::Int64(2));
    }

    #[test]
    fn count_star_counts_all_rows_including_null() {
        let agg = AggExpr::Count {
            arg: None,
            distinct: false,
        };
        let mut acc = Accumulator::new(&agg);
        acc.accumulate(&agg, &vec![ProximaValue::Null]).unwrap();
        acc.accumulate(&agg, &vec![ProximaValue::Int64(1)]).unwrap();
        // COUNT(*) → 2.
        assert_eq!(acc.finalize(), ProximaValue::Int64(2));
    }
}
