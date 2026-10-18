// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! # Native vectorized operators + lowering (ADR-054 Phase 2, TD-OLAP-10)
//!
//! The mechanics behind `native_engine::try_vectorized`:
//! * [`PhysExpr`] — the zero-DataFusion expression IR (the ADR-054 v2 BLOCKER-2
//!   fix), lowered from `proximadb_relational_types::Expr` and evaluated on an
//!   Arrow `RecordBatch` via `arrow` compute kernels (comparison + Kleene
//!   boolean + null-test).
//! * [`MemoryScanSource`] — an [`ExecutionOperator`] that emits in-memory
//!   `RecordBatch`es built from a `PhysicalPlan::Values` literal table.
//! * [`FilterProjectOperator`] — a fused filter+project operator (ADR-054 §4.3):
//!   one pass per batch — predicate mask via `filter_record_batch`, then column
//!   selection. No intermediate `RecordBatch`.
//! * [`lower_physical`] — lowers a supported `PhysicalPlan` shape
//!   (`Project`/`Filter`/`Limit` over `Values`) to a [`Pipeline`].
//! * [`execute_pipeline`] — drains a [`Pipeline`] to `Vec<RecordBatch>`.
//!
//! All operators work in the *contracts crate's* error space
//! (`proximadb_execution_contracts::ExecutionError`); `native_engine` converts
//! to the engine error at the `try_vectorized` seam.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryBuilder, BooleanArray, BooleanBuilder, Float32Builder, Float64Builder,
    Int8Builder, Int16Builder, Int32Builder, Int64Builder, RecordBatch, StringBuilder,
    UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{and_kleene, filter_record_batch, is_not_null, is_null, not, or_kleene};
use arrow::datatypes::{DataType, Field, SchemaRef};
use async_trait::async_trait;
use futures::stream::StreamExt;
use proximadb_data_model::{ProximaType, ProximaValue};
use proximadb_execution_contracts::{BatchStream, ExecutionError, ExecutionOperator, Pipeline};
use proximadb_functions::builtins;
use proximadb_relational_algebra::{AggregateExpr, NamedExpr};
use proximadb_relational_planner::PhysicalPlan;
use proximadb_relational_types::{BinaryOp, Expr, RelationalSchema, UnaryOp};

// =========================================================================
// PhysExpr — the zero-DataFusion expression IR
// =========================================================================

/// A physical expression evaluated on an Arrow `RecordBatch`, with zero
/// DataFusion dependency. A strict subset of `Expr` — lowering returns `Err`
/// for anything outside this set, which `try_vectorized` turns into a Volcano
/// fallback (never a hard failure of the experimental path).
#[derive(Debug, Clone)]
enum PhysExpr {
    /// `Expr::Column(ColumnRef)` — ordinals are pre-resolved at planning time.
    Column(usize),
    /// `Expr::Literal { value, .. }` — broadcast to the batch row count at eval.
    Literal(ProximaValue),
    /// The six comparison `BinaryOp`s, evaluated via `arrow::compute::kernels::cmp`.
    Compare {
        op: BinaryOp,
        left: Box<PhysExpr>,
        right: Box<PhysExpr>,
    },
    /// SQL `AND` — Kleene (NULL-aware), via `and_kleene`.
    And(Box<PhysExpr>, Box<PhysExpr>),
    /// SQL `OR` — Kleene (NULL-aware), via `or_kleene`.
    Or(Box<PhysExpr>, Box<PhysExpr>),
    /// SQL `NOT` — null-propagating, via `not`.
    Not(Box<PhysExpr>),
    /// `IS NULL` / `IS NOT NULL`, via `is_null` / `is_not_null`.
    IsNull { expr: Box<PhysExpr>, negated: bool },
}

impl PhysExpr {
    /// Lower a relational `Expr` to a `PhysExpr`. Returns `Err(NotImplemented)`
    /// for any unsupported variant — the caller (`try_vectorized`) maps that to
    /// a Volcano fallback.
    fn lower(expr: &Expr) -> Result<Self, ExecutionError> {
        match expr {
            Expr::Column(col) => Ok(PhysExpr::Column(col.ordinal)),
            Expr::Literal { value, .. } => Ok(PhysExpr::Literal(value.clone())),
            Expr::BinaryOp { op, left, right } => {
                let l = Box::new(Self::lower(left)?);
                let r = Box::new(Self::lower(right)?);
                match op {
                    BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::LtEq
                    | BinaryOp::Gt
                    | BinaryOp::GtEq => Ok(PhysExpr::Compare {
                        op: *op,
                        left: l,
                        right: r,
                    }),
                    BinaryOp::And => Ok(PhysExpr::And(l, r)),
                    BinaryOp::Or => Ok(PhysExpr::Or(l, r)),
                    other => Err(ExecutionError::NotImplemented(format!(
                        "binary op {other:?} not supported in native vectorized predicate"
                    ))),
                }
            }
            Expr::IsNull { expr, not } => Ok(PhysExpr::IsNull {
                expr: Box::new(Self::lower(expr)?),
                negated: *not,
            }),
            Expr::UnaryOp { op, expr } => match op {
                UnaryOp::Not => Ok(PhysExpr::Not(Box::new(Self::lower(expr)?))),
                UnaryOp::Neg => Err(ExecutionError::NotImplemented(
                    "unary Neg not supported in native vectorized predicate".into(),
                )),
            },
            other => Err(ExecutionError::NotImplemented(format!(
                "expr {other:?} not supported in native vectorized predicate"
            ))),
        }
    }

    /// Evaluate to a column array against `batch`. `Literal` is broadcast to the
    /// batch row count so that comparison operands share a length (the arrow
    /// `cmp` kernels require equal-length arrays).
    fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, ExecutionError> {
        match self {
            PhysExpr::Column(idx) => {
                let col = batch.column(*idx);
                if *idx >= batch.num_columns() {
                    return Err(ExecutionError::Schema(format!(
                        "column ordinal {idx} out of range ({} columns)",
                        batch.num_columns()
                    )));
                }
                Ok(col.clone())
            }
            PhysExpr::Literal(v) => broadcast_literal(v, batch.num_rows()),
            PhysExpr::Compare { op, left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                compare_arrays(op, &l, &r)
            }
            PhysExpr::And(l, r) => {
                let lv = l.eval_bool(batch)?;
                let rv = r.eval_bool(batch)?;
                let out = and_kleene(&lv, &rv)
                    .map_err(|e| ExecutionError::Execution(format!("and_kleene: {e}")))?;
                Ok(Arc::new(out))
            }
            PhysExpr::Or(l, r) => {
                let lv = l.eval_bool(batch)?;
                let rv = r.eval_bool(batch)?;
                let out = or_kleene(&lv, &rv)
                    .map_err(|e| ExecutionError::Execution(format!("or_kleene: {e}")))?;
                Ok(Arc::new(out))
            }
            PhysExpr::Not(e) => {
                let v = e.eval_bool(batch)?;
                let out = not(&v).map_err(|e| ExecutionError::Execution(format!("not: {e}")))?;
                Ok(Arc::new(out))
            }
            PhysExpr::IsNull { expr, negated } => {
                let a = expr.eval(batch)?;
                let out = if *negated {
                    is_not_null(a.as_ref())
                } else {
                    is_null(a.as_ref())
                }
                .map_err(|e| ExecutionError::Execution(format!("is_null: {e}")))?;
                Ok(Arc::new(out))
            }
        }
    }

    /// Evaluate as a boolean mask (downcasts the result of [`Self::eval`]).
    fn eval_bool(&self, batch: &RecordBatch) -> Result<BooleanArray, ExecutionError> {
        let arr = self.eval(batch)?;
        arr.as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
            .ok_or_else(|| {
                ExecutionError::Schema(format!(
                    "predicate evaluated to non-boolean array: {:?}",
                    arr.data_type()
                ))
            })
    }
}

/// Broadcast a scalar `ProximaValue` to a length-`n` array. Used so comparison
/// operands share a length (the arrow `cmp` kernels are array-vs-array). A
/// later phase can swap this for arrow's scalar/Datum comparison to avoid the
/// allocation; correctness-first here.
fn broadcast_literal(value: &ProximaValue, n: usize) -> Result<ArrayRef, ExecutionError> {
    match value {
        ProximaValue::Null => Err(ExecutionError::NotImplemented(
            "null literal in comparison not supported (use IS NULL)".into(),
        )),
        ProximaValue::Boolean(b) => Ok(Arc::new(BooleanArray::from(vec![*b; n]))),
        ProximaValue::Int8(x) => Ok(Arc::new(arrow::array::Int8Array::from_value(*x, n))),
        ProximaValue::Int16(x) => Ok(Arc::new(arrow::array::Int16Array::from_value(*x, n))),
        ProximaValue::Int32(x) => Ok(Arc::new(arrow::array::Int32Array::from_value(*x, n))),
        ProximaValue::Int64(x) => Ok(Arc::new(arrow::array::Int64Array::from_value(*x, n))),
        ProximaValue::UInt8(x) => Ok(Arc::new(arrow::array::UInt8Array::from_value(*x, n))),
        ProximaValue::UInt16(x) => Ok(Arc::new(arrow::array::UInt16Array::from_value(*x, n))),
        ProximaValue::UInt32(x) => Ok(Arc::new(arrow::array::UInt32Array::from_value(*x, n))),
        ProximaValue::UInt64(x) => Ok(Arc::new(arrow::array::UInt64Array::from_value(*x, n))),
        ProximaValue::Float32(x) => Ok(Arc::new(arrow::array::Float32Array::from_value(*x, n))),
        ProximaValue::Float64(x) => Ok(Arc::new(arrow::array::Float64Array::from_value(*x, n))),
        ProximaValue::String(s) => Ok(Arc::new(arrow::array::StringArray::from_iter_values(
            std::iter::repeat_n(s.as_str(), n),
        ))),
        other => Err(ExecutionError::NotImplemented(format!(
            "literal type {other:?} not supported in native vectorized predicate"
        ))),
    }
}

/// Apply a comparison `BinaryOp` to two equal-length arrays via the arrow-ord
/// `cmp` kernels (which take `&dyn Datum`). `&dyn Array` implements `Datum`, so
/// a reference to it coerces to `&dyn Datum`.
fn compare_arrays(
    op: &BinaryOp,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef, ExecutionError> {
    let l: &dyn Array = left.as_ref();
    let r: &dyn Array = right.as_ref();
    let mask = match op {
        BinaryOp::Eq => cmp::eq(&l, &r),
        BinaryOp::NotEq => cmp::neq(&l, &r),
        BinaryOp::Lt => cmp::lt(&l, &r),
        BinaryOp::LtEq => cmp::lt_eq(&l, &r),
        BinaryOp::Gt => cmp::gt(&l, &r),
        BinaryOp::GtEq => cmp::gt_eq(&l, &r),
        other => {
            return Err(ExecutionError::NotImplemented(format!(
                "comparison op {other:?} not supported"
            )));
        }
    }
    .map_err(|e| ExecutionError::Execution(format!("cmp {op:?}: {e}")))?;
    Ok(Arc::new(mask))
}

// =========================================================================
// MemoryScanSource — in-memory RecordBatch source (from PhysicalPlan::Values)
// =========================================================================

/// A scan source backed by in-memory `RecordBatch`es. Built once from a
/// `PhysicalPlan::Values` literal table. Phase 2.5 swaps this for a real scan
/// operator behind the same [`ExecutionOperator`] seam (non-breaking).
#[derive(Debug)]
pub(crate) struct MemoryScanSource {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
}

impl MemoryScanSource {
    fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> Self {
        Self { batches, schema }
    }
}

#[async_trait]
impl ExecutionOperator for MemoryScanSource {
    fn output_schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn execute(&self, _input: BatchStream) -> Result<BatchStream, ExecutionError> {
        let batches = self.batches.clone();
        let stream = futures::stream::iter(batches.into_iter().map(Ok::<_, ExecutionError>));
        Ok(Box::pin(stream))
    }
}

// =========================================================================
// FilterProjectOperator — fused filter + project (ADR-054 §4.3)
// =========================================================================

/// Fused filter + project: for each input batch, optionally applies a predicate
/// mask (`filter_record_batch`) and optionally selects a subset of columns
/// (`RecordBatch::project`) — in one pass, no intermediate batch. Either or
/// both of `predicate`/`projection` may be `None`.
#[derive(Debug)]
pub(crate) struct FilterProjectOperator {
    predicate: Option<PhysExpr>,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
}

impl FilterProjectOperator {
    fn new(
        predicate: Option<PhysExpr>,
        projection: Option<Vec<usize>>,
        output_schema: SchemaRef,
    ) -> Self {
        Self {
            predicate,
            projection,
            output_schema,
        }
    }
}

#[async_trait]
impl ExecutionOperator for FilterProjectOperator {
    fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    async fn execute(&self, input: BatchStream) -> Result<BatchStream, ExecutionError> {
        let predicate = self.predicate.clone();
        let projection = self.projection.clone();
        Ok(Box::pin(input.map(move |result| {
            let mut batch = result?;
            if let Some(pred) = &predicate {
                let mask = pred.eval_bool(&batch)?;
                batch = filter_record_batch(&batch, &mask)
                    .map_err(|e| ExecutionError::Execution(format!("filter: {e}")))?;
            }
            if let Some(proj) = &projection {
                batch = batch
                    .project(proj)
                    .map_err(|e| ExecutionError::Execution(format!("project: {e}")))?;
            }
            Ok(batch)
        })))
    }
}

// =========================================================================
// HashAggregateOperator — blocking GROUP BY + COUNT/SUM/AVG/MIN/MAX (Phase 2.1)
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggKind {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// One aggregate to compute. `arg` is the input column ordinal; `None` = `COUNT(*)`.
#[derive(Debug, Clone)]
struct AggSpec {
    arg: Option<usize>,
    kind: AggKind,
}

/// A blocking hash-aggregate operator (ADR-054 Phase 2.1). Consumes its entire
/// input, builds a `group_by → accumulators` hash table, then emits one row per
/// group. MVP scope: `COUNT(*)/COUNT(col)/SUM/AVG/MIN/MAX`, non-distinct,
/// column-argument only, no `HAVING` (the planner's `having` declines → Volcano).
///
/// Competitiveness vs the Volcano: the Volcano calls `Expr::eval` *per cell*;
/// this operator pre-extracts each column once and reads cells via a direct
/// typed downcast (`arrow_cell_to_proxima`), avoiding the per-cell expression
/// evaluation + function-registry lookup.
#[derive(Debug)]
pub(crate) struct HashAggregateOperator {
    group_by: Vec<usize>,
    aggregates: Vec<AggSpec>,
    output_schema: SchemaRef,
}

impl HashAggregateOperator {
    fn new(group_by: Vec<usize>, aggregates: Vec<AggSpec>, output_schema: SchemaRef) -> Self {
        Self {
            group_by,
            aggregates,
            output_schema,
        }
    }
}

#[async_trait]
impl ExecutionOperator for HashAggregateOperator {
    fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn is_blocking(&self) -> bool {
        true
    }

    async fn execute(&self, mut input: BatchStream) -> Result<BatchStream, ExecutionError> {
        let group_by = self.group_by.clone();
        let aggregates = self.aggregates.clone();
        let output_schema = self.output_schema.clone();

        // Vectorized fast path for UNGROUPED scalar aggregates (no GROUP BY): fold
        // Arrow aggregate kernels per batch instead of the row-wise accumulator
        // loop. This is the dominant OLAP scalar-aggregate shape and was measured
        // 3-31x slower row-wise at 100M (TD-OLAP-4). Grouped aggregation keeps the
        // row-wise path below (high-cardinality GROUP BY is routed to DataFusion).
        if group_by.is_empty() {
            return ungrouped_vectorized(input, &aggregates, output_schema).await;
        }

        // group key → (group_by values, per-agg accumulators)
        let mut groups: std::collections::HashMap<GroupKey, (Vec<ProximaValue>, Vec<Accumulator>)> =
            std::collections::HashMap::new();

        // Stream the input: fold each batch into the hash state, then DROP it —
        // bounded memory (one batch in flight + the group state), never
        // materializing the whole input. Collecting all input first OOMs at scale
        // (a full-width 100M scan is tens of GB). Group state is still unbounded in
        // the number of distinct groups — high-cardinality GROUP BY is routed away
        // upstream until a spilling aggregate lands.
        while let Some(batch) = input.next().await {
            let batch = batch?;
            let nrows = batch.num_rows();
            for r in 0..nrows {
                // Build the group key from the group_by columns.
                let key_vals: Vec<ProximaValue> = group_by
                    .iter()
                    .map(|&c| cell_value(batch.column(c).as_ref(), r))
                    .collect::<Result<_, _>>()?;
                let key = GroupKey::from_values(&key_vals);
                let entry = groups.entry(key).or_insert_with(|| {
                    (
                        key_vals.clone(),
                        aggregates
                            .iter()
                            .map(|s| Accumulator::new(s.kind, s.arg.is_some()))
                            .collect(),
                    )
                });
                // Accumulate each aggregate's argument.
                for (spec, acc) in aggregates.iter().zip(entry.1.iter_mut()) {
                    let v = match spec.arg {
                        Some(c) => cell_value(batch.column(c).as_ref(), r)?,
                        None => ProximaValue::Boolean(true), // COUNT(*): a non-null sentinel
                    };
                    acc.accumulate(spec.kind, v)?;
                }
            }
        }

        // Emit one row per group: [group_by values..., finalized aggregates...].
        let ngroups = groups.len();
        let ncols = group_by.len() + aggregates.len();
        let mut columns: Vec<Vec<ProximaValue>> =
            (0..ncols).map(|_| Vec::with_capacity(ngroups)).collect();
        // Stable order: sort group keys by their first component's raw ordering when
        // possible, else by insertion (HashMap order is nondeterministic). For
        // determinism in tests, sort by the group_by value ordering.
        let mut rows: Vec<(Vec<ProximaValue>, Vec<Accumulator>)> = groups.into_values().collect();
        rows.sort_by(|a, b| {
            for (x, y) in a.0.iter().zip(b.0.iter()) {
                let ord = compare_values(x, y);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
        for (gb_vals, accs) in rows {
            for (i, v) in gb_vals.into_iter().enumerate() {
                columns[i].push(v);
            }
            for (j, acc) in accs.iter().enumerate() {
                columns[group_by.len() + j].push(acc.finalize());
            }
        }

        // Build each output column as a typed Arrow array.
        let arrays: Vec<ArrayRef> = (0..columns.len())
            .map(|c| build_column_array(&columns[c], output_schema.field(c).data_type()))
            .collect::<Result<_, _>>()?;
        let batch = RecordBatch::try_new(output_schema, arrays)
            .map_err(|e| ExecutionError::Execution(format!("aggregate output: {e}")))?;
        Ok(Box::pin(futures::stream::once(async move { Ok(batch) })))
    }
}

// =========================================================================
// Vectorized UNGROUPED aggregation (TD-OLAP-4): Arrow kernels per batch instead
// of the row-wise Accumulator loop — the 3-31x compute win on scalar aggregates.
// Finalization semantics MATCH the row-wise `Accumulator::finalize` exactly
// (Count→Int64, Sum→Float64|Null, Avg→Float64|Null, Min/Max→value|Null).
// =========================================================================

/// Running state for one ungrouped aggregate, folded across batches.
enum UngroupedState {
    Count { n: i64 },
    Sum { sum: f64, any: bool },
    Avg { sum: f64, n: i64 },
    Min { cur: Option<ProximaValue> },
    Max { cur: Option<ProximaValue> },
}

impl UngroupedState {
    fn new(kind: AggKind) -> Self {
        match kind {
            AggKind::Count => Self::Count { n: 0 },
            AggKind::Sum => Self::Sum {
                sum: 0.0,
                any: false,
            },
            AggKind::Avg => Self::Avg { sum: 0.0, n: 0 },
            AggKind::Min => Self::Min { cur: None },
            AggKind::Max => Self::Max { cur: None },
        }
    }

    fn update(&mut self, arg: Option<usize>, batch: &RecordBatch) -> Result<(), ExecutionError> {
        let need_col = || {
            arg.ok_or_else(|| {
                ExecutionError::Execution("SUM/AVG/MIN/MAX requires a column argument".into())
            })
        };
        match self {
            // COUNT(*): every row; COUNT(col): non-null rows (null_count from the array).
            Self::Count { n } => match arg {
                None => *n += batch.num_rows() as i64,
                Some(c) => {
                    let col = batch.column(c);
                    *n += (col.len() - col.null_count()) as i64;
                }
            },
            Self::Sum { sum, any } => {
                let (s, cnt) = array_sum_count(batch.column(need_col()?))?;
                *sum += s;
                if cnt > 0 {
                    *any = true;
                }
            }
            Self::Avg { sum, n } => {
                let (s, cnt) = array_sum_count(batch.column(need_col()?))?;
                *sum += s;
                *n += cnt;
            }
            Self::Min { cur } => {
                if let Some(v) = array_min_max_proxima(batch.column(need_col()?), false)? {
                    match cur {
                        None => *cur = Some(v),
                        Some(c) if compare_values(&v, c) == std::cmp::Ordering::Less => {
                            *cur = Some(v)
                        }
                        _ => {}
                    }
                }
            }
            Self::Max { cur } => {
                if let Some(v) = array_min_max_proxima(batch.column(need_col()?), true)? {
                    match cur {
                        None => *cur = Some(v),
                        Some(c) if compare_values(&v, c) == std::cmp::Ordering::Greater => {
                            *cur = Some(v)
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    fn finalize(&self) -> ProximaValue {
        match self {
            Self::Count { n } => ProximaValue::Int64(*n),
            Self::Sum { sum, any } => {
                if *any {
                    ProximaValue::Float64(*sum)
                } else {
                    ProximaValue::Null
                }
            }
            Self::Avg { sum, n } => {
                if *n == 0 {
                    ProximaValue::Null
                } else {
                    ProximaValue::Float64(sum / (*n as f64))
                }
            }
            Self::Min { cur } | Self::Max { cur } => cur.clone().unwrap_or(ProximaValue::Null),
        }
    }
}

/// Vectorized SUM + non-null count for one column: cast to `Float64` (matching the
/// row-wise `numeric_to_f64` semantics) then the Arrow `sum` kernel.
fn array_sum_count(col: &dyn arrow::array::Array) -> Result<(f64, i64), ExecutionError> {
    let casted = arrow::compute::cast(col, &DataType::Float64)
        .map_err(|e| ExecutionError::Execution(format!("SUM/AVG cast to f64: {e}")))?;
    let a = casted
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| ExecutionError::Execution("f64 downcast".into()))?;
    let s = arrow::compute::kernels::aggregate::sum(a).unwrap_or(0.0);
    let n = (a.len() - a.null_count()) as i64;
    Ok((s, n))
}

/// Vectorized MIN (or MAX) of one column as a `ProximaValue`, preserving the
/// column type (type-dispatched Arrow `min`/`max` kernel). `None` on an all-null
/// or empty batch.
fn array_min_max_proxima(
    col: &dyn arrow::array::Array,
    want_max: bool,
) -> Result<Option<ProximaValue>, ExecutionError> {
    use arrow::array::*;
    use arrow::compute::kernels::aggregate as agg;
    macro_rules! mm {
        ($ArrTy:ty, $Prox:path) => {{
            let a = col
                .as_any()
                .downcast_ref::<$ArrTy>()
                .ok_or_else(|| ExecutionError::Execution("MIN/MAX downcast".into()))?;
            let r = if want_max { agg::max(a) } else { agg::min(a) };
            Ok(r.map($Prox))
        }};
    }
    match col.data_type() {
        DataType::Int8 => mm!(Int8Array, ProximaValue::Int8),
        DataType::Int16 => mm!(Int16Array, ProximaValue::Int16),
        DataType::Int32 => mm!(Int32Array, ProximaValue::Int32),
        DataType::Int64 => mm!(Int64Array, ProximaValue::Int64),
        DataType::UInt8 => mm!(UInt8Array, ProximaValue::UInt8),
        DataType::UInt16 => mm!(UInt16Array, ProximaValue::UInt16),
        DataType::UInt32 => mm!(UInt32Array, ProximaValue::UInt32),
        DataType::UInt64 => mm!(UInt64Array, ProximaValue::UInt64),
        DataType::Float32 => mm!(Float32Array, ProximaValue::Float32),
        DataType::Float64 => mm!(Float64Array, ProximaValue::Float64),
        DataType::Boolean => {
            let a = col
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| ExecutionError::Execution("MIN/MAX downcast".into()))?;
            let r = if want_max {
                agg::max_boolean(a)
            } else {
                agg::min_boolean(a)
            };
            Ok(r.map(ProximaValue::Boolean))
        }
        DataType::Utf8 => {
            let a = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ExecutionError::Execution("MIN/MAX downcast".into()))?;
            let r = if want_max {
                agg::max_string(a)
            } else {
                agg::min_string(a)
            };
            Ok(r.map(|s| ProximaValue::String(s.to_string())))
        }
        other => Err(ExecutionError::NotImplemented(format!(
            "vectorized MIN/MAX for {other:?}"
        ))),
    }
}

/// Fold the ungrouped aggregates over the input stream with Arrow kernels, then
/// emit the single result row. Bounded memory (one batch + O(#aggregates) state).
async fn ungrouped_vectorized(
    mut input: BatchStream,
    aggregates: &[AggSpec],
    output_schema: SchemaRef,
) -> Result<BatchStream, ExecutionError> {
    let mut states: Vec<UngroupedState> = aggregates
        .iter()
        .map(|s| UngroupedState::new(s.kind))
        .collect();
    while let Some(batch) = input.next().await {
        let batch = batch?;
        for (spec, st) in aggregates.iter().zip(states.iter_mut()) {
            st.update(spec.arg, &batch)?;
        }
    }
    let vals: Vec<ProximaValue> = states.iter().map(|s| s.finalize()).collect();
    let arrays: Vec<ArrayRef> = vals
        .iter()
        .enumerate()
        .map(|(i, v)| {
            build_column_array(std::slice::from_ref(v), output_schema.field(i).data_type())
        })
        .collect::<Result<_, _>>()?;
    let batch = RecordBatch::try_new(output_schema, arrays)
        .map_err(|e| ExecutionError::Execution(format!("aggregate output: {e}")))?;
    Ok(Box::pin(futures::stream::once(async move { Ok(batch) })))
}

/// Per-group, per-aggregate accumulator. Mirrors the Volcano `Accumulator`'s NULL
/// semantics (COUNT(*) counts rows; COUNT(col)/SUM/AVG/MIN/MAX ignore NULLs;
/// SUM of all-NULL → NULL; AVG of none → NULL).
enum Accumulator {
    Count { skip_null: bool, n: i64 },
    Sum { running: Option<f64> },
    Avg { sum: f64, n: i64 },
    Min { current: Option<ProximaValue> },
    Max { current: Option<ProximaValue> },
}

impl Accumulator {
    /// `count_skip_null`: for `COUNT(col)` (skip nulls in that column); false for
    /// `COUNT(*)` (count every row).
    fn new(kind: AggKind, count_skip_null: bool) -> Self {
        match kind {
            AggKind::Count => Accumulator::Count {
                skip_null: count_skip_null,
                n: 0,
            },
            AggKind::Sum => Accumulator::Sum { running: None },
            AggKind::Avg => Accumulator::Avg { sum: 0.0, n: 0 },
            AggKind::Min => Accumulator::Min { current: None },
            AggKind::Max => Accumulator::Max { current: None },
        }
    }

    fn accumulate(&mut self, kind: AggKind, v: ProximaValue) -> Result<(), ExecutionError> {
        match (&mut *self, kind) {
            (Accumulator::Count { skip_null, n }, AggKind::Count) => {
                if *skip_null && matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                *n += 1;
                Ok(())
            }
            (Accumulator::Sum { running }, AggKind::Sum) => {
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                let f = numeric_to_f64(&v)?;
                *running = Some(running.unwrap_or(0.0) + f);
                Ok(())
            }
            (Accumulator::Avg { sum, n }, AggKind::Avg) => {
                if matches!(v, ProximaValue::Null) {
                    return Ok(());
                }
                *sum += numeric_to_f64(&v)?;
                *n += 1;
                Ok(())
            }
            (Accumulator::Min { current }, AggKind::Min) => {
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
            (Accumulator::Max { current }, AggKind::Max) => {
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
            _ => Err(ExecutionError::Execution(
                "accumulator/aggregate kind mismatch".into(),
            )),
        }
    }

    fn finalize(&self) -> ProximaValue {
        match self {
            Accumulator::Count { n, .. } => ProximaValue::Int64(*n),
            Accumulator::Sum { running } => match running {
                Some(f) => ProximaValue::Float64(*f),
                None => ProximaValue::Null,
            },
            Accumulator::Avg { sum, n } => {
                if *n == 0 {
                    ProximaValue::Null
                } else {
                    ProximaValue::Float64(sum / (*n as f64))
                }
            }
            Accumulator::Min { current } | Accumulator::Max { current } => {
                current.clone().unwrap_or(ProximaValue::Null)
            }
        }
    }
}

// --- hashable group key (ProximaValue is PartialEq only; f64 needs bit-encoding) ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyComponent {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(u64), // f64::to_bits — NB: distinguishes -0.0/+0.0 (acceptable for grouping)
    Str(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<KeyComponent>);

impl GroupKey {
    fn from_values(values: &[ProximaValue]) -> Self {
        Self(values.iter().map(proxima_to_key).collect())
    }
}

fn proxima_to_key(v: &ProximaValue) -> KeyComponent {
    match v {
        ProximaValue::Null => KeyComponent::Null,
        ProximaValue::Boolean(b) => KeyComponent::Bool(*b),
        ProximaValue::Int8(x) => KeyComponent::Int(*x as i64),
        ProximaValue::Int16(x) => KeyComponent::Int(*x as i64),
        ProximaValue::Int32(x) => KeyComponent::Int(*x as i64),
        ProximaValue::Int64(x) => KeyComponent::Int(*x),
        ProximaValue::UInt8(x) => KeyComponent::UInt(*x as u64),
        ProximaValue::UInt16(x) => KeyComponent::UInt(*x as u64),
        ProximaValue::UInt32(x) => KeyComponent::UInt(*x as u64),
        ProximaValue::UInt64(x) => KeyComponent::UInt(*x),
        ProximaValue::Float32(x) => KeyComponent::Float((*x as f64).to_bits()),
        ProximaValue::Float64(x) => KeyComponent::Float(x.to_bits()),
        ProximaValue::String(s) => KeyComponent::Str(s.clone()),
        ProximaValue::Binary(b) => KeyComponent::Bytes(b.clone()),
        other => KeyComponent::Str(format!("{other:?}")), // fallback: stringify (Date/Timestamp/etc.)
    }
}

/// Extract one cell as a `ProximaValue` (reuses the Phase-1 conversion so the
/// native engine has one Arrow→ProximaValue path).
fn cell_value(array: &dyn arrow::array::Array, row: usize) -> Result<ProximaValue, ExecutionError> {
    Ok(super::native_engine::arrow_cell_to_proxima(array, row))
}

/// Numeric coercion to f64 for SUM/AVG.
fn numeric_to_f64(v: &ProximaValue) -> Result<f64, ExecutionError> {
    match v {
        ProximaValue::Int8(x) => Ok(*x as f64),
        ProximaValue::Int16(x) => Ok(*x as f64),
        ProximaValue::Int32(x) => Ok(*x as f64),
        ProximaValue::Int64(x) => Ok(*x as f64),
        ProximaValue::UInt8(x) => Ok(*x as f64),
        ProximaValue::UInt16(x) => Ok(*x as f64),
        ProximaValue::UInt32(x) => Ok(*x as f64),
        ProximaValue::UInt64(x) => Ok(*x as f64),
        ProximaValue::Float32(x) => Ok(*x as f64),
        ProximaValue::Float64(x) => Ok(*x),
        other => Err(ExecutionError::Execution(format!(
            "non-numeric value in SUM/AVG: {other:?}"
        ))),
    }
}

/// Total ordering over `ProximaValue` for MIN/MAX + deterministic group output.
fn compare_values(a: &ProximaValue, b: &ProximaValue) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    // NULLs sort first (and are equal to each other).
    let an = matches!(a, ProximaValue::Null);
    let bn = matches!(b, ProximaValue::Null);
    match (an, bn) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        (false, false) => {}
    }
    let af = numeric_to_f64(a).ok();
    let bf = numeric_to_f64(b).ok();
    if let (Some(x), Some(y)) = (af, bf) {
        return x.partial_cmp(&y).unwrap_or(Ordering::Equal);
    }
    // Fall back to Debug-string ordering for non-numeric (strings, etc.).
    format!("{a:?}").cmp(&format!("{b:?}"))
}

// =========================================================================
// PhysicalPlan → Pipeline lowering
// =========================================================================

/// A lowered pipeline plus an optional trailing `Limit { offset, count }`
/// (captured from a `PhysicalPlan::Limit` node and applied at collect).
#[derive(Debug)]
pub(crate) struct LoweredPipeline {
    /// The main (probe) pipeline.
    pub pipeline: Pipeline,
    /// An optional BUILD pipeline that must run + drain first (its blocking
    /// build operator publishes the hash table via a shared `OnceLock` consumed
    /// by a probe operator in `pipeline`). Set only by the `PhysicalPlan::Join`
    /// lowering (Phase 3, TD-OLAP-11).
    pub build_pipeline: Option<Pipeline>,
    pub limit: Option<LimitSpec>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LimitSpec {
    pub offset: u64,
    pub limit: Option<u64>,
}

/// One operator to build in the lowered chain. Held as a spec (not yet built) so
/// `walk` can fuse a contiguous Filter+Project into one `FilterProjectOperator`
/// (ADR-054 §4.3) before materializing.
enum OpSpec {
    FilterProject {
        predicate: Option<PhysExpr>,
        projection: Option<Vec<usize>>,
    },
    /// Blocking hash aggregate. `output_schema` is precomputed (group_by cols +
    /// agg result cols) so the operator and any post-aggregate projection agree.
    HashAggregate {
        group_by: Vec<usize>,
        aggregates: Vec<AggSpec>,
        output_schema: SchemaRef,
    },
}

/// Scan context for the native engine: pre-discovered PAX segments per table +
/// the filesystem to read them. Threaded through `lower_physical`/`walk` so the
/// `PhysicalPlan::Scan` arm can construct a `PaxScanOperator`. `None` when only
/// Values-backed queries are served (the default — Scan returns Err → Volcano).
pub struct ScanCtx {
    pub filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    pub tables: HashMap<String, ScanTableInfo>,
}

pub struct ScanTableInfo {
    pub splits: Vec<crate::storage::formats::FileSplit>,
    pub name_to_col_id: HashMap<String, i32>,
}

/// The pieces `walk` accumulates while descending a supported plan subtree.
/// `ops` is bottom-up: the source feeds the first, the last emits the output.
struct Walked {
    /// The scan leaf: `MemoryScanSource` for `Values`, a `PaxScanOperator` for a
    /// `Scan` with a threaded [`ScanCtx`] (PAX segments), or a
    /// `ParquetScanOperator` injected via [`lower_physical_over_source`] for a
    /// `Scan` over external parquet (TD-OLAP-4) — generalized so native serves
    /// real storage, not just literal tables.
    source: Box<dyn ExecutionOperator>,
    ops: Vec<OpSpec>,
    /// Output schema of the last op (or the source if `ops` is empty).
    cur_schema: SchemaRef,
    limit: Option<LimitSpec>,
}

thread_local! {
    /// Scan source injected by [`lower_physical_over_source`] for the `Scan` leaf
    /// (TD-OLAP-4 native-parquet). Set immediately before the sync `walk`, taken
    /// at the `Scan` arm, cleared after — never held across an await.
    static INJECTED_SCAN_SOURCE: std::cell::RefCell<Option<Box<dyn ExecutionOperator>>> =
        const { std::cell::RefCell::new(None) };
}

/// Lower `plan` to a native pipeline using `source` as the `Scan` leaf — the
/// entry the native-parquet shadow uses to run the SAME plan DataFusion runs, but
/// over a [`super::native_parquet_scan::ParquetScanOperator`]. Declines (like
/// `lower_physical`) for any op native does not support.
pub(crate) fn lower_physical_over_source(
    plan: &PhysicalPlan,
    source: Box<dyn ExecutionOperator>,
) -> Result<LoweredPipeline, ExecutionError> {
    INJECTED_SCAN_SOURCE.with(|s| *s.borrow_mut() = Some(source));
    // No `ScanCtx`: the `Scan` leaf resolves to the injected parquet source, which
    // the `Scan` arm checks before the PAX path.
    let result = lower_physical(plan, None);
    INJECTED_SCAN_SOURCE.with(|s| *s.borrow_mut() = None);
    result
}

/// Lower a `PhysicalPlan` to a [`LoweredPipeline`]. Returns `Err(NotImplemented)`
/// for any unsupported shape (Scan, Join, Sort, Distinct, SetOp, Union,
/// AssertMaxOneRow, nested Limit, non-column projection/aggregate-arg, distinct
/// aggregates, STRING_AGG/Custom aggregates, HAVING). `try_vectorized` maps the
/// `Err` to a Volcano fallback.
pub(crate) fn lower_physical(
    plan: &PhysicalPlan,
    scan_ctx: Option<&ScanCtx>,
) -> Result<LoweredPipeline, ExecutionError> {
    // Join is handled specially — it produces two pipelines (build + probe).
    if matches!(plan, PhysicalPlan::Join { .. }) {
        return lower_join(plan, scan_ctx);
    }
    let walked = walk(plan, scan_ctx)?;
    let (operators, limit) = walked_to_operators(walked);
    Ok(LoweredPipeline {
        pipeline: Pipeline::new(operators),
        build_pipeline: None,
        limit,
    })
}

// ADR-058 D1 — the *declared* capability of the native vectorized path: a
// shallow classification of which `PhysicalPlan` node variants the engine can
// lower. This is the single source of truth for "can native-vectorized handle
// this plan shape?" so the capability is stated once (and re-asserted by the
// drift test below), rather than re-derived by each caller.
//
// **Sound one-way filter:** a [`SupportLevel::Partial`] verdict GUARANTEES
// [`lower_physical`] declines — every [`SupportReason`] mirrors a concrete
// decline in `walk`/`lower_join`. [`SupportLevel::Full`] does NOT guarantee
// success: deeper constraints (distinct aggregates, non-column projections,
// equi-key extractability, scan-source availability, `MAX_WALK_DEPTH`) are
// checked only during lowering. Callers may safely short-circuit on `Partial`;
// on `Full` they must still attempt the lowering.
#[derive(Debug, Clone)]
pub(crate) enum SupportLevel {
    /// Every node variant on every path is in the supported set.
    Full,
    /// At least one unsupported node (or structural red-flag); carries the reasons.
    Partial(Vec<SupportReason>),
}

/// Why the vectorized path does not support a plan. Each variant mirrors a
/// concrete decline site in the lowering (`walk`/`lower_join`) — see the drift
/// test `vectorized_supports_drift_matches_lowering`.
#[derive(Debug, Clone)]
pub(crate) enum SupportReason {
    /// A `PhysicalPlan` variant the vectorized lowering does not handle
    /// ("Sort", "Distinct", "Union", "SetOp", "AssertMaxOneRow") — `walk`'s
    /// catch-all declines it.
    UnsupportedNode(&'static str),
    /// `Aggregate` with a `HAVING` clause — the native `HashAggregate` has no
    /// post-filter pass (`walk`'s Aggregate arm declines on `having.is_some()`).
    AggregateHaving,
    /// Two `Limit` nodes on one walk-path — the second is nested (`walk`'s Limit
    /// arm declines when `w.limit` is already set).
    NestedLimit,
}

impl SupportReason {
    /// Short tracing label — reads the carried node name for `UnsupportedNode`
    /// so the reason is observable in the decline log (not only via `Debug`).
    pub(crate) fn as_label(&self) -> &'static str {
        match self {
            SupportReason::UnsupportedNode(node) => node,
            SupportReason::AggregateHaving => "AggregateHaving",
            SupportReason::NestedLimit => "NestedLimit",
        }
    }
}

/// Classify a plan's support for the native vectorized path. Walks the same
/// node set `walk`/`lower_join` handle. `limit_count` is reset at each `Join`
/// (each side is lowered independently by `lower_join`, with its own limit slot).
pub(crate) fn vectorized_supports(plan: &PhysicalPlan) -> SupportLevel {
    let mut reasons = Vec::new();
    classify_support(plan, 0, &mut reasons);
    if reasons.is_empty() {
        SupportLevel::Full
    } else {
        SupportLevel::Partial(reasons)
    }
}

fn classify_support(plan: &PhysicalPlan, limit_count: u8, reasons: &mut Vec<SupportReason>) {
    match plan {
        PhysicalPlan::Values { .. } | PhysicalPlan::Scan { .. } => {}
        PhysicalPlan::Filter { input, .. } | PhysicalPlan::Project { input, .. } => {
            classify_support(input, limit_count, reasons);
        }
        PhysicalPlan::Aggregate { input, having, .. } => {
            if having.is_some() {
                reasons.push(SupportReason::AggregateHaving);
            }
            classify_support(input, limit_count, reasons);
        }
        PhysicalPlan::Limit { input, .. } => {
            if limit_count >= 1 {
                reasons.push(SupportReason::NestedLimit);
            }
            classify_support(input, limit_count + 1, reasons);
        }
        // `Join` is lowered by `lower_join`, which walks each side independently
        // (fresh limit slot per side) — reset the counter at the join boundary.
        PhysicalPlan::Join { left, right, .. } => {
            classify_support(left, 0, reasons);
            classify_support(right, 0, reasons);
        }
        PhysicalPlan::Sort { .. } => reasons.push(SupportReason::UnsupportedNode("Sort")),
        PhysicalPlan::Distinct { .. } => reasons.push(SupportReason::UnsupportedNode("Distinct")),
        PhysicalPlan::AssertMaxOneRow { .. } => {
            reasons.push(SupportReason::UnsupportedNode("AssertMaxOneRow"));
        }
        PhysicalPlan::Union { .. } => reasons.push(SupportReason::UnsupportedNode("Union")),
        PhysicalPlan::SetOp { .. } => reasons.push(SupportReason::UnsupportedNode("SetOp")),
    }
}

/// Convert a [`Walked`] (source + OpSpec chain) into a concrete operator chain
/// + the trailing `Limit`. Extracted so `lower_join` can reuse it for each side.
fn walked_to_operators(walked: Walked) -> (Vec<Box<dyn ExecutionOperator>>, Option<LimitSpec>) {
    let Walked {
        source, ops, limit, ..
    } = walked;
    let mut operators: Vec<Box<dyn ExecutionOperator>> = Vec::with_capacity(ops.len() + 1);
    operators.push(source);
    let mut cur_schema = operators[0].output_schema();
    for spec in ops {
        match spec {
            OpSpec::FilterProject {
                predicate,
                projection,
            } => {
                let out: SchemaRef = match &projection {
                    Some(idx) => Arc::new(arrow::datatypes::Schema::new(
                        idx.iter()
                            .map(|i| cur_schema.field(*i).clone())
                            .collect::<Vec<_>>(),
                    )),
                    None => cur_schema.clone(),
                };
                operators.push(Box::new(FilterProjectOperator::new(
                    predicate,
                    projection,
                    out.clone(),
                )));
                cur_schema = out;
            }
            OpSpec::HashAggregate {
                group_by,
                aggregates,
                output_schema,
            } => {
                operators.push(Box::new(HashAggregateOperator::new(
                    group_by,
                    aggregates,
                    output_schema.clone(),
                )));
                cur_schema = output_schema;
            }
        }
    }
    (operators, limit)
}

/// Lower a `PhysicalPlan::Join` into a two-pipeline `LoweredPipeline` (build +
/// probe) using the #779 `HashJoinBuildOperator`/`HashJoinProbeOperator`.
fn lower_join(
    plan: &PhysicalPlan,
    scan_ctx: Option<&ScanCtx>,
) -> Result<LoweredPipeline, ExecutionError> {
    use crate::query::execution::native_engine::native_join_enabled;
    use crate::query::execution::native_join_ops::{
        HashJoinBuildOperator, HashJoinProbeOperator, JoinColumn,
    };
    use proximadb_relational_algebra::{JoinKind, JoinSide, JoinStrategy};
    use std::sync::OnceLock;

    let PhysicalPlan::Join {
        left,
        right,
        kind,
        on,
        strategy,
    } = plan
    else {
        unreachable!()
    };

    if !native_join_enabled() {
        return Err(ExecutionError::NotImplemented(
            "native join disabled (set PROXIMADB_NATIVE_JOIN=1)".into(),
        ));
    }
    let JoinStrategy::Hash { build_side } = strategy else {
        return Err(ExecutionError::NotImplemented(
            "non-hash join strategy not supported in native path".into(),
        ));
    };

    // Equi-key extraction: ON is an AND-conjunction of `col = col` in
    // concatenated left++right schema space (left: 0..L-1, right: L..L+R-1).
    let left_width = left.output_schema().columns.len();
    let equi_keys = on
        .as_ref()
        .and_then(|e| extract_equi_keys(e, left_width))
        .unwrap_or_default();
    if equi_keys.is_empty() {
        return Err(ExecutionError::NotImplemented(
            "join has no equi-keys (non-equi ON not supported in native path)".into(),
        ));
    }

    // Walk both subtrees.
    let left_walked = walk(left, scan_ctx)?;
    let right_walked = walk(right, scan_ctx)?;
    let left_schema = left_walked.cur_schema.clone();
    let right_schema = right_walked.cur_schema.clone();
    let (left_ops, _) = walked_to_operators(left_walked);
    let (right_ops, _) = walked_to_operators(right_walked);

    // Build/probe assignment from `build_side`.
    let (build_ops, probe_ops, build_key_ords, probe_key_ords, build_schema, probe_schema) =
        match build_side {
            JoinSide::Right => (
                right_ops,
                left_ops,
                equi_keys.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
                equi_keys.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
                right_schema.clone(),
                left_schema.clone(),
            ),
            JoinSide::Left => (
                left_ops,
                right_ops,
                equi_keys.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
                equi_keys.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
                left_schema.clone(),
                right_schema.clone(),
            ),
        };

    let table_slot = Arc::new(OnceLock::new());

    // Build pipeline: subtree ops + HashJoinBuildOperator.
    let mut build_operators = build_ops;
    build_operators.push(Box::new(HashJoinBuildOperator {
        build_keys: build_key_ords,
        build_schema: build_schema.clone(),
        table_slot: table_slot.clone(),
        bloom_enabled: false,
    }));
    let build_pipeline = Pipeline::new(build_operators);

    // Probe pipeline: subtree ops + HashJoinProbeOperator.
    let probe_ncols = probe_schema.fields().len();
    let build_ncols = build_schema.fields().len();
    let output_columns: Vec<JoinColumn> = match kind {
        JoinKind::Semi | JoinKind::Anti => (0..probe_ncols).map(JoinColumn::Probe).collect(),
        _ => (0..probe_ncols)
            .map(JoinColumn::Probe)
            .chain((0..build_ncols).map(JoinColumn::Build))
            .collect(),
    };

    // Output schema: [probe fields, build fields] (probe-only for Semi/Anti).
    let mut out_fields: Vec<_> = probe_schema.fields().iter().cloned().collect();
    if !matches!(kind, JoinKind::Semi | JoinKind::Anti) {
        out_fields.extend(build_schema.fields().iter().cloned());
    }
    let output_schema = Arc::new(arrow::datatypes::Schema::new(out_fields));

    let mut probe_operators = probe_ops;
    probe_operators.push(Box::new(HashJoinProbeOperator {
        table_slot: table_slot.clone(),
        probe_keys: probe_key_ords,
        output_columns,
        kind: *kind,
        output_schema: output_schema.clone(),
    }));
    let probe_pipeline = Pipeline::new(probe_operators);

    Ok(LoweredPipeline {
        pipeline: probe_pipeline,
        build_pipeline: Some(build_pipeline),
        limit: None,
    })
}

/// Extract equi-join key pairs from the ON expression. The ON is an
/// AND-conjunction of `BinaryOp(Eq, Column, Column)` in concatenated left++right
/// schema space (left: `0..L-1`, right: `L..L+R-1`).
fn extract_equi_keys(expr: &Expr, left_width: usize) -> Option<Vec<(usize, usize)>> {
    fn collect(expr: &Expr, left_width: usize, out: &mut Vec<(usize, usize)>) -> bool {
        match expr {
            Expr::BinaryOp {
                op: BinaryOp::Eq,
                left,
                right,
            } => {
                let (l, r) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(c1), Expr::Column(c2)) => (c1.ordinal, c2.ordinal),
                    _ => return false,
                };
                if l < left_width && r >= left_width {
                    out.push((l, r - left_width));
                    true
                } else if r < left_width && l >= left_width {
                    out.push((r, l - left_width));
                    true
                } else {
                    false
                }
            }
            Expr::BinaryOp {
                op: BinaryOp::And,
                left,
                right,
            } => {
                collect(left.as_ref(), left_width, out) && collect(right.as_ref(), left_width, out)
            }
            _ => false,
        }
    }
    let mut keys = Vec::new();
    if collect(expr, left_width, &mut keys) {
        Some(keys)
    } else {
        None
    }
}

/// Stack red-zone / growth for the native lowering recursion (TD-EXEC-2 §1.a).
/// Retires the former `MAX_WALK_DEPTH=32` depth cap: rather than declining a
/// deeply nested plan to the Volcano for a stack reason, grow the stack on demand
/// so native serves any depth correctly on any worker stack. Deliberately generous
/// pending Slice-1 frame-cost calibration; the per-frame check is a TLS read +
/// pointer compare (~ns) and a segment allocates only when the descent nears the
/// limit.
const WALK_RED_ZONE: usize = 256 * 1024;
const WALK_STACK_GROW: usize = 4 * 1024 * 1024;

/// Lower a physical-plan subtree to a native pipeline. Guards each recursion level
/// with [`stacker::maybe_grow`] (TD-EXEC-2 §1.a) so a deep plan grows the stack on
/// demand rather than overflowing the worker thread; the recursive descent goes
/// back through this entry, so every level checks its own headroom.
fn walk(plan: &PhysicalPlan, scan_ctx: Option<&ScanCtx>) -> Result<Walked, ExecutionError> {
    stacker::maybe_grow(WALK_RED_ZONE, WALK_STACK_GROW, || {
        // TD-EXEC-2 Slice 1: sample the recursion's stack low-water mark when a
        // probe is armed (a TLS read + branch otherwise).
        proximadb_relational_planner::stack_probe::note();
        walk_inner(plan, scan_ctx)
    })
}

fn walk_inner(plan: &PhysicalPlan, scan_ctx: Option<&ScanCtx>) -> Result<Walked, ExecutionError> {
    match plan {
        PhysicalPlan::Values {
            rows,
            output_schema,
        } => {
            let arrow_schema = relational_schema_to_arrow(output_schema);
            let batch = values_to_record_batch(rows, arrow_schema.clone())?;
            let source = Box::new(MemoryScanSource::new(vec![batch], arrow_schema.clone()));
            Ok(Walked {
                source,
                ops: Vec::new(),
                cur_schema: arrow_schema,
                limit: None,
            })
        }
        PhysicalPlan::Scan {
            table,
            output_schema,
            predicate,
            ..
        } => {
            // TD-OLAP-4: a parquet source injected via `lower_physical_over_source`
            // takes precedence — this is how native serves external parquet for the
            // shadow probe. Absent an injection, fall through to the PAX path (#807).
            if let Some(source) = INJECTED_SCAN_SOURCE.with(|s| s.borrow_mut().take()) {
                let cur_schema = source.output_schema();
                // A pushed-down `Scan.predicate` must still be applied row-by-row:
                // the parquet source only *prunes* (row groups / pages, a superset),
                // so the exact filter is a FilterProject directly above it. An
                // expression native cannot lower declines the whole query (→ DF).
                let ops = match predicate {
                    Some(p) => vec![OpSpec::FilterProject {
                        predicate: Some(PhysExpr::lower(p)?),
                        projection: None,
                    }],
                    None => Vec::new(),
                };
                return Ok(Walked {
                    source,
                    ops,
                    cur_schema,
                    limit: None,
                });
            }
            use crate::query::execution::native_scan::{PaxScanOperator, expr_to_prune_predicates};
            let ctx = scan_ctx.ok_or_else(|| {
                ExecutionError::NotImplemented(
                    "native scan requires a ScanCtx (no storage context threaded)".into(),
                )
            })?;
            let info = ctx.tables.get(&table.name).ok_or_else(|| {
                ExecutionError::NotImplemented(format!(
                    "no PAX segments discovered for table {:?}",
                    table.name
                ))
            })?;
            let arrow_schema = relational_schema_to_arrow(output_schema);
            // Translate the WHERE predicate → block-level zone-map prune predicates.
            let prune_predicates = predicate
                .as_ref()
                .map(expr_to_prune_predicates)
                .unwrap_or_default();
            let source = Box::new(PaxScanOperator::new(
                info.splits.clone(),
                ctx.filesystem_factory.clone(),
                info.name_to_col_id.clone(),
                arrow_schema.clone(),
                prune_predicates,
            ));
            Ok(Walked {
                source,
                ops: Vec::new(),
                cur_schema: arrow_schema,
                limit: None,
            })
        }
        PhysicalPlan::Filter { input, predicate } => {
            let mut w = walk(input, scan_ctx)?;
            let p = PhysExpr::lower(predicate)?;
            // Fuse into a trailing FilterProject (AND-merge the predicate), else push.
            let fuse = matches!(w.ops.last(), Some(OpSpec::FilterProject { .. }));
            if fuse {
                if let Some(OpSpec::FilterProject {
                    predicate: pred, ..
                }) = w.ops.last_mut()
                {
                    *pred = Some(match pred.take() {
                        Some(existing) => PhysExpr::And(Box::new(existing), Box::new(p)),
                        None => p,
                    });
                }
            } else {
                w.ops.push(OpSpec::FilterProject {
                    predicate: Some(p),
                    projection: None,
                });
            }
            Ok(w) // a filter preserves the column set → cur_schema unchanged
        }
        PhysicalPlan::Project { input, outputs } => {
            let mut w = walk(input, scan_ctx)?;
            let indices = lower_column_indices(outputs)?;
            // Output schema of this projection (computed before `indices` is moved
            // into the op below).
            let projected = Arc::new(arrow::datatypes::Schema::new(
                indices
                    .iter()
                    .map(|i| w.cur_schema.field(*i).clone())
                    .collect::<Vec<_>>(),
            ));
            // Fuse into a trailing pure FilterProject (predicate set, no projection
            // yet) — i.e. a Filter directly below this Project. Else push a new
            // FilterProject (e.g. a post-aggregate projection).
            let fuse = matches!(
                w.ops.last(),
                Some(OpSpec::FilterProject {
                    projection: None,
                    ..
                })
            );
            if fuse {
                if let Some(OpSpec::FilterProject { projection, .. }) = w.ops.last_mut() {
                    *projection = Some(indices);
                }
            } else {
                w.ops.push(OpSpec::FilterProject {
                    predicate: None,
                    projection: Some(indices),
                });
            }
            w.cur_schema = projected;
            Ok(w)
        }
        PhysicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
            having,
            ..
        } => {
            if having.is_some() {
                return Err(ExecutionError::NotImplemented(
                    "HAVING not supported in native HashAggregate".into(),
                ));
            }
            let mut w = walk(input, scan_ctx)?;
            let gb: Vec<usize> = group_by
                .iter()
                .map(|ne| lower_col(&ne.expr))
                .collect::<Result<_, _>>()?;
            let mut specs = Vec::with_capacity(aggregates.len());
            for na in aggregates {
                specs.push(lower_aggregate(&na.agg)?);
            }
            let names: Vec<String> = aggregates.iter().map(|na| na.name.clone()).collect();
            let out_schema = agg_output_schema(&w.cur_schema, &gb, &specs, &names)?;
            w.ops.push(OpSpec::HashAggregate {
                group_by: gb,
                aggregates: specs,
                output_schema: out_schema.clone(),
            });
            w.cur_schema = out_schema;
            Ok(w)
        }
        PhysicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let mut w = walk(input, scan_ctx)?;
            if w.limit.is_some() {
                return Err(ExecutionError::NotImplemented(
                    "nested Limit not supported in native vectorized path".into(),
                ));
            }
            w.limit = Some(LimitSpec {
                offset: *offset,
                limit: *limit,
            });
            Ok(w)
        }
        other => Err(ExecutionError::NotImplemented(format!(
            "native vectorized path does not support PhysicalPlan::{other:?}"
        ))),
    }
}

/// Lower `Project` outputs to column ordinals (column references only).
fn lower_column_indices(outputs: &[NamedExpr]) -> Result<Vec<usize>, ExecutionError> {
    outputs.iter().map(|ne| lower_col(&ne.expr)).collect()
}

/// Resolve a single `Expr` to a column ordinal (column references only).
fn lower_col(expr: &Expr) -> Result<usize, ExecutionError> {
    match expr {
        Expr::Column(col) => Ok(col.ordinal),
        other => Err(ExecutionError::NotImplemented(format!(
            "column reference required here, got {other:?}"
        ))),
    }
}

/// Lower an `AggregateExpr` to an [`AggSpec`]. Declines distinct aggregates and
/// STRING_AGG/Custom (→ Volcano fallback).
fn lower_aggregate(agg: &AggregateExpr) -> Result<AggSpec, ExecutionError> {
    match agg {
        AggregateExpr::Count { arg, distinct } => {
            if *distinct {
                return Err(ExecutionError::NotImplemented(
                    "COUNT(DISTINCT) not supported in native HashAggregate".into(),
                ));
            }
            Ok(AggSpec {
                arg: arg.as_ref().map(lower_col).transpose()?,
                kind: AggKind::Count,
            })
        }
        AggregateExpr::Sum { arg, distinct } => {
            if *distinct {
                return Err(ExecutionError::NotImplemented(
                    "SUM(DISTINCT) not supported in native HashAggregate".into(),
                ));
            }
            Ok(AggSpec {
                arg: Some(lower_col(arg)?),
                kind: AggKind::Sum,
            })
        }
        AggregateExpr::Avg { arg, distinct } => {
            if *distinct {
                return Err(ExecutionError::NotImplemented(
                    "AVG(DISTINCT) not supported in native HashAggregate".into(),
                ));
            }
            Ok(AggSpec {
                arg: Some(lower_col(arg)?),
                kind: AggKind::Avg,
            })
        }
        AggregateExpr::Min { arg } => Ok(AggSpec {
            arg: Some(lower_col(arg)?),
            kind: AggKind::Min,
        }),
        AggregateExpr::Max { arg } => Ok(AggSpec {
            arg: Some(lower_col(arg)?),
            kind: AggKind::Max,
        }),
        other => Err(ExecutionError::NotImplemented(format!(
            "aggregate {other:?} not supported in native HashAggregate"
        ))),
    }
}

/// Output schema of a hash aggregate: [group_by columns, aggregate result columns].
/// COUNT→Int64, SUM/AVG→Float64, MIN/MAX→argument column type.
fn agg_output_schema(
    input: &SchemaRef,
    group_by: &[usize],
    aggregates: &[AggSpec],
    names: &[String],
) -> Result<SchemaRef, ExecutionError> {
    let mut fields: Vec<Field> = group_by.iter().map(|&c| input.field(c).clone()).collect();
    for (spec, name) in aggregates.iter().zip(names.iter()) {
        let dt = match spec.kind {
            AggKind::Count => DataType::Int64,
            AggKind::Sum | AggKind::Avg => DataType::Float64,
            AggKind::Min | AggKind::Max => {
                let arg = spec.arg.ok_or_else(|| {
                    ExecutionError::NotImplemented("MIN/MAX requires a column argument".into())
                })?;
                input.field(arg).data_type().clone()
            }
        };
        fields.push(Field::new(name, dt, true));
    }
    Ok(Arc::new(arrow::datatypes::Schema::new(fields)))
}

/// Drain a [`LoweredPipeline`] to materialized `RecordBatch`es, applying the
/// trailing `Limit` (offset + optional count) row-wise at the end.
/// Execute a single source operator as a one-op pipeline (no plan lowering) —
/// used by the metadata-elision path, where the source already emits the final
/// result row (TD-OLAP-4). Drains the source to `RecordBatch`es.
pub(crate) async fn execute_source(
    source: Box<dyn ExecutionOperator>,
) -> Result<Vec<RecordBatch>, ExecutionError> {
    let lowered = LoweredPipeline {
        pipeline: Pipeline::new(vec![source]),
        build_pipeline: None,
        limit: None,
    };
    execute_pipeline(&lowered).await
}

pub(crate) async fn execute_pipeline(
    lowered: &LoweredPipeline,
) -> Result<Vec<RecordBatch>, ExecutionError> {
    // If there is a BUILD pipeline (a hash join), run + drain it first. Its
    // blocking build operator publishes the hash table via a shared `OnceLock`
    // as a side-effect of being drained; its own (sentinel) output is discarded.
    if let Some(build) = &lowered.build_pipeline {
        let empty: BatchStream = Box::pin(futures::stream::empty());
        let mut build_stream = build.execute(empty).await?;
        while build_stream.next().await.transpose()?.is_some() {}
    }

    let empty: BatchStream = Box::pin(futures::stream::empty());
    let mut stream = lowered.pipeline.execute(empty).await?;

    let mut batches = Vec::new();
    while let Some(result) = stream.next().await {
        batches.push(result?);
    }

    if let Some(spec) = lowered.limit {
        apply_limit(&mut batches, spec)?;
    }
    Ok(batches)
}

/// Apply `offset` + optional `limit` across the collected batches, re-chunking
/// so the surviving rows are returned in a single batch.
fn apply_limit(batches: &mut Vec<RecordBatch>, spec: LimitSpec) -> Result<(), ExecutionError> {
    let schema = batches
        .first()
        .map(|b| b.schema())
        .unwrap_or_else(|| Arc::new(arrow::datatypes::Schema::empty()));

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let offset = (spec.offset as usize).min(total);
    let remaining = total - offset;
    let take = spec
        .limit
        .map(|l| (l as usize).min(remaining))
        .unwrap_or(remaining);

    if offset == 0 && take == total {
        return Ok(());
    }

    // Re-chunk: concatenate then slice [offset, offset+take).
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        arrow::compute::concat_batches(&schema, batches.iter())
            .map_err(|e| ExecutionError::Execution(format!("concat for limit: {e}")))?
    };
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| c.slice(offset, take))
        .collect();
    let sliced = RecordBatch::try_new(schema, columns)
        .map_err(|e| ExecutionError::Execution(format!("limit slice: {e}")))?;
    batches.clear();
    if sliced.num_rows() > 0 {
        batches.push(sliced);
    }
    Ok(())
}

// =========================================================================
// Values → RecordBatch (ProximaValue → Arrow builders)
// =========================================================================

/// Build a single `RecordBatch` from a `PhysicalPlan::Values` literal table by
/// evaluating each cell `Expr` (literals) to a `ProximaValue` and constructing
/// typed Arrow arrays column-major. Reuses `Expr::eval` (the same path the
/// Volcano `ValuesExec` uses) so e.g. `VALUES (1+1)` works.
fn values_to_record_batch(
    rows: &[Vec<Expr>],
    schema: SchemaRef,
) -> Result<RecordBatch, ExecutionError> {
    let ncols = schema.fields().len();
    let nrows = rows.len();
    if ncols == 0 {
        return Err(ExecutionError::NotImplemented(
            "zero-column Values table not supported in native MemoryScanSource".into(),
        ));
    }

    let mut columns: Vec<Vec<ProximaValue>> =
        (0..ncols).map(|_| Vec::with_capacity(nrows)).collect();
    for row in rows {
        if row.len() != ncols {
            return Err(ExecutionError::Schema(format!(
                "values row arity {} != schema column count {}",
                row.len(),
                ncols
            )));
        }
        for (c, expr) in row.iter().enumerate() {
            let empty_row: Vec<ProximaValue> = Vec::new();
            let v = expr
                .eval(&empty_row, builtins())
                .map_err(|e| ExecutionError::Execution(format!("values eval: {e}")))?;
            columns[c].push(v);
        }
    }

    let arrays: Vec<ArrayRef> = (0..ncols)
        .map(|c| build_column_array(&columns[c], schema.field(c).data_type()))
        .collect::<Result<_, _>>()?;

    RecordBatch::try_new(schema, arrays)
        .map_err(|e| ExecutionError::Execution(format!("values batch: {e}")))
}

macro_rules! numeric_col {
    ($values:expr, $builder:ty, $variant:ident) => {{
        let mut b = <$builder>::with_capacity($values.len());
        for v in $values {
            match v {
                ProximaValue::$variant(x) => b.append_value(*x),
                ProximaValue::Null => b.append_null(),
                other => {
                    return Err(ExecutionError::Execution(format!(
                        "type mismatch in values column: expected {}, got {other:?}",
                        stringify!($variant)
                    )));
                }
            }
        }
        Arc::new(b.finish()) as ArrayRef
    }};
}

/// Build a typed Arrow array for one column of `ProximaValue`s, dispatching on
/// the target `DataType`. Unsupported types return `Err` (→ Volcano fallback).
fn build_column_array(values: &[ProximaValue], dt: &DataType) -> Result<ArrayRef, ExecutionError> {
    use arrow::datatypes::DataType as D;
    match dt {
        D::Boolean => {
            let mut b = BooleanBuilder::with_capacity(values.len());
            for v in values {
                match v {
                    ProximaValue::Boolean(x) => b.append_value(*x),
                    ProximaValue::Null => b.append_null(),
                    other => {
                        return Err(ExecutionError::Execution(format!(
                            "type mismatch in Boolean column: got {other:?}"
                        )));
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        D::Int8 => Ok(numeric_col!(values, Int8Builder, Int8)),
        D::Int16 => Ok(numeric_col!(values, Int16Builder, Int16)),
        D::Int32 => Ok(numeric_col!(values, Int32Builder, Int32)),
        D::Int64 => Ok(numeric_col!(values, Int64Builder, Int64)),
        D::UInt8 => Ok(numeric_col!(values, UInt8Builder, UInt8)),
        D::UInt16 => Ok(numeric_col!(values, UInt16Builder, UInt16)),
        D::UInt32 => Ok(numeric_col!(values, UInt32Builder, UInt32)),
        D::UInt64 => Ok(numeric_col!(values, UInt64Builder, UInt64)),
        D::Float32 => Ok(numeric_col!(values, Float32Builder, Float32)),
        D::Float64 => Ok(numeric_col!(values, Float64Builder, Float64)),
        D::Utf8 => {
            let mut b = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for v in values {
                match v {
                    ProximaValue::String(s) => b.append_value(s),
                    ProximaValue::Null => b.append_null(),
                    other => {
                        return Err(ExecutionError::Execution(format!(
                            "type mismatch in Utf8 column: got {other:?}"
                        )));
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        D::Binary => {
            let mut b = BinaryBuilder::with_capacity(values.len(), values.len() * 8);
            for v in values {
                match v {
                    ProximaValue::Binary(x) => b.append_value(x),
                    ProximaValue::Null => b.append_null(),
                    other => {
                        return Err(ExecutionError::Execution(format!(
                            "type mismatch in Binary column: got {other:?}"
                        )));
                    }
                }
            }
            Ok(Arc::new(b.finish()))
        }
        other => Err(ExecutionError::NotImplemented(format!(
            "arrow type {other:?} not supported in native MemoryScanSource"
        ))),
    }
}

/// Convert a `RelationalSchema` (ProximaType) to an Arrow `Schema`.
fn relational_schema_to_arrow(schema: &RelationalSchema) -> SchemaRef {
    let fields: Vec<_> = schema
        .columns
        .iter()
        .map(|c| {
            arrow::datatypes::Field::new(c.name.clone(), proxima_type_to_arrow(&c.ty), c.nullable)
        })
        .collect();
    Arc::new(arrow::datatypes::Schema::new(fields))
}

fn proxima_type_to_arrow(ty: &ProximaType) -> DataType {
    use ProximaType as P;
    match ty {
        P::Boolean => DataType::Boolean,
        P::Int8 => DataType::Int8,
        P::Int16 => DataType::Int16,
        P::Int32 => DataType::Int32,
        P::Int64 => DataType::Int64,
        P::UInt8 => DataType::UInt8,
        P::UInt16 => DataType::UInt16,
        P::UInt32 => DataType::UInt32,
        P::UInt64 => DataType::UInt64,
        P::Float32 => DataType::Float32,
        P::Float64 => DataType::Float64,
        P::String => DataType::Utf8,
        P::Binary => DataType::Binary,
        P::Date => DataType::Date32,
        P::Timestamp(_) => DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
        P::TimestampTz(_) => DataType::Timestamp(
            arrow::datatypes::TimeUnit::Nanosecond,
            Some("+00:00".into()),
        ),
        other => {
            // Unsupported types fall back to Utf8 (stringified) — but note this
            // only matters if a Values table carries such a column; lower_physical
            // still attempts the build and the build_column_array path will Err
            // for non-builder types, sending the query to the Volcano.
            let _ = other;
            DataType::Utf8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use futures::StreamExt;

    fn int64_schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]))
    }

    fn int64_batch(values: &[i64]) -> RecordBatch {
        let schema = int64_schema("x");
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    #[tokio::test]
    async fn filter_project_filters_and_projects() {
        // MemoryScanSource(1..=10) → Filter{x>5} → Project{[x]} → expect 6..=10
        let schema = int64_schema("x");
        let batch = int64_batch(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let source = MemoryScanSource::new(vec![batch], schema.clone());

        // predicate: x > 5 (column ordinal 0 > literal 5)
        let pred = PhysExpr::Compare {
            op: BinaryOp::Gt,
            left: Box::new(PhysExpr::Column(0)),
            right: Box::new(PhysExpr::Literal(ProximaValue::Int64(5))),
        };
        let fp = FilterProjectOperator::new(Some(pred), Some(vec![0]), schema.clone());

        let empty: BatchStream = Box::pin(futures::stream::empty());
        let scanned = source.execute(empty).await.unwrap();
        let filtered = fp.execute(scanned).await.unwrap();

        let out: Vec<RecordBatch> = filtered
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(out.len(), 1);
        let arr = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let got: Vec<i64> = arr.iter().map(|o| o.unwrap()).collect();
        assert_eq!(got, vec![6, 7, 8, 9, 10]);
    }

    #[tokio::test]
    async fn is_null_predicate() {
        let schema = int64_schema("x");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![
                Some(1),
                None,
                Some(3),
                None,
            ]))],
        )
        .unwrap();
        let source = MemoryScanSource::new(vec![batch], schema.clone());
        // predicate: x IS NULL
        let pred = PhysExpr::IsNull {
            expr: Box::new(PhysExpr::Column(0)),
            negated: false,
        };
        let fp = FilterProjectOperator::new(Some(pred), None, schema.clone());
        let empty: BatchStream = Box::pin(futures::stream::empty());
        let scanned = source.execute(empty).await.unwrap();
        let filtered = fp.execute(scanned).await.unwrap();
        let out: Vec<RecordBatch> = filtered
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<_, _>>()
            .unwrap();
        let arr = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let got: Vec<Option<i64>> = arr.iter().collect();
        assert_eq!(got, vec![None, None]);
    }

    #[tokio::test]
    async fn and_kleene_three_valued_logic() {
        // Evaluate (x>2 AND x<8) on [1,5,9] → [false, true, false]
        let batch = int64_batch(&[1, 5, 9]);
        let pred = PhysExpr::And(
            Box::new(PhysExpr::Compare {
                op: BinaryOp::Gt,
                left: Box::new(PhysExpr::Column(0)),
                right: Box::new(PhysExpr::Literal(ProximaValue::Int64(2))),
            }),
            Box::new(PhysExpr::Compare {
                op: BinaryOp::Lt,
                left: Box::new(PhysExpr::Column(0)),
                right: Box::new(PhysExpr::Literal(ProximaValue::Int64(8))),
            }),
        );
        let mask = pred.eval_bool(&batch).unwrap();
        let got: Vec<bool> = mask.iter().map(|o| o.unwrap_or(false)).collect();
        assert_eq!(got, vec![false, true, false]);
    }

    // --- Phase 2.1 HashAggregateOperator tests ---

    async fn run_pipeline(ops: Vec<Box<dyn ExecutionOperator>>) -> Vec<RecordBatch> {
        let lowered = LoweredPipeline {
            pipeline: Pipeline::new(ops),
            build_pipeline: None,
            limit: None,
        };
        execute_pipeline(&lowered).await.unwrap()
    }

    #[tokio::test]
    async fn hash_aggregate_count_star_no_group() {
        // COUNT(*) over [1,2,3,4,5] → 5 (no GROUP BY → single group).
        let schema = int64_schema("x");
        let batch = int64_batch(&[1, 2, 3, 4, 5]);
        let source = MemoryScanSource::new(vec![batch], schema);
        let out_schema = Arc::new(Schema::new(vec![Field::new(
            "count",
            DataType::Int64,
            true,
        )]));
        let agg = HashAggregateOperator::new(
            vec![],
            vec![AggSpec {
                arg: None,
                kind: AggKind::Count,
            }],
            out_schema,
        );
        let batches = run_pipeline(vec![Box::new(source), Box::new(agg)]).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(count.value(0), 5);
    }

    #[tokio::test]
    async fn hash_aggregate_sum_avg_no_group() {
        // SUM=15.0, AVG=3.0 over [1,2,3,4,5].
        let schema = int64_schema("x");
        let batch = int64_batch(&[1, 2, 3, 4, 5]);
        let source = MemoryScanSource::new(vec![batch], schema);
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("sum", DataType::Float64, true),
            Field::new("avg", DataType::Float64, true),
        ]));
        let agg = HashAggregateOperator::new(
            vec![],
            vec![
                AggSpec {
                    arg: Some(0),
                    kind: AggKind::Sum,
                },
                AggSpec {
                    arg: Some(0),
                    kind: AggKind::Avg,
                },
            ],
            out_schema,
        );
        let batches = run_pipeline(vec![Box::new(source), Box::new(agg)]).await;
        assert_eq!(batches[0].num_rows(), 1);
        let sum = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        let avg = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert!((sum.value(0) - 15.0).abs() < f64::EPSILON);
        assert!((avg.value(0) - 3.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn hash_aggregate_group_by_min_max() {
        // g=[a,a,b], x=[10,30,5] → group a: min10/max30; group b: min5/max5.
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, false),
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Int64Array::from(vec![10, 30, 5])),
            ],
        )
        .unwrap();
        let source = MemoryScanSource::new(vec![batch], schema);
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8, true),
            Field::new("min_x", DataType::Int64, true),
            Field::new("max_x", DataType::Int64, true),
        ]));
        let agg = HashAggregateOperator::new(
            vec![0],
            vec![
                AggSpec {
                    arg: Some(1),
                    kind: AggKind::Min,
                },
                AggSpec {
                    arg: Some(1),
                    kind: AggKind::Max,
                },
            ],
            out_schema,
        );
        let batches = run_pipeline(vec![Box::new(source), Box::new(agg)]).await;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2); // groups a, b
        let g = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let minx = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let maxx = batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // Sorted output: group a first, then b.
        assert_eq!(g.value(0), "a");
        assert_eq!(minx.value(0), 10);
        assert_eq!(maxx.value(0), 30);
        assert_eq!(g.value(1), "b");
        assert_eq!(minx.value(1), 5);
        assert_eq!(maxx.value(1), 5);
    }

    // --- Phase 3 hash-join operator tests (TD-OLAP-11) ---

    use crate::query::execution::native_join_ops::{
        HashJoinBuildOperator, HashJoinProbeOperator, JoinColumn,
    };
    use arrow::array::StringArray;
    use proximadb_relational_algebra::JoinKind;
    use std::sync::{Arc, OnceLock};

    fn two_col_batch(
        k: &[i64],
        v: &[&str],
        k_name: &str,
        v_name: &str,
    ) -> (RecordBatch, SchemaRef) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(k_name, DataType::Int64, false),
            Field::new(v_name, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(k.to_vec())),
                Arc::new(StringArray::from(v.to_vec())),
            ],
        )
        .unwrap();
        (batch, schema)
    }

    /// Wire build + probe sources with the join operators sharing a `table_slot`,
    /// then run via `execute_pipeline` (which drains build first).
    async fn run_join(
        build_batch: RecordBatch,
        build_schema: SchemaRef,
        probe_batch: RecordBatch,
        probe_schema: SchemaRef,
        output_columns: Vec<JoinColumn>,
        kind: JoinKind,
        output_schema: SchemaRef,
    ) -> Vec<RecordBatch> {
        let build_source = MemoryScanSource::new(vec![build_batch], build_schema.clone());
        let probe_source = MemoryScanSource::new(vec![probe_batch], probe_schema.clone());
        let table_slot = Arc::new(OnceLock::new());
        let build_op = HashJoinBuildOperator {
            build_keys: vec![0],
            build_schema,
            table_slot: table_slot.clone(),
            bloom_enabled: false,
        };
        let probe_op = HashJoinProbeOperator {
            table_slot,
            probe_keys: vec![0],
            output_columns,
            kind,
            output_schema,
        };
        let lowered = LoweredPipeline {
            pipeline: Pipeline::new(vec![Box::new(probe_source), Box::new(probe_op)]),
            build_pipeline: Some(Pipeline::new(vec![
                Box::new(build_source),
                Box::new(build_op),
            ])),
            limit: None,
        };
        execute_pipeline(&lowered).await.unwrap()
    }

    #[tokio::test]
    async fn hash_join_inner_equi() {
        // build (right): k=[2,3,4] v=[x,y,z]; probe (left): k=[1,2,3] v=[a,b,c].
        // Inner on k → (2,b,x),(3,c,y). (1 unmatched left, 4 unmatched build.)
        let (build_batch, build_schema) = two_col_batch(&[2, 3, 4], &["x", "y", "z"], "kr", "vr");
        let (probe_batch, probe_schema) = two_col_batch(&[1, 2, 3], &["a", "b", "c"], "kl", "vl");
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("kl", DataType::Int64, true),
            Field::new("vl", DataType::Utf8, true),
            Field::new("vr", DataType::Utf8, true),
        ]));
        let batches = run_join(
            build_batch,
            build_schema,
            probe_batch,
            probe_schema,
            vec![
                JoinColumn::Probe(0),
                JoinColumn::Probe(1),
                JoinColumn::Build(1),
            ],
            JoinKind::Inner,
            out_schema,
        )
        .await;
        assert_eq!(batches.len(), 1);
        let out = &batches[0];
        assert_eq!(out.num_rows(), 2);
        let kl = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let vr = out
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // probe order: (k=2,vl=b,vr=x), (k=3,vl=c,vr=y)
        assert_eq!((kl.value(0), vr.value(0)), (2, "x"));
        assert_eq!((kl.value(1), vr.value(1)), (3, "y"));
    }

    #[tokio::test]
    async fn hash_join_left_unmatched() {
        // Left join: probe k=[1,2] v=[a,b]; build k=[2] v=[x]. k=1 unmatched → NULL vr.
        let (build_batch, build_schema) = two_col_batch(&[2], &["x"], "kr", "vr");
        let (probe_batch, probe_schema) = two_col_batch(&[1, 2], &["a", "b"], "kl", "vl");
        let out_schema = Arc::new(Schema::new(vec![
            Field::new("kl", DataType::Int64, true),
            Field::new("vl", DataType::Utf8, true),
            Field::new("vr", DataType::Utf8, true),
        ]));
        let batches = run_join(
            build_batch,
            build_schema,
            probe_batch,
            probe_schema,
            vec![
                JoinColumn::Probe(0),
                JoinColumn::Probe(1),
                JoinColumn::Build(1),
            ],
            JoinKind::Left,
            out_schema,
        )
        .await;
        let out = &batches[0];
        assert_eq!(out.num_rows(), 2); // matched (2,b,x) + unmatched (1,a,NULL)
        let kl = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let vr = out
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for r in 0..out.num_rows() {
            match kl.value(r) {
                2 => assert!(!vr.is_null(r) && vr.value(r) == "x", "matched row"),
                1 => assert!(vr.is_null(r), "unmatched probe row → NULL build col"),
                other => panic!("unexpected k={other}"),
            }
        }
    }

    #[tokio::test]
    async fn hash_join_semi_anti() {
        // probe k=[1,2,3]; build k=[2,4]. Semi → [2]; Anti → [1,3].
        let (build_batch, build_schema) = two_col_batch(&[2, 4], &["x", "z"], "kr", "vr");
        let (probe_batch, probe_schema) = two_col_batch(&[1, 2, 3], &["a", "b", "c"], "kl", "vl");

        let out_schema = Arc::new(Schema::new(vec![
            Field::new("kl", DataType::Int64, true),
            Field::new("vl", DataType::Utf8, true),
        ]));

        // Semi
        let semi = run_join(
            build_batch.clone(),
            build_schema.clone(),
            probe_batch.clone(),
            probe_schema.clone(),
            vec![JoinColumn::Probe(0), JoinColumn::Probe(1)],
            JoinKind::Semi,
            out_schema.clone(),
        )
        .await;
        let semi_k: Vec<i64> = semi[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|o| o.unwrap())
            .collect();
        assert_eq!(semi_k, vec![2]);

        // Anti
        let anti = run_join(
            build_batch,
            build_schema,
            probe_batch,
            probe_schema,
            vec![JoinColumn::Probe(0), JoinColumn::Probe(1)],
            JoinKind::Anti,
            out_schema,
        )
        .await;
        let anti_k: Vec<i64> = anti[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|o| o.unwrap())
            .collect();
        assert_eq!(anti_k, vec![1, 3]);
    }

    fn null_key_batch(
        k: &[Option<i64>],
        v: &[&str],
        k_name: &str,
        v_name: &str,
    ) -> (RecordBatch, SchemaRef) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(k_name, DataType::Int64, true),
            Field::new(v_name, DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(k.to_vec())),
                Arc::new(StringArray::from(v.to_vec())),
            ],
        )
        .unwrap();
        (batch, schema)
    }

    fn int_key_batch(k: &[i64], name: &str) -> (RecordBatch, SchemaRef) {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(k.to_vec()))])
                .unwrap();
        (batch, schema)
    }

    #[tokio::test]
    async fn hash_join_null_keys_never_match() {
        // build (right): k=[Some(2), None]; probe (left): k=[Some(2), None, Some(3)].
        // A NULL key must never match in ANY kind: NULL build keys are absent from
        // the map, and a NULL probe key short-circuits to "no match". (TD-OLAP-11
        // "highest risk" — asserted across all six hash-join kinds.)
        let out3 = || {
            Arc::new(Schema::new(vec![
                Field::new("kl", DataType::Int64, true),
                Field::new("vl", DataType::Utf8, true),
                Field::new("vr", DataType::Utf8, true),
            ]))
        };
        let out2 = || {
            Arc::new(Schema::new(vec![
                Field::new("kl", DataType::Int64, true),
                Field::new("vl", DataType::Utf8, true),
            ]))
        };
        let cols3 = || {
            vec![
                JoinColumn::Probe(0),
                JoinColumn::Probe(1),
                JoinColumn::Build(1),
            ]
        };
        let cols2 = || vec![JoinColumn::Probe(0), JoinColumn::Probe(1)];
        let bp = || null_key_batch(&[Some(2), None], &["x", "n"], "kr", "vr");
        let pp = || null_key_batch(&[Some(2), None, Some(3)], &["a", "b", "c"], "kl", "vl");
        let rows = |bs: Vec<RecordBatch>| bs.iter().map(|b| b.num_rows()).sum::<usize>();

        // Inner: only k=2 matches (NULL never matches).
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols3(), JoinKind::Inner, out3()).await),
            1
        );
        // Left: matched (k=2) + 2 unmatched probe rows (NULL-key, k=3) → 3.
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols3(), JoinKind::Left, out3()).await),
            3
        );
        // Semi: probe rows with a match → only k=2 → 1.
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols2(), JoinKind::Semi, out2()).await),
            1
        );
        // Anti: probe rows without a match → NULL-key row + k=3 → 2.
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols2(), JoinKind::Anti, out2()).await),
            2
        );
        // Right: matched (k=2) + drained unmatched build (the NULL-key build row) → 2.
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols3(), JoinKind::Right, out3()).await),
            2
        );
        // Full: matched (1) + unmatched probe (2) + unmatched build (1) → 4.
        let (b, bs) = bp();
        let (p, ps) = pp();
        assert_eq!(
            rows(run_join(b, bs, p, ps, cols3(), JoinKind::Full, out3()).await),
            4
        );
    }

    #[tokio::test]
    async fn hash_join_right_and_full_drain_unmatched_build() {
        // build (right): k=[2,3,4]; probe (left): k=[1,2,3].
        let out = || {
            Arc::new(Schema::new(vec![
                Field::new("kl", DataType::Int64, true),
                Field::new("vl", DataType::Utf8, true),
                Field::new("vr", DataType::Utf8, true),
            ]))
        };
        let cols = || {
            vec![
                JoinColumn::Probe(0),
                JoinColumn::Probe(1),
                JoinColumn::Build(1),
            ]
        };

        // Right: matched (2↔x),(3↔y) + unmatched build {4→z}, probe cols NULL → 3.
        let (b, bs) = two_col_batch(&[2, 3, 4], &["x", "y", "z"], "kr", "vr");
        let (p, ps) = two_col_batch(&[1, 2, 3], &["a", "b", "c"], "kl", "vl");
        let right = run_join(b, bs, p, ps, cols(), JoinKind::Right, out()).await;
        assert_eq!(right.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
        // Exactly one row has a NULL probe key — the drained unmatched build row,
        // which still carries its own value (vr = "z").
        let mut null_probe_rows = 0;
        for batch in &right {
            let kl = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let vr = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for r in 0..batch.num_rows() {
                if kl.is_null(r) {
                    null_probe_rows += 1;
                    assert_eq!(vr.value(r), "z");
                }
            }
        }
        assert_eq!(null_probe_rows, 1);

        // Full: matched (2,3) + unmatched probe {1} + unmatched build {4} → 4.
        let (b, bs) = two_col_batch(&[2, 3, 4], &["x", "y", "z"], "kr", "vr");
        let (p, ps) = two_col_batch(&[1, 2, 3], &["a", "b", "c"], "kl", "vl");
        let full = run_join(b, bs, p, ps, cols(), JoinKind::Full, out()).await;
        assert_eq!(full.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    }

    #[tokio::test]
    async fn hash_join_bloom_prefilter_direction() {
        // The bloom is built over BUILD keys and tested against PROBE keys. With
        // ≥ BLOOM_MIN_KEYS build keys the build op constructs the bloom; probe keys
        // disjoint from the build set must all miss (TD-OLAP-11: bloom over {build},
        // probe {disjoint} → all miss), and PRESENT probe keys must still match —
        // no false drops (the dangerous failure mode if the bloom were built over
        // probe keys and tested against build keys instead).
        let build_keys: Vec<i64> = (0..1500).collect(); // > 1024 → bloom is built
        let probe_keys: Vec<i64> = vec![5, 100000, 100, 100001, 900]; // 3 present, 2 absent

        let (build_batch, build_schema) = int_key_batch(&build_keys, "kr");
        let (probe_batch, probe_schema) = int_key_batch(&probe_keys, "kl");

        let table_slot = Arc::new(OnceLock::new());
        let build_op = HashJoinBuildOperator {
            build_keys: vec![0],
            build_schema: build_schema.clone(),
            table_slot: table_slot.clone(),
            bloom_enabled: true,
        };
        let probe_op = HashJoinProbeOperator {
            table_slot,
            probe_keys: vec![0],
            output_columns: vec![JoinColumn::Probe(0)],
            kind: JoinKind::Inner,
            output_schema: Arc::new(Schema::new(vec![Field::new("kl", DataType::Int64, true)])),
        };
        let lowered = LoweredPipeline {
            pipeline: Pipeline::new(vec![
                Box::new(MemoryScanSource::new(vec![probe_batch], probe_schema)),
                Box::new(probe_op),
            ]),
            build_pipeline: Some(Pipeline::new(vec![
                Box::new(MemoryScanSource::new(vec![build_batch], build_schema)),
                Box::new(build_op),
            ])),
            limit: None,
        };
        let batches = execute_pipeline(&lowered).await.unwrap();
        let mut matched: Vec<i64> = Vec::new();
        for batch in &batches {
            let kl = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for r in 0..batch.num_rows() {
                matched.push(kl.value(r));
            }
        }
        matched.sort_unstable();
        assert_eq!(matched, vec![5, 100, 900]);
    }

    // --- Join routing through lower_physical (ADR-054 Phase 3 routing) ---

    /// ADR-058 D1 drift guard: `vectorized_supports` must agree with
    /// `lower_physical` — if the descriptor marks a plan `Partial`, the lowering
    /// MUST decline (sound one-way filter); if `Full`, the lowering MAY succeed.
    /// Prevents the declared capability from drifting from the actual lowering.
    #[test]
    fn vectorized_supports_drift_matches_lowering() {
        use proximadb_relational_types::ColumnInfo;
        let schema = RelationalSchema::new(vec![ColumnInfo::new("k", ProximaType::Int64, true)]);
        let values_leaf = || PhysicalPlan::Values {
            rows: vec![vec![Expr::Literal {
                value: ProximaValue::Int64(1),
                ty: ProximaType::Int64,
            }]],
            output_schema: schema.clone(),
        };

        // Full: a bare Values leaf is fully supported, and the lowering succeeds.
        assert!(matches!(
            vectorized_supports(&values_leaf()),
            SupportLevel::Full
        ));
        assert!(lower_physical(&values_leaf(), None).is_ok());

        // Partial (NestedLimit): two Limits on one path → the descriptor flags
        // it AND the lowering declines. Proves the one-way invariant.
        let nested = PhysicalPlan::Limit {
            input: Box::new(PhysicalPlan::Limit {
                input: Box::new(values_leaf()),
                limit: Some(10),
                offset: 0,
            }),
            limit: Some(5),
            offset: 0,
        };
        match vectorized_supports(&nested) {
            SupportLevel::Partial(rs) => assert!(
                rs.iter().any(|r| matches!(r, SupportReason::NestedLimit)),
                "expected NestedLimit reason, got {rs:?}"
            ),
            SupportLevel::Full => panic!("nested Limit must be classified Partial"),
        }
        assert!(
            lower_physical(&nested, None).is_err(),
            "nested Limit must be declined by the lowering"
        );
    }

    #[tokio::test]
    async fn join_routes_through_lower_physical() {
        use proximadb_relational_algebra::{JoinKind, JoinSide, JoinStrategy};
        use proximadb_relational_types::{ColumnInfo, ColumnRef};

        // Enable the native join gate (nextest: fresh process, OnceLock fresh).
        // SAFETY: test-only; nextest process isolation ensures no concurrent access.
        unsafe { std::env::set_var("PROXIMADB_NATIVE_JOIN", "1") };

        let schema = RelationalSchema::new(vec![ColumnInfo::new("k", ProximaType::Int64, true)]);

        // Left table: k = [1, 2, 3]
        let left = PhysicalPlan::Values {
            rows: vec![
                vec![Expr::Literal {
                    value: ProximaValue::Int64(1),
                    ty: ProximaType::Int64,
                }],
                vec![Expr::Literal {
                    value: ProximaValue::Int64(2),
                    ty: ProximaType::Int64,
                }],
                vec![Expr::Literal {
                    value: ProximaValue::Int64(3),
                    ty: ProximaType::Int64,
                }],
            ],
            output_schema: schema.clone(),
        };

        // Right table: k = [2, 3, 4]
        let right = PhysicalPlan::Values {
            rows: vec![
                vec![Expr::Literal {
                    value: ProximaValue::Int64(2),
                    ty: ProximaType::Int64,
                }],
                vec![Expr::Literal {
                    value: ProximaValue::Int64(3),
                    ty: ProximaType::Int64,
                }],
                vec![Expr::Literal {
                    value: ProximaValue::Int64(4),
                    ty: ProximaType::Int64,
                }],
            ],
            output_schema: schema.clone(),
        };

        // Inner join on k (col0 = col1 in concatenated left++right schema).
        let on = Expr::BinaryOp {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Column(ColumnRef {
                name: "k".to_string(),
                ordinal: 0,
                ty: ProximaType::Int64,
                nullable: true,
            })),
            right: Box::new(Expr::Column(ColumnRef {
                name: "k".to_string(),
                ordinal: 1, // left_width(1) + 0
                ty: ProximaType::Int64,
                nullable: true,
            })),
        };
        let plan = PhysicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            kind: JoinKind::Inner,
            on: Some(on),
            strategy: JoinStrategy::Hash {
                build_side: JoinSide::Right,
            },
        };

        // Lower + execute.
        let lowered = lower_physical(&plan, None).expect("lower_physical for Join");
        let batches = execute_pipeline(&lowered)
            .await
            .expect("execute_pipeline for Join");

        // Assert: 2 matched rows (k=2↔2, k=3↔3). Output = [left_k, right_k].
        assert!(!batches.is_empty(), "expected at least one output batch");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "expected 2 matched rows (k=2, k=3)");
        let left_k = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array for left k");
        let right_k = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array for right k");
        let pairs: Vec<(i64, i64)> = (0..batches[0].num_rows())
            .map(|r| (left_k.value(r), right_k.value(r)))
            .collect();
        assert!(
            pairs.contains(&(2, 2)) && pairs.contains(&(3, 3)),
            "expected matched pairs (2,2) + (3,3), got {pairs:?}"
        );
    }

    // --- PAX Scan through lower_physical (Phase 2.5 wiring) ---

    #[tokio::test]
    async fn scan_reads_real_pax_through_lower_physical() {
        use crate::storage::engines::sst::segment_format::write_pax_segment;
        use crate::storage::formats::FileSplit;
        use crate::storage::persistence::filesystem::FilesystemFactory;
        use proximadb_block_format::{VectorQuant, col_id};
        use proximadb_records::ProximaRecord;

        fn record(oid: &str, tenant: &str, ts: i64) -> ProximaRecord {
            ProximaRecord {
                oid: oid.into(),
                tenant_id: tenant.into(),
                created_at_ns: ts,
                updated_at_ns: ts,
                ..Default::default()
            }
        }

        // Write a REAL .pax segment with 2 records (created_at = 1000, 3000).
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("seg.pax");
        let records = vec![record("r1", "t", 1000), record("r2", "t", 3000)];
        write_pax_segment(&path, &records, "col", 0, VectorQuant::Auto, None, None)
            .expect("write_pax_segment");

        // Build the ScanCtx: FilesystemFactory + pre-discovered splits + column mapping.
        let fs = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("FilesystemFactory"),
        );
        let split = FileSplit::new_block(path.to_str().unwrap().to_string(), 0, 0, 0, 0);
        let scan_ctx = ScanCtx {
            filesystem_factory: fs,
            tables: HashMap::from([(
                "test_table".to_string(),
                ScanTableInfo {
                    splits: vec![split],
                    name_to_col_id: HashMap::from([("created_at".to_string(), col_id::CREATED_AT)]),
                },
            )]),
        };

        // Construct PhysicalPlan::Scan pointing at the table.
        let schema = RelationalSchema::new(vec![proximadb_relational_types::ColumnInfo::new(
            "created_at",
            ProximaType::Int64,
            true,
        )]);
        let plan = PhysicalPlan::Scan {
            table: proximadb_relational_algebra::TableId::new("test_table"),
            output_schema: schema,
            projection: None,
            predicate: None,
            limit: None,
            access: proximadb_relational_planner::ScanAccess::FullScan,
        };

        // Lower + execute through the native engine.
        let lowered = lower_physical(&plan, Some(&scan_ctx)).expect("lower_physical for Scan");
        let batches = execute_pipeline(&lowered)
            .await
            .expect("execute_pipeline for Scan");

        // Assert: 2 rows with created_at = 1000, 3000.
        assert!(
            !batches.is_empty(),
            "expected at least one batch from the scan"
        );
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "expected 2 rows from the PAX segment");
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array for created_at");
        let vals: Vec<i64> = arr.iter().map(|o| o.unwrap()).collect();
        assert!(
            vals.contains(&1000) && vals.contains(&3000),
            "expected [1000, 3000], got {vals:?}"
        );
    }
}
