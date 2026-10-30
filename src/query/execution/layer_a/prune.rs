// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Row-group pruning for Layer A segments (ADR-097 P3.1, PRs 5–6).
//!
//! The scan's filters are pushed down `Inexact` — DataFusion re-applies them —
//! so pruning only has to be **safe**: a row group it skips must hold no row
//! that would have been served and matched. Three things make naive min/max
//! pruning unsafe here, and each has a rule:
//!
//! * **Newest wins.** A row group's rows are the newest copies of their oids
//!   only within its segment; an older segment may hold an older copy that
//!   DOES match. Skipping the newer copy would let the stale one through. So a
//!   pruned row group is not dropped but read as a **key spine** — its oid
//!   column alone — whose rows enter the merge invisible, still shadowing every
//!   older copy. The oldest segment shadows nothing, so its pruned row groups
//!   are skipped outright.
//! * **The residual.** A row's served value for a column comes from its
//!   residual instead of the typed column (which the statistics describe) when
//!   the residual names that column. A row group the statistics would prune
//!   but that holds residual rows is a *candidate*: its residual column alone
//!   is read, and the prune stands only if no residual row names a column the
//!   filters reference (compared case-insensitively), or replaces embeddings
//!   while a filter references a vector column (only those are served from
//!   embeddings) ([`SegmentPruner::residual_allows_prune`]). A residual the
//!   reader cannot read or decode keeps the row group.
//! * **Types and keys.** Statistics are used only for a column the segment
//!   holds under the catalog's own Arrow type (otherwise its values are
//!   conformed, and the stored ones say nothing about the served ones); and a
//!   primary-key column's statistics (bounds and null count) are withheld for
//!   a row group where it is NULL in some row, or may be — that row's key is
//!   its oid, which the statistics do not cover. Float columns are never
//!   pruned on: Parquet leaves NaN out of min/max, while DataFusion orders NaN
//!   above every number.
//!
//! Statistics that are absent read as "unknown", which keeps the row group.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use arrow::compute::nullif;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::utils::conjunction;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::ParquetMetaData;
use proximadb_storage_common::proxima_arrow::{SYS_COL_RESIDUAL, residual_overrides};

/// The scan's filters as a row-group pruning predicate, plus what it pruned.
pub(crate) struct SegmentPruner {
    predicate: PruningPredicate,
    /// The catalog Arrow schema (every live column) the predicate is over.
    catalog: SchemaRef,
    primary_key: Option<String>,
    /// The columns the pruning filters reference, as [`prop_head`]s.
    columns: HashSet<String>,
    /// Does a filter reference a vector column (served from embeddings)?
    vector_filter: bool,
    row_groups_total: Count,
    row_groups_pruned: Count,
    residual_checked: Count,
    residual_unreadable: Count,
}

/// A segment's pruning plan before the residual check.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RowGroupPlan {
    /// Per row group: may it hold a matching row?
    pub keep: Vec<bool>,
    /// Row groups the statistics prune but that hold (or may hold) residual
    /// rows: kept unless [`SegmentPruner::residual_allows_prune`] clears them.
    pub residual_candidates: Vec<usize>,
}

impl std::fmt::Debug for SegmentPruner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentPruner")
            .field("predicate", &self.predicate.predicate_expr())
            .finish_non_exhaustive()
    }
}

impl SegmentPruner {
    /// A pruner for `filters`, or `None` when there is nothing to prune with
    /// (no filters, or none DataFusion can turn into a pruning predicate).
    pub fn try_new(
        filters: &[Expr],
        catalog: SchemaRef,
        primary_key: Option<String>,
        state: &dyn Session,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Option<Self> {
        let df_schema = DFSchema::try_from(catalog.as_ref().clone()).ok()?;
        // A filter that cannot be planned here (a subquery, a placeholder) is
        // left out rather than disabling pruning for the rest.
        let usable: Vec<Expr> = filters
            .iter()
            .filter(|f| state.create_physical_expr((*f).clone(), &df_schema).is_ok())
            .cloned()
            .collect();
        let names: Vec<String> = usable
            .iter()
            .flat_map(|f| f.column_refs())
            .map(|c| c.name.clone())
            .collect();
        let physical = state
            .create_physical_expr(conjunction(usable)?, &df_schema)
            .ok()?;
        let counter = |name: &'static str| MetricBuilder::new(metrics).global_counter(name);
        Self::from_physical(
            physical,
            &names,
            catalog,
            primary_key,
            [
                counter("row_groups_total"),
                counter("row_groups_pruned"),
                counter("row_groups_residual_checked"),
                counter("row_groups_residual_unreadable"),
            ],
        )
    }

    /// A pruner for the current state of a dynamic (runtime) filter pushed
    /// into the scan: `expr` is over `schema`, the scan's output schema. Its
    /// counters are private — the scan counts what dynamic filters prune.
    pub fn from_dynamic(
        expr: Arc<dyn PhysicalExpr>,
        schema: SchemaRef,
        primary_key: Option<String>,
    ) -> Option<Self> {
        let names: Vec<String> = collect_columns(&expr)
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        Self::from_physical(
            expr,
            &names,
            schema,
            primary_key,
            std::array::from_fn(|_| Count::new()),
        )
    }

    fn from_physical(
        physical: Arc<dyn PhysicalExpr>,
        referenced: &[String],
        catalog: SchemaRef,
        primary_key: Option<String>,
        [
            row_groups_total,
            row_groups_pruned,
            residual_checked,
            residual_unreadable,
        ]: [Count; 4],
    ) -> Option<Self> {
        let columns = referenced.iter().map(|c| prop_head(c)).collect();
        let vector_filter = referenced.iter().any(|c| {
            catalog.field_with_name(c).is_ok_and(|f| {
                matches!(
                    f.data_type(),
                    DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
                )
            })
        });
        let predicate = PruningPredicateBuilder::new()
            .with_file_schema(catalog.clone())
            .try_build(physical)
            .ok()?;
        if predicate.always_true() {
            return None;
        }
        Some(Self {
            predicate,
            catalog,
            primary_key,
            columns,
            vector_filter,
            row_groups_total,
            row_groups_pruned,
            residual_checked,
            residual_unreadable,
        })
    }

    /// Per row group of a segment: may it hold a matching row? (`false` ⇒ it
    /// may be read as a key spine only.) A row group the statistics prune but
    /// that holds residual rows is kept here and listed as a candidate for
    /// [`Self::residual_allows_prune`]. Any failure keeps every row group.
    pub fn plan(&self, metadata: &ParquetMetaData, segment: &Schema) -> RowGroupPlan {
        let n = metadata.num_row_groups();
        let stats = SegmentStatistics {
            metadata,
            segment,
            pruner: self,
        };
        let mut keep = self
            .predicate
            .prune(&stats)
            .unwrap_or_else(|_| vec![true; n]);
        if keep.len() != n {
            keep = vec![true; n];
        }
        let mut residual_candidates = Vec::new();
        for (i, (k, has_residual)) in keep.iter_mut().zip(stats.residual_row_groups()).enumerate() {
            if has_residual && !*k {
                *k = true;
                residual_candidates.push(i);
            }
        }
        RowGroupPlan {
            keep,
            residual_candidates,
        }
    }

    /// May a candidate row group stay pruned, given (a batch of) its residual
    /// cells? Only when every cell decodes and none names a filter column (by
    /// [`prop_head`]) or replaces the embeddings under a vector filter.
    pub fn residual_allows_prune<'a>(&self, cells: impl IntoIterator<Item = &'a [u8]>) -> bool {
        cells.into_iter().all(|bytes| {
            residual_overrides(bytes).is_ok_and(|o| {
                !(o.embeddings && self.vector_filter)
                    && o.props
                        .iter()
                        .all(|p| !self.columns.contains(&prop_head(p)))
            })
        })
    }

    /// Count one candidate's residual check, and whether its column could not
    /// be read (which keeps it).
    pub fn count_residual_check(&self, unreadable: bool) {
        self.residual_checked.add(1);
        if unreadable {
            self.residual_unreadable.add(1);
        }
    }

    /// Count a segment's final fates.
    pub fn record(&self, keep: &[bool]) {
        self.row_groups_total.add(keep.len());
        self.row_groups_pruned
            .add(keep.iter().filter(|k| !**k).count());
    }
}

/// One segment's row-group statistics, restricted to what is safe to prune on.
struct SegmentStatistics<'a> {
    metadata: &'a ParquetMetaData,
    segment: &'a Schema,
    pruner: &'a SegmentPruner,
}

impl SegmentStatistics<'_> {
    fn converter(&self, name: &str) -> Option<StatisticsConverter<'_>> {
        StatisticsConverter::try_new(
            name,
            self.segment,
            self.metadata.file_metadata().schema_descr(),
        )
        .ok()
        .map(|c| c.with_missing_null_counts_as_zero(false))
    }

    /// A converter for a catalog column the segment holds under the catalog's
    /// own type — the only columns whose statistics describe served values —
    /// and that is not a float (NaN is absent from float min/max).
    fn typed(&self, column: &Column) -> Option<StatisticsConverter<'_>> {
        let catalog = self.pruner.catalog.field_with_name(&column.name).ok()?;
        let stored = self.segment.field_with_name(&column.name).ok()?;
        let usable =
            stored.data_type() == catalog.data_type() && !catalog.data_type().is_floating();
        usable.then(|| self.converter(&column.name))?
    }

    fn is_key(&self, column: &Column) -> bool {
        self.pruner
            .primary_key
            .as_deref()
            .is_some_and(|pk| pk.eq_ignore_ascii_case(&column.name))
    }

    /// For the primary key, every statistic of a row group where the key column
    /// is (or may be) NULL in some row is withheld — NULL reads as unknown.
    fn key_safe(
        &self,
        column: &Column,
        converter: &StatisticsConverter<'_>,
        values: ArrayRef,
    ) -> Option<ArrayRef> {
        if !self.is_key(column) {
            return Some(values);
        }
        let nulls = self.null_counts_of(converter)?;
        let unknown: BooleanArray = nulls.iter().map(|c| Some(c != Some(0))).collect();
        nullif(&values, &unknown).ok()
    }

    fn null_counts_of(&self, converter: &StatisticsConverter<'_>) -> Option<UInt64Array> {
        converter
            .row_group_null_counts(self.metadata.row_groups().iter())
            .ok()
    }

    fn bound(&self, column: &Column, max: bool) -> Option<ArrayRef> {
        let converter = self.typed(column)?;
        let row_groups = self.metadata.row_groups().iter();
        let values = if max {
            converter.row_group_maxes(row_groups).ok()?
        } else {
            converter.row_group_mins(row_groups).ok()?
        };
        self.key_safe(column, &converter, values)
    }

    /// Row groups holding any residual row (or whose residual statistics are
    /// missing) — pruned only after the residual check.
    fn residual_row_groups(&self) -> Vec<bool> {
        let n = self.metadata.num_row_groups();
        if self.segment.field_with_name(SYS_COL_RESIDUAL).is_err() {
            return vec![false; n];
        }
        let Some(converter) = self.converter(SYS_COL_RESIDUAL) else {
            return vec![true; n];
        };
        let Some(nulls) = self.null_counts_of(&converter) else {
            return vec![true; n];
        };
        self.metadata
            .row_groups()
            .iter()
            .zip(nulls.iter())
            .map(|(rg, null_count)| null_count != Some(rg.num_rows().max(0) as u64))
            .collect()
    }
}

impl PruningStatistics for SegmentStatistics<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, false)
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound(column, true)
    }

    fn num_containers(&self) -> usize {
        self.metadata.num_row_groups()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let converter = self.typed(column)?;
        let counts = Arc::new(self.null_counts_of(&converter)?) as ArrayRef;
        self.key_safe(column, &converter, counts)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        Some(Arc::new(UInt64Array::from_iter_values(
            self.metadata
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows().max(0) as u64),
        )))
    }

    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

/// The top-level prop a column or residual key resolves through, lowercased:
/// a column `a.b` reads the nested prop `a` → `b` (`tree_get`), so a residual
/// replacing `a` overrides it. Case is folded so the comparison only errs
/// towards keeping a row group.
fn prop_head(name: &str) -> String {
    name.split('.').next().unwrap_or(name).to_lowercase()
}

/// Runs of consecutive row groups with the same fate, in file order.
pub(crate) fn runs<T: Copy + PartialEq>(fates: &[T]) -> Vec<(T, Vec<usize>)> {
    let mut out: Vec<(T, Vec<usize>)> = Vec::new();
    for (i, &k) in fates.iter().enumerate() {
        match out.last_mut() {
            Some((fate, groups)) if *fate == k => groups.push(i),
            _ => out.push((k, vec![i])),
        }
    }
    out
}
