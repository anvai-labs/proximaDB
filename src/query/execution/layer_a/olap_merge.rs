// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! ADR-025 read-merge, streamed (ADR-097 P3.1 PR 9).
//!
//! A materialized table whose rows changed after its `MATERIALIZE` snapshot is
//! read as its cold Parquet base minus the changed oids, plus the current rows
//! for those oids. That merge used to collect the whole base into a second
//! `MemTable`. [`OlapMergeTable`] serves the same rows as a stream instead: the
//! base keeps its own scan — split pruning, filter pushdown, partitions — and
//! [`OlapMergeExec`] drops each changed oid's stale copy batch by batch. Only
//! the changed rows themselves are held in memory.

use std::collections::HashSet;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::compute::filter_record_batch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::{config::ConfigOptions, tree_node::TreeNodeRecursion},
    datasource::{TableProvider, TableType, memory::MemorySourceConfig},
    error::Result as DFResult,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::{Expr, TableProviderFilterPushDown},
    physical_expr::{EquivalenceProperties, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{FilterDescription, FilterPushdownPhase},
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::StreamExt;
use proximadb_storage_common::proxima_arrow::arrow_cell_to_proxima_value;

use crate::services::record_store::proxima_value_to_unique_text;

/// A materialized table's base reconciled with its post-snapshot delta.
#[derive(Debug)]
pub struct OlapMergeTable {
    base: Arc<dyn TableProvider>,
    schema: SchemaRef,
    pk_idx: usize,
    suppress: Arc<HashSet<String>>,
    appends: Vec<RecordBatch>,
}

impl OlapMergeTable {
    /// `suppress` holds the oids changed since the snapshot; `appends` their
    /// current rows, already on the base's schema.
    pub fn try_new(
        base: Arc<dyn TableProvider>,
        pk_column: &str,
        suppress: HashSet<String>,
        appends: Vec<RecordBatch>,
    ) -> DFResult<Self> {
        let schema = base.schema();
        let pk_idx = schema.index_of(pk_column)?;
        Ok(Self {
            base,
            schema,
            pk_idx,
            suppress: Arc::new(suppress),
            appends,
        })
    }
}

#[async_trait]
impl TableProvider for OlapMergeTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Inexact: the base scan only prunes with the filters, and the changed
    /// rows are not filtered at all — DataFusion applies every filter above.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    // No `statistics`: the base footer's counts and bounds include the stale
    // copies, so they must not answer an aggregate.

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let output: Vec<usize> = projection
            .cloned()
            .unwrap_or_else(|| (0..self.schema.fields().len()).collect());
        // The key is read even when not projected, and dropped after.
        let mut read = output.clone();
        let pk = match read.iter().position(|&i| i == self.pk_idx) {
            Some(pk) => pk,
            None => {
                read.push(self.pk_idx);
                read.len() - 1
            }
        };
        // No limit on the base: a suppressed row would count toward it. The
        // limit stays above the scan.
        let base = self.base.scan(state, Some(&read), filters, None).await?;
        let projected = Arc::new(self.schema.project(&output)?);
        let base: Arc<dyn ExecutionPlan> = Arc::new(OlapMergeExec::new(
            base,
            pk,
            self.suppress.clone(),
            projected,
        ));
        if self.appends.is_empty() {
            return Ok(base);
        }
        let appends = MemorySourceConfig::try_new_exec(
            std::slice::from_ref(&self.appends),
            self.schema.clone(),
            Some(output),
        )?;
        UnionExec::try_new(vec![base, appends])
    }
}

/// Drops the base rows whose oid changed since the snapshot, then the key
/// column if it was read only for that.
#[derive(Debug)]
pub struct OlapMergeExec {
    input: Arc<dyn ExecutionPlan>,
    pk: usize,
    suppress: Arc<HashSet<String>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl OlapMergeExec {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        pk: usize,
        suppress: Arc<HashSet<String>>,
        schema: SchemaRef,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            pk,
            suppress,
            schema,
            properties,
        }
    }
}

impl DisplayAs for OlapMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "OlapMergeExec: suppressed_oids={}", self.suppress.len())
    }
}

impl ExecutionPlan for OlapMergeExec {
    fn name(&self) -> &str {
        "OlapMergeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Runtime filters (hash-join build keys, TopK and aggregate thresholds)
    /// pass through to the base scan, which prunes splits with them: dropping a
    /// stale row is per row, and the output columns are a prefix of the
    /// input's, so a filter over the output reads the same columns below.
    /// Static predicates stay above — the base already took them in `scan`.
    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DFResult<FilterDescription> {
        let children = self.children();
        match phase {
            FilterPushdownPhase::Post => {
                FilterDescription::from_children(parent_filters, &children)
            }
            _ => Ok(FilterDescription::all_unsupported(
                &parent_filters,
                &children,
            )),
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let (pk, suppress, schema) = (self.pk, self.suppress.clone(), self.schema.clone());
        Ok(Arc::new(Self::new(
            children.swap_remove(0),
            pk,
            suppress,
            schema,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let (pk, suppress, schema) = (self.pk, self.suppress.clone(), self.schema.clone());
        let stream = input.map(move |batch| suppress_batch(&batch?, pk, &suppress, &schema));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }

    // No expressions of its own: runtime filters pass through to the input.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

/// Keep the rows whose oid is not in `suppress`, as `schema`'s leading columns.
/// Re-wrapping onto `schema` also drops the Parquet file's field metadata.
fn suppress_batch(
    batch: &RecordBatch,
    pk: usize,
    suppress: &HashSet<String>,
    schema: &SchemaRef,
) -> DFResult<RecordBatch> {
    let keys = batch.column(pk);
    let keep: BooleanArray = (0..batch.num_rows())
        .map(|row| {
            let oid = arrow_cell_to_proxima_value(keys, row)
                .map(|v| proxima_value_to_unique_text(&v))
                .unwrap_or_default();
            Some(!suppress.contains(&oid))
        })
        .collect();
    let kept = filter_record_batch(batch, &keep)?;
    let columns = kept.columns()[..schema.fields().len()].to_vec();
    let options = RecordBatchOptions::new().with_row_count(Some(kept.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        schema.clone(),
        columns,
        &options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, true),
        ]))
    }

    fn batch(ids: &[i64], labels: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(StringArray::from(labels.to_vec())),
            ],
        )
        .unwrap()
    }

    fn oid(id: i64) -> String {
        proxima_value_to_unique_text(&proximadb_data_model::ProximaValue::Int64(id))
    }

    /// The base holds 1..=4; 2 was updated and 3 deleted after the snapshot.
    async fn merged() -> SessionContext {
        let base = datafusion::datasource::MemTable::try_new(
            schema(),
            vec![
                vec![batch(&[1, 2], &["a", "b"])],
                vec![batch(&[3, 4], &["c", "d"])],
            ],
        )
        .unwrap();
        let table = OlapMergeTable::try_new(
            Arc::new(base),
            "id",
            [oid(2), oid(3)].into_iter().collect(),
            vec![batch(&[2], &["B"])],
        )
        .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx
    }

    async fn query(ctx: &SessionContext, sql: &str) -> Vec<String> {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        datafusion::arrow::util::pretty::pretty_format_batches(&batches)
            .unwrap()
            .to_string()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[tokio::test]
    async fn changed_rows_replace_their_stale_copies() {
        let ctx = merged().await;
        let got = query(&ctx, "select id, label from t order by id").await;
        assert_eq!(
            got,
            [
                "+----+-------+",
                "| id | label |",
                "+----+-------+",
                "| 1  | a     |",
                "| 2  | B     |",
                "| 4  | d     |",
                "+----+-------+",
            ]
        );
    }

    #[tokio::test]
    async fn the_key_is_read_without_being_projected() {
        let ctx = merged().await;
        let got = query(&ctx, "select label from t order by label").await;
        assert_eq!(
            got,
            [
                "+-------+",
                "| label |",
                "+-------+",
                "| B     |",
                "| a     |",
                "| d     |",
                "+-------+",
            ]
        );
        let got = query(&ctx, "select count(*) as n from t").await;
        assert_eq!(got[3], "| 3 |");
    }

    #[tokio::test]
    async fn filters_and_limits_apply_after_the_merge() {
        let ctx = merged().await;
        // `label = 'b'` matches only the stale copy of 2.
        let got = query(&ctx, "select count(*) as n from t where label = 'b'").await;
        assert_eq!(got[3], "| 0 |");
        let got = query(&ctx, "select id from t where id <= 3 order by id limit 2").await;
        assert_eq!(got[3..5], ["| 1  |", "| 2  |"]);
    }
}
