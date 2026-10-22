// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! `LayerATableProvider` / `LayerAScanExec`: the streaming DataFusion source over
//! one native table's relational record tier (ADR-097 P3.1, PR 4).

use std::collections::HashSet;
use std::fmt::Formatter;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::file::metadata::ParquetMetaData;
use proximadb_catalog_schema::CatalogTableSchema;
use proximadb_records::ProximaRecord;
use proximadb_storage_common::proxima_arrow::SYS_COL_OID;

use super::conform::{ConformContext, Conformed, SegmentShape, conform_segment_batch};
use super::merge::{ConformedStream, LayerAMerge};
use crate::services::layer_a_snapshot::{LayerARowAccess, LayerASnapshot, NativeLayerASnapshot};
use crate::storage::engines::sst::segment_format::read_segment_records;
use crate::storage::persistence::filesystem::FileSystem;
use crate::storage::persistence::filesystem::parquet_async_reader::FileSystemParquetReader;

/// A native table served by streaming its record tier: spill segments merged
/// newest-wins under the resident rows. Rows are exactly those the materialized
/// path (`full_table_records` → `MemTable`) serves, with no row cap.
pub struct LayerATableProvider {
    schema: SchemaRef,
    table: Arc<LayerATable>,
}

/// What every scan of one registration shares.
struct LayerATable {
    schema: CatalogTableSchema,
    snapshot: Option<LayerASnapshot>,
    access: Arc<LayerARowAccess>,
}

impl std::fmt::Debug for LayerATableProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerATableProvider")
            .field("table", &self.table.schema.name)
            .field("snapshot", &self.table.snapshot)
            .finish()
    }
}

impl LayerATableProvider {
    pub fn new(table: NativeLayerASnapshot) -> Self {
        let NativeLayerASnapshot {
            schema,
            snapshot,
            access,
        } = table;
        Self {
            schema: schema.to_arrow_schema(),
            table: Arc::new(LayerATable {
                schema,
                snapshot,
                access: Arc::new(access),
            }),
        }
    }
}

#[async_trait]
impl TableProvider for LayerATableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Inexact for every filter: DataFusion keeps evaluating them above the
    /// scan, so correctness never depends on what the scan prunes.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let ctx = ConformContext::new(
            self.table.schema.clone(),
            projection.map(Vec::as_slice),
            self.table.access.clone(),
            now_ns(),
        )
        .map_err(|e| DataFusionError::Plan(format!("{e:#}")))?;
        Ok(Arc::new(LayerAScanExec::new(
            self.table.clone(),
            Arc::new(ctx),
            limit,
            state.config().batch_size(),
        )))
    }
}

fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// One partition: the merge of every segment plus the resident rows.
pub struct LayerAScanExec {
    table: Arc<LayerATable>,
    ctx: Arc<ConformContext>,
    limit: Option<usize>,
    batch_size: usize,
    properties: Arc<PlanProperties>,
}

impl std::fmt::Debug for LayerAScanExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerAScanExec")
            .field("table", &self.table.schema.name)
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl LayerAScanExec {
    fn new(
        table: Arc<LayerATable>,
        ctx: Arc<ConformContext>,
        limit: Option<usize>,
        batch_size: usize,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(ctx.output.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            table,
            ctx,
            limit,
            batch_size,
            properties: Arc::new(properties),
        }
    }

    fn segment_count(&self) -> usize {
        self.table
            .snapshot
            .as_ref()
            .map_or(0, |s| s.view.segments.len())
    }
}

impl DisplayAs for LayerAScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "LayerAScanExec: table={}, segments={}, resident={}, records_only={}, limit={:?}",
            self.table.schema.name,
            self.segment_count(),
            self.table
                .snapshot
                .as_ref()
                .map_or(0, |s| s.view.resident.len()),
            self.ctx.records_only,
            self.limit
        )
    }
}

impl ExecutionPlan for LayerAScanExec {
    fn name(&self) -> &str {
        "LayerAScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "LayerAScanExec has one partition, asked for {partition}"
            )));
        }
        let output = self.ctx.output.clone();
        let Some(snapshot) = self.table.snapshot.clone() else {
            // Denied: no rows, and no partition state was read.
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                output,
                futures::stream::empty(),
            )));
        };
        let ctx = self.ctx.clone();
        let (limit, batch_size) = (self.limit, self.batch_size);
        let stream =
            futures::stream::once(
                async move { open_merge(snapshot, ctx, limit, batch_size).await },
            )
            .map_ok(|merge| {
                futures::stream::try_unfold(merge, |mut merge| async move {
                    Ok(merge.next_batch().await?.map(|batch| (batch, merge)))
                })
            })
            .try_flatten()
            .map_err(|e: anyhow::Error| DataFusionError::External(e.into()));
        Ok(Box::pin(RecordBatchStreamAdapter::new(output, stream)))
    }
}

/// Open every segment (footers concurrently) and assemble the merge.
async fn open_merge(
    snapshot: LayerASnapshot,
    ctx: Arc<ConformContext>,
    limit: Option<usize>,
    batch_size: usize,
) -> Result<LayerAMerge> {
    let LayerASnapshot {
        view,
        filesystem,
        table_name: _,
    } = snapshot;
    let tombstones: HashSet<&str> = view.tombstones.iter().map(String::as_str).collect();
    let resident: Vec<ProximaRecord> = view
        .resident
        .into_iter()
        .filter(|r| !tombstones.contains(r.oid.as_str()))
        .collect();
    let mut shadow: HashSet<String> = view.tombstones.iter().cloned().collect();
    shadow.extend(resident.iter().map(|r| r.oid.clone()));

    let segments =
        if view.segments.is_empty() {
            Vec::new()
        } else {
            let filesystem = filesystem.context("layer A: segments without a filesystem")?;
            futures::future::try_join_all(view.segments.iter().map(|path| {
                open_segment(filesystem.clone(), path.clone(), ctx.clone(), batch_size)
            }))
            .await?
        };
    let resident = (!resident.is_empty())
        .then(|| Conformed::from_records(&ctx, &resident))
        .transpose()?;
    Ok(LayerAMerge::new(
        segments,
        shadow,
        resident,
        ctx.output.clone(),
        batch_size,
        limit,
    ))
}

/// Is every row group declared sorted by the oid column, ascending?
fn declares_oid_order(metadata: &ParquetMetaData) -> bool {
    let schema = metadata.file_metadata().schema_descr();
    let Some(oid) =
        (0..schema.num_columns()).find(|&i| schema.column(i).path().string() == SYS_COL_OID)
    else {
        return false;
    };
    metadata.row_groups().iter().all(|rg| {
        rg.sorting_columns()
            .and_then(|cols| cols.first())
            .is_some_and(|c| c.column_idx as usize == oid && !c.descending)
    })
}

/// A segment as a stream of conformed batches in oid order.
///
/// Streamed row group by row group when it is a Parquet object declaring oid
/// order (every segment the spill writer produces); otherwise read whole through
/// the canonical segment router and sorted in memory — bounded by one segment,
/// and never a reason to refuse an older or differently-written object.
async fn open_segment(
    filesystem: Arc<dyn FileSystem>,
    path: String,
    ctx: Arc<ConformContext>,
    batch_size: usize,
) -> Result<ConformedStream> {
    let mut reader = FileSystemParquetReader::new(filesystem.clone(), path.clone());
    let metadata = reader.load_metadata(None).await.ok();
    if let Some(metadata) = metadata.filter(|m| declares_oid_order(m)) {
        let arrow = ArrowReaderMetadata::try_new(metadata.clone(), ArrowReaderOptions::new())
            .with_context(|| format!("layer A: read schema of segment '{path}'"))?;
        let mask = match ctx.segment_columns(arrow.schema()) {
            Some(roots) => {
                ProjectionMask::roots(arrow.metadata().file_metadata().schema_descr(), roots)
            }
            None => ProjectionMask::all(),
        };
        let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(
            reader.with_metadata(metadata),
            arrow,
        )
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()
        .with_context(|| format!("layer A: open segment '{path}'"))?;
        let shape = Arc::new(SegmentShape::of(&ctx, stream.schema())?);
        return Ok(stream
            .map(move |batch| {
                let batch = batch.with_context(|| format!("layer A: decode segment '{path}'"))?;
                conform_segment_batch(&ctx, &shape, &batch)
            })
            .boxed());
    }

    let bytes = filesystem
        .read(&path)
        .await
        .map_err(|e| anyhow::anyhow!("layer A: read segment '{path}' failed: {e}"))?;
    let mut records = read_segment_records(&bytes, &[], &[], None)
        .with_context(|| format!("layer A: decode segment '{path}'"))?;
    records.sort_by(|a, b| a.oid.cmp(&b.oid));
    let chunks: Vec<Result<Conformed>> = records
        .chunks(batch_size.max(1))
        .map(|chunk| Conformed::from_records(&ctx, chunk))
        .collect();
    Ok(futures::stream::iter(chunks).boxed())
}
