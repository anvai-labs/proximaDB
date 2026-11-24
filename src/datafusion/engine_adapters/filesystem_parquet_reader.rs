//! # Parquet-over-FileSystem Table Provider
//!
//! Reads external Parquet files through ProximaDB's canonical [`FileSystem`] trait — the
//! same abstraction NOVA uses — so the *same code path* serves a local `file://` Parquet
//! in dev and an `s3://` object-store Parquet in production. This is the I/O side of the
//! financial benchmark (`docs/12-design/PROXIMA_NOTEBOOK_PSEUDO_DISTRIBUTED_BLUEPRINT_2026_06_04.adoc`):
//! demonstrate object-storage reads + DataFusion parallel compute without a JVM.
//!
//! Each Parquet row group becomes one [`FileSplit`], so `ProximaScanExec` fans the row
//! groups across partitions for real intra-node parallelism. The file bytes are read once
//! through the trait (one object-store GET for S3) and shared cheaply across splits via
//! reference-counted [`Bytes`]; per-row-group ranged reads are a future optimization the
//! blueprint already tracks.

use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::prelude::SessionContext;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::super::proxima_scan_exec::{ProximaScanExec, SplitReader};
use super::super::proxima_table_provider::EngineType;
use crate::storage::formats::{FileSplit, SplitType};
use crate::storage::persistence::filesystem::{FileSystem, FilesystemFactory};

/// Project a full schema down to the given column indices (in projection order).
fn project_schema(full: &SchemaRef, projection: Option<&[usize]>) -> SchemaRef {
    match projection {
        Some(proj) => {
            let fields: Vec<_> = proj
                .iter()
                .filter_map(|&i| full.fields().get(i))
                .map(|f| f.as_ref().clone())
                .collect();
            Arc::new(Schema::new(fields))
        }
        None => full.clone(),
    }
}

// ============================================================================
// Split Reader
// ============================================================================

/// Reads individual Parquet row groups from an in-memory, reference-counted copy of the
/// file bytes (fetched once through the [`FileSystem`] trait at table-open time).
#[derive(Debug)]
pub struct FilesystemParquetSplitReader {
    schema: SchemaRef,
    bytes: Bytes,
}

#[async_trait]
impl SplitReader for FilesystemParquetSplitReader {
    async fn read_split(
        &self,
        split: &FileSplit,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        let row_group_index = match &split.split_type {
            SplitType::RowGroup {
                row_group_index, ..
            } => *row_group_index,
            other => {
                return Err(DataFusionError::Execution(format!(
                    "FilesystemParquetSplitReader expects RowGroup splits, got {other:?}"
                )));
            }
        };

        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| DataFusionError::Execution(format!("parquet open: {e}")))?
            .with_row_groups(vec![row_group_index])
            .with_batch_size(batch_size);
        let reader = builder
            .build()
            .map_err(|e| DataFusionError::Execution(format!("parquet reader build: {e}")))?;

        let proj_owned = projection.map(|p| p.to_vec());
        let out_schema = project_schema(&self.schema, projection);

        // The row group fits in memory; decode it (projecting to the requested columns in
        // projection order) and hand DataFusion the batches as a stream.
        let mut batches: Vec<DFResult<arrow_array::RecordBatch>> = Vec::new();
        for batch in reader {
            let batch =
                batch.map_err(|e| DataFusionError::Execution(format!("parquet decode: {e}")))?;
            let batch = match &proj_owned {
                Some(proj) => batch
                    .project(proj)
                    .map_err(|e| DataFusionError::Execution(format!("project: {e}")))?,
                None => batch,
            };
            batches.push(Ok(batch));
        }

        let stream = futures::stream::iter(batches);
        Ok(Box::pin(RecordBatchStreamAdapter::new(out_schema, stream)))
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn engine_type(&self) -> EngineType {
        // Parquet/columnar — reuse the NOVA engine label (VIPER removed, ADR-093).
        EngineType::Nova
    }

    fn supports_projection_pushdown(&self) -> bool {
        true
    }
}

// ============================================================================
// Table Provider
// ============================================================================

/// A DataFusion table backed by a single external Parquet file read through the canonical
/// [`FileSystem`] trait. One row group per [`FileSplit`].
#[derive(Debug)]
pub struct FilesystemParquetTable {
    schema: SchemaRef,
    bytes: Bytes,
    splits: Vec<FileSplit>,
    file_path: String,
}

impl FilesystemParquetTable {
    /// Open a Parquet file at `url` (e.g. `file:///data/options.parquet`, `s3://bucket/k`)
    /// using `fs`, reading the bytes once and enumerating its row groups as splits.
    pub async fn open(fs: Arc<dyn FileSystem>, url: &str) -> DFResult<Self> {
        let path = FilesystemFactory::resolve_path(url)
            .map_err(|e| DataFusionError::Execution(format!("resolve_path({url}): {e}")))?;
        let raw = fs
            .read(&path)
            .await
            .map_err(|e| DataFusionError::Execution(format!("filesystem read {path}: {e}")))?;
        let bytes = Bytes::from(raw);

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .map_err(|e| DataFusionError::Execution(format!("parquet metadata {path}: {e}")))?;
        let schema = builder.schema().clone();
        let metadata = builder.metadata().clone();

        let mut splits = Vec::with_capacity(metadata.num_row_groups());
        for i in 0..metadata.num_row_groups() {
            let rg = metadata.row_group(i);
            let byte_size = rg.total_byte_size().max(0) as u64;
            splits.push(FileSplit::new_row_group(
                path.clone(),
                i,
                0,
                byte_size,
                rg.num_rows(),
            ));
        }

        Ok(Self {
            schema,
            bytes,
            splits,
            file_path: path,
        })
    }

    /// Number of splits (row groups) — useful for asserting real parallelism in tests.
    pub fn split_count(&self) -> usize {
        self.splits.len()
    }

    fn reader(&self) -> Arc<FilesystemParquetSplitReader> {
        Arc::new(FilesystemParquetSplitReader {
            schema: self.schema.clone(),
            bytes: self.bytes.clone(),
        })
    }
}

#[async_trait]
impl TableProvider for FilesystemParquetTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let target_partitions = state.config().target_partitions();
        let exec = ProximaScanExec::builder()
            .schema(self.schema.clone())
            .splits(self.splits.clone())
            .reader(self.reader())
            .projection(projection.map(<[usize]>::to_vec))
            .filters(filters.to_vec())
            .limit(limit)
            .collection_name(self.file_path.clone())
            .batch_size(8192)
            .target_partitions(target_partitions)
            .build()?;
        Ok(Arc::new(exec))
    }
}

/// Register a Parquet file (read through `fs`) as a DataFusion table named `name`.
///
/// Returns the concrete table so callers can inspect split/partition counts.
pub async fn register_parquet_path(
    ctx: &SessionContext,
    fs: Arc<dyn FileSystem>,
    name: &str,
    url: &str,
) -> DFResult<Arc<FilesystemParquetTable>> {
    let table = Arc::new(FilesystemParquetTable::open(fs, url).await?);
    ctx.register_table(name, table.clone())?;
    Ok(table)
}
