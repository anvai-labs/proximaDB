//! DataFusion table provider for an authorized, producer-pinned Iceberg snapshot.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use futures::{StreamExt, TryStreamExt, stream};
use object_store::path::Path;
use proximadb_iceberg_engine::{IcebergObjectStoreBridge, PinnedIcebergReader};

use crate::query::execution::pinned_source::{PinnedBatchSource, PinnedBatchStream};
use crate::services::ddl::pinned_iceberg::PinnedIcebergLocation;

/// Read-only provider for a pinned source.
pub struct PinnedTable {
    reader: Arc<dyn PinnedBatchSource>,
}

/// Read-only provider constructed from a pinned Iceberg snapshot reader.
pub type PinnedIcebergTable = PinnedTable;

impl PinnedBatchSource for PinnedIcebergReader {
    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn split_count(&self) -> usize {
        self.snapshot().plan.files.data_files.len()
    }

    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream> {
        Ok(self
            .scan_files(splits, projection, batch_size)?
            .map_err(anyhow::Error::from)
            .boxed())
    }
}

impl PinnedTable {
    /// Wrap a reader that has already resolved the authorized metadata URI.
    pub fn new(reader: PinnedIcebergReader) -> Self {
        Self::from_source(reader)
    }

    pub(crate) fn from_source(reader: impl PinnedBatchSource + 'static) -> Self {
        Self {
            reader: Arc::new(reader),
        }
    }

    pub(crate) async fn open_catalog_snapshot(
        location: &PinnedIcebergLocation,
        columns: &[proximadb_catalog::CatalogColumn],
    ) -> anyhow::Result<Self> {
        let bridge = if location.table_root.starts_with("file://") {
            #[cfg(not(unix))]
            anyhow::bail!("confined local Iceberg reads require Unix");
            #[cfg(unix)]
            {
                use super::confined_file_store::ConfinedFileStore;
                use crate::services::ddl::external_location_root;
                use proximadb_object_store::ProximaObjectStore;

                let root = external_location_root(&location.table_root).ok_or_else(|| {
                    anyhow::anyhow!("external Iceberg root is outside the operator allowlist")
                })?;
                let relative = location
                    .table_root
                    .strip_prefix(&root)
                    .ok_or_else(|| anyhow::anyhow!("invalid external Iceberg root"))?;
                let relative = Path::parse(relative)?;
                let directory = url::Url::parse(&root)?.to_file_path().map_err(|_| {
                    anyhow::anyhow!("external Iceberg allowlist root is not a local directory URI")
                })?;
                let store = ConfinedFileStore::open(directory, relative).await?;
                IcebergObjectStoreBridge::new(ProximaObjectStore::new(Arc::new(store)))
                    .with_root_url(&location.table_root)
            }
        } else {
            IcebergObjectStoreBridge::from_url(&location.table_root)?
        };
        let reader = bridge
            .open_pinned_iceberg_reader(
                &Path::ROOT,
                &location.metadata_uri,
                location.expected_delta_version,
            )
            .await?;
        validate_external_schema(&reader.schema(), columns)
            .map_err(|e| anyhow::anyhow!("external Iceberg {e}"))?;
        Ok(Self::new(reader))
    }
}

pub(super) fn validate_external_schema(
    schema: &arrow_schema::Schema,
    columns: &[proximadb_catalog::CatalogColumn],
) -> anyhow::Result<()> {
    if schema.fields().len() != columns.len()
        || schema.fields().iter().zip(columns).any(|(field, column)| {
            use arrow_schema::DataType;
            field.name() != &column.name
                || !matches!(
                    field.data_type(),
                    DataType::Boolean
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Utf8
                        | DataType::Binary
                        | DataType::Date32
                        | DataType::Decimal128(_, _)
                )
                || crate::query::execution::datafusion_engine::arrow_type_to_proxima(
                    field.data_type(),
                ) != column.data_type
        })
    {
        anyhow::bail!(
            "catalog columns must match the supported pinned schema in name, type and order"
        );
    }
    Ok(())
}

impl fmt::Debug for PinnedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedTable")
            .field("splits", &self.reader.split_count())
            .finish()
    }
}

#[async_trait]
impl TableProvider for PinnedTable {
    fn schema(&self) -> SchemaRef {
        self.reader.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let schema = self.schema();
        let schema = match projection {
            Some(indices) => Arc::new(schema.project(indices)?),
            None => schema,
        };
        let files = self.reader.split_count();
        if files == 0 {
            return Ok(Arc::new(EmptyExec::new(schema)));
        }

        let files_per_partition = files.div_ceil(state.config().target_partitions().max(1));
        let projection: Option<Arc<[usize]>> = projection.map(Arc::from);
        let partitions: Vec<Arc<dyn PartitionStream>> = (0..files)
            .step_by(files_per_partition)
            .map(|start| {
                Arc::new(PinnedPartition {
                    reader: self.reader.clone(),
                    schema: schema.clone(),
                    files: start..start.saturating_add(files_per_partition).min(files),
                    projection: projection.clone(),
                    batch_size: state.config().batch_size(),
                }) as Arc<dyn PartitionStream>
            })
            .collect();
        // The source applies projection and its snapshot visibility rules.
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            partitions,
            None,
            [],
            false,
            limit,
        )?))
    }
}

struct PinnedPartition {
    reader: Arc<dyn PinnedBatchSource>,
    schema: SchemaRef,
    files: Range<usize>,
    projection: Option<Arc<[usize]>>,
    batch_size: usize,
}

impl fmt::Debug for PinnedPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedPartition")
            .field("files", &self.files)
            .finish()
    }
}

impl PartitionStream for PinnedPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let external = |err: anyhow::Error| DataFusionError::External(err.into());
        let batches = match self.reader.scan_splits(
            self.files.clone(),
            self.projection.as_deref(),
            self.batch_size,
        ) {
            Ok(batches) => batches.map_err(external).boxed(),
            Err(err) => stream::iter([Err(external(err))]).boxed(),
        };
        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), batches))
    }
}
