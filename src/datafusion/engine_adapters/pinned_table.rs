//! DataFusion execution adapter for a bound, read-only Arrow batch source.

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

use crate::query::execution::pinned_source::PinnedBatchSource;

/// Read-only provider for a pinned source.
pub struct PinnedTable {
    reader: Arc<dyn PinnedBatchSource>,
}

impl PinnedTable {
    pub(crate) fn from_source(reader: impl PinnedBatchSource + 'static) -> Self {
        Self {
            reader: Arc::new(reader),
        }
    }
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
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let schema = self.schema();
        let schema = match projection {
            Some(indices) => Arc::new(schema.project(indices)?),
            None => schema,
        };
        let files = self.reader.split_count();
        if files == 0 || limit == Some(0) {
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
            limit.filter(|_| filters.is_empty()),
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
        let reader = self.reader.clone();
        let files = self.files.clone();
        let projection = self.projection.clone();
        let batch_size = self.batch_size;
        // Execution plans can build partitions before downstream demands them.
        // Start the reader on first poll, so a dropped partition does no I/O.
        let batches =
            stream::once(
                async move { reader.scan_splits(files, projection.as_deref(), batch_size) },
            )
            .try_flatten()
            .map_err(|err| DataFusionError::External(err.into()))
            .boxed();
        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), batches))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{SessionConfig, SessionContext};

    use super::*;
    use crate::query::execution::pinned_source::PinnedBatchStream;

    struct CountingSource {
        schema: SchemaRef,
        opened: Arc<AtomicUsize>,
    }

    impl PinnedBatchSource for CountingSource {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }

        fn split_count(&self) -> usize {
            2
        }

        fn scan_splits(
            &self,
            splits: Range<usize>,
            projection: Option<&[usize]>,
            _batch_size: usize,
        ) -> anyhow::Result<PinnedBatchStream> {
            self.opened.fetch_add(1, Ordering::Relaxed);
            let values = Int64Array::from_iter_values(splits.map(|split| split as i64));
            let batch = RecordBatch::try_new(self.schema(), vec![Arc::new(values)])?;
            let batch = match projection {
                Some(indices) => batch.project(indices)?,
                None => batch,
            };
            Ok(stream::iter([Ok(batch)]).boxed())
        }
    }

    #[tokio::test]
    async fn partitions_open_on_demand_and_preserve_row_only_projection() {
        let opened = Arc::new(AtomicUsize::new(0));
        let provider = PinnedTable::from_source(CountingSource {
            schema: Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
            opened: opened.clone(),
        });
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));

        let empty = provider
            .scan(&ctx.state(), None, &[], Some(0))
            .await
            .unwrap();
        let rows = empty
            .execute(0, ctx.task_ctx())
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(rows.is_empty());
        assert_eq!(opened.load(Ordering::Relaxed), 0);

        let plan = provider
            .scan(&ctx.state(), Some(&[]), &[], None)
            .await
            .unwrap();
        drop(plan.execute(0, ctx.task_ctx()).unwrap());
        let rows = plan.execute(1, ctx.task_ctx()).unwrap();
        assert_eq!(opened.load(Ordering::Relaxed), 0);

        let rows = rows.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(opened.load(Ordering::Relaxed), 1);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].num_columns(), rows[0].num_rows()), (0, 1));
    }
}
