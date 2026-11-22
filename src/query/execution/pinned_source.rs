//! Batch boundary between an authorized pinned source and its executor.

use std::ops::Range;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::stream::BoxStream;

pub(crate) type PinnedBatchStream = BoxStream<'static, anyhow::Result<RecordBatch>>;

pub(crate) trait PinnedBatchSource: Send + Sync {
    fn schema(&self) -> SchemaRef;
    fn split_count(&self) -> usize;
    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream>;
}
