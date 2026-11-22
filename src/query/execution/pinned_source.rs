//! Batch boundary between an authorized pinned source and its executor.

use std::ops::Range;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::stream::BoxStream;

pub(crate) type PinnedBatchStream = BoxStream<'static, anyhow::Result<RecordBatch>>;

/// An already-bound, read-only inventory. Split indexes are local to this
/// binding and stable across scans of the same source. Each scan owns its
/// progress and releases pending work when the returned stream is dropped.
pub(crate) trait PinnedBatchSource: Send + Sync {
    fn schema(&self) -> SchemaRef;
    fn split_count(&self) -> usize;
    /// Apply source visibility before projecting. An empty projection must
    /// preserve row counts. Batch size is a row-count hint, since file
    /// decoders can have their own chunking limits.
    fn scan_splits(
        &self,
        splits: Range<usize>,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> anyhow::Result<PinnedBatchStream>;
}
